#!/usr/bin/env python3
"""GPU check of the SCA local branch: flex vs gather equivalence and per-layer attention timing.

  python scripts/bench_attn.py equiv              # max |flex - gather| on random q/k/v, bf16 and fp32
  python scripts/bench_attn.py bench --grid 256x256 [--variants]

Timing is one attention call (one layer, batch 1, 24 heads, head dim 128, 512 text tokens) from
pre-RoPE q/k/v, minimum of CUDA-event timings after warmup. Rows: full SDPA (RoPE + SDPA),
gather O_loc, flex O_loc (permutation copies included), global branch, and total SCA
(SCAProcessor.sca: shared RoPE, both branches and the pi mix).
"""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sca import attn as A  # noqa: E402


def grid_arg(s):
    h, w = s.lower().split("x")
    return int(h), int(w)


def setup(grid, L=512, heads=24, D=128, dtype=torch.bfloat16, seed=0):
    from diffusers.models.transformers.transformer_flux import FluxPosEmbed
    H, W = grid
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(1, L + H * W, heads, D, device="cuda", generator=g, dtype=dtype) for _ in range(3))
    ids = torch.zeros(H, W, 3, device="cuda")
    ids[..., 1] = torch.arange(H, device="cuda")[:, None]
    ids[..., 2] = torch.arange(W, device="cuda")[None, :]
    ids = ids.reshape(-1, 3)
    pe = FluxPosEmbed(theta=10000, axes_dim=[16, 56, 56])
    rot = pe(torch.cat([torch.zeros(L, 3, device="cuda"), ids]))
    return q, k, v, rot, ids, pe


def make_proc(grid, ids, pe, impl, tile=8, n0=64, pi=0.5):
    ctx = A.SCAContext(mode="sca", n0=n0, tile=tile, local_impl=impl, pi=A.PiTable("const", value=pi))
    ctx.pos_embed, ctx.n_layers = pe, 1
    ctx.set_inputs(ids, sigma=0.5)
    return A.SCAProcessor(ctx, 0), ctx


def timeit(fn, iters=15, warmup=3):
    """Minimum over iterations (robust to a co-tenant on the GPU), in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return min(ts)


@torch.inference_mode()
def equiv(args):
    rows = []
    for grid in args.grids:
        for dtype in (torch.bfloat16, torch.float32):
            q, k, v, rot, ids, pe = setup(grid, dtype=dtype, seed=1)
            q, k = q * 2, k * 2  # sharper attention than unit-variance logits
            L = 512
            for tile in (8, 16):
                _, ctx = make_proc(grid, ids, pe, "flex", tile=tile)
                _, t, win = ctx.geometry()
                qr, kr = A.rope(q, rot), A.rope(k, rot)
                ref = A.local_branch(qr, kr, v, L, grid, t, win)
                out = A.local_branch_flex(qr[:, L:], kr, v, L, ctx.flex_layout(L, t, win, q.device))
                d = (out.float() - ref.float()).abs()
                row = {"grid": f"{grid[0]}x{grid[1]}", "dtype": str(dtype).split(".")[-1], "tile": t, "window": win,
                       "max_abs_diff": d.max().item(), "mean_abs_diff": d.mean().item(),
                       "ref_abs_mean": ref.float().abs().mean().item()}
                if dtype == torch.bfloat16:  # both against an fp32 gather reference
                    r32 = A.local_branch(qr.float(), kr.float(), v.float(), L, grid, t, win)
                    row["gather_vs_fp32"] = (ref.float() - r32).abs().max().item()
                    row["flex_vs_fp32"] = (out.float() - r32).abs().max().item()
                # whole SCA attention (both branches + mix): gather path vs fused flex path
                if tile == 8:
                    pg, _ = make_proc(grid, ids, pe, "gather", tile=tile)
                    pf, _ = make_proc(grid, ids, pe, "flex", tile=tile)
                    sg, sf = pg.sca(q, k, v, rot).float(), pf.sca(q, k, v, rot).float()
                    row["sca_total_max_abs_diff"] = (sf - sg).abs().max().item()
                    row["sca_total_mean_abs_diff"] = (sf - sg).abs().mean().item()
                    del sg, sf
                print(json.dumps(row), flush=True)
                rows.append(row)
                del ref, out, d
                torch.cuda.empty_cache()
    return rows


def interleaved(cands, rounds=8, reps=3):
    """Min ms per candidate, timed round-robin so a GPU co-tenant affects all candidates alike."""
    out = {}
    for name, fn in cands:
        try:
            fn()
            torch.cuda.synchronize()
        except Exception as e:  # report and drop the candidate
            out[name] = f"error: {type(e).__name__}: {str(e)[:160]}"
    live = [(n, f) for n, f in cands if n not in out]
    best = {n: float("inf") for n, _ in live}
    for _ in range(rounds):
        for name, fn in live:
            best[name] = min(best[name], timeit(fn, iters=reps, warmup=1))
    out.update(best)
    return out


@torch.inference_mode()
def bench(args):
    grid, L = args.grid, 512
    q, k, v, rot, ids, pe = setup(grid)
    res = {"grid": f"{grid[0]}x{grid[1]}", "res": f"{grid[0] * 16}x{grid[1] * 16}"}
    qr, kr = A.rope(q, rot), A.rope(k, rot)
    qi = qr[:, L:].contiguous()
    cands = [("full_sdpa", lambda: A.sdpa(A.rope(q, rot), A.rope(k, rot), v))]
    for tile in args.tiles:
        for impl in ("gather", "flex"):
            if impl == "gather" and args.skip_gather:
                continue
            proc, ctx = make_proc(grid, ids, pe, impl, tile=tile)
            f, t, win = ctx.geometry()
            if impl == "gather":
                rq, rk = ctx.coarse_rope(f, q.device)
                txt = (rot[0][:L], rot[1][:L])
                cands.append((f"t{tile}_global", lambda f=f, rq=rq, rk=rk, txt=txt, ctx=ctx:
                              A.global_branch(q, k, v, L, grid, f, ctx.pool, txt, rq, rk)))
                cands.append((f"t{tile}_gather_local", lambda t=t, win=win: A.local_branch(qi, kr, v, L, grid, t, win)))
            cands.append((f"t{tile}_sca_total_{impl}", lambda proc=proc: proc.sca(q, k, v, rot)))
            if impl == "flex":
                proc2, ctx2 = make_proc(grid, ids, pe, impl, tile=tile)
                ctx2.fused = False
                cands.append((f"t{tile}_sca_total_flex_unfused", lambda proc=proc2: proc.sca(q, k, v, rot)))
        variants = [("default", None, None, None)]
        if args.variants:
            variants += [("q8x16", (8, 16), (8, 8), None), ("q16x16", (16, 16), (8, 8), None),
                         ("q16x16_k16x16", (16, 16), (16, 16), None),
                         ("tma", None, None, {"BLOCK_M": 64, "BLOCK_N": 64, "USE_TMA": True}),
                         ("q8x16_tma", (8, 16), (8, 8), {"BLOCK_M": 128, "BLOCK_N": 64, "USE_TMA": True}),
                         ("q16x16_tma", (16, 16), (8, 8), {"BLOCK_M": 128, "BLOCK_N": 64, "USE_TMA": True})]
        _, ctx = make_proc(grid, ids, pe, "flex", tile=tile)
        _, t, win = ctx.geometry()
        for name, qt, kt, ko in variants:
            try:
                lay = A.flex_layout(L, grid, t, win, q.device, qt, kt, ko)
            except Exception as e:
                res[f"t{tile}_flex_local_{name}"] = f"error: {type(e).__name__}: {str(e)[:160]}"
                continue
            res[f"t{tile}_flex_{name}_density"] = round(1 - float(lay.block_mask.sparsity()) / 100, 4)
            cands.append((f"t{tile}_flex_local_{name}", lambda lay=lay: A.local_branch_flex(qi, kr, v, L, lay)))
    res.update(interleaved(cands, rounds=args.rounds))
    res["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    print(json.dumps(res), flush=True)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("equiv", "bench"))
    p.add_argument("--grids", type=grid_arg, nargs="*", default=[(64, 64), (128, 128), (256, 256), (128, 256)])
    p.add_argument("--grid", type=grid_arg, default=(256, 256))
    p.add_argument("--tiles", type=int, nargs="*", default=[8])
    p.add_argument("--variants", action="store_true", help="Also time other flex block layouts")
    p.add_argument("--skip-gather", action="store_true")
    p.add_argument("--rounds", type=int, default=8)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    (equiv if args.what == "equiv" else bench)(args)


if __name__ == "__main__":
    main()
