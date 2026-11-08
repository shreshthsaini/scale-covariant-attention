#!/usr/bin/env python3
"""Ideogram 4 GPU check before the campaign (results to --out/check.json, images beside it).

1. Identity, one step: at 2048^2 (grid 128 <= n0 256, so pooling factor 1 and the window covers the grid) SCA with the
   mix disabled is exact attention for any pi. Both transformers' velocity outputs for one packed input at sigma 0.7,
   stock processor against: the installed processor in mode full (must be bit-identical), SCA with pi = 0 (global
   branch only), pi = 1 (flex local branch only) and pi = 0.5. Noise floor: the stock model with SDPA's flash kernel
   against its memory-efficient kernel (the same exact attention summed in another order).
2. Identity, whole trajectory: 1024^2, 48 steps, stock against SCA pi = 1 and pi = 0 (final latent and image PSNR).
3. (runs last) Flex kernel options at the 8192^2 geometry (head_dim 256, window 256, tile 8, 600 text tokens): time and peak memory
   of one local-branch call per option, next to SDPA full attention and the global branch.
4. Real per-step cost at 8192^2: one conditional and one unconditional transformer call each for full attention and
   for SCA's three step kinds (global only, 512 window, native 256 window, sca_v2 bridge); a per-image estimate.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sca.runtime import atomic_json, runtime_environment  # noqa: E402

PROMPT = ("Macro studio product photograph of a mechanical skeleton wristwatch resting on dark walnut wood, brushed "
          "steel gear trains, tiny ruby jewel bearings, a blued hairspring")


def rel(a, b):
    a, b = a.float(), b.float()
    return {"rel_l2": float((a - b).norm() / b.norm().clamp_min(1e-12)), "max_abs": float((a - b).abs().max())}


def psnr(a, b):
    import numpy as np
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    mse = float(((x - y) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def timed(fn, sync):
    import torch
    sync()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - t, torch.cuda.max_memory_allocated() / 1e9


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--parts", default="1,2,3,4")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    runtime_environment()
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from sca.attn import PiTable
    from sca.ideogram4 import (GUIDANCE_SCHEDULE, Ideogram4SCAContext, gate_from_steps, install, load_pipeline,
                               packed_inputs, sigma_schedule, transformers_of)
    want = {int(x) for x in args.parts.split(",")}
    args.out.mkdir(parents=True, exist_ok=True)
    res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    path = args.out / "check.json"
    if path.exists():
        res.update(json.loads(path.read_text()))
    save = lambda: atomic_json(path, res)
    sync = torch.cuda.synchronize
    t0 = time.perf_counter()
    pipe = load_pipeline(args.device)
    res["load_seconds"] = time.perf_counter() - t0
    tfs = dict(transformers_of(pipe))
    done_key = {1: "identity_one_step_2048", 2: "identity_trajectory_1024", 3: "flex_bench_8192", 4: "estimate_8192_seconds"}
    parts = {k for k in want if done_key[k] not in res}   # a retry resumes after the last saved part
    print("parts to run:", sorted(parts, key=(1, 2, 4, 3).index), flush=True)

    def call(name, x, ctx=None):
        tf = tfs[name]
        undo = install(tf, ctx, None) if ctx is not None else None
        try:
            with torch.inference_mode():
                return tf(**x)[0]
        finally:
            if undo:
                undo()

    if 1 in parts:
        g = torch.Generator("cpu").manual_seed(1)
        H = W = 2048
        lat = torch.randn(1, (H // 16) * (W // 16), 128, generator=g)
        cond, uncond = packed_inputs(pipe, PROMPT, H, W, lat, 0.3)       # sigma 0.7
        one = {}
        for name, x in (("cond", cond), ("uncond", uncond)):
            ref = call(name, x)
            r = {}
            for backend in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
                try:
                    with sdpa_kernel([backend]):
                        r["noise_floor"] = dict(rel(call(name, x), ref), backend=str(backend))
                    break
                except RuntimeError as e:
                    r[f"noise_floor_{backend}"] = f"unavailable: {str(e)[:160]}"
            out = call(name, x, Ideogram4SCAContext(mode="full"))
            r["installed_full_bit_identical"] = bool(torch.equal(out, ref))
            for pi in (0.0, 1.0, 0.5):
                ctx = Ideogram4SCAContext(mode="sca", n0=256, tile=8, window=256, pi=PiTable("const", value=pi))
                r[f"sca_pi{pi:g}"] = rel(call(name, x, ctx), ref)
                r[f"sca_pi{pi:g}"]["pool_factors"] = list(ctx.geometry()[0])
            one[name] = r
            print(name, json.dumps(r), flush=True)
        res["identity_one_step_2048"] = one
        save()

    if 2 in parts:
        traj, runs = {}, {}
        for tag, pi in (("stock", None), ("sca_pi1", 1.0), ("sca_pi0", 0.0)):
            undo, last = [], {}
            if pi is not None:
                for name, tf in transformers_of(pipe):
                    undo.append(install(tf, Ideogram4SCAContext(mode="sca", n0=256, tile=8, window=256,
                                                                pi=PiTable("const", value=pi)), pipe.scheduler))

            def keep(pipe_, i, t, kw):
                last["latents"] = kw["latents"].detach().clone()
                return kw
            try:
                gen = torch.Generator("cpu").manual_seed(0)
                with torch.inference_mode():
                    out, t, m = timed(lambda: pipe(prompt=PROMPT, height=1024, width=1024, num_inference_steps=48,
                                                   guidance_schedule=GUIDANCE_SCHEDULE, generator=gen,
                                                   callback_on_step_end=keep), sync)
            finally:
                for u in undo:
                    u()
            im = out.images[0]
            im.save(args.out / f"identity_1024_{tag}.png")
            runs[tag] = (last["latents"], im)
            traj[tag] = {"seconds": t, "peak_gb": m}
        for tag in ("sca_pi1", "sca_pi0"):
            traj[tag].update(latent=rel(runs[tag][0], runs["stock"][0]), psnr_db=psnr(runs[tag][1], runs["stock"][1]))
        res["identity_trajectory_1024"] = traj
        print("trajectory", json.dumps(traj), flush=True)
        save()

    if 4 in parts:
        H = W = 8192
        s = sigma_schedule(48, H, W)
        gate = gate_from_steps(s, 14)
        g = torch.Generator("cpu").manual_seed(2)
        lat = torch.randn(1, (H // 16) * (W // 16), 128, generator=g)
        steps = {}
        cond, uncond = packed_inputs(pipe, PROMPT, H, W, lat, 0.5)
        for name, x in (("cond", cond), ("uncond", uncond)):
            _, t, m = timed(lambda: call(name, x), sync)
            steps[f"full_{name}"] = {"seconds": t, "peak_gb": m}
            print("full", name, t, m, flush=True)
        save()
        steps["schedule"] = {"gate_sigma_14_global_steps": gate, "sigma_first": float(s[0]), "sigma_30": float(s[30])}
        for kind, sigma, win in (("global", float(s[0]), None), ("win512", float(s[30]), 512),
                                 ("win256", float(s[30]), None)):
            for name, x in (("cond", cond), ("uncond", uncond)):
                kw = dict(mode="sca", n0=256, tile=8, pi=PiTable("const", value=0.5), gate_sigma=gate,
                          bridge_cells=8, bridge_k=16)
                if win:
                    kw["window"] = win
                y = dict(x, timestep=torch.full_like(x["timestep"], 1 - sigma))
                ctx = Ideogram4SCAContext(**kw)
                _, tc, _ = timed(lambda: call(name, y, ctx), sync)    # compiles this layout
                _, t, m = timed(lambda: call(name, y, ctx), sync)     # same context: cached layout, warm graph
                steps[f"sca_{kind}_{name}"] = {"seconds": t, "first_call_seconds": tc, "peak_gb": m}
                print("sca", kind, name, t, tc, m, flush=True)
            save()
        full = 48 * (steps["full_cond"]["seconds"] + steps["full_uncond"]["seconds"])
        both = lambda k: steps[f"sca_{k}_cond"]["seconds"] + steps[f"sca_{k}_uncond"]["seconds"]
        # sca_v2 step mix at 48 steps: 14 global, 8 scheduled-window steps (bounded by the 512 window), 26 native window
        sca = 14 * both("global") + 8 * both("win512") + 26 * both("win256")
        res["step_cost_8192"] = steps
        res["estimate_8192_seconds"] = {"full_attention_denoise": full, "sca_v2_denoise_upper": sca,
                                        "note": "transformer calls only; text encoding and VAE decode extra"}
        print("estimate", json.dumps(res["estimate_8192_seconds"]), flush=True)
        save()

    if 3 in parts:
        from sca.attn import flex_layout, local_branch_flex, sdpa
        Hg = Wg = 512
        L, h, D = 600, 18, 256
        N = Hg * Wg
        gen = torch.Generator(args.device).manual_seed(0)
        mk = lambda n: torch.randn(1, n, h, D, device=args.device, dtype=torch.bfloat16, generator=gen)
        q, k, v = mk(L + N), mk(L + N), mk(L + N)
        bench = {}
        _, t, m = timed(lambda: sdpa(q, k, v), sync)
        _, t, m = timed(lambda: sdpa(q, k, v), sync)
        bench["sdpa_full"] = {"seconds": t, "peak_gb": m, "tflops": 4 * (L + N) ** 2 * h * D / t / 1e12}
        kc, vc = mk(L + N // 4), mk(L + N // 4)
        _, t, m = timed(lambda: sdpa(q, kc, vc), sync)
        bench["sdpa_global_branch"] = {"seconds": t, "peak_gb": m}
        for name, ko in (("torch_default", {}), ("m64_n64", {"BLOCK_M": 64, "BLOCK_N": 64}),
                         ("m64_n64_s2", {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 2}),
                         ("m128_n64_tma", {"BLOCK_M": 128, "BLOCK_N": 64, "USE_TMA": True})):
            try:
                lay = flex_layout(L, (Hg, Wg), (8, 8), (256, 256), args.device, kernel_options=ko)
                f = lambda: local_branch_flex(q[:, L:], k, v, L, lay, permuted=True)
                _, tc, _ = timed(f, sync)
                _, t, m = timed(f, sync)
                bench[f"flex_{name}"] = {"seconds": t, "first_call_seconds": tc, "peak_gb": m,
                                         "tflops_window": 4 * N * (L + 256 * 256) * h * D / t / 1e12}
            except Exception as e:   # noqa: BLE001 (a kernel option that does not compile is a result here)
                bench[f"flex_{name}"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(name, json.dumps(bench[f"flex_{name}"]), flush=True)
        res["flex_bench_8192"] = bench
        del q, k, v, kc, vc
        torch.cuda.empty_cache()
        save()

if __name__ == "__main__":
    main()
