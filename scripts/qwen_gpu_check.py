#!/usr/bin/env python3
"""GPU checks of the Qwen-Image port (sca/qwen.py) on the real model; writes a JSON report.

1. Embeddings: fills the prompt-embedding cache for the given prompt files plus the negative prompt " ".
2. Exactness at the native 1328^2 grid (83 x 83, n0 = 83, so the pool factor is 1 and the window covers the grid):
   one transformer call with the stock processors against SCA in mode full, SCA with the mix disabled (pi = 0, global
   branch only) and SCA with pi = 1 (local branch only), relative L2 error and max abs difference in bf16.
3. Pipeline latents after a few steps (true CFG, 1328^2): stock against mode full.
4. Cost at 4096^2 and 5312^2: one transformer call per configuration after a warm-up (global-only step, widest window
   step, native-window step for SCA with the sca_v2 flags; full attention), seconds and peak memory, for the cond and
   the negative prompt's text length."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sca.runtime import atomic_json, read_prompts, runtime_environment  # noqa: E402


def rel(a, b):
    a, b = a.float(), b.float()
    return {"rel_l2": float((a - b).norm() / b.norm()), "max_abs": float((a - b).abs().max()),
            "ref_rms": float(b.pow(2).mean().sqrt()), "exact": bool(torch_equal(a, b))}


def torch_equal(a, b):
    import torch
    return torch.equal(a, b)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, nargs="+", required=True, help="Prompt files whose embeddings to cache")
    p.add_argument("--count", type=int, nargs="+", help="Rows to take per prompt file (default all)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sizes", default="4096x4096,5312x5312")
    p.add_argument("--steps", type=int, default=4, help="Pipeline steps for the latent comparison")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    runtime_environment()
    import torch
    from sca.attn import PiTable
    from sca.qwen import QwenSCAContext, encode_prompts, install, load_pipeline, set_mu
    report = {"steps": args.steps}
    rows = []
    for i, f in enumerate(args.prompts):
        r = read_prompts(f)
        rows += r[: args.count[i]] if args.count else r
    t0 = time.perf_counter()
    embeds = encode_prompts([r["prompt"] for r in rows] + [" "], args.device)
    report["encode"] = {"prompts": len(rows) + 1, "seconds": time.perf_counter() - t0}
    neg = embeds[-1].to(args.device)
    cond = embeds[0].to(args.device)
    pipe = load_pipeline(args.device)
    tr = pipe.transformer
    set_mu("native")
    dev = torch.device(args.device)

    def call(H, W, emb, sigma, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(1, H * W, tr.config.in_channels, generator=g).to(dev, torch.bfloat16)
        with torch.inference_mode():
            return tr(hidden_states=x, encoder_hidden_states=emb, encoder_hidden_states_mask=None,
                      timestep=torch.tensor([sigma], device=dev, dtype=torch.bfloat16), img_shapes=[[(1, H, W)]],
                      return_dict=False)[0]

    # 2. exactness at the native grid
    H = W = 83
    exact = {}
    for sigma in (0.95, 0.5):
        ref = call(H, W, cond, sigma)
        for name, ctx in (("full", QwenSCAContext(mode="full")),
                          ("sca_pi0_global_only", QwenSCAContext(mode="sca", n0=83, tile=1, window=83,
                                                                 pi=PiTable("const", value=0.0))),
                          ("sca_pi1_local_only", QwenSCAContext(mode="sca", n0=83, tile=1, window=83,
                                                                pi=PiTable("const", value=1.0)))):
            undo = install(tr, ctx)
            try:
                out = call(H, W, cond, sigma)
            finally:
                undo()
            exact[f"{name}@sigma{sigma}"] = rel(out, ref)
        # bf16 noise floor: the stock model in bf16 against itself with the inputs perturbed by one bf16 ulp
        g = torch.Generator(device="cpu").manual_seed(0)
        x = torch.randn(1, H * W, tr.config.in_channels, generator=g).to(dev, torch.bfloat16)
        with torch.inference_mode():
            xa = tr(hidden_states=x, encoder_hidden_states=cond, timestep=torch.tensor([sigma], device=dev,
                    dtype=torch.bfloat16), img_shapes=[[(1, H, W)]], return_dict=False)[0]
            x2 = (x.float() * (1 + 2 ** -8)).to(torch.bfloat16)
            xb = tr(hidden_states=x2, encoder_hidden_states=cond, timestep=torch.tensor([sigma], device=dev,
                    dtype=torch.bfloat16), img_shapes=[[(1, H, W)]], return_dict=False)[0]
        exact[f"bf16_input_ulp_floor@sigma{sigma}"] = rel(xb, xa)
    report["exactness_1328"] = exact
    print(json.dumps(exact, indent=1), flush=True)

    # 3. pipeline latents, stock vs mode full
    def latents(ctx):
        undo = install(tr, ctx) if ctx is not None else None
        if ctx is not None:
            ctx.scheduler = pipe.scheduler
        try:
            g = torch.Generator(device="cpu").manual_seed(1234)
            with torch.inference_mode():
                return pipe(prompt=None, prompt_embeds=cond, negative_prompt_embeds=neg, true_cfg_scale=4.0,
                            height=1328, width=1328, num_inference_steps=args.steps, generator=g,
                            output_type="latent").images
        finally:
            if undo:
                undo()
    report["pipeline_latents_full_vs_stock"] = rel(latents(QwenSCAContext(mode="full")), latents(None))
    print("pipeline", report["pipeline_latents_full_vs_stock"], flush=True)

    # 4. cost per transformer call at the showcase sizes
    cost = {}
    for size in args.sizes.split(","):
        h, w = (int(v) for v in size.split("x"))
        H, W = h // 16, w // 16
        cells, per = (8, 16) if H % 8 == 0 and W % 8 == 0 else (4, 64)
        sca = dict(n0=83, tile=8, pool="center", gate_sigma=0.89, bridge_cells=cells, bridge_k=per,
                   win_sched=((0.88, 166), (0.86, 145), (0.84, 124), (0.82, 104)), win_pi_match=True,
                   win_pi_interp=True, pi=PiTable("const", value=0.5))
        for name, ctx in (("full", QwenSCAContext(mode="full")), ("sca", QwenSCAContext(mode="sca", **sca))):
            undo = install(tr, ctx)
            try:
                for label, emb in (("cond", cond), ("neg", neg)):
                    for sigma in ((0.5,) if name == "full" else (0.95, 0.87, 0.5)):
                        call(H, W, emb, sigma)   # warm-up (compiles the flex layout for this text length)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        t = time.perf_counter()
                        out = call(H, W, emb, sigma)
                        torch.cuda.synchronize()
                        key = f"{size}/{name}/{label}/sigma{sigma}"
                        cost[key] = {"seconds": time.perf_counter() - t,
                                     "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                                     "finite": bool(torch.isfinite(out).all()), "text_len": int(emb.shape[1])}
                        print(key, cost[key], flush=True)
            finally:
                undo()
            report["sca_log"] = sorted(set(ctx.log))
    report["cost"] = cost
    report["gpu"] = torch.cuda.get_device_name()
    atomic_json(args.out, report)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
