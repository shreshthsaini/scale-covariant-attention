#!/usr/bin/env python3
"""Generate one resumable FLUX.2-dev chunk with full attention or SCA. Output (PNG + JSON sidecar per
prompt, deterministic per-prompt seeds) has the same format as scripts/sample.py, so the eval scripts
run unchanged on --out.

Memory: the Mistral text encoder (~45 GB) and the transformer (~64 GB) do not fit together on a
96 GB GH200. Prompt embeddings are computed first (cached on scratch, see sca/flux2.py), the text
encoder is freed, then the transformer is loaded. --encode-only fills the cache and exits."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sca.runtime import (atomic_json, code_revision, file_sha256, fingerprint, image_seed, image_stem,
                         implementation_hash, read_prompts, resolution, runtime_environment)

VARIANT = {"full": None, "sca": "head", "sca_uniform": "uniform", "sca_layer": "layer", "sca_switch": "switch"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=tuple(VARIANT), default="full")
    p.add_argument("--label", help="Method name written to the sidecar (default --method)")
    p.add_argument("--pi-table", type=Path, help="npz from scripts/calibrate_flux2.py (sca variants)")
    p.add_argument("--pi-scale", type=int, help="Calibrated scale to use (default: nearest to target scale)")
    p.add_argument("--pi-const", type=float, help="Constant pi for every head and sigma (ablation)")
    p.add_argument("--sigma-star", type=float, help="Switch point for sca_switch (default: fitted value)")
    p.add_argument("--res", type=resolution, default=(1024, 1024), metavar="HxW")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=4.0, help="FLUX.2-dev embedded guidance (pipeline default 4)")
    p.add_argument("--out", type=Path)
    p.add_argument("--mu-mode", choices=("native", "default"), default="native",
                   help="native: mu of the 1024^2 sequence length at every resolution (FLUX.1 max_shift analogue); "
                        "default: the pipeline's linear extrapolation (mu 11.5 at 4096^2)")
    p.add_argument("--n0", type=int, default=64)
    p.add_argument("--tile", type=int, default=8)
    p.add_argument("--window", type=int)
    p.add_argument("--pool", choices=("center", "mean"), default="center")
    p.add_argument("--gate-sigma", type=float, default=0.0, help="Global-only (pi=0) above this sigma")
    p.add_argument("--route", type=float, default=0.0, help="Hard per-head routing threshold on pi (0 = soft mix)")
    p.add_argument("--chunk", type=int, help="Query tiles per local-branch chunk, gather only")
    p.add_argument("--local-impl", choices=("flex", "gather"), default="flex")
    p.add_argument("--encode-only", action="store_true", help="Fill the prompt-embedding cache and exit")
    p.add_argument("--device", default="cuda")
    return p


def build_context(args, grid):
    from sca.attn import PiTable, pool_factors
    from sca.flux2 import Flux2SCAContext
    variant = VARIANT[args.method]
    if variant is None:
        return Flux2SCAContext(mode="full"), {}
    target = max(pool_factors(grid, args.n0))
    if args.pi_const is not None:
        pi = PiTable("const", value=args.pi_const)
    elif variant == "switch" and args.sigma_star is not None:
        pi = PiTable("switch", sigma_star=args.sigma_star)
    elif args.pi_table is None:
        raise SystemExit("--pi-table is required for this method")
    else:
        pi = PiTable.load(args.pi_table, variant, args.pi_scale, args.sigma_star, target)
    ctx = Flux2SCAContext(mode="sca", known_grid=grid, n0=args.n0, tile=args.tile, window=args.window,
                          pool=args.pool, chunk=args.chunk, local_impl=args.local_impl, pi=pi,
                          gate_sigma=args.gate_sigma, route=args.route)
    info = {"n0": args.n0, "tile": args.tile, "window": args.window or args.n0, "pool": args.pool,
            "chunk": args.chunk, "gate_sigma": args.gate_sigma, "band": 0, "route": args.route,
            "pool_factors": list(pool_factors(grid, args.n0)), "pi_variant": pi.variant,
            "pi_const": args.pi_const, "sigma_star": pi.sigma_star, "pi_scale": getattr(pi, "scale", None),
            "pi_table_sha256": file_sha256(args.pi_table) if args.pi_const is None and args.pi_table else None}
    return ctx, info


def main(argv=None):
    args = parser().parse_args(argv)
    runtime_environment()
    prompts = read_prompts(args.prompts)
    end = len(prompts) if args.end is None else args.end
    if not 0 <= args.start < end <= len(prompts) or args.steps <= 0:
        raise ValueError("Require 0 <= start < end <= number of prompts and positive steps")
    rows = prompts[args.start:end]
    import diffusers
    import torch
    from sca.flux2 import REPO, empirical_mu, encode_prompts, free_cuda, install, load_pipeline, set_mu, snapshot
    if args.encode_only:
        encode_prompts([r["prompt"] for r in rows], args.device)
        print(f"embeddings cached for {len(rows)} prompts", flush=True)
        return
    if args.out is None:
        raise SystemExit("--out is required")
    h, w = args.res
    grid = (h // 16, w // 16)
    ctx, info = build_context(args, grid)
    label = args.label or args.method

    def paths(row):
        stem = image_stem(row["id"], args.seed)
        return args.out / f"{stem}.png", args.out / f"{stem}.json"

    def done(row):
        png, meta = paths(row)
        return png.exists() and meta.exists() and json.loads(meta.read_text()).get("complete")
    todo = [r for r in rows if not done(r)]
    embeds = dict(zip([r["id"] for r in todo], encode_prompts([r["prompt"] for r in todo], args.device)))
    free_cuda()
    pipe = load_pipeline(args.device)
    uninstall = install(pipe.transformer, ctx)
    ctx.scheduler = pipe.scheduler
    if h * w >= 2048 * 4096:
        pipe.vae.enable_tiling()
    set_mu(args.mu_mode)
    mu = empirical_mu(grid[0] * grid[1], args.steps, args.mu_mode)
    revision, impl = code_revision(), implementation_hash()
    try:
        for row in rows:
            png, meta = paths(row)
            config = {"schema_version": 1, "method": label, "res": f"{h}x{w}", "prompt_id": str(row["id"]),
                      "prompt": row["prompt"], "seed": int(args.seed), "derived_seed": image_seed(row["id"], args.seed),
                      "steps": args.steps, "guidance": args.guidance, "model": REPO, "snapshot": snapshot().name,
                      "dtype": "bfloat16", "batch_size": 1, "attention": f"sca:{args.method}",
                      "diffusers": diffusers.__version__, "torch": torch.__version__,
                      "implementation_sha256": impl, "grid": list(grid), "mu_mode": args.mu_mode, "mu": mu,
                      "text_encoder": "Mistral3 hidden states 10/20/30, 512 tokens, precomputed",
                      "sca": info, "scheduler": dict(pipe.scheduler.config)}
            digest = fingerprint(config)
            if png.exists() and meta.exists():
                saved = json.loads(meta.read_text())
                if saved.get("complete"):
                    if saved.get("config_fingerprint") != digest:
                        raise ValueError(f"Existing result has different settings: {meta}. Choose a new --out.")
                    if file_sha256(png) != saved["png_sha256"]:
                        raise ValueError(f"Existing PNG checksum differs: {png}")
                    print(f"skip {row['id']} {label} {config['res']}", flush=True)
                    continue
            ctx.pi_trace.clear()
            generator = torch.Generator(device="cpu").manual_seed(image_seed(row["id"], args.seed))
            cuda = args.device.startswith("cuda")
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.inference_mode():
                image = pipe(prompt=None, prompt_embeds=embeds[row["id"]].to(args.device), height=h, width=w,
                             num_inference_steps=args.steps, guidance_scale=args.guidance, generator=generator,
                             max_sequence_length=512).images[0]
            if cuda:
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            args.out.mkdir(parents=True, exist_ok=True)
            tmp = png.with_name(png.name + f".tmp.{os.getpid()}")
            image.save(tmp, format="PNG")
            os.replace(tmp, png)
            atomic_json(meta, {"complete": True, "config": config, "config_fingerprint": digest,
                               "code_revision": revision, "image": str(png.resolve()), "png_sha256": file_sha256(png),
                               "seconds": seconds, "seconds_scope": "denoising and VAE decode; prompt embeddings "
                               "precomputed (text encoder not resident); excludes model loading and PNG I/O",
                               "peak_memory_bytes": torch.cuda.max_memory_allocated() if cuda else None,
                               "sigmas": [float(s) for s in pipe.scheduler.sigmas[:-1]],
                               "pi_trace": list(ctx.pi_trace), "sca_log": sorted(set(ctx.log)),
                               "local_impl": args.local_impl})
            print(f"saved {row['id']} {label} {config['res']} seconds={seconds:.3f} "
                  f"peak_gb={(torch.cuda.max_memory_allocated() / 1e9) if cuda else 0:.1f}", flush=True)
    finally:
        uninstall()


if __name__ == "__main__":
    main()
