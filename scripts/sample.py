#!/usr/bin/env python3
"""Generate one resumable FLUX chunk with full attention or SCA. Output (PNG + JSON sidecar per
prompt, deterministic per-prompt seeds) matches constancy_rope/scripts/sample.py, so its eval.py
runs unchanged on --out."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sca.runtime import (atomic_json, code_revision, file_sha256, fingerprint, image_seed, image_stem,
                         implementation_hash, read_prompts, resolution, runtime_environment, set_mu)

VARIANT = {"full": None, "sca": "head", "sca_uniform": "uniform", "sca_layer": "layer", "sca_switch": "switch"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=tuple(VARIANT))
    p.add_argument("--label", help="Method name written to the sidecar (default --method); use for ablations")
    p.add_argument("--pi-table", type=Path, help="npz from scripts/calibrate.py (required for sca variants "
                   "except sca_switch with --sigma-star and sca_uniform with --pi-const)")
    p.add_argument("--pi-scale", type=int, help="Calibrated scale to use (default: nearest to target scale)")
    p.add_argument("--pi-const", type=float, help="Constant pi for every head and sigma (ablation)")
    p.add_argument("--sigma-star", type=float, help="Switch point for sca_switch (default: fitted value)")
    p.add_argument("--res", type=resolution, required=True, metavar="HxW")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mu-mode", choices=("max_shift", "default"), default="max_shift")
    p.add_argument("--n0", type=int, default=64)
    p.add_argument("--tile", type=int, default=8)
    p.add_argument("--window", type=int)
    p.add_argument("--pool", choices=("center", "mean"), default="center")
    p.add_argument("--gate-sigma", type=float, default=0.0, help="Global-only (pi=0) above this sigma")
    p.add_argument("--band", type=int, default=0, help="Low band (pool size, tokens) taken from O_glob")
    p.add_argument("--chunk", type=int, help="Query tiles per local-branch chunk, gather only (default: 2 GiB budget)")
    p.add_argument("--local-impl", choices=("flex", "gather"), default="flex",
                   help="Local branch implementation (same keys; flex uses a FlexAttention block mask)")
    p.add_argument("--device", default="cuda")
    return p


def build_context(args, grid):
    from sca.attn import PiTable, SCAContext, pool_factors
    variant = VARIANT[args.method]
    if variant is None:
        return SCAContext(mode="full"), {}
    target = max(pool_factors(grid, args.n0))
    if args.pi_const is not None:
        pi = PiTable("const", value=args.pi_const)
    elif variant == "switch" and args.sigma_star is not None:
        pi = PiTable("switch", sigma_star=args.sigma_star)
    elif args.pi_table is None:
        raise SystemExit("--pi-table is required for this method")
    else:
        pi = PiTable.load(args.pi_table, variant, args.pi_scale, args.sigma_star, target)
    ctx = SCAContext(mode="sca", n0=args.n0, tile=args.tile, window=args.window, pool=args.pool,
                     chunk=args.chunk, local_impl=args.local_impl, pi=pi,
                     gate_sigma=args.gate_sigma, band=args.band)
    info = {"n0": args.n0, "tile": args.tile, "window": args.window or args.n0, "pool": args.pool,
            "chunk": args.chunk, "gate_sigma": args.gate_sigma, "band": args.band, "pool_factors": list(pool_factors(grid, args.n0)), "pi_variant": pi.variant,
            "pi_const": args.pi_const, "sigma_star": pi.sigma_star, "pi_scale": getattr(pi, "scale", None),
            "pi_table_sha256": file_sha256(args.pi_table) if args.pi_const is None else None}
    return ctx, info


def main(argv=None):
    args = parser().parse_args(argv)
    runtime_environment()
    prompts = read_prompts(args.prompts)
    end = len(prompts) if args.end is None else args.end
    if not 0 <= args.start < end <= len(prompts) or args.steps <= 0:
        raise ValueError("Require 0 <= start < end <= number of prompts and positive steps")
    import diffusers
    import torch
    from PIL import Image
    from sca.attn import install
    from sca.runtime import load_pipeline
    h, w = args.res
    grid = (h // 16, w // 16)
    ctx, info = build_context(args, grid)
    pipe = load_pipeline(args.device)
    uninstall = install(pipe.transformer, ctx)
    ctx.scheduler = pipe.scheduler
    if h * w >= 2048 * 4096:
        pipe.vae.enable_tiling()
    set_mu(pipe, h, w, args.mu_mode)
    label = args.label or args.method
    revision, impl = code_revision(), implementation_hash()
    try:
        for row in prompts[args.start:end]:
            stem = image_stem(row["id"], args.seed)
            png, meta = args.out / f"{stem}.png", args.out / f"{stem}.json"
            config = {"schema_version": 1, "method": label, "res": f"{h}x{w}", "prompt_id": str(row["id"]),
                      "prompt": row["prompt"], "seed": int(args.seed), "derived_seed": image_seed(row["id"], args.seed),
                      "steps": args.steps, "guidance": args.guidance, "model": "black-forest-labs/FLUX.1-dev",
                      "dtype": "bfloat16", "batch_size": 1, "attention": f"sca:{args.method}",
                      "diffusers": diffusers.__version__, "torch": torch.__version__,
                      "implementation_sha256": impl, "grid": list(grid), "mu_mode": args.mu_mode,
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
                image = pipe(prompt=row["prompt"], height=h, width=w, num_inference_steps=args.steps,
                             guidance_scale=args.guidance, generator=generator, max_sequence_length=512).images[0]
            if cuda:
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            args.out.mkdir(parents=True, exist_ok=True)
            tmp = png.with_name(png.name + f".tmp.{os.getpid()}")
            image.save(tmp, format="PNG")
            os.replace(tmp, png)
            atomic_json(meta, {"complete": True, "config": config, "config_fingerprint": digest,
                               "code_revision": revision, "image": str(png.resolve()), "png_sha256": file_sha256(png),
                               "seconds": seconds, "seconds_scope": "prompt encoding, denoising, and VAE decode; "
                               "excludes model loading and PNG I/O",
                               "peak_memory_bytes": torch.cuda.max_memory_allocated() if cuda else None,
                               "sigmas": [float(s) for s in pipe.scheduler.sigmas[:-1]],
                               "pi_trace": list(ctx.pi_trace), "sca_log": sorted(set(ctx.log)),
                               "local_impl": args.local_impl})  # not fingerprinted: both give the same keys
            print(f"saved {row['id']} {label} {config['res']} seconds={seconds:.3f}", flush=True)
    finally:
        uninstall()


if __name__ == "__main__":
    main()
