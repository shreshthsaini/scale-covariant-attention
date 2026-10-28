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


def win_sched(text):
    """'0.8:128,0.7:96' -> ((0.8, 128), (0.7, 96)): local window 128 while sigma > 0.8, then 96 while > 0.7."""
    out = []
    for item in text.split(","):
        a, b = item.split(":")
        out.append((float(a), int(b)))
    return tuple(sorted(out, reverse=True))


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
    p.add_argument("--route", type=float, default=0.0, help="Hard per-head routing threshold on pi (0 = soft mix)")
    p.add_argument("--pi-ramp", type=float, default=0.0, help="Soft opening: hand pi over from 0 at --gate-sigma "
                   "to the table at this sigma (0 = off, published method)")
    p.add_argument("--pi-ramp-mode", choices=("linear", "head"), default="linear",
                   help="linear: pi * r^gamma; head: head h opens at ramp + (gate - ramp) * pi_h")
    p.add_argument("--pi-ramp-gamma", type=float, default=1.0)
    p.add_argument("--glob-pos", choices=("floor", "frac"), default="floor",
                   help="Global-branch query/key ids: floor(p / f) (published) or p / f")
    p.add_argument("--win-sched", type=win_sched, default=(), metavar="SIGMA:WIN,...",
                   help="Wider local windows at high noise, e.g. 0.8:128,0.7:96 (default: off)")
    p.add_argument("--text-local", type=float, default=0.0, help="Logit bias on text keys in the local branch (0 = off)")
    p.add_argument("--text-local-lo", type=float, default=0.0, help="--text-local applies while sigma > this")
    p.add_argument("--chunk", type=int, help="Query tiles per local-branch chunk, gather only (default: 2 GiB budget)")
    p.add_argument("--local-impl", choices=("flex", "gather"), default="flex",
                   help="Local branch implementation (same keys; flex uses a FlexAttention block mask)")
    p.add_argument("--model", choices=("dev", "krea"), default="dev",
                   help="dev = FLUX.1-dev; krea = FLUX.1-Krea-dev transformer with FLUX.1-dev encoders and VAE")
    p.add_argument("--device", default="cuda")
    return p


def build_context(args, grid):
    from sca.attn import PiTable, SCAContext, pool_factors
    variant = VARIANT[args.method]
    if variant is None:
        return SCAContext(mode="full", known_grid=grid), {}
    target = max(pool_factors(grid, args.n0))
    if args.pi_const is not None:
        pi = PiTable("const", value=args.pi_const)
    elif variant == "switch" and args.sigma_star is not None:
        pi = PiTable("switch", sigma_star=args.sigma_star)
    elif args.pi_table is None:
        raise SystemExit("--pi-table is required for this method")
    else:
        pi = PiTable.load(args.pi_table, variant, args.pi_scale, args.sigma_star, target)
    ctx = SCAContext(mode="sca", known_grid=grid, n0=args.n0, tile=args.tile, window=args.window, pool=args.pool,
                     chunk=args.chunk, local_impl=args.local_impl, pi=pi,
                     gate_sigma=args.gate_sigma, band=args.band, route=args.route, ramp_sigma=args.pi_ramp,
                     ramp_mode=args.pi_ramp_mode, ramp_gamma=args.pi_ramp_gamma, glob_pos=args.glob_pos,
                     win_sched=args.win_sched, text_local=args.text_local, text_local_lo=args.text_local_lo)
    info = {"n0": args.n0, "tile": args.tile, "window": args.window or args.n0, "pool": args.pool,
            "chunk": args.chunk, "gate_sigma": args.gate_sigma, "band": args.band, "route": args.route, "pool_factors": list(pool_factors(grid, args.n0)), "pi_variant": pi.variant,
            "pi_const": args.pi_const, "sigma_star": pi.sigma_star, "pi_scale": getattr(pi, "scale", None),
            "pi_table_sha256": file_sha256(args.pi_table) if args.pi_const is None else None}
    # duplication options enter the sidecar (and the fingerprint) only when set, so published configs are unchanged
    if args.pi_ramp:
        info.update(pi_ramp=args.pi_ramp, pi_ramp_mode=args.pi_ramp_mode, pi_ramp_gamma=args.pi_ramp_gamma)
    if args.glob_pos != "floor":
        info["glob_pos"] = args.glob_pos
    if args.win_sched:
        info["win_sched"] = [list(x) for x in args.win_sched]
    if args.text_local:
        info.update(text_local=args.text_local, text_local_lo=args.text_local_lo)
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
    from sca.runtime import MODELS, load_pipeline
    h, w = args.res
    grid = (h // 16, w // 16)
    ctx, info = build_context(args, grid)
    pipe = load_pipeline(args.device, args.model)
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
                      "steps": args.steps, "guidance": args.guidance, "model": MODELS[args.model],
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
