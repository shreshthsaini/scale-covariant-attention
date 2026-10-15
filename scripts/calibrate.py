#!/usr/bin/env python3
"""SCA calibration: run FLUX with exact attention at an in-distribution resolution, emulate
scales s_c (window n/s_c, pooling s_c), accumulate output-matching statistics per prompt
(resumable), then merge into a pi table npz and a JSON report (same path, .json)."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sca.runtime import atomic_json, image_seed, image_stem, read_prompts, resolution, runtime_environment, set_mu


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, default=Path(__file__).with_name("calib_prompts.jsonl"))
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--res", type=resolution, default=(1024, 1024), metavar="HxW")
    p.add_argument("--scales", default="2,4")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma-bins", type=int, default=14)
    p.add_argument("--mu-mode", choices=("max_shift", "default"), default="max_shift")
    p.add_argument("--n0", type=int, default=64)
    p.add_argument("--tile", type=int, default=8, help="Inference tile t; emulation uses t * Wc / Wn")
    p.add_argument("--window", type=int, help="Inference window Wn (default n0)")
    p.add_argument("--pool", choices=("center", "mean"), default="center")
    p.add_argument("--mass-queries", type=int, default=256)
    p.add_argument("--out", type=Path, required=True, help="pi table .npz; parts go to <out stem>_parts/")
    p.add_argument("--report-only", action="store_true", help="Merge existing parts without running FLUX")
    p.add_argument("--transfer-with", type=Path, nargs="*", default=[], help="Other calibration npz files")
    p.add_argument("--device", default="cuda")
    return p


def run(args, rows, parts, settings):
    import torch
    from sca.attn import SCAContext, install
    from sca.calib import CalibStats
    from sca.runtime import load_pipeline
    pipe = load_pipeline(args.device)
    ctx = SCAContext(mode="calib", n0=args.n0, tile=args.tile, window=args.window, pool=args.pool,
                     calib_scales=tuple(settings["scales"]))
    uninstall = install(pipe.transformer, ctx)
    ctx.scheduler = pipe.scheduler
    h, w = args.res
    set_mu(pipe, h, w, args.mu_mode)
    heads = pipe.transformer.config.num_attention_heads
    try:
        for j, row in enumerate(rows):
            part = parts / f"{image_stem(row['id'], args.seed)}.npz"
            if part.exists():
                print(f"skip {row['id']}", flush=True)
                continue
            ctx.calib = CalibStats(ctx.n_layers, heads, ctx.calib_scales, args.sigma_bins,
                                   mass_queries=args.mass_queries, device=args.device, seed=j)
            generator = torch.Generator(device="cpu").manual_seed(image_seed(row["id"], args.seed))
            with torch.inference_mode():
                pipe(prompt=row["prompt"], height=h, width=w, num_inference_steps=args.steps,
                     guidance_scale=args.guidance, generator=generator, max_sequence_length=512, output_type="latent")
            tmp = parts / f"{part.stem}.tmp{os.getpid()}.npz"
            import numpy as np
            np.savez(tmp, prompt_index=np.array(j), **ctx.calib.state())
            os.replace(tmp, part)
            print(f"calibrated {row['id']} ({j + 1}/{len(rows)})", flush=True)
    finally:
        uninstall()
    return ctx.log


def main(argv=None):
    args = parser().parse_args(argv)
    runtime_environment()
    rows = read_prompts(args.prompts)[: args.count]
    parts = args.out.parent / f"{args.out.stem}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    settings = {"res": f"{args.res[0]}x{args.res[1]}", "scales": [int(s) for s in args.scales.split(",")],
                "steps": args.steps, "guidance": args.guidance, "seed": args.seed, "sigma_bins": args.sigma_bins,
                "mu_mode": args.mu_mode, "n0": args.n0, "tile": args.tile, "window": args.window,
                "pool": args.pool, "mass_queries": args.mass_queries, "prompts": str(args.prompts.resolve())}
    cfg = parts / "calib_config.json"
    if cfg.exists() and json.loads(cfg.read_text()) != settings:
        raise SystemExit(f"{cfg} has different settings; choose a new --out")
    atomic_json(cfg, settings)
    log = [] if args.report_only else run(args, rows, parts, settings)

    from sca.calib import load_parts, merge, transfer
    paths = [parts / f"{image_stem(r['id'], args.seed)}.npz" for r in rows]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("no calibration parts to merge")
    report = merge(load_parts(paths), settings["res"], args.out)
    report["settings"] = settings
    report["geometry_log"] = sorted(set(log))
    report["transfer"] = transfer([args.out, *args.transfer_with])
    atomic_json(args.out.with_suffix(".json"), report)
    for s, r in report["scales"].items():
        print(f"s={s}", json.dumps(r["rel_err"]), "split_half", r.get("split_half_corr"),
              "frac_mid", r["frac_pi_mid"], "sigma_star", r["sigma_star"], flush=True)


if __name__ == "__main__":
    main()
