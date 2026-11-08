#!/usr/bin/env python3
"""Reach emulation for Ideogram 4 (same statistics and tables as scripts/calibrate.py): run Ideogram 4 with exact
attention at a native-scale size, emulate the target's image-to-window ratio s_c (window n / s_c, pooling s_c, tile
t * Wc / Wn), store one part per prompt and transformer (resumable), then merge into one reach table per transformer
plus a JSON report (same path, .json).

Both transformers run at inference (the conditional one on [text][image], the unconditional one on [image]), and they
have different weights, so each gets its own table: <out>_cond.npz and <out>_uncond.npz. Sampling follows the model
defaults (48 steps, guidance schedule 7.0 x 45 then 3.0 x 3, dual CFG, logit-normal schedule of the run's size), on the
upsampled captions of the emulation prompts (scripts/ideogram4_captions.py). --n0 defaults to the run's grid side, so
the emulated window is H / s_c and the tile keeps the inference tile-to-window ratio. --start/--end split the first
--count prompts across tasks; run with --no-merge per split and once with --report-only after all parts exist."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sca.runtime import atomic_json, image_seed, image_stem, read_prompts, resolution, runtime_environment  # noqa: E402

NAMES = ("cond", "uncond")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, default=Path(__file__).with_name("calib_prompts.jsonl"))
    p.add_argument("--captions", type=Path, required=True, help="Caption JSONL holding the emulation prompts")
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--start", type=int, default=0, help="First prompt index (within --count) of this split")
    p.add_argument("--end", type=int, help="End prompt index of this split (default --count)")
    p.add_argument("--res", type=resolution, default=(4096, 4096), metavar="HxW")
    p.add_argument("--scales", default="2")
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma-bins", type=int, default=14)
    p.add_argument("--n0", type=int, help="Emulation grid side (default: the run's grid side)")
    p.add_argument("--tile", type=int, default=8, help="Inference tile t; emulation uses t * Wc / Wn")
    p.add_argument("--window", type=int, help="Inference window Wn (default n0)")
    p.add_argument("--pool", choices=("center", "mean"), default="center")
    p.add_argument("--local-impl", choices=("flex", "gather"), default="flex")
    p.add_argument("--flex-kernel", type=json.loads, default={}, help="flex kernel options as JSON ({} = torch defaults)")
    p.add_argument("--mass-queries", type=int, default=256)
    p.add_argument("--out", type=Path, required=True, help="Table stem: writes <stem>_cond.npz, <stem>_uncond.npz")
    p.add_argument("--no-merge", action="store_true", help="Only write this split's parts")
    p.add_argument("--report-only", action="store_true", help="Merge existing parts without running Ideogram 4")
    p.add_argument("--transfer-with", type=Path, nargs="*", default=[], help="Other table stems (same transformer)")
    p.add_argument("--device", default="cuda")
    return p


def part_dir(out, name):
    return out.parent / f"{out.name}_{name}_parts"


def run(args, rows, offset, settings, captions):
    import numpy as np
    import torch
    from sca.calib import CalibStats
    from sca.ideogram4 import GUIDANCE_SCHEDULE, Ideogram4SCAContext, install, load_pipeline, transformers_of
    stem = lambda r: image_stem(r["id"], args.seed)
    todo = [(j, r) for j, r in enumerate(rows, offset)
            if not all((part_dir(args.out, n) / f"{stem(r)}.npz").exists() for n in NAMES)]
    if not todo:
        print("all parts present", flush=True)
        return []
    pipe = load_pipeline(args.device)
    h, w = args.res
    n0 = args.n0 or h // 16
    ctxs, undo = {}, []
    for name, tf in transformers_of(pipe):
        ctx = Ideogram4SCAContext(mode="calib", n0=n0, tile=args.tile, window=args.window, pool=args.pool,
                                  local_impl=args.local_impl, calib_scales=tuple(settings["scales"]))
        ctx.flex_kernel = dict(args.flex_kernel)
        undo.append(install(tf, ctx, pipe.scheduler))
        ctxs[name] = ctx
    if set(ctxs) != set(NAMES):
        raise SystemExit("reach emulation needs both transformers (dual guidance)")
    heads = pipe.transformer.config.num_attention_heads
    schedule = GUIDANCE_SCHEDULE if args.steps == len(GUIDANCE_SCHEDULE) else None
    kw = {"guidance_schedule": schedule} if schedule else {"guidance_scale": 7.0, "guidance_schedule": None}
    try:
        for j, row in todo:
            for i, ctx in enumerate(ctxs.values()):
                ctx.calib = CalibStats(ctx.n_layers, heads, ctx.calib_scales, args.sigma_bins,
                                       mass_queries=args.mass_queries, device=args.device, seed=2 * j + i)
            generator = torch.Generator(device="cpu").manual_seed(image_seed(row["id"], args.seed))
            with torch.inference_mode():
                pipe(prompt=captions[row["id"]], height=h, width=w, num_inference_steps=args.steps,
                     guidance_mode="dual", generator=generator, prompt_upsampling=False, output_type="latent", **kw)
            for name, ctx in ctxs.items():
                d = part_dir(args.out, name)
                d.mkdir(parents=True, exist_ok=True)
                tmp = d / f"{stem(row)}.tmp{os.getpid()}.npz"
                np.savez(tmp, prompt_index=np.array(j), **ctx.calib.state())
                os.replace(tmp, d / f"{stem(row)}.npz")
            print(f"emulated {row['id']} ({j + 1}/{args.count})", flush=True)
    finally:
        for u in undo:
            u()
    return sorted({m for c in ctxs.values() for m in c.log})


def main(argv=None):
    args = parser().parse_args(argv)
    runtime_environment()
    rows = read_prompts(args.prompts)[: args.count]
    end = len(rows) if args.end is None else min(args.end, len(rows))
    caps = {}
    for line in args.captions.read_text().splitlines():
        if line.strip():
            c = json.loads(line)
            caps[c["id"]] = c
    captions = {}
    for r in rows:
        c = caps.get(r["id"])
        if c is None or c["prompt"] != r["prompt"] or not c["valid"]:
            raise SystemExit(f"no valid caption for {r['id']} in {args.captions}")
        captions[r["id"]] = c["caption"]
    h, w = args.res
    settings = {"model": "ideogram-ai/ideogram-4-nf4-diffusers", "res": f"{h}x{w}",
                "scales": [int(s) for s in args.scales.split(",")], "steps": args.steps, "guidance_mode": "dual",
                "seed": args.seed, "sigma_bins": args.sigma_bins, "n0": args.n0 or h // 16, "tile": args.tile,
                "window": args.window, "pool": args.pool, "mass_queries": args.mass_queries,
                "prompts": str(args.prompts.resolve()), "captions": str(args.captions.resolve()),
                "caption_sha256": [caps[r["id"]]["caption_sha256"] for r in rows]}
    for name in NAMES:
        d = part_dir(args.out, name)
        d.mkdir(parents=True, exist_ok=True)
        cfg = d / "calib_config.json"
        if cfg.exists() and json.loads(cfg.read_text()) != settings:
            raise SystemExit(f"{cfg} has different settings; choose a new --out")
        if not cfg.exists():
            atomic_json(cfg, settings)
    log = [] if args.report_only else run(args, rows[args.start:end], args.start, settings, captions)
    if args.no_merge:
        return

    from sca.calib import load_parts, merge, transfer
    for name in NAMES:
        d = part_dir(args.out, name)
        paths = [d / f"{image_stem(r['id'], args.seed)}.npz" for r in rows]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} of {len(paths)} parts missing, e.g. {missing[:3]}")
        npz = args.out.parent / f"{args.out.name}_{name}.npz"
        report = merge(load_parts(paths), settings["res"], npz)
        report["settings"] = settings
        report["transformer"] = name
        report["geometry_log"] = log
        others = [o.parent / f"{o.name}_{name}.npz" for o in args.transfer_with]
        report["transfer"] = transfer([npz, *[o for o in others if o.exists()]])
        atomic_json(npz.with_suffix(".json"), report)
        for s, r in report["scales"].items():
            print(f"{name} s={s}", json.dumps(r["rel_err"]), "split_half", r.get("split_half_corr"),
                  "frac_mid", r["frac_pi_mid"], "sigma_star", r["sigma_star"], flush=True)


if __name__ == "__main__":
    main()
