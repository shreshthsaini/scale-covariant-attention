#!/usr/bin/env python3
"""SCA calibration for Qwen-Image (same statistics and tables as scripts/calibrate.py): run Qwen-Image with exact
attention near its native resolution, emulate scales s_c (window n/s_c, pooling s_c), store one part per prompt
(resumable), then merge into a pi table npz plus a JSON report (same path, .json).

The native 1328^2 grid (83 x 83 tokens) is prime, so ratio-2 and ratio-4 pooling cannot tile it; the default --res is
1344^2 (84 x 84 tokens), the nearest size where both do. Sampling follows the model card (true CFG 4.0, negative
prompt " ", 50 steps, mu of the native sequence); both transformer calls of a step add statistics, as both run SCA at
inference. --start/--end split the first --count prompts across tasks; run with --no-merge per split and once with
--report-only after all parts exist. Prompt embeddings are precomputed and cached (sca/qwen.py)."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sca.runtime import atomic_json, image_seed, image_stem, read_prompts, resolution, runtime_environment  # noqa: E402


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, default=Path(__file__).with_name("calib_prompts.jsonl"))
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--start", type=int, default=0, help="First prompt index (within --count) of this split")
    p.add_argument("--end", type=int, help="End prompt index of this split (default --count)")
    p.add_argument("--res", type=resolution, default=(1344, 1344), metavar="HxW")
    p.add_argument("--scales", default="2,4")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--true-cfg", type=float, default=4.0)
    p.add_argument("--negative", default=" ")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma-bins", type=int, default=14)
    p.add_argument("--mu-mode", choices=("native", "default"), default="native")
    p.add_argument("--n0", type=int, default=84, help="Calibration grid side (tile scaling t * Wc / Wn)")
    p.add_argument("--tile", type=int, default=8, help="Inference tile t; emulation uses t * Wc / Wn")
    p.add_argument("--window", type=int, help="Inference window Wn (default n0)")
    p.add_argument("--pool", choices=("center", "mean"), default="center")
    p.add_argument("--local-impl", choices=("flex", "gather"), default="flex")
    p.add_argument("--mass-queries", type=int, default=256)
    p.add_argument("--out", type=Path, required=True, help="pi table .npz; parts go to <out stem>_parts/")
    p.add_argument("--no-merge", action="store_true", help="Only write this split's parts")
    p.add_argument("--report-only", action="store_true", help="Merge existing parts without running Qwen-Image")
    p.add_argument("--transfer-with", type=Path, nargs="*", default=[], help="Other calibration npz files")
    p.add_argument("--device", default="cuda")
    return p


def run(args, rows, offset, parts, settings):
    import numpy as np
    import torch
    from sca.calib import CalibStats
    from sca.qwen import QwenSCAContext, encode_prompts, install, load_pipeline, set_mu
    todo = [(j, r) for j, r in enumerate(rows, offset) if not (parts / f"{image_stem(r['id'], args.seed)}.npz").exists()]
    if not todo:
        print("all parts present", flush=True)
        return []
    cfg_on = args.true_cfg > 1
    enc = encode_prompts([r["prompt"] for _, r in todo] + ([args.negative] if cfg_on else []), args.device)
    neg = enc.pop() if cfg_on else None
    pipe = load_pipeline(args.device)
    ctx = QwenSCAContext(mode="calib", n0=args.n0, tile=args.tile, window=args.window, pool=args.pool,
                         local_impl=args.local_impl, calib_scales=tuple(settings["scales"]))
    uninstall = install(pipe.transformer, ctx)
    ctx.scheduler = pipe.scheduler
    set_mu(args.mu_mode)
    h, w = args.res
    heads = pipe.transformer.config.num_attention_heads
    try:
        for (j, row), emb in zip(todo, enc):
            part = parts / f"{image_stem(row['id'], args.seed)}.npz"
            ctx.calib = CalibStats(ctx.n_layers, heads, ctx.calib_scales, args.sigma_bins,
                                   mass_queries=args.mass_queries, device=args.device, seed=j)
            generator = torch.Generator(device="cpu").manual_seed(image_seed(row["id"], args.seed))
            kw = {"negative_prompt_embeds": neg.to(args.device)} if cfg_on else {}
            with torch.inference_mode():
                pipe(prompt=None, prompt_embeds=emb.to(args.device), height=h, width=w, num_inference_steps=args.steps,
                     true_cfg_scale=args.true_cfg, generator=generator, output_type="latent", **kw)
            tmp = parts / f"{part.stem}.tmp{os.getpid()}.npz"
            np.savez(tmp, prompt_index=np.array(j), **ctx.calib.state())
            os.replace(tmp, part)
            print(f"calibrated {row['id']} ({j + 1}/{args.count})", flush=True)
    finally:
        uninstall()
    return ctx.log


def main(argv=None):
    args = parser().parse_args(argv)
    runtime_environment()
    rows = read_prompts(args.prompts)[: args.count]
    end = len(rows) if args.end is None else min(args.end, len(rows))
    parts = args.out.parent / f"{args.out.stem}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    settings = {"model": "Qwen/Qwen-Image", "res": f"{args.res[0]}x{args.res[1]}",
                "scales": [int(s) for s in args.scales.split(",")], "steps": args.steps, "true_cfg": args.true_cfg,
                "negative": args.negative, "seed": args.seed, "sigma_bins": args.sigma_bins, "mu_mode": args.mu_mode,
                "n0": args.n0, "tile": args.tile, "window": args.window, "pool": args.pool,
                "mass_queries": args.mass_queries, "prompts": str(args.prompts.resolve())}
    cfg = parts / "calib_config.json"
    if cfg.exists() and json.loads(cfg.read_text()) != settings:
        raise SystemExit(f"{cfg} has different settings; choose a new --out")
    if not cfg.exists():
        atomic_json(cfg, settings)
    log = [] if args.report_only else run(args, rows[args.start:end], args.start, parts, settings)
    if args.no_merge:
        return

    from sca.calib import load_parts, merge, transfer
    paths = [parts / f"{image_stem(r['id'], args.seed)}.npz" for r in rows]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(paths)} calibration parts missing, e.g. {missing[:3]}")
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
