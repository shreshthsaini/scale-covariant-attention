#!/usr/bin/env python3
"""Generate one resumable Ideogram 4 chunk with full attention, NTK, YaRN, SEGA or SCA. Output (PNG + JSON sidecar per
prompt, deterministic per-prompt seeds) has the format of scripts/sample.py, so the evaluation scripts run unchanged.

Every method runs the pinned SEGA copy of the pipeline and transformer (sca/ideogram4.py) with the model's defaults:
48 steps, the guidance schedule 7.0 x 45 then 3.0 x 3, asymmetric CFG with the unconditional transformer ("dual"), the
resolution-aware logit-normal schedule, bf16 compute on the nf4 checkpoint. The conditioning is the structured JSON
caption that scripts/ideogram4_captions.py wrote once per prompt (--captions); the sidecar's prompt is the user prompt
(what the metrics score), its caption field the text the model saw. ntk / yarn / sega are run_ideogram4.py's
--rope-scaling methods with --train-grid (default 256 = 4096 px); sca installs SCA on both transformers, each with its
own reach table (--pi-table: the conditional transformer's npz; --pi-table-uncond: the unconditional one's)."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sca import dedup  # noqa: E402
from sca.runtime import (atomic_json, code_revision, file_sha256, fingerprint, image_seed, image_stem,  # noqa: E402
                         implementation_hash, read_prompts, resolution, runtime_environment)

VARIANT = {"full": None, "ntk": None, "yarn": None, "sega": None, "sca": "head", "sca_uniform": "uniform",
           "sca_layer": "layer"}
ROPE = {"ntk", "yarn", "sega"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=tuple(VARIANT), default="full")
    p.add_argument("--label", help="Method name written to the sidecar (default --method)")
    p.add_argument("--pi-table", type=Path, help="Reach table npz of the conditional transformer (sca variants)")
    p.add_argument("--pi-table-uncond", type=Path, help="Reach table npz of the unconditional transformer")
    p.add_argument("--pi-scale", type=int, help="Emulated scale to use (default: nearest to the target ratio)")
    p.add_argument("--pi-const", type=float, help="Constant pi for every head and sigma (identity checks, ablation)")
    p.add_argument("--res", type=resolution, default=(4096, 4096), metavar="HxW")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--captions", type=Path, help="Caption JSONL from scripts/ideogram4_captions.py (required unless "
                   "--plain-prompts)")
    p.add_argument("--plain-prompts", action="store_true", help="Condition on the user prompt itself (checks only)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--ids", nargs="*", help="Only these prompt ids (in file order)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--train-grid", type=int, default=256, help="ntk / yarn / sega: trained grid extent in tokens")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n0", type=int, default=256)
    p.add_argument("--tile", type=int, default=8)
    p.add_argument("--window", type=int)
    p.add_argument("--pool", choices=("center", "mean"), default="center")
    p.add_argument("--gate-sigma", type=float, default=0.0, help="Global-only (pi=0) above this sigma")
    p.add_argument("--route", type=float, default=0.0, help="Hard per-head routing threshold on pi (0 = soft mix)")
    p.add_argument("--chunk", type=int, help="Query tiles per local-branch chunk, gather only")
    p.add_argument("--local-impl", choices=("flex", "gather"), default="flex")
    p.add_argument("--flex-kernel", type=json.loads, default={}, help="flex kernel options as JSON ({} = torch defaults)")
    dedup.add_args(p)  # bridge, window schedule, matched weights (sca_v2 options); all off by default
    p.add_argument("--vae-tiling", choices=("auto", "on", "off"), default="auto",
                   help="auto: tiled VAE decode at or above 2048 x 4096 (as scripts/sample_flux2.py)")
    p.add_argument("--device", default="cuda")
    return p


def build_context(args, grid, table):
    from sca.attn import PiTable, pool_factors
    from sca.ideogram4 import Ideogram4SCAContext
    variant = VARIANT[args.method]
    if variant is None:
        return None, {}
    if args.anchor_lo:
        raise SystemExit("--anchor-* works in scripts/sample.py (FLUX.1) only")
    target = max(pool_factors(grid, args.n0))
    if args.pi_const is not None:
        pi = PiTable("const", value=args.pi_const)
    elif table is None:
        raise SystemExit("--pi-table and --pi-table-uncond are required for this method")
    else:
        pi = PiTable.load(table, variant, args.pi_scale, None, target)
    ctx = Ideogram4SCAContext(mode="sca", known_grid=grid, n0=args.n0, tile=args.tile, window=args.window,
                              pool=args.pool, chunk=args.chunk, local_impl=args.local_impl, pi=pi,
                              gate_sigma=args.gate_sigma, route=args.route, **dedup.context_kwargs(args))
    ctx.flex_kernel = dict(args.flex_kernel)
    info = {"n0": args.n0, "tile": args.tile, "window": args.window or args.n0, "pool": args.pool,
            "chunk": args.chunk, "gate_sigma": args.gate_sigma, "band": 0, "route": args.route,
            "pool_factors": list(pool_factors(grid, args.n0)), "pi_variant": pi.variant,
            "pi_const": args.pi_const, "pi_scale": getattr(pi, "scale", None), "flex_kernel": ctx.flex_kernel,
            "pi_table": str(table) if args.pi_const is None and table else None,
            "pi_table_sha256": file_sha256(table) if args.pi_const is None and table else None}
    info.update(dedup.info(args))
    return ctx, info


def captions_for(args, rows):
    if args.plain_prompts:
        return {r["id"]: r["prompt"] for r in rows}, None
    if args.captions is None:
        raise SystemExit("--captions is required (or --plain-prompts for checks)")
    caps = {}
    for line in args.captions.read_text().splitlines():
        if line.strip():
            c = json.loads(line)
            caps[c["id"]] = c
    out = {}
    for r in rows:
        c = caps.get(r["id"])
        if c is None or c["prompt"] != r["prompt"] or not c["valid"]:
            raise SystemExit(f"no valid caption for {r['id']} in {args.captions}")
        out[r["id"]] = c["caption"]
    return out, caps


def main(argv=None):
    args = parser().parse_args(argv)
    runtime_environment()
    prompts = read_prompts(args.prompts)
    end = len(prompts) if args.end is None else args.end
    if not 0 <= args.start < end <= len(prompts) or args.steps <= 0:
        raise ValueError("Require 0 <= start < end <= number of prompts and positive steps")
    rows = prompts[args.start:end]
    if args.ids:
        rows = [r for r in rows if r["id"] in set(args.ids)]
        if len(rows) != len(set(args.ids)):
            raise SystemExit("some --ids are not in the prompt range")
    import diffusers
    import torch
    from sca.ideogram4 import (GUIDANCE_SCHEDULE, REPO, install, load_pipeline, set_rope_scaling, snapshot,
                               transformers_of)
    h, w = args.res
    grid = (h // 16, w // 16)
    label = args.label or args.method
    texts, cap_rows = captions_for(args, rows)
    contexts = {"cond": build_context(args, grid, args.pi_table), "uncond": build_context(args, grid, args.pi_table_uncond)}

    def paths(row):
        stem = image_stem(row["id"], args.seed)
        return args.out / f"{stem}.png", args.out / f"{stem}.json"

    def done(row):
        png, meta = paths(row)
        return png.exists() and meta.exists() and json.loads(meta.read_text()).get("complete")
    if all(done(r) for r in rows):
        print(f"all {len(rows)} images present in {args.out}", flush=True)
        return
    t_load = time.perf_counter()
    pipe = load_pipeline(args.device)
    load_seconds = time.perf_counter() - t_load
    if args.method in ROPE:
        set_rope_scaling(pipe, args.method, args.train_grid)
    uninstall = []
    for name, tf in transformers_of(pipe):
        ctx = contexts[name][0]
        if ctx is not None:
            uninstall.append(install(tf, ctx, pipe.scheduler))
    tiling = args.vae_tiling == "on" or (args.vae_tiling == "auto" and h * w >= 2048 * 4096)
    if tiling:
        pipe.vae.enable_tiling()
    schedule = GUIDANCE_SCHEDULE if args.steps == len(GUIDANCE_SCHEDULE) else None
    revision, impl = code_revision(), implementation_hash()
    try:
        for row in rows:
            png, meta = paths(row)
            cap = texts[row["id"]]
            config = {"schema_version": 1, "method": label, "res": f"{h}x{w}", "prompt_id": str(row["id"]),
                      "prompt": row["prompt"], "seed": int(args.seed), "derived_seed": image_seed(row["id"], args.seed),
                      "steps": args.steps, "guidance_schedule": list(schedule) if schedule else [7.0] * args.steps,
                      "guidance_mode": "dual", "model": REPO, "snapshot": snapshot().name, "dtype": "bfloat16 on nf4",
                      "batch_size": 1, "attention": f"ideogram4:{args.method}", "diffusers": diffusers.__version__,
                      "torch": torch.__version__, "implementation_sha256": impl, "grid": list(grid),
                      "conditioning": "plain prompt" if args.plain_prompts else "upsampled JSON caption",
                      "caption_sha256": None if cap_rows is None else cap_rows[row["id"]]["caption_sha256"],
                      "rope_scaling": ({"method": args.method, "train_grid": args.train_grid}
                                       if args.method in ROPE else None),
                      "sca": {k: v[1] for k, v in contexts.items()} if VARIANT[args.method] else {},
                      "vae_tiling": tiling, "pipeline": "third_party/sega/ideogram4_sega (pinned SEGA copy)",
                      "scheduler": dict(pipe.scheduler.config)}
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
            for name, (ctx, _) in contexts.items():
                if ctx is not None:
                    ctx.pi_trace.clear()
            generator = torch.Generator(device="cpu").manual_seed(image_seed(row["id"], args.seed))
            cuda = args.device.startswith("cuda")
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            kw = {"guidance_schedule": schedule} if schedule else {"guidance_scale": 7.0, "guidance_schedule": None}
            start = time.perf_counter()
            with torch.inference_mode():
                image = pipe(prompt=cap, height=h, width=w, num_inference_steps=args.steps, guidance_mode="dual",
                             generator=generator, prompt_upsampling=False, **kw).images[0]
            if cuda:
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            args.out.mkdir(parents=True, exist_ok=True)
            tmp = png.with_name(png.name + f".tmp.{os.getpid()}")
            image.save(tmp, format="PNG")
            os.replace(tmp, png)
            atomic_json(meta, {"complete": True, "config": config, "config_fingerprint": digest,
                               "code_revision": revision, "image": str(png.resolve()), "png_sha256": file_sha256(png),
                               "caption": cap, "seconds": seconds,
                               "seconds_scope": "text encoding, 48 steps of both transformers and VAE decode; caption "
                                                "upsampling precomputed; excludes model loading and PNG I/O",
                               "load_seconds": load_seconds,
                               "peak_memory_bytes": torch.cuda.max_memory_allocated() if cuda else None,
                               "gpu": torch.cuda.get_device_name() if cuda else None,
                               "sigmas": [float(s) for s in pipe.scheduler.sigmas[:-1]],
                               "pi_trace": {k: list(v[0].pi_trace) for k, v in contexts.items() if v[0] is not None},
                               "sca_log": sorted({m for v in contexts.values() if v[0] is not None for m in v[0].log}),
                               "local_impl": args.local_impl})
            print(f"saved {row['id']} {label} {config['res']} seconds={seconds:.1f} "
                  f"peak_gb={(torch.cuda.max_memory_allocated() / 1e9) if cuda else 0:.1f}", flush=True)
    finally:
        for undo in uninstall:
            undo()


if __name__ == "__main__":
    main()
