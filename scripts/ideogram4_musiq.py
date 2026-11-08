#!/usr/bin/env python3
"""MUSIQ for 8192^2 images: MUSIQ_c exactly as constancy_rope's eval.py (mean over four fixed 1024^2 native-pixel crops,
crop_boxes(size, prompt_id, seed 0)), and full-image MUSIQ on a 4096 px Lanczos downscale (musiq_4096).

Why not eval.py's full-image MUSIQ: pyiqa 0.1.16's MUSIQ keeps every native-resolution 32 px patch
(max_seq_len_from_original_res = -1) and materializes its full attention matrix; at 8192^2 that is ~65.7k tokens and
~100 GB of scores, beyond one 96 GB GPU. Same pyiqa model, same tensor conversion (crope.metrics.models). Output: a
per-image CSV with the columns of eval.py's per_image.csv (method, res, prompt_id, seed, path, metrics), resumable
through a JSONL cache keyed by PNG SHA256."""

import argparse
import csv
import glob
import hashlib
import json
import os
from pathlib import Path
import sys

DEFAULT_CROPE = "/work/09032/saini_2/vista/Projects/Exploration/constancy_rope"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="Eval root: <method>/<HxW>/<stem>.png + .json")
    p.add_argument("--out", type=Path, required=True, help="Per-image CSV")
    p.add_argument("--long-side", type=int, default=4096)
    p.add_argument("--crops", type=int, default=4)
    p.add_argument("--crop-seed", type=int, default=0)
    p.add_argument("--crope-root", default=DEFAULT_CROPE)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    sys.path.insert(0, args.crope_root)
    import torch
    from PIL import Image
    from crope.metrics.image import crop_boxes
    from crope.metrics.models import PyiqaScorer, tensor_rgb
    Image.MAX_IMAGE_PIXELS = None
    cache_path = args.out.with_suffix(".cache.jsonl")
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["sha256"]] = r
    rows, model = [], None
    for meta in sorted(glob.glob(str(args.root / "*" / "*" / "*.json"))):
        m = json.load(open(meta))
        png = meta[:-5] + ".png"
        if not m.get("complete") or not os.path.exists(png):
            continue
        c = m["config"]
        key = sha(png)
        if key not in cache:
            if model is None:
                model = PyiqaScorer("musiq", args.device).model
            with Image.open(png) as src, torch.inference_mode():
                img = src.convert("RGB")
                w, h = img.size
                s = args.long_side / max(w, h)
                small = img.resize((round(w * s), round(h * s)), Image.LANCZOS) if s < 1 else img
                full = float(model(tensor_rgb(small).to(args.device)).mean().item())
                boxes = crop_boxes(img.size, c["prompt_id"], seed=args.crop_seed, count=args.crops)
                crops = [float(model(tensor_rgb(img.crop(b)).to(args.device)).mean().item()) for b in boxes]
            cache[key] = {"sha256": key, f"musiq_{args.long_side}": full, "musiq_crop1024": sum(crops) / len(crops),
                          "crop_boxes": boxes}
            with open(cache_path, "a") as fh:
                fh.write(json.dumps(cache[key]) + "\n")
            print("musiq", c["method"], c["prompt_id"], round(full, 3), round(cache[key]["musiq_crop1024"], 3),
                  flush=True)
        r = cache[key]
        rows.append({"method": c["method"], "res": c["res"], "prompt_id": c["prompt_id"], "seed": c["seed"],
                     "path": png, f"musiq_{args.long_side}": r[f"musiq_{args.long_side}"],
                     "musiq_crop1024": r["musiq_crop1024"]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "res", "prompt_id", "seed", "path", f"musiq_{args.long_side}",
                                           "musiq_crop1024"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
