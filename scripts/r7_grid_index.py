#!/usr/bin/env python3
"""Token-checkerboard index (round 7): spectral energy of an image at the token period (16 px: FLUX's 2 x 2 latent
patch of 8 px VAE cells) relative to the energy in a ring of neighbouring frequencies.

For each image: luminance of the central 2048 x 2048 px, mean removed, Hann window, 2-D power spectrum; the index is
log10 of the mean power at the bins (0, k), (k, 0), (k, k), (k, -k), k = N / period, over the median power of the
bins at Chebyshev distance 3 to 6 from (k, k). VAE decodes carry some token-period energy (sca_v2 about 1.8 on a
flat illustration); a token checkerboard adds orders of magnitude.
Usage: r7_grid_index.py --dirs label=DIR [...] --ids ids.jsonl --out out.csv
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
from PIL import Image


def index(png, period=16):
    Image.MAX_IMAGE_PIXELS = None
    im = np.asarray(Image.open(png).convert("L"), dtype=np.float32)
    c0 = im.shape[0] // 2 - 1024
    im = im[c0:c0 + 2048, c0:c0 + 2048]
    n = im.shape[0]
    im = im - im.mean()
    w = np.hanning(n).astype(np.float32)
    P = np.abs(np.fft.fft2(im * w[:, None] * w[None, :])) ** 2
    k = n // period
    vals = [P[0, k], P[k, 0], P[k, k], P[k, -k]]
    ring = np.median([P[(k + dy) % n, (k + dx) % n] for dy in range(-6, 7) for dx in range(-6, 7)
                      if max(abs(dy), abs(dx)) >= 3])
    return float(np.log10(np.mean(vals) / (ring + 1e-12)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    ids = [json.loads(l)["id"] for l in open(a.ids) if l.strip()]
    rows = []
    for item in a.dirs:
        lab, d = item.split("=", 1)
        vals = []
        for i in ids:
            f = glob.glob(f"{d}/{glob.escape(i)}_*seed0.png")
            if f:
                v = index(f[0])
                vals.append(v)
                rows.append({"label": lab, "prompt_id": i, "grid_index": round(v, 4)})
        v = np.array(vals)
        print(f"{lab}: n {len(v)} median {np.median(v):.3f} mean {v.mean():.3f} p90 {np.quantile(v, 0.9):.3f}")
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "prompt_id", "grid_index"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
