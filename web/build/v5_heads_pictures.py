#!/usr/bin/env python3
"""Two plain pictures for "How it works", step 1: where a near-reaching and a far-reaching head look.

Data (read only): paper/figures/src/data/v6_attn_block12.npz (full attention of block 12, all 24 heads, at one query
of the 1024^2 sheep image, three noise levels; written by paper/figures/src/v6_attn_extract.py) and the image
paper/figures/attn_focus/image_1024.png. The two heads are the paper's (\\hlFocusHeadNear, \\hlFocusHeadFar from
paper/figures/snippets/attn_focus_numbers.tex) at its noise level (\\hlFocusSigma). Each token's attention weight is
drawn over the image, orange within 16 tokens of the query (the near set the paper counts) and blue beyond, with
opacity from the weight (square root, normalised to the head's own 99.5th percentile), a ring at the query and a
thin outline of the 16-token neighbourhood. No labels are burned in; the page writes them.

Writes web/assets/img/v5/head_near.webp and head_far.webp (900 x 900).
Usage: /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_heads_pictures.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
NPZ = REPO / "paper/figures/src/data/v6_attn_block12.npz"
IMG = REPO / "paper/figures/attn_focus/image_1024.png"
MACROS = REPO / "paper/figures/snippets/attn_focus_numbers.tex"
OUT = REPO / "web/assets/img/v5"
SIZE, RADIUS = 900, 16
NEAR, FAR = (240, 160, 75), (111, 163, 224)    # design.md dark-theme near and far (media sits on the screen ground)


def macro(name: str) -> str:
    return re.search(r"\\newcommand\{\\" + name + r"\}\{([^}]*)\}", MACROS.read_text()).group(1)


def picture(att: np.ndarray, base: Image.Image, q: tuple[int, int]) -> Image.Image:
    n = att.shape[0]
    cell = SIZE / n
    a = np.sqrt(np.clip(att / np.percentile(att, 99.5), 0, 1))
    rr, cc = np.mgrid[0:n, 0:n]
    near = np.maximum(np.abs(rr - q[0]), np.abs(cc - q[1])) <= RADIUS
    rgb = np.zeros((n, n, 3), np.float32)
    rgb[near] = NEAR
    rgb[~near] = FAR
    alpha = (a * 0.92)[..., None]
    over = Image.fromarray(np.concatenate([rgb, alpha * 255], -1).astype(np.uint8), "RGBA").resize((SIZE, SIZE), Image.NEAREST)
    g = base.convert("L").resize((SIZE, SIZE), Image.LANCZOS)
    g = Image.eval(g, lambda v: int(v * 0.42))
    im = Image.merge("RGB", (g, g, g)).convert("RGBA")
    im.alpha_composite(over)
    d = ImageDraw.Draw(im)
    x0, y0 = (q[1] - RADIUS) * cell, (q[0] - RADIUS) * cell
    x1, y1 = (q[1] + RADIUS + 1) * cell, (q[0] + RADIUS + 1) * cell
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(SIZE - 1, x1), min(SIZE - 1, y1)
    for i in range(int(x0), int(x1), 14):          # dashed outline of the near neighbourhood
        d.line([(i, y0), (min(i + 7, x1), y0)], fill=NEAR + (220,), width=2)
        d.line([(i, y1), (min(i + 7, x1), y1)], fill=NEAR + (220,), width=2)
    for j in range(int(y0), int(y1), 14):
        d.line([(x0, j), (x0, min(j + 7, y1))], fill=NEAR + (220,), width=2)
        d.line([(x1, j), (x1, min(j + 7, y1))], fill=NEAR + (220,), width=2)
    cx, cy = (q[1] + .5) * cell, (q[0] + .5) * cell
    d.ellipse([cx - 11, cy - 11, cx + 11, cy + 11], outline=(255, 255, 255, 255), width=3)
    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 255, 255, 255))
    return im.convert("RGB")


def main() -> None:
    z = np.load(NPZ, allow_pickle=True)
    sig = float(macro("hlFocusSigma"))
    k = int(np.argmin(np.abs(z["sigma"] - sig)))
    q = (int(z["query"][0]), int(z["query"][1]))
    base = Image.open(IMG).convert("RGB")
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, head in (("near", int(macro("hlFocusHeadNear"))), ("far", int(macro("hlFocusHeadFar")))):
        im = picture(z["full"][k, head], base, q)
        path = OUT / f"head_{tag}.webp"
        im.save(path, "WEBP", quality=84, method=6)
        print(f"{path.relative_to(REPO)}: head {head}, sigma {z['sigma'][k]:.2f}, query {q}, {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
