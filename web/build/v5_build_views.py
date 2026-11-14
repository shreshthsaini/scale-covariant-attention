"""Data of the two-views explorer (SCAComp.views, web/assets/js/c_views.js): method step 1, the two native-size views.

The picture is progressive SCA's 4096^2 output for the prompt of the paper's attention figures (Aesthetic-4K
a4k_eval_4096_minus8343371271909676658, seed 0; fig_attn_focus.TARGET), the same image whose 1024^2 stage the heads
and emulation components show. Geometry comes from the code, not typed:
  n0, tile   attn_focus_data.N0 and TILE (FLUX.1's native grid side, 64 tokens, and the inference tile, 8 tokens)
  patch      v6_scaling.PATCH (16 pixels per image token: VAE factor 8, 2 x 2 latent packing)
  window     sca/attn.py window_index: each tile x tile query tile attends to an n0 x n0 key window centred on the
             tile and clamped inside the grid
  grid       sca/attn.py pool_tokens(pool="center") with pool_factors: the keys of the block centres
             kappa(b) = r b + floor(r / 2), b = 0 .. n0 - 1, r = n / n0 (n tokens per side)
Weights (compact mode's two bars): block 12's heads 4 and 7 (the paper's near- and far-reaching examples) in the
r = 2 and r = 4 tables of results/evidence/calib/k1_1024.npz at the noise bin of the paper's method figure
(v6_method_assets.BIN, sigma 0.62).
Images written (web/assets/data/views/img/): base.webp, the 4096^2 picture at 2048 px, which the page draws the
views on and crops the window view from; grid_r<r>.webp for r = 1, 2, 4: what the grid view's keys cover, made by
taking the 16 x 16-pixel patch of every block-centre token of the picture at side 1024 r (the 4096^2 picture,
resized) and laying the n0 x n0 patches side by side (r = 1 is the 1024^2 picture itself).
The sizes 1024^2 and 2048^2 reuse the 4096^2 picture, resized (the boxes and dots are exact for each size; the
pixels at 2048^2 are not that run's own 2048^2 stage). fallback.webp: the picture with the 4096^2 views drawn on it.

Run (repository root, CPU, seconds):
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_build_views.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
REPO = os.path.dirname(WEB)
sys.path.insert(0, os.path.join(REPO, "paper", "figures", "src"))

from PIL import Image, ImageDraw  # noqa: E402

import attn_focus_data as afd  # noqa: E402
import numpy as np  # noqa: E402
import v6_method_assets as ma  # noqa: E402
import fig_attn_focus as faf  # noqa: E402
import v6_scaling as vs  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = os.path.join(WEB, "assets", "data", "views")
SIZES = (1024, 2048, 4096)
QUERY_1024 = tuple(faf.QUERY)   # the query token of the paper's attention figures at 1024^2


def mosaic(img, side, n0, patch):
    """n0 x n0 block-centre patches of img resized to side x side (r = side / (n0 patch))."""
    im = img if img.size[0] == side else img.resize((side, side), Image.LANCZOS)
    n = side // patch
    r = n // n0
    out = Image.new("RGB", (n0 * patch, n0 * patch))
    for bi in range(n0):
        for bj in range(n0):
            ci, cj = r * bi + r // 2, r * bj + r // 2
            out.paste(im.crop((cj * patch, ci * patch, (cj + 1) * patch, (ci + 1) * patch)), (bj * patch, bi * patch))
    return out, r


def window(n, n0, tile, row, col):
    w = min(n0, n)
    def start(p):
        return max(0, min(n - w, (p // tile) * tile + tile // 2 - w // 2))
    return start(row), start(col), w


def main():
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
    n0, tile, patch = afd.N0, afd.TILE, vs.PATCH

    src = Image.open(faf.TARGET).convert("RGB")
    assert src.size == (4096, 4096), src.size
    base = src.resize((2048, 2048), Image.LANCZOS)
    base.save(os.path.join(OUT, "img", "base.webp"), "WEBP", quality=80, method=6)
    grids = {}
    for side in SIZES:
        m, r = mosaic(src, side, n0, patch)
        name = f"img/grid_r{r}.webp"
        m.save(os.path.join(OUT, name), "WEBP", quality=80, method=6)
        grids[str(side)] = name
    # fallback: the 4096^2 views at the default query on the picture (no text)
    fb = src.resize((1200, 1200), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(fb, "RGBA")
    n = 4096 // patch
    q = (QUERY_1024[0] * n // n0 + n // n0 // 2, QUERY_1024[1] * n // n0 + n // n0 // 2)
    r0, c0, w = window(n, n0, tile, *q)
    s = 1200 / n
    r = n // n0
    for bi in range(n0):
        for bj in range(n0):
            cy, cx = (r * bi + r // 2 + .5) * s, (r * bj + r // 2 + .5) * s
            d.ellipse((cx - 1.6, cy - 1.6, cx + 1.6, cy + 1.6), fill=(111, 163, 224, 230))
    d.rectangle((c0 * s, r0 * s, (c0 + w) * s, (r0 + w) * s), outline=(240, 160, 75, 255), width=4)
    fb.save(os.path.join(OUT, "fallback.webp"), "WEBP", quality=80, method=6)
    # weights of the two example heads (block 12, heads 4 and 7) at each ratio, at the noise bin of the method figure
    z = np.load(ma.CALIB)
    weights = {}
    for r in (2, 4):
        t = z[f"s{r}_pi_head"][ma.BLOCK, :, ma.BIN]
        weights[str(r)] = dict(near=round(float(t[ma.NEAR_HEAD]), 3), far=round(float(t[ma.FAR_HEAD]), 3))
    rec = dict(n0=n0, tile=tile, patch=patch, sizes=list(SIZES), image="img/base.webp", image_px=base.size[0],
               heads=dict(block=ma.BLOCK + 1, near=ma.NEAR_HEAD + 1, far=ma.FAR_HEAD + 1,
                          sigma=round(float(z["sigma_centers"][ma.BIN]), 2)), weights=weights,
               grids=grids, query=[q[0] / n, q[1] / n],
               source=dict(script="web/build/v5_build_views.py", picture=str(os.path.relpath(faf.TARGET, "/scratch/09032/saini_2")),
                           geometry="sca/attn.py window_index, pool_tokens(center), pool_factors; attn_focus_data N0, TILE"))
    with open(os.path.join(OUT, "data.js"), "w") as f:
        f.write("/* Generated by web/build/v5_build_views.py; do not edit. */\n")
        f.write("window.SCA_DATA = window.SCA_DATA || {};\n")
        f.write("window.SCA_DATA.views = " + json.dumps(rec, separators=(",", ":")) + ";\n")
    tot = sum(os.path.getsize(os.path.join(p, f)) for p, _, fs in os.walk(OUT) for f in fs)
    print("n0", n0, "tile", tile, "patch", patch, "grids", grids, "query", rec["query"], f"data {tot / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
