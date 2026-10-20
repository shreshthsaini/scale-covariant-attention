"""Data and rasters for the appendix method zoom (paper/figures/src/method_zoom.tex).

Writes, under paper/figures/:
  method_zoom/pi_strip.png    panel (d): s4_pi_head, 1368 head columns x 14 noise bins
  method_zoom/block_mask.png  panel (b): exact local block mask of the drawn toy grid
  method_zoom_provenance.json
and prints the TeX data macros of panel (c) (example head, sampling schedule) plus
the statistics quoted in the report.

Run from the repo root with the sca env:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I gen_method_zoom.py
"""

import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = "/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention"
FIG = os.path.join(ROOT, "paper", "figures")
SRC = os.path.join(FIG, "src")
OUT = os.path.join(FIG, "method_zoom")
sys.path.insert(0, SRC)
import scafig as sf  # noqa: E402  (PI_CMAP, the paper-wide pi colormap)

DATA = os.path.join(SRC, "data", "k1_1024_pi.npz")
KEY = "s4_pi_head"
SIGMA_G = 0.89
N_DOUBLE = 19
EX_BLOCK, EX_HEAD = 29, 18          # one-based; the block of the main diagram's panel (b)
HX, VY = 4, 20                      # raster upsampling: pixels per head column, per bin row
TOY = dict(H=24, s=4, n0=6, t=2, w=6, text_cols=6)  # the grid drawn in panels (a) and (b)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lookup(pi, centers, sigma):
    """sca/attn.py PiTable.__call__: linear interpolation between bin centres, clamped."""
    j = np.searchsorted(centers, sigma)
    if j <= 0:
        return pi[..., 0]
    if j >= len(centers):
        return pi[..., -1]
    a = (sigma - centers[j - 1]) / (centers[j] - centers[j - 1])
    return (1 - a) * pi[..., j - 1] + a * pi[..., j]


def rgb(values):
    return (np.asarray(sf.PI_CMAP(np.clip(values, 0, 1)))[..., :3] * 255 + 0.5).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    z = np.load(DATA)
    pi, centers, pop = z[KEY], z["sigma_centers"], z["populated"].astype(bool)
    L, Hh, B = pi.shape
    assert (L, Hh, B) == (57, 24, 14)
    assert np.allclose(pi[..., 0], pi[..., 1]) and not pop[0] and pop[1:].all()
    with open(os.path.join(FIG, "method_provenance.json")) as fh:
        sched = np.array(json.load(fh)["schedule"]["sigma"])
    post = sched[sched <= SIGMA_G]
    applied = np.stack([lookup(pi, centers, s) for s in post], -1)       # [57, 24, 20]
    mean_applied = applied.mean(-1)                                      # [57, 24]

    # ---- panel (d): heads sorted within each block by their mean applied pi (descending)
    order = np.argsort(-mean_applied, axis=1, kind="stable")             # [57, 24]
    cols = np.take_along_axis(pi, order[:, :, None], axis=1).reshape(L * Hh, B)   # [1368, 14]
    img = rgb(cols.T[::-1])                                               # rows: bin 13 (top) .. bin 0
    img = np.repeat(np.repeat(img, VY, axis=0), HX, axis=1)
    strip = os.path.join(OUT, "pi_strip.png")
    Image.fromarray(img).save(strip, optimize=True)

    # ---- panel (b): exact block mask of the toy local operator, tile-major order
    H, t, w, tc = TOY["H"], TOY["t"], TOY["w"], TOY["text_cols"]
    nt = H // t
    a = np.clip(np.arange(nt) * t + t // 2 - w // 2, 0, H - w)          # window start per tile index
    m = np.zeros((nt * nt, nt * nt))
    for qy in range(nt):
        for qx in range(nt):
            for ky in range(nt):
                for kx in range(nt):
                    oy = max(0, min(a[qy] + w, ky * t + t) - max(a[qy], ky * t))
                    ox = max(0, min(a[qx] + w, kx * t + t) - max(a[qx], kx * t))
                    m[qy * nt + qx, ky * nt + kx] = oy * ox / (t * t)
    full, part = (m == 1).sum(), ((m > 0) & (m < 1)).sum()
    white = np.array([255, 255, 255], np.uint8)
    loc = np.array([int(sf.LOCAL[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
    txt = np.array([int(sf.MID[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
    glyph = np.empty((nt * nt, tc + nt * nt, 3), np.uint8)
    glyph[:, :tc] = txt
    glyph[:, tc:] = white
    glyph[:, tc:][m > 0] = loc
    glyph = np.repeat(np.repeat(glyph, 4, axis=0), 4, axis=1)
    mask = os.path.join(OUT, "block_mask.png")
    Image.fromarray(glyph).save(mask, optimize=True)

    # ---- panel (c): example head and the sampling schedule, as TeX macros
    l, h = EX_BLOCK - 1, EX_HEAD - 1
    ex = pi[l, h]
    print("% ---- data written by gen_method_zoom.py from src/data/k1_1024_pi.npz ----")
    print("\\def\\exbins{" + ",".join("%d/%.4f/%.4f" % (k, centers[k], ex[k]) for k in range(B) if pop[k]) + "}")
    print("\\def\\exlow{%.4f}  \\def\\exhigh{%.4f}" % (ex[1], ex[13]))
    print("\\def\\steps{" + ",".join("%.4f" % s for s in sched) + "}")

    # ---- statistics
    popi = pi[..., pop]
    stats = {
        "pi_min_populated": float(popi.min()), "pi_max_populated": float(popi.max()),
        "share_cells_pi_eq_0": float((popi == 0).mean()), "share_cells_pi_eq_1": float((popi == 1).mean()),
        "mean_pi_applied_post_gate": {
            "all": float(mean_applied.mean()),
            "double_stream_blocks_1_19": float(mean_applied[:N_DOUBLE].mean()),
            "single_stream_blocks_20_57": float(mean_applied[N_DOUBLE:].mean())},
        "mean_pi_populated_bins": {
            "all": float(popi.mean()), "double": float(popi[:N_DOUBLE].mean()), "single": float(popi[N_DOUBLE:].mean())},
        "block_mean_applied_min": [int(mean_applied.mean(1).argmin() + 1), float(mean_applied.mean(1).min())],
        "block_mean_applied_max": [int(mean_applied.mean(1).argmax() + 1), float(mean_applied.mean(1).max())],
        "share_heads_mean_applied_gt_0p5": float((mean_applied > 0.5).mean()),
        "applied_mean_first_and_last_post_gate_step": [float(applied[..., 0].mean()), float(applied[..., -1].mean())],
    }
    print("% stats", json.dumps(stats, indent=1), file=sys.stderr)
    print("toy block mask: full blocks %d, partial %d, of %d" % (full, part, m.size), file=sys.stderr)

    prov = {
        "figure": "paper/figures/method_zoom.pdf",
        "tikz": "paper/figures/src/method_zoom.tex",
        "generator": "gen_method_zoom.py (session scratch; recipe below is complete)",
        "data": "paper/figures/src/data/k1_1024_pi.npz",
        "data_sha256": sha256(DATA),
        "key": KEY,
        "table": "scale-4 per-head table used at inference: pi[block, head, bin], 57 x 24 x 14, "
                 "blocks one-based 1..19 double-stream then 20..57 single-stream",
        "bins": {
            "count": 14, "width": "1/14 in sigma", "index_0": "sigma in [0, 1/14), unpopulated, copies bin 1 (as stored)",
            "sigma_centers": [round(float(c), 6) for c in centers],
            "populated": [bool(p) for p in pop],
            "lookup": "sca/attn.py PiTable.__call__: linear interpolation between the 14 centres, clamped at the ends",
        },
        "gate": {"sigma_g": SIGMA_G, "global_only_steps": [int(i + 1) for i in np.where(sched > SIGMA_G)[0]],
                 "schedule_sigma_28_steps": [round(float(s), 4) for s in sched],
                 "schedule_source": "paper/figures/method_provenance.json (FLUX.1 shifted schedule, mu = 1.15)"},
        "panel_d": {
            "raster": "paper/figures/method_zoom/pi_strip.png", "raster_sha256": sha256(strip),
            "rows": "top to bottom: bin 13 (sigma near 1) to bin 0 (sigma near 0), equal height, so sigma is linear",
            "columns": "57 blocks left to right, 24 heads per block; within a block heads are sorted by their mean "
                       "applied pi over the 20 post-gate steps, descending (stable)",
            "head_order_one_based": (order + 1).tolist(),
            "upsampling": "nearest, %d px per head column, %d px per bin row" % (HX, VY),
            "colormap": "scafig.PI_CMAP: 0 #1F4F8A, 0.22 #2F6DB3, 0.5 #F4F1EA, 0.78 #E08A2B, 1 #9C5512 (linear RGB)",
            "veil": "sigma > sigma_g = 0.89 (global only at inference)",
        },
        "panel_c": {"example": "block %d, head %d (one-based)" % (EX_BLOCK, EX_HEAD),
                    "pi_by_bin": [round(float(v), 4) for v in ex]},
        "panel_b": {"raster": "paper/figures/method_zoom/block_mask.png", "raster_sha256": sha256(mask),
                    "toy": TOY, "window_start_per_tile_index": a.tolist(),
                    "rows": "query tiles (t x t), tile-major order", "cols": "%d text columns, then key tiles in tile-major order" % tc,
                    "full_blocks": int(full), "partial_blocks": int(part)},
        "panel_a": {"toy": TOY, "block_centres": [int(TOY["s"] * b + TOY["s"] // 2) for b in range(TOY["n0"])]},
        "image": "paper/figures/teaser/square_bright.jpg (brightened copy of teaser/square.jpg)",
        "stats": stats,
    }
    with open(os.path.join(FIG, "method_zoom_provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=1)


if __name__ == "__main__":
    main()
