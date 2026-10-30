"""Data and rasters for the appendix method zoom (paper/figures/src/method_zoom.tex, Figure D.1).

Writes, under paper/figures/:
  method_zoom/pi_strip.png    panel (d): s4_pi_head, 1368 head columns x 14 noise bins
  method_zoom/block_mask.png  panel (b): block mask of the drawn toy local operator (text, bridge, window)
  method_zoom/data.tex        TeX macros of panels (b) and (c): bridge dots, example head, sampling schedule
  method_zoom_provenance.json
and prints the statistics quoted in the caption.

Operator drawn (sca_v2, sca/attn.py): global branch as published; local softmax = text keys + bridge set B
(sca_dedup_forward, bridge_index; flex_layout built for the prefix L + R) + the window of side w(sigma)
(SCAContext.window_now: 128, 112, 96, 80 tokens while sigma > 0.88, 0.86, 0.84, 0.82, then n0 = 64); the mix
weight of a window step is read at the ratio r = max(H, W) / w (SCAContext.pi_table_now, table_interp: log-linear
between the scale-2 and scale-4 tables, the nearest table outside [2, 4]); native steps read the scale-4 table.

Run from paper/figures/src with the sca env (numpy, Pillow, matplotlib):
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I fig_method_zoom.py
then compile:  ~/.local/bin/tectonic --outdir .. method_zoom.tex
then record the PDF checks (needs pymupdf < 1.26):
  ~/.local/bin/uv run --no-project --with 'pymupdf<1.26' python -I fig_method_zoom.py --pdf
"""

import hashlib
import json
import math
import os
import sys

FIG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(FIG, "src")
OUT = os.path.join(FIG, "method_zoom")
PROV = os.path.join(FIG, "method_zoom_provenance.json")
PDF = os.path.join(FIG, "method_zoom.pdf")
DATA = os.path.join(SRC, "data", "k1_1024_pi.npz")
KEY = "s4_pi_head"
SIGMA_G = 0.89
MU, STEPS = 1.15, 28                # FLUX.1 shifted schedule at every size (runtime max_shift protocol)
N_DOUBLE = 19
# sca_v2 window schedule (--win-sched 0.88:128,0.86:112,0.84:96,0.82:80), at 4096^2: H = W = 256, n0 = 64
WIN_SCHED = ((0.88, 128), (0.86, 112), (0.84, 96), (0.82, 80))
GRID_4096, N0_FLUX = 256, 64
CALIB_SCALES = (2, 4)
# example head of panel (c), one-based: chosen because its scale-2 and scale-4 weights differ visibly on the
# window steps (see stats["example_choice"]); the main diagram's block 29, head 18 has nearly equal tables there
EX_BLOCK, EX_HEAD = 2, 8
READ_STEP = 23                      # one-based native step whose weight feeds the mix node (sigma = 0.4627)
HX, VY = 4, 20                      # raster upsampling: pixels per head column, per bin row
TOY = dict(H=24, s=4, n0=6, t=2, w=6, w_max=12, text_cols=6, bridge_cells=4)  # grid of panels (a) and (b)
QUERY_TILES = ((3, 4), (11, 11))    # (x, y) tile indices drawn in panel (b)
BRIDGE_SEED = 0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def schedule():
    """FLUX.1 shifted schedule, mu = 1.15: sigma_k = e^mu / (e^mu + 1 / t_k - 1), t = linspace(1, 1/28, 28)."""
    return [math.exp(MU) / (math.exp(MU) + (1.0 / t - 1.0))
            for t in (1.0 - k * (1.0 - 1.0 / STEPS) / (STEPS - 1) for k in range(STEPS))]


def window_now(sigma):
    """SCAContext.window_now: the first (sigma_k, w_k), sigma_k descending, with sigma > sigma_k; else n0."""
    for s, w in sorted(WIN_SCHED, reverse=True):
        if sigma > s:
            return w
    return N0_FLUX


def lookup(pi, centers, sigma):
    """sca/attn.py PiTable.__call__: linear interpolation between the 14 bin centres, clamped."""
    import numpy as np
    j = np.searchsorted(centers, sigma)
    if j <= 0:
        return pi[..., 0]
    if j >= len(centers):
        return pi[..., -1]
    a = (sigma - centers[j - 1]) / (centers[j] - centers[j - 1])
    return (1 - a) * pi[..., j - 1] + a * pi[..., j]


def applied_weight(tables, centers, sigma):
    """(w, r, t, pi) of an image query at sigma under sca_v2 at 4096^2 (SCAContext.pi_table_now): a window step
    reads pi at r = max(H, W) / w, log-linear between the bracketing calibrated scales (table_interp); a native
    step (w = n0) reads the published scale-4 table. t is the interpolation weight of the scale-4 table."""
    w = window_now(sigma)
    if w == N0_FLUX:
        return w, GRID_4096 / w, None, lookup(tables[4], centers, sigma)
    r = GRID_4096 / w
    lo = max([c for c in CALIB_SCALES if c <= r], default=CALIB_SCALES[0])
    hi = min([c for c in CALIB_SCALES if c >= r], default=CALIB_SCALES[-1])
    if lo == hi:
        return w, r, (0.0 if lo == 2 else 1.0), lookup(tables[lo], centers, sigma)
    t = math.log(r / lo) / math.log(hi / lo)
    return w, r, t, (1 - t) * lookup(tables[lo], centers, sigma) + t * lookup(tables[hi], centers, sigma)


def hexrgb(h):
    import numpy as np
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)


def main():
    import numpy as np
    from PIL import Image
    sys.path.insert(0, SRC)
    import scafig as sf  # noqa: E402  (PI_CMAP, the paper-wide pi colormap, and the palette)

    def rgb(values):
        return (np.asarray(sf.PI_CMAP(np.clip(values, 0, 1)))[..., :3] * 255 + 0.5).astype(np.uint8)

    os.makedirs(OUT, exist_ok=True)
    z = np.load(DATA)
    pi, centers, pop = z[KEY], z["sigma_centers"], z["populated"].astype(bool)
    tables = {2: z["s2_pi_head"], 4: z["s4_pi_head"]}
    L, Hh, B = pi.shape
    assert (L, Hh, B) == (57, 24, 14) and [int(s) for s in z["scales"]] == list(CALIB_SCALES)
    assert np.allclose(pi[..., 0], pi[..., 1]) and not pop[0] and pop[1:].all()
    sched = np.array(schedule())
    post = sched[sched <= SIGMA_G]
    assert len(post) == 20 and int((sched > SIGMA_G).sum()) == 8

    # ---- panel (d): heads sorted within each block by the mean of the scale-4 table at the 20 post-gate step
    # sigmas (descending, stable); unchanged from the published figure
    applied4 = np.stack([lookup(pi, centers, s) for s in post], -1)      # [57, 24, 20]
    mean_applied = applied4.mean(-1)                                     # [57, 24]
    order = np.argsort(-mean_applied, axis=1, kind="stable")             # [57, 24]
    cols = np.take_along_axis(pi, order[:, :, None], axis=1).reshape(L * Hh, B)   # [1368, 14]
    img = rgb(cols.T[::-1])                                               # rows: bin 13 (top) .. bin 0
    img = np.repeat(np.repeat(img, VY, axis=0), HX, axis=1)
    strip = os.path.join(OUT, "pi_strip.png")
    Image.fromarray(img).save(strip, optimize=True)

    # ---- panel (b): bridge dots (schematic) and the block mask of the toy local operator at a native step
    H, t, w, tc, nc = TOY["H"], TOY["t"], TOY["w"], TOY["text_cols"], TOY["bridge_cells"]
    cs = H // nc                                                         # 6 x 6 tokens per cell
    taken = {(t * qx + dx, t * qy + dy) for qx, qy in QUERY_TILES                 # query tiles and the ring
             for dx in range(-1, t + 1) for dy in range(-1, t + 1)}             # of tokens around them
    # keep the dots off the drawn frame edges (centre within one token of an edge): the n0 and 2 n0 windows of the
    # interior tile, the corner tile's clipped window and its unclipped (dashed) outline
    frames = ((4, 6, 10, 12), (1, 3, 13, 15), (18, 18, 24, 24), (20, 20, 26, 26))

    def near_edge(i, j):
        x, y = i + 0.5, j + 0.5
        for x0, y0, x1, y1 in frames:
            if (x0 - 1 <= x <= x1 + 1 and (abs(y - y0) <= 1 or abs(y - y1) <= 1)) or \
               (y0 - 1 <= y <= y1 + 1 and (abs(x - x0) <= 1 or abs(x - x1) <= 1)):
                return True
        return False
    strict = taken | {(i, j) for i in range(H) for j in range(H) if near_edge(i, j)}
    rng = np.random.default_rng(BRIDGE_SEED)
    dots = []
    for cy in range(nc):
        for cx in range(nc):
            inner = [(cx * cs + i, cy * cs + j) for j in range(1, cs - 1) for i in range(1, cs - 1)]
            cand = [q for q in inner if q not in strict] or [q for q in inner if q not in taken]
            dots.append(cand[int(rng.integers(len(cand)))])
    R = len(dots)                                                        # one per cell: 16
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
    white, loc, txt = hexrgb("#FFFFFF"), hexrgb(sf.LOCAL), hexrgb(sf.MID)
    brg = hexrgb("#9C5512")                                              # dark end of PI_CMAP (amber family)
    glyph = np.empty((nt * nt, tc + R + nt * nt, 3), np.uint8)
    glyph[:, :tc] = txt                                                  # text keys: every query sees them
    glyph[:, tc:tc + R] = brg                                            # bridge keys: every query sees them
    glyph[:, tc + R:] = white
    glyph[:, tc + R:][m > 0] = loc                                       # window key tiles
    glyph = np.repeat(np.repeat(glyph, 4, axis=0), 4, axis=1)
    mask = os.path.join(OUT, "block_mask.png")
    Image.fromarray(glyph).save(mask, optimize=True)
    a_max = [int(v) for v in np.clip(np.arange(nt) * t + t // 2 - TOY["w_max"] // 2, 0, H - TOY["w_max"])]

    # ---- panel (c): example head at r = 2 and r = 4, the weight applied at each post-gate step
    l, h = EX_BLOCK - 1, EX_HEAD - 1
    ex2, ex4 = tables[2][l, h], tables[4][l, h]
    steps = []
    for k, s in enumerate(sched, 1):
        if s > SIGMA_G:
            continue
        wk, rk, tk, pk = applied_weight({2: ex2, 4: ex4}, centers, s)
        steps.append(dict(step=k, sigma=float(s), w=int(wk), r=float(rk), t=tk, pi=float(pk)))
    read = next(d for d in steps if d["step"] == READ_STEP)
    tex = ["% Written by fig_method_zoom.py from src/data/k1_1024_pi.npz; do not edit by hand.",
           "%% panel (b): bridge tokens (x/y, schematic: one per cell of a %d x %d cell grid)" % (nc, nc),
           "\\def\\bridgedots{" + ",".join("%d/%d" % d for d in dots) + "}",
           "%% panel (c): block %d, head %d (one-based), centre/pi at r = 2 and r = 4" % (EX_BLOCK, EX_HEAD),
           "\\def\\extwo{" + ",".join("%.4f/%.4f" % (centers[k], ex2[k]) for k in range(B)) + "}",
           "\\def\\exfour{" + ",".join("%.4f/%.4f" % (centers[k], ex4[k]) for k in range(B)) + "}",
           "\\def\\extwolo{%.4f} \\def\\extwohi{%.4f} \\def\\exfourlo{%.4f} \\def\\exfourhi{%.4f}"
           % (ex2[0], ex2[-1], ex4[0], ex4[-1]),
           "% weight applied at the 20 post-gate steps: step/sigma/pi/kind (1 = window step, 0 = native step)",
           "\\def\\applied{" + ",".join("%d/%.4f/%.4f/%d" % (d["step"], d["sigma"], d["pi"], d["w"] != N0_FLUX)
                                        for d in steps) + "}",
           "\\def\\readsigma{%.4f} \\def\\readpi{%.4f}" % (read["sigma"], read["pi"]),
           "% the 28 sampling sigmas (FLUX.1 shifted schedule, mu = 1.15)",
           "\\def\\steps{" + ",".join("%.4f" % s for s in sched) + "}"]
    with open(os.path.join(OUT, "data.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")

    # ---- statistics
    popi = pi[..., pop]
    shr = [d["sigma"] for d in steps if d["w"] != N0_FLUX]
    gap = np.stack([np.abs(lookup(tables[4], centers, s) - lookup(tables[2], centers, s)) for s in shr], -1).mean(-1)
    rank = int((gap > gap[l, h]).sum()) + 1
    stats = {
        "pi_min_populated": float(popi.min()), "pi_max_populated": float(popi.max()),
        "share_cells_pi_eq_0": float((popi == 0).mean()), "share_cells_pi_eq_1": float((popi == 1).mean()),
        "mean_pi_applied_post_gate_scale4": {
            "all": float(mean_applied.mean()),
            "double_stream_blocks_1_19": float(mean_applied[:N_DOUBLE].mean()),
            "single_stream_blocks_20_57": float(mean_applied[N_DOUBLE:].mean())},
        "mean_pi_populated_bins": {
            "all": float(popi.mean()), "double": float(popi[:N_DOUBLE].mean()), "single": float(popi[N_DOUBLE:].mean())},
        "example_choice": {
            "rule": "mean over the four window steps of |pi_scale4 - pi_scale2| at the step sigma, per head",
            "example_gap": float(gap[l, h]), "example_rank_of_1368": rank,
            "median_gap_over_heads": float(np.median(gap)), "max_gap": float(gap.max())},
    }
    print(json.dumps(stats, indent=1))
    print("toy block mask: full blocks %d, partial %d, of %d" % (full, part, m.size))
    for d in steps[:6]:
        print("step %d sigma %.4f w %d r %.4f t %s pi %.4f" % (d["step"], d["sigma"], d["w"], d["r"],
              "native" if d["t"] is None else "%.4f" % d["t"], d["pi"]))

    prov = {
        "figure": "paper/figures/method_zoom.pdf (main.tex Figure D.1, fig:method_zoom)",
        "tikz": "paper/figures/src/method_zoom.tex",
        "generator": "paper/figures/src/fig_method_zoom.py (rasters, data.tex, this file; --pdf adds the PDF checks)",
        "operator": "sca_v2 (code sca/attn.py, flags --bridge-cells 8 --bridge-k 16 --win-sched "
                    "0.88:128,0.86:112,0.84:96,0.82:80 --win-pi-match --win-pi-interp, gate 0.89)",
        "data": "paper/figures/src/data/k1_1024_pi.npz (identical s2/s4 tables to calib/k1_1024.npz)",
        "data_sha256": sha256(DATA),
        "key": KEY,
        "table": "per-head tables pi[block, head, bin], 57 x 24 x 14, blocks one-based 1..19 double-stream then "
                 "20..57 single-stream; s4_pi_head at native steps of 4096^2, s2_pi_head and s4_pi_head mixed on "
                 "window steps",
        "bins": {
            "count": 14, "width": "1/14 in sigma", "index_0": "sigma in [0, 1/14), unpopulated, copies bin 1 (as stored)",
            "sigma_centers": [round(float(c), 6) for c in centers],
            "populated": [bool(p) for p in pop],
            "lookup": "sca/attn.py PiTable.__call__: linear interpolation between the 14 centres, clamped at the ends",
        },
        "gate": {"sigma_g": SIGMA_G, "global_only_steps": [int(i + 1) for i in np.where(sched > SIGMA_G)[0]],
                 "schedule_sigma_28_steps": [round(float(s), 4) for s in sched],
                 "schedule_source": "FLUX.1 shifted schedule, mu = 1.15 (computed in fig_method_zoom.schedule)"},
        "window_schedule": {
            "rule": "SCAContext.window_now: w = 128, 112, 96, 80 while sigma > 0.88, 0.86, 0.84, 0.82, else n0 = 64",
            "steps_one_based": {str(d["step"]): d["w"] for d in steps if d["w"] != N0_FLUX},
            "note": "SCAContext.step is the scheduler's zero-based step_index, so these are steps 8 to 11 in the "
                    "code's numbering",
            "ratio": "r = max(H, W) / w = 256 / w at 4096^2: " + ", ".join(
                "%d -> %.4f" % (d["w"], d["r"]) for d in steps if d["w"] != N0_FLUX) + ", 64 -> 4",
            "interpolation": "SCAContext.table_interp: t = log(r / 2) / log(4 / 2), pi = (1 - t) pi_scale2 + t pi_scale4; "
                             "nearest table outside [2, 4]; native steps (w = n0) read the published scale-4 table "
                             "(pi_table_now returns self.pi)"},
        "panel_d": {
            "raster": "paper/figures/method_zoom/pi_strip.png", "raster_sha256": sha256(strip),
            "rows": "top to bottom: bin 13 (sigma near 1) to bin 0 (sigma near 0), equal height, so sigma is linear",
            "columns": "57 blocks left to right, 24 heads per block; within a block heads are sorted by the mean of "
                       "the scale-4 table at the 20 post-gate step sigmas, descending (stable)",
            "head_order_one_based": (order + 1).tolist(),
            "upsampling": "nearest, %d px per head column, %d px per bin row" % (HX, VY),
            "colormap": "scafig.PI_CMAP: 0 #1F4F8A, 0.22 #2F6DB3, 0.5 #F4F1EA, 0.78 #E08A2B, 1 #9C5512 (linear RGB)",
            "veil": "sigma > sigma_g = 0.89 (global only at inference)",
        },
        "panel_c": {
            "example": "block %d, head %d (one-based)" % (EX_BLOCK, EX_HEAD),
            "why": "its scale-2 and scale-4 weights differ visibly on the window steps (mean gap %.3f, rank %d of "
                   "1368; median over heads %.3f), so the four window-step weights visibly move from the r = 2 to "
                   "the r = 4 curve" % (gap[l, h], rank, float(np.median(gap))),
            "pi_by_bin_r2": [round(float(v), 4) for v in ex2],
            "pi_by_bin_r4": [round(float(v), 4) for v in ex4],
            "applied_steps": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} for d in steps],
            "read_step_into_mix": READ_STEP,
            "curves": "dashed: r = 2 table, solid: r = 4 table, both light; dots: the weight applied at the 20 "
                      "post-gate steps (window steps between the curves, native steps on the r = 4 curve)",
        },
        "panel_b": {"raster": "paper/figures/method_zoom/block_mask.png", "raster_sha256": sha256(mask),
                    "toy": TOY, "window_start_per_tile_index": a.tolist(),
                    "window_start_per_tile_index_w_max": a_max,
                    "query_tiles_xy": [list(q) for q in QUERY_TILES],
                    "bridge_dots_xy": [list(d) for d in dots],
                    "bridge_layout": "schematic: one token per cell of a %d x %d grid of %d x %d-token cells, drawn "
                                     "uniformly at random (numpy default_rng(%d)) from each cell's inner %d x %d tokens, "
                                     "query-tile tokens and their one-token ring excluded, and tokens within one token of a drawn frame edge excluded where the cell has others; FLUX.1 at 4096^2 keeps the 16 tokens per cell of an "
                                     "8 x 8 grid whose pre-RoPE keys, summed over heads, lie farthest from the mean key "
                                     "(bridge_index), recomputed per block and step, 1,024 tokens"
                                     % (nc, nc, cs, cs, BRIDGE_SEED, cs - 2, cs - 2),
                    "dotted_frame": "the window of tile %s at the widest scheduled side w_max = 2 n0 = %d (128 = 2 x 64 "
                                    "at 4096^2), start a = clip(t tau + t/2 - w_max/2, 0, H - w_max)"
                                    % (list(QUERY_TILES[0]), TOY["w_max"]),
                    "rows": "query tiles (t x t), tile-major order",
                    "cols": "%d text columns (grey), %d bridge columns (dark amber, one per bridge token, visible to "
                            "every query as the flex prefix L + R), then key tiles in tile-major order (amber: inside "
                            "the n0 window)" % (tc, R),
                    "full_blocks": int(full), "partial_blocks": int(part)},
        "panel_a": {"toy": TOY, "block_centres": [int(TOY["s"] * b + TOY["s"] // 2) for b in range(TOY["n0"])]},
        "image": "paper/figures/teaser/square_bright.jpg (brightened copy of teaser/square.jpg)",
        "stats": stats,
    }
    with open(PROV, "w") as fh:
        json.dump(prov, fh, indent=1)
    print("wrote", OUT, PROV)


def record_pdf():
    """Add PDF size, fonts, smallest text span and words to the provenance (the figure is placed at 396 bp)."""
    import fitz  # pymupdf < 1.26
    with open(PROV) as fh:
        prov = json.load(fh)
    doc = fitz.open(PDF)
    page = doc[0]
    sizes, fonts = [], set()
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["text"].strip():
                    sizes.append(span["size"])
                    fonts.add(span["font"])
    words = [wd[4] for wd in page.get_text("words")]
    prov.update(pdf_sha256=sha256(PDF), size_pt=[round(page.rect.width, 2), round(page.rect.height, 2)],
                size_cm=[round(page.rect.width / 72 * 2.54, 2), round(page.rect.height / 72 * 2.54, 2)],
                min_text_pt=round(min(sizes), 2), fonts=sorted(fonts), label_words=len(words))
    with open(PROV, "w") as fh:
        json.dump(prov, fh, indent=1)
    print(json.dumps({k: prov[k] for k in ("size_pt", "min_text_pt", "label_words", "fonts")}))


if __name__ == "__main__":
    record_pdf() if "--pdf" in sys.argv else main()
