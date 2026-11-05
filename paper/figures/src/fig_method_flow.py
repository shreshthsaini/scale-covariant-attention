"""Assets and data for the SCA method figure (Figure 3, fig:method), operator sca_v2.

The figure is TikZ (figures/src/method.tex, compiled to figures/method.pdf). This script writes everything it
draws from into figures/method_flow/ and records provenance in figures/method_provenance.json.

Panels (redesign of 2026-11-02):
  (a) Calibrate at 1024^2. One calibration image; three key maps of one query at emulated ratio r = 4 on the
      64 x 64 token grid, to scale: full attention (every key, grey), global (the block-centre key of every 4 x 4
      block, a 16 x 16 lattice, blue), local (the 16 x 16 window around the query tile, amber); the closed-form
      fit of one table cell, J(pi) = E - 2 pi B + pi^2 A on [0, 1] with minimum pi* = clip(B/A, 0, 1); and a
      3 x 3 excerpt of the r = 4 table (the three heads of Figure 5, sigma = 0.10, 0.54, 0.90), the fitted cell
      outlined.
  (b) Mix per head. O^glob and O^loc into the weighted sum O_i; the 24 heads of one block at one sampling step
      as stacked bars (amber pi_h, blue 1 - pi_h), sorted, the head of panel (c) outlined.
  (c) Sample at 4096^2. Input noise; one head's pi_h over the 28 steps (0 before the gate, one value per step);
      the shrinking window as nested frames (w = 128, 112, 96, 80, 64 tokens, to scale) on the real crop of the
      output around one query tile; the output.

Files in figures/method_flow/:
  calib.jpg   calibration prompt calib00, seed 0, as a 1024 px stand-in (see CALIB_NOTE).
  noise.png   seeded Gaussian RGB noise on a 64 x 64 grid, nearest-neighbour upsampled (an icon for x_1).
  output.jpg  the sca_v2 output at 4096^2 (FLUX.1-dev, seed 0), downsampled.
  crop.jpg    the 128-token (2048 px) window around the query tile of that output, downsampled.
  data.tex    TikZ macros: the pi_h staircase, the bar heights, the table excerpt and its colours, the fit
              parabola, the frame offsets.

Rules (sca/attn.py with the sca_v2 flags --bridge-cells 8 --bridge-k 16 --win-sched
0.88:128,0.86:112,0.84:96,0.82:80 --win-pi-match --win-pi-interp, gate 0.89):
  gate     sigma > 0.89: every query takes O^glob, pi = 0 (eq:mix).
  window   w = 128, 112, 96, 80 while sigma > 0.88, 0.86, 0.84, 0.82 (SCAContext.window_now), else n0 = 64;
           start a = clip(t tau + floor(t/2) - floor(w/2), 0, H - w) per axis, t = 8.
  weight   pi at r = max(H, W) / w, log-linear between the r = 2 and r = 4 tables (SCAContext.table_interp);
           at w = n0 the r = 4 table. Lookup linear in sigma between the 14 bin centres (PiTable.__call__).
  check    the mean of that weight over all 57 x 24 heads reproduces the pi_trace stored with the output image at
           every post-gate step (asserted below, tolerance 1e-9).

Run from paper/figures/src on a compute node (no GPU needed):
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I fig_method_flow.py
then compile:  ~/.local/bin/tectonic --outdir .. method.tex
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imgsel  # noqa: E402  (checked final-operator images)
import scafig  # noqa: E402  (palette and the paper-wide pi colormap)

FIG = HERE.parent
REPO = FIG.parent.parent
OUT = FIG / "method_flow"
NPZ = REPO / "results" / "evidence" / "calib" / "k1_1024.npz"
PROV = FIG / "method_provenance.json"

PID = "a4k_eval_4096_minus8343371271909676658"     # church and sheep, as in Figures 1 and 5
CALIB_SRC = Path("/scratch/09032/saini_2/sca/out/smoke_sca_2048/calib00_f4378132a3_seed0.png")
CALIB_NOTE = ("stand-in: the calibration pass stores sufficient statistics, not images. This is calibration prompt "
              "calib00 (seed 0) rendered by SCA at 2048^2 (smoke run), resized to 1024 px; the image the pass "
              "observed is the full-attention 1024^2 sample of the same prompt and seed. Searched 2026-11-04: no "
              "1024^2 full-attention image of any calibration prompt exists under /scratch/09032/saini_2/sca/out or "
              "/scratch/09032/saini_2/sca/calib (calib holds only the per-prompt statistics in *_parts/*.npz), so "
              "the stand-in stays until a GPU run renders that sample.")

H = 256                      # latent tokens per side at 4096^2
N0, T = 64, 8                # native side, query tile
GATE = 0.89
WIN_SCHED = ((0.88, 128), (0.86, 112), (0.84, 96), (0.82, 80))
TILE = (12, 14)              # query tile (row, col), the tile of Figure 5a: tokens 96..103 x 112..119
HEADS = {"local": (6, 9), "mid": (24, 20), "global": (11, 23)}   # (block, head), zero-based, Figure 5's heads
CURVE = "global"             # head drawn in (c) and outlined in (b)
TABLE_R = 4                  # ratio of the excerpt and the key maps
TABLE_BINS = (1, 7, 12)      # sigma centres 0.105, 0.536, 0.896
FIT_BIN = 7                  # the outlined cell: CURVE head, sigma = 0.536
BAR_STEP = 20                # one-based sampling step of the bars (and the dot on the curve)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lookup(table: np.ndarray, centres: np.ndarray, sigma: float) -> np.ndarray:
    """PiTable.__call__: linear in sigma between bin centres, clamped at both ends."""
    j = int(np.searchsorted(centres, sigma))
    if j <= 0:
        return table[..., 0]
    if j >= len(centres):
        return table[..., -1]
    a = (sigma - centres[j - 1]) / (centres[j] - centres[j - 1])
    return (1 - a) * table[..., j - 1] + a * table[..., j]


def window(sigma: float) -> int:
    for s, w in WIN_SCHED:
        if sigma > s:
            return w
    return N0


def weight(z, sigma: float) -> np.ndarray:
    """pi for every (block, head) at one step of the 4096^2 sampler."""
    c = z["sigma_centers"]
    if sigma > GATE:
        return np.zeros(z["s4_pi_head"].shape[:2])
    w = window(sigma)
    if w == N0:
        return lookup(z["s4_pi_head"], c, sigma)
    t = math.log(H / w / 2) / math.log(2)
    return (1 - t) * lookup(z["s2_pi_head"], c, sigma) + t * lookup(z["s4_pi_head"], c, sigma)


def start(tau: int, w: int) -> int:
    return int(np.clip(T * tau + T // 2 - w // 2, 0, H - w))


def hexcol(rgb) -> str:
    return "".join(f"{int(round(255 * v)):02X}" for v in rgb[:3])


def pale(value: float, keep: float = 0.42) -> str:
    """Cell fill: the paper's pi colour mixed with white, so black text stays legible."""
    r, g, b, _ = scafig.PI_CMAP(float(np.clip(value, 0, 1)))
    return hexcol([keep * v + (1 - keep) for v in (r, g, b)])


def build_images(png: Path) -> dict:
    OUT.mkdir(exist_ok=True)
    img = Image.open(png).convert("RGB")
    assert img.size == (4096, 4096)
    img.resize((560, 560), Image.LANCZOS).save(OUT / "output.jpg", quality=92, subsampling=0)
    px = 4096 // H
    r0, c0 = start(TILE[0], 128), start(TILE[1], 128)
    box = (c0 * px, r0 * px, (c0 + 128) * px, (r0 + 128) * px)
    img.crop(box).resize((560, 560), Image.LANCZOS).save(OUT / "crop.jpg", quality=92, subsampling=0)
    cal = Image.open(CALIB_SRC).convert("RGB").resize((1024, 1024), Image.LANCZOS)
    cal.resize((300, 300), Image.LANCZOS).save(OUT / "calib.jpg", quality=92, subsampling=0)
    rng = np.random.default_rng(20261102)
    noise = np.clip(0.5 + 0.22 * rng.standard_normal((64, 64, 3)), 0, 1)
    Image.fromarray((255 * noise).astype(np.uint8)).resize((256, 256), Image.NEAREST).save(OUT / "noise.png")
    return {"crop_px_xyxy": list(box), "crop_tokens_rc": [r0, c0, 128]}


def build_data(z, meta: dict) -> dict:
    c = z["sigma_centers"]
    sig = [float(s) for s in meta["sigmas"][:28]]
    trace = {d["step"]: d["pi_mean"] for d in meta["pi_trace"]}
    steps = []
    for k, s in enumerate(sig):
        p = weight(z, s)
        if k in trace:
            assert abs(float(p.mean()) - trace[k]) < 1e-9, (k, float(p.mean()), trace[k])
        steps.append(p)
    assert len(trace) == 20, "expected the 20 post-gate steps in the stored pi_trace"
    bl, hd = HEADS[CURVE]
    curve = [float(p[bl, hd]) for p in steps]
    windows = [window(s) if s <= GATE else None for s in sig]

    lines = ["% Written by fig_method_flow.py; do not edit by hand.",
             f"% pi_h of block {bl + 1}, head {hd + 1} (one-based) at the 28 steps of the 4096^2 sampler."]
    stair = []
    for k, v in enumerate(curve, start=1):
        stair += [f"({k - 0.5:g},{v:.4f})", f"({k + 0.5:g},{v:.4f})"]
    lines.append(r"\def\PiStair{" + " -- ".join(stair) + "}")
    lines.append(rf"\def\PiDotStep{{{BAR_STEP}}}\def\PiDotVal{{{curve[BAR_STEP - 1]:.4f}}}")

    p = steps[BAR_STEP - 1][bl]
    order = np.argsort(-p, kind="stable")
    bars = ",".join(f"{p[h]:.4f}/{int(h == hd)}" for h in order)
    lines.append(rf"\def\PiBars{{{bars}}}")

    tab = z[f"s{TABLE_R}_pi_head"]
    rows = []
    for i, name in enumerate(("local", "mid", "global")):
        b, h = HEADS[name]
        vals = [float(tab[b, h, j]) for j in TABLE_BINS]
        rows.append(vals)
        for j, v in enumerate(vals):
            lines.append(rf"\definecolor{{tab{i}{j}}}{{HTML}}{{{pale(v)}}}\expandafter\def\csname tabv{i}{j}\endcsname{{{v:.2f}}}")
    lines.append(r"\def\TabSigmas{" + ",".join(f"{c[j]:.1f}" for j in TABLE_BINS) + "}")

    si = [2, 4].index(TABLE_R)
    A, B, E = (float(z["stats"][si, q, bl, hd, FIT_BIN]) for q in (0, 1, 2))
    pstar = float(np.clip(B / A, 0, 1))
    assert abs(pstar - float(tab[bl, hd, FIT_BIN])) < 1e-12
    a, b = A / E, B / E
    jmin, j0, j1 = 1 - b * b / a, 1.0, 1 - 2 * b + a
    lines.append(rf"\def\FitA{{{a:.5f}}}\def\FitB{{{b:.5f}}}\def\FitPi{{{pstar:.4f}}}\def\FitJmin{{{jmin:.5f}}}"
                 rf"\def\FitJlo{{{min(jmin, j0, j1):.5f}}}\def\FitJhi{{{max(j0, j1):.5f}}}")

    offs = [(128 - w) / 256 for w in (128, 112, 96, 80, 64)]
    lines.append(r"\def\FrameOffsets{" + ",".join(f"{o:.6f}" for o in offs) + "}")
    q0 = (T * TILE[0] - start(TILE[0], 128)) / 128
    q1 = (T * TILE[1] - start(TILE[1], 128)) / 128
    lines.append(rf"\def\QueryRow{{{q0:.6f}}}\def\QueryCol{{{q1:.6f}}}\def\QuerySide{{{T / 128:.6f}}}")
    (OUT / "data.tex").write_text("\n".join(lines) + "\n")

    return {
        "curve_head_one_based": {"block": bl + 1, "head": hd + 1},
        "curve_pi_per_step": [round(v, 4) for v in curve],
        "sigma_per_step": [round(s, 4) for s in sig],
        "window_per_step": windows,
        "pi_trace_check": "mean over 57 x 24 heads equals the stored pi_trace at all 20 post-gate steps (|d| < 1e-9)",
        "bars": {"block_one_based": bl + 1, "step_one_based": BAR_STEP, "sigma": round(sig[BAR_STEP - 1], 4),
                 "ratio": H / windows[BAR_STEP - 1], "pi_sorted": [round(float(p[h]), 4) for h in order],
                 "outlined_head_one_based": hd + 1},
        "table": {"ratio": TABLE_R, "sigma_bins": list(TABLE_BINS),
                  "sigma_centres": [round(float(c[j]), 4) for j in TABLE_BINS],
                  "rows": {n: {"block_one_based": HEADS[n][0] + 1, "head_one_based": HEADS[n][1] + 1,
                               "pi": [round(v, 4) for v in r]} for n, r in zip(("local", "mid", "global"), rows)},
                  "outlined": {"row": "global", "sigma_bin": FIT_BIN}},
        "fit": {"cell": {"block_one_based": bl + 1, "head_one_based": hd + 1, "ratio": TABLE_R, "sigma_bin": FIT_BIN},
                "A": A, "B": B, "E": E, "pi_star": pstar,
                "drawn": "J(pi)/E = 1 - 2 (B/E) pi + (A/E) pi^2 on [0, 1]"},
    }


def main():
    z = np.load(NPZ)
    png, digest = imgsel.final_png("4096x4096", PID)
    png = Path(png)
    meta = json.loads(png.with_suffix(".json").read_text())
    assert meta["config"]["sca"]["pi_table_sha256"] == sha256(NPZ), "image not generated with this table"
    geo = build_images(png)
    data = build_data(z, meta)
    prov = {
        "figure": "fig:method (Figure 3), operator sca_v2",
        "tikz": "paper/figures/src/method.tex",
        "script": "paper/figures/src/fig_method_flow.py",
        "assets_dir": "paper/figures/method_flow",
        "palette": {"method_green": scafig.SCA, "local_amber": scafig.LOCAL, "global_blue": scafig.GLOBAL,
                    "neutral_ink": scafig.INK, "full_grey": "#9AA3AC",
                    "table_cells": "scafig.PI_CMAP mixed 42% with white"},
        "calibration_table": {"path": str(NPZ.relative_to(REPO)), "sha256": sha256(NPZ)},
        "output_image": {"path": str(png), "sha256": digest, "prompt_id": PID,
                         "prompt": meta["config"]["prompt"], "method": meta["config"]["method"],
                         "res": meta["config"]["res"], "seed": meta["config"]["seed"],
                         "query_tile_rc": list(TILE), **geo},
        "calibration_image": {"path": str(CALIB_SRC), "sha256": sha256(CALIB_SRC), "note": CALIB_NOTE},
        "noise": "numpy default_rng(20261102), N(0.5, 0.22^2) per channel on 64 x 64, clipped; an icon, not the "
                 "sampler's latent noise",
        "key_maps": "vector, drawn to scale on the 64 x 64 grid at r = 4: 16 x 16 block-centre lattice "
                    "(kappa(b) = 4 b + 2), 16 x 16 window centred on the query tile",
        "frames": "w = 128, 112, 96, 80, 64 tokens, concentric on the query tile (no clamping at this tile), "
                  "crop = the w = 128 window",
        **data,
    }
    PROV.write_text(json.dumps(prov, indent=1) + "\n")
    print("wrote", OUT, PROV)


if __name__ == "__main__":
    main()
