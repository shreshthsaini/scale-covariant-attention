"""Appendix figure pi_heatmap (v6, paper/figures/v6/pi_heatmap.pdf): the fitted weight of every head at three noise levels.

FLUX.1-dev, reach emulation at 1024^2, 32 prompts, ratio-2 table: pi_h at the noise bins centred at 0.97, 0.54 and
0.10 (top to bottom, the order of denoising), 57 blocks by 24 heads, heads sorted within each block by their weight at
the middle bin (the same order in every panel). Blue 0 = far, amber 1 = near (sf.PI_CMAP); a white line separates the
19 double-stream from the 38 single-stream blocks; block numbers are one-based, as in Figure 2.
The data code is fig_pi_heatmap.py main() verbatim (k1data.load, k1data.check, the same sort); only the presentation
follows the v6 house style (v6_emulation.py): "(a) Name" titles, near and far wording, nothing under 6 pt.

The appendix places this file at 0.9 \\linewidth (\\apxgraphic[width=0.9\\linewidth]{pi_heatmap}), so it is built at
that printed width, 0.9 x 5.5 in, and prints at 1:1. Pass --width 5.5 for a full-width build.

  cd paper/figures/src && /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I v6_app_pi_heatmap.py
Writes paper/figures/v6/pi_heatmap.pdf and paper/figures/v6/pi_heatmap_provenance.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scafig as sf  # noqa: E402
from k1data import N_DOUBLE, check, load  # noqa: E402

ROOT = HERE.parents[2]
V6 = ROOT / "paper" / "figures" / "v6"
STEM = "pi_heatmap"
W = round(0.9 * sf.TEXT, 3)  # printed width under \apxgraphic[width=0.9\linewidth]
FS_T, FS_L, FS_S, FS_V = 8.5, 7.5, 7.0, 6.5
SCALE = 2
BINS = (1, 7, 13)  # lowest populated, middle, highest sigma bin (fig_pi_heatmap.py)
DRAW = (13, 7, 1)  # drawn top to bottom: the order of denoising
RH, BAND, BOTTOM, TOPPAD = 0.60, 0.23, 0.34, 0.18  # heatmap height, title band, bottom margin, top margin (in)
H = BOTTOM + 3 * RH + 2 * BAND + TOPPAD


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


# ------------------------------------------------------------------------------------------------------------ data
def data() -> dict:
    """fig_pi_heatmap.py main(), the data lines verbatim."""
    z, report = load()
    check(z, report)
    pi = z[f"s{SCALE}_pi_head"]  # [layers, heads, bins]
    sig = z["sigma_centers"]
    # Head index is arbitrary within a layer: order heads by their mid-sigma weight
    # (same order in every panel) so the within-layer spread is visible.
    order = np.argsort(-pi[:, :, BINS[1]], axis=1)
    sorted_pi = np.take_along_axis(pi, order[:, :, None], axis=1)
    spread = {}
    for b in BINS:  # numbers quoted in the caption
        lo, hi = pi[:, :, b].min(1), pi[:, :, b].max(1)
        spread[f"{sig[b]:.2f}"] = {"median_within_block_range": round(float(np.median(hi - lo)), 4),
                                   "min_within_block_range": round(float(np.min(hi - lo)), 4),
                                   "blocks_spanning_below_0.1_to_above_0.9": int(np.sum((lo < 0.1) & (hi > 0.9)))}
    return dict(pi=pi, sig=sig, sorted_pi=sorted_pi, spread=spread)


# ------------------------------------------------------------------------------------------------------------ draw
def render(d, w: float):
    sf.use_style()
    sf.script_floor(FS_S)

    def axes_in(x, y, ww, hh, **kw):
        return fig.add_axes([x / w, y / H, ww / w, hh / H], **kw)

    def title(x, y, letter, name):
        fig.text(x / w, y / H, f"({letter}) {name}", fontsize=FS_T, fontweight="bold", ha="left", va="baseline")

    fig = plt.figure(figsize=(w, H))
    left, right = 0.40, 0.70
    hw = w - left - right
    axes = []
    for i, b in enumerate(DRAW):
        y0 = BOTTOM + (2 - i) * (RH + BAND)
        ax = axes_in(left, y0, hw, RH, sharex=axes[0] if axes else None, sharey=axes[0] if axes else None)
        img = ax.imshow(d["sorted_pi"][:, :, b].T, cmap=sf.PI_CMAP, vmin=0, vmax=1, interpolation="nearest",
                        aspect="auto", origin="lower")
        ax.axvline(N_DOUBLE - 0.5, color="white", lw=0.8)
        ax.set_yticks([0, 23], ["1", "24"])
        ax.tick_params(labelsize=FS_S, length=2.2, width=0.5, pad=1.6)
        ax.grid(visible=False)
        for s in ax.spines.values():
            s.set_visible(False)
        if i < 2:
            ax.tick_params(axis="x", length=0, labelbottom=False)
        title(0.0, y0 + RH + 0.085, "abc"[i], f"Noise level {d['sig'][b]:.2f}")
        axes.append(ax)
    axes[1].set_ylabel("head, sorted", fontsize=FS_L, labelpad=2)
    blocks = [1, 10, 19, 29, 39, 49, 57]  # one-based, as in Figure 2
    axes[-1].set_xticks([k - 1 for k in blocks], [str(k) for k in blocks])
    axes[-1].tick_params(axis="x", length=2.2, width=0.5)
    # the two stream types, each named under its block range (one-based 1-19 and 20-57)
    yl = 0.05
    for lo, hi, name in ((0, N_DOUBLE - 1, "double-stream blocks"), (N_DOUBLE, 56, "single-stream blocks")):
        xc = left + hw * ((lo + hi) / 2 + 0.5) / 57
        fig.text(xc / w, yl / H, name, fontsize=FS_L, ha="center", va="bottom")
    # colour bar beside the middle panel: 0 = far (blue), 1 = near (amber)
    mid_y = BOTTOM + RH + BAND + RH / 2
    ch = 1.25
    cax = axes_in(w - right + 0.12, mid_y - ch / 2, 0.07, ch)
    cb = fig.colorbar(img, cax=cax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["0", "0.5", "1"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=FS_S, length=2, width=0.5, pad=1.5)
    cb.set_label(r"fitted weight $\pi_h$", fontsize=FS_L, labelpad=2)
    off = 0.06 / ch
    cax.text(0.5, 1 + off, "near", transform=cax.transAxes, ha="center", va="bottom", fontsize=FS_S,
             color=sf.pi_color(1.0))
    cax.text(0.5, -off, "far", transform=cax.transAxes, ha="center", va="top", fontsize=FS_S,
             color=sf.pi_color(0.0))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--width", type=float, default=W, help="printed width in inches (default 0.9 x 5.5)")
    a = ap.parse_args()
    d = data()
    V6.mkdir(parents=True, exist_ok=True)
    fig = render(d, a.width)
    out = V6 / f"{STEM}.pdf"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    ins = [HERE / "data" / "k1_1024_pi.npz", HERE / "data" / "k1_1024.json"]
    prov = {"figure": "Appendix figure pi_heatmap (fig:heatmap), v6", "script": rel(Path(__file__)),
            "data_code": "paper/figures/src/fig_pi_heatmap.py main(), data lines verbatim",
            "size_in": [a.width, round(H, 3)],
            "placement": "built at 0.9 x 5.5 in, the width \\apxgraphic[width=0.9\\linewidth] prints it at",
            "panels_top_to_bottom_sigma": [round(float(d["sig"][b]), 4) for b in DRAW],
            "head_order": "within each block by pi at the middle bin, descending (np.argsort, as fig_pi_heatmap.py)",
            "inputs": {rel(p): sha256(p) for p in ins}, "values": d["spread"], "output_sha256": sha256(out)}
    (V6 / f"{STEM}_provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    for k, v in d["spread"].items():
        print(f"sigma {k}: {v}")


if __name__ == "__main__":
    main()
