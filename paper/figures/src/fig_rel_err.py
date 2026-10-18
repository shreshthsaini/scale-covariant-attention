"""Relative output error of each mixing rule at emulated scales 2 and 4.

Rule colours keep their meaning across the paper: per-head (SCA) accent, the other scale's per-head
table light accent, local only amber, global only blue, the shared rules neutral greys.

Regenerate: <sca env python> paper/figures/src/fig_rel_err.py
"""

import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-identical PDF/SVG on rerun
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scafig as sf  # noqa: E402
from k1data import FIGDIR, check, errors, load  # noqa: E402

ROWS = [  # (key, label, color)
    ("per_head", r"per-head $\pi_h(\sigma)$, SCA", sf.SCA),
    ("transfer_from", "per-head, other scale", "#9AD3BA"),
    ("per_layer", r"per-layer $\pi_\ell(\sigma)$", "#7D8893"),
    ("uniform", r"shared $\pi(\sigma)$", "#A3ABB3"),
    ("switch", "best hard switch", "#C6CBD0"),
    ("loc", "local only", sf.LOCAL),
    ("glob", "global only", sf.GLOBAL),
]
W, H = sf.TEXT, 1.72


def main():
    sf.use_style()
    plt.rcParams["svg.hashsalt"] = "sca"
    z, report = load()
    check(z, report)
    left, gap, right, bottom, top = 1.30, 0.30, 0.06, 0.37, 0.22
    pw = (W - left - gap - right) / 2
    fig = plt.figure(figsize=(W, H))
    axes = []
    for i in range(2):
        rect = [(left + i * (pw + gap)) / W, bottom / H, pw / W, (H - bottom - top) / H]
        axes.append(fig.add_axes(rect, sharey=axes[0] if axes else None))
    y = np.arange(len(ROWS))[::-1]
    for i, (ax, s) in enumerate(zip(axes, (2, 4))):
        e = errors(z, s)
        vals = [e[k] for k, _, _ in ROWS]
        ax.barh(y, vals, color=[c for _, _, c in ROWS], height=0.68, lw=0, zorder=2)
        for yi, v in zip(y, vals):
            ax.annotate(f"{v:.3f}", (v, yi), xytext=(2.5, 0), textcoords="offset points", va="center",
                        ha="left", fontsize=sf.FS_SMALL)
        ax.set_xlim(0, max(vals) * 1.17)
        ax.set_xlabel("relative output error")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.spines["left"].set_visible(False)
        if i == 0:
            ax.set_yticks(y, [lbl for _, lbl, _ in ROWS])
            ax.tick_params(axis="y", length=0, pad=3)
            # The rule column gets a header in the panel-letter row instead of a rotated label.
            ax.set_ylabel("mixing rule", rotation=0, ha="left", va="baseline", color=sf.MID,
                          fontsize=sf.FS_TICK)
            ax.yaxis.set_label_coords(-(left - 0.04) / pw, 1 + 0.05 / (H - bottom - top))
        else:
            ax.tick_params(axis="y", length=0, labelleft=False)
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        bb = ax.get_position()
        fig.text(bb.x0, bb.y1 + 0.05 / H, "ab"[i], fontsize=sf.FS_PANEL, fontweight="bold", va="baseline")
        fig.text(bb.x0 + 0.15 / W, bb.y1 + 0.05 / H, f"emulated scale {s}", fontsize=sf.FS, va="baseline")
    sf.save(fig, os.path.join(FIGDIR, "rel_err"))
    for s in (2, 4):
        print(s, {k: round(v, 4) for k, v in errors(z, s).items()})


if __name__ == "__main__":
    main()
