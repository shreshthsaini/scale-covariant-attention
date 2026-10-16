"""Relative output error of each mixing rule at emulated scales 2 and 4.

Regenerate: <sca env python> paper/figures/src/fig_rel_err.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1data import FIGDIR, check, errors, load  # noqa: E402
from orx_figstyle import BASELINE, MUTED, PALETTE, TEXT, figure_grid, panel_labels, save, use_style  # noqa: E402

ROWS = [  # (key, label, color)
    ("per_head", r"per-head $\pi_h(\sigma)$ (SCA)", PALETTE["blue"]),
    ("transfer_from", r"per-head, fitted at the other scale", "#66AAD1"),
    ("per_layer", r"per-layer $\pi_\ell(\sigma)$", PALETTE["orange"]),
    ("uniform", r"shared $\pi(\sigma)$", "#F0C566"),
    ("switch", r"best hard switch", PALETTE["red"]),
    ("loc", "local only", BASELINE),
    ("glob", "global only", MUTED),
]


def main():
    use_style()
    z, report = load()
    check(z, report)
    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.30, sharey=True)
    y = np.arange(len(ROWS))[::-1]
    for ax, s in zip(axes, (2, 4)):
        e = errors(z, s)
        vals = [e[k] for k, _, _ in ROWS]
        ax.barh(y, vals, color=[c for _, _, c in ROWS], edgecolor="black", linewidth=0.3, height=0.72)
        top = max(vals)
        for yi, v in zip(y, vals):
            ax.text(v + 0.012 * top, yi, f"{v:.3f}", va="center", ha="left", fontsize=5.5)
        ax.set_xlim(0, top * 1.18)
        ax.set_xlabel(f"relative output error, scale {s}")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [lbl for _, lbl, _ in ROWS])
    axes[0].set_ylabel("mixing rule")
    panel_labels(axes)
    save(fig, os.path.join(FIGDIR, "rel_err"))
    for s in (2, 4):
        print(s, {k: round(v, 4) for k, v in errors(z, s).items()})


if __name__ == "__main__":
    main()
