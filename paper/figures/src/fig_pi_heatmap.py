"""Per-head mixing weight pi_h(sigma) at three noise levels (emulated scale 2).

Regenerate: <sca env python> paper/figures/src/fig_pi_heatmap.py
"""

import os
import sys

import matplotlib as mpl
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1data import FIGDIR, N_DOUBLE, check, load  # noqa: E402
from orx_figstyle import SEQUENTIAL, TEXT, figure_grid, panel_labels, save, use_style  # noqa: E402

SCALE = 2
BINS = (1, 7, 13)  # lowest populated, middle, highest sigma bin


def main():
    use_style()
    z, report = load()
    check(z, report)
    pi = z[f"s{SCALE}_pi_head"]  # [layers, heads, bins]
    sig = z["sigma_centers"]
    # Head index is arbitrary within a layer: order heads by their mid-sigma weight
    # (same order in every panel) so the within-layer spread is visible.
    order = np.argsort(-pi[:, :, BINS[1]], axis=1)
    sorted_pi = np.take_along_axis(pi, order[:, :, None], axis=1)

    fig, axes = figure_grid(3, 1, width=TEXT, ratio=0.52, sharex=True)
    for ax, b in zip(axes, BINS):
        img = ax.imshow(sorted_pi[:, :, b].T, cmap=SEQUENTIAL, vmin=0, vmax=1,
                        interpolation="nearest", aspect="auto", origin="lower")
        ax.axvline(N_DOUBLE - 0.5, color="white", lw=0.9)
        ax.set_yticks([0, 23], ["1", "24"])
        ax.set_ylabel(f"head (sorted)\n$\\sigma={sig[b]:.2f}$")
        ax.grid(visible=False)
    axes[-1].set_xticks([0, 9, 18, 28, 38, 48, 56], ["0", "9", "18", "28", "38", "48", "56"])
    axes[-1].set_xlabel("transformer block (0 to 18 double-stream, 19 to 56 single-stream)")
    fig.colorbar(img, ax=list(axes), label=r"local weight $\pi_h(\sigma)$", fraction=0.03, pad=0.015)
    panel_labels(axes)
    save(fig, os.path.join(FIGDIR, "pi_heatmap"))


if __name__ == "__main__":
    main()
