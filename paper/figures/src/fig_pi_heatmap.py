"""Per-head mixing weight pi_h(sigma) at three noise levels (emulated scale 2).

Same visual language as fig_calib.py: the pi colormap (blue = global at 0, amber = local at 1),
a subtle white line between double- and single-stream blocks, and one slim colourbar.

Regenerate: <sca env python> paper/figures/src/fig_pi_heatmap.py
"""

import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-identical PDF/SVG on rerun
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scafig as sf  # noqa: E402
from k1data import FIGDIR, N_DOUBLE, check, load  # noqa: E402

SCALE = 2
BINS = (1, 7, 13)  # lowest populated, middle, highest sigma bin
LEFT, RIGHT, BOTTOM, TOP, LETTER_ROW, RH = 0.42, 0.62, 0.40, 0.05, 0.20, 0.62  # inches
W, H = sf.TEXT, 0.40 + 3 * (0.62 + 0.20) + 0.05


def pi_colorbar(fig, cax, image):
    """Slim pi colorbar whose ends read global (0) and local (1), as in fig_calib.py."""
    cb = fig.colorbar(image, cax=cax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["0", "0.5", "1"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.5, pad=1.5)
    cb.set_label(r"local weight $\pi_h$", labelpad=3)
    off = 0.075 / (cax.get_position().height * H)
    cax.text(0.5, 1 + off, "local", transform=cax.transAxes, ha="center", va="bottom",
             fontsize=sf.FS_TICK, color=sf.pi_color(1.0))
    cax.text(0.5, -off, "global", transform=cax.transAxes, ha="center", va="top",
             fontsize=sf.FS_TICK, color=sf.pi_color(0.0))


def main():
    sf.use_style()
    plt.rcParams["svg.hashsalt"] = "sca"
    z, report = load()
    check(z, report)
    pi = z[f"s{SCALE}_pi_head"]  # [layers, heads, bins]
    sig = z["sigma_centers"]
    # Head index is arbitrary within a layer: order heads by their mid-sigma weight
    # (same order in every panel) so the within-layer spread is visible.
    order = np.argsort(-pi[:, :, BINS[1]], axis=1)
    sorted_pi = np.take_along_axis(pi, order[:, :, None], axis=1)

    left, right, bottom, top, letter_row, rh = LEFT, RIGHT, BOTTOM, TOP, LETTER_ROW, RH
    hw = W - left - right
    fig = plt.figure(figsize=(W, H))
    axes = []
    for i in range(3):
        y0 = bottom + (2 - i) * (rh + letter_row)
        rect = [left / W, y0 / H, hw / W, rh / H]
        axes.append(fig.add_axes(rect, sharex=axes[0] if axes else None, sharey=axes[0] if axes else None))
    for i, (ax, b) in enumerate(zip(axes, BINS)):
        img = ax.imshow(sorted_pi[:, :, b].T, cmap=sf.PI_CMAP, vmin=0, vmax=1,
                        interpolation="nearest", aspect="auto", origin="lower")
        ax.axvline(N_DOUBLE - 0.5, color="white", lw=0.6)
        ax.set_yticks([0, 23], ["1", "24"])
        ax.tick_params(axis="y", length=2, width=0.5)
        ax.grid(visible=False)
        for s in ax.spines.values():
            s.set_visible(False)
        if i < 2:
            ax.tick_params(axis="x", length=0, labelbottom=False)
        bb = ax.get_position()
        fig.text(bb.x0, bb.y1 + 0.04 / H, "abc"[i], fontsize=sf.FS_PANEL, fontweight="bold", va="baseline")
        fig.text(bb.x0 + 0.15 / W, bb.y1 + 0.04 / H, rf"$\sigma = {sig[b]:.2f}$", fontsize=sf.FS,
                 va="baseline")
    axes[1].set_ylabel("head (sorted within block)")
    axes[-1].set_xticks([0, 9, 18, 28, 38, 48, 56], ["0", "9", "18", "28", "38", "48", "56"])
    axes[-1].tick_params(axis="x", length=2, width=0.5)
    axes[-1].set_xlabel("transformer block (0 to 18 double-stream, 19 to 56 single-stream)")
    mid = axes[1].get_position()
    ch = 1.25
    cax = fig.add_axes([(W - right + 0.12) / W, (mid.y0 + mid.height / 2) - ch / 2 / H, 0.07 / W, ch / H])
    pi_colorbar(fig, cax, img)
    sf.save(fig, os.path.join(FIGDIR, "pi_heatmap"))
    for b in BINS:  # numbers quoted in the caption
        lo, hi = pi[:, :, b].min(1), pi[:, :, b].max(1)
        print(f"sigma {sig[b]:.2f}: within-block range of pi_h median {np.median(hi - lo):.2f}, "
              f"min {np.min(hi - lo):.2f}; blocks spanning <0.1 to >0.9: {np.sum((lo < 0.1) & (hi > 0.9))}/57")


if __name__ == "__main__":
    main()
