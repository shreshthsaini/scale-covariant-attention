"""Distribution of pi_h over heads at every noise level, against AReA's hard switch.

(a) Fraction of the 1368 heads in each pi bin at every sigma bin, with the mean of the per-head
    weights, the best shared weight, and AReA's schedule read as a weight.
(b) Relative output error per sigma bin of four mixing rules.
Rule colours keep their meaning across the paper: per-head (SCA) accent, shared rules neutral grey,
local only amber, global only blue, AReA's schedule the AReA hue, dashed. Direct labels, no legend.

Regenerate: <sca env python> paper/figures/src/fig_pi_sigma.py
"""

import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-identical PDF/SVG on rerun
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scafig as sf  # noqa: E402
from k1data import FIGDIR, check, load  # noqa: E402

SCALE = 2
SIGMA_STAR = 0.89  # AReA FLUX.1 switch (App. A.3); Phase I = first 8 of 28 steps
PI_BINS = np.linspace(0, 1, 21)
SHARED = "#6F7C89"            # neutral grey for the shared rule
AREA = sf.METHOD_COLOR["area"]  # dark grey, dashed for AReA's schedule
DENSITY = LinearSegmentedColormap.from_list("sca_density", ["#FFFFFF", "#86C9AC"])  # light accent ramp
W, H = sf.TEXT, 2.05


def box(x, y, w, h):
    return [x / W, y / H, w / W, h / H]


def end_labels(ax, lines, names, fontsize):
    """Label each line just right of its last point, outside the axes, nudged apart to stay legible."""
    fig = ax.figure
    fig.canvas.draw()
    probe = ax.text(0, 0, "Ag", fontsize=fontsize)
    gap = probe.get_window_extent(fig.canvas.get_renderer()).height * 1.05
    probe.remove()
    ends = sorted(((ax.transData.transform((ln.get_xdata()[-1], ln.get_ydata()[-1]))[1], ln, name)
                   for ln, name in zip(lines, names)), key=lambda e: e[0])
    placed = []
    for y_px, ln, name in ends:
        placed.append(max(y_px, placed[-1] + gap) if placed else y_px)
    for (y_px, ln, name), y in zip(ends, placed):
        ax.annotate(name, (ln.get_xdata()[-1], ln.get_ydata()[-1]), xytext=(4, (y - y_px) * 72 / fig.dpi),
                    textcoords="offset points", ha="left", va="center", fontsize=fontsize,
                    color=ln.get_color(), annotation_clip=False)


def main():
    sf.use_style()
    plt.rcParams["svg.hashsalt"] = "sca"
    z, report = load()
    check(z, report)
    pi, sig, pop = z[f"s{SCALE}_pi_head"], z["sigma_centers"], z["populated"]
    S = z[f"s{SCALE}_stats"]
    bins = np.flatnonzero(pop)
    edges = np.concatenate([[0.0], (sig[bins][1:] + sig[bins][:-1]) / 2, [1.0]])
    edges[0] = sig[bins][0] - (edges[1] - sig[bins][0])

    bottom, top = 0.37, 0.22
    ph = H - bottom - top
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes(box(0.42, bottom, 1.92, ph))
    cax = fig.add_axes(box(0.42 + 1.92 + 0.08, bottom + 0.15 * ph, 0.07, 0.7 * ph))
    bx = fig.add_axes(box(3.40, bottom, W - 3.40 - 0.74, ph))

    # (a) fraction of heads per pi bin at every sigma bin, mean and shared weights on top
    dens = np.stack([np.histogram(pi[:, :, b].ravel(), PI_BINS)[0] / pi[:, :, b].size for b in bins], 1)
    vmax = np.ceil(dens.max() * 20) / 20  # next 0.05 above the densest cell
    print(f"densest cell: {dens.max():.3f} of heads; colour scale 0 to {vmax:.2f}")
    mesh = ax.pcolormesh(edges, PI_BINS, dens, cmap=DENSITY, vmin=0, vmax=vmax, shading="flat",
                         edgecolors="face", lw=0.05, rasterized=False)
    xs = np.array([edges[0], SIGMA_STAR, SIGMA_STAR, 1.0])
    shared, mean = z[f"s{SCALE}_pi_uniform"][bins], pi[:, :, bins].mean((0, 1))
    ax.plot(xs, [1, 1, 0, 0], color=AREA, lw=1.1, ls=(0, (3.5, 2)), zorder=3)
    ax.plot(sig[bins], shared, color=SHARED, lw=1.3, zorder=4)
    ax.plot(sig[bins], mean, color=sf.SCA, lw=1.5, zorder=5)
    # direct labels (no legend box): the switch named left of its drop, the curves along themselves
    ax.text(SIGMA_STAR - 0.02, 0.42, "AReA hard switch", ha="right", va="center", fontsize=sf.FS_TICK,
            color=AREA)
    k = int(np.argmin(np.abs(sig[bins] - 0.28)))
    ax.text(sig[bins][k], shared[k] + 0.015, r"best shared $\pi(\sigma)$", ha="left", va="bottom",
            fontsize=sf.FS_TICK, color=SHARED)
    ax.text(sig[bins][k], mean[k] - 0.02, r"mean of per-head $\pi_h$", ha="left", va="top",
            fontsize=sf.FS_TICK, color=sf.SCA)
    ax.set_xlim(edges[0], 1.0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1"])
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_xlabel(r"noise level $\sigma$")
    ax.set_ylabel(r"local weight $\pi$")
    ax.grid(visible=False)
    cb = fig.colorbar(mesh, cax=cax, ticks=np.arange(0, vmax + 1e-9, 0.1))
    cb.ax.set_yticklabels([f"{t:g}" for t in np.arange(0, vmax + 1e-9, 0.1)])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.5, pad=1.5)
    cb.set_label("fraction of heads", labelpad=3)
    sf.panel_label(fig, ax, "a", dy=0.035 / H)

    # (b) relative output error per sigma bin
    S_b = lambda p: (S[2] - 2 * p * S[1] + p ** 2 * S[0]).sum((0, 1))[bins] / S[4].sum((0, 1))[bins]
    rules = [("global only", 0.0, sf.GLOBAL, 1.2),
             ("local only", 1.0, sf.LOCAL, 1.2),
             (r"shared $\pi(\sigma)$", z[f"s{SCALE}_pi_uniform"][None, None], SHARED, 1.2),
             (r"per-head $\pi_h$", z[f"s{SCALE}_pi_head"], sf.SCA, 1.6)]
    lines = []
    for name, p, c, lw in rules:
        (ln,) = bx.plot(sig[bins], S_b(p), color=c, lw=lw, solid_capstyle="round")
        lines.append(ln)
    bx.set_xlim(edges[0], 1.0)
    bx.set_ylim(0, None)
    bx.set_yticks([0, 0.1, 0.2, 0.3], ["0", "0.1", "0.2", "0.3"])
    bx.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1"])
    bx.set_xlabel(r"noise level $\sigma$")
    bx.set_ylabel("relative output error")
    bx.grid(visible=False)
    end_labels(bx, lines, [name for name, *_ in rules], fontsize=sf.FS_TICK)
    sf.panel_label(fig, bx, "b", dy=0.035 / H)
    sf.save(fig, os.path.join(FIGDIR, "pi_sigma"))


if __name__ == "__main__":
    main()
