"""Distribution of pi_h over heads at every noise level, against AReA's hard switch.

Regenerate: <sca env python> paper/figures/src/fig_pi_sigma.py
"""

import os
import sys

import matplotlib as mpl
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1data import FIGDIR, check, load  # noqa: E402
from orx_figstyle import (BASELINE, PALETTE, TEXT, figure_grid, panel_labels,  # noqa: E402
                          save, use_style)

SCALE = 2
SIGMA_STAR = 0.89  # AReA FLUX.1 switch (App. A.3); Phase I = first 8 of 28 steps
PI_BINS = np.linspace(0, 1, 21)


def main():
    use_style()
    z, report = load()
    check(z, report)
    pi, sig, pop = z[f"s{SCALE}_pi_head"], z["sigma_centers"], z["populated"]
    S = z[f"s{SCALE}_stats"]
    bins = np.flatnonzero(pop)
    edges = np.concatenate([[0.0], (sig[bins][1:] + sig[bins][:-1]) / 2, [1.0]])
    edges[0] = sig[bins][0] - (edges[1] - sig[bins][0])

    fig, (ax, bx) = figure_grid(1, 2, width=TEXT, ratio=0.33, gridspec_kw={"width_ratios": [1.5, 1]})

    # (a) fraction of heads per pi bin at every sigma bin, mean and shared weights on top
    dens = np.stack([np.histogram(pi[:, :, b].ravel(), PI_BINS)[0] / pi[:, :, b].size for b in bins], 1)
    cmap = mpl.colormaps["Greys"]
    mesh = ax.pcolormesh(edges, PI_BINS, dens, cmap=cmap, vmin=0, vmax=0.25, shading="flat")
    ax.plot(sig[bins], pi[:, :, bins].mean((0, 1)), color=PALETTE["blue"], lw=1.4, marker="o", ms=2.5,
            label=r"mean of per-head $\pi_h$")
    ax.plot(sig[bins], z[f"s{SCALE}_pi_uniform"][bins], color=PALETTE["orange"], lw=1.4, marker="s", ms=2.5,
            label=r"best shared $\pi(\sigma)$")
    xs = np.array([edges[0], SIGMA_STAR, SIGMA_STAR, 1.0])
    ax.plot(xs, [1, 1, 0, 0], color=PALETTE["red"], lw=1.2, ls="--", label="AReA hard switch")
    ax.set_xlim(edges[0], 1.0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"noise level $\sigma$")
    ax.set_ylabel(r"local weight $\pi$")
    ax.grid(visible=False)
    fig.colorbar(mesh, ax=ax, label="fraction of heads", fraction=0.05, pad=0.02)
    ax.legend(loc="lower left", fontsize=6, frameon=True, framealpha=0.9)

    # (b) relative error per sigma bin
    S_b = lambda p: (S[2] - 2 * p * S[1] + p ** 2 * S[0]).sum((0, 1))[bins] / S[4].sum((0, 1))[bins]
    lines = [("per-head $\\pi_h(\\sigma)$", z[f"s{SCALE}_pi_head"], PALETTE["blue"], "-"),
             ("best shared $\\pi(\\sigma)$", z[f"s{SCALE}_pi_uniform"][None, None], PALETTE["orange"], "-"),
             ("local only", 1.0, BASELINE, "-"),
             ("global only", 0.0, BASELINE, ":")]
    for name, p, c, ls in lines:
        bx.plot(sig[bins], S_b(p), color=c, ls=ls, lw=1.3, label=name)
    bx.set_xlabel(r"noise level $\sigma$")
    bx.set_ylabel("relative output error")
    bx.set_ylim(0, None)
    bx.legend(loc="center left", fontsize=6, frameon=True, framealpha=0.9)
    panel_labels([ax, bx])
    save(fig, os.path.join(FIGDIR, "pi_sigma"))


if __name__ == "__main__":
    main()
