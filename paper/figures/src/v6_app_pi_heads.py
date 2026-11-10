"""Appendix figure pi_heads (v6, paper/figures/v6/pi_heads.pdf): every head's fitted weight, and its split-half repeat.

(a to c) The fitted weight pi_h(sigma) of every head of the ratio-2 table (FLUX.1-dev, 1024^2, 32 prompts), one panel
         per third of the depth: blocks 1 to 19 (the double-stream blocks), 20 to 38 and 39 to 57 (single-stream), 456
         heads each, block numbers one-based as in Figure 2. Amber: the head's mean pi over the bins inference reads
         exceeds 0.8; blue: below 0.2; grey otherwise; green: the panel's median. sigma > sigma_g runs the far view
         only, so that region is veiled.
(d)      pi_h fitted on the even-numbered prompts against the odd-numbered ones, every populated cell, as a 2D
         histogram, with its Pearson rho.
All data come from fig_pi_heads.data() (imported, so every recomputation and assertion of fig_pi_heads.py runs and
nothing is written); only the presentation follows the v6 house style (v6_emulation.py, v6_observation.py): "(a) Name"
titles and the noise level falling from left to right as in Figure 2.

  cd paper/figures/src && /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I v6_app_pi_heads.py
Writes paper/figures/v6/pi_heads.pdf and paper/figures/v6/pi_heads_provenance.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_pi_heads as base  # noqa: E402
import scafig as sf  # noqa: E402

ROOT = HERE.parents[2]
V6 = ROOT / "paper" / "figures" / "v6"
STEM = "pi_heads"
W, H = sf.TEXT, 1.34
FS_T, FS_L, FS_S, FS_V = 8.5, 7.5, 7.0, 6.5
SIGMA_G = base.SIGMA_G
STYLE = {"other": (sf.MID, 0.18, 0.35), "global": (sf.GLOBAL, 0.5, 0.45), "local": (sf.LOCAL, 0.2, 0.45)}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


# ------------------------------------------------------------------------------------------------------------ draw
def axes_in(fig, x, y, w, h, **kw):
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def title(fig, x, y, letter, name):
    fig.text(x / W, y / H, f"({letter}) {name}", fontsize=FS_T, fontweight="bold", ha="left", va="baseline")


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=FS_S, length=2.2, width=0.5, pad=1.6)


def render(d):
    sf.use_style()
    sf.script_floor(FS_S)
    pi, pop, sig, cls = d["pi"], d["pop"], d["sig"], d["cls"]
    x = sig[pop]
    fig = plt.figure(figsize=(W, H))
    ty, top, bottom = 1.22, 1.10, 0.33
    ph = top - bottom
    left, gap, gap_d, right = 0.40, 0.12, 0.52, 0.04
    pw = (W - left - 2 * gap - gap_d - ph - right) / 3
    halo = [patheffects.withStroke(linewidth=2.6, foreground="white")]
    axes = []
    for k, (b0, b1, _) in enumerate(base.GROUPS):
        x0 = left + k * (pw + gap)
        ax = axes_in(fig, x0, bottom, pw, ph, sharey=axes[0] if axes else None)
        g, c = pi[b0:b1].reshape(-1, pi.shape[-1])[:, pop], cls[b0:b1].ravel()
        for kind in ("other", "global", "local"):  # grey underneath, near amber and far blue above
            col, alpha, lw = STYLE[kind]
            for row in g[c == kind]:
                ax.plot(x, row, color=col, alpha=alpha, lw=lw, solid_joinstyle="round", zorder=2)
        ax.plot(x, np.median(g, 0), color=sf.SCA, lw=1.4, zorder=4, path_effects=halo, solid_capstyle="round")
        # sigma > sigma_g: far view only, so inference never reads pi there
        ax.axvspan(SIGMA_G, 1.0, color="white", alpha=0.72, lw=0, zorder=5)
        ax.axvline(SIGMA_G, color=sf.RULE, lw=0.6, zorder=5)
        ax.set_xlim(1.0, 0.08)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([SIGMA_G, 0.5, 0.1], [r"$\sigma_g$", "0.5", "0.1"])
        ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
        clean(ax)
        if k:
            ax.tick_params(axis="y", labelleft=False)
        title(fig, 0.0 if k == 0 else x0, ty, "abc"[k], f"Blocks {b0 + 1}–{b1}")
        axes.append(ax)
    axes[0].set_ylabel(r"fitted weight $\pi_h$", fontsize=FS_L, labelpad=1.5)
    axes[1].set_xlabel(r"noise level $\sigma$  $\rightarrow$ denoising", fontsize=FS_L, labelpad=1.5)

    # (d) split halves: 2D histogram over every populated cell, neutral grey, log counts
    xd = W - right - ph
    dax = axes_in(fig, xd, bottom, ph, ph)
    edges = np.linspace(0, 1, base.HIST_BINS + 1)
    hist, _, _ = np.histogram2d(d["even"], d["odd"], bins=[edges, edges])
    cmap = LinearSegmentedColormap.from_list("grey", ["#E4E7EA", sf.INK])
    dax.pcolormesh(edges, edges, np.ma.masked_equal(hist.T, 0), cmap=cmap, norm=LogNorm(vmin=1, vmax=hist.max()),
                   shading="flat", lw=0, zorder=2, rasterized=True)
    dax.plot([0, 1], [0, 1], color=sf.RULE, lw=0.6, zorder=1)
    dax.set_xlim(0, 1)
    dax.set_ylim(0, 1)
    dax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    dax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    clean(dax)
    dax.set_xlabel(r"$\pi_h$, even half", fontsize=FS_L, labelpad=1.5)
    dax.set_ylabel(r"$\pi_h$, odd half", fontsize=FS_L, labelpad=1.5)
    dax.text(0.05, 0.95, rf"$\rho={d['r']:.3f}$", transform=dax.transAxes, ha="left", va="top", fontsize=FS_S)
    title(fig, xd - 0.40, ty, "d", "Split halves")
    return fig


def main() -> None:
    d = base.data()
    V6.mkdir(parents=True, exist_ok=True)
    fig = render(d)
    out = V6 / f"{STEM}.pdf"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    s = d["stats"]
    prov = {"figure": "Appendix figure pi_heads (fig:pi_heads), v6", "script": rel(Path(__file__)),
            "data_code": "paper/figures/src/fig_pi_heads.py data() (imported; every assertion runs, nothing written)",
            "size_in": [W, H], "inputs": d["sources"],
            "panel_blocks_one_based": {f"{b0 + 1}-{b1}": name.split()[0] for b0, b1, name in base.GROUPS},
            "values": {"split_half_rho": s["split_half_r"], "split_half_cells": s["split_half_cells"],
                       "groups": {name: {k: g[k] for k in ("heads", "local_heavy", "global_heavy", "other")}
                                  for name, g in s["groups"].items()}},
            "output_sha256": sha256(out)}
    (V6 / f"{STEM}_provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(f"rho = {s['split_half_r']:.6f} over {s['split_half_cells']} cells")


if __name__ == "__main__":
    main()
