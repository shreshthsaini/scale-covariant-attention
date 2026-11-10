"""Appendix figure pi_sigma (v6, paper/figures/v6/pi_sigma.pdf): the fitted weight and the output error per noise level.

FLUX.1-dev, reach emulation at 1024^2, 32 prompts, ratio-2 table.
(a) Fraction of the 1,368 heads in each bin of the fitted weight pi_h at every populated noise bin, with the mean of
    the per-head weights (green), the best weight shared over heads (grey) and AReA's hard switch at sigma = 0.89 read
    as a weight (purple, dashed).
(b) Relative output error per noise bin of four rules: far view only, near view only, the shared weight, the per-head
    table.
The data code is fig_pi_sigma.py main() verbatim (k1data.load and k1data.check, then the same histogram and error
algebra); only the presentation follows the v6 house style (v6_emulation.py, v6_observation.py): "(a) Name" titles,
near amber and far blue, and the noise level falling from left to right as in Figure 2.

  cd paper/figures/src && /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I v6_app_pi_sigma.py
Writes paper/figures/v6/pi_sigma.pdf and paper/figures/v6/pi_sigma_provenance.json.
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
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scafig as sf  # noqa: E402
from k1data import check, load  # noqa: E402

ROOT = HERE.parents[2]
V6 = ROOT / "paper" / "figures" / "v6"
STEM = "pi_sigma"
W, H = sf.TEXT, 2.05
FS_T, FS_L, FS_S, FS_V = 8.5, 7.5, 7.0, 6.5
SCA_LIGHT = "#9AD3BA"
G_DARK = "#7D8893"
SCALE = 2
SIGMA_STAR = 0.89  # AReA FLUX.1 switch (fig_pi_sigma.py)
PI_BINS = np.linspace(0, 1, 21)
SHARED = G_DARK
AREA = sf.AREA
DENSITY = LinearSegmentedColormap.from_list("sca_density", ["#FFFFFF", SCA_LIGHT])


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


# ------------------------------------------------------------------------------------------------------------ data
def data() -> dict:
    """fig_pi_sigma.py main(), the data lines verbatim."""
    z, report = load()
    check(z, report)
    pi, sig, pop = z[f"s{SCALE}_pi_head"], z["sigma_centers"], z["populated"]
    S = z[f"s{SCALE}_stats"]
    bins = np.flatnonzero(pop)
    edges = np.concatenate([[0.0], (sig[bins][1:] + sig[bins][:-1]) / 2, [1.0]])
    edges[0] = sig[bins][0] - (edges[1] - sig[bins][0])
    # (a) fraction of heads per pi bin at every sigma bin, mean and shared weights on top
    dens = np.stack([np.histogram(pi[:, :, b].ravel(), PI_BINS)[0] / pi[:, :, b].size for b in bins], 1)
    vmax = np.ceil(dens.max() * 20) / 20  # next 0.05 above the densest cell
    xs = np.array([edges[0], SIGMA_STAR, SIGMA_STAR, 1.0])
    shared, mean = z[f"s{SCALE}_pi_uniform"][bins], pi[:, :, bins].mean((0, 1))
    # (b) relative output error per sigma bin
    S_b = lambda p: (S[2] - 2 * p * S[1] + p ** 2 * S[0]).sum((0, 1))[bins] / S[4].sum((0, 1))[bins]  # noqa: E731
    rules = [("far only", 0.0, sf.GLOBAL, 1.2),
             ("near only", 1.0, sf.LOCAL, 1.2),
             ("shared", z[f"s{SCALE}_pi_uniform"][None, None], SHARED, 1.2),
             ("per head", z[f"s{SCALE}_pi_head"], sf.SCA, 1.6)]
    err = [(name, S_b(p), c, lw) for name, p, c, lw in rules]
    return dict(sig=sig[bins], edges=edges, dens=dens, vmax=vmax, xs=xs, shared=shared, mean=mean, err=err,
                n_heads=int(pi.shape[0] * pi.shape[1]))


# ------------------------------------------------------------------------------------------------------------ draw
def axes_in(fig, x, y, w, h, **kw):
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def title(fig, x, y, letter, name):
    fig.text(x / W, y / H, f"({letter}) {name}", fontsize=FS_T, fontweight="bold", ha="left", va="baseline")


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=FS_S, length=2.2, width=0.5, pad=1.6)


def sigma_axis(ax, d):
    """Noise level falling from left to right (Figure 2), ticks at 0.8 ... 0.2."""
    ax.set_xlim(1.0, d["edges"][0])
    ax.set_xticks([0.8, 0.6, 0.4, 0.2], ["0.8", "0.6", "0.4", "0.2"])
    ax.set_xlabel(r"noise level $\sigma$  $\rightarrow$ denoising", fontsize=FS_L, labelpad=1.5)


def end_labels(fig, ax, items, fontsize):
    """Name each line right of its rightmost point (the lowest noise level), nudged apart to stay legible."""
    fig.canvas.draw()
    probe = ax.text(0, 0, "Ag", fontsize=fontsize)
    gap = probe.get_window_extent(fig.canvas.get_renderer()).height * 1.0
    probe.remove()
    ends = []
    for x, y, name, col in items:
        k = int(np.argmin(x))
        ends.append((ax.transData.transform((x[k], y[k]))[1], x[k], y[k], name, col))
    ends.sort(key=lambda e: e[0])
    placed = []
    for e in ends:
        placed.append(max(e[0], placed[-1] + gap) if placed else e[0])
    for (y_px, x, y, name, col), yp in zip(ends, placed):
        ax.annotate(name, (x, y), xytext=(4, (yp - y_px) * 72 / fig.dpi), textcoords="offset points", ha="left",
                    va="center", fontsize=fontsize, color=col, annotation_clip=False)


def render(d):
    sf.use_style()
    sf.script_floor(FS_S)
    fig = plt.figure(figsize=(W, H))
    ty, top, bottom = 1.93, 1.80, 0.36
    ph = top - bottom

    # (a) where the heads' weights sit at every noise level
    xa, wa = 0.40, 1.92
    title(fig, 0.0, ty, "a", "Weight distribution")
    ax = axes_in(fig, xa, bottom, wa, ph)
    mesh = ax.pcolormesh(d["edges"], PI_BINS, d["dens"], cmap=DENSITY, vmin=0, vmax=d["vmax"], shading="flat",
                         edgecolors="face", lw=0.05, rasterized=False)
    ax.plot(d["xs"], [1, 1, 0, 0], color=AREA, lw=1.1, ls=(0, (3.5, 2)), zorder=3, clip_on=False)
    ax.plot(d["sig"], d["shared"], color=SHARED, lw=1.3, zorder=4, solid_capstyle="round")
    ax.plot(d["sig"], d["mean"], color=sf.SCA, lw=1.5, zorder=5, solid_capstyle="round")
    sigma_axis(ax, d)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_ylabel(r"fitted weight $\pi$", fontsize=FS_L, labelpad=1.5)
    clean(ax)
    # direct labels: the switch right of its rise, the two curves along themselves where they are flat
    ax.text(SIGMA_STAR - 0.025, 0.40, "AReA switch", ha="left", va="center", fontsize=FS_S, color=AREA)
    sig = d["sig"]
    for name, curve, col, above in (("shared", d["shared"], SHARED, True), ("per-head mean", d["mean"], sf.SCA,
                                                                              False)):
        x0 = 0.62
        lab = ax.text(x0, 0.0, name, ha="left", va="bottom" if above else "top", fontsize=FS_S, color=col)
        fig.canvas.draw()
        bb = ax.transData.inverted().transform(lab.get_window_extent())
        span = (sig <= max(bb[:, 0])) & (sig >= min(bb[:, 0]))
        lo, hi = np.interp(sorted(bb[:, 0]), sig, curve)
        y = max(hi, lo, curve[span].max() if span.any() else hi) + 0.025 if above else \
            min(hi, lo, curve[span].min() if span.any() else lo) - 0.03
        lab.set_y(y)
    cax = axes_in(fig, xa + wa + 0.09, bottom + 0.16 * ph, 0.07, 0.68 * ph)
    ticks = np.arange(0, d["vmax"] + 1e-9, 0.1)
    cb = fig.colorbar(mesh, cax=cax, ticks=ticks)
    cb.ax.set_yticklabels([f"{t:g}" for t in ticks])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=FS_S, length=2, width=0.5, pad=1.5)
    cb.set_label("fraction of heads", fontsize=FS_L, labelpad=3)

    # (b) the error of each rule per noise bin
    xb = 3.42
    title(fig, xb - 0.38, ty, "b", "Error per noise level")
    bx = axes_in(fig, xb, bottom, 4.98 - xb, ph)
    items = []
    for name, e, col, lw in d["err"]:
        bx.plot(sig, e, color=col, lw=lw, solid_capstyle="round", zorder=3 if name == "per head" else 2)
        items.append((sig, e, name, col))
    sigma_axis(bx, d)
    bx.set_ylim(0, None)
    bx.set_yticks([0, 0.1, 0.2, 0.3], ["0", "0.1", "0.2", "0.3"])
    bx.set_ylabel("relative output error", fontsize=FS_L, labelpad=2)  # no arrow: rotated, it would point right
    sf.value_grid(bx, "y")
    clean(bx)
    end_labels(fig, bx, items, FS_S)
    return fig


def main() -> None:
    d = data()
    V6.mkdir(parents=True, exist_ok=True)
    fig = render(d)
    out = V6 / f"{STEM}.pdf"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    ins = [HERE / "data" / "k1_1024_pi.npz", HERE / "data" / "k1_1024.json"]
    vals = {"sigma_bins": [round(float(s), 4) for s in d["sig"]], "densest_cell": round(float(d["dens"].max()), 4),
            "colour_max": float(d["vmax"]), "mean_per_head_pi": [round(float(v), 4) for v in d["mean"]],
            "shared_pi": [round(float(v), 4) for v in d["shared"]],
            "rel_err_per_bin": {name: [round(float(v), 4) for v in e] for name, e, _, _ in d["err"]}}
    prov = {"figure": "Appendix figure pi_sigma (fig:pisigma), v6", "script": rel(Path(__file__)),
            "data_code": "paper/figures/src/fig_pi_sigma.py main(), data lines verbatim",
            "size_in": [W, H], "inputs": {rel(p): sha256(p) for p in ins}, "values": vals,
            "output_sha256": sha256(out)}
    (V6 / f"{STEM}_provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(f"densest cell: {d['dens'].max():.3f} of heads; colour scale 0 to {d['vmax']:.2f}")


if __name__ == "__main__":
    main()
