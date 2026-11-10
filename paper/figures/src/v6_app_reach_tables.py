"""Appendix figure reach_tables (v6, paper/figures/v6/reach_tables.pdf): the reach table, and how well a table transfers.

(a to c) The fitted weight pi_h of the ratio-4 table that sampling reads (FLUX.1-dev, 1024^2, 32 prompts) at three noise
         bins below the gate, drawn in the order of denoising (sigma 0.83, 0.46, 0.10); heads sorted within each block
         by their mean weight over the used bins; blue 0 = far, amber 1 = near.
(d)      Mean (line) and interquartile range (band) of pi_h over all 1,368 heads against the noise level, for the three
         reach tables (1024^2 at ratios 2 and 4, 1280^2 at ratio 2); sigma > 0.89 runs the far view only.
(e)      Excess relative output error of each table scored at each emulated setting, over that setting's own per-head
         table, with the setting's own per-layer fit as the reference; one group per scored setting, each bar named
         by the table it uses, values in percent, the lowest of each group in bold.
All numbers come from fig_reach_tables.load_all() (imported: every error is recomputed from the per-cell statistics and
asserted against data/k1_1024.json and data/k1_1280.json) and from the lines of fig_reach_tables.py main() that derive
the drawn quantities, copied verbatim; nothing is written by the import. Only the presentation follows the v6 house
style (v6_emulation.py, v6_observation.py): "(a) Name" titles, near and far wording, the noise level falling from left
to right as in Figure 2, and horizontal labelled bars in (e) instead of rotated labels.

  cd paper/figures/src && /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I v6_app_reach_tables.py
Writes paper/figures/v6/reach_tables.pdf and paper/figures/v6/reach_tables_provenance.json.
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
from matplotlib import transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_reach_tables as base  # noqa: E402
import scafig as sf  # noqa: E402
from k1data import N_DOUBLE  # noqa: E402

ROOT = HERE.parents[2]
V6 = ROOT / "paper" / "figures" / "v6"
STEM = "reach_tables"
W, H = sf.TEXT, 3.50
FS_T, FS_L, FS_S, FS_V = 8.5, 7.5, 7.0, 6.5
SCA_LIGHT = "#9AD3BA"
G_DARK = "#7D8893"
GATE, HEAT_SCALE = base.GATE, base.HEAT_SCALE
# the three tables share the SCA hue; the table sampling reads (1024^2, ratio 4) is the accent itself
TABLE = {"1024x1024/s2": (r"$1024^2$, $r=2$", "#0A4A34"),
         "1024x1024/s4": (r"$1024^2$, $r=4$", sf.SCA),
         "1280x1280/s2": (r"$1280^2$, $r=2$", "#6CC09B")}
LAYER_COLOR = G_DARK


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


# ------------------------------------------------------------------------------------------------------------ data
def data() -> dict:
    """fig_reach_tables.load_all(), then the data lines of fig_reach_tables.py main() verbatim."""
    z, tables, err, layer = base.load_all()
    sig, pop = z["sigma_centers"], z["populated"]
    used = np.flatnonzero(pop & (sig < GATE))           # bins the mix actually uses
    heat_bins = (used[0], used[len(used) // 2], used[-1])
    # (a-c) heatmaps: heads sorted within each block by mean pi over the used bins
    pi = tables[f"1024x1024/s{HEAT_SCALE}"][1]
    order = np.argsort(-pi[:, :, used].mean(2), axis=1, kind="stable")
    spi = np.take_along_axis(pi, order[:, :, None], axis=1)
    # (d) mean and IQR of pi over all heads against sigma
    bins = np.flatnonzero(pop)
    curves = {}
    for key, _, _ in base.CONFIGS:
        p = tables[key][1][:, :, bins].reshape(-1, len(bins))
        q1, q3 = np.percentile(p, [25, 75], axis=0)
        curves[key] = (q1, q3, p.mean(0))
    # (e) transfer: excess error over the target's own per-head table, per source table and the per-layer fit
    bars = {}
    for tgt, _, _ in base.CONFIGS:
        own = err[tgt, tgt]
        row = [(src, 100 * (err[src, tgt] / own - 1)) for src, _, _ in base.CONFIGS if src != tgt]
        row.append(("per_layer", 100 * (layer[tgt] / own - 1)))
        bars[tgt] = row
    return dict(sig=sig, bins=bins, heat_bins=heat_bins, spi=spi, curves=curves, bars=bars, err=err, layer=layer,
                used=used)


# ------------------------------------------------------------------------------------------------------------ draw
def axes_in(fig, x, y, w, h, **kw):
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def title(fig, x, y, letter, name):
    fig.text(x / W, y / H, f"({letter}) {name}", fontsize=FS_T, fontweight="bold", ha="left", va="baseline")


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=FS_S, length=2.2, width=0.5, pad=1.6)


def end_labels(fig, ax, items, fontsize):
    """Name each line right of its rightmost point (the lowest noise level), nudged apart to stay legible."""
    fig.canvas.draw()
    probe = ax.text(0, 0, "Ag", fontsize=fontsize)
    gap = probe.get_window_extent(fig.canvas.get_renderer()).height * 1.12
    probe.remove()
    ends = []
    for x, y, name, col in items:
        k = int(np.argmin(x))
        ends.append((ax.transData.transform((x[k], y[k]))[1], x[k], y[k], name, col))
    ends.sort(key=lambda e: e[0])
    placed = []
    for e in ends:
        placed.append(max(e[0], placed[-1] + gap) if placed else e[0])
    shift = max(0.0, (placed[-1] - ends[-1][0]) / 2)  # centre the stack on the lines
    for (y_px, x, y, name, col), yp in zip(ends, placed):
        ax.annotate(name, (x, y), xytext=(4, (yp - shift - y_px) * 72 / fig.dpi), textcoords="offset points",
                    ha="left", va="center", fontsize=fontsize, color=col, annotation_clip=False)


def panel_heatmaps(fig, d, y_top, hh):
    left, gap, cb_gap, cb_w, right = 0.40, 0.10, 0.12, 0.07, 0.56
    hw = (W - left - right - 2 * gap - cb_gap - cb_w) / 3
    y0 = y_top - hh
    axes = []
    for i, b in enumerate(d["heat_bins"][::-1]):          # the order of denoising: highest noise first
        x0 = left + i * (hw + gap)
        ax = axes_in(fig, x0, y0, hw, hh, sharey=axes[0] if axes else None)
        img = ax.imshow(d["spi"][:, :, b].T, cmap=sf.PI_CMAP, vmin=0, vmax=1, interpolation="nearest",
                        aspect="auto", origin="lower")
        ax.axvline(N_DOUBLE - 0.5, color="white", lw=0.7)
        ax.set_xticks([(N_DOUBLE - 1) / 2, (N_DOUBLE + 56) / 2], ["double", "single"])
        ax.tick_params(axis="x", length=0, pad=2.5, labelsize=FS_S)
        ax.grid(visible=False)
        for s in ax.spines.values():
            s.set_visible(False)
        if i == 0:
            ax.set_yticks([0, 23], ["1", "24"])
            ax.tick_params(axis="y", length=2.2, width=0.5, labelsize=FS_S, pad=1.6)
            ax.set_ylabel("head, sorted", fontsize=FS_L, labelpad=1)
        else:
            ax.tick_params(axis="y", length=0, labelleft=False)
        title(fig, 0.0 if i == 0 else x0, y_top + 0.10, "abc"[i], f"Noise level {d['sig'][b]:.2f}")
        axes.append(ax)
    axes[1].set_xlabel("transformer block", fontsize=FS_L, labelpad=2)
    cax = axes_in(fig, left + 3 * hw + 2 * gap + cb_gap, y0, cb_w, hh)
    cb = fig.colorbar(img, cax=cax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["0", "0.5", "1"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=FS_S, length=2, width=0.5, pad=1.5)
    cb.set_label(r"fitted weight $\pi_h$", fontsize=FS_L, labelpad=2)
    off = 0.05 / hh
    cax.text(0.5, 1 + off, "near", transform=cax.transAxes, ha="center", va="bottom", fontsize=FS_S,
             color=sf.pi_color(1.0))
    cax.text(0.5, -off, "far", transform=cax.transAxes, ha="center", va="top", fontsize=FS_S,
             color=sf.pi_color(0.0))


def panel_mean(fig, d, x, y, w, h):
    ax = axes_in(fig, x, y, w, h)
    sig, bins = d["sig"], d["bins"]
    ax.axvspan(GATE, 1.0, color=sf.GLOBAL_LIGHT, lw=0, zorder=0)
    ax.annotate("far only", (1.0, 0.04), xytext=(3, 0), textcoords="offset points", ha="left", va="bottom",
                fontsize=FS_S, color=sf.GLOBAL, zorder=1)
    items = []
    for key, _, _ in base.CONFIGS:
        lab, col = TABLE[key]
        q1, q3, mean = d["curves"][key]
        ax.fill_between(sig[bins], q1, q3, color=col, alpha=0.10, lw=0, zorder=1)
        ax.plot(sig[bins], mean, color=col, lw=1.4, zorder=3, solid_capstyle="round")
        items.append((sig[bins], mean, lab, col))
    ax.set_xlim(1.0, 0.08)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.8, 0.6, 0.4, 0.2], ["0.8", "0.6", "0.4", "0.2"])
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_yticks([0.25, 0.75], minor=True)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_xlabel(r"noise level $\sigma$  $\rightarrow$ denoising", fontsize=FS_L, labelpad=1.5)
    ax.set_ylabel(r"fitted weight $\pi_h$", fontsize=FS_L, labelpad=1.5)
    sf.value_grid(ax, "y")
    clean(ax)
    end_labels(fig, ax, items, FS_S)
    return ax


def panel_transfer(fig, d, x, y, w, h, lab_w):
    """One group per scored setting: a header, then one bar per table it can borrow and its own per-layer fit."""
    ax = axes_in(fig, x + lab_w, y, w - lab_w, h)
    rows, yy = [], 0.0                                     # (y, kind, text, value, colour, bold)
    for tgt, _, _ in base.CONFIGS:
        rows.append((yy, "head", rf"scored at {TABLE[tgt][0]}", None, None, False))
        best = min(v for _, v in d["bars"][tgt])
        for src, v in d["bars"][tgt]:
            yy += 1.0
            lab, col = ("per layer", LAYER_COLOR) if src == "per_layer" else TABLE[src]
            rows.append((yy, "bar", lab, v, col, v == best))
        yy += 1.45                                         # a gap before the next group
    top = max(r[3] for r in rows if r[3] is not None)
    ax.set_xlim(0, top * 1.16)
    ax.set_ylim(rows[-1][0] + 0.55, -0.7)
    head_tr = transforms.blended_transform_factory(fig.transFigure, ax.transData)
    for i, kind, text, v, col, bold in rows:
        if kind == "head":
            ax.text(x / W, i + 0.08, text, transform=head_tr, ha="left", va="center", fontsize=FS_S, color=sf.INK)
            continue
        ax.barh([i], [v], height=0.66, color=col, lw=0, zorder=2)
        ax.annotate(text, (0, i), xytext=(-3, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=FS_S, color=sf.INK, annotation_clip=False)
        ax.annotate(f"{v:.1f}", (v, i), xytext=(2, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=FS_V, color=sf.INK, fontweight="bold" if bold else "normal")
    ax.set_yticks([])
    ax.set_xticks([0, 10, 20], ["0", "10", "20"])
    sf.value_grid(ax, "x")
    clean(ax)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(r"excess error (%) $\downarrow$", fontsize=FS_L, labelpad=1.5)
    return ax


def render(d):
    sf.use_style()
    sf.script_floor(FS_S)
    fig = plt.figure(figsize=(W, H))
    y_top, hh = H - 0.22, 0.80
    panel_heatmaps(fig, d, y_top, hh)
    ty2 = y_top - hh - 0.47
    bottom = 0.36
    btop = ty2 - 0.12
    title(fig, 0.0, ty2, "d", "Three tables agree")
    panel_mean(fig, d, 0.40, bottom, 1.52, btop - bottom)
    xe = 2.86
    title(fig, xe, ty2, "e", "Transfer cost")
    panel_transfer(fig, d, xe, bottom, W - xe - 0.30, btop - bottom, lab_w=0.72)
    return fig


def main() -> None:
    d = data()
    V6.mkdir(parents=True, exist_ok=True)
    fig = render(d)
    out = V6 / f"{STEM}.pdf"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    ins = [HERE / "data" / "k1_1024_pi.npz", HERE / "data" / "k1_1024.json", HERE / "data" / "k1_1280_pi.npz",
           HERE / "data" / "k1_1280.json"]
    keys = [k for k, _, _ in base.CONFIGS]
    vals = {"heatmap_sigma_drawn_left_to_right": [round(float(d["sig"][b]), 4) for b in d["heat_bins"][::-1]],
            "rel_err_source_to_target": {f"{s} -> {t}": d["err"][s, t] for s in keys for t in keys},
            "rel_err_per_layer": d["layer"],
            "excess_percent": {t: {s: round(v, 3) for s, v in row} for t, row in d["bars"].items()},
            "mean_pi_over_heads": {k: [round(float(v), 4) for v in d["curves"][k][2]] for k in keys}}
    prov = {"figure": "Appendix figure reach_tables (fig:reach_tables), v6", "script": rel(Path(__file__)),
            "data_code": "paper/figures/src/fig_reach_tables.py load_all() (imported) and the data lines of its "
                         "main() (copied verbatim)",
            "size_in": [W, H], "inputs": {rel(p): sha256(p) for p in ins}, "values": vals,
            "output_sha256": sha256(out)}
    (V6 / f"{STEM}_provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    for t, row in d["bars"].items():
        print(t, {s: round(v, 2) for s, v in row})


if __name__ == "__main__":
    main()
