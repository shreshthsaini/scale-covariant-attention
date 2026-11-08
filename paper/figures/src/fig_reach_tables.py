"""Calibrated mixing weights pi_h(sigma) and how well a table transfers across scale and resolution.

(a-c) pi_h(sigma) of the scale-4 table used at inference, three sigma bins below the gate.
(d)   Mean and interquartile range of pi over all 57 x 24 heads against sigma, three calibrations.
(e)   Relative output error of each table on each calibration target, as excess over the
      target's own per-head table, with the target's own per-layer fit as a reference. Each bar
      carries its value (%, one decimal, 6.5 pt) above it, the lowest of each target in bold.
Panels (d) and (e) carry thin light-grey horizontal gridlines behind the data (plot standard of
2026-10-19); the heatmaps (a-c) carry none.

Data (compact copies, so the figure does not depend on scratch):
  data/k1_1024_pi.npz, data/k1_1024.json   1024^2, 32 prompts, emulated s = 2 and 4
  data/k1_1280_pi.npz, data/k1_1280.json   1280^2, 16 prompts, emulated s = 2
    copied from /scratch/09032/saini_2/sca/calib/k1_1280b.{npz,json} (2026-10-16):
    npz keeps sigma_centers, populated, n_prompts, res, s2_pi_*, s2_stats = stats[0];
    json keeps res, n_prompts, sigma_centers, populated_bins, scales, settings, transfer.
Every error shown is recomputed from the per-cell statistics and asserted against the reports.

Regenerate: <sca env python> paper/figures/src/fig_calib.py
"""

import json
import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-identical PDF/SVG on rerun
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scafig as sf  # noqa: E402
from k1data import FIGDIR, HERE, N_DOUBLE, check, load, rel_err  # noqa: E402

GATE = 0.89          # sigma above which every head uses O_glob only (first 8 of 28 steps)
HEAT_SCALE = 4       # the table used at inference for 4096^2 (target_scale 4)

# All three are SCA tables, so they share the accent hue; the inference table is the accent itself.
CONFIGS = [  # key, label, color
    ("1024x1024/s2", "1024\u00b2, $s{=}2$", "#0A4A34"),
    ("1024x1024/s4", "1024\u00b2, $s{=}4$", sf.SCA),
    ("1280x1280/s2", "1280\u00b2, $s{=}2$", "#6CC09B"),
]
LAYER_COLOR = sf.RULE  # the per-layer reference is a neutral grey
W, H = sf.TEXT, 3.30   # inches, final printed size
VALUE_PAD, NAME_PAD = 1.2, 10.0  # points above a bar: its value label; a source name above that
E_HEADROOM = 1.3       # (e) y-axis reaches 1.3 times the tallest bar: room for value and name


def load_all():
    z, report = load()
    check(z, report)
    z12 = dict(np.load(os.path.join(HERE, "data", "k1_1280_pi.npz")))
    with open(os.path.join(HERE, "data", "k1_1280.json")) as fh:
        r12 = json.load(fh)
    assert np.allclose(z12["sigma_centers"], z["sigma_centers"])
    tables = {  # config -> (stats, pi_head, pi_layer)
        "1024x1024/s2": (z["s2_stats"], z["s2_pi_head"], z["s2_pi_layer"]),
        "1024x1024/s4": (z["s4_stats"], z["s4_pi_head"], z["s4_pi_layer"]),
        "1280x1280/s2": (z12["s2_stats"], z12["s2_pi_head"], z12["s2_pi_layer"]),
    }
    # transfer matrix recomputed from statistics, checked against the 1280 report
    err = {}
    for src, _, _ in CONFIGS:
        for tgt, _, _ in CONFIGS:
            err[src, tgt] = float(rel_err(tables[tgt][0], tables[src][1]))
            ref = r12["transfer"][f"{src} -> {tgt}"]
            assert abs(err[src, tgt] - ref["rel_err"]) < 1e-9, (src, tgt, err[src, tgt], ref)
    layer = {t: float(rel_err(tables[t][0], tables[t][2][:, None])) for t, _, _ in CONFIGS}
    assert abs(layer["1280x1280/s2"] - r12["scales"]["2"]["rel_err"]["per_layer"]) < 1e-9
    return z, tables, err, layer


def box(x, y, w, h):
    """Axes rectangle in inches from the bottom-left corner of the figure."""
    return [x / W, y / H, w / W, h / H]


def pi_colorbar(fig, cax, image):
    """Slim pi colorbar whose ends read global (0) and local (1), in the colours of the map's ends."""
    cb = fig.colorbar(image, cax=cax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["0", "0.5", "1"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.5, pad=1.5)
    cb.set_label(r"local weight $\pi$", labelpad=3)  # no subscript: it would print below 6 pt
    off = 0.075 / (cax.get_position().height * H)  # clear of the end tick labels
    cax.text(0.5, 1 + off, "local", transform=cax.transAxes, ha="center", va="bottom",
             fontsize=sf.FS_TICK, color=sf.pi_color(1.0))
    cax.text(0.5, -off, "global", transform=cax.transAxes, ha="center", va="top",
             fontsize=sf.FS_TICK, color=sf.pi_color(0.0))


def end_labels(fig, ax, lines, names, fontsize):
    """Label each line just right of its last point, outside the axes, nudged apart to stay legible."""
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


def letter(fig, ax, text, note=None):
    """Bold panel letter above the axes' top-left corner, with an optional plain note after it."""
    bb = ax.get_position()
    y = bb.y1 + 0.035 / H
    fig.text(bb.x0, y, text, fontsize=sf.FS_PANEL, fontweight="bold", ha="left", va="baseline")
    if note:
        fig.text(bb.x0 + 0.15 / W, y, note, fontsize=sf.FS, ha="left", va="baseline")


def main():
    sf.use_style()
    plt.rcParams["svg.hashsalt"] = "sca"
    z, tables, err, layer = load_all()
    sig, pop = z["sigma_centers"], z["populated"]
    used = np.flatnonzero(pop & (sig < GATE))           # bins the mix actually uses
    heat_bins = (used[0], used[len(used) // 2], used[-1])

    fig = plt.figure(figsize=(W, H))

    # (a-c) heatmaps: heads sorted within each block by mean pi over the used bins
    left, gap, cb_gap, cb_w = 0.40, 0.09, 0.10, 0.07
    hw = (W - left - 0.42 - 2 * gap - cb_gap - cb_w) / 3
    hh, hy = 0.86, H - 0.22 - 0.86
    hax = [fig.add_axes(box(left, hy, hw, hh))]
    hax += [fig.add_axes(box(left + i * (hw + gap), hy, hw, hh), sharey=hax[0]) for i in (1, 2)]
    cax = fig.add_axes(box(left + 3 * hw + 2 * gap + cb_gap, hy, cb_w, hh))
    pi = tables[f"1024x1024/s{HEAT_SCALE}"][1]
    order = np.argsort(-pi[:, :, used].mean(2), axis=1, kind="stable")
    spi = np.take_along_axis(pi, order[:, :, None], axis=1)
    for i, (ax, b) in enumerate(zip(hax, heat_bins)):
        img = ax.imshow(spi[:, :, b].T, cmap=sf.PI_CMAP, vmin=0, vmax=1, interpolation="nearest",
                        aspect="auto", origin="lower")
        ax.axvline(N_DOUBLE - 0.5, color="white", lw=0.6)
        ax.set_xticks([(N_DOUBLE - 1) / 2, (N_DOUBLE + 56) / 2], ["double", "single"])
        ax.tick_params(axis="x", length=0, pad=2.5)
        ax.grid(visible=False)
        for s in ax.spines.values():
            s.set_visible(False)
        if i == 0:
            ax.set_yticks([0, 23], ["1", "24"])
            ax.tick_params(axis="y", length=2, width=0.5)
            ax.set_ylabel("head (sorted)", labelpad=1)
        else:
            ax.tick_params(axis="y", length=0, labelleft=False)
        letter(fig, ax, "abc"[i])  # the sigma of each panel is stated in the caption
    hax[1].set_xlabel("transformer block (stream type)", labelpad=2)
    pi_colorbar(fig, cax, img)

    # bottom row geometry
    by, bh = 0.37, hy - 0.58 - 0.37
    dw, ex = 1.80, left + 1.80 + 1.02  # room right of (d) for the direct labels
    dax = fig.add_axes(box(left, by, dw, bh))
    eax = fig.add_axes(box(ex, by, W - ex - 0.04, bh))

    # (d) mean and IQR of pi over all heads against sigma
    bins = np.flatnonzero(pop)
    dax.axvspan(GATE, 1.0, color=sf.GLOBAL_LIGHT, lw=0, zorder=0)
    dax.text(1.0, 1.02, "global-only opening", ha="right", va="bottom",
             fontsize=sf.FS_SMALL, color=sf.GLOBAL, transform=dax.get_xaxis_transform())
    lines = []
    for key, lbl, col in CONFIGS:
        p = tables[key][1][:, :, bins].reshape(-1, len(bins))
        q1, q3 = np.percentile(p, [25, 75], axis=0)
        dax.fill_between(sig[bins], q1, q3, color=col, alpha=0.08, lw=0, zorder=1)
        lines += dax.plot(sig[bins], p.mean(0), color=col, lw=1.4, zorder=3, solid_capstyle="round")
    dax.set_xlim(0.08, 1.0)
    dax.set_ylim(0, 1)
    dax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    dax.set_yticks([0.25, 0.75], minor=True)
    dax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1"])
    dax.set_xlabel(r"noise level $\sigma$")
    dax.set_ylabel(r"local weight $\pi$ over heads")
    sf.value_grid(dax, "y")
    end_labels(fig, dax, lines, [lbl for _, lbl, _ in CONFIGS], sf.FS_TICK)  # direct labels, no legend
    letter(fig, dax, "d")

    # (e) transfer: excess error over the target's own per-head table, bars coloured by source table
    width, step = 0.22, 0.25
    top, drawn = 0.0, []  # (x, value, colour, source name) of every bar
    source_name = {col: lbl + " table" for _, lbl, col in CONFIGS}
    source_name[LAYER_COLOR] = "per-layer fit"
    for ti, (tgt, _, _) in enumerate(CONFIGS):
        own = err[tgt, tgt]
        bars = [(100 * (err[src, tgt] / own - 1), col) for src, _, col in CONFIGS if src != tgt]
        bars.append((100 * (layer[tgt] / own - 1), LAYER_COLOR))
        best = min(val for val, _ in bars)
        for k, (val, col) in enumerate(bars):
            x = ti + (k - 1) * step
            eax.bar(x, val, width=width, color=col, lw=0, zorder=2)
            # value above the bar; the lowest excess for this target in bold
            eax.annotate(f"{val:.1f}", (x, val), xytext=(0, VALUE_PAD), textcoords="offset points",
                         ha="center", va="bottom", fontsize=sf.FS_VALUE, color=sf.INK,
                         fontweight="bold" if val == best else "normal")
            drawn.append((x, val, col))
            top = max(top, val)
    # Direct labels instead of a legend: each source is named once, above the value of the tallest
    # bar of its colour; the greens read upward, the grey reference sits flat above its value.
    for col, name in source_name.items():
        x, val, _ = max((b for b in drawn if b[2] == col), key=lambda b: b[1])
        flat = col == LAYER_COLOR
        eax.annotate(name, (x, val), xytext=(0, NAME_PAD), textcoords="offset points", ha="center",
                     va="bottom", rotation=0 if flat else 90, fontsize=sf.FS_SMALL,
                     color=sf.MID if flat else col)
    eax.set_xticks(range(len(CONFIGS)), [lbl for _, lbl, _ in CONFIGS])
    eax.tick_params(axis="x", length=0)
    eax.set_xlim(-0.5, len(CONFIGS) - 0.5)
    eax.set_ylim(0, top * E_HEADROOM)
    sf.value_grid(eax, "y")
    eax.set_xlabel("calibration target")
    eax.set_ylabel("excess error (%)")
    letter(fig, eax, "e")
    sf.save(fig, os.path.join(FIGDIR, "calib"))

    print("heatmap scale", HEAT_SCALE, "bins", [f"{sig[b]:.3f}" for b in heat_bins])
    for key, _, _ in CONFIGS:
        p = tables[key][1][:, :, used]
        print(f"{key}: frac pi in [0.2,0.8] over used bins {np.mean((p >= 0.2) & (p <= 0.8)):.3f}, "
              f"mean std over heads {p.reshape(-1, len(used)).std(0).mean():.3f}")
    for tgt, _, _ in CONFIGS:
        own = err[tgt, tgt]
        row = ", ".join(f"{src} {err[src, tgt]:.4f} ({100 * (err[src, tgt] / own - 1):+.1f}%)"
                        for src, _, _ in CONFIGS if src != tgt)
        print(f"target {tgt}: own {own:.4f}; {row}; per-layer {layer[tgt]:.4f} "
              f"({100 * (layer[tgt] / own - 1):+.1f}%)")


if __name__ == "__main__":
    main()
