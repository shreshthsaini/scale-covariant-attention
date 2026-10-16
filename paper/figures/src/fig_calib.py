"""Calibrated mixing weights pi_h(sigma) and how well a table transfers across scale and resolution.

(a-c) pi_h(sigma) of the scale-4 table used at inference, three sigma bins below the gate.
(d)   Mean and interquartile range of pi over all 57 x 24 heads against sigma, three calibrations.
(e)   Relative output error of each table on each calibration target, as excess over the
      target's own per-head table, with the own per-layer fit as a reference.

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
from k1data import FIGDIR, HERE, N_DOUBLE, check, load, rel_err  # noqa: E402
from orx_figstyle import BASELINE, PALETTE, SEQUENTIAL, TEXT, panel_labels, save, use_style  # noqa: E402

GATE = 0.89          # sigma above which every head uses O_glob only (first 8 of 28 steps)
HEAT_SCALE = 4       # the table used at inference for 4096^2 (target_scale 4)

CONFIGS = [  # key, label, color
    ("1024x1024/s2", r"$1024^2$, $s{=}2$", PALETTE["blue"]),
    ("1024x1024/s4", r"$1024^2$, $s{=}4$", PALETTE["orange"]),
    ("1280x1280/s2", r"$1280^2$, $s{=}2$", PALETTE["green"]),
]


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


def main():
    use_style()
    plt.rcParams["svg.hashsalt"] = "sca"
    z, tables, err, layer = load_all()
    sig, pop = z["sigma_centers"], z["populated"]
    used = np.flatnonzero(pop & (sig < GATE))           # bins the mix actually uses
    heat_bins = (used[0], used[len(used) // 2], used[-1])

    fig = plt.figure(figsize=(TEXT, 3.55), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[1.0, 1.25], hspace=0.04)

    # (a-c) heatmaps: heads sorted within each block by mean pi over the used bins
    pi = tables[f"1024x1024/s{HEAT_SCALE}"][1]
    order = np.argsort(-pi[:, :, used].mean(2), axis=1, kind="stable")
    spi = np.take_along_axis(pi, order[:, :, None], axis=1)
    hax = top.subplots(1, 3, sharey=True)
    for ax, b in zip(hax, heat_bins):
        img = ax.imshow(spi[:, :, b].T, cmap=SEQUENTIAL, vmin=0, vmax=1, interpolation="nearest",
                        aspect="auto", origin="lower")
        ax.axvline(N_DOUBLE - 0.5, color="white", lw=1.0)
        ax.set_xticks([(N_DOUBLE - 1) / 2, (N_DOUBLE + 56) / 2], ["double", "single"])
        ax.tick_params(axis="x", length=0)
        ax.grid(visible=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
    hax[0].set_yticks([0, 23], ["1", "24"])
    hax[0].set_ylabel("head (sorted)")
    hax[1].set_xlabel("transformer block")
    top.colorbar(img, ax=list(hax), label=r"$\pi_h(\sigma)$", fraction=0.04, pad=0.02, aspect=12)
    panel_labels(hax, [f"({c}) $\\sigma={sig[b]:.2f}$" for c, b in zip("abc", heat_bins)])

    dax, eax = bot.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.1]})

    # (d) mean and IQR of pi over all heads against sigma
    bins = np.flatnonzero(pop)
    for key, lbl, col in CONFIGS:
        p = tables[key][1][:, :, bins].reshape(-1, len(bins))
        q1, q3 = np.percentile(p, [25, 75], axis=0)
        dax.fill_between(sig[bins], q1, q3, color=col, alpha=0.06, lw=0)
        for q in (q1, q3):
            dax.plot(sig[bins], q, color=col, lw=0.5, alpha=0.7)
        dax.plot(sig[bins], p.mean(0), color=col, marker="o", ms=2.2, lw=1.2, label=lbl)
    dax.axvspan(GATE, 1.0, color=BASELINE, alpha=0.15, lw=0)
    dax.axvline(GATE, color=BASELINE, lw=0.8, ls="--")
    dax.text((GATE + 1.0) / 2, 0.03, "gate:\nglobal\nonly", fontsize=5.5, color="#4D4D4D", va="bottom", ha="center",
             transform=dax.get_xaxis_transform())
    dax.set_xlim(0.08, 1.0)
    dax.set_ylim(0, 1)
    dax.set_xlabel(r"noise level $\sigma$")
    dax.set_ylabel(r"$\pi$ over heads (mean, IQR)")
    dax.legend(loc="lower left", fontsize=6, handlelength=1.2)

    # (e) transfer: excess error over the target's own table
    nC = len(CONFIGS)
    off = np.linspace(-0.24, 0.24, nC)
    for ti, (tgt, _, _) in enumerate(CONFIGS):
        own = err[tgt, tgt]
        for si, (src, lbl, col) in enumerate(CONFIGS):
            x = ti + off[si]
            ex = 100 * (err[src, tgt] / own - 1)
            if src == tgt:
                eax.plot(x, ex, marker="o", ms=3.6, mfc="white", mec=col, mew=1.0, ls="none")
            else:
                eax.plot([x, x], [0, ex], color=col, lw=1.0)
                eax.plot(x, ex, marker="o", ms=3.6, color=col, ls="none")
        lx = 100 * (layer[tgt] / own - 1)
        eax.plot([ti - 0.36, ti + 0.36], [lx, lx], color=BASELINE, lw=1.2, ls="-")
        eax.text(ti, lx + 0.8, f"per-layer +{lx:.0f}%", ha="center", va="bottom", fontsize=5.5, color="#4D4D4D")
        eax.text(ti, -2.2, f"own {own:.3f}", ha="center", va="top", fontsize=5.5, color="#4D4D4D")
    eax.axhline(0, color="black", lw=0.5)
    eax.set_xticks(range(nC), [lbl for _, lbl, _ in CONFIGS])
    eax.tick_params(axis="x", length=0)
    eax.set_xlim(-0.55, nC - 0.45)
    top_y = max(100 * (layer[t] / err[t, t] - 1) for t, _, _ in CONFIGS)
    eax.set_ylim(-6.5, top_y + 5)
    eax.set_xlabel("calibration target (dot color = source table)")
    eax.set_ylabel("excess rel. error vs own (%)")
    panel_labels([dax, eax], ["(d)", "(e)"])
    save(fig, os.path.join(FIGDIR, "calib"))

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
