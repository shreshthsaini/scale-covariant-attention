"""Figure 2 (fig:concept): one switch for every head (AReA) against one calibrated mix per head (SCA).

x is the noise level sigma, from 1 (left, start of sampling) to 0, so the figure holds for every configuration:
single-pass SCA and AReA read it over their 28 steps at 4096^2, progressive SCA over its 2048^2 stage (ratio 2) and
4096^2 stage (ratio 4), which start at sigma 0.81 and 0.61. Drawn is the ratio-4 table, the weight every head puts on
its native window at 4096^2, as the sampler reads it: linear in sigma between the centres of the occupied noise bins,
constant beyond them (paper Eq. sched; scripts/sample.py). Above the gate sigma_g = 0.89 every head sees the global
view only (shaded). AReA in SCA's notation: weight 0 on its windows above the gate, then 1 for every head.

Two stages, both deterministic:
  1. write results/heads_view/concept_curves.csv from calib/k1_1024.npz (FLUX.1-dev, 32 prompts at 1024^2): one row
     per (layer, head, sigma) on a fixed grid of 80 sigma values from 0.89 to 0.10, column pi (ratio 4), plus
     columns layer, head, mean_after_gate (the head's mean weight over the grid, used for its colour);
  2. draw paper/figures/concept.pdf from that CSV only (0.42 text width; text at 6.5 pt or more).
Run from anywhere with the sca env python:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python fig_concept.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scafig as sf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

ROOT = HERE.parents[2]
NPZ = ROOT / "calib" / "k1_1024.npz"
CSV = ROOT / "results" / "heads_view" / "concept_curves.csv"
OUT = ROOT / "paper" / "figures" / "concept"
GATE = 0.89
GRID = np.round(np.linspace(GATE, 0.10, 80), 6)


def write_csv() -> None:
    z = np.load(NPZ, allow_pickle=False)
    pop = z["populated"]
    centres = z["sigma_centers"][pop]
    P4 = z["s4_pi_head"][:, :, pop]  # [layer, head, occupied bin], bins in increasing sigma

    def at_sigma(s: float) -> np.ndarray:
        if s <= centres[0]:
            return P4[..., 0]
        if s >= centres[-1]:
            return P4[..., -1]
        j = int(np.searchsorted(centres, s) - 1)
        u = (s - centres[j]) / (centres[j + 1] - centres[j])
        return P4[..., j] + u * (P4[..., j + 1] - P4[..., j])

    curves = np.stack([at_sigma(float(s)) for s in GRID], axis=-1)  # [layer, head, grid]
    mean_after = curves.mean(axis=-1)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "head", "sigma", "pi", "mean_after_gate"])
        for l in range(curves.shape[0]):
            for h in range(curves.shape[1]):
                for s, p in zip(GRID, curves[l, h]):
                    w.writerow([l, h, f"{s:.6f}", f"{p:.6f}", f"{mean_after[l, h]:.6f}"])
    print(f"wrote {CSV}: {curves.shape[0] * curves.shape[1]} heads x {len(GRID)} sigma values")


def draw() -> None:
    heads: dict[tuple[int, int], list[tuple[float, float]]] = {}
    mean_after: dict[tuple[int, int], float] = {}
    for r in csv.DictReader(open(CSV)):
        key = (int(r["layer"]), int(r["head"]))
        heads.setdefault(key, []).append((float(r["sigma"]), float(r["pi"])))
        mean_after[key] = float(r["mean_after_gate"])
    keys = sorted(heads)
    sig = np.array([s for s, _ in heads[keys[0]]])
    Y = np.array([[p for _, p in heads[k]] for k in keys])
    m = np.array([mean_after[k] for k in keys])
    # the sampler switches in one step at the gate: each curve jumps from 0 (global only) to its weight
    gx = np.concatenate([[1.0, GATE, GATE], sig])
    def jump(y):
        return np.concatenate([[0.0, 0.0, y[0]], y])

    sf.use_style()
    fig, ax = plt.subplots(figsize=(0.42 * sf.TEXT, 1.62))
    ax.axvspan(GATE, 1.0, color=sf.GRID, lw=0, zorder=0)
    ax.text((1.0 + GATE) / 2, 0.5, "global view", ha="center", va="center", rotation=90, fontsize=sf.FS_SMALL,
            color=sf.MID)
    order = np.argsort(np.abs(m - 0.5))[::-1]  # extreme heads first, mid heads on top
    segs = [np.column_stack([gx, jump(Y[k])]) for k in order]
    lc = LineCollection(segs, colors=[sf.pi_color(m[k]) for k in order], linewidths=0.35, alpha=0.12,
                        rasterized=True, zorder=1)
    ax.add_collection(lc)
    med = np.median(Y, axis=0)
    ax.plot(gx, jump(med), color=sf.SCA, lw=1.4, zorder=3)
    ax.plot([1.0, GATE, GATE, 0.10], [0, 0, 1, 1], color=sf.AREA, lw=1.4, ls=(0, (3.2, 1.6)), zorder=4)
    ax.text(0.10, 1.035, "AReA: every head", ha="right", va="bottom", fontsize=sf.FS_SMALL, color=sf.AREA)
    ax.text(0.10, 0.40, "SCA: one\nmix per head", ha="right", va="center", fontsize=sf.FS_SMALL, color=sf.SCA,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
    ax.set_xlim(1.0, 0.08)
    ax.set_ylim(-0.04, 1.16)
    ax.set_xticks([1.0, GATE, 0.6, 0.3, 0.1])
    ax.set_xticklabels(["1", "0.89", "0.6", "0.3", "0.1"])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"noise level $\sigma$")
    ax.set_ylabel(r"weight on the window, $\pi_h$")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.2)
    print(sf.save(fig, str(OUT), formats=("pdf",)))
    print(f"heads {len(keys)} | median of head means {np.median(m):.3f} | heads with mean < 0.3: "
          f"{int((m < 0.3).sum())}, > 0.9: {int((m > 0.9).sum())}")


if __name__ == "__main__":
    write_csv()
    draw()
