"""MOCK of the proposed Figure 2 concept panel (positioning proposal, 2026-11-05).

One schedule for every head (AReA, in SCA's notation) against one calibrated mix
per head (SCA), over the 28 sampling steps at 4096^2 on FLUX.1-dev. Every value
comes from calib/k1_1024.npz, the table behind every FLUX.1-dev result; the
per-step weight follows Eqs. (mix) and (sched) of the paper: global only above
the gate, then the window schedule 128, 112, 96, 80, 64 read at r = 256 / w,
linear in sigma between occupied bins and linear in log r between ratios 2 and 4.
This is a proposal mock, not a paper figure: the paper version belongs in
paper/figures/src/ with its numbers archived under results/heads_view/.

Run on a compute node:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I concept_mock.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "paper" / "figures" / "src"))
import scafig as sf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

z = np.load(REPO / "calib" / "k1_1024.npz", allow_pickle=False)
pop = z["populated"]
centres = z["sigma_centers"][pop]                 # increasing
P2, P4 = z["s2_pi_head"][:, :, pop], z["s4_pi_head"][:, :, pop]


def at_sigma(T: np.ndarray, s: float) -> np.ndarray:
    """Linear in sigma between occupied-bin centres, constant beyond them."""
    if s <= centres[0]:
        return T[..., 0]
    if s >= centres[-1]:
        return T[..., -1]
    j = int(np.searchsorted(centres, s) - 1)
    u = (s - centres[j]) / (centres[j + 1] - centres[j])
    return T[..., j] + u * (T[..., j + 1] - T[..., j])


# FLUX.1-dev, 28 Euler steps at maximum shift (mu = 1.15): sigma = e^mu / (e^mu + 1/t - 1).
steps = np.arange(1, 29)
t_lin = 1 - (steps - 1) / 28
sigma = np.exp(1.15) / (np.exp(1.15) + 1 / t_lin - 1)
GRID_SIDE, GATE = 256, 0.89


def window(s: float):
    if s > GATE:
        return None
    for lo, w in ((0.88, 128), (0.86, 112), (0.84, 96), (0.82, 80)):
        if s > lo:
            return w
    return 64


pi = np.zeros(P2.shape[:2] + (len(steps),))
for i, s in enumerate(sigma):
    w = window(s)
    if w is None:
        continue
    lam = float(np.clip(np.log2(GRID_SIDE / w) - 1, 0, 1))
    pi[..., i] = (1 - lam) * at_sigma(P2, s) + lam * at_sigma(P4, s)
heads = pi.reshape(-1, len(steps))
first_local = int(np.argmax(sigma <= GATE))       # index of step 9
assert steps[first_local] == 9, steps[first_local]
mean_after = heads[:, first_local:].mean(axis=1)

sf.use_style()
fig, ax = plt.subplots(figsize=(0.42 * sf.TEXT, 1.62))
ax.axvspan(0.5, first_local + 0.5, color=sf.GRID, lw=0, zorder=0)
ax.text((first_local + 1) / 2, 0.5, "global\nview", ha="center", va="center",
        fontsize=sf.FS_SMALL, color=sf.MID)
order = np.argsort(np.abs(mean_after - 0.5))[::-1]  # extremes drawn first, mid heads on top
gx = np.concatenate([steps[:first_local], [first_local + 0.5, first_local + 0.5], steps[first_local:]])
def jump(y):  # vertical jump at the gate, as the sampler switches in one step
    return np.concatenate([y[:first_local], [0.0, y[first_local]], y[first_local:]])
segs = [np.column_stack([gx, jump(heads[k])]) for k in order]
lc = LineCollection(segs, colors=[sf.pi_color(mean_after[k]) for k in order],
                    linewidths=0.35, alpha=0.12, rasterized=True, zorder=1)
ax.add_collection(lc)
med = np.median(heads, axis=0)
ax.plot(gx, jump(med), color=sf.SCA, lw=1.4, zorder=3)
area = (sigma <= GATE).astype(float)
ax.step(steps, area, where="mid", color=sf.AREA, lw=1.4, ls=(0, (3.2, 1.6)), zorder=4)
ax.text(28, 1.035, "AReA: every head", ha="right", va="bottom", fontsize=sf.FS_SMALL,
        color=sf.AREA)
ax.text(28, 0.40, "SCA: one\nmix per head", ha="right", va="center", fontsize=sf.FS_SMALL,
        color=sf.SCA, bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
ax.set_xlim(0.5, 28.5)
ax.set_ylim(-0.04, 1.16)
ax.set_xticks([1, 9, 13, 20, 28])
ax.set_yticks([0, 0.5, 1])
ax.set_xlabel("sampling step at $4096^2$")
ax.set_ylabel(r"weight on the window, $\pi_h$")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout(pad=0.2)
print(sf.save(fig, str(HERE / "concept_mock"), formats=("pdf", "png")))
print("heads", heads.shape[0], "| median weight after step 8:", round(float(np.median(mean_after)), 3),
      "| heads with mean weight < 0.3:", int((mean_after < 0.3).sum()), "> 0.9:", int((mean_after > 0.9).sum()))
