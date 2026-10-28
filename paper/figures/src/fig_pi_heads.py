"""What the heads want: per-head local weight pi_h(sigma) by depth, and its split-half stability.

(a-c) pi_h(sigma) of every head of the scale-2 table (FLUX.1-dev, 1024^2, 32 prompts), one panel
      per third of the depth: double-stream blocks 0-18, single-stream 19-37 and 38-56, 456 heads
      each. A head is local-heavy (amber) if its mean pi over the bins inference reads
      (sigma <= sigma_g) exceeds 0.8, global-heavy (blue) if below 0.2, otherwise grey; the green
      curve is the panel's median. Curves join the bin centres linearly, which is how inference
      reads the table; sigma > sigma_g runs the global branch only, so that region is veiled.
(d)   pi_h fitted on the even-numbered prompts against the odd-numbered ones, every populated
      (block, head, bin) cell, as a 2D histogram. Its correlation is the one Section 3 quotes.

Data (no GPU work; every number is recomputed and asserted):
  data/k1_1024_pi.npz, data/k1_1024.json   compact copy of the merged calibration (k1data.py)
  results/evidence/calib/k1_1024.npz       the archived merge; its table must equal the copy
  results/evidence/calib/k1_1024_split_half_stats.npz   per-half sufficient statistics
The pi fit is sca/calib.py's `fit` (clip(B / A, 0, 1), empty bins copy the nearest populated
bin), re-implemented here because sca/calib.py imports torch; refitting the summed halves must
reproduce the archived table exactly.

Regenerate: <sca env python> paper/figures/src/fig_pi_heads.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-identical PDF on rerun
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scafig as sf  # noqa: E402
from k1data import check, load  # noqa: E402

ROOT = HERE.parents[2]
EVID = ROOT / "results/evidence/calib"
OUT = ROOT / "paper/figures/pi_heads"

SCALE = 2            # the table Section 3 describes (its statistics are quoted at emulated scale 2)
SIGMA_G = 0.89       # gate: sigma > SIGMA_G uses the global branch only
LOCAL_HEAVY, GLOBAL_HEAVY = 0.8, 0.2  # thresholds on a head's mean pi over the read bins
GROUPS = [(0, 19, "double-stream 0–18"), (19, 38, "single-stream 19–37"),
          (38, 57, "single-stream 38–56")]
W, H = sf.TEXT, 1.22  # full text width (397.5 TeX pt) by 88.2 TeX pt
HIST_BINS = 40


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(b, a):  # sca/calib.py _ratio
    return np.clip(np.where(a > 0, b / np.where(a > 0, a, 1.0), 1.0), 0.0, 1.0)


def fill(pi, pop):  # sca/calib.py fill: copy the nearest populated bin into empty bins
    idx = np.flatnonzero(pop)
    near = idx[np.abs(np.arange(pop.size)[:, None] - idx[None]).argmin(1)]
    return pi[..., near]


def data():
    z, report = load()
    check(z, report)  # errors recomputed from the statistics agree with the merge report
    arch = np.load(EVID / "k1_1024.npz")
    halves = np.load(EVID / "k1_1024_split_half_stats.npz")
    i = list(arch["scales"]).index(SCALE)
    pi = z[f"s{SCALE}_pi_head"]                        # [57 blocks, 24 heads, 14 bins]
    pop, sig = z["populated"], z["sigma_centers"]
    S = arch["stats"][i]
    assert np.array_equal(pi, arch[f"s{SCALE}_pi_head"]), "compact copy differs from the archive"
    assert np.allclose(halves["even"][i] + halves["odd"][i], S, rtol=1e-12, atol=0), "halves do not sum"
    assert np.array_equal(fill(ratio(S[1], S[0]), pop), pi), "refit does not reproduce the table"
    even = fill(ratio(halves["even"][i][1], halves["even"][i][0]), pop)
    odd = fill(ratio(halves["odd"][i][1], halves["odd"][i][0]), pop)
    mask = (S[0] > 0) & pop[None, None]               # sca/calib.py merge(): the split-half mask
    r = float(np.corrcoef(even[mask], odd[mask])[0, 1])
    rep = report["scales"][str(SCALE)]
    assert abs(r - rep["split_half_corr"]["per_head"]) < 1e-12, (r, rep["split_half_corr"])

    # Section 3 statistics, recomputed from the table (populated bins only).
    cells = pi[:, :, pop]
    mid = float(((cells >= 0.2) & (cells <= 0.8)).mean())
    assert abs(mid - rep["frac_pi_mid"]) < 1e-12
    sd, mean = cells.std((0, 1)), cells.mean((0, 1))
    read = pop & (sig <= SIGMA_G)
    head_mean = pi[:, :, read].mean(-1)                # [57, 24]
    cls = np.where(head_mean > LOCAL_HEAVY, "local", np.where(head_mean < GLOBAL_HEAVY, "global", "other"))
    stats = dict(
        heads=int(pi.shape[0] * pi.shape[1]), populated_bins=int(pop.sum()),
        sd_over_heads_per_bin=[round(float(x), 4) for x in sd],
        sd_range=[round(float(sd.min()), 3), round(float(sd.max()), 3)],
        mean_range=[round(float(mean.min()), 3), round(float(mean.max()), 3)],
        frac_cells_in_0p2_0p8=round(mid, 4), frac_cells_above_0p95=round(float((cells > 0.95).mean()), 4),
        frac_cells_below_0p05=round(float((cells < 0.05).mean()), 4),
        split_half_r=r, split_half_cells=int(mask.sum()),
        split_half_mean_abs_diff=round(float(np.abs(even - odd)[mask].mean()), 4),
        split_half_median_abs_diff=round(float(np.median(np.abs(even - odd)[mask])), 4),
        frac_heads_local_heavy=round(float((cls == "local").mean()), 4),
        frac_heads_global_heavy=round(float((cls == "global").mean()), 4),
        block_mean_pi_read_bins=[round(float(x), 3) for x in pi[:, :, read].mean((1, 2))],
        local_heavy_per_block=[int(x) for x in (cls == "local").sum(1)],
        global_heavy_per_block=[int(x) for x in (cls == "global").sum(1)],
        groups={})
    for b0, b1, name in GROUPS:
        g, c = pi[b0:b1].reshape(-1, pi.shape[-1]), cls[b0:b1].ravel()
        stats["groups"][name] = dict(
            blocks=[b0, b1 - 1], heads=int(c.size),
            local_heavy=int((c == "local").sum()), global_heavy=int((c == "global").sum()),
            other=int((c == "other").sum()),
            median_curve_populated=[round(float(x), 3) for x in np.median(g, 0)[pop]],
            sd_range_populated=[round(float(g[:, pop].std(0).min()), 3), round(float(g[:, pop].std(0).max()), 3)])
    return dict(pi=pi, pop=pop, sig=sig, cls=cls, even=even[mask], odd=odd[mask], r=r, stats=stats,
                sources={"paper/figures/src/data/k1_1024_pi.npz": sha(HERE / "data/k1_1024_pi.npz"),
                         "paper/figures/src/data/k1_1024.json": sha(HERE / "data/k1_1024.json"),
                         "results/evidence/calib/k1_1024.npz": sha(EVID / "k1_1024.npz"),
                         "results/evidence/calib/k1_1024_split_half_stats.npz":
                             sha(EVID / "k1_1024_split_half_stats.npz"),
                         "results/evidence/calibration_split_half_audit.json":
                             sha(ROOT / "results/evidence/calibration_split_half_audit.json")})


def plot(d):
    sf.use_style()
    sf.script_floor(sf.FS_TICK)  # sigma_g tick labels are 7 pt: every script prints at 6 pt or more
    plt.rcParams["svg.hashsalt"] = "sca-pi-heads"
    pi, pop, sig, cls = d["pi"], d["pop"], d["sig"], d["cls"]
    x = sig[pop]
    left, right, bottom, top = 0.36, 0.06, 0.33, 0.15
    gap, gap_d = 0.10, 0.44
    ph = H - bottom - top
    pw = (W - left - right - 2 * gap - gap_d - ph) / 3
    fig = plt.figure(figsize=(W, H))
    axes = []
    for k in range(3):
        rect = [(left + k * (pw + gap)) / W, bottom / H, pw / W, ph / H]
        axes.append(fig.add_axes(rect, sharex=axes[0] if axes else None, sharey=axes[0] if axes else None))
    dax = fig.add_axes([(W - right - ph) / W, bottom / H, ph / W, ph / H])

    style = {"other": (sf.MID, 0.18, 0.35), "global": (sf.GLOBAL, 0.5, 0.45),
             "local": (sf.LOCAL, 0.2, 0.45)}
    halo = [patheffects.withStroke(linewidth=2.6, foreground="white")]
    for k, (ax, (b0, b1, name)) in enumerate(zip(axes, GROUPS)):
        g, c = pi[b0:b1].reshape(-1, pi.shape[-1])[:, pop], cls[b0:b1].ravel()
        for kind in ("other", "global", "local"):  # grey underneath, the two branch colours above
            col, alpha, lw = style[kind]
            for row in g[c == kind]:
                ax.plot(x, row, color=col, alpha=alpha, lw=lw, solid_joinstyle="round", zorder=2)
        ax.plot(x, np.median(g, 0), color=sf.SCA, lw=1.4, zorder=4, path_effects=halo,
                solid_capstyle="round")
        # sigma > sigma_g: global branch only, so inference never reads pi there
        ax.axvspan(SIGMA_G, 1.0, color="white", alpha=0.72, lw=0, zorder=5)
        ax.axvline(SIGMA_G, color=sf.RULE, lw=0.6, zorder=5)
        ax.set_xlim(0.08, 1.0)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0.1, 0.5, SIGMA_G], ["0.1", "0.5", r"$\sigma_g$"])
        ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if k:
            ax.tick_params(axis="y", labelleft=False)
        bb = ax.get_position()
        fig.text(bb.x0, bb.y1 + 0.05 / H, "abc"[k], fontsize=sf.FS_PANEL, fontweight="bold", va="baseline")
        fig.text(bb.x0 + 0.11 / W, bb.y1 + 0.05 / H, name, fontsize=sf.FS_TICK, color=sf.INK, va="baseline")
    # labelpad 1.5 (not 2.5): the 6.9 pt subscript deepens the label's descent, which matplotlib pins to
    # the tick labels; this keeps the label where it was, inside the left edge.
    axes[0].set_ylabel(r"local weight $\pi_h$", labelpad=1.5)
    axes[1].set_xlabel(r"noise level $\sigma$")

    # (d) split halves: 2D histogram over every populated cell, neutral grey, log counts
    edges = np.linspace(0, 1, HIST_BINS + 1)
    hist, _, _ = np.histogram2d(d["even"], d["odd"], bins=[edges, edges])
    cmap = LinearSegmentedColormap.from_list("grey", ["#E4E7EA", sf.INK])
    dax.pcolormesh(edges, edges, np.ma.masked_equal(hist.T, 0), cmap=cmap,
                   norm=LogNorm(vmin=1, vmax=hist.max()), shading="flat", lw=0, zorder=2)
    dax.plot([0, 1], [0, 1], color=sf.RULE, lw=0.6, zorder=1)
    dax.set_xlim(0, 1)
    dax.set_ylim(0, 1)
    dax.set_aspect("equal")
    dax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    dax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    for side in ("top", "right"):
        dax.spines[side].set_visible(False)
    dax.set_xlabel(r"$\pi_h$, even half")
    dax.set_ylabel(r"$\pi_h$, odd half", labelpad=1.5)
    dax.text(0.04, 0.96, f"r = {d['r']:.3f}", transform=dax.transAxes, ha="left", va="top",
             fontsize=sf.FS_TICK, color=sf.INK)
    bb = dax.get_position()
    fig.text(bb.x0, bb.y1 + 0.05 / H, "d", fontsize=sf.FS_PANEL, fontweight="bold", va="baseline")
    fig.text(bb.x0 + 0.11 / W, bb.y1 + 0.05 / H, "split halves", fontsize=sf.FS_TICK, color=sf.INK, va="baseline")
    return fig


def main():
    d = data()
    fig = plot(d)
    sf.save(fig, str(OUT), formats=("pdf",))
    provenance = dict(
        figure="paper/figures/pi_heads.pdf", script="paper/figures/src/fig_pi_heads.py",
        printed_size_in=[W, H], printed_size_texpt=[round(W * 72.27, 1), round(H * 72.27, 1)],
        table=f"per-head pi of the emulated scale-{SCALE} table, FLUX.1-dev, 1024x1024, 32 prompts, "
              "28 steps, guidance 4.5, mu = 1.15; 57 blocks x 24 heads x 14 sigma bins (13 populated)",
        curves="pi at the populated bin centres (sigma 0.105 to 0.969), joined linearly as inference "
               "interpolates; sigma > 0.89 veiled (global branch only at inference)",
        colour_rule=f"head mean pi over populated bins with centre <= {SIGMA_G}: > {LOCAL_HEAVY} amber "
                    f"(local-heavy), < {GLOBAL_HEAVY} blue (global-heavy), otherwise grey; green = panel median",
        split_half="pi refitted on even- and odd-indexed calibration prompts (sufficient statistics "
                   "summed per half); every populated cell with A > 0; Pearson r as in sca/calib.py",
        statistics=d["stats"], sources=d["sources"])
    OUT.with_name(OUT.name + "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    s = d["stats"]
    print(f"r = {s['split_half_r']:.6f} over {s['split_half_cells']} cells; SD {s['sd_range']}; "
          f"mean {s['mean_range']}; in [0.2,0.8] {s['frac_cells_in_0p2_0p8']}; >0.95 "
          f"{s['frac_cells_above_0p95']}; <0.05 {s['frac_cells_below_0p05']}")
    for name, g in s["groups"].items():
        print(f"{name}: local {g['local_heavy']}, global {g['global_heavy']}, other {g['other']}; "
              f"median {g['median_curve_populated'][0]} -> {g['median_curve_populated'][-1]}")


if __name__ == "__main__":
    main()
