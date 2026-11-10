"""Figure 4 (v6, paper/figures/v6/emulation.pdf): does the measurement hold?

FLUX.1-dev, reach emulation at 1024^2 on 32 emulation prompts (and 16 at 1280^2). Every number is read from the
archived statistics and recomputed here; nothing is typed.
(a) The fitted weight pi_h of the r = 2 table against the head's measured near share (share of its full-attention
    image mass within 16 tokens), all 1,368 heads at the noise bin centred at sigma = 0.62; Pearson rho
    (results/heads_view/pi_vs_mass.csv and summary.csv, written by fig_heads_view.py from k1_1024.npz; recomputed).
    Heads 4 and 7 of block 12 (Figure 2) in amber and blue.
(b) Relative output error sum J / sum ||O_full||^2 of four mixing rules at emulated ratios 2 and 4, pooled over blocks,
    heads and steps: per head (SCA), per layer, shared over heads, the best hard switch from far to near
    (k1data.errors on data/k1_1024_pi.npz, asserted against data/k1_1024.json).
(c) Transfer. Candidate A: tables fitted on disjoint prompt halves (2D histogram over every populated cell, Pearson
    rho; results/evidence/calib/k1_1024_split_half_stats.npz, as fig_pi_heads.py), then the r = 2 table scored at
    r = 4 against the r = 4 table, and the 1024^2 table scored at 1280^2 against the 1280^2 table
    (data/k1_1280_pi.npz, data/k1_1280.json; recomputed as fig_reach_tables.py).
    Candidate B: the same two transfers plus held-out prompts (each half's table scored on the other half), as bars.

  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I paper/figures/src/v6_emulation.py [--keep A|B]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scafig as sf  # noqa: E402
from k1data import check, errors, load, rel_err  # noqa: E402

ROOT = HERE.parents[2]
FIGS = ROOT / "paper" / "figures"
V6 = FIGS / "v6"
HV = ROOT / "results" / "heads_view"
EVID = ROOT / "results" / "evidence" / "calib"
BUILD = Path(os.environ.get("V6_BUILD", "/scratch/09032/saini_2/sca/paper_builds/v6_methodfigs_1")) / "emu"
W, H = sf.TEXT, 1.60
FS_T, FS_L, FS_S, FS_V = 8.5, 7.5, 7.0, 6.5
BIN, BLOCK, HEADS = 8, 11, (3, 6)
SCA_LIGHT = "#9AD3BA"
G_DARK, G_LIGHT = "#7D8893", "#C9CED3"
RULES = [("per_head", "per head"), ("per_layer", "per layer"), ("uniform", "shared"), ("switch", "hard switch")]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def ratio(b, a):
    return np.clip(np.where(a > 0, b / np.where(a > 0, a, 1.0), 1.0), 0.0, 1.0)


def fill(pi, pop):
    idx = np.flatnonzero(pop)
    near = idx[np.abs(np.arange(pop.size)[:, None] - idx[None]).argmin(1)]
    return pi[..., near]


def data() -> dict:
    z, report = load()
    check(z, report)
    e = {s: errors(z, s) for s in (2, 4)}
    for s in (2, 4):
        for k in ("per_head", "per_layer", "uniform", "switch"):
            assert abs(e[s][k] - report["scales"][str(s)]["rel_err"][k]) < 1e-12
    # (a) pi vs near share at bin 8 (sigma 0.62), r = 2 table
    rows = [r for r in csv.DictReader(open(HV / "pi_vs_mass.csv")) if int(r["bin"]) == BIN]
    x = np.array([float(r["share_r16"]) for r in rows])
    y = np.array([float(r["pi_s2"]) for r in rows])
    bh = [(int(r["block"]), int(r["head"])) for r in rows]
    rho = float(np.corrcoef(x, y)[0, 1])
    summ = [r for r in csv.DictReader(open(HV / "summary.csv")) if r["kind"] == "pearson" and r["scale"] == "2"
            and r["radius"] == "16" and r["bin"] == str(BIN)]
    assert len(summ) == 1 and abs(float(summ[0]["value"]) - rho) < 1e-5, (rho, summ)
    a = np.load(EVID / "k1_1024.npz")
    mm = a["mass_mean"][:, :, :, BIN]
    share = mm[4] / (1 - mm[0])
    assert np.allclose(share[[b for b, _ in bh], [h for _, h in bh]], x, atol=1e-6)
    assert np.allclose(z["s2_pi_head"][[b for b, _ in bh], [h for _, h in bh], BIN], y, atol=1e-6)
    hl = {h: bh.index((BLOCK, h)) for h in HEADS}
    # (c) split halves (fig_pi_heads.py logic)
    halves = np.load(EVID / "k1_1024_split_half_stats.npz")
    pop = z["populated"]
    i2 = list(a["scales"]).index(2)
    S = a["stats"][i2]
    assert np.allclose(halves["even"][i2] + halves["odd"][i2], S, rtol=1e-12, atol=0)
    even = fill(ratio(halves["even"][i2][1], halves["even"][i2][0]), pop)
    odd = fill(ratio(halves["odd"][i2][1], halves["odd"][i2][0]), pop)
    mask = (S[0] > 0) & pop[None, None]
    r_half = float(np.corrcoef(even[mask], odd[mask])[0, 1])
    assert abs(r_half - report["scales"]["2"]["split_half_corr"]["per_head"]) < 1e-12
    # transfer across resolution (fig_reach_tables.py logic)
    z12 = dict(np.load(HERE / "data" / "k1_1280_pi.npz"))
    r12 = json.load(open(HERE / "data" / "k1_1280.json"))
    res_own = float(rel_err(z12["s2_stats"], z12["s2_pi_head"]))
    res_tr = float(rel_err(z12["s2_stats"], z["s2_pi_head"]))
    assert abs(res_tr - r12["transfer"]["1024x1024/s2 -> 1280x1280/s2"]["rel_err"]) < 1e-9
    assert abs(res_own - r12["transfer"]["1280x1280/s2 -> 1280x1280/s2"]["rel_err"]) < 1e-9
    # transfer across ratio: the r = 2 table scored at r = 4 (k1data transfer_from)
    rat_own, rat_tr = e[4]["per_head"], e[4]["transfer_from"]
    assert abs(rat_tr - report["scales"]["4"]["rel_err"]["per_head"]) > 0  # sanity: differ
    half_own = report["scales"]["2"]["rel_err"]["per_head"]
    half_tr = report["scales"]["2"]["cross_half_rel_err_per_head"]
    return dict(e=e, x=x, y=y, rho=rho, hl=hl, even=even[mask], odd=odd[mask], r_half=r_half,
                res=(res_own, res_tr), rat=(rat_own, rat_tr), half=(half_own, half_tr),
                n_prompts=(int(z["n_prompts"]), int(r12["n_prompts"])))


# ------------------------------------------------------------------------------------------------------------ draw
def axes_in(fig, x, y, w, h, **kw):
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def title(fig, x, y, letter, name):
    fig.text(x / W, y / H, f"({letter}) {name}", fontsize=FS_T, fontweight="bold", ha="left", va="baseline")


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=FS_S, length=2.2, width=0.5, pad=1.6)


def panel_scatter(fig, d, x, y, s):
    ax = axes_in(fig, x, y, s, s)
    ax.plot([0, 1], [0, 1], color=sf.RULE, lw=0.6, ls=(0, (2.5, 1.5)), zorder=1)
    ax.scatter(d["x"], d["y"], s=2.2, c="#8E979F", alpha=0.55, lw=0, zorder=2, rasterized=True)
    for h, col in zip(HEADS, (sf.LOCAL, sf.GLOBAL)):
        i = d["hl"][h]
        ax.scatter([d["x"][i]], [d["y"][i]], s=16, c=col, ec="white", lw=0.6, zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    clean(ax)
    ax.set_xlabel("near share", fontsize=FS_L, labelpad=1.5)
    ax.set_ylabel(r"fitted weight $\pi_h$", fontsize=FS_L, labelpad=1.5)
    ax.text(0.97, 0.05, rf"$\rho={d['rho']:.2f}$", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS_L)
    return ax


def panel_rules_grouped(fig, d, x, y, w, h):
    """(b) A: one panel, rules as rows, r = 2 (light) and r = 4 (dark) bars per rule."""
    ax = axes_in(fig, x, y, w, h)
    n = len(RULES)
    yc = np.arange(n)[::-1].astype(float)
    bh = 0.36
    for s, off, light in ((2, bh / 2 + 0.01, True), (4, -bh / 2 - 0.01, False)):
        vals = [d["e"][s][k] for k, _ in RULES]
        cols = [(SCA_LIGHT if light else sf.SCA) if k == "per_head" else (G_LIGHT if light else G_DARK)
                for k, _ in RULES]
        ax.barh(yc + off, vals, height=bh, color=cols, lw=0, zorder=2)
        best = min(vals)
        for yy, v in zip(yc + off, vals):
            ax.text(v + 0.006, yy, f"{v:.3f}", va="center", ha="left", fontsize=FS_V,
                    fontweight="bold" if v == best else "normal", color=sf.INK)
    ax.set_yticks(yc, [lab for _, lab in RULES])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlim(0, 0.335)
    ax.set_xticks([0, 0.1, 0.2, 0.3], ["0", "0.1", "0.2", "0.3"])
    ax.set_ylim(-0.6, n - 0.4)
    sf.value_grid(ax, "x")
    clean(ax)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(r"relative output error $\downarrow$", fontsize=FS_L, labelpad=1.5)
    # ratio key, top right: light = r 2, dark = r 4
    for k, (lab, col) in enumerate(((r"$r=2$", G_LIGHT), (r"$r=4$", G_DARK))):
        ax.add_patch(plt.Rectangle((0.17 + k * 0.085, n - 0.30), 0.016, 0.22, color=col, lw=0, clip_on=False))
        ax.text(0.19 + k * 0.085, n - 0.19, lab, fontsize=FS_S, va="center", ha="left")
    return ax


def panel_rules_split(fig, d, x, y, w, h, gap=0.10):
    """(b) B: two panels, r = 2 and r = 4, four bars each."""
    pw = (w - gap) / 2
    axes = []
    n = len(RULES)
    yc = np.arange(n)[::-1].astype(float)
    for j, s in enumerate((2, 4)):
        ax = axes_in(fig, x + j * (pw + gap), y, pw, h)
        vals = [d["e"][s][k] for k, _ in RULES]
        cols = [sf.SCA if k == "per_head" else G_DARK for k, _ in RULES]
        ax.barh(yc, vals, height=0.62, color=cols, lw=0, zorder=2)
        best = min(vals)
        for yy, v in zip(yc, vals):
            ax.text(v + max(vals) * 0.03, yy, f"{v:.3f}", va="center", ha="left", fontsize=FS_V,
                    fontweight="bold" if v == best else "normal")
        ax.set_xlim(0, max(vals) * 1.42)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks(yc, [lab for _, lab in RULES] if j == 0 else [])
        ax.tick_params(axis="y", length=0, pad=3)
        sf.value_grid(ax, "x")
        clean(ax)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([0, 0.1] if s == 2 else [0, 0.2], ["0", "0.1"] if s == 2 else ["0", "0.2"])
        ax.text(0.0, n - 0.35, rf"$r={s}$", fontsize=FS_S, ha="left", va="bottom", transform=ax.transData)
        axes.append(ax)
    fig.text((x + w / 2) / W, 0.04 / H, r"relative output error $\downarrow$", ha="center", va="bottom",
             fontsize=FS_L)
    return axes


def panel_hist(fig, d, x, y, s):
    ax = axes_in(fig, x, y, s, s)
    edges = np.linspace(0, 1, 41)
    hist, _, _ = np.histogram2d(d["even"], d["odd"], bins=[edges, edges])
    cmap = LinearSegmentedColormap.from_list("grey", ["#E4E7EA", sf.INK])
    ax.pcolormesh(edges, edges, np.ma.masked_equal(hist.T, 0), cmap=cmap, norm=LogNorm(vmin=1, vmax=hist.max()),
                  shading="flat", lw=0, zorder=2, rasterized=True)
    ax.plot([0, 1], [0, 1], color=sf.RULE, lw=0.6, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1], ["0", "1"])
    ax.set_yticks([0, 1], ["0", "1"])
    clean(ax)
    ax.set_xlabel(r"$\pi_h$, half 1", fontsize=FS_L, labelpad=0.5)
    ax.set_ylabel(r"$\pi_h$, half 2", fontsize=FS_L, labelpad=0.5)
    ax.text(0.05, 0.95, rf"$\rho={d['r_half']:.3f}$", transform=ax.transAxes, ha="left", va="top", fontsize=FS_S)
    return ax


def pair_rows(fig, d, x, y, w, h, items, xmax=None, key=True):
    """Own table (green) against a transferred table (light green), one bar pair per transfer, on one shared
    relative-error axis (so pairs are comparable with each other and with panel b); values printed."""
    ax = axes_in(fig, x, y, w, h)
    n = len(items)
    top = xmax or max(max(o, t) for _, o, t, _ in items) * 1.35
    ax.set_xlim(0, top)
    ax.set_ylim(-0.02, n)
    for i, (lab, own, tr, fmt) in enumerate(items):
        y0 = n - 1 - i
        for k, (v, col) in enumerate(((own, sf.SCA), (tr, SCA_LIGHT))):
            yy = y0 + 0.43 - k * 0.27
            ax.barh([yy], [v], height=0.24, color=col, lw=0, zorder=2)
            ax.text(v + top * 0.025, yy, fmt.format(v), fontsize=FS_V, va="center", ha="left",
                    fontweight="bold" if k == 0 else "normal")
        ax.text(0, y0 + 0.64, lab, fontsize=FS_S, va="bottom", ha="left")
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=FS_S, length=2.2, width=0.5, pad=1.6)
    ticks = [0, 0.1, 0.2] if top > 0.2 else [0, 0.05]
    ax.set_xticks(ticks, ["0"] + [f"{t:g}" for t in ticks[1:]])
    sf.value_grid(ax, "x")
    if key:
        for k, (lab, col) in enumerate((("own table", sf.SCA), ("transferred", SCA_LIGHT))):
            xx = k * 0.50
            ax.add_patch(plt.Rectangle((top * xx, n + 0.02), top * 0.06, 0.20, color=col, lw=0, clip_on=False))
            ax.text(top * (xx + 0.09), n + 0.12, lab, fontsize=FS_S, va="center", ha="left")
    return ax


def render(d, cand: str):
    sf.use_style()
    sf.script_floor(FS_S)
    fig = plt.figure(figsize=(W, H))
    ty = 1.49
    s = 1.02
    title(fig, 0.0, ty, "a", "Weight tracks reach")
    panel_scatter(fig, d, 0.38, 0.33, s)
    if cand == "A":
        title(fig, 1.62, ty, "b", "Per-head fits best")
        panel_rules_grouped(fig, d, 2.24, 0.33, 1.12, 1.02)
        title(fig, 3.66, ty, "c", "One table transfers")
        panel_hist(fig, d, 3.92, 0.42, 0.74)
        pair_rows(fig, d, 4.82, 0.33, 0.60, 0.96, [
            (r"$r=2\to4$", d["rat"][0], d["rat"][1], "{:.3f}"),
            (r"$1024^2\to1280^2$", d["res"][0], d["res"][1], "{:.4f}")], key=False)
    elif cand == "C":
        s = 0.94
        panel_scatter_c = panel_scatter
        fig.clf()
        title(fig, 0.0, ty, "a", "Weight tracks reach")
        panel_scatter_c(fig, d, 0.36, 0.33, s)
        title(fig, 1.50, ty, "b", "Per-head fits best")
        panel_rules_split(fig, d, 2.10, 0.36, 1.36, 0.98)
        title(fig, 3.62, ty, "c", "One table transfers")
        panel_hist(fig, d, 3.86, 0.42, 0.72)
        pair_rows(fig, d, 4.80, 0.33, 0.64, 0.96, [
            (r"$r=2\to4$", d["rat"][0], d["rat"][1], "{:.3f}"),
            (r"$1024^2\to1280^2$", d["res"][0], d["res"][1], "{:.4f}")], key=False)
    else:
        title(fig, 1.62, ty, "b", "Per-head fits best")
        panel_rules_split(fig, d, 2.24, 0.36, 1.50, 0.98)
        title(fig, 4.02, ty, "c", "One table transfers")
        pair_rows(fig, d, 4.06, 0.33, 1.38, 1.02, [
            (rf"held-out prompts, $\rho={d['r_half']:.3f}$", d["half"][0], d["half"][1], "{:.4f}"),
            (r"ratio-2 table at $r=4$", d["rat"][0], d["rat"][1], "{:.3f}"),
            (r"$1024^2$ table at $1280^2$", d["res"][0], d["res"][1], "{:.4f}")], xmax=0.30, key=False)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keep", choices=("A", "B", "C"), default="B")
    a = ap.parse_args()
    d = data()
    BUILD.mkdir(parents=True, exist_ok=True)
    V6.mkdir(parents=True, exist_ok=True)
    for cand in ("A", "B", "C"):
        fig = render(d, cand)
        fig.savefig(BUILD / f"emulation_{cand}.pdf", dpi=600)
        plt.close(fig)
    fig = render(d, a.keep)
    fig.savefig(V6 / "emulation.pdf", dpi=600)
    plt.close(fig)
    vals = {"rho_pi_vs_share": d["rho"], "split_half_rho": d["r_half"],
            "rel_err": {str(s): {k: d["e"][s][k] for k in ("per_head", "per_layer", "uniform", "switch",
                                                            "transfer_from")} for s in (2, 4)},
            "ratio_transfer_own_vs_r2_table_at_r4": list(d["rat"]),
            "resolution_transfer_own1280_vs_1024table": list(d["res"]),
            "held_out_half_insample_vs_cross": list(d["half"]), "n_prompts_1024_1280": list(d["n_prompts"])}
    ins = [HV / "pi_vs_mass.csv", HV / "summary.csv", EVID / "k1_1024.npz", EVID / "k1_1024_split_half_stats.npz",
           HERE / "data" / "k1_1024_pi.npz", HERE / "data" / "k1_1024.json", HERE / "data" / "k1_1280_pi.npz",
           HERE / "data" / "k1_1280.json"]
    prov = {"figure": "Figure 4 (emulation), v6", "script": rel(Path(__file__)), "kept": a.keep, "size_in": [W, H],
            "inputs": {rel(p): sha256(p) for p in ins}, "values": vals,
            "output_sha256": sha256(V6 / "emulation.pdf")}
    (V6 / "emulation_provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(json.dumps(vals, indent=1))


if __name__ == "__main__":
    main()
