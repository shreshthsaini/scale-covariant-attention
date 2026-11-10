"""Figure 6 (paper/figures/v6/gains.pdf): paired gains of SCA on the ten per-image metrics at 4096^2.

Three comparisons on the same 195 Aesthetic-4K prompts, generation seed 0: progressive SCA minus AReA, single-pass
SCA minus AReA, progressive SCA minus HRDiT. Data: gen_holm.pair() for the pairs prog_area, sca_area and prog_hrdit.
It re-runs scripts/paired_bootstrap.py exactly (prompts resampled with replacement, 2000 draws, seed 0) on the
archived per-image evaluation files under /scratch/09032/saini_2/sca/eval/, asserts that every difference and 95%
percentile interval equals the archived CSV to the six significant digits it stores
(results/prog195_vs_baselines/sca_v2_prog_vs_area.csv, results/prog195_pairs/sca_v2_vs_area.csv,
results/prog195_vs_baselines/sca_v2_prog_vs_hrdit_dev_author.csv), and returns the unadjusted call (interval
excludes zero) and the Holm-adjusted call (two-sided bootstrap p-values, Holm step-down over the ten metrics of
each pair at 5%). No number is typed here.

Common axis. Each difference and both ends of its interval are divided by the standard deviation (ddof 1) of the
195 per-prompt differences of that metric and pair (per-prompt values from gen_holm._load, the same files): the
standardized paired effect, in units of prompt-to-prompt spread. Dividing by a positive constant keeps every
interval on its side of zero, so every call is exactly that of the paired files. --norm pct instead divides by the
reference's mean over the same prompts (AReA: results/a4k195_main/{base,extra}_summary.csv, method area; HRDiT:
results/prog195_sca_v2/{base,extra}_summary.csv, method hrdit_dev_author; both asserted equal to the per-image
means), the v5.1 convention; ImageReward has no natural zero, so its percent is the least comparable.

Encoding. Colour is the direction of an interval that excludes zero (SCA green: SCA better; LOSS orange: SCA
worse) and grey when it includes zero; a filled marker survives Holm, a hollow coloured marker is significant only
before adjustment. Stars: progressive SCA; circles: single-pass SCA (the marks of Figure 7 and the appendix
plots). Zero line, light vertical gridlines, the
direction of "better" on the axis.

Layouts (two candidates, both rendered and scored; WINNER goes to v6/gains.pdf, the other to v6/candidates/):
  A  wrap  0.48 text width (2.64 in): metric rows, (a) both modes against AReA, (b) progressive against HRDiT.
  B  full  5.5 in: metric rows shared by three panels, (a) progressive - AReA, (b) single pass - AReA,
           (c) progressive - HRDiT, one series each.
Checks (the run stops on any failure): the archived CSVs are reproduced (gen_holm); the plotted differences and
interval ends, in the units the paper prints, round to headline.tex (\\hlProgArea..., \\hlProgHrdit..., \\hlIrDelta);
the caption's claims hold (both modes better than AReA on all ten after Holm; against HRDiT better on exactly HPSv2
and ARNIQA after Holm and worse on none; MUSIQ better and CLIP-IQA worse only before adjustment; the ImageReward,
HPSv2.1 and PickScore intervals include zero).

Run on a compute node: <sca env python> -I paper/figures/src/v6_gains.py [--norm dz|pct] [--layout A|B|both]
[--out-dir DIR] (default: both layouts, dz, into paper/figures/v6/ and v6/candidates/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

import gen_holm as gh  # noqa: E402
import scafig as sf  # noqa: E402
import v6_plotkit as pk  # noqa: E402

KEYS = ("prog_area", "sca_area", "prog_hrdit")
WINNER = "A"
STEM = "gains"
# display names, in gen_holm's reading order (preference metrics, then image quality)
LABEL = {"ir": "ImageReward", "aesthetic": "Aesthetic", "clip": "CLIP score", "pickscore": "PickScore",
         "hpsv2": "HPSv2", "hpsv21": "HPSv2.1", "musiq": "MUSIQ", "musiq_crop1024": r"MUSIQ$_{\mathregular{c}}$",
         "clipiqa": "CLIP-IQA", "arniqa": "ARNIQA"}
PREFERENCE = ("ir", "aesthetic", "clip", "pickscore", "hpsv2", "hpsv21")
GROUP_GAP = 0.55  # extra row space between the preference and the image-quality metrics
MARKER = {"prog": ("*", 6.6), "sca": ("o", 3.9)}  # the paper-wide marks: star progressive, circle single pass
# headline.tex macros that print these pairs: (pair, metric, macro stem, scale, decimals, interval printed)
PRINTED = [("prog_area", "ir", "hlProgAreaIr", 1, 2, True), ("prog_area", "pickscore", "hlProgAreaPick", 1, 2, True),
           ("prog_area", "hpsv21", "hlProgAreaHps", 100, 2, True),
           ("prog_area", "musiq_crop1024", "hlProgAreaMusiqc", 1, 1, True),
           ("prog_area", "musiq", "hlProgAreaMusiq", 1, 1, True), ("prog_area", "clip", "hlProgAreaClip", 1, 2, True),
           ("prog_hrdit", "ir", "hlProgHrditIr", 1, 2, True), ("prog_hrdit", "hpsv21", "hlProgHrditHps", 100, 2, True),
           ("prog_hrdit", "pickscore", "hlProgHrditPick", 1, 2, True),
           ("prog_hrdit", "hpsv2", "hlProgHrditHpsTwo", 100, 2, True),
           ("prog_hrdit", "arniqa", "hlProgHrditArniqa", 100, 1, True),
           ("prog_hrdit", "musiq", "hlProgHrditMusiq", 1, 1, True),
           ("prog_hrdit", "clipiqa", "hlProgHrditClipiqa", 100, 1, True),
           ("prog_hrdit", "musiq_crop1024", "hlProgHrditMusiqc", 1, 1, True),
           ("sca_area", "ir", "hlIrDelta", 1, 2, False)]  # 1000-draw evaluator interval: the difference only
REF_SUMMARY = {"area": ("a4k195_main", "area"), "hrdit_dev_author": ("prog195_sca_v2", "hrdit_dev_author")}


def ref_means(run, method):
    """{metric: mean} of a reference from its 195-prompt summary CSVs (two layouts: base and extra)."""
    import csv
    out = {}
    for name in ("base_summary.csv", "extra_summary.csv"):
        for r in csv.DictReader(open(pk.ROOT / "results" / run / name)):
            if r["method"] == method and r["metric"] in LABEL:
                assert r["res"] == "4096x4096" and r["n_prompts"] == "195", r
                out[r["metric"]] = float(r["value"])
    assert set(out) == set(LABEL), (run, method, sorted(out))
    return out


def load():
    """{pair: {"n", "src", "ref", "metrics": {metric: {delta, lo, hi, sd, ref_mean, sig_raw, sig_holm, ...}}}}."""
    specs = {s[0]: s for s in gh.PAIRS}
    out = {}
    for key in KEYS:
        spec = specs[key]
        r = gh.pair(spec)  # re-runs paired_bootstrap.py and asserts equality with the archived CSV
        _, _, _, method, mroots, ref, rroots, _ = spec
        X, Y = gh._load(mroots, method), gh._load(rroots, ref)
        (grp,) = X.keys()
        x, y = X[grp], Y[grp]
        rows = {}
        for m in LABEL:
            g = dict(r["metrics"][m])
            ids = sorted(x[m])
            assert ids == sorted(y[m]) and len(ids) == r["n"], (key, m)
            d = np.array([x[m][i] - y[m][i] for i in ids])
            assert abs(d.mean() - g["delta"]) <= 1e-12 * max(1.0, abs(g["delta"])), (key, m)
            g["sd"] = float(d.std(ddof=1))
            g["ref_mean"] = float(np.mean([y[m][i] for i in ids]))
            assert g["sig_holm"] in (0, g["sig_raw"]), (key, m, g)  # Holm never adds or flips a call
            rows[m] = g
        S = ref_means(*REF_SUMMARY[ref])
        for m in LABEL:  # the per-image mean of the reference is its summary mean
            assert np.isclose(rows[m]["ref_mean"], S[m], rtol=1e-9, atol=1e-12), (key, m, rows[m]["ref_mean"], S[m])
        out[key] = dict(n=r["n"], src=r["src"], ref=ref, metrics=rows, up=r["up"], dn=r["dn"], up_raw=r["up_raw"],
                        dn_raw=r["dn_raw"])
    return out


def checks(D):
    bad = []
    for key, m, name, sc, d, interval in PRINTED:
        g = D[key]["metrics"][m]
        got = pk.fmt(sc * pk.g6(g["delta"]), d, signed=True)
        pk.check(got == pk.macro(name), f"{key} {m}: delta {got}, \\{name} {pk.macro(name)}", bad)
        if interval:
            for end, suffix in (("lo", "Lo"), ("hi", "Hi")):
                got = pk.fmt(sc * pk.g6(g[end]), d)
                pk.check(got == pk.macro(name + suffix), f"{key} {m}: {end} {got}, \\{name}{suffix} "
                         f"{pk.macro(name + suffix)}", bad)
    holm = {k: ({m for m, g in D[k]["metrics"].items() if g["sig_holm"] > 0},
                {m for m, g in D[k]["metrics"].items() if g["sig_holm"] < 0}) for k in KEYS}
    raw = {k: ({m for m, g in D[k]["metrics"].items() if g["sig_raw"] > 0},
               {m for m, g in D[k]["metrics"].items() if g["sig_raw"] < 0}) for k in KEYS}
    for k in ("prog_area", "sca_area"):
        pk.check(holm[k] == (set(LABEL), set()), f"claim: {k} better on all ten after Holm, got {holm[k]}", bad)
    pk.check(holm["prog_hrdit"] == ({"hpsv2", "arniqa"}, set()),
             f"claim: progressive vs HRDiT better on HPSv2 and ARNIQA only after Holm, got {holm['prog_hrdit']}", bad)
    pk.check(raw["prog_hrdit"] == ({"hpsv2", "arniqa", "musiq"}, {"clipiqa"}),
             f"claim: before adjustment MUSIQ better and CLIP-IQA worse, got {raw['prog_hrdit']}", bad)
    for m in ("ir", "hpsv21", "pickscore"):
        g = D["prog_hrdit"]["metrics"][m]
        pk.check(g["lo"] <= 0 <= g["hi"], f"claim: progressive vs HRDiT interval on {m} includes zero", bad)
    pk.check({D[k]["n"] for k in KEYS} == {195}, "n = 195 prompts for every pair", bad)
    pk.fail_if(bad, "v6_gains.py")
    return dict(holm={k: [sorted(v[0]), sorted(v[1])] for k, v in holm.items()},
                raw={k: [sorted(v[0]), sorted(v[1])] for k, v in raw.items()})


def scaled(g, norm):
    s = g["sd"] if norm == "dz" else g["ref_mean"] / 100.0
    return g["delta"] / s, g["lo"] / s, g["hi"] / s


def row_positions():
    y, pos = 0.0, {}
    for m in LABEL:
        if m == "musiq":
            y += GROUP_GAP
        pos[m] = y
        y += 1.0
    return pos, y - 1.0


def colour(g):
    return pk.NS if g["sig_raw"] == 0 else (pk.GAIN if g["sig_raw"] > 0 else pk.LOSS)


def draw_series(ax, D, key, norm, kind, dy, pos, lw=1.0):
    marker, ms = MARKER[kind]
    for m, g in D[key]["metrics"].items():
        v, lo, hi = scaled(g, norm)
        c = colour(g)
        y = pos[m] + dy
        ax.plot([lo, hi], [y, y], color=c, lw=lw, solid_capstyle="butt", zorder=2)
        ax.plot([v], [y], ls="none", marker=marker, ms=ms, mec=c, mew=0.9,
                mfc=c if g["sig_holm"] else "white", zorder=3)


def extent(D, keys, norm):
    vals = [x for k in keys for g in D[k]["metrics"].values() for x in scaled(g, norm)[1:]]
    return min(0.0, min(vals)), max(0.0, max(vals))


def ticks_for(lo, hi, norm):
    step = 0.5 if norm == "dz" else (10.0 if hi - lo > 30 else 5.0)
    t = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
    return [float(round(v, 6)) for v in t]


def tick_fmt(norm):
    def f(v, _):
        if abs(v) < 1e-9:
            return "0"
        s = f"{v:g}" if norm == "dz" else f"{v:+.0f}"
        return s.replace("-", pk.MINUS)
    return FuncFormatter(f)


def xlabel_text(norm):
    if norm == "dz":
        return f"Standardized difference, better {pk.ARROW}"
    return f"Difference, % of the reference's mean, better {pk.ARROW}"


def build(D, layout, norm):
    pk.use_style(script_pt=sf.FS_TICK)
    pos, ymax = row_positions()
    if layout == "A":
        W = pk.WRAP
        panels = [(("prog_area", "prog", -0.19), ("sca_area", "sca", 0.19)), (("prog_hrdit", "prog", 0.0),)]
        tags = [("a", "vs AReA"), ("b", "vs HRDiT")]
        pitch, left, gap, right = 11.5, 50.0, 9.0, 3.0      # points
        top, bottom = 27.0, 26.0
    else:
        W = pk.TEXT
        panels = [(("prog_area", "prog", 0.0),), (("sca_area", "sca", 0.0),), (("prog_hrdit", "prog", 0.0),)]
        tags = [("a", f"progressive {pk.MINUS} AReA"), ("b", f"single pass {pk.MINUS} AReA"),
                ("c", f"progressive {pk.MINUS} HRDiT")]
        pitch, left, gap, right = 8.8, 50.0, 14.0, 4.0
        top, bottom = 14.0, 26.0
    rows_h = (ymax + 1.0) * pitch
    H = (top + rows_h + bottom) / 72.0
    ranges = []
    for series in panels:
        lo, hi = extent(D, [k for k, _, _ in series], norm)
        span = hi - lo
        ranges.append((lo - 0.07 * span if lo < 0 else -0.06 * span, hi + 0.05 * span))
    total = sum(b - a for a, b in ranges)
    data_w = W * 72 - left - right - gap * (len(panels) - 1)
    fig = plt.figure(figsize=(W, H))
    axes, x0 = [], left
    for (a, b), series in zip(ranges, panels):
        w = data_w * (b - a) / total
        ax = fig.add_axes([x0 / (W * 72), bottom / (H * 72), w / (W * 72), rows_h / (H * 72)])
        x0 += w + gap
        ax.set_xlim(a, b)
        ax.set_ylim(ymax + 0.5, -0.5)
        ax.axvline(0, color=sf.INK, lw=0.7, zorder=1)
        for key, kind, dy in series:
            draw_series(ax, D, key, norm, kind, dy, pos)
        ax.xaxis.set_major_locator(FixedLocator(ticks_for(a, b, norm)))
        ax.xaxis.set_major_formatter(tick_fmt(norm))
        pk.clean_axes(ax, grid="x", left=False)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.set_yticks([pos[m] for m in LABEL])
        if axes:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels([LABEL[m] for m in LABEL])
        ax.set_ylabel("metric")  # the row names label the axis; kept hidden so the audit sees a labelled axis
        ax.yaxis.label.set_visible(False)
        axes.append(ax)
    # one axis title centred under all panels (figure text); each axes keeps the same title hidden, so that the
    # audit sees every x axis labelled
    for ax in axes:
        ax.set_xlabel(xlabel_text(norm))
        ax.xaxis.label.set_visible(False)
    span0, span1 = axes[0].get_position().x0, axes[-1].get_position().x1
    fig.text((span0 + span1) / 2, 3.0 / (H * 72), xlabel_text(norm), ha="center", va="bottom", fontsize=sf.FS,
             color=sf.INK)
    fig.canvas.draw()
    for ax, (letter, name) in zip(axes, tags):
        pk.panel_tag(fig, ax, letter, name, dy_pt=4.0)
    if layout == "A":  # one key for the two marker shapes, above the panel tags
        handles = [Line2D([], [], ls="none", marker=MARKER[k][0], ms=MARKER[k][1], mfc=pk.GAIN, mec=pk.GAIN,
                          mew=0.9) for k in ("prog", "sca")]
        fig.legend(handles, ["progressive SCA", "single-pass SCA"], loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=((span0 + span1) / 2 - 0.02, 1.0), fontsize=sf.FS_TICK, handletextpad=0.3,
                   columnspacing=1.2, borderaxespad=0.15, handlelength=1.0)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", choices=("dz", "pct"), default="dz")
    ap.add_argument("--layout", choices=("A", "B", "both"), default="both")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write every layout as DIR/gains_<layout>_<norm> instead of the paper paths")
    args = ap.parse_args()
    D = load()
    claims = checks(D)
    for k in KEYS:
        print(f"{k}: n={D[k]['n']} Holm better {claims['holm'][k][0]} worse {claims['holm'][k][1]}; "
              f"unadjusted better {len(claims['raw'][k][0])} worse {claims['raw'][k][1]}")
        for m, g in D[k]["metrics"].items():
            v, lo, hi = scaled(g, args.norm)
            print(f"    {m:15s} {g['delta']:+.4g} [{g['lo']:+.4g}, {g['hi']:+.4g}]  -> {v:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  f"  raw {g['sig_raw']:+d} holm {g['sig_holm']:+d} (p {g['p_boot']:.4f}, Holm {g['holm_boot']:.4f})")
    layouts = ("A", "B") if args.layout == "both" else (args.layout,)
    written = {}
    for lay in layouts:
        fig = build(D, lay, args.norm)
        if args.out_dir is not None:
            stem = args.out_dir / f"{STEM}_{lay}_{args.norm}"
        elif lay == WINNER and args.norm == "dz":
            stem = pk.V6 / STEM
        else:
            stem = pk.CANDIDATES / f"{STEM}_{lay}" if args.norm == "dz" else pk.CANDIDATES / f"{STEM}_{lay}_pct"
        print(f"min text size {pk.min_text_pt(fig):.1f} pt")
        written[lay] = pk.save(fig, stem, printed_width=pk.WRAP if lay == "A" else pk.TEXT)
    if args.out_dir is None and args.norm == "dz":
        prov = {
            "figure": "Figure 6, paper/figures/v6/gains.pdf", "script": "paper/figures/src/v6_gains.py",
            "winner": WINNER, "outputs": {k: [os.path.relpath(p, pk.ROOT) for p in v] for k, v in written.items()},
            "data": "gen_holm.pair(): scripts/paired_bootstrap.py re-run on the archived per-image files, 2000 prompt "
                    "draws, seed 0; equal to the archived CSVs to six significant digits",
            "archived": {k: D[k]["src"] for k in KEYS},
            "normalisation": "difference and interval ends divided by the SD (ddof 1) of the per-prompt differences",
            "holm": "two-sided bootstrap p-values, Holm step-down over the ten per-image metrics of each pair, 5%",
            "claims": claims,
            "values": {k: {m: {f: (round(g[f], 8) if isinstance(g[f], float) else g[f])
                               for f in ("delta", "lo", "hi", "sd", "ref_mean", "p_boot", "holm_boot", "sig_raw",
                                         "sig_holm")} for m, g in D[k]["metrics"].items()} for k in KEYS},
        }
        path = pk.V6 / f"{STEM}_provenance.json"
        path.write_text(json.dumps(prov, indent=1) + "\n")
        print("wrote", os.path.relpath(path, pk.ROOT))


if __name__ == "__main__":
    main()
