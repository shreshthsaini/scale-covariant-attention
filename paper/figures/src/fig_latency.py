"""Seconds per image against resolution for full attention, AReA and SCA (batch 1, one GH200).

Input, one of:
  --csv PATH   columns res,method,seconds; res like 4096x4096, one row per image (the median
               over rows of a (res, method) pair is plotted). Default:
               results/latency_curve.csv (the verified clean batch-1 sweep)
  results/latency_4096_batch1.csv (repo), used when the curve CSV does not exist or with
               --source 4096. Adapter: full and AReA from run "latency", SCA final (no band
               guard, code 2d11740) from run "latency3"; the band-guard SCA runs are dropped.
The lines and filled markers are clean batch-1 measurements only.

Campaign means are a separate, marked layer: hollow markers at 4096^2 for SCA, HRDiT and
UltraImage, read from results/baseline46/base_summary.csv (metric "seconds", the same 46 prompts
for every method, run on shared nodes). They are never joined to the clean curves. UltraImage's
191-prompt campaign mean (results/e1_ultraimage_base_summary.csv) is only recorded in the provenance.
A clean batch-1 HRDiT timing is queued but not measured; it is not plotted.

With several resolutions the figure is one log-log panel:
  median seconds per image against resolution, whiskers = observed min and max, a light fill
  between AReA and SCA, the least-squares exponent b of seconds ~ N^b (N = image tokens,
  (W/16)(H/16)) fitted to each method's medians, SCA's speedup over AReA as a top row, the
  campaign means as hollow markers, and a note on why full attention stops at 4096^2.
With one resolution it is a bar chart. Speedups and exponents are printed, and the caption
snippet figures/snippets/latency.tex is written from the same numbers (trial runs with --out
write it beside the output instead).

Regenerate: <sca env python> paper/figures/src/fig_latency.py [--csv PATH] [--out STEM]
"""

import argparse
import csv
import hashlib
import json
import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-identical PDF/SVG on rerun
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scafig as sf  # noqa: E402
from k1data import FIGDIR  # noqa: E402

REPO = os.path.normpath(os.path.join(FIGDIR, "..", ".."))
CURVE_CSV = os.path.join(REPO, "results", "latency_curve.csv")
CSV_4096 = os.path.join(REPO, "results", "latency_4096_batch1.csv")
CAMPAIGN_CSV = os.path.join(REPO, "results", "baseline46", "base_summary.csv")
CAMPAIGN_191_CSV = os.path.join(REPO, "results", "e1_ultraimage_base_summary.csv")
SNIPPET = os.path.join(FIGDIR, "snippets", "latency.tex")

METHODS = {  # canonical key -> (label, color, marker); drawn in this order (SCA on top)
    "full": ("Full attention", sf.METHOD_COLOR["full"], "^"),
    "area": ("AReA", sf.METHOD_COLOR["area"], "s"),
    "sca": ("SCA (ours)", sf.SCA, "o"),
}
ALIASES = {"full": "full", "full_attention": "full", "sdpa": "full", "area": "area",
           "sca": "sca", "sca_nb": "sca", "sca_final": "sca"}
CAMPAIGN_RES = "4096x4096"
CAMPAIGN = {  # base_summary.csv method -> (label, color, marker); baselines in neutral greys
    "sca_nb": ("SCA (ours)", sf.SCA, "o"),
    "hrdit_dev_author": ("HRDiT", "#6F7C89", "D"),
    "ultraimage": ("UltraImage", sf.METHOD_COLOR["ultraimage"], "D"),
}
FULL_STOP_NOTE = ("6144\u00b2 timing pending,", "8192\u00b2 not run by design (quadratic cost)")
PATCH = 16  # pixels per image token side: VAE factor 8 times 2x2 latent packing
W, H = sf.TEXT, 2.05  # inches; the main-text figure must not grow


def read_curve(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("warmup_verified", "True").lower() != "true":
                raise ValueError("Latency figure requires verified warm-up ordering")
            m = ALIASES.get(r["method"].strip().lower())
            if m is None:
                print(f"skipping unknown method {r['method']!r}")
                continue
            rows.append((r["res"].strip(), m, float(r["seconds"])))
    return rows


def read_4096(path):
    """Adapter: latency_4096_batch1.csv (run,code,host,method,image,seconds) -> res,method,seconds."""
    keep = {("latency", "full"): "full", ("latency", "area"): "area", ("latency3", "sca"): "sca"}
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            m = keep.get((r["run"], r["method"]))
            if m is not None:
                rows.append(("4096x4096", m, float(r["seconds"])))
    return rows


def read_campaign(path, methods, res=CAMPAIGN_RES):
    """Campaign mean seconds per image (and bootstrap 95% interval) from a *_base_summary.csv."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["metric"] == "seconds" and r["res"] == res and r["method"] in methods:
                if r["method"] in out:
                    raise ValueError(f"duplicate seconds row for {r['method']} in {path}")
                out[r["method"]] = dict(mean=float(r["value"]), ci_low=float(r["ci_low"]),
                                        ci_high=float(r["ci_high"]), n_prompts=int(r["n_prompts"]))
    missing = set(methods) - set(out)
    if missing:
        raise ValueError(f"no campaign seconds for {sorted(missing)} in {path}")
    return out


def medians(rows):
    acc = defaultdict(list)
    for res, m, s in rows:
        acc[res, m].append(s)
    if any(len(v) != 3 for v in acc.values()):
        raise ValueError("Each reported latency group must contain three timed images")
    return {k: (float(np.median(v)), len(v)) for k, v in acc.items()}


def parse_res(res):
    w, h = (int(v) for v in res.lower().split("x"))
    return w, h


def res_label(res):
    w, h = parse_res(res)
    return f"{w}\u00b2" if w == h else f"{w}\u00d7{h}"  # plain text keeps the sans tick font


def tokens(res):
    w, h = parse_res(res)
    return (w // PATCH) * (h // PATCH)


def box(x, y, w, h):
    return [x / W, y / H, w / W, h / H]


def bar_chart(med, speed, big):
    """Single-resolution fallback: horizontal bars of median seconds."""
    fig = plt.figure(figsize=(sf.base.COLUMN, 1.2))
    ax = fig.add_axes([0.28, 0.32, 0.62, 0.62])
    order = [m for m in ("sca", "full", "area") if (big, m) in med]
    y = np.arange(len(order))[::-1]
    vals = [med[big, m][0] for m in order]
    ax.barh(y, vals, color=[METHODS[m][1] for m in order], height=0.62, edgecolor="none")
    for yi, m, v in zip(y, order, vals):
        note = f"{v:.0f} s"
        if m == "sca":
            note += "  (" + ", ".join(f"{speed[big][b]:.2f}$\\times$ vs {METHODS[b][0]}"
                                      for b in ("full", "area") if b in speed[big]) + ")"
        ax.text(v + 3, yi, note, va="center", ha="left", fontsize=sf.FS_SMALL)
    ax.set_yticks(y, [METHODS[m][0] for m in order])
    ax.set_xlim(0, max(vals) * 2.5)
    ax.set_xlabel(f"seconds per {res_label(big)} image (batch 1, one GH200)")
    ax.grid(visible=False)
    return fig


def curve_figure(med, ranges, resolutions, speed, expo, camp):
    """Seconds per image (log-log) with SCA's speedup over AReA printed above each resolution."""
    fig = plt.figure(figsize=(W, H))
    left, right, bottom, top = 0.47, 0.04, 0.37, 0.36
    ax = fig.add_axes(box(left, bottom, W - left - right, H - bottom - top))
    n = {r: tokens(r) for r in resolutions}
    xt = [n[r] for r in resolutions]

    both = [r for r in resolutions if (r, "area") in med and (r, "sca") in med]
    ax.fill_between([n[r] for r in both], [med[r, "sca"][0] for r in both],
                    [med[r, "area"][0] for r in both], color=sf.SCA_LIGHT, lw=0, zorder=0)
    ends = {}
    for m, (lbl, col, mk) in METHODS.items():
        rs = [r for r in resolutions if (r, m) in med]
        if not rs:
            continue
        xs, ys = [n[r] for r in rs], [med[r, m][0] for r in rs]
        # whiskers: observed minimum and maximum over the three timed prompts, not intervals
        ax.vlines(xs, [min(ranges[r, m]) for r in rs], [max(ranges[r, m]) for r in rs], color=col,
                  lw=0.8, zorder=2)
        ax.plot(xs, ys, color=col, lw=1.6 if m == "sca" else 1.1, marker=mk,
                ms=3.8 if m == "sca" else 3.3, mec="white", mew=0.4, zorder=3 if m == "sca" else 2.5,
                solid_capstyle="round")
        ends[m] = (rs[-1], xs[-1], ys[-1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([res_label(r) for r in resolutions])
    ys = [v for v, _ in med.values()]
    lo, hi = np.floor(np.log10(min(ys))), np.ceil(np.log10(max(ys)))
    yt = [k * 10 ** e for e in np.arange(lo, hi + 1) for k in (1, 2, 5)]
    yt = [t for t in yt if min(ys) / 1.5 <= t <= max(ys) * 1.5]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{t:g}" for t in yt])
    ax.set_xlim(xt[0] / 1.2, xt[-1] * 2.3)
    ax.set_ylim(min(ys) / 1.35, max(ys) * 1.3)
    ax.grid(visible=False)
    ax.set_xlabel("output resolution (log scale)")
    ax.set_ylabel("seconds per image")

    # direct labels: name and fitted exponent at each line's end (full attention gets a note)
    for m, (_, x, y) in ends.items():
        lbl, col, _ = METHODS[m]
        if m == "full":
            continue
        ax.annotate(lbl, (x, y), xytext=(7, 2.5), textcoords="offset points", ha="left", va="center",
                    fontsize=sf.FS, color=col, fontweight="bold" if m == "sca" else "normal")
        # "proportional to N^b": the exponent is set as its own 7 pt text so it stays legible
        base = ax.annotate(r"$\propto N$", (x, y), xytext=(7, -3.5), textcoords="offset points",
                           ha="left", va="top", fontsize=sf.FS, color=col)
        fig.canvas.draw()
        bb = base.get_window_extent()
        px = 72 / fig.dpi
        x_disp, y_disp = ax.transData.transform((x, y))
        ax.annotate(f"{expo[m]:.2f}", (x, y), xytext=((bb.x1 - x_disp) * px + 0.6, (bb.y1 - y_disp) * px + 2.2),
                    textcoords="offset points", ha="left", va="top", fontsize=sf.FS_SMALL, color=col)

    # Why the full-attention curve stops: a note below the SCA line, tied to the last point.
    if "full" in ends and ends["full"][0] != resolutions[-1]:
        _, fx, fy = ends["full"]
        lbl, col, mk = METHODS["full"]
        nx, ny = fx * 1.16, fy / 3.3  # note anchor (top left), in data coordinates
        gx, gy = nx * 1.035, ny / 1.075  # the note's line glyph, where the leader ends
        ax.annotate("", (fx, fy), xytext=(gx, gy), textcoords="data",
                    arrowprops=dict(arrowstyle="-", color=sf.MID, lw=0.45, shrinkA=2.8, shrinkB=3.2),
                    zorder=1.5)
        ax.plot([gx], [gy], color=col, marker=mk, ms=3.3, mec="white", mew=0.4, ls="none", zorder=4)
        ax.annotate(f"{lbl}: " + FULL_STOP_NOTE[0] + "\n" + FULL_STOP_NOTE[1], (nx, ny),
                    xytext=(9.5, 0), textcoords="offset points", ha="left", va="top",
                    fontsize=sf.FS_TICK, color=sf.MID, linespacing=1.15)

    # Campaign means at 4096^2 (shared nodes): hollow markers on the clean column, never joined.
    xc = n[CAMPAIGN_RES]
    for key, (lbl, col, mk) in CAMPAIGN.items():
        v = camp[key]["mean"]
        if key == "sca_nb":  # a ring around SCA's clean point, which it nearly equals
            ax.plot([xc], [v], ls="none", marker="o", ms=7.4, mfc="none", mec=col, mew=0.85, zorder=3.6)
            continue
        ax.plot([xc], [v], ls="none", marker=mk, ms=4.0, mfc="white", mec=col, mew=0.9, zorder=3.6)
        ax.annotate(lbl, (xc, v), xytext=(-5.5, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=sf.FS_TICK, color=col)

    # key: filled = clean batch-1 sweep, hollow = campaign mean (grey baselines, green SCA)
    kx, ky, dy, dx = 0.020, 0.955, 0.13, 0.019  # axes fractions
    base_col = METHODS["area"][1]
    glyphs = [  # (row, slot, marker, size, face, edge, edge width)
        (0, 0, "s", 3.3, base_col, "white", 0.4), (0, 1, "o", 3.8, sf.SCA, "white", 0.4),
        (1, 0, "D", 3.6, "white", base_col, 0.85), (1, 1, "o", 5.0, "none", sf.SCA, 0.85)]
    for row, slot, mk, ms, face, edge, mew in glyphs:
        ax.plot([kx + slot * dx], [ky - row * dy], transform=ax.transAxes, ls="none", marker=mk, ms=ms,
                mfc=face, mec=edge, mew=mew, clip_on=False, zorder=4)
    texts = ["batch 1, one GH200, median of three prompts",
             f"campaign mean at {res_label(CAMPAIGN_RES)}, shared nodes, {camp['sca_nb']['n_prompts']} prompts"]
    for row, t in enumerate(texts):
        ax.text(kx + dx + 0.017, ky - row * dy, t, transform=ax.transAxes, ha="left", va="center",
                fontsize=sf.FS_TICK, color=sf.MID)

    # speedup row: SCA over AReA at every resolution, as tick labels on a top axis
    top_ax = ax.secondary_xaxis("top")
    top_ax.xaxis.set_major_locator(FixedLocator([n[r] for r in both]))
    top_ax.xaxis.set_minor_locator(NullLocator())
    top_ax.set_xticklabels([f"{speed[r]['area']:.2f}\u00d7" for r in both])
    top_ax.tick_params(axis="x", length=0, pad=2, labelsize=sf.FS_TICK, labelcolor=sf.SCA)
    top_ax.spines["top"].set_visible(False)
    labels = top_ax.get_xticklabels()
    labels[-1].set_fontweight("bold")
    labels[-1].set_fontsize(sf.FS)
    top_ax.set_xlabel("speedup of SCA over AReA", fontsize=sf.FS_TICK, color=sf.MID, labelpad=3)
    return fig, ax, top_ax


def tex_res(res):
    w, h = parse_res(res)
    return f"${w}^2$" if w == h else rf"${w}\times{h}$"


HRDIT_CLEAN = "/scratch/09032/saini_2/sca/out/latency_hrdit/hrdit_dev_author/4096x4096"


def hrdit_clean_seconds():
    """Median of the three batch-1 HRDiT sidecars on the latency prompts, or None if they are absent."""
    import glob
    import json
    vals = [json.load(open(f))["seconds"] for f in sorted(glob.glob(os.path.join(HRDIT_CLEAN, "*_seed0.json")))]
    return float(np.median(vals)) if len(vals) == 3 else None


def write_snippet(path, resolutions, speed, expo, camp):
    """Caption numbers come from the same CSVs as the plot (HRDiT's clean time from its three sidecars)."""
    hrdit_clean = hrdit_clean_seconds()
    big = resolutions[-1]
    n_camp = camp["sca_nb"]["n_prompts"]
    cap = (
        rf"\textbf{{\sca{{}} is the fastest of the methods measured at each size, and its lead over AReA "
        rf"grows to ${speed[big]['area']:.2f}\times$ at {tex_res(big)}.}}" "\n"
        r"  Filled markers: batch-1 seconds per image on one GH200 (median of three prompts, whiskers min to "
        rf"max); full attention scales as $N^{{{expo['full']:.2f}}}$ ($N$: image tokens)." "\n"
        rf"  Hollow markers: {n_camp}-prompt campaign means at {tex_res(CAMPAIGN_RES)} on shared nodes; HRDiT "
        r"runs 30, 17, 10 steps at $1024^2$, $2048^2$, $4096^2$, UltraImage two 30-step stages. "
        + (rf"HRDiT's clean batch-1 time is {hrdit_clean:.1f}\,s (median of three)." if hrdit_clean
           else r"HRDiT's clean batch-1 timing is pending.")
    )
    with open(path, "w") as fh:
        fh.write("% Generated by paper/figures/src/fig_latency.py from results/latency_curve.csv (clean "
                 "batch-1 sweep)\n% and results/baseline46/base_summary.csv (46-prompt campaign means).\n")
        fh.write("\\begin{figure}[tp]\n  \\centering\n  \\includegraphics{figures/latency.pdf}\n")
        fh.write("  \\caption{" + cap + "}\n  \\label{fig:latency}\n\\end{figure}\n")
    return path


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CURVE_CSV)
    ap.add_argument("--source", choices=("auto", "curve", "4096"), default="auto")
    default_out = os.path.join(FIGDIR, "latency")
    ap.add_argument("--out", default=default_out)
    args = ap.parse_args()
    # The paper snippet is written only for the default output; trial runs put it beside --out.
    snippet = SNIPPET if os.path.abspath(args.out) == default_out else args.out + "_snippet.tex"

    if args.source == "4096" or (args.source == "auto" and not os.path.exists(args.csv)):
        src, rows = CSV_4096, read_4096(CSV_4096)
    else:
        src, rows = args.csv, read_curve(args.csv)
    print(f"latency source: {src}")
    med = medians(rows)
    ranges = {}
    for res, method, seconds in rows:
        ranges.setdefault((res, method), []).append(seconds)
    resolutions = sorted({r for r, _ in med}, key=lambda r: np.prod(parse_res(r)))
    for (res, m), (s, n) in sorted(med.items()):
        print(f"  {res} {m}: median {s:.2f} s over {n} images")

    speed = {r: {b: med[r, b][0] / med[r, "sca"][0] for b in ("full", "area")
                 if (r, b) in med and (r, "sca") in med} for r in resolutions}
    for r in resolutions:
        for b, v in speed[r].items():
            print(f"speedup of SCA over {b} at {r}: {v:.2f}x")
    expo = {}
    for m in METHODS:
        rs = [r for r in resolutions if (r, m) in med]
        if len(rs) >= 2:
            expo[m] = float(np.polyfit(np.log([tokens(r) for r in rs]), np.log([med[r, m][0] for r in rs]), 1)[0])
            print(f"least-squares exponent of seconds ~ N^b for {m} over {len(rs)} resolutions: b = {expo[m]:.3f}")

    camp = read_campaign(CAMPAIGN_CSV, CAMPAIGN)
    if len({v["n_prompts"] for v in camp.values()}) != 1:
        raise ValueError("campaign means must come from one prompt set")
    ultra191 = read_campaign(CAMPAIGN_191_CSV, {"ultraimage": None})["ultraimage"]
    for k, v in camp.items():
        print(f"campaign mean at {CAMPAIGN_RES}, {k}: {v['mean']:.2f} s "
              f"[{v['ci_low']:.2f}, {v['ci_high']:.2f}] over {v['n_prompts']} prompts")
    print(f"campaign mean at {CAMPAIGN_RES}, ultraimage: {ultra191['mean']:.2f} s over "
          f"{ultra191['n_prompts']} prompts (provenance only)")
    fastest = all(med[r, "sca"][0] == min(v for (rr, _), (v, _) in med.items() if rr == r) for r in resolutions)
    fastest &= camp["sca_nb"]["mean"] == min(v["mean"] for v in camp.values())
    if not fastest:
        raise ValueError("caption lead no longer holds: SCA is not the fastest measured method at some size")

    sf.use_style()
    plt.rcParams["svg.hashsalt"] = "sca"
    if len(resolutions) == 1:
        fig = bar_chart(med, speed, resolutions[0])
    else:
        fig, _, _ = curve_figure(med, ranges, resolutions, speed, expo, camp)
        print("wrote", write_snippet(snippet, resolutions, speed, expo, camp))
    sf.save(fig, args.out)
    full_res = sorted((r for r in resolutions if (r, "full") in med), key=tokens)
    with open(args.out + "_provenance.json", "w") as handle:
        json.dump({"source": src, "sha256": sha256(src), "statistic": "median of three timed prompts",
                   "whiskers": "observed minimum and maximum, not a confidence interval",
                   "exponent_fit": "least squares of log median seconds on log N, N = (W/16)(H/16) tokens",
                   "exponents": {m: round(v, 6) for m, v in sorted(expo.items())},
                   "groups": [{"res": r, "method": m, "median_seconds": value, "n": n}
                              for (r, m), (value, n) in sorted(med.items())],
                   "speedup_sca_over_area": {r: round(v["area"], 4) for r, v in speed.items() if "area" in v},
                   "full_attention": {"measured": full_res,
                                      "not_plotted": "6144x6144 clean timing run pending; 8192x8192 not run "
                                                     "by design (quadratic attention cost)"},
                   "campaign_means": {
                       "source": CAMPAIGN_CSV, "sha256": sha256(CAMPAIGN_CSV), "res": CAMPAIGN_RES,
                       "statistic": "mean seconds per image over the campaign's prompts on shared nodes, "
                                    "bootstrap 95% interval; not clean batch-1 latency",
                       "drawn_as": "hollow markers at 4096^2, not joined to the clean curves",
                       "methods": {k: {**v, "label": CAMPAIGN[k][0]} for k, v in camp.items()},
                       "protocols": {"hrdit_dev_author": "progressive 30 + 17 + 10 steps at 1024^2, 2048^2, "
                                                         "4096^2 (scripts/sample_progressive.py HRDIT)",
                                     "ultraimage": "released two-stage recipe, 30 steps per stage",
                                     "sca_nb": "28 steps at the target size"},
                       "ultraimage_191_prompts": {"source": CAMPAIGN_191_CSV, "sha256": sha256(CAMPAIGN_191_CSV),
                                                  **ultra191, "use": "not plotted; recorded to reconcile with the 191-prompt tables"},
                       "pending": "clean batch-1 HRDiT timing queued, not measured; not plotted"}},
                  handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
