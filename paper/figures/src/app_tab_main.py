#!/usr/bin/env python3
"""Appendix tables of the main comparison (gen_app_tables.py module app_tab_main). Their \\sca{} cells switch to the
final operator (sca_v2) as its runs land; nothing is typed by hand.

  name               label                  kind     runs (v2swap.RUNS, v2swap.LATENCY)
  all_metrics        tab:all_metrics        tabular  a4k195_main_sca_v2, a4k195_2048x4096_sca_v2,
                                                     a4k195_2048x6144_sca_v2, latency_curve_v2
  app_4k             tab:app_4k             tabular  a4k195_main_sca_v2
  app_wide           tab:app_wide           tabular  a4k195_2048x4096_sca_v2, a4k195_2048x6144_sca_v2
  latency            tab:latency            tabular  latency_curve_v2
  published_compare  tab:published_compare  float    krea195_sca_v2, a4k195_main_sca_v2, hrgen1k_krea_sca_v2
  published_hrgen    tab:published_hrgen    float    hrgen1k_krea_sca_v2, hrgen1k_dev_sca_v2

Since 2026-11-07 (progressive SCA is the paper's headline): the final operator's rows of all_metrics read "Single-pass
\\sca{}" (app_4k heads its column the same way), and the 4096^2 block of all_metrics adds HRDiT and progressive SCA
from results/prog195_sca_v2 (prog_rows: clean latency from the third sweep, results/prog_latency; progressive SCA
marked against AReA from results/prog195_pairs; HRDiT unmarked like the other baselines).

Paper v6 layout (2026-11-10; the numbers, intervals, colours and bold cells of v5.1 are unchanged): every table is set
at 7 pt with the shared macros of main_tables/tabstyle.tex and no \\resizebox. all_metrics: one table, columns grouped
by metric family (time, preference, image quality, set level), our rows shaded, second best per column underlined in
blocks of more than two rows. app_4k, app_wide: one row per metric under family heads, each difference beside its grey
interval in its own column. latency: same cells. published_*: our rows shaded instead of bold, single-line heads, v6
captions (lead, what it expands, setup, marks, one key-result sentence), every claim recomputed from the cells.
--check compared with main.tex regions that moved to appendix.tex; it no longer applies (every run has landed).

Published rendering (every run pending; gen_app_tables.py --check compares it with today's tables):
  all_metrics  values as printed; bold = best printed value of each column within a size block, ties share;
               underline = second best in blocks of more than two rows.
               \\up / \\dn (blue / orange, the paper's one significance convention): the 95% paired interval
               against AReA excludes zero in the row's favour / against it (sig_mark gives \\sg / \\sw, which
               am_block maps to the colour wrappers).
               Full attention: full_vs_area.csv (scripts/paired_bootstrap.py, 2000 draws), per-image metrics only.
               \\sca{} band row: the 191-prompt final4k run as gen_tables.py reads it (corrected FID points with
               eval.py intervals, corrected patch FID without). \\sca{} (final): eval.py / eval_extra.py deltas
               against \\sca{}, negated (1000 draws), FID, KID and patch versions from corrected_fidp_ci
               (data195.versus). 6144^2 block: results/e2_6144 as gen_late_tables.py reads it. Latency: clean
               batch-1 medians (results/latency_curve_summary.csv; the band row: median of run latency2 in
               results/latency_4096_batch1.csv) or, with the double dagger, campaign means (base_summary seconds).
  app_4k,      gen_tables.py's own cell formatters (f, dcell, vcell, loaded by AST as fig_paired_area.py does);
  app_wide     Delta = method minus AReA through data195.versus, as above.
  latency      clean batch-1 medians; full attention at 6144^2 only in results/latency_curve_batch1.csv (median
               of its three timed images); bold = fastest printed value of each row.
  published_*  results/published_compare/build.py's logic: its source is executed into a namespace for its
               readers and constants (main() and its writers never run), and its two table writers are restated
               below with our \\sca{} rows replaced.
Landed rendering: each switching cell takes the sca_v2 run's value, interval and mark, read the same way
(v2swap.load and v2swap.versus; eval.py intervals, 1000 draws), and bold, underline and the published tables' key
results are recomputed. The z98x runs have no corrected_fidp_ci, so FID and KID keep eval.py's interval (reference
rows resampled, not reference-image clusters) and the patch versions print the point alone, without interval or
mark. Pending: the published cell wrapped in \\pend (the published tables also grey the key-result clauses about a
pending \\sca{} row). Each table switches cell by cell, so a partly landed table mixes both.
Values that never get an sca_v2 counterpart stay and say so (the subs): the 6144^2 row of all_metrics is
"\\sca{} w/o schedule". The band-guard row of all_metrics and the band column of app_4k were removed on 2026-11-07
(the paper no longer describes the band guard); band() and band_clean_seconds() stay only as readers.
AReA keeps its published values everywhere; each landed run's own AReA values are compared with them at printed
precision and any difference is a warning (the Delta columns pair against the landed run's AReA). Caption warnings
re-read the caption in main.tex and stop once it no longer carries the phrase they flag.
"""
from __future__ import annotations

import ast
import csv
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data195  # noqa: E402  (published a4k195_* runs, read-only)
import v2swap  # noqa: E402  (sca_v2 runs, read-only)

ROOT = HERE.parents[2]
RESULTS = ROOT / "results"
BUILD = RESULTS / "published_compare" / "build.py"
PEND = v2swap.pend
HEADER = "% Generated by paper/figures/src/gen_app_tables.py (module app_tab_main.py) from results/; do not edit."
# Paper v6 table style (2026-11-10): the shared macros of main_tables/tabstyle.tex (written by gen_main_tables.py).
# \scaFitTab{colspec}{n}{rows} sets a booktabs tabular at \linewidth by spreading the spare width over 2n cell
# paddings; \scaRowOurs shades our rows; \camp marks a campaign mean; colour scaCtx greys intervals. Every table is
# \scriptsize (7 pt) without \resizebox, so no font is scaled.
STYLE = r"\ifdefined\scaFitTab\else\input{figures/snippets/main_tables/tabstyle.tex}\fi"
OURS = r"\scaRowOurs "
# An estimate and its grey interval as one unit: two columns 3 pt apart. Each such pair removes two of the cell
# paddings that \scaFitTab spreads the spare width over, so its column count argument drops by one per pair.
# PAIR_GAP ends a column group with 6 pt more room after the interval, so the groups read apart.
PAIR = r"r@{\hspace{3pt}}l"
PAIR_GAP = r"r@{\hspace{3pt}}l<{\hspace{6pt}}"
# Superscript marks are smashed so that a row carrying one is no taller than its neighbours.
CAMP = r"\smash{\camp}"
DAGGER = r"\rlap{\smash{$^\dagger$}}"


def bound(s):
    """One formatted bound of an interval: math dollars and a leading plus dropped, the {-} minus kept."""
    s = s.strip().strip("$")
    return s[1:] if s.startswith("+") else s


def ci(lo, hi):
    """A 95% interval in grey from two formatted bounds, as the v6 tables print it: $[0.25, 0.41]$."""
    return r"{\color{scaCtx}$[" + bound(lo) + ", " + bound(hi) + r"]$}"


def dsplit(g, t, m):
    """gen_tables.dcell's cell split into (estimate, interval) for two columns: the same digits and the same colour
    (dcell decides both), the interval in grey; '' when the cell has no interval."""
    s = g["dcell"](t, m)
    if " {\\tiny[" not in s:
        return s, ""
    est, rest = s.split(" {\\tiny[", 1)
    if not rest.endswith("]}") or rest.count(", ") != 1:
        raise ValueError(f"unexpected dcell output {s!r}")
    lo, hi = rest[:-2].split(", ")
    return est, ci(lo, hi)


# Metric families of the v6 tables, for tables with one row per metric (app_4k, app_wide); the arrow in a family
# head holds for every row below it, the set-level rows carry their own.
FAMILIES = [(r"Preference $\uparrow$", ["ir", "aesthetic", "clip", "pickscore", "hpsv2", "hpsv21"]),
            (r"Image quality $\uparrow$", ["musiq", "musiq_crop1024", "clipiqa", "arniqa"]),
            ("Set level and time", ["is", "fid", "kid", "patch_fid", "patch_kid", "seconds"])]
ROW_LABEL = {"ir": "ImageReward", "aesthetic": "Aesthetic", "clip": "CLIP score", "pickscore": "PickScore",
             "hpsv2": r"HPSv2 ($\times100$)", "hpsv21": r"HPSv2.1 ($\times100$)", "musiq": "MUSIQ",
             "musiq_crop1024": r"MUSIQ$_\text{c}$", "clipiqa": r"CLIP-IQA ($\times100$)",
             "arniqa": r"ARNIQA ($\times100$)", "is": r"Inception score $\uparrow$", "fid": r"FID $\downarrow$",
             "kid": r"KID ($\times\smash{10^3}$) $\downarrow$", "patch_fid": r"FID$_\text{p}$ $\downarrow$",
             "patch_kid": r"KID$_\text{p}$ ($\times\smash{10^3}$) $\downarrow$",
             "seconds": r"Seconds per image\smash{$^\ddagger$} $\downarrow$"}


def metric_rows(ncols, cells_of):
    """Body of a table with one row per metric: an italic head per family, \\midrule between families;
    cells_of(metric) gives the row's cells after its label."""
    out = []
    for i, (head, keys) in enumerate(FAMILIES):
        if i:
            out.append(r"\midrule")
        out.append(rf"\multicolumn{{{ncols}}}{{l}}{{\emph{{{head}}}}} \\")
        out += [ROW_LABEL[m] + " & " + " & ".join(cells_of(m)) + r" \\" for m in keys]
    return out


def wins_losses(cells):
    """(better, worse) metric keys of {metric: estimate cell} by the colour the cell carries."""
    return ([m for m, c in cells.items() if c.lstrip().startswith(r"\up{")],
            [m for m, c in cells.items() if c.lstrip().startswith(r"\dn{")])

MAIN, V2_MAIN = "a4k195_main", "a4k195_main_sca_v2"
WIDE = {"2048x4096": ("a4k195_2048x4096", "a4k195_2048x4096_sca_v2"),
        "2048x6144": ("a4k195_2048x6144", "a4k195_2048x6144_sca_v2")}
for _pub, _v2 in [(MAIN, V2_MAIN)] + list(WIDE.values()):
    assert v2swap.RUNS[_v2].published_run == _pub and v2swap.RUNS[_v2].final == "sca_v2", _v2
FINAL = "sca_v2"
# gen_tables.py's metric order (Tables app_4k and app_wide); the last five are lower-better
ALL = ["ir", "aesthetic", "clip", "pickscore", "hpsv2", "hpsv21", "musiq", "musiq_crop1024", "clipiqa", "arniqa",
       "is", "fid", "kid", "patch_fid", "patch_kid", "seconds"]
PER_IMAGE = ALL[:10]
LOWER = {"fid", "kid", "patch_fid", "patch_kid", "seconds"}
N_MAIN = 195  # \nmain in main.tex
assert sorted(k for _, keys in FAMILIES for k in keys) == sorted(ALL)
WORDS = "no one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen".split()


# shared readers =====================================================================================================
def gen_tables_ns():
    """gen_tables.py's readers and formatters bound to results/, loaded by AST (its module body reads sys.argv and
    writes files), exactly as fig_paired_area.py does."""
    tree = ast.parse((HERE / "gen_tables.py").read_text())
    names = {"rows", "summary", "deltas", "corrected_distributions", "f", "dcell", "vcell"}
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in ("MET", "METD") for t in n.targets))]
    ns = {"csv": csv, "Path": Path, "R": RESULTS}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(HERE / "gen_tables.py"), "exec"), ns)
    return ns


def band(g):
    """The 191-prompt \\sca{} + band guard run (method "sca", reference AReA) as gen_tables.py reads it."""
    s_b = g["summary"]("final4k")
    d_b = g["deltas"](["final4k_base_paired_deltas.csv", "final4k_extra_paired_deltas.csv"])
    g["corrected_distributions"](s_b, d_b, ["final4k_base"], "area")
    return s_b, d_b


def published(run):
    """(S, vs): a published run's values and its method-minus-AReA deltas (data195.versus)."""
    S, D, ref, _ = data195.load(run)
    return S, data195.versus(S, D, ref, run, "area", ALL)


def final(run):
    """(S, vs, clustered) of a landed sca_v2 run: values, method minus AReA (eval.py deltas against sca_v2,
    negated), and whether FID, KID and their patch versions have the reference-cluster intervals (corrected_fidp_ci;
    without it FID and KID keep eval.py's reference-row interval and the patch versions have none)."""
    S, D, ref, clustered = v2swap.load(run)
    assert ref == FINAL, (run, ref)
    return S, v2swap.versus(S, D, ref, run, "area", ALL), clustered


CLUSTER_NOTE = ("FID and KID intervals of {runs} resample reference rows (eval.py), not reference-image clusters, and "
                "their patch versions have no interval (no corrected_fidp_ci)")


def n_prompts(run):
    d = (v2swap.V2_RESULTS if run in v2swap.RUNS else RESULTS) / run
    return data195.n_prompts(run, d.parent)


def batch1_median(res, method):
    """Median of the timed (not warm-up) images of a cell in results/latency_curve_batch1.csv, or None."""
    secs = [float(r["seconds"]) for r in csv.DictReader(open(RESULTS / "latency_curve_batch1.csv"))
            if r["res"] == res and r["method"] == method and not r["image"].startswith("sca_latency_warmup")]
    return statistics.median(secs) if secs else None


def band_clean_seconds():
    """Clean batch-1 median of the single-pass compiled band guard (run latency2, results/latency_4096_batch1.csv)."""
    secs = [float(r["seconds"]) for r in csv.DictReader(open(RESULTS / "latency_4096_batch1.csv"))
            if r["run"] == "latency2" and r["method"] == "sca"]
    assert len(secs) == 3, secs
    return statistics.median(secs)


def area_check(show, S_pub, S_new, run, metrics, where):
    """Warnings for AReA values of a landed run that differ from the published ones as printed (show(metric, v))."""
    out = []
    for m in metrics:
        a, b = S_pub.get(("area", m)), S_new.get(("area", m))
        if a is None or b is None:
            out.append(f"{where}: AReA {m} missing in {'the published run' if a is None else run}")
            continue
        pa, pb = show(m, a), show(m, b)
        if pa != pb:
            out.append(f"{where}: AReA {m} prints {pa} (published, kept in the table) but {pb} in {run}, against "
                       f"which the sca_v2 interval is paired")
    return out


def interval_check(vs, run, metrics, where):
    """Warnings for metrics of a landed run without a paired interval (the patch versions are expected to lack one)."""
    gone = [m for m in metrics if m not in ("patch_fid", "patch_kid")
            and (vs.get((FINAL, m)) is None or vs[(FINAL, m)][1] is None or vs[(FINAL, m)][2] is None)]
    return [f"{where}: {run} has no paired interval for {', '.join(gone)}; those cells print without one"] \
        if gone else []


def caption(label):
    """The one-line \\caption of a table whose float lives in appendix.tex or main.tex (read-only), so that a caption
    warning stops once the caption no longer carries the phrase; None if it cannot be found (every phrase then counts
    as present). Since v6 the floats of the tabular snippets sit in appendix.tex."""
    for name in ("appendix.tex", "main.tex"):
        try:
            lines = (ROOT / "paper" / name).read_text().splitlines()
            i = next(k for k, line in enumerate(lines) if f"\\label{{{label}}}" in line)
            return next(lines[k] for k in range(i, max(i - 6, -1), -1) if "\\caption{" in lines[k])
        except (OSError, StopIteration):
            continue
    return None


def says(cap, phrase):
    return cap is None or phrase in cap


def prompt_check(run, want=N_MAIN):
    n = n_prompts(run)
    return [] if n == {want} else [f"{run}: prompt counts {sorted(n)}, the caption says {want}"]


# tab:all_metrics ====================================================================================================
# (key, scale, decimals, lower is better), in the column order of the v6 table: time, then the metric families
AM_COLS = [("seconds", 1, 1, True), ("ir", 1, 2, False), ("aesthetic", 1, 2, False), ("clip", 1, 2, False),
           ("pickscore", 1, 2, False), ("hpsv2", 100, 2, False), ("hpsv21", 100, 2, False),
           ("musiq", 1, 1, False), ("musiq_crop1024", 1, 1, False), ("clipiqa", 100, 1, False),
           ("arniqa", 100, 1, False), ("fid", 1, 1, True), ("patch_fid", 1, 1, True)]
AM_HEAD = [
    r"\toprule",
    r" & Time & \multicolumn{6}{c}{Preference $\uparrow$} & \multicolumn{4}{c}{Image quality $\uparrow$} & "
    r"\multicolumn{2}{c}{Set level $\downarrow$} \\",
    r"\cmidrule(lr){2-2}\cmidrule(lr){3-8}\cmidrule(lr){9-12}\cmidrule(l){13-14}",
    r"Method & s $\downarrow$ & IR & Aesth. & CLIP & Pick & HPSv2 & HPSv2.1 & MUSIQ & MUSIQ$_\text{c}$ & "
    r"CLIP-IQA & ARNIQA & FID & FID$_\text{p}$ \\",
    r"\midrule"]
# the time column keeps 3 pt after its digits for the hanging campaign mark
AM_SPEC = r"lr<{\hspace{3pt}}*{12}{r}"
AM_BLOCKS = {"4096x4096": r"$4096\times4096$, scale factors (4, 4)",
             "2048x4096": r"$2048\times4096$, scale factors (2, 4)",
             "2048x6144": r"$2048\times6144$, scale factors (2, 6)",
             "6144x6144": r"$6144\times6144$, scale factors (6, 6), 100 prompts"}
SCA_FINAL = r"Single-pass \sca{}"  # the final operator (sca_v2), 28 steps at the target size
SCA_FINAL_OLD = r"\sca{} (final)"   # its label before the paper made progressive SCA the headline (2026-11-07)
BAND_LABEL, BAND_OLD = r"\sca{} band w/o sched.", r"\sca{} + band guard"  # no wider than "AReA (released code)"
E2_LABEL = r"\sca{} w/o schedule"  # "without the schedule" in the caption; the full words would widen every column
AREA_LABEL = "AReA"  # our run of the released code (v5.1: "AReA (released code)"; the caption says so)
# Multi-stage rows of Table tab:main (2026-11-07): HRDiT and progressive SCA (HRDiT's three-stage schedule with the
# final operator), both evaluated in results/prog195_sca_v2 (base and extra summaries; FID, KID and patch versions
# from its corrected_fidp, 168 reference identities). Progressive SCA's marks against AReA: per-image
# results/prog195_pairs/sca_v2_prog_vs_area.csv (paired_bootstrap.py, 2000 prompt draws, seed 0) and set-level
# results/prog195_pairs/cfid_vs_area/paired_deltas.csv (1000 draws over prompts and reference-image clusters).
# HRDiT stays unmarked, like the other baselines that are not full attention. Both rows take their clean latency from
# the third batch-1 sweep (results/prog_latency/latency_summary.csv: progressive SCA, HRDiT and AReA back to back on
# one GH200, median of three timed images after an excluded warm-up), as Table tab:main does.
PROG_RUN, PROG, PROG_PAIRS = "prog195_sca_v2", "sca_v2_prog", "prog195_pairs"
PROG_LATENCY = RESULTS / "prog_latency" / "latency_summary.csv"


def sweep3(method):
    """Clean batch-1 median of `method` at 4096^2 in the third latency sweep."""
    rows = [r for r in csv.DictReader(open(PROG_LATENCY)) if r["res"] == "4096x4096" and r["method"] == method]
    if len(rows) != 1 or int(rows[0]["n_timed"]) != 3:
        raise ValueError(f"{PROG_LATENCY}: expected one 4096x4096 row with three timed images for {method}")
    from decimal import ROUND_HALF_UP, Decimal  # 49.15 must print 49.2, as in Table tab:main
    return float(Decimal(rows[0]["median_s"]).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
PROG_LABEL = r"Progressive \sca{}"


def cluster_deltas(path, method, ref):
    """{metric: (delta, lo, hi)} of `method` minus `ref` for FID, KID and their patch versions from a
    repro/evaluate_corrected.py paired_deltas.csv (1000 draws over prompt ids and the 168 reference identities)."""
    out = {}
    for r in csv.DictReader(open(path)):
        if r["method"] != method or r["metric"] not in data195.DIST:
            continue
        if (r["reference_method"], int(r["n_prompts"]), int(r["n_reference_prompts"]), int(r["bootstrap_valid"])) \
                != (ref, N_MAIN, 168, 1000):
            raise ValueError(f"{path}: unexpected row {r}")
        out[r["metric"]] = (float(r["value"]), float(r["ci_low"]), float(r["ci_high"]))
    if set(out) != set(data195.DIST):
        raise ValueError(f"{path}: {method} minus {ref} lacks {sorted(set(data195.DIST) - set(out))}")
    return out


def prog_rows(S_area):
    """(HRDiT row, progressive SCA row, warnings) for the 4096^2 block of tab:all_metrics."""
    warn = prompt_check(PROG_RUN)
    P = v2swap.load(PROG_RUN)[0]
    keys = [k for k, _, _, _ in AM_COLS]
    hrdit = am_row("HRDiT", dict({k: P[("hrdit_dev_author", k)] for k in keys[1:]}, seconds=sweep3("hrdit_dev_author")))
    if am_show("seconds", sweep3("area")) != am_show("seconds", S_area["seconds"]):
        warn.append(f"{PROG_LATENCY}: AReA {sweep3('area'):.2f} s prints differently from the AReA row")
    per_image = RESULTS / PROG_PAIRS / f"{PROG}_vs_area.csv"
    if {int(r["n_prompts"]) for r in csv.DictReader(open(per_image))} != {N_MAIN}:
        raise ValueError(f"{per_image}: not {N_MAIN} prompts")
    marks = data195.pairs(PROG_PAIRS, PROG, "area")
    cfid = RESULTS / PROG_PAIRS / "cfid_vs_area"
    marks.update(cluster_deltas(cfid / "paired_deltas.csv", PROG, "area"))
    for r in csv.DictReader(open(cfid / "summary.csv")):  # the cluster run must score the same images
        if r["metric"] in ("fid", "patch_fid"):
            if r["method"] == PROG and abs(float(r["value"]) - P[(PROG, r["metric"])]) > 1e-6:
                raise ValueError(f"{cfid}/summary.csv: {PROG} {r['metric']} differs from {PROG_RUN}/corrected_fidp")
            if r["method"] == "area" and am_show(r["metric"], float(r["value"])) != \
                    am_show(r["metric"], S_area[r["metric"]]):
                warn.append(f"{cfid}: AReA {r['metric']} {float(r['value']):.2f} prints differently from the "
                            "published AReA row")
    prog = am_row(PROG_LABEL, dict({k: P[(PROG, k)] for k in keys[1:]}, seconds=sweep3(PROG)),
                  {k: marks.get(k) for k in keys[1:]}, ours=True)
    missing = [k for k in keys[1:] if marks.get(k) is None]
    if missing:
        warn.append(f"tab:all_metrics progressive row: no interval against AReA for {', '.join(missing)}")
    return hrdit, prog, warn


def num(x, d):
    """Absolute value at d decimals, a negative one in math mode ($-2.18$)."""
    s = f"{x:.{d}f}"
    return f"${s}$" if s.startswith("-") else s


def am_show(key, v):
    """A value as tab:all_metrics prints it (scale and decimals of its column)."""
    sc, d = next((sc, d) for k, sc, d, _ in AM_COLS if k == key)
    return num(v * sc, d)


def sig_mark(v, lower):
    """\\sg if the interval excludes zero in the row's favour, \\sw if against it, nothing without an interval."""
    if v is None or v[1] is None or v[2] is None:
        return ""
    _, lo, hi = v
    good, bad = (hi < 0, lo > 0) if lower else (lo > 0, hi < 0)
    return r"\sg" if good else (r"\sw" if bad else "")


def am_row(label, vals, marks=None, campaign=False, pend=None, ours=False):
    """vals {key: value}, marks {key: (delta, lo, hi)} against AReA, pend {key: bool} (or one bool for all); ours
    shades the row."""
    return dict(label=label, vals=vals, marks=marks or {}, campaign=campaign, pend=pend or {}, ours=ours)


def am_rank(rows):
    """{key: (best, second)} printed values per column of a size block; second is None in blocks of two rows, where
    it would only mark the other row. Ties share a mark."""
    rank = {}
    for key, sc, d, lower in AM_COLS:
        shown = sorted({float(f"{r['vals'][key] * sc:.{d}f}") for r in rows}, reverse=not lower)
        rank[key] = (shown[0], shown[1] if len(rows) > 2 and len(shown) > 1 else None)
    return rank


def am_block(rows):
    rank = am_rank(rows)
    out = []
    for r in rows:
        cells = []
        for key, sc, d, lower in AM_COLS:
            v = r["vals"][key] * sc
            txt = num(v, d)
            shown = float(f"{v:.{d}f}")
            if shown == rank[key][0]:
                txt = rf"\textbf{{{txt}}}"
            elif shown == rank[key][1]:
                txt = rf"\underline{{{txt}}}"
            if key == "seconds":
                txt += CAMP if r["campaign"] else ""
            else:  # one significance convention for every table (review round 1): colour, not superscripts
                txt = {r"\sg": rf"\up{{{txt}}}", r"\sw": rf"\dn{{{txt}}}"}.get(sig_mark(r["marks"].get(key), lower),
                                                                            txt)
            pend = r["pend"] if isinstance(r["pend"], bool) else r["pend"].get(key, False)
            cells.append(PEND(txt, pend))
        out.append((OURS if r["ours"] else "") + f"{r['label']} & " + " & ".join(cells) + r" \\")
    return out


AM_NAMES = {"seconds": "time", "ir": "ImageReward", "aesthetic": "Aesthetic", "clip": "CLIP", "pickscore": "PickScore",
            "hpsv2": "HPSv2", "hpsv21": "HPSv2.1", "musiq": "MUSIQ", "musiq_crop1024": r"MUSIQ$_\text{c}$",
            "clipiqa": "CLIP-IQA", "arniqa": "ARNIQA", "fid": "FID", "patch_fid": r"FID$_\text{p}$"}


def am_claims(rows, cap):
    """Warnings for the claims of the proposed v6 caption of tab:all_metrics about the 4096^2 block (best per column,
    the fastest method, all ten per-image metrics significantly better than AReA for both SCA rows)."""
    warn, rank = [], am_rank(rows)
    best = {}
    for key, sc, d, _ in AM_COLS:
        best[key] = [r["label"] for r in rows if float(f"{r['vals'][key] * sc:.{d}f}") == rank[key][0]]
    want = {PROG_LABEL: ["seconds", "clip", "hpsv2", "hpsv21", "arniqa"],
            SCA_FINAL: ["patch_fid", "musiq", "musiq_crop1024"]}
    for label, keys in want.items():
        wrong = [AM_NAMES[k] for k in keys if best[k] != [label]]
        if wrong and says(cap, "is best on"):
            warn.append(f"tab:all_metrics caption: {label} is no longer alone best at 4096^2 on {', '.join(wrong)}")
    for label in (SCA_FINAL, PROG_LABEL):
        r = next(x for x in rows if x["label"] == label)
        failing = [k for k in PER_IMAGE if sig_mark(r["marks"].get(k), False) != r"\sg"]
        if failing and says(cap, "all ten per-image metrics"):
            warn.append(f"tab:all_metrics caption 'all ten per-image metrics': {label} is not significantly better "
                        f"than AReA on {', '.join(AM_NAMES[k] for k in failing)}")
    return warn


def all_metrics(g):
    import gen_late_tables as late  # read-only use of its two readers (its writers are never called)
    warn, keys, cap = [], [k for k, _, _, _ in AM_COLS], caption("tab:all_metrics")
    lat = v2swap.latency(v2=False)
    lat_landed = v2swap.landed(v2swap.LATENCY)
    blocks = {}

    # 4096^2
    S, vs = published(MAIN)
    P_full = data195.pairs(MAIN, "full", "area")
    clean = {m: lat[("4096x4096", m)][0] for m in ("full", "area", "sca_nb")}
    rows = [am_row("Full attention", dict({k: S[("full", k)] for k in keys[1:]}, seconds=clean["full"]),
                   {k: P_full[k] for k in keys if k in P_full and k in PER_IMAGE})]
    for m, label in (("dype", "DyPE"), ("sega_official", "SEGA"), ("ultraimage", r"UltraImage$^{\circ}$")):
        rows.append(am_row(label, {k: S[(m, k)] for k in keys}, campaign=True))
    hrdit_row, prog_row, prog_warn = prog_rows(dict({k: S[("area", k)] for k in keys[1:]}, seconds=clean["area"]))
    warn += prog_warn
    rows.append(hrdit_row)
    rows.append(am_row(AREA_LABEL, dict({k: S[("area", k)] for k in keys[1:]}, seconds=clean["area"])))
    landed, unclustered = v2swap.landed(V2_MAIN), []
    if landed:
        S2, vs2, clustered = final(V2_MAIN)
        unclustered += [] if clustered else [V2_MAIN]
        vals = {k: S2[(FINAL, k)] for k in keys[1:]}
        marks = {k: vs2.get((FINAL, k)) for k in keys[1:]}
        warn += prompt_check(V2_MAIN) + interval_check(vs2, V2_MAIN, keys[1:], "tab:all_metrics 4096^2") + \
            area_check(am_show, S, S2, V2_MAIN, keys[1:], "tab:all_metrics 4096^2")
    else:
        vals = {k: S[("sca_nb", k)] for k in keys[1:]}
        marks = {k: vs.get(("sca_nb", k)) for k in keys[1:]}
    sca_lat = clean["sca_nb"]
    if lat_landed:
        v2lat = v2swap.latency(v2=True).get(("4096x4096", FINAL))
        if v2lat is None:
            warn.append("latency_curve_v2 has no sca_v2 row at 4096x4096; the published latency stays")
            lat_landed = False
        else:
            sca_lat = v2lat[0]
            if v2lat[3] != 3 and says(cap, "median of three"):
                warn.append(f"tab:all_metrics caption 'median of three': sca_v2 at 4096^2 has n={v2lat[3]}")
    pend = {k: not landed for k in keys[1:]}
    pend["seconds"] = not lat_landed
    rows.append(am_row(SCA_FINAL, dict(vals, seconds=sca_lat), marks, pend=pend, ours=True))
    rows.append(prog_row)
    blocks["4096x4096"] = rows
    warn += am_claims(rows, cap)
    if landed:
        failing = [k for k in PER_IMAGE if sig_mark(marks[k], False) != r"\sg"]
        if failing and says(cap, "every per-image metric"):
            warn.append("tab:all_metrics caption 'beats AReA on every per-image metric': sca_v2 is not "
                        f"significantly better on {', '.join(failing)}")
    if lat_landed:
        third = sca_lat / clean["area"]
        if abs(third - 1 / 3) > 0.05 and says(cap, "a third of its latency"):
            warn.append(f"tab:all_metrics caption 'about a third of its latency': sca_v2 / AReA = {third:.3f}")

    # non-square sizes: AReA and SCA (final), campaign times
    for res, (run, run2) in WIDE.items():
        S, vs = published(run)
        landed = v2swap.landed(run2)
        if landed:
            S2, vs2, clustered = final(run2)
            unclustered += [] if clustered else [run2]
            vals, marks = {k: S2[(FINAL, k)] for k in keys}, {k: vs2.get((FINAL, k)) for k in keys}
            warn += prompt_check(run2) + interval_check(vs2, run2, keys[1:], f"tab:all_metrics {res}") + \
                area_check(am_show, S, S2, run2, keys, f"tab:all_metrics {res}")
        else:
            vals, marks = {k: S[("sca_nb", k)] for k in keys}, {k: vs.get(("sca_nb", k)) for k in keys}
        blocks[res] = [am_row(AREA_LABEL, {k: S[("area", k)] for k in keys}, campaign=True),
                       am_row(SCA_FINAL, vals, {k: marks[k] for k in keys[1:]}, campaign=True, pend=not landed,
                              ours=True)]
    if unclustered and says(cap, "none for full attention's FID"):
        warn.append("tab:all_metrics caption '\\sg / \\sw ... (95% paired interval; none for full attention's FID and "
                    "FID_p)': the \\sca{} (final) FID_p cells now have no interval or mark either; "
                    + CLUSTER_NOTE.format(runs=", ".join(unclustered)))

    # 6144^2, 100 prompts: no sca_v2 run
    s6 = late.load_summary("e2_6144")
    d6 = late.load_deltas("e2_6144", "sca_nb")
    get = lambda m, k: s6[(m, late.norm(k))]  # noqa: E731
    blocks["6144x6144"] = [
        am_row(AREA_LABEL, {k: get("area", k) for k in keys}, campaign=True),
        am_row(E2_LABEL, {k: get("sca_nb", k) for k in keys}, {k: d6[late.norm(k)] for k in keys[1:]},
               campaign=True, ours=True)]

    body = []
    for i, (res, head) in enumerate(AM_BLOCKS.items()):
        if i:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{{len(AM_COLS) + 1}}}{{l}}{{\emph{{{head}}}}} \\")
        body += am_block(blocks[res])
    assert len(AM_COLS) == 13, "AM_SPEC holds one time column and twelve metric columns"
    lines = [HEADER, STYLE, v2swap.PEND_FALLBACK, r"\centering\scriptsize",
             rf"\scaFitTab{{{AM_SPEC}}}{{{len(AM_COLS) + 1}}}{{%"]
    lines += ["  " + x for x in AM_HEAD + body + [r"\bottomrule"]] + ["}"]
    return dict(name="all_metrics", kind="tabular", label="tab:all_metrics",
                runs=[V2_MAIN] + [r2 for _, r2 in WIDE.values()] + [v2swap.LATENCY, PROG_RUN],
                note="pendAppAllMetrics", tex="\n".join(lines) + "\n",
                subs=[(E2_LABEL, SCA_FINAL_OLD), (SCA_FINAL, SCA_FINAL_OLD)], warnings=warn)


# tab:app_4k and tab:app_wide ========================================================================================


def stale_source(cap, old_dir, new_run):
    """True if the caption still names a published results directory as the source of a landed sca_v2 column (the
    v5 captions ended in 'Sources: results/<run>/'); captions without a sources list pass."""
    return cap is not None and old_dir.replace("_", "\\_") in cap and new_run.replace("_", "\\_") not in cap


def app_4k(g):
    vcell = g["vcell"]
    S, vs = published(MAIN)
    landed, warn = v2swap.landed(V2_MAIN), []
    cap = caption("tab:app_4k")
    if landed:
        S2, vs2, clustered = final(V2_MAIN)
        warn += prompt_check(V2_MAIN) + interval_check(vs2, V2_MAIN, ALL, "tab:app_4k") + \
            area_check(lambda m, v: vcell({0: v}, 0, m), S, S2, V2_MAIN, ALL, "tab:app_4k")
        if not clustered and says(cap, "resample reference-image clusters"):
            warn.append("tab:app_4k caption 'Intervals use 1000 prompt draws for \\sca{} final (FID, KID, and patch "
                        "versions also resample reference-image clusters)': " + CLUSTER_NOTE.format(runs=V2_MAIN))
        if stale_source(cap, "results/a4k195_main/", V2_MAIN):
            warn.append("tab:app_4k caption 'Sources: results/a4k195_main/': \\sca{} final now comes from "
                        f"results/{V2_MAIN}/")
    est = {}

    def cells(m):
        if landed:
            e, iv = dsplit(g, vs2.get((FINAL, m)), m)
            sca = [vcell(S2, (FINAL, m), m), e, iv]
        else:
            e, iv = dsplit(g, vs.get(("sca_nb", m)), m)
            sca = [PEND(vcell(S, ("sca_nb", m), m), True), PEND(e, True), PEND(iv, True)]
        est[m] = e
        return [vcell(S, ("area", m), m)] + sca + [vcell(S, ("full", m), m), *dsplit(g, vs.get(("full", m)), m)]

    body = metric_rows(8, cells)
    # proposed caption: "... beats AReA on all ten per-image metrics, FID, and patch KID ... Key result: KID, patch FID,
    # and Inception score are unresolved"
    wins, losses = wins_losses({m: c for m, c in est.items() if m != "seconds"})
    unresolved = [m for m in ALL if m != "seconds" and m not in wins + losses]
    if says(cap, "and patch KID") and (sorted(wins) != sorted(PER_IMAGE + ["fid", "patch_kid"]) or losses):
        warn.append(f"tab:app_4k caption 'all ten per-image metrics, FID, and patch KID': single-pass \\sca{{}} is "
                    f"significantly better on {', '.join(wins)} and worse on {', '.join(losses) or 'none'}")
    if says(cap, "are unresolved") and unresolved != ["is", "kid", "patch_fid"]:
        warn.append(f"tab:app_4k caption 'KID, patch FID, and Inception score are unresolved': unresolved now "
                    f"{', '.join(unresolved) or 'none'}")
    head = [r"\toprule",
            r" & & \multicolumn{3}{c}{Single-pass \sca{}} & \multicolumn{3}{c}{Full attention} \\",
            r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
            r"Metric & AReA & Value & $\Delta$ & 95\% interval & Value & $\Delta$ & 95\% interval \\",
            r"\midrule"]
    lines = [HEADER, STYLE, v2swap.PEND_FALLBACK, r"\centering\scriptsize",
             rf"\scaFitTab{{lrr{PAIR_GAP}r{PAIR}}}{{6}}{{%"]
    lines += ["  " + x for x in head + body + [r"\bottomrule"]] + ["}"]
    return dict(name="app_4k", kind="tabular", label="tab:app_4k", runs=[V2_MAIN], note="pendAppFourK",
                tex="\n".join(lines) + "\n", subs=[], warnings=warn)


def app_wide(g):
    vcell = g["vcell"]
    cols, warn, unclustered, est = {}, [], [], {}
    cap = caption("tab:app_wide")
    for res, (run, run2) in WIDE.items():
        S, vs = published(run)
        landed = v2swap.landed(run2)
        cols[res], est[res] = {}, {}
        if landed:
            S2, vs2, clustered = final(run2)
            unclustered += [] if clustered else [run2]
            warn += prompt_check(run2) + interval_check(vs2, run2, ALL, f"tab:app_wide {res}") + \
                area_check(lambda m, v: vcell({0: v}, 0, m), S, S2, run2, ALL, f"tab:app_wide {res}")
        for m in ALL:
            if landed:
                e, iv = dsplit(g, vs2.get((FINAL, m)), m)
                cols[res][m] = [vcell(S, ("area", m), m), vcell(S2, (FINAL, m), m), e, iv]
            else:
                e, iv = dsplit(g, vs.get(("sca_nb", m)), m)
                cols[res][m] = [vcell(S, ("area", m), m), PEND(vcell(S, ("sca_nb", m), m), True), PEND(e, True),
                                PEND(iv, True)]
            est[res][m] = e
    if unclustered and says(cap, "resample reference-image clusters"):
        warn.append("tab:app_wide caption 'for FID, KID, and their patch versions 1000 draws that also resample "
                    "reference-image clusters': " + CLUSTER_NOTE.format(runs=", ".join(unclustered)))
    stale = [r2 for (r, r2) in WIDE.values() if v2swap.landed(r2) and stale_source(cap, r + "/", r2)]
    if stale:
        warn.append("tab:app_wide caption 'Sources: results/a4k195_2048x4096/, a4k195_2048x6144/': the \\sca{} "
                    f"columns now come from {', '.join('results/' + r + '/' for r in stale)}")
    # proposed caption, key result: "at 2048x6144 single-pass SCA wins seven of 15 metrics and loses two, at 2048x4096
    # it wins three and loses three, and it is faster at both"
    counts = {}
    for res in WIDE:
        wins, losses = wins_losses({m: c for m, c in est[res].items() if m != "seconds"})
        counts[res] = (len(wins), len(losses))
        if not est[res]["seconds"].startswith(r"\up{") and says(cap, "faster at both"):
            warn.append(f"tab:app_wide caption 'faster at both': single-pass \\sca{{}} is not significantly faster at "
                        f"{res}")
    if says(cap, "wins seven of 15") and counts != {"2048x4096": (3, 3), "2048x6144": (7, 2)}:
        warn.append("tab:app_wide caption 'at 2048x6144 ... wins seven of 15 metrics and loses two, at 2048x4096 it "
                    f"wins three and loses three': now (wins, losses) {counts}")
    body = metric_rows(9, lambda m: cols["2048x4096"][m] + cols["2048x6144"][m])
    single_pass = r"\shortstack[r]{Single-pass\\\sca{}}"
    head = [r"\toprule",
            r" & \multicolumn{4}{c}{$2048\times4096$, factors (2, 4)} & "
            r"\multicolumn{4}{c}{$2048\times6144$, factors (2, 6)} \\",
            r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
            rf"Metric & AReA & {single_pass} & $\Delta$ & 95\% interval & AReA & {single_pass} & $\Delta$ & "
            r"95\% interval \\",
            r"\midrule"]
    lines = [HEADER, STYLE, v2swap.PEND_FALLBACK, r"\centering\scriptsize",
             rf"\scaFitTab{{lrr{PAIR_GAP}rr{PAIR}}}{{7}}{{%"]
    lines += ["  " + x for x in head + body + [r"\bottomrule"]] + ["}"]
    return dict(name="app_wide", kind="tabular", label="tab:app_wide", runs=[r2 for _, r2 in WIDE.values()],
                note="pendAppWide", tex="\n".join(lines) + "\n", subs=[], warnings=warn)


# tab:latency ========================================================================================================
LAT_SIZES = ["2048x2048", "3072x3072", "4096x4096", "6144x6144", "8192x8192"]


def latency_table():
    pub = v2swap.latency(v2=False)
    landed = v2swap.landed(v2swap.LATENCY)
    v2 = v2swap.latency(v2=True) if landed else {}
    warn, body, slower = [], [], []
    for res in LAT_SIZES:
        w, h = (int(x) for x in res.split("x"))
        assert w == h, res
        label = f"${w}^2$ ({(w // 16) * (h // 16):,})".replace(",", "{,}")
        full = pub[(res, "full")][0] if (res, "full") in pub else batch1_median(res, "full")
        cells = {"full": None if full is None else num(full, 1), "area": num(pub[(res, "area")][0], 1)}
        pending = not landed or (res, FINAL) not in v2
        if landed and pending:
            warn.append(f"latency_curve_v2 has no sca_v2 row at {res}; the published value stays")
        if not pending and v2[(res, FINAL)][3] != 3 and says(caption("tab:latency"), "median of three"):
            warn.append(f"tab:latency caption 'median of three timed images': sca_v2 at {res} has "
                        f"n={v2[(res, FINAL)][3]}")
        cells["sca"] = num((pub[(res, "sca_nb")] if pending else v2[(res, FINAL)])[0], 1)
        fastest = min(float(c) for c in cells.values() if c is not None)
        out = []
        for k in ("full", "area", "sca"):
            c = cells[k]
            if c is None:
                out.append(r"{\color{scaCtx}not run}")
                continue
            if float(c) == fastest:
                c = rf"\textbf{{{c}}}"
            out.append(PEND(c, pending) if k == "sca" else c)
        if not pending and float(cells["sca"]) != fastest:
            slower.append(res)
        body.append(f"  {label} & " + " & ".join(out) + r" \\")
    if slower:
        warn.append("main text '\\sca{} is the fastest method we timed at every size': sca_v2 is not the fastest at "
                    + ", ".join(slower))
    # caption key result 'single-pass SCA is 2.6 times faster than AReA at 4096^2', from the printed cells
    cap = caption("tab:latency")
    said = re.search(r"([0-9.]+) times faster than AReA at \$4096\^2\$", cap or "")
    row = next(b for b in body if b.lstrip().startswith("$4096^2$"))
    printed = [float(x) for x in re.findall(r"(?<![\d.])(\d+\.\d)(?!\d)", row.split("&", 2)[2])]
    if said and len(printed) == 2 and f"{printed[0] / printed[1]:.1f}" != said.group(1):
        warn.append(f"tab:latency caption '{said.group(0)}': the printed times give "
                    f"{printed[0] / printed[1]:.1f} times")
    lines = [HEADER, STYLE, v2swap.PEND_FALLBACK, r"\centering\scriptsize", r"\scaFitTab{lrrr}{4}{%",
             r"  \toprule", r"   & \multicolumn{3}{c}{Seconds per image $\downarrow$} \\", r"  \cmidrule(l){2-4}",
             r"  Size ($N$ tokens) & Full attention & AReA & Single-pass \sca{} \\",
             r"  \midrule"] + body + [r"  \bottomrule", "}"]
    return dict(name="latency", kind="tabular", label="tab:latency", runs=[v2swap.LATENCY], note="pendAppLatency",
                tex="\n".join(lines) + "\n", subs=[], warnings=warn)


# tab:published_compare and tab:published_hrgen ======================================================================
# (block id in build.py, our SCA method there) -> sca_v2 run whose final label replaces it
SWITCH = {("krea", "sca_krea"): "krea195_sca_v2", ("dev", "sca_nb"): "a4k195_main_sca_v2",
          ("hrgen_krea", "sca_krea"): "hrgen1k_krea_sca_v2", ("hrgen_dev", "sca_nb"): "hrgen1k_dev_sca_v2"}
AREA_RUN = {"krea": "krea195_sca_v2", "dev": "a4k195_main_sca_v2", "hrgen_dev": "hrgen1k_dev_sca_v2"}


def build_ns():
    """results/published_compare/build.py executed into a namespace: its constants, readers and formatters. Its
    main(), write_tex() and write_hrgen_appendix() write files and are never called; compiling from source writes no
    bytecode into results/."""
    ns = {"__name__": "published_compare_build", "__file__": str(BUILD)}
    exec(compile(BUILD.read_text(), str(BUILD), "exec"), ns)
    return ns


def our_values(B):
    """({(block, method, metric): (value, n)}, {(block, method): pending}, warnings): build.py's `ours` as its main()
    computes it, with each switching \\sca{} row replaced by its landed sca_v2 run (final label), read like gather()
    reads a run directory (summaries first, patch FID from corrected_fidp/summary.csv)."""
    ours = {}
    for bid, _, _, _, _, methods, sources in B["BLOCKS"]:
        vals, _ = B["gather"](sources)
        for m in methods:
            for metric in B["METRICS"]:
                if (m, metric) in vals:
                    ours[(bid, m, metric)] = vals[(m, metric)]
    pending, warn = {}, {}
    for (bid, m), run in SWITCH.items():
        pending[(bid, m)] = not v2swap.landed(run)
        if pending[(bid, m)]:
            continue
        d, final_m = v2swap.V2_RESULTS / run, v2swap.RUNS[run].final
        new = {}
        for p in (d / "base_summary.csv", d / "extra_summary.csv"):
            for (meth, metric), v in B["read_summary"](p).items():
                if meth == final_m and metric in B["METRICS"]:
                    new.setdefault(metric, v)
        for (meth, metric), v in B["read_summary"](d / "corrected_fidp" / "summary.csv").items():
            if meth == final_m and metric == "patch_fid":
                new.setdefault("fidp", v)
        old = {k for k in ours if k[:2] == (bid, m)}
        lost = sorted({k[2] for k in old} - set(new))
        if lost:
            warn.setdefault(run, []).append(f"{run}: no {', '.join(lost)} for {final_m}; those cells print '--'")
        for k in old:
            del ours[k]
        for metric, v in new.items():
            ours[(bid, m, metric)] = v
    return ours, pending, warn


def area_check_published(B, ours, warn):
    """AReA in the switched blocks: published value (as printed) against the landed run's own AReA."""
    for bid, run in AREA_RUN.items():
        if not v2swap.landed(run):
            continue
        d = v2swap.V2_RESULTS / run
        new = {}
        for p in (d / "base_summary.csv", d / "extra_summary.csv"):
            for (meth, metric), v in B["read_summary"](p).items():
                if meth == "area":
                    new.setdefault(metric, v[0])
        for metric in B["FMT"]:
            if (bid, "area", metric) in ours and metric in new:
                a, b = B["fmt"](metric, ours[(bid, "area", metric)][0]), B["fmt"](metric, new[metric])
                if a != b:
                    warn.setdefault(run, []).append(f"{bid} block: our AReA {metric} prints {a} (published run) but "
                                                    f"{b} in {run}")


def cell(B, c, vals, rankable, best, pend, text=None):
    if c not in vals:
        return "--"
    s = text if text is not None else B["fmt"](c, vals[c])
    if s.startswith("-"):
        s = "$-$" + s[1:]
    if rankable and best[c] and round(vals[c], 6) == best[c][0]:
        s = f"\\textbf{{{s}}}"
    elif rankable and len(best[c]) > 1 and round(vals[c], 6) == best[c][1]:
        s = f"\\underline{{{s}}}"
    return PEND(s, pend)


def row_tex(B, row, best, cols, full_n):
    """build.py's row_tex in the v6 style: our rows shaded (\\scaRowOurs), \\pend on the cells of a pending row
    (row = group, label, vals, n, rankable, ours, pending)."""
    _, label, vals, n, rankable, is_ours, pend = row
    cells = [cell(B, c, vals, rankable, best, pend) for c in cols]
    ntxt = f"{n}" + ("" if n == full_n else DAGGER)
    return (OURS if is_ours else "") + f"{label} & {ntxt} & " + " & ".join(cells) + " \\\\"


OUR_ROW = "Single-pass \\sca{} (ours)"  # the checkpoint is named by the row's group head


def hrgen_ours(B, ours, pending):
    """build.py's hrgen_ours with the pending flag appended."""
    rows = []
    for bid, m, label in (("hrgen_krea", "sca_krea", OUR_ROW), ("hrgen_dev", "area", "AReA"),
                          ("hrgen_dev", "sca_nb", OUR_ROW)):
        vals = {k: ours[(bid, m, k)][0] for k in B["METRICS"] if (bid, m, k) in ours}
        if vals:
            n = min(ours[(bid, m, k)][1] for k in vals)
            rows.append((bid, label, vals, n, bid == "hrgen_krea" and n == 1000, m.startswith("sca"),
                         pending.get((bid, m), False)))
    return rows


def join_words(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + (", and " if len(xs) > 2 else " and ") + xs[-1]


def pub_name(B, c):
    """A metric's name in the captions of the published tables (build.py's METRICS label; HPS is HPSv2 there)."""
    return "HPSv2" if c == "hpsv2" else B["METRICS"][c][0]


def pub_head(B, c):
    return {"ir": "ImageReward"}.get(c, pub_name(B, c)) + ("$\\uparrow$" if B["METRICS"][c][1] else "$\\downarrow$")


def compare_text(B, rows):
    """(lead, key result) of tab:published_compare from the printed values, each (text, pending). Lead: how closely
    our FLUX.1-Krea-dev run of AReA's code reproduces AReA's published Aesthetic-4K row (FID left out of the
    tolerance, as in v5.1, and counted against it). Key result: single-pass SCA against that run, point estimates
    on the same prompts (build.py's key_result, second clause). The HRGen-1K clause moved to tab:published_hrgen."""
    METRICS, COLS, fmt = B["METRICS"], B["COLS"], B["fmt"]
    pubarea = next(r for r in rows if r[0] == "pub" and r[1] == "AReA")[2]
    ours_area = next((r for r in rows if r[0] == "krea" and r[1] == "AReA"), None)
    ours_sca = next((r for r in rows if r[0] == "krea" and r[5]), None)
    shown = lambda c, v: float(fmt(c, v))  # noqa: E731
    lead = ("Published numbers at $4096^2$ next to our runs", False)
    if ours_area:
        both = [c for c in COLS if c in ours_area[2] and c in pubarea]
        diff = {c: abs(shown(c, ours_area[2][c]) - shown(c, pubarea[c])) for c in both}
        gap = max(diff[c] for c in both if c != "fid")
        n = sum(diff[c] <= gap + 1e-9 for c in both)
        what = (f"all {WORDS[n]} of its published Aesthetic-4K scores" if n == len(both) else
                f"{WORDS[n]} of its {WORDS[len(both)]} published Aesthetic-4K scores")
        lead = (f"Our run of AReA's code on FLUX.1-Krea-dev reproduces {what} within {gap:.2f}", False)
    key = ("no paired run of single-pass \\sca{} on FLUX.1-Krea-dev", False)
    if ours_sca and ours_area and ours_sca[3] == ours_area[3]:
        a, s_ = ours_area[2], ours_sca[2]
        cols = [c for c in COLS if c in a and c in s_]
        delta = {c: (shown(c, s_[c]) - shown(c, a[c])) * (1 if METRICS[c][1] else -1) for c in cols}
        parts = []
        for words, keep in (("better on", lambda x: x > 0.1 + 1e-9), ("within 0.1 on", lambda x: abs(x) <= 0.1 + 1e-9),
                            ("worse on", lambda x: x < -0.1 - 1e-9)):
            names = [pub_name(B, c) for c in cols if keep(delta[c])]
            if names:
                parts.append(f"{words} {join_words(names)}")
        # a point-estimate comparison of two of our runs on the same prompts; krea_6144 holds its paired intervals
        key = ("against our FLUX.1-Krea-dev run of AReA, the point estimates of single-pass \\sca{} are "
               + join_words(parts) + "; Table~\\ref{tab:krea_6144} gives the paired intervals", ours_sca[6])
    return lead, key


def hrgen_text(B, rows, hpub):
    """(lead, key result) of tab:published_hrgen, each (text, pending). Lead: single-pass SCA on FLUX.1-Krea-dev
    against every score AReA publishes for HRGen-1K (build.py's key). Key result: that comparison by metric, and the
    paired FLUX.1-dev runs, ours against AReA's code, by metric (point estimates of the printed values)."""
    METRICS, HCOLS, fmt = B["METRICS"], B["HCOLS"], B["fmt"]
    krea = next((r for r in rows if r[0] == "hrgen_krea"), None)
    area = hpub["AReA"]
    if not krea:
        return ("HRGen-1K: AReA's published Table~2 next to our runs", False), ("pending", False)
    cols = [c for c in HCOLS if c in krea[2] and c in area]
    hi = [c for c in cols if (krea[2][c] >= area[c]) == METRICS[c][1]]
    lo = [c for c in cols if c not in hi]
    some = f" ({krea[3]} of 1000 prompts)" if krea[3] != 1000 else ""
    if hi and not lo:
        lead = f"On HRGen-1K, single-pass \\sca{{}} exceeds every score AReA publishes{some}"
    elif hi:
        lead = f"On HRGen-1K, single-pass \\sca{{}} exceeds {WORDS[len(hi)]} of the {WORDS[len(cols)]} scores AReA publishes{some}"
    else:
        lead = f"On HRGen-1K, single-pass \\sca{{}} does not exceed the scores AReA publishes{some}"
    key = ("single-pass \\sca{} on FLUX.1-Krea-dev " + (f"exceeds AReA's published {join_words([pub_name(B, c) for c in hi])}"
           if hi else "does not exceed AReA's published numbers")
           + (f" and trails it on {join_words([pub_name(B, c) for c in lo])}" if lo else ""))
    pend = krea[6]
    dev_area = next((r for r in rows if r[0] == "hrgen_dev" and not r[5]), None)
    dev_sca = next((r for r in rows if r[0] == "hrgen_dev" and r[5]), None)
    if dev_area and dev_sca:
        shown = lambda c, v: float(fmt(c, v))  # noqa: E731
        dcols = [c for c in HCOLS if c in dev_area[2] and c in dev_sca[2]]
        up = [c for c in dcols if (shown(c, dev_sca[2][c]) - shown(c, dev_area[2][c])) * (1 if METRICS[c][1] else -1) > 0]
        if len(up) == len(dcols):
            key += f"; on FLUX.1-dev its point estimates exceed our AReA run on all {WORDS[len(dcols)]} metrics"
        elif up:
            key += (f"; on FLUX.1-dev its point estimates exceed our AReA run on {join_words([pub_name(B, c) for c in up])}"
                    f" of {WORDS[len(dcols)]} metrics")
        else:
            key += "; on FLUX.1-dev its point estimates exceed our AReA run on none of the metrics"
        pend = pend or dev_sca[6]
    return (lead, krea[6]), (key, pend)


def sentence(text, pending):
    """A caption clause with its first letter capitalised, greyed while its \\sca{} run is pending."""
    return PEND(text[0].upper() + text[1:], pending)


def published_compare(B, pub, ours, pending):
    """build.py's write_tex, returned as text, with our \\sca{} rows switched; v6 table style and caption."""
    METRICS, COLS = B["METRICS"], B["COLS"]
    pubv, prec1 = {}, set()
    for p in pub:
        if p["source"] == "AReA" and p["benchmark"] == "Aesthetic-4K" and p["res"] == "4096x4096":
            pubv.setdefault(p["method"], {}).setdefault(p["metric"], float(p["value"]))
            if p["table"].startswith("Table 6") and p["metric"] == "clip":
                prec1.add(p["method"])
    rows = []
    for name in ["Base (full attention)", "UltraImage", "DyPE", "SigMa", "SEGA", "AReA"]:
        rows.append(("pub", name.replace("Base (full attention)", "Full attention"), pubv[name], 195, True, False,
                     False))
    for bid in ("krea", "dev"):
        ms = ["area", "dype", "sega_official", "sca_krea"] if bid == "krea" else ["area", "sca_nb"]
        for m in ms:
            vals = {k: ours[(bid, m, k)][0] for k in COLS if (bid, m, k) in ours}
            ns = {ours[(bid, m, k)][1] for k in COLS if (bid, m, k) in ours}
            if not vals:
                continue
            n = min(ns)
            rows.append((bid, OUR_ROW if m.startswith("sca") else B["OUR_NAME"][m], vals, n, n == 195,
                         m.startswith("sca"), pending.get((bid, m), False)))
    # Review round 1: rank within each block, never across blocks (published and our runs use different seeds and
    # checkpoints). Blocks with more than two ranked rows mark best and second; two-row blocks mark the best only.
    best = {}
    for g in {r[0] for r in rows}:
        ranked = [r for r in rows if r[0] == g and r[4]]
        best[g] = {}
        for c in COLS:
            cand = sorted({round(r[2][c], 6) for r in ranked if c in r[2]}, reverse=METRICS[c][1])
            best[g][c] = cand[:2] if len(ranked) > 2 else cand[:1]
    hpub = B["hrgen_published"](pub)
    hrows = [("hpub", "AReA, published", hpub["AReA"], 1000, True, False, False)] + \
        [r for r in hrgen_ours(B, ours, pending) if r[0] == "hrgen_krea"]
    hrows = [(g, lab, {c: v for c, v in vals.items() if c in COLS}, n, rk, o, p) for g, lab, vals, n, rk, o, p in hrows]
    hbest = {}
    for c in COLS:
        cand = sorted({round(r[2][c], 6) for r in hrows if r[4] and c in r[2]}, reverse=METRICS[c][1])
        hbest[c] = cand[:1] if len(cand) > 1 else []
    (lead, lead_p), (key, key_p) = compare_text(B, rows)
    ncol = len(COLS) + 2
    lines = [
        HEADER + " Logic of results/published_compare/build.py.", STYLE, v2swap.PEND_FALLBACK,
        "\\begin{table}[t]",
        "\\caption{\\textbf{" + sentence(lead, lead_p) + ".} Expands the AReA comparison of Table~\\ref{tab:main} with "
        "AReA's own published numbers. Aesthetic-4K, \\nmain{} prompts at $4096^2$. Rows: AReA's Table~1 (HPS and MUSIQ "
        "from its Table~6; ``FLUX.1''), our runs on FLUX.1-Krea-dev, which AReA's released code uses (28 steps, "
        "guidance 4.5), and our FLUX.1-dev protocol. The last block repeats the HRGen-1K comparison of "
        "Table~\\ref{tab:published_hrgen}. Published numbers have no intervals, so every comparison here is between "
        "point estimates. HPSv2 $\\times100$; MUSIQ on the full image; FID at $299^2$ against the 195 reference rows; "
        "patch FID is omitted because AReA does not state its crops; dashes: not published. \\textbf{Bold} / "
        "\\underline{underline}: best / second best within each block (bold only in two-row blocks); "
        "$^\\dagger$fewer than all prompts, not ranked; ours shaded. Key result: " + PEND(key, key_p)
        + ".\\pendAppPublishedCompare}",
        "\\label{tab:published_compare}",
        "\\centering\\scriptsize",
        f"\\scaFitTab{{lr*{{{len(COLS)}}}{{r}}}}{{{ncol}}}{{%",
        "  \\toprule",
        "  Method & $n$ & " + " & ".join(pub_head(B, c) for c in COLS) + " \\\\",
        "  \\midrule",
    ]
    titles = {"pub": "Published by AReA, single pass (FLUX.1)", "pubms": "Published by AReA, multi-stage (FLUX.1)",
              "krea": "Our runs, FLUX.1-Krea-dev, same prompts and metrics",
              "dev": "Our runs, FLUX.1-dev, shared sampler"}
    group = None
    for g, label, vals, n, rankable, is_ours, pend in rows:
        if g != group:
            if group is not None:
                lines.append("  \\midrule")
            lines.append(f"  \\multicolumn{{{ncol}}}{{l}}{{\\emph{{{titles[g]}}}}} \\\\")
            group = g
        cells = [cell(B, c, vals, rankable, best[g], pend,
                      f"{vals[c]:.1f}" if (c == "clip" and g == "pubms" and label in prec1 and c in vals) else None)
                 for c in COLS]
        ntxt = f"{n}" + ("" if n == 195 else DAGGER)
        lines.append("  " + (OURS if is_ours else "") + f"{label} & {ntxt} & " + " & ".join(cells) + " \\\\")
    lines.append("  \\midrule")
    lines.append(f"  \\multicolumn{{{ncol}}}{{l}}{{\\emph{{HRGen-1K, 1000 prompts without references: AReA's Table~2 "
                 "and our FLUX.1-Krea-dev run}} \\\\")
    for r in hrows:
        lines.append("  " + row_tex(B, r, hbest, COLS, 1000))
    lines += ["  \\bottomrule", "}", "\\end{table}"]
    runs = [SWITCH[("krea", "sca_krea")], SWITCH[("dev", "sca_nb")], SWITCH[("hrgen_krea", "sca_krea")]]
    return dict(name="published_compare", kind="float", label="tab:published_compare", old="published_compare.tex",
                runs=runs, note="pendAppPublishedCompare", tex="\n".join(lines) + "\n", subs=[], warnings=[])


def published_hrgen(B, pub, ours, pending):
    """build.py's write_hrgen_appendix, returned as text, with our \\sca{} rows switched; v6 table style and
    caption. Our rows are split by checkpoint, each under its own group head."""
    METRICS, HCOLS = B["METRICS"], B["HCOLS"]
    hpub = B["hrgen_published"](pub)
    rows = [("hpub", n.replace("Base (full attention)", "Full attention"), hpub[n], 1000, True, False, False)
            for n in ["Base (full attention)", "UltraImage", "DyPE", "SigMa", "SEGA", "AReA"]] + \
        hrgen_ours(B, ours, pending)
    best = {}
    for c in HCOLS:
        cand = sorted({round(r[2][c], 6) for r in rows if r[4] and c in r[2]}, reverse=METRICS[c][1])
        best[c] = cand[:2] if len(cand) > 1 else []
    (lead, lead_p), (key, key_p) = hrgen_text(B, rows, hpub)
    ncol = len(HCOLS) + 2
    lines = [
        HEADER + " Logic of results/published_compare/build.py.", STYLE, v2swap.PEND_FALLBACK,
        "\\begin{table}[!htbp]",
        "\\caption{\\textbf{" + sentence(lead, lead_p) + ".} Expands the HRGen-1K block of "
        "Table~\\ref{tab:published_compare} to every row of AReA's Table~2. All 1000 HRGen-1K prompts at $4096^2$, "
        "without reference images. Top: AReA's Table~2 (``FLUX.1'', which its released code runs as "
        "FLUX.1-Krea-dev). Below: single-pass \\sca{} on FLUX.1-Krea-dev with a reach table fitted on that checkpoint "
        "(28 steps, guidance 4.5), then our paired FLUX.1-dev runs, whose differences Table~\\ref{tab:generality} "
        "reports. Published numbers have no intervals, so every comparison here is between point estimates. HPSv2 "
        "$\\times100$; IS: Inception score; MUSIQ on the full image; dashes: not published. \\textbf{Bold} / "
        "\\underline{underline}: best / second best among the published rows and our FLUX.1-Krea-dev row; ours "
        "shaded. Key result: " + PEND(key, key_p) + ".\\pendAppPublishedHrgen}",
        "\\label{tab:published_hrgen}",
        "\\centering\\scriptsize",
        f"\\scaFitTab{{lr*{{{len(HCOLS)}}}{{r}}}}{{{ncol}}}{{%",
        "  \\toprule",
        "  Method & $n$ & " + " & ".join(pub_head(B, c) for c in HCOLS) + " \\\\",
        "  \\midrule",
    ]
    titles = {"hpub": "Published by AReA (FLUX.1)", "hrgen_krea": "Our run, FLUX.1-Krea-dev",
              "hrgen_dev": "Our paired runs, FLUX.1-dev, shared sampler"}
    group = None
    for r in rows:
        if r[0] != group:
            if group is not None:
                lines.append("  \\midrule")
            lines.append(f"  \\multicolumn{{{ncol}}}{{l}}{{\\emph{{{titles[r[0]]}}}}} \\\\")
            group = r[0]
        lines.append("  " + row_tex(B, r, best, HCOLS, 1000))
    lines += ["  \\bottomrule", "}", "\\end{table}"]
    runs = [SWITCH[("hrgen_krea", "sca_krea")], SWITCH[("hrgen_dev", "sca_nb")]]
    return dict(name="published_hrgen", kind="float", label="tab:published_hrgen", old="published_hrgen.tex",
                runs=runs, note="pendAppPublishedHrgen", tex="\n".join(lines) + "\n", subs=[], warnings=[])


def published_tables():
    B = build_ns()
    pub = list(csv.DictReader(open(BUILD.parent / "published.csv")))
    ours, pending, warn = our_values(B)
    area_check_published(B, ours, warn)
    for (bid, _), run in SWITCH.items():
        if v2swap.landed(run):
            warn.setdefault(run, []).extend(prompt_check(run, 1000 if bid.startswith("hrgen") else N_MAIN))
    tabs = [published_compare(B, pub, ours, pending), published_hrgen(B, pub, ours, pending)]
    for t in tabs:
        t["warnings"] = [w for run in t["runs"] for w in warn.get(run, [])]
    return tabs


# entry point ========================================================================================================
def tables():
    g = gen_tables_ns()
    return [all_metrics(g), app_4k(g), app_wide(g), latency_table()] + published_tables()


if __name__ == "__main__":
    for t in tables():
        print(f"{t['name']} ({t['label']}, {t['kind']}): runs {t['runs']}, note \\{t['note']}")
        for w in t["warnings"]:
            print("    WARNING", w)
