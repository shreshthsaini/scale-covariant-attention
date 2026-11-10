"""Main-text Tables 1 to 3, two appendix tables and the headline numbers of the prose, generated from results/
(nothing typed by hand).

One command, run from the repository root, regenerates every file:

    PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/work/09032/saini_2/vista/library/uv-cache uv run --no-project \\
        --python /work/09032/saini_2/vista/library/uv-envs/sca/bin/python python paper/figures/src/gen_main_tables.py

Outputs, in paper/figures/snippets/:
  main_tables/tab_main.tex, tab_ablation.tex, tab_generality.tex
      the \\setlength{\\tabcolsep} line and the tabular of main-text Tables tab:main, tab:ablation and
      tab:generality. The float, caption, label and font line stay in main.tex, which \\input's the file in place
      of the tabular. Each file ends in "\\end{tabular}%" so the \\input adds no extra space.
  main_tables/tab_main_compact.tex, tab_ablation_compact.tex (2026-11-10)
      the main text's lean Table 1 (time, IR, HPSv2.1, PickScore, MUSIQ_c, patch FID; bold best per group) and
      ablation (reach weights, then views); tab_main.tex and tab_ablation.tex keep the full versions.
      tab_ablation_compact_prog.tex adds the progressive-schedule rows of results/prog195_ablation/ and is written
      only once that folder has SHA256SUMS (main.tex switch \\ifresprogabl).
  main_tables/tab_panel46.tex
      the tabular of appendix Table tab:panel46 (SigMa and the sampling budgets of the operator without the
      schedule); its float and caption live in appendix.tex.
  app_tables/prog_all.tex
      the complete float of appendix Table tab:prog_all: progressive SCA minus every 4096^2 baseline on every metric.
  v6 (2026-11-10, tablesplots owner): main_tables/tabstyle.tex (the shared table macros: \\scaFitTab, \\scaRowOurs,
      \\camp, \\scaCI, colours scaRowTint and scaCtx; every v6 snippet inputs it once); Table 1 at full width with the
      speed ratio and the Holm-adjusted ten-metric record; Table 2 at full width with a per-head reference row per
      group; Table 3 at 0.48 width; appendix floats app_tables/holm.tex (tab:holm, with gen_holm.py), gen_all.tex
      (tab:gen_all), app_abl.tex, app_ps2.tex, phase2_sensitivity.tex, phase2_budget.tex (generated replacements of
      tables v5.1 typed into appendix.tex), and the restyled prog_all.tex, prog_abl_all.tex, tab_panel46.tex; headline
      macros \\holm..., \\hlLat..., and the numbers the v6 text requested (request_macros).
  app_tables/holm_short.tex (2026-11-10, appendix owner)
      panel (a) of holm.tex alone, with its own caption, for the short review edition (main_short.tex inputs it in
      place of holm.tex, so no table of that edition prints the record panel).
  main_tables/headline.tex
      \\newcommand{\\hl...} macros for the numbers the prose repeats, and the caption notes \\pendTabMain,
      \\pendTabPanel, \\pendTabAbl, \\pendTabGen; \\input it once in the preamble. A comment above each macro names
      its file and column.

Configurations (2026-11-07 rewrite, user decision): the headline is progressive SCA (sca_v2_prog: HRDiT's three-stage
schedule with the final operator sca_v2 at its 2048^2 and 4096^2 stages); single-pass SCA (sca_v2, 28 steps) is the
fast mode; SCA without the schedule (sca_nb) appears only in the ablation and generality rows that exist for it.
The paper does not report duplication: no table has a Dup. column and headline.tex has no duplication macro.

Inputs: results/*.csv (the a4k195_* runs through data195.py, the sca_v2 runs through scav2_runs.py), the paired files
of progressive SCA against every baseline (results/prog195_vs_baselines/sca_v2_prog_vs_<baseline>.csv, per-image,
2000 prompt draws; set-level intervals from results/prog195_pairs/cfid_vs_{area,sca_v2}/paired_deltas.csv and, once
it lands, results/prog195_vs_baselines/cfid/paired_deltas.csv, 1000 draws over prompts and 168 reference
identities), the sca_v2 runs' eval directories on scratch for what their tasks do not copy (eval.py paired deltas,
per-image CSVs), the 46-prompt runs' eval directories (eval/dup46_<label>), and one archived JSON:
results/evidence/calib/k1_1024.json, scales/4/rel_err (checked against Table K.5 and Section 3).

Rules, all set below rather than typed into the cells:
  * numbers use Python's format() at each column's precision; a negative value is set in math mode;
  * Table 1: bold = best and underline = second best of each column within its group (single pass; multi-stage),
    ranked on the printed values (ties share the mark); the campaign-time column is not ranked;
  * Table 1 colours: every row except progressive SCA is coloured against progressive SCA, the headline: \\up{}
    (blue) where the row's 95% paired interval against progressive SCA excludes zero in the row's favour, \\dn{}
    (orange) where it excludes zero against the row, plain where it includes zero or no interval exists (the caption
    names those cells). This is the paper's one significance convention (main.tex, Section 4, Metrics);
  * clean seconds: median of the batch-1 timings at 4096^2 where a method was timed, otherwise "n/m";
  * Table tab:panel46 (appendix): bold = best within each group; colours against single-pass SCA at 28 steps
    (per-image metrics through scav2_runs.derived_pairs; no FID interval exists, so those cells stay plain), except
    the budget rows, which are coloured against the published operator from results/a4k195_panel;
  * Table 2 (ablations): change against the group's reference; Table 3: SCA minus the reference with the interval
    printed under each difference; rows of runs not yet landed print "pending";
  * shading: both of our rows in Table 1, the default row of Table 2.
After writing, the script checks the prompt counts and the qualitative claims printed next to these numbers;
if one fails it lists it and exits with status 1 (the files are still written).
"""
from __future__ import annotations

import csv
import json
import math
from decimal import ROUND_HALF_UP, Decimal
import re
import statistics
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data195  # noqa: E402  (results/a4k195_* loader; reads CSVs only)
import scav2_runs as V  # noqa: E402  (sca_v2 runs: landed or pending, loaders; read-only)

ROOT = HERE.parents[2]
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures" / "snippets" / "main_tables"
OUT_APP = ROOT / "paper" / "figures" / "snippets" / "app_tables"
CALIB_JSON = "evidence/calib/k1_1024.json"
RES4K = "4096x4096"
MAIN_RUN, PANEL_RUN, REF = "a4k195_main", "a4k195_panel", "sca_nb"
MAIN_V2, PROG_V2, WIDE_V2, KREA_V2 = "a4k195_main_sca_v2", "prog195_sca_v2", "a4k195_2048x6144_sca_v2", "krea195_sca_v2"
PROG = "sca_v2_prog"  # progressive SCA's method label in every progressive file
PROG_PAIRS = "prog195_vs_baselines"  # results/<dir>/sca_v2_prog_vs_<baseline>.csv
PROG_CFID = {"area": "prog195_pairs/cfid_vs_area/paired_deltas.csv",
             "sca_v2": "prog195_pairs/cfid_vs_sca_v2/paired_deltas.csv"}
PROG_CFID_ALL = "prog195_vs_baselines/cfid/paired_deltas.csv"  # reference sca_v2_prog; lands after the CPU job
PROG_LAT = "prog_latency/latency_summary.csv"  # sweep 3: progressive SCA, HRDiT, AReA back to back on one GH200
DEFAULT46 = "dup_i_shrink"  # the final operator on the 46-prompt subset
SHADE = r"\rowcolor[HTML]{EAF4F1}"
HEADER = "% Generated by paper/figures/src/gen_main_tables.py from results/; do not edit by hand."
# Shared table style (tabstyle.tex, written by this script): every v6 table snippet loads it once.
TABSTYLE_INPUT = r"\ifdefined\scaFitTab\else\input{figures/snippets/main_tables/tabstyle.tex}\fi"
SPEED_T1: dict = {}


def tabstyle_tex():
    """main_tables/tabstyle.tex: the macros every generated v6 table uses (all \providecommand / \providecolor, so the
    file can be input more than once and never overrides a definition of main.tex)."""
    return "\n".join([
        HEADER,
        "% Shared style of the generated tables (paper v6). Each table snippet inputs this file once:",
        "%   " + TABSTYLE_INPUT,
        r"\providecolor{scaRowTint}{HTML}{EAF4F1}% our rows: pale tint of the method green (scafig.SCA #13865F)",
        r"\providecolor{scaCtx}{HTML}{6E7781}% intervals and context text",
        r"\providecommand{\scaRowOurs}{\rowcolor{scaRowTint}}",
        r"% campaign mean on shared nodes, where a method has no clean time; hangs right so digits stay aligned",
        r"\providecommand{\camp}{\rlap{$^\ddagger$}}",
        r"% \scaCI{lo}{hi}: a 95% interval in grey after its estimate, at the cell's size",
        r"\providecommand{\scaCI}[2]{{\color{scaCtx}\,[#1,\,#2]}}",
        r"% \scaFitTab{colspec}{columns}{rows}: a booktabs tabular as wide as \linewidth, made by spreading the spare",
        r"% width evenly over the 2 x columns cell paddings (tabcolsep kept between 2 and 9 pt). Column specs carry no",
        r"% @{} so that shaded rows end where the rules end; unlike tabular*, shaded rows stay seamless. The rows are",
        r"% set twice: once to measure, once to print.",
        r"\providecommand{\scaFitTab}[3]{\begingroup\setlength{\tabcolsep}{0pt}%",
        r"  \sbox0{\begin{tabular}{#1}#3\end{tabular}}%",
        r"  \setlength{\tabcolsep}{\dimexpr(\linewidth-\wd0)/\numexpr2*#2\relax\relax}%",
        r"  \ifdim\tabcolsep<2pt\setlength{\tabcolsep}{2pt}\fi\ifdim\tabcolsep>9pt\setlength{\tabcolsep}{9pt}\fi",
        r"  \begin{tabular}{#1}#3\end{tabular}\endgroup}",
    ]) + "\n"
FAILS: list[str] = []
NOTES: list[str] = []
PENDING: dict[str, bool] = {}
NO_CI: dict[str, list[str]] = {}
_DATA: dict = {}


def check(ok, message):
    if not ok:
        FAILS.append(message)


def rows(rel):
    path = RES / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return list(csv.DictReader(open(path)))


def data(run):
    """(S, D, ref, corrected_ci, deltas source) of a run, loaded once."""
    if run not in _DATA:
        _DATA[run] = V.load(run)
    return _DATA[run]


def num(x, d):
    """Absolute values: plain digits, a negative value in math mode ($-2.18$)."""
    s = f"{x:.{d}f}"
    return f"${s}$" if s.startswith("-") else s


def sgn(x, d, minus="-"):
    """Paired differences: $+0.818$, $-0.02$; minus="{-}" gives ${-}0.052$."""
    s = f"{x:+.{d}f}"
    return "$" + (minus + s[1:] if s.startswith("-") else s) + "$"


def shown(cell):
    """The value a formatted cell prints, for ranking."""
    return float(cell.replace("$", "").replace("{-}", "-"))


def ranks(cells, lower):
    """{row: 'best' | 'second'}: every row printing the best value, then every row printing the next one."""
    vals = sorted({shown(c) for c in cells if c != "n/m"}, reverse=not lower)
    out = {}
    for i, c in enumerate(cells):
        if c != "n/m" and shown(c) in vals[:2]:
            out[i] = "best" if shown(c) == vals[0] else "second"
    return out


def word(n, capital=False):
    w = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen"][n]
    return w.capitalize() if capital else w


def pend(text, pending):
    return rf"\pend{{{text}}}" if pending and text else text


def tabular(setlength, colspec, head, body, indent):
    lines = [setlength, rf"\begin{{tabular}}{{{colspec}}}"] + [indent + x for x in head + body + [r"\bottomrule"]]
    return "\n".join([HEADER] + lines + [r"\end{tabular}%"]) + "\n"


def sig(v, lower):
    """Significance colour of a (delta, lo, hi) of the row's method minus its reference: 'up' if the 95% interval
    excludes zero in the row's favour, 'dn' if against it, '' if it includes zero, None if no interval exists."""
    if v is None:
        return None
    _, lo, hi = v
    if lo is None or hi is None:
        return None
    if lower:
        return "up" if hi < 0 else ("dn" if lo > 0 else "")
    return "up" if lo > 0 else ("dn" if hi < 0 else "")


def paint(cell, s):
    """Wrap a cell in main.tex's \\up or \\dn (colour only; bold and underline keep their ranking meaning)."""
    return rf"\{s}{{{cell}}}" if s else cell


def neg(v):
    """(delta, lo, hi) of a minus b -> b minus a."""
    return None if v is None else (-v[0], -v[2], -v[1])


# ---------------------------------------------------------------- progressive SCA's pairs -------------------------
def prog_pairs(m):
    """({metric: (delta, lo, hi)}, n, source): progressive SCA minus method m, per-image metrics, 2000 draws."""
    rel = f"{PROG_PAIRS}/{PROG}_vs_{m}.csv"
    dd, n = V.paired_csv(rel, PROG, m)
    check(n == 195, f"{rel}: {n} prompts, not 195")
    return dd, n, f"results/{rel}"


def prog_setlevel(m):
    """({metric: (delta, lo, hi)} for fid, kid, patch_fid, patch_kid of progressive SCA minus m, source) or
    (None, reason) while no corrected interval with 168 reference identities exists for the pair."""
    out = {}
    if (RES / PROG_CFID_ALL).exists():
        for r in rows(PROG_CFID_ALL):
            if r["method"] == m and r["reference_method"] == PROG and r["metric"] in data195.DIST:
                assert r["bootstrap_valid"] == "1000" and r["n_reference_prompts"] == "168", r
                out[r["metric"]] = neg((float(r["value"]), float(r["ci_low"]), float(r["ci_high"])))
        if out:
            return out, f"results/{PROG_CFID_ALL} (method {m}, reference {PROG}, negated)"
    if m in PROG_CFID:
        for r in rows(PROG_CFID[m]):
            if r["method"] == PROG and r["reference_method"] == m and r["metric"] in data195.DIST:
                assert r["bootstrap_valid"] == "1000" and r["n_reference_prompts"] == "168", r
                out[r["metric"]] = (float(r["value"]), float(r["ci_low"]), float(r["ci_high"]))
        return out, f"results/{PROG_CFID[m]} (method {PROG}, reference {m})"
    return None, f"no corrected interval for {PROG} against {m} yet ({PROG_CFID_ALL} pending)"


# ---------------------------------------------------------------- who is shown -----------------------------------
def sca_shown(run_published):
    """(run, method, pending) of the single-pass SCA row: the final operator once a4k195_main_sca_v2 lands."""
    return (MAIN_V2, "sca_v2", False) if V.landed(MAIN_V2) else (run_published, REF, True)


def prog_shown():
    return (PROG_V2, PROG, False) if V.landed(PROG_V2) else (PANEL_RUN, "sca_prog", True)


# ---------------------------------------------------------------- Table 1 and Table panel46 (4096^2, 195) --------
# (column key, scale, decimals, lower is better); "clean" is the batch-1 latency, "seconds" the campaign mean
COLS = [("clean", 1, 1, True), ("seconds", 1, 1, True), ("ir", 1, 2, False), ("hpsv21", 100, 2, False),
        ("pickscore", 1, 2, False), ("clip", 1, 2, False), ("musiq_crop1024", 1, 1, False),
        ("fid", 1, 1, True), ("patch_fid", 1, 1, True)]
QUALITY = ("ir", "hpsv21", "pickscore", "clip", "musiq_crop1024", "fid", "patch_fid")
PER_IMAGE_COLS = ("ir", "hpsv21", "pickscore", "clip", "musiq_crop1024")
DIST_COLS = ("fid", "patch_fid")
SCA_LABEL = r"Single-pass \sca{} (ours)"
PROG_LABEL = r"\textbf{Progressive \sca{} (ours)}"
PANEL_SCA_LABEL = r"Single-pass \sca{}"
# (group heading, [(kind, run, method, label, steps)]); "sca" and "prog" rows show the final operator once its run
# lands (run and method here are the published placeholders); "ctl" rows exist only for the published operator.
TAB_MAIN = [
    ("Single pass, shared sampler (28 steps, guidance 4.5)",
     [("pub", MAIN_RUN, "full", "Full attention", "28"), ("pub", MAIN_RUN, "dype", "DyPE", "28"),
      ("pub", MAIN_RUN, "sega_official", "SEGA", "28"), ("pub", MAIN_RUN, "area", "AReA", "28"),
      ("sca", MAIN_RUN, REF, SCA_LABEL, "28")]),
    ("Multi-stage, own recipe and budget",
     [("pub", MAIN_RUN, "ultraimage", "UltraImage", "30+30"),
      ("pub", PANEL_RUN, "hrdit_dev_author", "HRDiT", "30+17+10"),
      ("prog", PANEL_RUN, "sca_prog", PROG_LABEL, "30+17+10")]),
]
TAB_PANEL = [
    (r"Single pass, shared sampler (28 steps, guidance 4.5; other rows in Table~\ref{tab:main})",
     [("pub", PANEL_RUN, "sigma_dev_matched", "SigMa", "28"), ("sca", PANEL_RUN, REF, PANEL_SCA_LABEL, "28")]),
    (r"\sca{} without the schedule, other budgets",
     [("ctl", PANEL_RUN, "sca_s56", r"\quad 56 steps", "56"), ("ctl", PANEL_RUN, "sca_g7", r"\quad guidance 7", "28"),
      ("ctl", PANEL_RUN, "sca_s42", r"\quad 42 steps", "42")]),
]
HEAD11 = [r"\toprule",
          r" & & \multicolumn{2}{c}{Time (s)} & & & & & & & \\",
          r"\cmidrule(lr){3-4}",
          r"Method & Steps & clean$\downarrow$ & camp. & IR$\uparrow$ & HPS$\uparrow$ & Pick$\uparrow$ & CLIP$\uparrow$ & "
          r"M$_\text{c}\!\uparrow$ & FID$\downarrow$ & FID$_\text{p}^{\dagger}\!\downarrow$ \\"]
NCOL11 = 11


def clean_seconds():
    """{method: median clean batch-1 seconds per image at 4096^2 on one GH200}.

    results/latency_curve_summary.csv, median_seconds (the audited sweep: full attention, AReA, SCA; three timed
    prompts after a warm-up image), checked against its per-image rows in results/latency_curve.csv. HRDiT, which
    the sweep lacks: median of its hrdit_dev_author rows in results/latency_curve_batch1.csv, which has no HRDiT
    warm-up row, so its cold first run is included (as the Table 1 caption says). The final operator (sca_v2) and
    published SCA re-timed beside it: results/latency_curve_v2/summary.csv (collect_latency_v2.py)."""
    per = {}
    for r in rows("latency_curve.csv"):
        if r["res"] == RES4K:
            check(r["warmup_verified"] == "True", f"latency_curve.csv: unverified warm-up {r}")
            per.setdefault(r["method"], []).append(float(r["seconds"]))
    out = {}
    for r in rows("latency_curve_summary.csv"):
        if r["res"] == RES4K:
            m, med = r["method"], float(r["median_seconds"])
            assert int(r["n"]) == len(per[m]) == 3 and abs(statistics.median(per[m]) - med) < 1e-9, (m, per[m])
            out[m] = med
    hrdit = [float(r["seconds"]) for r in rows("latency_curve_batch1.csv")
             if r["res"] == RES4K and r["method"] == "hrdit_dev_author"]
    assert len(hrdit) == 3, hrdit
    out["hrdit_dev_author"] = statistics.median(hrdit)
    # Sweep 3 (results/prog_latency/latency_summary.csv): progressive SCA, HRDiT and AReA timed back to back on one
    # idle GH200, three timed prompts after an excluded warm-up. It replaces HRDiT's cold-start median above and gives
    # the matched AReA time for progressive SCA's speed ratio (AReA agrees with sweep 1 to 0.1 s).
    if (RES / PROG_LAT).exists():
        # median_s is stored at two decimals (49.15 for a raw median of 49.153 s); round half up to the printed
        # one decimal here, since binary floats would print 49.15 as 49.1
        s3 = {r["method"]: float(Decimal(r["median_s"]).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
              for r in rows(PROG_LAT) if r["res"] == RES4K}
        for r in rows(PROG_LAT):
            assert r["n_timed"] == "3", r
        out[PROG], out["hrdit_dev_author"], out["area_s3"] = s3[PROG], s3["hrdit_dev_author"], s3["area"]
        check(abs(s3["area"] - out["area"]) < 0.5, f"AReA clean time: sweep 3 {s3['area']} vs sweep 1 {out['area']}")
    if V.landed(V.LATENCY):
        lat = V.latency_v2()
        out["sca_v2"] = lat[(RES4K, "sca_v2")][0]
        if (RES4K, "sca_nb") in lat:
            out["sca_nb_v2"] = lat[(RES4K, "sca_nb")][0]
            drift = out["sca_nb_v2"] / out[REF] - 1
            check(abs(drift) <= 0.05, f"latency_curve_v2: published SCA re-timed at {out['sca_nb_v2']:.1f} s against "
                                      f"{out[REF]:.1f} s ({100 * drift:+.1f}%); the GPUs differ, so sca_v2 cannot be "
                                      "compared with the published sweep directly")
    return out


def check_runs():
    for run in (MAIN_RUN, PANEL_RUN):
        check(data195.n_prompts(run, RES) == {195}, f"results/{run}: prompt counts {data195.n_prompts(run, RES)}, main.tex "
                                               r"says \nmain = \npanel = 195")
        _, _, ref, corrected_ci, _ = data(run)
        assert ref == REF and corrected_ci, f"results/{run}: reference {ref}, corrected_fidp_ci {corrected_ci}"
    for run in (MAIN_V2, PROG_V2):
        if V.landed(run):
            check(data195.n_prompts(run, RES) == {195}, f"results/{run}: prompt counts {data195.n_prompts(run, RES)}")
    if V.landed(MAIN_V2):
        # the final operator's run re-evaluates AReA and published SCA: their numbers must match the published runs
        S2, Sp = data(MAIN_V2)[0], data(MAIN_RUN)[0]
        for m in ("area", REF):
            for k in QUALITY:
                a, b = S2[(m, k)], Sp[(m, k)]
                check(abs(a - b) <= 1e-3 * max(1.0, abs(b)), f"results/{MAIN_V2}: {m} {k} = {a:.6g}, results/"
                                                              f"{MAIN_RUN} has {b:.6g}; the evaluations differ")
    if V.landed(PROG_V2):
        # HRDiT is scored in both the panel run and the progressive run: the printed values must agree
        S2, Sp = data(PROG_V2)[0], data(PANEL_RUN)[0]
        for k in QUALITY:
            a, b = S2[("hrdit_dev_author", k)], Sp[("hrdit_dev_author", k)]
            check(abs(a - b) <= 2e-3 * max(1.0, abs(b)), f"HRDiT {k}: {a:.6g} in results/{PROG_V2}, {b:.6g} in "
                                                          f"results/{PANEL_RUN}")


def table_rows(spec, clean):
    """Flat rows (kind, run, method, label, steps, pending, cells, cell_pending) of Table 1 or panel46, unmarked."""
    out = []
    for _, group in spec:
        for kind, run, m, label, steps in group:
            pending = False
            if kind == "sca":
                run, m, pending = sca_shown(run)
            elif kind == "prog":
                run, m, pending = prog_shown()
            S = data(run)[0]
            cells, cp = [], []
            for k, sc, d, _ in COLS:
                if k == "clean":
                    if kind == "sca":
                        lat = V.landed(V.LATENCY)
                        cells.append(num(clean["sca_v2" if lat else REF], d))
                        cp.append(not lat)
                    elif kind == "prog":
                        cells.append(num(clean[PROG], d) if PROG in clean else "n/m")
                        cp.append(False)
                    else:
                        cells.append(num(clean[m], d) if (kind == "pub" and m in clean) else "n/m")
                        cp.append(False)
                else:
                    cells.append(num(sc * S[(m, k)], d))
                    cp.append(pending)
            out.append((kind, run, m, label, steps, pending, cells, cp))
    return out


def table1_marks(row):
    """{column: colour} of a Table 1 row against progressive SCA (row minus progressive SCA)."""
    m = row[2]
    pi, _, _ = prog_pairs(m)
    sl, src = prog_setlevel(m)
    out = {k: sig(neg(pi[k]), lower=False) for k in PER_IMAGE_COLS}
    for k in DIST_COLS:
        out[k] = sig(neg(sl[k]), lower=True) if sl else None
        if out[k] is None:
            NO_CI.setdefault("tab_main", []).append(f"{m}:{k}")
    return out


def table2_marks(row, sca_row):
    """{column: (colour, grey)} for a panel46 row other than SCA: against single-pass SCA at 28 steps (the shown
    operator), or, for the budget rows, against the published operator. grey: the colour will be recomputed when
    SCA's run lands."""
    kind, run, m = row[0], row[1], row[2]
    ref_run, ref_m, ref_pending = (PANEL_RUN, REF, False) if kind == "ctl" else (sca_row[1], sca_row[2], sca_row[5])
    published_pair = run == PANEL_RUN and (ref_run, ref_m) == (PANEL_RUN, REF)
    if published_pair:
        P = data195.pairs(PANEL_RUN, m, REF, RES)
    else:
        P, n = V.derived_pairs(run, m, ref_run, ref_m)
        check(n == 195, f"Table panel46 marks {m} vs {ref_m}: {n} paired prompts")
    out = {}
    for k in PER_IMAGE_COLS:
        out[k] = (sig(P[k], lower=False), ref_pending)
    for k in DIST_COLS:
        v = data(PANEL_RUN)[1][(m, k)] if published_pair else (None, None, None)
        assert v[1] is not None or not published_pair, (m, k)
        out[k] = (sig(v, lower=True), ref_pending)
        if out[k][0] is None:
            NO_CI.setdefault("tab_panel46", []).append(f"{m}:{k}")
    return out


def rank_groups(name, spec, cells, underline):
    starts, k0 = [], 0
    for _, group in spec:
        starts.append((k0, k0 + len(group)))
        k0 += len(group)
    for a, b in starts:
        for j, (key, _, _, lower) in enumerate(COLS):
            if name == "tab_main" and key == "seconds":  # campaign time is not ranked in Table 1
                continue
            for i, r in ranks([c[j] for c in cells[a:b]], lower).items():
                if r == "best":
                    cells[a + i][j] = rf"\textbf{{{cells[a + i][j]}}}"
                elif underline:
                    cells[a + i][j] = rf"\underline{{{cells[a + i][j]}}}"
    return starts


def table_main(clean):
    flat = table_rows(TAB_MAIN, clean)
    cells = [list(r[6]) for r in flat]
    rank_groups("tab_main", TAB_MAIN, cells, underline=True)
    for i, row in enumerate(flat):
        marks = {} if row[0] == "prog" else table1_marks(row)
        for j, (k, *_) in enumerate(COLS):
            painted = cells[i][j] if row[7][j] else paint(cells[i][j], marks.get(k) or "")
            cells[i][j] = pend(painted, row[7][j])
        if marks:
            NOTES.append(f"Table 1 {row[2]} against progressive SCA: " +
                         ", ".join(f"{k} {marks[k] if marks.get(k) is not None else 'no interval'}"
                                   for k in QUALITY))
    PENDING["tab_main"] = any(any(r[7]) for r in flat)
    body, i = [], 0
    for g, (heading, group) in enumerate(TAB_MAIN):
        body += ([r"\midrule"] if g else []) + [rf"\multicolumn{{{NCOL11}}}{{@{{}}l}}{{\emph{{{heading}}}}} \\"]
        for _ in group:
            shade = SHADE if flat[i][0] in ("sca", "prog") else ""
            body.append(shade + " & ".join([flat[i][3], flat[i][4]] + cells[i]) + r" \\")
            i += 1
    # caption claims: single-pass SCA prints the best value of every ranked column among single-pass rows
    single = flat[:len(TAB_MAIN[0][1])]
    i = [r[0] for r in single].index("sca")
    lost = [COLS[j][0] for j in range(len(COLS)) if COLS[j][0] != "seconds" and
            ranks([r[6][j] for r in single], COLS[j][3]).get(i) != "best"]
    check(not lost, f"Table 1 caption: single-pass SCA is not best among single-pass rows on {lost}")
    return tabular(r"\setlength{\tabcolsep}{3.1pt}", r"@{}lcrr*{7}{r}@{}", HEAD11 + [r"\midrule"], body, "  ")


# Main-text Table 1 (v6, 2026-11-10): full text width (\scaFitTab of tabstyle.tex spreads the spare width over the
# column gaps, so shaded rows stay seamless), two-tier heads with direction arrows, best bold and second underlined
# within each group, both of our rows shaded. Columns: clean seconds with the ratio to progressive SCA's time of the
# same kind (a method with no clean time prints its campaign mean, marked \camp, and the ratio of campaign means),
# the three preference scores, MUSIQ on crops, FID and patch FID, and the ten-metric record: of the ten per-image
# metrics, on how many progressive SCA is significantly better / worse than the row (paired bootstrap, 2000 draws,
# Holm-adjusted over the ten; gen_holm.py). No significance colour: the record column and Figure gains carry it.
# The full table (steps, campaign time, CLIP) stays in tab_main.tex; tab:all_metrics expands it in the appendix.
COMPACT = ("ir", "hpsv21", "pickscore", "musiq_crop1024", "fid", "patch_fid")
HEAD_COMPACT = [r"\toprule",
                r" & \multicolumn{2}{c}{Time per image} & \multicolumn{3}{c}{Preference $\uparrow$} & "
                r"\multicolumn{1}{c}{Detail $\uparrow$} & \multicolumn{2}{c}{Set level $\downarrow$} & "
                r"\multicolumn{1}{c}{Ten metrics} \\",
                r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-7}\cmidrule(lr){8-9}\cmidrule(l){10-10}",
                r"Method & s $\downarrow$ & vs.\ ours & IR & HPSv2.1 & Pick & M$_\text{c}$ & FID & "
                r"FID$_\text{p}^{\dagger}$ & $+$\,/\,$-$ \\"]
GROUPS_COMPACT = ("Single pass, AReA's sampler (28 steps)", "Multi-stage, own recipes (UltraImage 30+30 steps; "
                  "HRDiT and ours 30+17+10)")
HOLM_KEY = {"area": "prog_area", "hrdit_dev_author": "prog_hrdit", "ultraimage": "prog_ultra", "dype": "prog_dype",
            "sega_official": "prog_sega", "full": "prog_full", "sca_v2": "prog_sca"}
_HOLM: dict = {}


def holm():
    """{pair key: gen_holm.pair result}, computed once (reproduces the archived paired CSVs; about 5 s)."""
    if not _HOLM:
        import gen_holm  # noqa: E402  (per-image files on scratch; read only)
        _HOLM.update({r["key"]: r for r in gen_holm.run()})
    return _HOLM


def half_up(x, d=1):
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def table_main_compact(clean):
    flat = table_rows(TAB_MAIN, clean)
    j_of = {c[0]: j for j, c in enumerate(COLS)}
    prog = next(r for r in flat if r[0] == "prog")
    S_prog = data(prog[1])[0]
    t_clean_prog, t_camp_prog = half_up(clean[PROG]), half_up(S_prog[(prog[2], "seconds")])
    speed = {}
    cells, kinds = [], []
    for kind, run, m, label, steps, pending, raw, cp in flat:
        timed = raw[j_of["clean"]] != "n/m"
        t = shown(raw[j_of["clean"]]) if timed else shown(raw[j_of["seconds"]])
        ratio = t / (t_clean_prog if timed else t_camp_prog)
        speed[m] = (f"{ratio:.1f}", timed)
        time_cell = raw[j_of["clean"]] if timed else raw[j_of["seconds"]] + r"\camp"
        ratio_cell = rf"{ratio:.1f}$\times$" + ("" if timed else r"\camp")
        row = [time_cell, ratio_cell] + [raw[j_of[k]] for k in COMPACT]
        if kind == "prog":
            row.append(r"\emph{ref.}")
        else:
            h = holm()[HOLM_KEY[m]]
            check(h["n"] == 195, f"Table 1 record {m}: {h['n']} prompts")
            row.append(rf"{h['up']}\,/\,{h['dn']}")
        cells.append(row)
        kinds.append((kind, timed, cp))
    # rank within each group: clean times among the rows timed clean (campaign means are never ranked), the metrics
    # among all rows; the ratio and record columns are not ranked
    k0 = 0
    for _, group in TAB_MAIN:
        a, b = k0, k0 + len(group)
        k0 = b
        cols = [(0, True, [i for i in range(a, b) if kinds[i][1]])] + \
               [(2 + jj, COLS[j_of[k]][3], list(range(a, b))) for jj, k in enumerate(COMPACT)]
        for jj, lower, members in cols:
            for pos, r in ranks([cells[i][jj] for i in members], lower).items():
                i = members[pos]
                cells[i][jj] = (rf"\textbf{{{cells[i][jj]}}}" if r == "best" else rf"\underline{{{cells[i][jj]}}}")
    for i, row in enumerate(flat):  # grey values of runs still pending (none since the final operator landed)
        for jj, k in enumerate(COMPACT):
            cells[i][2 + jj] = pend(cells[i][2 + jj], row[7][j_of[k]])
    body, i = [], 0
    ncol = 10
    for g, (_, group) in enumerate(TAB_MAIN):
        body += ([r"\midrule"] if g else []) + [rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{{GROUPS_COMPACT[g]}}}}} \\"]
        for _ in group:
            shade = "\\scaRowOurs " if flat[i][0] in ("sca", "prog") else ""
            body.append(shade + " & ".join([flat[i][3]] + cells[i]) + r" \\")
            i += 1
    # the printed ratios are the prose's ratios (headline.tex and fig_speed.py's speed_numbers.tex)
    want = {"area": r"\hlSpeedProgArea", "hrdit_dev_author": r"\hlSpeedProgHrdit", "full": r"\hlSpeedProgFull",
            "dype": r"\hlSpeedProgDypeCamp", "sega_official": r"\hlSpeedProgSegaCamp",
            "ultraimage": r"\hlSpeedProgUltraCamp"}
    sn = (OUT.parent / "speed_numbers.tex").read_text()
    for m, macro in want.items():
        hit = re.search(r"\\newcommand\{\\" + macro[1:] + r"\}\{([^}]*)\}", sn)
        if hit:
            check(hit.group(1) == speed[m][0], f"Table 1 ratio {m} {speed[m][0]} != {macro} {hit.group(1)}")
    SPEED_T1.update(speed)
    check(speed["area"][0] == f"{clean['area_s3'] / clean[PROG]:.1f}", "Table 1 AReA ratio != \\hlSpeedProgArea")
    check(speed["hrdit_dev_author"][0] == f"{clean['hrdit_dev_author'] / clean[PROG]:.1f}",
          "Table 1 HRDiT ratio != \\hlSpeedProgHrdit")
    # caption claims: single-pass SCA prints the best value of the single-pass group on every metric column; the
    # record of progressive SCA is all ten against every single-pass baseline
    single = flat[:len(TAB_MAIN[0][1])]
    s = [r[0] for r in single].index("sca")
    lost = [k for k in COMPACT if ranks([r[6][j_of[k]] for r in single], COLS[j_of[k]][3]).get(s) != "best"]
    check(not lost, f"Table 1: single-pass SCA is not best among single-pass rows on {lost}")
    for m in ("area", "dype", "sega_official", "full"):
        h = holm()[HOLM_KEY[m]]
        check((h["up"], h["dn"]) == (10, 0), f"Table 1 record: progressive vs {m} is {h['up']}/{h['dn']}")
    colspec = r"l*{8}{r}c"
    lines = [HEADER, TABSTYLE_INPUT, rf"\scaFitTab{{{colspec}}}{{10}}{{%"]
    lines += ["  " + x for x in HEAD_COMPACT + [r"\midrule"] + body + [r"\bottomrule"]]
    return "\n".join(lines + ["}%"]) + "\n"


def table_panel(clean):
    """Appendix Table tab:panel46 (v6 style): SigMa and three budgets of SCA without the schedule beside single-pass
    SCA. One time column, the campaign mean on shared nodes, which every row has (single-pass SCA's clean time is in
    Table 1); colours as before (against single-pass SCA at 28 steps; the budget rows against SCA without the
    schedule at 28 steps); bold best and underline second within each group."""
    flat = table_rows(TAB_PANEL, clean)
    j_of = {c[0]: j for j, c in enumerate(COLS)}
    keys = ("seconds", "ir", "hpsv21", "pickscore", "clip", "musiq_crop1024", "fid", "patch_fid")
    cells = [[r[6][j_of[k]] for k in keys] for r in flat]
    k0 = 0
    for _, group in TAB_PANEL:
        a_, b_ = k0, k0 + len(group)
        k0 = b_
        for jj, k in enumerate(keys):
            for pos, rk in ranks([cells[i][jj] for i in range(a_, b_)], COLS[j_of[k]][3]).items():
                i = a_ + pos
                if rk == "best":
                    cells[i][jj] = rf"\textbf{{{cells[i][jj]}}}"
                elif b_ - a_ >= 3:  # a second place means something only among three or more rows
                    cells[i][jj] = rf"\underline{{{cells[i][jj]}}}"
    sca_row = next(r for r in flat if r[0] == "sca")
    for i, row in enumerate(flat):
        marks = table2_marks(row, sca_row) if row[0] != "sca" else {}
        for jj, k in enumerate(keys):
            colour, grey = marks.get(k, ("", False))
            painted = cells[i][jj] if (grey or row[7][j_of[k]]) else paint(cells[i][jj], colour or "")
            cells[i][jj] = pend(painted, row[7][j_of[k]])
    PENDING["tab_panel46"] = any(any(r[7]) for r in flat) or any(r"\pend" in c for c in sum(cells, []))
    heads = [r"Single pass, AReA's sampler (28 steps, guidance 4.5)", r"\sca{} without the schedule, other budgets"]
    body, i = [], 0
    for g, (_, group) in enumerate(TAB_PANEL):
        body += ([r"\midrule"] if g else []) + [rf"\multicolumn{{10}}{{l}}{{\emph{{{heads[g]}}}}} \\"]
        for _ in group:
            shade = r"\scaRowOurs " if flat[i][0] == "sca" else ""
            label = flat[i][3].replace(r"\quad ", "")
            body.append(shade + " & ".join([label, flat[i][4]] + cells[i]) + r" \\")
            i += 1
    head = [r"\toprule",
            r" & \multicolumn{2}{c}{Cost} & \multicolumn{4}{c}{Preference $\uparrow$} & "
            r"\multicolumn{1}{c}{Detail $\uparrow$} & \multicolumn{2}{c}{Set level $\downarrow$} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-8}\cmidrule(lr){9-10}",
            r"Method & Steps & s$\downarrow$ & IR & HPSv2.1 & Pick & CLIP & M$_\text{c}$ & FID & "
            r"FID$_\text{p}^{\dagger}$ \\", r"\midrule"]
    lines = [HEADER, TABSTYLE_INPUT, r"\scaFitTab{lc*{8}{r}}{10}{%"]
    lines += ["  " + x for x in head + body + [r"\bottomrule"]]
    return "\n".join(lines + ["}%"]) + "\n"


# ---------------------------------------------------------------- Table 2 (ablations, 46 prompts) ----------------
ABL_COLS = [("ir", 1, 2), ("hpsv21", 100, 2), ("pickscore", 1, 2), ("musiq_crop1024", 1, 1)]
# Schedule block: paired change against single-pass SCA (dup_i_shrink) on the same 46 prompts; its variants of
# the queued 46-prompt runs appear once they land (no placeholder rows).
VARIANT_LABELS = {"dup_i_160": "first window 160", "dup_i_bridge32": "2,048 bridge tokens",
                  "dup_i_ext75": "five-step schedule"}
# Base-operator block: (label, method in results/ablnb_<method>_deltas.csv, k1_1024.json scales/4/rel_err key),
# paired change against the published operator (SCA without the schedule).
ABL_ROWS = [("no opening (v0)", "abl_v0", "per_head"), ("local only", "abl_localonly_nb", "loc"),
            ("global only", "abl_globalonly", "glob"), (r"shared $\pi(\sigma)$", "abl_uniform_nb", "uniform"),
            (r"per-layer $\pi_\ell(\sigma)$", "abl_layer_nb", "per_layer"), ("mean pooling", "abl_mean_nb", None)]
ABL_HEAD_SCHED = "Schedule and bridge"
ABL_HEAD_BASE = r"Base operator, against \sca{} without the schedule"


def ablation_deltas(method):
    out = {}
    for r in rows(f"ablnb_{method}_deltas.csv"):
        assert (r["method"], r["reference"], r["n_prompts"]) == (method, REF, "46"), r
        out[r["metric"]] = (float(r["delta"]), float(r["ci_low"]), float(r["ci_high"]))
    return out


def delta_cells(dd):
    return [paint(sgn(sc * dd[k][0], d), sig(dd[k], lower=False) or "") for k, sc, d in ABL_COLS]


def table_ablation():
    # The Err. column (relative output error at scale 4, results/evidence/calib/k1_1024.json) lives in Table K.5;
    # the checks below keep the K.5 row and Section 3 tied to the archive.
    calib = json.load(open(RES / CALIB_JSON))
    assert calib["settings"]["pool"] == "center", calib["settings"]["pool"]
    err = calib["scales"]["4"]["rel_err"]
    for key, printed in (("per_head", "0.200"), ("loc", "0.261"), ("glob", "0.627"), ("uniform", "0.230"),
                         ("per_layer", "0.223")):
        check(f"{err[key]:.3f}" == printed, f"Table K.5 output-error row prints {printed} for {key}, the archive "
                                            f"gives {err[key]:.3f}")
    S46 = V.summary46(DEFAULT46)
    sca = [num(sc * S46[k], d) for k, sc, d in ABL_COLS]
    body = [SHADE + r"Single-pass \sca{} & " + " & ".join(sca) + r" \\", r"\midrule",
            rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{ABL_HEAD_SCHED}}}}} \\"]
    p = V.pair46(DEFAULT46)
    sched = [("without the schedule", REF, {k: (-v, -hi, -lo) for k, (v, lo, hi) in p.items()})]
    for label, method, eval_dir in [("bridge only", "dup_bridge", "dup46_dup_bridge")] + \
            [(VARIANT_LABELS[v], v, f"dup46_{v}") for v in V.VARIANTS46 if V.landed(v)]:
        dd, n = V.derived_pairs(eval_dir, method, f"dup46_{DEFAULT46}", DEFAULT46)
        check(n == 46, f"Table 2 '{label}': {n} paired prompts, \\nabl is 46")
        sched.append((label, method, dd))
    for label, method, dd in sched:
        body.append(" & ".join([label] + delta_cells(dd)) + r" \\")
        better = [k for k, _, _ in ABL_COLS if dd[k][1] > 0]
        worse = [k for k, _, _ in ABL_COLS if dd[k][2] < 0]
        NOTES.append(f"Table 2 schedule row '{label}' against single-pass SCA: significantly better on "
                     f"{better or 'none'}, worse on {worse or 'none'}; deltas " +
                     ", ".join(f"{k} {sc * dd[k][0]:+.3f}" for k, sc, _ in ABL_COLS))
    body.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{ABL_HEAD_BASE}}}}} \\")
    for label, method, key in ABL_ROWS:
        dd = ablation_deltas(method)
        for k, _, _ in ABL_COLS:
            check(dd[k][1] <= 0, f"Table 2: base-operator row '{label}' is significantly better on {k}")
        check(dd["hpsv21"][2] < 0 and dd["pickscore"][2] < 0,
              f"Table 2 caption: '{label}' does not significantly lower both HPSv2.1 and PickScore")
        body.append(" & ".join([label] + delta_cells(dd)) + r" \\")
    PENDING["tab_ablation"] = False  # rows of runs not yet landed are left out, never shown grey
    return tabular(r"\setlength{\tabcolsep}{2.4pt}", "@{}lrrrr@{}",
                   [r"\toprule", r"Variant & IR & HPS & Pick & M$_\text{c}$ \\", r"\midrule"], body, "  ")


# Compact main-text ablation (2026-11-10 rewrite): the measurement rows first (shared and per-layer weights against
# per-head weights), then the views; the schedule and bridge rows stay in the appendix (Table tab:app_abl).
# Progressive-schedule rows (results/prog195_ablation/, method agent) fold in only when that folder has SHA256SUMS:
# then tab_ablation_compact_prog.tex is written as well, and main.tex's switch \ifresprogabl selects it.
# Expected files: results/prog195_ablation/*.csv in the paired format of results/prog195_sca_v2/sca_v2_prog_vs_*.csv
# (method,reference,metric,delta,ci_low,ci_high,n_prompts,...), each row a variant minus progressive SCA.
PROG_ABL_DIR = RES / "prog195_ablation"
PROG_ABL_REFS = ("sca_v2_prog",)
# Method names of HANDOVER Section 40, experiment B (slurm/reach_anneal_tasks.py): prog_headmean (the table's mean over
# every layer and head per noise bin), prog_layermean (mean over each layer's heads), prog_win (pi = 1: window and bridge
# only). Checked in order; the first substring found in the method name gives the row label.
PROG_ABL_LABELS = (("headmean", r"shared $\pi(\sigma)$"), ("layermean", r"per-layer $\pi_\ell(\sigma)$"),
                   ("_win", "window only"), ("uniform", r"shared $\pi(\sigma)$"),
                   ("shared", r"shared $\pi(\sigma)$"), ("layer", r"per-layer $\pi_\ell(\sigma)$"))
ABL_COMPACT_GROUPS = [(r"Reach weights, against per-head $\pi_h$", ("abl_uniform_nb", "abl_layer_nb")),
                      ("Views", ("abl_v0", "abl_localonly_nb", "abl_globalonly", "abl_mean_nb"))]


def prog_ablation_rows():
    """[(label, dd, n)] of the progressive-schedule ablation, or None while results/prog195_ablation/ lacks
    SHA256SUMS. Every CSV in the folder must be in the paired format with progressive SCA as the reference."""
    if not (PROG_ABL_DIR / "SHA256SUMS").exists():
        return None
    out = {}
    for path in sorted(PROG_ABL_DIR.glob("*.csv")):
        table = list(csv.DictReader(open(path)))
        if not table or not {"method", "reference", "metric", "delta", "ci_low", "ci_high"} <= set(table[0]):
            NOTES.append(f"prog195_ablation: {path.name} is not a paired file; skipped")
            continue
        for r in table:
            if r["reference"] not in PROG_ABL_REFS:
                continue
            d = out.setdefault(r["method"], {"n": r.get("n_prompts", "")})
            d[r["metric"]] = (float(r["delta"]), float(r["ci_low"]), float(r["ci_high"]))
    rows_ = []
    for method, dd in sorted(out.items()):
        if not all(k in dd for k, _, _ in ABL_COLS):
            NOTES.append(f"prog195_ablation: {method} lacks one of {[k for k, _, _ in ABL_COLS]}; skipped")
            continue
        label = next((lab for key, lab in PROG_ABL_LABELS if key in method),
                     method.replace("_", r"\_"))
        rows_.append((label, dd, dd["n"]))
        NOTES.append(f"prog195_ablation row '{label}' ({method}, n {dd['n']}): " +
                     ", ".join(f"{k} {sc * dd[k][0]:+.3f}" for k, sc, _ in ABL_COLS))
    return rows_


def prog_ablation_setlevel():
    """{method: {metric: (delta, lo, hi)}} for FID, KID and their patch versions of each progressive-ablation variant
    minus progressive SCA. Intervals come from results/prog195_ablation/cfid_ci/*_paired_deltas.csv (CPU bootstrap,
    1000 draws, 168 reference identities); until those exist the point estimates of corrected_fidp/ are used with
    lo = hi = None."""
    out = {}

    def take(path):
        for r in csv.DictReader(open(path)):
            if r["reference_method"] not in PROG_ABL_REFS or r["metric"] not in LOWER:
                continue
            has_ci = bool(r["ci_low"] and r["ci_high"])
            if has_ci:
                assert r["bootstrap_valid"] == "1000" and r["n_reference_prompts"] == "168", r
            cur = out.setdefault(r["method"], {})
            if has_ci or r["metric"] not in cur:
                cur[r["metric"]] = (float(r["value"]), float(r["ci_low"]) if has_ci else None,
                                    float(r["ci_high"]) if has_ci else None)

    point = PROG_ABL_DIR / "corrected_fidp" / "paired_deltas.csv"
    if point.exists():
        take(point)
    for path in sorted((PROG_ABL_DIR / "cfid_ci").glob("*_paired_deltas.csv")):
        take(path)
    return out


def prog_ablation_methods():
    """[(method, label)] of the progressive-ablation variants, in the row order of the compact table."""
    seen = set()
    for path in sorted(PROG_ABL_DIR.glob("*.csv")):
        table = list(csv.DictReader(open(path)))
        if table and {"method", "reference", "metric", "delta"} <= set(table[0]):
            seen |= {r["method"] for r in table if r["reference"] in PROG_ABL_REFS}
    return [(m, next((lab for key, lab in PROG_ABL_LABELS if key in m), m.replace("_", r"\_"))) for m in sorted(seen)]


def prog_ablation_full():
    """[(method, label, {metric: (delta, lo, hi)})]: all ten per-image metrics plus the set-level ones."""
    setlevel = prog_ablation_setlevel()
    out = []
    for method, label in prog_ablation_methods():
        dd = {}
        for path in sorted(PROG_ABL_DIR.glob(f"{method}_vs_*.csv")):
            for r in csv.DictReader(open(path)):
                if r["method"] == method and r["reference"] in PROG_ABL_REFS:
                    check(r["n_prompts"] == "195", f"{path.name}: {r['n_prompts']} prompts, not 195")
                    dd[r["metric"]] = (float(r["delta"]), float(r["ci_low"]), float(r["ci_high"]))
        check(all(k in dd for k in PER_IMAGE10), f"prog195_ablation: {method} lacks a per-image metric")
        dd.update(setlevel.get(method, {}))
        out.append((method, label, dd))
    return out


def table_prog_abl_all(full):
    """Appendix table prog_abl_all.tex: every metric of the progressive-schedule ablation with its interval."""
    no_ci = any(dd[k][1] is None for _, _, dd in full for k in dd if k in LOWER)
    lost = {m: sum(dd[k][2] < 0 for k in PER_IMAGE10) for m, _, dd in full}
    won = {m: sum(dd[k][1] > 0 for k in PER_IMAGE10) for m, _, dd in full}
    NOTES.append(f"prog_abl_all: per-image metrics significantly lost {lost}, won {won}")
    NOTES.append("prog_abl_all: patch FID " + ", ".join(
        f"{m} {dd['patch_fid'][0]:+.1f} [{dd['patch_fid'][1]}, {dd['patch_fid'][2]}]" for m, _, dd in full
        if "patch_fid" in dd))
    rows_ = [(ABL6_PROG_LABEL.get(m, label), dd) for m, label, dd in full]
    check(all(dd["patch_fid"][2] < 0 for m, _, dd in full if m != "prog_win") and
          full[[m for m, _, _ in full].index("prog_win")][2]["patch_fid"][1] > 0,
          "prog_abl_all key result: patch FID directions")
    cap = (r"\caption{\textbf{Inside the progressive sampler, per-head weights win every per-image metric; coarser "
           r"weights have lower patch FID.} Expands the progressive rows of Table~\ref{tab:ablation}. FLUX.1-dev, "
           r"\nmain{} Aesthetic-4K prompts at $4096^2$, seed 0. Each cell is the variant minus progressive \sca{} with "
           r"per-head weights, with its 95\% paired interval below it in grey: 2000 prompt draws for the per-image "
           r"metrics, and for FID, KID, and their patch versions 1000 draws that also resample the 168 reference-image "
           r"clusters" + (r" (a point estimate alone where that interval is not yet computed)" if no_ci else "") +
           r". $^{\times}$: $\times100$; $^{\ast}$: $\times1000$. \up{Blue} / \dn{orange}: the interval excludes zero "
           r"in favour of / against the variant; for FID and KID lower is better. All four runs share the schedule, "
           r"the window, the bridge, and the seeds, and differ only in the weights $\pi$. Key result: shared and "
           r"per-layer weights lose all ten per-image metrics and lower patch FID; window-only attention raises it.}")
    return "\n".join([
        HEADER.replace("% Generated", "% Generated (app_tables/prog_abl_all.tex)"), TABSTYLE_INPUT,
        r"\begin{table}[!htbp]", cap, r"\label{tab:prog_abl_all}", r"\centering\scriptsize"]
        + paired_panels(rows_, r"Variant $-$ per-head") + [r"\end{table}"]) + "\n"


def prog_ablation_sentence(full):
    """Section 4.3's sentences (\\progAblResult): how many of the ten per-image metrics each variant loses against
    per-head weights, the ImageReward and MUSIQ-crop changes, and the patch FID change, which can read the other way.
    The wording is chosen from the intervals, so it cannot claim more than the cells of tab:prog_abl_all show."""
    def ci(v, d):
        return (rf"{sgn(v[0], d)} $[{v[1]:.{d}f},{v[2]:.{d}f}]$".replace("[-", "[{-}").replace(",-", ",{-}")
                if v[1] is not None else sgn(v[0], d))

    def joined(parts):
        return " and ".join(parts) if len(parts) < 3 else ", ".join(parts[:-1]) + ", and " + parts[-1]

    def count(n):
        return "all ten" if n == 10 else word(n)

    lost = {m: sum(dd[k][2] < 0 for k in PER_IMAGE10) for m, _, dd in full}
    won = {m: sum(dd[k][1] > 0 for k in PER_IMAGE10) for m, _, dd in full}
    check(all(lost[m] > won[m] for m in lost), f"prog195_ablation: a variant does not lose to per-head weights: {lost} {won}")
    coarse = [(m, lab, dd) for m, lab, dd in full if "window" not in lab]
    rest = [(m, lab, dd) for m, lab, dd in full if "window" in lab]
    parts = []
    if coarse:
        n = {lost[m] for m, _, _ in coarse}
        each = " each" if len(coarse) > 1 else ""
        head = (rf"{joined([lab for _, lab, _ in coarse])}{each} lose to per-head weights on {count(n.pop())} "
                "per-image metrics" if len(n) == 1 else
                joined([rf"{lab} loses to per-head weights on {count(lost[m])}" for m, lab, _ in coarse]) +
                " per-image metrics")
        parts.append(head + " (ImageReward " + " and ".join(ci(dd["ir"], 3) for _, _, dd in coarse) +
                     r", MUSIQ$_\text{c}$ " + " and ".join(sgn(dd["musiq_crop1024"][0], 1) for _, _, dd in coarse) + ")")
    for m, lab, dd in rest:
        gain = (rf" and gains {sgn(dd['musiq_crop1024'][0], 1)[1:-1].lstrip('+')} MUSIQ$_\text{{c}}$"
                if dd["musiq_crop1024"][1] > 0 else "")
        parts.append(rf"{lab} loses on {count(lost[m])}{gain}")
    sentence = ("Inside the progressive sampler the gaps are smaller but point the same way: " +
                (", and ".join(parts) if len(parts) == 2 else joined(parts)) + ".")
    pf = [(lab, dd["patch_fid"]) for _, lab, dd in full if "patch_fid" in dd]
    if pf:
        have_ci = all(v[1] is not None for _, v in pf)
        lower = [(lab, v) for lab, v in pf if have_ci and v[2] < 0]
        higher = [(lab, v) for lab, v in pf if have_ci and v[1] > 0]
        open_ = [(lab, v) for lab, v in pf if (lab, v) not in lower + higher]
        bits = []
        if lower:
            bits.append("is lower with " + joined([rf"{lab} ({ci(v, 1)})" for lab, v in lower]))
        if higher:
            bits.append("higher with " + joined([rf"{lab} ({ci(v, 1)})" for lab, v in higher]))
        if open_:
            bits.append(("changes by " if not have_ci else "unresolved with ") +
                        joined([rf"{lab} ({ci(v, 1)})" if have_ci else rf"{ci(v, 1)} with {lab}" for lab, v in open_]))
        sentence += (" Patch FID, a set-level metric, does not follow them: it " + joined(bits) +
                     r" (Tables~\ref{tab:ablation} and~\ref{tab:prog_abl_all})." if lower else
                     " Patch FID, a set-level metric, " + joined(bits) +
                     r" (Tables~\ref{tab:ablation} and~\ref{tab:prog_abl_all}).")
    return sentence.replace(r") (Tables", "; Tables")  # one bracket, not two in a row


def table_ablation_compact(prog_rows=None):
    S46 = V.summary46(DEFAULT46)
    sca = [num(sc * S46[k], d) for k, sc, d in ABL_COLS]
    body = [SHADE + r"Single-pass \sca{} & " + " & ".join(sca) + r" \\"]
    labels = {m: lab for lab, m, _ in ABL_ROWS}
    for heading, methods in ABL_COMPACT_GROUPS:
        body += [r"\midrule", rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{heading}}}}} \\"]
        for method in methods:
            body.append(" & ".join([labels[method]] + delta_cells(ablation_deltas(method))) + r" \\")
        if prog_rows and heading.startswith("Reach weights"):
            body.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{Progressive \sca{{}}, against per-head $\pi_h$}}}} \\")
            for label, dd, _ in prog_rows:
                body.append(" & ".join([label] + delta_cells(dd)) + r" \\")
    return tabular(r"\setlength{\tabcolsep}{2.4pt}", "@{}lrrrr@{}",
                   [r"\toprule", r"Variant & IR & HPS & Pick & M$_\text{c}$ \\", r"\midrule"], body, "  ")


# ---------------------------------------------------------------- Table 2, v6 (2026-11-10) -----------------------
# Full text width. Each group opens with its shaded per-head reference row in absolute values (v5.1 printed
# single-pass SCA's values above deltas that were taken against SCA without the schedule); the rows below are the
# variant minus that reference, coloured by the paper's convention (\dn: the 95% paired interval excludes zero
# against the variant, \up: in its favour). FID_p: corrected patch FID; the progressive rows carry the cluster
# bootstrap interval (1000 draws, 168 reference identities), the 46-prompt rows a point estimate only. The record
# column counts, over the ten per-image metrics, where per-head weights are significantly better / worse than the
# variant after Holm adjustment (gen_holm.py). Views pair with the first shaded row (the same 46-prompt reference).
ABL6_SINGLE = [("abl_uniform_nb", r"shared $\pi(\sigma)$"), ("abl_layer_nb", r"per-layer $\pi_\ell(\sigma)$")]
ABL6_VIEWS = [("abl_v0", "no opening"), ("abl_localonly_nb", "local only"), ("abl_globalonly", "global only"),
              ("abl_mean_nb", "mean pooling")]
ABL6_PROG_LABEL = {"prog_headmean": r"shared $\pi(\sigma)$", "prog_layermean": r"per-layer $\pi_\ell(\sigma)$",
                   "prog_win": r"window only ($\pi=1$)"}
HEAD_ABL6 = [r"\toprule",
             r" & \multicolumn{3}{c}{Preference $\uparrow$} & \multicolumn{1}{c}{Detail $\uparrow$} & "
             r"\multicolumn{1}{c}{Set level $\downarrow$} & \multicolumn{1}{c}{Ten metrics} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}",
             r"Variant & IR & HPSv2.1 & Pick & M$_\text{c}$ & FID$_\text{p}$ & $+$\,/\,$-$ \\"]
CORR_PFID46 = "evidence/distributions/checks/ablnb_base_rows_rows_patch_fid_0.csv"  # corrected, no interval


def abl46_reference():
    """Absolute values of the 46-prompt reference (SCA without the schedule, per-head weights) and the corrected patch
    FID of every 46-prompt ablation run."""
    S = {(r["method"], r["metric"]): float(r["value"]) for r in rows("ablnb_base_summary.csv") +
         rows("ablnb_extra_summary.csv")}
    pfid = {r["method"]: float(r["value"]) for r in rows(CORR_PFID46)
            if r["metric"] == "patch_fid" and r["reference_method"] == REF}
    for (m, k), v in S.items():
        if m == REF and k in ("ir", "pickscore"):
            check(int(float(next(r["n_prompts"] for r in rows("ablnb_extra_summary.csv") + rows("ablnb_base_summary.csv")
                                 if r["method"] == m and r["metric"] == k))) == 46, f"ablnb {k}: not 46 prompts")
    return S, pfid


def abl6_cells(dd, pfid_delta, record):
    """Delta cells of one variant row: the four per-image columns, patch FID, the record."""
    out = [paint(sgn(sc * dd[k][0], d), sig(dd[k], lower=False) or "") for k, sc, d in ABL_COLS]
    v = pfid_delta
    out.append(paint(sgn(v[0], 1), sig(v, lower=True) or "") if v[1] is not None else sgn(v[0], 1))
    out.append(record)
    return out


def abl6_record(key):
    h = holm()[key]  # variant minus per-head reference: the variant's losses are per-head weights' wins
    return rf"{h['dn']}\,/\,{h['up']}", (h["dn"], h["up"])


def table_ablation_v6(prog_full=None):
    S, pfid = abl46_reference()
    ref46 = [num(sc * S[(REF, k)], d) for k, sc, d in ABL_COLS] + [num(pfid[REF], 1), r"\emph{ref.}"]
    body = [r"\multicolumn{7}{l}{\emph{Reach weights, single pass without the schedule (\nabl{} prompts)}} \\",
            r"\scaRowOurs per-head $\pi_h$ (ours) & " + " & ".join(ref46) + r" \\"]
    rec = {}
    for m, label in ABL6_SINGLE:
        record, rec[m] = abl6_record(m)
        body.append(" & ".join([label] + abl6_cells(ablation_deltas(m), (pfid[m] - pfid[REF], None, None), record))
                    + r" \\")
    if prog_full:
        SG = data(PROG_V2)[0]
        refp = [num(sc * SG[(PROG, k)], d) for k, sc, d in ABL_COLS] + [num(SG[(PROG, "patch_fid")], 1),
                                                                         r"\emph{ref.}"]
        body += [r"\midrule", r"\multicolumn{7}{l}{\emph{Reach weights, progressive \sca{} (\nmain{} prompts)}} \\",
                 r"\scaRowOurs per-head $\pi_h$ (ours) & " + " & ".join(refp) + r" \\"]
        for m, _, dd in prog_full:
            record, rec[m] = abl6_record(m)
            body.append(" & ".join([ABL6_PROG_LABEL[m]] + abl6_cells(dd, dd["patch_fid"], record)) + r" \\")
    body += [r"\midrule", r"\multicolumn{7}{l}{\emph{Views, single pass without the schedule (\nabl{} prompts; against "
                          r"the first shaded row)}} \\"]
    for m, label in ABL6_VIEWS:
        record, rec[m] = abl6_record(m)
        body.append(" & ".join([label] + abl6_cells(ablation_deltas(m), (pfid[m] - pfid[REF], None, None), record))
                    + r" \\")
    # the claims the caption and Section 4 make from this table
    check(all(rec[m][1] == 0 for m in rec if m != "abl_v0" and m != "prog_win"),
          f"Table 2: a coarser rule or view wins a metric after Holm: {rec}")
    if prog_full:
        check(rec["prog_headmean"] == (10, 0) and rec["prog_layermean"] == (10, 0),
              f"Table 2: progressive per-head weights do not win all ten after Holm: {rec}")
    check(min(v[0] for v in rec.values()) >= 5, f"Table 2: a variant loses fewer than five metrics: {rec}")
    ABL6_RECORD.update(rec)
    lines = [HEADER, TABSTYLE_INPUT, r"\scaFitTab{l*{5}{r}c}{7}{%"]
    lines += ["  " + x for x in HEAD_ABL6 + [r"\midrule"] + body + [r"\bottomrule"]]
    return "\n".join(lines + ["}%"]) + "\n"


ABL6_RECORD: dict = {}


# ---------------------------------------------------------------- Table 3, v6 (2026-11-10) -----------------------
# 0.48 text width at 8 pt: one line per setting (n, the three preference differences coloured by the convention, the
# Holm-adjusted ten-metric record of the row's method against its reference). The intervals of every cell, and the
# other seven metrics, are in the appendix table gen_all (app_tables/gen_all.tex), written below.
GEN6 = [(r"Progressive \sca{} $-$ AReA", [("Seed 0", r"\nmain", "prog_area"), ("Seed 1", r"\nseed", "prog_area_s1"),
                                         ("Seed 2", r"\nseed", "prog_area_s2"),
                                         ("HRGen-1K", r"\nhrgen", "prog_area_hrgen")]),
        (r"Single-pass \sca{} $-$ AReA", [("Seed 0", r"\nmain", "sca_area"), ("Seed 1", r"\nseed", "sca_area_s1"),
                                         ("Seed 2", r"\nseed", "sca_area_s2"),
                                         ("HRGen-1K", r"\nhrgen", "sca_area_hrgen"),
                                         ("FLUX.1-Krea", r"\nkrea", "sca_area_krea")]),
        (r"Without the schedule $-$ full attention", [("FLUX.2-dev", None, "flux2")])]
HEAD_GEN6 = [r"\toprule",
             r" & & \multicolumn{3}{c}{Paired difference $\uparrow$} & \multicolumn{1}{c}{Holm} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-6}",
             r"Setting & $n$ & IR & HPS & Pick & $+$\,/\,$-$ \\"]


def table_generality_v6():
    H = holm()
    PENDING["tab_generality"] = False  # every setting has landed; no grey cells
    body = []
    for g, (heading, settings) in enumerate(GEN6):
        body += ([r"\midrule"] if g else []) + [rf"\multicolumn{{6}}{{l}}{{\emph{{{heading}}}}} \\"]
        for label, ncell, key in settings:
            h = H[key]
            check(ncell is None or N_MACROS[ncell] == h["n"], f"Table 3 '{label}': {h['n']} prompts, {ncell}")
            cells = [label, f"${ncell}$" if ncell else str(h["n"])]
            for k, sc, d in GEN_COLS:
                g_ = h["metrics"][k]
                cells.append(paint(sgn(sc * g_["delta"], d), sig((g_["delta"], g_["lo"], g_["hi"]), lower=False) or ""))
            cells.append(rf"{h['up']}\,/\,{h['dn']}")
            body.append(" & ".join(cells) + r" \\")
    # the caption's claims: every comparison against AReA is significantly better on all three columns except the
    # FLUX.1-Krea ImageReward, and no row is significantly worse on any of the ten metrics after Holm
    for heading, settings in GEN6:
        for label, _, key in settings:
            h = H[key]
            check(h["dn"] == 0, f"Table 3 '{label}' ({key}): significantly worse on {h['dn']} metrics after Holm")
            open_ = [k for k, _, _ in GEN_COLS if not h["metrics"][k]["lo"] > 0]
            check(open_ == ([] if key != "sca_area_krea" else ["ir"]), f"Table 3 '{label}': unresolved {open_}")
    lines = [HEADER, TABSTYLE_INPUT, r"\scaFitTab{lr*{3}{r}c}{6}{%"]
    lines += ["  " + x for x in HEAD_GEN6 + [r"\midrule"] + body + [r"\bottomrule"]]
    return "\n".join(lines + ["}%"]) + "\n"


# ---------------------------------------------------------------- Table 3 (generality) ---------------------------
# Every difference carries its 95% interval (printed under it). Progressive rows whose runs have not landed print
# "pending" in one line; they fill in when their paired CSVs appear under results/.
GEN_COLS = [("ir", 1, 2), ("hpsv21", 100, 2), ("pickscore", 1, 2)]
N_MACROS = {r"\nhrgen": 1000, r"\nseed": 195, r"\nabl": 46, r"\nkrea": 195, r"\nmain": 195}  # their values in main.tex
# Progressive SCA minus AReA: (label, n macro, candidate CSVs under results/ with method sca_v2_prog and reference
# area; the first that exists is used).
GEN_PROG = [("Seed 0", r"\nmain", [f"{PROG_PAIRS}/{PROG}_vs_area.csv"]),
            ("Seed 1", r"\nseed", ["prog195_s1/sca_v2_prog_vs_area.csv", "prog195_seed1/sca_v2_prog_vs_area.csv"]),
            ("Seed 2", r"\nseed", ["prog195_s2/sca_v2_prog_vs_area.csv", "prog195_seed2/sca_v2_prog_vs_area.csv"]),
            ("HRGen-1K", r"\nhrgen", ["prog_hrgen1k/sca_v2_prog_vs_area.csv", "hrgen1k_prog/sca_v2_prog_vs_area.csv"])]
# Single-pass SCA minus AReA (final operator).
GEN_AREA = [("HRGen-1K", r"\nhrgen", "hrgen1k_dev_sca_v2"), ("Seed 1", r"\nseed", "a4k195_seed1_sca_v2"),
            ("Seed 2", r"\nseed", "a4k195_seed2_sca_v2"), ("FLUX.1-Krea", r"\nkrea", KREA_V2)]
# FLUX.2-dev: the operator without the schedule against full attention.
GEN_FLUX2 = ("FLUX.2-dev", None, "e8_flux2_sca_vs_full.csv", "flux2_sca", "flux2_full")
KEY_AREA_ROWS = ("HRGen-1K", "Seed 1", "Seed 2")  # the caption's key result: gains over AReA hold on these
NCOL3 = 4


def gen_area_pair(run):
    """(deltas, n, source) of a single-pass 'minus AReA' row: the final operator's paired CSV."""
    assert V.landed(run), f"Table 3: {run} has not landed"
    rel = f"{run}/{V.RUNS[run].final}_vs_area.csv"
    dd, n = V.paired_csv(rel, V.RUNS[run].final, "area")
    return dd, n, rel


def table3_rows(label, ncell, dd, n):
    """Two lines per setting: the differences (coloured by the convention), then n and the intervals."""
    check(ncell is None or N_MACROS[ncell] == n, f"Table 3 '{label}': {n} prompts, {ncell} is "
                                                 f"{N_MACROS.get(ncell)} in main.tex")
    top, low = [label], [rf"\scriptsize $n={ncell or n}$"]
    for k, sc, d in GEN_COLS:
        v, lo, hi = dd[k]
        top.append(paint(sgn(sc * v, d), sig(dd[k], lower=False) or ""))
        low.append(rf"\scriptsize $[{sc * lo:.{d}f}, {sc * hi:.{d}f}]$".replace("[-", "[{-}").replace(", -", ", {-}"))
    return [" & ".join(top) + r" \\", " & ".join(low) + r" \\[0.5pt]"]


def table_generality():
    body = [rf"\multicolumn{{{NCOL3}}}{{@{{}}l}}{{\emph{{Progressive \sca{{}} minus AReA}}}} \\"]
    pending = []
    for label, ncell, candidates in GEN_PROG:
        rel = next((c for c in candidates if (RES / c).exists()), None)
        if rel is None:
            body.append(rf"{label} & \multicolumn{{3}}{{c}}{{\color{{pendgrey}}pending}} \\[1.5pt]")
            pending.append(label)
            continue
        dd, n = V.paired_csv(rel, PROG, "area")
        body += table3_rows(label, ncell, dd, n)
    body.append(rf"\multicolumn{{{NCOL3}}}{{@{{}}l}}{{\emph{{Single-pass \sca{{}} minus AReA}}}} \\")
    for label, ncell, run in GEN_AREA:
        dd, n, _ = gen_area_pair(run)
        if label in KEY_AREA_ROWS:
            check(all(dd[k][1] > 0 for k, _, _ in GEN_COLS),
                  f"Table 3 caption: single-pass '{label}' is not significantly better than AReA on every column")
        body += table3_rows(label, ncell, dd, n)
    body.append(rf"\multicolumn{{{NCOL3}}}{{@{{}}l}}{{\emph{{Without the schedule, minus full attention}}}} \\")
    label, ncell, rel, method, reference = GEN_FLUX2
    dd, n = V.paired_csv(rel, method, reference)
    body += table3_rows(label, ncell, dd, n)
    body[-1] = body[-1].replace(r" \\[0.5pt]", r" \\")
    PENDING["tab_generality"] = bool(pending)
    NOTES.append(f"Table 3 progressive rows pending: {pending or 'none'}")
    return tabular(r"\setlength{\tabcolsep}{2.0pt}", "@{}lrrr@{}",
                   [r"\toprule", r"Setting & $\Delta$IR & $\Delta$HPS & $\Delta$Pick \\", r"\midrule"], body, "  ")


# ---------------------------------------------------------------- appendix Table prog_all -----------------------
PROG_ALL_REFS = [("area", "AReA"), ("hrdit_dev_author", "HRDiT"), ("ultraimage", "UltraImage"), ("dype", "DyPE"),
                 ("sega_official", "SEGA"), ("full", "Full attn."), ("sca_v2", r"Single-pass")]
PROG_ALL_METRICS = [("ir", "ImageReward", 1, 3), ("aesthetic", "Aesthetic", 1, 2), ("clip", "CLIP score", 1, 2),
                    ("pickscore", "PickScore", 1, 2), ("hpsv2", r"HPSv2 ($\times100$)", 100, 2),
                    ("hpsv21", r"HPSv2.1 ($\times100$)", 100, 2), ("musiq", "MUSIQ", 1, 1),
                    ("musiq_crop1024", "MUSIQ, crops", 1, 1), ("clipiqa", r"CLIP-IQA ($\times100$)", 100, 1),
                    ("arniqa", r"ARNIQA ($\times100$)", 100, 1), ("fid", "FID", 1, 1), ("kid", r"KID ($\times$1000)", 1000, 2),
                    ("patch_fid", "Patch FID", 1, 1), ("patch_kid", r"Patch KID ($\times$1000)", 1000, 2)]


def bound(x, d, keep_sign):
    """An interval end at d decimals. With keep_sign (the interval excludes zero), an end that would print as zero
    gets more decimals (up to d + 3) so that the reader sees it is not zero, as gen_tables.py printed them
    (e.g. [0.0002, 0.062])."""
    s = f"{x:.{d}f}"
    for dd in range(d + 1, d + 4):
        if not keep_sign or float(s) != 0 or x == 0:
            break
        s = f"{x:.{dd}f}"
    return s


def interval(lo, hi, d):
    k = lo > 0 or hi < 0
    return rf"$[{bound(lo, d, k)}, {bound(hi, d, k)}]$".replace("[-", "[{-}").replace(", -", ", {-}")


# Appendix paired tables in the v6 style: comparisons as rows, metrics as columns, split into stacked panels by metric
# family so that every cell fits at 7 pt without \resizebox; each estimate carries its 95% interval on the line below
# in grey; colour by the paper's convention. Precision per metric as PROG_ALL_METRICS.
PANELS_V6 = [("(a) Preference", ["ir", "aesthetic", "clip", "pickscore", "hpsv2", "hpsv21"]),
             ("(b) Image quality", ["musiq", "musiq_crop1024", "clipiqa", "arniqa"]),
             ("(c) Set level", ["fid", "kid", "patch_fid", "patch_kid"])]
SHORT_V6 = {"ir": "IR", "aesthetic": "Aesthetic", "clip": "CLIP", "pickscore": "PickScore",
            "hpsv2": r"HPSv2$^{\times}$", "hpsv21": r"HPSv2.1$^{\times}$", "musiq": "MUSIQ",
            "musiq_crop1024": r"MUSIQ$_\text{c}$", "clipiqa": r"CLIP-IQA$^{\times}$", "arniqa": r"ARNIQA$^{\times}$",
            "fid": r"FID$\downarrow$", "kid": r"KID$^{\ast}\!\downarrow$", "patch_fid": r"FID$_\text{p}\!\downarrow$",
            "patch_kid": r"KID$_\text{p}^{\ast}\!\downarrow$"}


def paired_panels(rows_, label_head, shade=()):
    """rows_: [(label, {metric: (delta, lo, hi) or (delta, None, None)})]. Returns the LaTeX lines of the stacked
    panels. A missing metric prints "--"; a point estimate without interval prints no second line."""
    spec = {k: (sc, d) for k, _, sc, d in PROG_ALL_METRICS}
    out = []
    for p_i, (title, keys) in enumerate(PANELS_V6):
        body = []
        for label, dd in rows_:
            top, low = [label], [""]
            for k in keys:
                sc, d = spec[k]
                if k not in dd:
                    top.append("--")
                    low.append("")
                    continue
                v, lo, hi = dd[k]
                top.append(paint(sgn(sc * v, d, "{-}"), sig(dd[k], lower=k in LOWER) or ""))
                low.append(rf"{{\color{{scaCtx}}{interval(sc * lo, sc * hi, d)}}}" if lo is not None else "")
            pre = r"\scaRowOurs " if label in shade else ""
            body += [pre + " & ".join(top) + r" \\", (pre + " & ".join(low)) + r" \\[1pt]"]
        head = [r"\toprule", label_head + " & " + " & ".join(SHORT_V6[k] for k in keys) + r" \\", r"\midrule"]
        out += ([r"\par\medskip"] if p_i else []) + [rf"\textit{{{title}}}\par\smallskip",
                                                     rf"\scaFitTab{{l*{{{len(keys)}}}{{r}}}}{{{len(keys) + 1}}}{{%"]
        out += ["  " + x for x in head + body + [r"\bottomrule"]] + ["}"]
    return out



def table_prog_all():
    cols = []
    for m, _ in PROG_ALL_REFS:
        pi, n, _ = prog_pairs(m)
        sl, _ = prog_setlevel(m)
        cols.append((pi, sl))
    pending_refs = [name for (m, name), (_, sl) in zip(PROG_ALL_REFS, cols) if sl is None]
    metric_name = {"fid": "FID", "kid": "KID", "patch_fid": "patch FID", "patch_kid": "patch KID"}
    outside = [f"{name} {metric_name[key]}" for (m, name), (pi, sl) in zip(PROG_ALL_REFS, cols)
               for key in data195.DIST if sl and key in sl and not (sl[key][1] <= sl[key][0] <= sl[key][2])]
    rows_ = []
    for (m, name), (pi, sl) in zip(PROG_ALL_REFS, cols):
        dd = dict(pi)
        dd.update(sl or {})
        rows_.append((name if m != "sca_v2" else r"Single-pass \sca{}", dd))
    note = (" Set-level intervals against " + ", ".join(pending_refs) + " are pending." if pending_refs else "")
    # key result, checked against the cells
    wins = {m: sum(pi[k][1] > 0 for k in PER_IMAGE10) for (m, _), (pi, _) in zip(PROG_ALL_REFS, cols)}
    sl_of = {m: sl for (m, _), (_, sl) in zip(PROG_ALL_REFS, cols)}
    check(all(wins[m] == 10 for m in ("area", "dype", "sega_official", "full")) and wins["ultraimage"] == 9,
          f"prog_all key result: per-image wins {wins}")
    worse_pfid = [m for m in ("area", "hrdit_dev_author", "sca_v2") if sl_of[m] and sl_of[m]["patch_fid"][1] > 0]
    check(worse_pfid == ["area", "hrdit_dev_author", "sca_v2"], f"prog_all key result: patch FID worse vs {worse_pfid}")
    key = (r" Key result: progressive \sca{} is significantly better than AReA, DyPE, SEGA, and full attention on every "
           r"per-image metric and than UltraImage on nine; its differences from HRDiT on ImageReward, HPSv2.1, and "
           r"PickScore include zero, and its patch FID is higher than AReA's, HRDiT's, and single-pass \sca{}'s."
           if not pending_refs else "")
    if outside:  # a percentile interval of a biased statistic need not contain the point estimate
        note += (" The point estimate lies outside its percentile interval for " + ", ".join(outside) +
                 ", as it can for a biased statistic.")
    NOTES.append(f"prog_all: points outside their interval: {outside or 'none'}")
    cap = (r"\caption{\textbf{Progressive \sca{} beats every single-pass baseline on all ten per-image metrics and "
           r"pays in patch FID.} Expands the progressive row of Table~\ref{tab:main}. FLUX.1-dev at $4096^2$, \nmain{} "
           r"Aesthetic-4K prompts, seed 0. Each cell is progressive \sca{} minus the row's method, with its 95\% paired "
           r"interval below it in grey: 2000 prompt draws for the per-image metrics, and for FID, KID, and their patch "
           r"versions 1000 draws that also resample the 168 reference-image clusters. $^{\times}$: $\times100$; "
           r"$^{\ast}$: $\times1000$. \up{Blue} / \dn{orange}: the interval excludes zero in favour of / against "
           r"progressive \sca{}; for FID and KID lower is better, so a negative difference is a gain. Inception score "
           r"has no paired interval (Table~\ref{tab:all_metrics}); Table~\ref{tab:holm} gives the Holm-adjusted calls."
           + note + key + "}")
    text = "\n".join([
        HEADER.replace("% Generated", "% Generated (app_tables/prog_all.tex)"), TABSTYLE_INPUT,
        r"\begin{table}[!htbp]", cap, r"\label{tab:prog_all}", r"\centering\scriptsize"]
        + paired_panels(rows_, r"Progressive \sca{} $-$") + [r"\end{table}"]) + "\n"
    NOTES.append(f"prog_all: set-level pending against {pending_refs or 'none'}")
    return text


# ---------------------------------------------------------------- appendix Table variance_split ------------------
VS_ROWS = [("flux1_dev", "1024x1024", r"FLUX.1-dev, $1024^2$", 32), ("flux1_dev", "1280x1280", r"FLUX.1-dev, $1280^2$", 16),
           ("flux1_krea_dev", "1024x1024", r"FLUX.1-Krea-dev, $1024^2$", 16), ("flux2_dev", "1024x1024",
                                                                             r"FLUX.2-dev, $1024^2$", 16)]


def table_variance_split():
    """Appendix table: variance of each head's image attention within r tokens, split into head and noise level."""
    rel = "heads_view/variance_split.csv"
    R = {(r["checkpoint"], r["res"], r["sigma_range"], r["quantity"], r["radius"]): r for r in rows(rel)}
    body = []
    for rng, heading in (("post_gate", r"After the gate ($\sigma<0.89$)"), ("all", "All noise levels")):
        body.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{heading}}}}} \\")
        for ckpt, res, label, n in VS_ROWS:
            cells = [label, str(n)]
            for radius in ("8", "16"):
                r = R[(ckpt, res, rng, "image_share", radius)]
                head = 1 - float(r["noise_within_head"])
                cells += [f"{100 * head:.1f}", f"{100 * float(r['noise_within_head']):.1f}",
                          f"{100 * float(r['head_avg_curve']):.2f}"]
            body.append(" & ".join(cells) + r" \\")
        body.append(r"\midrule" if rng == "post_gate" else "")
    body = [b for b in body if b]
    min_head = min(1 - float(R[(c, s_, "post_gate", "image_share", rr)]["noise_within_head"])
                   for c, s_, _, _ in VS_ROWS for rr in ("8", "16"))
    cap = (r"\caption{\textbf{Reach is set by the head.} Variance of the share of each head's image attention within "
           r"8 or 16 tokens of the query (Chebyshev distance), over all (block, head, noise bin) cells. It is split into "
           r"the head (between blocks plus between heads of one block) and the noise level within a head; the two "
           r"sum to 100\%. Avg.\ is the variance of the head-averaged curve over noise levels, the part one schedule shared "
           r"by all heads could follow. 256 seeded queries per block and step, full attention at each row's "
           r"resolution. AReA's statistic is the distance to dominant keys; ours is the mass share, which sets the output "
           r"by Lemma~\ref{lem:split}. Key result: after the gate the head carries at least "
           + f"{100 * min_head:.1f}" + r"\% of the variance in every model and size.}")
    text = "\n".join([
        HEADER.replace("% Generated", "% Generated (app_tables/variance_split.tex)"),
        r"\begin{table}[!htbp]", cap, r"\label{tab:variance_split}",
        r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lr rrr rrr@{}}", r"\toprule",
        r" & & \multicolumn{3}{c}{within 8 tokens (\%)} & \multicolumn{3}{c}{within 16 tokens (\%)} \\",
        r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
        r"Model and size & Prompts & Head & Noise & Avg. & Head & Noise & Avg. \\", r"\midrule"]
        + body + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]) + "\n"
    post = [1 - float(R[(c, s, "post_gate", "image_share", rr)]["noise_within_head"])
            for c, s, _, _ in VS_ROWS for rr in ("8", "16")]
    check(min(post) >= 0.969, f"variance_split caption: minimum post-gate head share {min(post):.4f} < 0.969")
    return text


# ---------------------------------------------------------------- appendix Table holm (v6) ----------------------
# (a) the three headline comparisons metric by metric: difference, unadjusted and Holm-adjusted bootstrap p-values;
# (b) the record of every paired comparison of Tables 1 to 3 before and after adjustment. gen_holm.py reproduces each
# archived paired CSV before it computes a p-value, so every difference here equals the one in the other tables.
HOLM_SCALE = {"ir": (1, 3), "aesthetic": (1, 2), "clip": (1, 2), "pickscore": (1, 2), "hpsv2": (100, 2),
              "hpsv21": (100, 2), "musiq": (1, 1), "musiq_crop1024": (1, 1), "clipiqa": (100, 1), "arniqa": (100, 1)}
HOLM_HEAD = [("prog_area", r"Progressive $-$ AReA"), ("prog_hrdit", r"Progressive $-$ HRDiT"),
             ("sca_area", r"Single-pass $-$ AReA")]
HOLM_GROUPS = [("Table~\\ref{tab:main}: progressive \\sca{} against each method (\\nmain{} prompts, seed 0)",
                [("prog_area", "AReA"), ("prog_hrdit", "HRDiT"), ("prog_ultra", "UltraImage"), ("prog_dype", "DyPE"),
                 ("prog_sega", "SEGA"), ("prog_full", "full attention"), ("prog_sca", r"single-pass \sca{}")]),
               ("Table~\\ref{tab:generality}: \\sca{} against AReA, other seeds, prompts, checkpoints",
                [("sca_area", r"single-pass, seed 0"), ("prog_area_s1", "progressive, seed 1"),
                 ("prog_area_s2", "progressive, seed 2"), ("prog_area_hrgen", "progressive, HRGen-1K"),
                 ("sca_area_s1", "single-pass, seed 1"), ("sca_area_s2", "single-pass, seed 2"),
                 ("sca_area_hrgen", "single-pass, HRGen-1K"), ("sca_area_krea", "single-pass, FLUX.1-Krea"),
                 ("flux2", r"w/o schedule $-$ full attn., FLUX.2-dev")]),
               (r"Table~\ref{tab:ablation}: per-head weights against each variant (reversed: variant losses are per-head wins)",
                [("abl_uniform_nb", r"shared $\pi(\sigma)$, single pass"),
                 ("abl_layer_nb", r"per-layer $\pi_\ell(\sigma)$, single pass"),
                 ("prog_headmean", r"shared $\pi(\sigma)$, progressive"),
                 ("prog_layermean", r"per-layer $\pi_\ell(\sigma)$, progressive"),
                 ("prog_win", r"window only, progressive"), ("abl_v0", "no opening"),
                 ("abl_localonly_nb", "local only"), ("abl_globalonly", "global only"),
                 ("abl_mean_nb", "mean pooling")])]


def fmt_p(p):
    """A p-value at three decimals; the smallest value 2000 draws resolve prints as <=0.001."""
    return r"${\le}0.001$" if p < 0.0015 else f"{p:.3f}"


def table_holm():
    import gen_holm
    H = holm()
    body = []
    for grp, metrics in (("Preference", ["ir", "aesthetic", "clip", "pickscore", "hpsv2", "hpsv21"]),
                         ("Image quality", ["musiq", "musiq_crop1024", "clipiqa", "arniqa"])):
        body.append(rf"\multicolumn{{10}}{{l}}{{\emph{{{grp}}}}} \\")
        for k in metrics:
            sc, d = HOLM_SCALE[k]
            label = dict(gen_holm.METRICS)[k] + (r" ($\times100$)" if sc == 100 else "")
            cells = [label]
            for key, _ in HOLM_HEAD:
                g = H[key]["metrics"][k]
                s = {1: "up", -1: "dn", 0: ""}[g["sig_holm"]]
                cells += [paint(sgn(sc * g["delta"], d, "{-}"), s), fmt_p(g["p_boot"]), fmt_p(g["holm_boot"])]
            body.append(" & ".join(cells) + r" \\")
    head_a = [r"\toprule",
              " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{name}}}" for _, name in HOLM_HEAD) + r" \\",
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
              r"Metric" + r" & $\Delta$ & $p$ & $p_\text{Holm}$" * 3 + r" \\", r"\midrule"]
    rec = []
    for g_i, (heading, pairs) in enumerate(HOLM_GROUPS):
        rec += ([r"\midrule"] if g_i else []) + [rf"\multicolumn{{5}}{{l}}{{\emph{{{heading}}}}} \\"]
        for key, label in pairs:
            h = H[key]
            rev = h["group"] in ("abl", "ablprog")
            pair_ = (lambda u, dn: f"{dn}\\,/\\,{u}") if rev else (lambda u, dn: f"{u}\\,/\\,{dn}")
            rec.append(" & ".join([label, str(h["n"]), pair_(h["up_raw"], h["dn_raw"]), pair_(h["up"], h["dn"]),
                                   pair_(h["up_flip"], h["dn_flip"])]) + r" \\")
    head_b = [r"\toprule", r"Comparison & $n$ & 95\% interval & Holm, bootstrap & Holm, sign-flip \\", r"\midrule"]
    hr = H["prog_hrdit"]
    lost = [dict(gen_holm.METRICS)[k] for k, _ in gen_holm.METRICS
            if hr["metrics"][k]["sig_raw"] and not hr["metrics"][k]["sig_holm"]]
    kept = [dict(gen_holm.METRICS)[k] for k, _ in gen_holm.METRICS if hr["metrics"][k]["sig_holm"]]
    check((H["prog_area"]["up"], H["prog_area"]["dn"], H["sca_area"]["up"], H["sca_area"]["dn"]) == (10, 0, 10, 0),
          "tab:holm key result: a headline win over AReA does not survive Holm")
    check(kept == ["HPSv2", "ARNIQA"] and hr["dn"] == 0, f"tab:holm key result: HRDiT survivors {kept}")
    cap = (r"\caption{\textbf{Every win over AReA survives Holm correction; against HRDiT, two differences of four do.} "
           r"FLUX.1-dev at $4096^2$. (a) Paired differences, method minus reference, with two-sided bootstrap "
           r"$p$-values from the same 2000 prompt draws (seed 0) that give every interval of the paper, before and "
           r"after Holm's step-down adjustment over the ten per-image metrics of each comparison. \up{Blue} / "
           r"\dn{orange}: significantly better / worse after adjustment at 5\%. $p\le0.001$ is the smallest value 2000 "
           r"draws resolve. (b) Significantly better / worse metrics of ten for every paired comparison of "
           r"Tables~\ref{tab:main} to~\ref{tab:ablation}: unadjusted (the 95\% interval excludes zero), Holm on the "
           r"bootstrap $p$-values, and Holm on paired sign-flip $p$-values (2000 sign vectors, seed 0) as a check. "
           r"Every archived interval is reproduced to six digits before a $p$-value is taken. Key result: progressive "
           r"and single-pass \sca{} keep all ten wins over AReA at every seed and on HRGen-1K; against HRDiT, "
           + " and ".join(kept) + r" remain better and " + " and ".join(lost) +
           r" no longer differ.}")
    text = "\n".join(
        [HEADER.replace("% Generated", "% Generated (app_tables/holm.tex, with gen_holm.py)"), TABSTYLE_INPUT,
         r"\begin{table}[!htbp]", cap, r"\label{tab:holm}", r"\centering\scriptsize",
         r"\textit{(a) The headline comparisons, metric by metric}\par\smallskip",
         r"\scaFitTab{l*{9}{r}}{10}{%"] + ["  " + x for x in head_a + body + [r"\bottomrule"]] + ["}", r"\par\medskip",
         r"\textit{(b) Wins\,/\,losses of ten for every paired comparison}\par\smallskip",
         r"\scaFitTab{lrccc}{5}{%"] + ["  " + x for x in head_b + rec + [r"\bottomrule"]] + ["}", r"\end{table}"])
    return text + "\n"


def table_holm_short():
    """Panel (a) of table_holm() alone, for the short review edition (main_short.tex inputs app_tables/holm_short.tex
    in place of holm.tex): the same cells and the same label tab:holm; the caption drops panel (b) and names the full
    version for the record of every paired comparison. Added 2026-11-10 (appendix owner, short edition)."""
    import gen_holm
    full = table_holm().splitlines()
    i0 = full.index(r"\scaFitTab{l*{9}{r}}{10}{%")
    i1 = full.index(r"\par\medskip", i0)
    H = holm()
    hr = H["prog_hrdit"]
    lost = [dict(gen_holm.METRICS)[k] for k, _ in gen_holm.METRICS
            if hr["metrics"][k]["sig_raw"] and not hr["metrics"][k]["sig_holm"]]
    kept = [dict(gen_holm.METRICS)[k] for k, _ in gen_holm.METRICS if hr["metrics"][k]["sig_holm"]]
    cap = (r"\caption{\textbf{Every win over AReA survives Holm correction; against HRDiT, two differences of four do.} "
           r"FLUX.1-dev at $4096^2$. Paired differences, method minus reference, with two-sided bootstrap "
           r"$p$-values from the same 2000 prompt draws (seed 0) that give every interval of the paper, before and "
           r"after Holm's step-down adjustment over the ten per-image metrics of each comparison. \up{Blue} / "
           r"\dn{orange}: significantly better / worse after adjustment at 5\%. $p\le0.001$ is the smallest value 2000 "
           r"draws resolve. The full version adds the record of every paired comparison of Tables~\ref{tab:main} "
           r"to~\ref{tab:ablation}, with a sign-flip check. Key result: progressive and single-pass \sca{} keep all "
           r"ten wins over AReA; against HRDiT, " + " and ".join(kept) + r" remain better and " + " and ".join(lost) +
           r" no longer differ.}")
    text = "\n".join(
        [HEADER.replace("% Generated", "% Generated (app_tables/holm_short.tex: panel (a) of holm.tex for the short "
                                       "edition, with gen_holm.py)"), TABSTYLE_INPUT,
         r"\begin{table}[!htbp]", cap, r"\label{tab:holm}", r"\centering\scriptsize"] + full[i0:i1] + [r"\end{table}"])
    return text + "\n"


def holm_macros():
    """\\holm<Pair>Better / Worse (digits) and ...Word (words) for every comparison; the HRDiT survivors as a list."""
    import gen_holm
    H = holm()
    names = {"prog_area": "ProgArea", "prog_hrdit": "ProgHrdit", "sca_area": "ScaArea", "prog_ultra": "ProgUltra",
             "prog_dype": "ProgDype", "prog_sega": "ProgSega", "prog_full": "ProgFull", "prog_sca": "ProgSca",
             "prog_area_s1": "ProgAreaSeedOne", "prog_area_s2": "ProgAreaSeedTwo", "prog_area_hrgen": "ProgAreaHrgen",
             "sca_area_s1": "ScaAreaSeedOne", "sca_area_s2": "ScaAreaSeedTwo", "sca_area_hrgen": "ScaAreaHrgen",
             "sca_area_krea": "ScaAreaKrea", "flux2": "FluxTwo", "abl_uniform_nb": "AblShared",
             "abl_layer_nb": "AblLayer", "prog_headmean": "ProgAblShared", "prog_layermean": "ProgAblLayer",
             "prog_win": "ProgAblWindow", "abl_v0": "AblNoOpening", "abl_localonly_nb": "AblLocal",
             "abl_globalonly": "AblGlobal", "abl_mean_nb": "AblMean"}
    out = []
    for key, nm in names.items():
        h = H[key]
        src = f"gen_holm.py on the per-image files behind {h['src']}: metrics of ten, Holm-adjusted bootstrap p <= 0.05"
        for tag, v in (("Better", h["up"]), ("Worse", h["dn"])):
            out.append((f"holm{nm}{tag}", str(v), f"{src}, {tag.lower()} for the first-named method", False))
            out.append((f"holm{nm}{tag}Word", word(v), "the same count as a word", False))
        out.append((f"holm{nm}BetterRaw", str(h["up_raw"]), f"{src.split(':')[0]}: unadjusted count (interval)", False))
        out.append((f"holm{nm}WorseRaw", str(h["dn_raw"]), f"{src.split(':')[0]}: unadjusted count (interval)", False))
    hr = H["prog_hrdit"]
    lab = dict(gen_holm.METRICS)
    kept = [lab[k] for k, _ in gen_holm.METRICS if hr["metrics"][k]["sig_holm"] > 0]
    lost = [lab[k] for k, _ in gen_holm.METRICS if hr["metrics"][k]["sig_raw"] and not hr["metrics"][k]["sig_holm"]]
    out += [("holmProgHrditKept", " and ".join(kept), "gen_holm.py: metrics on which progressive SCA stays "
                                                       "significantly better than HRDiT after Holm", False),
            ("holmProgHrditDropped", " and ".join(lost), "gen_holm.py: metrics significant before but not after Holm "
                                                         "(progressive SCA minus HRDiT)", False)]
    for k, nm in (("musiq", "Musiq"), ("clipiqa", "Clipiqa")):
        g = hr["metrics"][k]
        out += [(f"holmProgHrdit{nm}P", f"{g['p_boot']:.3f}", "gen_holm.py: unadjusted bootstrap p, progressive SCA "
                                                               f"minus HRDiT, {k}", False),
                (f"holmProgHrdit{nm}PHolm", f"{g['holm_boot']:.2f}", f"gen_holm.py: Holm-adjusted p, {k}", False)]
    g = H["prog_sca"]["metrics"]["ir"]
    out += [("holmProgScaIrP", f"{g['p_boot']:.3f}", "gen_holm.py: unadjusted bootstrap p, progressive minus single-pass "
                                                      "SCA, ImageReward", False),
            ("holmProgScaIrPHolm", f"{g['holm_boot']:.3f}", "gen_holm.py: Holm-adjusted bootstrap p, the same", False),
            ("holmProgScaIrPHolmFlip", f"{g['holm_flip']:.3f}", "gen_holm.py: Holm-adjusted sign-flip p, the same",
             False)]
    return out


# ---------------------------------------------------------------- appendix Table app_abl (v6) -------------------
# Generated replacement of the hand-typed Table tab:app_abl (v5.1 kept gen_tables.py's output cell for cell): the
# single-pass variants on the fixed 46-prompt subset, each minus SCA without the schedule (results/ablnb_<m>_deltas.csv,
# 2000 draws), the reference row in absolute values; panel (c) gives corrected patch FID (point estimates, no
# interval) and the relative output error of each weight rule at emulated ratio 4 (k1_1024.json; mean pooling has
# none, since reach emulation scores block-centre keys only; no opening keeps the per-head table).
APP_ABL_ROWS = [("abl_uniform_nb", r"shared $\pi(\sigma)$", "uniform"), ("abl_layer_nb", r"per-layer $\pi_\ell(\sigma)$",
                                                                       "per_layer"),
                ("abl_localonly_nb", "local only", "loc"), ("abl_globalonly", "global only", "glob"),
                ("abl_mean_nb", "mean pooling", None), ("abl_v0", "no opening", "per_head")]


def table_app_abl():
    S, pfid = abl46_reference()
    err = json.load(open(RES / CALIB_JSON))["scales"]["4"]["rel_err"]
    spec = {k: (sc, d) for k, _, sc, d in PROG_ALL_METRICS}
    ref_label = r"per-head $\pi_h$"
    out = []
    for p_i, (title, keys) in enumerate(PANELS_V6[:2]):
        head = [r"\toprule", "Variant & " + " & ".join(SHORT_V6[k] for k in keys) + r" \\", r"\midrule"]
        body = [r"\scaRowOurs " + " & ".join([ref_label] + [num(spec[k][0] * S[(REF, k)], spec[k][1]) for k in keys])
                + r" \\[1pt]"]
        for m, label, _ in APP_ABL_ROWS:
            dd = ablation_deltas(m)
            top, low = [label], [""]
            for k in keys:
                sc, d = spec[k]
                v, lo, hi = dd[k]
                top.append(paint(sgn(sc * v, d, "{-}"), sig(dd[k], lower=False) or ""))
                low.append(rf"{{\color{{scaCtx}}{interval(sc * lo, sc * hi, d)}}}")
            body += [" & ".join(top) + r" \\", " & ".join(low) + r" \\[1pt]"]
        out += ([r"\par\medskip"] if p_i else []) + [rf"\textit{{{title}}}\par\smallskip",
                                                     rf"\scaFitTab{{l*{{{len(keys)}}}{{r}}}}{{{len(keys) + 1}}}{{%"]
        out += ["  " + x for x in head + body + [r"\bottomrule"]] + ["}"]
    rows_c = [r"\scaRowOurs " + ref_label + rf" & {num(pfid[REF], 1)} & {err['per_head']:.3f} \\"]
    for m, label, key in APP_ABL_ROWS:
        rows_c.append(rf"{label} & {num(pfid[m], 1)} & {f'{err[key]:.3f}' if key else '--'} \\")
    out += [r"\par\medskip", r"\textit{(c) Patch FID and output error, absolute values}\par\smallskip",
            r"\scaFitTab{lrr}{3}{%"] + ["  " + x for x in [r"\toprule",
                                                       r"Variant & FID$_\text{p}\downarrow$ & Output error$\downarrow$ \\",
                                                       r"\midrule"] + rows_c + [r"\bottomrule"]] + ["}"]
    lost = {m: sum(ablation_deltas(m)[k][2] < 0 for k in PER_IMAGE10) for m, _, _ in APP_ABL_ROWS}
    check(min(lost.values()) >= 5 and max(lost.values()) <= 10, f"app_abl key result: losses {lost}")
    check(all(f"{err[k]:.3f}" > f"{err['per_head']:.3f}" for _, _, k in APP_ABL_ROWS if k and k != "per_head"),
          "app_abl: a coarser rule has lower output error than per-head weights")
    cap = (r"\caption{\textbf{Every single-pass variant is significantly worse than per-head weights on five to ten "
           r"per-image metrics.} Expands the single-pass rows of Table~\ref{tab:ablation}. FLUX.1-dev at $4096^2$, the "
           r"fixed \nabl-prompt ablation subset, \sca{} without the schedule throughout. The shaded row gives absolute "
           r"values; every other cell is the variant minus that reference, with its 95\% paired interval below it in "
           r"grey (2000 prompt draws). $^{\times}$: $\times100$. \up{Blue} / \dn{orange}: the interval excludes zero in "
           r"favour of / against the variant. (c) Patch FID is set-level and has no paired interval at 46 prompts; "
           r"output error is the relative error of each weight rule at emulated ratio 4 (Section~\ref{sec:ablations}); "
           r"mean pooling has none because reach emulation scores block-centre keys only, and the no-opening variant "
           r"keeps the per-head table. Table~\ref{tab:holm} gives the Holm-adjusted counts. Key result: shared and "
           r"per-layer weights, which raise the output error, lose " + word(lost["abl_uniform_nb"]) + " and " +
           word(lost["abl_layer_nb"]) + r" per-image metrics.}")
    text = "\n".join([HEADER.replace("% Generated", "% Generated (app_tables/app_abl.tex)"), TABSTYLE_INPUT,
                      r"\begin{table}[!htbp]", cap, r"\label{tab:app_abl}", r"\centering\scriptsize"] + out +
                     [r"\end{table}"])
    return text + "\n"


# ---------------------------------------------------------------- appendix Tables app_ps2, phase2_sensitivity,
# phase2_budget (v6): generated replacements of three tables v5.1 typed into appendix.tex from gen_tables.py's output.
# Comparisons as rows, metrics as columns in two panels (preference, image quality), each difference with its 95%
# paired interval below it in grey (2000 prompt draws, scripts/paired_bootstrap.py files under results/).
def _per_image_panels(rows_, head):
    spec = {k: (sc, d) for k, _, sc, d in PROG_ALL_METRICS}
    out = []
    for p_i, (title, keys) in enumerate(PANELS_V6[:2]):
        body = []
        for label, dd in rows_:
            top, low = [label], [""]
            for k in keys:
                sc, d = spec[k]
                v, lo, hi = dd[k]
                top.append(paint(sgn(sc * v, d, "{-}"), sig(dd[k], lower=False) or ""))
                low.append(rf"{{\color{{scaCtx}}{interval(sc * lo, sc * hi, d)}}}")
            body += [" & ".join(top) + r" \\", " & ".join(low) + r" \\[1pt]"]
        out += ([r"\par\medskip"] if p_i else []) + [rf"\textit{{{title}}}\par\smallskip",
                                                     rf"\scaFitTab{{l*{{{len(keys)}}}{{r}}}}{{{len(keys) + 1}}}{{%"]
        out += ["  " + x for x in [r"\toprule", head + " & " + " & ".join(SHORT_V6[k] for k in keys) + r" \\",
                                   r"\midrule"] + body + [r"\bottomrule"]] + ["}"]
    return out


def deltas_csv(rel, method, reference):
    out = {}
    for r in rows(rel):
        assert (r["method"], r["reference"]) == (method, reference), (rel, r)
        out[r["metric"]] = (float(r["delta"]), float(r["ci_low"]), float(r["ci_high"]))
    return out


def counts(dd):
    up = [k for k in PER_IMAGE10 if dd[k][1] > 0]
    dn = [k for k in PER_IMAGE10 if dd[k][2] < 0]
    return up, dn


def float_table(name, label, cap, body):
    return "\n".join([HEADER.replace("% Generated", f"% Generated (app_tables/{name}.tex)"), TABSTYLE_INPUT,
                      r"\begin{table}[!htbp]", cap, rf"\label{{{label}}}", r"\centering\scriptsize"] + body +
                     [r"\end{table}"]) + "\n"


def table_app_ps2():
    ps2 = deltas_csv("ps2_abl_ps2_vs_scanb.csv", "abl_ps2", REF)
    area = deltas_csv("ps2_area_vs_scanb.csv", "area", REF)
    up, dn = counts(ps2)
    sig_ = up + dn
    big = max(abs(ps2[k][0]) * {"hpsv2": 100, "hpsv21": 100, "clipiqa": 100, "arniqa": 100}.get(k, 1) for k in sig_)
    check(len(sig_) == 3 and f"{big:.2f}" == "0.17", f"app_ps2 caption: {sig_} significant, largest {big:.3f}")
    check(not dn, f"app_ps2 caption: the ratio-2 table is worse on {dn}")
    cap = (r"\caption{\textbf{At $2048\times4096$ the ratio-2 table raises " + word(len(sig_)) + r" metrics "
           rf"significantly, by at most {big:.2f}, and lowers none.}} Checks the ratio rule of "
           r"Section~\ref{sec:emulation} at a non-square size. FLUX.1-dev, the \nabl{} ablation prompts, \sca{} without "
           r"the schedule throughout. Each row is paired against the same operator with the default ratio-4 table: "
           r"the operator with the ratio-2 table, and AReA (released code) for scale. $^{\times}$: $\times100$. "
           r"\up{Blue} / \dn{orange}: the 95\% paired interval (2000 prompt draws) excludes zero in favour of / against "
           r"the row. Key result: the default ratio-4 table trails the ratio-2 table on " + word(len(sig_)) +
           rf" of ten metrics, by at most {big:.2f}, and leads on none.}}")
    rows_ = [(r"ratio-2 table", ps2), ("AReA", area)]
    return float_table("app_ps2", "tab:app_ps2", cap, _per_image_panels(rows_, r"Row $-$ ratio-4 table"))


def table_phase2_sensitivity():
    rows_ = [(f"{n}-prompt table", deltas_csv(f"e6_sca_cal{n}_deltas.csv", f"sca_cal{n}", REF)) for n in (4, 8, 16)]
    rows_ += [(rf"gate $\sigma_g=0.{g}$", deltas_csv(f"e7_sca_g{g}_deltas.csv", f"sca_g{g}", REF)) for g in (85, 93)]
    nsig = sum(len(sum(counts(dd), [])) for _, dd in rows_[:3])
    g85_dn = len(counts(rows_[3][1])[1])
    musiq_up = [lab for lab, dd in rows_[:3] if dd["musiq"][1] > 0]
    check(nsig == 1 and musiq_up == ["16-prompt table"] and g85_dn == 7,
          f"phase2_sensitivity caption: {nsig} significant, MUSIQ up {musiq_up}, gate 0.85 lowers {g85_dn}")
    cap = (r"\caption{\textbf{Reach tables fitted on 4 to 16 prompts change one of 30 differences significantly; a gate "
           r"at $\sigma_g=0.85$ lowers " + word(g85_dn) + r" metrics.} Supports the prompt-count check of Section~\ref{sec:generality}. "
           r"FLUX.1-dev at $4096^2$, the \nabl{} ablation prompts, \sca{} without the schedule throughout. Each row is "
           r"the variant minus the default (the 32-prompt table, gate $\sigma_g=0.89$). $^{\times}$: $\times100$. "
           r"\up{Blue} / \dn{orange}: the 95\% paired interval (2000 prompt draws) excludes zero in favour of / against "
           r"the variant. Key result: tables fitted on 4 to 16 prompts move ImageReward by at most "
           + f"{max(abs(dd['ir'][0]) for _, dd in rows_[:3]):.3f}" + r", and one of their 30 differences is "
           r"significant.}")
    return float_table("phase2_sensitivity", "tab:phase2_sensitivity", cap,
                       _per_image_panels(rows_, r"Variant $-$ default"))


def table_phase2_budget():
    rows_ = [(r"56 vs.\ 28 steps", data195.pairs(PANEL_RUN, "sca_s56", REF)),
             (r"guidance 7 vs.\ 4.5", data195.pairs(PANEL_RUN, "sca_g7", REF)),
             (r"56 steps vs.\ UltraImage", data195.pairs(PANEL_RUN, "sca_s56", "ultraimage"))]
    up56, dn56 = counts(rows_[0][1])
    upg, dng = counts(rows_[1][1])
    up_u, dn_u = counts(rows_[2][1])
    check(len(up56) == 8 and not dn56 and len(dng) == 7 and upg == ["musiq"],
          f"phase2_budget caption: 56 steps {up56}/{dn56}, guidance 7 {upg}/{dng}")
    cap = (r"\caption{\textbf{Doubling the steps raises " + word(len(up56)) + r" metrics; guidance 7 lowers " +
           word(len(dng)) + r" and raises MUSIQ.} Expands Table~\ref{tab:panel46}. FLUX.1-dev at $4096^2$, \npanel{} "
           r"prompts, all with \sca{} without the schedule: the first two rows compare with its default 28-step run at "
           r"guidance 4.5, the last compares its 56-step run with UltraImage. Each row is the first-named setting "
           r"minus the second. $^{\times}$: $\times100$. \up{Blue} / \dn{orange}: the 95\% paired interval (2000 prompt "
           r"draws) excludes zero in favour of / against the first-named setting. Key result: at 56 steps \sca{} "
           r"without the schedule is better than UltraImage on " + word(len(up_u)) + r" metrics and worse on "
           + word(len(dn_u)) + ".}")
    return float_table("phase2_budget", "tab:phase2_budget", cap, _per_image_panels(rows_, "Comparison"))


# ---------------------------------------------------------------- appendix Table gen_all (v6) -------------------
# Expands Table 3: every generality comparison on all ten per-image metrics, each difference with its 95% interval
# on the same line (gen_holm.py's reproduction of the archived paired CSVs, 2000 prompt draws, seed 0).
def table_gen_all():
    H = holm()
    blocks = [("Progressive \\sca{} $-$ AReA", [("prog_area", "Seed 0"), ("prog_area_s1", "Seed 1"),
                                                 ("prog_area_s2", "Seed 2"), ("prog_area_hrgen", "HRGen-1K")]),
              ("Single-pass \\sca{} $-$ AReA", [("sca_area", "Seed 0"), ("sca_area_s1", "Seed 1"),
                                                 ("sca_area_s2", "Seed 2"), ("sca_area_hrgen", "HRGen-1K"),
                                                 ("sca_area_krea", "FLUX.1-Krea")]),
              ("\\sca{} without the schedule $-$ full attention", [("flux2", "FLUX.2-dev")])]
    short = {"ir": "IR", "aesthetic": "Aesthetic", "clip": "CLIP", "pickscore": "PickScore", "hpsv2": "HPSv2",
             "hpsv21": "HPSv2.1", "musiq": "MUSIQ", "musiq_crop1024": r"MUSIQ$_\text{c}$", "clipiqa": "CLIP-IQA",
             "arniqa": "ARNIQA"}
    panels = []
    for p_i, (pname, keys) in enumerate((("Preference", ["ir", "aesthetic", "clip", "pickscore", "hpsv2", "hpsv21"]),
                                          ("Image quality", ["musiq", "musiq_crop1024", "clipiqa", "arniqa"]))):
        body = []
        for b_i, (heading, settings) in enumerate(blocks):
            body += ([r"\midrule"] if b_i else []) + [rf"\multicolumn{{{len(keys) + 2}}}{{l}}{{\emph{{{heading}}}}} \\"]
            for key, label in settings:
                h = H[key]
                top, low = [label, str(h["n"])], ["", ""]
                for k in keys:
                    sc, d = HOLM_SCALE[k]  # ImageReward at three decimals, as phase2_general and prog_all print it
                    g = h["metrics"][k]
                    top.append(paint(sgn(sc * g["delta"], d, "{-}"),
                                     sig((g["delta"], g["lo"], g["hi"]), lower=False) or ""))
                    low.append(rf"{{\color{{scaCtx}}{interval(sc * g['lo'], sc * g['hi'], d)}}}")
                body += [" & ".join(top) + r" \\", " & ".join(low) + r" \\[1pt]"]
        heads = [short[k] + (r" ($\times100$)" if HOLM_SCALE[k][0] == 100 else "") for k in keys]
        head = [r"\toprule", "Setting & $n$ & " + " & ".join(heads) + r" \\", r"\midrule"]
        panels += ([r"\par\medskip"] if p_i else []) + [rf"\textit{{({'ab'[p_i]}) {pname}}}\par\smallskip",
                                                          rf"\scaFitTab{{lr*{{{len(keys)}}}{{r}}}}{{{len(keys) + 2}}}{{%"]
        panels += ["  " + x for x in head + body + [r"\bottomrule"]] + ["}"]
    cap = (r"\caption{\textbf{Table~\ref{tab:generality} with all ten per-image metrics.} FLUX.1-dev at $4096^2$ unless "
           r"named; each cell is the paired difference (HPS, CLIP-IQA, ARNIQA $\times100$) with its 95\% interval below "
           r"it in grey (2000 prompt draws, seed 0). \up{Blue} / \dn{orange}: the interval excludes zero in favour of / "
           r"against \sca{}. Seeds 0 to 2 each cover all \nmain{} Aesthetic-4K prompts; HRGen-1K has \nhrgen{} prompts; "
           r"FLUX.1-Krea-dev runs the same \nkrea{} prompts as seed 0; FLUX.2-dev uses 50. Table~\ref{tab:holm} gives "
           r"the Holm-adjusted calls. Key result: no comparison is significantly worse on any metric, and every "
           r"comparison on FLUX.1-dev is significantly better on all ten.}")
    for _, settings in blocks:
        for key, _ in settings:
            check(H[key]["dn_raw"] == 0, f"gen_all key result: {key} loses a metric")
    for key in ("prog_area", "prog_area_s1", "prog_area_s2", "prog_area_hrgen", "sca_area", "sca_area_s1",
                "sca_area_s2", "sca_area_hrgen"):
        check(H[key]["up_raw"] == 10, f"gen_all key result: {key} wins {H[key]['up_raw']} of ten")
    text = "\n".join(
        [HEADER.replace("% Generated", "% Generated (app_tables/gen_all.tex, with gen_holm.py)"), TABSTYLE_INPUT,
         r"\begin{table}[!htbp]", cap, r"\label{tab:gen_all}", r"\centering\scriptsize"] + panels + [r"\end{table}"])
    return text + "\n"


# ---------------------------------------------------------------- headline macros ---------------------------------
# The 15 metrics of the SCA-versus-AReA comparison (Table app_4k, Figure paired_area); the last four are lower-better.
AREA_METRICS = ["ir", "aesthetic", "clip", "pickscore", "hpsv2", "hpsv21", "musiq", "musiq_crop1024", "clipiqa",
                "arniqa", "is", "fid", "kid", "patch_fid", "patch_kid"]
LOWER = {"fid", "kid", "patch_fid", "patch_kid"}
PER_IMAGE10 = AREA_METRICS[:10]


def area_counts(vs):
    """(wins, losses, significant, better, no-interval metrics) of SCA minus AReA over AREA_METRICS."""
    sg = {m: vs[m][1] is not None and (vs[m][1] > 0 or vs[m][2] < 0) for m in AREA_METRICS}
    better = {m: (vs[m][0] < 0) == (m in LOWER) for m in AREA_METRICS}
    wins = sum(sg[m] and better[m] for m in AREA_METRICS)
    losses = sum(sg[m] and not better[m] for m in AREA_METRICS)
    return wins, losses, sg, better, [m for m in AREA_METRICS if vs[m][1] is None]


def sca_vs_area(run, m):
    """{metric: (delta, lo, hi)}: m minus AReA in `run` over AREA_METRICS, as the headline counts wins (the run's
    eval.py deltas negated, FID, KID and patch versions corrected as data195.load describes)."""
    S, D, ref, _, _ = data(run)
    vs = data195.versus(S, D, ref, run, "area", AREA_METRICS, RES)
    return {k: vs[(m, k)] for k in AREA_METRICS}


def deltas_source(run, m, metric_desc):
    """Provenance of `m` minus AReA in `run` (sca_vs_area): the files, and how the delta is read from them."""
    S, D, ref, corrected_ci, src = data(run)
    fid = (f"results/{run}/corrected_fidp_ci/paired_deltas.csv" if corrected_ci else
           f"results/{run}/corrected_fidp/paired_deltas.csv (points; whole-image FID and KID intervals from eval.py, "
           f"patch FID and KID without one)")
    how = (f"method {m} against area" if ref == "area" else
           f"method area against {ref}, negated to {m} minus AReA (columns value, ci_high, ci_low)")
    return f"{src} (1000 prompt draws) and {fid}, {how}, {metric_desc}"


def triple(out, name, v, sc, d, src, signed=True):
    """Append name, nameLo, nameHi for a (delta, lo, hi) scaled by sc at d decimals."""
    out += [(name, (f"{sc * v[0]:+.{d}f}" if signed else f"{sc * v[0]:.{d}f}"), f"{src}, delta x {sc}", False),
            (f"{name}Lo", f"{sc * v[1]:.{d}f}", f"{src}, lower end x {sc}", False),
            (f"{name}Hi", f"{sc * v[2]:.{d}f}", f"{src}, upper end x {sc}", False)]


def prog_macros(clean):
    """Numbers of the progressive headline: against AReA, HRDiT, UltraImage, DyPE, SEGA, full attention and
    single-pass SCA (per-image pairs and corrected set-level intervals), and campaign times."""
    out = []
    names = {"clipiqa": "CLIP-IQA", "arniqa": "ARNIQA", "musiq": "MUSIQ", "musiq_crop1024": "MUSIQ_c",
             "ir": "ImageReward", "aesthetic": "aesthetic", "clip": "CLIP", "pickscore": "PickScore",
             "hpsv2": "HPSv2", "hpsv21": "HPSv2.1"}
    wins = {}
    for m in ("area", "dype", "sega_official", "full", "ultraimage", "hrdit_dev_author", "sca_v2"):
        pi, _, src = prog_pairs(m)
        wins[m] = [k for k in PER_IMAGE10 if pi[k][1] > 0]
        losses = [k for k in PER_IMAGE10 if pi[k][2] < 0]
        NOTES.append(f"progressive minus {m}: wins {len(wins[m])}/10 {wins[m]}; losses {losses}")
    for m, nm in (("area", "Area"), ("dype", "Dype"), ("sega_official", "Sega"), ("full", "Full"),
                  ("ultraimage", "Ultra")):
        out.append((f"hlProgWins{nm}", str(len(wins[m])), f"results/{PROG_PAIRS}/{PROG}_vs_{m}.csv: per-image "
                                                           "metrics (of 10) with ci_low > 0", False))
        out.append((f"hlProgWins{nm}Word", word(len(wins[m])), "the same count as a word", False))
    ultra_open = [names[k] for k in PER_IMAGE10 if k not in wins["ultraimage"]]
    out.append(("hlProgUltraOpen", ", ".join(ultra_open), f"results/{PROG_PAIRS}/{PROG}_vs_ultraimage.csv: "
                                                         "per-image metrics without a significant win", False))
    # against AReA
    pa, _, srca = prog_pairs("area")
    for k, nm, sc, d in (("ir", "Ir", 1, 2), ("pickscore", "Pick", 1, 2), ("hpsv21", "Hps", 100, 2),
                         ("musiq_crop1024", "Musiqc", 1, 1), ("musiq", "Musiq", 1, 1), ("clip", "Clip", 1, 2)):
        triple(out, f"hlProgArea{nm}", pa[k], sc, d, f"{srca}, metric {k}")
    sla, srcsla = prog_setlevel("area")
    for k, nm, sc, d in (("fid", "Fid", 1, 1), ("patch_fid", "FidP", 1, 1), ("kid", "Kid", 100, 3),
                         ("patch_kid", "KidP", 100, 3)):
        triple(out, f"hlProgArea{nm}", sla[k], sc, d, f"{srcsla}, metric {k}")
    check(sig(sla["patch_fid"], True) == "dn" and sig(sla["patch_kid"], True) == "dn",
          "text: progressive SCA significantly worse than AReA on patch FID and patch KID")
    check(sig(sla["fid"], True) == "" and sig(sla["kid"], True) == "",
          "text: progressive SCA against AReA unresolved on FID and KID")
    # against HRDiT
    ph, _, srch = prog_pairs("hrdit_dev_author")
    for k, nm, sc, d in (("ir", "Ir", 1, 2), ("hpsv21", "Hps", 100, 2), ("pickscore", "Pick", 1, 2),
                         ("hpsv2", "HpsTwo", 100, 2), ("arniqa", "Arniqa", 100, 1), ("musiq", "Musiq", 1, 1),
                         ("clipiqa", "Clipiqa", 100, 1), ("musiq_crop1024", "Musiqc", 1, 1)):
        triple(out, f"hlProgHrdit{nm}", ph[k], sc, d, f"{srch}, metric {k}")
    hr_better = [k for k in PER_IMAGE10 if ph[k][1] > 0]
    hr_worse = [k for k in PER_IMAGE10 if ph[k][2] < 0]
    check(sorted(hr_better) == ["arniqa", "hpsv2", "musiq"] and hr_worse == ["clipiqa"],
          f"text: progressive vs HRDiT better on {hr_better}, worse on {hr_worse} (text names HPSv2, ARNIQA, MUSIQ; "
          "CLIP-IQA)")
    check(all(ph[k][1] <= 0 <= ph[k][2] for k in ("ir", "hpsv21", "pickscore")),
          "text: progressive vs HRDiT intervals on IR, HPSv2.1, PickScore include zero")
    slh, srcslh = prog_setlevel("hrdit_dev_author")
    if slh:
        for k, nm, sc, d in (("fid", "Fid", 1, 1), ("patch_fid", "FidP", 1, 1)):
            triple(out, f"hlProgHrdit{nm}", slh[k], sc, d, f"{srcslh}, metric {k}")
    # against single-pass SCA
    ps, _, srcs = prog_pairs("sca_v2")
    for k, nm, sc, d in (("ir", "Ir", 1, 2), ("pickscore", "Pick", 1, 2), ("hpsv21", "Hps", 100, 2),
                         ("musiq", "Musiq", 1, 1), ("musiq_crop1024", "Musiqc", 1, 1)):
        triple(out, f"hlProgSca{nm}", ps[k], sc, d, f"{srcs}, metric {k}")
    sls, srcsls = prog_setlevel("sca_v2")
    for k, nm, sc, d in (("fid", "Fid", 1, 1), ("patch_fid", "FidP", 1, 1), ("kid", "Kid", 100, 3),
                         ("patch_kid", "KidP", 100, 3)):
        triple(out, f"hlProgSca{nm}", sls[k], sc, d, f"{srcsls}, metric {k}")
    check(sig(ps["musiq_crop1024"], False) == "dn" and sig(sls["patch_fid"], True) == "dn",
          "text: progressive SCA trails single-pass SCA on MUSIQ_c and patch FID")
    check(all(sig(ps[k], False) == "up" for k in ("ir", "pickscore", "hpsv21")),
          "text: progressive SCA beats single-pass SCA on IR, PickScore, HPSv2.1")
    # progressive SCA on other seeds against AReA (Table 3's progressive rows), once their paired files land
    for label, nm, _, cands in [(g[0], {"Seed 1": "SeedOne", "Seed 2": "SeedTwo", "HRGen-1K": "Hrgen"}.get(g[0]), g[1], g[2])
                                for g in GEN_PROG if g[0] != "Seed 0"]:
        rel = next((c for c in cands if (RES / c).exists()), None)
        if rel:
            dd, n = V.paired_csv(rel, PROG, "area")
            w = sum(dd[k][1] > 0 for k in PER_IMAGE10)
            out += [(f"hlProg{nm}WinsArea", str(w), f"results/{rel}: per-image metrics (of 10) with ci_low > 0", False),
                    (f"hlProg{nm}WinsAreaWord", word(w), "the same count as a word", False)]
            triple(out, f"hlProg{nm}AreaIr", dd["ir"], 1, 2, f"results/{rel}, metric ir")
            triple(out, f"hlProg{nm}AreaPick", dd["pickscore"], 1, 2, f"results/{rel}, metric pickscore")
            if label.startswith("Seed"):
                check(w == 10, f"text 'all ten per-image metrics for each of three seeds': {label} wins {w}")
            # patch FID against AReA at this seed (files hold AReA minus progressive): interval if the 1000-draw
            # cluster bootstrap has landed (corrected_fidp_ci), else the point and the word "pending"
            run_dir = rel.split("/")[0]
            for sub in ("corrected_fidp_ci", "corrected_fidp"):
                pfile = RES / run_dir / sub / "paired_deltas.csv"
                if not pfile.exists():
                    continue
                r = next(r for r in rows(f"{run_dir}/{sub}/paired_deltas.csv")
                         if r["method"] == "area" and r["reference_method"] == PROG and r["metric"] == "patch_fid")
                src = f"results/{run_dir}/{sub}/paired_deltas.csv, area minus {PROG}, patch_fid, negated"
                if r["ci_low"] and r["ci_high"] and r["bootstrap_valid"] == "1000":
                    v = neg((float(r["value"]), float(r["ci_low"]), float(r["ci_high"])))
                    triple(out, f"hlProg{nm}AreaFidP", v, 1, 1, src)
                    sg = sig(v, lower=True)
                    out.append((f"hlProg{nm}AreaFidPText",
                                f"${v[0]:+.1f}$ $[{v[1]:.1f},{v[2]:.1f}]$".replace("[-", "[{-}").replace(",-", ",{-}"),
                                src, False))
                    out.append((f"hlProg{nm}AreaFidPSig", {"dn": "significantly higher", "up": "significantly lower",
                                                         "": "not significantly different"}[sg], src, False))
                    # main text: "significant at seeds 0 and 2 but not at seed 1" (seed 0 is checked in prog_macros)
                    check(sg == {"SeedOne": "", "SeedTwo": "dn"}.get(nm, sg),
                          f"text: patch FID gap to AReA at {label} is '{sg}', text says seed 1 unresolved, seed 2 higher")
                else:
                    out.append((f"hlProg{nm}AreaFidPText", f"${-float(r['value']):+.1f}$ (interval pending)", src,
                                False))
                break
    # absolute values and times
    SG = data(PROG_V2)[0]
    SA, SP = data(MAIN_RUN)[0], data(PANEL_RUN)[0]
    camp = "base_summary.csv, metric seconds, column value"
    out += [("hlProgIrAbs", f"{SG[(PROG, 'ir')]:.2f}", f"results/{PROG_V2}/extra_summary.csv, {PROG}, ir", False),
            ("hlAreaIrAbs", f"{SA[('area', 'ir')]:.2f}", f"results/{MAIN_RUN}/extra_summary.csv, area, ir", False),
            ("hlCampArea", f"{SA[('area', 'seconds')]:.1f}", f"results/{MAIN_RUN}/{camp}, area", False),
            ("hlCampProgAreaX", f"{SA[('area', 'seconds')] / SG[(PROG, 'seconds')]:.1f}",
             f"results/{MAIN_RUN}/{camp} area divided by results/{PROG_V2}/{camp} {PROG}", False),
            ("hlCampProgHrditPct", f"{100 * SG[(PROG, 'seconds')] / SP[('hrdit_dev_author', 'seconds')]:.0f}",
             f"results/{PROG_V2}/{camp} {PROG} divided by results/{PANEL_RUN}/{camp} hrdit, percent", False)]
    if PROG in clean:
        src3 = f"results/{PROG_LAT}, column median_s, res 4096x4096"
        out += [("hlSecProg", f"{clean[PROG]:.1f}", f"{src3}, method {PROG}", False),
                ("hlSecHrditThree", f"{clean['hrdit_dev_author']:.1f}", f"{src3}, method hrdit_dev_author", False),
                ("hlSecAreaThree", f"{clean['area_s3']:.1f}", f"{src3}, method area", False),
                ("hlSpeedProgArea", f"{clean['area_s3'] / clean[PROG]:.1f}", f"{src3}: area / {PROG}", False),
                ("hlSpeedProgHrdit", f"{clean['hrdit_dev_author'] / clean[PROG]:.1f}",
                 f"{src3}: hrdit_dev_author / {PROG}", False),
                ("hlProgHrditPct", f"{100 * clean[PROG] / clean['hrdit_dev_author']:.0f}",
                 f"{src3}: {PROG} / hrdit_dev_author, percent", False)]
        check(clean[PROG] / clean["hrdit_dev_author"] < 0.5, "'less than half HRDiT's time' (clean, sweep 3)")
    ratio = SG[(PROG, "seconds")] / SP[("hrdit_dev_author", "seconds")]
    check(ratio < 0.5, f"'less than half HRDiT's time': progressive / HRDiT campaign time {ratio:.3f}")
    for m in ("area", "dype", "sega_official", "full"):
        check(len(wins[m]) == 10, f"text: progressive SCA beats {m} on all ten per-image metrics ({len(wins[m])})")
    return out


def headline(clean):
    out = []  # (name, value, source, pending)
    lat = V.landed(V.LATENCY)
    sca_run, sca_m, sca_pend = sca_shown(MAIN_RUN)
    S = data(sca_run)[0]
    SP = data(PANEL_RUN)[0]
    SM = data(MAIN_RUN)[0]
    vs = sca_vs_area(sca_run, sca_m)
    wins, losses, sgf, better, no_ci = area_counts(vs)
    vs15 = deltas_source(sca_run, sca_m, f"over the 15 metrics of Table app_4k "
                                  f"(no interval, counted unresolved: {', '.join(no_ci) or 'none'}):")
    lat_src = ("results/latency_curve_v2/summary.csv (collect_latency_v2.py), column median_seconds, res 4096x4096, "
               "method sca_v2" if lat else "results/latency_curve_summary.csv, column median_seconds, res 4096x4096, "
                                           "method sca_nb")
    area_src = "results/latency_curve_summary.csv, column median_seconds, res 4096x4096, method area"
    sca_clean = clean["sca_v2" if lat else REF]
    camp = "base_summary.csv, metric seconds, column value,"
    ir, fid = vs["ir"], vs["fid"]
    ir_src = deltas_source(sca_run, sca_m, "metric ir,")
    fid_src = deltas_source(sca_run, sca_m, "metric fid,")
    out += [
        ("hlSpeedArea", f"{clean['area'] / sca_clean:.2f}", f"{area_src} divided by {lat_src}", not lat),
        ("hlSecSca", f"{sca_clean:.1f}", lat_src, not lat),
        ("hlSecArea", f"{clean['area']:.1f}", area_src, False),
        ("hlSecHrdit", f"{clean['hrdit_dev_author']:.1f}", "results/prog_latency/latency_summary.csv (sweep 3) if "
                                                          "present, else results/latency_curve_batch1.csv median of 3 "
                                                          "(cold first run included)", False),
        ("hlWinsArea", str(wins), f"{vs15} count whose 95% interval excludes zero in SCA's favour", sca_pend),
        ("hlWinsAreaWord", word(wins, capital=True), f"{vs15} the same count as a capitalised word", sca_pend),
        ("hlMetricsArea", str(len(AREA_METRICS)), "the number of metrics of Table app_4k (fixed)", False),
        ("hlUnresolvedAreaWord", word(len(AREA_METRICS) - wins - losses),
         f"{vs15} count whose interval includes zero or is missing, as a word", sca_pend),
        ("hlLossesAreaWord", word(losses), f"{vs15} count significantly worse for SCA, as a word", sca_pend),
        ("hlIrDelta", f"{ir[0]:+.2f}", f"{ir_src} column value", sca_pend),
        ("hlIrLo", f"{ir[1]:.2f}", f"{ir_src} column ci_high, negated (lower end)", sca_pend),
        ("hlIrHi", f"{ir[2]:.2f}", f"{ir_src} column ci_low, negated (upper end)", sca_pend),
        ("hlFidDelta", f"{fid[0]:+.1f}", f"{fid_src} column value", sca_pend),
    ]
    if fid[1] is None:
        FAILS.append(f"hlFidLo/hlFidHi: no FID interval for {sca_m} minus AReA in {sca_run}")
        out += [("hlFidLo", "n/a", "no interval", sca_pend), ("hlFidHi", "n/a", "no interval", sca_pend)]
    else:
        out += [("hlFidLo", f"{fid[1]:.1f}", f"{fid_src} column ci_high, negated (lower end)", sca_pend),
                ("hlFidHi", f"{fid[2]:.1f}", f"{fid_src} column ci_low, negated (upper end)", sca_pend)]
    out += [
        ("hlIrScaAbs", f"{S[(sca_m, 'ir')]:.2f}", f"results/{sca_run}/extra_summary.csv, method {sca_m}, metric ir",
         sca_pend),
        ("hlFidScaAbs", f"{S[(sca_m, 'fid')]:.1f}",
         f"results/{sca_run}/{'corrected_fidp_ci' if data(sca_run)[3] else 'corrected_fidp'}/summary.csv, method "
         f"{sca_m}, metric fid, column value", sca_pend),
        ("hlCampSca", f"{S[(sca_m, 'seconds')]:.1f}", f"results/{sca_run}/{camp} method {sca_m}", sca_pend),
    ]
    # 2048 x 6144: wins against AReA over the same 15 metrics
    wide_run, wide_m, wide_pend = ((WIDE_V2, "sca_v2", False) if V.landed(WIDE_V2) else
                                   ("a4k195_2048x6144", REF, True))
    wvs = sca_vs_area(wide_run, wide_m)
    wwins, _, _, _, wno = area_counts(wvs)
    out.append(("hlWideSixWins", f"{word(wwins)} of {len(AREA_METRICS)}",
                deltas_source(wide_run, wide_m, f"over the 15 metrics (no interval: "
                                        f"{', '.join(wno) or 'none'}): count whose interval excludes zero in SCA's "
                                        "favour, as a word, 'of 15'"), wide_pend))
    # campaign times (shared nodes)
    prog_run, prog_m, prog_pend = prog_shown()
    SG = data(prog_run)[0]
    out += [
        ("hlCampProg", f"{SG[(prog_m, 'seconds')]:.1f}", f"results/{prog_run}/{camp} method {prog_m}", prog_pend),
        ("hlCampHrdit", f"{SP[('hrdit_dev_author', 'seconds')]:.1f}",
         f"results/{PANEL_RUN}/{camp} method hrdit_dev_author", False),
    ]
    sca_camp_run = sca_run if not sca_pend else PANEL_RUN
    out += [
        ("hlHrditTimeX", f"{SP[('hrdit_dev_author', 'seconds')] / data(sca_camp_run)[0][(sca_m, 'seconds')]:.1f}",
         f"results/{PANEL_RUN}/{camp} hrdit_dev_author, divided by results/{sca_camp_run}/{camp} {sca_m}", sca_pend),
        ("hlUltraTimeOver", str(math.floor(SM[("ultraimage", "seconds")] / S[(sca_m, "seconds")])),
         f"floor of results/{MAIN_RUN}/{camp} ultraimage, divided by results/{sca_run}/{camp} {sca_m}, printed as "
         "'over N times'", sca_pend),
    ]
    # Krea: per-image wins against AReA (single-pass SCA; the progressive sampler does not support Krea)
    krea_run, krea_m, krea_pend = ((KREA_V2, "sca_v2_krea", False) if V.landed(KREA_V2) else
                                   ("krea195", "sca_krea", True))
    krea, _ = V.paired_csv(f"{krea_run}/{krea_m}_vs_area.csv", krea_m, "area")
    out.append(("hlKreaWinsWord", word(sum(krea[m][1] > 0 for m in PER_IMAGE10)),
                f"results/{krea_run}/{krea_m}_vs_area.csv, column ci_low: count of the 10 per-image metrics with "
                "ci_low > 0, as a word", krea_pend))
    # progressive SCA against HRDiT (kept names), from the progressive run's own paired file
    prog_rel = f"{prog_run}/{prog_m}_vs_hrdit_dev_author.csv"
    pairs, _ = V.paired_csv(prog_rel, prog_m, "hrdit_dev_author")
    for key, name, sc in (("ir", "Ir", 1), ("hpsv21", "Hps", 100), ("pickscore", "Pick", 1)):
        d, lo, hi = (sc * x for x in pairs[key])
        times = " times 100" if sc != 1 else ""
        out += [(f"hlProgHrdit{name}", f"{d:+.2f}", f"results/{prog_rel}, metric {key}, column delta{times}", prog_pend),
                (f"hlProgHrdit{name}Lo", f"{lo:.2f}", f"results/{prog_rel}, metric {key}, column ci_low{times}",
                 prog_pend),
                (f"hlProgHrdit{name}Hi", f"{hi:.2f}", f"results/{prog_rel}, metric {key}, column ci_high{times}",
                 prog_pend)]
    out += [x for x in prog_macros(clean) if x[0] not in {o[0] for o in out}]
    # variance split (Section 3 and the introduction), results/heads_view/variance_split.csv (scripts/variance_split.py)
    out += variance_macros()
    out += overhead_macros()
    # qualitative claims printed next to these numbers
    check(losses == 0, f"'loses none': {sca_m} significantly worse than AReA on {losses} metrics")
    check(all(sgf[m] and better[m] for m in PER_IMAGE10), f"'every per-image metric': {sca_m} does not significantly "
          f"beat AReA on {[m for m in PER_IMAGE10 if not (sgf[m] and better[m])]}")
    third = sca_clean / clean["area"]
    check(abs(third - 1 / 3) <= 0.05, f"'about a third of AReA's time': {third:.3f}")
    return out


def variance_macros():
    """\\hlVarHead = 100 (1 - noise_within_head), \\hlVarStep = 100 noise_within_head, \\hlVarAvg = 100 head_avg_curve
    (FLUX.1-dev, 1024^2, post-gate bins, image share within 16 tokens), and the head shares of Krea and FLUX.2."""
    rel = "heads_view/variance_split.csv"
    R = {(r["checkpoint"], r["res"], r["sigma_range"], r["quantity"], r["radius"]): r for r in rows(rel)}
    out = []
    for name, ckpt, field in (("hlVarHead", "flux1_dev", "head"), ("hlVarStep", "flux1_dev", "noise"),
                              ("hlVarAvg", "flux1_dev", "avg"), ("hlVarHeadKrea", "flux1_krea_dev", "head"),
                              ("hlVarHeadFluxTwo", "flux2_dev", "head")):
        r = R[(ckpt, "1024x1024", "post_gate", "image_share", "16")]
        x = {"head": 1 - float(r["noise_within_head"]), "noise": float(r["noise_within_head"]),
             "avg": float(r["head_avg_curve"])}[field]
        out.append((name, f"{100 * x:.1f}", f"results/{rel}: {ckpt}, 1024x1024, post_gate, image_share, radius 16, "
                                             f"{field}", False))
    r = R[("flux1_dev", "1024x1024", "post_gate", "image_share", "16")]
    out += [("hlVarCurveLo", f"{float(r['curve_min']):.2f}", f"results/{rel}: flux1_dev post_gate r16 curve_min", False),
            ("hlVarCurveHi", f"{float(r['curve_max']):.2f}", f"results/{rel}: flux1_dev post_gate r16 curve_max", False)]
    return out


def latency_macros():
    """Section 4.4's scaling numbers: least-squares exponents b of clean seconds ~ N^b (N = (W/16)(H/16) image
    tokens, all five sizes of each method) for single-pass SCA (results/latency_curve_v2/summary.csv, sca_v2) and AReA
    (results/latency_curve_summary.csv), and both times at 8192^2 with their ratio. main.tex had these typed as
    \\providecommand defaults (0.99, 309.7, 3.07) and as literals (1.12, 952.0); these definitions replace them."""
    def tok(res):
        w, h = (int(v) for v in res.split("x"))
        return (w // 16) * (h // 16)

    def fit(rel, m):
        rs = [r for r in rows(rel) if r["method"] == m]
        check(len(rs) == 5, f"{rel} {m}: {len(rs)} sizes, expected 5")
        x = [math.log(tok(r["res"])) for r in rs]
        y = [math.log(float(r["median_seconds"])) for r in rs]
        mx, my = sum(x) / len(x), sum(y) / len(y)
        b = sum((u - mx) * (v - my) for u, v in zip(x, y)) / sum((u - mx) ** 2 for u in x)
        eight = next(float(r["median_seconds"]) for r in rs if r["res"] == "8192x8192")
        return b, eight

    v2, s1 = "latency_curve_v2/summary.csv", "latency_curve_summary.csv"
    bs, es = fit(v2, "sca_v2")
    ba, ea = fit(s1, "area")
    check(f"{bs:.2f}" == "0.99" and f"{ba:.2f}" == "1.12", f"latency exponents {bs:.3f} {ba:.3f} (paper: 0.99, 1.12)")
    src = "least squares of log median_seconds on log N over 2048^2 to 8192^2"
    return [("hlLatExpSca", f"{bs:.2f}", f"results/{v2}, method sca_v2: {src}", False),
            ("hlLatExpArea", f"{ba:.2f}", f"results/{s1}, method area: {src}", False),
            ("hlLatEightSca", f"{es:.1f}", f"results/{v2}, method sca_v2, res 8192x8192, median_seconds", False),
            ("hlLatEightArea", f"{ea:.1f}", f"results/{s1}, method area, res 8192x8192, median_seconds", False),
            ("hlLatEightRatio", f"{float(f'{ea:.1f}') / float(f'{es:.1f}'):.2f}",
             "the two printed 8192^2 times divided (AReA over single-pass SCA)", False)]


def request_macros():
    """Numbers the v6 text owner asked for (main.tex \\reqnum placeholders, .agent/notes/paper_v6/text.md), each read
    from its archived source. Each value is checked against the v5.1 value main.tex keeps as the placeholder (no
    number may change); a mismatch fails the run."""
    out = []

    def add(name, value, src):
        out.append((name, value, src, False))

    # emulation statistics (Section 3.2 and Figure 4): results/evidence/calib/k1_1024.json and k1_1280b.json
    k1 = json.load(open(RES / CALIB_JSON))
    k1280 = json.load(open(RES / "evidence/calib/k1_1280b.json"))
    for s, nm in (("2", "Two"), ("4", "Four")):
        add(f"hlEmuSplitHalf{nm}", f"{k1['scales'][s]['split_half_corr']['per_head']:.3f}",
            f"results/{CALIB_JSON}: scales/{s}/split_half_corr/per_head (tables fitted on odd and even prompts)")
        for key, rule in (("per_head", "Head"), ("per_layer", "Layer"), ("uniform", "Shared"), ("switch", "Switch")):
            add(f"hlEmuErr{rule}{nm}", f"{k1['scales'][s]['rel_err'][key]:.3f}",
                f"results/{CALIB_JSON}: scales/{s}/rel_err/{key} (relative output error at emulated ratio {s})")
    t = k1["transfer"]["1024x1024/s2 -> 1024x1024/s4"]
    add("hlEmuXferRatio", f"{t['rel_err']:.3f}", f"results/{CALIB_JSON}: transfer 's2 -> s4' rel_err (own "
                                                 f"{t['own']:.3f})")
    t = k1280["transfer"]["1024x1024/s2 -> 1280x1280/s2"]
    add("hlEmuXferRes", f"{t['rel_err']:.4f}", "results/evidence/calib/k1_1280b.json: transfer '1024x1024/s2 -> "
                                               "1280x1280/s2' rel_err")
    add("hlEmuOwnRes", f"{t['own']:.4f}", "the same entry, own (the 1280^2 table on 1280^2)")
    rho = [r for r in rows("heads_view/summary.csv") if (r["kind"], r["metric"], r["scale"], r["radius"], r["bin"])
           == ("pearson", "pearson_r", "2", "16", "8")]
    check(len(rho) == 1, "heads_view/summary.csv: one Pearson row at scale 2, radius 16, bin 8")
    add("hlEmuShareRho", f"{float(rho[0]['value']):.2f}", "results/heads_view/summary.csv: kind pearson, scale 2, "
                                                          "radius 16, bin 8 (fitted weight against near share, 1368 heads)")
    irs = [float(r["delta"]) for n in (4, 8, 16) for r in rows(f"e6_sca_cal{n}_deltas.csv") if r["metric"] == "ir"]
    check(len(irs) == 3, f"e6_sca_cal*_deltas.csv: {len(irs)} ImageReward rows")
    add("hlEmuBudgetIrMax", f"{max(abs(x) for x in irs):.3f}", "results/e6_sca_cal{4,8,16}_deltas.csv: largest |ir "
                                                                "delta| of the 4-, 8- and 16-prompt tables")
    # single-pass ablations, 46 prompts, variant minus SCA without the schedule
    for m, nm in (("abl_uniform_nb", "Shared"), ("abl_localonly_nb", "Local"), ("abl_mean_nb", "Mean"),
                  ("abl_v0", "NoOpen")):
        dd = ablation_deltas(m)
        add(f"hlAbl{nm}Ir", f"{dd['ir'][0]:.2f}", f"results/ablnb_{m}_deltas.csv, metric ir, delta")
        if nm != "Mean":
            add(f"hlAbl{nm}IrLo", f"{dd['ir'][1]:.2f}", f"results/ablnb_{m}_deltas.csv, metric ir, ci_low")
            add(f"hlAbl{nm}IrHi", f"{dd['ir'][2]:.2f}", f"results/ablnb_{m}_deltas.csv, metric ir, ci_high")
        if nm == "Shared":
            add("hlAblSharedMusiqc", f"{dd['musiq_crop1024'][0]:.1f}", f"results/ablnb_{m}_deltas.csv, musiq_crop1024")
    p46 = V.pair46(DEFAULT46)  # single-pass SCA minus SCA without the schedule, 46 prompts
    add("hlAblNoSchedMusiqc", f"{-p46['musiq_crop1024'][0]:+.1f}", f"eval/dup46_{DEFAULT46}/paired/{DEFAULT46}_vs_"
                                                                     "sca_nb.csv, musiq_crop1024, negated")
    # generality, sizes, latency
    add("hlFidAreaAbs", f"{data(MAIN_RUN)[0][('area', 'fid')]:.1f}", f"results/{MAIN_RUN}, corrected FID of AReA")
    gaps = []
    for r in rows("published_compare/compare.csv"):
        if (r["benchmark"], r["method"], r["ours_model"]) == ("Aesthetic-4K", "AReA", "FLUX.1-Krea-dev") and \
                r["metric"] in ("ir", "aesthetic", "clip", "pickscore", "hpsv2", "musiq"):
            gaps.append(abs(float(r["ours_minus_published"])) * (100 if r["metric"] == "hpsv2" else 1))
    check(len(gaps) >= 6, f"published_compare/compare.csv: {len(gaps)} AReA rows")
    add("hlAreaReproGap", f"{math.ceil(10 * max(gaps) - 1e-9) / 10:.1f}",
        "results/published_compare/compare.csv: largest |ours - published| of AReA on FLUX.1-Krea-dev over ImageReward, "
        "aesthetic, CLIP, PickScore, HPSv2 (x100), MUSIQ, rounded up to 0.1")
    kd, _ = V.paired_csv("krea195_sca_v2/sca_v2_krea_vs_area.csv", "sca_v2_krea", "area")
    add("hlKreaIr", f"{kd['ir'][0]:+.3f}", "results/krea195_sca_v2/sca_v2_krea_vs_area.csv, ir, delta")
    add("hlKreaIrLo", f"{kd['ir'][1]:.3f}", "the same, ci_low")
    add("hlKreaIrHi", f"{kd['ir'][2]:.3f}", "the same, ci_high")
    kp = [r for r in rows("krea195_sca_v2/corrected_fidp_ci/paired_deltas.csv")
          if (r["method"], r["reference_method"], r["metric"]) == ("area", "sca_v2_krea", "patch_fid")]
    check(len(kp) == 1 and kp[0]["bootstrap_valid"] == "1000", "krea195_sca_v2 corrected patch FID row")
    v, lo, hi = neg((float(kp[0]["value"]), float(kp[0]["ci_low"]), float(kp[0]["ci_high"])))
    add("hlKreaFidP", f"{v:+.1f}", "results/krea195_sca_v2/corrected_fidp_ci/paired_deltas.csv, patch_fid, area minus "
                                   "sca_v2_krea, negated")
    add("hlKreaFidPLo", f"{lo:.1f}", "the same, lower end")
    add("hlKreaFidPHi", f"{hi:.1f}", "the same, upper end")
    add("hlFluxTwoWinsWord", word(holm()["flux2"]["up_raw"]), "results/e8_flux2_sca_vs_full.csv: per-image metrics "
                                                               "(of ten) with ci_low > 0, as a word")
    wide_run = "a4k195_2048x4096_sca_v2" if V.landed("a4k195_2048x4096_sca_v2") else "a4k195_2048x4096"
    wvs = sca_vs_area(wide_run, "sca_v2" if wide_run.endswith("sca_v2") else REF)
    w4, l4, _, _, _ = area_counts(wvs)
    add("hlWideFourWinsWord", word(w4), f"results/{wide_run}: of the 15 metrics, significant wins over AReA")
    add("hlWideFourLossesWord", word(l4), f"results/{wide_run}: of the 15 metrics, significant losses")
    import gen_late_tables as late
    d6 = late.load_deltas("e2_6144", "sca_nb")
    spec6 = [(k, h) for k, _, _, _, h in late.PER_IMAGE + late.DIST if k != "seconds"]
    wins6 = sum((v6[1] > 0) if h else (v6[2] < 0) for k, h in spec6 for v6 in [d6[k]])
    add("hlSixKWins", str(wins6), f"results/e2_6144 (sca_nb minus area, 100 prompts): of {len(spec6)} metrics, "
                                  "intervals excluding zero in SCA's favour")
    e8 = {r["metric"]: r for r in rows("e2_8192_sca_nb_vs_area.csv")}
    add("hlEightKFavWord", word(sum(float(e8[k]["ci_low"]) > 0 for k in PER_IMAGE10)),
        "results/e2_8192_sca_nb_vs_area.csv (12 prompts): per-image metrics with ci_low > 0, as a word")
    # the worked cost example (Section 3.1, Table tab:flops): analytic attention FLOPs of one 4096^2 image with
    # FLUX.1-dev's 57 blocks, 24 heads, head dimension 128, 28 steps, T = 512 text tokens; single-pass SCA: the global
    # view (n0^2 block centres + T keys for all N + T queries) at every step, the local view (window^2 + T + 1,024
    # bridge keys for the N image queries) after the 8 opening steps, windows 128, 112, 96, 80, then 16 steps at 64
    n0, T, bridge, side = 64, 512, 1024, 256
    N = side * side
    per = 57 * 24 * 128 * 4  # blocks x heads x head dim x (QK^T + PV, 2 FLOP per MAC)
    full = 28 * per * (N + T) ** 2
    glob = 28 * per * (N + T) * (n0 * n0 + T)
    loc = sum(per * N * (w * w + T + bridge) for w in [128, 112, 96, 80] + [n0] * 16)
    check(f"{full:.2e}" == "8.56e+16" and f"{glob + loc:.2e}" == "1.24e+16",
          f"worked cost: {full:.3e} / {glob + loc:.3e} (Table tab:flops prints 8.56e16 / 1.24e16)")
    add("hlWorkedLocalKeys", f"{n0 * n0 + T + bridge:,}".replace(",", "{,}"),
        "n0^2 + T + 1,024: keys of one image query in the native local view (method constants)")
    add("hlWorkedPairsRatio", f"{full / (glob + loc):.1f}", "analytic attention FLOPs of full attention over single-pass "
                                                            "SCA at 4096^2 (see request_macros)")
    # progressive SCA minus UltraImage on MUSIQ over 1024^2 crops (text owner's optional request for Figure 6)
    pu, _, srcu = prog_pairs("ultraimage")
    v, lo, hi = pu["musiq_crop1024"]
    add("hlProgUltraMusiqc", f"{v:+.1f}", f"{srcu}, metric musiq_crop1024, delta")
    add("hlProgUltraMusiqcLo", f"{lo:.1f}", f"{srcu}, metric musiq_crop1024, ci_low")
    add("hlProgUltraMusiqcHi", f"{hi:.1f}", f"{srcu}, metric musiq_crop1024, ci_high")
    # every value must equal the v5.1 number main.tex keeps as its placeholder
    main_tex = (ROOT / "paper" / "main.tex").read_text()
    for name, value, _, _ in out:
        hit = re.search(r"\\providecommand\{\\" + name + r"\}\{\\reqnum\{" + name + r"\}\{([^}]*(?:\{,\}[^}]*)?)\}\}",
                        main_tex)
        if hit:
            check(hit.group(1).replace("{,}", ",") == value.replace("{,}", ","),
                  f"requested macro \\{name}: generated {value}, main.tex placeholder {hit.group(1)}")
    return out


def overhead_macros():
    """The time the schedule and the bridge add to single-pass SCA at 4096^2: sca_v2 against sca_nb timed back to back
    on one GPU (latency_curve_v2), and the operator without them in seconds."""
    lat = V.latency_v2()
    lv, lp = lat[(RES4K, "sca_v2")][0], lat[(RES4K, "sca_nb")][0]
    src = "results/latency_curve_v2/summary.csv (collect_latency_v2.py), column median_seconds, res 4096x4096"
    return [("hlTimeOverhead", f"{100 * (lv / lp - 1):.0f}\\%", f"{src}: sca_v2 / sca_nb - 1, in percent", False),
            ("hlSecScaNoSched", f"{lp:.1f}", f"{src}, method sca_nb", False)]


def main():
    print("final operator (sca_v2) runs:")
    for line in V.status_lines():
        print("  " + line)
    check_runs()
    clean = clean_seconds()
    OUT.mkdir(parents=True, exist_ok=True)
    files = {OUT / "tabstyle.tex": tabstyle_tex(),
             OUT / "tab_main.tex": table_main(clean), OUT / "tab_main_compact.tex": table_main_compact(clean),
             OUT / "tab_panel46.tex": table_panel(clean),
             OUT / "tab_ablation_compact.tex": table_ablation_v6(),
             OUT / "tab_ablation.tex": table_ablation(), OUT / "tab_generality.tex": table_generality_v6(),
             OUT_APP / "prog_all.tex": table_prog_all(), OUT_APP / "variance_split.tex": table_variance_split(),
             OUT_APP / "holm.tex": table_holm(), OUT_APP / "holm_short.tex": table_holm_short(),
             OUT_APP / "gen_all.tex": table_gen_all(),
             OUT_APP / "app_abl.tex": table_app_abl(), OUT_APP / "app_ps2.tex": table_app_ps2(),
             OUT_APP / "phase2_sensitivity.tex": table_phase2_sensitivity(),
             OUT_APP / "phase2_budget.tex": table_phase2_budget()}
    prog_abl = prog_ablation_rows()
    if prog_abl:
        # Section 4.3's sentences (\progAblResult) and the all-metric appendix table (tab:prog_abl_all)
        full = prog_ablation_full()
        files[OUT / "tab_ablation_compact_prog.tex"] = table_ablation_v6(full)
        files[OUT_APP / "prog_abl_all.tex"] = table_prog_abl_all(full)
        sentence = prog_ablation_sentence(full)
        lat = [float(r["median_s"]) for r in csv.DictReader(open(PROG_ABL_DIR / "latency.csv"))]
        check(len(lat) == len(full) + 1, f"prog195_ablation/latency.csv: {len(lat)} rows")
        seconds = f"{min(lat):.1f} to {max(lat):.1f}"  # clean batch-1 medians of the four runs
        files[OUT / "prog_abl_text.tex"] = "\n".join([
            HEADER, rf"\newcommand{{\progAblResult}}{{{sentence}}}",
            rf"\newcommand{{\progAblSeconds}}{{{seconds}}}"]) + "\n"
    print("progressive ablation: " + ("waiting for results/prog195_ablation/SHA256SUMS" if prog_abl is None
                                       else f"{len(prog_abl)} rows -> tab_ablation_compact_prog.tex"))
    macros = headline(clean) + holm_macros() + latency_macros() + request_macros()
    lines = [HEADER, "% Headline numbers the prose repeats; each prints exactly the literal used in the text.",
             "% Fallback grey for values of runs that have not landed; main.tex defines \\pend after this file.",
             r"\providecommand{\pend}[1]{\textcolor{gray}{#1}}",
             "% Caption notes: expand to the note while a table shows a grey value, to nothing once its runs land."]
    for name, key in (("pendTabMain", "tab_main"), ("pendTabPanel", "tab_panel46"),
                      ("pendTabAbl", "tab_ablation"), ("pendTabGen", "tab_generality")):
        lines.append(rf"\newcommand{{\{name}}}{{{V.NOTE if PENDING[key] and key != 'tab_generality' else ''}}}")
    seen = set()
    for name, value, source, grey in macros:
        assert name not in seen, name
        seen.add(name)
        lines += [f"% {c}" for c in textwrap.wrap(source + (" [grey: run pending]" if grey else ""), 112)]
        lines.append(rf"\newcommand{{\{name}}}{{{pend(value, grey)}}}")
    files[OUT / "headline.tex"] = "\n".join(lines) + "\n"
    for path, text in files.items():
        path.write_text(text)
        print(f"wrote {path}")
    print("caption notes: " + ", ".join(f"{k} {'pending' if v else 'empty'}" for k, v in PENDING.items()))
    for k, v in NO_CI.items():
        print(f"no interval ({k}): {', '.join(v)}; the caption must say so")
    print("headline: " + ", ".join(f"\\{n}={v}{' (grey)' if g else ''}" for n, v, _, g in macros))
    for note in NOTES:
        print("note: " + note)
    if FAILS:
        print("CHECKS FAILED (files written; update main.tex text or captions):", *FAILS, sep="\n  ")
        sys.exit(1)
    print("checks passed")


if __name__ == "__main__":
    main()
