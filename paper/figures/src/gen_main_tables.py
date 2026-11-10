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


# Compact main-text Table 1 (2026-11-10 rewrite): the columns that carry the argument (clean time, the three
# preference scores, crop-level detail, patch FID). Bold marks the best printed value within each group; no underline
# and no significance colour (the appendix Table tab:all_metrics and Figure paired_area carry the intervals). The full
# table (steps, campaign time, CLIP, FID) stays in tab_main.tex for the appendix.
COMPACT = ("clean", "ir", "hpsv21", "pickscore", "musiq_crop1024", "patch_fid")
HEAD_COMPACT = [r"\toprule",
                r"Method & Time (s)$\downarrow$ & IR$\uparrow$ & HPS$\uparrow$ & Pick$\uparrow$ & "
                r"M$_\text{c}\!\uparrow$ & FID$_\text{p}^{\dagger}\!\downarrow$ \\"]
GROUPS_COMPACT = ("Single pass, shared sampler (28 steps)", "Multi-stage, own recipe")


def table_main_compact(clean):
    flat = table_rows(TAB_MAIN, clean)
    idx = [j for j, c in enumerate(COLS) if c[0] in COMPACT]
    check([COLS[j][0] for j in idx] == list(COMPACT), "compact Table 1 column order")
    cells = [[r[6][j] for j in idx] for r in flat]
    k0 = 0
    for _, group in TAB_MAIN:
        a, b = k0, k0 + len(group)
        k0 = b
        for jj, j in enumerate(idx):
            for i, r in ranks([c[jj] for c in cells[a:b]], COLS[j][3]).items():
                if r == "best":
                    cells[a + i][jj] = rf"\textbf{{{cells[a + i][jj]}}}"
    for i, row in enumerate(flat):
        for jj, j in enumerate(idx):
            cells[i][jj] = "--" if cells[i][jj] == "n/m" else pend(cells[i][jj], row[7][j])
    body, i = [], 0
    ncol = 1 + len(COMPACT)
    for g, (_, group) in enumerate(TAB_MAIN):
        body += ([r"\midrule"] if g else []) + [rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\emph{{{GROUPS_COMPACT[g]}}}}} \\"]
        for _ in group:
            shade = SHADE if flat[i][0] in ("sca", "prog") else ""
            body.append(shade + " & ".join([flat[i][3]] + cells[i]) + r" \\")
            i += 1
    # caption claims of the compact table: progressive SCA prints the best time and HPS of the multi-stage group and
    # single-pass SCA every best value of the single-pass group
    single = flat[:len(TAB_MAIN[0][1])]
    s = [r[0] for r in single].index("sca")
    lost = [COLS[j][0] for j in idx if ranks([r[6][j] for r in single], COLS[j][3]).get(s) != "best"]
    check(not lost, f"compact Table 1: single-pass SCA is not best among single-pass rows on {lost}")
    return tabular(r"\setlength{\tabcolsep}{3.4pt}", r"@{}l*{6}{r}@{}", HEAD_COMPACT + [r"\midrule"], body, "  ")


def table_panel(clean):
    flat = table_rows(TAB_PANEL, clean)
    cells = [list(r[6]) for r in flat]
    rank_groups("tab_panel46", TAB_PANEL, cells, underline=False)
    sca_row = next(r for r in flat if r[0] == "sca")
    for i, row in enumerate(flat):
        marks = table2_marks(row, sca_row) if row[0] != "sca" else {}
        for j, (k, *_) in enumerate(COLS):
            colour, grey = marks.get(k, ("", False))
            painted = cells[i][j] if (grey or row[7][j]) else paint(cells[i][j], colour or "")
            cells[i][j] = pend(painted, row[7][j])
    PENDING["tab_panel46"] = any(any(r[7]) for r in flat) or any(r"\pend" in c for c in sum(cells, []))
    body, i = [], 0
    for g, (heading, group) in enumerate(TAB_PANEL):
        body += ([r"\midrule"] if g else []) + [rf"\multicolumn{{{NCOL11}}}{{@{{}}l}}{{\emph{{{heading}}}}} \\"]
        for _ in group:
            shade = SHADE if flat[i][0] == "sca" else ""
            body.append(shade + " & ".join([flat[i][3], flat[i][4]] + cells[i]) + r" \\")
            i += 1
    return tabular(r"\setlength{\tabcolsep}{3.1pt}", r"@{}lcrr*{7}{r}@{}", HEAD11 + [r"\midrule"], body, "  ")


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
PROG_ABL_LABELS = (("uniform", r"shared $\pi(\sigma)$"), ("shared", r"shared $\pi(\sigma)$"),
                   ("layer", r"per-layer $\pi_\ell(\sigma)$"), ("switch", "hard switch"), ("hard", "hard switch"))
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
                body.append(" & ".join([r"\quad " + label] + delta_cells(dd)) + r" \\")
    return tabular(r"\setlength{\tabcolsep}{2.4pt}", "@{}lrrrr@{}",
                   [r"\toprule", r"Variant & IR & HPS & Pick & M$_\text{c}$ \\", r"\midrule"], body, "  ")


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


def interval(lo, hi, d):
    return rf"$[{lo:.{d}f}, {hi:.{d}f}]$".replace("[-", "[{-}").replace(", -", ", {-}")


def table_prog_all():
    cols = []
    for m, _ in PROG_ALL_REFS:
        pi, n, _ = prog_pairs(m)
        sl, _ = prog_setlevel(m)
        cols.append((pi, sl))
    lines = []
    pending_refs = [name for (m, name), (_, sl) in zip(PROG_ALL_REFS, cols) if sl is None]
    metric_name = {"fid": "FID", "kid": "KID", "patch_fid": "patch FID", "patch_kid": "patch KID"}
    outside = [f"{name} {metric_name[key]}" for (m, name), (pi, sl) in zip(PROG_ALL_REFS, cols)
               for key in data195.DIST if sl and key in sl and not (sl[key][1] <= sl[key][0] <= sl[key][2])]
    for key, label, sc, d in PROG_ALL_METRICS:
        top, low = [label], [""]
        for (m, _), (pi, sl) in zip(PROG_ALL_REFS, cols):
            src = pi if key in pi else (sl or {})
            if key not in src:
                top.append(r"{\color{pendgrey}pending}")
                low.append("")
                continue
            v, lo, hi = src[key]
            top.append(paint(sgn(sc * v, d, "{-}"), sig(src[key], lower=key in data195.DIST) or ""))
            low.append(interval(sc * lo, sc * hi, d))
        lines += [" & ".join(top) + r" \\", " & ".join(low) + r" \\[0.5pt]"]
    lines[-1] = lines[-1].replace(r" \\[0.5pt]", r" \\")
    head = " & ".join(["Metric"] + [name for _, name in PROG_ALL_REFS]) + r" \\"
    note = (" Set-level intervals against " + ", ".join(pending_refs) + " are pending." if pending_refs else "")
    # key result, checked against the cells
    wins = {m: sum(pi[k][1] > 0 for k in PER_IMAGE10) for (m, _), (pi, _) in zip(PROG_ALL_REFS, cols)}
    sl_of = {m: sl for (m, _), (_, sl) in zip(PROG_ALL_REFS, cols)}
    check(all(wins[m] == 10 for m in ("area", "dype", "sega_official", "full")) and wins["ultraimage"] == 9,
          f"prog_all key result: per-image wins {wins}")
    worse_pfid = [m for m in ("area", "hrdit_dev_author", "sca_v2") if sl_of[m] and sl_of[m]["patch_fid"][1] > 0]
    check(worse_pfid == ["area", "hrdit_dev_author", "sca_v2"], f"prog_all key result: patch FID worse vs {worse_pfid}")
    key = (r" Key result: progressive \sca{} is significantly better than AReA, DyPE, SEGA, and full attention on every "
           r"per-image metric and than UltraImage on nine. Its differences from HRDiT on ImageReward, HPSv2.1, and "
           r"PickScore span zero, and its patch FID is higher than AReA's, HRDiT's, and single-pass \sca{}'s."
           if not pending_refs else "")
    if outside:  # a percentile interval of a biased statistic need not contain the point estimate
        note += (" The point estimate lies outside its percentile interval for " + ", ".join(outside) +
                 ", as it can for a biased statistic.")
    NOTES.append(f"prog_all: points outside their interval: {outside or 'none'}")
    cap = (r"\caption{\textbf{Progressive \sca{} against every $4096^2$ baseline.} FLUX.1-dev, \nmain{} Aesthetic-4K "
           r"prompts, seed label 0. Each cell is progressive \sca{} minus the column's method, with its 95\% paired "
           r"interval below it. Per-image metrics use 2000 prompt draws; FID, KID and their patch versions use 1000 "
           r"draws that also resample the 168 reference-image clusters. Single-pass: single-pass \sca{}. \up{Blue} / \dn{orange}: the interval excludes zero "
           r"in favour of / against progressive \sca{}; lower is better for FID and KID, so their signs read the other "
           r"way. Inception score has no paired interval (Table~\ref{tab:all_metrics})." + note + key + "}")
    text = "\n".join([
        HEADER.replace("% Generated", "% Generated (app_tables/prog_all.tex)"),
        r"\begin{table}[!htbp]", cap, r"\label{tab:prog_all}",
        r"\centering\scriptsize\setlength{\tabcolsep}{2.6pt}",
        r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
        r"\begin{tabular}{@{}l" + "r" * len(PROG_ALL_REFS) + "@{}}", r"\toprule", head, r"\midrule"]
        + lines + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]) + "\n"
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
           r"by all heads could follow. 256 random queries per block and step, full attention at each row's "
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
    files = {OUT / "tab_main.tex": table_main(clean), OUT / "tab_main_compact.tex": table_main_compact(clean),
             OUT / "tab_panel46.tex": table_panel(clean),
             OUT / "tab_ablation_compact.tex": table_ablation_compact(),
             OUT / "tab_ablation.tex": table_ablation(), OUT / "tab_generality.tex": table_generality(),
             OUT_APP / "prog_all.tex": table_prog_all(), OUT_APP / "variance_split.tex": table_variance_split()}
    prog_abl = prog_ablation_rows()
    if prog_abl:
        files[OUT / "tab_ablation_compact_prog.tex"] = table_ablation_compact(prog_abl)
        # one neutral sentence for Section 4.3 (\progAblResult): each variant's ImageReward change with its interval
        parts = [rf"{label} weights change ImageReward by {sgn(dd['ir'][0], 2)} "
                 rf"$[{dd['ir'][1]:.2f},{dd['ir'][2]:.2f}]$".replace("[-", "[{-}").replace(",-", ",{-}")
                 for label, dd, _ in prog_abl]
        sentence = ("In the progressive sampler, " + "; ".join(parts) +
                    r" against per-head weights (Table~\ref{tab:ablation}).")
        files[OUT / "prog_abl_text.tex"] = "\n".join([HEADER, rf"\newcommand{{\progAblResult}}{{{sentence}}}"]) + "\n"
    print("progressive ablation: " + ("waiting for results/prog195_ablation/SHA256SUMS" if prog_abl is None
                                       else f"{len(prog_abl)} rows -> tab_ablation_compact_prog.tex"))
    macros = headline(clean)
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
