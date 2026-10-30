"""Main-text Tables 1 to 4 and the headline numbers of the prose, generated from results/ (nothing typed by hand).

One command, run from the repository root, regenerates all five files:

    PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/work/09032/saini_2/vista/library/uv-cache uv run --no-project \\
        --python /work/09032/saini_2/vista/library/uv-envs/sca/bin/python python paper/figures/src/gen_main_tables.py

Outputs, in paper/figures/snippets/main_tables/:
  tab_main.tex, tab_panel46.tex, tab_ablation.tex, tab_generality.tex
      the \\setlength{\\tabcolsep} line and the tabular of Tables tab:main, tab:panel46, tab:ablation and
      tab:generality. The float, caption, label and font line stay in main.tex, which \\input's the file in place
      of the tabular. Each file ends in "\\end{tabular}%" so the \\input adds no extra space.
  headline.tex
      \\newcommand{\\hl...} macros for the numbers that the abstract, introduction, Sections 4 and 5 and the
      conclusion repeat, and the caption notes \\pendTabMain, \\pendTabPanel, \\pendTabAbl, \\pendTabGen; \\input it
      once in the preamble. A comment above each macro names its file and column.

The final operator. "SCA (ours)" is sca_v2: published SCA (sca_nb) plus the window schedule after the opening and
1,024 bridge tokens (scav2_runs.py). Its runs land one at a time; the script prints one status line per run. Until
a run lands, every cell and prose number it will replace shows the published operator's value as \\pend{value}
(grey; main.tex defines \\pend, headline.tex only provides a fallback), significance marks that will be recomputed
against it are grey, and the caption note of each table with a grey entry expands to scav2_runs.NOTE (empty once all
of that table's inputs have landed). Published SCA appears only in Table 3, as "without the schedule", and in the
rows that exist only for it (Table 2's other budgets, Table 4's sensitivity group), which are labelled so.
SCA_V2_PENDING=1 renders every queued run as pending.

Inputs: results/*.csv (the a4k195_* runs through data195.py, the sca_v2 runs through scav2_runs.py), the sca_v2
runs' eval directories on scratch for what their tasks do not copy (eval.py paired deltas, per-image CSVs), the
46-prompt runs' eval directories (eval/dup46_<label>), and one archived JSON for Table 3's Err. column, which no CSV
holds: results/evidence/calib/k1_1024.json, scales/4/rel_err.

Rules, all set below rather than typed into the cells:
  * numbers use Python's format() at each column's precision; a negative value is set in math mode;
  * Dup. (Tables 1 and 2): percent of the 95 prompts naming a person or animal with a duplicated one, rounded to an
    integer; Table 3: count of the 26 such prompts among the 46;
  * Table 1: bold = best and underline = second best of each column over all seven rows, ranked on the printed
    values (ties share the mark); Table 2: bold = best over its rows, no underline;
  * clean seconds: median of the batch-1 timings at 4096^2 where a method was timed, otherwise "n/m"; the final
    operator's from results/latency_curve_v2/summary.csv (collect_latency_v2.py);
  * marks, + significantly better, circ interval includes zero, none significantly worse, ast no interval exists:
    Table 1, SCA row only (TABLE1_MARKS), against AReA, from the same deltas as the headline count of wins (eval.py
    deltas, FID and patch FID corrected as data195.py describes); Table 2, against SCA at 28 steps (the final operator
    once landed: per-image metrics through scav2_runs.derived_pairs, no FID interval exists, so ast), except the
    budget rows, which are marked against the published operator from results/a4k195_panel as before;
  * Table 3: \\dn{} (orange) if the 95% interval lies below zero, \\up{} if above;
  * Table 4: signed paired differences; a cell is set in \\dn{} when its interval lies below zero;
  * shading: SCA in Tables 1 to 3 and progressive SCA in Table 2.
After writing, the script checks the prompt counts and the qualitative claims printed next to these numbers;
if one fails it lists it and exits with status 1 (the files are still written).
"""
from __future__ import annotations

import csv
import json
import math
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
CALIB_JSON = "evidence/calib/k1_1024.json"
RES4K = "4096x4096"
MAIN_RUN, PANEL_RUN, REF = "a4k195_main", "a4k195_panel", "sca_nb"
MAIN_V2, PROG_V2, WIDE_V2, KREA_V2 = "a4k195_main_sca_v2", "prog195_sca_v2", "a4k195_2048x6144_sca_v2", "krea195_sca_v2"
DEFAULT46 = "dup_i_shrink"  # the final operator on the 46-prompt subset
SHADE = r"\rowcolor[HTML]{EAF4F1}"
PLUS, CIRC, AST = r"\rlap{$^{+}$}", r"\rlap{$^{\circ}$}", r"\rlap{$^{\ast}$}"
TABLE1_MARKS = False  # marks against AReA on Table 1's SCA row (the caption must then explain +, circ and ast)
HEADER = "% Generated by paper/figures/src/gen_main_tables.py from results/; do not edit by hand."
FAILS: list[str] = []
NOTES: list[str] = []
PENDING: dict[str, bool] = {}
AST_USED: dict[str, int] = {}
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


def mark(v, lower):
    """+ significantly better, circ interval includes zero, none significantly worse, ast no interval."""
    _, lo, hi = v
    if lo is None or hi is None:
        return AST
    if lower:
        return PLUS if hi < 0 else ("" if lo > 0 else CIRC)
    return PLUS if lo > 0 else (CIRC if hi >= 0 else "")


# ---------------------------------------------------------------- who is shown -----------------------------------
def sca_shown(run_published):
    """(run, method, pending) of the SCA row: the final operator once a4k195_main_sca_v2 lands, else the published
    operator in `run_published` (a4k195_main for Table 1 and the headline, a4k195_panel for Table 2)."""
    return (MAIN_V2, "sca_v2", False) if V.landed(MAIN_V2) else (run_published, REF, True)


def prog_shown():
    return (PROG_V2, "sca_v2_prog", False) if V.landed(PROG_V2) else (PANEL_RUN, "sca_prog", True)


# ---------------------------------------------------------------- Tables 1 and 2 (4096^2, 195 prompts) ----------
# (column key, scale, decimals, lower is better); "clean" is the batch-1 latency, "seconds" the campaign mean
COLS = [("clean", 1, 1, True), ("seconds", 1, 1, True), ("ir", 1, 2, False), ("hpsv21", 100, 2, False),
        ("pickscore", 1, 2, False), ("clip", 1, 2, False), ("musiq_crop1024", 1, 1, False),
        ("fid", 1, 1, True), ("patch_fid", 1, 1, True), ("dup", 1, 0, True)]
QUALITY = ("ir", "hpsv21", "pickscore", "clip", "musiq_crop1024", "fid", "patch_fid")
PER_IMAGE_COLS = ("ir", "hpsv21", "pickscore", "clip", "musiq_crop1024")
DIST_COLS = ("fid", "patch_fid")
SCA_LABEL = r"\textbf{\sca{} (ours)}"
PROG_LABEL = r"\textbf{Prog.\ \sca{}}"  # "Progressive SCA" would widen Table 2 by 16 pt
# (group heading, [(kind, run, method, label, steps)]); kind "sca" and "prog" rows show the final operator once its
# run lands (run and method here are the published placeholders); "ctl" rows exist only for the published operator.
TAB_MAIN = [
    ("Single pass, shared sampler (28 steps, guidance 4.5)",
     [("pub", MAIN_RUN, "full", "Full attention", "28"), ("pub", MAIN_RUN, "dype", "DyPE", "28"),
      ("pub", MAIN_RUN, "sega_official", "SEGA", "28"), ("pub", MAIN_RUN, "area", "AReA", "28"),
      ("sca", MAIN_RUN, REF, SCA_LABEL, "28")]),
    ("Multi-stage, own recipe and budget",
     [("pub", MAIN_RUN, "ultraimage", "UltraImage", "30+30"),
      ("pub", PANEL_RUN, "hrdit_dev_author", "HRDiT", "30+17+10")]),
]
TAB_PANEL = [
    (r"Single pass, shared sampler (28 steps, guidance 4.5; other rows in Table~\ref{tab:main})",
     [("pub", PANEL_RUN, "sigma_dev_matched", "SigMa", "28"), ("sca", PANEL_RUN, REF, SCA_LABEL, "28")]),
    ("Multi-stage, own recipe and budget",
     [("pub", PANEL_RUN, "ultraimage", "UltraImage", "30+30"),
      ("pub", PANEL_RUN, "hrdit_dev_author", "HRDiT", "30+17+10"),
      ("prog", PANEL_RUN, "sca_prog", PROG_LABEL, "30+17+10")]),
    ("Without the schedule, other budgets",
     [("ctl", PANEL_RUN, "sca_s56", r"\quad 56 steps", "56"), ("ctl", PANEL_RUN, "sca_g7", r"\quad guidance 7", "28"),
      ("ctl", PANEL_RUN, "sca_s42", r"\quad 42 steps", "42")]),
]
HEAD12 = [r"\toprule",
          r" & & \multicolumn{2}{c}{Time (s)$\downarrow$} & Image$\uparrow$ & $\uparrow$ & Pick$\uparrow$ & $\uparrow$ & "
          r"$\uparrow$ & $\downarrow$ & $\downarrow$ & $\downarrow$ \\",
          r"\cmidrule(lr){3-4}",
          r"Method & Steps & clean & camp. & Reward & HPSv2.1 & Score & CLIP & MUSIQ$_\text{c}$ & FID & "
          r"FID$_\text{p}$ & Dup. \\"]
NCOL12 = 12


def clean_seconds():
    """{method: median clean batch-1 seconds per image at 4096^2 on one GH200}.

    results/latency_curve_summary.csv, median_seconds (the audited sweep: full attention, AReA, SCA; three timed
    prompts after a warm-up image), checked against its per-image rows in results/latency_curve.csv. HRDiT, which
    the sweep lacks: median of its hrdit_dev_author rows in results/latency_curve_batch1.csv, which has no HRDiT
    warm-up row, so its cold first run is included (as the Table 1 caption says). The final operator (sca_v2) and
    published SCA re-timed beside it: results/latency_curve_v2/summary.csv, once collect_latency_v2.py has written
    the 4096x4096 rows (key "sca_nb_v2" for the re-timed published operator)."""
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


def table12_rows(spec, clean):
    """Flat rows (kind, run, method, label, steps, pending, cells, cell_pending) of Table 1 or 2, unmarked."""
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
                    else:
                        cells.append(num(clean[m], d) if (kind == "pub" and m in clean) else "n/m")
                        cp.append(False)
                elif k == "dup":
                    kk, n = V.dup195(run, m)
                    check(n == 95, f"duplication rate of {m} in {run} over {n} prompts, not 95")
                    cells.append(f"{100 * kk / n:.0f}")
                    cp.append(pending)
                else:
                    cells.append(num(sc * S[(m, k)], d))
                    cp.append(pending)
            out.append((kind, run, m, label, steps, pending, cells, cp))
    return out


def sca_vs_area(run, m):
    """{metric: (delta, lo, hi)}: m minus AReA in `run` over AREA_METRICS, as the headline counts wins (the run's
    eval.py deltas negated, FID, KID and patch versions corrected as data195.load describes)."""
    S, D, ref, _, _ = data(run)
    vs = data195.versus(S, D, ref, run, "area", AREA_METRICS, RES)
    return {k: vs[(m, k)] for k in AREA_METRICS}


def table2_marks(row, sca_row):
    """{column: (mark, grey)} for a Table 2 row other than SCA: against SCA at 28 steps (the shown operator), or, for
    the budget rows, against the published operator. grey: the mark will be recomputed when SCA's run lands."""
    kind, run, m = row[0], row[1], row[2]
    ref_run, ref_m, ref_pending = (PANEL_RUN, REF, False) if kind == "ctl" else (sca_row[1], sca_row[2], sca_row[5])
    published_pair = run == PANEL_RUN and (ref_run, ref_m) == (PANEL_RUN, REF)
    if published_pair:
        P = data195.pairs(PANEL_RUN, m, REF, RES)
    else:
        P, n = V.derived_pairs(run, m, ref_run, ref_m)
        check(n == 195, f"Table 2 marks {m} vs {ref_m}: {n} paired prompts")
    out = {}
    for k in PER_IMAGE_COLS:
        out[k] = (mark(P[k], lower=False), ref_pending)
    for k in DIST_COLS:
        v = data(PANEL_RUN)[1][(m, k)] if published_pair else (None, None, None)
        assert v[1] is not None or not published_pair, (m, k)
        out[k] = (mark(v, lower=True), ref_pending)
    return out


def table12(name, spec, clean, underline):
    flat = table12_rows(spec, clean)
    cells = [list(r[6]) for r in flat]
    for j, (_, _, _, lower) in enumerate(COLS):
        for i, r in ranks([c[j] for c in cells], lower).items():
            if r == "best":
                cells[i][j] = rf"\textbf{{{cells[i][j]}}}"
            elif underline:
                cells[i][j] = rf"\underline{{{cells[i][j]}}}"
    sca_row = next(r for r in flat if r[0] == "sca")
    for i, row in enumerate(flat):
        marks = {}
        if name == "tab_main" and row[0] == "sca" and TABLE1_MARKS:
            vs = sca_vs_area(row[1], row[2])
            marks = {k: (mark(vs[k], lower=k in DIST_COLS), False) for k in QUALITY}
        elif name == "tab_panel46" and row[0] != "sca":
            marks = table2_marks(row, sca_row)
        for j, (k, *_) in enumerate(COLS):
            mk, grey = marks.get(k, ("", False))
            if mk == AST:
                AST_USED[name] = AST_USED.get(name, 0) + 1
            cells[i][j] = pend(cells[i][j] + pend(mk, grey and not row[7][j]), row[7][j])
    PENDING[name] = any(any(r[7]) for r in flat) or any(r"\pend" in c for c in sum(cells, []))
    body, i = [], 0
    for g, (heading, group) in enumerate(spec):
        body += ([r"\midrule"] if g else []) + [rf"\multicolumn{{{NCOL12}}}{{@{{}}l}}{{\emph{{{heading}}}}} \\"]
        for _ in group:
            shade = SHADE if flat[i][0] in ("sca", "prog") else ""
            body.append(shade + " & ".join([flat[i][3], flat[i][4]] + cells[i]) + r" \\")
            i += 1
    if name == "tab_main":
        # Table 1 caption: among the single-pass rows (first group), SCA prints the best value of every column but Dup.
        single = flat[:len(spec[0][1])]
        i = [r[0] for r in single].index("sca")
        lost = [COLS[j][0] for j in range(len(COLS)) if COLS[j][0] != "dup" and
                ranks([r[6][j] for r in single], COLS[j][3]).get(i) != "best"]
        check(not lost, f"Table 1 caption: SCA is not best among single-pass methods on {lost}")
    # 2.4 pt column gaps and 2.5 pt room for the marks (3.1 and 3 before the Dup. column) keep both tables
    # inside the 5.5 in text width at \footnotesize
    return tabular(r"\setlength{\tabcolsep}{2.4pt}", r"@{}lcrr*{7}{r<{\hspace{2.5pt}}}r@{}", HEAD12 + [r"\midrule"],
                   body, "  ")


# ---------------------------------------------------------------- Table 3 (ablations, 46 prompts) ----------------
ABL_COLS = [("ir", 1, 2), ("hpsv21", 100, 2), ("pickscore", 1, 2), ("musiq_crop1024", 1, 1)]
# Schedule block: paired change against the final operator (dup_i_shrink) on the same 46 prompts. Its variants of
# the queued 46-prompt runs appear once they land (no placeholder rows).
VARIANT_LABELS = {"dup_i_160": "first window 160", "dup_i_bridge32": "2,048 bridge tokens",
                  "dup_i_ext75": "five-step schedule"}
# Base-operator block: (label, method in results/ablnb_<method>_deltas.csv, k1_1024.json scales/4/rel_err key),
# paired change against the published operator (SCA without the schedule). Err. is the output error of the weight
# rule used after the opening phase: v0 keeps the per-head table and only drops the opening phase; the calibration
# report evaluates centre pooling only, so mean pooling has none (n/a). The schedule rows keep the per-head table.
ABL_ROWS = [("no opening (v0)", "abl_v0", "per_head"), ("local only", "abl_localonly_nb", "loc"),
            ("global only", "abl_globalonly", "glob"), (r"shared $\pi(\sigma)$", "abl_uniform_nb", "uniform"),
            (r"per-layer $\pi_\ell(\sigma)$", "abl_layer_nb", "per_layer"), ("mean pooling", "abl_mean_nb", None)]
ABL_HEAD_SCHED = "Schedule"
ABL_HEAD_BASE = r"Base operator (against \sca{} without the schedule)"


def ablation_deltas(method):
    out = {}
    for r in rows(f"ablnb_{method}_deltas.csv"):
        assert (r["method"], r["reference"], r["n_prompts"]) == (method, REF, "46"), r
        out[r["metric"]] = (float(r["delta"]), float(r["ci_low"]), float(r["ci_high"]))
    return out


def delta_cells(dd):
    cells = []
    for k, sc, d in ABL_COLS:
        v, lo, hi = dd[k]
        c = sgn(sc * v, d)
        cells.append(rf"\dn{{{c}}}" if hi < 0 else (rf"\up{{{c}}}" if lo > 0 else c))
    return cells


def dup_cell(method):
    k, n, _ = V.dup46(method)
    check(n == 26, f"Table 3 Dup.: {method} over {n} people/animal prompts, not 26")
    return str(k)


def table3():
    calib = json.load(open(RES / CALIB_JSON))
    assert calib["settings"]["pool"] == "center", calib["settings"]["pool"]
    err = calib["scales"]["4"]["rel_err"]
    S46 = V.summary46(DEFAULT46)
    sca = [f"{err['per_head']:.3f}"] + [num(sc * S46[k], d) for k, sc, d in ABL_COLS] + [dup_cell(DEFAULT46)]
    body = [SHADE + r"\sca{} (ours) & " + " & ".join(sca) + r" \\", r"\midrule",
            rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{ABL_HEAD_SCHED}}}}} \\"]
    p = V.pair46(DEFAULT46)
    sched = [("without the schedule", REF, {k: (-v, -hi, -lo) for k, (v, lo, hi) in p.items()})]
    for label, method, eval_dir in [("bridge only", "dup_bridge", "dup46_dup_bridge")] + \
            [(VARIANT_LABELS[v], v, f"dup46_{v}") for v in V.VARIANTS46 if V.landed(v)]:
        dd, n = V.derived_pairs(eval_dir, method, f"dup46_{DEFAULT46}", DEFAULT46)
        check(n == 46, f"Table 3 '{label}': {n} paired prompts, \\nabl is 46")
        sched.append((label, method, dd))
    for label, method, dd in sched:
        cells = [f"{err['per_head']:.3f}"] + delta_cells(dd) + [dup_cell(method)]
        body.append(" & ".join([label] + cells) + r" \\")
        better = [k for k, _, _ in ABL_COLS if dd[k][1] > 0]
        worse = [k for k, _, _ in ABL_COLS if dd[k][2] < 0]
        NOTES.append(f"Table 3 schedule row '{label}' against SCA: significantly better on {better or 'none'}, "
                     f"worse on {worse or 'none'}")
    body.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{ABL_HEAD_BASE}}}}} \\")
    for label, method, key in ABL_ROWS:
        dd = ablation_deltas(method)
        cells = [f"{err[key]:.3f}" if key else "n/a"] + delta_cells(dd) + [dup_cell(method)]
        for k, _, _ in ABL_COLS:
            check(dd[k][1] <= 0, f"Table 3: base-operator row '{label}' is significantly better on {k}")
        check(dd["hpsv21"][2] < 0 and dd["pickscore"][2] < 0,
              f"Table 3 caption: '{label}' does not significantly lower both HPSv2.1 and PickScore")
        body.append(" & ".join([label] + cells) + r" \\")
    PENDING["tab_ablation"] = False  # rows of runs not yet landed are left out, never shown grey
    # with the Dup. column the table needs \scriptsize to fit the 0.47\linewidth minipage (226 pt at \footnotesize)
    return tabular(r"\scriptsize\setlength{\tabcolsep}{1.1pt}", "@{}lrrrrrr@{}",
                   [r"\toprule", r"Variant & Err. & IR & HPS & Pick & M$_\text{c}$ & Dup. \\", r"\midrule"], body,
                   "  ")


# ---------------------------------------------------------------- Table 4 (generality and sensitivity) -----------
GEN_COLS = [("ir", 1, 3), ("hpsv21", 100, 2), ("musiq_crop1024", 1, 2)]
N_MACROS = {r"\nhrgen": 1000, r"\nseed": 195, r"\nabl": 46, r"\nkrea": 195}  # their values in main.tex
# Against AReA: (label, n cell, sca_v2 run, its final label, published CSV, published label); the published row is
# shown grey until the run lands. Published operator only: (label, n cell or None for the CSV's count, CSV, method,
# reference), unchanged.
GEN_AREA = [("HRGen-1K", r"\nhrgen", "hrgen1k_dev_sca_v2", "hrgen1k_dev/sca_nb_vs_area.csv", REF),
            ("Seed 1", r"\nseed", "a4k195_seed1_sca_v2", "a4k195_seed1/sca_nb_vs_area.csv", REF),
            ("Seed 2", r"\nseed", "a4k195_seed2_sca_v2", "a4k195_seed2/sca_nb_vs_area.csv", REF),
            ("FLUX.1-Krea", r"\nkrea", KREA_V2, "krea195/sca_krea_vs_area.csv", "sca_krea")]
GEN_PUB_HEAD = "Without the schedule (published operator)"
GEN_PUB = [("4 prompts", r"\nabl", "e6_sca_cal4_deltas.csv", "sca_cal4", REF),
           ("8 prompts", r"\nabl", "e6_sca_cal8_deltas.csv", "sca_cal8", REF),
           ("16 prompts", r"\nabl", "e6_sca_cal16_deltas.csv", "sca_cal16", REF),
           (r"$\sigma_g=0.85$", r"\nabl", "e7_sca_g85_deltas.csv", "sca_g85", REF),
           (r"$\sigma_g=0.93$", r"\nabl", "e7_sca_g93_deltas.csv", "sca_g93", REF),
           (r"FLUX.2 vs.\ full", None, "e8_flux2_sca_vs_full.csv", "flux2_sca", "flux2_full")]
BRACED_MINUS = {"HRGen-1K", "FLUX.1-Krea"}  # rows apply_krea_hrgen.py wrote: its minus is {-} (same glyph)
KEY_AREA_ROWS = ("HRGen-1K", "Seed 1", "Seed 2")  # the caption's key result: gains over AReA hold on these


def gen_area_pair(label, run, pub_csv, pub_m):
    """(deltas, n, pending, source) of an 'Against AReA' row."""
    if V.landed(run):
        rel = f"{run}/{V.RUNS[run].final}_vs_area.csv"
        dd, n = V.paired_csv(rel, V.RUNS[run].final, "area")
        return dd, n, False, rel
    dd, n = V.paired_csv(pub_csv, pub_m, "area")
    return dd, n, True, pub_csv


def table4_row(label, ncell, dd, n, grey):
    check(ncell is None or N_MACROS[ncell] == n, f"Table 4 '{label}': {n} prompts, {ncell} is "
                                                 f"{N_MACROS.get(ncell)} in main.tex")
    minus = "{-}" if label in BRACED_MINUS else "-"
    cells = [ncell or str(n)]
    for k, sc, d in GEN_COLS:
        c = sgn(sc * dd[k][0], d, minus)
        cells.append(pend(rf"\dn{{{c}}}" if dd[k][2] < 0 else c, grey))
    return " & ".join([label] + cells) + r" \\"


def table4():
    body = [r"\multicolumn{5}{@{}l}{\emph{Against AReA}} \\"]
    pending = False
    for label, ncell, run, pub_csv, pub_m in GEN_AREA:
        dd, n, grey, _ = gen_area_pair(label, run, pub_csv, pub_m)
        pending |= grey
        if label in KEY_AREA_ROWS:
            check(all(dd[k][1] > 0 for k, _, _ in GEN_COLS),
                  f"Table 4 caption: '{label}' is not significantly better than AReA on every column")
        body.append(table4_row(label, ncell, dd, n, grey))
    body.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{GEN_PUB_HEAD}}}}} \\")
    for label, ncell, rel, method, reference in GEN_PUB:
        dd, n = V.paired_csv(rel, method, reference)
        body.append(table4_row(label, ncell, dd, n, False))
    PENDING["tab_generality"] = pending
    return tabular(r"\setlength{\tabcolsep}{2.6pt}", "@{}lrrrr@{}",
                   [r"\toprule", r"Setting & $n$ & $\Delta$IR & $\Delta$HPS & $\Delta$M$_\text{c}$ \\", r"\midrule"],
                   body, "  ")


# ---------------------------------------------------------------- headline macros ---------------------------------
# The 15 metrics of the SCA-versus-AReA comparison (Table app_4k, Figure paired_area); the last four are lower-better.
AREA_METRICS = ["ir", "aesthetic", "clip", "pickscore", "hpsv2", "hpsv21", "musiq", "musiq_crop1024", "clipiqa",
                "arniqa", "is", "fid", "kid", "patch_fid", "patch_kid"]
LOWER = {"fid", "kid", "patch_fid", "patch_kid"}
PER_IMAGE10 = AREA_METRICS[:10]


def area_counts(vs):
    """(wins, losses, significant, better, no-interval metrics) of SCA minus AReA over AREA_METRICS."""
    sig = {m: vs[m][1] is not None and (vs[m][1] > 0 or vs[m][2] < 0) for m in AREA_METRICS}
    better = {m: (vs[m][0] < 0) == (m in LOWER) for m in AREA_METRICS}
    wins = sum(sig[m] and better[m] for m in AREA_METRICS)
    losses = sum(sig[m] and not better[m] for m in AREA_METRICS)
    return wins, losses, sig, better, [m for m in AREA_METRICS if vs[m][1] is None]


def deltas_source(run, m, metric_desc):
    """Provenance of `m` minus AReA in `run` (sca_vs_area): the files, and how the delta is read from them."""
    S, D, ref, corrected_ci, src = data(run)
    fid = (f"results/{run}/corrected_fidp_ci/paired_deltas.csv" if corrected_ci else
           f"results/{run}/corrected_fidp/paired_deltas.csv (points; whole-image FID and KID intervals from eval.py, "
           f"patch FID and KID without one)")
    how = (f"method {m} against area" if ref == "area" else
           f"method area against {ref}, negated to {m} minus AReA (columns value, ci_high, ci_low)")
    return f"{src} (1000 prompt draws) and {fid}, {how}, {metric_desc}"


def headline(clean):
    out = []  # (name, value, source, pending)
    lat = V.landed(V.LATENCY)
    sca_run, sca_m, sca_pend = sca_shown(MAIN_RUN)
    S = data(sca_run)[0]
    SP = data(PANEL_RUN)[0]
    SM = data(MAIN_RUN)[0]
    vs = sca_vs_area(sca_run, sca_m)
    wins, losses, sig, better, no_ci = area_counts(vs)
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
    # Krea: per-image wins against AReA
    krea_run, krea_m, krea_pend = ((KREA_V2, "sca_v2_krea", False) if V.landed(KREA_V2) else
                                   ("krea195", "sca_krea", True))
    krea, _ = V.paired_csv(f"{krea_run}/{krea_m}_vs_area.csv", krea_m, "area")
    out.append(("hlKreaWinsWord", word(sum(krea[m][1] > 0 for m in PER_IMAGE10)),
                f"results/{krea_run}/{krea_m}_vs_area.csv, column ci_low: count of the 10 per-image metrics with "
                "ci_low > 0, as a word", krea_pend))
    # progressive SCA against HRDiT
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
    out += duplication_macros()
    # qualitative claims printed next to these numbers (for whichever operator is shown)
    check(losses == 0, f"'loses none': {sca_m} significantly worse than AReA on {losses} metrics")
    check(all(sig[m] and better[m] for m in PER_IMAGE10), f"'every per-image metric': {sca_m} does not significantly "
          f"beat AReA on {[m for m in PER_IMAGE10 if not (sig[m] and better[m])]}")
    ratio = SG[(prog_m, "seconds")] / SP[("hrdit_dev_author", "seconds")]
    check(ratio < 0.5, f"'less than half the time': progressive / HRDiT campaign time {ratio:.3f}")
    third = sca_clean / clean["area"]
    check(abs(third - 1 / 3) <= 0.05, f"conclusion 'about a third of AReA's time': {third:.3f}")
    worse = [m for m in PER_IMAGE10 if pairs[m][2] < 0]
    check(not worse, f"progressive SCA significantly worse than HRDiT on {worse}")
    return out


def duplication_macros():
    """People/animal duplication (scripts/dup_detect.py and dup_rate.py at threshold 0.35) and the 46-prompt
    comparison of the final operator (dup_i_shrink) with published SCA (sca_nb) and AReA."""
    out = []
    rates = "results/dup_audit_20261027/rates_a4k195_people.csv, columns dup and n, 100 dup / n rounded,"
    run, m, grey = (MAIN_V2, "sca_v2", False) if V.landed(MAIN_V2) else (MAIN_RUN, REF, True)
    k, n = V.dup195(run, m)
    src = (f"results/{MAIN_V2}/dup_rates_people.csv, method sca_v2, columns dup and n, 100 dup / n rounded"
           if not grey else f"{rates} method sca_nb")
    out.append(("hlDupV", f"{100 * k / n:.0f}", src, grey))
    for name, method in (("hlDupPub", REF), ("hlDupArea", "area"), ("hlDupFullAttn", "full")):
        k, n = V.dup195(MAIN_RUN, method)
        out.append((name, f"{100 * k / n:.0f}", f"{rates} method {method}", False))
    out.append(("hlDupN", str(n), "results/dup_audit_20261027/rates_a4k195_people.csv, column n (prompts naming a "
                                  "person or animal, of 195)", False))
    for name, method in (("hlDupFsPub", REF), ("hlDupFsV", DEFAULT46), ("hlDupFsArea", "area"),
                         ("hlDupGateNinetyThree", "sca_g93"), ("hlDupGateEightyNine", REF),
                         ("hlDupGateEightyFive", "sca_g85"), ("hlDupGlobalOnly", "abl_globalonly")):
        k, n46, src = V.dup46(method)
        check(n46 == 26, f"{name}: {n46} people/animal prompts of the 46, not 26")
        out.append((name, str(k), f"{src}, method {method}: prompts with a duplicated person or animal, of {n46}",
                    False))
    out.append(("hlDupFsN", str(n46), "results/dup_audit_20261027/rates_46_all_people.csv, column n (prompts of the 46 "
                                      "naming a person or animal)", False))
    sec = V.seconds46(DEFAULT46)
    (tv, nv), (tp, np_) = sec[DEFAULT46], sec[REF]
    check(nv == np_ == 46, f"46-prompt seconds over {nv} and {np_} images")
    tsrc = f"/scratch/09032/saini_2/sca/eval/dup46_{DEFAULT46}/base/per_image.csv, column seconds, median over 46"
    out += [("hlTimeV", f"{tv:.1f}", f"{tsrc}, method {DEFAULT46} (the final operator)", False),
            ("hlTimePub", f"{tp:.1f}", f"{tsrc}, method sca_nb", False),
            ("hlTimeOverhead", f"{100 * (tv / tp - 1):.0f}\\%", f"{tsrc}: {DEFAULT46} / sca_nb - 1, in percent", False)]
    p = V.pair46(DEFAULT46)
    psrc = f"/scratch/09032/saini_2/sca/eval/dup46_{DEFAULT46}/paired/{DEFAULT46}_vs_sca_nb.csv (2000 draws),"
    for name, key, d in (("hlFsArniqa", "arniqa", 4), ("hlFsMusiqc", "musiq_crop1024", 2)):
        v, lo, hi = p[key]
        out += [(name, f"{v:+.{d}f}", f"{psrc} metric {key}, column delta", False),
                (f"{name}Lo", f"{lo:.{d}f}", f"{psrc} metric {key}, column ci_low", False),
                (f"{name}Hi", f"{hi:.{d}f}", f"{psrc} metric {key}, column ci_high", False)]
    for name, label in (("hlFsIrAreaPub", REF), ("hlFsIrAreaV", DEFAULT46)):
        v, lo, hi, src = V.vs_area46(label, "ir")
        out += [(name, f"{v:+.2f}", f"{src}; {label} minus AReA, delta", False),
                (f"{name}Lo", f"{lo:.2f}", f"{src}; {label} minus AReA, lower end", False),
                (f"{name}Hi", f"{hi:.2f}", f"{src}; {label} minus AReA, upper end", False)]
    return out


def main():
    print("final operator (sca_v2) runs:")
    for line in V.status_lines():
        print("  " + line)
    check_runs()
    clean = clean_seconds()
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"tab_main.tex": table12("tab_main", TAB_MAIN, clean, underline=True),
             "tab_panel46.tex": table12("tab_panel46", TAB_PANEL, clean, underline=False),
             "tab_ablation.tex": table3(), "tab_generality.tex": table4()}
    macros = headline(clean)
    lines = [HEADER, "% Headline numbers the prose repeats; each prints exactly the literal used in the text.",
             "% Fallback grey for values of runs that have not landed; main.tex defines \\pend after this file.",
             r"\providecommand{\pend}[1]{\textcolor{gray}{#1}}",
             "% Caption notes: expand to the note while a table shows a grey value, to nothing once its runs land."]
    for name, key in (("pendTabMain", "tab_main"), ("pendTabPanel", "tab_panel46"),
                      ("pendTabAbl", "tab_ablation"), ("pendTabGen", "tab_generality")):
        lines.append(rf"\newcommand{{\{name}}}{{{V.NOTE if PENDING[key] else ''}}}")
    for name, value, source, grey in macros:
        lines += [f"% {c}" for c in textwrap.wrap(source + (" [grey: run pending]" if grey else ""), 112)]
        lines.append(rf"\newcommand{{\{name}}}{{{pend(value, grey)}}}")
    files["headline.tex"] = "\n".join(lines) + "\n"
    for name, text in files.items():
        (OUT / name).write_text(text)
        print(f"wrote {OUT / name}")
    print("caption notes: " + ", ".join(f"{k} {'pending' if v else 'empty'}" for k, v in PENDING.items()))
    if AST_USED:
        print("ast (no interval) marks: " + ", ".join(f"{k} {v}" for k, v in AST_USED.items()) +
              "; the captions must say so")
    print("headline: " + ", ".join(f"\\{n}={v}{' (grey)' if g else ''}" for n, v, _, g in macros))
    for note in NOTES:
        print("note: " + note)
    if FAILS:
        print("CHECKS FAILED (files written; update main.tex text or captions):", *FAILS, sep="\n  ")
        sys.exit(1)
    print("checks passed: prompt counts, Table 1, 3 and 4 captions, 'loses none', 'every per-image metric', "
          "'less than half the time', 'about a third', progressive not worse than HRDiT")


if __name__ == "__main__":
    main()
