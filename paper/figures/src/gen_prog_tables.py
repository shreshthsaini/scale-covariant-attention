"""Progressive SCA numbers for the paper, generated from the archived CSVs (no hand-typed values).

Writes figures/snippets/prog_deltas.tex (Table tab:prog_deltas: paired per-image intervals for progressive SCA,
the 42-step control, HRDiT and 28-step SCA on the 46-prompt panel) and checks that the two Table 2 rows typed in
main.tex (42 steps, progressive) match results/progressive46 to the printed precision, including the
significance marks against 28-step SCA.

Inputs: results/progressive46/{base_summary,extra_summary,timing}.csv, the paired CSVs there, and
figures/src/data/progressive46/sca_s42_vs_sca_nb_paired.csv (computed with the same script and inputs; see the
provenance.json beside it). Run from anywhere: python gen_prog_tables.py [--check-only]
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
REPO = PAPER.parent
RES = REPO / "results" / "progressive46"
S42 = HERE / "data" / "progressive46" / "sca_s42_vs_sca_nb_paired.csv"
SNIPPET = PAPER / "figures" / "snippets" / "prog_deltas.tex"

# (csv metric, row label, scale, decimals), in the order of Table tab:phase2_budget
METRICS = [("ir", "ImageReward", 1, 3), ("aesthetic", "Aesthetic", 1, 3), ("clip", "CLIP score", 1, 2),
           ("pickscore", "PickScore", 1, 2), ("hpsv2", r"HPSv2 ($\times100$)", 100, 2),
           ("hpsv21", r"HPSv2.1 ($\times100$)", 100, 2), ("musiq", "MUSIQ", 1, 1),
           ("musiq_crop1024", r"MUSIQ$_\text{c}$", 1, 1), ("clipiqa", r"CLIP-IQA ($\times100$)", 100, 1),
           ("arniqa", r"ARNIQA ($\times100$)", 100, 1)]
# (file, method, reference, column head)
PAIRS = [(RES / "sca_prog_vs_hrdit_dev_author_paired.csv", "sca_prog", "hrdit_dev_author",
          r"Progressive vs.\ HRDiT"),
         (RES / "sca_prog_vs_sca_nb_paired.csv", "sca_prog", "sca_nb", r"Progressive vs.\ \sca{}"),
         (S42, "sca_s42", "sca_nb", r"42 steps vs.\ \sca{}"),
         (RES / "hrdit_dev_author_vs_sca_s42_paired.csv", "hrdit_dev_author", "sca_s42",
          r"HRDiT vs.\ 42 steps")]


def paired(path, method, reference):
    out = {}
    for r in csv.DictReader(open(path)):
        assert (r["method"], r["reference"], r["n_prompts"], r["res"]) == (method, reference, "46", "4096x4096"), r
        out[r["metric"]] = tuple(float(r[k]) for k in ("delta", "ci_low", "ci_high"))
    return out


def signed(x, d):
    s = f"{x:+.{d}f}"
    if float(s) == 0:
        s = ("-" if x < 0 else "+") + s[1:]
    return s.replace("-", "{-}") if s.startswith("-") else s


def cell(v, scale, d):
    delta, lo, hi = (x * scale for x in v)
    body = f"${signed(delta, d)}$"
    if lo > 0:
        body = r"\up{" + body + "}"
    elif hi < 0:
        body = r"\dn{" + body + "}"
    return body + r" {\tiny[$" + signed(lo, d) + "$, $" + signed(hi, d) + "$]}"


def mark(v):
    """Table 2 mark against 28-step SCA: + better, circ includes zero, none worse (higher is better)."""
    _, lo, hi = v
    return r"\rlap{$^{+}$}" if lo > 0 else (r"\rlap{$^{\circ}$}" if hi >= 0 else "")


def summaries():
    vals = {}
    for r in csv.DictReader(open(RES / "base_summary.csv")):
        vals[(r["method"], r["metric"])] = float(r["value"])
    for r in csv.DictReader(open(RES / "extra_summary.csv")):
        vals[(r["method"], r["metric"])] = float(r["value"])
    return vals


def dist_mark(method, metric):
    """Table 2 mark for a distribution metric (lower is better) against 28-step SCA, from the
    reference-cluster bootstrap in base_paired_deltas_distribution.csv."""
    for r in csv.DictReader(open(RES / "base_paired_deltas_distribution.csv")):
        if (r["method"], r["metric"], r["reference_method"]) == (method, metric, "sca_nb"):
            lo, hi = float(r["ci_low"]), float(r["ci_high"])
            return r"\rlap{$^{+}$}" if hi < 0 else ("" if lo > 0 else r"\rlap{$^{\circ}$}")
    return r"\rlap{$^{\tbd}$}"


def table2_rows(vals, prog_nb, s42_nb):
    def row(m, cmp, label, steps, camp):
        v = lambda k: vals[(m, k)]
        cells = [f"{v('ir'):.2f}" + mark(cmp["ir"]), f"{100 * v('hpsv21'):.2f}" + mark(cmp["hpsv21"]),
                 f"{v('pickscore'):.2f}" + mark(cmp["pickscore"]), f"{v('clip'):.2f}" + mark(cmp["clip"]),
                 f"{v('musiq_crop1024'):.1f}" + mark(cmp["musiq_crop1024"]),
                 f"{v('fid'):.1f}" + dist_mark(m, "fid"), f"{v('patch_fid'):.1f}" + dist_mark(m, "patch_fid")]
        return label, steps, camp, cells
    return [row("sca_s42", s42_nb, "42 steps", "42", f"{vals[('sca_s42', 'seconds')]:.1f}"),
            row("sca_prog", prog_nb, "progressive", "30+17+10", f"{vals[('sca_prog', 'seconds')]:.1f}")]


def check_main(rows):
    tex = (PAPER / "main.tex").read_text()
    for label, steps, camp, cells in rows:
        line = next(l for l in tex.splitlines() if re.search(r"\\quad " + label + r" &", l))
        got = [c.strip() for c in line.split("\\\\")[0].split("&")]
        strip = lambda s: re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
        want = [steps, "n/m", camp] + cells
        have = [strip(c) for c in got[1:]]
        assert have == want, f"Table 2 row '{label}' differs:\n have {have}\n want {want}"
        print(f"Table 2 row '{label}' matches the CSVs: {want}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    data = [paired(*p[:3]) for p in PAIRS]
    vals = summaries()
    check_main(table2_rows(vals, data[1], data[2]))
    t = {r["method"]: r for r in csv.DictReader(open(RES / "timing.csv"))}
    print("campaign mean s:", {k: round(float(v["mean_seconds"]), 1) for k, v in t.items()},
          "HRDiT / progressive:", round(float(t["hrdit_dev_author"]["mean_seconds"])
                                        / float(t["sca_prog"]["mean_seconds"]), 2))
    if a.check_only:
        return
    lines = [r"% Generated by src/gen_prog_tables.py from results/progressive46; do not edit by hand.",
             r"\begin{table}[!htbp]",
             r"\caption{\textbf{Against HRDiT, progressive \sca{} is not significantly worse on any per-image metric; "
             r"single-pass \sca{} with 42 steps is.} $4096^2$, 46 prompts, seed label 0. Per-image differences (first-named setting minus "
             r"second) with 95\% paired intervals over 2000 prompt draws; \up{blue} / \dn{orange}: the interval "
             r"excludes zero in favour of / against the first-named setting. Progressive: HRDiT's schedule with "
             r"\sca{} attention (Section~\ref{sec:progressive}); 42 steps: single-pass \sca{} sized to HRDiT's "
             r"campaign time. Intervals that include zero do not establish equivalence.}",
             r"\label{tab:prog_deltas}",
             r"\centering\scriptsize\setlength{\tabcolsep}{2.4pt}\let\tiny\scriptsize",
             r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
             r"\begin{tabular}{l" + "c" * len(PAIRS) + "}", r"\toprule",
             "Metric & " + " & ".join(p[3] for p in PAIRS) + r" \\", r"\midrule"]
    for key, label, scale, d in METRICS:
        lines.append(label + " & " + " & ".join(cell(x[key], scale, d) for x in data) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    SNIPPET.write_text("\n".join(lines))
    print("wrote", SNIPPET)


if __name__ == "__main__":
    main()
