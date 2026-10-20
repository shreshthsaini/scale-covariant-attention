"""Progressive SCA numbers for the paper, generated from the archived CSVs (no hand-typed values).

Writes figures/snippets/prog_deltas.tex (Table tab:prog_deltas: paired per-image intervals for progressive SCA,
the 42-step control, HRDiT and 28-step SCA on all 195 Aesthetic-4K prompts) and checks that the two Table 2 rows
typed in main.tex (42 steps, progressive) match results/a4k195_panel to the printed precision, including the
significance marks against 28-step SCA. The snippet is written first; a failed check then exits non-zero.

Inputs, all read through data195.py: results/a4k195_panel/{base,extra}_summary.csv and corrected_fidp (values,
seconds, and the FID / patch-FID deltas against sca_nb: data195.load), and the paired per-image files
results/a4k195_panel/<a>_vs_<b>.csv (data195.pairs; 2000 prompt draws, seed 0). Table 2's distribution marks
come only from the reference-cluster bootstrap, as the 46-prompt version took them from
base_paired_deltas_distribution.csv: results/a4k195_panel/corrected_fidp_ci (data195.load's corrected_ci).
Until it exists, FID and patch FID carry Table 2's asterisk (no paired interval); data195's fallback eval.py
interval for whole-image FID belongs to the uncorrected reference bank and gives no mark.
Run from anywhere: python gen_prog_tables.py [--check-only]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data195  # noqa: E402  (results/a4k195_* loader)

PAPER = HERE.parent.parent
REPO = PAPER.parent
RUN, REF = "a4k195_panel", "sca_nb"
N_PROMPTS, RES = 195, "4096x4096"
DRAWS = 2000  # scripts/paired_bootstrap.py default (--draws), stated in data195.pairs; checked in main()
SNIPPET = PAPER / "figures" / "snippets" / "prog_deltas.tex"

# (csv metric, row label, scale, decimals), in the order of Table tab:phase2_budget
METRICS = [("ir", "ImageReward", 1, 3), ("aesthetic", "Aesthetic", 1, 3), ("clip", "CLIP score", 1, 2),
           ("pickscore", "PickScore", 1, 2), ("hpsv2", r"HPSv2 ($\times100$)", 100, 2),
           ("hpsv21", r"HPSv2.1 ($\times100$)", 100, 2), ("musiq", "MUSIQ", 1, 1),
           ("musiq_crop1024", r"MUSIQ$_\text{c}$", 1, 1), ("clipiqa", r"CLIP-IQA ($\times100$)", 100, 1),
           ("arniqa", r"ARNIQA ($\times100$)", 100, 1)]
# (method, reference, column head); each pair is read from results/a4k195_panel/<method>_vs_<reference>.csv
PAIRS = [("sca_prog", "hrdit_dev_author", r"Progressive vs.\ HRDiT"),
         ("sca_prog", "sca_nb", r"Progressive vs.\ \sca{}"),
         ("sca_s42", "sca_nb", r"42 steps vs.\ \sca{}"),
         ("hrdit_dev_author", "sca_s42", r"HRDiT vs.\ 42 steps")]


def paired(method, reference):
    path = data195.RESULTS / RUN / f"{method}_vs_{reference}.csv"
    rows = data195._rows(path)
    if not rows:
        raise FileNotFoundError(path)
    for r in rows:
        assert (r["method"], r["reference"], r["n_prompts"], r["res"]) == (method, reference, str(N_PROMPTS), RES), r
    out = data195.pairs(RUN, method, reference)
    assert len(out) == len(rows) and all(k in out for k, *_ in METRICS), (path, sorted(out))
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
    """(S, D): values and deltas against 28-step SCA from data195.load, after checking the run's prompt count."""
    if data195.n_prompts(RUN) != {N_PROMPTS}:
        raise ValueError(f"results/{RUN}: prompt counts {sorted(data195.n_prompts(RUN))}, not {N_PROMPTS}")
    S, D, ref, corrected_ci = data195.load(RUN)
    if ref != REF:
        raise ValueError(f"results/{RUN}: deltas are against {ref}, not {REF}")
    print(f"results/{RUN}: corrected reference-cluster intervals {'used' if corrected_ci else 'not yet available'}")
    return S, D, corrected_ci


def dist_mark(D, corrected_ci, method, metric):
    """Table 2 mark for a distribution metric (lower is better) against 28-step SCA, from the reference-cluster
    bootstrap (corrected_fidp_ci, through data195.load): + significantly lower, none significantly higher, circ
    interval includes zero; ast no such interval yet; tbd no delta at all."""
    if (method, metric) not in D:
        return r"\rlap{$^{\tbd}$}"
    _, lo, hi = D[(method, metric)]
    if not corrected_ci or lo is None or hi is None:
        return r"\rlap{$^{\ast}$}"
    return r"\rlap{$^{+}$}" if hi < 0 else ("" if lo > 0 else r"\rlap{$^{\circ}$}")


def table2_rows(S, D, corrected_ci, prog_nb, s42_nb):
    def row(m, cmp, label, steps, camp):
        v = lambda k: S[(m, k)]
        cells = [f"{v('ir'):.2f}" + mark(cmp["ir"]), f"{100 * v('hpsv21'):.2f}" + mark(cmp["hpsv21"]),
                 f"{v('pickscore'):.2f}" + mark(cmp["pickscore"]), f"{v('clip'):.2f}" + mark(cmp["clip"]),
                 f"{v('musiq_crop1024'):.1f}" + mark(cmp["musiq_crop1024"]),
                 f"{v('fid'):.1f}" + dist_mark(D, corrected_ci, m, "fid"),
                 f"{v('patch_fid'):.1f}" + dist_mark(D, corrected_ci, m, "patch_fid")]
        return label, steps, camp, cells
    return [row("sca_s42", s42_nb, "42 steps", "42", f"{S[('sca_s42', 'seconds')]:.1f}"),
            row("sca_prog", prog_nb, "progressive", "30+17+10", f"{S[('sca_prog', 'seconds')]:.1f}")]


def check_main(rows):
    """Compare the Table 2 rows typed in main.tex with the CSVs; returns the failures (empty when they match)."""
    tex = (PAPER / "main.tex").read_text()
    failures = []
    for label, steps, camp, cells in rows:
        line = next((l for l in tex.splitlines() if re.search(r"\\quad " + label + r" &", l)), None)
        want = [steps, "n/m", camp] + cells
        if line is None:
            failures.append(f"Table 2 row '{label}' not found in main.tex; want {want}")
            continue
        got = [c.strip() for c in line.split("\\\\")[0].split("&")]
        strip = lambda s: re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
        have = [strip(c) for c in got[1:]]
        if have != want:
            failures.append(f"Table 2 row '{label}' differs:\n have {have}\n want {want}")
        else:
            print(f"Table 2 row '{label}' matches the CSVs: {want}")
    return failures


def check_caption(data):
    """The caption's bold claim: against HRDiT, progressive SCA has no significantly worse per-image metric, and
    single-pass SCA with 42 steps has at least one (HRDiT minus 42 steps significantly positive)."""
    prog_hrdit, hrdit_s42 = data[0], data[3]
    worse = [k for k, *_ in METRICS if prog_hrdit[k][2] < 0]
    s42_worse = [k for k, *_ in METRICS if hrdit_s42[k][1] > 0]
    if worse or not s42_worse:
        raise ValueError(f"caption claim fails: progressive worse on {worse}; 42 steps worse on {s42_worse}")
    return s42_worse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if f"{DRAWS} draws" not in (data195.pairs.__doc__ or ""):
        raise ValueError(f"data195.pairs no longer documents {DRAWS} draws; update DRAWS and the caption")
    data = [paired(*p[:2]) for p in PAIRS]
    S, D, corrected_ci = summaries()
    s42_worse = check_caption(data)
    print("42 steps significantly worse than HRDiT on:", ", ".join(s42_worse))
    secs = {m: S[(m, "seconds")] for m in ("sca_nb", "sca_s42", "sca_prog", "hrdit_dev_author")}
    print("campaign mean s:", {k: round(v, 1) for k, v in secs.items()},
          "HRDiT / progressive:", round(secs["hrdit_dev_author"] / secs["sca_prog"], 2))
    if not a.check_only:
        lines = [rf"% Generated by src/gen_prog_tables.py from results/{RUN} (through data195.py); do not edit by hand.",
                 r"\begin{table}[!htbp]",
                 r"\caption{\textbf{Against HRDiT, progressive \sca{} is not significantly worse on any per-image metric; "
                 rf"single-pass \sca{{}} with 42 steps is.}} $4096^2$, {N_PROMPTS} prompts, seed label 0. Per-image differences (first-named setting minus "
                 rf"second) with 95\% paired intervals over {DRAWS} prompt draws; \up{{blue}} / \dn{{orange}}: the interval "
                 r"excludes zero in favour of / against the first-named setting. Progressive: HRDiT's schedule with "
                 r"\sca{} attention (Section~\ref{sec:progressive}); 42 steps: single-pass \sca{} sized to HRDiT's "
                 r"campaign time. Intervals that include zero do not establish equivalence.}",
                 r"\label{tab:prog_deltas}",
                 r"\centering\scriptsize\setlength{\tabcolsep}{2.4pt}\let\tiny\scriptsize",
                 r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
                 r"\begin{tabular}{l" + "c" * len(PAIRS) + "}", r"\toprule",
                 "Metric & " + " & ".join(p[2] for p in PAIRS) + r" \\", r"\midrule"]
        for key, label, scale, d in METRICS:
            lines.append(label + " & " + " & ".join(cell(x[key], scale, d) for x in data) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
        SNIPPET.write_text("\n".join(lines))
        print("wrote", SNIPPET)
    failures = check_main(table2_rows(S, D, corrected_ci, data[1], data[2]))
    if failures:
        sys.exit("main.tex check FAILED (snippet " + ("not written" if a.check_only else "written") + "):\n"
                 + "\n".join(failures))
    print("main.tex check passed")


if __name__ == "__main__":
    main()
