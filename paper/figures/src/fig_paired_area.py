"""Paired SCA minus AReA differences at 4096^2 for the 15 quality metrics of Table app_4k.

Switch. While the final operator's 4096^2 run is pending (v2swap.landed("a4k195_main_sca_v2") is
false) this draws the published figure, SCA = sca_nb. Once that run lands it draws sca_v2 (published
SCA plus the window schedule and the 1,024 bridge tokens) minus AReA with the same construction.
Every run prints one status line first: "fig_paired_area.py: pending (a4k195_main_sca_v2)" or
"fig_paired_area.py: landed (a4k195_main_sca_v2)". With SCA_V2_MOCK=DIR (v2swap.py --make-mock) the
figure and its provenance go to DIR/figures/, never into paper/figures.

Data while pending. All 195 Aesthetic-4K prompts, read through data195.py from results/a4k195_main
(base and extra summaries; eval.py paired deltas against SCA, negated to SCA minus AReA, which is
exact for the percentile bootstrap; FID, KID and their patch versions from corrected_fidp_ci, 1000
draws over prompt ids and the 168 distinct reference images). Cells are formatted by gen_tables.py's
own `f`, `dcell` and `vcell`, loaded by AST (its table writer does not run). Every plotted interval
is then checked cell by cell against Table app_4k in paper/main.tex and against the main-text
sentence (count of significant metrics, count unresolved, ImageReward and FID quotes), so the figure
cannot disagree with either. No metric is computed here.

Data once landed. v2swap.load and v2swap.versus on a4k195_main_sca_v2: its results/ summaries, its
eval.py and eval_extra paired deltas against sca_v2 (1000 prompt draws, read from the run's scratch
eval directory, base/paired_deltas.csv and extra/extra_paired_deltas.csv) negated to sca_v2 minus
AReA, and FID, KID and their patch versions from corrected_fidp. Without a cluster bootstrap
(corrected_fidp_ci) FID and KID keep eval.py's reference-row interval around the corrected point and
patch FID and KID have none: they are drawn as open grey points without a whisker, labelled "no
interval", and are not counted as significant. A significant loss (none in the published run) is
drawn in the loss colour. The checks against Table app_4k (typed in main.tex or \\input from a
snippet) and the main-text sentence still run but print warnings instead of failing, because the
table and the prose switch through other generators and may lag; --strict makes them errors.

Common axis. x = 100 * sign * delta / mean_AReA, sign = -1 for the lower-is-better metrics
(FID, KID and their patch versions), so right always means SCA better. Dividing by a positive
constant keeps each interval on the same side of zero, so the significance calls are exactly
the table's. A standardized effect (delta / SD) is undefined for the five set-level metrics, and
delta / SE would hide effect size. ImageReward has no natural zero, so its percent is relative
to AReA's mean and is the least comparable of the 15.

Regenerate: <sca env python> paper/figures/src/fig_paired_area.py [--strict]
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-identical PDF on rerun
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scafig as sf  # noqa: E402
import data195  # noqa: E402  (results/a4k195_* loader)
import v2swap  # noqa: E402  (the sca_v2 switch: landed runs and their loaders)

ROOT = HERE.parents[2]
RESULTS = ROOT / "results"
MAIN = ROOT / "paper/main.tex"
HEADLINE = ROOT / "paper/figures/snippets/main_tables/headline.tex"  # gen_main_tables.py: prose numbers
FIGDIR = v2swap.MOCK / "figures" if v2swap.MOCK else ROOT / "paper/figures"  # a mock run never writes into paper/
OUT = FIGDIR / "paired_area"
RUN = "a4k195_main"  # results/<RUN>: all 195 prompts at 4096^2, published SCA (sca_nb)
V2_RUN = "a4k195_main_sca_v2"  # the same prompts and AReA images with the final operator (sca_v2)

W, H = 2.53, 2.40  # inches: 0.46 text width (182.8 TeX pt) by 173.4 TeX pt, built at printed size
UNRESOLVED = "#8A939C"  # the open grey of fig_baseline_deltas.py
# gen_main_tables.word: the number words the headline macros print
WORDS = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen"]

# Table app_4k order, three groups as in the Evaluation paragraph; short names as in the tables.
GROUPS = [
    [("ir", "ImageReward"), ("aesthetic", "Aesthetic"), ("clip", "CLIP score"),
     ("pickscore", "PickScore"), ("hpsv2", "HPSv2"), ("hpsv21", "HPSv2.1")],
    [("musiq", "MUSIQ"), ("musiq_crop1024", r"MUSIQ$_\mathsf{c}$"), ("clipiqa", "CLIP-IQA"),
     ("arniqa", "ARNIQA")],
    [("is", "Inception score"), ("fid", "FID"), ("kid", "KID"), ("patch_fid", r"FID$_\mathsf{p}$"),
     ("patch_kid", r"KID$_\mathsf{p}$")],
]
GROUP_GAP = 0.55  # extra row units between groups


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    """Repository-relative name where possible, else the absolute path."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def expand_headline(tex: str) -> str:
    """main.tex with the headline.tex macros (\\hlWinsAreaWord{} and so on) replaced by the literals they print.

    A value whose run is pending is wrapped in \\pend{...} (grey); the wrapper is dropped, the literal kept."""
    body = r"((?:[^{}]|\{[^{}]*\})*)"  # one level of nested braces, enough for \pend{value}
    macros = {name: re.sub(r"\\pend\{([^{}]*)\}", r"\1", value) for name, value in
              re.findall(r"\\newcommand\{\\(hl[A-Za-z]+)\}\{" + body + r"\}", HEADLINE.read_text())}
    return re.sub(r"\\(hl[A-Za-z]+)(?:\{\})?", lambda m: macros.get(m.group(1), m.group(0)), tex)


def table_logic(read: set) -> dict:
    """Namespace with gen_tables.py's own functions bound to RESULTS; records every file read."""
    tree = ast.parse((HERE / "gen_tables.py").read_text())
    names = {"rows", "summary", "deltas", "corrected_distributions", "f", "dcell", "vcell"}
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in ("MET", "METD") for t in n.targets))]
    ns = {"csv": csv, "Path": Path, "R": RESULTS}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(HERE / "gen_tables.py"), "exec"), ns)
    plain = ns["rows"]

    def rows(name):
        if (RESULTS / name).exists():
            read.add(name)
        return plain(name)

    ns["rows"] = rows  # the other functions look `rows` up in this namespace at call time
    return ns


def table_rows(tex: str) -> list:
    """Stripped lines of Table app_4k: the body typed in main.tex, or the snippet that body inputs."""
    block = tex[tex.index(r"\label{tab:app_4k}"):]
    block = block[:block.index(r"\end{table}")]
    for name in re.findall(r"\\input\{([^}]*)\}", block):
        path = ROOT / "paper" / (name if name.endswith(".tex") else name + ".tex")
        if path.exists():
            block += "\n" + path.read_text()
    return [unpend(ln.strip()) for ln in block.splitlines()]


def unpend(line: str) -> str:
    """The line with every \\pend{...} wrapper (grey pending value, gen_app_tables.py) replaced by its content."""
    out, i = [], 0
    while True:
        j = line.find(r"\pend{", i)
        if j < 0:
            return "".join(out) + line[i:]
        out.append(line[i:j])
        k, depth = j + len(r"\pend{"), 1
        start = k
        while depth:
            depth += {"{": 1, "}": -1}.get(line[k], 0)
            k += 1
        out.append(line[start:k - 1])
        i = k


def load_run(landed: bool):
    """(S, deltas, SCA label, corrected_ci, files) where deltas[(SCA label, metric)] = SCA minus AReA."""
    metrics = [m for g in GROUPS for m, _ in g]
    if not landed:  # The 195-prompt 4096^2 run, as in Table app_4k: SCA minus AReA.
        S, D, ref, corrected_ci = data195.load(RUN)
        assert ref == "sca_nb" and corrected_ci, "a4k195_main needs its corrected_fidp_ci intervals"
        assert data195.n_prompts(RUN) == {195}, "prompt count"
        return S, data195.versus(S, D, ref, RUN, "area", metrics), ref, corrected_ci, []
    S, D, ref, corrected_ci = v2swap.load(V2_RUN)
    assert ref == "sca_v2", f"{V2_RUN}: eval.py reference {ref}, expected sca_v2"
    assert data195.n_prompts(V2_RUN, v2swap.V2_RESULTS) == {195}, "prompt count"
    d, e = v2swap.V2_RESULTS / V2_RUN, v2swap.V2_EVAL / V2_RUN
    dist = d / ("corrected_fidp_ci" if corrected_ci else "corrected_fidp")
    files = [d / "base_summary.csv", d / "extra_summary.csv", e / "base" / "paired_deltas.csv",
             e / "extra" / "extra_paired_deltas.csv", dist / "summary.csv", dist / "paired_deltas.csv"]
    return S, v2swap.versus(S, D, ref, V2_RUN, "area", metrics), ref, corrected_ci, [p for p in files if p.exists()]


def quote(r: dict, digits: int) -> str:
    """A difference as the main-text sentence prints it: +0.31$ $[0.23,0.38] (the point alone without interval)."""
    q = f"{r['delta']:+.{digits}f}$"
    if r["ci_low"] is not None:
        q += f" $[{r['ci_low']:.{digits}f},{r['ci_high']:.{digits}f}]"
    return q


def span(r: dict, digits: int) -> str:
    if r["ci_low"] is None:
        return f"{r['delta']:+.{digits}f} (no interval)"
    return f"{r['delta']:+.{digits}f} [{r['ci_low']:.{digits}f},{r['ci_high']:.{digits}f}]"


def statistics(landed: bool = False, strict: bool = False):
    """Records of the 15 plotted differences, the provenance, and the check warnings (landed mode only)."""
    read: set = set()
    ns = table_logic(read)
    problems: list = []

    def check(ok, message):
        """Pending: the published figure must agree with main.tex. Landed: warn, unless --strict."""
        if ok:
            return
        if not landed or strict:
            raise AssertionError(message)
        problems.append(message)

    S, deltas, sca, corrected_ci, files = load_run(landed)
    if not landed:
        for name in ("base_summary.csv", "extra_summary.csv", "base_paired_deltas.csv", "extra_paired_deltas.csv",
                     "corrected_fidp_ci/summary.csv", "corrected_fidp_ci/paired_deltas.csv"):
            read.add(f"{RUN}/{name}")

    # Cell-by-cell check against the table.
    tex = MAIN.read_text()
    lines = table_rows(tex)
    records = []
    for metric, short in [m for g in GROUPS for m in g]:
        key, label, scale, digits, lower = ns["METD"][metric]
        delta, lo, hi = deltas[(sca, metric)]
        has_ci = lo is not None and hi is not None
        if not has_ci and not landed:
            raise ValueError(f"{metric}: no interval, cannot plot")
        cells = [label, ns["vcell"](S, ("area", metric), metric),
                 ns["vcell"](S, (sca, metric), metric), ns["dcell"]((delta, lo, hi), metric)]
        prefix = " & ".join(cells) + " & "
        check(any(ln.startswith(prefix) for ln in lines),
              f"Table app_4k disagrees for {metric}: expected row prefix {prefix!r}")
        mean = S[("area", metric)]
        sign = -1.0 if lower else 1.0
        pct = [100.0 * sign * x / mean for x in ((delta, lo, hi) if has_ci else (delta,))]
        record = dict(
            metric=metric, label=short, table_label=label, lower_is_better=lower,
            area_mean=mean, sca_mean=S[(sca, metric)], delta=delta, ci_low=lo, ci_high=hi,
            gain_pct=pct[0], gain_pct_low=min(pct[1:]) if has_ci else None,
            gain_pct_high=max(pct[1:]) if has_ci else None,
            significant=bool(has_ci and (lo > 0 or hi < 0)), better=bool((delta < 0) if lower else (delta > 0)),
            table_cell=cells[3])
        if landed:
            record["has_interval"] = has_ci
        records.append(record)

    # The main-text sentence: the count of significant metrics, the count unresolved, ImageReward and FID as
    # quoted, none worse (published run).
    sig = [r for r in records if r["significant"]]
    wins = [r for r in sig if r["better"]]
    unresolved = [r for r in records if not r["significant"]]
    assert len(records) == 15, "metric count"
    if not landed:
        assert len(wins) == len(sig), "a significant metric is worse"
    by = {r["metric"]: r for r in records}
    ir, fid = by["ir"], by["fid"]
    count = WORDS[len(wins)].capitalize()
    prose = expand_headline(tex)  # the main-text sentence and the caption print their numbers through macros
    sentence = re.search(count + r" of 15 metrics\b[^\n]*?\bunresolved\b", prose)
    check(sentence, f"main-text sentence with {count} of 15 not found")
    for q in (quote(ir, 2), quote(fid, 1), f"{WORDS[len(unresolved)]} are unresolved"):
        check(sentence and q in sentence.group(0), f"main-text sentence lacks {q}")
    figcap = re.search(r"\\textbf\{" + count + r" of 15 metrics improve\.\}", prose)
    check(figcap, "figure caption count disagrees")

    sources = {str((RESULTS / n).relative_to(ROOT)): sha(RESULTS / n) for n in sorted(read)}
    sources.update({rel(p): sha(p) for p in files})
    sources["paper/figures/src/gen_tables.py"] = sha(HERE / "gen_tables.py")
    sources["paper/main.tex"] = sha(MAIN)
    sources["paper/figures/snippets/main_tables/headline.tex"] = sha(HEADLINE)
    if landed:
        comparison = ("SCA final (sca_v2: published SCA, sca_nb, plus the window schedule after the opening and "
                      "1,024 bridge tokens) minus AReA (released code), 4096x4096, 195 prompts, seed label 0, "
                      f"run {V2_RUN}")
        interval = ("95% paired bootstrap, 1000 draws over prompt ids: eval.py and eval_extra paired deltas against "
                    "sca_v2 (base/paired_deltas.csv, extra/extra_paired_deltas.csv in the run's eval directory), "
                    "negated to SCA minus AReA; " +
                    ("FID/KID and patch versions: corrected crops, 1000 draws over prompt ids and the distinct "
                     "reference images (corrected_fidp_ci)" if corrected_ci else
                     "FID and KID: corrected-crop point with eval.py's reference-row bootstrap interval; FID_p and "
                     "KID_p: corrected-crop point without interval (the run has no corrected_fidp_ci)"))
    else:
        comparison = "SCA final (sca_nb) minus AReA (released code), 4096x4096, 195 prompts, seed label 0"
        interval = ("95% paired bootstrap, 1000 draws over prompt ids, as in Table app_4k (FID/KID and patch "
                    "versions: corrected crops, 1000 draws over prompt ids and the 168 distinct reference images, "
                    "results/a4k195_main/corrected_fidp_ci)")
    provenance = dict(
        figure=rel(OUT.with_suffix(".pdf")), script="paper/figures/src/fig_paired_area.py",
        printed_size_in=[W, H], printed_size_texpt=[round(W * 72.27, 1), round(H * 72.27, 1)],
        comparison=comparison,
        interval=interval,
        axis="gain = 100 * sign * (SCA - AReA) / mean_AReA; sign = -1 for FID, KID, FID_p, KID_p",
        significance="interval excludes zero (the table's rule); unchanged by the positive rescaling",
        checked_against=["paper/main.tex Table app_4k: AReA, SCA final and Delta [95% CI] cells of all 15 rows",
                         f"paper/main.tex Section 4.1 sentence: {len(wins)} of 15 significant, "
                         f"{len(unresolved)} unresolved; ImageReward {span(ir, 2)}; FID {span(fid, 1)}",
                         "Figure caption: same count"],
        excluded=["Seconds per image (a cost, not a quality metric; not among the 15)"],
        caveats=["One generation seed: intervals cover prompt sampling only.",
                 f"ImageReward has no natural zero; its percent is relative to AReA's mean of {ir['area_mean']:.3f}.",
                 "Intervals are not adjusted for multiplicity."],
        sources=sources, records=records)
    if landed:
        provenance.update(
            state=f"landed: {V2_RUN}" + (f" (mock tree {v2swap.MOCK})" if v2swap.MOCK else ""),
            no_interval=[r["metric"] for r in records if r["ci_low"] is None],
            significant_losses=[r["metric"] for r in sig if not r["better"]],
            check_warnings=problems)
    return records, provenance, problems


def plot(records):
    sf.use_style()
    sf.script_floor(sf.FS_TICK)  # MUSIQ_c, FID_p, KID_p: 7 pt tick labels, subscripts at 6 pt or more
    plt.rcParams["svg.hashsalt"] = "sca-paired-area"
    fig = plt.figure(figsize=(W, H))
    left, right, bottom, top = 0.80, 0.07, 0.40, 0.05
    ax = fig.add_axes([left / W, bottom / H, (W - left - right) / W, (H - bottom - top) / H])

    ys, y = [], 0.0
    for gi, group in enumerate(GROUPS):
        for _ in group:
            ys.append(y)
            y -= 1.0
        y -= GROUP_GAP
    ax.axvline(0, color=sf.INK, lw=0.6, zorder=2)
    for r, yy in zip(records, ys):
        col = (sf.SCA if r["better"] else sf.LOSS) if r["significant"] else UNRESOLVED
        g, lo, hi = r["gain_pct"], r["gain_pct_low"], r["gain_pct_high"]
        if lo is not None:  # a difference without an interval is a point without a whisker
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.1, solid_capstyle="butt", zorder=3)
        ax.plot([g], [yy], "o", ms=3.6, mew=0.9, mec=col, mfc=col if r["significant"] else "white", zorder=4)

    ax.set_yticks(ys, [r["label"] for r in records])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(ys[-1] - 0.7, ys[0] + 0.7)
    lo_x = min(r["gain_pct"] if r["gain_pct_low"] is None else r["gain_pct_low"] for r in records)
    hi_x = max(r["gain_pct"] if r["gain_pct_high"] is None else r["gain_pct_high"] for r in records)
    ax.set_xlim(min(-20.0, lo_x - 2), max(45.0, hi_x + 2))
    ax.xaxis.set_major_locator(FixedLocator([-20, 0, 20, 40]))
    ax.set_xticklabels(["−20", "0", "+20", "+40"])
    sf.value_grid(ax, "x")
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("SCA gain over AReA (% of AReA)")

    # Direct labels replace a legend: one beside the first significant row, one beside the first
    # unresolved row, each in its own colour and in the empty space next to its dot; once landed, also
    # one beside the first point without an interval and the first significant loss, if any.
    def first(test):
        return next((i for i, r in enumerate(records) if test(r)), None)

    i = first(lambda r: r["significant"] and r["better"])
    if i is not None:
        ax.text(records[i]["gain_pct_low"] - 1.5, ys[i], "significant", ha="right", va="center",
                fontsize=sf.FS_SMALL, color=sf.SCA)
    i = first(lambda r: not r["significant"] and r["gain_pct_low"] is not None)
    if i is not None:
        ax.text(records[i]["gain_pct_high"] + 1.5, ys[i], "unresolved", ha="left", va="center",
                fontsize=sf.FS_SMALL, color=UNRESOLVED)
    i = first(lambda r: r["gain_pct_low"] is None)
    if i is not None:
        ax.text(records[i]["gain_pct"] + 2.5, ys[i], "no interval", ha="left", va="center",
                fontsize=sf.FS_SMALL, color=UNRESOLVED)
    i = first(lambda r: r["significant"] and not r["better"])
    if i is not None:
        ax.text(records[i]["gain_pct_low"] - 1.5, ys[i], "worse", ha="right", va="center",
                fontsize=sf.FS_SMALL, color=sf.LOSS)
    return fig


def main():
    ap = argparse.ArgumentParser(description="Figure fig:paired_area (SCA minus AReA at 4096^2).")
    ap.add_argument("--strict", action="store_true",
                    help="once landed, a mismatch with Table app_4k or the main-text sentence is an error")
    args = ap.parse_args()
    landed = v2swap.landed(V2_RUN)
    print(f"fig_paired_area.py: {'landed' if landed else 'pending'} ({V2_RUN})"
          + (f" [mock {v2swap.MOCK}]" if v2swap.MOCK else ""), flush=True)
    records, provenance, problems = statistics(landed, args.strict)
    fig = plot(records)
    sf.save(fig, str(OUT), formats=("pdf",))
    OUT.with_name(OUT.name + "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for r in records:
        if r["ci_low"] is None:
            ci, gain, state = "[no interval]", "[no interval]", "no interval"
        else:
            ci = f'[{r["ci_low"]:+.6g}, {r["ci_high"]:+.6g}]'
            gain = f'[{r["gain_pct_low"]:+6.2f}, {r["gain_pct_high"]:+6.2f}]'
            state = ("sig" if r["better"] else "sig worse") if r["significant"] else "unresolved"
        print(f'{r["metric"]:15s} delta {r["delta"]:+.6g} {ci}  gain {r["gain_pct"]:+6.2f}% {gain}  {state}')
    for p in problems:
        print(f"WARNING (main.tex not yet switched?): {p}")


if __name__ == "__main__":
    main()
