"""Paired SCA minus AReA differences at 4096^2 for the 15 quality metrics of Table app_4k.

Data. All 195 Aesthetic-4K prompts, read through data195.py from results/a4k195_main (base and
extra summaries; eval.py paired deltas against SCA, negated to SCA minus AReA, which is exact for
the percentile bootstrap; FID, KID and their patch versions from corrected_fidp_ci, 1000 draws over
prompt ids and the 168 distinct reference images). Cells are formatted by gen_tables.py's own `f`,
`dcell` and `vcell`, loaded by AST (its table writer does not run). Every plotted interval is then
checked cell by cell against Table app_4k in paper/main.tex and against the main-text sentence
(count of significant metrics, ImageReward and FID quotes), so the figure cannot disagree with
either. No metric is computed here.

Common axis. x = 100 * sign * delta / mean_AReA, sign = -1 for the lower-is-better metrics
(FID, KID and their patch versions), so right always means SCA better. Dividing by a positive
constant keeps each interval on the same side of zero, so the significance calls are exactly
the table's. A standardized effect (delta / SD) is undefined for the five set-level metrics, and
delta / SE would hide effect size. ImageReward has no natural zero, so its percent is relative
to AReA's 0.917 and is the least comparable of the 15.

Regenerate: <sca env python> paper/figures/src/fig_paired_area.py
"""
from __future__ import annotations

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
import data195  # noqa: E402,F401  (results/a4k195_* loader)

ROOT = HERE.parents[2]
RESULTS = ROOT / "results"
MAIN = ROOT / "paper/main.tex"
OUT = ROOT / "paper/figures/paired_area"
RUN = "a4k195_main"  # results/<RUN>: all 195 prompts at 4096^2

W, H = 2.53, 2.40  # inches: 0.46 text width (182.8 TeX pt) by 173.4 TeX pt, built at printed size
UNRESOLVED = "#8A939C"  # the open grey of fig_baseline_deltas.py

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


def statistics():
    read: set = set()
    ns = table_logic(read)
    # The 195-prompt 4096^2 run, as in Table app_4k: SCA minus AReA.
    import data195
    s_nb, D, ref, corrected_ci = data195.load(RUN)
    assert ref == "sca_nb" and corrected_ci, "a4k195_main needs its corrected_fidp_ci intervals"
    assert data195.n_prompts(RUN) == {195}, "prompt count"
    metrics = [m for g in GROUPS for m, _ in g]
    d_nb = data195.versus(s_nb, D, ref, RUN, "area", metrics)
    for name in ("base_summary.csv", "extra_summary.csv", "base_paired_deltas.csv", "extra_paired_deltas.csv",
                 "corrected_fidp_ci/summary.csv", "corrected_fidp_ci/paired_deltas.csv"):
        read.add(f"{RUN}/{name}")
    area4 = {("area", m): v for (meth, m), v in s_nb.items() if meth == "area"}

    # Cell-by-cell check against the published table.
    tex = MAIN.read_text()
    block = tex[tex.index(r"\label{tab:app_4k}"):]
    block = block[:block.index(r"\end{tabular}")]
    lines = [ln.strip() for ln in block.splitlines()]
    records = []
    for metric, short in [m for g in GROUPS for m in g]:
        key, label, scale, digits, lower = ns["METD"][metric]
        delta, lo, hi = d_nb[("sca_nb", metric)]
        if lo is None or hi is None:
            raise ValueError(f"{metric}: no interval, cannot plot")
        cells = [label, ns["vcell"](area4, ("area", metric), metric),
                 ns["vcell"](s_nb, ("sca_nb", metric), metric), ns["dcell"]((delta, lo, hi), metric)]
        prefix = " & ".join(cells) + " & "
        if not any(ln.startswith(prefix) for ln in lines):
            raise AssertionError(f"Table app_4k disagrees for {metric}: expected row prefix {prefix!r}")
        mean = area4[("area", metric)]
        sign = -1.0 if lower else 1.0
        pct = [100.0 * sign * x / mean for x in (delta, lo, hi)]
        records.append(dict(
            metric=metric, label=short, table_label=label, lower_is_better=lower,
            area_mean=mean, sca_mean=s_nb[("sca_nb", metric)], delta=delta, ci_low=lo, ci_high=hi,
            gain_pct=pct[0], gain_pct_low=min(pct[1:]), gain_pct_high=max(pct[1:]),
            significant=bool(lo > 0 or hi < 0), better=bool((delta < 0) if lower else (delta > 0)),
            table_cell=cells[3]))

    # The main-text sentence: the count of significant metrics, ImageReward and FID as quoted, none worse.
    sig = [r for r in records if r["significant"]]
    assert len(records) == 15 and all(r["better"] for r in sig), "a significant metric is worse"
    words = {9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen", 15: "All 15"}
    by = {r["metric"]: r for r in records}
    ir, fid = by["ir"], by["fid"]
    sentence = re.search(words[len(sig)] + r" of 15 metrics improve significantly[^\n]*?unresolved", tex)
    assert sentence, f"main-text sentence with {words[len(sig)]} of 15 not found"
    for quote in (f'{ir["delta"]:+.2f}$ $[{ir["ci_low"]:.2f},{ir["ci_high"]:.2f}]',
                  f'{fid["delta"]:+.1f}$ $[{fid["ci_low"]:.1f},{fid["ci_high"]:.1f}]'):
        assert quote in sentence.group(0), f"main-text sentence lacks {quote}"
    figcap = re.search(r"\\textbf\{" + words[len(sig)] + r" of 15 metrics improve\.\}", tex)
    assert figcap, "figure caption count disagrees"

    sources = {str((RESULTS / n).relative_to(ROOT)): sha(RESULTS / n) for n in sorted(read)}
    sources["paper/figures/src/gen_tables.py"] = sha(HERE / "gen_tables.py")
    sources["paper/main.tex"] = sha(MAIN)
    provenance = dict(
        figure="paper/figures/paired_area.pdf", script="paper/figures/src/fig_paired_area.py",
        printed_size_in=[W, H], printed_size_texpt=[round(W * 72.27, 1), round(H * 72.27, 1)],
        comparison="SCA final (sca_nb) minus AReA (released code), 4096x4096, 195 prompts, seed label 0",
        interval="95% paired bootstrap, 1000 draws over prompt ids, as in Table app_4k (FID/KID and patch "
                 "versions: corrected crops, 1000 draws over prompt ids and the 168 distinct reference images, "
                 "results/a4k195_main/corrected_fidp_ci)",
        axis="gain = 100 * sign * (SCA - AReA) / mean_AReA; sign = -1 for FID, KID, FID_p, KID_p",
        significance="interval excludes zero (the table's rule); unchanged by the positive rescaling",
        checked_against=["paper/main.tex Table app_4k: AReA, SCA final and Delta [95% CI] cells of all 15 rows",
                         f"paper/main.tex Section 4.1 sentence: {len(sig)} of 15 significant; ImageReward "
                         f"{ir['delta']:+.2f} [{ir['ci_low']:.2f},{ir['ci_high']:.2f}]; FID {fid['delta']:+.1f} "
                         f"[{fid['ci_low']:.1f},{fid['ci_high']:.1f}]", "Figure caption: same count"],
        excluded=["Seconds per image (a cost, not a quality metric; not among the 15)"],
        caveats=["One generation seed: intervals cover prompt sampling only.",
                 "ImageReward has no natural zero; its percent is relative to AReA's mean of 0.917.",
                 "Intervals are not adjusted for multiplicity."],
        sources=sources, records=records)
    return records, provenance


def plot(records):
    sf.use_style()
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
        col = sf.SCA if r["significant"] else UNRESOLVED
        g, lo, hi = r["gain_pct"], r["gain_pct_low"], r["gain_pct_high"]
        ax.plot([lo, hi], [yy, yy], color=col, lw=1.1, solid_capstyle="butt", zorder=3)
        ax.plot([g], [yy], "o", ms=3.6, mew=0.9, mec=col, mfc=col if r["significant"] else "white", zorder=4)

    ax.set_yticks(ys, [r["label"] for r in records])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(ys[-1] - 0.7, ys[0] + 0.7)
    lo_x = min(r["gain_pct_low"] for r in records)
    hi_x = max(r["gain_pct_high"] for r in records)
    ax.set_xlim(min(-20.0, lo_x - 2), max(45.0, hi_x + 2))
    ax.xaxis.set_major_locator(FixedLocator([-20, 0, 20, 40]))
    ax.set_xticklabels(["−20", "0", "+20", "+40"])
    sf.value_grid(ax, "x")
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("SCA gain over AReA (% of AReA)")

    # Two direct labels replace a legend: one beside the first significant row, one beside the
    # first unresolved row, each in its own colour and in the empty space next to its dot.
    first_sig = next(i for i, r in enumerate(records) if r["significant"])
    first_unr = next(i for i, r in enumerate(records) if not r["significant"])
    rs, ru = records[first_sig], records[first_unr]
    ax.text(rs["gain_pct_low"] - 1.5, ys[first_sig], "significant", ha="right", va="center",
            fontsize=sf.FS_SMALL, color=sf.SCA)
    ax.text(ru["gain_pct_high"] + 1.5, ys[first_unr], "unresolved", ha="left", va="center",
            fontsize=sf.FS_SMALL, color=UNRESOLVED)
    return fig


def main():
    records, provenance = statistics()
    fig = plot(records)
    sf.save(fig, str(OUT), formats=("pdf",))
    OUT.with_name(OUT.name + "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for r in records:
        print(f'{r["metric"]:15s} delta {r["delta"]:+.6g} [{r["ci_low"]:+.6g}, {r["ci_high"]:+.6g}]  '
              f'gain {r["gain_pct"]:+6.2f}% [{r["gain_pct_low"]:+6.2f}, {r["gain_pct_high"]:+6.2f}]  '
              f'{"sig" if r["significant"] else "unresolved"}')


if __name__ == "__main__":
    main()
