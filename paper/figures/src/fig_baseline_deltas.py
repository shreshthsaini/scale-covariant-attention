"""Paired scalar quality differences of the shared-protocol baselines against SCA (fig:baseline_quality).

Switch. While the final operator's 4096^2 run is pending (v2swap.landed("a4k195_main_sca_v2") is false) this
draws the published figure from the files below. Once that run lands, each difference is sca_v2 of that run minus
the baseline of results/a4k195_main, from v2swap.derived_pairs("a4k195_main_sca_v2", "sca_v2", "a4k195_main",
baseline): scripts/paired_bootstrap.py on the two runs' per-image eval CSVs, DRAWS prompt draws, seed 0, cached
under /scratch/09032/saini_2/sca/eval/app_tables_pairs/. The same checks run on it (rows of the pairs file, paired
mean equal to the difference of the two runs' summary means, the caption claims); a failed claim still stops the
script, because the caption snippet states it. Every run first prints one status line,
"fig_baseline_deltas.py: pending (a4k195_main_sca_v2)" or "fig_baseline_deltas.py: landed (a4k195_main_sca_v2)".
--no-results-csv skips writing results/baseline_quality_deltas.csv. With SCA_V2_MOCK=DIR (v2swap.py --make-mock)
the outputs go to DIR/figures/ as a trial run (snippet beside the figure, no results CSV), never into paper/.

All 195 Aesthetic-4K prompts at 4096^2, generation seed 0, read through data195.py from results/a4k195_main
(paths relative to the repository root):
  results/a4k195_main/<baseline>_vs_sca_nb.csv
      scripts/paired_bootstrap.py on per-image metrics, baseline minus SCA, DRAWS prompt draws, seed 0
      (data195.pairs). The figure plots SCA minus baseline: under the same prompt draws the percentile
      interval of the negated mean is the negated interval, so the sign reversal is exact.
  results/a4k195_main/{base,extra}_summary.csv
      point estimates (data195.load). Every paired mean difference must equal the difference of the two
      summary means over the same prompts; this check replaces the 191-prompt reproduction of the archived
      intervals from per-image scores, which the 195-prompt runs do not archive under results/.
Writes figures/baseline_deltas.pdf/.svg, figures/baseline_deltas_provenance.json (sources, checks and the
plotted statistics, also written to results/baseline_quality_deltas.csv) and figures/snippets/baseline_deltas.tex, whose
prompt count, draw count and claims are read from the same data; the script stops if a claim fails.

Run with the sca uv environment on a compute node. No generation or scoring is performed.
Whiskers quantify prompt sampling at one generation seed, not seed variability.
Thin light-grey gridlines run along the value (x) axis, behind the data (plot standard of
2026-10-19, which puts the gridlines of horizontal bar-like charts on the value axis).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data195  # noqa: E402  (results/a4k195_* loader)
import scafig as sf  # noqa: E402
import v2swap  # noqa: E402  (the sca_v2 switch: landed runs and their loaders)

ROOT = Path(__file__).resolve().parents[3]
SNIPPET = ROOT / "paper/figures/snippets/baseline_deltas.tex"
FIGDIR = v2swap.MOCK / "figures" if v2swap.MOCK else ROOT / "paper/figures"  # a mock run never writes into paper/
RUN, REF = "a4k195_main", "sca_nb"  # every baseline is paired with SCA (28 steps) inside one run
V2_RUN, V2_REF = "a4k195_main_sca_v2", "sca_v2"  # the final operator's run on the same prompts, once it lands
N_PROMPTS, RES, SEED = 195, "4096x4096", "0"
DRAWS = 2000  # scripts/paired_bootstrap.py default (--draws), stated in data195.pairs; checked below
METHODS = [("full", "Full attention"),
           ("area", "AReA"),
           ("dype", "DyPE"),
           ("sega_official", "SEGA"),
           ("ultraimage", "UltraImage†")]
SHARED = ("full", "area", "dype", "sega_official")  # shared sampler (28 steps, guidance 4.5)
METRICS = [("ir", "ImageReward", 1), ("pickscore", "PickScore", 1),
           ("hpsv21", "HPSv2.1 × 100", 100),
           ("musiq_crop1024", "Crop MUSIQ", 1)]


def paired_statistics(landed=False):
    if f"{DRAWS} draws" not in (data195.pairs.__doc__ or ""):
        raise ValueError(f"data195.pairs no longer documents {DRAWS} draws; update DRAWS and the caption")
    if data195.n_prompts(RUN) != {N_PROMPTS}:
        raise ValueError(f"results/{RUN}: prompt counts {sorted(data195.n_prompts(RUN))}, not {N_PROMPTS}")
    S, _, ref, _ = data195.load(RUN)
    if ref != REF:
        raise ValueError(f"results/{RUN}: deltas are against {ref}, not {REF}")
    sources = {}

    def track(name=None, path=None):
        path = data195.RESULTS / RUN / name if path is None else Path(path)
        try:
            key = str(path.relative_to(ROOT))
        except ValueError:  # a mock tree outside the repository
            key = str(path)
        sources[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    for name in ("base_summary.csv", "extra_summary.csv"):
        track(name)
    if landed:
        S2, _, ref2, _ = v2swap.load(V2_RUN)
        if ref2 != V2_REF:
            raise ValueError(f"{V2_RUN}: deltas are against {ref2}, not {V2_REF}")
        if data195.n_prompts(V2_RUN, v2swap.V2_RESULTS) != {N_PROMPTS}:
            raise ValueError(f"{V2_RUN}: prompt counts {sorted(data195.n_prompts(V2_RUN, v2swap.V2_RESULTS))}, "
                             f"not {N_PROMPTS}")
        for name in ("base_summary.csv", "extra_summary.csv"):
            track(path=v2swap.V2_RESULTS / V2_RUN / name)
    records, worst = [], 0.0
    for method, label in METHODS:
        if landed:  # sca_v2 of the new run minus the baseline of the published run, over the same prompts
            paired = v2swap.derived_pairs(V2_RUN, V2_REF, RUN, method, draws=DRAWS)
            path = track(path=v2swap.PAIR_CACHE / f"{V2_RUN}__{V2_REF}_vs_{RUN}__{method}.csv")  # its cache file
            sca_mean = {m: S2[(V2_REF, m)] for m, _, _ in METRICS}
        else:
            path = track(f"{method}_vs_{REF}.csv")
            sca_mean = {m: S[(REF, m)] for m, _, _ in METRICS}
        seen = set()
        for row in data195._rows(path):
            if (row["n_prompts"], row["res"], row["seed"]) != (str(N_PROMPTS), RES, SEED):
                raise ValueError(f"unexpected prompt count/resolution/seed in {path}: {row}")
            if row["metric"] in seen:
                raise ValueError(f"repeated metric {row['metric']} in {path}")
            seen.add(row["metric"])
        if not landed:
            paired = data195.pairs(RUN, method, REF)  # baseline minus SCA
        for metric, _, scale in METRICS:
            if metric not in paired:
                raise ValueError(f"no {metric} row in {path}")
            delta, lo, hi = paired[metric]
            if not all(np.isfinite([delta, lo, hi])) or not lo <= delta <= hi:
                raise ValueError(f"malformed interval: {method}/{metric}")
            # Over the same prompts the mean paired difference is the difference of the summary means.
            means = (sca_mean[metric] - S[(method, metric)]) if landed else (S[(method, metric)] - S[(REF, metric)])
            if not np.isclose(delta, means, rtol=1e-5, atol=1e-6):
                raise ValueError(f"paired mean {delta} and summary difference {means} disagree: {method}/{metric}")
            worst = max(worst, abs(delta - means) * scale)
            if not landed:  # baseline minus SCA, reversed to SCA minus baseline
                delta, lo, hi = -delta, -hi, -lo
            records.append(dict(baseline=method, baseline_label=label, metric=metric,
                                scale=scale, n_pairs=N_PROMPTS, sca_mean=sca_mean[metric] * scale,
                                baseline_mean=S[(method, metric)] * scale, delta=delta * scale,
                                ci_low=lo * scale, ci_high=hi * scale,
                                protocol="two_stage_30_steps_per_stage" if method == "ultraimage"
                                         else "shared_28_steps_guidance4.5"))
    provenance = dict(sources=sources, loader="paper/figures/src/data195.py", run=f"results/{RUN}",
                      difference="SCA minus named baseline", resolution=RES,
                      generation_seed_label=int(SEED), n_prompts=N_PROMPTS, bootstrap_seed=int(SEED),
                      bootstrap_draws=DRAWS, bootstrap_unit="matched prompt id",
                      interval="two-sided percentile 95%", multiplicity_adjusted=False,
                      seed_variability_estimated=False,
                      canonical_intervals=f"results/{RUN}/<baseline>_vs_{REF}.csv (baseline minus SCA), sign reversed",
                      consistency_check=dict(rule="paired mean difference equals the difference of the summary "
                                                  "means (data195.load), rtol 1e-5, atol 1e-6",
                                             max_abs_discrepancy_plotted_units=worst),
                      caveats=["One generation seed; intervals measure prompt sampling uncertainty only.",
                               "UltraImage uses a two-stage recipe, not matched compute or latent trajectories.",
                               "DyPE and SEGA use FLUX.1-dev, not their preferred Krea checkpoint."])
    if landed:
        provenance.update(
            loader="paper/figures/src/v2swap.py (load, derived_pairs) and data195.py",
            run=f"results/{V2_RUN} (sca_v2) and results/{RUN} (baselines)",
            difference="SCA final (sca_v2) minus named baseline",
            canonical_intervals=(f"v2swap.derived_pairs: scripts/paired_bootstrap.py over the per-image eval CSVs of "
                                 f"{V2_RUN} (sca_v2) and {RUN} (baseline), {DRAWS} draws, seed 0, cached in "
                                 f"{v2swap.PAIR_CACHE}"),
            consistency_check=dict(rule="paired mean difference equals sca_v2's summary mean (v2swap.load) minus the "
                                        "baseline's (data195.load), rtol 1e-5, atol 1e-6",
                                   max_abs_discrepancy_plotted_units=worst),
            state=f"landed: {V2_RUN}" + (f" (mock tree {v2swap.MOCK})" if v2swap.MOCK else ""))
    return records, provenance


def claims(records):
    """The caption's bold claim, re-checked on every run: SCA significantly better than every shared-protocol
    baseline on all four metrics; against UltraImage at least one significant gain and one significant loss."""
    shared = [r for r in records if r["baseline"] in SHARED]
    ultra = [r for r in records if r["baseline"] == "ultraimage"]
    if len(shared) != len(SHARED) * len(METRICS) or any(r["ci_low"] <= 0 for r in shared):
        raise ValueError("caption claim fails: SCA does not beat every shared-protocol baseline on all metrics")
    gains, losses = [r["metric"] for r in ultra if r["ci_low"] > 0], [r["metric"] for r in ultra if r["ci_high"] < 0]
    if not gains or not losses:
        raise ValueError("caption claim fails: the UltraImage comparison is not mixed")
    return dict(ultraimage_gains=gains, ultraimage_losses=losses,
                ultraimage_unresolved=[r["metric"] for r in ultra if r["ci_low"] <= 0 <= r["ci_high"]])


def write_snippet(path, landed=False):
    cap = (r"\textbf{SCA beats every shared-protocol baseline on all four metrics; against UltraImage the result "
           r"is mixed.}" "\n"
           rf"  SCA minus baseline, mean over the same {N_PROMPTS} prompts at $4096^2$, whiskers 95\% "
           rf"prompt-bootstrap intervals ({DRAWS} draws, one seed); open markers: interval contains zero; "
           r"orange: significant loss." "\n"
           r"  $^\dagger$Own two-stage recipe, not an equal-cost comparison.")
    origin = (f"results/{V2_RUN} and results/{RUN} (paired per-image differences through v2swap.derived_pairs, "
              "and summaries)" if landed else
              f"results/{RUN} (paired per-image differences and summaries, through data195.py)")
    path.write_text(f"% Generated by src/fig_baseline_deltas.py from {origin}.\n"
                    "\\begin{figure}[!htbp]\n  \\centering\n  \\includegraphics{figures/baseline_deltas.pdf}\n"
                    "  \\caption{" + cap + "}\n  \\label{fig:baseline_quality}\n\\end{figure}\n")
    return path


# Display names and label precision for the plot only (the CSV keeps METHODS/METRICS labels).
SHOW = {"ir": ("ImageReward", 2, 2), "pickscore": ("PickScore", 2, 4),  # name, label digits, tick step
        "hpsv21": (r"HPSv2.1 $(\times 100)$", 2, 20), "musiq_crop1024": ("crop MUSIQ", 1, 20)}
ROW = {"full": 4.4, "area": 3.4, "dype": 2.4, "sega_official": 1.4, "ultraimage": 0.0}
NAME = {"full": "Full attention", "area": "AReA", "dype": "DyPE", "sega_official": "SEGA",
        "ultraimage": "UltraImage\u2020"}
UNRESOLVED = "#8A939C"


def plot_deltas(records):
    """One row of four panels sharing the baseline axis; dot = mean paired difference, whisker = 95%."""
    W, H = sf.TEXT, 1.55  # the figure must not grow; the value labels and footer are gone
    left, gap, right, bottom, top = 0.72, 0.15, 0.05, 0.40, 0.08
    pw = (W - left - right - 3 * gap) / 4
    fig = plt.figure(figsize=(W, H))
    axes = []
    for i in range(4):
        rect = [(left + i * (pw + gap)) / W, bottom / H, pw / W, (H - bottom - top) / H]
        axes.append(fig.add_axes(rect, sharey=axes[0] if axes else None))
    for i, (ax, (metric, _, _)) in enumerate(zip(axes, METRICS)):
        name, digits, step = SHOW[metric]
        rows = [r for r in records if r["metric"] == metric]
        ax.axvline(0, color=sf.INK, lw=0.6, zorder=1)
        ax.axhline(0.7, color=sf.RULE, lw=0.5, zorder=1)
        # No per-dot values (house rule 9): the tables hold the numbers. Colour carries the verdict:
        # SCA green for a significant gain, LOSS orange for a significant loss, open grey otherwise.
        for r in rows:
            yy, d, lo, hi = ROW[r["baseline"]], r["delta"], r["ci_low"], r["ci_high"]
            col = sf.SCA if lo > 0 else sf.LOSS if hi < 0 else UNRESOLVED
            ax.errorbar(d, yy, xerr=[[d - lo], [hi - d]], fmt="o", ms=4.2, mew=0.9, mec=col,
                        mfc="white" if lo <= 0 <= hi else col, ecolor=col, elinewidth=1.0,
                        capsize=1.8, capthick=0.8, zorder=3)
        lo_x = min(0.0, min(r["ci_low"] for r in rows))
        hi_x = max(r["ci_high"] for r in rows)
        pad = 0.06 * (hi_x - lo_x)
        ax.set_xlim(lo_x - pad, hi_x + pad)
        ax.xaxis.set_major_locator(MultipleLocator(step))
        sf.value_grid(ax, "x")
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(rf"$\Delta$ {name}")
        if i == 0:
            ax.set_yticks(list(ROW.values()), [NAME[k] for k in ROW])
            ax.tick_params(axis="y", length=0, pad=3)
        else:
            ax.tick_params(axis="y", length=0, labelleft=False)
        ax.set_ylim(-0.6, 5.0)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=FIGDIR / "baseline_deltas")
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--no-results-csv", action="store_true",
                        help="do not write results/baseline_quality_deltas.csv")
    args = parser.parse_args()
    landed = v2swap.landed(V2_RUN)
    print(f"fig_baseline_deltas.py: {'landed' if landed else 'pending'} ({V2_RUN})"
          + (f" [mock {v2swap.MOCK}]" if v2swap.MOCK else ""), flush=True)
    trial = args.out.resolve() != (ROOT / "paper/figures/baseline_deltas").resolve()
    records, provenance = paired_statistics(landed)
    provenance["claims"] = claims(records)
    sf.use_style()
    plt.rcParams["svg.hashsalt"] = "sca-baseline-deltas"
    fig = plot_deltas(records)
    sf.save(fig, str(args.out))
    if args.preview:
        fig.savefig(args.preview, dpi=150, facecolor="white")
    snippet = write_snippet(args.out.with_name(args.out.name + "_snippet.tex") if trial else SNIPPET, landed)
    provenance["statistics"] = records
    if not trial and not args.no_results_csv:  # the plotted statistics as a results CSV, same columns as before
        import csv
        with open(ROOT / "results/baseline_quality_deltas.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader(); w.writerows(records)
    provenance["outputs"] = dict(pdf=f"{args.out}.pdf", svg=f"{args.out}.svg", snippet=str(snippet))
    if not trial:
        provenance["outputs"] = {k: str(Path(v).resolve().relative_to(ROOT)) for k, v in provenance["outputs"].items()}
    args.out.with_name(args.out.name + "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for row in records:
        print(f'{row["baseline"]:14s} {row["metric"]:18s} {row["delta"]:+.6f} '
              f'[{row["ci_low"]:+.6f}, {row["ci_high"]:+.6f}]')
    print("wrote", snippet)


if __name__ == "__main__":
    main()
