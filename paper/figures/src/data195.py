"""The 195-prompt Aesthetic-4K evaluations (results/a4k195_*) in gen_tables.py's dictionary format.

Every run directory written by the z96x evaluation tasks holds
  base_summary.csv, extra_summary.csv        point estimates (method, metric, value, n_prompts, ...)
  base_paired_deltas.csv, extra_paired_deltas.csv
                                             eval.py paired deltas, method minus sca_nb, 1000 prompt draws
  <a>_vs_<b>.csv                             scripts/paired_bootstrap.py on per-image metrics, a minus b,
                                             2000 prompt draws, seed 0
  corrected_fidp/{summary,paired_deltas}.csv FID, KID and their patch versions on the corrected crop bank
  corrected_fidp_ci/{summary,paired_deltas}.csv
                                             the same with 1000 draws over prompt ids and the 168 distinct
                                             reference images (hold_cpu_finalize/<run>_bootstrap.sh)
As gen_tables.corrected_distributions does for the 191-prompt files, corrected values replace the eval.py
ones for fid, kid, patch_fid and patch_kid. Without corrected_fidp_ci, whole-image FID and KID keep the
eval.py interval and the patch versions keep none.
"""
import csv
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[3] / "results"
DIST = ("fid", "kid", "patch_fid", "patch_kid")


def _rows(path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def load(run, results=RESULTS):
    """(S, D, ref, corrected_ci): S[(method, metric)] -> value; D[(method, metric)] -> (delta, lo, hi), method minus
    the run's reference method `ref` (sca_nb for the main and panel runs, area for the non-square ones), with
    lo = hi = None where no interval exists; corrected_ci says whether corrected_fidp_ci was used."""
    d = results / run
    if not (d / "base_summary.csv").exists():
        raise FileNotFoundError(d / "base_summary.csv")
    S, D, refs = {}, {}, set()
    for r in _rows(d / "base_summary.csv") + _rows(d / "extra_summary.csv"):
        S[(r["method"], r["metric"])] = float(r["value"])
    for r in _rows(d / "base_paired_deltas.csv") + _rows(d / "extra_paired_deltas.csv"):
        refs.add(r["reference_method"])
        D[(r["method"], r["metric"])] = (float(r["value"]), float(r["ci_low"]), float(r["ci_high"]))
    if len(refs) != 1:
        raise ValueError(f"{run}: references {refs}")
    ref = refs.pop()
    ci_dir = d / "corrected_fidp_ci"
    corrected_ci = (ci_dir / "paired_deltas.csv").exists()
    src = ci_dir if corrected_ci else d / "corrected_fidp"
    for r in _rows(src / "summary.csv"):
        if r["metric"] in DIST:
            S[(r["method"], r["metric"])] = float(r["value"])
    for r in _rows(src / "paired_deltas.csv"):
        if r["metric"] not in DIST:
            continue
        if r["reference_method"] != ref:
            raise ValueError(f"{run}: corrected reference {r['reference_method']}, eval.py reference {ref}")
        key = (r["method"], r["metric"])
        if r["ci_low"] and r["ci_high"]:
            D[key] = (float(r["value"]), float(r["ci_low"]), float(r["ci_high"]))
        elif r["metric"] in ("fid", "kid") and key in D:
            D[key] = (float(r["value"]), D[key][1], D[key][2])  # historical eval.py interval, corrected point
        else:
            D[key] = (float(r["value"]), None, None)
    for key in [k for k in D if k[1] in ("patch_fid", "patch_kid")]:
        if not corrected_ci and D[key][1] is not None:
            D[key] = (D[key][0], None, None)  # patch intervals only from the corrected cluster bootstrap
    return S, D, ref, corrected_ci


def pairs(run, a, b, results=RESULTS):
    """{metric: (delta, lo, hi)} from results/<run>/<a>_vs_<b>.csv (per-image metrics, a minus b, 2000 draws)."""
    out = {}
    for r in _rows(results / run / f"{a}_vs_{b}.csv"):
        if r["method"] != a or r["reference"] != b:
            raise ValueError(f"{run}/{a}_vs_{b}.csv holds {r['method']} vs {r['reference']}")
        out[r["metric"]] = (float(r["delta"]), float(r["ci_low"]), float(r["ci_high"]))
    return out


def versus(S, D, run_ref, run, target, metrics, results=RESULTS):
    """Deltas against `target`, keyed (method, metric), from a run whose deltas D are against `run_ref`.

    If target is run_ref, D itself. Otherwise run_ref minus target negates D[(target, metric)]: under the same
    prompt draws the percentile interval of the negated statistic is the negated interval, so this is exact. Any
    other method m uses <m>_vs_<target>.csv for per-image metrics and a point difference S[m] - S[target] without
    interval for the set-level ones."""
    if target == run_ref:
        return {k: v for k, v in D.items() if k[1] in metrics}
    out = {}
    for (m, metric), (v, lo, hi) in D.items():
        if m == target and metric in metrics:
            out[(run_ref, metric)] = (-v, None if hi is None else -hi, None if lo is None else -lo)
    methods = {m for (m, _) in S if m not in (target, run_ref)}
    for m in sorted(methods):
        per_image = pairs(run, m, target, results)
        for metric in metrics:
            if metric in per_image:
                out[(m, metric)] = per_image[metric]
            elif (m, metric) in S and (target, metric) in S:
                out[(m, metric)] = (S[(m, metric)] - S[(target, metric)], None, None)
    return out


def n_prompts(run, results=RESULTS):
    """The set of prompt counts recorded in a run's summaries (should be {195})."""
    d = results / run
    return {int(float(r["n_prompts"])) for r in _rows(d / "base_summary.csv") + _rows(d / "extra_summary.csv")
            if r.get("n_prompts")}
