#!/usr/bin/env python3
"""The sca_v2 switch shared by gen_app_tables.py, fig_paired_area.py and fig_latency.py.

sca_v2 is the final operator: published SCA (label sca_nb) plus the window schedule after the opening (128, 112, 96,
80, then 64 tokens, each head's weight read at the matching ratio) and 1,024 bridge tokens in the local softmax
(.agent/notes/duplication_fix_20261027.md, end). Published SCA becomes "SCA without the schedule", an ablation.
Its evaluations land one run at a time (taskq z980 to z988): /scratch/09032/saini_2/sca/eval/<run>/DONE, with the
CSVs copied to results/<run>/. Until a run lands, every table cell that will change shows the published value as
\\pend{value} (grey), and the caption of each affected table carries a note macro that expands to NOTE while pending
and to nothing after. The latency curve lands as results/latency_curve_v2/summary.csv (no marker).

Inputs are read-only: results/*.csv and /scratch/09032/saini_2/sca/eval/<run>/ CSVs. The only files written here are
derived paired intervals (a method of one run minus a method of another, e.g. a baseline minus sca_v2), computed with
scripts/paired_bootstrap.py (2000 prompt draws, seed 0, as the eval tasks use) from the two runs' per-image eval CSVs
and cached under /scratch/09032/saini_2/sca/eval/app_tables_pairs/.

    python v2swap.py     # one status line per run: landed or pending (and where its eval task sits)

Environment: SCA_V2_PENDING=1 treats every run as pending, to test the published rendering after runs land.
SCA_V2_MOCK=DIR reads the sca_v2 runs (and the latency curve) from DIR/results and DIR/eval instead, and makes the
generators write into DIR (never into paper/); `python v2swap.py --make-mock DIR` builds such a tree in which every
sca_v2 label is a relabelled copy of the published SCA rows, so a landed rendering should reproduce today's numbers
(less the intervals the z98x tasks do not compute). Mock numbers are for testing the code path only.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data195  # noqa: E402  (read-only; published runs)

ROOT = HERE.parents[2]
RESULTS = ROOT / "results"
EVAL = Path("/scratch/09032/saini_2/sca/eval")
TASKQ = Path("/scratch/09032/saini_2/sca/taskq")
MOCK = Path(os.environ["SCA_V2_MOCK"]).resolve() if os.environ.get("SCA_V2_MOCK") else None
V2_RESULTS = MOCK / "results" if MOCK else RESULTS  # where the sca_v2 runs are read from
V2_EVAL = MOCK / "eval" if MOCK else EVAL
PAIR_CACHE = (MOCK / "pairs") if MOCK else EVAL / "app_tables_pairs"
LATENCY_PUBLISHED = RESULTS / "latency_curve_summary.csv"
LATENCY_V2 = V2_RESULTS / "latency_curve_v2" / "summary.csv"
DIST = data195.DIST

# run -> final operator's method label in that run, the published run it replaces, the published SCA label there,
# and the eval task that writes it. Every eval run also re-evaluates the published label (and AReA where listed).
Run = namedtuple("Run", "final published_run published_method task")
RUNS = {
    "a4k195_main_sca_v2": Run("sca_v2", "a4k195_main", "sca_nb", "z980_eval_a4k195_main_sca_v2.sh"),
    "a4k195_2048x4096_sca_v2": Run("sca_v2", "a4k195_2048x4096", "sca_nb", "z981_eval_a4k195_2048x4096_sca_v2.sh"),
    "a4k195_2048x6144_sca_v2": Run("sca_v2", "a4k195_2048x6144", "sca_nb", "z982_eval_a4k195_2048x6144_sca_v2.sh"),
    "a4k195_seed1_sca_v2": Run("sca_v2", "a4k195_seed1", "sca_nb", "z983_eval_a4k195_seed1_sca_v2.sh"),
    "a4k195_seed2_sca_v2": Run("sca_v2", "a4k195_seed2", "sca_nb", "z984_eval_a4k195_seed2_sca_v2.sh"),
    "prog195_sca_v2": Run("sca_v2_prog", "a4k195_panel", "sca_prog", "z985_eval_prog195_sca_v2.sh"),
    "krea195_sca_v2": Run("sca_v2_krea", "krea195", "sca_krea", "z986_eval_krea195_sca_v2.sh"),
    "hrgen1k_dev_sca_v2": Run("sca_v2", "hrgen1k_dev", "sca_nb", "z987_eval_hrgen1k_dev_sca_v2.sh"),
    "hrgen1k_krea_sca_v2": Run("sca_v2_krea", "hrgen1k_krea", "sca_krea", "z988_eval_hrgen1k_krea_sca_v2.sh"),
}
LATENCY = "latency_curve_v2"  # pseudo-run name for the latency curve

NOTE = " Grey: published values without the schedule; the final operator's run is pending."
# Fallback for the shared grey wrapper, placed at the top of every generated snippet. \providecommand inside the
# float is local and loses to any preamble definition, so the main tables' \pend wins if one exists. It greys the
# whole cell, including \up / \dn marks (kept bold / italic), which would otherwise keep their colours.
PEND_FALLBACK = r"\providecommand{\pend}[1]{{\color{gray}\let\up\textbf\let\dn\textit#1}}%"


def forced_pending():
    return os.environ.get("SCA_V2_PENDING", "") not in ("", "0")


def landed(run):
    """True once a run's results can be read: DONE on scratch and both summaries copied to results/<run>/."""
    if forced_pending():
        return False
    if run == LATENCY:
        return LATENCY_V2.exists()
    if run not in RUNS:
        raise KeyError(run)
    d = V2_RESULTS / run
    return (V2_EVAL / run / "DONE").exists() and (d / "base_summary.csv").exists() and (d / "extra_summary.csv").exists()


def task_state(run):
    """Where the run's eval task sits in the queue (hold_dup_eval, pending, running, done, failed) or '?'."""
    name = RUNS[run].task
    for sub in ("done", "running", "failed", "pending", "hold_dup_eval"):
        d = TASKQ / sub
        if d.is_dir() and any(p.name.startswith(name) for p in d.iterdir()):
            return sub
    return "?"


def status_lines():
    out = []
    for run in RUNS:
        if landed(run):
            out.append(f"{run}: landed ({(V2_EVAL / run / 'DONE').read_text().strip()})")
        else:
            why = "forced by SCA_V2_PENDING" if forced_pending() else f"eval task {task_state(run)}"
            if not forced_pending() and (V2_EVAL / run / "DONE").exists():
                why = "DONE on scratch but results/ copy incomplete"
            out.append(f"{run}: pending ({why})")
    out.append(f"{LATENCY}: {'landed' if landed(LATENCY) else 'pending'} ({LATENCY_V2})")
    return out


# ---------------------------------------------------------------- loaders ----------------------------------------
def _rows(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def load(run):
    """(S, D, ref, corrected_ci) in data195.load's format for a published run or a landed sca_v2 run.

    For an sca_v2 run, ref is its final label (sca_v2, sca_v2_prog or sca_v2_krea) and D holds method minus ref. The
    results/ copy has no eval.py paired deltas, so they are read from the scratch eval directory
    (base/paired_deltas.csv and extra/extra_paired_deltas.csv, 1000 prompt draws, written by the same eval task);
    corrected FID/KID and patch versions follow data195's rules (corrected_fidp_ci if it ever appears, else
    corrected_fidp, whose z98x run has no intervals: whole-image FID/KID keep eval.py's interval, patch ones none)."""
    if run not in RUNS:
        return data195.load(run, RESULTS)
    if not landed(run):
        raise RuntimeError(f"{run} has not landed")
    d, e = V2_RESULTS / run, V2_EVAL / run
    S, D, refs = {}, {}, set()
    for r in _rows(d / "base_summary.csv") + _rows(d / "extra_summary.csv"):
        S[(r["method"], r["metric"])] = float(r["value"])
    for r in _rows(e / "base" / "paired_deltas.csv") + _rows(e / "extra" / "extra_paired_deltas.csv"):
        refs.add(r["reference_method"])
        lo, hi = (float(r["ci_low"]), float(r["ci_high"])) if r["ci_low"] and r["ci_high"] else (None, None)
        D[(r["method"], r["metric"])] = (float(r["value"]), lo, hi)
    ref = RUNS[run].final
    if refs - {ref}:
        raise ValueError(f"{run}: eval.py references {refs}, expected {ref}")
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
            raise ValueError(f"{run}: corrected reference {r['reference_method']}, expected {ref}")
        key = (r["method"], r["metric"])
        if r["ci_low"] and r["ci_high"]:
            D[key] = (float(r["value"]), float(r["ci_low"]), float(r["ci_high"]))
        elif r["metric"] in ("fid", "kid") and key in D:
            D[key] = (float(r["value"]), D[key][1], D[key][2])
        else:
            D[key] = (float(r["value"]), None, None)
    for key in [k for k in D if k[1] in ("patch_fid", "patch_kid")]:
        if not corrected_ci and D[key][1] is not None:
            D[key] = (D[key][0], None, None)
    return S, D, ref, corrected_ci


def pairs(run, a, b):
    """{metric: (delta, lo, hi)} from results/<run>/<a>_vs_<b>.csv (paired_bootstrap.py, 2000 draws)."""
    return data195.pairs(run, a, b, V2_RESULTS if run in RUNS else RESULTS)


def versus(S, D, run_ref, run, target, metrics):
    """data195.versus over results/: deltas against `target`, keyed (method, metric)."""
    return data195.versus(S, D, run_ref, run, target, metrics, V2_RESULTS if run in RUNS else RESULTS)


def _per_image_files(run):
    e = (V2_EVAL if run in RUNS else EVAL) / run
    return [p for p in (e / "extra" / "extra_per_image.csv", e / "base" / "per_image.csv") if p.exists()]


def derived_pairs(run_a, a, run_b, b, draws=2000):
    """{metric: (delta, lo, hi)}: method a of run_a minus method b of run_b, per-image metrics only.

    Both methods must cover the same prompts, resolution and seed label. Rows of the two runs' per-image eval CSVs
    are filtered to the one method each and passed to scripts/paired_bootstrap.py; the result is cached under
    PAIR_CACHE and recomputed when missing or older than either run's per-image CSVs."""
    files_a, files_b = _per_image_files(run_a), _per_image_files(run_b)
    if not files_a or not files_b:
        raise FileNotFoundError(f"per-image CSVs for {run_a} or {run_b}")
    tag = f"{run_a}__{a}_vs_{run_b}__{b}"
    out = PAIR_CACHE / f"{tag}.csv"
    newest = max(p.stat().st_mtime for p in files_a + files_b)
    if not out.exists() or out.stat().st_mtime < newest:
        work = PAIR_CACHE / f".{tag}.inputs"
        work.mkdir(parents=True, exist_ok=True)
        inputs = []
        for method, files in ((a, files_a), (b, files_b)):
            for src in files:
                rows = [r for r in csv.DictReader(open(src)) if r["method"] == method]
                if not rows:
                    continue
                dst = work / f"{method}__{src.parent.name}__{src.name}"
                with open(dst, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                inputs.append(str(dst))
        tmp = out.with_suffix(".tmp")
        subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "paired_bootstrap.py"), "--method", a,
                        "--reference", b, "--draws", str(draws), "--out", str(tmp)] + inputs, check=True)
        tmp.replace(out)
    res = {}
    for r in _rows(out):
        res[r["metric"]] = (float(r["delta"]), float(r["ci_low"]), float(r["ci_high"]))
    return res


def latency(v2=None):
    """{(res, method): (median, minimum, maximum, n)} seconds; v2=None: the sca_v2 curve if landed, else published.

    The sca_v2 file has sca_v2 and sca_nb rows (and possibly others); published rows fill the methods it lacks."""
    rows = {}
    for r in _rows(LATENCY_PUBLISHED):
        rows[(r["res"], r["method"])] = (float(r["median_seconds"]), float(r["minimum_seconds"]),
                                         float(r["maximum_seconds"]), int(r["n"]))
    if v2 is None:
        v2 = landed(LATENCY)
    if v2:
        for r in _rows(LATENCY_V2):
            rows[(r["res"], r["method"])] = (float(r["median_seconds"]), float(r["minimum_seconds"]),
                                             float(r["maximum_seconds"]), int(r["n"]))
    return rows


# ---------------------------------------------------------------- LaTeX helpers ----------------------------------
def pend(cell, pending):
    """Wrap a cell that will change when its run lands; empty cells stay empty."""
    return rf"\pend{{{cell}}}" if pending and cell.strip() else cell


def note_macro(name, pending):
    return rf"\providecommand{{\{name}}}{{{NOTE if pending else ''}}}"


# ---------------------------------------------------------------- mock tree (tests only) -------------------------
EVAL_METHODS = {  # methods each z98x task evaluates besides the final label (see its GATE spec)
    "prog195_sca_v2": ("sca_prog", "hrdit_dev_author"), "hrgen1k_krea_sca_v2": ("sca_krea",),
    "krea195_sca_v2": ("sca_krea", "area")}


def make_mock(root, with_ci=False):
    """Fake landed tree: each sca_v2 run = its published run with the published SCA rows copied under the final
    label (and as themselves), restricted to the methods the z98x task evaluates; deltas re-referenced to the final
    label. corrected_fidp_ci is copied only with with_ci (the z98x tasks do not compute it)."""
    import shutil
    root = Path(root).resolve()

    def relabel(src, dst, keep, final, pub, ref_cols=()):
        rows = _rows(src)
        if not rows:
            return
        out = []
        for r in rows:
            if r.get("method") not in keep:
                continue
            if ref_cols == ("reference_method",):  # deltas: re-reference to the final label
                if r["reference_method"] == pub and r["method"] != pub:
                    out.append(dict(r, reference_method=final))
                elif r["method"] == pub and r["reference_method"] in keep:  # pub - x  ->  x - final
                    lo, hi = r.get("ci_low", ""), r.get("ci_high", "")
                    out.append(dict(r, method=r["reference_method"], reference_method=final,
                                    value=str(-float(r["value"])), ci_low=str(-float(hi)) if hi else "",
                                    ci_high=str(-float(lo)) if lo else ""))
                continue
            for col in ref_cols:
                if r.get(col) == pub:
                    r = dict(r, **{col: final})
            out.append(r)
            if r.get("method") == pub and not ref_cols:
                out.append(dict(r, method=final))
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(out)

    for run, spec in RUNS.items():
        pub, final, src = spec.published_method, spec.final, RESULTS / spec.published_run
        keep = {pub} | set(EVAL_METHODS.get(run, ("sca_nb", "area")))
        d, e = root / "results" / run, root / "eval" / run
        for name in ("base_summary.csv", "extra_summary.csv"):
            relabel(src / name, d / name, keep, final, pub)
        for sub in ["corrected_fidp"] + (["corrected_fidp_ci"] if with_ci else []):
            if (src / sub / "summary.csv").exists():
                relabel(src / sub / "summary.csv", d / sub / "summary.csv", keep, final, pub)
                relabel(src / sub / "paired_deltas.csv", d / sub / "paired_deltas.csv", keep, final, pub,
                        ("reference_method",))
        for p in src.glob(f"{pub}_vs_*.csv"):  # published SCA minus x -> final minus x
            other = p.stem.split("_vs_", 1)[1]
            if other in keep:
                relabel(p, d / f"{final}_vs_{other}.csv", {pub}, final, pub, ("method",))
        pe = EVAL / spec.published_run
        relabel(pe / "base" / "per_image.csv", e / "base" / "per_image.csv", keep, final, pub)
        relabel(pe / "extra" / "extra_per_image.csv", e / "extra" / "extra_per_image.csv", keep, final, pub)
        relabel(pe / "base" / "paired_deltas.csv", e / "base" / "paired_deltas.csv", keep, final, pub,
                ("reference_method",))
        relabel(pe / "extra" / "extra_paired_deltas.csv", e / "extra" / "extra_paired_deltas.csv", keep, final,
                pub, ("reference_method",))
        (e / "DONE").write_text("mock\n")
    lat = root / "results" / "latency_curve_v2"
    lat.mkdir(parents=True, exist_ok=True)
    rows = [dict(r) for r in _rows(LATENCY_PUBLISHED) if r["method"] == "sca_nb"]
    with open(lat / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows + [dict(r, method="sca_v2") for r in rows])
    shutil.rmtree(root / "pairs", ignore_errors=True)
    print(f"mock tree at {root}: SCA_V2_MOCK={root}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--make-mock":
        make_mock(sys.argv[2], with_ci="--with-ci" in sys.argv)
    else:
        print("\n".join(status_lines()))
