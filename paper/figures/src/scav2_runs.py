"""The final operator's (sca_v2) runs for the main tables: which have landed, and their numbers (read-only).

sca_v2 is published SCA (label sca_nb) plus a window schedule after the global-only opening (local windows of 128, 112,
96 and 80 tokens on the four steps after the gate, then 64, each head's weight read from the calibration table at the
matching ratio) and 1,024 bridge tokens in the local softmax (.agent/notes/duplication_fix_20261027.md, end).
Published SCA becomes the ablation row "without the schedule".

Landed means:
  * a 195-prompt run (RUNS): /scratch/09032/saini_2/sca/eval/<run>/DONE exists and its eval task has copied both
    summaries to results/<run>/ (the task copies before it writes DONE);
  * a 46-prompt variant (VARIANTS46): eval/dup46_<label>/DONE exists;
  * the clean latency: results/latency_curve_v2/summary.csv (collect_latency_v2.py) has the 4096x4096 sca_v2 row.
SCA_V2_PENDING=1 in the environment treats every queued run as pending (to check the published rendering).

Sources, all read-only:
  results/<run>/{base,extra}_summary.csv, <a>_vs_<b>.csv (scripts/paired_bootstrap.py, 2000 prompt draws, seed 0),
      corrected_fidp/{summary,paired_deltas}.csv (no intervals: the z98x tasks run --bootstrap 0) and
      dup_rates_people.csv, as copied by the z98x eval tasks;
  eval/<run>/base/paired_deltas.csv and extra/extra_paired_deltas.csv: eval.py and eval_extra.py deltas against the
      run's final label (1000 prompt draws). The z98x tasks do not copy them, so they are read from the eval directory
      (or from results/<run>/{base,extra}_paired_deltas.csv if those are ever added);
  eval/<run>/{extra/extra_per_image.csv, base/per_image.csv}: per-image metrics, for intervals between methods
      evaluated in two different runs (derived_pairs: scripts/paired_bootstrap.py on the rows of the two methods,
      2000 draws, seed 0, run in a temporary directory that is deleted afterwards);
  results/dup_audit_20261027/: duplication rates and the subject list;
  eval/dup46_<label>/: base/, extra/, paired/<label>_vs_sca_nb.csv, dup_per_image.csv of the 46-prompt runs.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
from collections import namedtuple
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data195  # noqa: E402

ROOT = HERE.parents[2]
RESULTS = ROOT / "results"
S = Path("/scratch/09032/saini_2/sca")
EVAL = S / "eval"
TASKQ = S / "taskq"
DUP = RESULTS / "dup_audit_20261027"
LAT_PUB = RESULTS / "latency_curve_summary.csv"
LAT_V2 = RESULTS / "latency_curve_v2" / "summary.csv"
NOTE = " Grey: published values without the schedule; the final operator's run is pending."
LATENCY = "latency_curve_v2"
PEOPLE = {"person", "animal"}

# run -> the final operator's label in it, the published run it replaces, the published label there, its eval task
Run = namedtuple("Run", "final published_run published task")
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
# queued 46-prompt variants of sca_v2 (label -> eval task); each changes one thing (generation task flags):
# dup_i_160 first window 160 (160, 112, 96, 80); dup_i_bridge32 32 bridge keys per cell (2,048 tokens);
# dup_i_ext75 five-step schedule (128, 120, 112, 96, 80 down to sigma 0.80)
VARIANTS46 = {"dup_i_160": "x_d90z_z97_eval_dup46_dup_i_160.sh",
              "dup_i_bridge32": "x_d90z_z97_eval_dup46_dup_i_bridge32.sh",
              "dup_i_ext75": "x_d90z_z97_eval_dup46_dup_i_ext75.sh"}
# eval.py method label -> label in results/dup_audit_20261027/rates_a4k195_people.csv
DUP_NAME = {"sega_official": "sega", "hrdit_dev_author": "hrdit", "sigma_dev_matched": "sigma"}


def forced_pending():
    return os.environ.get("SCA_V2_PENDING", "") not in ("", "0")


def landed(run):
    if forced_pending():
        return False
    if run == LATENCY:
        return ("4096x4096", "sca_v2") in latency_v2()
    if run in VARIANTS46:
        return (EVAL / f"dup46_{run}" / "DONE").exists()
    d = RESULTS / run
    return (EVAL / run / "DONE").exists() and (d / "base_summary.csv").exists() and (d / "extra_summary.csv").exists()


def task_state(name):
    for sub in ("done", "running", "failed", "pending", "hold_dup_eval"):
        d = TASKQ / sub
        if d.is_dir() and any(p.name.startswith(name) for p in d.iterdir()):
            return sub
    return "not queued"


def status_lines():
    out = []
    for run, r in RUNS.items():
        if landed(run):
            out.append(f"{run:26s} landed  ({(EVAL / run / 'DONE').read_text().strip()})")
        elif forced_pending():
            out.append(f"{run:26s} pending (forced by SCA_V2_PENDING)")
        elif (EVAL / run / "DONE").exists():
            out.append(f"{run:26s} pending (DONE on scratch, results/{run}/ summaries missing)")
        else:
            out.append(f"{run:26s} pending (eval task {r.task}: {task_state(r.task)})")
    for label, task in VARIANTS46.items():
        state = "landed " if landed(label) else f"pending (eval task: {task_state(task)})"
        out.append(f"{'dup46_' + label:26s} {state}")
    lat = sorted(res for res, m in latency_v2() if m == "sca_v2")
    out.append(f"{LATENCY:26s} {'landed ' if landed(LATENCY) else 'pending'} (sca_v2 sizes in "
               f"results/latency_curve_v2/summary.csv: {', '.join(lat) or 'none'}; collect_latency_v2.py writes it)")
    return out


def _rows(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def latency(path):
    return {(r["res"], r["method"]): (float(r["median_seconds"]), int(r["n"])) for r in _rows(path)}


def latency_v2():
    return latency(LAT_V2)


# ---------------------------------------------------------------- 195-prompt runs ---------------------------------
def load(run):
    """(S, D, ref, corrected_ci, deltas_src) as data195.load; for an sca_v2 run, D is method minus the final label.

    Corrected FID/KID and their patch versions follow data195's rules: corrected_fidp_ci if it exists, else
    corrected_fidp's points, where whole-image FID and KID keep the eval.py interval and the patch versions none."""
    if run not in RUNS:
        S_, D_, ref, ci = data195.load(run, RESULTS)
        return S_, D_, ref, ci, f"results/{run}/{{base,extra}}_paired_deltas.csv"
    d, e = RESULTS / run, EVAL / run
    S_, D_, refs = {}, {}, set()
    for r in _rows(d / "base_summary.csv") + _rows(d / "extra_summary.csv"):
        S_[(r["method"], r["metric"])] = float(r["value"])
    if (d / "base_paired_deltas.csv").exists():
        files, src = [d / "base_paired_deltas.csv", d / "extra_paired_deltas.csv"], f"results/{run}"
    else:
        files, src = [e / "base" / "paired_deltas.csv", e / "extra" / "extra_paired_deltas.csv"], str(e)
    if not all(f.exists() for f in files):
        raise FileNotFoundError(f"{run}: eval.py paired deltas missing ({files})")
    for r in _rows(files[0]) + _rows(files[1]):
        refs.add(r["reference_method"])
        lo, hi = (float(r["ci_low"]), float(r["ci_high"])) if r["ci_low"] and r["ci_high"] else (None, None)
        D_[(r["method"], r["metric"])] = (float(r["value"]), lo, hi)
    ref = RUNS[run].final
    if refs != {ref}:
        raise ValueError(f"{run}: eval.py deltas against {refs}, expected {ref}")
    ci_dir = d / "corrected_fidp_ci"
    corrected_ci = (ci_dir / "paired_deltas.csv").exists()
    cdir = ci_dir if corrected_ci else d / "corrected_fidp"
    for r in _rows(cdir / "summary.csv"):
        if r["metric"] in data195.DIST:
            S_[(r["method"], r["metric"])] = float(r["value"])
    for r in _rows(cdir / "paired_deltas.csv"):
        if r["metric"] not in data195.DIST:
            continue
        if r["reference_method"] != ref:
            raise ValueError(f"{run}: corrected deltas against {r['reference_method']}, expected {ref}")
        key = (r["method"], r["metric"])
        if r["ci_low"] and r["ci_high"]:
            D_[key] = (float(r["value"]), float(r["ci_low"]), float(r["ci_high"]))
        elif r["metric"] in ("fid", "kid") and key in D_:
            D_[key] = (float(r["value"]), D_[key][1], D_[key][2])
        else:
            D_[key] = (float(r["value"]), None, None)
    for key in [k for k in D_ if k[1] in ("patch_fid", "patch_kid")]:
        if not corrected_ci and D_[key][1] is not None:
            D_[key] = (D_[key][0], None, None)
    return S_, D_, ref, corrected_ci, f"{src}/{{base/paired_deltas,extra/extra_paired_deltas}}.csv"


def paired_csv(rel, method, reference):
    """({metric: (delta, lo, hi)}, n_prompts) from a paired_bootstrap.py CSV under results/."""
    out, ns = {}, set()
    rows = _rows(RESULTS / rel)
    if not rows:
        raise FileNotFoundError(RESULTS / rel)
    for r in rows:
        if (r["method"], r["reference"]) != (method, reference):
            raise ValueError(f"{rel} holds {r['method']} vs {r['reference']}")
        ns.add(int(r["n_prompts"]))
        out[r["metric"]] = (float(r["delta"]), float(r["ci_low"]), float(r["ci_high"]))
    if len(ns) != 1:
        raise ValueError(f"{rel}: prompt counts {ns}")
    return out, ns.pop()


def per_image_files(eval_dir):
    files = [EVAL / eval_dir / "extra" / "extra_per_image.csv", EVAL / eval_dir / "base" / "per_image.csv"]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"per-image CSVs missing: {missing}")
    return files


def derived_pairs(eval_a, a, eval_b, b, draws=2000):
    """({metric: (delta, lo, hi)}, n): method a of eval/<eval_a> minus method b of eval/<eval_b>, per-image metrics.

    The rows of each method are copied from its run's per-image CSVs into a temporary directory and passed to
    scripts/paired_bootstrap.py (2000 draws, seed 0, as every eval task runs it); nothing is kept."""
    with tempfile.TemporaryDirectory(prefix="scav2_pairs_") as tmp:
        inputs = []
        for name, method in ((eval_a, a), (eval_b, b)):
            for src in per_image_files(name):
                rows = [r for r in csv.DictReader(open(src)) if r["method"] == method]
                if not rows:
                    raise ValueError(f"{src}: no rows for {method}")
                dst = Path(tmp) / f"{method}__{name}__{src.name}"
                with open(dst, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
                inputs.append(str(dst))
        out = Path(tmp) / "pairs.csv"
        subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "paired_bootstrap.py"), "--method", a,
                        "--reference", b, "--draws", str(draws), "--out", str(out)] + inputs, check=True)
        rows = _rows(out)
    groups = {(r["res"], r["seed"]) for r in rows}
    if len(groups) != 1:
        raise ValueError(f"{a} vs {b}: resolution/seed groups {groups}")
    ns = {int(r["n_prompts"]) for r in rows}
    if len(ns) != 1:
        raise ValueError(f"{a} vs {b}: prompt counts {ns}")
    return {r["metric"]: (float(r["delta"]), float(r["ci_low"]), float(r["ci_high"])) for r in rows}, ns.pop()


def dup195(run, method):
    """(k, n): prompts naming a person or animal (95) with a duplicated person or animal.

    Published methods: results/dup_audit_20261027/rates_a4k195_people.csv; sca_v2 runs: results/<run>/
    dup_rates_people.csv (scripts/dup_rate.py --kinds person,animal, written by the run's eval task)."""
    path = (RESULTS / run / "dup_rates_people.csv") if run in RUNS else DUP / "rates_a4k195_people.csv"
    name = method if run in RUNS else DUP_NAME.get(method, method)
    for r in _rows(path):
        if r["method"] == name:
            return int(r["dup"]), int(r["n"])
    raise KeyError(f"{path}: no row for {name}")


# ---------------------------------------------------------------- 46-prompt subset ---------------------------------
def ids46():
    ids = [json.loads(line)["id"] for line in open(S / "abl50.jsonl") if line.strip()]
    dev = {json.loads(line)["id"] for line in open(S / "dev4.jsonl") if line.strip()}
    return [i for i in ids if i not in dev]


def _people_flags(per_image_csv, method):
    """{prompt: flag} for one method from a dup_rate.py --per-image CSV written without --kinds: its counts column
    lists query:count/implied for every subject with an implied count, in subjects_a4k195.json order, so keeping the
    person and animal subjects reproduces dup_rate.py --kinds person,animal exactly."""
    subj = {r["id"]: [s for s in r["subjects"] if s.get("implied") is not None]
            for r in json.load(open(DUP / "subjects_a4k195.json"))}
    out = {}
    for r in _rows(per_image_csv):
        if r["method"] != method:
            continue
        counts = r["counts"].split(";") if r["counts"] else []
        subjects = subj[r["prompt_id"]]
        if len(counts) != len(subjects):
            raise ValueError(f"{per_image_csv}: {r['prompt_id']} has {len(counts)} counts, {len(subjects)} subjects")
        flag, eligible = False, False
        for c, s in zip(counts, subjects):
            query, n = c.rsplit(":", 1)
            seen, implied = n.split("/")
            if query != s["query"] or int(implied) != s["implied"]:
                raise ValueError(f"{per_image_csv}: {r['prompt_id']} count {c} does not match subject {s}")
            if s.get("kind") in PEOPLE:
                eligible = True
                flag |= int(seen) > int(implied)
        if eligible:
            out[r["prompt_id"]] = flag
    return out


def dup46(method):
    """(k, n, source) of the 26 people/animal prompts of the 46: rates_46_all_people.csv, then rates_abl46_people.csv,
    then (a variant that neither lists yet) eval/dup46_<method>/dup_per_image.csv filtered to people and animals."""
    for name in ("rates_46_all_people.csv", "rates_abl46_people.csv"):
        for r in _rows(DUP / name):
            if r["method"] == method:
                return int(r["dup"]), int(r["n"]), f"results/dup_audit_20261027/{name}, columns dup, n"
    path = EVAL / f"dup46_{method}" / "dup_per_image.csv"
    flags = _people_flags(path, method)
    keep = set(ids46())
    flags = {p: f for p, f in flags.items() if p in keep}
    return sum(flags.values()), len(flags), f"{path} (person/animal subjects of subjects_a4k195.json)"


def summary46(label):
    """{metric: value} for `label` from eval/dup46_<label>/{extra/extra_summary.csv, base/summary.csv} (46 prompts)."""
    e = EVAL / f"dup46_{label}"
    out = {}
    for r in _rows(e / "extra" / "extra_summary.csv") + _rows(e / "base" / "summary.csv"):
        if r["method"] == label:
            if int(float(r["n_prompts"])) != 46:
                raise ValueError(f"{e}: {r['metric']} over {r['n_prompts']} prompts")
            out[r["metric"]] = float(r["value"])
    return out


def pair46(label):
    """{metric: (delta, lo, hi)}: label minus sca_nb, eval/dup46_<label>/paired/<label>_vs_sca_nb.csv (2000 draws)."""
    path = EVAL / f"dup46_{label}" / "paired" / f"{label}_vs_sca_nb.csv"
    out = {}
    for r in _rows(path):
        if (r["method"], r["reference"], r["n_prompts"]) != (label, "sca_nb", "46"):
            raise ValueError(f"{path}: {r}")
        out[r["metric"]] = (float(r["delta"]), float(r["ci_low"]), float(r["ci_high"]))
    if not out:
        raise FileNotFoundError(path)
    return out


def seconds46(label):
    """{method: (median seconds, n)} for label and sca_nb from eval/dup46_<label>/base/per_image.csv, column seconds
    (the generation time recorded with each image; scripts/dup_decide.py takes the same medians from the sidecars)."""
    t = {}
    for r in _rows(EVAL / f"dup46_{label}" / "base" / "per_image.csv"):
        t.setdefault(r["method"], []).append(float(r["seconds"]))
    return {m: (statistics.median(v), len(v)) for m, v in t.items()}


def _bootstrap4000(d, draws=4000):
    """scripts/dup_vs_area.py's interval: 4000 prompt draws, random.Random(0), percentile 2.5 and 97.5."""
    rng = random.Random(0)
    b = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(draws))
    return sum(d) / len(d), b[int(0.025 * draws)], b[int(0.975 * draws) - 1]


def vs_area46(label, metric="ir"):
    """(delta, lo, hi, source): label minus AReA on the 46 prompts, recomputed as scripts/dup_vs_area.py does (AReA and
    published SCA from eval/a4k195_main, a variant from eval/dup46_<label>) and checked against the archived
    results/dup_audit_20261027/dup46_20261029/vs_area_46.txt at its printed precision; that text is the fallback when
    the per-image CSVs are gone."""
    txt = DUP / "dup46_20261029" / "vs_area_46.txt"
    m = re.search(rf"^{re.escape(label)}: .*?\b{metric} ([+-][0-9.e+-]+)\[([+-][0-9.e+-]+),([+-][0-9.e+-]+)\]",
                  txt.read_text(), re.M)
    archived = tuple(float(x) for x in m.groups()) if m else None

    def values(eval_dir):
        out = {}
        for f in (EVAL / eval_dir / "extra" / "extra_per_image.csv", EVAL / eval_dir / "base" / "per_image.csv"):
            for r in _rows(f):
                if r.get(metric) not in (None, ""):
                    out[(r["method"], r["prompt_id"])] = float(r[metric])
        return out

    main = values("a4k195_main")
    src = main if label == "sca_nb" else values(f"dup46_{label}")
    ids = ids46()
    d = [src[(label, i)] - main[("area", i)] for i in ids if (label, i) in src and ("area", i) in main]
    if len(d) == len(ids):
        res = _bootstrap4000(d)
        if archived and any(f"{x:+.3g}" != f"{y:+.3g}" for x, y in zip(res, archived)):
            raise ValueError(f"vs_area46({label}): recomputed {res} but {txt.name} prints {archived}")
        return (*res, f"recomputed from eval/a4k195_main and eval/dup46_{label} per-image {metric} (dup_vs_area.py "
                      f"protocol: 4000 draws, Random(0)); equals results/dup_audit_20261027/dup46_20261029/{txt.name} at its printed precision")
    if archived is None:
        raise FileNotFoundError(f"vs_area46({label}): no per-image CSVs and no line in {txt}")
    return (*archived, f"results/dup_audit_20261027/dup46_20261029/{txt.name}, line '{label}', {metric} (three significant digits; the per-image "
                       f"CSVs are gone)")
