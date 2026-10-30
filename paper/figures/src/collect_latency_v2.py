#!/usr/bin/env python3
"""Clean batch-1 latency of the final operator (sca_v2) and published SCA (sca_nb), timed back to back on one GPU.

Inputs (read-only): /scratch/09032/saini_2/sca/out/latency_curve_v2/<HxW>/{sca_nb,sca_v2}/*.json, the sample.py
sidecars written by taskq x_d96z_lat_<HxW>.sh (planned as x_e_lat_<HxW>.sh). That task runs sca_nb and then sca_v2
on the same GPU, each in its own process over latency4_warmup.jsonl, whose first prompt (sca_latency_warmup) warms up
and compiles; the next three are timed. The protocol is scripts/audit_latency.py's (results/latency_curve_summary.csv):
  * per size and method, exactly the warm-up and the three timed prompts, each a complete sidecar with the method's
    label, the size, 28 steps, guidance 4.5 and seed 0 (sca_v2 also with its schedule and bridge flags, sca_nb with
    neither);
  * the task log (logs/task_<task>_<HxW>.log or .latest.log, either task name, the newest) must show, for each
    method, the warm-up and then the three timed prompts in order, no 'skip' line (a resumed process would time a
    cold image), and the same seconds as the sidecars (the log prints three decimals);
  * median, minimum and maximum of the three timed images.
A size is written only when every timed sidecar of both methods exists and passes; incomplete sizes are reported
and left out, and nothing is written while no size is complete.

Outputs, in results/latency_curve_v2/ (the only results/ directory this script writes):
  summary.csv   res, method, n, median_seconds, minimum_seconds, maximum_seconds, log_sha256
                (the columns of results/latency_curve_summary.csv)
  samples.csv   res, method, prompt_id, seconds, peak_memory_bytes, warmup_verified (as results/latency_curve.csv)
  SHA256SUMS

    PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/work/09032/saini_2/vista/library/uv-cache uv run --no-project \\
        --python /work/09032/saini_2/vista/library/uv-envs/sca/bin/python python paper/figures/src/collect_latency_v2.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "latency_curve_v2"
S = Path("/scratch/09032/saini_2/sca")
SRC = S / "out" / "latency_curve_v2"
LOGS = S / "logs"
TASKS = ("x_d96z_lat", "x_e_lat")  # the task name that ran, then the planned one
PROMPTS = S / "latency4_warmup.jsonl"
SIZES = ("2048x2048", "3072x3072", "4096x4096", "6144x6144", "8192x8192")
METHODS = ("sca_nb", "sca_v2")
WARM = "sca_latency_warmup"
V2_FLAGS = {"win_sched": [[0.88, 128], [0.86, 112], [0.84, 96], [0.82, 80]], "bridge_cells": 8, "bridge_k": 16,
            "win_pi_match": True, "win_pi_interp": True}


def operator_ok(method, sca):
    if method == "sca_v2":
        return all(sca.get(k) == v for k, v in V2_FLAGS.items())
    return not sca.get("win_sched") and not sca.get("bridge_k")


def task_log(res):
    names = [LOGS / f"task_{t}_{res}{suffix}.log" for t in TASKS for suffix in ("", ".latest")]
    found = [p for p in names if p.exists()]
    return max(found, key=lambda p: p.stat().st_mtime_ns) if found else None


def collect(res, timed):
    """(groups, samples, None) when the size passes, else (None, None, reason)."""
    expected = [WARM, *timed]
    sidecars = {}
    for method in METHODS:
        found = {}
        for path in sorted((SRC / res / method).glob("*.json")):
            data = json.loads(path.read_text())
            pid = data["config"]["prompt_id"]
            if pid in found:
                return None, None, f"{method}: duplicate sidecar for {pid}"
            found[pid] = data
        missing = [p for p in expected if p not in found]
        if missing:
            return None, None, f"{method}: {len(expected) - len(missing)} of {len(expected)} sidecars"
        if set(found) != set(expected):
            return None, None, f"{method}: unexpected prompts {sorted(set(found) - set(expected))}"
        for pid, data in found.items():
            c = data["config"]
            if not data.get("complete") or c["res"] != res or c["method"] != method:
                return None, None, f"{method}/{pid}: incomplete or mislabelled sidecar"
            if (c["steps"], c["guidance"], c["seed"]) != (28, 4.5, 0):
                return None, None, f"{method}/{pid}: steps/guidance/seed {c['steps']}/{c['guidance']}/{c['seed']}"
            if not operator_ok(method, c.get("sca", {})):
                return None, None, f"{method}/{pid}: operator flags do not match {method}"
            if data.get("first_call_in_process") is True and pid != WARM:
                return None, None, f"{method}/{pid}: timed sample was the first call in its process"
        sidecars[method] = found
    log = task_log(res)
    if log is None:
        return None, None, f"no task log for {' or '.join(t + '_' + res for t in TASKS)} under {LOGS}"
    raw = log.read_bytes()
    text = raw.decode(errors="replace")
    saved = re.findall(r"saved (\S+) (\S+) (\d+x\d+) seconds=([0-9.]+)", text)
    groups, samples = [], []
    for method in METHODS:
        events = [(pid, float(sec)) for pid, label, shape, sec in saved if label == method and shape == res]
        if [pid for pid, _ in events] != expected:
            return None, None, f"{method}: log {log.name} does not show the warm-up then the three timed prompts"
        if re.search(rf"skip \S+ {re.escape(method)} {res}", text):
            return None, None, f"{method}: log {log.name} shows a resumed process (skip lines)"
        for pid, logged in events:
            if abs(sidecars[method][pid]["seconds"] - logged) > 0.0006:
                return None, None, f"{method}/{pid}: log and sidecar seconds differ"
        seconds = [sidecars[method][pid]["seconds"] for pid in timed]
        groups.append({"res": res, "method": method, "n": len(seconds), "median_seconds": statistics.median(seconds),
                       "minimum_seconds": min(seconds), "maximum_seconds": max(seconds),
                       "log_sha256": hashlib.sha256(raw).hexdigest()})
        samples += [{"res": res, "method": method, "prompt_id": pid, "seconds": sidecars[method][pid]["seconds"],
                     "peak_memory_bytes": sidecars[method][pid].get("peak_memory_bytes"), "warmup_verified": True}
                    for pid in timed]
    return groups, samples, None


def write_csv(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def main():
    ids = [json.loads(line)["id"] for line in PROMPTS.read_text().splitlines() if line.strip()]
    if ids[0] != WARM or len(ids) != 4:
        raise ValueError(f"{PROMPTS}: expected the warm-up then three timed prompts, got {ids}")
    groups, samples = [], []
    for res in SIZES:
        g, s, why = collect(res, ids[1:])
        if why:
            print(f"latency_curve_v2 {res}: pending ({why})")
            continue
        groups += g
        samples += s
        print(f"latency_curve_v2 {res}: complete, median " +
              ", ".join(f"{r['method']} {r['median_seconds']:.1f} s" for r in g))
    if not groups:
        print("latency_curve_v2: no size complete; nothing written")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "summary.csv", groups)
    write_csv(OUT / "samples.csv", samples)
    sums = "".join(f"{hashlib.sha256((OUT / n).read_bytes()).hexdigest()}  {n}\n" for n in ("samples.csv", "summary.csv"))
    (OUT / "SHA256SUMS").write_text(sums)
    print(f"wrote {OUT}/summary.csv, samples.csv, SHA256SUMS ({len(groups) // 2} sizes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
