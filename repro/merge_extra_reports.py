#!/usr/bin/env python3
"""Merge complete split metric reports with identical recorded input identities."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import tempfile
import time

from extra_report_identity import IDENTITY_FILE, POLICY, REPORT_FILES, key, sha256, write_json

FIELDS = ("method", "res", "prompt_id", "seed", "path")
TEXT_FIELDS = set(FIELDS) | {"metric", "reference_method", "delta_definition"}
METRICS = {"ir", "aesthetic", "hpsv2", "hpsv21", "clipiqa", "arniqa", "is", "latency"}
PROTOCOL = ("reference_method", "bootstrap", "bootstrap_seed", "iqa_long_side", "is_splits")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index(rows, fields, label):
    result = {}
    for row in rows:
        identity = tuple(str(row[name]) for name in fields)
        require(identity not in result, f"Duplicate {label}: {identity}")
        result[identity] = row
    return result


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def canonical(row):
    return {k: str(v) if k in TEXT_FIELDS else float(v)
            for k, v in row.items() if v is not None and v != ""}


def check_csv(path, expected, fields):
    with path.open(newline="") as handle:
        actual = list(csv.DictReader(handle))
    left, right = index(actual, fields, path.name), index(expected, fields, path.name + " JSON")
    require(left.keys() == right.keys(), f"CSV/JSON identity mismatch: {path}")
    for identity in left:
        require(canonical(left[identity]) == canonical(right[identity]), f"CSV/JSON value mismatch: {path}, {identity}")


def mean_by_prompt(values):
    grouped = defaultdict(list)
    for prompt, value in values:
        grouped[prompt].append(value)
    return statistics.mean(statistics.mean(v) for v in grouped.values()), len(grouped)


def check_stat(row, estimate, count, draws, label):
    require(finite(row.get("value")), f"Non-finite statistic: {label}")
    if estimate is not None:
        require(math.isclose(row["value"], estimate, rel_tol=1e-12, abs_tol=1e-12), f"Wrong scalar mean: {label}")
    require(row.get("n_prompts") == count, f"Prompt coverage mismatch: {label}")
    valid = row.get("bootstrap_valid")
    require(isinstance(valid, int) and 0 <= valid <= draws, f"Invalid bootstrap count: {label}")
    low, high = row.get("ci_low"), row.get("ci_high")
    require((low is None and high is None) or (finite(low) and finite(high) and low <= high), f"Invalid interval: {label}")


def load_source(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / IDENTITY_FILE).read_text())
    require(manifest.get("policy") == POLICY and manifest.get("phase") == "complete",
            f"Missing completed before/after image identities: {folder}; use the portable evaluator in a new output directory")
    for name in REPORT_FILES:
        require(manifest.get("report_files", {}).get(name) == sha256(folder / name), f"Report hash mismatch: {folder / name}")
    report = json.loads((folder / "extra_metrics.json").read_text())
    require(report.get("complete") is True and not report.get("errors"), f"Incomplete report: {folder}")
    metrics = report["metrics"]
    require(metrics and len(set(metrics)) == len(metrics) and set(metrics) <= METRICS, f"Invalid metrics: {folder}")
    rows = index(report["per_image"], FIELDS[:4], "per-image row")
    identities = index(manifest["identities"], FIELDS[:4], "input identity")
    require(rows and rows.keys() == identities.keys(), f"Image coverage mismatch: {folder}")
    scalar = set(metrics) - {"is"}
    for identity, row in rows.items():
        proof = identities[identity]
        require(row["path"] == proof["path"], f"Image path mismatch: {identity}")
        require(isinstance(proof.get("image_bytes"), int) and proof["image_bytes"] > 0, f"Invalid image size: {identity}")
        for field in ("image_sha256", "prompt_sha256", "sidecar_sha256", "config_hash"):
            value = proof.get(field, "")
            require(len(value) == 64 and all(c in "0123456789abcdef" for c in value), f"Missing SHA256 identity {field}: {identity}")
        require(set(row["metrics"]) == scalar and all(finite(v) for v in row["metrics"].values()),
                f"Missing, undeclared or non-finite per-image metric: {identity}")
    config = report["config"]
    require(all(name in config for name in PROTOCOL), f"Missing protocol metadata: {folder}")
    draws = config["bootstrap"]
    require(isinstance(draws, int) and draws >= 0, f"Invalid bootstrap setting: {folder}")
    require(isinstance(manifest.get("environment"), dict) and isinstance(report.get("versions"), dict), f"Missing source environment: {folder}")
    require(all(m == "latency" or m in report.get("provenance", {}) for m in metrics), f"Missing model provenance: {folder}")
    summaries = index(report["summary"], ("method", "res", "metric"), "summary")
    paired = index(report["paired_deltas"], ("method", "res", "reference_method", "metric"), "paired delta")
    groups = defaultdict(list)
    for row in rows.values():
        groups[(row["method"], row["res"])].append(row)
    require(set(summaries) == {(m, r, metric) for m, r in groups for metric in metrics}, f"Summary coverage mismatch: {folder}")
    expected_paired = set()
    for (method, res), group in groups.items():
        for metric in metrics:
            stat = summaries[(method, res, metric)]
            count = len({r["prompt_id"] for r in group})
            estimate = None if metric == "is" else mean_by_prompt((r["prompt_id"], r["metrics"][metric]) for r in group)[0]
            check_stat(stat, estimate, count, draws, (method, res, metric))
            if metric == "is":
                require(stat.get("n_images") == len(group), f"IS image coverage mismatch: {method}/{res}")
        reference = config["reference_method"]
        require((reference, res) in groups, f"Reference method absent at {res}")
        if method == reference:
            continue
        ref = {(r["prompt_id"], str(r["seed"])): r for r in groups[(reference, res)]}
        shared = [(r, ref[(r["prompt_id"], str(r["seed"]))]) for r in group if (r["prompt_id"], str(r["seed"])) in ref]
        if not shared:
            continue
        count = len({r["prompt_id"] for r, _ in shared})
        for metric in metrics:
            identity = (method, res, reference, metric)
            expected_paired.add(identity)
            require(identity in paired, f"Missing paired statistic: {identity}")
            stat = paired[identity]
            require(stat.get("delta_definition") == "method_minus_reference", f"Wrong delta convention: {identity}")
            estimate = None if metric == "is" else mean_by_prompt((r["prompt_id"], r["metrics"][metric] - b["metrics"][metric]) for r, b in shared)[0]
            check_stat(stat, estimate, count, draws, identity)
            if metric == "is":
                require(stat.get("n_images") == len(shared), f"Paired IS coverage mismatch: {identity}")
    require(set(paired) == expected_paired, f"Unexpected paired rows: {folder}")
    flat = [{**{k: r[k] for k in FIELDS}, **r["metrics"]} for r in rows.values()]
    check_csv(folder / "extra_per_image.csv", flat, FIELDS[:4])
    check_csv(folder / "extra_summary.csv", report["summary"], ("method", "res", "metric"))
    check_csv(folder / "extra_paired_deltas.csv", report["paired_deltas"], ("method", "res", "reference_method", "metric"))
    return folder, manifest, report, rows, identities, summaries, paired


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def merge(sources, destination):
    start = time.monotonic()
    destination = Path(destination).resolve()
    require(not destination.exists(), f"Refusing existing output directory: {destination}")
    require(len(sources) >= 2, "Supply at least two source reports")
    loaded = [load_source(folder) for folder in sources]
    require(len({entry[0] for entry in loaded}) == len(loaded), "Duplicate source directory")
    _, first_manifest, first, rows, identities, _, _ = loaded[0]
    combined_rows = {k: {**copy.deepcopy(r), "metrics": {}} for k, r in rows.items()}
    combined_summary, combined_paired, provenance, metric_sources, source_records = {}, {}, {}, defaultdict(list), []
    metrics = []
    for number, (folder, manifest, report, source_rows, source_ids, summaries, paired) in enumerate(loaded):
        require(source_ids == identities, f"Prompt/method/resolution/seed/path/image identity mismatch: {folder}")
        require({k: report["config"][k] for k in PROTOCOL} == {k: first["config"][k] for k in PROTOCOL}, f"Protocol mismatch: {folder}")
        source_id = f"source_{number}"
        source_records.append({"id": source_id, "directory": str(folder),
            "identity_manifest_sha256": sha256(folder / IDENTITY_FILE), "report_files": manifest["report_files"],
            "environment": manifest["environment"], "versions": report["versions"],
            "config": report["config"], "provenance": report["provenance"],
            "source_hashes": manifest.get("source_hashes", {}), "seconds": report.get("seconds")})
        for metric in report["metrics"]:
            metric_sources[metric].append(source_id)
            definition = {k: v for k, v in report.get("provenance", {}).get(metric, {}).items() if k != "load_seconds"}
            if metric in metrics:
                previous = {k: v for k, v in provenance.get(metric, {}).items() if k != "load_seconds"}
                require(definition == previous, f"Conflicting metric provenance: {metric}")
            else:
                metrics.append(metric)
                if metric in report.get("provenance", {}):
                    provenance[metric] = copy.deepcopy(report["provenance"][metric])
        for identity, row in source_rows.items():
            target = combined_rows[identity]["metrics"]
            for metric, value in row["metrics"].items():
                require(metric not in target or target[metric] == value, f"Conflicting per-image score: {identity}/{metric}")
                target[metric] = value
        for target, incoming in ((combined_summary, summaries), (combined_paired, paired)):
            for identity, stat in incoming.items():
                require(identity not in target or target[identity] == stat, f"Conflicting aggregate statistic: {identity}")
                target[identity] = copy.deepcopy(stat)
    common_versions = {package: version for package, version in first["versions"].items()
                       if all(entry[2]["versions"].get(package) == version for entry in loaded)}
    report = {"config": {**first["config"], "out": str(destination), "metrics": ",".join(metrics)},
        "metrics": metrics, "versions": common_versions, "provenance": provenance,
        "errors": [], "summary": [combined_summary[k] for k in sorted(combined_summary)],
        "paired_deltas": [combined_paired[k] for k in sorted(combined_paired)],
        "per_image": [combined_rows[k] for k in sorted(combined_rows)],
        "uncertainty": first["uncertainty"], "paired_note": first["paired_note"], "is_note": first.get("is_note"),
        "complete": True, "seconds": None,
        "merge": {"policy": "strict_input_sha256_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "sources": source_records, "metric_sources": dict(metric_sources),
                  "aggregate_policy": "Source statistics and intervals preserved; scalar points/coverage checked; IS points not recomputed without logits",
                  "versions_policy": "Only common source versions at top level; full environments retained per source",
                  "seconds": time.monotonic() - start}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".staging-", dir=destination.parent))
    write_csv(staging / "extra_per_image.csv", [{**{k: r[k] for k in FIELDS}, **r["metrics"]} for r in report["per_image"]])
    write_csv(staging / "extra_summary.csv", report["summary"])
    write_csv(staging / "extra_paired_deltas.csv", report["paired_deltas"])
    write_json(staging / "extra_metrics.json", report)
    write_json(staging / IDENTITY_FILE, {"policy": POLICY, "phase": "complete", "identities": first_manifest["identities"],
        "environment": {"by_source": {row["id"]: row["environment"] for row in source_records}},
        "report_files": {name: sha256(staging / name) for name in REPORT_FILES},
        "source_hashes": {"repro/merge_extra_reports.py": sha256(__file__)}})
    staging.rename(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("sources", nargs="+", type=Path)
    args = parser.parse_args()
    report = merge(args.sources, args.out)
    print(f"Merged {len(report['per_image'])} identical inputs, {len(report['metrics'])} metrics into {args.out}")


if __name__ == "__main__":
    main()
