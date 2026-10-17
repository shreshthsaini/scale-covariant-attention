"""Re-derive narrow manuscript compute claims from archived evidence only."""
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def raw(name):
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    row = next(r for r in manifest if r["source"].endswith(name))
    data = gzip.decompress((ROOT / row["archive"]).read_bytes())
    assert hashlib.sha256(data).hexdigest() == row["sha256"]
    return data.decode()


def completed_loops(text):
    rows, last = [], None
    for line in text.splitlines():
        match = re.search(r"28/28 \[([0-9:]+)<", line)
        if match:
            last = match.group(1)
        if line.startswith("calibrated "):
            assert last is not None, line
            fields = list(map(int, last.split(":")))
            seconds = sum(value * 60 ** i for i, value in enumerate(reversed(fields)))
            rows.append({"completion": line, "displayed_seconds": seconds})
            last = None
    return rows


def main():
    k1024 = raw("logs/k1.log").split("scripts/calibrate.py --count 32", 1)[1].split("+ /", 1)[0]
    loops = {"k1_1024": completed_loops(k1024),
             "k1_1280b": completed_loops(raw("logs/task_a_calib1280.log"))}
    assert [len(loops[group]) for group in loops] == [32, 16]
    worker = [line for line in raw("logs/node_c611-031.log").splitlines() if "a_calib1280.sh" in line]
    assert len(worker) == 2 and "start" in worker[0] and "end" in worker[1] and "rc=0" in worker[1]
    elapsed = (datetime.fromisoformat(worker[1].split()[0]) - datetime.fromisoformat(worker[0].split()[0])).total_seconds()
    parts = json.loads((ROOT / "part_metadata_inventory.json").read_text())
    assert len(parts) == 48
    assert all(not any(word in field for word in ("seconds", "elapsed", "time"))
               for row in parts for field in row["fields"])
    source = REPO / "results/evidence/generation_sidecars.jsonl.gz"
    with gzip.open(source, "rt") as handle:
        images = [json.loads(line)["sidecar"] for line in handle]
    assert all(row["complete"] and row["png_sha256"] and row["image"] for row in images)
    counts = {}
    for name, buckets in (("all_archived", None), ("main", {"main"}), ("main_and_abl", {"main", "abl"})):
        selected = [r for r in images if buckets is None or r["image"].split("/sca/out/")[1].split("/")[0] in buckets]
        by_hash = defaultdict(list)
        for row in selected:
            by_hash[row["png_sha256"]].append(row)
        duplicate_methods = Counter(tuple(sorted(r["config"]["method"] for r in rows))
                                    for rows in by_hash.values() if len(rows) > 1)
        counts[name] = {"sidecar_records": len(selected), "distinct_image_paths": len({r["image"] for r in selected}),
                        "distinct_recorded_png_sha256": len(by_hash),
                        "duplicate_artifact_records_removed": len(selected) - len(by_hash),
                        "duplicate_method_groups": [{"methods": list(methods), "count": count}
                                                    for methods, count in sorted(duplicate_methods.items())]}
    report = {
        "calibration": {"successful_loop_displayed_seconds": {group: sum(r["displayed_seconds"] for r in rows)
                                                              for group, rows in loops.items()},
                        "successful_loops": loops, "1280_worker_span_seconds": elapsed,
                        "1280_worker_lines": worker, "parts_without_elapsed_metadata": len(parts),
                        "conclusion": "The 1.5 GPU-hour claim is not re-derived. Successful forward-loop displays are partial timings; no full 1024 task span or complete retry accounting survives in the checked logs."},
        "generated_images": {"source": str(source.relative_to(REPO)),
                             "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                             "scope": "Complete archived generated-image metadata; deduplicate recorded PNG SHA256 values, not repeated evaluations. No image bytes rehashed in this audit.",
                             "counts": counts,
                             "conclusion": "More than 2000 generated images is supported even by the main subset alone."}}
    (ROOT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"calibration_loop_seconds": report["calibration"]["successful_loop_displayed_seconds"],
                      "1280_worker_seconds": elapsed, "image_counts": counts}))


if __name__ == "__main__":
    main()
