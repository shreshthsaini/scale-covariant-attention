#!/bin/bash
# E3 summary: median seconds per (res, method) into eval/latency_curve.csv
export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton OMP_NUM_THREADS=8
unset CROPE_TRANSFORMER CROPE_ROPE
while ls /scratch/09032/saini_2/sca/taskq/pending /scratch/09032/saini_2/sca/taskq/running 2>/dev/null | grep -q -E '^(a_)?y00[0-9]_latency_'; do sleep 60; done
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I - <<'EOF'
import csv, glob, json, statistics
rows = []
for f in sorted(glob.glob("/scratch/09032/saini_2/sca/out/latency_curve/*/*/*.json")):
    d = json.load(open(f)); c = d["config"]
    if c["prompt_id"] == "sca_latency_warmup": continue
    rows.append((c["res"], c["method"], c["prompt_id"], d["seconds"], d.get("peak_memory_bytes"), d.get("first_call_in_process")))
with open("/scratch/09032/saini_2/sca/eval/latency_curve.csv", "w", newline="") as h:
    w = csv.writer(h); w.writerow(["res", "method", "prompt_id", "seconds", "peak_memory_bytes", "first_call_in_process"]); w.writerows(rows)
groups = {}
for r in rows: groups.setdefault((r[0], r[1]), []).append(r[3])
for k, v in sorted(groups.items()): print(*k, "n", len(v), "median_s", round(statistics.median(v), 1))
EOF
