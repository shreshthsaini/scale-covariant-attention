#!/bin/bash
# Assemble eval/prog195_pairs as an r7_decide evaluation directory (label sca_v2_prog, references sca_v2 and AReA,
# 195 prompts) from the cached evaluations; CPU only. Method owner, 2026-11-07.
set -euo pipefail
S=/scratch/09032/saini_2/sca; E=$S/eval/prog195_pairs; PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python
mkdir -p $E/extra $E/base $E/paired $E/corrected_fidp_ci
$PY - $E <<'PYEOF'
import csv, sys
E = sys.argv[1]
S = "/scratch/09032/saini_2/sca/eval"
def cat(srcs, out, keep):
    rows, fields = [], None
    for f in srcs:
        r = list(csv.DictReader(open(f)))
        fields = fields or list(r[0].keys())
        rows += [x for x in r if x["method"] in keep]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(out, len(rows))
cat([f"{S}/a4k195_main_sca_v2/extra/extra_per_image.csv", f"{S}/prog195_sca_v2/extra/extra_per_image.csv"],
    f"{E}/extra/extra_per_image.csv", {"sca_v2", "area", "sca_v2_prog"})
cat([f"{S}/a4k195_main_sca_v2/base/per_image.csv", f"{S}/prog195_sca_v2/base/per_image.csv"],
    f"{E}/base/per_image.csv", {"sca_v2", "area", "sca_v2_prog"})
rows, fields = [], None
for f in (f"{E}/cfid_vs_sca_v2/paired_deltas.csv", f"{E}/cfid_vs_area/paired_deltas.csv"):
    r = list(csv.DictReader(open(f))); fields = fields or list(r[0].keys()); rows += r
with open(f"{E}/corrected_fidp_ci/paired_deltas.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
PYEOF
for p in sca_v2_prog_vs_area sca_v2_prog_vs_sca_v2 sca_v2_vs_area; do cp $E/$p.csv $E/paired/; done
cd $S/code/d440b71
$PY scripts/dup_rate.py --dets $S/eval/dup_detect_20261027 $S/eval/a4k195_main_sca_v2/det $S/eval/prog195_sca_v2/det --subjects results/dup_audit_20261027/subjects_a4k195.json --ids /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl --thr 0.35 --kinds person,animal --methods sca_v2,sca_v2_prog,area --ref sca_v2 --per-image $E/dup_per_image_people.csv --out $E/dup_rates_people.csv > $E/dup_rates_people.txt
$PY scripts/dup_rate.py --dets $S/eval/dup_detect_20261027 $S/eval/a4k195_main_sca_v2/det $S/eval/prog195_sca_v2/det --subjects results/dup_audit_20261027/subjects_a4k195.json --ids /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl --thr 0.35 --methods sca_v2,sca_v2_prog,area --ref sca_v2 --per-image $E/dup_per_image_any.csv --out $E/dup_rates_any.csv > $E/dup_rates_any.txt
$PY - > $E/timing.txt <<'PYEOF'
import glob, json, statistics
ids = {json.loads(l)["id"] for l in open("/scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl") if l.strip()}
for name, d in (("sca_v2_prog", "/scratch/09032/saini_2/sca/out/dupfix_prog/sca_v2_prog/4096x4096"), ("sca_v2", "/scratch/09032/saini_2/sca/out/dupfix/sca_v2/4096x4096"), ("area", "/scratch/09032/saini_2/sca/out/main/area/4096x4096")):
    t = [json.load(open(f)) for f in glob.glob(d + "/*.json")]
    t = [x["seconds"] for x in t if x.get("config", {}).get("prompt_id") in ids]
    print(name, "n", len(t), "median s", round(statistics.median(t), 2) if t else None)
PYEOF
cat $E/timing.txt; date -u +%FT%TZ > $E/DONE
