#!/usr/bin/env python3
"""Progressive SCA (sca_v2_prog) headline tasks for the submission (2026-11-07; lead relay of the user's decision).

Writes into taskq/pending, gens prefixed 0a1_/0a2_ so they sort ahead of everything; evaluations 0a0_* are parked in
taskq/hold_dup_eval and released by the last generation task of their group.
  seeds  sca_v2_prog seeds 1 and 2 on the 195 Aesthetic-4K prompts at 4096^2 (out/dupfix_prog_seed{s}), chunks of 15;
         evaluation eval/prog195_s{s} -> results/prog195_s{s}: sca_v2_prog vs AReA and sca_v2 at the same seed, the
         a4k195 metric set (IR, aesthetic, HPSv2, HPSv2.1, CLIP-IQA, ARNIQA, IS, CLIP, PickScore, MUSIQ, MUSIQ crop,
         FID, KID, patch FID, patch KID), corrected FID/patch FID (bootstrap 0; intervals on CPU afterwards), the
         person/animal duplicate detector, paired bootstraps (2000 draws, seed 0), time medians. AReA and sca_v2 metrics
         come from the cache of eval/a4k195_seed{s}_sca_v2.
  hrgen  sca_v2_prog on HRGen-1K (1000 prompts, FLUX.1-dev, out/dupfix_prog_hrgen), chunks of 25; evaluation
         eval/prog_hrgen1k -> results/prog_hrgen1k against AReA and sca_v2 (the HRGen-1K metric set: IR, aesthetic,
         HPSv2, HPSv2.1, CLIP-IQA, ARNIQA, IS, CLIP, PickScore, MUSIQ; caches from eval/hrgen1k_dev_sca_v2).
  latency one batch-1 sweep on one GPU: sca_v2_prog, HRDiT (author profile) and AReA at 4096^2 on latency4_warmup.jsonl
         (row 0 an excluded warm-up, rows 1 to 3 timed, median), commands cloned from the review-round sweep
         (taskq/failed/b0_r4_latency_sweep.sh) -> results/prog_latency.
FLUX.1-Krea-dev is not run: scripts/sample_progressive.py supports --model dev only.
Usage: slurm/prog_headline_tasks.py [--dry]
"""

import argparse
import json
from pathlib import Path

S = "/scratch/09032/saini_2/sca"
P = "/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention"
PY = "/work/09032/saini_2/vista/library/uv-envs/sca/bin/python"
PYM = "/work/09032/saini_2/vista/library/uv-envs/sca-metrics/bin/python"
CR = "/work/09032/saini_2/vista/Projects/Exploration/constancy_rope"
A4K = "/scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl"
REFS = "/scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl"
HRGEN = f"{S}/hrgen1k.jsonl"
PROG_CODE = f"{S}/code/7bd64c0"           # the tree of out/dupfix_prog (seed 0)
EVAL_CODE = f"{S}/code/023918f"           # detector and rate scripts
ENV = ("export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 "
       "TRANSFORMERS_OFFLINE=1 XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton "
       "OMP_NUM_THREADS=8\nunset CROPE_TRANSFORMER CROPE_ROPE\nexport CC=/usr/bin/gcc CXX=/usr/bin/g++\n")
PROG_FLAGS = ("--pi-table /scratch/09032/saini_2/sca/calib/k1_1024.npz --gate-sigma 0.89 --label sca_v2_prog --bridge-cells 8 "
              "--bridge-k 16 --win-sched 0.88:128,0.86:112,0.84:96,0.82:80 --win-pi-match --win-pi-interp")
LINKER = """{py} -I - "{prompts}" "{spec}" "4096x4096" "$E/root" <<'PYEOF'
import glob, json, os, sys
prompts, spec, res, link = sys.argv[1:5]
ids = [json.loads(l)["id"] for l in open(prompts) if l.strip()]
bad = 0
for item in spec.split(";"):
    method, seed, dirs = item.split("|")
    n = 0
    for i in ids:
        pngs = [p for d in dirs.split(",") for p in glob.glob(f"{{d}}/{{glob.escape(i)}}_*seed{{seed}}.png")]
        if len(pngs) != 1 or not json.load(open(pngs[0][:-4] + ".json")).get("complete"):
            bad += 1
            continue
        d = os.path.join(link, method, res); os.makedirs(d, exist_ok=True)
        for src in (pngs[0], pngs[0][:-4] + ".json"):
            dst = os.path.join(d, os.path.basename(src))
            if not os.path.lexists(dst):
                os.symlink(src, dst)
        n += 1
    print("GATE", method, n, "of", len(ids), flush=True)
sys.exit(1 if bad else 0)
PYEOF
[ $? -eq 0 ] || {{ echo "GATE failed"; exit 6; }}
"""


def write(path, text, dry):
    path = Path(path)
    if path.exists():
        print("exists, kept:", path)
        return
    print("write", path)
    if not dry:
        path.write_text(text)
        path.chmod(0o755)


def gens(names_rows, prompts, out, seed, marker_dir, ename, dry):
    pat = "|".join(n[:-3] for n, _, _ in names_rows)
    for name, a, b in names_rows:
        text = (f"#!/bin/bash\n# SCA_MIN_SECONDS: 3600\n# progressive SCA headline ({name}): sca_v2_prog seed {seed} on "
                f"{Path(prompts).name} [{a}, {b}), the recipe and tree of out/dupfix_prog\n" + ENV
                + f"cd {PROG_CODE}\n"
                + f"C=\"{PY} scripts/sample_progressive.py --prompts {prompts} --seed {seed} --out {out} --start {a} --end {b} {PROG_FLAGS}\"\n"
                + "$C || { sleep 30; $C; } || exit $?\n"
                + f"G={marker_dir}; mkdir -p $G && touch $G/{name} || echo \"WARNING could not write $G/{name}\"\n"
                + f"if [ \"$(ls $G 2>/dev/null | grep -c -E '^({pat})[.]sh$')\" -eq {len(names_rows)} ]; then\n"
                + f"  mv {S}/taskq/hold_dup_eval/{ename} {S}/taskq/pending/ 2>/dev/null && echo \"released {ename}\"\nfi\nexit 0\n")
        write(f"{S}/taskq/pending/{name}", text, dry)
    return pat, len(names_rows)


def guard(pat, n, marker_dir):
    return (f"if [ \"$(ls {marker_dir} 2>/dev/null | grep -c -E '^({pat})[.]sh$')\" -lt {n} ]; then "
            f"echo \"generation incomplete ({marker_dir})\"; exit 5; fi\n")


def seed_eval(seed, dry):
    out = f"{S}/out/dupfix_prog_seed{seed}/sca_v2_prog/4096x4096"
    starts = list(range(0, 195, 15))
    rows = [(f"0a1_pgs{seed}_{a:04d}_{min(a + 15, 195):04d}.sh", a, min(a + 15, 195)) for a in starts]
    E = f"{S}/eval/prog195_s{seed}"
    ename = f"0a0_eval_prog195_s{seed}.sh"
    pat, n = gens(rows, A4K, out, seed, f"{E}/gen_done", ename, dry)
    src = f"{S}/eval/a4k195_seed{seed}_sca_v2"
    spec = (f"sca_v2_prog|{seed}|{out};area|{seed}|{S}/out/seed{seed}/area/4096x4096;"
            f"sca_v2|{seed}|{S}/out/dupfix_seed{seed}/sca_v2/4096x4096")
    R = f"{P}/results/prog195_s{seed}"
    text = (f"#!/bin/bash\n# SCA_MIN_SECONDS: 5400\n# progressive SCA headline: sca_v2_prog seed {seed} vs AReA and sca_v2 (seed {seed}) on the "
            f"195 prompts at 4096^2 (slurm/prog_headline_tasks.py)\n" + ENV + "set -uo pipefail\n"
            + f"E={E}; R={R}\nexport E\n" + guard(pat, n, f"{E}/gen_done")
            + "mkdir -p $E/paired $E/extra $E/base/features $R\n"
            + "run(){ \"$@\" 2>&1 | tee -a $E/run.log; local rc=${PIPESTATUS[0]}; if [ $rc -ne 0 ]; then echo \"step failed rc=$rc: $*\"; exit $rc; fi; }\n"
            + LINKER.format(py=PY, prompts=A4K, spec=spec)
            + f"[ -f {src}/extra/extra_cache.jsonl ] && [ ! -f $E/extra/extra_cache.jsonl ] && cp -p {src}/extra/extra_cache.jsonl $E/extra/\n"
            + f"[ -f {src}/base/evaluation_cache.json ] && [ ! -f $E/base/evaluation_cache.json ] && cp -p {src}/base/evaluation_cache.json $E/base/ && cp -p {src}/base/features/*.npz $E/base/features/ 2>/dev/null\n"
            + f"cd {EVAL_CODE}\n"
            + f"run {PY} scripts/dup_detect.py --subjects results/dup_audit_20261027/subjects_a4k195.json --device cuda --threads 8 --sides 800,1333 --out $E/det --dirs sca_v2_prog={out}\n"
            + f"{PY} scripts/dup_rate.py --dets {S}/eval/dup_detect_20261027 {src}/det $E/det --subjects results/dup_audit_20261027/subjects_a4k195.json --thr 0.35 --kinds person,animal --methods sca_v2_prog,area,sca_v2 --ref area --per-image $E/dup_per_image_people.csv --out $E/dup_rates_people.csv | tee -a $E/run.log\n"
            + f"run {PYM} {P}/scripts/eval_extra.py --root $E/root --prompts {A4K} --out $E/extra --metrics ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa,is --reference-method sca_v2_prog --device cuda --batch-size 16 --threads 8\n"
            + f"run {PY} {CR}/scripts/eval.py --root $E/root --prompts {A4K} --references {REFS} --reference-method sca_v2_prog --res 4096x4096 --out $E/base --metrics clip,pickscore,musiq,fid,kid,patch_fid,patch_kid --device cuda\n"
            + "mkdir -p $E/corrected_fidp/features && cp -p $E/base/features/*.npz $E/corrected_fidp/features/ && cp -p $E/base/evaluation_cache.json $E/corrected_fidp/ || { echo \"feature copy failed\"; exit 7; }\n"
            + f"(cd {P} && WANDB_MODE=disabled run {PY} repro/evaluate_corrected.py --root $E/root --prompts {A4K} --references {REFS} --reference-method sca_v2_prog --reference-clusters images --res 4096x4096 --out $E/corrected_fidp --metrics fid,kid,patch_fid,patch_kid --device cuda --bootstrap 0 --threads 8 --no-wandb)\n"
            + "for pair in sca_v2_prog:area sca_v2_prog:sca_v2 sca_v2:area; do a=${pair%%:*}; b=${pair#*:}\n"
            + f"  run {PY} -I {P}/scripts/paired_bootstrap.py --method $a --reference $b --draws 2000 --out $E/paired/${{a}}_vs_${{b}}.csv $E/extra/extra_per_image.csv $E/base/per_image.csv\ndone\n"
            + timing(out, A4K, seed)
            + "cp $E/base/summary.csv $R/base_summary.csv && cp $E/extra/extra_summary.csv $R/extra_summary.csv && cp $E/paired/*.csv $R/ && cp $E/dup_rates_people.csv $E/timing.txt $R/\n"
            + "mkdir -p $R/corrected_fidp && cp $E/corrected_fidp/summary.csv $E/corrected_fidp/paired_deltas.csv $R/corrected_fidp/\n"
            + "(cd $R && sha256sum $(find . -name '*.csv' -o -name '*.txt' | sort) > SHA256SUMS)\n"
            + f"date -u +%FT%TZ > $E/DONE; echo \"DONE prog195_s{seed}\"\n")
    write(f"{S}/taskq/hold_dup_eval/{ename}", text, dry)


def timing(out, prompts, seed):
    return f"""{PY} - <<'PYEOF' | tee $E/timing.txt
import glob, json, os, statistics
root = os.environ.get("E", "") + "/root"
ids = {{json.loads(l)["id"] for l in open("{prompts}") if l.strip()}}
for m in sorted(os.listdir(root)):
    t = [json.load(open(f)) for f in glob.glob(f"{{root}}/{{m}}/4096x4096/*.json")]
    t = [x["seconds"] for x in t if x.get("config", {{}}).get("prompt_id") in ids and "seconds" in x]
    print(m, "n", len(t), "median s", round(statistics.median(t), 2) if t else None)
PYEOF
"""


def hrgen_eval(dry):
    out = f"{S}/out/dupfix_prog_hrgen/sca_v2_prog/4096x4096"
    rows = [(f"0a2_pghk_{a:04d}_{min(a + 25, 1000):04d}.sh", a, min(a + 25, 1000)) for a in range(0, 1000, 25)]
    E = f"{S}/eval/prog_hrgen1k"
    ename = "0a0_eval_prog_hrgen1k.sh"
    pat, n = gens(rows, HRGEN, out, 0, f"{E}/gen_done", ename, dry)
    src = f"{S}/eval/hrgen1k_dev_sca_v2"
    spec = (f"sca_v2_prog|0|{out};area|0|{S}/out/hrgen/area/4096x4096;sca_v2|0|{S}/out/dupfix_hrgen/sca_v2/4096x4096")
    R = f"{P}/results/prog_hrgen1k"
    text = (f"#!/bin/bash\n# SCA_MIN_SECONDS: 7200\n# progressive SCA headline: sca_v2_prog vs AReA and sca_v2 on HRGen-1K (1000 prompts, "
            f"FLUX.1-dev, 4096^2; slurm/prog_headline_tasks.py)\n" + ENV + "set -uo pipefail\n"
            + f"E={E}; R={R}\nexport E\n" + guard(pat, n, f"{E}/gen_done")
            + "mkdir -p $E/paired $E/extra $E/base $R\n"
            + "run(){ \"$@\" 2>&1 | tee -a $E/run.log; local rc=${PIPESTATUS[0]}; if [ $rc -ne 0 ]; then echo \"step failed rc=$rc: $*\"; exit $rc; fi; }\n"
            + LINKER.format(py=PY, prompts=HRGEN, spec=spec)
            + f"[ -f {src}/extra/extra_cache.jsonl ] && [ ! -f $E/extra/extra_cache.jsonl ] && cp -p {src}/extra/extra_cache.jsonl $E/extra/\n"
            + f"[ -f {src}/base/evaluation_cache.json ] && [ ! -f $E/base/evaluation_cache.json ] && cp -p {src}/base/evaluation_cache.json $E/base/\n"
            + f"run {PYM} {P}/scripts/eval_extra.py --root $E/root --prompts {HRGEN} --out $E/extra --metrics ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa,is --reference-method sca_v2_prog --device cuda --batch-size 16 --threads 8\n"
            + f"run {PY} {CR}/scripts/eval.py --root $E/root --prompts {HRGEN} --reference-method sca_v2_prog --res 4096x4096 --out $E/base --metrics clip,pickscore,musiq --device cuda\n"
            + "for pair in sca_v2_prog:area sca_v2_prog:sca_v2 sca_v2:area; do a=${pair%%:*}; b=${pair#*:}\n"
            + f"  run {PY} -I {P}/scripts/paired_bootstrap.py --method $a --reference $b --draws 2000 --out $E/paired/${{a}}_vs_${{b}}.csv $E/extra/extra_per_image.csv $E/base/per_image.csv\ndone\n"
            + timing(out, HRGEN, 0)
            + "cp $E/base/summary.csv $R/base_summary.csv && cp $E/extra/extra_summary.csv $R/extra_summary.csv && cp $E/paired/*.csv $E/timing.txt $R/\n"
            + "(cd $R && sha256sum $(find . -name '*.csv' -o -name '*.txt' | sort) > SHA256SUMS)\n"
            + "date -u +%FT%TZ > $E/DONE; echo \"DONE prog_hrgen1k\"\n")
    write(f"{S}/taskq/hold_dup_eval/{ename}", text, dry)


def latency(dry):
    sweep = Path(f"{S}/taskq/failed/b0_r4_latency_sweep.sh").read_text().splitlines()
    area = next(l for l in sweep if l.startswith('step "4096x4096 area"'))
    hrdit = next(l for l in sweep if l.startswith('step "4096x4096 hrdit_dev_author"'))
    summ = sweep[sweep.index(next(l for l in sweep if l.startswith("nvidia-smi -i $GPU_ID --query-compute-apps=pid,used_memory --format=csv >"))):]
    summ = [l.replace("results/review1_latency", "results/prog_latency") for l in summ]
    prog = (f"step \"4096x4096 sca_v2_prog\" '(cd {PROG_CODE} && {PY} scripts/sample_progressive.py --prompts {S}/latency4_warmup.jsonl "
            f"--seed 0 --out $RUN/4096x4096/sca_v2_prog --start 0 --end 4 {PROG_FLAGS})'")
    text = ("#!/bin/bash\n# SCA_MIN_SECONDS: 3600\n# progressive SCA headline: one clean batch-1 latency sweep on ONE GPU: sca_v2_prog, HRDiT "
            "(author profile) and AReA at 4096^2, each in its own process on latency4_warmup.jsonl (row 0 an excluded warm-up, rows "
            "1..3 timed, median); commands cloned from taskq/failed/b0_r4_latency_sweep.sh (slurm/prog_headline_tasks.py)\n" + ENV
            + "set -uo pipefail\n"
            + f"C=/scratch/09032/saini_2/sca/code/f479459\nRUN={S}/out/prog_latency/$(date +%Y%m%dT%H%M%S)_$(hostname -s)\n"
            + f"R={P}/results/prog_latency\nmkdir -p $RUN $R\ncd $C\n"
            + "GPU_ID=${CUDA_VISIBLE_DEVICES:-0}; GPU_ID=${GPU_ID%%,*}\n"
            + "nvidia-smi -i $GPU_ID --query-gpu=name,uuid,driver_version,utilization.gpu,memory.used,clocks.max.sm --format=csv | tee $RUN/gpu.csv\n"
            + "nvidia-smi -i $GPU_ID --query-compute-apps=pid,used_memory --format=csv | tee $RUN/gpu_apps_before.csv\n"
            + "[ \"$(wc -l < $RUN/gpu_apps_before.csv)\" -gt 1 ] && echo \"WARNING: other processes hold this GPU; timings are not clean\"\n"
            + "FAIL=0\nstep(){ local tag=$1; shift; echo \"$(date -u +%FT%TZ) start $tag\"; eval \"$@\"; local rc=$?\n"
            + "  [ $rc -eq 0 ] || { echo \"FAILED $tag rc=$rc\"; FAIL=$((FAIL+1)); }; echo \"$(date -u +%FT%TZ) end $tag\"; }\n"
            + prog + "\n" + hrdit + "\n" + area + "\n" + "\n".join(summ) + "\n")
    write(f"{S}/taskq/pending/0a0_latency_prog_hrdit_area.sh", text, dry)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry", action="store_true")
    p.add_argument("--only", default="latency,seeds,hrgen")
    a = p.parse_args()
    only = a.only.split(",")
    if "latency" in only:
        latency(a.dry)
    if "seeds" in only:
        for s in (1, 2):
            seed_eval(s, a.dry)
    if "hrgen" in only:
        hrgen_eval(a.dry)


if __name__ == "__main__":
    main()
