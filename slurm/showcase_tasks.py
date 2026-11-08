#!/usr/bin/env python3
"""GPU showcase campaign (HANDOVER Section 32): spool tasks for the 40 showcase prompts at seed 0.

Groups (task names sort ahead of every other spool task with the 00s prefix; lower digit runs first):
  flux1   FLUX.1-dev: progressive SCA (sca_v2_prog, the paper's headline) at 4096^2; single-pass SCA (sca_v2) at
          5120^2 and at 4096x6144 (landscape prompts) or 6144x4096 (portrait prompts); native full attention at 1024^2
          and 1024x1536 / 1536x1024 (Lanczos reference). Tree code/7bd64c0, the tree of the paper's sca_v2 and
          sca_v2_prog runs, with their exact flags.
  area    AReA on FLUX.1-dev at the same three sizes (4096^2, 5120^2, aspect), AReA flux1 defaults (run_area.py).
  flux2   FLUX.2-dev: single-pass SCA at 4096^2 and 5120^2 (the review-round flux2_sca_v2 flags), native 1024^2.
  flux2full  FLUX.2-dev full attention at 4096^2 and 5120^2.
  qwen    Qwen-Image: single-pass SCA at 4096^2 and 5312^2 (4x native), native 1328^2.
  qwenfull   Qwen-Image full attention at 4096^2 and 5312^2.
Outputs: /scratch/09032/saini_2/sca/out/showcase/<model>_<method>/<HxW>/ (PNG + JSON sidecar per prompt).
Usage: slurm/showcase_tasks.py GROUP [GROUP ...] [--code DIR] [--dry]
"""

import argparse
import json
from pathlib import Path

S = "/scratch/09032/saini_2/sca"
P = "/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention"
PY = "/work/09032/saini_2/vista/library/uv-envs/sca/bin/python"
OUT = f"{S}/out/showcase"
PROMPTS = f"{S}/prompts/showcase40.jsonl"
LAND = f"{S}/prompts/showcase40_land.jsonl"
PORT = f"{S}/prompts/showcase40_port.jsonl"
FLUX1_CODE = f"{S}/code/7bd64c0"
AREA_CODE = f"{S}/code/7bd64c0r"     # 7bd64c0 plus repro/evaluator_snapshot, which scripts/run_area.py imports
ENV = ("export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 "
       "TRANSFORMERS_OFFLINE=1 XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton "
       "OMP_NUM_THREADS=8\nunset CROPE_TRANSFORMER CROPE_ROPE\nexport CC=/usr/bin/gcc CXX=/usr/bin/g++\n"
       "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n")
SCA_V2 = ("--pi-table /scratch/09032/saini_2/sca/calib/k1_1024.npz --gate-sigma 0.89 --bridge-cells 8 --bridge-k 16 "
          "--win-sched 0.88:128,0.86:112,0.84:96,0.82:80 --win-pi-match --win-pi-interp")
FLUX2_SCA = ("--steps 28 --guidance 4.0 --mu-mode native --method sca --pi-table /scratch/09032/saini_2/sca/calib/flux2_1024.npz "
             "--gate-sigma 0.959 --bridge-cells 8 --bridge-k 16 --win-sched 0.95165:128,0.94355:112,0.93465:96,0.9249:80 "
             "--win-pi-match --win-pi-interp")


def count(path):
    return sum(1 for line in open(path) if line.strip())


GUARD = """G=${{CUDA_VISIBLE_DEVICES:-0}}; G=${{G%%,*}}; NEED={need}; LOG={out}/_gpu; mkdir -p $LOG
free(){{ nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }}
for i in $(seq 1 {polls}); do F=$(free); [ "${{F:-0}}" -ge $((NEED*1024)) ] && break
  [ $i -eq 1 ] && echo "$(date +%FT%T) waiting for $NEED GB free on GPU $G (free $F MiB): other processes hold it"; sleep 30; done
if [ "${{F:-0}}" -lt $((NEED*1024)) ]; then
  echo "$(date +%FT%T) GPU $G still busy (free $F MiB); requeue {name} for another worker"
  nvidia-smi -i $G --query-compute-apps=pid,used_memory --format=csv,noheader >> $LOG/{name}.requeue
  cp "$0" {pending}/{name}; sleep 60; exit 0
fi
snap(){{ echo "$1 $(date +%FT%T) $(hostname -s) gpu$G"; nvidia-smi -i $G --query-compute-apps=pid,used_memory --format=csv,noheader; }}
snap start >> $LOG/{name}.apps
"""


WAIT = """for i in $(seq 1 20); do [ -e {path} ] && break; sleep 30; done
if [ ! -e {path} ]; then echo "$(date +%FT%T) {path} not there yet; requeue {name}"; cp "$0" {pending}/{name}; sleep 60; exit 0; fi
"""


def task(name, minutes, comment, code, cmd, dry, need=60, polls=40, wait=None, post=""):
    """One spool task. need: GB of free GPU memory the task waits for (polls x 30 s), else it puts itself back in
    pending for another worker (other projects' co-tenant processes hold some GPUs). The GPU's compute apps at start
    and end go to out/showcase/_gpu/<task>.apps, so timings measured on a shared GPU can be flagged."""
    text = (f"#!/bin/bash\n# SCA_MIN_SECONDS: {int(minutes * 60)}\n# showcase (slurm/showcase_tasks.py): {comment}\n" + ENV
            + "set -uo pipefail\n" + (WAIT.format(path=wait, name=name, pending=f"{S}/taskq/pending") if wait else "")
            + GUARD.format(need=need, out=OUT, polls=polls, name=name, pending=f"{S}/taskq/pending")
            + f"cd {code}\n" + f"C=\"{cmd}\"\n" + f"echo \"cmd $C\" >> $LOG/{name}.apps\n"
            + "$C || { sleep 30; $C; }; rc=$?\nsnap end >> $LOG/" + name + ".apps\n"
            + "[ $rc -eq 0 ] || exit $rc\n" + post + f"mkdir -p {OUT}/_done && touch {OUT}/_done/{name}\nexit 0\n")
    path = Path(f"{S}/taskq/pending/{name}")
    if Path(f"{OUT}/_done/{name}").exists():
        print("done, kept:", name)
        return
    for d in ("running",):
        hits = list(Path(f"{S}/taskq/{d}").glob(name + "*"))
        if hits:
            print("running, kept:", hits[0])
            return
    old = Path(f"{S}/taskq/failed/{name}")
    if old.exists() and not dry:   # superseded failure: keep the old script for the record, write a fresh one
        sup = Path(f"{S}/taskq/superseded_showcase")
        sup.mkdir(exist_ok=True)
        old.rename(sup / f"{name}.{int(old.stat().st_mtime)}")
    print("write", path)
    if not dry:
        tmp = path.with_name(f".{name}.tmp")
        tmp.write_text(text)
        tmp.chmod(0o755)
        tmp.rename(path)


def chunks(n, size):
    return [(a, min(a + size, n)) for a in range(0, n, size)]


def flux1(dry, code):
    for a, b in chunks(count(PROMPTS), 10):
        task(f"00s1_prog4096_{a:02d}_{b:02d}.sh", 12, "FLUX.1-dev progressive SCA (sca_v2_prog) at 4096^2", code,
             f"{PY} scripts/sample_progressive.py --prompts {PROMPTS} --seed 0 --out {OUT}/flux1_sca_v2_prog/4096x4096 "
             f"--start {a} --end {b} {SCA_V2} --label sca_v2_prog", dry, need=46)
    native = (f"{PY} scripts/sample.py --method full --seed 0 --guidance 3.5 --steps 28 --label full")
    task("00s1_native_flux1_sq.sh", 10, "FLUX.1-dev full attention at 1024^2 (Lanczos reference)", code,
         f"{native} --res 1024x1024 --prompts {PROMPTS} --out {OUT}/flux1_full/1024x1024", dry, need=36)
    for tag, prompts, res in (("land", LAND, "1024x1536"), ("port", PORT, "1536x1024")):
        task(f"00s1_native_flux1_{tag}.sh", 8, f"FLUX.1-dev full attention at {res} (Lanczos reference)", code,
             f"{native} --res {res} --prompts {prompts} --out {OUT}/flux1_full/{res}", dry, need=36)
    sca = f"{PY} scripts/sample.py --method sca --seed 0 --guidance 4.5 {SCA_V2} --label sca_v2"
    for a, b in chunks(count(PROMPTS), 5):
        task(f"00s2_sca5120_{a:02d}_{b:02d}.sh", 15, "FLUX.1-dev single-pass SCA (sca_v2) at 5120^2", code,
             f"{sca} --res 5120x5120 --prompts {PROMPTS} --out {OUT}/flux1_sca_v2/5120x5120 --start {a} --end {b}", dry, need=62)
    for tag, prompts, res in (("land", LAND, "4096x6144"), ("port", PORT, "6144x4096")):
        for a, b in chunks(count(prompts), 5):
            task(f"00s2_sca{tag}_{a:02d}_{b:02d}.sh", 15, f"FLUX.1-dev single-pass SCA (sca_v2) at {res}", code,
                 f"{sca} --res {res} --prompts {prompts} --out {OUT}/flux1_sca_v2/{res} --start {a} --end {b}", dry, need=62)


def area(dry, code):
    run = f"{PY} scripts/run_area.py --seed 0 --guidance 4.5"
    for a, b in chunks(count(PROMPTS), 4):
        task(f"00s6_area4096_{a:02d}_{b:02d}.sh", 20, "AReA on FLUX.1-dev at 4096^2", code,
             f"{run} --res 4096x4096 --prompts {PROMPTS} --out {OUT}/flux1_area/4096x4096 --start {a} --end {b}", dry, need=50)
    for a, b in chunks(count(PROMPTS), 3):
        task(f"00s7_area5120_{a:02d}_{b:02d}.sh", 25, "AReA on FLUX.1-dev at 5120^2", code,
             f"{run} --res 5120x5120 --prompts {PROMPTS} --out {OUT}/flux1_area/5120x5120 --start {a} --end {b}", dry, need=66)
    for tag, prompts, res in (("land", LAND, "4096x6144"), ("port", PORT, "6144x4096")):
        for a, b in chunks(count(prompts), 3):
            task(f"00s7_area{tag}_{a:02d}_{b:02d}.sh", 25, f"AReA on FLUX.1-dev at {res}", code,
                 f"{run} --res {res} --prompts {prompts} --out {OUT}/flux1_area/{res} --start {a} --end {b}", dry, need=66)


QWEN_TABLE = f"{S}/calib/qwen_1344.npz"
# Gate and window schedule matched in unshifted time t (fraction of the trajectory), as the FLUX.2 port's gate rule (8
# of 28 steps global, windows on the next 4): 14 of 50 steps global, then windows 2, 1.75, 1.5, 1.25 x n0 for two steps
# each (sca/qwen.py gate_from_steps(s, 14), win_sched_from_steps(s, 14, (166, 145, 124, 104), 2) at mu native).
QWEN_SCA = ("--method sca --steps 50 --true-cfg 4.0 --mu-mode native --n0 83 --gate-sigma 0.85799 "
            "--win-sched 0.83248:166,0.80553:145,0.77701:124,0.74679:104 --win-pi-match --win-pi-interp")


def qwen(dry, code):
    task("00s0_a_qwen_check.sh", 25, "Qwen-Image port GPU check (exactness at 1328^2, latents, cost at 4096^2 and 5312^2) "
         "and the embedding cache for the calibration and showcase prompts", code,
         f"{PY} scripts/qwen_gpu_check.py --prompts scripts/calib_prompts.jsonl {PROMPTS} --count 16 40 "
         f"--out {OUT}/_checks/qwen_gpu_check.json", dry, need=88)
    for i, (a, b) in enumerate(chunks(16, 4)):
        merge = (f"if [ $(ls {S}/calib/qwen_1344_parts/*.npz 2>/dev/null | wc -l) -ge 16 ] && mkdir {S}/calib/qwen_1344_merge.lock "
                 f"2>/dev/null; then {PY} scripts/calibrate_qwen.py --out {QWEN_TABLE} --report-only || exit 4; fi\n")
        task(f"00s0_b_qcal_{a:02d}_{b:02d}.sh", 30, "Qwen-Image calibration at 1344^2 (16 prompts, ratios 2 and 4)", code,
             f"{PY} scripts/calibrate_qwen.py --out {QWEN_TABLE} --start {a} --end {b} --no-merge", dry, need=70,
             post=merge)
    for a, b in chunks(count(PROMPTS), 5):
        task(f"00s4_a_qsca4096_{a:02d}_{b:02d}.sh", 40, "Qwen-Image single-pass SCA at 4096^2", code,
             f"{PY} scripts/sample_qwen.py {QWEN_SCA} --pi-table {QWEN_TABLE} --bridge-cells 8 --bridge-k 16 "
             f"--label qwen_sca --res 4096x4096 --prompts {PROMPTS} --out {OUT}/qwen_sca/4096x4096 --start {a} --end {b}",
             dry, need=70, wait=QWEN_TABLE[:-4] + ".json")
    for a, b in chunks(count(PROMPTS), 4):
        task(f"00s4_b_qsca5312_{a:02d}_{b:02d}.sh", 50, "Qwen-Image single-pass SCA at 5312^2 (4 x native)", code,
             f"{PY} scripts/sample_qwen.py {QWEN_SCA} --pi-table {QWEN_TABLE} --bridge-cells 4 --bridge-k 64 "
             f"--label qwen_sca --res 5312x5312 --prompts {PROMPTS} --out {OUT}/qwen_sca/5312x5312 --start {a} --end {b}",
             dry, need=75, wait=QWEN_TABLE[:-4] + ".json")
    for a, b in chunks(count(PROMPTS), 20):
        task(f"00s4_c_qnative_{a:02d}_{b:02d}.sh", 30, "Qwen-Image at its native 1328^2 (Lanczos reference)", code,
             f"{PY} scripts/sample_qwen.py --method full --steps 50 --true-cfg 4.0 --mu-mode native --label qwen_full "
             f"--res 1328x1328 --prompts {PROMPTS} --out {OUT}/qwen_full/1328x1328 --start {a} --end {b}", dry, need=60)


def qwen_full(dry, code):
    for size, n, mins in (("4096x4096", 4, 50), ("5312x5312", 2, 60)):
        for a, b in chunks(count(PROMPTS), n):
            task(f"00s9_qfull{size[:4]}_{a:02d}_{b:02d}.sh", mins, f"Qwen-Image full attention at {size}", code,
                 f"{PY} scripts/sample_qwen.py --method full --steps 50 --true-cfg 4.0 --mu-mode native --label qwen_full "
                 f"--res {size} --prompts {PROMPTS} --out {OUT}/qwen_full/{size} --start {a} --end {b}", dry, need=75)


def flux2(dry, code):
    base = f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {PY} scripts/sample_flux2.py"
    task("00s3_c_f2native.sh", 40, "FLUX.2-dev at its native 1024^2 (Lanczos reference; fills the Mistral embedding cache)",
         code, f"{base} --method full --steps 28 --guidance 4.0 --mu-mode native --label flux2_full --res 1024x1024 "
         f"--prompts {PROMPTS} --out {OUT}/flux2_full/1024x1024", dry, need=88)
    for m, flags in (("sca", FLUX2_SCA + " --label flux2_sca_v2"),
                     ("full", "--steps 28 --guidance 4.0 --mu-mode native --method full --label flux2_full")):
        task(f"00s3_d_f2smoke5120_{m}.sh", 20, f"FLUX.2-dev {m} memory smoke at 5120^2 (2 steps, one prompt)", code,
             f"{base} {flags} --steps 2 --res 5120x5120 --prompts {PROMPTS} --out {OUT}/_smoke/flux2_{m}_5120 --start 0 --end 1",
             dry, need=90)
    for a, b in chunks(count(PROMPTS), 5):
        task(f"00s4_d_f2sca4096_{a:02d}_{b:02d}.sh", 25, "FLUX.2-dev single-pass SCA at 4096^2", code,
             f"{base} {FLUX2_SCA} --label flux2_sca_v2 --res 4096x4096 --prompts {PROMPTS} "
             f"--out {OUT}/flux2_sca_v2/4096x4096 --start {a} --end {b}", dry, need=88)
    for a, b in chunks(count(PROMPTS), 4):
        task(f"00s4_e_f2sca5120_{a:02d}_{b:02d}.sh", 30, "FLUX.2-dev single-pass SCA at 5120^2", code,
             f"{base} {FLUX2_SCA} --label flux2_sca_v2 --res 5120x5120 --prompts {PROMPTS} "
             f"--out {OUT}/flux2_sca_v2/5120x5120 --start {a} --end {b}", dry, need=90)


def flux2_full(dry, code):
    base = f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {PY} scripts/sample_flux2.py"
    for size, n, mins in (("4096x4096", 4, 35), ("5120x5120", 3, 55)):
        for a, b in chunks(count(PROMPTS), n):
            task(f"00s8_f2full{size[:4]}_{a:02d}_{b:02d}.sh", mins, f"FLUX.2-dev full attention at {size}", code,
                 f"{base} --steps 28 --guidance 4.0 --mu-mode native --method full --label flux2_full --res {size} "
                 f"--prompts {PROMPTS} --out {OUT}/flux2_full/{size} --start {a} --end {b}", dry, need=90)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("groups", nargs="+", choices=("prompts", "flux1", "area", "qwen", "qwenfull", "flux2", "flux2full"))
    p.add_argument("--code", help="Frozen code tree (default: code/7bd64c0 for FLUX.1-dev groups)")
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    if "prompts" in a.groups:
        rows = [json.loads(line) for line in open(f"{P}/web/build/showcase_prompts.jsonl") if line.strip()]
        for path, keep in ((PROMPTS, rows), (LAND, [r for r in rows if r["orient"] == "landscape"]),
                           (PORT, [r for r in rows if r["orient"] == "portrait"])):
            text = "".join(json.dumps(r) + "\n" for r in keep)
            if Path(path).exists() and Path(path).read_text() != text:
                raise SystemExit(f"{path} exists with different content; outputs are keyed by prompt id, refusing")
            print("write", path, len(keep))
            if not a.dry:
                Path(path).write_text(text)
    if "flux1" in a.groups:
        flux1(a.dry, a.code or FLUX1_CODE)
    if "area" in a.groups:
        area(a.dry, AREA_CODE)
    for g, fn in (("qwen", qwen), ("qwenfull", qwen_full), ("flux2", flux2), ("flux2full", flux2_full)):
        if g in a.groups:
            if not a.code:
                raise SystemExit(f"--code (a frozen tree with sca/qwen.py) is required for {g}")
            fn(a.dry, a.code)


if __name__ == "__main__":
    main()
