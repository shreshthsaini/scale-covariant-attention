#!/usr/bin/env python3
"""Ideogram 4 at 8192^2 (HANDOVER Section 35): spool tasks, seed 0, 10 showcase prompts.

Priority (lead, 2026-11-08): the enabling work sorts first as 00a1_*: the GPU check (identity test, kernel options,
step cost), captions, the reach tables (2048^2 and 4096^2, ratio 2) and single-pass SCA at 8192^2 on all 10 prompts.
SEGA at 8192^2 and Ideogram 4 at its native 4096^2 run as 00s1a1_* (after the Qwen check 00s0); full attention (10
prompts), NTK and YaRN (3 prompts) at 8192^2 run last as 00z_a1_*. Dependencies by release: the caption task moves
the held reach-emulation tasks from taskq/hold_ideogram4 into pending; the last part of each table moves its merge
task (00a1_28, 00a1_29); the 4096^2 merge moves the SCA tasks. Every task waits for its free-memory need and otherwise requeues itself
(other projects' processes share some GPUs); co-tenant logs go to out/ideogram4/_gpu/<task>.apps.
Outputs: /scratch/09032/saini_2/sca/out/ideogram4/ideogram4_<method>/<HxW>/ (PNG + JSON sidecar per prompt).
Usage: slurm/ideogram4_tasks.py GROUP [GROUP ...] --code DIR [--dry]   (groups: prompts, a1, s1, z)
"""

import argparse
import json
from pathlib import Path

S = "/scratch/09032/saini_2/sca"
P = "/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention"
PY = "/work/09032/saini_2/vista/library/uv-envs/sca/bin/python"
OUT = f"{S}/out/ideogram4"
PENDING, HOLD = f"{S}/taskq/pending", f"{S}/taskq/hold_ideogram4"
PROMPTS = f"{S}/prompts/ideogram4_10.jsonl"
CAPTIONS = f"{S}/ideogram4/captions/captions_1x1_seed0.jsonl"
CALIB = f"{S}/calib"
TABLE = f"{CALIB}/ideogram4_4096"            # the native-scale fit the SCA runs use
ORDER = ("sc_p01_skeleton_watch", "sc_l04_hillside_town", "sc_h01_silver_braids", "sc_l06_atoll",
         "sc_l03_opera_house", "sc_p08_sapphire_ring", "sc_h03_watchmaker", "sc_l05_palm_house",
         "sc_p09_filigree_necklace", "sc_p11_honey_jars")   # showcase_page.json prompts in the four categories
ROPE_IDS = ORDER[:3]                          # NTK and YaRN: the two hero prompts and a portrait
ENV = ("export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 "
       "TRANSFORMERS_OFFLINE=1 XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton "
       "OMP_NUM_THREADS=8\nunset CROPE_TRANSFORMER CROPE_ROPE\nexport CC=/usr/bin/gcc CXX=/usr/bin/g++\n"
       "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n")
# single-pass SCA (sca_v2) on the 48-step schedule at 8192^2, matched to FLUX.1's 8 of 28 global steps and the Qwen port's
# two steps per scheduled window: 14 global steps (sigma > 0.97444), windows 2, 1.75, 1.5, 1.25 x n0 for two steps each
SCA_V2 = ("--method sca --n0 256 --tile 8 --gate-sigma 0.97444 --bridge-cells 8 --bridge-k 16 "
          "--win-sched 0.96957:512,0.96413:448,0.95802:384,0.95112:320 --win-pi-match --win-pi-interp "
          f"--pi-table {TABLE}_cond.npz --pi-table-uncond {TABLE}_uncond.npz")
GEN = f"{PY} scripts/sample_ideogram4.py --prompts {PROMPTS} --captions {CAPTIONS} --seed 0"

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


def release(pattern):
    """Shell lines moving held tasks matching pattern into pending (atomic rename; a task already moved is skipped)."""
    return f"for t in {HOLD}/{pattern}; do [ -e \"$t\" ] && mv -n \"$t\" {PENDING}/ && echo \"released $(basename $t)\"; done\n"


def parts_done(stem, n):
    """Shell test: both transformers' part folders of a reach table hold n finished parts."""
    c = lambda name: f"$(ls {stem}_{name}_parts/ 2>/dev/null | grep -c '_seed0.npz$')"
    return f"[ {c('cond')} -ge {n} ] && [ {c('uncond')} -ge {n} ]"


def task(name, minutes, comment, code, cmd, dry, need=60, polls=40, wait=None, post="", hold=False):
    text = (f"#!/bin/bash\n# SCA_MIN_SECONDS: {int(minutes * 60)}\n# ideogram4 (slurm/ideogram4_tasks.py): {comment}\n"
            + ENV + "set -uo pipefail\n"
            + (WAIT.format(path=wait, name=name, pending=PENDING) if wait else "")
            + (GUARD.format(need=need, out=OUT, polls=polls, name=name, pending=PENDING) if need else
               f"LOG={OUT}/_gpu; mkdir -p $LOG\nsnap(){{ echo \"$1 $(date +%FT%T) $(hostname -s)\"; }}\n")
            + f"cd {code}\n" + f"C=\"{cmd}\"\n" + f"echo \"cmd $C\" >> $LOG/{name}.apps\n"
            + "$C || { sleep 30; $C; }; rc=$?\nsnap end >> $LOG/" + name + ".apps\n"
            + "[ $rc -eq 0 ] || exit $rc\n" + post + f"mkdir -p {OUT}/_done && touch {OUT}/_done/{name}\nexit 0\n")
    if Path(f"{OUT}/_done/{name}").exists():
        print("done, kept:", name)
        return
    for d in ("running", "pending", "hold_ideogram4"):
        hits = list(Path(f"{S}/taskq/{d}").glob(name + "*"))
        if hits:
            print(f"{d}, kept:", hits[0])
            return
    old = Path(f"{S}/taskq/failed/{name}")
    if old.exists() and not dry:   # superseded failure: keep the old script for the record, write a fresh one
        sup = Path(f"{S}/taskq/superseded_ideogram4")
        sup.mkdir(exist_ok=True)
        old.rename(sup / f"{name}.{int(old.stat().st_mtime)}")
    path = Path(HOLD if hold else PENDING) / name
    print("write", path)
    if not dry:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{name}.tmp")
        tmp.write_text(text)
        tmp.chmod(0o755)
        tmp.rename(path)


def write_prompts(dry):
    rows = {json.loads(line)["id"]: json.loads(line) for line in open(f"{P}/web/build/showcase_prompts.jsonl") if line.strip()}
    text = "".join(json.dumps(rows[i]) + "\n" for i in ORDER)
    for path in (PROMPTS, f"{P}/web/build/ideogram4_prompts.jsonl"):
        if Path(path).exists() and Path(path).read_text() != text:
            raise SystemExit(f"{path} exists with different content; outputs are keyed by prompt id, refusing")
        print("write", path, len(ORDER))
        if not dry:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)


def group_a1(dry, code):
    task("00a1_00_check", 45, "GPU identity test, flex kernel options, step cost at 8192^2", code,
         f"{PY} scripts/ideogram4_gpu_check.py --out {S}/ideogram4/check", dry, need=80)
    task("00a1_01_captions", 20, "upsampled JSON captions: 10 showcase + 16 emulation prompts, aspect 1:1, seed 0",
         code, f"{PY} scripts/ideogram4_captions.py --prompts {PROMPTS} scripts/calib_prompts.jsonl --count 10 16 "
         f"--out {CAPTIONS} --res 8192x8192 --seed 0", dry, need=30, post=release("00a1_1*"))
    for res, n_per, need in ((2048, 8, 40), (4096, 4, 56)):
        stem, tag, merge = f"{CALIB}/ideogram4_{res}", {2048: "10", 4096: "11"}[res], {2048: "28", 4096: "29"}[res]
        for a in range(0, 16, n_per):
            b = a + n_per
            task(f"00a1_{tag}_fit{res}_{a:02d}_{b:02d}", 25 if res == 2048 else 90,
                 f"reach emulation at {res}^2, ratio 2, prompts {a}-{b}", code,
                 f"{PY} scripts/calibrate_ideogram4.py --captions {CAPTIONS} --count 16 --start {a} --end {b} "
                 f"--res {res}x{res} --scales 2 --sigma-bins 28 --out {stem} --no-merge", dry, need=need,
                 wait=CAPTIONS, hold=True,
                 post=f"if {parts_done(stem, 16)}; then\n" + release(f"00a1_{merge}_merge{res}") + "fi\n")
        other = f" --transfer-with {CALIB}/ideogram4_2048" if res == 4096 else ""
        task(f"00a1_{merge}_merge{res}", 5, f"merge the {res}^2 reach tables (CPU)", code,
             f"{PY} scripts/calibrate_ideogram4.py --captions {CAPTIONS} --count 16 --res {res}x{res} --scales 2 "
             f"--sigma-bins 28 --out {stem} --report-only{other}", dry, need=0, hold=True,
             post=release("00a1_3*") if res == 4096 else "")
    for k, pid in enumerate(ORDER):
        task(f"00a1_3{k}_sca8k_{pid}", 120, f"single-pass SCA (sca_v2) at 8192^2, {pid}", code,
             f"{GEN} --res 8192x8192 --ids {pid} {SCA_V2} --out {OUT}/ideogram4_sca/8192x8192", dry, need=72,
             wait=f"{TABLE}_uncond.npz", hold=True)


def group_s1(dry, code):
    for k, pid in enumerate(ORDER):
        task(f"00s1a1_2{k}_sega8k_{pid}", 180, f"SEGA at 8192^2 (train grid 256), {pid}", code,
             f"{GEN} --res 8192x8192 --ids {pid} --method sega --train-grid 256 --out {OUT}/ideogram4_sega/8192x8192",
             dry, need=72, wait=CAPTIONS)
    for a, b in ((0, 5), (5, 10)):
        task(f"00s1a1_5{a // 5}_nat4k_{a:02d}_{b:02d}", 75, f"Ideogram 4 at its native 4096^2, prompts {a}-{b}", code,
             f"{GEN} --res 4096x4096 --start {a} --end {b} --method full --out {OUT}/ideogram4_full/4096x4096", dry,
             need=50, wait=CAPTIONS)


def group_z(dry, code):
    for k, pid in enumerate(ORDER):
        task(f"00z_a1_3{k}_full8k_{pid}", 180, f"full attention at 8192^2, {pid}", code,
             f"{GEN} --res 8192x8192 --ids {pid} --method full --out {OUT}/ideogram4_full/8192x8192", dry, need=72,
             wait=CAPTIONS)
    for m, tag in (("ntk", "40"), ("yarn", "41")):
        for k, pid in enumerate(ROPE_IDS):
            task(f"00z_a1_{tag}{k}_{m}8k_{pid}", 180, f"{m.upper()} at 8192^2 (train grid 256), {pid}", code,
                 f"{GEN} --res 8192x8192 --ids {pid} --method {m} --train-grid 256 --out {OUT}/ideogram4_{m}/8192x8192",
                 dry, need=72, wait=CAPTIONS)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("groups", nargs="+", choices=("prompts", "a1", "s1", "z"))
    p.add_argument("--code", help="Frozen code tree with sca/ideogram4.py and third_party/sega")
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    if "prompts" in a.groups:
        write_prompts(a.dry)
    for g, fn in (("a1", group_a1), ("s1", group_s1), ("z", group_z)):
        if g in a.groups:
            if not a.code or not (Path(a.code) / "sca/ideogram4.py").exists():
                raise SystemExit("--code must be a frozen tree with sca/ideogram4.py")
            fn(a.dry, a.code)


if __name__ == "__main__":
    main()
