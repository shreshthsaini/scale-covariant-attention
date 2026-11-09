#!/usr/bin/env python3
"""AReA's released Qwen-Image code against single-pass SCA (HANDOVER Section 38): spool tasks named 000q_* so they
run right after the 000p_* showcase pairs and ahead of every 00s* task.

Prompts: /scratch/09032/saini_2/sca/prompts/qwen_area15.jsonl, the first three prompts of each of the five prompt
groups of web/build/showcase_prompts.jsonl (product, landscape and architecture, portrait, animal, dense) in file
order; rows identical to showcase40.jsonl, so ids, stems and derived seeds match the showcase outputs.
  000q_a     AReA at 5312^2 on prompts 1-3 (sc_p01..p03, the pairs already on the page with SCA and full attention)
  000q_b     AReA at 4096^2 on prompts 1-3 (AReA's README states its Qwen-Image schedule at 4096^2)
  000q_c     SCA at 4096^2 on prompts 1-3 (same tree and flags as the queued 00s4_a tasks, which then skip them)
  000q_<k>1  AReA at 5312^2 on prompts 4-15, three per task
  000q_<k>2  SCA at 5312^2 on the same three (same tree and flags as 00s4_b; outputs shared with it)
  000q_z_sca40  SCA at 5312^2 with 40 steps, AReA's step count (lead, 2026-11-09), on all 15 prompts, three per task;
             every other flag as 000q_<k>2. SCA's gate and window thresholds are noise levels, so at 40 steps the
             global opening covers 11 steps (14 of 50 scales to 11.2) and the windows 166, 166, 145, 124, 124, 104,
             104 cover the same noise range as at 50 steps; outputs out/showcase/qwen_sca40/5312x5312
  000q_z     evaluation (scripts/qwen_area_eval.py -> results/qwen_5312_vs_area); kept in taskq/hold_area_qwen and
             moved to pending by the last generation task to finish; then slurm/area_qwen_finish.sh refreshes the page
             data and the Qwen-Image paper snippets in the live repository (nothing committed)
AReA runs scripts/run_area_qwen.py from the frozen tree --code (third_party/AReA_3efb7e5 inside it); SCA runs
code/fa20267, the tree of every Qwen-Image SCA image so far. Outputs out/showcase/qwen_{area,sca}/<HxW>/.

Timing hygiene on top of showcase_tasks.py's memory guard: before the sampler starts the task waits up to 5 min for
the GPU to hold no other compute process, then polls every 15 s while it runs and appends any other process to
out/showcase/_gpu/<task>.apps as "poll <epoch> <time> <pid>, <MiB>" (the manifest builder then flags the task's
images as shared; qwen_area_eval.py flags single images by time). run_area_qwen.py also polls in-process.
Usage: slurm/area_qwen_tasks.py --code /scratch/09032/saini_2/sca/code/<commit> [--dry]
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import showcase_tasks as st  # noqa: E402

S, PY, OUT = st.S, st.PY, st.OUT
SCA_CODE = f"{S}/code/fa20267"
PROMPTS15 = f"{S}/prompts/qwen_area15.jsonl"
HOLD = f"{S}/taskq/hold_area_qwen"
GROUPS = ("sc_p", "sc_l", "sc_h", "sc_a", "sc_d")
RESULTS = f"{st.P}/results/qwen_5312_vs_area"

EXCL = """for i in $(seq 1 10); do A=$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  [ "${{A:-0}}" -eq 0 ] && break; [ $i -eq 1 ] && echo "$(date +%FT%T) waiting up to 5 min for an exclusive GPU $G ($A other processes)"; sleep 30; done
ours(){{ local p=$1; for k in $(seq 1 12); do [ "$p" = "$$" ] && return 0; p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  [ -z "$p" ] && return 1; [ "$p" -le 1 ] && return 1; done; return 1; }}
poller(){{ while :; do nvidia-smi -i $G --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null |
  while IFS=, read -r pid mem; do pid=${{pid// /}}; [ -z "$pid" ] && continue; ours "$pid" || echo "poll $(date +%s) $(date +%FT%T) $pid,$mem"; done >> $LOG/{name}.apps
  sleep 15; done; }}
echo "start poller $(date +%s) $(date +%FT%T) $(hostname -s) gpu$G" >> $LOG/{name}.apps
poller & POLLER=$!
trap 'kill $POLLER 2>/dev/null' EXIT
trap 'kill $POLLER 2>/dev/null; exit 143' TERM INT HUP
"""

RELEASE = """ok=1; for t in {gen}; do [ -e {done}/$t ] || ok=0; done
if [ $ok -eq 1 ] && mkdir {hold}/.released 2>/dev/null; then
  cp {hold}/000q_z_eval.sh {pending}/000q_z_eval.sh && echo "$(date +%FT%T) released 000q_z_eval.sh"
fi
"""


def task(name, minutes, comment, code, cmd, dry, need=60, extra_env="", post="", folder=None, exclusive=True):
    """One spool task in showcase_tasks.py's format (memory guard, .apps snapshots, retry once, _done marker) plus,
    for generation tasks, the exclusivity wait and the co-tenant poller."""
    text = (f"#!/bin/bash\n# SCA_MIN_SECONDS: {int(minutes * 60)}\n# AReA vs SCA on Qwen-Image (slurm/area_qwen_tasks.py): "
            f"{comment}\n" + st.ENV + extra_env + "set -uo pipefail\n"
            + st.GUARD.format(need=need, out=OUT, polls=40, name=name, pending=f"{S}/taskq/pending")
            + (EXCL.format(name=name) if exclusive else "POLLER=\n")
            + f"cd {code}\n" + f"C=\"{cmd}\"\n" + f"echo \"cmd $C\" >> $LOG/{name}.apps\n"
            + "$C || { sleep 30; $C; }; rc=$?\n[ -n \"$POLLER\" ] && kill $POLLER 2>/dev/null\nsnap end >> $LOG/"
            + name + ".apps\n"
            + "[ $rc -eq 0 ] || exit $rc\n" + f"mkdir -p {OUT}/_done && touch {OUT}/_done/{name}\n" + post + "exit 0\n")
    path = Path(folder or f"{S}/taskq/pending") / name
    if Path(f"{OUT}/_done/{name}").exists():
        print("done, kept:", name)
        return
    if list(Path(f"{S}/taskq/running").glob(name + "*")):
        print("running, kept:", name)
        return
    print("write", path)
    if not dry:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{name}.tmp")
        tmp.write_text(text)
        tmp.chmod(0o755)
        tmp.rename(path)


def prompts15(dry):
    rows = [json.loads(x) for x in open(st.PROMPTS) if x.strip()]
    keep = [r for g in GROUPS for r in [r for r in rows if r["id"].startswith(g)][:3]]
    assert len(keep) == 15 and keep[0]["id"] == "sc_p01_skeleton_watch"
    text = "".join(json.dumps(r) + "\n" for r in keep)
    p = Path(PROMPTS15)
    if p.exists() and p.read_text() != text:
        raise SystemExit(f"{p} exists with different content; outputs are keyed by prompt id, refusing")
    print("write", p, [r["id"] for r in keep])
    if not dry:
        p.write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", required=True, help="Frozen tree with scripts/run_area_qwen.py and third_party/AReA_3efb7e5")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if not (Path(a.code) / "scripts/run_area_qwen.py").exists() or not (Path(a.code) / "third_party/AReA_3efb7e5").is_dir():
        raise SystemExit(f"{a.code} lacks scripts/run_area_qwen.py or third_party/AReA_3efb7e5")
    prompts15(a.dry)
    rev = f"export SCA_CODE_REVISION={Path(a.code).name}\n"
    area = f"{PY} scripts/run_area_qwen.py --seed 0 --prompts {PROMPTS15}"
    sca = (f"{PY} scripts/sample_qwen.py {st.QWEN_SCA} --pi-table {st.QWEN_TABLE} --label qwen_sca "
           f"--prompts {PROMPTS15}")
    bridge = {"5312x5312": "--bridge-cells 4 --bridge-k 64", "4096x4096": "--bridge-cells 8 --bridge-k 16"}
    specs = [  # (name, minutes, comment, code, cmd, need, extra_env)
        ("000q_a_area5312_00_03.sh", 45, "AReA (official, qwen-image defaults) at 5312^2, prompts 1-3", a.code,
         f"{area} --res 5312x5312 --start 0 --end 3 --out {OUT}/qwen_area/5312x5312", 60, rev),
        ("000q_b_area4096_00_03.sh", 25, "AReA (official, qwen-image defaults) at 4096^2, prompts 1-3", a.code,
         f"{area} --res 4096x4096 --start 0 --end 3 --out {OUT}/qwen_area/4096x4096", 55, rev),
        ("000q_c_qsca4096_00_03.sh", 30, "single-pass SCA at 4096^2, prompts 1-3 (00s4_a flags)", SCA_CODE,
         f"{sca} {bridge['4096x4096']} --res 4096x4096 --start 0 --end 3 --out {OUT}/qwen_sca/4096x4096", 60, "")]
    for k, (s, e) in zip("defg", ((3, 6), (6, 9), (9, 12), (12, 15))):
        specs += [(f"000q_{k}1_area5312_{s:02d}_{e:02d}.sh", 45, f"AReA at 5312^2, prompts {s + 1}-{e}", a.code,
                   f"{area} --res 5312x5312 --start {s} --end {e} --out {OUT}/qwen_area/5312x5312", 60, rev),
                  (f"000q_{k}2_qsca5312_{s:02d}_{e:02d}.sh", 50,
                   f"single-pass SCA at 5312^2, prompts {s + 1}-{e} (00s4_b flags)", SCA_CODE,
                   f"{sca} {bridge['5312x5312']} --res 5312x5312 --start {s} --end {e} --out {OUT}/qwen_sca/5312x5312",
                   60, "")]
    sca40 = sca.replace("--steps 50 ", "--steps 40 ").replace("--label qwen_sca ", "--label qwen_sca40 ")
    assert sca40.count("--steps 40 ") == 1 and "--label qwen_sca40 " in sca40
    for s, e in ((0, 3), (3, 6), (6, 9), (9, 12), (12, 15)):
        specs.append((f"000q_z_sca40_{s:02d}_{e:02d}.sh", 40, f"single-pass SCA at 5312^2, 40 steps, prompts {s + 1}-{e}",
                      SCA_CODE, f"{sca40} {bridge['5312x5312']} --res 5312x5312 --start {s} --end {e} "
                      f"--out {OUT}/qwen_sca40/5312x5312", 60, ""))
    # the evaluation waits in the hold folder; the generation task that completes the set moves it to pending
    release = RELEASE.format(pending=f"{S}/taskq/pending", hold=HOLD, gen=" ".join(x[0] for x in specs),
                             done=f"{OUT}/_done")
    for name, minutes, comment, code, cmd, need, env in specs:
        task(name, minutes, comment, code, cmd, a.dry, need=need, extra_env=env, post=release)
    task("000q_z_eval.sh", 40, "evaluation: SCA vs AReA on Qwen-Image (scripts/qwen_area_eval.py)", a.code,
         f"{PY} scripts/qwen_area_eval.py --prompts {PROMPTS15} --results {RESULTS}", a.dry, need=40, folder=HOLD,
         exclusive=False, post=f"bash {st.P}/slurm/area_qwen_finish.sh\n")


if __name__ == "__main__":
    main()
