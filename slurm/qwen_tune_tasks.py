#!/usr/bin/env python3
"""Tuning SCA on Qwen-Image against AReA (HANDOVER Section 39): spool tasks named 000r_* so they run ahead of every
00s* showcase task.

Protocol (anti-overfitting): the 15 prompts of results/qwen_5312_vs_area (qwen_area15.jsonl) are the held-out TEST
set and are never used to choose a configuration. TUNE is a separate set of 12 prompts from
web/build/showcase_prompts.jsonl chosen by a fixed rule: in each prompt group, the prompts that follow the three TEST
prompts, in file order (product p04-p06, landscape and architecture l04-l06, portrait h04-h06, animal a04-a05, dense
d04). AReA runs at its own defaults on TUNE as the reference; every SCA configuration is scored by paired differences
against it (scripts/qwen_tune_eval.py). The chosen configuration then runs once on TEST at 5312^2.

Groups (each writes spool tasks; outputs /scratch/09032/saini_2/sca/out/qwen_tune/<label>/<HxW>/):
  prompts    write prompts/qwen_tune12.jsonl (refuses to overwrite different content)
  area4096   AReA (scripts/run_area_qwen.py, its qwen-image defaults) at 4096^2 on TUNE, the single-pass reference
  base4096   untuned SCA at 40 steps (the qwen_sca40 flags of the TEST comparison) at 4096^2 on TUNE
Usage: slurm/qwen_tune_tasks.py GROUP [GROUP ...] [--dry]
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import area_qwen_tasks as aq  # noqa: E402
import showcase_tasks as st  # noqa: E402

S, PY = st.S, st.PY
TUNE = f"{S}/prompts/qwen_tune12.jsonl"
TEST = aq.PROMPTS15
OUT_TUNE = f"{S}/out/qwen_tune"
GROUPS = ("sc_p", "sc_l", "sc_h", "sc_a", "sc_d")
TUNE_COUNTS = (3, 3, 3, 2, 1)           # product, landscape/architecture, portrait, animal, dense
AREA_CODE = f"{S}/code/5e945ef"         # AReA-Qwen runner and evaluator tree of the TEST comparison
SCA_CODE = aq.SCA_CODE                  # code/fa20267, every Qwen-Image SCA image so far
BRIDGE = {"5312x5312": "--bridge-cells 4 --bridge-k 64", "4096x4096": "--bridge-cells 8 --bridge-k 16"}
SCA40 = st.QWEN_SCA.replace("--steps 50 ", "--steps 40 ")
assert SCA40.count("--steps 40 ") == 1


def tune_prompts(dry):
    """The fixed TUNE rule: per group, the prompts after the first three (TEST), in file order."""
    rows = [json.loads(x) for x in open(st.PROMPTS) if x.strip()]
    test = {json.loads(x)["id"] for x in open(TEST) if x.strip()}
    keep = []
    for g, n in zip(GROUPS, TUNE_COUNTS):
        rest = [r for r in rows if r["id"].startswith(g)][3:]
        keep += rest[:n]
    ids = [r["id"] for r in keep]
    assert len(keep) == 12 and not set(ids) & test, ids
    text = "".join(json.dumps(r) + "\n" for r in keep)
    p = Path(TUNE)
    if p.exists() and p.read_text() != text:
        raise SystemExit(f"{p} exists with different content; outputs are keyed by prompt id, refusing")
    print("write", p, ids)
    if not dry:
        p.write_text(text)


def chunks(n, size):
    return [(a, min(a + size, n)) for a in range(0, n, size)]


def area4096(dry):
    rev = f"export SCA_CODE_REVISION={Path(AREA_CODE).name}\n"
    for a, b in chunks(12, 4):
        aq.task(f"000r_a_area4096_{a:02d}_{b:02d}.sh", 35, f"TUNE reference: AReA (qwen-image defaults) at 4096^2, "
                f"prompts {a + 1}-{b}", AREA_CODE,
                f"{PY} scripts/run_area_qwen.py --seed 0 --prompts {TUNE} --res 4096x4096 --start {a} --end {b} "
                f"--label qt_area --out {OUT_TUNE}/qt_area/4096x4096", dry, need=55, extra_env=rev)


def base4096(dry):
    for a, b in chunks(12, 4):
        aq.task(f"000r_b_base4096_{a:02d}_{b:02d}.sh", 35, f"TUNE: untuned SCA (qwen_sca40 flags) at 4096^2, prompts "
                f"{a + 1}-{b}", SCA_CODE,
                f"{PY} scripts/sample_qwen.py {SCA40} --pi-table {st.QWEN_TABLE} {BRIDGE['4096x4096']} "
                f"--label qt_sca40 --res 4096x4096 --prompts {TUNE} --start {a} --end {b} "
                f"--out {OUT_TUNE}/qt_sca40/4096x4096", dry, need=55)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+", choices=("prompts", "area4096", "base4096"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    for g in a.groups:
        {"prompts": tune_prompts, "area4096": area4096, "base4096": base4096}[g](a.dry)


if __name__ == "__main__":
    main()
