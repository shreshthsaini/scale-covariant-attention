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
  area5312   AReA at 5312^2 on TUNE, the reference of the progressive screens (their stage factors are exact there)
  sp4096     single-pass SCA screens at 4096^2 (SP below; 40 steps, the untuned thresholds, which are noise levels)
  prog5312   progressive SCA screens at 5312^2 (PROG below; scripts/sample_progressive_qwen.py)
Configurations (labels are the sidecar methods and output folders):
  qt_sca40     untuned: native mu 0.834, negative " ", no suffix (the TEST comparison's qwen_sca40)
  qt_mu        + resolution time shift (--mu-mode resolution: mu 1.96 at 4096^2, 2.22 at 5312^2; AReA uses 2.0 / 2.26)
  qt_sfx       + AReA's conditioning: its QWEN_SUFFIX and the empty negative prompt (metrics score the benchmark prompt)
  qt_mu_sfx    both
  qt_prog      progressive: 1328^2 stock attention (30 steps), 2656^2 SCA (last 20 steps, from sigma 0.81), 5312^2
               SCA (last 13, from sigma 0.62); HRDiT's guidance, true CFG 4 in every stage, bridge 1024 tokens
  qt_prog_sfx  qt_prog with AReA's conditioning
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
NEW_CODE = f"{S}/code/a7034a3"          # resolution mu, --prompt-suffix, sample_progressive_qwen.py (tests pass)
AREA_COND = "--prompt-suffix area --negative="
SP = {"qt_mu": SCA40.replace("--mu-mode native ", "--mu-mode resolution "),
      "qt_sfx": f"{SCA40} {AREA_COND}",
      "qt_mu_sfx": SCA40.replace("--mu-mode native ", "--mu-mode resolution ") + f" {AREA_COND}"}
assert all(v != SCA40 for v in SP.values())
PROG_BASE = ("--stages 1328x1328,2656x2656,5312x5312 --base-steps 30 --highres-steps 20,13 --true-cfg 4.0 "
             f"--pi-table {st.QWEN_TABLE} --bridge-tokens 1024")
PROG = {"qt_prog": PROG_BASE, "qt_prog_sfx": f"{PROG_BASE} {AREA_COND}"}


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


def area5312(dry):
    rev = f"export SCA_CODE_REVISION={Path(AREA_CODE).name}\n"
    for a, b in chunks(12, 3):
        aq.task(f"000r_c_area5312_{a:02d}_{b:02d}.sh", 45, f"TUNE reference: AReA (qwen-image defaults) at 5312^2, "
                f"prompts {a + 1}-{b}", AREA_CODE,
                f"{PY} scripts/run_area_qwen.py --seed 0 --prompts {TUNE} --res 5312x5312 --start {a} --end {b} "
                f"--label qt_area --out {OUT_TUNE}/qt_area/5312x5312", dry, need=60, extra_env=rev)


def sp4096(dry, labels=None):
    for label, flags in SP.items():
        if labels and label not in labels:
            continue
        for a, b in chunks(12, 4):
            aq.task(f"000r_d_{label[3:]}4096_{a:02d}_{b:02d}.sh", 35, f"TUNE screen: single-pass SCA {label} at "
                    f"4096^2, prompts {a + 1}-{b}", NEW_CODE,
                    f"{PY} scripts/sample_qwen.py {flags} --pi-table {st.QWEN_TABLE} {BRIDGE['4096x4096']} "
                    f"--label {label} --res 4096x4096 --prompts {TUNE} --start {a} --end {b} "
                    f"--out {OUT_TUNE}/{label}/4096x4096", dry, need=55)


def prog5312(dry, labels=None):
    for label, flags in PROG.items():
        if labels and label not in labels:
            continue
        for a, b in chunks(12, 4):
            pre = "000r_0" if (label, a) == ("qt_prog", 0) else "000r_c"   # first GPU run of the new sampler goes first
            aq.task(f"{pre}_{label[3:]}5312_{a:02d}_{b:02d}.sh", 40, f"TUNE screen: progressive SCA {label} at "
                    f"5312^2, prompts {a + 1}-{b}", NEW_CODE,
                    f"{PY} scripts/sample_progressive_qwen.py {flags} --label {label} --prompts {TUNE} --start {a} "
                    f"--end {b} --seed 0 --out {OUT_TUNE}/{label}/5312x5312", dry, need=60)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+", choices=("prompts", "area4096", "base4096", "area5312", "sp4096", "prog5312"))
    ap.add_argument("--labels", nargs="*", help="Only these configuration labels (sp4096, prog5312)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    for g in a.groups:
        if g in ("sp4096", "prog5312"):
            {"sp4096": sp4096, "prog5312": prog5312}[g](a.dry, a.labels)
        else:
            {"prompts": tune_prompts, "area4096": area4096, "base4096": base4096, "area5312": area5312}[g](a.dry)


if __name__ == "__main__":
    main()
