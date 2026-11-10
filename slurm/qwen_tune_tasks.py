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
  sp5312     a single-pass screen at 5312^2 (the protocol's tie check; then eval5312 --labels ... --tag _sp)
Configurations (labels are the sidecar methods and output folders):
  qt_sca40     untuned: native mu 0.834, negative " ", no suffix (the TEST comparison's qwen_sca40)
  qt_mu        + resolution time shift (--mu-mode resolution: mu 1.96 at 4096^2, 2.22 at 5312^2; AReA uses 2.0 / 2.26)
  qt_sfx       + AReA's conditioning: its QWEN_SUFFIX and the empty negative prompt (metrics score the benchmark prompt)
  qt_mu_sfx    both
  qt_prog      progressive: 1328^2 stock attention (30 steps), 2656^2 SCA (last 20 steps, from sigma 0.81), 5312^2
               SCA (last 13, from sigma 0.62); HRDiT's guidance, true CFG 4 in every stage, bridge 1024 tokens
  qt_prog_sfx  qt_prog with AReA's conditioning
  qt_prog_late qt_prog with HRDiT's FLUX step counts (17 at 2656^2 from sigma 0.74, 10 at 5312^2 from sigma 0.51)
  qt_prog_late_sfx  qt_prog_late with AReA's conditioning (completes the 2 x 2 of stage starts and conditioning)
  qt_prog_deep_sfx  round 2: qt_prog_sfx with a deeper 5312^2 stage (25 steps from sigma 0.915 instead of 13 from 0.62)
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
PROG = {"qt_prog": PROG_BASE, "qt_prog_sfx": f"{PROG_BASE} {AREA_COND}",
        # HRDiT's FLUX step counts (17 and 10 of 30) instead of its noise levels: stages from sigma 0.74 and 0.51
        "qt_prog_late": PROG_BASE.replace("--highres-steps 20,13 ", "--highres-steps 17,10 ")}
PROG["qt_prog_late_sfx"] = f"{PROG['qt_prog_late']} {AREA_COND}"
assert PROG["qt_prog_late"] != PROG_BASE
# Round 2 (added after the first TUNE evaluation, TEST untouched): the round-1 progressive images are soft at native
# pixels (MUSIQ_c 7.5 to 13.7 below AReA) because the 5312^2 stage starts at sigma 0.62 or 0.51, which at 4x the native
# side keeps nearly all of the upsampled image. A deeper last stage: 25 steps from sigma 0.915 (step 5 of 30), still
# anchored by HRDiT's low-pass structure guidance, with AReA's conditioning (the better progressive variant).
PROG["qt_prog_deep_sfx"] = PROG_BASE.replace("--highres-steps 20,13 ", "--highres-steps 20,25 ") + f" {AREA_COND}"
assert "--highres-steps 20,25 " in PROG["qt_prog_deep_sfx"]


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


def sp5312(dry, labels):
    """The protocol's tie case: when the best single-pass and the best progressive configuration are within 0.10 mean
    d_z, the single-pass one also runs at 5312^2 on TUNE (against qt_area at 5312^2), three prompts per task."""
    for label in labels:
        for a, b in chunks(12, 3):
            aq.task(f"000r_d_{label[3:]}5312_{a:02d}_{b:02d}.sh", 45, f"TUNE screen: single-pass SCA {label} at "
                    f"5312^2 (tie check), prompts {a + 1}-{b}", NEW_CODE,
                    f"{PY} scripts/sample_qwen.py {SP[label]} --pi-table {st.QWEN_TABLE} {BRIDGE['5312x5312']} "
                    f"--label {label} --res 5312x5312 --prompts {TUNE} --start {a} --end {b} "
                    f"--out {OUT_TUNE}/{label}/5312x5312", dry, need=60)


def prog5312(dry, labels=None):
    for label, flags in PROG.items():
        if labels and label not in labels:
            continue
        for a, b in chunks(12, 4):
            pre = "000r_0" if (label, a) == ("qt_prog", 0) else "000r_c"   # first GPU run of the new sampler goes first
            if label in ROUND2:
                pre = "000r_gz"        # round 2 sorts after the TEST run and its evaluation (000r_f, 000r_g)
            aq.task(f"{pre}_{label[3:]}5312_{a:02d}_{b:02d}.sh", 40, f"TUNE screen: progressive SCA {label} at "
                    f"5312^2, prompts {a + 1}-{b}", NEW_CODE,
                    f"{PY} scripts/sample_progressive_qwen.py {flags} --label {label} --prompts {TUNE} --start {a} "
                    f"--end {b} --seed 0 --out {OUT_TUNE}/{label}/5312x5312", dry, need=60)


def full5312(dry):
    """Full attention at 5312^2 for the TEST prompts that lack it (the review page's grids): exactly the queued
    00s9_qfull5312 tasks' tree, flags, prompt file and output folder, one prompt per task, so those skip them later."""
    rows = [json.loads(x)["id"] for x in open(st.PROMPTS) if x.strip()]
    have = {p.name.split("_seed")[0].rsplit("_", 1)[0] for p in Path(f"{st.OUT}/qwen_full/5312x5312").glob("*.png")}
    for pid in [json.loads(x)["id"] for x in open(TEST) if x.strip()]:
        if pid in have:
            continue
        k = rows.index(pid)
        aq.task(f"000r_h_qfull5312_{k:02d}.sh", 40, f"TEST grids: full attention at 5312^2, {pid} (00s9_qfull5312 flags)",
                SCA_CODE, f"{PY} scripts/sample_qwen.py --method full --steps 50 --true-cfg 4.0 --mu-mode native "
                f"--label qwen_full --res 5312x5312 --prompts {st.PROMPTS} --out {st.OUT}/qwen_full/5312x5312 "
                f"--start {k} --end {k + 1}", dry, need=60, post=FINISH.format(name=f"000r_h_qfull5312_{k:02d}.sh"))


OUT_TEST = f"{S}/out/qwen_tuned"
EVAL = f"{S}/eval"
# Review page hook (slurm/qwen_review_finish.sh): run after 000r_g and every 000r_h task; the last one builds the page
FINISH = f"bash {st.P}/slurm/qwen_review_finish.sh {{name}}\n"


ROUND2 = ("qt_prog_deep_sfx",)


def eval_tune(dry, res, labels, code, tag=""):
    """Screen evaluation at one size: every label that has a folder there, against AReA (qt_area)."""
    meth = " ".join(f"--method {lab}={OUT_TUNE}/{lab}/{res}" for lab in ["qt_area"] + list(labels))
    name = f"000r_gz_zeval_tune{res[:4]}{tag}.sh" if set(labels) & set(ROUND2) else f"000r_e_eval_tune{res[:4]}{tag}.sh"
    aq.task(name, 100, f"TUNE screen evaluation at {res} (scripts/qwen_tune_eval.py); "
            "named to be claimed after every screen task, it scores images as they land until all 12 per method exist",
            code, f"{PY} scripts/qwen_tune_eval.py --prompts {TUNE} --res {res} --reference qt_area {meth} "
            f"--eval {EVAL}/qwen_tune{res[:4]} --results {st.P}/results/qwen_tune/{res[:4]} --expect 12 "
            "--wait-max 5400", dry, need=40, exclusive=False)


def test5312(dry, labels, code, name=None):
    """The chosen configuration(s) on the 15 TEST prompts at 5312^2, three prompts per task, on the tree of the TUNE
    screens (NEW_CODE; samplers identical in every later tree), so each runs exactly as screened. The first label is
    the chosen one: with --name (the final choice), out/qwen_tuned/chosen.env names it for the review page hook; a
    configuration enqueued before the final choice (the best single pass while round 2 runs) passes no --name."""
    code = NEW_CODE
    env = Path(OUT_TEST) / "chosen.env"
    text = f"TUNED={labels[0]}\nTUNED_NAME={json.dumps(name or labels[0])}\n"
    if name and env.exists() and env.read_text() != text:
        raise SystemExit(f"{env} exists with a different choice; the TEST run happens once")
    if name:
        print("write", env, text.split())
        if not dry:
            env.parent.mkdir(parents=True, exist_ok=True)
            env.write_text(text)
    for label in labels:
        if label in SP:
            cmd = (f"{PY} scripts/sample_qwen.py {SP[label]} --pi-table {st.QWEN_TABLE} {BRIDGE['5312x5312']} "
                   f"--label {label} --res 5312x5312")
        elif label in PROG:
            cmd = f"{PY} scripts/sample_progressive_qwen.py {PROG[label]} --label {label} --seed 0"
        else:
            raise SystemExit(f"unknown configuration {label}")
        for a, b in chunks(15, 3):
            aq.task(f"000r_f_{label[3:]}5312_{a:02d}_{b:02d}.sh", 45, f"TEST: {label} at 5312^2, prompts {a + 1}-{b}",
                    code, f"{cmd} --prompts {TEST} --start {a} --end {b} --out {OUT_TEST}/{label}/5312x5312", dry,
                    need=60)


def eval_test(dry, labels, code):
    """TEST evaluation at 5312^2: the chosen configuration(s), AReA, untuned SCA (40 and 50 steps), full attention."""
    meth = [f"--method qwen_area={st.OUT}/qwen_area/5312x5312", f"--method qwen_sca40={st.OUT}/qwen_sca40/5312x5312",
            f"--method qwen_sca={st.OUT}/qwen_sca/5312x5312", f"--method qwen_full={st.OUT}/qwen_full/5312x5312"]
    meth += [f"--method {lab}={OUT_TEST}/{lab}/5312x5312" for lab in labels]
    meth += [f"--pair {lab}=qwen_sca40" for lab in labels]          # tuned minus untuned SCA (40 steps), paired
    aq.task("000r_g_eval_test5312.sh", 120, "TEST evaluation at 5312^2 (scripts/qwen_tune_eval.py); scores images as "
            "they land until the tuned configuration has all 15", code,
            f"{PY} scripts/qwen_tune_eval.py --prompts {TEST} --res 5312x5312 --reference qwen_area {' '.join(meth)} "
            f"--eval {EVAL}/qwen_5312_tuned --results {st.P}/results/qwen_5312_tuned --expect 15 --partial qwen_full "
            "--wait-max 6000", dry, need=40, exclusive=False, post=FINISH.format(name="000r_g_eval_test5312.sh"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+", choices=("prompts", "area4096", "base4096", "area5312", "sp4096", "prog5312",
                                                  "sp5312", "full5312", "eval4096", "eval5312", "test", "evaltest"))
    ap.add_argument("--labels", nargs="*", help="Configuration labels (sp4096, prog5312, eval*, test)")
    ap.add_argument("--code", help="Frozen tree for eval*, test (default NEW_CODE)")
    ap.add_argument("--tag", default="", help="Suffix of the evaluation task name (a re-run after more images)")
    ap.add_argument("--name", help="Display name of the chosen configuration (test; the review page)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    code = a.code or NEW_CODE
    for g in a.groups:
        if g in ("sp4096", "prog5312"):
            {"sp4096": sp4096, "prog5312": prog5312}[g](a.dry, a.labels)
        elif g == "sp5312":
            sp5312(a.dry, a.labels)
        elif g == "eval4096":
            eval_tune(a.dry, "4096x4096", a.labels or ["qt_sca40", *SP], code, a.tag)
        elif g == "eval5312":
            eval_tune(a.dry, "5312x5312", a.labels or list(PROG), code, a.tag)
        elif g == "test":
            test5312(a.dry, a.labels, code, a.name)
        elif g == "evaltest":
            eval_test(a.dry, a.labels, code)
        else:
            {"prompts": tune_prompts, "area4096": area4096, "base4096": base4096, "area5312": area5312,
             "full5312": full5312}[g](a.dry)


if __name__ == "__main__":
    main()
