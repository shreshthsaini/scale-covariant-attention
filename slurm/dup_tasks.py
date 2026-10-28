#!/usr/bin/env python3
"""Spool tasks for the late-duplication fix (.agent/notes/duplication_fix_20261027.md).

pilot: one generation task per variant on the 15 pilot prompts (4 development + the 11 audited uncurated 4096^2
prompts), seed 0, plus one evaluation task (detector, then the extra and base metric sets on the 11 non-development
prompts, timing summary). abl46: one variant on abl50 (46 ablation prompts + 4 development prompts) in chunks
of 5, plus the E7-style evaluation with paired bootstraps against sca_nb. Never overwrites an existing task.
"""

import argparse
from pathlib import Path

S = "/scratch/09032/saini_2/sca"
P = "/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention"
PY = "/work/09032/saini_2/vista/library/uv-envs/sca/bin/python"
PYM = "/work/09032/saini_2/vista/library/uv-envs/sca-metrics/bin/python"
CR = "/work/09032/saini_2/vista/Projects/Exploration/constancy_rope"
ENV = (f"export HF_HOME={S}/../tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 "
       "TRANSFORMERS_OFFLINE=1 XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache "
       "TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton OMP_NUM_THREADS=8\nunset CROPE_TRANSFORMER CROPE_ROPE\n")
ENV = ENV.replace(f"{S}/../tmp/hf", "/scratch/09032/saini_2/tmp/hf")
BASE = ["--method", "sca", "--seed", "0", "--guidance", "4.5", "--pi-table", f"{S}/calib/k1_1024.npz",
        "--gate-sigma", "0.89", "--res", "4096x4096"]

# label -> extra sample.py flags (the published method is "base"; gate flags replace --gate-sigma 0.89)
VARIANTS = {  # in priority order: the worker takes the alphabetically first task
    "dup_base": [],
    "dup_bridge": ["--bridge-cells", "8", "--bridge-k", "16"],
    "dup_ramp70": ["--pi-ramp", "0.7"],
    "dup_lse1": ["--lse-tau", "1.0"],
    "dup_bridge_ramp70": ["--bridge-cells", "8", "--bridge-k", "16", "--pi-ramp", "0.7"],
    "dup_head50": ["--pi-ramp", "0.5", "--pi-ramp-mode", "head"],
    "dup_bridge_mid": ["--bridge-cells", "8", "--bridge-k", "16", "--bridge-lo", "0.5"],
    "dup_txtmid": ["--text-local", "-2.77", "--text-local-lo", "0.5"],
    "dup_win128": ["--win-sched", "0.80:128,0.70:96"],
    "dup_frac_ramp60": ["--glob-pos", "frac", "--pi-ramp", "0.6"],
}


def gen_cmd(label, flags, prompts, out, start=None, end=None):
    args = list(BASE)
    if "--gate-sigma" in flags:
        i = args.index("--gate-sigma")
        del args[i:i + 2]
    args += ["--out", out, "--label", label, "--prompts", prompts] + flags
    if start is not None:
        args += ["--start", str(start), "--end", str(end)]
    return f"{PY} scripts/sample.py " + " ".join(args)


def write(path, text, dry):
    path = Path(path)
    for d in ("pending", "running", "done", "failed", path.parent.name):
        if any((path.parent.parent / d).glob(path.name + "*")):
            print("exists, kept:", d, path.name)
            return
    print("write", path)
    if not dry:
        path.write_text(text)
        path.chmod(0o755)


def pilot(code, hold, dry, variants):
    prompts = f"{S}/pilot15_dup.jsonl"
    names = []
    for k, label in enumerate(variants):
        out = f"{S}/out/dup/{label}/4096x4096"
        name = f"d1{k:02d}_{label}.sh"
        names.append(name)
        write(Path(hold) / name, "#!/bin/bash\n# duplication pilot: " + label + " " + " ".join(VARIANTS[label]) + "\n" + ENV
              + f"cd {code}\n" + gen_cmd(label, VARIANTS[label], prompts, out) + "\n", dry)
    dirs = " ".join([f"sca_nb={S}/out/main/sca_nb/4096x4096"] + [f"{v}={S}/out/dup/{v}/4096x4096" for v in variants])
    pat = "|".join(n[:-3] for n in names)
    ev = f"""#!/bin/bash
# duplication pilot evaluation: detector on all 15 prompts, extra and base metrics on the 11 non-development prompts
{ENV}set -uo pipefail
Q={S}/taskq; E={S}/eval/dup_pilot; mkdir -p $E
while ls $Q/pending $Q/running 2>/dev/null | grep -q -E '^({pat})[.]sh'; do sleep 60; done
ls $Q/failed 2>/dev/null | grep -E '^({pat})[.]sh' | sed 's/^/WARNING failed generation task: /'
cd {code}
{PY} scripts/dup_detect.py --subjects results/dup_audit_20261027/subjects_a4k195.json --device cuda --threads 8 --sides 800,1333 --out $E/det --dirs {dirs}
{PY} scripts/dup_rate.py --dets {S}/eval/dup_detect_20261027 $E/det --subjects results/dup_audit_20261027/subjects_a4k195.json --ids {S}/pilot15_dup.jsonl --thr 0.35 --per-image $E/dup_per_image.csv --out $E/dup_rates.csv | tee $E/dup_rates.txt
{PY} -I {P}/scripts/link_common.py --root {S}/out/evalroot/dup_pilot --res 4096x4096 --seed 0 --prompts {S}/pilot15_dup.jsonl --min-count 1 {dirs}
{PYM} {P}/scripts/eval_extra.py --root {S}/out/evalroot/dup_pilot --prompts {S}/pilot15_dup.jsonl --out $E/extra --metrics ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa --reference-method sca_nb --device cuda --batch-size 16 --threads 8
{PY} {CR}/scripts/eval.py --root {S}/out/evalroot/dup_pilot --prompts {S}/pilot15_dup.jsonl --reference-method sca_nb --res 4096x4096 --out $E/base --metrics clip,pickscore,musiq --device cuda
{PY} - <<'PYEOF'
import glob, json, statistics
for d in sorted(glob.glob("{S}/out/dup/*/4096x4096")) + ["{S}/out/main/sca_nb/4096x4096"]:
    t = [json.load(open(f))["seconds"] for f in sorted(glob.glob(d + "/*.json"), key=lambda f: json.load(open(f)).get("seconds", 0))]
    if t:
        print(d.split("/")[-2], "n", len(t), "median s", round(statistics.median(t), 2), "max s", round(max(t), 2))
PYEOF
date -u +%FT%TZ > $E/DONE
"""
    write(Path(hold) / "d199_eval_dup_pilot.sh", ev, dry)


def abl46(code, hold, dry, label, flags):
    prompts = f"{S}/abl50.jsonl"
    out = f"{S}/out/dup46/{label}/4096x4096"
    names = []
    for a in range(0, 50, 5):
        name = f"d2{a:02d}_{label}.sh"
        names.append(name)
        write(Path(hold) / name, f"#!/bin/bash\n# duplication fix, 46-prompt ablation subset: {label} {' '.join(flags)}\n" + ENV
              + f"cd {code}\n" + gen_cmd(label, flags, prompts, out, a, a + 5) + "\n", dry)
    pat = "|".join(n[:-3] for n in names)
    E = f"{S}/eval/dup46_{label}"
    dirs = f"sca_nb={S}/out/main/sca_nb/4096x4096 {label}={out}"
    ev = f"""#!/bin/bash
# duplication fix: {label} vs sca_nb on the 46 ablation prompts (E7 protocol) plus detector and paired bootstrap
{ENV}set -uo pipefail
Q={S}/taskq; E={E}; mkdir -p $E/paired
while ls $Q/pending $Q/running 2>/dev/null | grep -q -E '^({pat})[.]sh'; do sleep 60; done
ls $Q/failed 2>/dev/null | grep -E '^({pat})[.]sh' | sed 's/^/WARNING failed generation task: /'
cd {code}
{PY} scripts/dup_detect.py --subjects results/dup_audit_20261027/subjects_a4k195.json --device cuda --threads 8 --sides 800,1333 --out $E/det --dirs {label}={out}
{PY} scripts/dup_rate.py --dets {S}/eval/dup_detect_20261027 $E/det --subjects results/dup_audit_20261027/subjects_a4k195.json --ids {prompts} --exclude-ids {S}/dev4.jsonl --thr 0.35 --methods sca_nb,{label} --per-image $E/dup_per_image.csv --out $E/dup_rates.csv | tee $E/dup_rates.txt
{PY} -I {P}/scripts/link_common.py --root {S}/out/evalroot/dup46_{label} --res 4096x4096 --seed 0 --prompts {prompts} --min-count 1 {dirs}
{PYM} {P}/scripts/eval_extra.py --root {S}/out/evalroot/dup46_{label} --prompts {prompts} --out $E/extra --metrics ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa --reference-method sca_nb --device cuda --batch-size 16 --threads 8
{PY} {CR}/scripts/eval.py --root {S}/out/evalroot/dup46_{label} --prompts {prompts} --references /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl --reference-method sca_nb --res 4096x4096 --out $E/base --metrics clip,pickscore,musiq,patch_fid,patch_kid --device cuda
{PY} -I {P}/scripts/paired_bootstrap.py --method {label} --reference sca_nb --draws 2000 --out $E/paired/{label}_vs_sca_nb.csv $E/extra/extra_per_image.csv $E/base/per_image.csv
date -u +%FT%TZ > $E/DONE
"""
    write(Path(hold) / f"d299_eval_dup46_{label}.sh", ev, dry)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", required=True, help="frozen code copy, e.g. /scratch/09032/saini_2/sca/code/<commit>")
    p.add_argument("--hold", default=f"{S}/taskq/hold_dup")
    p.add_argument("--stage", choices=("pilot", "abl46"), default="pilot")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--label", help="abl46: variant label")
    p.add_argument("--flags", default="", help="abl46: sample.py flags for the variant (quoted)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    Path(a.hold).mkdir(parents=True, exist_ok=True)
    if a.stage == "pilot":
        pilot(a.code, a.hold, a.dry_run, a.variants.split(","))
    else:
        abl46(a.code, a.hold, a.dry_run, a.label, a.flags.split())


if __name__ == "__main__":
    main()
