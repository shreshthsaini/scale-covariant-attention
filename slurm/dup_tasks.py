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
    "dup_bridge32": ["--bridge-cells", "8", "--bridge-k", "32"],
    "dup_bridge_lse1": ["--bridge-cells", "8", "--bridge-k", "16", "--lse-tau", "1.0"],
    "dup_head50": ["--pi-ramp", "0.5", "--pi-ramp-mode", "head"],
    "dup_ramp60": ["--pi-ramp", "0.6"],
    "dup_frac_ramp60": ["--glob-pos", "frac", "--pi-ramp", "0.6"],
    "dup_bridge_mid": ["--bridge-cells", "8", "--bridge-k", "16", "--bridge-lo", "0.5"],
    "dup_txtmid": ["--text-local", "-2.77", "--text-local-lo", "0.5"],
    "dup_win128": ["--win-sched", "0.80:128,0.70:96"],
    "dup_frac_g80": ["--glob-pos", "frac", "--gate-sigma", "0.80"],
    "dup_head40": ["--pi-ramp", "0.4", "--pi-ramp-mode", "head"],
    "dup_ss50": ["--scale-sched", "--scale-lo", "0.5"],
    "dup_ss40": ["--scale-sched", "--scale-lo", "0.4"],
}


GEN_PREFIX = ""                 # set by --prefix: generation task names (the worker takes the first in sort order)
EVAL_HOLD = f"{S}/taskq/hold_dup_eval"   # evaluations wait here; slurm/release_dup_evals.sh releases them


def guard(name, pat):
    """Evaluation guard: never sleeps. While a generation task of its group is pending or running, the evaluation
    parks a copy of itself in EVAL_HOLD and exits 3 (as the a4k195 z96x tasks did); release_dup_evals.sh moves it
    back to pending once the group has drained."""
    return (f"# RELEASE_AFTER: ^({pat})[.]sh\nNAME={name}; HOLD={S}/taskq/hold_dup_eval\n"
            f"if ls $Q/pending $Q/running 2>/dev/null | grep -q -E '^({pat})[.]sh'; then mkdir -p $HOLD; "
            f"cp \"$0\" $HOLD/.$NAME.tmp.$$ && mv -f $HOLD/.$NAME.tmp.$$ $HOLD/$NAME; "
            f"echo \"parked $NAME: generation still pending or running\"; exit 3; fi\n")


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


def pilot(code, hold, dry, variants, ename="dup_pilot"):
    prompts = f"{S}/pilot15_dup.jsonl"
    names = []
    for k, label in enumerate(variants):
        out = f"{S}/out/dup/{label}/4096x4096"
        name = f"{GEN_PREFIX}d1{k:02d}_{label}.sh"
        names.append(name)
        write(Path(hold) / name, "#!/bin/bash\n# duplication pilot: " + label + " " + " ".join(VARIANTS[label]) + "\n" + ENV
              + f"cd {code}\n" + gen_cmd(label, VARIANTS[label], prompts, out) + "\n", dry)
    dirs = " ".join([f"sca_nb={S}/out/main/sca_nb/4096x4096"] + [f"{v}={S}/out/dup/{v}/4096x4096" for v in variants])
    pat = "|".join(n[:-3] for n in names)
    ev = f"""#!/bin/bash
# duplication pilot evaluation: detector on all 15 prompts, extra and base metrics on the 11 non-development prompts
{ENV}set -uo pipefail
Q={S}/taskq; E={S}/eval/{ename}; mkdir -p $E
""" + guard(f"{GEN_PREFIX}d199_eval_{ename}.sh", pat) + f"""ls $Q/failed 2>/dev/null | grep -E '^({pat})[.]sh' | sed 's/^/WARNING failed generation task: /'
cd {code}
{PY} scripts/dup_detect.py --subjects results/dup_audit_20261027/subjects_a4k195.json --device cuda --threads 8 --sides 800,1333 --out $E/det --dirs {dirs}
{PY} scripts/dup_rate.py --dets {S}/eval/dup_detect_20261027 $E/det --subjects results/dup_audit_20261027/subjects_a4k195.json --ids {S}/pilot15_dup.jsonl --thr 0.35 --per-image $E/dup_per_image.csv --out $E/dup_rates.csv | tee $E/dup_rates.txt
{PY} scripts/dup_rate.py --dets {S}/eval/dup_detect_20261027 $E/det --subjects results/dup_audit_20261027/subjects_a4k195.json --ids {S}/pilot15_dup.jsonl --thr 0.35 --kinds person,animal --per-image $E/dup_per_image_people.csv --out $E/dup_rates_people.csv | tee $E/dup_rates_people.txt
{PY} -I {P}/scripts/link_common.py --root {S}/out/evalroot/{ename} --res 4096x4096 --seed 0 --prompts {S}/pilot15_dup.jsonl --min-count 1 {dirs}
{PYM} {P}/scripts/eval_extra.py --root {S}/out/evalroot/{ename} --prompts {S}/pilot15_dup.jsonl --out $E/extra --metrics ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa --reference-method sca_nb --device cuda --batch-size 16 --threads 8
{PY} {CR}/scripts/eval.py --root {S}/out/evalroot/{ename} --prompts {S}/pilot15_dup.jsonl --reference-method sca_nb --res 4096x4096 --out $E/base --metrics clip,pickscore,musiq --device cuda
{PY} - <<'PYEOF'
import glob, json, statistics
for d in sorted(glob.glob("{S}/out/dup/*/4096x4096")) + ["{S}/out/main/sca_nb/4096x4096"]:
    t = [json.load(open(f))["seconds"] for f in sorted(glob.glob(d + "/*.json"), key=lambda f: json.load(open(f)).get("seconds", 0))]
    if t:
        print(d.split("/")[-2], "n", len(t), "median s", round(statistics.median(t), 2), "max s", round(max(t), 2))
PYEOF
date -u +%FT%TZ > $E/DONE
"""
    write(Path(EVAL_HOLD) / f"{GEN_PREFIX}d199_eval_{ename}.sh", ev, dry)


def abl46(code, hold, dry, label, flags):
    prompts = f"{S}/abl50.jsonl"
    out = f"{S}/out/dup46/{label}/4096x4096"
    names = []
    for a in range(0, 50, 5):
        name = f"{GEN_PREFIX}d2{a:02d}_{label}.sh"
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
""" + guard(f"z97_eval_dup46_{label}.sh", pat) + f"""ls $Q/failed 2>/dev/null | grep -E '^({pat})[.]sh' | sed 's/^/WARNING failed generation task: /'
cd {code}
{PY} scripts/dup_detect.py --subjects results/dup_audit_20261027/subjects_a4k195.json --device cuda --threads 8 --sides 800,1333 --out $E/det --dirs {label}={out}
{PY} scripts/dup_rate.py --dets {S}/eval/dup_detect_20261027 $E/det --subjects results/dup_audit_20261027/subjects_a4k195.json --ids {prompts} --exclude-ids {S}/dev4.jsonl --thr 0.35 --methods sca_nb,{label} --per-image $E/dup_per_image.csv --out $E/dup_rates.csv | tee $E/dup_rates.txt
{PY} -I {P}/scripts/link_common.py --root {S}/out/evalroot/dup46_{label} --res 4096x4096 --seed 0 --prompts {prompts} --min-count 1 {dirs}
{PYM} {P}/scripts/eval_extra.py --root {S}/out/evalroot/dup46_{label} --prompts {prompts} --out $E/extra --metrics ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa --reference-method sca_nb --device cuda --batch-size 16 --threads 8
{PY} {CR}/scripts/eval.py --root {S}/out/evalroot/dup46_{label} --prompts {prompts} --references /scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl --reference-method sca_nb --res 4096x4096 --out $E/base --metrics clip,pickscore,musiq,patch_fid,patch_kid --device cuda
{PY} -I {P}/scripts/paired_bootstrap.py --method {label} --reference sca_nb --draws 2000 --out $E/paired/{label}_vs_sca_nb.csv $E/extra/extra_per_image.csv $E/base/per_image.csv
date -u +%FT%TZ > $E/DONE
"""
    write(Path(EVAL_HOLD) / f"z97_eval_dup46_{label}.sh", ev, dry)


# ----------------------------------------------------------------------------- 195-prompt headline rerun
A4K = "/scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl"
REFS = "/scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl"
HRGEN = f"{S}/hrgen1k.jsonl"
ROWS = {  # key: (prompts, n, chunk, res, seed, model, old SCA (label, dir), AReA dir or None, eval name)
    "m4k": (A4K, 195, 5, "4096x4096", 0, "dev", ("sca_nb", f"{S}/out/main/sca_nb/4096x4096"), f"{S}/out/main/area/4096x4096", "a4k195_main"),
    "m24": (A4K, 195, 5, "2048x4096", 0, "dev", ("sca_nb", f"{S}/out/main/sca_nb/2048x4096"), f"{S}/out/main/area/2048x4096", "a4k195_2048x4096"),
    "m26": (A4K, 195, 5, "2048x6144", 0, "dev", ("sca_nb", f"{S}/out/main/sca_nb/2048x6144"), f"{S}/out/main/area/2048x6144", "a4k195_2048x6144"),
    "s1": (A4K, 195, 5, "4096x4096", 1, "dev", ("sca_nb", f"{S}/out/seed1/sca_nb/4096x4096"), f"{S}/out/seed1/area/4096x4096", "a4k195_seed1"),
    "s2": (A4K, 195, 5, "4096x4096", 2, "dev", ("sca_nb", f"{S}/out/seed2/sca_nb/4096x4096"), f"{S}/out/seed2/area/4096x4096", "a4k195_seed2"),
    "pg": (A4K, 195, 5, "4096x4096", 0, "prog", ("sca_prog", f"{S}/out/prog/sca_prog/4096x4096"), None, "prog195"),
    "kr": (A4K, 195, 5, "4096x4096", 0, "krea", ("sca_krea", f"{S}/out/krea/sca_krea/4096x4096"), f"{S}/out/krea/area/4096x4096", "krea195"),
    "hd": (HRGEN, 1000, 10, "4096x4096", 0, "dev", ("sca_nb", f"{S}/out/hrgen/sca_nb/4096x4096"), f"{S}/out/hrgen/area/4096x4096", "hrgen1k_dev"),
    "hk": (HRGEN, 1000, 10, "4096x4096", 0, "krea", ("sca_krea", f"{S}/out/hrgen_krea/sca_krea/4096x4096"), None, "hrgen1k_krea"),
}


def rerun_gen_cmd(row, label, flags, a, b):
    prompts, n, chunk, res, seed, model, old, area, ev = ROWS[row]
    if model == "prog":
        return (f"{PY} scripts/sample_progressive.py --prompts {prompts} --seed {seed} --out {rerun_dir(row, label)} "
                f"--pi-table {S}/calib/k1_1024.npz --gate-sigma 0.89 --label {label}_prog --start {a} --end {b} " + " ".join(flags))
    args = ["--method", "sca", "--seed", str(seed), "--guidance", "4.5", "--res", res, "--gate-sigma", "0.89",
            "--out", rerun_dir(row, label), "--prompts", prompts, "--start", str(a), "--end", str(b)]
    if model == "krea":
        args += ["--model", "krea", "--pi-table", f"{S}/calib/krea_1024.npz", "--label", f"{label}_krea"]
    else:
        args += ["--pi-table", f"{S}/calib/k1_1024.npz", "--label", label]
    return f"{PY} scripts/sample.py " + " ".join(args + list(flags))


def rerun_dir(row, label):
    prompts, n, chunk, res, seed, model, old, area, ev = ROWS[row]
    tag = {"dev": label, "krea": f"{label}_krea", "prog": f"{label}_prog"}[model]
    camp = {"m4k": "dupfix", "m24": "dupfix", "m26": "dupfix", "s1": "dupfix_seed1", "s2": "dupfix_seed2",
            "pg": "dupfix_prog", "kr": "dupfix_krea", "hd": "dupfix_hrgen", "hk": "dupfix_hrgen_krea"}[row]
    return f"{S}/out/{camp}/{tag}/{res}"


def rerun_eval(row, label, names, code):
    prompts, n, chunk, res, seed, model, old, area, ev = ROWS[row]
    tag = {"dev": label, "krea": f"{label}_krea", "prog": f"{label}_prog"}[model]
    spec = [f"{tag}|{seed}|{rerun_dir(row, label)}", f"{old[0]}|{seed}|{old[1]}"]
    pairs = [f"{tag}:{old[0]}"]
    if area:
        spec.append(f"area|{seed}|{area}")
        pairs += [f"{tag}:area", f"{old[0]}:area"]
    if row == "pg":
        hr = ",".join(f"{S}/out/{d}/hrdit_dev_author/4096x4096" for d in ("baseline_expansion_20261017_rest145", "baseline_expansion_20261017_v1", "baseline_expansion_20261017_dev4"))
        spec.append(f"hrdit_dev_author|0|{hr}")
        pairs += [f"{tag}:hrdit_dev_author"]
    refs = prompts == A4K
    base_metrics = "clip,pickscore,musiq,fid,kid,patch_fid,patch_kid" if refs else "clip,pickscore,musiq"
    extra_metrics = "ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa,is"
    name = f"z98{list(ROWS).index(row)}_eval_{ev}_{label}.sh"  # sorts after every generation task
    E, R = f"{S}/eval/{ev}_{label}", f"{P}/results/{ev}_{label}"
    pat = "|".join(x[:-3] for x in names)
    ref_arg = f"--references {REFS} " if refs else ""
    corrected = (f"""mkdir -p $E/corrected_fidp/features && cp -p $E/base/features/*.npz $E/corrected_fidp/features/ && cp -p $E/base/evaluation_cache.json $E/corrected_fidp/ || {{ echo "feature copy failed"; exit 7; }}
(cd {P} && WANDB_MODE=disabled run {PY} repro/evaluate_corrected.py --root $E/root --prompts {prompts} --references {REFS} --reference-method {tag} --reference-clusters images --res {res} --out $E/corrected_fidp --metrics fid,kid,patch_fid,patch_kid --device cuda --bootstrap 0 --threads 8 --no-wandb)
""" if refs else "")
    det = "" if prompts != A4K else f"""run {PY} scripts/dup_detect.py --subjects results/dup_audit_20261027/subjects_a4k195.json --device cuda --threads 8 --sides 800,1333 --out $E/det --dirs {tag}={rerun_dir(row, label)}
{PY} scripts/dup_rate.py --dets {S}/eval/dup_detect_20261027 $E/det --subjects results/dup_audit_20261027/subjects_a4k195.json --thr 0.35 --kinds person,animal --methods {tag} --out $E/dup_rates_people.csv | tee -a $E/run.log
{PY} scripts/dup_rate.py --dets {S}/eval/dup_detect_20261027 $E/det --subjects results/dup_audit_20261027/subjects_a4k195.json --thr 0.35 --methods {tag} --out $E/dup_rates_any.csv | tee -a $E/run.log
"""
    text = f"""#!/bin/bash
# Duplication-fix rerun, row {row} ({ev}): {tag} vs {old[0]}{' and AReA' if area else ''}; clone of taskq/done/a000_z960_eval_a4k195_main.sh
# (same metric commands, corrected patch FID for Aesthetic-4K rows, paired bootstrap 2000 draws) plus the duplicate detector.
{ENV}set -uo pipefail
Q={S}/taskq; E={E}; R={R}
{guard(name, pat)}if ls $Q/failed 2>/dev/null | grep -q -E '^({pat})[.]sh'; then echo "generation task failed; not evaluating"; exit 3; fi
if [ -e $E/DONE ]; then echo "already complete: $E"; exit 0; fi
[ -e $E ] && mv $E $E.partial_$(date +%Y%m%dT%H%M%S)
mkdir -p $E/paired $R
run(){{ "$@" 2>&1 | tee -a $E/run.log; local rc=${{PIPESTATUS[0]}}; if [ $rc -ne 0 ]; then echo "step failed rc=$rc: $*"; exit $rc; fi; }}
{PY} -I - "{prompts}" "{';'.join(spec)}" "{res}" "$E/root" <<'PYEOF'
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
            os.symlink(src, os.path.join(d, os.path.basename(src)))
        n += 1
    print("GATE", method, n, "of", len(ids), flush=True)
sys.exit(1 if bad else 0)
PYEOF
[ $? -eq 0 ] || {{ echo "GATE failed"; exit 6; }}
cd {code}
{det}run {PYM} {P}/scripts/eval_extra.py --root $E/root --prompts {prompts} --out $E/extra --metrics {extra_metrics} --reference-method {tag} --device cuda --batch-size 16 --threads 8
run {PY} {CR}/scripts/eval.py --root $E/root --prompts {prompts} {ref_arg}--reference-method {tag} --res {res} --out $E/base --metrics {base_metrics} --device cuda
{corrected}for pair in {' '.join(pairs)}; do a=${{pair%%:*}}; b=${{pair#*:}}
  run {PY} -I {P}/scripts/paired_bootstrap.py --method $a --reference $b --draws 2000 --out $E/paired/${{a}}_vs_${{b}}.csv $E/extra/extra_per_image.csv $E/base/per_image.csv
done
cp $E/base/summary.csv $R/base_summary.csv && cp $E/extra/extra_summary.csv $R/extra_summary.csv && cp $E/paired/*.csv $R/ && cp $E/dup_rates_*.csv $R/ 2>/dev/null
[ -d $E/corrected_fidp ] && mkdir -p $R/corrected_fidp && cp $E/corrected_fidp/summary.csv $E/corrected_fidp/paired_deltas.csv $R/corrected_fidp/
(cd $R && sha256sum $(find . -name '*.csv' | sort) > SHA256SUMS)
date -u +%FT%TZ > $E/DONE; echo "DONE {ev}_{label}: $E -> $R"
"""
    return name, text


def rerun195(code, hold, dry, label, flags, rows):
    total = 0
    for row in rows:
        prompts, n, chunk, res, seed, model, old, area, ev = ROWS[row]
        names = []
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            name = f"{GEN_PREFIX}d9{list(ROWS).index(row)}_{row}_{label}_{a:04d}_{b:04d}.sh"
            names.append(name)
            write(Path(hold) / name, f"#!/bin/bash\n# duplication-fix rerun {row}: {label} {' '.join(flags)}\n" + ENV
                  + f"cd {code}\n" + rerun_gen_cmd(row, label, flags, a, b) + "\n", dry)
        total += len(names)
        ename, etext = rerun_eval(row, label, names, code)
        write(Path(EVAL_HOLD) / ename, etext, dry)
    print(f"{total} generation tasks")


def main():
    global GEN_PREFIX, EVAL_HOLD
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", required=True, help="frozen code copy, e.g. /scratch/09032/saini_2/sca/code/<commit>")
    p.add_argument("--hold", default=f"{S}/taskq/hold_dup")
    p.add_argument("--stage", choices=("pilot", "abl46", "rerun195"), default="pilot")
    p.add_argument("--rows", default="m4k,m24,m26,s1,s2,pg,kr,hd,hk", help="rerun195 rows")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--label", help="abl46: variant label")
    p.add_argument("--flags", default="", help="abl46: sample.py flags for the variant (quoted)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--prefix", default="", help="generation task name prefix, e.g. a_ to run first")
    p.add_argument("--eval-name", default="dup_pilot", help="pilot: evaluation directory name under eval/")
    p.add_argument("--eval-hold", help="write evaluation tasks here instead of taskq/hold_dup_eval (staging)")
    a = p.parse_args()
    Path(a.hold).mkdir(parents=True, exist_ok=True)
    Path(EVAL_HOLD).mkdir(parents=True, exist_ok=True)
    GEN_PREFIX = a.prefix
    if a.eval_hold:
        EVAL_HOLD = a.eval_hold
        Path(EVAL_HOLD).mkdir(parents=True, exist_ok=True)
    if a.stage == "pilot":
        pilot(a.code, a.hold, a.dry_run, a.variants.split(","), a.eval_name)
    elif a.stage == "abl46":
        abl46(a.code, a.hold, a.dry_run, a.label, a.flags.split())
    else:
        rerun195(a.code, a.hold, a.dry_run, a.label, a.flags.split(), a.rows.split(","))


if __name__ == "__main__":
    main()
