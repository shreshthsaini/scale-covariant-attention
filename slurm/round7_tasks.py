#!/usr/bin/env python3
"""Round-7 duplication-fix GPU tasks (.agent/notes/duplication_fix_round7.md); prefix a7_ so they sort before the
review-round r1_ tasks. Never overwrites a task (dup_tasks.write).

Stages
  gen      LABEL with FLAGS (on top of sca_v2's flags) on PROMPTS into OUT, chunks of --chunk prompts, tasks
           a7_<tag>NN_<label>.sh. Each task leaves a marker in --markers; the last one moves --release (an evaluation
           parked in taskq/hold_dup_eval) to pending.
  abl46    generation of an r7_ label on the 46-prompt subset (r6/abl46.jsonl) into r7/out/<label>/4096x4096 and its
           evaluation a7_<eval_tag>_eval_<label> (eval/r7_46_<label>): detector, duplication rates vs sca_v2 and AReA,
           the E7 metrics, paired bootstraps label vs sca_v2, label vs AReA and sca_v2 vs AReA, time medians. Metric
           caches are seeded from the round-6 evaluation of the same prompts, so sca_v2 and AReA are not rescored.
  eval195  full evaluation of LABEL (images in --out) on the 195 Aesthetic-4K prompts at 4096^2 against sca_v2 and AReA
           (eval/r7_195_<label>, results/r7_195_<label>): detector and people/animal and any-subject duplication rates
           (the people/animal rate is over the 95 eligible prompts), the 15 metrics of the a4k195 rows (IR, aesthetic,
           HPSv2, HPSv2.1, CLIP-IQA, ARNIQA, IS, CLIP, PickScore, MUSIQ, MUSIQ crop, FID, KID, patch FID, patch KID),
           corrected FID / patch FID, paired bootstraps label vs sca_v2, label vs AReA, sca_v2 vs AReA, time medians.
           Caches seeded from eval/a4k195_main_sca_v2. Set-level intervals: CPU bootstrap afterwards (cpu_boot.sh).
Usage: slurm/round7_tasks.py --code /scratch/09032/saini_2/sca/code/<freeze> --stage abl46 --label r7_x --flags "..."
"""

import argparse
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dup_tasks as dt  # noqa: E402

S, P, PY, PYM, CR, ENV = dt.S, dt.P, dt.PY, dt.PYM, dt.CR, dt.ENV
R6, R7 = f"{S}/r6", f"{S}/r7"
ABL46 = f"{R6}/abl46.jsonl"
A4K = dt.A4K
REFS = "/scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl"
SCA_V2_DIR = f"{S}/out/dupfix/sca_v2/4096x4096"
AREA_DIR = f"{S}/out/main/area/4096x4096"
SCA_V2 = ["--bridge-cells", "8", "--bridge-k", "16", "--win-sched", "0.88:128,0.86:112,0.84:96,0.82:80",
          "--win-pi-match", "--win-pi-interp"]
QUEUE = f"{S}/taskq/pending"
HOLD = dt.EVAL_HOLD
SEED46 = f"{S}/eval/r6_46_r6_ffl"            # round-6 evaluation that scored sca_v2 and AReA on the 46 prompts
SEED195 = f"{S}/eval/a4k195_main_sca_v2"     # evaluation that scored sca_v2 and AReA on the 195 prompts
SUB = "results/dup_audit_20261027/subjects_a4k195.json"


def header(name, what, seconds):
    return f"#!/bin/bash\n# SCA_MIN_SECONDS: {seconds}\n# round 7 ({name}): {what}\n" + ENV


def gen_cmd(label, flags, prompts, out, a=None, b=None):
    """sample.py command with one retry after 30 s (stale Lustre handles in the shared Triton cache, round 6)."""
    c = dt.gen_cmd(label, SCA_V2 + flags, prompts, out, a, b)
    return f"{c} || {{ sleep 30; {c}; }}"


def n_prompts(path):
    return sum(1 for ln in open(path) if ln.strip())


def gen(code, dry, label, flags, prompts, out, chunk, tag, markers, release, queue=QUEUE):
    n = n_prompts(prompts)
    starts = list(range(0, n, chunk))
    names = [f"a7_{tag}{i:02d}_{label}.sh" for i in range(len(starts))]
    pat = "|".join(x[:-3] for x in names)
    for a, name in zip(starts, names):
        text = (header(name, f"{label} on {Path(prompts).name} [{a}, {min(a + chunk, n)}): {' '.join(flags)}", 3600)
                + f"cd {code}\n" + gen_cmd(label, flags, prompts, out, a, min(a + chunk, n)) + " || exit $?\n")
        text += dt.self_release(markers, name, pat, len(names), release) if release else "exit 0\n"
        dt.write(Path(queue) / name, text, dry)
    return names, pat


def seed_caches(E, src):
    """Shell lines that copy the metric caches of an earlier evaluation (same images) into $E before scoring."""
    return (f"mkdir -p {E}/extra {E}/base/features\n"
            f"[ -f {src}/extra/extra_cache.jsonl ] && [ ! -f {E}/extra/extra_cache.jsonl ] && cp -p {src}/extra/extra_cache.jsonl {E}/extra/\n"
            f"[ -f {src}/base/evaluation_cache.json ] && [ ! -f {E}/base/evaluation_cache.json ] && cp -p {src}/base/evaluation_cache.json {E}/base/ "
            f"&& cp -p {src}/base/features/*.npz {E}/base/features/ 2>/dev/null\n")


def timing(label, out, ids_file):
    return f"""{PY} - <<'PYEOF' | tee $E/timing.txt
import glob, json, statistics
ids = {{json.loads(l)["id"] for l in open("{ids_file}") if l.strip()}}
for name, d in (("{label}", "{out}"), ("sca_v2", "{SCA_V2_DIR}"), ("area", "{AREA_DIR}")):
    t = [json.load(open(f)) for f in glob.glob(d + "/*.json")]
    t = [x["seconds"] for x in t if x.get("config", {{}}).get("prompt_id") in ids]
    print(name, "n", len(t), "median s", round(statistics.median(t), 2) if t else None)
PYEOF
"""


def abl46(code, dry, label, flags, chunk, gen_tag, eval_tag, queue=QUEUE):
    out = f"{R7}/out/{label}/4096x4096"
    E = f"{S}/eval/r7_46_{label}"
    ename = f"a7_{eval_tag}_eval_{label}.sh"
    names, pat = gen(code, dry, label, flags, ABL46, out, chunk, gen_tag, f"{E}/gen_done", ename, queue)
    dirs = f"sca_v2={SCA_V2_DIR} {label}={out} area={AREA_DIR}"
    det = f"{S}/eval/dup_detect_20261027 {S}/eval/a4k195_main_sca_v2/det $E/det"
    root = f"{S}/out/evalroot/r7_46_{label}"
    ev = header(ename, f"{label} vs sca_v2 and AReA on the 46 prompts: detector, metrics, paired bootstrap", 5400) + f"""set -uo pipefail
Q={S}/taskq; E={E}; mkdir -p $E/paired
""" + dt.guard(ename, pat, (f"{E}/gen_done", len(names))) + f"""ls $Q/failed 2>/dev/null | grep -E '^({pat})[.]sh' | sed 's/^/WARNING failed generation task: /'
cd {code}
{PY} scripts/dup_detect.py --subjects {SUB} --device cuda --threads 8 --sides 800,1333 --out $E/det --dirs {label}={out}
{PY} scripts/dup_rate.py --dets {det} --subjects {SUB} --ids {ABL46} --thr 0.35 --kinds person,animal --methods sca_v2,{label},area --ref sca_v2 --per-image $E/dup_per_image_people.csv --out $E/dup_rates_people.csv | tee $E/dup_rates_people.txt
{PY} scripts/dup_rate.py --dets {det} --subjects {SUB} --ids {ABL46} --thr 0.35 --methods sca_v2,{label},area --ref sca_v2 --per-image $E/dup_per_image_any.csv --out $E/dup_rates_any.csv | tee $E/dup_rates_any.txt
{PY} -I {P}/scripts/link_common.py --root {root} --res 4096x4096 --seed 0 --prompts {ABL46} --min-count 46 {dirs}
""" + seed_caches("$E", SEED46) + f"""{PYM} {P}/scripts/eval_extra.py --root {root} --prompts {ABL46} --out $E/extra --metrics ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa --reference-method sca_v2 --device cuda --batch-size 16 --threads 8
{PY} {CR}/scripts/eval.py --root {root} --prompts {ABL46} --references {REFS} --reference-method sca_v2 --res 4096x4096 --out $E/base --metrics clip,pickscore,musiq,patch_fid,patch_kid --device cuda
for pair in "{label} sca_v2" "{label} area" "sca_v2 area"; do set -- $pair
  {PY} -I {P}/scripts/paired_bootstrap.py --method $1 --reference $2 --draws 2000 --out $E/paired/$1_vs_$2.csv $E/extra/extra_per_image.csv $E/base/per_image.csv
done
""" + timing(label, out, ABL46) + "date -u +%FT%TZ > $E/DONE\n"
    dt.write(Path(HOLD) / ename, ev, dry)


def eval195(code, dry, label, out, tag, gen_names=None, gen_pat=None):
    """Full 195-prompt evaluation; parked in taskq/hold_dup_eval when gen_pat is given (released by the last
    generation task), else written to pending."""
    E, R = f"{S}/eval/r7_195_{label}", f"{P}/results/r7_195_{label}"
    ename = f"a7_{tag}_eval195_{label}.sh"
    root = "$E/root"
    det = f"{S}/eval/dup_detect_20261027 {SEED195}/det $E/det"
    spec = f"{label}|0|{out};sca_v2|0|{SCA_V2_DIR};area|0|{AREA_DIR}"
    guard = (dt.guard(ename, gen_pat, (f"{E}/gen_done", len(gen_names))) +
             f"if ls $Q/failed 2>/dev/null | grep -q -E '^({gen_pat})[.]sh'; then echo \"generation task failed; not evaluating\"; exit 3; fi\n"
             ) if gen_pat else ""
    text = header(ename, f"{label} vs sca_v2 and AReA on the 195 prompts at 4096^2: 15 metrics, detector, paired bootstrap", 7200) + f"""set -uo pipefail
Q={S}/taskq; E={E}; R={R}
{guard}if [ -e $E/DONE ]; then echo "already complete: $E"; exit 0; fi
mkdir -p $E/paired $R
run(){{ "$@" 2>&1 | tee -a $E/run.log; local rc=${{PIPESTATUS[0]}}; if [ $rc -ne 0 ]; then echo "step failed rc=$rc: $*"; exit $rc; fi; }}
rm -rf {root}
{PY} -I - "{A4K}" "{spec}" "4096x4096" "{root}" <<'PYEOF'
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
run {PY} scripts/dup_detect.py --subjects {SUB} --device cuda --threads 8 --sides 800,1333 --out $E/det --dirs {label}={out}
{PY} scripts/dup_rate.py --dets {det} --subjects {SUB} --thr 0.35 --kinds person,animal --methods sca_v2,{label},area --ref sca_v2 --per-image $E/dup_per_image_people.csv --out $E/dup_rates_people.csv | tee $E/dup_rates_people.txt
{PY} scripts/dup_rate.py --dets {det} --subjects {SUB} --thr 0.35 --methods sca_v2,{label},area --ref sca_v2 --per-image $E/dup_per_image_any.csv --out $E/dup_rates_any.csv | tee $E/dup_rates_any.txt
""" + seed_caches("$E", SEED195) + f"""run {PYM} {P}/scripts/eval_extra.py --root {root} --prompts {A4K} --out $E/extra --metrics ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa,is --reference-method sca_v2 --device cuda --batch-size 16 --threads 8
run {PY} {CR}/scripts/eval.py --root {root} --prompts {A4K} --references {REFS} --reference-method sca_v2 --res 4096x4096 --out $E/base --metrics clip,pickscore,musiq,fid,kid,patch_fid,patch_kid --device cuda
mkdir -p $E/corrected_fidp/features && cp -p $E/base/features/*.npz $E/corrected_fidp/features/ && cp -p $E/base/evaluation_cache.json $E/corrected_fidp/ || {{ echo "feature copy failed"; exit 7; }}
(cd {P} && WANDB_MODE=disabled run {PY} repro/evaluate_corrected.py --root {root} --prompts {A4K} --references {REFS} --reference-method sca_v2 --reference-clusters images --res 4096x4096 --out $E/corrected_fidp --metrics fid,kid,patch_fid,patch_kid --device cuda --bootstrap 0 --threads 8 --no-wandb)
for pair in {label}:sca_v2 {label}:area sca_v2:area; do a=${{pair%%:*}}; b=${{pair#*:}}
  run {PY} -I {P}/scripts/paired_bootstrap.py --method $a --reference $b --draws 2000 --out $E/paired/${{a}}_vs_${{b}}.csv $E/extra/extra_per_image.csv $E/base/per_image.csv
done
""" + timing(label, out, A4K) + f"""cp $E/base/summary.csv $R/base_summary.csv && cp $E/extra/extra_summary.csv $R/extra_summary.csv && cp $E/paired/*.csv $R/ && cp $E/dup_rates_*.csv $E/dup_per_image_*.csv $E/timing.txt $R/ 2>/dev/null
[ -d $E/corrected_fidp ] && mkdir -p $R/corrected_fidp && cp $E/corrected_fidp/summary.csv $E/corrected_fidp/paired_deltas.csv $R/corrected_fidp/
(cd $R && sha256sum $(find . -name '*.csv' -o -name '*.txt' | sort) > SHA256SUMS)
date -u +%FT%TZ > $E/DONE; echo "DONE r7_195_{label}: $E -> $R"
"""
    dt.write(Path(HOLD if gen_pat else QUEUE) / ename, text, dry)
    return ename


def smoke(code, dry, candidates, hold):
    """a7_a000_smoke: ffl near "bridge" with an empty head set is sca_v2 computed through the split (two prompts; PSNR
    against the sca_v2 images, >= 24 dB as in round 6), then one prompt of every candidate (crash check, time). On
    PASS it moves the generation tasks parked in `hold` to pending."""
    out = f"{R7}/smoke"
    name = "a7_a000_smoke.sh"
    ident = ["--text-mode", "ffl", "--text-mode-lo", "0.5", "--ffl-near", "bridge", "--ffl-head-frac", "0.0001"]
    lines = [header(name, "round-7 GPU identity and crash check", 3000), f"set -uo pipefail\ncd {code}\n",
             dt.gen_cmd("r7_id0", SCA_V2 + ident, ABL46, f"{out}/id0", 4, 6) + "\n"]
    for lab, fl in candidates.items():
        lines.append(dt.gen_cmd(lab, SCA_V2 + fl, ABL46, f"{out}/{lab}", 4, 5) + f" || echo 'SMOKE FAIL {lab}'\n")
    lines.append(f"""{PY} - <<'PYEOF'
import glob, json, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
rows = []
for f in sorted(glob.glob("{out}/id0/*.png")):
    ref = "{SCA_V2_DIR}/" + f.split("/")[-1]
    a = np.asarray(Image.open(f).convert("RGB"), dtype=np.float64); b = np.asarray(Image.open(ref).convert("RGB"), dtype=np.float64)
    mse = ((a - b) ** 2).mean(); rows.append((f.split("/")[-1], 10 * np.log10(255 ** 2 / max(mse, 1e-12))))
for r in rows: print("identity PSNR id0 vs sca_v2", r[0], round(r[1], 2), "dB")
for d in sorted(glob.glob("{out}/*/")):
    for j in sorted(glob.glob(d + "*.json")):
        x = json.load(open(j)); print(d.split("/")[-2], x["config"]["prompt_id"], "seconds", round(x["seconds"], 1), x.get("sca_log", [])[:3])
labels = {sorted(candidates)!r}
ok = len(rows) == 2 and min(r[1] for r in rows) >= 24.0 and all(glob.glob("{out}/" + l + "/*.png") for l in labels)
print("SMOKE", "PASS" if ok else "FAIL")
open("{out}/VERDICT", "w").write("PASS" if ok else "FAIL")
PYEOF
date -u +%FT%TZ > {out}/DONE
if [ "$(cat {out}/VERDICT)" = PASS ]; then n=$(ls {hold}/a7_*.sh 2>/dev/null | wc -l); mv {hold}/a7_*.sh {QUEUE}/ 2>/dev/null; echo "released $n generation tasks"; else echo "smoke failed: generation tasks stay in {hold}"; exit 4; fi
""")
    dt.write(Path(QUEUE) / name, "".join(lines), dry)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--code", required=True)
    p.add_argument("--stage", choices=("gen", "abl46", "eval195", "gen195", "smoke"), required=True)
    p.add_argument("--label", default="")
    p.add_argument("--candidates", default="{}", help="smoke: JSON {label: flags string}")
    p.add_argument("--hold", help="abl46 / gen195: park generation tasks here (the smoke task releases them)")
    p.add_argument("--flags", default="")
    p.add_argument("--prompts", help="gen / gen195: prompt file (default: all 195)")
    p.add_argument("--out", help="gen / gen195 / eval195: image directory")
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--gen-tag", default="b0")
    p.add_argument("--eval-tag", default="z0")
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    if not Path(a.code, "FREEZE_COMMIT").exists():
        raise SystemExit(f"{a.code} is not a frozen code tree (no FREEZE_COMMIT)")
    flags = shlex.split(a.flags)
    if a.hold:
        Path(a.hold).mkdir(parents=True, exist_ok=True)
    if a.stage == "smoke":
        smoke(a.code, a.dry, {k: shlex.split(v) for k, v in json.loads(a.candidates).items()}, a.hold or f"{S}/taskq/hold_r7")
        return
    if a.stage == "abl46":
        if not a.label.startswith("r7_"):
            raise SystemExit("--label r7_<name> is required for abl46")
        abl46(a.code, a.dry, a.label, flags, a.chunk, a.gen_tag, a.eval_tag, a.hold or QUEUE)
    elif a.stage == "gen195":   # generation on --prompts (default the 195) into --out, then the 195 evaluation
        prompts = a.prompts or A4K
        ename = f"a7_{a.eval_tag}_eval195_{a.label}.sh"
        names, pat = gen(a.code, a.dry, a.label, flags, prompts, a.out, a.chunk, a.gen_tag,
                         f"{S}/eval/r7_195_{a.label}/gen_done", ename, a.hold or QUEUE)
        eval195(a.code, a.dry, a.label, a.out, a.eval_tag, names, pat)
    elif a.stage == "eval195":
        eval195(a.code, a.dry, a.label, a.out, a.eval_tag)
    else:
        gen(a.code, a.dry, a.label, flags, a.prompts, a.out, a.chunk, a.gen_tag, f"{R7}/markers/{a.label}", None)


if __name__ == "__main__":
    main()
