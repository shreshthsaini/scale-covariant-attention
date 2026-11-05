#!/usr/bin/env python3
"""Round-6 duplication-fix GPU tasks (.agent/notes/duplication_fix_round6.md); prefix a6_ so they sort before the
review-round r1_ tasks. Never overwrites a task (dup_tasks.write).

Stages
  smoke   a6_a000_smoke: text-mode forward on GPU. gtg with gamma 0 is the sca_v2 operator computed through the exact
          split of the window softmax (two prompts; PSNR against the sca_v2 campaign images is the identity check),
          then one prompt of every candidate (crash check).
  trace   a6_t*: per-step x0 traces (scripts/sample_trace.py) of sca_v2 and AReA on the round-6 selection
          (r6/trace_selection.json: 10 detector-flagged, human-confirmed sca_v2 duplicates, crop on one copy).
  abl46   per label: generation a6_d2NN_<label> on the 46-prompt subset (r6/abl46.jsonl, chunks of --chunk), and
          a6_z97_eval_<label> (parked in taskq/hold_dup_eval, released by the last generation task): detector,
          duplication rates vs sca_v2 and AReA, the E7 metrics of sca_v2, the label and AReA, paired bootstraps
          label vs sca_v2, label vs AReA and sca_v2 vs AReA, and the time medians.
Usage: slurm/round6_tasks.py --code /scratch/09032/saini_2/sca/code/<freeze> --stage abl46 --label r6_ctw \
          --flags "--text-mode ctw" [--dry]
"""

import argparse
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dup_tasks as dt  # noqa: E402

S, P, PY, PYM, CR, ENV = dt.S, dt.P, dt.PY, dt.PYM, dt.CR, dt.ENV
R6 = f"{S}/r6"
ABL46 = f"{R6}/abl46.jsonl"
REFS = "/scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl"
SCA_V2_DIR = f"{S}/out/dupfix/sca_v2/4096x4096"
AREA_DIR = f"{S}/out/main/area/4096x4096"
SCA_V2 = ["--bridge-cells", "8", "--bridge-k", "16", "--win-sched", "0.88:128,0.86:112,0.84:96,0.82:80",
          "--win-pi-match", "--win-pi-interp"]
QUEUE = f"{S}/taskq/pending"
HOLD = f"{S}/taskq/hold_r6"      # abl46 generation waits here until the smoke task releases it


def header(name, what, seconds):
    return f"#!/bin/bash\n# SCA_MIN_SECONDS: {seconds}\n# round 6 ({name}): {what}\n" + ENV


def gen(label, flags, prompts, out, a=None, b=None):
    return dt.gen_cmd(label, SCA_V2 + flags, prompts, out, a, b)


def smoke(code, dry, candidates):
    out = f"{R6}/smoke"
    lines = [header("a6_a000_smoke", "text-mode GPU identity and crash check", 2400), f"set -uo pipefail\ncd {code}\n",
             gen("r6_split0", ["--text-mode", "gtg", "--text-gamma", "0"], ABL46, f"{out}/split0", 4, 6) + "\n"]
    for lab, fl in candidates.items():
        lines.append(gen(lab, fl, ABL46, f"{out}/{lab}", 4, 5) + f" || echo 'SMOKE FAIL {lab}'\n")
    lines.append(f"""{PY} - <<'PYEOF'
import glob, json, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
rows = []
for f in sorted(glob.glob("{out}/split0/*.png")):
    ref = "{SCA_V2_DIR}/" + f.split("/")[-1]
    a = np.asarray(Image.open(f).convert("RGB"), dtype=np.float64); b = np.asarray(Image.open(ref).convert("RGB"), dtype=np.float64)
    mse = ((a - b) ** 2).mean(); rows.append((f.split("/")[-1], 10 * np.log10(255 ** 2 / max(mse, 1e-12))))
for r in rows: print("identity PSNR split0 vs sca_v2", r[0], round(r[1], 2), "dB")
for d in sorted(glob.glob("{out}/*/")):
    for j in sorted(glob.glob(d + "*.json")):
        x = json.load(open(j)); print(d.split("/")[-2], x["config"]["prompt_id"], "seconds", round(x["seconds"], 1))
labels = {sorted(candidates)!r}
ok = len(rows) == 2 and min(r[1] for r in rows) >= 24.0 and all(glob.glob("{out}/" + l + "/*.png") for l in labels)
print("SMOKE", "PASS" if ok else "FAIL")
open("{out}/VERDICT", "w").write("PASS" if ok else "FAIL")
PYEOF
date -u +%FT%TZ > {out}/DONE
if [ "$(cat {out}/VERDICT)" = PASS ]; then n=$(ls {HOLD}/a6_d2*.sh 2>/dev/null | wc -l); mv {HOLD}/a6_d2*.sh {S}/taskq/pending/ 2>/dev/null; echo "released $n generation tasks"; else echo "smoke failed: generation tasks stay in {HOLD}"; exit 4; fi
""")
    dt.write(Path(QUEUE) / "a6_a000_smoke.sh", "".join(lines), dry)


def trace(code, dry, sids, per_task):
    sel = f"{R6}/trace_selection.json"
    for method in ("sca_v2", "area"):
        for k in range(0, len(sids), per_task):
            part = sids[k:k + per_task]
            name = f"a6_t{k // per_task:02d}_trace_{method}.sh"
            text = (header(name, f"x0 traces of {method} at the copies' crops", 5400) + f"set -uo pipefail\ncd {code}\n"
                    + f"{PY} scripts/sample_trace.py --method {method} --selection {sel} --out-root {R6}/trace "
                    + "--sid " + " ".join(part) + "\n")
            dt.write(Path(QUEUE) / name, text, dry)


def abl46(code, dry, label, flags, chunk, hold=False):
    out = f"{R6}/out/{label}/4096x4096"
    E = f"{S}/eval/r6_46_{label}"
    ename = f"a6_z97_eval_{label}.sh"
    starts = list(range(0, 46, chunk))
    names = [f"a6_d2{i:02d}_{label}.sh" for i in range(len(starts))]
    pat = "|".join(n[:-3] for n in names)
    for a, name in zip(starts, names):
        dt.write(Path(HOLD if hold else QUEUE) / name, header(name, f"{label} on the 46-prompt subset: {' '.join(flags)}", 3600)
                 + f"cd {code}\n" + gen(label, flags, ABL46, out, a, min(a + chunk, 46)) + " || exit $?\n"
                 + dt.self_release(f"{E}/gen_done", name, pat, len(names), ename), dry)
    dirs = f"sca_v2={SCA_V2_DIR} {label}={out} area={AREA_DIR}"
    det = f"{S}/eval/dup_detect_20261027 {S}/eval/a4k195_main_sca_v2/det $E/det"
    sub = "results/dup_audit_20261027/subjects_a4k195.json"
    ev = header(ename, f"{label} vs sca_v2 and AReA on the 46 prompts: detector, metrics, paired bootstrap", 5400) + f"""set -uo pipefail
Q={S}/taskq; E={E}; mkdir -p $E/paired
""" + dt.guard(ename, pat, (f"{E}/gen_done", len(names))) + f"""ls $Q/failed 2>/dev/null | grep -E '^({pat})[.]sh' | sed 's/^/WARNING failed generation task: /'
cd {code}
{PY} scripts/dup_detect.py --subjects {sub} --device cuda --threads 8 --sides 800,1333 --out $E/det --dirs {label}={out}
{PY} scripts/dup_rate.py --dets {det} --subjects {sub} --ids {ABL46} --thr 0.35 --kinds person,animal --methods sca_v2,{label},area --ref sca_v2 --per-image $E/dup_per_image_people.csv --out $E/dup_rates_people.csv | tee $E/dup_rates_people.txt
{PY} scripts/dup_rate.py --dets {det} --subjects {sub} --ids {ABL46} --thr 0.35 --methods sca_v2,{label},area --ref sca_v2 --per-image $E/dup_per_image_any.csv --out $E/dup_rates_any.csv | tee $E/dup_rates_any.txt
{PY} -I {P}/scripts/link_common.py --root {S}/out/evalroot/r6_46_{label} --res 4096x4096 --seed 0 --prompts {ABL46} --min-count 46 {dirs}
{PYM} {P}/scripts/eval_extra.py --root {S}/out/evalroot/r6_46_{label} --prompts {ABL46} --out $E/extra --metrics ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa --reference-method sca_v2 --device cuda --batch-size 16 --threads 8
{PY} {CR}/scripts/eval.py --root {S}/out/evalroot/r6_46_{label} --prompts {ABL46} --references {REFS} --reference-method sca_v2 --res 4096x4096 --out $E/base --metrics clip,pickscore,musiq,patch_fid,patch_kid --device cuda
for pair in "{label} sca_v2" "{label} area" "sca_v2 area"; do set -- $pair
  {PY} -I {P}/scripts/paired_bootstrap.py --method $1 --reference $2 --draws 2000 --out $E/paired/$1_vs_$2.csv $E/extra/extra_per_image.csv $E/base/per_image.csv
done
{PY} - <<'PYEOF' | tee $E/timing.txt
import glob, json, statistics
ids = {{json.loads(l)["id"] for l in open("{ABL46}")}}
for name, d in (("{label}", "{out}"), ("sca_v2", "{SCA_V2_DIR}"), ("area", "{AREA_DIR}")):
    t = [json.load(open(f)) for f in glob.glob(d + "/*.json")]
    t = [x["seconds"] for x in t if x.get("config", {{}}).get("prompt_id") in ids]
    print(name, "n", len(t), "median s", round(statistics.median(t), 2) if t else None)
PYEOF
date -u +%FT%TZ > $E/DONE
"""
    dt.write(Path(dt.EVAL_HOLD) / ename, ev, dry)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--code", required=True)
    p.add_argument("--stage", choices=("smoke", "trace", "abl46"), required=True)
    p.add_argument("--label")
    p.add_argument("--flags", default="")
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--candidates", default="{}", help="smoke: JSON {label: flags string}")
    p.add_argument("--sids", nargs="*", default=[])
    p.add_argument("--per-task", type=int, default=5)
    p.add_argument("--hold", action="store_true", help="abl46: park generation in taskq/hold_r6 (the smoke releases it)")
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    if not Path(a.code, "FREEZE_COMMIT").exists():
        raise SystemExit(f"{a.code} is not a frozen code tree (no FREEZE_COMMIT)")
    if a.stage == "smoke":
        smoke(a.code, a.dry, {k: shlex.split(v) for k, v in json.loads(a.candidates).items()})
    elif a.stage == "trace":
        trace(a.code, a.dry, a.sids, a.per_task)
    else:
        if not a.label or not a.label.startswith("r6_"):
            raise SystemExit("--label r6_<name> is required for abl46")
        abl46(a.code, a.dry, a.label, shlex.split(a.flags), a.chunk, a.hold)


if __name__ == "__main__":
    main()
