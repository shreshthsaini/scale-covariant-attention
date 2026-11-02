#!/usr/bin/env python3
"""Spool tasks for the GPU experiments of paper review round 1 (.agent/notes/review_round1/reviewer{1,2}_*.md).

Items: (1) the schedule alone and the bridge alone on the 195 Aesthetic-4K prompts at 4096^2; (2) a controlled pair on
the 46 ablation prompts: SCA with AReA's 20 Phase II windows and bridge size, and AReA with SCA's 4-step shrink;
(3) sca_v2 at 6144^2, at 8192^2 and on FLUX.2-dev; (4) one batch-1 latency sweep on one GPU; (5) DyPE and SEGA added
to the FLUX.1-Krea-dev 195-prompt block. Everything is written to --out (default taskq/hold_review_round), nothing to
pending. Each generation task leaves a marker in eval/review1_gen_done/<group>/; the last one of a group moves the
group's evaluation from --out to taskq/pending (GPU) or taskq_cpu/pending (CPU only). Evaluations park themselves
back in --out (exit 3) while their generation is unfinished and check that every image is complete before scoring.
A GPU evaluation that needs set-level intervals releases its CPU bootstrap task into taskq_cpu/pending when it ends.
README.md in --out is written from the same constants. Never overwrites an existing task.
"""

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dup_tasks import A4K, ENV, P, PY, PYM, REFS, S  # noqa: E402  (same paths and environment as the dup tasks)

Q, QC = f"{S}/taskq", f"{S}/taskq_cpu"
HOLD = f"{Q}/hold_review_round"
ABL50, DEV4, EVAL100, HI12 = f"{S}/abl50.jsonl", f"{S}/dev4.jsonl", f"{S}/eval100.jsonl", f"{S}/hi12.jsonl"
FLUX2P, LAT4 = f"{S}/prompts/flux2_eval50.jsonl", f"{S}/latency4_warmup.jsonl"
K1, KREA_T, FLUX2_T = f"{S}/calib/k1_1024.npz", f"{S}/calib/krea_1024.npz", f"{S}/calib/flux2_1024.npz"
SUBJ = "results/dup_audit_20261027/subjects_a4k195.json"     # relative to the frozen code
CROPE_FREEZE = f"{S}/code/crope_c05ebdc"                      # DyPE and SEGA runner of the main campaign
BASELINE_FREEZE = f"{S}/code/baseline_expansion_20261017_v1"   # HRDiT and SigMa runner of the main campaign
OUT = f"{S}/out/review1"
MARK = f"{S}/eval/review1_gen_done"
OVERHEAD = 45   # seconds per generation task beyond its images (model load, first compile): x_d90_m4k_sca_v2_0000_0005

V2_SCHED = "0.88:128,0.86:112,0.84:96,0.82:80"
V2 = ["--bridge-cells", "8", "--bridge-k", "16", "--win-sched", V2_SCHED, "--win-pi-match", "--win-pi-interp"]
SCHED_ONLY = ["--win-sched", V2_SCHED, "--win-pi-match", "--win-pi-interp"]
BRIDGE_ONLY = ["--bridge-cells", "8", "--bridge-k", "16"]
AREA_WINDOWS = (180, 170, 161, 153, 145, 137, 130, 123, 116, 110, 104, 99, 94, 89, 84, 80, 75, 71, 68, 64)  # 4096^2
FLUX2_SIGMAS = {7: 0.9627, 8: 0.9555, 9: 0.9478, 10: 0.9393, 11: 0.9300, 12: 0.9198}  # sidecars of out/flux2/flux2_sca


def flux1_sigmas(steps=28, mu=1.15):
    """FLUX.1 sigmas at mu = 1.15 (every size >= 1024^2 with mu_mode max_shift; AReA forces the same):
    exp(mu) / (exp(mu) + 1/t - 1), t = linspace(1, 1/steps, steps). Step 8 = 0.88758, step 27 = 0.10472."""
    e = math.exp(mu)
    return [e / (e + 1 / t - 1) for t in (1 - i * (1 - 1 / steps) / (steps - 1) for i in range(steps))]


def sched_after_gate(sigmas, first, windows):
    """--win-sched text that gives windows[k] on step first + k (threshold halfway to the next step's sigma). The last
    window is the base window n0 and is not listed."""
    return ",".join(f"{round((sigmas[first + k] + sigmas[first + k + 1]) / 2, 5)}:{w}" for k, w in enumerate(windows[:-1]))


AREA_SCHED = sched_after_gate(flux1_sigmas(), 8, AREA_WINDOWS)
SCA_AREA20 = ["--win-sched", AREA_SCHED, "--win-pi-match", "--win-pi-interp", "--bridge-cells", "8", "--bridge-ratio", "0.5"]
FLUX2_V2_SCHED = sched_after_gate(FLUX2_SIGMAS, 8, (128, 112, 96, 80, 64))
FLUX2_V2 = ["--bridge-cells", "8", "--bridge-k", "16", "--win-sched", FLUX2_V2_SCHED, "--win-pi-match", "--win-pi-interp"]
AREA_SCA4 = ["--phase2-windows", "128,112,96,80", "--label", "area_sca4"]


def sca_cmd(label, flags, prompts, out, res, a, b, model="dev"):
    table = f"--model krea --pi-table {KREA_T}" if model == "krea" else f"--pi-table {K1}"
    return (f"{PY} scripts/sample.py --method sca --seed 0 --guidance 4.5 --res {res} --gate-sigma 0.89 {table} "
            f"--label {label} --out {out} --prompts {prompts} --start {a} --end {b} " + " ".join(flags))


def area_cmd(out, prompts, res, a, b, flags=()):
    return (f"{PY} scripts/run_area.py --res {res} --prompts {prompts} --seed 0 --guidance 4.5 --out {out} "
            f"--start {a} --end {b} " + " ".join(flags)).rstrip()


def flux2_cmd(label, flags, out, a, b):
    return (f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {PY} scripts/sample_flux2.py --res 4096x4096 --prompts {FLUX2P} --seed 0 --steps 28 --guidance 4.0 "
            f"--mu-mode native --method sca --pi-table {FLUX2_T} --gate-sigma 0.959 --label {label} --out {out} "
            f"--start {a} --end {b} " + " ".join(flags))


class Writer:
    def __init__(self, out, dry):
        self.out, self.dry, self.names = Path(out), dry, []

    def __call__(self, name, text):
        path = self.out / name
        self.names.append(name)
        for d in (Path(Q) / x for x in ("pending", "running", "done", "failed")):
            if any(d.glob(name + "*")):
                print("exists in", d.name, "kept:", name)
                return
        if path.exists():
            print("exists, kept:", path)
            return
        print("write", path)
        if not self.dry:
            path.write_text(text)
            path.chmod(0o755)


def header(seconds, *lines):
    return "#!/bin/bash\n" + f"# SCA_MIN_SECONDS={int(seconds)}\n" + "".join(f"# {x}\n" for x in lines) + ENV


def gen_task(code, cmd, group, name, pat, n, ename, target, seconds, note):
    """Generation task: run cmd (resumable: the samplers skip finished images), leave a marker, and as the last of n
    move the evaluation ename from the hold directory to target/pending."""
    return (header(seconds, f"Review round 1: {note}", "Resumable: the sampler skips images whose sidecar is complete.")
            + f"set -uo pipefail\ncd {code}\n{cmd} || exit $?\n"
            + f"G={MARK}/{group}; mkdir -p $G && touch $G/{name} || echo \"WARNING could not write $G/{name}\"\n"
            + f"if [ \"$(ls $G 2>/dev/null | grep -c -E '^({pat})[.]sh$')\" -ge {n} ]; then\n"
            + f"  mkdir -p {target}/pending && mv {HOLD}/{ename} {target}/pending/ 2>/dev/null && echo \"released {ename} to {target}/pending\"\n"
            + "fi\nexit 0\n")


GATE_PY = r'''
import glob, json, os, sys
prompts, exclude, spec, res, link = sys.argv[1:6]
read = lambda f: [json.loads(l)["id"] for l in open(f) if l.strip()]
skip = set(read(exclude)) if exclude else set()
ids = [i for i in read(prompts) if i not in skip]
assert len(ids) == len(set(ids)), "duplicate prompt ids"
bad, plan = 0, []
for item in spec.split(";"):
    method, seed, dirs = item.split("|"); miss = 0
    for i in ids:
        pngs = [p for d in dirs.split(",") for p in glob.glob(f"{d}/{glob.escape(i)}_*seed{seed}.png")]
        ok = len(pngs) == 1
        if ok:
            try:
                c = json.load(open(pngs[0][:-4] + ".json"))
                cfg = c["config"]
                ok = (c.get("complete") is True and str(cfg["prompt_id"]) == i and cfg["method"] == method
                      and cfg["res"] == res and str(cfg["seed"]) == seed)
            except Exception:
                ok = False
        if ok:
            plan.append((method, pngs[0]))
        else:
            miss += 1
    print(f"GATE {method} seed{seed}: {len(ids) - miss}/{len(ids)} complete", flush=True); bad += miss
if bad:
    sys.exit(1)
if link:
    for method, png in plan:
        d = os.path.join(link, method, res); os.makedirs(d, exist_ok=True)
        for src in (png, png[:-4] + ".json"):
            os.symlink(src, os.path.join(d, os.path.basename(src)))
    print("linked", len(plan), "images under", link, flush=True)
'''


def eval_head(code, name, E, R, group, pat, n, seconds, lines, prompts, exclude, spec, res, cpu=False):
    """Common part of an evaluation task: park while generation runs, completeness gate, fresh output directory."""
    G = f"{MARK}/{group}"
    target = "CUDA_VISIBLE_DEVICES= " if cpu else ""
    rel = f"# RELEASE_AFTER: ^({pat})[.]sh\n" if n else ""
    return (header(seconds, *lines) + rel + (f"export {target}WANDB_MODE=disabled\n" if cpu else "")
            + f"""set -uo pipefail
Q={Q}; QC={QC}; H={HOLD}; NAME={name}; C={code}; E={E}; R={R}
PR={prompts}; EXCL={exclude}; RES={res}
SPEC='{spec}'
park(){{ mkdir -p $H; [ -e "$H/$NAME" ] || {{ cp "$0" "$H/.$NAME.tmp.$$" && mv -f "$H/.$NAME.tmp.$$" "$H/$NAME"; }}; echo "parked $NAME in $H: $1"; }}
requeue(){{ cp "$0" "$Q/pending/.$NAME.tmp.$$" && mv -f "$Q/pending/.$NAME.tmp.$$" "$Q/pending/$NAME" && echo "requeued $NAME: $1"; }}
gate(){{ {PY} -I - "$PR" "$EXCL" "$SPEC" "$RES" "${{1:-}}" <<'GATEPY'{GATE_PY}GATEPY
}}
GEN_DONE=1
""" + (f"""[ "$(ls {G} 2>/dev/null | grep -c -E '^({pat})[.]sh$')" -ge {n} ] || GEN_DONE=0
if [ $GEN_DONE -eq 0 ] && ls $Q/pending $Q/running 2>/dev/null | grep -q -E '^({pat})[.]sh'; then park "generation still pending or running"; exit 3; fi
ls $Q/failed 2>/dev/null | grep -E '^({pat})[.]sh' | sed 's/^/WARNING generation task in failed: /'
""" if n else "") + f"""if [ -e $E/DONE ]; then echo "already complete: $E"; exit 0; fi
if ! gate; then
  [ $GEN_DONE -eq 1 ] && {{ echo "GATE failed although every generation task finished: restore or regenerate the missing images"; exit 6; }}
  park "images incomplete"; exit 3
fi
if [ -e $E ]; then mv $E $E.partial_$(date +%Y%m%dT%H%M%S); fi
mkdir -p $E/paired $R
gate $E/root || {{ echo "GATE: linking failed"; exit 6; }}
sha256sum $C/scripts/eval_extra.py $C/repro/evaluator_snapshot/scripts/eval.py $C/repro/evaluate_corrected.py $C/scripts/paired_bootstrap.py $C/scripts/dup_detect.py $C/scripts/dup_rate.py $PR > $E/code_sha256.txt
run(){{ "$@" 2>&1 | tee -a $E/run.log; local rc=${{PIPESTATUS[0]}}
  if [ $rc -ne 0 ]; then
    if grep -q -E 'OutOfMemoryError|CUDA out of memory|CUDA error: out of memory|cudaErrorMemoryAllocation' $E/run.log; then requeue "CUDA OOM from co-tenant GPU processes"; exit 75; fi
    echo "step failed rc=$rc: $*"; exit $rc
  fi; }}
cd $C
""")


def extra_cmd(prompts, ref, metrics, device="cuda", batch=16, threads=8):
    return (f"run {PYM} $C/scripts/eval_extra.py --crope-root $C/repro/evaluator_snapshot --root $E/root --prompts {prompts} "
            f"--out $E/extra --metrics {metrics} --reference-method {ref} --device {device} --batch-size {batch} --threads {threads}")


def base_cmd(prompts, ref, metrics, refs=True):
    r = f"--references {REFS} " if refs else ""
    return (f"run {PY} $C/repro/evaluator_snapshot/scripts/eval.py --root $E/root --prompts {prompts} {r}--reference-method {ref} "
            f"--res $RES --out $E/base --metrics {metrics} --device cuda")


def corrected_cmd(prompts, ref):
    return ("mkdir -p $E/corrected_fidp/features && cp -p $E/base/features/*.npz $E/corrected_fidp/features/ && cp -p "
            "$E/base/evaluation_cache.json $E/corrected_fidp/ || { echo \"feature copy failed\"; exit 7; }\n"
            f"WANDB_MODE=disabled run {PY} $C/repro/evaluate_corrected.py --root $E/root --prompts {prompts} --references {REFS} "
            f"--reference-method {ref} --reference-clusters images --res $RES --out $E/corrected_fidp --metrics "
            "fid,kid,patch_fid,patch_kid --device cuda --bootstrap 0 --threads 8 --no-wandb")


def paired(pairs):
    return (f"for pair in {' '.join(pairs)}; do a=${{pair%%:*}}; b=${{pair#*:}}\n"
            "  run $PY -I $C/scripts/paired_bootstrap.py --method $a --reference $b --draws 2000 --out "
            "$E/paired/${a}_vs_${b}.csv $E/extra/extra_per_image.csv $E/base/per_image.csv\ndone\n").replace("$PY", PY)


def detector(dirs, rates, device="cuda"):
    """dup_detect on dirs (label=dir), then dup_rate per (tag, dets, methods, ref, ids args) for people/animal and any."""
    out = (f"run {PY} $C/scripts/dup_detect.py --subjects $C/{SUBJ} --device {device} --threads 8 --sides 800,1333 "
           f"--out $E/det --dirs {' '.join(dirs)}\n") if dirs else ""
    for tag, dets, methods, ref, ids in rates:
        for kind, kargs in (("people", "--kinds person,animal "), ("any", "")):
            out += (f"run {PY} $C/scripts/dup_rate.py --dets {dets} --subjects $C/{SUBJ} --thr 0.35 {kargs}{ids}"
                    f"--methods {methods} --ref {ref} --per-image $E/dup_per_image_{kind}{tag}.csv "
                    f"--out $E/dup_rates_{kind}{tag}.csv\n")
    return out


def finish(R_files, cpu_task=None):
    rel = (f"mkdir -p $QC/pending && mv $H/{cpu_task} $QC/pending/ 2>/dev/null && echo \"released {cpu_task} to $QC/pending\"\n"
           if cpu_task else "")
    return (f"{R_files}\n(cd $R && sha256sum $(find . -name '*.csv' | sort) > SHA256SUMS)\n"
            f"date -u +%FT%TZ > $E/DONE\n{rel}echo \"DONE $NAME: $E -> $R\"\n")


COPY = ("cp $E/base/summary.csv $R/base_summary.csv && cp $E/base/paired_deltas.csv $R/base_paired_deltas.csv && "
        "cp $E/extra/extra_summary.csv $R/extra_summary.csv && cp $E/extra/extra_paired_deltas.csv $R/extra_paired_deltas.csv "
        "&& cp $E/paired/*.csv $R/ || { echo \"copy to $R failed\"; exit 8; }\ncp $E/dup_rates_*.csv $R/ 2>/dev/null\n")
COPY_FIDP = ("mkdir -p $R/corrected_fidp && cp $E/corrected_fidp/summary.csv $E/corrected_fidp/paired_deltas.csv "
             "$R/corrected_fidp/ || { echo \"copy to $R failed\"; exit 8; }")


def boot_task(code, name, E, R, ref, res, prompts, seconds, note, out_dir="corrected_fidp_ci", src="corrected_fidp",
              metrics="fid,kid,patch_fid,patch_kid", extra=""):
    """CPU task: set-level 95% intervals (1000 draws over prompt ids and reference identities) on the cached features,
    released into taskq_cpu/pending by its GPU evaluation."""
    return (header(seconds, f"Review round 1, CPU only: {note}", "Released into taskq_cpu/pending by the GPU evaluation "
                   "when it writes $E/DONE; parks itself in hold_review_round if run earlier.")
            + "export CUDA_VISIBLE_DEVICES= WANDB_MODE=disabled OMP_NUM_THREADS=16\nset -euo pipefail\n"
            + f"H={HOLD}; NAME={name}; C={code}; E={E}; R={R}\n"
            + "if [ ! -e $E/DONE ]; then mkdir -p $H; [ -e $H/$NAME ] || cp \"$0\" $H/$NAME; echo \"parked $NAME: $E/DONE missing\"; exit 3; fi\n"
            + f"if [ -e $E/CPU_BOOTSTRAP_COMPLETE ]; then echo \"already complete\"; exit 0; fi\n"
            + (f"mkdir -p $E/{out_dir}/features && cp -p $E/{src}/features/*.npz $E/{out_dir}/features/ && "
               f"cp -p $E/{src}/evaluation_cache.json $E/{out_dir}/\n" if out_dir != src else "")
            + f"cd $C\n{PY} $C/repro/evaluate_corrected.py --reference-clusters images {extra}--root $E/root --prompts {prompts} "
            f"--references {REFS} --reference-method {ref} --res {res} --out $E/{out_dir} --metrics {metrics} --device cpu "
            "--bootstrap 1000 --threads 16 --no-wandb\n"
            + f"mkdir -p $R/{out_dir} && cp $E/{out_dir}/summary.csv $E/{out_dir}/paired_deltas.csv $R/{out_dir}/\n"
            + "(cd $R && sha256sum $(find . -name '*.csv' | sort) > SHA256SUMS)\ntouch $E/CPU_BOOTSTRAP_COMPLETE\n")


def chunks(a, b, size):
    return [(x, min(b, x + size)) for x in range(a, b, size)]


def safe(seconds):
    """SCA_MIN_SECONDS: 1.5 x the estimate plus 5 minutes, so the worker admits a task only with room to finish."""
    return 1.5 * seconds + 300


def generation(w, code, group, ename, target, jobs, note):
    """jobs: (name, command, estimated seconds). Returns the RELEASE_AFTER pattern and the task count."""
    pat = "|".join(n[:-3] for n, _, _ in jobs)
    for name, cmd, sec in jobs:
        w(name, gen_task(code, cmd, group, name, pat, len(jobs), ename, target, safe(sec), note))
    return pat, len(jobs)


DETS_DEV = "$S/eval/dup_detect_20261027 $S/eval/a4k195_main_sca_v2/det"   # archived sca_nb, area and sca_v2 detections
NEED_DETS = ("for d in " + DETS_DEV + "; do ls $d/dets_*.jsonl >/dev/null 2>&1 || { echo \"missing archived detections "
             "$d (restore eval outputs, MIGRATION.md Section 3)\"; exit 6; }; done\n").replace("$S", S)


def item1(w, code):
    group, name, cname = "a4k195_sched_bridge", "r1_z_eval_a4k195_sched_bridge.sh", "c1_boot_a4k195_sched_bridge.sh"
    E, R = f"{S}/eval/review1_{group}", f"{P}/results/review1_{group}"
    jobs = []
    for label, flags, sec in (("sca_sched", SCHED_ONLY, 74.7), ("sca_bridge", BRIDGE_ONLY, 73.6)):
        for a, b in chunks(0, 195, 5):
            jobs.append((f"r1_g_{label}_{a:04d}_{b:04d}.sh", sca_cmd(label, flags, A4K, f"{OUT}/{label}/4096x4096",
                         "4096x4096", a, b), (b - a) * sec + OVERHEAD))
    pat, n = generation(w, code, group, name, Q, jobs, "item 1, sca_v2 parts alone on the 195 prompts at 4096^2")
    spec = ";".join([f"sca_sched|0|{OUT}/sca_sched/4096x4096", f"sca_bridge|0|{OUT}/sca_bridge/4096x4096",
                     f"sca_v2|0|{S}/out/dupfix/sca_v2/4096x4096", f"sca_nb|0|{S}/out/main/sca_nb/4096x4096"])
    body = (NEED_DETS + extra_cmd(A4K, "sca_v2", "ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa,is") + "\n"
            + base_cmd(A4K, "sca_v2", "clip,pickscore,musiq,fid,kid,patch_fid,patch_kid") + "\n" + corrected_cmd(A4K, "sca_v2") + "\n"
            + paired(["sca_sched:sca_v2", "sca_bridge:sca_v2", "sca_sched:sca_nb", "sca_bridge:sca_nb", "sca_sched:sca_bridge",
                      "sca_v2:sca_nb"])
            + detector([f"sca_sched={OUT}/sca_sched/4096x4096", f"sca_bridge={OUT}/sca_bridge/4096x4096"],
                       [("", f"{DETS_DEV} $E/det".replace("$S", S), "sca_nb,sca_v2,sca_sched,sca_bridge", "sca_v2", ""),
                        ("_vs_sca_nb", f"{DETS_DEV} $E/det".replace("$S", S), "sca_nb,sca_v2,sca_sched,sca_bridge", "sca_nb", "")])
            + finish(COPY + COPY_FIDP, cname))
    w(name, eval_head(code, name, E, R, group, pat, n, 4 * 3600, [
        "Review round 1, item 1: schedule alone (sca_sched) and bridge alone (sca_bridge) vs sca_v2 and published SCA",
        "(sca_nb), 195 prompts, 4096^2. Clone of the a4k195 protocol (taskq/done/a000_z960_eval_a4k195_main.sh, z980):",
        "extra and base metric sets, corrected patch FID, paired bootstrap (2000 draws), detector with McNemar via",
        "dup_rate.py --ref (sca_v2 and sca_nb). Released by the last generation task; parks itself while generation runs."],
        A4K, "", spec, "4096x4096") + body)
    w(cname, boot_task(code, cname, E, R, "sca_v2", "4096x4096", A4K, safe(3600),
                       "item 1 set-level FID/KID/patch FID/patch KID intervals (clone of eval/a4k195_main_sca_v2/cpu_bootstrap_sca_v2.sh)"))
    return n + 2


def item2(w, code):
    group, name = "abl46_pair", "r2_z_eval_abl46_pair.sh"
    E, R = f"{S}/eval/review1_{group}", f"{P}/results/review1_{group}"
    bounds = [4, 9, 14, 19, 24, 29, 34, 39, 44, 50]   # abl50 rows 4..49 = the 46 ablation prompts (dev4 are rows 0..3)
    jobs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        jobs.append((f"r2_g_sca_area20_{a:04d}_{b:04d}.sh", sca_cmd("sca_area20", SCA_AREA20, ABL50,
                     f"{OUT}/sca_area20/4096x4096", "4096x4096", a, b), (b - a) * 108.3 + OVERHEAD + 300))
        jobs.append((f"r2_g_area_sca4_{a:04d}_{b:04d}.sh", area_cmd(f"{OUT}/area_sca4/4096x4096", ABL50, "4096x4096", a, b,
                     AREA_SCA4), (b - a) * 224.9 + OVERHEAD))
    pat, n = generation(w, code, group, name, Q, jobs, "item 2, controlled pair on the 46 ablation prompts at 4096^2")
    spec = ";".join([f"sca_area20|0|{OUT}/sca_area20/4096x4096", f"area_sca4|0|{OUT}/area_sca4/4096x4096",
                     f"sca_v2|0|{S}/out/dupfix/sca_v2/4096x4096", f"area|0|{S}/out/main/area/4096x4096"])
    ids = f"--ids {ABL50} --exclude-ids {DEV4} "
    body = (NEED_DETS + extra_cmd(ABL50, "sca_v2", "ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa") + "\n"
            + base_cmd(ABL50, "sca_v2", "clip,pickscore,musiq,patch_fid,patch_kid") + "\n"
            + paired(["sca_area20:sca_v2", "sca_area20:area", "area_sca4:area", "area_sca4:sca_v2", "sca_v2:area",
                      "sca_area20:area_sca4"])
            + detector([f"sca_area20={OUT}/sca_area20/4096x4096", f"area_sca4={OUT}/area_sca4/4096x4096"],
                       [("", f"{DETS_DEV} $E/det".replace("$S", S), "sca_v2,area,sca_area20,area_sca4", "sca_v2", ids),
                        ("_vs_area", f"{DETS_DEV} $E/det".replace("$S", S), "sca_v2,area,sca_area20,area_sca4", "area", ids)])
            + finish(COPY))
    w(name, eval_head(code, name, E, R, group, pat, n, 2 * 3600, [
        "Review round 1, item 2: (a) sca_area20 = SCA with AReA's 20 Phase II windows (180..64, one per step 8..27),",
        "weights at each window's emulated ratio (log-linear between the scale-2 and scale-4 tables; ratios below 2,",
        "windows 180..130, read the scale-2 table) and AReA's bridge size round(0.5 w)^2 (--bridge-ratio 0.5);",
        "(b) area_sca4 = AReA with windows 128,112,96,80 on its first four Phase II steps, 64 after (bridge, Halo and",
        "jitter unchanged). Both vs sca_v2 and AReA on the 46 prompts (abl50 minus dev4), abl46 (E7) metric protocol,",
        "paired bootstrap, detector with McNemar vs sca_v2 and vs AReA."], ABL50, DEV4, spec, "4096x4096") + body)
    return n + 1


def item3(w, code):
    count = 0
    # (a) 6144^2, eval100: protocol of taskq/done/i998_eval_e2_6144.sh plus a_cpu_e2_6144_bootstrap.sh
    group, name, cname = "6144", "r3_z_eval_6144.sh", "c3_boot_6144.sh"
    E, R = f"{S}/eval/review1_{group}", f"{P}/results/review1_{group}"
    jobs = [(f"r3_g_6144_sca_v2_{a:04d}_{b:04d}.sh", sca_cmd("sca_v2", V2, EVAL100, f"{OUT}/sca_v2/6144x6144", "6144x6144", a, b),
             (b - a) * 175.0 + OVERHEAD) for a, b in chunks(0, 100, 4)]
    pat, n = generation(w, code, group, name, Q, jobs, "item 3, sca_v2 at 6144^2 on eval100")
    spec = ";".join([f"sca_v2|0|{OUT}/sca_v2/6144x6144", f"sca_nb|0|{S}/out/main/sca_nb/6144x6144",
                     f"area|0|{S}/out/main/area/6144x6144"])
    body = (extra_cmd(EVAL100, "sca_v2", "ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa,is") + "\n"
            + f"run {PY} $C/repro/evaluate_corrected.py --reference-clusters images --musiq-query-chunk 256 --root $E/root "
            f"--prompts {EVAL100} --references {REFS} --reference-method sca_v2 --res $RES --out $E/base --metrics "
            "clip,pickscore,musiq,fid,kid,patch_fid,patch_kid --device cuda --bootstrap 0 --no-wandb\n"
            + paired(["sca_v2:sca_nb", "sca_v2:area", "sca_nb:area"])
            + detector([f"sca_v2={OUT}/sca_v2/6144x6144", f"sca_nb={S}/out/main/sca_nb/6144x6144", f"area={S}/out/main/area/6144x6144"],
                       [("", "$E/det", "sca_v2,sca_nb,area", "sca_v2", f"--ids {EVAL100} ")])
            + finish(COPY, cname))
    w(name, eval_head(code, name, E, R, group, pat, n, 3 * 3600, [
        "Review round 1, item 3a: sca_v2 vs published SCA (sca_nb) and AReA at 6144^2, 100 prompts (eval100). Protocol of",
        "taskq/done/i998_eval_e2_6144.sh (extra with IS, base via evaluate_corrected.py with the 4096 px references,",
        "reference clusters images, MUSIQ query chunk 256), reference method now sca_v2, plus paired bootstrap and the",
        "detector. The 1000-draw set-level intervals run on CPU (c3_boot_6144.sh, released at DONE)."],
        EVAL100, "", spec, "6144x6144") + body)
    w(cname, boot_task(code, cname, E, R, "sca_v2", "6144x6144", EVAL100, safe(2 * 3600),
                       "item 3a 1000-draw intervals of the base metrics (clone of taskq/done/a_cpu_e2_6144_bootstrap.sh)",
                       out_dir="base", src="base", metrics="clip,pickscore,musiq,fid,kid,patch_fid,patch_kid",
                       extra="--musiq-query-chunk 256 "))
    count += n + 2
    # (b) 8192^2, hi12: protocol of the i999 CPU evaluation and the MUSIQ shards (query chunk 256, CPU, fp32)
    group, name = "8192", "c3_eval_8192_cpu.sh"
    E, R = f"{S}/eval/review1_{group}", f"{P}/results/review1_{group}"
    jobs = [(f"r3_g_8192_sca_v2_{a:04d}_{b:04d}.sh", sca_cmd("sca_v2", V2, HI12, f"{OUT}/sca_v2/8192x8192", "8192x8192", a, b),
             (b - a) * 325.0 + OVERHEAD) for a, b in chunks(0, 12, 2)]
    pat, n = generation(w, code, group, name, QC, jobs, "item 3b, sca_v2 at 8192^2 on hi12")
    spec = ";".join([f"sca_v2|0|{OUT}/sca_v2/8192x8192", f"sca_nb|0|{S}/out/main/sca_nb/8192x8192",
                     f"area|0|{S}/out/main/area/8192x8192"])
    body = (extra_cmd(HI12, "sca_v2", "ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa", "cpu", 2, 16) + "\n"
            + f"run {PY} $C/repro/evaluate_corrected.py --musiq-query-chunk 256 --root $E/root --prompts {HI12} "
            "--reference-method sca_v2 --res $RES --out $E/base --metrics clip,pickscore,musiq --device cpu --bootstrap 1000 "
            "--threads 16 --no-wandb\n"
            + paired(["sca_v2:sca_nb", "sca_v2:area", "sca_nb:area"])
            + detector([f"sca_v2={OUT}/sca_v2/8192x8192", f"sca_nb={S}/out/main/sca_nb/8192x8192", f"area={S}/out/main/area/8192x8192"],
                       [("", "$E/det", "sca_v2,sca_nb,area", "sca_v2", f"--ids {HI12} ")], device="cpu")
            + finish(COPY))
    w(name, eval_head(code, name, E, R, group, pat, n, 12 * 3600, [
        "Review round 1, item 3b, CPU only (fp32 scorers, as taskq/recovered/i999_eval_e2_8192.sh and",
        "eval/e2_8192_musiq_shards_20261017): sca_v2 vs published SCA (sca_nb) and AReA at 8192^2, 12 prompts (hi12).",
        "Extra without IS on CPU, clip/pickscore/musiq via evaluate_corrected.py --musiq-query-chunk 256 (MUSIQ at",
        "native 8192 px, about 700 s per image on 8 threads), no references, paired bootstrap, detector on CPU.",
        "Released into taskq_cpu/pending by the last 8192 generation task."], HI12, "", spec, "8192x8192", cpu=True)
      .replace("OMP_NUM_THREADS=8", "OMP_NUM_THREADS=16") + body)
    count += n + 1
    # (c) FLUX.2-dev, flux2_eval50: protocol of taskq/done/f900_flux2_eval.sh
    group, name = "flux2", "r3_z_eval_flux2.sh"
    E, R = f"{S}/eval/review1_{group}", f"{P}/results/review1_{group}"
    w("r3_e_flux2_encode.sh", header(safe(1200), "Review round 1, item 3c: fill the FLUX.2 prompt-embedding cache "
                                     "(sca/flux2.py EMBED_CACHE) for flux2_eval50 once, before the generation tasks.")
      + f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\nset -uo pipefail\ncd {code}\n"
      f"{PY} scripts/sample_flux2.py --encode-only --prompts {FLUX2P}\n")
    jobs = [(f"r3_g_flux2_sca_v2_{a:04d}_{b:04d}.sh", flux2_cmd("flux2_sca_v2", FLUX2_V2, f"{OUT}/flux2_sca_v2/4096x4096", a, b),
             (b - a) * 237.0 + OVERHEAD + 120) for a, b in chunks(0, 50, 5)]
    pat, n = generation(w, code, group, name, Q, jobs, "item 3c, sca_v2 on FLUX.2-dev (flux2_eval50, gate 0.959, "
                        "FLUX.2 table, windows on the four steps after the gate)")
    spec = ";".join([f"flux2_sca_v2|0|{OUT}/flux2_sca_v2/4096x4096", f"flux2_sca|0|{S}/out/flux2/flux2_sca/4096x4096",
                     f"flux2_full|0|{S}/out/flux2/flux2_full/4096x4096"])
    body = (extra_cmd(A4K, "flux2_full", "ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa,is,latency") + "\n"
            + base_cmd(A4K, "flux2_full", "clip,pickscore,musiq,fid,kid,patch_fid,patch_kid") + "\n"
            + paired(["flux2_sca_v2:flux2_full", "flux2_sca_v2:flux2_sca", "flux2_sca:flux2_full"])
            + detector([f"flux2_sca_v2={OUT}/flux2_sca_v2/4096x4096", f"flux2_sca={S}/out/flux2/flux2_sca/4096x4096",
                        f"flux2_full={S}/out/flux2/flux2_full/4096x4096"],
                       [("", "$E/det", "flux2_sca_v2,flux2_sca,flux2_full", "flux2_sca_v2", f"--ids {FLUX2P} ")])
            + finish(COPY))
    w(name, eval_head(code, name, E, R, group, pat, n, 2 * 3600, [
        "Review round 1, item 3c: sca_v2 on FLUX.2-dev vs the published operator (flux2_sca) and full attention",
        "(flux2_full), 50 prompts (flux2_eval50), 4096^2. Protocol of taskq/done/f900_flux2_eval.sh (reference method",
        "flux2_full, extra with IS and latency, base with the Aesthetic-4K references), plus paired bootstrap and the",
        "detector."], FLUX2P, "", spec, "4096x4096") + body)
    count += n + 2
    return count


def item4(w, code):
    hf = "/scratch/09032/saini_2/tmp/hf"
    dev = f"{hf}/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
    site = "/work/09032/saini_2/vista/library/uv-envs/sca-baseline-expansion-20261017/lib/python3.11/site-packages"
    bx = ("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
          "TOKENIZERS_PARALLELISM=false UV_CACHE_DIR=/work/09032/saini_2/vista/library/uv-cache "
          f"TORCHINDUCTOR_CACHE_DIR={S}/cache/baseline_expansion_20261017/inductor "
          f"TRITON_CACHE_DIR={S}/cache/baseline_expansion_20261017/triton CC=/usr/bin/gcc CXX=/usr/bin/g++")
    v2 = " ".join(V2)
    sca = lambda res, label, flags="": (f"{PY} scripts/sample.py --method sca --res {res} --prompts {LAT4} --seed 0 --guidance 4.5 "
                                        f"--out $RUN/{res}/{label} --pi-table {K1} --gate-sigma 0.89 --label {label} {flags}").rstrip()
    area = lambda res: f"{PY} scripts/run_area.py --res {res} --prompts {LAT4} --seed 0 --guidance 4.5 --out $RUN/{res}/area"
    steps = [("4096x4096 sca_nb", sca("4096x4096", "sca_nb")), ("4096x4096 sca_v2", sca("4096x4096", "sca_v2", v2)),
             ("4096x4096 area", area("4096x4096")),
             ("4096x4096 full", f"{PY} scripts/sample.py --method full --res 4096x4096 --prompts {LAT4} --seed 0 --guidance 4.5 "
                                "--out $RUN/4096x4096/full")]
    for m in ("dype", "sega_official"):
        steps.append((f"4096x4096 {m}", f"(cd {CROPE_FREEZE} && {PY} scripts/sample.py --method {m} --seed 0 --guidance 4.5 "
                                        f"--steps 28 --out $RUN/4096x4096/{m} --res 4096x4096 --prompts {LAT4})"))
    steps.append(("4096x4096 ultraimage", f"ULTRA_BLOCKAVG_ROOT=$RUN/ultraimage_blockavg {PY} scripts/run_ultraimage.py --seed 0 "
                                          f"--out $RUN/4096x4096/ultraimage --res 4096x4096 --prompts {LAT4}"))
    for m, extra in (("hrdit_dev_author", f" --extra-site {site}"), ("sigma_dev_matched", "")):
        steps.append((f"4096x4096 {m}", f"(export {bx}; ${{UV:-uv}} run --no-project --python {PY} python {BASELINE_FREEZE}/run.py "
                                        f"--profile {m} --prompts {LAT4} --start 0 --end 4 --seed 0 --out $RUN/4096x4096/{m}_root "
                                        f"--dev-snapshot {dev} --mode generate{extra})"))
    for res in ("2048x2048", "3072x3072", "6144x6144", "8192x8192"):
        steps += [(f"{res} area", area(res)), (f"{res} sca_nb", sca(res, "sca_nb")), (f"{res} sca_v2", sca(res, "sca_v2", v2))]
    assert all("'" not in c for _, c in steps)
    body = "".join(f"step \"{t}\" '{c}'\n" for t, c in steps)
    name = "r4_latency_sweep.sh"
    summary = r'''
import collections, csv, glob, json, os, re, statistics, sys
run, gpu = sys.argv[1], open(sys.argv[2]).read().strip().splitlines()[-1] if os.path.exists(sys.argv[2]) else ""
rows = collections.defaultdict(list); warm = {}
for f in sorted(glob.glob(f"{run}/*/*/**/*.json", recursive=True)):
    rel = os.path.relpath(f, run).split(os.sep)
    if not re.fullmatch(r"[0-9]+x[0-9]+", rel[0]) or "intermediates" in rel or "wandb" in rel:
        continue
    try:
        j = json.load(open(f))
    except Exception:
        continue
    if not isinstance(j, dict) or not j.get("complete") or "seconds" not in j or "png_sha256" not in j:
        continue
    res, method = rel[0], rel[1].removesuffix("_root")
    pid = str(j.get("config", {}).get("prompt_id", ""))
    if pid == "sca_latency_warmup":
        warm[(res, method)] = float(j["seconds"])
    else:
        rows[(res, method)].append(float(j["seconds"]))
w = csv.writer(sys.stdout); out = [["res", "method", "n_timed", "median_s", "min_s", "max_s", "warmup_s", "gpu"]]
for (res, method), t in sorted(rows.items(), key=lambda x: (int(x[0][0].split("x")[0]), x[0][1])):
    out.append([res, method, len(t), round(statistics.median(t), 2), round(min(t), 2), round(max(t), 2),
                round(warm.get((res, method), float("nan")), 2), gpu])
for r in out:
    w.writerow(r)
csv.writer(open(f"{run}/latency_summary.csv", "w")).writerows(out)
'''
    text = (header(36000, "Review round 1, item 4: one clean batch-1 latency sweep on ONE otherwise idle GPU, all methods in one",
                   "task so GPU and conditions are shared. Every method runs in its own process on latency4_warmup.jsonl:",
                   "row 0 is an excluded warm-up, rows 1..3 are timed (median reported). 4096^2: published SCA (sca_nb),",
                   "sca_v2, AReA, full attention, DyPE, SEGA, UltraImage (fresh block-average cache, so its extra pass is",
                   "timed), HRDiT, SigMa; then AReA, sca_nb and sca_v2 at 2048^2, 3072^2, 6144^2, 8192^2. Commands cloned",
                   "from taskq/done/a_y00*_latency_*.sh, a_latency*.sh, a_latency_hrdit_batch1.sh and the campaign tasks",
                   "(g010_dype, g015_sega_official, g015_ultraimage, a_b_pilot46_sigma_dev_matched). Never resumes timed",
                   "samples: each attempt writes a fresh run directory. Prints a CSV of median seconds at the end.")
            + f"set -uo pipefail\nC={code}\nRUN={S}/out/review1_latency/$(date +%Y%m%dT%H%M%S)_$(hostname -s)\n"
            + f"R={P}/results/review1_latency\nmkdir -p $RUN $R\ncd $C\n"
            + "GPU_ID=${CUDA_VISIBLE_DEVICES:-0}; GPU_ID=${GPU_ID%%,*}\n"
            + "nvidia-smi -i $GPU_ID --query-gpu=name,uuid,driver_version,utilization.gpu,memory.used,clocks.max.sm --format=csv | tee $RUN/gpu.csv\n"
            + "nvidia-smi -i $GPU_ID --query-compute-apps=pid,used_memory --format=csv | tee $RUN/gpu_apps_before.csv\n"
            + "[ \"$(wc -l < $RUN/gpu_apps_before.csv)\" -gt 1 ] && echo \"WARNING: other processes hold this GPU; timings are not clean\"\n"
            + "FAIL=0\nstep(){ local tag=$1; shift; echo \"$(date -u +%FT%TZ) start $tag\"; eval \"$@\"; local rc=$?\n"
            + "  [ $rc -eq 0 ] || { echo \"FAILED $tag rc=$rc\"; FAIL=$((FAIL+1)); }; echo \"$(date -u +%FT%TZ) end $tag\"; }\n"
            + body
            + "nvidia-smi -i $GPU_ID --query-compute-apps=pid,used_memory --format=csv > $RUN/gpu_apps_after.csv\n"
            + f"{PY} -I - $RUN $RUN/gpu.csv <<'SUMPY' | tee $RUN/latency_summary.txt{summary}SUMPY\n"
            + "cp $RUN/latency_summary.csv $R/latency_summary_$(basename $RUN).csv && cp $RUN/latency_summary.csv $R/latency_summary.csv\n"
            + "(cd $R && sha256sum *.csv > SHA256SUMS)\necho \"latency run: $RUN ($FAIL failed steps)\"\n[ $FAIL -eq 0 ]\n")
    w(name, text)
    return 1


def item5(w, code):
    group, name, cname = "krea195_dype_sega", "r0_k5_eval_krea195_dype_sega.sh", "c5_boot_krea195_dype_sega.sh"
    E, R = f"{S}/eval/review1_{group}", f"{P}/results/review1_{group}"
    CRK = "/scratch/09032/saini_2/constancy_rope/out_krea/main"
    spec = ";".join([f"sca_v2_krea|0|{S}/out/dupfix_krea/sca_v2_krea/4096x4096", f"sca_krea|0|{S}/out/krea/sca_krea/4096x4096",
                     f"area|0|{S}/out/krea/area/4096x4096", f"dype|0|{CRK}/dype/4096x4096",
                     f"sega_official|0|{CRK}/sega_official/4096x4096"])
    body = (extra_cmd(A4K, "sca_v2_krea", "ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa,is") + "\n"
            + base_cmd(A4K, "sca_v2_krea", "clip,pickscore,musiq,fid,kid,patch_fid,patch_kid") + "\n"
            + corrected_cmd(A4K, "sca_v2_krea") + "\n"
            + paired(["sca_v2_krea:dype", "sca_v2_krea:sega_official", "area:dype", "area:sega_official", "sca_krea:dype",
                      "sca_krea:sega_official", "sca_v2_krea:area", "sca_v2_krea:sca_krea"])
            + detector([f"sca_v2_krea={S}/out/dupfix_krea/sca_v2_krea/4096x4096", f"sca_krea={S}/out/krea/sca_krea/4096x4096",
                        f"area_krea={S}/out/krea/area/4096x4096", f"dype_krea={CRK}/dype/4096x4096",
                        f"sega_krea={CRK}/sega_official/4096x4096"],
                       [("", "$E/det", "sca_v2_krea,area_krea,sca_krea,dype_krea,sega_krea", "sca_v2_krea", "")])
            + finish(COPY + COPY_FIDP, cname))
    w(name, eval_head(code, name, E, R, group, "", 0, 4 * 3600, [
        "Review round 1, item 5: DyPE and SEGA join the FLUX.1-Krea-dev 195-prompt block. No generation: the 195 DyPE and",
        "195 SEGA images in constancy_rope/out_krea/main are complete (checked 2026-11-02: one PNG plus a complete",
        "sidecar per prompt id; the 56 constancy_rope generation tasks export CROPE_TRANSFORMER = FLUX.1-Krea-dev",
        "snapshot 8162a9c, seed 0, 28 steps, guidance 4.5; their sidecars name the FLUX.1-dev base pipeline only).",
        "Same protocol as taskq/done/a000_z966_eval_krea195.sh, reference method sca_v2_krea as in krea195_sca_v2,",
        "with sca_v2_krea, sca_krea and AReA rescored in the same root, plus the detector (McNemar vs sca_v2_krea).",
        "Move it to pending by hand; the gate checks every image first."], A4K, "", spec, "4096x4096") + body)
    w(cname, boot_task(code, cname, E, R, "sca_v2_krea", "4096x4096", A4K, safe(1.5 * 3600),
                       "item 5 set-level intervals (clone of eval/krea195_sca_v2/cpu_bootstrap_sca_v2.sh)"))
    return 2


def readme(code, commit, counts):
    h = lambda s: f"{s / 3600:.1f}"
    g1 = 195 * (74.7 + 73.6) + 78 * OVERHEAD
    g2a, g2b = 46 * 108.3 + 9 * (OVERHEAD + 300), 46 * 224.9 + 9 * OVERHEAD
    g36, g38, g3f = 100 * 175.0 + 25 * OVERHEAD, 12 * 325.0 + 6 * OVERHEAD, 50 * 237.0 + 10 * (OVERHEAD + 120) + 600
    lat4096 = 4 * (71.1 + 77.0 + 203.4 + 186.6 + 187.6 + 189.0 + 303.3 + 107.4 + 186.5)
    latsize = 4 * ((46.0 + 18.2 + 19.8) + (114.5 + 40.4 + 44.0) + (495.3 + 161.6 + 176.0) + (959.8 + 287.8 + 313.4))
    g4 = lat4096 + latsize + 21 * 60
    t1, t2, t3, t4, t5 = g1 + 3600, g2a + g2b + 1200, g36 + 3600 + g38 + g3f + 1500, g4, 4500
    return f"""# Review round 1: held GPU task scripts

Generated by `slurm/review_tasks.py` (repository commit `{commit}`). Every task runs the frozen code
`{code}` (its `FREEZE_COMMIT` holds the full hash). Nothing here is in `taskq/pending`: the Vista allocation is
exhausted and these tasks are meant for another machine. What the reviewers asked: `.agent/notes/review_round1/`.

| Item | What | Tasks (generation + GPU eval + CPU) | Outputs |
|---|---|---|---|
| 1 | schedule alone (`sca_sched`) and bridge alone (`sca_bridge`), 195 prompts, 4096^2 | {counts['1']} (78 + 1 + 1) | `out/review1/{{sca_sched,sca_bridge}}`, `eval/review1_a4k195_sched_bridge` |
| 2 | `sca_area20` (SCA, AReA's 20 windows and bridge size) and `area_sca4` (AReA, SCA's 4-step shrink), 46 prompts | {counts['2']} (18 + 1) | `out/review1/{{sca_area20,area_sca4}}`, `eval/review1_abl46_pair` |
| 3 | sca_v2 at 6144^2 (100), 8192^2 (12), FLUX.2-dev (50) | {counts['3']} (25 + 6 + 1 encode + 10, evals 6144 and FLUX.2 on GPU, 8192 eval and the 6144 bootstrap on CPU) | `out/review1/sca_v2/{{6144x6144,8192x8192}}`, `out/review1/flux2_sca_v2`, `eval/review1_{{6144,8192,flux2}}` |
| 4 | one batch-1 latency sweep, one GPU (`r4_latency_sweep.sh`) | {counts['4']} | `out/review1_latency/<time>_<host>`, `results/review1_latency` |
| 5 | DyPE and SEGA added to the Krea-195 block (evaluation only) | {counts['5']} (1 + 1) | `eval/review1_krea195_dype_sega` |

CSV summaries go to `results/review1_<name>/` in the repository (with `SHA256SUMS`), as for every earlier evaluation.

## Exact definitions

- sca_v2 (adopted operator): `{' '.join(V2)}`, gate 0.89, table `calib/k1_1024.npz`.
- Item 1: `sca_sched` = `{' '.join(SCHED_ONLY)}`; `sca_bridge` = `{' '.join(BRIDGE_ONLY)}`.
- Item 2a `sca_area20`: `--win-sched` with AReA's 4096^2 Phase II windows 180, 170, ..., 68 on steps 8 to 26 (one per
  step; thresholds halfway between consecutive step sigmas at mu 1.15) and the base window 64 on step 27, weights from
  `--win-pi-match --win-pi-interp` at ratio 256/w, and `--bridge-cells 8 --bridge-ratio 0.5` (new option): per cell
  round(round(0.5 w)^2 / 64) tokens, from 127 at w = 180 down to 16 at w = 64. Ratios below 2 (windows 180 to 130, the
  first seven steps) read the scale-2 table unchanged: `SCAContext.table_interp` clamps to the nearest calibrated scale
  outside [2, 4]; no extrapolation. AReA's own cost cap (`fit_extra_keys`) can cut its bridge on its widest steps; the
  SCA variant uses the nominal size. Full flag: `{AREA_SCHED}`.
- Item 2b `area_sca4`: `scripts/run_area.py --phase2-windows 128,112,96,80 --label area_sca4` (new options): AReA's
  Phase I unchanged, Phase II windows 128, 112, 96, 80 on steps 8 to 11 and 64 on the remaining sixteen; bridge
  (0.5 w)^2, Halo 8 and jitter as in AReA. Implemented by wrapping the backbone's `schedule()`; no vendored file edited.
- Item 3: 6144^2 and 8192^2 use the sca_v2 flags unchanged (the sigma schedule is the same, mu 1.15). FLUX.2-dev
  (`scripts/sample_flux2.py`, which gained the same options) keeps its gate 0.959 and table `calib/flux2_1024.npz`
  (it has the scale-2 entry, so no recalibration) and puts the four windows on its steps 8 to 11:
  `--win-sched {FLUX2_V2_SCHED}` (FLUX.2 sigmas 0.9555, 0.9478, 0.9393, 0.9300 at those steps, native mu).
- Item 5: the 195 DyPE and 195 SEGA images in `/scratch/09032/saini_2/constancy_rope/out_krea/main/{{dype,sega_official}}`
  were checked on 2026-11-02: one PNG and a complete sidecar for each of the 195 prompt ids; the 56 constancy_rope
  generation tasks that wrote them export `CROPE_TRANSFORMER` = the FLUX.1-Krea-dev snapshot 8162a9c and use seed 0,
  28 steps, guidance 4.5 (the sidecars name only the FLUX.1-dev base pipeline). So only the evaluation is written.

## Run order

5, 1, 2, 3, 4. Item 5 needs no generation and about 1.3 GPU-h, so it checks the metric, detector and reference setup on
the new machine before any long generation starts. Items 1 and 2 answer the blocking reviewer points. Item 3 is the
largest. Item 4 needs a GPU with nothing else running on it or its node, so it goes last, alone.

## GPU-hour estimates (GH200 96 GB; other GPUs scale with their FLUX speed)

Per-image seconds are medians of the sidecars of existing runs; each generation task adds {OVERHEAD} s (model load and
first compile: `x_d90_m4k_sca_v2_0000_0005` took 434 s for five images of about 78 s).

| Item | Numbers used | GPU-h |
|---|---|---|
| 1 | bridge only 73.6 s (`out/dup46/dup_bridge`, n = 50); schedule only 74.7 s (predicted: published SCA 70.7 s plus 1.42e-4 s per local key and step, fitted on sca_nb, dup_bridge and sca_v2 = 77.0 s); 390 images; eval about 1.0 h (a4k195_main_sca_v2 took 40 min for three sets) | {h(t1)} |
| 2 | sca_area20 108 s predicted by the same key model (checked on dup_bridge_win128: 82.3 s predicted, 83.0 s measured) plus 300 s per task for 20 new attention layouts; area_sca4 224.9 s (AReA median, an upper bound: AReA's time is dominated by per-window work, not FLOPs; its base-tiling attention cost falls from 6.85 to 2.23 full steps); eval 0.3 h | {h(t2)} |
| 3 | 6144^2: 175 s (sca_nb 160.7 s x 77.0/70.7) x 100; 8192^2: 325 s (298.4 s x 1.089) x 12; FLUX.2: 237 s (flux2_sca 218.1 s x 1.089) x 50 plus 120 s per task for the transformer load and 0.2 h of prompt encoding; GPU evals of 6144 (1 h, e2_6144 took 33 min for two sets) and FLUX.2 (0.4 h) | {h(t3)} |
| 4 | four images per method (one warm-up, three timed): 4096^2 sca_nb 71.1, sca_v2 77.0, AReA 203.4, full 186.6, DyPE 187.6, SEGA 189.0, UltraImage 303.3, HRDiT 107.4, SigMa 186.5 s; AReA / sca_nb / sca_v2 at 2048^2 46.0 / 18.2 / 19.8, 3072^2 114.5 / 40.4 / 44.0, 6144^2 495.3 / 161.6 / 176.0, 8192^2 959.8 / 287.8 / 313.4 s (`out/latency_curve`, campaign medians where no clean run exists); 21 model loads of about 60 s | {h(t4)} |
| 5 | five sets of 195 images rescored plus the detector (krea195_sca_v2 took 43 min for three sets) | {h(t5)} |
| Total | | {h(t1 + t2 + t3 + t4 + t5)} |

CPU only (no GPU): item 1 bootstrap about 0.5 h, item 3 6144 bootstrap about 1.5 h (a_cpu_e2_6144_bootstrap took 63
min for two sets), item 3 8192 evaluation about 6 h (MUSIQ at native 8192 px took 702 s per image on 8 threads; 36
images), item 5 bootstrap about 1 h.

## Before running: inputs this machine must have

Paths stay absolute Vista paths, as in every earlier task (MIGRATION.md Section 2: either recreate the same paths with
symlinks, or rewrite them with `repro/migration/relocate_paths.sh --apply`, passing this directory and `taskq_cpu`).

1. Frozen code: `{code}`. If it is not there, recreate it from the repository:
   `C={code}; mkdir -p $C && git archive {commit} sca scripts slurm tests results/dup_audit_20261027 HANDOVER.md repro/evaluator_snapshot repro/evaluate_corrected.py | tar -x -C $C && git rev-parse {commit} > $C/FREEZE_COMMIT`.
   `repro/evaluator_snapshot` and `repro/evaluate_corrected.py` are part of this freeze (earlier freezes lacked them):
   `run_area.py` and `run_ultraimage.py` import `crope.utils` from there, and every evaluation uses that snapshot
   (byte-identical to the constancy_rope evaluator used for the published rows).
2. Other runners: `code/crope_c05ebdc` (DyPE, SEGA) and `code/baseline_expansion_20261017_v1` (HRDiT, SigMa) from the
   Hugging Face `sca-frozen-code` set; AReA at `$SCA_SCRATCH/sca/third_party/AReA` (or `AREA_DIR`), UltraImage
   (`ULTRA_DIR`); envs `sca`, `sca-metrics`, `sca-baseline-expansion-20261017`; `uv` on PATH for HRDiT and SigMa.
3. Image sets on the Hugging Face mirror (phase 1): `out/main/sca_nb/4096x4096`, `out/main/area/4096x4096`,
   `out/dupfix/sca_v2/4096x4096`, `out/dupfix_krea/sca_v2_krea/4096x4096`; archived detections
   `eval/dup_detect_20261027` and `eval/a4k195_main_sca_v2/det` (items 1 and 2 exit 6 without them).
4. NOT on the mirror (phase 2 or never copied; copy them from Vista scratch before its purge, or upload phase 2 as in
   MIGRATION.md Section 6): item 3 needs `out/main/{{sca_nb,area}}/6144x6144` (5.0 + 4.7 GB), `out/main/{{sca_nb,area}}/8192x8192`
   (1.0 + 1.0 GB), `out/flux2/{{flux2_sca,flux2_full}}/4096x4096` (1.4 + 1.1 GB); item 5 needs `out/krea/sca_krea/4096x4096`
   (about 5 GB), `out/krea/area/4096x4096` (4.4 GB) and `/scratch/09032/saini_2/constancy_rope/out_krea/main/{{dype,sega_official}}/4096x4096`
   (4.5 + 4.6 GB). Without them the generation tasks still run, and the evaluation gate exits 6 naming the missing set.
5. The FLUX.2 prompt-embedding cache was not migrated; `r3_e_flux2_encode.sh` refills it.

## Moving tasks into pending

```bash
S=/scratch/09032/saini_2/sca; H=$S/taskq/hold_review_round; Q=$S/taskq
mv $H/r0_k5_eval_krea195_dype_sega.sh $Q/pending/   # item 5 (evaluation only)
mv $H/r1_g_*.sh $Q/pending/                          # item 1 generation
mv $H/r2_g_*.sh $Q/pending/                          # item 2 generation
mv $H/r3_e_*.sh $H/r3_g_*.sh $Q/pending/             # item 3 encode + generation
mv $H/r4_latency_sweep.sh $Q/pending/                # item 4, only with one worker on an otherwise idle node
```

Leave every `*_z_eval_*` and `c*_` task in this directory: they are released automatically. A worker
(`slurm/worker_loop_v2.py`, any fleet sbatch of this repository) takes `pending/*.sh` in name order and admits a task
only if its `# SCA_MIN_SECONDS` header fits the time left. CPU tasks are released into `$S/taskq_cpu/pending`; drain
that spool on a CPU node with the same worker, for example
`CUDA_VISIBLE_DEVICES= python slurm/worker_loop_v2.py worker --queue $S/taskq_cpu --logs $S/logs/cpu --deadline $(( $(date +%s) + 86400 ))`,
or run each script there with `bash`.

## How evaluations are released

Each generation task, after its sampler succeeds, touches `eval/review1_gen_done/<group>/<task name>`. The task that
completes the set (all markers present) moves the group's evaluation from this directory to `taskq/pending` (the 8192
evaluation to `taskq_cpu/pending`). An evaluation started early (moved by hand, or released twice) parks a copy of
itself back here and exits 3 while any generation task of its group is pending or running. Before scoring, its gate
checks every method of the evaluation: exactly one PNG per prompt id with a sidecar that is complete and names the
expected method, resolution, seed and prompt id. If images are missing it parks (generation unfinished) or exits 6
(generation finished, so inputs are missing: see the list above). The samplers skip finished images, so a failed
generation task can simply be moved back to pending. A GPU evaluation that needs set-level intervals releases its CPU
bootstrap (`c1_`, `c3_boot_6144`, `c5_`) into `taskq_cpu/pending` when it writes `DONE`. A CUDA out-of-memory error
caused by another process requeues the evaluation (exit 75), as in the Krea-195 task.

## Code added for these tasks (tests in tests/test_review_round.py, CPU)

- `sca/attn.py`, `sca/dedup.py`: `--bridge-ratio RHO` (`SCAContext.bridge_ratio`), the bridge size tied to the window.
- `scripts/run_area.py`: `--phase2-windows`, `--label`.
- `scripts/sample_flux2.py`: the late-duplication options of `scripts/sample.py` (bridge, window schedule, matched
  weights) for FLUX.2.
- `slurm/review_tasks.py`: this generator (`--code <freeze> --out <dir>`; it never overwrites a task).
"""


def main():
    global HOLD
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--code", required=True, help="frozen code copy, e.g. /scratch/09032/saini_2/sca/code/<commit>")
    p.add_argument("--out", default=HOLD, help="directory for the held tasks and README.md")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    code, HOLD = a.code.rstrip("/"), a.out.rstrip("/")
    f = Path(code) / "FREEZE_COMMIT"
    commit = f.read_text().strip()[:7] if f.exists() else Path(code).name
    Path(HOLD).mkdir(parents=True, exist_ok=True)
    w = Writer(HOLD, a.dry_run)
    counts = {k: fn(w, code) for k, fn in (("5", item5), ("1", item1), ("2", item2), ("3", item3), ("4", item4))}
    print("tasks per item:", counts, "total", len(w.names))
    if not a.dry_run:
        (Path(HOLD) / "README.md").write_text(readme(code, commit, counts))


if __name__ == "__main__":
    main()
