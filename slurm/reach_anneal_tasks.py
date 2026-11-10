#!/usr/bin/env python3
"""HANDOVER Section 40 spool tasks: experiment A (reach-annealed hand-off) and experiment B (does the per-head table
matter inside the progressive schedule). Prefix 000a_ (A) and 000b_ (B) so they sort ahead of the 000r_ Qwen tasks
(the worker's `ls | sort` ignores '_' under the locale and is bytewise otherwise; both orders agree here).
Never overwrites a task (dup_tasks.write). Every task runs the frozen tree --code.

A  a-screen: every variant on copyprone21 (r7/copyprone21.jsonl, chunks of 7, 000a_a<k>NN_<label>) with a GPU detector
   triage and a blind label packet r7/label/pRA1..pRA5 released by the last chunk (000a_0a1_tri_<label>), and on the
   46-prompt subset (round7_tasks.abl46: chunks of 8, 000a_b<k>NN_<label>, evaluation 000a_0a0_eval_<label> ->
   eval/r7_46_<label>, released by the last chunk). Each evaluation then copies the 195 gate 000a_0a2_gate195 into
   pending; the gate decides once every candidate's 46-prompt evaluation is DONE (see GATE_RULE) and releases the staged
   195-prompt run of the chosen variant (taskq/hold_ra195/<label>: generation 000a_c..., evaluation
   000a_0a0_eval195_<label> -> eval/ra195_<label> and results/ra195, clean batch-1 timing 000a_0a3_latency_<label>).
   The 195 run is provisional until the hand labels confirm the copy counts (HANDOVER Section 40).
B  b: progressive schedule unchanged, tables ablated in both upper stages: prog_headmean (--pi-variant headmean),
   prog_layermean (--pi-variant layermean), prog_win (--pi-const 1.0, window plus bridge only); 195 prompts, seed 0,
   chunks of 15 (000b_g<k>NN_<label>), evaluation 000b_0a0_eval_prog195_ablation (eval/prog195_ablation ->
   results/prog195_ablation, against sca_v2_prog with HRDiT alongside; metric caches of eval/prog195_sca_v2) and one
   clean batch-1 timing task 000b_0a1_latency_prog_ablation.
Usage: slurm/reach_anneal_tasks.py --code /scratch/09032/saini_2/sca/code/<freeze> --stage a-screen|a-stage195|b [--dry]
"""

import argparse
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dup_tasks as dt  # noqa: E402
import round7_tasks as rt  # noqa: E402

S, P, PY, PYM, CR, ENV = dt.S, dt.P, dt.PY, dt.PYM, dt.CR, dt.ENV
Q = f"{S}/taskq"
C21 = f"{S}/r7/copyprone21.jsonl"
A4K, REFS = dt.A4K, dt.REFS
SUB = rt.SUB
OP = "--lattice-pool mean --bridge-cells 8 --bridge-k 16"
# label: (sample.py flags on top of BASE, 46-prompt run in the screen, 195 candidate). Measured schedules from
# scripts/handoff_from_mass.py (image share, head mean; steps of the 28-step 4096^2 schedule):
#   ra_fixed  control: global steps 0-7 (gate 0.89), r = 2 on steps 8-13, native window from step 14
#   ra_m      tau_2 0.5 / tau_4 0.4 (the only measured pair that walks 1 -> 2 -> 4): global 0-4, r = 2 5-12, r = 4 13-27
#   ra_t40    tau 0.4: m_2 > 0.4 at every step, so r = 2 from step 0 (no global opening) to 12, r = 4 from 13
#   ra_t50    tau 0.5: global 0-4, r = 2 from step 5 to the end (m_4 never reaches 0.5)
#   ra_t60    tau 0.6: global 0-19, r = 2 from step 20 to the end (m_4 never reaches 0.6)
A_VARIANTS = {
    "ra_fixed": (f"--gate-sigma 0.89 --scale-fixed 0.77:2 {OP}", True, True),
    "ra_m": (f"--gate-sigma 0.9404 --scale-fixed 0.7892:2 {OP}", True, True),
    "ra_t40": (f"--gate-sigma 0.0 --scale-fixed 0.7892:2 {OP}", True, True),
    "ra_t50": (f"--gate-sigma 0.9404 --scale-fixed 0.0:2 {OP}", True, True),
    "ra_t60": (f"--gate-sigma 0.5896 --scale-fixed 0.0:2 {OP}", False, False),
}
PACKET_SEED = {"ra_fixed": 4101, "ra_m": 4102, "ra_t40": 4103, "ra_t50": 4104, "ra_t60": 4105}
PACKET = {"ra_fixed": "pRA1", "ra_m": "pRA2", "ra_t40": "pRA3", "ra_t50": "pRA4", "ra_t60": "pRA5"}  # neutral names: labelers stay blind to the variant
HOLD_RA = f"{Q}/hold_ra"            # gate template and parked runs
HOLD_195 = f"{Q}/hold_ra195"        # staged 195-prompt runs, one directory per candidate
GATE_NAME = "000a_0a2_gate195.sh"
GATE_RULE = ("metric part of r7_decide's relaxed bar on the 46 (no significant loss vs sca_v2 on IR, HPSv2.1, PickScore; "
             "every other significant loss still better than AReA; median time < 202 s) AND copyprone21 detector flags "
             "<= 6 of 21 (round-7 rule for sending a screen to blind labels) AND people/animal detector flags on the 46 "
             "<= 2 (sca_v2 4, AReA 1); among those, fewest copyprone21 flags, ties by the larger PickScore delta")
PROG_CODE_FLAGS = ("--pi-table /scratch/09032/saini_2/sca/calib/k1_1024.npz --gate-sigma 0.89 --bridge-cells 8 "
                   "--bridge-k 16 --win-sched 0.88:128,0.86:112,0.84:96,0.82:80 --win-pi-match --win-pi-interp")
B_VARIANTS = {"prog_headmean": "--pi-variant headmean", "prog_layermean": "--pi-variant layermean",
              "prog_win": "--pi-const 1.0"}
PROG_DIR = f"{S}/out/dupfix_prog/sca_v2_prog/4096x4096"
HRDIT_DIRS = ",".join(f"{S}/out/{d}/hrdit_dev_author/4096x4096" for d in
                      ("baseline_expansion_20261017_rest145", "baseline_expansion_20261017_v1", "baseline_expansion_20261017_dev4"))
PROG_EVAL = f"{S}/eval/prog195_sca_v2"   # scored sca_v2_prog and HRDiT on the 195 prompts (caches, per-image rows)

CAPTURE = {}


def _write(path, text, dry):
    """dup_tasks.write with a hook: 46-prompt evaluations of the A screen get the gate copy appended."""
    path = Path(path)
    hook = CAPTURE.get(path.name)
    if hook:
        text = hook(text)
    return ORIG_WRITE(path, text, dry)


ORIG_WRITE = dt.write
dt.write = _write


def gate_release_line():
    return (f"cp {HOLD_RA}/gate195.template {Q}/pending/.{GATE_NAME}.tmp.$$ && "
            f"mv -f {Q}/pending/.{GATE_NAME}.tmp.$$ {Q}/pending/{GATE_NAME} && echo \"queued the 195 gate\"\n")


def triage_task(code, label, dry):
    name = f"000a_0a1_tri_{label}.sh"
    O = f"{S}/r7/cpu_det/{label}"
    out = f"{S}/r7/out/{label}/4096x4096"
    text = rt.header(name, f"detector triage of {label} on copyprone21 (GPU) and its blind label packet {PACKET[label]}", 1200) + f"""set -uo pipefail
O={O}; mkdir -p $O/imgs
n=$({PY} -c "import glob,json; ids=[json.loads(l)['id'] for l in open('{C21}') if l.strip()]; print(sum(1 for i in ids for f in glob.glob('{out}/'+glob.escape(i)+'_*.json') if json.load(open(f)).get('complete')))")
[ "$n" -ge 21 ] || {{ echo "only $n of 21 images"; exit 5; }}
cd {code}
{PY} - "{C21}" "{out}" "$O/imgs" <<'PYEOF'
import glob, json, os, sys
ids, src, dst = sys.argv[1:4]
for i in (json.loads(l)["id"] for l in open(ids) if l.strip()):
    for f in glob.glob(f"{{src}}/{{glob.escape(i)}}_*.png") + glob.glob(f"{{src}}/{{glob.escape(i)}}_*.json"):
        t = os.path.join(dst, os.path.basename(f))
        if not os.path.exists(t):
            os.symlink(f, t)
PYEOF
{PY} scripts/dup_detect.py --subjects {SUB} --device cuda --threads 8 --sides 800,1333 --out $O/det --dirs {label}=$O/imgs > $O/detect.log 2>&1 || {{ echo "detector failed"; tail -5 $O/detect.log; exit 4; }}
{PY} scripts/dup_rate.py --dets {S}/eval/dup_detect_20261027 {S}/eval/a4k195_main_sca_v2/det $O/det --subjects {SUB} --ids {C21} --thr 0.35 --kinds person,animal --methods {label},sca_v2,area --ref sca_v2 --per-image $O/per_image_people.csv --out $O/rates_people.csv | tee $O/rates_people.txt
echo "$(date +%H:%M) $(head -1 $O/rates_people.txt)" >> {S}/r7/cpu_det/auto_triage.log
bash {S}/r7/label_people.sh {PACKET[label]} {PACKET_SEED[label]} {label}={out}:{C21} > {S}/r7/label/{PACKET[label]}.log 2>&1 && echo "packet r7/label/{PACKET[label]} built" || echo "WARNING packet build failed (see r7/label/{PACKET[label]}.log)"
exit 0
"""
    dt.write(Path(dt.EVAL_HOLD) / name, text, dry)
    return name


def a_screen(code, dry):
    rt.PREFIX, rt.SCA_V2 = "000a_", []
    hooked = []
    for k, (label, (flags, run46, _)) in enumerate(A_VARIANTS.items()):
        fl = shlex.split(flags)
        tri = triage_task(code, label, dry)
        rt.gen(code, dry, label, fl, C21, f"{S}/r7/out/{label}/4096x4096", 7, f"a{k}", f"{S}/r7/markers/{label}_c21", tri)
        ename = f"000a_0a0_eval_{label}.sh"
        CAPTURE[ename] = lambda t: t.replace("date -u +%FT%TZ > $E/DONE\n", "date -u +%FT%TZ > $E/DONE\n" + gate_release_line())
        hooked.append(ename)
        queue = f"{Q}/pending" if run46 else f"{HOLD_RA}/park46_{label}"
        Path(queue).mkdir(parents=True, exist_ok=True)
        rt.abl46(code, dry, label, fl, 8, f"b{k}", "0a0", queue)
    gate_template(code, dry)


def gate_template(code, dry):
    cands = [lab for lab, (_, run46, c195) in A_VARIANTS.items() if run46 and c195]
    G = f"{S}/eval/ra_gate195"
    text = rt.header(GATE_NAME, "decide which A variant (if any) gets the provisional 195-prompt run", 900) + f"""set -uo pipefail
G={G}; mkdir -p $G
[ -e $G/DECIDED ] && {{ echo "already decided: $(cat $G/DECIDED)"; exit 0; }}
cd {code}
{PY} -I - "$G" <<'PYEOF'
import csv, json, subprocess, sys
from pathlib import Path
G = Path(sys.argv[1]); S = "{S}"
cands = {cands!r}
missing = [c for c in cands if not Path(f"{{S}}/eval/r7_46_{{c}}/DONE").exists()]
if missing:
    print("waiting for the 46-prompt evaluations of", missing); sys.exit(0)
fr = G / "frontier.csv"
args = [f"{{S}}/eval/r7_46_{{c}}:{{c}}" for c in cands]
r = subprocess.run([sys.executable, "scripts/r7_decide.py", "--frontier", str(fr)] + args, capture_output=True, text=True)
(G / "r7_decide.txt").write_text(r.stdout + r.stderr)
rows = {{x["label"]: x for x in csv.DictReader(open(fr))}} if fr.exists() else {{}}
def c21(c):
    f = Path(f"{{S}}/r7/cpu_det/{{c}}/rates_people.csv")
    if not f.exists():
        return None
    return next((int(x["dup"]) for x in csv.DictReader(open(f)) if x["method"] == c), None)
def pick(c):
    for x in csv.DictReader(open(f"{{S}}/eval/r7_46_{{c}}/paired/{{c}}_vs_sca_v2.csv")):
        if x["metric"] == "pickscore":
            return float(x["delta"])
    return float("nan")
lines, ok = [], []
for c in cands:
    x = rows.get(c)
    if x is None:
        lines.append(f"{{c}}: no r7_decide row"); continue
    reasons = [s for s in x["relaxed_fail"].split("; ") if s and not s.startswith("rate above")]
    n21 = c21(c)
    if n21 is None or n21 > 6:
        reasons.append(f"copyprone21 detector {{n21}} > 6")
    if int(x["detector_flags"]) > 2:
        reasons.append(f"46 detector flags {{x['detector_flags']}} > 2")
    lines.append(f"{{c}}: copyprone21 detector {{n21}}/21, 46 flags {{x['detector_flags']}}, losses {{x['n_losses']}} "
                 f"[{{x['losses']}}], time x{{x['time_ratio']}} -> {{'ELIGIBLE' if not reasons else 'out: ' + '; '.join(reasons)}}")
    if not reasons:
        ok.append((n21, -pick(c), c))
choice = sorted(ok)[0][2] if ok else "NONE"
rule = {GATE_RULE!r}
(G / "decision.txt").write_text("rule: " + rule + "\\n" + "\\n".join(lines) + f"\\nchoice: {{choice}}\\n")
print((G / "decision.txt").read_text())
(G / "CHOICE").write_text(choice + "\\n")
PYEOF
[ -e $G/CHOICE ] || exit 0
mkdir $G/lock 2>/dev/null || {{ echo "another gate run is deciding"; exit 0; }}
L=$(cat $G/CHOICE); echo "$L" > $G/DECIDED
if [ "$L" != NONE ] && [ -d {HOLD_195}/$L ]; then n=$(ls {HOLD_195}/$L/*.sh 2>/dev/null | wc -l); mv {HOLD_195}/$L/*.sh {Q}/pending/ && echo "released $n tasks of the 195-prompt run of $L"; fi
exit 0
"""
    p = Path(HOLD_RA)
    p.mkdir(parents=True, exist_ok=True)
    if not dry:
        (p / "gate195.template").write_text(text)
        (p / "gate195.template").chmod(0o755)
    print("write", p / "gate195.template")


def latency_task(name, R, methods, dry, where):
    """One clean batch-1 sweep on one GPU: each method in its own process on latency4_warmup.jsonl (row 0 an excluded
    warm-up, rows 1-3 timed), medians into R/latency.csv. methods: [(tag, code dir, command prefix without --out)]."""
    lines = [rt.header(name, "clean batch-1 latency sweep (one GPU, one process per method, warm-up row excluded)", 3600),
             "set -uo pipefail\n",
             f"RUN={S}/out/ra_latency/$(date +%Y%m%dT%H%M%S)_$(hostname -s)_{name[:-3]}\nR={R}\nmkdir -p $RUN $R\n",
             "GPU_ID=${CUDA_VISIBLE_DEVICES:-0}; GPU_ID=${GPU_ID%%,*}\n",
             "nvidia-smi -i $GPU_ID --query-gpu=name,uuid,driver_version,utilization.gpu,memory.used,clocks.max.sm --format=csv | tee $RUN/gpu.csv\n",
             "nvidia-smi -i $GPU_ID --query-compute-apps=pid,used_memory --format=csv | tee $RUN/gpu_apps_before.csv\n",
             "[ \"$(wc -l < $RUN/gpu_apps_before.csv)\" -gt 1 ] && echo \"WARNING: other processes hold this GPU; timings are not clean\" | tee $RUN/NOT_CLEAN\n"]
    for tag, code, cmd in methods:
        lines.append(f"echo \"$(date -u +%FT%TZ) start {tag}\"; (cd {code} && {cmd} --prompts {S}/latency4_warmup.jsonl "
                     f"--start 0 --end 4 --out $RUN/{tag}) || echo \"FAILED {tag}\"\n")
    lines.append(f"""{PY} - "$RUN" "$R" <<'PYEOF'
import csv, glob, json, os, statistics, sys
run, R = sys.argv[1:3]
ids = [json.loads(l)["id"] for l in open("{S}/latency4_warmup.jsonl") if l.strip()]
rows = []
for d in sorted(glob.glob(run + "/*/")):
    tag = os.path.basename(d.rstrip("/"))
    sec = {{}}
    for f in glob.glob(d + "*.json"):
        x = json.load(open(f))
        if x.get("complete"):
            sec[x["config"]["prompt_id"]] = x["seconds"]
    t = [sec[i] for i in ids[1:] if i in sec]
    rows.append({{"method": tag, "n_timed": len(t), "median_s": round(statistics.median(t), 3) if t else "",
                 "min_s": round(min(t), 3) if t else "", "max_s": round(max(t), 3) if t else "",
                 "warmup_s": round(sec[ids[0]], 3) if ids[0] in sec else "", "clean": int(not os.path.exists(run + "/NOT_CLEAN")),
                 "run": run}})
with open(os.path.join(R, "latency.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
PYEOF
(cd $R && sha256sum $(find . -name '*.csv' -o -name '*.txt' | sort) > SHA256SUMS)
""")
    dt.write(Path(where) / name, "".join(lines), dry)


def a_stage195(code, dry):
    """Stage the 195-prompt run of every candidate in taskq/hold_ra195/<label> (generation and timing; the evaluation
    waits in hold_dup_eval and is released by the last generation task)."""
    rt.PREFIX, rt.SCA_V2 = "000a_", []
    for k, (label, (flags, run46, c195)) in enumerate(A_VARIANTS.items()):
        if not c195:
            continue
        fl = shlex.split(flags)
        hold = f"{HOLD_195}/{label}"
        Path(hold).mkdir(parents=True, exist_ok=True)
        out = f"{S}/out/ra195/{label}/4096x4096"
        E = f"{S}/eval/ra195_{label}"
        ename = f"000a_0a0_eval195_{label}.sh"
        names, pat = rt.gen(code, dry, label, fl, A4K, out, 15, f"c{k}", f"{E}/gen_done", ename, hold)
        eval195(code, dry, label, out, ename, names, pat)
        latency_task(f"000a_0a3_latency_{label}.sh", f"{P}/results/ra195",
                     [("sca_v2", code, f"{PY} scripts/sample.py " + " ".join(base_args(rt_sca_v2())) + " --label sca_v2"),
                      (label, code, f"{PY} scripts/sample.py " + " ".join(base_args(fl)) + f" --label {label}")], dry, hold)


def rt_sca_v2():
    return ["--bridge-cells", "8", "--bridge-k", "16", "--win-sched", "0.88:128,0.86:112,0.84:96,0.82:80",
            "--win-pi-match", "--win-pi-interp"]


def base_args(flags):
    args = list(dt.BASE)
    if "--gate-sigma" in flags:
        i = args.index("--gate-sigma")
        del args[i:i + 2]
    return args + list(flags)


def eval195(code, dry, label, out, ename, gen_names, gen_pat):
    """rt.eval195 (label vs sca_v2 and AReA: detector, 15 metrics, corrected FID / patch FID, paired bootstraps, time)
    plus per-image paired bootstraps of label vs sca_v2_prog and vs HRDiT from eval/prog195_sca_v2's rows, into
    results/ra195 with SHA256SUMS."""
    captured = {}
    orig = dt.write

    def grab(path, text, dry_):
        captured["text"] = text
    dt.write = grab
    try:
        rt.eval195(code, True, label, out, "0a0", gen_names, gen_pat)
    finally:
        dt.write = orig
    text = captured["text"]
    text = text.replace(f"{S}/eval/r7_195_{label}", f"{S}/eval/ra195_{label}").replace(
        f"{P}/results/r7_195_{label}", f"{P}/results/ra195")
    text = text.replace(f"a7_0a0_eval195_{label}.sh", ename)
    extra = f"""mkdir -p $E/merged
{PY} -I - "$E" "{PROG_EVAL}" "{label}" <<'PYEOF'
import csv, sys
E, PE, label = sys.argv[1:4]
for name in ("extra/extra_per_image.csv", "base/per_image.csv"):
    rows, fields = [], []
    for src, keep in ((f"{{E}}/{{name}}", None), (f"{{PE}}/{{name}}", ("sca_v2_prog", "hrdit_dev_author"))):
        for r in csv.DictReader(open(src)):
            if keep is None or r["method"] in keep:
                rows.append(r)
                fields += [k for k in r if k not in fields]
    with open(f"{{E}}/merged/" + name.split("/")[-1], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print("merged", name, len(rows), "rows")
PYEOF
for ref in sca_v2_prog hrdit_dev_author; do
  run {PY} -I {P}/scripts/paired_bootstrap.py --method {label} --reference $ref --draws 2000 --out $E/paired/{label}_vs_${{ref}}.csv $E/merged/extra_per_image.csv $E/merged/per_image.csv
done
"""
    text = text.replace("cp $E/base/summary.csv $R/base_summary.csv", extra + "cp $E/base/summary.csv $R/base_summary.csv")
    dt.write(Path(dt.EVAL_HOLD) / ename, text, dry)


def b_stage(code, dry):
    E = f"{S}/eval/prog195_ablation"
    R = f"{P}/results/prog195_ablation"
    ename = "000b_0a0_eval_prog195_ablation.sh"
    rows = []
    for k, (label, extra) in enumerate(B_VARIANTS.items()):
        out = f"{S}/out/prog195_ablation/{label}/4096x4096"
        for a in range(0, 195, 15):
            rows.append((f"000b_g{k}{a // 15:02d}_{label}.sh", label, extra, out, a, min(a + 15, 195)))
    pat = "|".join(n[:-3] for n, *_ in rows)
    for name, label, extra, out, a, b in rows:
        cmd = (f"{PY} scripts/sample_progressive.py --prompts {A4K} --seed 0 --out {out} --start {a} --end {b} "
               f"{PROG_CODE_FLAGS} --label {label} {extra}")
        text = (rt.header(name, f"prog195 ablation {label} ({extra}) on the 195 prompts [{a}, {b}), progressive schedule "
                          "unchanged", 3600) + f"cd {code}\nC=\"{cmd}\"\n$C || {{ sleep 30; $C; }} || exit $?\n"
                + dt.self_release(f"{E}/gen_done", name, pat, len(rows), ename))
        dt.write(Path(Q) / "pending" / name, text, dry)
    spec = ";".join([f"sca_v2_prog|0|{PROG_DIR}", f"hrdit_dev_author|0|{HRDIT_DIRS}"]
                    + [f"{lab}|0|{S}/out/prog195_ablation/{lab}/4096x4096" for lab in B_VARIANTS])
    import prog_headline_tasks as ph
    pairs = " ".join([f"{lab}:sca_v2_prog" for lab in B_VARIANTS] + ["sca_v2_prog:hrdit_dev_author"]
                     + [f"{lab}:hrdit_dev_author" for lab in B_VARIANTS])
    text = (rt.header(ename, "prog195 ablation: (ii) headmean, (iii) layermean, (iv) window only vs (i) sca_v2_prog, "
                      "HRDiT alongside; 195 prompts, 4096^2, seed 0", 7200) + "set -uo pipefail\n"
            + f"Q={Q}; E={E}; R={R}\nexport E\n"
            + dt.guard(ename, pat, (f"{E}/gen_done", len(rows)))
            + f"if ls $Q/failed 2>/dev/null | grep -q -E '^({pat})[.]sh'; then echo \"generation task failed; not evaluating\"; exit 3; fi\n"
            + "if [ -e $E/DONE ]; then echo \"already complete: $E\"; exit 0; fi\n"
            + "mkdir -p $E/paired $E/extra $E/base/features $R\n"
            + "run(){ \"$@\" 2>&1 | tee -a $E/run.log; local rc=${PIPESTATUS[0]}; if [ $rc -ne 0 ]; then echo \"step failed rc=$rc: $*\"; exit $rc; fi; }\n"
            + ph.LINKER.format(py=PY, prompts=A4K, spec=spec)
            + f"[ -f {PROG_EVAL}/extra/extra_cache.jsonl ] && [ ! -f $E/extra/extra_cache.jsonl ] && cp -p {PROG_EVAL}/extra/extra_cache.jsonl $E/extra/\n"
            + f"[ -f {PROG_EVAL}/base/evaluation_cache.json ] && [ ! -f $E/base/evaluation_cache.json ] && cp -p {PROG_EVAL}/base/evaluation_cache.json $E/base/ && cp -p {PROG_EVAL}/base/features/*.npz $E/base/features/ 2>/dev/null\n"
            + f"cd {code}\n"
            + f"run {PY} scripts/dup_detect.py --subjects {SUB} --device cuda --threads 8 --sides 800,1333 --out $E/det --dirs "
            + " ".join(f"{lab}={S}/out/prog195_ablation/{lab}/4096x4096" for lab in B_VARIANTS) + "\n"
            + f"{PY} scripts/dup_rate.py --dets {S}/eval/dup_detect_20261027 {S}/eval/a4k195_main_sca_v2/det {PROG_EVAL}/det $E/det --subjects {SUB} --thr 0.35 --kinds person,animal --methods {','.join(B_VARIANTS)},sca_v2_prog --ref sca_v2_prog --per-image $E/dup_per_image_people.csv --out $E/dup_rates_people.csv | tee -a $E/run.log\n"
            + f"run {PYM} {P}/scripts/eval_extra.py --root $E/root --prompts {A4K} --out $E/extra --metrics ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa,is --reference-method sca_v2_prog --device cuda --batch-size 16 --threads 8\n"
            + f"run {PY} {CR}/scripts/eval.py --root $E/root --prompts {A4K} --references {REFS} --reference-method sca_v2_prog --res 4096x4096 --out $E/base --metrics clip,pickscore,musiq,fid,kid,patch_fid,patch_kid --device cuda\n"
            + "mkdir -p $E/corrected_fidp/features && cp -p $E/base/features/*.npz $E/corrected_fidp/features/ && cp -p $E/base/evaluation_cache.json $E/corrected_fidp/ || { echo \"feature copy failed\"; exit 7; }\n"
            + f"(cd {P} && WANDB_MODE=disabled run {PY} repro/evaluate_corrected.py --root $E/root --prompts {A4K} --references {REFS} --reference-method sca_v2_prog --reference-clusters images --res 4096x4096 --out $E/corrected_fidp --metrics fid,kid,patch_fid,patch_kid --device cuda --bootstrap 0 --threads 8 --no-wandb)\n"
            + f"for pair in {pairs}; do a=${{pair%%:*}}; b=${{pair#*:}}\n"
            + f"  run {PY} -I {P}/scripts/paired_bootstrap.py --method $a --reference $b --draws 2000 --out $E/paired/${{a}}_vs_${{b}}.csv $E/extra/extra_per_image.csv $E/base/per_image.csv\ndone\n"
            + ph.timing(PROG_DIR, A4K, 0)
            + "cp $E/base/summary.csv $R/base_summary.csv && cp $E/extra/extra_summary.csv $R/extra_summary.csv && cp $E/paired/*.csv $R/ && cp $E/dup_rates_people.csv $E/timing.txt $R/\n"
            + "mkdir -p $R/corrected_fidp && cp $E/corrected_fidp/summary.csv $E/corrected_fidp/paired_deltas.csv $R/corrected_fidp/\n"
            + "(cd $R && sha256sum $(find . -name '*.csv' -o -name '*.txt' | sort) > SHA256SUMS)\n"
            + "date -u +%FT%TZ > $E/DONE; echo \"DONE prog195_ablation\"\n")
    dt.write(Path(dt.EVAL_HOLD) / ename, text, dry)
    prog = f"{PY} scripts/sample_progressive.py --seed 0 {PROG_CODE_FLAGS}"
    latency_task("000b_0a1_latency_prog_ablation.sh", R,
                 [("sca_v2_prog", code, prog + " --label sca_v2_prog")]
                 + [(lab, code, prog + f" --label {lab} {extra}") for lab, extra in B_VARIANTS.items()], dry, f"{Q}/pending")



def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--code", required=True)
    p.add_argument("--stage", choices=("a-screen", "a-stage195", "b"), required=True)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    if not Path(a.code, "FREEZE_COMMIT").exists():
        raise SystemExit(f"{a.code} is not a frozen code tree (no FREEZE_COMMIT)")
    {"a-screen": a_screen, "a-stage195": a_stage195, "b": b_stage}[a.stage](a.code, a.dry)


if __name__ == "__main__":
    main()
