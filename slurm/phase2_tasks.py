#!/usr/bin/env python3
"""Write the Phase 2 spool tasks (E1 to E7, E9) and their gated evaluation tasks.

Generation tasks go to taskq/pending (or taskq/hold_phase2 for tasks that need the PHASE2
frozen code). Each evaluation task waits until its generation tasks have left pending/ and
running/, links the common prompt ids (scripts/link_common.py, dev4 excluded) and runs
eval_extra.py (sca-metrics env) and constancy_rope eval.py (sca env).
Existing task files are never overwritten; rerun with --dry-run to list what would be written.
Historical builder: the corrected final evaluations and deadline-aware dispatch
are preserved in results/evidence/dispatch/task_scripts and HANDOVER.md. Existing
tasks are intentionally not rewritten by this generator.
Notes: .agent/notes/phase2_experiments.md.
"""

import argparse
from pathlib import Path

S = Path("/scratch/09032/saini_2/sca")
Q = S / "taskq"
REPO = Path("/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention")
CROPE_LIVE = Path("/work/09032/saini_2/vista/Projects/Exploration/constancy_rope")
CODE = S / "code/2d11740"          # final campaign code
PHASE2 = S / "code/PHASE2"         # frozen by the lead after the --model and UltraImage changes
CROPE = S / "code/crope_c05ebdc"   # frozen constancy_rope clone for DyPE and SEGA
PY = "/work/09032/saini_2/vista/library/uv-envs/sca/bin/python"
PYM = "/work/09032/saini_2/vista/library/uv-envs/sca-metrics/bin/python"
EVAL = "/scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl"
REFS = "/scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_refs.jsonl"
ABL50, EVAL100, HI12, HRGEN, LAT3 = (str(S / n) for n in
                                      ("abl50.jsonl", "eval100.jsonl", "hi12.jsonl", "hrgen200.jsonl", "lat3.jsonl"))
PI = str(S / "calib/k1_1024.npz")
KREA_PI = str(S / "calib/krea_1024.npz")
SCA = f"--pi-table {PI} --gate-sigma 0.89 --label sca_nb"
ENV = ("export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 "
       "XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton "
       "OMP_NUM_THREADS=8\nunset CROPE_TRANSFORMER CROPE_ROPE")
OUT = {}   # name -> text; filled by the builders below, then written
HOLD = set()


def nprompts(path):
    return sum(1 for line in open(path) if line.strip())


def task(name, body, hold=False, comment=""):
    assert name not in OUT, name
    OUT[name] = "#!/bin/bash\n" + (f"# {comment}\n" if comment else "") + body.rstrip() + "\n"
    if hold:
        HOLD.add(name)


def gen(prefix, tag, res, prompts, chunk, cmd, code=CODE, hold=False, comment="", pre=""):
    """One task per prompt chunk: <prefix><start>_<tag>_<res>.sh. cmd gets --res/--prompts/--start/--end."""
    n, names = nprompts(prompts), []
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        name = f"{prefix}{a:03d}_{tag}_{res}.sh"
        task(name, f"{ENV}\n{pre}cd {code}\n{PY} {cmd} --res {res} --prompts {prompts} --start {a} --end {b}",
             hold, comment)
        names.append(name)
    return names


def sca_cmd(out, seed=0, extra=SCA):
    return f"scripts/sample.py --method sca --seed {seed} --guidance 4.5 --out {out} {extra}"


def area_cmd(out, seed=0, extra=""):
    return f"scripts/run_area.py --seed {seed} --guidance 4.5 --out {out} {extra}".rstrip()


def wait_regex(names):
    """Regex over task names (pending/ and running/<name>.<host>) of the generation tasks."""
    groups = sorted({n.split("_", 1)[1][:-3] for n in names})
    prefixes = sorted({n[0] for n in names})
    return f"^(a_)?[{''.join(prefixes)}][0-9]{{3}}_({'|'.join(g.replace('.', '[.]') for g in groups)})[.]sh"


def evaluation(name, gens, root, res, seed, prompts, methods, ref, extra, base, refs=True, hold=False,
               comment="", min_count=1, allow_missing=False):
    rx = wait_regex(gens)
    pairs = " ".join(f"{k}={v}" for k, v in methods)
    r = S / "out/evalroot" / root
    body = f"""{ENV}
while ls {Q}/pending {Q}/running 2>/dev/null | grep -q -E '{rx}'; do sleep 60; done
ls {Q}/failed 2>/dev/null | grep -E '{rx}' | sed 's/^/WARNING failed generation task: /'
{PY} -I {REPO}/scripts/link_common.py --root {r} --res {res} --seed {seed} --prompts {prompts} --min-count {min_count} {pairs}
{PYM} {REPO}/scripts/eval_extra.py --root {r} --prompts {prompts} --out {S}/eval/{root}_extra --metrics {extra} --reference-method {ref} --device cuda --batch-size 16 --threads 8
{PY} {CROPE_LIVE}/scripts/eval.py --root {r} --prompts {prompts}{f' --references {REFS}' if refs else ''} --reference-method {ref} --res {res} --out {S}/eval/{root}_base --metrics {base} --device cuda{' --allow-missing' if allow_missing else ''}
"""
    task(name, body, hold, comment)


EX_FULL, EX_SMALL = "ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa,is", "ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa"
BASE_FULL, BASE_ABL = "clip,pickscore,musiq,fid,kid,patch_fid,patch_kid", "clip,pickscore,musiq,patch_fid,patch_kid"
MAIN = S / "out/main"
R4 = "4096x4096"


def build():
    # E6 calibration budget (tables built on CPU: calib/k1_1024_n{4,8,16}.npz) and E7 gate sensitivity.
    e6, e7 = [], []
    for k in (4, 8, 16):
        lbl = f"sca_cal{k}"
        e6 += gen("e", lbl, R4, ABL50, 5, sca_cmd(S / f"out/calib_budget/{lbl}/{R4}",
                  extra=f"--pi-table {S}/calib/k1_1024_n{k}.npz --gate-sigma 0.89 --label {lbl}"),
                  comment=f"E6: pi table from the first {k} calibration prompts")
    for g in ("85", "93"):
        lbl = f"sca_g{g}"
        e7 += gen("e", lbl, R4, ABL50, 5, sca_cmd(S / f"out/gate/{lbl}/{R4}",
                  extra=f"--pi-table {PI} --gate-sigma 0.{g} --label {lbl}"), comment=f"E7: gate sigma* = 0.{g}")
    evaluation("e998_eval_e6_calib.sh", e6, "e6_calib", R4, 0, ABL50,
               [("sca_nb", MAIN / f"sca_nb/{R4}")] + [(f"sca_cal{k}", S / f"out/calib_budget/sca_cal{k}/{R4}") for k in (4, 8, 16)],
               "sca_nb", EX_SMALL, BASE_ABL, comment="E6 eval: 46 prompts, reference sca_nb (32-prompt table)")
    evaluation("e999_eval_e7_gate.sh", e7, "e7_gate", R4, 0, ABL50,
               [("sca_nb", MAIN / f"sca_nb/{R4}")] + [(f"sca_g{g}", S / f"out/gate/sca_g{g}/{R4}") for g in ("85", "93")],
               "sca_nb", EX_SMALL, BASE_ABL, comment="E7 eval: 46 prompts, reference sca_nb (gate 0.89)")

    # E5 seeds 1 and 2 on abl50.
    for seed in (1, 2):
        names = gen("f", f"sca_nb_s{seed}", R4, ABL50, 5, sca_cmd(S / f"out/seed{seed}/sca_nb/{R4}", seed),
                    comment=f"E5: SCA final, seed {seed}")
        names += gen("f", f"area_s{seed}", R4, ABL50, 5, area_cmd(S / f"out/seed{seed}/area/{R4}", seed),
                     comment=f"E5: AReA, seed {seed}")
        evaluation(f"f99{7 + seed}_eval_e5_seed{seed}.sh", names, f"e5_seed{seed}", R4, seed, ABL50,
                   [("sca", S / f"out/seed{seed}/sca_nb/{R4}"), ("area", S / f"out/seed{seed}/area/{R4}")],
                   "area", EX_SMALL, BASE_ABL, comment=f"E5 eval: seed {seed}, 46 prompts, reference area")

    # E1 baselines with the frozen constancy_rope harness (FLUX.1-dev, guidance 4.5, mu = max_shift).
    e1 = []
    for m in ("dype", "sega_official"):
        e1 += gen("g", m, R4, EVAL, 5, f"scripts/sample.py --method {m} --seed 0 --guidance 4.5 --steps 28 "
                  f"--out {MAIN / m / R4}", code=CROPE, comment=f"E1: {m} (constancy_rope c05ebdc, FLUX.1-dev)")
    evaluation("g999_eval_e1_baselines.sh", e1, "e1_baselines", R4, 0, EVAL,
               [("sca_nb", MAIN / f"sca_nb/{R4}"), ("area", MAIN / f"area/{R4}"),
                ("dype", MAIN / f"dype/{R4}"), ("sega_official", MAIN / f"sega_official/{R4}")],
               "sca_nb", EX_FULL, BASE_FULL, comment="E1 eval: 191 prompts, reference sca_nb")
    # E1 UltraImage: new wrapper, so it runs from PHASE2 (held until the lead freezes it).
    ui = gen("g", "ultraimage", R4, EVAL, 5, f"scripts/run_ultraimage.py --seed 0 --out {MAIN / 'ultraimage' / R4}",
             code=PHASE2, hold=True, comment="E1: UltraImage official code, its own FLUX.1-dev defaults")
    evaluation("g998_eval_e1_ultraimage.sh", ui, "e1_ultraimage", R4, 0, EVAL,
               [("sca_nb", MAIN / f"sca_nb/{R4}"), ("area", MAIN / f"area/{R4}"), ("ultraimage", MAIN / f"ultraimage/{R4}")],
               "sca_nb", EX_FULL, BASE_FULL, hold=True, comment="E1 eval: UltraImage, 191 prompts, reference sca_nb")

    # E4 HRGen-1K, first 200 prompts, no references.
    hr = gen("h", "sca_nb", R4, HRGEN, 5, sca_cmd(S / f"out/hrgen/sca_nb/{R4}"), comment="E4: HRGen-1K SCA")
    hr += gen("h", "area", R4, HRGEN, 5, area_cmd(S / f"out/hrgen/area/{R4}"), comment="E4: HRGen-1K AReA")
    evaluation("h999_eval_e4_hrgen.sh", hr, "e4_hrgen", R4, 0, HRGEN,
               [("sca", S / f"out/hrgen/sca_nb/{R4}"), ("area", S / f"out/hrgen/area/{R4}")],
               "area", EX_FULL, "clip,pickscore,musiq", refs=False, comment="E4 eval: 200 HRGen prompts, no references")

    # E2 higher resolution.
    for res, prompts, chunk, extra, base, allow in (("6144x6144", EVAL100, 4, EX_FULL, BASE_FULL, False),
                                                    ("8192x8192", HI12, 2, EX_SMALL, "clip,pickscore,musiq", True)):
        names = gen("i", "sca_nb", res, prompts, chunk, sca_cmd(MAIN / f"sca_nb/{res}"), comment=f"E2: SCA {res}")
        names += gen("i", "area", res, prompts, chunk, area_cmd(MAIN / f"area/{res}"), comment=f"E2: AReA {res}")
        evaluation(f"i99{8 if res.startswith('6') else 9}_eval_e2_{res.split('x')[0]}.sh", names,
                   f"e2_{res.split('x')[0]}", res, 0, prompts,
                   [("sca_nb", MAIN / f"sca_nb/{res}"), ("area", MAIN / f"area/{res}")], "area", extra, base,
                   refs=res.startswith("6"), allow_missing=allow,
                   comment=f"E2 eval {res}, reference area (Aesthetic-4K references are 4096 px)")

    # E3 latency curve: one task per resolution, sequential methods on one otherwise idle GPU.
    lat = []
    for i, res in enumerate(("2048x2048", "3072x3072", "4096x4096", "6144x6144", "8192x8192")):
        o = S / f"out/latency_curve/{res}"
        lines = [f"{ENV}", "nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv",
                 "nvidia-smi --query-compute-apps=pid,used_memory --format=csv", f"cd {CODE}",
                 f"{PY} scripts/sample.py --method sca --res {res} --prompts {LAT3} --seed 0 --guidance 4.5 --out {o}/sca_nb {SCA}",
                 f"{PY} scripts/run_area.py --res {res} --prompts {LAT3} --seed 0 --guidance 4.5 --out {o}/area"]
        if res != "8192x8192":
            lines.append(f"{PY} scripts/sample.py --method full --res {res} --prompts {LAT3} --seed 0 --guidance 4.5 --out {o}/full")
        else:
            lines.append("echo 'full attention skipped at 8192x8192 (262144 image tokens, impractical)'")
        lines.append(f"grep -H -o '\"seconds\": [0-9.]*' {o}/*/*.json")
        name = f"y00{i}_latency_{res}.sh"
        task(name, "\n".join(lines), comment=f"E3: batch-1 latency at {res}; needs an otherwise idle GPU")
        lat.append(name)
    rx = "^y00[0-9]_latency_"
    task("y999_summary_latency.sh", f"""{ENV}
while ls {Q}/pending {Q}/running 2>/dev/null | grep -q -E '{rx}'; do sleep 60; done
{PY} -I - <<'EOF'
import csv, glob, json, statistics
rows = []
for f in sorted(glob.glob("{S}/out/latency_curve/*/*/*.json")):
    d = json.load(open(f)); c = d["config"]
    rows.append((c["res"], c["method"], c["prompt_id"], d["seconds"], d.get("peak_memory_bytes"), d.get("first_call_in_process")))
with open("{S}/eval/latency_curve.csv", "w", newline="") as h:
    w = csv.writer(h); w.writerow(["res", "method", "prompt_id", "seconds", "peak_memory_bytes", "first_call_in_process"]); w.writerows(rows)
groups = {{}}
for r in rows: groups.setdefault((r[0], r[1]), []).append(r[3])
for k, v in sorted(groups.items()): print(*k, "n", len(v), "median_s", round(statistics.median(v), 1))
EOF
""", comment="E3 summary: median seconds per (res, method) into eval/latency_curve.csv")

    # E9 FLUX.1-Krea-dev: needs the --model flag, so everything runs from PHASE2 and is held.
    task("d000_calib_krea.sh", f"""{ENV}
cd {PHASE2}
{PY} scripts/calibrate.py --model krea --prompts {REPO}/scripts/calib_prompts.jsonl --count 16 --res 1024x1024 --scales 2,4 --guidance 4.5 --out {KREA_PI} --transfer-with {PI}
""", hold=True, comment="E9: SCA calibration on FLUX.1-Krea-dev, 16 prompts at 1024x1024, scales 2,4")
    wait_pi = (f"while [ ! -f {KREA_PI[:-4]}.json ]; do ls {Q}/failed | grep -q '^d000_calib_krea' && "
               f"{{ echo 'Krea calibration failed'; exit 1; }}; sleep 60; done\n")
    k9 = gen("j", "sca_krea", R4, EVAL100, 5, sca_cmd(S / f"out/krea/sca_krea/{R4}",
             extra=f"--model krea --pi-table {KREA_PI} --gate-sigma 0.89 --label sca_krea"), code=PHASE2, hold=True,
             pre=wait_pi, comment="E9: SCA on Krea with the Krea pi table")
    k9 += gen("j", "area_krea", R4, EVAL100, 5, area_cmd(S / f"out/krea/area/{R4}", extra="--model krea"),
              code=PHASE2, hold=True, comment="E9: AReA on Krea (its own default checkpoint)")
    kr = Path("/scratch/09032/saini_2/constancy_rope/out_krea/main")
    evaluation("j999_eval_e9_krea.sh", k9, "e9_krea", R4, 0, EVAL100,
               [("sca_krea", S / f"out/krea/sca_krea/{R4}"), ("area", S / f"out/krea/area/{R4}"),
                ("dype", kr / f"dype/{R4}"), ("sega_official", kr / f"sega_official/{R4}")],
               "area", EX_FULL, BASE_FULL, hold=True,
               comment="E9 eval: Krea, 100 prompts; DyPE/SEGA Krea outputs reused from constancy_rope out_krea/main")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    build()
    written = skipped = 0
    for name, text in sorted(OUT.items()):
        dest = Q / ("hold_phase2" if name in HOLD else "pending") / name
        # Operators may park dependency-bound or long tasks in additional hold folders,
        # and latency tasks gain an a_ priority prefix. Never recreate those tasks.
        variants = (name, "a_" + name)
        exists = [d for d in Q.iterdir() if d.is_dir()
                  and any(list(d.glob(candidate + "*")) for candidate in variants)]
        if exists:
            skipped += 1
            continue
        if not a.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
        written += 1
    counts = {}
    for name in OUT:
        key = ("hold:" if name in HOLD else "") + name[0]
        counts[key] = counts.get(key, 0) + 1
    print(f"{'would write' if a.dry_run else 'wrote'} {written}, skipped existing {skipped}; per prefix {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
