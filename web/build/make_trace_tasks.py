#!/usr/bin/env python3
"""Write the held GPU tasks for the denoising-trace videos of the project page.

One task per (method, prompt): w_trace_01..08 run SCA (sca_nb) and w_trace_09..16 run AReA on the eight web
prompts, each through scripts/sample_trace.py in a frozen code copy. Tasks and a README go to
/scratch/09032/saini_2/sca/taskq/hold_web/, never to pending/ (the SU freeze of 2026-10-24 holds; taskq/STOP
is never touched). Rerunning overwrites the 16 task files and the README and nothing else.

Final operator (2026-10-30): --methods sca_v2 --commit 7bd64c0 --code-tag web_trace2_7bd64c0 --prefix zw2_trace
--pending writes zw2_trace_01..08 (sca_v2 on the same eight prompts and seeds) straight into taskq/pending. The
spool workers take tasks in plain sorted order, so the zw2_ prefix runs them after the x_ reruns and the z98x
evaluations already queued; taskq/STOP is never touched.

--freeze first creates the frozen copy /scratch/09032/saini_2/sca/code/<code tag>/ from `git archive <commit>`
(code paths only) plus the uncommitted overlay (scripts/sample_trace.py, web/build/compare_selection.json),
with COMMIT and OVERLAY.md. An existing frozen copy is never modified.

Run with the sca env python: python web/build/make_trace_tasks.py [--freeze] [--print]
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
TASKQ = Path("/scratch/09032/saini_2/sca/taskq")
CODE_ROOT = Path("/scratch/09032/saini_2/sca/code")
OUT_ROOT = Path("/scratch/09032/saini_2/sca/web_assets/trace")
PY = "/work/09032/saini_2/vista/library/uv-envs/sca/bin/python"
COMMIT = "75baaa8"                       # HEAD on 2026-10-27; sca/attn.py identical to the campaign's 2d11740
CODE_TAG = f"web_trace_{COMMIT}"
CAMPAIGN_CODE = CODE_ROOT / "2d11740"
FROZEN_PATHS = ["sca", "scripts", "slurm", "tests", "repro/evaluator_snapshot", "README.md", "HANDOVER.md",
                "PROPOSAL.md", ".gitignore"]
OVERLAY = ["scripts/sample_trace.py", "web/build/compare_selection.json"]
SIDS = ["m8343371271909676658", "5849277588805396896", "2232973817337527860", "m548913239957610051",
        "m8737279395568207187", "997293625843621481", "1516768667427572324", "3943607367684487957"]
METHODS = ["sca", "area"]                # SCA first: w_trace_01..08, then AReA 09..16
NAME = {"sca": "SCA (sca_nb)", "area": "AReA (official code)", "sca_v2": "SCA, final operator (sca_v2)"}
CAMPAIGN_DIR = {"sca": "out/main/sca_nb/4096x4096", "area": "out/main/area/4096x4096",
                "sca_v2": "out/dupfix/sca_v2/4096x4096"}
# GPU minutes per task on one GH200: load plus flex compile ~2.5, campaign sampling (sidecar seconds: SCA 71 s,
# AReA 209 s), 28 tiled 4096^2 VAE decodes at ~12 s, final PNG and checks ~0.5.
MINUTES = {"sca": 10, "area": 13, "sca_v2": 10}
MIN_SECONDS = {"sca": 1200, "area": 1500, "sca_v2": 1200}   # worker_loop_v2 deadline header, with margin
ENV = ("export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 "
       "XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton "
       "OMP_NUM_THREADS=8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(commit, dest, campaign_code=CAMPAIGN_CODE):
    if dest.exists():
        raise SystemExit(f"{dest} exists; frozen copies are never modified (choose another --code-tag)")
    tmp = dest.with_name(dest.name + f".partial.{os.getpid()}")
    tmp.mkdir(parents=True)
    tar = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", commit, *FROZEN_PATHS],
                         check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=tar, check=True)
    lines = []
    for rel in OVERLAY:
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, tmp / rel)
        lines.append(f"| `{rel}` | `{sha256(tmp / rel)}` |")
    same = (campaign_code / "sca/attn.py").exists() and sha256(campaign_code / "sca/attn.py") == sha256(tmp / "sca/attn.py")
    (tmp / "COMMIT").write_text(commit + "\n")
    (tmp / "OVERLAY.md").write_text(
        f"# {dest.name}\n\nFrozen copy for the held web-trace tasks (taskq/hold_web). Never edit it.\n\n"
        f"- `git archive {commit}` of: {', '.join(FROZEN_PATHS)}.\n"
        f"- `sca/attn.py` identical to the campaign copy `{campaign_code}`: {'yes' if same else 'NO'}.\n"
        "- Uncommitted files added on top (written by web/build/make_trace_tasks.py --freeze):\n\n"
        "| File | SHA-256 |\n|---|---|\n" + "\n".join(lines) + "\n")
    os.rename(tmp, dest)
    print(f"froze {dest} (commit {commit}, attn.py same as campaign: {same})")


def task_text(nn, method, sid, code_dir, out_root, held=True):
    return (
        "#!/bin/bash\n"
        f"# SCA_MIN_SECONDS={MIN_SECONDS[method]}\n"
        f"# Web trace W{nn:02d}: {NAME[method]} denoising trace of {sid} for the project page. Same derived seed,\n"
        f"# settings and code path as {CAMPAIGN_DIR[method]}, plus per-step x0 decodes (scripts/sample_trace.py).\n"
        + ("# Held: release via hold_web/README.md. " if held else "# Queued in taskq/pending. ")
        + "Generated by web/build/make_trace_tasks.py.\n"
        f"{ENV}\n"
        "unset CROPE_TRANSFORMER CROPE_ROPE\n"
        f"cd {code_dir}\n"
        f"{PY} scripts/sample_trace.py --method {method} --sid {sid} --out-root {out_root}\n")


def readme(code_dir, names, out_root):
    rows = []
    for m in METHODS:
        mine = [n for n in names if f"_{m}_" in n]
        rows.append(f"| {NAME[m]} | {mine[0][:10]} to {mine[-1][8:10]} | {len(mine)} | {MINUTES[m]} | "
                    f"{MINUTES[m] * len(mine)} |")
    rows = "\n".join(rows)
    total = sum(MINUTES[m] * len(SIDS) for m in METHODS)
    return f"""# hold_web: denoising traces for the project page (2026-10-27)

Generated by `web/build/make_trace_tasks.py`; rerun it instead of editing by hand.

{len(names)} GPU tasks, one per (method, prompt), on the eight web prompts ({', '.join(SIDS)}; the last two are
random draws of `web/build/compare_selection.json`). `w_trace_01` to `08` run SCA (sca_nb), `09` to `16` AReA.
Each runs `scripts/sample_trace.py` in the frozen copy `{code_dir}` (git {COMMIT} plus the overlay listed in its
OVERLAY.md) with the campaign's derived seed, settings and code path, and writes
`{out_root}/<method>/<sid>/`: `steps/x0_NN.webp` (decoded x0 prediction, 1024 px), `steps/crop_NN.png`
(native 512 px crop at the prompt's box), `final.png` and `trace.json`. A complete trace is skipped on rerun.

## How to release (only after the SU freeze of 2026-10-24 is lifted)

```
mv /scratch/09032/saini_2/sca/taskq/hold_web/w_trace_*.sh /scratch/09032/saini_2/sca/taskq/pending/
```

Workers must be running and `taskq/STOP` removed; both are the lead's decision. `w_` tasks sort after every
other spool prefix, so they run last. Each task needs one GPU; nothing waits on another task.

## Cost (one GH200 per task; estimates, not yet measured on GPU)

| Method | Tasks | Count | Minutes each | GPU-minutes |
|---|---|---|---|---|
{rows}
| Total | | {len(names)} | | {total} (about {total / 60:.1f} GPU-h) |

Per task: model load and FlexAttention compile about 2.5 min, the campaign's own sampling time (sidecar seconds:
SCA 71 s, AReA 209 s), 28 tiled 4096^2 VAE decodes at about 12 s each, final PNG and checks about 0.5 min. Running
one task per method with all eight `--sid` values would save seven model loads per method (about 35 GPU-min).

## After the tasks finish (CPU only)

1. Check each `trace.json`: `final.sha256_matches_campaign`, `final.pixels_identical`, `final.psnr_db`,
   `config_check.match`, and for SCA `pi_trace_matches_campaign`.
2. `cd /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention && nice -n 19 {PY} web/build/render_trace_videos.py`
   writes `web/assets/video/trace/<sid>_sca.mp4`, `<sid>_vs_area.mp4`, their WebP posters and `web/assets/js/traces.js`.
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hold-dir", type=Path, default=TASKQ / "hold_web")
    p.add_argument("--code-tag", default=CODE_TAG)
    p.add_argument("--code-root", type=Path, default=CODE_ROOT)
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--commit", default=COMMIT, help="commit to archive with --freeze")
    p.add_argument("--freeze", action="store_true", help="create the frozen code copy first (must not exist)")
    p.add_argument("--print", action="store_true", help="print the tasks instead of writing them")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=list(NAME))
    p.add_argument("--prefix", default="w_trace", help="task file prefix (spool order is plain sorted order)")
    p.add_argument("--campaign-code", type=Path, default=CAMPAIGN_CODE,
                   help="campaign code copy whose sca/attn.py the frozen copy must match")
    p.add_argument("--pending", action="store_true",
                   help="write the tasks straight into taskq/pending (no README there); never STOP")
    args = p.parse_args(argv)
    hold = (TASKQ / "pending").resolve() if args.pending else args.hold_dir.resolve()
    spool = {(TASKQ / d).resolve() for d in ("", "running", "done", "failed", "stale", "recovered")}
    if not args.pending:
        spool.add((TASKQ / "pending").resolve())
    if hold in spool or hold.name == "STOP":
        raise SystemExit(f"refusing to write tasks into {hold}: held tasks go to a hold_* folder")
    code_dir = args.code_root / args.code_tag
    if args.freeze:
        freeze(args.commit, code_dir, args.campaign_code)
    if not (code_dir / "scripts/sample_trace.py").exists():
        raise SystemExit(f"{code_dir}/scripts/sample_trace.py missing: run with --freeze or pick --code-tag")
    tasks = []
    for m in args.methods:
        for sid in SIDS:
            nn = len(tasks) + 1
            tasks.append((f"{args.prefix}_{nn:02d}_{m}_{sid}.sh",
                          task_text(nn, m, sid, code_dir, args.out_root, held=not args.pending)))
    text = None if args.pending else readme(code_dir, [n for n, _ in tasks], args.out_root)
    if args.print:
        for name, body in tasks:
            print(f"== {name}\n{body}")
        print(text or "")
        return 0
    hold.mkdir(parents=True, exist_ok=True)
    for name, body in tasks + ([("README.md", text)] if text else []):
        tmp = hold / f".{name}.tmp.{os.getpid()}"
        tmp.write_text(body)
        os.replace(tmp, hold / name)
    print(f"wrote {len(tasks)} tasks" + ("" if args.pending else " and README.md") + f" to {hold}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
