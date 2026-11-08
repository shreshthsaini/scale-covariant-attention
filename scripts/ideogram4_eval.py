#!/usr/bin/env python3
"""Evaluate the Ideogram 4 sets (HANDOVER Section 36) and archive the numbers in results/ideogram4_8k.

8192^2, per image: ImageReward, aesthetic, HPSv2, HPSv2.1, CLIP-IQA and ARNIQA (scripts/eval_extra.py, sca-metrics env;
IQA inputs capped at a 1024 long side as everywhere in the project), CLIP score and PickScore (constancy_rope eval.py,
sca env), MUSIQ_c on four 1024^2 native crops and full-image MUSIQ on a 4096 px downscale (scripts/ideogram4_musiq.py;
pyiqa's full-resolution MUSIQ does not fit at 8192^2). Paired bootstrap (scripts/paired_bootstrap.py, 2000 draws, seed 0)
of SCA minus each baseline over the prompts both have. Ideogram 4 at 4096^2 is scored with the same metrics and kept
apart (native reference, no pairing). Timing and peak memory come from the sidecars; a timing is flagged shared when
the task's GPU log lists another process. Re-running adds new images (every evaluator caches per image)."""

import argparse
import csv
import glob
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

P = Path(__file__).resolve().parents[1]
S = Path("/scratch/09032/saini_2/sca")
OUT = S / "out" / "ideogram4"
PY = "/work/09032/saini_2/vista/library/uv-envs/sca/bin/python"
PYM = "/work/09032/saini_2/vista/library/uv-envs/sca-metrics/bin/python"
CROPE = "/work/09032/saini_2/vista/Projects/Exploration/constancy_rope"
METHODS = ("sca", "full", "sega", "ntk", "yarn")
EXTRA = "ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa"


def run(cmd, log):
    print("+", " ".join(cmd), flush=True)
    with open(log, "a") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if rc:
        raise SystemExit(f"step failed rc={rc}: {' '.join(cmd)} (log {log})")


def link_root(root, res, methods, ids, out=OUT):
    """Symlink complete PNG + sidecar pairs of the requested prompt ids into root/<method>/<res>/."""
    found = {}
    for m in methods:
        n = 0
        for meta in sorted(glob.glob(str(out / f"ideogram4_{m}" / res / "*.json"))):
            d = json.load(open(meta))
            png = meta[:-5] + ".png"
            c = d.get("config", {})
            if not d.get("complete") or not os.path.exists(png) or c.get("prompt_id") not in ids or c.get("method") != m:
                continue
            dst = root / m / res
            dst.mkdir(parents=True, exist_ok=True)
            for src in (png, meta):
                t = dst / os.path.basename(src)
                if not os.path.lexists(t):
                    os.symlink(src, t)
            n += 1
        found[m] = n
    return found


def read_rows(paths):
    rows = []
    for p in paths:
        if Path(p).exists():
            rows += list(csv.DictReader(open(p)))
    return rows


def subset(paths, keep_methods, keep_ids, out_dir):
    """Copies of the per-image CSVs restricted to two methods and the prompts both have (paired_bootstrap needs equal
    prompt sets)."""
    out = []
    for p in paths:
        rows = [r for r in csv.DictReader(open(p)) if r["method"] in keep_methods and r["prompt_id"] in keep_ids]
        if not rows:
            continue
        q = out_dir / f"{Path(p).stem}__{'_'.join(sorted(keep_methods))}.csv"
        with open(q, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        out.append(str(q))
    return out


def shared(task_glob, out=None):
    """True when a GPU log of the task lists any compute process at its start or end snapshot (the task's own process
    is not running at either, so any listed process is another tenant of the GPU)."""
    for f in glob.glob(str((out or OUT) / "_gpu" / task_glob)):
        lines = [x for x in open(f).read().splitlines() if x and not x.startswith(("start ", "end ", "cmd "))]
        if lines:
            return True
    return False


def timing(res, ids, out=OUT):
    rows = []
    for m in METHODS:
        for meta in sorted(glob.glob(str(out / f"ideogram4_{m}" / res / "*.json"))):
            d = json.load(open(meta))
            c = d.get("config", {})
            if not d.get("complete") or c.get("prompt_id") not in ids or c.get("method") != m:
                continue
            tag = {"sca": "sca8k", "full": "full8k" if res == "8192x8192" else "nat4k", "sega": "sega8k",
                   "ntk": "ntk8k", "yarn": "yarn8k"}[m]
            pattern = f"*{tag}_{c['prompt_id']}.apps" if res == "8192x8192" else f"*{tag}_*.apps"
            rows.append({"method": m, "res": res, "prompt_id": c["prompt_id"], "seconds": round(d["seconds"], 2),
                         "peak_memory_gb": round((d.get("peak_memory_bytes") or 0) / 1e9, 2), "gpu": d.get("gpu"),
                         "gpu_shared": shared(pattern, out)})
    seen, out = set(), []
    for r in rows:
        k = (r["method"], r["prompt_id"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def write_csv(path, rows, fields=None):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(per_image, methods):
    """Mean of every metric per method over its images."""
    vals = {}
    for r in per_image:
        for k, v in r.items():
            if k in ("method", "res", "prompt_id", "seed", "path", "seconds") or v in ("", None):
                continue
            try:
                vals.setdefault(r["method"], {}).setdefault(k, []).append(float(v))
            except ValueError:
                pass
    out = []
    for m in methods:
        if m in vals:
            row = {"method": m, "n": max(len(v) for v in vals[m].values())}
            row.update({k: round(statistics.fmean(v), 5) for k, v in sorted(vals[m].items())})
            out.append(row)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, default=S / "prompts" / "ideogram4_10.jsonl")
    p.add_argument("--eval", type=Path, default=S / "eval" / "ideogram4_8k")
    p.add_argument("--results", type=Path, default=P / "results" / "ideogram4_8k")
    p.add_argument("--min-pairs", type=int, default=3, help="Pair a baseline only with at least this many prompts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--outputs", type=Path, default=OUT, help="Generation root (ideogram4_<method>/<HxW>)")
    p.add_argument("--skip-metrics", action="store_true", help="Reuse the per-image CSVs already in --eval (tests)")
    args = p.parse_args(argv)
    ids = [json.loads(line)["id"] for line in open(args.prompts) if line.strip()]
    E, R = args.eval, args.results
    for d in (E / "paired", E / "pairs_tmp", R):
        d.mkdir(parents=True, exist_ok=True)
    log = E / "run.log"
    found = link_root(E / "root8k", "8192x8192", METHODS, set(ids), args.outputs)
    native = link_root(E / "root4k", "4096x4096", ("full",), set(ids), args.outputs)
    print("8192^2 images per method:", found, "| 4096^2:", native, flush=True)
    if not found.get("sca"):
        raise SystemExit("no SCA images yet")
    env = {"HF_HOME": "/scratch/09032/saini_2/tmp/hf", "WANDB_MODE": "disabled"}
    os.environ.update(env)
    for tag, root, res in (("8k", E / "root8k", "8192x8192"), ("4k", E / "root4k", "4096x4096")):
        if not root.exists() or args.skip_metrics:
            continue
        ref = "sca" if tag == "8k" else "full"
        run([PYM, str(P / "scripts/eval_extra.py"), "--root", str(root), "--prompts", str(args.prompts), "--out",
             str(E / f"extra{tag}"), "--metrics", EXTRA, "--reference-method", ref, "--device", args.device,
             "--batch-size", "4", "--threads", "8", "--allow-missing"], log)
        run([PY, f"{CROPE}/scripts/eval.py", "--root", str(root), "--prompts", str(args.prompts), "--reference-method",
             ref, "--res", res, "--out", str(E / f"base{tag}"), "--metrics", "clip,pickscore", "--device",
             args.device, "--no-wandb"], log)
        run([PY, str(P / "scripts/ideogram4_musiq.py"), "--root", str(root), "--out", str(E / f"musiq{tag}.csv"),
             "--device", args.device], log)
    files8 = [str(E / "extra8k/extra_per_image.csv"), str(E / "base8k/per_image.csv"), str(E / "musiq8k.csv")]
    rows8 = read_rows(files8)
    have = {m: {r["prompt_id"] for r in rows8 if r["method"] == m} for m in METHODS}
    pairs = []
    for b in METHODS[1:]:
        common = have["sca"] & have[b]
        if len(common) < args.min_pairs:
            continue
        sub = subset(files8, {"sca", b}, common, E / "pairs_tmp")
        out = E / "paired" / f"sca_vs_{b}.csv"
        run([PY, "-I", str(P / "scripts/paired_bootstrap.py"), "--method", "sca", "--reference", b, "--draws", "2000",
             "--out", str(out), *sub], log)
        pairs.append(out)
    # archive
    merged = {}
    for r in rows8:
        k = (r["method"], r["prompt_id"])
        merged.setdefault(k, {"method": r["method"], "res": r.get("res"), "prompt_id": r["prompt_id"]})
        merged[k].update({c: v for c, v in r.items() if c not in ("path", "seconds", "seed")})
    per8 = sorted(merged.values(), key=lambda r: (METHODS.index(r["method"]), ids.index(r["prompt_id"])))
    fields = sorted({c for r in per8 for c in r} - {"method", "res", "prompt_id"})
    write_csv(R / "per_image_8192.csv", per8, ["method", "res", "prompt_id", *fields])
    write_csv(R / "summary_8192.csv", summarize(per8, METHODS))
    rows4 = read_rows([E / "extra4k/extra_per_image.csv", E / "base4k/per_image.csv", E / "musiq4k.csv"])
    if rows4:
        m4 = {}
        for r in rows4:
            m4.setdefault(r["prompt_id"], {"method": "native_4096", "res": "4096x4096", "prompt_id": r["prompt_id"]})
            m4[r["prompt_id"]].update({c: v for c, v in r.items() if c not in ("path", "seconds", "seed", "method", "res")})
        per4 = [m4[i] for i in ids if i in m4]
        f4 = sorted({c for r in per4 for c in r} - {"method", "res", "prompt_id"})
        write_csv(R / "per_image_native4096.csv", per4, ["method", "res", "prompt_id", *f4])
        write_csv(R / "summary_native4096.csv", summarize(per4, ("native_4096",)))
    for q in pairs:
        shutil.copy(q, R / q.name)
    t8, t4 = timing("8192x8192", set(ids), args.outputs), timing("4096x4096", set(ids), args.outputs)
    write_csv(R / "timing.csv", t8 + t4)
    med = []
    for m in METHODS:
        for res, rows in (("8192x8192", t8), ("4096x4096", t4)):
            xs = [r for r in rows if r["method"] == m]
            if xs:
                clean = [r["seconds"] for r in xs if not r["gpu_shared"]]
                med.append({"method": m if res == "8192x8192" else "native_4096", "res": res, "n": len(xs),
                            "median_seconds": round(statistics.median(r["seconds"] for r in xs), 1),
                            "median_seconds_clean": round(statistics.median(clean), 1) if clean else "",
                            "n_clean": len(clean), "median_gpu_hours": round(statistics.median(r["seconds"] for r in xs) / 3600, 3),
                            "max_peak_memory_gb": max(r["peak_memory_gb"] for r in xs)})
    write_csv(R / "timing_summary.csv", med)
    for name in ("check.json",):
        src = S / "ideogram4" / "check" / name
        if src.exists():
            shutil.copy(src, R / "gpu_check.json")
    for t in ("ideogram4_4096_cond", "ideogram4_4096_uncond", "ideogram4_2048_cond", "ideogram4_2048_uncond"):
        src = S / "calib" / f"{t}.json"
        if src.exists():
            shutil.copy(src, R / f"reach_{t.split('ideogram4_')[1]}.json")
    caps = S / "ideogram4" / "captions" / "captions_1x1_seed0.jsonl"
    if caps.exists():
        shutil.copy(caps, R / "captions.jsonl")
    shutil.copy(args.prompts, R / "prompts.jsonl")
    files = sorted(str(x.relative_to(R)) for x in R.rglob("*") if x.is_file() and x.name != "SHA256SUMS")
    subprocess.check_call(["bash", "-c", f"cd {R} && sha256sum {' '.join('./' + f for f in files)} > SHA256SUMS"])
    print("archived", len(files), "files in", R, flush=True)


if __name__ == "__main__":
    main()
