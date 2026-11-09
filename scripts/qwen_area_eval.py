#!/usr/bin/env python3
"""Score Qwen-Image SCA against AReA's released Qwen-Image code (HANDOVER Section 38); archive in
results/qwen_5312_vs_area.

Images: /scratch/09032/saini_2/sca/out/showcase/qwen_{sca,area,full}/<HxW>/ (sample_qwen.py and run_area_qwen.py
sidecars), restricted to the prompts of --prompts (15: the first three of each prompt group of
web/build/showcase_prompts.jsonl). 5312^2 is the comparison (SCA and AReA on all 15, full attention on the first 3);
4096^2 (first 3 prompts) is scored and summarized apart.

Metrics, the project's set (as scripts/ideogram4_eval.py): ImageReward, aesthetic, HPSv2, HPSv2.1, CLIP-IQA, ARNIQA
(scripts/eval_extra.py, sca-metrics env; IQA inputs capped at a 1024 long side), CLIP score and PickScore
(constancy_rope eval.py, sca env), MUSIQ_c on four fixed 1024^2 native crops and full-image MUSIQ on a 4096 px
downscale (scripts/ideogram4_musiq.py). Every metric scores the benchmark prompt; AReA's quality suffix is part of its
conditioning only. Paired bootstrap (scripts/paired_bootstrap.py, 2000 draws, seed 0) of SCA minus AReA over the
prompts both have (and minus full attention when at least --min-pairs prompts are common).

Timing: per image from the sidecars (seconds: denoising and VAE decode, prompt embeddings precomputed, for both
methods). An image is shared when another process held its GPU while it was generated: AReA sidecars record this
from a 10 s in-process poll; tasks of slurm/area_qwen_tasks.py also poll every 15 s into
out/showcase/_gpu/<task>.apps ("poll <epoch> ..." lines; a poll counts for an image when it falls inside
[PNG mtime - seconds - 90 s, PNG mtime]); older tasks (000p) have only start and end snapshots, and any process there
marks every image of the task. Medians over all images, over clean ones, and over clean ones that were not the first
image of their process (compile and allocator warm-up). Re-running adds new images (the evaluators cache per image).
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess

P = Path(__file__).resolve().parents[1]
S = Path("/scratch/09032/saini_2/sca")
OUT = S / "out" / "showcase"
PY = "/work/09032/saini_2/vista/library/uv-envs/sca/bin/python"
PYM = "/work/09032/saini_2/vista/library/uv-envs/sca-metrics/bin/python"
CROPE = "/work/09032/saini_2/vista/Projects/Exploration/constancy_rope"
METHODS = ("qwen_sca", "qwen_area", "qwen_full")
EXTRA = "ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa"
SIZES = ("5312x5312", "4096x4096")


def run(cmd, log):
    print("+", " ".join(cmd), flush=True)
    with open(log, "a") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if rc:
        raise SystemExit(f"step failed rc={rc}: {' '.join(cmd)} (log {log})")


def sidecars(res, ids, out=OUT):
    """{(method, prompt id): (sidecar dict, png path)} of complete images of the requested prompts."""
    found = {}
    for m in METHODS:
        for meta in sorted(glob.glob(str(out / m / res / "*.json"))):
            try:
                d = json.load(open(meta))
            except ValueError:
                continue
            png = meta[:-5] + ".png"
            c = d.get("config", {})
            if d.get("complete") and os.path.exists(png) and c.get("prompt_id") in ids and c.get("method") == m:
                found[(m, c["prompt_id"])] = (d, png)
    return found


def link_root(root, res, found):
    for (m, _), (_, png) in found.items():
        dst = root / m / res
        dst.mkdir(parents=True, exist_ok=True)
        for src in (png, png[:-4] + ".json"):
            t = dst / os.path.basename(src)
            if not os.path.lexists(t):
                os.symlink(src, t)
    return {m: sum(1 for k in found if k[0] == m) for m in METHODS}


def read_rows(paths):
    rows = []
    for p in paths:
        if Path(p).exists():
            rows += list(csv.DictReader(open(p)))
    return rows


def write_csv(path, rows, fields=None):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def subset(paths, keep_methods, keep_ids, out_dir):
    out = []
    for p in paths:
        if not Path(p).exists():
            continue
        rows = [r for r in csv.DictReader(open(p)) if r["method"] in keep_methods and r["prompt_id"] in keep_ids]
        if rows:
            q = out_dir / f"{Path(p).parent.name}_{Path(p).stem}__{'_'.join(sorted(keep_methods))}.csv"
            write_csv(q, rows)
            out.append(str(q))
    return out


def merge(rows, ids):
    merged = {}
    for r in rows:
        k = (r["method"], r["prompt_id"])
        merged.setdefault(k, {"method": r["method"], "res": r.get("res"), "prompt_id": r["prompt_id"]})
        merged[k].update({c: v for c, v in r.items()
                          if c not in ("path", "seconds", "seed") and not c.startswith("eval_seconds")})
    per = sorted(merged.values(), key=lambda r: (METHODS.index(r["method"]), ids.index(r["prompt_id"])))
    fields = sorted({c for r in per for c in r} - {"method", "res", "prompt_id"})
    return per, ["method", "res", "prompt_id", *fields]


def summarize(per_image):
    vals = {}
    for r in per_image:
        for k, v in r.items():
            if k in ("method", "res", "prompt_id") or v in ("", None):
                continue
            try:
                vals.setdefault(r["method"], {}).setdefault(k, []).append(float(v))
            except ValueError:
                pass
    out = []
    for m in METHODS:
        if m in vals:
            row = {"method": m, "n": max(len(v) for v in vals[m].values())}
            row.update({k: round(statistics.fmean(v), 5) for k, v in sorted(vals[m].items())})
            out.append(row)
    return out


# ----------------------------------------------------------------------------- GPU sharing per image

def task_records(out=OUT):
    """[(task, image keys (out dir, prompt id), snapshot_shared, polled, poll epochs)] from out/showcase/_gpu/*.apps."""
    recs = []
    for f in sorted((out / "_gpu").glob("*.apps")):
        lines = [x for x in f.read_text().splitlines() if x.strip()]
        cmds = [x[4:] for x in lines if x.startswith("cmd ")]
        if not cmds:
            continue
        c = cmds[-1]

        def arg(k, d=None):
            hit = re.search(rf"--{k} (\S+)", c)
            return hit.group(1) if hit else d
        try:
            prompts = [json.loads(x)["id"] for x in open(arg("prompts")) if x.strip()] if arg("prompts") else []
        except OSError:
            continue
        a, b = int(arg("start", 0)), int(arg("end", len(prompts)))
        keys = {(str(Path(arg("out")).resolve()), pid) for pid in prompts[a:b]} if arg("out") else set()
        polls = [x for x in lines if x.startswith("poll ")]
        snaps = [x for x in lines if not x.startswith(("start ", "end ", "cmd ", "poll "))]
        polled = any(x.startswith("start poller") for x in lines)
        epochs = []
        for x in polls:
            try:
                epochs.append(float(x.split()[1]))
            except (IndexError, ValueError):
                pass
        recs.append({"task": f.name[:-5], "keys": keys, "snapshot_shared": bool(snaps), "polled": polled,
                     "poll_epochs": epochs})
    return recs


def sharing(png, seconds, sidecar, recs):
    """(shared, evidence) for one image."""
    key = (str(Path(png).parent.resolve()), sidecar["config"]["prompt_id"])
    mine = [r for r in recs if key in r["keys"]]
    end = os.path.getmtime(png)
    lo = end - float(seconds) - 90.0
    shared, evidence = False, []
    if sidecar.get("gpu_shared") is not None:
        shared = shared or bool(sidecar["gpu_shared"])
        evidence.append("in-process poll")
    for r in mine:
        if r["polled"]:
            shared = shared or any(lo <= t <= end for t in r["poll_epochs"])
            evidence.append("task poll")
        else:
            shared = shared or r["snapshot_shared"]
            evidence.append("task start/end snapshots")
    return shared, "+".join(sorted(set(evidence))) or "none"


def first_in_process(found, method, res):
    """Prompt ids whose image was the first one generated in its process: AReA sidecars say so; for SCA the earliest
    PNG of each task chunk (tasks run 3 prompts per process; a retried attempt would make this approximate)."""
    firsts = set()
    recs = task_records()
    for (m, pid), (d, png) in found.items():
        if m != method:
            continue
        if "first_call_in_process" in d:
            if d["first_call_in_process"]:
                firsts.add(pid)
            continue
        key = (str(Path(png).parent.resolve()), pid)
        for r in recs:
            if key in r["keys"]:
                group = [(os.path.getmtime(p2), p) for (m2, p), (_, p2) in found.items()
                         if m2 == method and (str(Path(p2).parent.resolve()), p) in r["keys"]]
                if group and min(group)[1] == pid:
                    firsts.add(pid)
    return firsts


def timing(res, found):
    recs = task_records()
    rows = []
    for m in METHODS:
        firsts = first_in_process(found, m, res)
        for (mm, pid), (d, png) in sorted(found.items()):
            if mm != m:
                continue
            shared, ev = sharing(png, d["seconds"], d, recs)
            steps = int(d["config"]["steps"])
            rows.append({"method": m, "res": res, "prompt_id": pid, "steps": steps, "seconds": round(d["seconds"], 2),
                         "seconds_per_step": round(d["seconds"] / steps, 3),
                         "peak_memory_gb": round((d.get("peak_memory_bytes") or 0) / 1e9, 2),
                         "first_in_process": pid in firsts, "gpu_shared": shared, "shared_evidence": ev,
                         "host": d.get("host", "")})
    return rows


def timing_summary(rows):
    out = []
    med = lambda v: round(statistics.median(v), 1) if v else ""
    for m in METHODS:
        for res in SIZES:
            xs = [r for r in rows if r["method"] == m and r["res"] == res]
            if not xs:
                continue
            clean = [r for r in xs if not r["gpu_shared"]]
            warm = [r for r in clean if not r["first_in_process"]]
            out.append({"method": m, "res": res, "steps": xs[0]["steps"], "n": len(xs),
                        "median_seconds_all": med([r["seconds"] for r in xs]),
                        "n_clean": len(clean), "median_seconds_clean": med([r["seconds"] for r in clean]),
                        "n_clean_warm": len(warm), "median_seconds_clean_warm": med([r["seconds"] for r in warm]),
                        "median_seconds_per_step_clean": round(statistics.median(r["seconds_per_step"] for r in clean), 3)
                        if clean else "",
                        "max_peak_memory_gb": max(r["peak_memory_gb"] for r in xs)})
    return out


# ----------------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", type=Path, default=S / "prompts" / "qwen_area15.jsonl")
    p.add_argument("--eval", type=Path, default=S / "eval" / "qwen_5312_vs_area")
    p.add_argument("--results", type=Path, default=P / "results" / "qwen_5312_vs_area")
    p.add_argument("--min-pairs", type=int, default=5, help="Bootstrap a pair only with at least this many prompts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-metrics", action="store_true", help="Reuse the per-image CSVs already in --eval")
    args = p.parse_args(argv)
    ids = [json.loads(x)["id"] for x in open(args.prompts) if x.strip()]
    E, R = args.eval, args.results
    for d in (E / "paired", E / "pairs_tmp", R):
        d.mkdir(parents=True, exist_ok=True)
    log = E / "run.log"
    found = {res: sidecars(res, set(ids)) for res in SIZES}
    counts = {res: link_root(E / f"root{res[:4]}", res, found[res]) for res in SIZES}
    print("images per method:", counts, flush=True)
    if not counts["5312x5312"].get("qwen_sca") or not counts["5312x5312"].get("qwen_area"):
        raise SystemExit("need SCA and AReA images at 5312^2")
    os.environ.update({"HF_HOME": "/scratch/09032/saini_2/tmp/hf", "WANDB_MODE": "disabled"})
    files = {}
    for res in SIZES:
        tag, root = res[:4], E / f"root{res[:4]}"
        if not root.exists():
            continue
        if not args.skip_metrics:
            run([PYM, str(P / "scripts/eval_extra.py"), "--root", str(root), "--prompts", str(args.prompts), "--out",
                 str(E / f"extra{tag}"), "--metrics", EXTRA, "--reference-method", "qwen_area", "--device",
                 args.device, "--batch-size", "4", "--threads", "8", "--allow-missing"], log)
            run([PY, f"{CROPE}/scripts/eval.py", "--root", str(root), "--prompts", str(args.prompts),
                 "--reference-method", "qwen_area", "--res", res, "--out", str(E / f"base{tag}"), "--metrics",
                 "clip,pickscore", "--device", args.device, "--no-wandb"], log)
            run([PY, str(P / "scripts/ideogram4_musiq.py"), "--root", str(root), "--out", str(E / f"musiq{tag}.csv"),
                 "--long-side", "4096", "--device", args.device], log)
        files[res] = [str(E / f"extra{tag}/extra_per_image.csv"), str(E / f"base{tag}/per_image.csv"),
                      str(E / f"musiq{tag}.csv")]
    rows5 = read_rows(files["5312x5312"])
    have = {m: {r["prompt_id"] for r in rows5 if r["method"] == m} for m in METHODS}
    pairs = []
    for ref in ("qwen_area", "qwen_full"):
        common = have["qwen_sca"] & have[ref]
        if len(common) < args.min_pairs:
            print(f"no bootstrap against {ref}: {len(common)} common prompts", flush=True)
            continue
        sub = subset(files["5312x5312"], {"qwen_sca", ref}, common, E / "pairs_tmp")
        q = E / "paired" / f"sca_minus_{ref.split('_')[1]}_5312.csv"
        run([PY, "-I", str(P / "scripts/paired_bootstrap.py"), "--method", "qwen_sca", "--reference", ref,
             "--draws", "2000", "--out", str(q), *sub], log)
        pairs.append(q)
    per5, f5 = merge(rows5, ids)
    write_csv(R / "per_image_5312.csv", per5, f5)
    write_csv(R / "summary_5312.csv", summarize(per5))
    sca_area = sorted(have["qwen_sca"] & have["qwen_area"], key=ids.index)
    common_rows = [r for r in per5 if r["prompt_id"] in sca_area and r["method"] in ("qwen_sca", "qwen_area")]
    write_csv(R / "summary_5312_paired_prompts.csv", summarize(common_rows))
    if "4096x4096" in files:
        rows4 = read_rows(files["4096x4096"])
        if rows4:
            per4, f4 = merge(rows4, ids)
            write_csv(R / "per_image_4096.csv", per4, f4)
            write_csv(R / "summary_4096.csv", summarize(per4))
    for q in pairs:
        shutil.copy(q, R / q.name)
    t = timing("5312x5312", found["5312x5312"]) + timing("4096x4096", found["4096x4096"])
    write_csv(R / "timing.csv", t)
    write_csv(R / "timing_summary.csv", timing_summary(t))
    area = next((d for (m, _), (d, _) in sorted(found["5312x5312"].items()) if m == "qwen_area"), None)
    sca = next((d for (m, _), (d, _) in sorted(found["5312x5312"].items()) if m == "qwen_sca"), None)
    about = {"prompts": [x for x in ids], "paired_prompts_5312": sca_area,
             "images": {res: {m: sorted(pid for (mm, pid) in found[res] if mm == m) for m in METHODS} for res in SIZES},
             "area": None if area is None else {k: area["config"][k] for k in
                                                ("area", "steps", "true_cfg_scale", "negative_prompt", "prompt_suffix",
                                                 "mu_rule", "implementation_sha256", "diffusers", "torch")},
             "area_schedule_5312": None if area is None else area.get("area_schedule"),
             "sca": None if sca is None else {k: sca["config"][k] for k in
                                              ("steps", "true_cfg_scale", "negative_prompt", "mu_mode", "mu", "sca",
                                               "implementation_sha256", "diffusers", "torch")},
             "metrics": "ImageReward, aesthetic, HPSv2, HPSv2.1, CLIP-IQA, ARNIQA (eval_extra.py); CLIP, PickScore "
                        "(constancy_rope eval.py); MUSIQ_c on four 1024^2 crops and MUSIQ on a 4096 px downscale",
             "bootstrap": "scripts/paired_bootstrap.py, prompts resampled, 2000 draws, seed 0, 95% percentile CI"}
    (R / "about.json").write_text(json.dumps(about, indent=1) + "\n")
    shutil.copy(args.prompts, R / "prompts.jsonl")
    files_out = sorted(str(x.relative_to(R)) for x in R.rglob("*") if x.is_file() and x.name != "SHA256SUMS")
    subprocess.check_call(["bash", "-c", f"cd {R} && sha256sum {' '.join('./' + f for f in files_out)} > SHA256SUMS"])
    print("archived", len(files_out), "files in", R, flush=True)


if __name__ == "__main__":
    main()
