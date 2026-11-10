#!/usr/bin/env python3
"""Score Qwen-Image SCA configurations against AReA on one prompt set and size (HANDOVER Section 39): the TUNE
screens (results/qwen_tune/<HxW>) and the TEST run of the chosen configuration (results/qwen_5312_tuned).

Same metric code path as scripts/qwen_area_eval.py (its helpers are imported): ImageReward, aesthetic, HPSv2, HPSv2.1,
CLIP-IQA, ARNIQA (eval_extra.py, sca-metrics env), CLIP score and PickScore (constancy_rope eval.py), MUSIQ_c on four
fixed 1024^2 crops and MUSIQ on a 4096 px downscale (ideogram4_musiq.py). Every metric scores the benchmark prompt
(sidecar "prompt"), also for images conditioned on a suffixed prompt. Paired bootstrap (paired_bootstrap.py, 2000
draws, seed 0) of every method against --reference over the prompts both have.

Selection score (fixed before any TUNE result, .agent/notes/qwen_tune_20261109.md): for PickScore, HPSv2.1 and
ImageReward, d_z = mean paired difference / SD of the differences; screen.csv ranks the methods by the mean d_z over
the three (ties within 0.10 go to the faster one).

Timing from the sidecars (seconds: denoising, stage transitions if any, VAE decode; prompt embeddings precomputed),
with the shared-GPU and first-in-process flags of qwen_area_eval.py (out/showcase/_gpu/<task>.apps polls).
Usage: qwen_tune_eval.py --prompts P --res HxW --reference LABEL --method LABEL=DIR [--method ...] --eval E --results R
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_area_eval as qae  # noqa: E402

P, PY, PYM, CROPE = qae.P, qae.PY, qae.PYM, qae.CROPE
SCREEN_METRICS = ("pickscore", "hpsv21", "ir")


def load(methods, res, ids):
    """{(label, prompt id): (sidecar, png)} of complete images in each label's folder (sidecar method may differ from
    the label only for images reused from another campaign, e.g. qwen_sca40)."""
    found = {}
    for label, folder in methods.items():
        for meta in sorted(Path(folder).glob("*.json")):
            try:
                d = json.loads(meta.read_text())
            except ValueError:
                continue
            png = str(meta)[:-5] + ".png"
            c = d.get("config", {})
            if d.get("complete") and os.path.exists(png) and c.get("prompt_id") in ids and c.get("res") == res:
                found[(label, c["prompt_id"])] = (d, png)
    return found


def link(root, res, found):
    """Eval root <label>/<res>/<stem>.{png,json}; the json is rewritten with method = label (the metric tools group
    by the sidecar method), everything else as generated."""
    for (label, _), (d, png) in found.items():
        dst = root / label / res
        dst.mkdir(parents=True, exist_ok=True)
        t = dst / os.path.basename(png)
        if not os.path.lexists(t):
            os.symlink(png, t)
        j = dst / (os.path.basename(png)[:-4] + ".json")
        if not j.exists():
            dd = json.loads(json.dumps(d))
            dd["config"]["method"] = label
            j.write_text(json.dumps(dd))


def d_z(diffs):
    if len(diffs) < 2:
        return float("nan")
    sd = statistics.stdev(diffs)
    return statistics.fmean(diffs) / sd if sd > 0 else float("nan")


def screen(per_image, methods, reference, timing_rows):
    """Per method: mean paired difference and d_z for the three ranking metrics, their mean d_z, median seconds."""
    by = {(r["method"], r["prompt_id"]): r for r in per_image}
    med = {}
    for m in methods:
        xs = [float(r["seconds"]) for r in timing_rows if r["method"] == m]
        med[m] = round(statistics.median(xs), 1) if xs else ""
    out = []
    for m in methods:
        if m == reference:
            continue
        common = sorted(p for (mm, p) in by if mm == m and (reference, p) in by)
        row = {"method": m, "reference": reference, "n_prompts": len(common)}
        zs = []
        for k in SCREEN_METRICS:
            diffs = [float(by[(m, p)][k]) - float(by[(reference, p)][k]) for p in common
                     if by[(m, p)].get(k) not in ("", None) and by[(reference, p)].get(k) not in ("", None)]
            row[f"{k}_delta"] = round(statistics.fmean(diffs), 5) if diffs else ""
            row[f"{k}_dz"] = round(d_z(diffs), 4) if diffs else ""
            row[f"{k}_wins"] = sum(x > 0 for x in diffs)
            if diffs:
                zs.append(d_z(diffs))
        for k in ("musiq_crop1024", "clipiqa", "aesthetic", "hpsv2", "clip"):
            diffs = [float(by[(m, p)][k]) - float(by[(reference, p)][k]) for p in common
                     if by[(m, p)].get(k) not in ("", None) and by[(reference, p)].get(k) not in ("", None)]
            row[f"{k}_delta"] = round(statistics.fmean(diffs), 5) if diffs else ""
        row["mean_dz"] = round(statistics.fmean(zs), 4) if len(zs) == len(SCREEN_METRICS) else ""
        row["median_seconds"] = med[m]
        row["reference_median_seconds"] = med.get(reference, "")
        out.append(row)
    out.sort(key=lambda r: -(r["mean_dz"] if r["mean_dz"] != "" else -1e9))
    return out


def metrics(args, root, E, res, log):
    """The TEST comparison's metric tools on the eval root (each caches per image, so a re-run scores new images)."""
    qae.run([PYM, str(P / "scripts/eval_extra.py"), "--root", str(root), "--prompts", str(args.prompts), "--out",
             str(E / "extra"), "--metrics", qae.EXTRA, "--reference-method", args.reference, "--device",
             args.device, "--batch-size", "4", "--threads", "8", "--allow-missing"], log)
    qae.run([PY, f"{CROPE}/scripts/eval.py", "--root", str(root), "--prompts", str(args.prompts),
             "--reference-method", args.reference, "--res", res, "--out", str(E / "base"), "--metrics",
             "clip,pickscore", "--device", args.device, "--no-wandb"], log)
    qae.run([PY, str(P / "scripts/ideogram4_musiq.py"), "--root", str(root), "--out", str(E / "musiq.csv"),
             "--long-side", str(args.musiq_long_side), "--device", args.device], log)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--res", required=True)
    p.add_argument("--reference", required=True, help="Label of AReA's images")
    p.add_argument("--method", action="append", required=True, metavar="LABEL=DIR",
                   help="Image folder of one method (repeat); the reference too")
    p.add_argument("--eval", type=Path, required=True, help="Scratch work directory (metric caches)")
    p.add_argument("--results", type=Path, required=True, help="Archive directory (CSV, about.json, SHA256SUMS)")
    p.add_argument("--min-pairs", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-metrics", action="store_true")
    p.add_argument("--musiq-long-side", type=int, default=4096)
    p.add_argument("--expect", type=int, default=0, help="Wait until every method (but --partial ones) has this many "
                   "images, scoring the ones that exist meanwhile (0 = score what exists now)")
    p.add_argument("--partial", nargs="*", default=[], help="Labels exempt from --expect (e.g. full attention)")
    p.add_argument("--wait-max", type=float, default=3600.0, help="Seconds to wait for --expect at most")
    args = p.parse_args(argv)
    methods = dict(x.split("=", 1) for x in args.method)
    if args.reference not in methods:
        raise SystemExit("--reference must be one of the --method labels")
    qae.METHODS = tuple(methods)       # ordering for qae.merge / summarize / timing
    qae.SIZES = (args.res,)
    ids = [json.loads(x)["id"] for x in open(args.prompts) if x.strip()]
    E, R, res = args.eval, args.results, args.res
    for d in (E / "paired", E / "pairs_tmp", R):
        d.mkdir(parents=True, exist_ok=True)
    log = E / "run.log"
    root = E / "root"
    os.environ.update({"HF_HOME": "/scratch/09032/saini_2/tmp/hf", "WANDB_MODE": "disabled"})
    t_end = time.time() + args.wait_max
    while True:   # with --expect: score what exists, then re-score as images land (metric tools cache per image)
        found = load(methods, res, set(ids))
        link(root, res, found)
        counts = {m: sum(1 for k in found if k[0] == m) for m in methods}
        print(time.strftime("%H:%M:%S"), "images per method:", counts, flush=True)
        if not counts[args.reference]:
            if args.expect and time.time() < t_end:
                time.sleep(60)
                continue
            raise SystemExit("no reference images")
        complete = all(counts[m] >= min(args.expect, len(ids)) for m in methods if m not in args.partial)
        if args.skip_metrics or not args.expect or complete or time.time() >= t_end:
            break
        metrics(args, root, E, res, log)
        time.sleep(30)
    if not args.skip_metrics:
        metrics(args, root, E, res, log)
    files = [str(E / "extra/extra_per_image.csv"), str(E / "base/per_image.csv"), str(E / "musiq.csv")]
    rows = qae.read_rows(files)
    have = {m: {r["prompt_id"] for r in rows if r["method"] == m} for m in methods}
    pairs = []
    for m in methods:
        if m == args.reference:
            continue
        common = have[m] & have[args.reference]
        if len(common) < args.min_pairs:
            print(f"no bootstrap of {m}: {len(common)} common prompts", flush=True)
            continue
        sub = qae.subset(files, {m, args.reference}, common, E / "pairs_tmp")
        q = E / "paired" / f"{m}_minus_{args.reference}.csv"
        qae.run([PY, "-I", str(P / "scripts/paired_bootstrap.py"), "--method", m, "--reference", args.reference,
                 "--draws", "2000", "--out", str(q), *sub], log)
        pairs.append(q)
    per, fields = qae.merge(rows, ids)
    qae.write_csv(R / "per_image.csv", per, fields)
    qae.write_csv(R / "summary.csv", qae.summarize(per))
    for q in pairs:
        shutil.copy(q, R / q.name)
    t = qae.timing(res, {k: v for k, v in found.items()})
    qae.write_csv(R / "timing.csv", t)
    qae.write_csv(R / "timing_summary.csv", qae.timing_summary(t))
    sc = screen(per, list(methods), args.reference, t)
    qae.write_csv(R / "screen.csv", sc)
    configs = {}
    for (m, _), (d, _) in sorted(found.items()):
        if m not in configs:
            c = d["config"]
            configs[m] = {k: c.get(k) for k in ("method", "steps", "true_cfg_scale", "negative_prompt", "mu_mode", "mu",
                                                 "prompt_suffix", "progressive", "sca", "area", "mu_rule",
                                                 "implementation_sha256", "attention")}
            configs[m]["folder"] = methods[m]
    about = {"prompts": ids, "res": res, "reference": args.reference, "images": {m: sorted(p for (mm, p) in found if mm == m)
                                                                                 for m in methods},
             "configs": configs, "screen_rule": "mean d_z over PickScore, HPSv2.1, ImageReward vs the reference; ties "
             "within 0.10 to the faster", "metrics": "as results/qwen_5312_vs_area (scripts/qwen_area_eval.py)",
             "bootstrap": "scripts/paired_bootstrap.py, prompts resampled, 2000 draws, seed 0, 95% percentile CI"}
    (R / "about.json").write_text(json.dumps(about, indent=1) + "\n")
    shutil.copy(args.prompts, R / "prompts.jsonl")
    out = sorted(str(x.relative_to(R)) for x in R.rglob("*") if x.is_file() and x.name != "SHA256SUMS")
    subprocess.check_call(["bash", "-c", f"cd {R} && sha256sum {' '.join('./' + f for f in out)} > SHA256SUMS"])
    for r in sc:
        print(f"{r['method']:16s} n={r['n_prompts']:2d} mean_dz={r['mean_dz']} pick={r['pickscore_delta']} "
              f"hps21={r['hpsv21_delta']} ir={r['ir_delta']} musiq_c={r['musiq_crop1024_delta']} "
              f"sec={r['median_seconds']}", flush=True)
    print("archived", len(out), "files in", R, flush=True)


if __name__ == "__main__":
    main()
