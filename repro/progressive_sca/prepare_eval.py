#!/usr/bin/env python3
"""Progressive-SCA evaluation on the 46 common prompts at 4096^2, no jobs.

prepare: validate every input pair and build a new isolated workspace (sample symlinks, prompts, references,
         evaluator code copied from the baseline46 workspace with hash checks, inputs.json, score scripts).
verify:  coverage and finiteness barrier on the GPU score CSVs; snapshot them as scored_* (immutable).
timing:  per-method seconds and progressive per-stage timing from the sidecars (timing.csv, timing.json).

Rows: hrdit_dev_author (HRDiT as released, existing outputs), sca_prog (HRDiT schedule with the SCA operator),
sca_s42 (single-pass SCA at roughly matched wall clock), sca_nb (single-pass SCA, 28 steps, the reference
method of the baseline46 evaluation) and ultraimage.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import statistics

BASE46 = Path("/scratch/09032/saini_2/sca/eval/baseline_expansion46_v1")
PROG = Path("/scratch/09032/saini_2/sca/out/prog/sca_prog/4096x4096")
S42 = Path("/scratch/09032/saini_2/sca/out/budget/sca_s42/4096x4096")
REUSED = ("hrdit_dev_author", "sca_nb", "ultraimage")
METHODS = ("sca_prog", "hrdit_dev_author", "sca_s42", "sca_nb", "ultraimage")
PY = "/work/09032/saini_2/vista/library/uv-envs/sca/bin/python"
EXTRA_PY = "/work/09032/saini_2/vista/library/uv-envs/sca-metrics/bin/python"
BASE_METRICS = ("clip", "pickscore", "musiq", "musiq_crop1024")
EXTRA_METRICS = ("ir", "aesthetic", "hpsv2", "hpsv21", "clipiqa", "arniqa")
BASE_REPORTS = ("metrics.json", "per_image.csv", "summary.csv", "paired_deltas.csv")
PAIRS = (("sca_prog", "hrdit_dev_author"), ("sca_prog", "sca_nb"), ("sca_prog", "sca_s42"),
         ("hrdit_dev_author", "sca_s42"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stem(prompt_id, seed=0):
    import re
    key = str(prompt_id)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:80].strip(".") or "prompt"
    return f"{safe}_{hashlib.sha256(key.encode()).hexdigest()[:10]}_seed{int(seed)}"


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def check_sample(image, method, row):
    """Same identity rules as repro/baseline_evaluation_20261017/prepare.py:check_sample."""
    image = Path(image)
    sidecar = image.with_suffix(".json")
    if not image.is_file() or not sidecar.is_file():
        raise ValueError(f"Missing pair: {image}")
    record = json.loads(sidecar.read_text())
    cfg = record["config"]
    expected = dict(method=method, prompt_id=str(row["id"]), prompt=row["prompt"], res="4096x4096", seed=0)
    if record.get("complete") is not True or any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Incomplete or wrong identity: {sidecar}")
    digest = sha(image)
    if record.get("png_sha256") != digest:
        raise ValueError(f"PNG checksum mismatch: {image}")
    from PIL import Image
    with Image.open(image) as opened:
        if opened.size != (4096, 4096):
            raise ValueError(f"Wrong pixel dimensions: {image}")
        opened.verify()
    return dict(method=method, prompt_id=str(row["id"]), image=str(image.resolve()), image_sha256=digest,
                sidecar=str(sidecar.resolve()), sidecar_sha256=sha(sidecar),
                config_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest())


def gather(base46=BASE46, prog=PROG, s42=S42):
    receipt = json.loads((base46 / "inputs.json").read_text())
    rows = read_jsonl(base46 / "prompts.jsonl")
    if sha(base46 / "prompts.jsonl") != receipt["prompts_sha256"] or len(rows) != 46:
        raise ValueError("baseline46 prompt manifest changed")
    selected = []
    old = {(x["method"], x["prompt_id"]): x for x in receipt["inputs"]}
    for method in REUSED:
        for row in rows:
            item = old[method, str(row["id"])]
            fresh = check_sample(item["image"], method, row)
            if fresh["image_sha256"] != item["image_sha256"] or fresh["sidecar_sha256"] != item["sidecar_sha256"]:
                raise ValueError(f"baseline46 input changed: {item['image']}")
            selected.append(fresh)
    for method, root in (("sca_prog", prog), ("sca_s42", s42)):
        for row in rows:
            selected.append(check_sample(root / (stem(row["id"]) + ".png"), method, row))
    return receipt, rows, selected


def command(python, script, args):
    return shlex.join(["uv", "run", "--no-project", "--python", python, "python", str(script), *map(str, args)])


def prepare(workspace, base46=BASE46, prog=PROG, s42=S42):
    workspace = Path(workspace).resolve()
    if workspace.exists():
        raise ValueError("Use a new evaluation workspace; existing results are preserved")
    receipt, rows, selected = gather(base46, prog, s42)
    workspace.mkdir(parents=True)
    code = workspace / "code"
    for name, digest in receipt["sources"].items():
        source = base46 / "code" / name
        if sha(source) != digest:
            raise ValueError(f"Frozen evaluator source changed: {name}")
        (code / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, code / name)
    view = workspace / "samples"
    for item in selected:
        folder = view / item["method"]
        folder.mkdir(parents=True, exist_ok=True)
        for kind in ("image", "sidecar"):
            source = Path(item[kind])
            (folder / source.name).symlink_to(source)
    for name in ("prompts.jsonl", "references.jsonl"):
        shutil.copyfile(base46 / name, workspace / name)
    out = dict(size=len(rows), methods=list(METHODS), inputs=selected, sources=receipt["sources"],
               prompts_sha256=sha(workspace / "prompts.jsonl"), references_sha256=sha(workspace / "references.jsonl"),
               reference_inputs=receipt["reference_inputs"], derived_from=str(base46 / "inputs.json"),
               derived_from_sha256=sha(base46 / "inputs.json"), status="inputs_complete_scoring_not_run")
    (workspace / "inputs.json").write_text(json.dumps(out, indent=2) + "\n")
    common = ["--root", view, "--prompts", workspace / "prompts.jsonl", "--reference-method", "sca_nb",
              "--res", "4096x4096", "--threads", "4"]
    base = common + ["--references", workspace / "references.jsonl", "--reference-clusters", "images",
                     "--out", workspace / "base"]
    extra = common + ["--crope-root", code / "repro/evaluator_snapshot", "--out", workspace / "extra"]
    me = command(PY, Path(__file__).resolve(), ["verify", "--workspace", workspace])
    header = ("#!/bin/bash\nset -euo pipefail\nexport HF_HOME=/scratch/09032/saini_2/tmp/hf "
              "TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
              "export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4\n")
    gpu = [command(PY, code / "repro/evaluate_corrected.py",
                   base + ["--metrics", "clip,pickscore,musiq,fid,kid,patch_fid,patch_kid", "--device", "cuda",
                           "--bootstrap", "0"]),
           command(EXTRA_PY, code / "repro/eval_extra_portable.py",
                   extra + ["--metrics", "ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa", "--device", "cuda",
                            "--bootstrap", "0"]),
           me]
    for a, b in PAIRS:
        gpu.append(command(PY, code / "scripts/paired_bootstrap.py",
                           ["--method", a, "--reference", b, "--draws", "2000", "--out",
                            workspace / f"{a}_vs_{b}_paired.csv", workspace / "base/scored_per_image.csv",
                            workspace / "extra/extra_per_image.csv"]))
    gpu.append(command(PY, Path(__file__).resolve(), ["timing", "--workspace", workspace]))
    # cpu_cache_only.py hard-wires the eight-method barrier, so the CPU stage calls the same estimator directly
    # after this workspace's own barrier; Inception features come from the GPU feature cache in base/features.
    cpu = [me, command(PY, code / "repro/evaluate_corrected.py",
                       base + ["--metrics", "fid,kid,patch_fid,patch_kid", "--device", "cpu", "--bootstrap", "1000"]),
           me]
    for name, lines in (("score_gpu.sh", gpu), ("intervals_cpu.sh", cpu)):
        (workspace / name).write_text(header + "\n".join(lines) + "\n")
    return out


def check_csv(path, expected, metrics):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    keys = [(r["method"], r["prompt_id"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"Wrong paired coverage: {path}")
    for row in rows:
        if row["res"] != "4096x4096" or str(row["seed"]) != "0":
            raise ValueError("Wrong resolution/seed")
        for metric in metrics:
            if not math.isfinite(float(row[metric])):
                raise ValueError(f"Missing/nonfinite {metric}: {row['method']}/{row['prompt_id']}")


def verify(workspace):
    workspace = Path(workspace)
    receipt = json.loads((workspace / "inputs.json").read_text())
    for row in receipt["inputs"]:
        for kind in ("image", "sidecar"):
            if sha(row[kind]) != row[kind + "_sha256"]:
                raise ValueError(f"Input changed: {row[kind]}")
    expected = {(x["method"], x["prompt_id"]) for x in receipt["inputs"]}
    if len(expected) != len(METHODS) * 46:
        raise ValueError("Receipt lacks exact paired coverage")
    scored = workspace / "base/scored_per_image.csv"
    check_csv(scored if scored.exists() else workspace / "base/per_image.csv", expected, BASE_METRICS)
    check_csv(workspace / "extra/extra_per_image.csv", expected, EXTRA_METRICS)
    for item in ("base/metrics.json", "extra/extra_metrics.json"):
        report = json.loads((workspace / item).read_text())
        if not report.get("complete") or report.get("errors"):
            raise ValueError(f"Scoring report incomplete: {item}")
    for name in BASE_REPORTS:  # immutable copy before any CPU interval run rewrites base/
        source = workspace / "base" / name
        target = source.with_name("scored_" + name)
        if not target.exists():
            target.write_bytes(source.read_bytes())
    (workspace / "GPU_SCORES_COMPLETE.json").write_text(json.dumps(
        {"inputs_sha256": sha(workspace / "inputs.json"),
         "reports": {n: sha(workspace / n) for n in ["base/scored_" + x for x in BASE_REPORTS] +
                     ["extra/extra_metrics.json", "extra/extra_per_image.csv"]}}, indent=2) + "\n")


def timing(workspace):
    workspace = Path(workspace)
    receipt = json.loads((workspace / "inputs.json").read_text())
    per = {}
    for row in receipt["inputs"]:
        per.setdefault(row["method"], []).append(json.loads(Path(row["sidecar"]).read_text()))
    summary, lines = {}, []
    for method in METHODS:
        records = per[method]
        sec = [r["seconds"] for r in records]
        item = dict(method=method, n=len(sec), mean_seconds=statistics.fmean(sec), median_seconds=statistics.median(sec),
                    steps=records[0]["config"].get("steps"))
        if method == "sca_prog":
            for key in ("stage_seconds", "transition_seconds"):
                cols = list(zip(*[r[key] for r in records]))
                item[key + "_mean"] = [statistics.fmean(c) for c in cols]
            item["text_seconds_mean"] = statistics.fmean(r["text_seconds"] for r in records)
            item["decode_seconds_mean"] = statistics.fmean(r["decode_seconds"] for r in records)
            item["peak_memory_gib_max"] = max(r["peak_memory_bytes"] or 0 for r in records) / 2 ** 30
        summary[method] = item
        lines.append([method, item["n"], f"{item['mean_seconds']:.3f}", f"{item['median_seconds']:.3f}", item["steps"]])
    summary["caveat"] = ("Seconds come from each method's own campaign sidecars on GH200 nodes. HRDiT timing "
                         "includes per-step finite checks and fp16; sca_prog seconds are CUDA synchronized and "
                         "include text encoding, all stages, transitions and the final decode.")
    (workspace / "timing.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (workspace / "timing.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n", "mean_seconds", "median_seconds", "steps"])
        w.writerows(lines)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("prepare", "verify", "timing", "gather"))
    p.add_argument("--workspace", type=Path)
    a = p.parse_args()
    if a.action == "gather":
        _, _, s = gather()
        print(json.dumps(dict(status="all inputs valid", samples=len(s))))
    elif a.workspace is None:
        p.error("--workspace required")
    elif a.action == "prepare":
        r = prepare(a.workspace)
        print(json.dumps(dict(status=r["status"], samples=len(r["inputs"]), workspace=str(a.workspace))))
    elif a.action == "verify":
        verify(a.workspace)
        print("GPU score barrier passed")
    else:
        timing(a.workspace)
