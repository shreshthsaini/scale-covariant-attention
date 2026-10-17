#!/usr/bin/env python3
"""Bounded CPU restoration parity using unchanged campaign scorer classes.

The fixed tolerances are declared before scoring. This tests environment
restoration on selected existing images, not GPU parity or benchmark accuracy.
"""
from __future__ import annotations

import argparse
import base64
import csv
import gc
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("aesthetic", "hpsv2", "hpsv21", "clipiqa", "arniqa", "is", "ir")
TOLERANCES = {name: {"atol": 1e-5, "rtol": 1e-5} for name in METRICS}
TOLERANCES["is"] = {"atol": 1e-4, "rtol": 1e-5}
POLICY = "sca_extra_restoration_cpu_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def manifest(args):
    prompts = {r["id"]: r["prompt"] for r in map(json.loads, args.prompts.read_text().splitlines())}
    with args.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = []
    for row in rows:
        if row["method"] != "area":
            continue
        path = Path(row["path"])
        sidecar = path.with_suffix(".json")
        prompt = prompts[row["prompt_id"]]
        selected.append({k: row[k] for k in ("method", "res", "prompt_id", "seed", "path")})
        selected[-1].update(prompt=prompt, image_sha256=sha256(path),
                            sidecar=str(sidecar), sidecar_sha256=sha256(sidecar),
                            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise ValueError("Insufficient fixed input rows")
    write(args.out, {"policy": POLICY, "tolerances": TOLERANCES, "inputs": selected,
                     "selection": "first area rows in archived final4k_full_extra CSV",
                     "source_csv_sha256": sha256(args.csv), "prompts_sha256": sha256(args.prompts)})


def check_inputs(records):
    for row in records:
        for key, filename in (("image_sha256", "path"), ("sidecar_sha256", "sidecar")):
            if sha256(row[filename]) != row[key]:
                raise ValueError(f"Input identity changed: {row[filename]}")
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError("Prompt identity changed")


def score(args):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ.update(HF_HOME=str(args.hf_home), HF_HUB_CACHE=str(args.hf_home / "hub"),
                      HUGGINGFACE_HUB_CACHE=str(args.hf_home / "hub"), TORCH_HOME=str(args.torch_home),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_XET="1",
                      TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
    fixed = json.loads(args.manifest.read_text())
    if fixed["policy"] != POLICY or fixed["tolerances"] != TOLERANCES:
        raise ValueError("Unrecognized predeclared policy or tolerances")
    check_inputs(fixed["inputs"])
    spec = importlib.util.spec_from_file_location("sca_extra_parity_scorers", ROOT / "scripts/eval_extra.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.AESTHETIC_WEIGHTS = str(args.torch_home / "hub/checkpoints/laion_aesthetic/sac+logos+ava1-l14-linearMSE.pth")
    import torch
    from PIL import Image
    from huggingface_hub import constants
    if Path(constants.HF_HUB_CACHE).resolve() != (args.hf_home / "hub").resolve():
        raise ValueError("HF cache isolation failed")
    if Path(torch.hub.get_dir()).resolve() != (args.torch_home / "hub").resolve():
        raise ValueError("Torch cache isolation failed")
    assets = args.hf_home.parent / "cache_copy_receipt.json"
    if not assets.is_file():
        raise ValueError("The verified asset-copy receipt is required")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    packages = {dist.metadata["Name"]: dist.version for dist in metadata.distributions()}
    report = {"policy": POLICY, "manifest_sha256": sha256(args.manifest), "inputs": fixed["inputs"],
              "tolerances": TOLERANCES, "device": "cpu", "threads": args.threads,
              "executable": sys.executable, "environment": packages, "results": {}, "errors": {},
              "asset_receipt": {"path": str(assets), "sha256": sha256(assets)},
              "cache_configuration": {"hf_hub_cache": constants.HF_HUB_CACHE,
                                      "torch_hub": torch.hub.get_dir(), "network": "offline"},
              "source_hashes": {"scripts/eval_extra.py": sha256(ROOT / "scripts/eval_extra.py"),
                                "repro/check_metric_extra_parity.py": sha256(__file__)},
              "limitations": ["Two existing 4K images, CPU float32 inference; GPU mixed precision untested.",
                              "Inception logits parity is tested; two images cannot validate IS estimates."],
              "complete": False}
    for name in args.metrics.split(","):
        started = time.monotonic()
        scorer = None
        try:
            scorer = module.build_scorer(name, argparse.Namespace(device="cpu", iqa_long_side=1024))
            outputs = []
            for record in fixed["inputs"]:
                with Image.open(record["path"]) as image:
                    image = image.convert("RGB")
                    if f"{image.height}x{image.width}" != record["res"]:
                        raise ValueError("Recorded image dimensions disagree")
                    prepared = scorer.prepare(image, record)
                value = scorer.run([prepared], [record])[0]
                if name == "is":
                    import numpy as np
                    values = np.frombuffer(base64.b64decode(value["is_logits_b64"]), dtype=np.float32).tolist()
                else:
                    values = [value[name]]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("Nonfinite score")
                outputs.append(values)
            report["results"][name] = {"values": outputs, "settings": scorer.settings,
                                         "seconds": time.monotonic() - started}
        except Exception:
            report["errors"][name] = traceback.format_exc()
        finally:
            del scorer
            gc.collect()
        check_inputs(fixed["inputs"])
        write(args.out, report)
        print(json.dumps({"metric": name, "success": name in report["results"],
                          "seconds": time.monotonic() - started}), flush=True)
    report["complete"] = not report["errors"] and set(report["results"]) == set(args.metrics.split(","))
    write(args.out, report)
    return 0 if report["complete"] else 1


def compare_values(reference, candidate, metric):
    if not reference or len(reference) != len(candidate):
        raise ValueError("Score row coverage differs")
    atol, rtol = TOLERANCES[metric]["atol"], TOLERANCES[metric]["rtol"]
    differences, accepted = [], True
    for left, right in zip(reference, candidate):
        if not left or len(left) != len(right):
            raise ValueError("Score vector dimensions differ")
        for a, b in zip(left, right):
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("Nonfinite parity input")
            delta = abs(a - b)
            differences.append(delta)
            accepted &= delta <= atol + rtol * abs(a)
    return {"pass": bool(accepted), "max_abs_error": max(differences),
            "values_compared": len(differences), "atol": atol, "rtol": rtol}


def compare(args):
    reference, candidate = (json.loads(path.read_text()) for path in (args.reference, args.candidate))
    for field in ("policy", "manifest_sha256", "inputs", "tolerances", "device", "source_hashes",
                  "asset_receipt", "cache_configuration"):
        if reference[field] != candidate[field]:
            raise ValueError(f"Cannot compare changed {field}")
    requested = args.metrics.split(",")
    checks = {}
    for metric in requested:
        if metric not in reference["results"] or metric not in candidate["results"]:
            checks[metric] = {"pass": False, "reason": "Missing successful score output"}
        else:
            if reference["results"][metric]["settings"] != candidate["results"][metric]["settings"]:
                raise ValueError(f"Cannot compare changed {metric} scorer settings")
            checks[metric] = compare_values(reference["results"][metric]["values"],
                                            candidate["results"][metric]["values"], metric)
    passed = all(check["pass"] for check in checks.values())
    write(args.out, {"policy": POLICY, "passed": passed, "checks": checks,
                     "reference": {"path": str(args.reference), "sha256": sha256(args.reference)},
                     "candidate": {"path": str(args.candidate), "sha256": sha256(args.candidate)},
                     "limitations": candidate["limitations"]})
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    select = subs.add_parser("manifest")
    select.add_argument("--csv", type=Path, default=ROOT / "results/evidence/eval/final4k_full_extra/extra_per_image.csv")
    select.add_argument("--prompts", type=Path, default=ROOT / "results/evidence/prompts/aesthetic4k_eval.jsonl")
    select.add_argument("--count", type=int, default=2)
    select.add_argument("--out", type=Path, required=True)
    select.set_defaults(func=manifest)
    run = subs.add_parser("score")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--metrics", required=True)
    run.add_argument("--hf-home", type=Path, required=True)
    run.add_argument("--torch-home", type=Path, required=True)
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--out", type=Path, required=True)
    run.set_defaults(func=score)
    check = subs.add_parser("compare")
    check.add_argument("--reference", type=Path, required=True)
    check.add_argument("--candidate", type=Path, required=True)
    check.add_argument("--metrics", required=True)
    check.add_argument("--out", type=Path, required=True)
    check.set_defaults(func=compare)
    args = parser.parse_args()
    if hasattr(args, "threads") and not 1 <= args.threads <= 16:
        parser.error("threads must be between 1 and 16")
    if hasattr(args, "count") and not 1 <= args.count <= 8:
        parser.error("count must be between 1 and 8")
    if hasattr(args, "metrics"):
        selected = args.metrics.split(",")
        if any(name not in METRICS for name in selected) or len(selected) != len(set(selected)):
            parser.error("metrics must be unique known metric names")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
