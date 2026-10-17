#!/usr/bin/env python3
"""Bounded same-input CPU restoration parity, using the frozen base adapters.

CLIP, PickScore and Inception use the original 4K images. MUSIQ uses one
deterministic 1024 crop per image. This tests restoration on bounded inputs,
not native-4K MUSIQ end-to-end parity, full-campaign reruns or CUDA parity.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "repro/evaluator_snapshot"
# Fixed before running either environment; never adapt these to observed errors.
ATOL, RTOL = 1e-5, 1e-6
PROTOCOL = {"version": 1, "device": "cpu", "threads": 4,
            "atol": ATOL, "rtol": RTOL, "crop_seed": 0, "crops": 4,
            "musiq": "first frozen deterministic 1024 crop, no resize",
            "clip_pickscore": "original 4K image, frozen processor",
            "features": "original 4K whole image plus four frozen 1024 crops"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_hashes():
    return {str(path.relative_to(ROOT)): sha(path) for path in
            [Path(__file__), SNAPSHOT / "crope/metrics/models.py",
             SNAPSHOT / "crope/metrics/image.py"]}


def prepare(args):
    from PIL import Image
    sys.path.insert(0, str(SNAPSHOT))
    from crope.metrics.image import crop_boxes
    rows = [json.loads(line) for line in Path(args.prompts).read_text().splitlines()]
    rows = [row for row in rows if all(len(list((Path(args.images) / method / "4096x4096").glob(
        row["id"] + "_*_seed0.png"))) == 1 for method in ("area", "sca"))][:2]
    if len(rows) != 2:
        raise ValueError("Need two declared prompt rows with paired existing images")
    records = []
    for row in rows:
        for method in ("area", "sca"):
            matches = sorted((Path(args.images) / method / "4096x4096").glob(row["id"] + "_*_seed0.png"))
            if len(matches) != 1:
                raise ValueError(f"Expected one image for {method}/{row['id']}: {matches}")
            path = matches[0]
            with Image.open(path) as source:
                image = source.convert("RGB")
                if image.size != (4096, 4096):
                    raise ValueError(f"Unexpected image dimensions: {path}")
                boxes = crop_boxes(image.size, row["id"], seed=0, count=4)
                cropped = Path(args.out).parent / "musiq_inputs" / f"{method}_{row['id']}.png"
                cropped.parent.mkdir(parents=True, exist_ok=True)
                if cropped.exists():
                    with Image.open(cropped) as existing:
                        if existing.convert("RGB").tobytes() != image.crop(boxes[0]).tobytes():
                            raise ValueError(f"Existing crop differs: {cropped}")
                else:
                    image.crop(boxes[0]).save(cropped)
            sidecar = path.with_suffix(".json")
            records.append({"prompt_id": row["id"], "prompt": row["prompt"],
                            "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                            "method": method, "path": str(path), "image_sha256": sha(path),
                            "sidecar_sha256": sha(sidecar), "size": [4096, 4096],
                            "crop_boxes": boxes, "musiq_path": str(cropped),
                            "musiq_sha256": sha(cropped)})
    write_json(args.out, {"protocol": PROTOCOL, "records": records,
                          "selection": "first two prompt rows with exactly one existing image per method",
                          "prompts_sha256": sha(args.prompts), "sources": source_hashes()})


def verify_inputs(manifest):
    if manifest["protocol"] != PROTOCOL or manifest["sources"] != source_hashes():
        raise ValueError("Protocol or code changed since manifest preparation")
    for record in manifest["records"]:
        for path, identity in ((record["path"], "image_sha256"),
                               (record["musiq_path"], "musiq_sha256"),
                               (str(Path(record["path"]).with_suffix(".json")), "sidecar_sha256")):
            if sha(path) != record[identity]:
                raise ValueError(f"Input changed: {path}")


def score(args):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ.update(HF_HOME=args.hf_home, TORCH_HOME=args.torch_home,
                      HF_HUB_CACHE=str(Path(args.hf_home) / "hub"),
                      HUGGINGFACE_HUB_CACHE=str(Path(args.hf_home) / "hub"),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(SNAPSHOT))
    from crope.metrics.models import load_scorer, InceptionFeatures
    manifest = json.loads(Path(args.manifest).read_text())
    verify_inputs(manifest)
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    inventory = {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()}
    write_json(target / "environment.json", {"executable": sys.executable, "python": sys.version,
               "platform": platform.platform(), "packages": inventory, "sources": source_hashes(),
               "cache_environment": {name: os.environ.get(name) for name in
                   ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "TRANSFORMERS_CACHE")}})
    for metric in args.metrics.split(","):
        result_path = target / f"{metric}.json"
        if result_path.exists():
            raise ValueError(f"Refusing to overwrite previous result: {result_path}")
        started = time.monotonic()
        print(f"START {metric}", flush=True)
        model = InceptionFeatures("cpu") if metric == "features" else load_scorer(metric, "cpu")
        values, arrays = [], {}
        for index, original in enumerate(manifest["records"]):
            record = dict(original)
            if metric == "features":
                for patch in (False, True):
                    key = f"record{index}_{'patch' if patch else 'whole'}"
                    arrays[key] = model.extract(record, patch=patch, crop_seed=0, crops=4)
                    if not np.isfinite(arrays[key]).all():
                        raise ValueError(f"Nonfinite features: {key}")
            else:
                if metric == "musiq":
                    record["path"] = record["musiq_path"]
                value = model.score(record, crop_seed=0, crops=1)
                if not np.isfinite(list(value.values())).all():
                    raise ValueError(f"Nonfinite score: {index}/{metric}")
                values.append(value)
        if arrays:
            np.savez_compressed(target / "features.npz", **arrays)
        verify_inputs(manifest)
        write_json(result_path, {"metric": metric, "manifest_sha256": sha(args.manifest),
                   "environment_sha256": sha(target / "environment.json"),
                   "provenance": model.provenance, "values": values,
                   "feature_sha256": sha(target / "features.npz") if arrays else None,
                   "seconds": time.monotonic() - started, "cuda_initialized": torch.cuda.is_initialized()})
        del model
        gc.collect()
        print(f"DONE {metric}: {time.monotonic() - started:.3f}s", flush=True)


def compare(args):
    import numpy as np
    comparisons, passed = {}, True
    for metric in ("clip", "pickscore", "musiq", "features"):
        left = json.loads((Path(args.inherited) / f"{metric}.json").read_text())
        right = json.loads((Path(args.restored) / f"{metric}.json").read_text())
        if left["manifest_sha256"] != right["manifest_sha256"]:
            raise ValueError("Different input manifests")
        if left["provenance"] != right["provenance"]:
            raise ValueError(f"Different model provenance for {metric}")
        if metric == "features":
            for directory, report in ((args.inherited, left), (args.restored, right)):
                if sha(Path(directory) / "features.npz") != report["feature_sha256"]:
                    raise ValueError("Feature artifact changed")
            with np.load(Path(args.inherited) / "features.npz", allow_pickle=False) as a, \
                    np.load(Path(args.restored) / "features.npz", allow_pickle=False) as b:
                if a.files != b.files:
                    raise ValueError("Feature keys differ")
                first = np.concatenate([a[key].ravel() for key in a.files])
                second = np.concatenate([b[key].ravel() for key in b.files])
        else:
            first = np.asarray([list(row.values()) for row in left["values"]])
            second = np.asarray([list(row.values()) for row in right["values"]])
        if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError(f"Invalid numerical arrays: {metric}")
        good = bool(np.allclose(first, second, atol=ATOL, rtol=RTOL))
        comparisons[metric] = {"passed": good, "count": first.size,
                               "max_absolute_error": float(np.max(np.abs(first - second))),
                               "inherited_sha256": sha(Path(args.inherited) / f"{metric}.json"),
                               "restored_sha256": sha(Path(args.restored) / f"{metric}.json")}
        passed = passed and good
    write_json(args.out, {"passed": passed, "protocol": PROTOCOL, "comparisons": comparisons,
               "manifest_sha256": left["manifest_sha256"],
               "scope": __doc__, "inherited": args.inherited, "restored": args.restored})
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--images", required=True)
    prepare_parser.add_argument("--prompts", required=True)
    prepare_parser.add_argument("--out", required=True)
    score_parser = commands.add_parser("score")
    for name in ("manifest", "hf-home", "torch-home", "out"):
        score_parser.add_argument("--" + name, required=True)
    score_parser.add_argument("--metrics", default="clip,pickscore,musiq,features",
                              choices=None)
    compare_parser = commands.add_parser("compare")
    for name in ("inherited", "restored", "out"):
        compare_parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "score": score, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
