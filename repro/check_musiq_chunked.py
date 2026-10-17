#!/usr/bin/env python3
"""Bounded pretrained MUSIQ equivalence check; never loads a full 8K image."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import time

import torch
import pyiqa

from musiq_chunked import install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--query-chunk", type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = pyiqa.create_metric("musiq", device="cpu").eval()
    torch.manual_seed(672)
    inputs = [torch.rand(1, 3, height, width) for height, width in [(512, 512), (768, 1024), (1024, 1024)]]
    original = []
    with torch.inference_mode():
        for image in inputs:
            start = time.perf_counter()
            score = float(model(image).item())
            original.append((score, time.perf_counter() - start))
    provenance = install(model, args.query_chunk)
    rows = []
    with torch.inference_mode():
        for image, (score, seconds) in zip(inputs, original):
            start = time.perf_counter()
            observed = float(model(image).item())
            elapsed = time.perf_counter() - start
            assert abs(observed - score) <= 1e-4, (score, observed)
            rows.append({"shape": list(image.shape), "original": score, "adapted": observed,
                         "abs_difference": abs(observed - score), "original_seconds": seconds,
                         "adapted_seconds": elapsed})
    report = {"provenance": provenance, "device": "cpu", "threads": torch.get_num_threads(),
              "torch_version": torch.__version__, "pyiqa_version": importlib.metadata.version("pyiqa"),
              "absolute_score_tolerance": 1e-4,
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "results": rows}
    checkpoint = Path(torch.hub.get_dir()) / "pyiqa/musiq_koniq_ckpt-e95806b9.pth"
    if checkpoint.is_file():
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        report["checkpoint"] = {"filename": checkpoint.name, "sha256": digest.hexdigest(),
                                "bytes": checkpoint.stat().st_size}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
