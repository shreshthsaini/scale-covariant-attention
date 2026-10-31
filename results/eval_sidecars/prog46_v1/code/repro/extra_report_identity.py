"""Durable input identities for newly computed split extra-metric reports."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_FILES = ("extra_per_image.csv", "extra_summary.csv", "extra_paired_deltas.csv", "extra_metrics.json")
IDENTITY_FILE = "extra_input_identities.json"
POLICY = "sha256_before_after_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row):
    return tuple(str(row[name]) for name in ("method", "res", "prompt_id", "seed"))


def identities(records):
    result, seen = [], set()
    for row in sorted(records, key=key):
        identity = key(row)
        if identity in seen:
            raise ValueError(f"Duplicate input identity: {identity}")
        seen.add(identity)
        image, sidecar = Path(row["path"]), Path(row["sidecar"])
        before = (image.stat().st_size, image.stat().st_mtime_ns, sidecar.stat().st_mtime_ns)
        item = dict(zip(("method", "res", "prompt_id", "seed"), identity))
        item.update(path=str(image.resolve()), image_sha256=sha256(image), image_bytes=before[0],
                    prompt_sha256=hashlib.sha256(row["prompt"].encode()).hexdigest(),
                    sidecar_sha256=sha256(sidecar), config_hash=row["config_hash"])
        after = (image.stat().st_size, image.stat().st_mtime_ns, sidecar.stat().st_mtime_ns)
        if before != after:
            raise ValueError(f"Input changed while hashing: {image}")
        result.append(item)
    return result


def environment():
    result = {}
    for package in ("torch", "torchvision", "transformers", "huggingface-hub", "image-reward",
                    "hpsv2", "pyiqa", "torch-fidelity", "open-clip-torch", "timm", "numpy", "scipy"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def source_hashes():
    return {str(path.relative_to(ROOT)): sha256(path) for path in (
        ROOT / "scripts/eval_extra.py", ROOT / "repro/eval_extra_portable.py", Path(__file__),
        ROOT / "repro/evaluator_snapshot/crope/metrics/io.py",
        ROOT / "repro/evaluator_snapshot/crope/metrics/statistics.py")}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)
