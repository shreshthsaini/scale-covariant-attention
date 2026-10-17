#!/usr/bin/env python3
"""Restore HPSv2's omitted package-data file from hash-identical OpenAI CLIP."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys

EXPECTED = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore(source, target):
    source, target = Path(source), Path(target)
    if sha256(source) != EXPECTED:
        raise ValueError("CLIP tokenizer hash differs from archived HPS tokenizer")
    if target.exists():
        if sha256(target) != EXPECTED:
            raise ValueError("Refusing to overwrite a different HPS tokenizer")
        action = "already_present_verified"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as out, source.open("rb") as inp:
            shutil.copyfileobj(inp, out)
        if sha256(target) != EXPECTED:
            raise ValueError("Copied HPS tokenizer failed verification; preserve it")
        action = "copied_missing_package_data"
    return {"action": action, "source": str(source), "target": str(target),
            "bytes": target.stat().st_size, "sha256": EXPECTED,
            "algorithm_change": False, "network_requests": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("Refusing to overwrite an existing receipt")
    if metadata.version("hpsv2") != "1.2.0" or metadata.version("openai-clip") != "1.0.1":
        raise ValueError("This repair requires the hash-locked HPSv2/CLIP versions")
    clip = metadata.distribution("openai-clip")
    hps = metadata.distribution("hpsv2")
    source = clip.locate_file("clip/bpe_simple_vocab_16e6.txt.gz")
    target = hps.locate_file("hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz")
    if not Path(target).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("Refusing to modify a package outside the current environment")
    result = restore(source, target)
    result.update(environment=sys.prefix, hpsv2="1.2.0", openai_clip="1.0.1",
                  explanation="HPSv2 PyPI wheel omits its BPE data; CLIP's locked asset is byte-identical")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
