#!/usr/bin/env python3
"""Build an evaluation root of symlinks over the prompt ids that every method has finished.

Usage: link_common.py --root R --res HxW --seed S --prompts P.jsonl [--exclude dev4.jsonl] NAME=DIR ...

For each prompt id in --prompts (minus --exclude), the PNG and JSON sidecar named
image_stem(id, seed) (constancy_rope / sca naming) must exist and be complete in every DIR;
those ids are linked into R/NAME/<res>/. R is recreated from scratch. Stdlib only.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def image_stem(prompt_id, seed):  # identical to sca/runtime.py and crope/utils.py
    key = str(prompt_id)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:80].strip(".") or "prompt"
    return f"{safe}_{hashlib.sha256(key.encode()).hexdigest()[:10]}_seed{int(seed)}"


def ids(path):
    return [str(json.loads(line)["id"]) for line in open(path) if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--res", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompts", required=True)
    p.add_argument("--exclude", default="/scratch/09032/saini_2/sca/dev4.jsonl")
    p.add_argument("--min-count", type=int, default=1, help="Exit nonzero if fewer common ids")
    p.add_argument("methods", nargs="+", metavar="NAME=DIR")
    a = p.parse_args()
    methods = [m.split("=", 1) for m in a.methods]
    skip = set(ids(a.exclude)) if a.exclude else set()
    if a.root.exists():
        shutil.rmtree(a.root)
    common, missing = [], {name: 0 for name, _ in methods}
    for pid in ids(a.prompts):
        if pid in skip:
            continue
        stem, ok = image_stem(pid, a.seed), True
        for name, d in methods:
            png, js = Path(d) / f"{stem}.png", Path(d) / f"{stem}.json"
            if not (png.is_file() and js.is_file() and json.loads(js.read_text()).get("complete", True) is True):
                missing[name] += 1
                ok = False
        if ok:
            common.append(stem)
    for name, d in methods:
        out = a.root / name / a.res
        out.mkdir(parents=True, exist_ok=True)
        for stem in common:
            for ext in (".png", ".json"):
                os.symlink(Path(d) / f"{stem}{ext}", out / f"{stem}{ext}")
    print(f"common ids: {len(common)}; missing per method: {missing}", flush=True)
    if len(common) < a.min_count:
        raise SystemExit(f"only {len(common)} common ids (< {a.min_count})")


if __name__ == "__main__":
    main()
