#!/usr/bin/env python3
"""Publish held E2/E9 evaluation only after strict generation coverage checks.

Inspection is read-only by default. --apply uses a nonblocking per-experiment
flock and publishes the complete held script with an atomic, no-clobber hard
link, then removes the held name, matching dispatcher v2. A crash between these
operations leaves an identifiable alias, never permission to publish a duplicate.
No scheduler calls, GPU work, polling, imports from mutable project code, or
generation-output changes occur. Freeze this standalone file before deployment.

--print-footer emits a shell footer for a generation task. Append it immediately
after the sampler command; its final exit status is always the sampler's status.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import struct
import sys


SCRATCH = Path("/scratch/09032/saini_2/sca")
BASELINES = Path("/scratch/09032/saini_2/constancy_rope/out_krea/main")
PROFILES = {
    "e2_6144": ("i998_eval_e2_6144.sh", "6144x6144", ("area", "sca_nb")),
    "e9_krea": ("j999_eval_e9_krea.sh", "4096x4096", ("area", "sca_krea", "dype", "sega_official")),
}


def image_stem(prompt_id, seed):
    key = str(prompt_id)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:80].strip(".") or "prompt"
    return f"{safe}_{hashlib.sha256(key.encode()).hexdigest()[:10]}_seed{seed}"


def manifest(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if (len(records) != 100 or len({r["id"] for r in records}) != 100
            or any(not isinstance(r["id"], str) or not isinstance(r["prompt"], str) for r in records)):
        raise ValueError("Evaluation requires exactly 100 unique string prompt IDs and prompt texts")
    return records


def sources(experiment, scratch, baselines):
    _, resolution, methods = PROFILES[experiment]
    return {m: ((baselines / m / resolution) if m in ("dype", "sega_official")
                else scratch / "out" / ("main" if experiment == "e2_6144" else "krea") / m / resolution)
            for m in methods}


def stamp(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def check_image(png, sidecar, record, method, resolution):
    before = stamp(png), stamp(sidecar)
    meta = json.loads(sidecar.read_text())
    config = meta.get("config", {})
    expected = {"method": method, "prompt_id": record["id"], "prompt": record["prompt"],
                "res": resolution, "seed": 0}
    if (meta.get("complete") is not True or type(config.get("seed")) is not int
            or any(config.get(key) != value for key, value in expected.items())):
        raise ValueError("incomplete or mismatched sidecar identity")
    expected_hash = meta.get("png_sha256", "")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("missing PNG SHA-256")
    height, width = map(int, resolution.split("x"))
    with png.open("rb") as stream:
        header = stream.read(24)
        if (len(header) != 24 or header[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                or struct.unpack(">II", header[16:24]) != (width, height)):
            raise ValueError("PNG dimensions or header mismatch")
        stream.seek(-12, os.SEEK_END)
        if stream.read(12) != b"\x00\x00\x00\x00IEND\xaeB`\x82":
            raise ValueError("PNG is truncated or has trailing bytes")
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_hash:
        raise ValueError("PNG SHA-256 mismatch")
    if before != (stamp(png), stamp(sidecar)):
        raise ValueError("artifact changed during validation")


def inventory(records, directories):
    missing, artifacts = {}, []
    for method, directory in directories.items():
        missing[method] = []
        for record in records:
            stem = image_stem(record["id"], 0)
            png, sidecar = directory / (stem + ".png"), directory / (stem + ".json")
            if not png.is_file() or not sidecar.is_file():
                missing[method].append(record["id"])
            else:
                artifacts.append((png, sidecar, record, method))
    return missing, artifacts


def presence(records, directories):
    """Incomplete producers never hold the publisher lock or hash image bytes."""
    missing, _ = inventory(records, directories)
    if any(missing.values()):
        return {"status": "not_ready", "missing": {m: len(ids) for m, ids in missing.items()}}
    return {"status": "ready"}


def readiness(records, directories, resolution):
    """Extra source-directory prompts are allowed; the selected manifest is exact."""
    missing, artifacts = inventory(records, directories)
    if any(missing.values()):
        return {"status": "not_ready", "missing": {m: len(ids) for m, ids in missing.items()}}
    invalid = []
    for png, sidecar, record, method in artifacts:
        try:
            check_image(png, sidecar, record, method, resolution)
        except (OSError, ValueError, TypeError, KeyError) as error:
            invalid.append(f"{method}/{record['id']}: {error}")
    if invalid:
        return {"status": "invalid_outputs", "invalid_count": len(invalid), "examples": invalid[:5]}
    return {"status": "ready", "prompts_per_method": len(records), "methods": list(directories),
            "resolution": resolution, "seed": 0, "verified_png_hashes": len(artifacts)}


def dispatched(queue, task):
    return [str(path.relative_to(queue)) for state in ("pending", "running", "done", "failed", "quarantine")
            for path in (queue / state).glob(task + "*")
            if path.name == task or path.name.startswith(task + ".")]


def promote(queue, task, validate, apply=False, precheck=None):
    """Validate under the publisher lock; never overwrite another task state."""
    lock = None
    try:
        existing = dispatched(queue, task)
        if existing:
            return {"status": "already_dispatched", "states": existing}
        if precheck is not None:
            result = precheck()
            if result["status"] != "ready":
                return result
        if apply:
            lock = (queue / (".promote_" + task + ".lock")).open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"status": "publisher_busy"}
        existing = dispatched(queue, task)
        if existing:
            return {"status": "already_dispatched", "states": existing}
        held = queue / "hold_cpu" / task
        if not held.is_file() or held.is_symlink():
            raise ValueError(f"Held evaluation script missing or not a regular file: {held}")
        task_stamp = stamp(held)
        result = validate()
        if result["status"] != "ready" or not apply:
            return result
        if stamp(held) != task_stamp or dispatched(queue, task):
            raise ValueError("Evaluation task changed state during readiness check")
        # os.link is atomic and fails if the pending name exists; no overwrite.
        os.link(held, queue / "pending" / task, follow_symlinks=False)
        try:
            held.unlink()
        except OSError as error:
            result["held_alias_warning"] = str(error)
        return dict(result, status="promoted", task=task)
    finally:
        if lock is not None:
            lock.close()


def footer(command):
    invocation = shlex.join(command)
    return ("sca_generation_rc=$?\n"
            'if [ "$sca_generation_rc" -eq 0 ]; then\n'
            f"  if ! {invocation}; then\n"
            "    printf '%s\\n' 'WARNING: evaluation promotion failed; generation status preserved' >&2\n"
            "  fi\nfi\n"
            'exit "$sca_generation_rc"\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=PROFILES)
    parser.add_argument("--scratch", type=Path, default=SCRATCH)
    parser.add_argument("--baselines", type=Path, default=BASELINES)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--print-footer", action="store_true")
    args = parser.parse_args()
    if args.print_footer:
        print(footer([sys.executable, str(Path(__file__).resolve()), args.experiment,
                      "--scratch", str(args.scratch), "--baselines", str(args.baselines), "--apply"]), end="")
        return 0
    task, resolution, _ = PROFILES[args.experiment]
    records = manifest(args.scratch / "eval100.jsonl")
    directories = sources(args.experiment, args.scratch, args.baselines)
    result = promote(args.scratch / "taskq", task,
                     lambda: readiness(records, directories, resolution), apply=args.apply,
                     precheck=lambda: presence(records, directories))
    print(json.dumps(dict(experiment=args.experiment, **result), sort_keys=True), flush=True)
    return 2 if result["status"] == "invalid_outputs" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Evaluation promotion failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error
