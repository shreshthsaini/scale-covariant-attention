#!/usr/bin/env python3
"""One-shot, fail-closed release of the two reviewed SCA runtime repairs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OWNING_JOB = "1075814"
RECOVERY = "recovered/runtime_repairs_1076947"
EXPECTED = {
    "a_000_repro_gpu_smoke.sh": "a_000_repro_gpu_smoke_gcc_1076947.sh",
    "a_y003_latency_6144x6144_full.sh": "a_y003_latency_6144x6144_full_gridfix_1076947.sh",
}
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED", "PREEMPTED"}


class Refusal(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(path, expected):
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise Refusal(f"Not an ordinary file: {path}")
    if sha256(path) != expected:
        raise Refusal(f"Hash mismatch: {path}")


def load_manifest(path, repo_root=ROOT):
    value = json.loads(Path(path).read_text())
    if value.get("owning_job") != OWNING_JOB or value.get("schema_version", 1) != 1:
        raise Refusal("Manifest has an unexpected owner or schema")
    rows = value.get("repairs", [])
    if len(rows) != len(EXPECTED):
        raise Refusal("Manifest must contain exactly the two declared repairs")
    seen = set()
    for row in rows:
        original, replacement = row.get("original"), row.get("replacement")
        if original not in EXPECTED or EXPECTED[original] != replacement or original in seen:
            raise Refusal("Manifest filenames do not match the reviewed repair set")
        seen.add(original)
        for key in ("original_sha256", "replacement_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", row.get(key, "")):
                raise Refusal("Manifest requires exact SHA256 guards")
    evidence = value.get("terminal_failure_evidence")
    if not isinstance(evidence, dict):
        raise Refusal("Terminal-failure evidence identity is required")
    relative = Path(evidence.get("path", ""))
    candidate = (Path(repo_root) / relative).resolve()
    if relative.is_absolute() or not candidate.is_relative_to(Path(repo_root).resolve()):
        raise Refusal("Evidence must be repository relative")
    checked_file(candidate, evidence.get("sha256"))
    return value


def query(command):
    return subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)


def terminal_state(raw):
    # sacct can append the cancelling UID, but a truncation '+' is ambiguous.
    token = raw.strip().split(" ", 1)[0]
    if token not in TERMINAL:
        raise Refusal(f"State is active or unclear: {raw!r}")


def owned_identifier(identifier):
    return identifier == OWNING_JOB or any(identifier.startswith(OWNING_JOB + mark) for mark in (".", "_", "+"))


def scheduler_gate(receipt, runner=query):
    commands = [
        ["squeue", "--noheader", "--user", "saini_2", "--format", "%i|%T"],
        ["sacct", "--noheader", "--parsable2", "--jobs", OWNING_JOB,
         "--format", "JobIDRaw,State%40,ExitCode"],
        ["squeue", "--steps", "--noheader", "--user", "saini_2", "--format", "%i"],
    ]
    receipt["scheduler"] = []
    outputs = []
    for command in commands:
        checked_at = now()
        try:
            result = runner(command)
        except Exception as exc:
            raise Refusal(f"Scheduler query failed: {command[0]}: {exc}") from exc
        receipt["scheduler"].append({"command": command, "checked_utc": checked_at,
                                     "returncode": result.returncode,
                                     "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode != 0 or result.stderr.strip():
            raise Refusal("Scheduler returned an error or warning; release refused")
        outputs.append(result.stdout)
    for line in outputs[0].splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split("|")]
        if len(fields) != 2 or not re.fullmatch(r"[0-9][0-9A-Za-z_.+\[\]%,:-]*", fields[0]):
            raise Refusal("Unexpected squeue job identity or format")
        if owned_identifier(fields[0]):
            terminal_state(fields[1])
    root_found = False
    for line in outputs[1].splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.rstrip("|").split("|")]
        if len(fields) != 3 or not re.fullmatch(r"\d+:\d+", fields[2]):
            raise Refusal("Unclear sacct record")
        if fields[0] == OWNING_JOB:
            root_found = True
        elif not fields[0].startswith(OWNING_JOB + "."):
            raise Refusal("Unexpected sacct job identity")
        terminal_state(fields[1])
    if not root_found:
        raise Refusal("sacct has no terminal owning-job record")
    for line in outputs[2].splitlines():
        identifier = line.strip()
        if not identifier:
            continue
        if not re.fullmatch(r"[0-9][0-9A-Za-z_.+\[\]%,:-]*", identifier):
            raise Refusal("Unexpected squeue step identity or format")
        if owned_identifier(identifier):
            raise Refusal("Owning job still has a visible scheduler step")
    receipt["scheduler_gate_passed"] = True


def matches(directory, name):
    return [path for path in directory.iterdir()
            if path.name == name or path.name.startswith(name + ".")]


def safe_root(taskq):
    taskq = Path(taskq).resolve()
    if not taskq.is_dir():
        raise Refusal("Task queue does not exist")
    for path in taskq.iterdir():
        if path.is_symlink():
            raise Refusal(f"Unexpected queue symlink: {path}")
    return taskq


def inspect_repair(taskq, row):
    original, replacement = row["original"], row["replacement"]
    archived = taskq / RECOVERY / original
    if archived.parent.is_symlink() or not archived.parent.resolve().is_relative_to(taskq):
        raise Refusal("Recovery directory escapes the queue")
    source = taskq / "failed" / original
    locations = []
    for directory in taskq.iterdir():
        if not directory.is_dir():
            continue
        found = list(directory.rglob(replacement + "*")) if directory.name == "recovered" else matches(directory, replacement)
        locations.extend((directory.name, path) for path in found)
        originals = matches(directory, original)
        if directory.name == "failed" and any(path != source for path in originals):
            raise Refusal("Original has an ambiguous failed-task suffix or duplicate")
        if directory.name != "failed" and originals:
            raise Refusal(f"Original task also exists outside failed: {directory}")
    archive_link_cleanup = source.exists() and archived.exists() and os.path.samefile(source, archived)
    if source.exists() and archived.exists() and not archive_link_cleanup:
        raise Refusal("Original exists in failed and recovery; ambiguous duplicate")
    if source.exists():
        checked_file(source, row["original_sha256"])
    if archived.exists():
        checked_file(archived, row["original_sha256"])
    if not source.exists() and not archived.exists():
        raise Refusal(f"Missing original failed task and recovery: {original}")
    held, active = [], []
    for state, path in locations:
        checked_file(path, row["replacement_sha256"])
        if state == "hold_review" and path.name == replacement:
            held.append(path)
        elif state in {"pending", "running", "done"}:
            active.append((state, path))
        else:
            raise Refusal(f"Replacement appears in refused state: {state}: {path}")
    if len(held) > 1 or len(active) > 1:
        raise Refusal("Replacement appears in multiple queue locations")
    if active:
        if not archived.exists() or source.exists():
            raise Refusal("Published replacement lacks the preserved original")
        if held and not os.path.samefile(held[0], active[0][1]):
            raise Refusal("Held and published replacements are independent duplicates")
        action = "cleanup_published_link" if held else "already_released"
    elif len(held) == 1:
        action = ("finish_archive_and_publish" if archive_link_cleanup else
                  "archive_and_publish" if source.exists() else "publish_after_archive")
    else:
        raise Refusal("Replacement is neither held nor already released")
    return {"original": original, "replacement": replacement, "action": action,
            "original_sha256": row["original_sha256"], "replacement_sha256": row["replacement_sha256"],
            "active_state": active[0][0] if active else None,
            "active_path": str(active[0][1]) if active else None}


def exclusive_move(source, target):
    # Linking publishes complete bytes atomically and cannot overwrite a racer.
    # The source stays preserved if interrupted between the two operations.
    os.link(source, target, follow_symlinks=False)
    source.unlink()


@contextmanager
def apply_lock(taskq):
    with (taskq / ".runtime_repairs_1076947.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Refusal("Another repair-release process holds the lock") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def release(manifest, taskq, receipt, apply=False, runner=query):
    taskq = safe_root(taskq)
    receipt.update(taskq=str(taskq), owning_job=OWNING_JOB, apply_requested=apply)
    def work():
        scheduler_gate(receipt, runner)
        plans = [inspect_repair(taskq, row) for row in manifest["repairs"]]
        receipt["plans"] = plans
        if not apply:
            receipt["status"] = "dry_run"
            return
        receipt["actions"] = []
        for row, plan in zip(manifest["repairs"], plans):
            # Validate again immediately before this task's mutation.
            plan = inspect_repair(taskq, row)
            source = taskq / "failed" / row["original"]
            archived = taskq / RECOVERY / row["original"]
            held = taskq / "hold_review" / row["replacement"]
            pending = taskq / "pending" / row["replacement"]
            if plan["action"] == "archive_and_publish":
                archived.parent.mkdir(parents=True, exist_ok=True)
                exclusive_move(source, archived)
                checked_file(archived, row["original_sha256"])
            elif plan["action"] == "finish_archive_and_publish":
                source.unlink()
            if plan["action"] in {"archive_and_publish", "finish_archive_and_publish", "publish_after_archive"}:
                exclusive_move(held, pending)
            elif plan["action"] == "cleanup_published_link":
                held.unlink()
            receipt["actions"].append(plan)
        receipt["status"] = "applied" if any(p["action"] != "already_released" for p in plans) else "already_released"
    if apply:
        with apply_lock(taskq):
            work()
    else:
        work()
    return receipt


def save_receipt(path, receipt):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f".part.{os.getpid()}")
    with partial.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, path)
    partial.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--taskq", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("Refusing to overwrite an existing receipt")
    receipt = {"schema_version": 1, "started_utc": now(), "helper_sha256": sha256(__file__),
               "manifest": str(args.manifest), "status": "refused"}
    code = 0
    try:
        manifest = load_manifest(args.manifest)
        receipt["manifest_sha256"] = sha256(args.manifest)
        release(manifest, args.taskq, receipt, args.apply)
    except Exception as exc:
        receipt.update(status="refused", error=f"{type(exc).__name__}: {exc}")
        code = 2
    receipt["finished_utc"] = now()
    save_receipt(args.receipt, receipt)
    print(json.dumps(receipt))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
