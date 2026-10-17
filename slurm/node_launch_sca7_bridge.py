#!/usr/bin/env python3
"""Bridge only pending SCA7 job 1075814 to the frozen deadline-aware launcher.

No submission or topology change. Only rank zero may query scontrol, once, if
SLURM_JOB_END_TIME is unavailable. Peers wait on an atomic job/step receipt.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

JOB_ID = "1075814"
DISPATCH = Path("/scratch/09032/saini_2/sca/dispatch/20261017_v2")
COORDINATION = Path("/scratch/09032/saini_2/sca/dispatch/sca7_deadlines")
FROZEN_HASHES = {
    "worker_loop_v2.py": "837fc3622ac90026f6153828dde05eb96ac3c0f1b33f97b2a267121b21db244c",
    "node_launch_v2.sh": "417ea4e63c1c9ecb7f49dba8682d129712f167ecbd6266ef99ecaaa2136a398f",
    "telemetry.sh": "249cfe0a6d7a72791c3613dba573e480b3e509850c466fa93c39cbd8b1277323",
}


def scope(env):
    if env.get("SLURM_JOB_ID") != JOB_ID:
        raise ValueError(f"This bridge is restricted to job {JOB_ID}")
    if any(env.get(key) != "4" for key in ("SLURM_JOB_NUM_NODES", "SLURM_STEP_NUM_NODES", "SLURM_STEP_NUM_TASKS")):
        raise ValueError("SCA7 requires exactly four job nodes and four step tasks on four nodes")
    if env.get("SLURM_LOCALID") != "0" or env.get("SLURM_PROCID") not in ("0", "1", "2", "3"):
        raise ValueError("Expected one local task per node and global rank zero through three")
    step = env.get("SLURM_STEP_ID", "")
    restart = env.get("SLURM_RESTART_COUNT", "0")
    if not re.fullmatch(r"\d+", step) or not re.fullmatch(r"\d+", restart):
        raise ValueError("Numeric Slurm step ID and restart count are required")
    return {"job_id": JOB_ID, "step_id": step, "restart_count": restart}


def future_epoch(value, now=None):
    if not re.fullmatch(r"[0-9]+", str(value)):
        return None
    result = int(value)
    return result if result > (time.time() if now is None else now) else None


def query_deadline(job_id):
    result = subprocess.run(["scontrol", "show", "job", job_id, "-o"],
                            check=True, capture_output=True, text=True, timeout=10)
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z_]+)=([^\s]+)", result.stdout))
    if fields.get("JobId") != job_id or "EndTime" not in fields:
        raise ValueError("scontrol returned no matching job EndTime")
    # Slurm's unzoned timestamp and datetime.timestamp both use the process's
    # local timezone. ISO timestamps with explicit offsets are accepted too.
    deadline = int(datetime.fromisoformat(fields["EndTime"]).timestamp())
    if future_epoch(deadline) is None:
        raise ValueError("Scheduler allocation EndTime has expired")
    return deadline


def write_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def read_packet(path, identity):
    packet = json.loads(path.read_text())
    if any(packet.get(key) != value for key, value in identity.items()):
        raise ValueError("Shared deadline receipt belongs to a different job/step/restart")
    if "error" in packet:
        raise RuntimeError("Rank zero deadline discovery failed: " + packet["error"])
    deadline = future_epoch(packet.get("deadline_epoch"))
    if deadline is None:
        raise ValueError("Shared deadline is invalid or expired")
    return deadline


def obtain_deadline(env, root, *, wait_seconds=45, poll_seconds=0.25, query=query_deadline):
    identity = scope(env)
    rank = int(env["SLURM_PROCID"])
    directory = Path(root) / f"job_{identity['job_id']}" / f"restart_{identity['restart_count']}" / f"step_{identity['step_id']}"
    path = directory / "deadline.json"
    supplied = future_epoch(env.get("SLURM_JOB_END_TIME", ""))
    if supplied is not None:
        if rank == 0:
            packet = {**identity, "deadline_epoch": supplied, "source": "SLURM_JOB_END_TIME",
                      "coordinator_host": socket.gethostname(), "created_utc": datetime.now(timezone.utc).isoformat()}
            if path.exists():
                if read_packet(path, identity) != supplied:
                    raise ValueError("Conflicting existing step deadline")
            else:
                write_atomic(path, packet)
        return supplied, path
    if rank == 0:
        if path.exists():
            return read_packet(path, identity), path
        packet = {**identity, "source": "scontrol_once_rank_zero",
                  "coordinator_host": socket.gethostname(), "created_utc": datetime.now(timezone.utc).isoformat()}
        try:
            packet["deadline_epoch"] = query(identity["job_id"])
            if future_epoch(packet["deadline_epoch"]) is None:
                raise ValueError("Invalid scheduler deadline")
        except Exception as exc:
            packet["error"] = f"{type(exc).__name__}: {exc}"
            write_atomic(path, packet)
            raise
        write_atomic(path, packet)
        return packet["deadline_epoch"], path
    until = time.monotonic() + wait_seconds
    while time.monotonic() < until:
        if path.is_file():
            return read_packet(path, identity), path
        time.sleep(min(poll_seconds, max(0, until - time.monotonic())))
    raise TimeoutError(f"Rank {rank} waited {wait_seconds}s for rank-zero deadline: {path}")


def validate_dispatch(dispatch):
    for name, expected in FROZEN_HASHES.items():
        path = Path(dispatch) / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen dispatcher hash mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordination-root", type=Path, default=COORDINATION)
    parser.add_argument("--wait-seconds", type=float, default=45)
    parser.add_argument("--print-deadline", action="store_true", help="Resolve/print only; never start a worker")
    args = parser.parse_args()
    if not 0 < args.wait_seconds <= 60:
        parser.error("--wait-seconds must be positive and at most60")
    try:
        # Validate the frozen executable before any production scheduler query.
        if not args.print_deadline:
            validate_dispatch(DISPATCH)
        deadline, receipt = obtain_deadline(os.environ, args.coordination_root, wait_seconds=args.wait_seconds)
        print(f"SCA7 rank={os.environ['SLURM_PROCID']} deadline={deadline} receipt={receipt}", flush=True)
        if args.print_deadline:
            return 0
        env = dict(os.environ, SCA_DEADLINE_EPOCH=str(deadline), SLURM_EXPORT_ENV="ALL")
        os.execve("/bin/bash", ["bash", str(DISPATCH / "node_launch_v2.sh")], env)
    except Exception as exc:
        print(f"SCA7 bridge failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
