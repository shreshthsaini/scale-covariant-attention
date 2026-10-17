"""Deadline-aware SCA spool worker and node supervisor (Linux, standard library).

Freeze this file, node_launch_v2.sh, and the telemetry script together before
submission. The batch root supplies an actual allocation-end epoch. Task scripts
must keep their descendants in the task session; daemonizing outside it is not a
supported task contract. Ambiguous cleanup is quarantined, never retried.
"""
import argparse
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import uuid


STOP = False
EVENT_LOG = None
MINIMUM = re.compile(r"^\s*#\s*SCA_MIN_SECONDS\s*=\s*([0-9]+)\s*$")


def stopping(signum, frame):
    global STOP
    STOP = True


def subreaper():
    # Reap grandchildren orphaned during group shutdown instead of leaving zombies.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot become a child subreaper")


def proc_record(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]),
                    pgid=int(fields[2]), sid=int(fields[3]), ticks=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_records():
    result = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            record = proc_record(int(path.name))
            if record:
                result.append(record)
    return result


def members(pgid):
    return [r for r in process_records() if r["pgid"] == pgid and r["state"] != "Z"]


def escaped_children(pid, pgid):
    records = process_records()
    descendants = {pid}
    while True:
        found = {r["pid"] for r in records if r["ppid"] in descendants}
        if found <= descendants:
            break
        descendants |= found
    return [r for r in records if r["pid"] in descendants
            and r["pgid"] != pgid and r["state"] != "Z"]


@dataclass(frozen=True)
class Identity:
    pid: int
    ppid: int
    pgid: int
    sid: int
    started: str
    ticks: int


def inspect_identity(pid):
    """Use ps before signaling; /proc start ticks additionally prevent PID reuse."""
    try:
        check = subprocess.run(
            ["ps", "-o", "pid=,ppid=,pgid=,sid=,lstart=,stat=,args=", "-p", str(pid)],
            text=True, capture_output=True, check=False, timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        return None
    fields = check.stdout.strip().split(maxsplit=10)
    record = proc_record(pid)
    if check.returncode or len(fields) < 11 or record is None:
        return None
    return Identity(*(int(x) for x in fields[:4]), " ".join(fields[4:9]), record["ticks"])


def guard(command):
    """Keep the session leader alive until the command and adopted children exit."""
    subreaper()
    signal.signal(signal.SIGTERM, lambda *_: None)
    signal.signal(signal.SIGINT, lambda *_: None)
    signal.signal(signal.SIGHUP, lambda *_: None)
    if sys.stdin.buffer.readline() != b"start\n":
        return 125
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    code = child.wait()
    # Children orphaned by shell exit are adopted because this is a subreaper.
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            return code if code >= 0 else 128 - code
        except InterruptedError:
            continue


class OwnedGroup:
    def __init__(self, command, output, allow_child_sessions=False):
        self.allow_child_sessions = allow_child_sessions
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "guard", "--", *command],
            stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True)
        self.identity = inspect_identity(self.process.pid)
        if (self.identity is None or self.identity.ppid != os.getpid()
                or self.identity.pgid != self.identity.pid
                or self.identity.sid != self.identity.pid):
            self.process.stdin.close()  # Guardian exits on EOF before starting work.
            self.process.wait(timeout=5)
            raise RuntimeError("cannot establish owned process-group identity")
        self.process.stdin.write(b"start\n")
        self.process.stdin.close()

    def signal(self, signum):
        current = inspect_identity(self.identity.pid)
        if current != self.identity:
            return False
        # Negative targets are never inferred from names or user-wide process lists.
        try:
            os.killpg(self.identity.pgid, signum)
        except ProcessLookupError:
            return False
        return True

    def terminated(self):
        code = self.process.poll()
        return code is not None and not members(self.identity.pgid)

    def stop(self, grace, poll=0.1):
        if self.terminated():
            return True
        if not self.allow_child_sessions and escaped_children(self.identity.pid, self.identity.pgid):
            return False
        if not self.signal(signal.SIGTERM):
            return self.terminated()
        until = time.monotonic() + grace
        while time.monotonic() < until:
            if self.terminated():
                return True
            time.sleep(poll)
        # Guardian deliberately survives TERM while children remain, so its
        # recorded identity can be rechecked before escalating the same group.
        if not self.allow_child_sessions and escaped_children(self.identity.pid, self.identity.pgid):
            return False
        if not self.signal(signal.SIGKILL):
            return self.terminated()
        until = time.monotonic() + max(grace, 2)
        while time.monotonic() < until:
            if self.terminated():
                reap_adopted(self.identity.pgid)
                return True
            time.sleep(poll)
        return False


def reap_adopted(pgid):
    for record in process_records():
        if record["pgid"] != pgid or record["ppid"] != os.getpid():
            continue
        try:
            os.waitpid(record["pid"], os.WNOHANG)
        except ChildProcessError:
            continue


def task_seconds(path, unknown):
    # Metadata is data, never sourced or evaluated as shell code.
    with path.open() as stream:
        for _ in range(64):
            line = stream.readline(4096)
            if not line:
                break
            match = MINIMUM.fullmatch(line.rstrip("\n"))
            if match:
                value = int(match[1])
                if value < 1:
                    raise ValueError("SCA_MIN_SECONDS must be positive")
                return value
            if "SCA_MIN_SECONDS" in line:
                raise ValueError("malformed SCA_MIN_SECONDS header")
    return unknown


def fits(seconds, deadline, cleanup, now=None):
    return seconds + cleanup <= deadline - (time.time() if now is None else now)


def transfer(source, destination):
    """Publish without overwriting another worker's pending/done record."""
    os.link(source, destination)
    source.unlink()


def event(worker, action, task, rc=None):
    suffix = "" if rc is None else f" rc={rc}"
    line = f"{datetime.now():%Y-%m-%dT%H:%M:%S} {worker} {action} {task}{suffix}"
    print(line, flush=True)
    if EVENT_LOG is not None:
        # Existing account_compute.py discovers worker*.log and deduplicates
        # identical events mirrored in the fleet's Slurm stdout.
        with EVENT_LOG.open("a") as stream:
            stream.write(line + "\n")


def worker(args):
    global EVENT_LOG
    queue, logs = Path(args.queue), Path(args.logs)
    for name in ("pending", "running", "done", "failed", "quarantine"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    attempts = logs / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    hostname = socket.gethostname().split(".")[0]
    worker_id = f"{hostname}-g{os.environ.get('CUDA_VISIBLE_DEVICES', '0')}-j{os.environ.get('SLURM_JOB_ID', 'na')}-p{os.getpid()}-{uuid.uuid4().hex[:8]}"
    EVENT_LOG = logs / f"worker_{worker_id}.log"
    idle_since = time.monotonic()
    while not STOP and not (queue / "STOP").exists():
        candidates = sorted((queue / "pending").glob("*.sh"))
        claimed = None
        for source in candidates:
            try:
                if not fits(task_seconds(source, args.unknown_seconds), args.deadline, args.cleanup):
                    continue
                destination = queue / "running" / f"{source.name}.{worker_id}"
                source.rename(destination)  # Atomic claim; legacy workers race on same source.
                claimed = destination
                break
            except FileNotFoundError:
                continue
            except ValueError as exc:
                print(f"skip {source.name}: {exc}", flush=True)
        if claimed is None:
            if candidates:
                print(f"{worker_id} exit: no pending task fits remaining allocation time", flush=True)
                return 0
            if (time.monotonic() - idle_since >= args.idle_seconds
                    or not fits(0, args.deadline, args.cleanup)):
                return 0
            time.sleep(args.poll)
            continue
        idle_since = time.monotonic()
        task = claimed.name.removesuffix("." + worker_id)
        attempt = f"{task.removesuffix('.sh')}.{time.time_ns()}.{worker_id}"
        logpath = attempts / f"{attempt}.log"
        # Separate from legacy task_*.log, which old workers reopen with truncation.
        latest = logs / f"task_{task.removesuffix('.sh')}.latest.log"
        temporary = logs / f".latest-{uuid.uuid4().hex}"
        temporary.symlink_to(logpath.resolve())
        temporary.replace(latest)
        event(worker_id, "start", task)
        group = None
        try:
            with logpath.open("xb") as output:
                if STOP or not fits(task_seconds(claimed, args.unknown_seconds), args.deadline, args.cleanup):
                    transfer(claimed, queue / "pending" / task)
                    event(worker_id, "requeued", task)
                    return 0
                group = OwnedGroup(["bash", str(claimed.resolve())], output)
                metadata = dict(task=task, worker=worker_id, log=str(logpath),
                                events=str(EVENT_LOG), deadline=args.deadline,
                                identity=asdict(group.identity))
                (attempts / f"{attempt}.json").write_text(json.dumps(metadata, indent=2) + "\n")
                while group.process.poll() is None:
                    if STOP or time.time() >= args.deadline - args.cleanup or (queue / "STOP").exists():
                        stopped = group.stop(args.terminate_grace)
                        if stopped:
                            transfer(claimed, queue / "pending" / task)
                            event(worker_id, "requeued", task)
                        else:
                            transfer(claimed, queue / "quarantine" / claimed.name)
                            event(worker_id, "end", task, 125)
                            print(f"QUARANTINED {task}: process termination not confirmed", flush=True)
                        return 0 if stopped else 1
                    time.sleep(args.poll)
                code = group.process.returncode
                if not group.terminated():
                    raise RuntimeError("live task-group members after guardian exit")
                # Slurm can signal worker and task simultaneously. A task that
                # has already exited when we observe STOP is still an interrupted
                # attempt, and must not silently become an ordinary failed task.
                if STOP or time.time() >= args.deadline - args.cleanup or (queue / "STOP").exists():
                    transfer(claimed, queue / "pending" / task)
                    event(worker_id, "requeued", task)
                    return 0
                transfer(claimed, queue / ("done" if code == 0 else "failed") / task)
                event(worker_id, "end", task, code)
        except Exception as exc:
            stopped = group is None or group.stop(args.terminate_grace)
            if claimed.exists():
                transfer(claimed, queue / "quarantine" / claimed.name)
            event(worker_id, "end", task, 125)
            print(f"QUARANTINED {task}: {exc}; termination_confirmed={stopped}", flush=True)
            return 1
    return 0


def validate_step(env):
    nodes = int(env["SLURM_STEP_NUM_NODES"])
    tasks = int(env["SLURM_STEP_NUM_TASKS"])
    local = int(env["SLURM_LOCALID"])
    if nodes < 1 or tasks != nodes or local != 0:
        raise ValueError("expected exactly one launcher task per allocated step node")
    return nodes, tasks


def supervise(args):
    nodes, tasks = validate_step(os.environ)
    telemetry = Path(args.telemetry_script)
    if not telemetry.is_absolute() or not telemetry.is_file():
        raise ValueError("pass an existing frozen absolute telemetry script")
    print(f"step_nodes={nodes} step_tasks={tasks} deadline={args.deadline}", flush=True)
    command = [sys.executable, str(Path(__file__).resolve()), "worker",
               "--queue", args.queue, "--logs", args.logs, "--deadline", str(args.deadline),
               "--unknown-seconds", str(args.unknown_seconds), "--cleanup", str(args.cleanup),
               "--idle-seconds", str(args.idle_seconds), "--poll", str(args.poll),
               "--terminate-grace", str(args.terminate_grace)]
    logger = OwnedGroup(["bash", str(telemetry), args.telemetry_dir, "30"], None)
    runner = None
    try:
        # Worker owns and supervises separate task sessions itself.
        runner = OwnedGroup(command, None, allow_child_sessions=True)
        while runner.process.poll() is None:
            if logger.process.poll() is not None:
                raise RuntimeError("telemetry exited while worker was active")
            if STOP or time.time() >= args.deadline - args.cleanup:
                if not runner.stop(args.terminate_grace + 5):
                    raise RuntimeError("worker shutdown not confirmed")
                break
            time.sleep(args.poll)
        return runner.process.returncode or 0
    finally:
        if runner and not runner.terminated():
            runner.stop(args.terminate_grace + 5)
        if not logger.stop(args.terminate_grace):
            raise RuntimeError("telemetry shutdown not confirmed")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "guard":
        return guard(sys.argv[3:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "node"))
    parser.add_argument("--queue", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--unknown-seconds", type=int, default=2700)
    parser.add_argument("--cleanup", type=float, default=120)
    parser.add_argument("--terminate-grace", type=float, default=10)
    parser.add_argument("--idle-seconds", type=float, default=30)
    parser.add_argument("--poll", type=float, default=1)
    parser.add_argument("--telemetry-script")
    parser.add_argument("--telemetry-dir", default="/scratch/09032/saini_2/sca/telemetry")
    args = parser.parse_args()
    durations = (args.deadline, args.cleanup, args.terminate_grace, args.poll, args.idle_seconds)
    if (not all(math.isfinite(x) for x in durations) or args.unknown_seconds < 1 or args.cleanup < 0
            or args.terminate_grace <= 0 or args.poll <= 0 or args.idle_seconds < 0):
        parser.error("invalid duration")
    subreaper()
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stopping)
    return worker(args) if args.mode == "worker" else supervise(args)


if __name__ == "__main__":
    sys.exit(main())
