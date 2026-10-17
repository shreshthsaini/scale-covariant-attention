"""Isolated dispatcher tests. No scheduler, GPU, or live spool access."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from datetime import datetime

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "slurm" / "worker_loop_v2.py"
SPEC = importlib.util.spec_from_file_location("sca_worker_v2", SOURCE)
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


def setup_queue(tmp_path):
    queue, logs = tmp_path / "queue", tmp_path / "logs"
    (queue / "pending").mkdir(parents=True)
    logs.mkdir()
    return queue, logs


def command(queue, logs, *, mode="worker", seconds=30):
    return [sys.executable, str(SOURCE), mode, "--queue", str(queue),
            "--logs", str(logs), "--deadline", str(time.time() + seconds),
            "--unknown-seconds", "1", "--cleanup", "0.2", "--idle-seconds", "0",
            "--terminate-grace", "0.2", "--poll", "0.02"]


def wait_for(predicate, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition did not become true")


def verified_signal(process, signum):
    identity = worker.inspect_identity(process.pid)
    if process.poll() is None:
        assert identity is not None and identity.ppid == os.getpid()
        assert worker.inspect_identity(process.pid) == identity
        os.kill(process.pid, signum)


def finish(process):
    if process.poll() is None:
        verified_signal(process, signal.SIGTERM)
    process.wait(timeout=8)


def test_metadata_and_deadline(tmp_path):
    task = tmp_path / "task.sh"
    task.write_text("echo test\n")
    assert worker.task_seconds(task, 2700) == 2700
    task.write_text("#!/bin/bash\n# SCA_MIN_SECONDS=42\necho test\n")
    assert worker.task_seconds(task, 2700) == 42
    assert worker.fits(42, 164, 120, now=2)
    assert not worker.fits(42, 163, 120, now=2)
    task.write_text("# SCA_MIN_SECONDS=0\n")
    with pytest.raises(ValueError):
        worker.task_seconds(task, 2700)


def test_step_scope_uses_allocated_step_not_request_minimum():
    env = dict(SLURM_STEP_NUM_NODES="8", SLURM_STEP_NUM_TASKS="8",
               SLURM_LOCALID="0", SLURM_JOB_NUM_NODES="2")
    assert worker.validate_step(env) == (8, 8)
    with pytest.raises(ValueError):
        worker.validate_step(dict(env, SLURM_STEP_NUM_TASKS="16"))


def test_identity_mismatch_never_signals(monkeypatch):
    group = object.__new__(worker.OwnedGroup)
    group.identity = worker.Identity(123, 10, 123, 123, "original", 55)
    monkeypatch.setattr(worker, "inspect_identity", lambda _: None)
    calls = []
    monkeypatch.setattr(os, "killpg", lambda *args: calls.append(args))
    assert not group.signal(signal.SIGTERM)
    assert calls == []


def test_scan_past_long_task_and_preserve_legacy_log(tmp_path):
    queue, logs = setup_queue(tmp_path)
    (queue / "pending" / "a_long.sh").write_text("# SCA_MIN_SECONDS=9999\nexit 99\n")
    (queue / "pending" / "b_short.sh").write_text("# SCA_MIN_SECONDS=1\necho short-ok\n")
    legacy = logs / "task_b_short.log"
    legacy.write_text("preserved legacy log\n")
    run = subprocess.run(command(queue, logs), capture_output=True, text=True, timeout=8)
    assert run.returncode == 0, run.stdout + run.stderr
    assert (queue / "pending" / "a_long.sh").exists()
    assert (queue / "done" / "b_short.sh").exists()
    assert not list((queue / "running").iterdir())
    assert legacy.read_text() == "preserved legacy log\n"
    assert (logs / "task_b_short.latest.log").read_text() == "short-ok\n"
    assert " start b_short.sh" in run.stdout and " end b_short.sh rc=0" in run.stdout
    events = next(logs.glob("worker*.log")).read_text()
    assert " start b_short.sh" in events and " end b_short.sh rc=0" in events


def test_all_nonfitting_tasks_exit_without_claim(tmp_path):
    queue, logs = setup_queue(tmp_path)
    (queue / "pending" / "long.sh").write_text("# SCA_MIN_SECONDS=9999\nexit 0\n")
    run = subprocess.run(command(queue, logs), capture_output=True, text=True, timeout=3)
    assert run.returncode == 0
    assert not list((queue / "running").iterdir())
    assert not list((logs / "attempts").iterdir())


def test_competing_workers_claim_each_task_once(tmp_path):
    queue, logs = setup_queue(tmp_path)
    seen = tmp_path / "seen"
    for index in range(6):
        (queue / "pending" / f"task{index}.sh").write_text(f"echo {index} >> '{seen}'\nsleep 0.05\n")
    processes = [subprocess.Popen(command(queue, logs), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                 for _ in range(2)]
    try:
        for process in processes:
            output, error = process.communicate(timeout=10)
            assert process.returncode == 0, output + error
        assert sorted(seen.read_text().splitlines()) == list("012345")
        assert len(list((queue / "done").iterdir())) == 6
    finally:
        for process in processes:
            finish(process)


def test_term_kills_owned_descendants_before_requeue(tmp_path):
    queue, logs = setup_queue(tmp_path)
    child_pid = tmp_path / "child.pid"
    task = queue / "pending" / "interrupt.sh"
    task.write_text(f"trap '' TERM\n{sys.executable} -c \"import os,time; open('{child_pid}', 'w').write(str(os.getpid())); time.sleep(30)\"\n")
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    process = subprocess.Popen(command(queue, logs), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        wait_for(child_pid.exists)
        pid = int(child_pid.read_text())
        assert worker.proc_record(pid) is not None
        verified_signal(process, signal.SIGTERM)
        output, error = process.communicate(timeout=8)
        assert process.returncode == 0, output + error
        assert task.exists()
        assert not list((queue / "running").iterdir())
        assert worker.proc_record(pid) is None
        assert sentinel.poll() is None
        assert b" requeued interrupt.sh" in output
    finally:
        finish(process)
        finish(sentinel)


def test_deadline_stops_current_task_and_requeues(tmp_path):
    queue, logs = setup_queue(tmp_path)
    task = queue / "pending" / "deadline.sh"
    task.write_text("# SCA_MIN_SECONDS=1\nsleep 30\n")
    run = subprocess.run(command(queue, logs, seconds=1.8), capture_output=True, text=True, timeout=8)
    assert run.returncode == 0, run.stdout + run.stderr
    assert task.exists() and " requeued deadline.sh" in run.stdout


def test_supervisor_reaps_logger_and_children(tmp_path):
    queue, logs = setup_queue(tmp_path)
    (queue / "pending" / "done.sh").write_text("sleep 0.2\necho done\n")
    telemetry = tmp_path / "telemetry.sh"
    directory = tmp_path / "telemetry"
    directory.mkdir()
    telemetry.write_text('echo $$ > "$1/logger.pid"\ntrap "" TERM\nsleep 30 &\necho $! > "$1/sleep.pid"\nwait\n')
    cmd = command(queue, logs, mode="node") + ["--telemetry-script", str(telemetry), "--telemetry-dir", str(directory)]
    env = dict(os.environ, SLURM_STEP_NUM_NODES="3", SLURM_STEP_NUM_TASKS="3", SLURM_LOCALID="0")
    run = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=8)
    assert run.returncode == 0, run.stdout + run.stderr
    assert (queue / "done" / "done.sh").exists()
    for path in directory.glob("*.pid"):
        assert worker.proc_record(int(path.read_text())) is None


def test_attempt_logs_survive_retry(tmp_path):
    queue, logs = setup_queue(tmp_path)
    task = queue / "pending" / "retry.sh"
    task.write_text("echo recorded-attempt\nexit 1\n")
    for _ in range(2):
        run = subprocess.run(command(queue, logs), capture_output=True, text=True, timeout=8)
        assert run.returncode == 0
        (queue / "failed" / task.name).rename(task)
    attempts = list((logs / "attempts").glob("*.log"))
    assert len(attempts) == 2
    assert all(path.read_text() == "recorded-attempt\n" for path in attempts)
    for path in (logs / "attempts").glob("*.json"):
        record = json.loads(path.read_text())
        assert record["identity"]["pid"] == record["identity"]["pgid"]


def test_supervisor_signal_requeues_task_and_reaps_telemetry(tmp_path):
    queue, logs = setup_queue(tmp_path)
    task_pid = tmp_path / "task.pid"
    task = queue / "pending" / "interrupt.sh"
    task.write_text(f"echo $$ > '{task_pid}'\nsleep 30\n")
    telemetry = tmp_path / "telemetry.sh"
    directory = tmp_path / "telemetry"
    directory.mkdir()
    telemetry.write_text('echo $$ > "$1/logger.pid"\nsleep 30\n')
    cmd = command(queue, logs, mode="node") + ["--telemetry-script", str(telemetry), "--telemetry-dir", str(directory)]
    env = dict(os.environ, SLURM_STEP_NUM_NODES="2", SLURM_STEP_NUM_TASKS="2", SLURM_LOCALID="0")
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        wait_for(task_pid.exists)
        pid = int(task_pid.read_text())
        verified_signal(process, signal.SIGTERM)
        output, error = process.communicate(timeout=8)
        assert process.returncode == 0, output + error
        assert task.exists() and worker.proc_record(pid) is None
        assert worker.proc_record(int((directory / "logger.pid").read_text())) is None
        assert b" requeued interrupt.sh" in output
    finally:
        finish(process)


def test_account_compute_event_compatibility(capsys):
    worker.event("c620-001-g0-j123-p456-abcd", "start", "task.sh")
    worker.event("c620-001-g0-j123-p456-abcd", "requeued", "task.sh")
    text = capsys.readouterr().out
    source = SOURCE.parents[1] / "scripts" / "account_compute.py"
    spec = importlib.util.spec_from_file_location("sca_accounting_v2_test", source)
    accounting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(accounting)
    events = accounting.parse_events(text, "fleet_123.out")
    spans, active, orphan, _ = accounting.pair_events(events)
    assert len(spans) == 1 and spans[0]["outcome"] == "requeued"
    assert active == [] and orphan == []


def test_transfer_does_not_overwrite_existing_pending_task(tmp_path):
    running, pending = tmp_path / "running.sh", tmp_path / "pending.sh"
    running.write_text("owned attempt")
    pending.write_text("already queued")
    with pytest.raises(FileExistsError):
        worker.transfer(running, pending)
    assert running.read_text() == "owned attempt"
    assert pending.read_text() == "already queued"


def test_telemetry_failure_stops_and_requeues_worker(tmp_path):
    queue, logs = setup_queue(tmp_path)
    task = queue / "pending" / "long.sh"
    task.write_text("sleep 30\n")
    telemetry = tmp_path / "telemetry.sh"
    telemetry.write_text("sleep 0.5\nexit 1\n")
    cmd = command(queue, logs, mode="node") + ["--telemetry-script", str(telemetry)]
    env = dict(os.environ, SLURM_STEP_NUM_NODES="2", SLURM_STEP_NUM_TASKS="2", SLURM_LOCALID="0")
    run = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=8)
    assert run.returncode != 0
    assert task.exists() and not list((queue / "running").iterdir())
    assert "telemetry exited" in run.stderr


def test_unconfirmed_cleanup_quarantines_instead_of_retry(tmp_path, monkeypatch, capsys):
    queue, logs = setup_queue(tmp_path)
    (queue / "pending" / "uncertain.sh").write_text("echo not-executed\n")
    monkeypatch.setattr(worker, "STOP", False)
    monkeypatch.setattr(worker, "EVENT_LOG", None)

    class UncertainGroup:
        def __init__(self, *_):
            self.identity = worker.Identity(555, 10, 555, 555, "recorded", 42)
            self.process = SimpleNamespace(poll=lambda: None)
            worker.STOP = True

        def stop(self, *_):
            return False

    monkeypatch.setattr(worker, "OwnedGroup", UncertainGroup)
    args = SimpleNamespace(queue=str(queue), logs=str(logs), deadline=time.time() + 100,
                           unknown_seconds=1, cleanup=2, terminate_grace=0.1,
                           idle_seconds=0, poll=0.01)
    assert worker.worker(args) == 1
    assert not list((queue / "pending").iterdir())
    assert len(list((queue / "quarantine").iterdir())) == 1
    assert " requeued " not in capsys.readouterr().out


@pytest.mark.parametrize("valid_env", [True, False])
def test_batch_uses_actual_nodes_and_one_deadline_lookup(tmp_path, valid_env):
    batch_source = SOURCE.with_name("fleet_SCA9.sbatch").read_text()
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    for name in ["worker_loop_v2.py", "node_launch_v2.sh", "telemetry.sh"]:
        (frozen / name).touch()
    batch = tmp_path / "batch.sh"
    batch.write_text(batch_source.replace("/scratch/09032/saini_2/sca/dispatch/20261017_v2", str(frozen)))
    binaries = tmp_path / "bin"
    binaries.mkdir()
    deadline = int(time.time()) + 3600
    calls, launched = tmp_path / "calls", tmp_path / "launched"
    control = binaries / "scontrol"
    control.write_text(f"#!/bin/bash\necho called >> '{calls}'\necho 'JobId=123 EndTime={datetime.fromtimestamp(deadline).isoformat()}'\n")
    control.chmod(0o755)
    srun = binaries / "srun"
    srun.write_text(f'#!/bin/bash\nprintf "%s\\n" "$SCA_DEADLINE_EPOCH" "$*" > "{launched}"\n')
    srun.chmod(0o755)
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
               SLURM_JOB_NUM_NODES="3", SLURM_JOB_ID="123",
               SLURM_JOB_END_TIME=str(deadline) if valid_env else "Unknown")
    run = subprocess.run(["bash", str(batch)], env=env, capture_output=True, text=True, timeout=5)
    assert run.returncode == 0, run.stdout + run.stderr
    text = launched.read_text()
    assert text.splitlines()[0] == str(deadline)
    assert "--nodes=3 --ntasks=3 --ntasks-per-node=1" in text
    assert (not calls.exists()) if valid_env else calls.read_text().splitlines() == ["called"]
