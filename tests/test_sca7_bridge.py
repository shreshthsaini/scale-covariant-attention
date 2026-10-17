"""SCA7 bridge tests use isolated receipts and mocked scheduler responses only."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "slurm/node_launch_sca7_bridge.py"
SPEC = importlib.util.spec_from_file_location("sca7_bridge", SOURCE)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def environment(rank=0, **updates):
    result = dict(SLURM_JOB_ID="1075814", SLURM_JOB_NUM_NODES="4", SLURM_STEP_NUM_NODES="4",
                  SLURM_STEP_NUM_TASKS="4", SLURM_LOCALID="0", SLURM_PROCID=str(rank),
                  SLURM_STEP_ID="3", SLURM_RESTART_COUNT="0")
    result.update(updates)
    return result


def test_valid_allocation_end_never_queries_scheduler(tmp_path):
    deadline = int(time.time()) + 3600
    def forbidden(job):
        raise AssertionError("Valid environment deadline must not query scheduler")
    for rank in (2, 0, 3, 1):
        value, path = bridge.obtain_deadline(environment(rank, SLURM_JOB_END_TIME=str(deadline)),
                                            tmp_path, query=forbidden)
        assert value == deadline
    assert path.relative_to(tmp_path) == Path("job_1075814/restart_0/step_3/deadline.json")
    assert json.loads(path.read_text())["source"] == "SLURM_JOB_END_TIME"


def test_four_ranks_share_one_rank_zero_query(tmp_path):
    calls = []
    deadline = int(time.time()) + 3600
    def query(job):
        calls.append(job)
        time.sleep(0.03)
        return deadline
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(bridge.obtain_deadline, environment(rank, SLURM_JOB_END_TIME="expired"),
                               tmp_path, query=query, wait_seconds=1, poll_seconds=0.005)
                   for rank in (3, 2, 1, 0)]
        results = [future.result(timeout=2) for future in futures]
    assert calls == ["1075814"] and all(value == deadline for value, _ in results)
    assert len({path for _, path in results}) == 1
    # A restarted rank-zero wrapper in the same step reuses the receipt.
    assert bridge.obtain_deadline(environment(), tmp_path, query=query)[0] == deadline
    assert calls == ["1075814"]
    assert not list(tmp_path.rglob("*.tmp"))


def test_scheduler_failure_is_shared_and_not_requeried(tmp_path):
    calls = []
    def fail(job):
        calls.append(job)
        raise subprocess.TimeoutExpired("scontrol", 10)
    with pytest.raises(subprocess.TimeoutExpired):
        bridge.obtain_deadline(environment(), tmp_path, query=fail)
    for rank in (1, 2, 3, 0):
        with pytest.raises(RuntimeError, match="Rank zero deadline discovery failed"):
            bridge.obtain_deadline(environment(rank), tmp_path, query=fail, wait_seconds=0.1)
    assert calls == ["1075814"]


def test_peer_timeout_never_falls_back_to_scheduler(tmp_path):
    def forbidden(job):
        raise AssertionError("A peer must never query scheduler")
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="waited"):
        bridge.obtain_deadline(environment(2), tmp_path, query=forbidden, wait_seconds=0.04, poll_seconds=0.005)
    assert time.monotonic() - start < 0.5


@pytest.mark.parametrize("updates", [
    {"SLURM_JOB_ID": "1075813"}, {"SLURM_JOB_NUM_NODES": "2"},
    {"SLURM_STEP_NUM_NODES": "2"}, {"SLURM_STEP_NUM_TASKS": "8"},
    {"SLURM_LOCALID": "1"}, {"SLURM_PROCID": "4"}, {"SLURM_STEP_ID": "batch"},
])
def test_wrong_job_or_topology_rejected_before_coordination(tmp_path, updates):
    with pytest.raises(ValueError):
        bridge.obtain_deadline(environment(**updates), tmp_path)
    assert not list(tmp_path.iterdir())


def test_receipt_scope_and_expiry_are_validated(tmp_path):
    identity = bridge.scope(environment())
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({**identity, "step_id": "another", "deadline_epoch": int(time.time()) + 3600}))
    with pytest.raises(ValueError, match="different job/step/restart"):
        bridge.read_packet(path, identity)
    path.write_text(json.dumps({**identity, "deadline_epoch": 1}))
    with pytest.raises(ValueError, match="invalid or expired"):
        bridge.read_packet(path, identity)


def test_scontrol_parsing_uses_actual_endtime_once(monkeypatch):
    deadline = int(time.time()) + 3600
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=f"JobId=1075814 NumNodes=4 EndTime={datetime.fromtimestamp(deadline).isoformat()}\n")
    monkeypatch.setattr(bridge.subprocess, "run", run)
    assert bridge.query_deadline("1075814") == deadline
    assert len(calls) == 1
    assert calls[0][0] == ["scontrol", "show", "job", "1075814", "-o"]
    assert calls[0][1]["timeout"] == 10


def test_exec_preserves_four_node_topology(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SLURM_"):
            monkeypatch.delenv(key)
    deadline = int(time.time()) + 3600
    for key, value in environment(SLURM_JOB_END_TIME=str(deadline)).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, "argv", [str(SOURCE), "--coordination-root", str(tmp_path)])
    checked, executed = [], []
    monkeypatch.setattr(bridge, "validate_dispatch", lambda path: checked.append(path))
    monkeypatch.setattr(bridge.os, "execve", lambda *args: executed.append(args))
    bridge.main()
    assert checked == [bridge.DISPATCH]
    executable, args, env = executed[0]
    assert executable == "/bin/bash" and args == ["bash", str(bridge.DISPATCH / "node_launch_v2.sh")]
    assert env["SCA_DEADLINE_EPOCH"] == str(deadline)
    assert all(env[k] == "4" for k in ("SLURM_JOB_NUM_NODES", "SLURM_STEP_NUM_NODES", "SLURM_STEP_NUM_TASKS"))
    assert env["SLURM_LOCALID"] == "0" and env["SLURM_EXPORT_ENV"] == "ALL"
