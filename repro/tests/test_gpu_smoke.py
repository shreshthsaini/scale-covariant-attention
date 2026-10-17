"""CPU-only checks of the GPU smoke supervisor; no torch import or CUDA call."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "environment/gpu_smoke.py"
SPEC = importlib.util.spec_from_file_location("sca_gpu_smoke", SOURCE)
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def test_supervisor_keeps_nonzero_exit_and_log(tmp_path):
    log = tmp_path / "worker.log"
    rc, timed_out = SMOKE.supervise([sys.executable, "-c", "print('diagnostic'); raise SystemExit(7)"], 10, log)
    assert rc == 7 and not timed_out
    assert "diagnostic" in log.read_text()


def test_supervisor_bounds_owned_worker(tmp_path):
    log = tmp_path / "worker.log"
    rc, timed_out = SMOKE.supervise([sys.executable, "-c", "import time; time.sleep(60)"], 1, log)
    assert timed_out and rc != 0
    assert "verified owned process group" in log.read_text()


def test_compiler_preflight_overrides_inherited_nvidia_compilers(tmp_path, monkeypatch):
    monkeypatch.setenv("CC", "/unavailable/nvc")
    monkeypatch.setenv("CXX", "/unavailable/nvc++")
    receipt = {}
    SMOKE.compiler_preflight("gcc", "g++", tmp_path, receipt)
    assert receipt["passed"]
    assert receipt["inherited"]["CC"] == "/unavailable/nvc"
    assert all(probe["load_result"] == 42 for probe in receipt["probes"])
    assert all("-Wno-psabi" in probe["command"] for probe in receipt["probes"])
    assert [probe["variable"] for probe in receipt["probes"]] == ["CC", "CXX"]
    assert SMOKE.os.environ["CC"] == receipt["selected"]["CC"]
    assert SMOKE.os.environ["CXX"] == receipt["selected"]["CXX"]


def test_compiler_preflight_refuses_missing_compiler(tmp_path, monkeypatch):
    monkeypatch.setattr(SMOKE.shutil, "which", lambda command: None)
    with pytest.raises(RuntimeError, match="compiler is unavailable"):
        SMOKE.compiler_preflight("missing", "missing++", tmp_path, {})


def test_compiler_preflight_preserves_rejected_flags(tmp_path, monkeypatch):
    compiler = tmp_path / "reject_compiler"
    compiler.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then echo rejecting-compiler; exit 0; fi\necho "rejected -Wno-psabi" >&2\nexit 9\n')
    compiler.chmod(0o700)
    monkeypatch.setenv("CC", "old-compiler")
    monkeypatch.setenv("CXX", "old-compiler++")
    receipt = {}
    with pytest.raises(RuntimeError, match="rejected -Wno-psabi"):
        SMOKE.compiler_preflight(str(compiler), str(compiler), tmp_path / "probe", receipt)
    assert receipt["probes"][0]["exit_code"] == 9
    assert not receipt.get("passed", False)
    assert (tmp_path / "probe/host_probe.c").is_file()


def test_compiler_only_cli_stops_before_torch(tmp_path):
    out = tmp_path / "compiler.json"
    result = subprocess.run([sys.executable, str(SOURCE), "--compiler-preflight-only", "--out", str(out)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    record = json.loads(out.read_text())
    assert record["status"] == "passed"
    assert record["stage"] == "compiler_preflight_complete"
    assert record["compiler_preflight"]["passed"]
    assert "torch" not in record
    assert record["checks"] == []
    assert not record["timed_out"]


def test_compiler_only_cli_failure_writes_partial_evidence(tmp_path):
    out = tmp_path / "failed.json"
    result = subprocess.run([sys.executable, str(SOURCE), "--compiler-preflight-only", "--cc", "/missing/cc", "--out", str(out)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    record = json.loads(out.read_text())
    assert record["status"] == "failed"
    assert record["stage"] == "host_compiler_preflight"
    assert "compiler is unavailable" in record["error"]
    assert "torch" not in record
