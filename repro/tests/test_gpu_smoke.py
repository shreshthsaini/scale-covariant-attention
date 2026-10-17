"""CPU-only checks of the GPU smoke supervisor; no torch import or CUDA call."""
import importlib.util
from pathlib import Path
import sys

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
