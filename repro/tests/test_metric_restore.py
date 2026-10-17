"""Ensure metric restoration stays read-only without opt-in and rejects bad assets."""
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "metrics_environment/restore_assets.py"
SPEC = importlib.util.spec_from_file_location("sca_metric_restore", SOURCE)
RESTORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESTORE)


def test_plan_does_not_create_cache_or_download(tmp_path, monkeypatch, capsys):
    hf, torch = tmp_path / "hf", tmp_path / "torch"
    monkeypatch.setattr(sys, "argv", [str(SOURCE), "--hf-home", str(hf), "--torch-home", str(torch)])
    monkeypatch.setattr(RESTORE.urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected download"))
    RESTORE.main()
    assert not hf.exists() and not torch.exists()
    assert '"download_enabled": false' in capsys.readouterr().out


def test_checksum_rejects_changed_asset_without_deleting(tmp_path):
    path = tmp_path / "asset.pt"
    path.write_bytes(b"known checkpoint")
    RESTORE.verify(path, hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        RESTORE.verify(path, "0" * 64)
    assert path.read_bytes() == b"known checkpoint"
