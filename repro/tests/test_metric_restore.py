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


def test_cache_copy_verifies_both_paths_and_preserves_changed_target(tmp_path):
    source, target = tmp_path / "original", tmp_path / "isolated" / "asset"
    source.write_bytes(b"verified asset")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    receipt = []
    RESTORE.copy_verified(source, target, expected, receipt)
    assert source.read_bytes() == target.read_bytes()
    assert receipt[0]["sha256"] == expected
    target.write_bytes(b"changed target")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        RESTORE.copy_verified(source, target, expected, [])
    assert target.read_bytes() == b"changed target"


def test_cache_restore_authenticates_git_tokenizer_blob(tmp_path):
    payload = b'{"tokenizer": "synthetic"}'
    blob_id = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    source = tmp_path / blob_id
    source.write_bytes(payload)
    record = {"hub": {"org/model": {"revision": "fixed-revision",
              "restore_files": ["tokenizer.json"], "files": [{"name": "tokenizer.json",
              "resolved_path": str(source), "bytes": len(payload)}]}}, "torch_files": {}}
    hf, torch = tmp_path / "hf", tmp_path / "torch"
    receipt = RESTORE.restore_from_cache(record, hf, torch)
    assert len(receipt) == 1
    assert (hf / "hub/models--org--model/refs/main").read_text().strip() == "fixed-revision"
    source.write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="Git blob identity mismatch"):
        RESTORE.restore_from_cache(record, hf, torch)


def test_cache_restore_refuses_existing_revision(tmp_path):
    hf = tmp_path / "hf"
    ref = hf / "hub/models--org--model/refs/main"
    ref.parent.mkdir(parents=True)
    ref.write_text("other-revision")
    record = {"hub": {"org/model": {"revision": "fixed-revision"}}, "torch_files": {}}
    with pytest.raises(RuntimeError, match="Refusing to change existing cache revision"):
        RESTORE.restore_from_cache(record, hf, tmp_path / "torch")
    assert ref.read_text() == "other-revision"
