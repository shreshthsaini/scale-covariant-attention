"""Readiness and publication tests use temporary files, never the live spool."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "slurm/promote_ready_eval.py"
SPEC = importlib.util.spec_from_file_location("promote_ready_eval", SOURCE)
promoter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promoter)


def png_bytes(width=4, height=4):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + b"\0\0\0" * width) * height)) + chunk(b"IEND", b""))


def images(tmp_path, methods=("area", "sca_nb")):
    records = [{"id": "prompt_0", "prompt": "A tree"}, {"id": "prompt_1", "prompt": "A lake"}]
    directories = {method: tmp_path / method for method in methods}
    for method, directory in directories.items():
        directory.mkdir()
        for record in records:
            stem = promoter.image_stem(record["id"], 0)
            png = directory / (stem + ".png")
            png.write_bytes(png_bytes())
            meta = {"complete": True, "config": dict(record, prompt_id=record["id"], method=method, res="4x4", seed=0),
                    "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest()}
            png.with_suffix(".json").write_text(json.dumps(meta))
    return records, directories


def queue(tmp_path):
    root = tmp_path / "queue"
    for state in ("hold_cpu", "pending", "running", "done", "failed", "quarantine"):
        (root / state).mkdir(parents=True)
    (root / "hold_cpu/eval.sh").write_text("#!/bin/bash\n# SCA_MIN_SECONDS=2700\necho evaluate\n")
    return root


def test_fixed_profiles_and_manifest(tmp_path):
    path = tmp_path / "eval100.jsonl"
    records = [{"id": str(i), "prompt": "prompt"} for i in range(100)]
    path.write_text("\n".join(map(json.dumps, records)))
    assert len(promoter.manifest(path)) == 100
    records[-1] = records[0]
    path.write_text("\n".join(map(json.dumps, records)))
    with pytest.raises(ValueError):
        promoter.manifest(path)
    path.write_text(json.dumps({"id": "only_one", "prompt": "prompt"}))
    with pytest.raises(ValueError):
        promoter.manifest(path)
    sources = promoter.sources("e9_krea", tmp_path, tmp_path / "reused")
    assert set(sources) == {"area", "sca_krea", "dype", "sega_official"}
    assert sources["dype"] == tmp_path / "reused/dype/4096x4096"


def test_complete_paired_outputs_and_extra_unselected_files(tmp_path):
    records, directories = images(tmp_path)
    (directories["area"] / "outside_manifest.json").write_text("irrelevant")
    result = promoter.readiness(records, directories, "4x4")
    assert result["status"] == "ready" and result["verified_png_hashes"] == 4


def test_missing_outputs_skip_expensive_validation(tmp_path, monkeypatch):
    records, directories = images(tmp_path)
    next(directories["area"].glob("*.png")).unlink()
    monkeypatch.setattr(promoter, "check_image", lambda *a: pytest.fail("Should not hash incomplete coverage"))
    result = promoter.readiness(records, directories, "4x4")
    assert result == {"status": "not_ready", "missing": {"area": 1, "sca_nb": 0}}


@pytest.mark.parametrize("field,value", [("method", "wrong"), ("prompt_id", "wrong"),
                                         ("prompt", "wrong"), ("seed", 1), ("seed", False), ("res", "8x8")])
def test_sidecar_identity_mismatch_blocks(tmp_path, field, value):
    records, directories = images(tmp_path)
    path = next(directories["area"].glob("*.json"))
    meta = json.loads(path.read_text()); meta["config"][field] = value
    path.write_text(json.dumps(meta))
    assert promoter.readiness(records, directories, "4x4")["status"] == "invalid_outputs"


@pytest.mark.parametrize("change", ["complete", "missing_complete", "hash", "truncated", "dimensions"])
def test_corrupt_or_incomplete_artifact_blocks(tmp_path, change):
    records, directories = images(tmp_path)
    path = next(directories["area"].glob("*.json")); meta = json.loads(path.read_text())
    if change == "complete": meta["complete"] = False
    if change == "missing_complete": del meta["complete"]
    if change == "hash": meta["png_sha256"] = "0" * 64
    if change == "truncated": path.with_suffix(".png").write_bytes(png_bytes()[:-12])
    if change == "dimensions": path.with_suffix(".png").write_bytes(png_bytes(5, 4))
    path.write_text(json.dumps(meta))
    assert promoter.readiness(records, directories, "4x4")["status"] == "invalid_outputs"


def test_inspection_and_not_ready_leave_spool_unchanged(tmp_path):
    root = queue(tmp_path)
    assert promoter.promote(root, "eval.sh", lambda: {"status": "ready"})["status"] == "ready"
    assert promoter.promote(root, "eval.sh", lambda: {"status": "not_ready"}, True)["status"] == "not_ready"
    assert (root / "hold_cpu/eval.sh").is_file() and not list((root / "pending").iterdir())


def test_publication_preserves_script_and_is_idempotent(tmp_path):
    root = queue(tmp_path); held = root / "hold_cpu/eval.sh"; before = held.read_bytes()
    assert promoter.promote(root, "eval.sh", lambda: {"status": "ready"}, True)["status"] == "promoted"
    assert not held.exists() and (root / "pending/eval.sh").read_bytes() == before
    assert promoter.promote(root, "eval.sh", lambda: pytest.fail("already published"), True)["status"] == "already_dispatched"


@pytest.mark.parametrize("state,name", [("running", "eval.sh.node.123"), ("done", "eval.sh"),
                                       ("failed", "eval.sh"), ("quarantine", "eval.sh.node.123")])
def test_existing_state_never_republished(tmp_path, state, name):
    root = queue(tmp_path); (root / state / name).write_text("previous attempt")
    assert promoter.promote(root, "eval.sh", lambda: pytest.fail("existing task"), True)["status"] == "already_dispatched"
    assert not list((root / "pending").iterdir())


def test_concurrent_publisher_returns_without_waiting(tmp_path):
    root = queue(tmp_path)
    with (root / ".promote_eval.sh.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert promoter.promote(root, "eval.sh", lambda: pytest.fail("lock held"), True)["status"] == "publisher_busy"


def test_incomplete_coverage_never_holds_publisher_lock(tmp_path):
    root = queue(tmp_path)
    result = promoter.promote(root, "eval.sh", lambda: pytest.fail("incomplete coverage"), True,
                              precheck=lambda: {"status": "not_ready"})
    assert result["status"] == "not_ready"
    assert not (root / ".promote_eval.sh.lock").exists()


def test_worker_can_claim_immediately_after_publication(tmp_path, monkeypatch):
    root = queue(tmp_path); link = os.link
    def claim(source, target, **kwargs):
        link(source, target, **kwargs)
        os.rename(target, root / "running/eval.sh.node.123")
    monkeypatch.setattr(promoter.os, "link", claim)
    assert promoter.promote(root, "eval.sh", lambda: {"status": "ready"}, True)["status"] == "promoted"
    assert (root / "running/eval.sh.node.123").is_file() and not (root / "hold_cpu/eval.sh").exists()


def test_publication_collision_never_overwrites(tmp_path, monkeypatch):
    root = queue(tmp_path); link = os.link
    def collide(source, target, **kwargs):
        target.write_text("other task")
        link(source, target, **kwargs)
    monkeypatch.setattr(promoter.os, "link", collide)
    with pytest.raises(FileExistsError):
        promoter.promote(root, "eval.sh", lambda: {"status": "ready"}, True)
    assert (root / "pending/eval.sh").read_text() == "other task"
    assert (root / "hold_cpu/eval.sh").is_file()


def test_crash_left_held_alias_does_not_republish(tmp_path):
    root = queue(tmp_path)
    os.link(root / "hold_cpu/eval.sh", root / "running/eval.sh.node.123")
    assert promoter.promote(root, "eval.sh", lambda: pytest.fail("existing hard link"), True)["status"] == "already_dispatched"


@pytest.mark.parametrize("sampler_status,promoter_status", [(0, 0), (0, 7), (3, 0), (3, 7)])
def test_footer_preserves_sampler_status(tmp_path, sampler_status, promoter_status):
    marker = tmp_path / "promotion marker"
    command = [sys.executable, "-c", f"from pathlib import Path;Path({str(marker)!r}).touch();raise SystemExit({promoter_status})"]
    script = f"sh -c 'exit {sampler_status}'\n" + promoter.footer(command)
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == sampler_status
    assert marker.exists() == (sampler_status == 0)


def test_footer_handles_errexit_without_masking_failure(tmp_path):
    command = [sys.executable, "-c", "raise SystemExit(7)"]
    result = subprocess.run(["bash", "-c", "set -e\ntrue\n" + promoter.footer(command)], capture_output=True)
    assert result.returncode == 0
