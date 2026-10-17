"""Synthetic rejection tests for bounded restoration comparison, no learned models."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("base_parity", Path(__file__).parents[1] / "check_metric_base_parity.py")
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


@pytest.fixture
def reports(tmp_path):
    for name in ("inherited", "restored"):
        directory = tmp_path / name
        directory.mkdir()
        np.savez_compressed(directory / "features.npz", record0_whole=np.ones((1, 2048), np.float32))
        for metric in ("clip", "pickscore", "musiq", "features"):
            parity.write_json(directory / f"{metric}.json", {
                "metric": metric, "manifest_sha256": "same-inputs",
                "provenance": {"model": metric}, "values": [{metric: 2.0}],
                "feature_sha256": parity.sha(directory / "features.npz") if metric == "features" else None})
    return SimpleNamespace(inherited=str(tmp_path / "inherited"),
                           restored=str(tmp_path / "restored"), out=str(tmp_path / "comparison.json"))


def change(args, metric, field, value):
    path = Path(args.restored) / f"{metric}.json"
    document = json.loads(path.read_text())
    document[field] = value
    parity.write_json(path, document)


def test_identical_reports_pass(reports):
    parity.compare(reports)
    result = json.loads(Path(reports.out).read_text())
    assert result["passed"]
    assert result["protocol"]["atol"] == 1e-5
    assert result["comparisons"]["features"]["count"] == 2048


def test_different_inputs_refused(reports):
    change(reports, "clip", "manifest_sha256", "different")
    with pytest.raises(ValueError, match="Different input"):
        parity.compare(reports)


def test_different_model_refused(reports):
    change(reports, "pickscore", "provenance", {"model": "other"})
    with pytest.raises(ValueError, match="Different model"):
        parity.compare(reports)


def test_nonfinite_scores_refused(reports):
    change(reports, "musiq", "values", [{"musiq": float("nan")}])
    with pytest.raises(ValueError, match="Invalid numerical"):
        parity.compare(reports)


def test_changed_feature_artifact_refused(reports):
    np.savez_compressed(Path(reports.restored) / "features.npz", record0_whole=np.zeros((1, 2048), np.float32))
    with pytest.raises(ValueError, match="Feature artifact changed"):
        parity.compare(reports)


def test_score_difference_fails_fixed_tolerance(reports):
    change(reports, "clip", "values", [{"clip": 2.001}])
    with pytest.raises(SystemExit) as failure:
        parity.compare(reports)
    assert failure.value.code == 1
    assert not json.loads(Path(reports.out).read_text())["passed"]


def test_changed_input_refused(tmp_path, monkeypatch):
    source = tmp_path / "image.png"
    source.write_bytes(b"original image")
    monkeypatch.setattr(parity, "source_hashes", lambda: {"source": "fixed"})
    document = {"protocol": parity.PROTOCOL, "sources": {"source": "fixed"},
                "records": [{"path": str(source), "image_sha256": "wrong",
                             "musiq_path": str(source), "musiq_sha256": "wrong"}]}
    with pytest.raises(ValueError, match="Input changed"):
        parity.verify_inputs(document)


def test_changed_protocol_refused(monkeypatch):
    monkeypatch.setattr(parity, "source_hashes", lambda: {})
    with pytest.raises(ValueError, match="Protocol or code"):
        parity.verify_inputs({"protocol": {"atol": 100}, "sources": {}, "records": []})
