"""Regression: distinct prompt crops must survive shared reference image paths."""

import argparse
import json

import numpy as np
from PIL import Image
import pytest

from repro import evaluate_corrected as corrected
from crope.metrics import models
from crope.metrics.evaluate import Evaluation as HistoricalEvaluation
from crope.metrics.image import crop_boxes
from crope.metrics.statistics import distribution_bootstrap, fid, kid


class TinyFeatures:
    """Real image/crop means replace a learned extractor in this assembly test."""
    provenance = {"backend": "test_rgb_means", "feature_dim": 3}

    def __init__(self, device):
        assert device == "cpu"

    def extract(self, record, *, patch, crop_seed, crops):
        with Image.open(record["path"]) as opened:
            image = opened.convert("RGB")
            boxes = crop_boxes(image.size, record["prompt_id"], seed=crop_seed, count=crops) if patch else [None]
            return np.stack([
                np.asarray(image if box is None else image.crop(box), dtype=np.float64).mean((0, 1)) / 255
                for box in boxes
            ])


def fixture_args(tmp_path):
    size = 1152
    ramp = np.linspace(0, 255, size).astype(np.uint8)
    array = np.stack([np.broadcast_to(ramp[None, :], (size, size)),
                      np.broadcast_to(ramp[:, None], (size, size)),
                      np.full((size, size), 91, dtype=np.uint8)], axis=-1)
    root = tmp_path / "generated"
    for j, method in enumerate(("baseline", "candidate")):
        folder = root / method
        folder.mkdir(parents=True)
        for i in range(2):
            path = folder / f"{i}.png"
            Image.fromarray(np.roll(array, 40 + i * 180 + j * 90, axis=1)).save(path)
            path.with_suffix(".json").write_text(json.dumps({"complete": True, "seconds": 1,
                "config": {"method": method, "res": f"{size}x{size}", "seed": 0,
                           "prompt_id": str(i), "prompt": f"prompt {i}"}}))
    shared = tmp_path / "shared_reference.png"
    Image.fromarray(array).save(shared)
    refs = tmp_path / "references.jsonl"
    refs.write_text("\n".join(json.dumps({"id": f"reference_{i}", "path": str(shared)}) for i in range(2)))
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("\n".join(json.dumps({"id": str(i), "prompt": f"prompt {i}"}) for i in range(2)))
    return argparse.Namespace(root=str(root), prompts=str(prompts), references=str(refs),
        out=str(tmp_path / "evaluation"), metrics="fid,kid,patch_fid,patch_kid", res=None,
        reference_method="baseline", device="cpu", bootstrap=20, bootstrap_seed=7,
        crop_seed=0, crops=4, allow_missing=False, no_wandb=True, threads=1)


def test_reference_crop_rows_survive_aliases_and_cache_resume(tmp_path, monkeypatch):
    args = fixture_args(tmp_path)
    monkeypatch.setattr(models, "InceptionFeatures", TinyFeatures)
    historical = HistoricalEvaluation(args)
    historical.distributions()
    assert not historical.errors
    assert len({r["path"] for r in historical.references}) == 1
    extractor = TinyFeatures("cpu")
    real = {r["prompt_id"]: extractor.extract(r, patch=True, crop_seed=0, crops=4)
            for r in historical.references}
    assert not np.array_equal(*real.values())
    generated = {method: {r["prompt_id"]: extractor.extract(r, patch=True, crop_seed=0, crops=4)
                         for r in historical.records if r["method"] == method}
                 for method in ("baseline", "candidate")}

    class NoExtraction:
        def __init__(self, device):
            raise AssertionError("Corrected evaluator must reuse the existing per-record feature caches")

    monkeypatch.setattr(corrected, "InceptionFeatures", NoExtraction)
    current = corrected.Evaluation(args)
    current.distributions()
    assert not current.errors
    for metric, statistic in (("patch_fid", fid), ("patch_kid", kid)):
        observed = next(r for r in current.summaries if r["metric"] == metric and r["method"] == "candidate")
        expected = distribution_bootstrap(generated["candidate"], real, statistic, draws=20, seed=7)
        for field, value in expected.items():
            assert observed[field] == pytest.approx(value)
        assert observed["n_reference_prompts"] == 2
        assert observed["n_reference_feature_rows"] == 8
        old = next(r for r in historical.summaries if r["metric"] == metric and r["method"] == "candidate")
        assert observed["value"] != pytest.approx(old["value"], abs=1e-10)
        delta = next(r for r in current.paired if r["metric"] == metric)
        paired = distribution_bootstrap(generated["candidate"], real, statistic,
                                        baseline=generated["baseline"], draws=20, seed=7)
        for field, value in paired.items():
            assert delta[field] == pytest.approx(value)
        assert current.status[metric]["feature_assembly"] == "reference_prompt_row_v2"
    for metric in ("fid", "kid"):
        assert [r for r in current.summaries if r["metric"] == metric] == [
            r for r in historical.summaries if r["metric"] == metric]
        assert [r for r in current.paired if r["metric"] == metric] == [
            r for r in historical.paired if r["metric"] == metric]


def test_duplicate_reference_ids_are_reported_not_silently_collapsed(tmp_path):
    args = fixture_args(tmp_path)
    evaluation = corrected.Evaluation(args)
    evaluation.references[1]["prompt_id"] = evaluation.references[0]["prompt_id"]
    evaluation.distributions()
    assert len(evaluation.errors) == 4
    assert all("Reference prompt IDs must be unique" in row["error"] for row in evaluation.errors)
