"""Synthetic report fixtures only; no learned scores or weights are generated."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import pytest

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
import merge_extra_reports as merger
from extra_report_identity import IDENTITY_FILE, POLICY, REPORT_FILES, sha256, write_json


def fixture(folder, metrics, seeds=(0,), metric_offset=0):
    folder.mkdir()
    rows, identities = [], []
    for method in ("area", "sca_nb"):
        for prompt in ("p0", "p1"):
            for seed in seeds:
                row = dict(method=method, res="8x8", prompt_id=prompt, seed=seed,
                           path=f"/synthetic/{method}/{prompt}_{seed}.png")
                value = (method == "sca_nb") + int(prompt[-1]) + seed / 4 + metric_offset
                row["metrics"] = {metric: value + 1 for metric in metrics if metric != "is"}
                rows.append(row)
                proof = {k: str(row[k]) for k in merger.FIELDS[:4]}
                proof.update(path=row["path"], image_bytes=12)
                for field in ("image_sha256", "sidecar_sha256", "config_hash"):
                    proof[field] = hashlib.sha256(row["path"].encode()).hexdigest()
                proof["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
                identities.append(proof)
    summary, paired = [], []
    for method in ("area", "sca_nb"):
        group = [r for r in rows if r["method"] == method]
        for metric in metrics:
            value = 2.5 if metric == "is" else statistics.mean(r["metrics"][metric] for r in group)
            stat = dict(method=method, res="8x8", metric=metric, value=value,
                        ci_low=value - 0.1, ci_high=value + 0.1, n_prompts=2, bootstrap_valid=4)
            if metric == "is":
                stat.update(n_images=len(group), split_std=0.25)
            summary.append(stat)
            if method != "area":
                value = 0.0 if metric == "is" else 1.0
                stat = dict(method=method, res="8x8", reference_method="area", metric=metric,
                    delta_definition="method_minus_reference", value=value, ci_low=value - 0.1,
                    ci_high=value + 0.1, n_prompts=2, bootstrap_valid=4)
                if metric == "is":
                    stat["n_images"] = len(group)
                paired.append(stat)
    report = dict(config=dict(reference_method="area", bootstrap=4, bootstrap_seed=0,
                    iqa_long_side=1024, is_splits=2, out=str(folder)),
        metrics=list(metrics), versions={"torch": "synthetic", "timm": "old" if metrics == ["ir"] else "new"},
        provenance={m: {"model": "synthetic fixture", "load_seconds": 0.1} for m in metrics},
        errors=[], complete=True, per_image=rows, summary=summary, paired_deltas=paired,
        uncertainty="synthetic CI fixture", paired_note="method minus reference", is_note="synthetic set statistic")
    manifest = dict(policy=POLICY, phase="complete", identities=identities,
                    environment=report["versions"], source_hashes={})
    save(folder, report, manifest)
    return report, manifest


def save(folder, report, manifest):
    merger.write_csv(folder / "extra_per_image.csv", [{**{k: r[k] for k in merger.FIELDS}, **r["metrics"]} for r in report["per_image"]])
    merger.write_csv(folder / "extra_summary.csv", report["summary"])
    merger.write_csv(folder / "extra_paired_deltas.csv", report["paired_deltas"])
    write_json(folder / "extra_metrics.json", report)
    manifest["report_files"] = {name: sha256(folder / name) for name in REPORT_FILES}
    write_json(folder / IDENTITY_FILE, manifest)


def test_merge_preserves_statistics_sources_and_paper_reader(tmp_path):
    ir, extra = tmp_path / "ir", tmp_path / "extra"
    original, _ = fixture(ir, ["ir"])
    others, _ = fixture(extra, ["aesthetic", "hpsv2", "hpsv21", "clipiqa", "arniqa", "is"])
    out = tmp_path / "merged"
    result = merger.merge([ir, extra], out)
    assert result["complete"] and len(result["per_image"]) == 4
    assert result["versions"] == {"torch": "synthetic"}
    assert [s["environment"]["timm"] for s in result["merge"]["sources"]] == ["old", "new"]
    assert next(r for r in result["summary"] if r["metric"] == "ir") == original["summary"][0]
    assert [r for r in result["summary"] if r["metric"] == "is"] == [r for r in others["summary"] if r["metric"] == "is"]
    merger.load_source(out)
    # Exercise the actual paper table reader with all ten expected scalar columns.
    base = tmp_path / "base"
    base.mkdir()
    base_rows = [{**{k: r[k] for k in merger.FIELDS}, **{m: 3.0 for m in ("clip", "pickscore", "musiq", "musiq_crop1024")}}
                 for r in original["per_image"]]
    merger.write_csv(base / "per_image.csv", base_rows)
    merger.write_csv(base / "summary.csv", [dict(method=m, res="8x8", metric=k, value=3.0, n_prompts=2)
        for m in ("area", "sca_nb") for k in ("clip", "pickscore", "musiq", "musiq_crop1024")])
    write_json(base / "metrics.json", {"complete": True, "all_metrics_available": True, "errors": []})
    spec = importlib.util.spec_from_file_location("table_reader", REPRO.parent / "paper/figures/src/gen_final_tables.py")
    table = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(table)
    means, _ = table.per_image(base, out, ("area", "sca_nb"), ("p0", "p1"), "8x8")
    assert len(means) == 20 and means[("area", "ir")] == 1.5


@pytest.mark.parametrize("field", ["image_sha256", "prompt_sha256", "sidecar_sha256", "config_hash", "path", "seed", "method", "res"])
def test_reject_changed_identity(tmp_path, field):
    a, b = tmp_path / "a", tmp_path / "b"
    fixture(a, ["ir"])
    report, proof = fixture(b, ["aesthetic"])
    item = proof["identities"][0]
    item[field] = "f" * 64 if field.endswith("sha256") or field == "config_hash" else "changed"
    save(b, report, proof)
    with pytest.raises(ValueError):
        merger.merge([a, b], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_reject_missing_seed_even_when_each_source_is_complete(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    fixture(a, ["ir"], seeds=(0, 1))
    fixture(b, ["aesthetic"], seeds=(0,))
    with pytest.raises(ValueError, match="identity mismatch"):
        merger.merge([a, b], tmp_path / "bad")


def test_overlap_identical_is_allowed_but_changed_scores_rejected(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    fixture(a, ["ir"], seeds=(0, 1))
    fixture(b, ["ir"], seeds=(0, 1))
    fixture(c, ["ir"], seeds=(0, 1), metric_offset=1)
    result = merger.merge([a, b], tmp_path / "same")
    assert result["metrics"] == ["ir"] and len(result["merge"]["metric_sources"]["ir"]) == 2
    with pytest.raises(ValueError, match="Conflicting per-image"):
        merger.merge([a, c], tmp_path / "bad")


@pytest.mark.parametrize("change", ["incomplete", "missing_value", "wrong_mean", "wrong_protocol", "wrong_interval", "duplicate"])
def test_reject_inconsistent_reports(tmp_path, change):
    a, b = tmp_path / "a", tmp_path / "b"
    fixture(a, ["ir"])
    report, proof = fixture(b, ["aesthetic"])
    if change == "incomplete":
        report["complete"] = False
    elif change == "missing_value":
        report["per_image"][0]["metrics"] = {}
    elif change == "wrong_mean":
        report["summary"][0]["value"] += 1
    elif change == "wrong_protocol":
        report["config"]["bootstrap_seed"] = 17
    elif change == "wrong_interval":
        report["summary"][0]["ci_low"] = 100
    else:
        report["per_image"].append(copy.deepcopy(report["per_image"][0]))
    save(b, report, proof)
    with pytest.raises(ValueError):
        merger.merge([a, b], tmp_path / "bad")


def test_reject_changed_file_and_missing_identity_proof(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    fixture(a, ["ir"])
    fixture(b, ["aesthetic"])
    (b / "extra_summary.csv").write_text("edited\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        merger.merge([a, b], tmp_path / "bad")
    # A historical report cannot acquire scoring-time proof from a path alone.
    (b / IDENTITY_FILE).write_text("{}")
    with pytest.raises(ValueError, match="before/after"):
        merger.merge([a, b], tmp_path / "bad")
