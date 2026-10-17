"""Completion-gated galleries must not select from partial or mislabeled pools."""

import copy
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper/figures/src"))
import fig_phase2 as gallery


def source(root, pid, *, suffix="0123456789", **overrides):
    root.mkdir(parents=True, exist_ok=True)
    image = root / f"{pid}_{suffix}_seed0.png"
    Image.new("RGB", (32, 32), (20, 40, 60)).save(image)
    config = dict(prompt_id=pid, prompt=f"prompt {pid}", res="32x32", seed=0, steps=28, guidance=4.5)
    config["model"] = ("black-forest-labs/FLUX.1-Krea-dev" if root.name in ("krea_area", "krea_sca")
                       else "black-forest-labs/FLUX.1-dev")
    config.update(overrides)
    image.with_suffix(".json").write_text(json.dumps({"complete": True, "config": config}))
    return image


def pool(tmp_path, monkeypatch, tag="6144_complete"):
    ids = [f"p{i:03d}" for i in range(100)]
    spec = copy.deepcopy(gallery.SPECS[tag])
    manifest = tmp_path / "eval100.jsonl"
    manifest.write_text("".join(json.dumps({"id": pid, "prompt": f"prompt {pid}"}) + "\n" for pid in ids))
    spec.update(res="32x32", prompt_manifest=manifest)
    if "provenance_receipts" in spec:
        receipt = tmp_path / "receipt.json"
        receipt.write_text(json.dumps({"transformer_snapshot": "8162a9c7b05a641be098422bf2fcf335615c2f28",
            "coverage": {m: {"selected_prompts": 100} for m in ("dype", "sega_official")}}))
        spec["provenance_receipts"] = [receipt]
    monkeypatch.setitem(gallery.SPECS, tag, spec)
    monkeypatch.setattr(gallery.imgsel, "dev4_ids", lambda: set())
    monkeypatch.setattr(gallery.imgsel, "FIGDIR", str(tmp_path / "figures"))
    roots = {}
    for method in spec["methods"]:
        roots[method] = tmp_path / method
        monkeypatch.setitem(gallery.imgsel.ROOTS, method, str(roots[method]))
        for pid in ids:
            source(roots[method], pid)
    return ids, roots


def test_exact_100_ids_not_just_count_and_no_partial_artifact(tmp_path, monkeypatch):
    ids, roots = pool(tmp_path, monkeypatch)
    # A wrong extra prompt cannot replace the missing evaluation prompt.
    missing = roots["sca"] / f"{ids[-1]}_0123456789_seed0.png"
    missing.with_suffix(".json").write_text('{"complete": false, "config": {"seed": 0}}')
    source(roots["sca"], "outside")
    with pytest.raises(ValueError, match="paired=99/100"):
        gallery.build("6144_complete")
    assert not (tmp_path / "figures").exists()
    source(roots["sca"], ids[-1])
    _, eligible, provenance = gallery.ready_pool("6144_complete")
    assert eligible == ids and provenance["count"] == 100
    assert gallery.select_prompts(reversed(eligible), 4) == ["p038", "p046", "p053", "p066"]


@pytest.mark.parametrize("override, message", [
    ({"prompt": "wrong text"}, "Prompt differs"),
    ({"res": "64x64"}, "identity/resolution mismatch"),
    ({"guidance": 3.5}, "Unexpected sampling protocol"),
    ({"model": "wrong-checkpoint"}, "Unexpected base/model identity"),
])
def test_complete_file_with_wrong_metadata_is_rejected(tmp_path, monkeypatch, override, message):
    ids, roots = pool(tmp_path, monkeypatch)
    source(roots["sca"], ids[0], **override)
    with pytest.raises(ValueError, match=message):
        gallery.ready_pool("6144_complete")


def test_duplicate_complete_prompt_is_not_silently_selected(tmp_path, monkeypatch):
    ids, roots = pool(tmp_path, monkeypatch)
    source(roots["sca"], ids[0], suffix="aaaaaaaaaa")
    with pytest.raises(ValueError, match="Duplicate completed"):
        gallery.ready_pool("6144_complete")


def test_krea_requires_all_four_methods_and_checkpoint_receipt(tmp_path, monkeypatch):
    ids, roots = pool(tmp_path, monkeypatch, "krea_complete")
    missing = roots["krea_sega"] / f"{ids[-1]}_0123456789_seed0.png"
    missing.with_suffix(".json").write_text('{"complete": false, "config": {"seed": 0}}')
    with pytest.raises(ValueError, match="krea_sega=99/100"):
        gallery.ready_pool("krea_complete")
    source(roots["krea_sega"], ids[-1])
    _, eligible, _ = gallery.ready_pool("krea_complete")
    assert eligible == ids
    gallery.SPECS["krea_complete"]["provenance_receipts"] = [tmp_path / "not_archived.json"]
    with pytest.raises(ValueError, match="required provenance receipt"):
        gallery.ready_pool("krea_complete")
