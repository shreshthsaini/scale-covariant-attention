"""Check that archived generation code and wrapper helpers stand alone on WORK."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_generation_snapshot_matches_frozen_manifest():
    snapshot = ROOT / "repro/generation_snapshot"
    manifest = json.loads((snapshot / "SOURCE.json").read_text())
    assert manifest["source_commit"] == "c05ebdcccddafd6b8fe4ec2e14247d72775df9a1"
    for row in manifest["files"]:
        assert hashlib.sha256((snapshot / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    fingerprint = hashlib.sha256()
    for name in manifest["fingerprint_files"]:
        fingerprint.update(name.encode())
        fingerprint.update((snapshot / name).read_bytes())
    assert fingerprint.hexdigest() == manifest["generation_implementation_sha256"]
    assert (snapshot / "crope/utils.py").read_bytes() == (
        ROOT / "repro/evaluator_snapshot/crope/utils.py").read_bytes()


@pytest.mark.parametrize("script", ["run_area.py", "run_ultraimage.py"])
def test_wrappers_import_vendored_helpers_in_fresh_process(script):
    program = (
        "import json,runpy,sys; "
        "module=runpy.run_path(sys.argv[1]); "
        "print(json.dumps({'root':str(module['CROPE']), 'module':sys.modules['crope.utils'].__file__, "
        "'seed':module['image_seed']('reproduction',7), 'stem':module['image_stem']('reproduction',7)}))"
    )
    output = subprocess.check_output([sys.executable, "-c", program, str(ROOT / "scripts" / script)], text=True)
    value = json.loads(output)
    assert Path(value["root"]) == ROOT / "repro/evaluator_snapshot"
    assert Path(value["module"]) == ROOT / "repro/evaluator_snapshot/crope/utils.py"
    assert value["stem"].endswith("_seed7")


def test_archived_model_metadata_hashes():
    report = json.loads((ROOT / "repro/model_snapshots.json").read_text())
    for model in report["models"]:
        assert len(model["observed_cached_revision"]) == 40
        for row in model["files"]:
            if "archived_metadata" in row:
                data = (ROOT / "repro" / row["archived_metadata"]).read_bytes()
                assert hashlib.sha256(data).hexdigest() == row["metadata_sha256"]
