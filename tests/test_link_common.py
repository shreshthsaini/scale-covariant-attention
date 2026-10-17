"""Evaluation linking preserves prior evidence and rejects incomplete coverage."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/link_common.py'
SPEC = importlib.util.spec_from_file_location('link_common', SCRIPT)
LINK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LINK)


def invoke(tmp_path, minimum):
    prompts = tmp_path / 'prompts.jsonl'
    prompts.write_text(json.dumps({'id': 'one'}) + '\n')
    images = tmp_path / 'images'
    images.mkdir(exist_ok=True)
    stem = LINK.image_stem('one', 0)
    (images / (stem + '.png')).write_bytes(b'unchanged image bytes')
    (images / (stem + '.json')).write_text('{"complete":true}')
    return subprocess.run([sys.executable, str(SCRIPT), '--root', str(tmp_path / 'root'),
                           '--prompts', str(prompts), '--exclude', '', '--res', '4096x4096',
                           '--min-count', str(minimum), 'area=' + str(images)],
                          text=True, capture_output=True)


def test_insufficient_coverage_leaves_prior_root_untouched(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'retained.txt').write_text('prior evidence')
    result = invoke(tmp_path, 2)
    assert result.returncode != 0
    assert (root / 'retained.txt').read_text() == 'prior evidence'
    assert not list(tmp_path.glob('root.building-*'))


def test_replacement_archives_prior_root_and_keeps_image_targets(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'retained.txt').write_text('prior evidence')
    result = invoke(tmp_path, 1)
    assert result.returncode == 0, result.stderr
    previous, = tmp_path.glob('root.previous-*')
    assert (previous / 'retained.txt').read_text() == 'prior evidence'
    linked, = (root / 'area/4096x4096').glob('*.png')
    assert linked.is_symlink()
    assert linked.read_bytes() == b'unchanged image bytes'
    assert linked.resolve().parent == tmp_path / 'images'
