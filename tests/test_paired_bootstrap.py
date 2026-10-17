"""Prevent silent overwrite or selective pairing in paper confidence intervals."""
import csv
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/paired_bootstrap.py'


def run(tmp_path, rows):
    source, out = tmp_path / 'in.csv', tmp_path / 'out.csv'
    with source.open('w') as fh:
        writer = csv.DictWriter(fh, fieldnames=['method', 'res', 'seed', 'prompt_id', 'ir'])
        writer.writeheader()
        writer.writerows(rows)
    result = subprocess.run([sys.executable, str(SCRIPT), '--method', 'sca', '--reference',
                             'area', '--draws', '100', '--out', str(out), str(source)],
                            capture_output=True, text=True)
    return result, list(csv.DictReader(out.open())) if out.exists() else []


def fixture_rows():
    return [dict(method=m, res=res, seed=seed, prompt_id=pid,
                 ir=(10 * seed + scale) if m == 'sca' else 0)
            for res, scale in [('2048x2048', 2), ('4096x4096', 4)]
            for seed in (0, 1) for pid in ('p1', 'p2') for m in ('sca', 'area')]


def test_seeds_and_resolutions_are_separate(tmp_path):
    result, rows = run(tmp_path, fixture_rows())
    assert result.returncode == 0, result.stderr
    assert {(r['res'], int(r['seed'])): float(r['delta']) for r in rows} == {
        ('2048x2048', 0): 2, ('2048x2048', 1): 12,
        ('4096x4096', 0): 4, ('4096x4096', 1): 14}
    assert all(r['n_prompts'] == '2' for r in rows)


def test_missing_pair_is_rejected(tmp_path):
    result, _ = run(tmp_path, fixture_rows()[1:])
    assert result.returncode != 0
    assert 'Unpaired metric' in result.stderr


def test_duplicate_is_rejected(tmp_path):
    rows = fixture_rows()
    result, _ = run(tmp_path, rows + [rows[0]])
    assert result.returncode != 0
    assert 'Duplicate metric' in result.stderr
