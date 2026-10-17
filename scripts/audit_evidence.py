"""Recheck archived scalar means, prompt coverage, and calibration least squares."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1] / 'results/evidence'
    manifest = json.loads((root / 'manifest.json').read_text())
    for entry in manifest['files']:
        assert hashlib.sha256((root / entry['archive']).read_bytes()).hexdigest() == entry['sha256'], entry
    assert hashlib.sha256((root / 'generation_sidecars.jsonl.gz').read_bytes()).hexdigest() == manifest['generation_sidecars_sha256']
    means, calibration, coverage = [], [], []
    dev = {json.loads(x)['id'] for x in (root / 'prompts/dev4.jsonl').read_text().splitlines()}
    for folder in sorted((root / 'eval').iterdir()):
        prefix = 'extra_' if folder.name.endswith('_extra') else ''
        src, summary = folder / (prefix + 'per_image.csv'), folder / (prefix + 'summary.csv')
        if not src.exists() or not summary.exists():
            continue
        rows = list(csv.DictReader(src.open()))
        for s in csv.DictReader(summary.open()):
            vals = [float(r[s['metric']]) for r in rows if r['method'] == s['method']
                    and r['res'] == s['res'] and r.get(s['metric'], '')]
            if vals:
                error = abs(float(np.mean(vals)) - float(s['value']))
                assert np.isfinite(vals).all() and error < 1e-8, (folder, s, error)
                means.append({'dataset': folder.name, 'method': s['method'],
                              'metric': s['metric'], 'n': len(vals), 'abs_error': error})
        groups = {}
        for row in rows:
            groups.setdefault(row['method'], set()).add((row['res'], row['seed'], row['prompt_id']))
        same = len({tuple(sorted(ids)) for ids in groups.values()}) == 1
        coverage.append({'dataset': folder.name, 'all_methods_same_ids': same,
                         'methods': {m: {'n': len(ids), 'development_ids': sorted({i[2] for i in ids} & dev)}
                                     for m, ids in groups.items()}})

    for file in sorted((root / 'calib').glob('*.npz')):
        if not file.with_suffix('.json').exists():
            continue
        arrays = np.load(file)
        report = json.loads(file.with_suffix('.json').read_text())
        pop = arrays['populated']
        indices = np.flatnonzero(pop)
        nearest = indices[np.abs(np.arange(pop.size)[:, None] - indices[None]).argmin(1)]
        for i, scale in enumerate(arrays['scales']):
            stats = arrays['stats'][i]
            A, B = stats[0], stats[1]
            pi = np.clip(np.where(A > 0, B / np.where(A > 0, A, 1), 1), 0, 1)[..., nearest]
            error = float(np.max(np.abs(pi - arrays[f's{scale}_pi_head'])))
            rel = float((stats[2] - 2*pi*B + pi*pi*A).sum() / stats[4].sum())
            report_error = abs(rel - report['scales'][str(scale)]['rel_err']['per_head'])
            assert error < 1e-12 and report_error < 1e-12, (file, scale, error, report_error)
            calibration.append({'table': file.name, 'scale': int(scale), 'pi_max_error': error,
                                'relative_squared_error': rel, 'report_abs_error': report_error})
    result = {'archive_hashes_checked': len(manifest['files']) + 1,
              'scalar_means_checked': len(means), 'max_mean_abs_error': max(r['abs_error'] for r in means),
              'coverage': coverage, 'calibration': calibration,
              'scope': 'Scalar means and calibration are rederived; FID/KID/IS bootstrap is preserved, not rerun.'}
    (root / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'means_checked': len(means), 'max_mean_abs_error': result['max_mean_abs_error'],
                      'calibration_scales_checked': len(calibration),
                      'unequal_coverage': [c['dataset'] for c in coverage if not c['all_methods_same_ids']]}))


if __name__ == '__main__':
    main()
