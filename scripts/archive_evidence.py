"""Archive compact paper evidence and audit prompt pairing without loading models.

Run with the existing sca uv environment on a compute node. Bulk image/feature caches
stay on scratch. Source hashes describe the files at archival time, not necessarily
the evaluator revision at generation time; recorded metric configs remain authoritative.
"""
import argparse
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scratch', type=Path, default=Path('/scratch/09032/saini_2/sca'))
    ap.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    a = ap.parse_args()
    archive = a.results / 'evidence'
    archive.mkdir(parents=True, exist_ok=True)
    files, audits, sidecars = [], [], {}

    def copy(src, rel):
        dst = archive / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files.append({'source': str(src), 'archive': str(rel), 'bytes': dst.stat().st_size,
                      'sha256': sha(dst)})
        return dst

    names = ('per_image.csv', 'summary.csv', 'paired_deltas.csv', 'metrics.json',
             'extra_per_image.csv', 'extra_summary.csv', 'extra_paired_deltas.csv', 'extra_metrics.json')
    upcoming = {'e2_6144': (100, {'area', 'sca_nb'}),
                'e2_8192': (12, {'area', 'sca_nb'}),
                'e9_krea': (100, {'area', 'sca_krea', 'dype', 'sega_official'})}
    ready = set()
    for experiment, (expected, methods) in upcoming.items():
        valid = True
        pair_sets = []
        prompt_file = a.scratch / ('hi12.jsonl' if experiment == 'e2_8192' else 'eval100.jsonl')
        prompt_ids = {str(json.loads(line)['id']) for line in prompt_file.read_text().splitlines() if line}
        for suffix, prefix in (('base', ''), ('extra', 'extra_')):
            folder = a.scratch / 'eval' / f'{experiment}_{suffix}'
            metadata = folder / (prefix + 'metrics.json')
            per_image = folder / (prefix + 'per_image.csv')
            if not metadata.exists() or not per_image.exists():
                valid = False
                break
            meta = json.loads(metadata.read_text())
            if not meta.get('complete') or meta.get('all_metrics_available') is False:
                valid = False
                break
            groups = defaultdict(set)
            for row in csv.DictReader(per_image.open()):
                groups[row['method']].add((row['res'], row['seed'], row['prompt_id']))
            if set(groups) != methods or any(len(v) != expected for v in groups.values()):
                valid = False
                break
            if any({key[2] for key in values} != prompt_ids for values in groups.values()):
                valid = False
                break
            pair_sets.extend(groups.values())
        if valid and len({tuple(sorted(s)) for s in pair_sets}) == 1:
            ready.add(experiment)
    for directory in sorted((a.scratch / 'eval').iterdir()):
        if not directory.is_dir():
            continue
        experiment = directory.name.rsplit('_', 1)[0]
        if experiment in upcoming and experiment not in ready:
            continue
        prefix = 'extra_' if directory.name.endswith('_extra') else ''
        metadata = directory / (prefix + 'metrics.json')
        if not metadata.exists() or not json.loads(metadata.read_text()).get('complete'):
            continue
        for name in names:
            src = directory / name
            if not src.exists():
                continue
            dst = copy(src, Path('eval') / directory.name / name)
            if name not in ('per_image.csv', 'extra_per_image.csv'):
                continue
            rows = list(csv.DictReader(dst.open()))
            groups = defaultdict(list)
            for row in rows:
                groups[(row['method'], row['res'], row['seed'])].append(row)
                path = Path(row['path']).with_suffix('.json')
                if str(path) not in sidecars:
                    sidecars[str(path)] = json.loads(path.read_text()) if path.exists() else None
            for (method, res, seed), rr in sorted(groups.items()):
                ids = [r['prompt_id'] for r in rr]
                duplicates = [pid for pid, count in Counter(ids).items() if count > 1]
                if duplicates:
                    raise ValueError(f'{directory.name}/{name}: duplicate keys {duplicates}')
                audits.append({'dataset': directory.name, 'file': name, 'method': method,
                               'res': res, 'seed': seed, 'n_rows': len(rr), 'ids': sorted(ids)})

    for name in ('dev4.jsonl', 'abl50.jsonl', 'eval100.jsonl', 'hi12.jsonl', 'hrgen200.jsonl', 'lat3.jsonl'):
        src = a.scratch / name
        if src.exists():
            copy(src, Path('prompts') / name)
    refs = Path('/scratch/09032/saini_2/constancy_rope/prompts')
    for name in ('aesthetic4k_eval.jsonl', 'aesthetic4k_refs.jsonl'):
        if (refs / name).exists():
            copy(refs / name, Path('prompts') / name)
    for stem in ('k1_1024', 'k1_1280b', 'k1_1024_n4', 'k1_1024_n8', 'k1_1024_n16',
                 'krea_1024', 'flux2_1024'):
        for ext in ('.json', '.npz'):
            src = a.scratch / 'calib' / (stem + ext)
            if src.exists():
                copy(src, Path('calib') / src.name)
    payload = ''.join(
        json.dumps({'source': p, 'sidecar': obj}, sort_keys=True) + '\n'
        for p, obj in sorted(sidecars.items())).encode()
    with (archive / 'generation_sidecars.jsonl.gz').open('wb') as fh:
        fh.write(gzip.compress(payload, mtime=0))
    (archive / 'coverage.json').write_text(json.dumps(audits, indent=2) + '\n')
    (archive / 'manifest.json').write_text(json.dumps({
        'archived_utc': datetime.now(timezone.utc).isoformat(), 'files': files,
        'generation_sidecars': len(sidecars),
        'generation_sidecars_sha256': sha(archive / 'generation_sidecars.jsonl.gz'),
        'coverage_sha256': sha(archive / 'coverage.json'),
        'missing_sidecars': [p for p, obj in sidecars.items() if obj is None],
        'scope': 'Numeric evidence, prompt manifests, calibration tables; no images or feature caches.'
    }, indent=2) + '\n')

    # Preserve familiar flat summary names and independently recompute paired means/CIs
    # using the existing script, only within a single resolution and run seed.
    comparisons = {'e4_hrgen': [('sca_nb', 'area')],
                   'budget': [('sca_g7', 'sca_nb'), ('sca_s56', 'sca_nb'),
                              ('ultraimage', 'sca_nb'), ('sca_s56', 'ultraimage'),
                              ('sca_g7', 'ultraimage')]}
    for experiment in ready:
        method = 'sca_krea' if experiment == 'e9_krea' else 'sca_nb'
        refs = ['area', 'dype', 'sega_official'] if experiment == 'e9_krea' else ['area']
        comparisons[experiment] = [(method, ref) for ref in refs]
    for experiment, pairs in comparisons.items():
        inputs = []
        for suffix, prefix in (('base', ''), ('extra', 'extra_')):
            folder = archive / 'eval' / f'{experiment}_{suffix}'
            for name in ('summary.csv', 'paired_deltas.csv'):
                src = folder / (prefix + name)
                if src.exists():
                    shutil.copy2(src, a.results / f'{experiment}_{suffix}_{name}')
            src = folder / (prefix + 'per_image.csv')
            if src.exists():
                rows = list(csv.DictReader(src.open()))
                assert len({(r['res'], r['seed']) for r in rows}) == 1
                inputs.append(str(src))
        for method, reference in pairs:
            subprocess.run([sys.executable, str(Path(__file__).with_name('paired_bootstrap.py')),
                            '--method', method, '--reference', reference, '--out',
                            str(a.results / f'{experiment}_{method}_vs_{reference}.csv'), *inputs], check=True)
    print(json.dumps({'archived_files': len(files), 'bytes': sum(f['bytes'] for f in files),
                      'coverage_groups': len(audits), 'sidecars': len(sidecars),
                      'missing_sidecars': sum(obj is None for obj in sidecars.values()),
                      'new_experiments_ready': sorted(ready)}))


if __name__ == '__main__':
    main()
