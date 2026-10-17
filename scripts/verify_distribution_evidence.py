"""Archive cached Inception banks and rederive distribution metrics on CPU.

This validates numerical statistics from existing features, not image generation or
encoder inference. Requires the pinned evaluator in repro/evaluator_snapshot. Limit
BLAS threads before invocation, for example OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'repro/evaluator_snapshot'))
from crope.metrics.io import digest, file_signature, load_references
from crope.metrics.statistics import distribution_bootstrap, fid, kid

DEFAULT = ['final4k_nb_base', 'e1_baselines_base', 'e1_ultraimage_base', 'flux2_4k_base',
           'final4k_full_base', 'final_2048x4096_base', 'final_2048x6144_base', 'budget_base',
           'ablnb_base', 'localnb_base', 'ps2_base', 'route_base', 'ablfinal_base',
           'e6_calib_base', 'e7_gate_base', 'e5_seed1_base', 'e5_seed2_base']


def write_csv(path, rows):
    with path.open('w') as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def make_bank(records, patch, report, source, out, *, alias_by_path=False):
    settings = {'schema': report['metric_schema'], 'crop_seed': report['config']['crop_seed'],
                'crops': report['config']['crops'], 'versions': report['versions'],
                'model_revisions': report['model_revisions']}
    arrays, metadata = [], []
    manifest_order = {r['prompt_id']: i for i, r in enumerate(records)}
    last_record = {r['path']: r for r in records}
    records = sorted(records, key=lambda r: (r['prompt_id'], str(r.get('seed', 'reference'))))
    for record in records:
        signature = file_signature(record['path'])
        key = digest({'file': signature, 'prompt': record.get('prompt'), 'prompt_id': record['prompt_id'],
                      'metric': 'inception_features', 'settings': settings,
                      'source_config_hash': record.get('config_hash'), 'extra': {'patch': patch}})
        file = source / 'features' / (key + '.npz')
        if not file.exists():
            raise FileNotFoundError(f'Feature key no longer resolves: {file} for {record["path"]}')
        raw = file.read_bytes()
        with np.load(file, allow_pickle=False) as saved:
            features = saved['features']
            provenance = json.loads(str(saved['provenance']))
        assert features.shape == (4 if patch else 1, 2048)
        arrays.append(features)
        metadata.append({'prompt_id': record['prompt_id'], 'seed': record.get('seed', 'reference'),
                         'manifest_order': manifest_order[record['prompt_id']],
                         'image_signature': signature, 'source_feature': str(file),
                         'source_feature_sha256': hashlib.sha256(raw).hexdigest(),
                         'config_hash': record.get('config_hash'), 'provenance': provenance})
    values = np.stack(arrays)
    ids = np.asarray([r['prompt_id'] for r in records])
    seeds = np.asarray([str(r.get('seed', 'reference')) for r in records])
    raw_identity = hashlib.sha256(values.tobytes() + ids.tobytes() + seeds.tobytes()).hexdigest()
    row_file = out / 'banks' / (raw_identity + '.npz')
    if not row_file.exists():
        np.savez_compressed(row_file, features=values, prompt_ids=ids, seeds=seeds)
    if alias_by_path:
        by_identity = {(r['prompt_id'], str(r.get('seed', 'reference'))): i for i, r in enumerate(records)}
        selected = [by_identity[(last_record[r['path']]['prompt_id'],
                                str(last_record[r['path']].get('seed', 'reference')))] for r in records]
        values = values[selected]
        for row, index in zip(metadata, selected):
            row['effective_feature_source'] = metadata[index]['source_feature']
            row['effective_feature_prompt_id'] = records[index]['prompt_id']
    identity = hashlib.sha256(values.tobytes() + ids.tobytes() + seeds.tobytes()).hexdigest()
    file = out / 'banks' / (identity + '.npz')
    if not file.exists():
        np.savez_compressed(file, features=values, prompt_ids=ids, seeds=seeds)
    bank = {}
    for pid in sorted(set(ids)):
        bank[str(pid)] = values[ids == pid].reshape(-1, 2048)
    return bank, {'bank': str(file.relative_to(out)), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                  'row_bank': str(row_file.relative_to(out)),
                  'row_bank_sha256': hashlib.sha256(row_file.read_bytes()).hexdigest(),
                  'n_prompts': len(bank), 'feature_shape': list(values.shape), 'records': metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, default=Path('/scratch/09032/saini_2/sca'))
    parser.add_argument('--out', type=Path, default=REPO / 'results/evidence/distributions')
    parser.add_argument('--experiments', nargs='+', default=DEFAULT)
    parser.add_argument('--bootstrap-experiment')
    parser.add_argument('--bootstrap-metrics', nargs='+', default=['fid', 'kid', 'patch_fid', 'patch_kid'])
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--reference-policy', choices=['legacy', 'rows', 'unique'], default='legacy')
    args = parser.parse_args()
    (args.out / 'banks').mkdir(parents=True, exist_ok=True)
    point_rows, indices, ci_rows = [], {}, []
    for experiment in args.experiments:
        source = args.scratch / 'eval' / experiment
        report = json.loads((source / 'metrics.json').read_text())
        metrics = [m for m in report['config']['metrics'].split(',') if m in ('fid', 'kid', 'patch_fid', 'patch_kid')]
        if not metrics:
            continue
        assert report['complete'] and report['all_metrics_available'], experiment
        records = report['per_image']
        references = load_references(report['config']['references'])
        if args.reference_policy == 'unique':
            references = list({r['path']: r for r in reversed(references)}.values())
        methods = sorted({r['method'] for r in records})
        banks, entries = {}, {}
        for patch in sorted({m.startswith('patch_') for m in metrics}):
            for method in [*methods, 'reference_images']:
                selected = references if method == 'reference_images' else [r for r in records if r['method'] == method]
                bank, entry = make_bank(selected, patch, report, source, args.out,
                                       alias_by_path=args.reference_policy == 'legacy')
                banks[method, patch] = bank
                entries[f'{method}:patch={patch}'] = entry
        indices[experiment] = {'source_metrics': str(source / 'metrics.json'),
                               'source_metrics_sha256': hashlib.sha256((source / 'metrics.json').read_bytes()).hexdigest(),
                               'config': report['config'], 'banks': entries}
        indices[experiment]['reference_policy'] = args.reference_policy
        for metric in metrics:
            patch = metric.startswith('patch_')
            statistic = kid if metric.endswith('kid') else fid
            real = banks['reference_images', patch]
            y = np.concatenate([real[k] for k in sorted(real)])
            for method in methods:
                generated = banks[method, patch]
                x = np.concatenate([generated[k] for k in sorted(generated)])
                value = statistic(x, y)
                saved = next(r for r in report['summary'] if r['method'] == method and r['metric'] == metric)
                delta = abs(value - saved['value'])
                if delta > 1e-7 and args.reference_policy == 'legacy':
                    raise ValueError(f'{experiment}/{method}/{metric} differs by {delta}')
                point_rows.append({'experiment': experiment, 'method': method, 'metric': metric,
                                   'recomputed': value, 'reported': saved['value'], 'absolute_error': delta,
                                   'n_generated_prompts': len(generated), 'n_reference_prompts': len(real)})
        print(json.dumps({'experiment': experiment, 'point_estimates_checked': len(methods) * len(metrics),
                          'max_abs_error': max(r['absolute_error'] for r in point_rows if r['experiment'] == experiment)}), flush=True)
        (args.out / f'{experiment}_banks.json').write_text(json.dumps(indices[experiment], indent=2) + '\n')
        write_csv(args.out / f'{experiment}_points.csv', [r for r in point_rows if r['experiment'] == experiment])
        if experiment == args.bootstrap_experiment:
            reference = report['config']['reference_method']
            for metric in args.bootstrap_metrics:
                patch = metric.startswith('patch_')
                for method in methods:
                    if method == reference:
                        continue
                    start = time.monotonic()
                    result = distribution_bootstrap(banks[method, patch], banks['reference_images', patch],
                        kid if metric.endswith('kid') else fid, draws=args.draws,
                        seed=report['config']['bootstrap_seed'], baseline=banks[reference, patch])
                    saved = next(r for r in report['paired_deltas'] if r['method'] == method and r['metric'] == metric)
                    errors = {k: abs(result[k] - saved[k]) for k in ('value', 'ci_low', 'ci_high')}
                    if args.reference_policy == 'legacy' and args.draws == report['config']['bootstrap'] and max(errors.values()) > 1e-7:
                        raise ValueError(f'Bootstrap mismatch: {experiment}/{method}/{metric}: {errors}')
                    row = dict(experiment=experiment, method=method, reference=reference, metric=metric,
                               **result, max_original_difference=max(errors.values()),
                               draws=args.draws, compute_seconds=time.monotonic() - start)
                    ci_rows.append(row)
                    print(json.dumps(row), flush=True)
                    write_csv(args.out / f'{experiment}_paired_ci.csv', ci_rows)
    (args.out / 'verification.json').write_text(json.dumps({
        'point_estimates_checked_this_run': len(point_rows),
        'max_absolute_point_error': max(r['absolute_error'] for r in point_rows),
        'scope': 'Cached Inception numerical statistics only; image generation and encoder inference were not rerun.',
        'evaluator_snapshot': str(REPO / 'repro/evaluator_snapshot'),
        'bootstrap_experiment': args.bootstrap_experiment,
        'bootstrap_rows_checked': len(ci_rows), 'bootstrap_draws': args.draws if ci_rows else 0,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
