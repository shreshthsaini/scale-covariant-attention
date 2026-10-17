"""Validate or study FID/KID from archived features without scratch or model loading.

--policy legacy reproduces historical path-overwritten crops; rows uses each row's
own cached crop features at the same 195-row weights. --reference-clusters images
resamples unique reference image identities while retaining their row multiplicity.
--policy unique is a separately labeled reference-pool sensitivity.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'repro/evaluator_snapshot'))
from crope.metrics.statistics import distribution_bootstrap, fid, kid


def load_bank(root, entry, policy='legacy', clusters='rows'):
    key = 'bank' if policy == 'legacy' else 'row_bank'
    file = root / entry[key]
    assert hashlib.sha256(file.read_bytes()).hexdigest() == entry[key + '_sha256' if key != 'bank' else 'sha256']
    with np.load(file, allow_pickle=False) as saved:
        features, ids, seeds = saved['features'], saved['prompt_ids'], saved['seeds']
    info = {(r['prompt_id'], str(r['seed'])): r for r in entry['records']}
    first_per_image = {}
    if policy == 'unique':
        for row in sorted(entry['records'], key=lambda r: r['manifest_order']):
            first_per_image.setdefault(row['image_signature']['path'], row['prompt_id'])
    bank = defaultdict(list)
    for array, pid, seed in zip(features, ids, seeds):
        row = info[str(pid), str(seed)]
        image = row['image_signature']['path']
        if policy == 'unique' and first_per_image[image] != pid:
            continue
        cluster = image if clusters == 'images' else str(pid)
        bank[cluster].append(array)
    return {key: np.concatenate(value) for key, value in bank.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO / 'results/evidence/distributions')
    parser.add_argument('--experiments', nargs='+')
    parser.add_argument('--policy', choices=['legacy', 'rows', 'unique'], default='legacy')
    parser.add_argument('--reference-clusters', choices=['rows', 'images'], default='rows')
    parser.add_argument('--metrics', nargs='+', default=['fid', 'kid', 'patch_fid', 'patch_kid'])
    parser.add_argument('--draws', type=int, default=0)
    parser.add_argument('--reference-method')
    parser.add_argument('--methods', nargs='+', help='Only compute these method comparisons; baseline remains loaded.')
    args = parser.parse_args()
    out = args.root / 'checks'
    out.mkdir(exist_ok=True)
    experiments = args.experiments or [p.name.removesuffix('_banks.json') for p in sorted(args.root.glob('*_banks.json'))]
    for experiment in experiments:
        index = json.loads((args.root / f'{experiment}_banks.json').read_text())
        report = json.loads((REPO / 'results/evidence/eval' / experiment / 'metrics.json').read_text())
        available = {row['metric'] for row in report['summary']}
        methods = sorted({r['method'] for r in report['per_image']})
        selected_methods = args.methods or methods
        assert set(selected_methods) <= set(methods), (experiment, selected_methods, methods)
        reference = args.reference_method or report['config']['reference_method']
        for metric in [m for m in args.metrics if m in available]:
            start = time.monotonic()
            patch = metric.startswith('patch_')
            entry = index['banks'][f'reference_images:patch={patch}']
            real = load_bank(args.root, entry, args.policy, args.reference_clusters)
            banks = {method: load_bank(args.root, index['banks'][f'{method}:patch={patch}']) for method in methods}
            statistic = kid if metric.endswith('kid') else fid
            combine = lambda bank: np.concatenate([bank[key] for key in sorted(bank)])
            y = combine(real)
            if args.draws and reference not in banks:
                raise ValueError(f'{experiment}: historical reference {reference} missing; supply --reference-method')
            baseline_value = statistic(combine(banks[reference]), y) if reference in banks else None
            rows = []
            for method in selected_methods:
                value = statistic(combine(banks[method]), y)
                saved = next(r for r in report['summary'] if r['method'] == method and r['metric'] == metric)
                row = dict(experiment=experiment, policy=args.policy, reference_clusters=args.reference_clusters,
                           method=method, reference_method=reference, metric=metric, value=value,
                           historical_value=saved['value'], difference_from_historical=value-saved['value'],
                           paired_delta=value-baseline_value if baseline_value is not None else None, ci_low=None, ci_high=None,
                           n_generated_prompts=len(banks[method]), n_reference_clusters=len(real),
                           n_reference_rows=len(y) // (4 if patch else 1), draws=args.draws,
                           historical_ci_max_abs_error=None)
                if args.draws and method != reference:
                    estimate = distribution_bootstrap(banks[method], real, statistic, draws=args.draws,
                        seed=report['config']['bootstrap_seed'], baseline=banks[reference])
                    row.update(paired_delta=estimate['value'], ci_low=estimate['ci_low'], ci_high=estimate['ci_high'])
                    old = next(r for r in report['paired_deltas'] if r['method'] == method and r['metric'] == metric)
                    errors = [abs(estimate[k] - old[k]) for k in ('value', 'ci_low', 'ci_high')]
                    row['historical_ci_max_abs_error'] = max(errors)
                    if args.policy == 'legacy' and args.reference_clusters == 'rows' and args.draws == report['config']['bootstrap']:
                        assert max(errors) < 1e-7, (experiment, metric, errors)
                rows.append(row)
            suffix = '_methods-' + '-'.join(selected_methods) if args.methods else ''
            file = out / f'{experiment}_{args.policy}_{args.reference_clusters}_{metric}_{args.draws}{suffix}.csv'
            with file.open('w') as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps({'experiment': experiment, 'policy': args.policy, 'clusters': args.reference_clusters,
                              'metric': metric, 'draws': args.draws, 'seconds': time.monotonic()-start,
                              'results': rows}), flush=True)


if __name__ == '__main__':
    main()
