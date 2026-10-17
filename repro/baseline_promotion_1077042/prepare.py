#!/usr/bin/env python3
"""Prepare reviewed, duration-bounded baseline candidates without queue mutation."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / 'results/evidence/continuation/baseline_smoke_deployment_20261017.json'
METHODS = ('sigma_dev_matched', 'hrdit_dev_author')
STAGES = {'pilot12': (1, 12), 'pilot46': (12, 46)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(deployment=DEPLOY):
    receipt = json.loads(deployment.read_text())
    root = Path(receipt['frozen_root'])
    for name, digest in receipt['files'].items():
        if sha(root / name) != digest:
            raise ValueError(f'Frozen generation file changed: {name}')
    spec = importlib.util.spec_from_file_location('promotion_frozen_run', root / 'run.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    # The immutable upstream task writer imports its sibling as `run`.
    previous = sys.modules.get('run')
    sys.modules['run'] = adapter
    try:
        spec = importlib.util.spec_from_file_location('promotion_frozen_tasks', root / 'prepare_tasks.py')
        writer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(writer)
    finally:
        if previous is None:
            sys.modules.pop('run', None)
        else:
            sys.modules['run'] = previous
    return receipt, adapter, writer


def inspect_outputs(receipt, adapter, stage):
    required, size = STAGES[stage]
    rows = adapter.read_prompts(adapter.ROOT / 'pilot46.jsonl', 0, size)
    sources = adapter.verify_sources()
    records = []
    for method in METHODS:
        args = SimpleNamespace(profile=method, prompts=adapter.ROOT / 'pilot46.jsonl', seed=0)
        for index, row in enumerate(rows):
            image = Path(receipt['output_root']) / method / '4096x4096' / (adapter.stem_for(row['id'], 0) + '.png')
            sidecar = image.with_suffix('.json')
            complete = adapter.check_resume(image, sidecar, adapter.make_config(args, row, sources))
            if not complete:
                if index < required:
                    raise ValueError(f'Prerequisite incomplete: {method}/{row["id"]}')
                continue
            from PIL import Image
            with Image.open(image) as opened:
                opened.verify()
            with Image.open(image) as opened:
                opened.load()
            record = json.loads(sidecar.read_text())
            seconds = record.get('wall_seconds')
            if (type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds <= 0
                    or record.get('seconds') != seconds):
                raise ValueError(f'Invalid pipeline timing: {sidecar}')
            if record.get('finite_checks') != ['latents_each_step', 'vae_decoder_before_pixel_conversion']:
                raise ValueError(f'Missing finite-generation checks: {sidecar}')
            records.append(dict(method=method, index=index, prompt_id=str(row['id']),
                image=str(image), image_sha256=sha(image), sidecar=str(sidecar),
                sidecar_sha256=sha(sidecar), pipeline_seconds=seconds, prerequisite=index < required))
    return rows, records


def review_template(stage, deployment, records):
    return dict(stage=stage, deployment_sha256=sha(deployment), reviewer='', notes='',
        observations=[dict(method=r['method'], prompt_id=r['prompt_id'],
            image_sha256=r['image_sha256'], sidecar_sha256=r['sidecar_sha256'],
            fidelity_reviewed=False, image_valid=False, notes='')
            for r in records if r['prerequisite']])


def validate_review(review, expected):
    if (review.get('stage') != expected['stage']
            or review.get('deployment_sha256') != expected['deployment_sha256']
            or not str(review.get('reviewer', '')).strip()):
        raise ValueError('Review stage, deployment, or reviewer is missing or different')
    observations = review.get('observations', [])
    if not isinstance(observations, list):
        raise ValueError('Review observations must be a list')
    keys = [(x['method'], x['prompt_id']) for x in observations]
    wanted = {(x['method'], x['prompt_id']): x for x in expected['observations']}
    if len(set(keys)) != len(keys) or set(keys) != set(wanted):
        raise ValueError('Review must cover exactly the prerequisite outputs once')
    for row in observations:
        reference = wanted[row['method'], row['prompt_id']]
        if (any(row.get(k) != reference[k] for k in ('image_sha256', 'sidecar_sha256'))
                or row.get('fidelity_reviewed') is not True or row.get('image_valid') is not True
                or not str(row.get('notes', '')).strip()):
            raise ValueError(f'Output needs valid, hash-bound fidelity review: {row["method"]}/{row["prompt_id"]}')


def task_ranges(indices, worst_seconds, budget, startup):
    if (not math.isfinite(worst_seconds) or worst_seconds <= 0
            or type(budget) is not int or type(startup) is not int or startup < 0 or budget <= startup):
        raise ValueError('Invalid duration budget')
    capacity = math.floor((budget - startup) / (2 * worst_seconds))
    if capacity < 1:
        raise ValueError(f'One prompt needs at least {math.ceil(2 * worst_seconds + startup)} seconds; use a larger fleet window')
    ranges = []
    for index in sorted(indices):
        if ranges and index == ranges[-1][1] and ranges[-1][1] - ranges[-1][0] < capacity:
            ranges[-1][1] += 1
        else:
            ranges.append([index, index + 1])
    return [(start, end, math.ceil(2 * worst_seconds * (end - start) + startup)) for start, end in ranges]


def plan(receipt, records, stage, budget, startup):
    _, size = STAGES[stage]
    result = {}
    for method in METHODS:
        observed = [x for x in records if x['method'] == method]
        completed = {x['index'] for x in observed}
        missing = set(range(size)) - completed
        durations = [x['pipeline_seconds'] for x in observed]
        mean, worst = statistics.mean(durations), max(durations)
        ranges = task_ranges(missing, worst, budget, startup) if missing else []
        result[method] = dict(observed_images=len(observed), remaining_images=len(missing),
            mean_pipeline_seconds=mean, maximum_pipeline_seconds=worst,
            incremental_pipeline_gpu_hours=dict(central=len(missing) * mean / 3600,
                planning_range=[len(missing) * mean / 3600, 2 * len(missing) * worst / 3600]),
            startup_allowance_seconds_per_task=startup,
            task_reservation_hours=sum(x[2] for x in ranges) / 3600,
            ranges=[dict(start=a, end=b, minimum_seconds=c) for a, b, c in ranges])
    return result


def write_candidates(directory, receipt, writer, stage, result, review_path, deployment=DEPLOY):
    # Validate and render everything before creating any candidate directory.
    scripts = []
    for method, data in result.items():
        for item in data['ranges']:
            start, end, minimum = (item[k] for k in ('start', 'end', 'minimum_seconds'))
            name = f'a_b_{stage}_{method}_{start:03d}_{end:03d}.sh'
            scripts.append((name, writer.task_text(Path(receipt['frozen_root']), Path(receipt['output_root']),
                method, start, end, minimum)))
    directory.mkdir(parents=True, exist_ok=False)
    tasks = []
    for name, script in scripts:
        path = directory / name
        with path.open('x') as target:
            target.write(script)
        tasks.append(dict(path=str(path.resolve()), sha256=sha(path)))
    report = dict(status='candidates_only_not_enqueued', stage=stage,
        checked_utc=datetime.now(timezone.utc).isoformat(), deployment_sha256=sha(deployment),
        review_path=str(review_path.resolve()), review_sha256=sha(review_path),
        source_sha256=sha(__file__), profiles=result, tasks=tasks,
        limitations=[
            'Planning range is not a confidence interval; observed pipeline times exclude model loading.',
            'Startup allowance is explicit planning headroom, not a measured model-load time.',
            'Dispatcher adds 120 seconds cleanup beyond each SCA_MIN_SECONDS admission value.',
            'Evaluation, retries, SU charge rates and pending-job reservations require separate budget review.',
            'Inspect task ownership in pending/running/done before publication; this helper never enqueues.',
            'Full191 promotion requires a separately frozen manifest and provenance audit.'])
    (directory / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--deployment', type=Path, default=DEPLOY)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--review-template', type=Path)
    parser.add_argument('--tasks-dir', type=Path)
    parser.add_argument('--max-task-seconds', type=int, default=6000,
        help='Admission budget excluding dispatcher cleanup120s; must fit the fleet window')
    parser.add_argument('--startup-seconds', type=int, default=600,
        help='Unmeasured model-load/setup allowance added per task')
    args = parser.parse_args()
    if args.review_template is None and (args.review is None or args.tasks_dir is None):
        parser.error('--review and --tasks-dir required unless writing --review-template')
    receipt, adapter, writer = load_frozen(args.deployment)
    _, records = inspect_outputs(receipt, adapter, args.stage)
    template = review_template(args.stage, args.deployment, records)
    if args.review_template is not None:
        with args.review_template.open('x') as target:
            target.write(json.dumps(template, indent=2) + '\n')
        print(json.dumps(dict(status='review_required', path=str(args.review_template))))
        return
    validate_review(json.loads(args.review.read_text()), template)
    result = plan(receipt, records, args.stage, args.max_task_seconds, args.startup_seconds)
    report = write_candidates(args.tasks_dir, receipt, writer, args.stage, result, args.review, args.deployment)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
