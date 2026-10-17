"""Reconstruct attributable SCA worker time without charging unrelated allocations.

Closed task spans include loading, evaluation, calibration, failed attempts, and any
waiting inside the task. They measure occupied single-GPU worker time, not GPU kernel
activity. Unclosed spans are censored, never extrapolated. Sidecar generation seconds
overlap worker spans and lack placement timestamps, so are reported separately.
"""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
from collections import Counter, defaultdict
from zoneinfo import ZoneInfo

TZ = ZoneInfo('America/Chicago')
EVENT = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d) (c\d+-\d+[^ ]*) '
                   r'(start|end|requeued) (\S+)(?: rc=(-?\d+))?')
SAFE = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d) '
                  r'(start|done|failed|stale) (\S+)')
HOST = re.compile(r'(c\d+-\d+)')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse_events(text, filename):
    events = []
    for lineno, line in enumerate(text.splitlines(), 1):
        match = EVENT.match(line)
        if match:
            stamp, worker, action, task, rc = match.groups()
        elif (match := SAFE.match(line)) and 'preemptible_' in filename:
            stamp, action, task = match.groups()
            host = HOST.search(filename)
            if not host:
                continue
            worker = host[1] + '-guest-safe'
            task = task.split('.sh')[0] + '.sh'
            rc = '0' if action == 'done' else None
        else:
            continue
        events.append(dict(timestamp=stamp, epoch=datetime.fromisoformat(stamp).replace(tzinfo=TZ).timestamp(),
                           worker=worker, host=HOST.search(worker)[1], action=action,
                           task=task, rc=rc, source=filename, line=lineno))
    return events


def category(task):
    if 'eval' in task:
        return 'evaluation'
    if 'calib' in task or 'merge' in task:
        return 'calibration_or_merge'
    if 'latency' in task or 'bench' in task:
        return 'latency_or_benchmark'
    return 'generation_or_other'


def pair_events(events):
    unique = {}
    for event in events:
        key = tuple(event[k] for k in ('timestamp', 'worker', 'action', 'task', 'rc'))
        unique.setdefault(key, event)
    buckets = defaultdict(list)
    for event in unique.values():
        buckets[(event['epoch'], event['worker'], event['task'])].append(event)
    active, closed, orphan = {}, [], []
    # Resolve equal-second boundaries from lifecycle state, independent of which
    # archived snapshot supplied a duplicated event.
    for _, bucket in sorted(buckets.items()):
        key = (bucket[0]['worker'], bucket[0]['task'])
        end_first = key in active
        for event in sorted(bucket, key=lambda e: (e['action'] == 'start') if end_first else (e['action'] != 'start')):
            key = (event['worker'], event['task'])
            if event['action'] == 'start':
                if key in active:
                    orphan.append(dict(active[key], reason='restarted_without_end'))
                active[key] = event
            elif key not in active:
                orphan.append(dict(event, reason='end_without_start'))
            else:
                start = active.pop(key)
                seconds = event['epoch'] - start['epoch']
                if seconds < 0:
                    raise ValueError((start, event))
                closed.append(dict(host=event['host'], worker=event['worker'], task=event['task'],
                                   start=start['timestamp'], end=event['timestamp'], start_epoch=start['epoch'],
                                   end_epoch=event['epoch'], seconds=seconds, rc=event['rc'],
                                   outcome=event['action'], category=category(event['task']),
                                   start_source=start['source'], end_source=event['source']))
    return closed, list(active.values()), orphan, len(events) - len(unique)


def union_seconds(spans, conservative=False):
    by_host = defaultdict(list)
    for span in spans:
        start = span['start_epoch'] + int(conservative)
        end = span['end_epoch']
        if end > start:
            by_host[span['host']].append((start, end))
    total = 0.0
    for intervals in by_host.values():
        end = float('-inf')
        for start, stop in sorted(intervals):
            total += max(0.0, stop - max(start, end))
            end = max(end, stop)
    return total


def write_csv(path, rows):
    with path.open('w') as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, default=Path('/scratch/09032/saini_2/sca'))
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'results/compute')
    parser.add_argument('--scheduler', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / 'raw'
    raw.mkdir(exist_ok=True)
    snapshot = datetime.now(timezone.utc).isoformat()
    logs = sorted({p for pattern in ('guest*.log', 'fleet*.out', 'worker*.log', 'preemptible*.log')
                   for p in (args.scratch / 'logs').glob(pattern)})
    log_archive = raw / 'worker_logs'
    log_archive.mkdir(exist_ok=True)
    manifest_path = args.out / 'source_manifest.json'
    sources = json.loads(manifest_path.read_text()).get('sources', []) if manifest_path.exists() else []
    events = []
    for path in logs:
        data = path.read_bytes()
        snapshot_name = f'{path.stem}_{digest(data)[:16]}.log'
        (log_archive / snapshot_name).write_bytes(data)
        entry = {'source': str(path), 'snapshot': str(Path('raw/worker_logs') / snapshot_name),
                 'sha256': digest(data), 'bytes': len(data)}
        if entry not in sources:
            sources.append(entry)
    for path in sorted(log_archive.glob('*.log')):
        events.extend(parse_events(path.read_text(errors='replace'), path.name))
    for pattern in ('guest*.log', 'fleet*.out', 'worker*.log', 'preemptible*.log'):
        for path in sorted(raw.glob(pattern)):
            events.extend(parse_events(path.read_text(errors='replace'), path.name))
    closed, censored, orphan, duplicates = pair_events(events)
    write_csv(args.out / 'closed_task_spans.csv', closed)
    (args.out / 'unpaired_events.json').write_text(json.dumps({'censored_starts': censored,
                                                          'unpaired_events': orphan}, indent=2) + '\n')
    span_seconds = union_seconds(closed)
    conservative = union_seconds(closed, conservative=True)

    # Symlinked evaluation roots and imported external-project images cannot add cost.
    seen_paths, seen_artifacts, images, excluded = set(), set(), [], Counter()
    compact_metadata = []
    output_root = (args.scratch / 'out').resolve()
    for path in sorted(output_root.rglob('*.json')):
        resolved = path.resolve()
        if resolved in seen_paths:
            excluded['duplicate_resolved_path'] += 1
            continue
        seen_paths.add(resolved)
        if not resolved.is_relative_to(output_root):
            excluded['external_project_path'] += 1
            continue
        data = path.read_bytes()
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            excluded['unreadable_or_incomplete_json'] += 1
            continue
        cfg = obj.get('config', {})
        seconds = obj.get('seconds')
        if not obj.get('complete') or not isinstance(seconds, (float, int)) or seconds < 0 or not cfg:
            excluded['not_completed_timed_image'] += 1
            continue
        if obj.get('image') and not Path(obj['image']).resolve().is_relative_to(output_root):
            excluded['metadata_points_to_external_project'] += 1
            continue
        if cfg.get('batch_size', 1) != 1:
            excluded['non_unit_batch_requires_separate_accounting'] += 1
            continue
        png_hash = obj.get('png_sha256')
        if not png_hash:
            excluded['missing_image_hash'] += 1
            continue
        # Identical generated artifacts are deliberately counted only once, even if
        # separate reruns really occurred. This keeps the artifact subtotal conservative.
        if png_hash in seen_artifacts:
            excluded['duplicate_image_hash'] += 1
            continue
        seen_artifacts.add(png_hash)
        images.append(dict(source=str(resolved), source_sha256=digest(data), png_sha256=png_hash,
                           method=cfg.get('method'), res=cfg.get('res'), seed=cfg.get('seed'),
                           prompt_id=cfg.get('prompt_id'), seconds=seconds,
                           seconds_scope=obj.get('seconds_scope', 'unspecified')))
        compact_metadata.append({'source': str(resolved), 'complete': True, 'config': cfg,
                                 'png_sha256': png_hash, 'seconds': seconds,
                                 'seconds_scope': obj.get('seconds_scope', 'unspecified')})
    write_csv(args.out / 'generation_seconds.csv', images)
    metadata_bytes = ''.join(json.dumps(x, sort_keys=True) + '\n' for x in compact_metadata).encode()
    (raw / 'generation_metadata.jsonl.gz').write_bytes(gzip.compress(metadata_bytes, mtime=0))
    generation_seconds = sum(row['seconds'] for row in images)

    sched = args.scheduler
    if sched is None:
        candidates = sorted(args.scratch.glob('sca_scheduler_accounting_*.psv'))
        sched = candidates[-1] if candidates else None
    scheduler_records = []
    if sched:
        data = sched.read_bytes()
        (raw / sched.name).write_bytes(data)
        entry = {'source': str(sched), 'snapshot': str(Path('raw') / sched.name),
                 'sha256': digest(data), 'bytes': len(data)}
        if entry not in sources:
            sources.append(entry)
        for line in data.decode().splitlines():
            fields = line.split('|')
            if len(fields) < 7 or not fields[0].isdigit() or not fields[1].startswith('SCA'):
                continue
            job, name, state, nodes, elapsed, start, end = fields[:7]
            scheduler_records.append(dict(job_id=job, name=name, state=state, nodes=int(nodes),
                                          elapsed_seconds=int(elapsed), start=start, end=end))
    allocated_seconds = sum(r['nodes'] * r['elapsed_seconds'] for r in scheduler_records)
    observed_done = {p.name for p in (args.scratch / 'taskq/done').glob('*.sh')}
    running_names = {p.name for p in (args.scratch / 'taskq/running').glob('*.sh.*')}
    logged_done = {s['task'] for s in closed if s['rc'] == '0'}
    failed = [s for s in closed if s['rc'] not in (None, '0') or s['outcome'] in ('failed', 'stale', 'requeued')]
    result = dict(snapshot_utc=snapshot, timezone='America/Chicago', gpu_assumption='One GH200 per worker node',
                  closed_task_attempts=len(closed), failed_or_interrupted_closed_attempts=len(failed),
                  duplicate_log_events_removed=duplicates, worker_log_files=len(logs),
                  worker_closed_gpu_node_hours=span_seconds / 3600,
                  worker_closed_conservative_hours=conservative / 3600,
                  summed_span_hours_before_overlap_removal=sum(s['seconds'] for s in closed) / 3600,
                  failed_or_interrupted_span_hours=union_seconds(failed) / 3600,
                  by_category_hours={c: union_seconds([s for s in closed if s['category'] == c]) / 3600
                                     for c in sorted({s['category'] for s in closed})},
                  censored_start_count=len(censored), unpaired_event_count=len(orphan),
                  censored_starts_still_in_running=sum(e['task'] + '.' + e['worker'] in running_names for e in censored),
                  done_tasks_without_closed_success_span=sorted(observed_done - logged_done),
                  unique_timed_artifacts=len(images), generation_only_hours=generation_seconds / 3600,
                  sidecar_exclusions=dict(excluded), scheduler_dedicated_node_hours=allocated_seconds / 3600,
                  scheduler_records=scheduler_records,
                  attributable_resource_hours_lower_bound=conservative / 3600,
                  counting_rule='Union completed worker intervals by physical node; generation durations and dedicated allocation time are separate overlapping measurements.',
                  limitations=[
                      'Worker spans include model loading, evaluation, calibration, host work, and in-task waits; these are occupied GPU-node hours, not active GPU kernel hours.',
                      'Conservative closed intervals advance each start by one second to allow for timestamp rounding.',
                      'Running or interrupted starts without a matching end are excluded; overwritten or missing logs and manual tasks can increase the true cost.',
                      'Artifact durations exclude loading and PNG I/O where recorded; identical output hashes are counted once, so reruns may be omitted.',
                      'Artifact durations lack host placement and start/end timestamps, so their sum is not independently a lower bound on GPU-node hours and is never added to worker time.',
                      'Scheduler dedicated-node time overlaps worker time and is reported separately. No MF9/CF2 or other shared parent allocation is charged in full.',
                      'Dedicated scheduler accounting is the supplied saved snapshot; refresh it separately after jobs finish.',
                  ])
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / 'source_manifest.json').write_text(json.dumps({'snapshot_utc': snapshot, 'sources': sources,
             'generation_metadata_sha256': digest((raw / 'generation_metadata.jsonl.gz').read_bytes())}, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('snapshot_utc', 'closed_task_attempts', 'worker_closed_gpu_node_hours',
          'worker_closed_conservative_hours', 'generation_only_hours', 'attributable_resource_hours_lower_bound',
          'scheduler_dedicated_node_hours', 'censored_start_count', 'unpaired_event_count')}))


if __name__ == '__main__':
    main()
