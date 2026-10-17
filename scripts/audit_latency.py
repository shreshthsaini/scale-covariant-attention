"""Verify warm-up ordering and archive clean batch-one latency measurements.

Historical SCA sidecars do not record first_call_in_process. The uninterrupted task
log must therefore show the excluded warm-up and all three timed images in order.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, default=Path('/scratch/09032/saini_2/sca'))
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    prompts = [json.loads(line)['id'] for line in (args.scratch / 'lat3.jsonl').read_text().splitlines() if line]
    warm = 'sca_latency_warmup'
    expected = [warm, *prompts]
    archive = args.results / 'evidence/latency'
    archive.mkdir(parents=True, exist_ok=True)
    samples, groups, missing = [], [], []
    log_snapshots = {}
    for index, size in enumerate((2048, 3072, 4096, 6144, 8192)):
        res = f'{size}x{size}'
        for method in ('sca_nb', 'area', 'full') if size != 8192 else ('sca_nb', 'area'):
            folder = args.scratch / 'out/latency_curve' / res / method
            sidecars = {}
            for path in folder.glob('*.json'):
                data = json.loads(path.read_text())
                pid = data['config']['prompt_id']
                if pid in sidecars:
                    raise ValueError(f'Duplicate latency prompt: {folder}/{pid}')
                sidecars[pid] = (path, data)
            task = f'a_y00{index}_latency_{res}' + (f'_{method}' if index >= 3 else '')
            # V2 keeps immutable attempt logs; its alias must take precedence
            # over a preserved historical attempt using the legacy filename.
            log = args.scratch / 'logs' / f'task_{task}.latest.log'
            if not log.exists():
                log = args.scratch / 'logs' / f'task_{task}.log'
            if set(sidecars) != set(expected) or not log.exists():
                missing.append(f'{res}/{method}')
                continue
            if log not in log_snapshots:
                log_snapshots[log] = log.read_bytes()
            log_bytes = log_snapshots[log]
            text = log_bytes.decode(errors='replace')
            matches = re.findall(r'saved (\S+) (\S+) (\d+x\d+) seconds=([0-9.]+)', text)
            events = [(pid, float(sec)) for pid, label, shape, sec in matches if label == method and shape == res]
            if [pid for pid, _ in events] != expected:
                raise ValueError(f'No uninterrupted warm-up followed by three samples: {task}/{method}')
            if re.search(rf'skip \S+ {re.escape(method)} {res}', text):
                raise ValueError(f'Resumed latency process needs separate provenance: {task}/{method}')
            for pid, logged_seconds in events:
                path, data = sidecars[pid]
                config = data['config']
                if not data['complete'] or config['res'] != res or config['method'] != method:
                    raise ValueError(f'Invalid completed sidecar: {path}')
                if config['steps'] != 28 or config['guidance'] != 4.5 or config['seed'] != 0:
                    raise ValueError(f'Unexpected latency protocol: {path}')
                if abs(data['seconds'] - logged_seconds) > 0.0006:
                    raise ValueError(f'Log and sidecar timing differ: {path}')
                if pid != warm and data.get('first_call_in_process') is True:
                    raise ValueError(f'Timed sample was cold: {path}')
                destination = archive / res / method / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                if pid != warm:
                    samples.append({'res': res, 'method': method, 'prompt_id': pid,
                                    'seconds': data['seconds'], 'peak_memory_bytes': data.get('peak_memory_bytes'),
                                    'warmup_verified': True})
            (archive / log.name).write_bytes(log_bytes)
            seconds = [sidecars[pid][1]['seconds'] for pid in prompts]
            groups.append({'res': res, 'method': method, 'n': len(seconds),
                           'median_seconds': statistics.median(seconds),
                           'minimum_seconds': min(seconds), 'maximum_seconds': max(seconds),
                           'log_sha256': hashlib.sha256(log_bytes).hexdigest()})
    if missing and not args.allow_partial:
        raise ValueError(f'Incomplete latency groups: {missing}')
    for name, rows in (('latency_curve.csv', samples), ('latency_curve_summary.csv', groups)):
        if rows:
            with (args.results / name).open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    report = {'complete': not missing, 'groups': groups, 'missing': missing,
              'protocol': 'One generated warm-up, followed by three timed images in the same process; batch1.'}
    (archive / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
