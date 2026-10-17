"""Log a bounded campaign snapshot to WandB, then close the client.

Runs on a compute node. Uses configured WandB credentials without exporting them.
No resident monitor, scheduler polling loop, model loads, or image uploads.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    import wandb

    scratch = Path('/scratch/09032/saini_2/sca')
    repo = Path(__file__).resolve().parents[1]
    directory = scratch / 'wandb'
    directory.mkdir(exist_ok=True)
    queue = scratch / 'taskq'
    tasks = {folder.name: sorted(p.name for p in folder.iterdir() if p.is_file())
             for folder in queue.iterdir() if folder.is_dir()}
    values = {f'queue/{name}': len(names) for name, names in tasks.items()}
    current_hosts = {match[0] for name in tasks.get('running', [])
                     if (match := re.search(r'c\d+-\d+', name))}
    values['campaign/active_workers'] = len(tasks.get('running', []))
    now = datetime.now(timezone.utc)
    for host in sorted(current_hosts):
        files = sorted((scratch / 'telemetry').glob(f'{host}_*.csv'), key=lambda p: p.stat().st_mtime)
        if not files:
            continue
        path = files[-1]
        with path.open('rb') as handle:
            handle.seek(max(0, path.stat().st_size - 8192))
            last = handle.read().decode(errors='replace').strip().splitlines()[-1].split(',')
        if len(last) == 5:
            try:
                for key, value in zip(('memory_mib', 'total_memory_mib', 'utilization_percent', 'power_w'), last[1:]):
                    values[f'gpu/{host}/{key}'] = float(value)
                values[f'gpu/{host}/sample_age_seconds'] = now.timestamp() - path.stat().st_mtime
            except ValueError:
                pass
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    mode = 'offline' if args.offline else 'online'
    run = wandb.init(project='scale-covariant-attention', id='sca-phase2-20261017',
                     resume='allow' if mode == 'online' else None, mode=mode, dir=str(directory),
                     job_type='campaign-monitor', settings=wandb.Settings(init_timeout=45),
                     config={'protocol': 'FLUX.1-dev28steps guidance4.5; registered Phase2 extensions',
                             'reference_assembly': 'reference_prompt_row_v2'})
    try:
        run.log(values)
        run.summary.update({'last_snapshot_utc': now.isoformat(), 'source_revision': revision,
                            'running_tasks': tasks.get('running', []),
                            'active_gpu_hosts': sorted(current_hosts),
                            'held_long_tasks': tasks.get('hold_long', [])})
        receipt = {'timestamp_utc': now.isoformat(), 'mode': mode, 'run_id': run.id,
                   'url': run.url if mode == 'online' else None, 'source_revision': revision,
                   'monitor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'values': values}
    finally:
        run.finish()
    output = repo / 'results/evidence/tracking'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'latest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
