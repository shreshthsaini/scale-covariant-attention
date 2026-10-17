"""Event-driven, single-owner continuation for the SCA campaign.

Freeze this module, batch file, prompt and schema before submission. A successor
is armed before Codex starts, so timeouts and node failures leave a queued run.
Only queued controller successors can be cancelled by this module.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time

ROOT = Path('/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention')
SCRATCH = Path('/scratch/09032/saini_2/sca')
CONTROL = SCRATCH / 'continuation'
FROZEN = Path(__file__).resolve().parent
ACCOUNT = 'ASC26048'
CHECKS = ('e2_6144', 'e3_latency', 'e9_krea', 'fresh_generation_gpu',
          'metric_environment_parity', 'manuscript_reconciled',
          'figures_reviewed', 'independent_source_build')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def command(args):
    result = subprocess.run(args, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'{args[0]} exited {result.returncode}: {result.stderr[-2000:]}')
    return result.stdout


def jobs():
    output = command(['squeue', '-u', 'saini_2', '-h', '-o', '%i|%j|%T'])
    return {job: (name, state) for line in output.splitlines()
            for job, name, state in [line.strip().split('|')]}


def gpu_dependency(records):
    terms = []
    for job, (name, state) in sorted(records.items()):
        if not re.fullmatch(r'SCA[0-9]+', name) or not job.isdigit():
            continue
        if state == 'PENDING':
            terms.append(f'after:{job}')
        terms.append(f'afterany:{job}')
    return '?'.join(terms)


def submit(dependency=None):
    """One short SSH, light login commands only, budget check at each submission."""
    local = command(['ps', '-u', str(os.getuid()), '-o', 'comm=']).splitlines()
    if sum(name.strip() == 'ssh' for name in local) >= 3:
        raise RuntimeError('SSH cap reached; refusing another connection')
    args = ['sbatch', '--parsable']
    if dependency:
        if not re.fullmatch(r'(after|afterany):[0-9]+(\?(after|afterany):[0-9]+)*', dependency):
            raise ValueError('Invalid dependency')
        args.append('--dependency=' + dependency)
    args.append(str(FROZEN / 'continuation.sbatch'))
    # 1600/40000 SU is the 96% stop boundary. Reserve a conservative 576 SU
    # (144 cores x four hours) rather than assuming a cheaper CPU billing rate.
    remote = '''set -eu
count=$(ps -u "$USER" --no-headers | wc -l)
[ "$count" -lt 40 ] || { echo 'Login process guard refused submission' >&2; exit 70; }
balance=$(/usr/local/etc/taccinfo)
printf '%s\n' "$balance"
available=$(printf '%s\n' "$balance" | awk '$2=="ASC26048" {print $3}')
awk -v value="$available" 'BEGIN {if (value !~ /^[0-9]+([.][0-9]+)?$/ || value <= 2176) exit 1}' || {
  echo 'Budget unavailable or within stop boundary plus reservation' >&2; exit 71;
}
''' + shlex.join(args)
    result = command(['ssh', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=10',
                      '-o', 'ServerAliveCountMax=2', 'login2', remote])
    match = re.search(r'^([0-9]+)(?:;[^\n]+)?$', result, flags=re.M)
    if not match:
        raise RuntimeError('No job ID in submission response: ' + result)
    job = match.group(1)
    save(CONTROL / 'submissions' / f'{job}.json',
         dict(job_id=job, dependency=dependency, submitted=time.time(), receipt=result))
    return job


def cancel_successor(job):
    # No live allocation may be cancelled here. The ID is our saved successor.
    record = jobs().get(job)
    if record is None:
        return
    if record != ('SCAC', 'PENDING'):
        raise RuntimeError(f'Refusing to cancel non-pending controller {job}: {record}')
    command(['scancel', job])


def completion_errors(root=ROOT, scratch=SCRATCH):
    errors = []
    try:
        report = json.loads((root / 'results/evidence/project_completion.json').read_text())
        if report.get('complete') is not True:
            errors.append('Completion report is not true')
        for key in CHECKS:
            if report.get('checks', {}).get(key) is not True:
                errors.append('Missing verified check: ' + key)
        artifacts = report.get('artifacts', [])
        paths = set()
        for item in artifacts:
            relative = Path(item['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Completion artifacts must be repository-relative')
            paths.add(str(relative))
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != item['sha256']:
                errors.append('Artifact hash mismatch: ' + str(relative))
        for required in ('paper/main.pdf', 'paper/sca_overleaf.zip'):
            if required not in paths:
                errors.append('Missing delivery artifact: ' + required)
        delivery = json.loads((root / 'paper/delivery_receipt.json').read_text())
        if delivery.get('standalone_build_passed') is not True:
            errors.append('Independent source build missing')
        for key, name in (('paper_sha256', 'main.pdf'), ('source_zip_sha256', 'sca_overleaf.zip')):
            if hashlib.sha256((root / 'paper' / name).read_bytes()).hexdigest() != delivery.get(key):
                errors.append('Stale delivery receipt: ' + name)
        manuscript = (root / 'paper/main.tex').read_text()
        if re.search(r'\\todo\{E[239](?:\b|,)', manuscript):
            errors.append('Experiment TODO remains in manuscript')
        smoke = json.loads((root / 'results/evidence/environment/fresh_gpu_smoke.json').read_text())
        if smoke.get('status') != 'passed' or smoke.get('timed_out') is not False or smoke.get('worker_exit_code') != 0:
            errors.append('Fresh GPU smoke has not passed')
        with (root / 'results/latency_curve.csv').open() as stream:
            latency = list(csv.DictReader(stream))
        for resolution, method in (('6144x6144', 'area'), ('6144x6144', 'full'), ('8192x8192', 'area')):
            group = [row for row in latency if row['res'] == resolution and row['method'] == method]
            if len(group) != 3 or any(row['warmup_verified'] != 'True' or float(row['seconds']) <= 0 for row in group):
                errors.append('Missing verified latency group: ' + resolution + '/' + method)
        for experiment in ('e2_6144', 'e9_krea'):
            folder = scratch / 'eval' / (experiment + '_base')
            if not (folder / 'CPU_BOOTSTRAP_COMPLETE').is_file():
                errors.append('Missing final bootstrap marker: ' + experiment)
            metadata = json.loads((folder / 'metrics.json').read_text())
            if metadata.get('complete') is not True or metadata.get('config', {}).get('bootstrap', 0) < 1000:
                errors.append('Unfinished final metrics: ' + experiment)
        for name in ('pending', 'running', 'failed'):
            if any((scratch / 'taskq' / name).glob('*.sh')):
                errors.append('Unfinished task queue: ' + name)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
    return errors


def validate_freeze():
    expected = json.loads((FROZEN / 'SHA256.json').read_text())
    for name, digest in expected.items():
        if hashlib.sha256((FROZEN / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Frozen controller changed: ' + name)


def run():
    validate_freeze()
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit() or not socket.gethostname().startswith('i'):
        raise RuntimeError('Continuation requires a Slurm CPU compute allocation')
    CONTROL.mkdir(parents=True, exist_ok=True)
    with (CONTROL / 'owner.lock').open('a') as owner:
        # Retargeting a pending successor to an already-satisfied GPU event can
        # start it before this owner has returned. Allow that brief handoff.
        lock_deadline = time.time() + 120
        while True:
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() >= lock_deadline:
                    raise RuntimeError('Another continuation worker owns this project')
                time.sleep(10)
        directory = CONTROL / ('job_' + job)
        directory.mkdir(exist_ok=True)
        state = dict(job=job, host=socket.gethostname(), started=time.time(), status='starting')
        save(CONTROL / 'status.json', state)
        if (CONTROL / 'STOP').exists():
            state.update(status='stopped_by_request')
            save(CONTROL / 'status.json', state)
            return
        # Arm before agent work. Slurm then handles timeout/node failure as well
        # as normal exit. Repeated pre-agent crashes stop after three attempts.
        failures_path = CONTROL / 'consecutive_failures.json'
        failures = json.loads(failures_path.read_text()) if failures_path.exists() else {'count': 0}
        if failures['count'] >= 3:
            state.update(status='blocked', error='Three consecutive worker failures; inspect job logs')
            save(CONTROL / 'status.json', state)
            return
        save(failures_path, {'count': failures['count'] + 1})
        successor = submit('afterany:' + job)
        state.update(successor=successor, status='working')
        save(CONTROL / 'status.json', state)
        record = command(['scontrol', 'show', 'job', job, '-o'])
        end = re.search(r'\bEndTime=([^ ]+)', record)
        if end is None:
            raise RuntimeError('Cannot obtain actual allocation end time')
        deadline = int(command(['date', '-d', end.group(1), '+%s']).strip())
        iteration = 0
        invalid = 0
        while time.time() < deadline - 1200:
            if (CONTROL / 'STOP').exists():
                cancel_successor(successor)
                state.update(status='stopped_by_request')
                save(CONTROL / 'status.json', state)
                return
            iteration += 1
            prefix = directory / f'turn_{iteration:03d}'
            output = prefix.with_suffix('.final.json')
            prompt = (FROZEN / 'continuation_prompt.txt').read_text()
            prompt += f'\nThis allocation ends at {end.group(1)}. Current job={job}; successor={successor}.\n'
            prompt += 'Read continuation/status.json on scratch for the previous unit or error.\n'
            args = ['/home1/09032/saini_2/.local/bin/codex', 'exec',
                    '-m', 'gpt-6.1-sol', '-c', 'model_reasoning_effort="high"',
                    '-c', 'model_context_window=200000',
                    '-c', 'model_auto_compact_token_limit=170000',
                    '--json', '--output-schema', str(FROZEN / 'continuation_schema.json'),
                    '-C', str(ROOT), '-o', str(output), '-']
            # Explicit commands and no ignored configs/rules/hook-trust bypass.
            # Slurm owns cleanup of this allocation's complete process tree.
            with prefix.with_suffix('.events.jsonl').open('w') as stdout, \
                    prefix.with_suffix('.stderr.log').open('w') as stderr:
                child = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=stdout,
                                         stderr=stderr, text=True, cwd=ROOT)
                save(prefix.with_suffix('.process.json'), dict(pid=child.pid, parent=os.getpid(),
                     job=job, started=time.time(), command=args))
                child.communicate(prompt)
            try:
                if child.returncode:
                    raise RuntimeError(f'Codex exited {child.returncode}')
                result = json.loads(output.read_text())
                status = result['status']
                if status not in ('continue', 'waiting', 'complete', 'blocked'):
                    raise ValueError('Invalid structured status')
                invalid = 0
                save(failures_path, {'count': 0})
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                invalid += 1
                state.update(status='retrying', error=str(exc), iteration=iteration)
                save(CONTROL / 'status.json', state)
                if invalid >= 3:
                    state.update(status='blocked')
                    save(CONTROL / 'status.json', state)
                    cancel_successor(successor)
                    return
                time.sleep(30)
                continue
            state.update(result=result, iteration=iteration, updated=time.time(), status=status)
            if status == 'complete':
                errors = completion_errors()
                if errors:
                    state.update(status='continue', completion_rejected=errors)
                else:
                    cancel_successor(successor)
                    save(CONTROL / 'status.json', state)
                    save(ROOT / 'results/evidence/continuation_complete.json', state)
                    return
            elif status == 'waiting':
                dependency = gpu_dependency(jobs())
                if dependency:
                    command(['scontrol', 'update', 'JobId=' + successor,
                             'Dependency=' + dependency])
                    state.update(dependency=dependency)
                    save(CONTROL / 'status.json', state)
                    return
                state.update(status='continue', error='Waiting rejected: no live SCA GPU jobs. Recover work or report concrete blocker.')
            elif status == 'blocked':
                cancel_successor(successor)
                save(CONTROL / 'status.json', state)
                return
            save(CONTROL / 'status.json', state)
            time.sleep(10)
        state.update(status='continuing_on_successor', updated=time.time())
        save(CONTROL / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'submit', 'check-completion'))
    parser.add_argument('--dependency')
    options = parser.parse_args()
    if options.action == 'run':
        try:
            run()
        except Exception as exc:
            save(CONTROL / ('failure_' + os.environ.get('SLURM_JOB_ID', 'unknown') + '.json'),
                 dict(error=str(exc), timestamp=time.time()))
            raise
    elif options.action == 'submit':
        validate_freeze()
        print(submit(options.dependency))
    else:
        print(json.dumps(completion_errors(), indent=2))
