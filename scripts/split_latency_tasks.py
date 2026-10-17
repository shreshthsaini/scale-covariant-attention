"""Split untouched large-resolution latency bundles by method without changing samples.

Each task still warms up once and times three images in the same model process.
Tasks stay held until the operator verifies enough allocation time is available.
"""
import json
from pathlib import Path
import subprocess


def main():
    root = Path('/scratch/09032/saini_2/sca')
    queue = root / 'taskq'
    originals = queue / 'hold_original'
    originals.mkdir(exist_ok=True)
    receipt = []
    for res, index in (('6144x6144', 3), ('8192x8192', 4)):
        source = queue / 'hold_long' / f'a_y00{index}_latency_{res}.sh'
        if not source.exists():
            continue
        if list((root / 'out/latency_curve' / res).glob('*/*.json')):
            raise RuntimeError(f'Refusing to split started measurement: {res}')
        lines = source.read_text().splitlines()
        commands = [line for line in lines if '/bin/python scripts/' in line]
        header = [line for line in lines if line.startswith(('export ', 'unset ', 'nvidia-smi ', 'cd '))]
        for command in commands:
            method = command.split(' --out ', 1)[1].split()[0].rsplit('/', 1)[1]
            task = source.with_name(f'a_y00{index}_latency_{res}_{method}.sh')
            if task.exists():
                raise RuntimeError(f'Task already exists: {task}')
            task.write_text('#!/bin/bash\nset -e\n# One excluded warm-up, then three timed images.\n'
                            + '\n'.join(header + [command]) + '\n')
            subprocess.run(['bash', '-n', str(task)], check=True)
            receipt.append({'source': source.name, 'task': task.name, 'method': method,
                            'res': res, 'warmup_id': 'sca_latency_warmup'})
        source.rename(originals / source.name)
    path = root / 'latency_split_preparation.json'
    if receipt:
        if path.exists():
            raise RuntimeError('Receipt already exists')
        path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
