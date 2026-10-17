"""Prioritize untouched E3 tasks and add one explicitly excluded warm-up image.

This changes pending task plumbing only, never frozen generation code. The CPU-only
summary is parked for the main operator instead of occupying a GPU worker.
"""
import json
from pathlib import Path
import subprocess


def main():
    root = Path('/scratch/09032/saini_2/sca')
    queue = root / 'taskq'
    original = root / 'lat3.jsonl'
    rows = [json.loads(s) for s in original.read_text().splitlines() if s.strip()]
    warm = {'id': 'sca_latency_warmup', 'prompt': rows[0]['prompt']}
    prompts = root / 'latency4_warmup.jsonl'
    prompts.write_text('\n'.join(json.dumps(r) for r in [warm, *rows]) + '\n')
    changes = []
    for task in sorted((queue / 'pending').glob('y00?_latency_*.sh')):
        assert not any((root / 'out/latency_curve' / task.stem.split('_latency_')[1]).glob('*/*.json')), task
        text = task.read_text().replace(str(original), str(prompts))
        text = text.replace('# E3:', '# E3: first image is an excluded warm-up;')
        # All three timed images follow a warm-up in the same process.
        new = task.with_name('a_' + task.name)
        assert not new.exists(), new
        task.write_text(text)
        subprocess.run(['bash', '-n', str(task)], check=True)
        task.rename(new)
        changes.append({'old': task.name, 'new': new.name})
    summary = queue / 'pending/y999_summary_latency.sh'
    if summary.exists():
        text = summary.read_text().replace("'^y00[0-9]_latency_'", "'^(a_)?y00[0-9]_latency_'")
        text = text.replace('d = json.load(open(f)); c = d["config"]',
                            'd = json.load(open(f)); c = d["config"]\n    if c["prompt_id"] == "sca_latency_warmup": continue')
        summary.write_text(text)
        subprocess.run(['bash', '-n', str(summary)], check=True)
        hold = queue / 'hold_cpu'
        hold.mkdir(exist_ok=True)
        summary.rename(hold / summary.name)
    receipt = root / 'latency_task_preparation.json'
    receipt.write_text(json.dumps({'changes': changes, 'warmup_id': warm['id'],
                                  'timed_prompt_ids': [r['id'] for r in rows]}, indent=2) + '\n')
    print(receipt.read_text())


if __name__ == '__main__':
    main()
