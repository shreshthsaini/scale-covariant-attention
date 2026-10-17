"""Failure-path checks for unattended continuation; no live scheduler or SSH."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('continuation', Path(__file__).parents[1] / 'slurm/continuation.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def test_pending_gpu_wakes_on_start_or_termination():
    records = {'1075814': ('SCA7', 'PENDING'), '1076923': ('SCA9', 'RUNNING'),
               '20': ('SCAC', 'PENDING'), '21': ('MF9', 'RUNNING')}
    assert controller.gpu_dependency(records) == 'after:1075814?afterany:1075814?afterany:1076923'


def test_no_gpu_means_no_wait_dependency():
    assert controller.gpu_dependency({'20': ('SCAC', 'PENDING')}) == ''


@pytest.mark.parametrize('record', [('SCAC', 'RUNNING'), ('SCA9', 'PENDING'), ('foreign', 'PENDING')])
def test_cancel_never_signals_live_or_foreign_job(monkeypatch, record):
    monkeypatch.setattr(controller, 'jobs', lambda: {'20': record})
    monkeypatch.setattr(controller, 'command', lambda args: pytest.fail('must not cancel'))
    with pytest.raises(RuntimeError):
        controller.cancel_successor('20')


def test_cancel_saved_pending_controller(monkeypatch):
    calls = []
    monkeypatch.setattr(controller, 'jobs', lambda: {'20': ('SCAC', 'PENDING')})
    monkeypatch.setattr(controller, 'command', calls.append)
    controller.cancel_successor('20')
    assert calls == [['scancel', '20']]


def test_missing_completion_is_not_success(tmp_path):
    assert controller.completion_errors(tmp_path, tmp_path)


def test_self_attestation_without_artifacts_is_rejected(tmp_path):
    destination = tmp_path / 'results/evidence/project_completion.json'
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps({'complete': True, 'checks': dict.fromkeys(controller.CHECKS, True), 'artifacts': []}))
    errors = controller.completion_errors(tmp_path, tmp_path)
    assert any('Missing delivery artifact' in e for e in errors)


def test_path_escape_in_completion_receipt_is_rejected(tmp_path):
    destination = tmp_path / 'results/evidence/project_completion.json'
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps({'complete': True, 'checks': dict.fromkeys(controller.CHECKS, True),
                                      'artifacts': [{'path': '../outside', 'sha256': '0' * 64}]}))
    assert any('repository-relative' in e for e in controller.completion_errors(tmp_path, tmp_path))


def test_submission_refuses_ssh_cap(monkeypatch):
    monkeypatch.setattr(controller, 'command', lambda args: 'ssh\nssh\nssh\n')
    with pytest.raises(RuntimeError, match='SSH cap'):
        controller.submit()


def test_submission_budgets_before_sbatch_and_records_receipt(monkeypatch, tmp_path):
    calls = []
    def fake(args):
        calls.append(args)
        return 'python\n' if args[0] == 'ps' else '| ASC26048 8000 2027-04-28 |\n123456;vista\n'
    monkeypatch.setattr(controller, 'command', fake)
    monkeypatch.setattr(controller, 'CONTROL', tmp_path)
    assert controller.submit('afterany:20') == '123456'
    remote = calls[-1][-1]
    assert remote.index('/usr/local/etc/taccinfo') < remote.index('sbatch --parsable')
    assert 'value <= 2176' in remote
    assert '--dependency=afterany:20' in remote
    assert (tmp_path / 'submissions/123456.json').is_file()


def test_rejects_dependency_shell_injection(monkeypatch):
    monkeypatch.setattr(controller, 'command', lambda args: 'python\n')
    with pytest.raises(ValueError, match='Invalid dependency'):
        controller.submit('afterany:20; arbitrary')


@pytest.fixture
def completed(tmp_path):
    root, scratch = tmp_path / 'repo', tmp_path / 'scratch'
    def write(relative, data):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
        return controller.hashlib.sha256(path.read_bytes()).hexdigest()
    pdf = write('paper/main.pdf', 'test PDF bytes')
    zip_hash = write('paper/sca_overleaf.zip', 'test source ZIP bytes')
    write('paper/main.tex', 'Verified manuscript')
    write('paper/delivery_receipt.json', json.dumps({'standalone_build_passed': True,
          'paper_sha256': pdf, 'source_zip_sha256': zip_hash}))
    write('results/evidence/environment/fresh_gpu_smoke.json', json.dumps(
          {'status': 'passed', 'timed_out': False, 'worker_exit_code': 0}))
    rows = 'res,method,seconds,warmup_verified\n'
    for resolution, method in (('6144x6144', 'area'), ('6144x6144', 'full'), ('8192x8192', 'area')):
        rows += f'{resolution},{method},1.0,True\n' * 3
    write('results/latency_curve.csv', rows)
    write('results/evidence/project_completion.json', json.dumps({'complete': True,
          'checks': dict.fromkeys(controller.CHECKS, True), 'artifacts': [
              {'path': 'paper/main.pdf', 'sha256': pdf},
              {'path': 'paper/sca_overleaf.zip', 'sha256': zip_hash}]}))
    for experiment in ('e2_6144', 'e9_krea'):
        folder = scratch / 'eval' / (experiment + '_base')
        folder.mkdir(parents=True)
        (folder / 'CPU_BOOTSTRAP_COMPLETE').touch()
        (folder / 'metrics.json').write_text(json.dumps({'complete': True, 'config': {'bootstrap': 1000}}))
    return root, scratch


def test_complete_fixture_passes_mechanical_gate(completed):
    assert controller.completion_errors(*completed) == []


def test_pending_task_prevents_completion(completed):
    root, scratch = completed
    pending = scratch / 'taskq/pending'
    pending.mkdir(parents=True)
    (pending / 'unfinished.sh').touch()
    assert 'Unfinished task queue: pending' in controller.completion_errors(root, scratch)


def test_stale_delivery_rejected(completed):
    root, scratch = completed
    (root / 'paper/main.pdf').write_text('changed')
    errors = controller.completion_errors(root, scratch)
    assert 'Artifact hash mismatch: paper/main.pdf' in errors
    assert 'Stale delivery receipt: main.pdf' in errors


def test_unfinished_bootstrap_rejected(completed):
    root, scratch = completed
    path = scratch / 'eval/e2_6144_base/metrics.json'
    path.write_text(json.dumps({'complete': True, 'config': {'bootstrap': 0}}))
    assert 'Unfinished final metrics: e2_6144' in controller.completion_errors(root, scratch)
