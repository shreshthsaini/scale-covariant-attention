"""Exercise promotion barriers and admission sizing without generation or queue writes."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from PIL import Image
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('promotion', HERE / 'prepare.py')
promotion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promotion)


@pytest.fixture
def environment(tmp_path):
    _, real, _ = promotion.load_frozen()
    prompts = tmp_path / 'pilot46.jsonl'
    rows = [dict(id=str(i), prompt=f'fixed prompt {i}') for i in range(46)]
    prompts.write_text(''.join(json.dumps(x) + '\n' for x in rows))
    adapter = SimpleNamespace(ROOT=tmp_path, read_prompts=real.read_prompts,
        verify_sources=lambda: {}, stem_for=real.stem_for, check_resume=real.check_resume,
        make_config=lambda args, row, source: dict(method=args.profile, prompt_id=row['id']))
    receipt = dict(output_root=str(tmp_path / 'out'))
    buffer = io.BytesIO()
    Image.new('RGB', (4096, 4096), (40, 80, 120)).save(buffer, format='PNG')
    content = buffer.getvalue()

    def output(method, index, seconds=100):
        image = Path(receipt['output_root']) / method / '4096x4096' / (real.stem_for(str(index), 0) + '.png')
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(content)
        image.with_suffix('.json').write_text(json.dumps(dict(complete=True,
            config=dict(method=method, prompt_id=str(index)), png_sha256=promotion.sha(image),
            wall_seconds=seconds, seconds=seconds,
            finite_checks=['latents_each_step', 'vae_decoder_before_pixel_conversion'])))
        return image

    return receipt, adapter, output


def test_real_frozen_import_and_task_admission(tmp_path):
    receipt, adapter, writer = promotion.load_frozen()
    assert adapter.ROOT == Path(receipt['frozen_root'])
    before = {k: promotion.sha(adapter.ROOT / k) for k in receipt['files']}
    script = writer.task_text(adapter.ROOT, Path(receipt['output_root']), promotion.METHODS[0], 12, 14, 6000)
    path = tmp_path / 'candidate.sh'
    path.write_text(script)
    subprocess.run(['bash', '-n', str(path)], check=True)
    assert '--start 12 --end 14' in script and '--wandb' in script
    assert before == {k: promotion.sha(adapter.ROOT / k) for k in receipt['files']}


def test_missing_smoke_closes_barrier(environment):
    receipt, adapter, _ = environment
    with pytest.raises(ValueError, match='Prerequisite incomplete'):
        promotion.inspect_outputs(receipt, adapter, 'pilot12')


def test_future_completed_output_excluded_and_source_bytes_preserved(environment):
    receipt, adapter, output = environment
    paths = [output(method, index) for method in promotion.METHODS for index in [0, 4]]
    before = {str(p): (promotion.sha(p), promotion.sha(p.with_suffix('.json'))) for p in paths}
    rows, records = promotion.inspect_outputs(receipt, adapter, 'pilot12')
    assert len(rows) == 12 and len(records) == 4
    result = promotion.plan(receipt, records, 'pilot12', 1000, 100)
    for data in result.values():
        planned = [i for item in data['ranges'] for i in range(item['start'], item['end'])]
        assert planned == [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
        assert all(item['minimum_seconds'] <= 1000 for item in data['ranges'])
    assert before == {str(p): (promotion.sha(p), promotion.sha(p.with_suffix('.json'))) for p in paths}


def test_partial_future_output_refuses_instead_of_overwriting(environment):
    receipt, adapter, output = environment
    for method in promotion.METHODS:
        output(method, 0)
    existing = output(promotion.METHODS[0], 0)
    image = existing.with_name(adapter.stem_for('4', 0) + '.png')
    image.write_bytes(existing.read_bytes())
    with pytest.raises(ValueError, match='Partial output retained'):
        promotion.inspect_outputs(receipt, adapter, 'pilot12')
    assert image.exists()


def test_pilot46_requires_all_twelve_before_candidate_creation(environment):
    receipt, adapter, output = environment
    for method in promotion.METHODS:
        for index in range(11):
            output(method, index)
    with pytest.raises(ValueError, match='Prerequisite incomplete'):
        promotion.inspect_outputs(receipt, adapter, 'pilot46')


@pytest.mark.parametrize('field,value', [
    ('wall_seconds', float('nan')), ('wall_seconds', True), ('seconds', 0), ('finite_checks', []),
])
def test_invalid_timing_or_missing_finite_checks_closes_gate(environment, field, value):
    receipt, adapter, output = environment
    image = output(promotion.METHODS[0], 0)
    path = image.with_suffix('.json')
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='timing|finite-generation'):
        promotion.inspect_outputs(receipt, adapter, 'pilot12')


def test_crc_corruption_rejected_even_with_updated_hash(environment):
    receipt, adapter, output = environment
    image = output(promotion.METHODS[0], 0)
    content = bytearray(image.read_bytes())
    content[50] ^= 1
    image.write_bytes(content)
    path = image.with_suffix('.json')
    record = json.loads(path.read_text())
    record['png_sha256'] = promotion.sha(image)
    path.write_text(json.dumps(record))
    with pytest.raises((OSError, SyntaxError)):
        promotion.inspect_outputs(receipt, adapter, 'pilot12')


def test_review_requires_exact_coverage_current_bytes_and_actual_review(tmp_path):
    deployment = tmp_path / 'deployment.json'
    deployment.write_text('{}')
    records = [dict(method=m, prompt_id='0', image_sha256='a', sidecar_sha256='b', prerequisite=True)
        for m in promotion.METHODS]
    template = promotion.review_template('pilot12', deployment, records)
    review = json.loads(json.dumps(template))
    review['reviewer'] = 'operator'
    with pytest.raises(ValueError, match='fidelity review'):
        promotion.validate_review(review, template)
    for row in review['observations']:
        row.update(fidelity_reviewed=True, image_valid=True, notes='Inspected image and recorded source settings.')
    promotion.validate_review(review, template)
    review['observations'][0]['image_sha256'] = 'stale'
    with pytest.raises(ValueError, match='hash-bound'):
        promotion.validate_review(review, template)
    review['observations'][0]['image_sha256'] = 'a'
    review['observations'].append(review['observations'][0])
    with pytest.raises(ValueError, match='exactly'):
        promotion.validate_review(review, template)


def test_duration_bound_refuses_smaller_fleet_and_splits_gaps():
    with pytest.raises(ValueError, match='larger fleet window'):
        promotion.task_ranges([1], 3100, 6000, 600)
    assert promotion.task_ranges([1, 2, 4, 5, 6], 1000, 6600, 600) == [(1, 3, 4600), (4, 7, 6600)]


def test_no_result_based_promotion_and_exact46_coverage():
    records = [dict(method=m, index=i, pipeline_seconds=200 + i * 10)
        for m in promotion.METHODS for i in range(12)]
    result = promotion.plan({}, records, 'pilot46', 6000, 600)
    for data in result.values():
        planned = [i for x in data['ranges'] for i in range(x['start'], x['end'])]
        assert planned == list(range(12, 46))
        assert data['remaining_images'] == 34
        assert data['incremental_pipeline_gpu_hours']['central'] == pytest.approx(34 * 255 / 3600)
        assert data['task_reservation_hours'] >= 34 * 310 * 2 / 3600


def test_existing_candidate_directory_is_preserved(tmp_path):
    existing = tmp_path / 'existing'
    existing.mkdir()
    sentinel = existing / 'keep'
    sentinel.write_text('retain')
    with pytest.raises(FileExistsError):
        promotion.write_candidates(existing, {}, None, 'pilot46', {}, tmp_path / 'review')
    assert sentinel.read_text() == 'retain'


def test_candidates_bind_review_and_use_original_frozen_manifest(tmp_path):
    receipt, _, writer = promotion.load_frozen()
    review = tmp_path / 'review.json'
    review.write_text('{"fixture_only":true}\n')
    result = {m: dict(ranges=[dict(start=12, end=13, minimum_seconds=6000)]) for m in promotion.METHODS}
    directory = tmp_path / 'candidates'
    report = promotion.write_candidates(directory, receipt, writer, 'pilot46', result, review)
    assert report['status'] == 'candidates_only_not_enqueued'
    assert report['review_sha256'] == promotion.sha(review)
    assert len(report['tasks']) == 2
    for task in report['tasks']:
        path = Path(task['path'])
        assert task['sha256'] == promotion.sha(path)
        text = path.read_text()
        assert receipt['frozen_root'] + '/pilot46.jsonl' in text
        assert '--out ' + receipt['output_root'] in text
        assert 'sbatch' not in text and 'scancel' not in text
        subprocess.run(['bash', '-n', str(path)], check=True)
    assert json.loads((directory / 'receipt.json').read_text()) == report
