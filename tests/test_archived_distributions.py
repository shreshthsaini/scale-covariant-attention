"""Reference rows and shared-image bootstrap clusters preserve different invariants."""
import hashlib
import numpy as np
from scripts.check_archived_distributions import load_bank


def test_reference_row_policy_and_image_clusters(tmp_path):
    raw = np.array([[[1., 2.]], [[7., 8.]]], dtype=np.float32)
    for name, values in [('rows', raw), ('legacy', raw[[1, 1]])]:
        np.savez(tmp_path / f'{name}.npz', features=values, prompt_ids=['p1', 'p2'], seeds=['reference', 'reference'])
    entry = dict(bank='legacy.npz', row_bank='rows.npz',
                 sha256=hashlib.sha256((tmp_path / 'legacy.npz').read_bytes()).hexdigest(),
                 row_bank_sha256=hashlib.sha256((tmp_path / 'rows.npz').read_bytes()).hexdigest(),
                 records=[dict(prompt_id=p, seed='reference', manifest_order=i,
                               image_signature={'path': '/same/image.png'}) for i, p in enumerate(['p1', 'p2'])])
    legacy = load_bank(tmp_path, entry)
    rows = load_bank(tmp_path, entry, 'rows')
    clustered = load_bank(tmp_path, entry, 'rows', 'images')
    unique = load_bank(tmp_path, entry, 'unique', 'images')
    np.testing.assert_array_equal(legacy['p1'], legacy['p2'])
    np.testing.assert_array_equal(rows['p1'], raw[0])
    assert len(clustered) == 1
    np.testing.assert_array_equal(clustered['/same/image.png'], raw[:, 0])
    np.testing.assert_array_equal(unique['/same/image.png'], raw[0])
