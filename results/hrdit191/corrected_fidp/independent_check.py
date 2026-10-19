"""Independent recomputation of corrected (per-reference-row) patch FID/KID from cached
npz features, with a scipy sqrtm FID instead of the evaluator's low-rank identity."""
import importlib.util, sys
import numpy as np
from pathlib import Path
from scipy import linalg
REPO = Path('/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention')
sys.path.insert(0, str(REPO / 'repro'))
spec = importlib.util.spec_from_file_location('evaluate_corrected', REPO / 'repro/evaluate_corrected.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
args = mod.configure_cli()
ev = mod.Evaluation(args)
def feats(r, patch):
    with np.load(ev.out / 'features' / (ev.key(r, 'inception_features', {'patch': patch}) + '.npz'), allow_pickle=False) as z:
        return z['features'].astype(np.float64)
def fid_sqrtm(x, y):
    mx, my = x.mean(0), y.mean(0); cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    s, _ = linalg.sqrtm(cx @ cy, disp=False); s = s.real
    return float(((mx - my) ** 2).sum() + np.trace(cx) + np.trace(cy) - 2 * np.trace(s))
def kid_poly(x, y):
    d = x.shape[1]; k = lambda a, b: (a @ b.T / d + 1) ** 3
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y); n, m = len(x), len(y)
    return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
patch = True
y_rows = np.concatenate([feats(r, patch) for r in ev.references])          # corrected: every manifest row, own crops
by_path = {r['path']: feats(r, patch) for r in ev.references}              # historical: last row per path wins
y_hist = np.concatenate([by_path[r['path']] for r in ev.references])
print('reference rows', len(ev.references), 'feature rows', y_rows.shape, 'distinct paths', len(by_path))
for method in ('sca_nb', 'area', 'hrdit_dev_author', 'ultraimage'):
    x = np.concatenate([feats(r, patch) for r in sorted(ev.records, key=lambda r: r['prompt_id']) if r['method'] == method])
    print(method, x.shape, 'corrected patch_fid', repr(fid_sqrtm(x, y_rows)), 'patch_kid', repr(kid_poly(x, y_rows)),
          '| historical patch_fid', repr(fid_sqrtm(x, y_hist)))
