"""Check that every generated and reference record has a cached Inception feature file
under --out/features, so evaluate_corrected.py runs without loading Inception."""
import importlib.util, sys
from collections import Counter
from pathlib import Path
REPO = Path('/work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention')
sys.path.insert(0, str(REPO / 'repro'))
spec = importlib.util.spec_from_file_location('evaluate_corrected', REPO / 'repro/evaluate_corrected.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
args = mod.configure_cli()
ev = mod.Evaluation(args)
print('generated records', len(ev.records), Counter(r['method'] for r in ev.records))
print('reference rows', len(ev.references), 'distinct paths', len({r['path'] for r in ev.references}))
missing = 0
for patch in (False, True):
    for kind, recs in (('gen', ev.records), ('ref', ev.references)):
        miss = [r for r in recs if not (ev.out / 'features' / (ev.key(r, 'inception_features', {'patch': patch}) + '.npz')).is_file()]
        print(f'patch={patch} {kind}: {len(recs)-len(miss)}/{len(recs)} cached')
        missing += len(miss)
print('MISSING', missing)
