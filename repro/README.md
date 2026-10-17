# Reproducing the evaluation

`evaluator_snapshot/` preserves the account-authored evaluator previously imported
from the adjacent `constancy_rope` project. Its ten Python files are unchanged
copies, including the existing numerical tests. Every file also matches the
`c05ebdc` generation freeze. `SOURCE.json` records source commits and SHA256 hashes.
No other project checkout or live experiment code was changed.

The snapshot provides CLIP, PickScore, MUSIQ, FID/KID, crop metrics, sample discovery
and prompt bootstrap. `scripts/eval_extra.py --crope-root repro/evaluator_snapshot`
uses the same archived helpers for ImageReward, HPS, aesthetic, CLIP-IQA and ARNIQA.
The snapshot contains no model weights or third-party baseline implementation.
Its source project has no repository license file; this archive does not assign
a new redistribution license to that account-authored code.

## Checks run on October 17, 2026

From the SCA repository on a Vista compute node:

```bash
SCA_PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python
METRICS_PY=/work/09032/saini_2/vista/library/uv-envs/sca-metrics/bin/python
PYTHONPATH="$PWD/repro/evaluator_snapshot" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --no-project --python "$SCA_PY" python -m pytest -q \
  repro/evaluator_snapshot/tests/test_metrics.py
uv run --no-project --python "$SCA_PY" python \
  repro/evaluator_snapshot/scripts/eval.py --help
uv run --no-project --python "$METRICS_PY" python scripts/eval_extra.py \
  --crope-root "$PWD/repro/evaluator_snapshot" --help
```

All 14 tests passed. They compare FID/KID with explicit numerical references and
check paired prompt bootstrap, crop coordinates, metadata discovery, evaluator
resume and missing-metric failures. Imports were verified to resolve inside this
snapshot. These checks load no learned metric models and do not reproduce GPU
generation or the full benchmark. `validation.json` records their scope.

## Scoring existing campaign images

The commands below preserve the campaign metric choices and use a new output
directory. They were prepared from the campaign commands; full rescoring was not
rerun for this archive. The existing `e1_baselines` image tree contains the paired
191-prompt subset after development exclusions. Rebuild that tree with
`scripts/link_common.py` when restoring images.

```bash
export HF_HOME=/scratch/09032/saini_2/tmp/hf
export TORCH_HOME=/scratch/09032/saini_2/tmp/torch
export HF_HUB_OFFLINE=1
export WANDB_PROJECT=scale-covariant-attention
export WANDB_DIR=/scratch/09032/saini_2/sca/wandb
export WANDB_MODE=offline
SCA_EVAL_ROOT=/scratch/09032/saini_2/sca/out/evalroot/e1_baselines
uv run --no-project --python "$SCA_PY" python repro/evaluator_snapshot/scripts/eval.py \
  --root "$SCA_EVAL_ROOT" --prompts results/evidence/prompts/aesthetic4k_eval.jsonl \
  --references results/evidence/prompts/aesthetic4k_refs.jsonl \
  --reference-method sca_nb --res 4096x4096 \
  --out /scratch/09032/saini_2/sca/eval/repro_e1_base \
  --metrics clip,pickscore,musiq,fid,kid,patch_fid,patch_kid --device cuda
uv run --no-project --python "$METRICS_PY" python scripts/eval_extra.py \
  --crope-root "$PWD/repro/evaluator_snapshot" \
  --root "$SCA_EVAL_ROOT" --prompts results/evidence/prompts/aesthetic4k_eval.jsonl \
  --reference-method sca_nb --res 4096x4096 \
  --out /scratch/09032/saini_2/sca/eval/repro_e1_extra \
  --metrics ir,aesthetic,hpsv2,hpsv21,clipiqa,arniqa,is \
  --device cuda --batch-size 16 --threads 8
```

Generated PNGs and reference photographs remain required for fresh scoring.
Reference JSONL paths refer to scratch and must be restored or remapped after a
purge. Learned scorers require cached weights and their original access terms;
model identities/revisions and scorer settings are recorded in the archived
`results/evidence/eval/*/metrics.json` reports. The aesthetic scorer additionally
uses the fixed checkpoint path and SHA256 in `scripts/eval_extra.py`.

`generation-freeze.txt` and `metrics-freeze.txt` describe installed environments,
not a verified fresh installation recipe. The uv environments themselves resolve
to scratch. Restore them and metric caches separately after a purge. No environment
was installed or modified for these checks.

## Baseline sources and remaining dependencies

`baseline_metadata/sources.json` records exact commits, remotes, clean tracked
status, source/configuration hashes, and one actual generation configuration per
baseline. License files are copied verbatim from the inspected checkouts.

| Baseline | Exact upstream commit | License file inspected | Integration used |
| --- | --- | --- | --- |
| AReA | `9e38ec2d8457f2aa7800766e8a22d4960f160c69` | Apache-2.0 | `scripts/run_area.py` imports the upstream package |
| DyPE | `436fe612fb7695fa2d5dc32d63390c048ba6033c` | MIT, Guy Yariv | Numerical port in frozen `crope/rope.py` |
| SEGA | `ddaab08552c7b7ceba2ebd8f579a0f65dee90b5e` | No LICENSE, COPYING or NOTICE in tracked files | Account-authored numerical port, `crope/sega_official.py`; upstream code is not copied here |
| UltraImage | `586ae882099ee768e9bbc6757118d572a1e71b25` | Apache-2.0 | `scripts/run_ultraimage.py` imports upstream `flux/` modules |

Upstreams are [AReA](https://github.com/KooroshRH/AReA),
[DyPE](https://github.com/guyyariv/DyPE),
[SEGA](https://github.com/rajabi2001/sega), and
[UltraImage](https://github.com/thu-ml/DiT-Extrapolation), branch `ultra-flux`.
Use the exact commits above when restoring clean clones. The metadata hashes allow
restored files to be checked against the inspected versions. UltraImage's original
`flux/base.yaml` is preserved as `baseline_metadata/UltraImage/base.yaml`; effective
wrapper overrides are in the observed generation configuration.

The representative AReA and UltraImage sidecars report `code_revision=uncommitted`
because their scratch freezes have no Git metadata. Their composite implementation
fingerprints were rederived exactly from the frozen wrapper and dependency files.
The matching wrapper bytes are recoverable from SCA commits recorded in
`matching_frozen_wrapper`, with Git blob IDs and SHA256 hashes. Resume fingerprints
include absolute source paths, so relocating a restored wrapper can change its
fingerprint even when the source bytes match; use a fresh output directory.

AReA and UltraImage wrappers still contain an absolute `CROPE` helper path and use
external source checkouts, configurable through `AREA_DIR` and `ULTRA_DIR`.
Generation also embeds machine-specific cache/output locations. This archive makes
evaluation independent of the sibling project, but leaves those generation paths
unchanged to preserve the live protocol and resume fingerprints.

DyPE and SEGA generation used the local account-authored `constancy_rope` commit
`c05ebdcccddafd6b8fe4ec2e14247d72775df9a1`. It has no remote. Its Git history on work
can restore the scratch freeze with `git archive` into a new directory; the selected
runtime hashes in `sources.json` were checked against that commit. This archive
does not contain that generation harness. A portable public reproduction still
needs a separately prepared harness, dependency installation checks, and model/data
retrieval instructions. Numerical port parity is not a reproduction of upstream
papers under their original checkpoints or sampler settings.
