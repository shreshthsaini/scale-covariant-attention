# Learned-metric restoration

Three dependency sets resolve on Linux aarch64 with CPython 3.11.13 and uv 0.8.23.
Each lock pins package versions and distribution hashes. They have **not** been
installed as complete fresh environments or used for image rescoring. This audit
downloaded no model weights and changed neither campaign environment. A separate
probe installed only the small `timm==0.6.13` wheel, then read the inherited
dependencies to check ImageReward imports.

| Recipe | Metrics | Selected packages | Main versions |
| --- | --- | ---: | --- |
| `base.lock` | CLIP, PickScore, MUSIQ, FID/KID and crop FID/KID | 125 | PyIQA 0.1.16, Transformers 5.19.0, Hub 2.2.0, NumPy 2.4.6 |
| `extra.lock` | Aesthetic, HPSv2/2.1, CLIP-IQA, ARNIQA, Inception Score | 123 | PyIQA 0.1.15, Transformers 4.45.2, Hub 0.36.2, NumPy 2.2.6 |
| `imagereward.lock` | ImageReward | 77 | ImageReward 1.5, Transformers 4.45.2, timm 0.6.13 |

All retain Torch 2.11.0+cu128, torchvision 0.26.0+cu128 and Triton 3.6.0.
The `.constraints` files preserve inherited versions where selected; the `.in`
files state the required packages. The `*_resolve.json` files record exact
resolver commands. The base recipe does not install Diffusers, so its inherited
Hub 2.2.0 is coherent here; the generation recipe's Diffusers constraint does not
apply to this environment.

## Why the recipes are separate

The inherited `sca-metrics` environment imports the scorers successfully but has
four package-version conflicts plus the NVIDIA platform-tag warning:

- ImageReward requires `timm==0.6.13`, while PyIQA 0.1.15 requires `timm>=0.8`.
  The inherited timm is 1.0.30. ImageReward therefore gets a separate environment.
  Its imports also pass with the isolated 0.6.13 wheel over inherited dependency
  paths. That check is recorded in `timm_compatibility_probe.json`; it is not a
  full fresh-environment or numerical-equivalence test.
- HPSv2 requires `protobuf<4`, `pytest==7.2.0`, and `pytest-split==0.8.0`.
  The extra recipe uses protobuf 3.20.3 and those exact test-package versions.
  Inherited TensorBoard 2.21.0 requires protobuf >=6.31.1, so the extra recipe
  selects TensorBoard 2.16.2. The resulting set resolves without version conflicts.
- PyIQA 0.1.15 requires Transformers <5; the main metric campaign used PyIQA
  0.1.16 and Transformers 5.19.0. Preserve these two metric implementations in
  separate recipes instead of silently changing either campaign version.

`inherited_pip_check.log` and `extra_tensorboard_conflict.log` preserve the
conflicts. `inherited_metadata.json` contains the exact requirements. All five
inherited scorer imports passed without CUDA initialization; the environment
still has the same 96 missing zero-byte Torch/Triton modules as inherited SCA.

## Installing later

Run on a compute node, using new environment names and scratch caches:

```bash
export UV_PYTHON_INSTALL_DIR=/work/09032/saini_2/vista/library/uv-python
bash repro/metrics_environment/install.sh base \
  /work/09032/saini_2/vista/library/uv-envs/sca-metric-base-NEW \
  /scratch/09032/saini_2/sca/metric_uv_cache_NEW /path/to/existing/python3.11
```

Repeat with `extra` and `imagereward`, using distinct environment names. The
installer refuses existing environments. These are resolved restoration commands,
not completed installation evidence. Do not alter active campaign environments.

Some dependencies, including openai-clip and fairscale, have source-only releases.
`wheel_only_failure.log` records why a wholly wheel-only recipe fails. The locks
require binary packages for Torch, NumPy, SciPy, pandas and the other selected
native components, while allowing the small Python source packages. Source-build
execution and build-tool isolation have not been validated; runtime artifact
hashes do not separately pin every build-system dependency. Stop if a native
toolchain build is requested. The final `uv pip check` may also report the
preserved cuSPARSELt `manylinux2014_sbsa` tag warning documented in the
[generation recipe](../environment/README.md). Do not ignore additional errors.

## Required model assets

`checkpoints.json` inventories six HF snapshots and seven Torch asset files.
Five HF revisions match the historical completed main evaluation records.
The BERT tokenizer revision is retrospective cache evidence. The legacy
`TRANSFORMERS_CACHE` directory contains the same revisions and blob identities
as `HF_HOME/hub`; both were checked. The old extra reports' `checkpoint_sha`
fields are HF revision IDs, not hashes of checkpoint bytes.

All seven Torch files were freshly SHA256-hashed: MUSIQ KonIQ, CLIP RN50,
ARNIQA encoder and KonIQ regressor, the ResNet50 weights loaded by ARNIQA's
constructor, canonical FID Inception, and the aesthetic MLP. Their paths, sizes
and exact hashes are recorded. Large HF model files were not rehashed; their
content-addressed cache blob names and pinned revisions are recorded.

The restoration plan needs about 12.08 GB of assets before cache overhead.
It excludes the redundant PickScore `.bin` copy and does not download a BERT
model or the LAION ViT-H model: only their tokenizer/processor files are needed.
Weights remain outside this repository and retain their original access terms.

The following command prints a plan and makes no directories or downloads:

```bash
SCA_FRESH_PY=/work/09032/saini_2/vista/library/uv-envs/sca-repro-20261017/bin/python
"$SCA_FRESH_PY" repro/metrics_environment/restore_assets.py \
  --hf-home /scratch/09032/saini_2/sca/metric_restore_NEW/hf \
  --torch-home /scratch/09032/saini_2/sca/metric_restore_NEW/torch
```

Add `--download` only when restoring assets on a compute node. The helper fetches
the pinned HF revisions, verifies recorded SHA256 identities, and creates their
offline `main` references in the isolated cache. Torch URLs are recorded in
`torch_asset_urls.json`; mutable upstream URLs must still match the exact archived
hash. A mismatch stops restoration and preserves the file for inspection.
No restoration downloads were executed during this audit.

## Scoring after restoration

Use the same images, prompt rows, methods and preprocessing settings as the parent
reproduction guide. First unset legacy cache overrides, then use the new paths:

```bash
unset TRANSFORMERS_CACHE
export HF_HOME=/scratch/09032/saini_2/sca/metric_restore_NEW/hf
export TORCH_HOME=/scratch/09032/saini_2/sca/metric_restore_NEW/torch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
```

Run `repro/evaluate_corrected.py` with the base environment and
`--reference-clusters images` for the final main/wide distribution intervals.
For extras, `repro/eval_extra_portable.py` uses archived helpers by default and
locates the aesthetic checkpoint under `--torch-home`. It changes only those
paths; the existing scorer implementation and checkpoint hash check remain intact.

Use the extra and ImageReward environments in separate output directories, then
merge their completed reports. For example, after restoring the images:

```bash
EXTRA_PY=/work/09032/saini_2/vista/library/uv-envs/sca-metric-extra-NEW/bin/python
IR_PY=/work/09032/saini_2/vista/library/uv-envs/sca-metric-imagereward-NEW/bin/python
IMAGE_ROOT=/scratch/09032/saini_2/sca/metric_restore_NEW/images
RESULT_ROOT=/scratch/09032/saini_2/sca/metric_restore_NEW/evaluation
COMMON=(--root "$IMAGE_ROOT" --prompts results/evidence/prompts/aesthetic4k_eval.jsonl
  --reference-method area --res 4096x4096 --bootstrap 1000 --bootstrap-seed 0
  --iqa-long-side 1024 --is-splits 10 --hf-home "$HF_HOME" --torch-home "$TORCH_HOME")
"$EXTRA_PY" repro/eval_extra_portable.py "${COMMON[@]}" \
  --metrics aesthetic,hpsv2,hpsv21,clipiqa,arniqa,is --out "$RESULT_ROOT/extra_split"
"$IR_PY" repro/eval_extra_portable.py "${COMMON[@]}" \
  --metrics ir --out "$RESULT_ROOT/ir_split"
"$SCA_FRESH_PY" repro/merge_extra_reports.py --out "$RESULT_ROOT/extra_merged" \
  "$RESULT_ROOT/ir_split" "$RESULT_ROOT/extra_split"
```

Each portable scorer hashes image bytes, prompt text and sidecars before and after
scoring. `extra_input_identities.json` records those identities, the source
environment/code hashes and hashes of all four report files. Resume is allowed
only when these inputs, code and environment remain unchanged. Existing caches
without this proof require a new output directory. Hashing reads the image files
twice but performs no additional model inference.

The merger requires identical prompt/method/resolution/seed/path/image identities,
complete metric coverage, matching reference/bootstrap/preprocessing settings,
and consistent JSON/CSV reports. It checks scalar point means and paired counts.
It rejects overlapping metrics with conflicting values, intervals or model
provenance. Identical duplicate metrics are retained once, with both sources
recorded. Historical reports without scoring-time identity proof are rejected;
the merger does not retrospectively infer image identity from a path.

The output contains standard `extra_per_image.csv`, `extra_summary.csv`,
`extra_paired_deltas.csv` and `extra_metrics.json`, plus its identity manifest.
The paper's table reader accepts this format. Each source environment, full
configuration, metric provenance and file hashes remain in the merged JSON;
top-level versions contain only values shared by all sources. Source confidence
intervals and IS statistics are preserved. IS is not recomputed without logits.
No extra scorer run or invented score is used to combine the reports. Preserve
the identity manifest when archiving the merged and source reports.

The remaining verification is concrete: install each complete lock, run its CLI
and numerical tests, then score the same small, hash-identified image/prompt set
in the inherited and restored environments. Compare every metric before accepting
full rescoring. In particular, the coherent ImageReward timm pin differs from the
historical environment. Fresh locks, import checks and matching checkpoints alone
do not establish equal scores. The supplied restore-plan/checksum tests cover
default no-download behavior and rejection of changed assets.

`repro/tests/test_extra_report_merge.py` uses explicitly synthetic report fixtures
to cover identity/seed mismatches, partial results, duplicate/conflicting metrics,
stale files, invalid statistics and compatibility with the actual paper table
reader. `merge_validation.json` also records a tiny end-to-end synthetic PNG test
using sidecar latency only, including refusal to resume after an image changes.
