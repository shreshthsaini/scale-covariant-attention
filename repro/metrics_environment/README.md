# Learned-metric restoration

Three distinct fresh environments were installed on Linux aarch64 with CPython
3.11.13 and uv 0.8.23. Bounded inherited/restored CPU scoring is exactly equal for
all tested outputs. Model assets were restored into isolated scratch caches from
verified local copies, without network requests or campaign-environment changes.
This is environment parity on fixed existing images, not full benchmark reproduction.

| Recipe | Metrics | Selected packages | Main versions |
| --- | --- | ---: | --- |
| `base.lock` | CLIP, PickScore, MUSIQ, FID/KID and crop FID/KID | 125 | PyIQA 0.1.16, Transformers 5.19.0, Hub 2.2.0, NumPy 2.4.6 |
| `extra.lock` | Aesthetic, HPSv2/2.1, CLIP-IQA, ARNIQA, Inception Score | 123 | PyIQA 0.1.15, Transformers 4.45.2, Hub 0.36.2, NumPy 2.2.6 |
| `imagereward.lock` | ImageReward | 80 | ImageReward 1.5, Transformers 4.45.2, timm 0.6.13 |

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

## Restoring another installation

Run on a compute node, using new environment names and scratch caches:

```bash
export UV_PYTHON_INSTALL_DIR=/work/09032/saini_2/vista/library/uv-python
bash repro/metrics_environment/install.sh base \
  /work/09032/saini_2/vista/library/uv-envs/sca-metric-base-NEW \
  /scratch/09032/saini_2/sca/metric_uv_cache_NEW /path/to/existing/python3.11
```

Repeat with `extra` and `imagereward`, using distinct environment names. The
installer refuses existing environments. The completed October17 installations are
recorded below. Do not alter active campaign environments.

Some dependencies, including openai-clip and fairscale, have source-only releases.
`wheel_only_failure.log` records why a wholly wheel-only recipe fails. The locks
require binary packages for Torch, NumPy, SciPy, pandas and the other selected
native components, while allowing the small Python source packages. Source-build
execution was exercised for the selected small Python wheels, with no native
extensions built. Runtime artifact hashes do not separately pin every build-system
dependency. Stop if a native
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
and exact hashes are recorded. The later isolated restoration rehashed all selected HF source and destination
files as well. Pinned revisions and content-addressed cache identities are retained.

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
offline `main` references in the isolated cache, without a trailing newline. Torch URLs are recorded in
`torch_asset_urls.json`; mutable upstream URLs must still match the exact archived
hash. A mismatch stops restoration and preserves the file for inspection.
No restoration downloads were executed during this audit.

## Reusing verified local assets

On a compute node with the archived source paths still present, restore isolated
caches without downloading weights:

```bash
"$SCA_FRESH_PY" repro/metrics_environment/restore_assets.py \
  --hf-home /scratch/09032/saini_2/sca/metric_restore_NEW/hf \
  --torch-home /scratch/09032/saini_2/sca/metric_restore_NEW/torch \
  --from-cache --receipt /scratch/09032/saini_2/sca/metric_restore_NEW/cache_copy_receipt.json
```

This verifies each source and destination SHA256, preserves the original caches,
and refuses conflicting destination bytes or revisions. Three tokenizer files
have archived Git blob SHA1 identities rather than SHA256 values; the helper
checks those identities and recorded sizes before computing their SHA256.
Partial copies stay available for diagnosis. The receipt records every copied
file, its size and SHA256. This checks asset restoration only; numerical parity
requires actual scoring in each restored environment.

The continuation CPU unit copied and verified all selected assets without network
requests. Its receipt is archived at
`results/evidence/metric_environment/cache_copy_receipt.json`.

## Scoring after restoration

Use the same images, prompt rows, methods and preprocessing settings as the parent
reproduction guide. First unset legacy cache overrides, then use the new paths:

```bash
unset TRANSFORMERS_CACHE
export HF_HOME=/scratch/09032/saini_2/sca/metric_restore_NEW/hf
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME=/scratch/09032/saini_2/sca/metric_restore_NEW/torch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
```

Both Hub cache variables are explicit because account-level overrides can take
precedence over `HF_HOME`. Keeping an old override would test the inherited cache
instead of the restored one. The parity checks record the resolved cache paths.

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

## Completed bounded validation, October17

| Environment | Fixed-image checks | Result |
| --- | --- | --- |
| Base,125 packages | Four original4K CLIP/PickScore inputs, eight MUSIQ outputs on deterministic1024 crops,40960 Inception feature scalars | Bit exact |
| Extra,123 packages | Two original4K inputs; aesthetic,HPSv2,HPSv2.1,CLIP-IQA,ARNIQA;2016 Inception logits | Bit exact |
| ImageReward,80 packages | Same two original4K inputs; timm0.6.13 rather than inherited1.0.30 | Bit exact |

Receipts: `results/evidence/metric_environment/base_validation.json`,
`extra_restore_validation.json`, `imagereward_restore_validation.json` and
`validation_summary_1076947.json`. They retain commands, package inventories,
input/checkpoint/code hashes, declared tolerances, score outputs and logs. The real
portable extra and ImageReward CLIs completed and their seven-metric reports
passed the strict merger. Forty-two focused integration/rejection tests pass.

The exercised recipes needed four repairs, with original failures retained:

- Both HF cache override variables must point at the isolated cache. Setting
  `HF_HOME` alone can leave an inherited account cache active.
- Hub0.36.2 reads `refs/main` verbatim; a newline prevents offline lookup.
  Restoration now writes the exact revision and rejects conflicting references.
- ImageReward1.5 imports undeclared `clip`; its lock now includes
  `openai-clip==1.0.1` plus its dependencies. The old lock remains archived.
- HPSv2's wheel omits BPE tokenizer data. `restore_hps_tokenizer.py` copies the
  pinned CLIP asset only after checking its SHA256 against inherited HPS data.
  The extra installer invokes it; mismatched existing files are preserved/rejected.

All three final `uv pip check` commands return1 solely for the documented NVIDIA
cuSPARSELt wheel-tag warning; they have zero dependency-version conflicts. This
warning remains visible. Native4K MUSIQ, CUDA/mixed-precision metric parity,
full distributional estimates and full benchmark reruns are outside these bounded
checks. Two-image one-split IS only checks the CLI/logit path, not benchmark IS.
The parity harness receipts contain exact commands for another fixed-image run.

`repro/tests/test_extra_report_merge.py` uses explicitly synthetic report fixtures
to cover identity/seed mismatches, partial results, duplicate/conflicting metrics,
stale files, invalid statistics and compatibility with the actual paper table
reader. `merge_validation.json` also records a tiny end-to-end synthetic PNG test
using sidecar latency only, including refusal to resume after an image changes.
