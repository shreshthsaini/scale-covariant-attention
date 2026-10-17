# Reproducing the evaluation

`evaluator_snapshot/` preserves the account-authored evaluator previously imported
from the adjacent `constancy_rope` project. Its ten Python files are unchanged
copies, including the existing numerical tests. Every file also matches the
`c05ebdc` generation freeze. `SOURCE.json` records source commits and SHA256 hashes.
No other project checkout or live experiment code was changed.

Use `evaluate_corrected.py` for new distribution evaluations. The original snapshot
has a reference crop-feature assembly bug: references with different prompt IDs can
resolve to the same image path, and the last reference's crops overwrite earlier
ones. The corrected entrypoint keeps a separate feature bank per reference prompt
ID, records policy `reference_prompt_row_v2`, and retains the original reference
row weights. The historical snapshot remains unchanged for reproducing old reports.
Use `--reference-clusters images` for the final manuscript's corrected main and wide
distribution intervals. This groups repeated references by content SHA256 when
available, including SHA256-named Hugging Face blobs, and otherwise by resolved path.
All 195 reference rows retain their point-estimate weights inside the 168 image
clusters. The default `rows` retains the earlier reference-row bootstrap for
compatibility. `reference_cluster_validation.json` records exact numerical agreement
with the independent archived-feature checker for all four distribution metrics.

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

Additional regression tests in `tests/test_reference_rows.py` exercise the
corrected evaluator. Different prompt IDs share one reference PNG, produce distinct
crop features, and reuse historical per-record caches. Corrected point estimates
and paired intervals match direct per-row calculations, whole-image FID/KID stay
unchanged, and all reference rows remain present. Image-cluster checks match the
archived-feature checker while retaining every row's weight. Another test checks that optional
MUSIQ execution changes use a separate cache while other metric caches remain
compatible. Run these checks together with the historical numerical tests:

```bash
PYTHONPATH="$PWD/repro/evaluator_snapshot:$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --no-project --python "$SCA_PY" python -m pytest -q \
  repro/tests/test_reference_rows.py repro/evaluator_snapshot/tests/test_metrics.py
uv run --no-project --python "$SCA_PY" python repro/evaluate_corrected.py --help
```

The pinned Aesthetic-4K source has 195 filenames and prompt IDs but 168 distinct
image-content hashes. This was verified against the upstream Hugging Face tree at
revision `8c5d5cb8b94230ff897d87bf060451257d1c7bf8`, then against the local cache.
All exported filenames/prompts match the upstream metadata. The repeated content
comes from the source dataset, rather than a local export mistake.
`reference_dataset_audit.json` records the comparison and the 15 repeated-content
groups. The corrected main evaluation preserves all 195 rows; any deduplicated
168-image evaluation is a separate sensitivity analysis.

## Scoring existing campaign images

Native 8K MUSIQ has 65,730 attention tokens including its two resized scales and
class token. Its original six-head float32 attention matrix alone occupies about
104 GB, before masking and softmax temporaries. `--musiq-query-chunk 256` opts into
`musiq_chunked.py`, which computes independent query rows in chunks against all
keys. It retains every original-resolution patch, the original resized scales,
the finite -1000 mask, and the original softmax. It requires inference mode.
This reduces intermediate memory but keeps quadratic attention computation.
Other metrics reuse their original caches; adapted MUSIQ uses a separate cache
namespace and records adapter/source hashes in its provenance.
PyIQA attribution and the installed package's license files are preserved in
`baseline_metadata/PyIQA/`; its principal license is PolyForm Noncommercial 1.0.0.

`musiq_chunked_validation.json` records a bounded CPU check using the cached
KONIQ checkpoint. Original and adapted scores were identical at float32 output
precision for deterministic 512x512, 768x1024 and 1024x1024 random inputs.
Attention regression tests cover unmasked, partially masked and fully masked
queries, unequal chunk boundaries, and inference-only enforcement. These checks
support numerical equivalence, not a claim of bitwise identity at every size.
The suite also includes the cache-merge regression.
`musiq_chunked_8k_benchmark.json` records successful full native 8192x8192 scoring
plus four original 1024x1024 crops: 702 seconds at four CPU threads, with a process
peak of 21.76 GiB. Its image hash identifies the exact benchmark input. The original
8K implementation was not rerun for this check; numerical agreement was established
on the bounded inputs above. The first three eight-thread shard scores took
447 to 452 seconds each under concurrent CPU load.

```bash
HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch \
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  uv run --no-project --python "$SCA_PY" python repro/check_musiq_chunked.py \
  --out /scratch/09032/saini_2/sca/eval/musiq_chunked_validation.json
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  uv run --no-project --python "$SCA_PY" python -m pytest -q \
  repro/tests/test_cache_merge.py repro/tests/test_musiq_chunked.py repro/tests/test_reference_rows.py \
  repro/evaluator_snapshot/tests/test_metrics.py
```

The 8K CPU rollout uses four disjoint prompt manifests, each with three prompts
and both methods, in separate output directories. Prompt IDs are sorted and
assigned by index modulo four. Workers use eight threads and chunk size 256;
the first-image benchmark uses four threads. Do not merge while workers write.
After every worker exits successfully, combine caches with
`merge_evaluation_caches.py --destination <combined>/evaluation_cache.json
<shard-0>/evaluation_cache.json ... <shard-3>/evaluation_cache.json` and rerun the
corrected evaluator on the full prompt manifest with `--musiq-query-chunk 256`.
The merge rejects any unequal entry sharing a key before changing the destination.
It does not combine shard summary statistics; the final evaluator recomputes
summaries and paired intervals from all cached per-image scores.

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
uv run --no-project --python "$SCA_PY" python repro/evaluate_corrected.py \
  --root "$SCA_EVAL_ROOT" --prompts results/evidence/prompts/aesthetic4k_eval.jsonl \
  --references results/evidence/prompts/aesthetic4k_refs.jsonl \
  --reference-method sca_nb --reference-clusters images --res 4096x4096 \
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

`generation-freeze.txt` and `metrics-freeze.txt` describe the inherited environments.
The separate [fresh generation recipe](environment/README.md) was installed and
checked on Linux aarch64 using an isolated uv environment. It preserves the main
framework versions and resolves the inherited Hub version conflict; CPU tests and
baseline dry runs pass, with an explicit NVIDIA wheel-tag warning. It does not
claim GPU equivalence or a fresh learned-metric installation. The uv environments
resolve to scratch. Restore them and metric caches separately after a purge.

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

AReA and UltraImage wrappers now import the byte-identical helper at
`repro/evaluator_snapshot/crope/utils.py` relative to this repository. Their upstream
checkouts remain external, configurable through `AREA_DIR` and `ULTRA_DIR`; set
`ULTRA_DIR` to the checkout's `flux/` directory. Future code freezes must include
the vendored helper directory. Existing campaign freezes are unchanged. The moved
helper path and updated wrapper bytes change wrapper fingerprints, so use fresh
output directories with the portable wrappers.

`generation_snapshot/` now contains the unchanged DyPE and SEGA generation code
from account-authored commit `c05ebdcccddafd6b8fe4ec2e14247d72775df9a1`. Its 14 source
and test files have Git-blob and SHA256 provenance. The composite implementation
fingerprint matches every archived DyPE/SEGA sidecar exactly. It includes all local
runtime dependencies and the two extra scripts included in the historical fingerprint.
The selected CPU tests passed 33 cases, including direct upstream numerical parity,
seed identity and resume integrity. This does not reproduce upstream papers under
their preferred checkpoints or sampler settings, or regenerate the benchmark images.
`generation_package_validation.json` records these checks, the 16 package/FLUX.2
checks, wrapper source hashes, and the mocked embedding-cache argument check.

Run the frozen generation tests in a separate process from evaluator tests because
both preserved packages use the `crope` name:

```bash
PYTHONPATH="$PWD/repro/generation_snapshot" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-project --python "$SCA_PY" python -m pytest -q \
  repro/generation_snapshot/tests/test_rope.py repro/generation_snapshot/tests/test_runtime.py
uv run --no-project --python "$SCA_PY" python repro/generation_snapshot/scripts/sample.py --help
```

Set `SCA_REPRO_OUTPUT` to a scratch directory. The manuscript's DyPE generation
command is:

```bash
uv run --no-project --python "$SCA_PY" python repro/generation_snapshot/scripts/sample.py \
  --method dype --res 4096x4096 --steps 28 --guidance 4.5 --seed 0 \
  --prompts results/evidence/prompts/aesthetic4k_eval.jsonl --start 0 --end 195 \
  --out "$SCA_REPRO_OUTPUT/dype"
```

Replace `dype` with `sega_official` for SEGA. The generation manifest includes all
195 prompts; the paper excludes
the four IDs in `results/evidence/prompts/dev4.jsonl` when constructing the common
191-prompt evaluation root. Baseline inference uses maximum scheduler shift,
prompt-derived seeds and tiled VAE decoding, as recorded in the original sidecars.

## Pinned model and data restoration

`model_snapshots.json` records the local cache observed on October 17, including
file sizes, content-addressed blob names, and hashes of archived small JSON metadata
under `model_metadata/`. The following revisions are the reproduction pins:

| Repository | Observed cached revision |
| --- | --- |
| `black-forest-labs/FLUX.1-dev` | `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` |
| `black-forest-labs/FLUX.1-Krea-dev` | `8162a9c7b05a641be098422bf2fcf335615c2f28` |
| `black-forest-labs/FLUX.2-dev` | `26afe3a78bb242c0a8bb181dcc8937bb16e5c66c` |

This inventory is retrospective. AReA, DyPE, SEGA and UltraImage sidecars directly
record the listed FLUX.1-dev revision; historical SCA and FLUX.2 sidecars record the
model identity but omit its snapshot revision. Observing the current cache does not
retroactively prove those missing run-time fields. Large weight files were not
rehashed or copied in this audit. Krea uses the FLUX.1-dev encoders and VAE together
with the Krea transformer, so restore both repositories for that comparison.

Restore into a dedicated reproduction cache, keeping live campaign caches separate.
The code below downloads only the paths present in the audited snapshot and pins the
local `refs/main` used by the preserved loaders. Model access must already be granted.
Choose the repository needed for the run; repeat for other models as needed.

```bash
export HF_HOME="${SCA_REPRO_HF_HOME:?Set a dedicated scratch cache first}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=0
uv run --no-project --python "$SCA_PY" python - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
repo_id = "black-forest-labs/FLUX.1-dev"
model = next(row for row in json.loads(Path("repro/model_snapshots.json").read_text())["models"]
             if row["repo_id"] == repo_id)
cache = Path(os.environ["HF_HOME"]) / "hub"
revision = model["observed_cached_revision"]
snapshot_download(repo_id, revision=revision, cache_dir=str(cache),
                  allow_patterns=[row["path"] for row in model["files"]])
ref = cache / ("models--" + repo_id.replace("/", "--")) / "refs/main"
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text(revision)
PY
export HF_HUB_OFFLINE=1
```

Set `TORCH_HOME`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`, `MPLCONFIGDIR` and
`WANDB_DIR` to scratch locations on the reproduction machine; the historical files
retain Vista defaults when these variables are absent. For FLUX.2 generation, pass
`--embedding-cache "$SCA_REPRO_OUTPUT/flux2_embeddings"` to relocate its embedding
cache; omitting the flag retains the original campaign default.
For UltraImage, set `ULTRA_BLOCKAVG_ROOT` to a scratch directory for its per-prompt
block-average cache.

The Aesthetic-4K manifests pin dataset revision
`8c5d5cb8b94230ff897d87bf060451257d1c7bf8` of `zhang0jhon/Aesthetic-4K`.
Restore its `eval/size_4096/` files at that revision, then rebuild reference paths
using each archived row's `source_file_name`. Update `path`, `image` and
`reference_image` fields where present in copied manifests; preserve IDs, prompts,
all 195 rows, and source revision metadata. `upstream_image_tree.json` preserves
the expected per-file LFS identities. Generate image outputs again or restore them
from a separate artifact store for fresh learned-metric scoring. The archived
numeric/feature banks remain sufficient for the offline checks described above.

The [fresh generation recipe](environment/README.md) supplies a hash-pinned
dependency resolution and bounded CPU validation for Linux aarch64. Separate
[metric restoration recipes](metrics_environment/README.md) now provide three
installed hash-pinned dependency sets, checkpoint identities and isolated asset
restoration. Bounded CPU learned-score and feature checks match inherited outputs
exactly; the linked guide records inputs, repairs, receipts and limitations.
These checks do not reproduce full benchmark scoring or GPU metric parity.
