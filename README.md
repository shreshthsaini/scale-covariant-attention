# Scale-Covariant Attention

SCA modifies frozen diffusion-transformer attention for high-resolution generation.
Each image head mixes a block-centre global branch with a native-extent local window.
The mixing weights are fitted against full-attention outputs at native resolution.
An initial global-only phase preserves large-scale layout.

The implementation supports FLUX.1-dev, the FLUX.1-Krea-dev transformer, and FLUX.2-dev.
This is the research workspace. The paper reports the tested protocols and their limitations;
it does not establish superiority on every metric or every checkpoint.

## Paper and evidence

- [Paper PDF](paper/main.pdf) and [LaTeX source ZIP](paper/sca_overleaf.zip).
- [Live handover](HANDOVER.md) records the remaining campaign work.
- `results/` contains numeric summaries; `results/evidence/` archives per-image metrics,
  calibration tables, prompt manifests, generation sidecars and file hashes.
- Figure sources and selection provenance are in `paper/figures/`.
- `PROPOSAL.md` is the original design, not the final implementation specification.

## Tested environment

Runs used NVIDIA GH200 GPUs on Vista and Python 3.11.13 in uv environments.
The generation environment is `/work/09032/saini_2/vista/library/uv-envs/sca`;
preference and image-quality metrics use the separate `sca-metrics` environment.
Exact installed-package snapshots are in `repro/generation-freeze.txt` and
`repro/metrics-freeze.txt`. These record the experiment environment, not a newly
validated portable dependency resolution. In particular, the inherited generation
environment has a diffusers/huggingface-hub version-constraint mismatch; the tested
imports and generation work, but a fresh solver may reject that snapshot.

The generation environment also has scratch-purged zero-byte modules.
`sca/envfix.py` supplies only these recorded empty modules in memory. Do not reinstall
packages in the shared environment while campaign processes are active.

All commands below run from this repository on a compute node. Use scratch for images,
caches and model weights. Model downloads require the upstream model access permissions.
The loaders use local Hugging Face snapshot caches and do not download weights automatically.

```bash
export HF_HOME=/scratch/09032/saini_2/tmp/hf
export TORCH_HOME=/scratch/09032/saini_2/tmp/torch
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton
export OMP_NUM_THREADS=8
SCA_PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python
```

## Generate with the calibrated FLUX.1 method

Each prompt file contains one JSON object per line with `id` and `prompt` fields.
Resolution arguments are height by width. The seed is derived from the prompt ID
and the requested seed, so task chunking does not change an image's noise.

```bash
uv run --no-project --python "$SCA_PY" python scripts/sample.py \
  --method sca --res 4096x4096 --prompts scripts/calib_prompts.jsonl \
  --start 0 --end 1 --seed 0 --steps 28 --guidance 4.5 \
  --pi-table results/evidence/calib/k1_1024.npz --gate-sigma 0.89 \
  --label sca_nb --out /scratch/09032/saini_2/sca/out/readme_smoke
```

That command uses a calibration prompt only as a runnable example, not an evaluation
sample. Evaluation uses the archived evaluation manifests and excludes development prompts.
Outputs are PNGs and JSON sidecars with configuration, timing, implementation fingerprint,
and completion status. Resume refuses incompatible settings. Use a new output directory
when changing settings or source code.

For Krea, add `--model krea` and use `results/evidence/calib/krea_1024.npz`.
For FLUX.2, use `scripts/sample_flux2.py`, the `flux2_1024.npz` table, guidance 4.0,
`--mu-mode native`, and `--gate-sigma 0.959`. FLUX.2 caches text embeddings before
loading the transformer; its reported sampling latency excludes that cached encoding.
The FLUX.1 and FLUX.2 timing scopes must not be pooled.

## Calibrate, test and evaluate

```bash
uv run --no-project --python "$SCA_PY" python scripts/calibrate.py \
  --prompts scripts/calib_prompts.jsonl --count 32 --res 1024x1024 \
  --scales 2,4 --steps 28 --guidance 4.5 \
  --out /scratch/09032/saini_2/sca/calib/new_1024.npz
uv run --no-project --python "$SCA_PY" python -m pytest -q tests
```

`scripts/eval_extra.py` computes preference and no-reference quality metrics.
Use `repro/evaluate_corrected.py` for CLIP/PickScore/MUSIQ/FID evaluation. It preserves
reference crop features for every prompt row, including rows sharing an image file.
The historical evaluator remains unchanged in `repro/evaluator_snapshot/` for reproducing
the original reports; its patch FID/KID reference assembly contains the documented bug.
Pass `--crope-root "$PWD/repro/evaluator_snapshot"` to `scripts/eval_extra.py` to use
the same archived helpers. [Evaluation reproduction instructions](repro/README.md)
include tested imports, numerical checks, upstream commits and remaining dependencies.
The generation wrappers still use external upstream checkouts documented there.
Archived paired per-image metrics can be checked without loading any model.

## Build the paper

```bash
uv run --no-project --python "$SCA_PY" python paper/build_delivery.py
```

This runs Tectonic, packages the LaTeX sources and rendered figures, extracts the ZIP,
and checks a second independent build before replacing the deliverables. It preserves
build logs on scratch and records hashes in `paper/delivery_receipt.json`.
