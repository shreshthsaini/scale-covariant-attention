# HRDiT and SigMa baseline preparation

These are prepared, unrun GPU baselines. They do not add paper results. The existing
campaign already measures full attention, AReA, DyPE, SEGA and guided UltraImage.
The primary-paper comparison is in `.agent/notes/baseline_audit_20261017.md`.

Official sources are byte-preserved under `vendor/`, with commit, Git blob and
SHA256 receipts in `sources.json`. HRDiT and SigMa repositories use MIT licenses;
inherited Apache source headers remain intact. Model licenses are separate. No
weights or upstream paper PDFs are redistributed here.

| Profile | Backbone | Steps and guidance | Purpose |
|---|---|---|---|
| `sigma_dev_matched` | FLUX.1-dev |28;4.5| Initial shared-protocol comparison |
| `sigma_krea_matched` | Krea transformer, dev encoders/VAE |28;4.5| Optional shared Krea comparison |
| `sigma_krea_native` | Krea transformer, dev encoders/VAE |28;4.5| Released CLI settings; schedule numerically duplicates matched Krea |
| `sigma_dev_paper_settings` | FLUX.1-dev |28;3.5| Final paper's stated backbone/guidance, released sampler |
| `hrdit_dev_author` | FLUX.1-dev |1K:30/3.5;2K:17/4.5;4K:10/6| Released progressive protocol, separate budget |

SigMa's final paper states dev/guidance3.5, while its released CLI hardcodes
Krea/guidance4.5. Its pipeline explicitly caps mu at max_shift=1.15 using the target
sequence length, so the two Krea profiles do not supply an independent schedule
ablation. Do not spend GPU time on both as if they did. The initial smoke uses
`sigma_dev_matched`; the other profiles remain explicit optional controls.

HRDiT retains the shipped scope plan, fp16, SPA3/0, group80 and global-anchor32.
Its57 steps are spread across three resolutions. SigMa uses bf16 and YaRN.
All adapters use campaign CPU per-prompt generators, local pinned model components,
resident CUDA models and tiled VAE decoding. These differ from upstream CLI seed,
device/offload and decode choices and are recorded in each sidecar. HRDiT's global
Torch RNG is also reset per prompt because GaussianBlur draws random sigma.
These runs are controlled adaptations, not exact reproductions of author tables.

`run.py` defaults to plan mode. It refuses partial/mismatched outputs, verifies
vendor files, pins snapshots, checks model metadata hashes, blob identities and
sizes, and refuses network downloads. Weight bytes were not independently rehashed.
Runtime package versions and source/manifest hashes are part of resume identity.
Actual sigmas/timesteps, PNG checksum and pipeline time are recorded. Latents are
checked for finite values after every step; VAE decoder outputs are checked before
conversion to integer pixels, including tiles. These checks add synchronization
overhead to the reported campaign timing. Runtime
includes text encoding and decode but excludes model loading, without an excluded
warmup; it is not a clean latency benchmark.

CPU checks use the campaign environment, unchanged:

```bash
SCA_BASELINE_PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python
SCA_BASELINE_SRC=$PWD/repro/baseline_expansion_20261017
uv run --no-project --python "$SCA_BASELINE_PY" python -m pytest -q \
  "$SCA_BASELINE_SRC/test_adapter.py"
```

Both official transformers pass tiny CPU forwards. SigMa uses its enabled YaRN
modulation on16 tokens with coordinates spanning0–255. This exercises the
extrapolation branch but is not a full4K attention test. HRDiT's CPU check excludes
the HAP GPU mask/kernel path. No pretrained generation or GPU compatibility is
claimed yet. The exact reports are `*_cpu_preflight.json`.

HRDiT wavelet dependencies are isolated in the managed uv environment
`/work/09032/saini_2/vista/library/uv-envs/sca-baseline-expansion-20261017`.
The runner adds only its site-packages to the existing interpreter. It contains
PyWavelets1.9.0 and pytorch-wavelets1.3.0;102 package/metadata files are hash-checked
against `dependency_overlay.json`. No shadow Torch or NumPy is installed there.
Restore it, if needed, on compute with:

```bash
uv venv /work/09032/saini_2/vista/library/uv-envs/sca-baseline-expansion-20261017 \
  --python /work/09032/saini_2/vista/library/uv-envs/sca/bin/python
uv pip install --python \
  /work/09032/saini_2/vista/library/uv-envs/sca-baseline-expansion-20261017/bin/python \
  --no-deps PyWavelets==1.9.0 pytorch-wavelets==1.3.0
```

Restoration must pass receipt checks; regenerated package metadata may require a
new reviewed freeze. Do not alter a dependency receipt while tasks use it.

The main operator freezes this directory at
`/scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1`.
Two scripts under `task_candidates/` point to exactly that path and to the new output
namespace `/scratch/09032/saini_2/sca/out/baseline_expansion_20261017_v1`.
They have not been queued. They validate all source hashes, run offline, select GNU
compilers, and request3600seconds of allocation time plus dispatcher cleanup.
After copying the package, the main operator can independently regenerate them:

```bash
uv run --no-project --python "$SCA_BASELINE_PY" python \
  "$SCA_BASELINE_SRC/prepare_tasks.py" \
  --frozen-root /scratch/09032/saini_2/sca/code/baseline_expansion_20261017_v1 \
  --out-root /scratch/09032/saini_2/sca/out/baseline_expansion_20261017_v1 \
  --tasks-dir /scratch/09032/saini_2/sca/baseline_candidates_20261017/smoke \
  --stage smoke
```

This creates candidates only. The root operator owns publication into the existing
spool; no new Slurm jobs are submitted by these tools. First perform one4K image
per method, inspect finite/non-corrupt output and implementation fidelity, and
measure runtime/cost. Then the same command with a new task directory and
`--stage pilot12` validates both completed smoke records and prepares the remaining
11 images per method in three chunks. Pilot tasks opt into WandB logging in
`scale-covariant-attention`, group `baseline-expansion-20261017`; per-image completion
and timing are logged, without image uploads. Smoke has no WandB run. Existing
dispatcher telemetry remains responsible for GPU utilization logging.
It refuses if smoke records, runtime versions,
or source provenance do not match. It does not judge image quality scores or enqueue.

`pilot46.jsonl` is the existing abl50 ordering with fixed dev4 removed. First12
form a nested pilot; no prompt selection used outcomes. Keep the same46-line
manifest for smoke, pilot and expansion so resume identity is stable. Use existing
191-prompt methods on exactly these IDs when comparing. Do not mix these12 with
the distinct exploratory8K hi12 subset.

Promotion depends on fidelity, valid outputs and affordable measured cost, never
favorable metrics. Retain all unfavorable results. After12/46 protocol checks,
the continuation should consider matched191-prompt comparison within the available
SU budget. The initial46 is bounded preflight scope, not a final completion gate.
No automatic191 task expansion is prepared. Review an immutable new manifest and
output namespace before extending beyond46.

Evaluation should use the unchanged corrected evaluator and split metric tools.
Extract features/scores on GPU with bootstrap0, then run1000 reference-image-cluster
distribution draws on CPU. Compute paired scalar95% intervals with2000 draws,
seed0. Keep small-sample distribution results exploratory and retain each method's
sampling budget. The output PNG/JSON schema is compatible with
`repro/evaluator_snapshot/crope/metrics/io.py`; use unique method labels above.
