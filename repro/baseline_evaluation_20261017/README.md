# Evaluation handoff for the new baselines

The generation package and its two queued smoke tasks remain unchanged. This
directory prepares evaluation only after exact12/46 coverage. It submits no jobs,
does no generation and implements no metric estimator. Use it on a compute node.

```bash
SCA_EVAL_PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python
uv run --no-project --python "$SCA_EVAL_PY" python \
  repro/baseline_evaluation_20261017/prepare.py --cost-only
uv run --no-project --python "$SCA_EVAL_PY" python \
  repro/baseline_evaluation_20261017/prepare.py --size 12 \
  --workspace /scratch/09032/saini_2/sca/eval/baseline_expansion12_v1
```

The first command reports actual smoke duration if present, otherwise explicitly
reports that an estimate is unavailable. It projects additional generation beyond
the first smoke for12,46 and191 prompts, with a1–2times planning envelope.
This is not a confidence interval or measured clean latency. Model loading,
evaluation and retries remain unmeasured additions. Refresh after the pilot,
subtract images already completed, include those observed costs, check the current SU balance and existing job
reservations, and convert GPU hours using the actual allocation's charge rate.
Do not infer a total campaign budget from one cold image.

The second command refuses incomplete generation before creating a workspace. It
verifies every file in the root deployment receipt, new-output resume provenance
and runtime versions, complete flags, PNG hashes, dimensions, prompt text/ID,
method, seed0 and4096². It uses the frozen46-line manifest, selecting first12 or all46.
The distinct8K hi12 set is not used. Every new result is paired with the same IDs
from full attention, AReA, DyPE, SEGA, guided UltraImage and SCA. Historical control
paths come from archived reports, and original PNG/sidecar checksums are verified.
No missing methods or prompts are silently dropped.

After passing, the helper creates a new scratch workspace with a symlink-only
sample view, exact prompt manifest, full195-row reference manifest, hashed input
receipt, and a separate copy of the unchanged evaluator sources with SHA256 manifest.
The original168 reference image identities retain195 row weights. Reference bytes
are hashed once per distinct resolved image and checked again at every input/score
barrier, including CPU reruns. All evaluator commands explicitly use four threads.
It prepares:

- `score_gpu.sh`: corrected base metrics/features and portable extra metrics with
  bootstrap0. Both retain the existing GPU metric precision and preprocessing.
  It validates all inputs before scoring and all metric/feature coverage afterward.
- `intervals_cpu.sh`: rechecks the score barrier, uses cached Inception features
  for1000 reference-image-cluster distribution draws, and uses the existing paired
  bootstrap script for2000 scalar draws, seed0, comparing each new method to SCA.
  It preserves all four GPU reports, including paired deltas and the scalar CSV,
  before the distribution-only reports are written.

The main operator may run these scripts through existing allocations/spools after
review; neither script requests GPUs or polls Slurm. GPU scoring must finish before
CPU intervals. The CPU barrier requires exact eight-method coverage, finite scalar
values, complete error-free reports, unchanged source/input bytes, matching metric
package/model revisions, and every required finite2048-dimensional feature array.
It compares fresh sample discovery to the input receipt, verifies the extra-metric
identity receipt, and rejects missing caches. A separate cache-only guard also
refuses Inception construction if a cache disappears after validation. CPU reruns
read immutable GPU score snapshots, whose hashes remain bound to the completion
marker, while the live distribution report may be replaced. The current scripts
do not automatically enqueue scoring or bootstrap tasks.

Cache reuse is conditional, not promised:

- Base cache keys include resolved PNG path, size/mtime, prompt text/ID, source
  config hash, metric schema, crop seed/count, package versions and model revisions.
  Symlinks preserve original resolved paths. Keys do not record extraction device,
  so matching keys or package versions alone do not establish GPU precision parity.
  This workflow starts with an empty output directory. Do not import base caches
  until a separate audit verifies source-report device, extraction provenance,
  exact input identities, settings and hashes. Never modify source caches.
- Keep the existing crop defaults: `--crop-seed 0 --crops 4`. Changed environments,
  weights, source configs, crop settings or file signatures require new scoring.
- The portable extra evaluator binds its cache to the complete input-identity set.
  Do not transplant historical caches lacking this receipt, or reuse a12-image-set
  portable cache as a46-image-set cache. Use a new46 workspace and rescore extras.
  Archived control scalar rows may be reused in a separately audited table merge
  only when exact image/protocol/version/precision identity is established. The
  prepared workflow does not silently perform such a merge.

For46, rerun preparation with `--size 46` and a new workspace. Any base cache transfer
from12 requires the separate extraction-provenance audit above. Preserve12 reports before extending.
For191, prepare an explicitly reviewed full manifest and retain valid earlier
outputs through an audited combined view; do not change active frozen manifests.

Promotion depends on implementation fidelity, finite/non-corrupt outputs and
affordable measured runtime, never whether results favor SCA. Retain every loss
and failure.12/46 are bounded checks, not the final completion criterion. HRDiT's
30+17+10 progressive steps remain a separate protocol from direct28-step methods
and guided UltraImage's60 steps. Existing SCA56-step and UltraImage budget controls
should inform that separate-system comparison; similar total step counts across
resolutions do not establish equal compute. The final191 campaign should reconcile
the complete paper metric suite, including IS if retained in the final table.
Small-sample distribution estimates are exploratory.
After final checks, archive reports, scalar CSVs, source/input receipts and required
feature banks on work, then update tables and paper claims from actual results.
