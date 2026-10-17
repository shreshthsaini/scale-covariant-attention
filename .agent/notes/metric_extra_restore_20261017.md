# Extra metrics and ImageReward restoration, 2026-10-17

Owner: metric_extra subagent under CPU controller allocation1076947. Completed
06:31CDT. No SSH, submissions, task mutations, campaign environment edits, or
controller changes. Every owned exec session is complete and reaped.

## Completed

- Full distinct managed envs sca-metric-extra-20261017 (123packages) and
  sca-metric-imagereward-20261017 (80packages after repair) installed through
  uv hash-locked sync, reusing the managed UV cache.
- Two fixed existing full4K area image/prompt/sidecar identities are archived in
  results/evidence/metric_environment/extra_parity_inputs.json.
- Predeclared scalar atol1e-5/rtol1e-5 and Inception-logit atol1e-4/rtol1e-5
  were retained. All seven scorers match inherited sca-metrics CPU float32
  outputs bit for bit:12 scalar values and2016 Inception logits, max_abs0.
- Six parity acceptance tests, three tokenizer repair tests and five CLI
  argument/help checks pass. Root's broader suite separately covers integration.
- Real portable extra6 and IR1 CLIs each finish with complete identity manifests,
  two rows, bootstrap0 and no errors. All2028 scalar/logit values match the
  direct parity harness. Actual split-report merger accepts their seven metrics.
- extra_restore_validation.json and imagereward_restore_validation.json
  contain exact source/lock/input/log/report/identity/asset hashes and commands.
  All paths here are relative to results/evidence/metric_environment unless noted.

## Concrete failures and repairs

- Account HUGGINGFACE_HUB_CACHE overrode HF_HOME. Initial all-seven inherited
  success is preserved as extra_parity_inherited_legacy_cache_attempt.json.
  Explicit HF_HUB_CACHE/HUGGINGFACE_HUB_CACHE settings now select the isolated
  cache, and the harness verifies actual imported HF/Torch cache directories.
- Restored refs/main contained trailing newlines, breaking offline HF lookup.
  Root fixed helper/refs and archived ref_repair_receipt.json. The failed run
  remains extra_parity_inherited_newline_ref_failure.json.
- ImageReward1.5 imports clip unconditionally without declaring the dependency.
  imagereward.in now explicitly requires openai-clip1.0.1; the regenerated lock
  adds its ftfy/wcwidth dependencies. Old/new locks and hashes, failed run and
  sync logs are retained in imagereward_dependency_repair.json and its artifacts.
- HPSv2's wheel lacks its BPE data. The helper at repro/metrics_environment/
  restore_hps_tokenizer.py copies the pinned openai-clip asset only after
  verifying its SHA256 equals the archived inherited HPS asset. Existing
  mismatched files are preserved and rejected. Extra install.sh invokes this
  helper before uv check. The real fresh repair receipt and prior scoring failure
  are archived. No scorer implementation changed.
- uv pip check returns1 solely for the preserved nvidia-cusparselt-cu12 platform
  tag. Zero dependency-version conflicts. The warning is recorded, not suppressed.
- args/clint/fairscale source wheels have no native extension files; no native
  toolchain build was requested. Build-system dependencies are not independently
  hash pinned, consistent with the recipe's existing limitation.

## Scope and next action

The bounded restoration task is complete. This establishes CPU environment
parity only, not CUDA/mixed-precision parity or full benchmark reproduction.
Two-image IS with one split is a CLI diagnostic, not an IS benchmark.
Root owns shared README/STATE/HANDOVER integration, local commit and delivery.
No processes or long commands remain for this subtask.
