# GPU smoke compiler repair, job 1076947

Completed on CPU compute host i615-074, 2026-10-17.
Root owns all task mutations, ownership checks and requeue decisions.
This unit made no scheduler, SSH, task, frozen dispatcher or campaign changes.

## Cause and repair

The first fresh GPU smoke passed bf16 matrix multiplication, SDPA and the tiny
FLUX stock/full comparison. Native-window compiled FlexAttention failed because
the account exports `CC=/home1/apps/nvidia/Linux_aarch64/24.7/compilers/bin/nvc`.
Triton 3.6.0 reads `CC` directly and passes `-Wno-psabi` when compiling its CUDA
host shim. NVIDIA nvc rejects that flag. This is a compiler configuration failure,
not a scientific negative result and not a successful compiled SCA check.

The repaired worker selects explicit GNU gcc/g++ before importing Torch, replacing
inherited CC/CXX. It records the inherited and selected paths, compiler versions
and binary hashes. It compiles/loads C with Python.h and Triton's host flags plus
C++17, requiring both shared-library probes to return42. Failures retain commands,
diagnostics, sources and partial JSON. Each subprocess has a bounded timeout.
The numerical tolerance stays atol0.02/rtol0.02. Both compiled FlexAttention
checks and mandatory compiled-cache assertions are unchanged, with no fallback.

## CPU evidence

Scratch root: `/scratch/09032/saini_2/sca/repro_env_gpu_compiler_repair_1076947`.
All owned subprocesses finished and were reaped.

- `gcc_preflight_final.json`: passed with GNU gcc/g++11.5.0, both load values42.
  SHA256 `ed006291617171309bcc28925f26beb563c8d4c9a78b852031be2e7f6988a3fc`.
- `nvc_expected_failure_final.json`: independently reproduced the original
  `nvc-Error-Unknown switch: -Wno-psabi` before importing Torch.
  SHA256 `fe120c525cf5442ed5dac1ce99f265b4605b08428db624a79455083044b469ff`.
- `tests_final.log`:7 passed in3.09s, including real GCC compile/load, explicit
  selection over inherited NVIDIA settings, missing/rejecting compilers,
  compiler-only CLI evidence, supervisor failure preservation and timeout cleanup.
- Each JSON has its corresponding `.log`. `original_failure/` preserves the
  original GPU failure JSON/log copied before editing. Those originals also remain
  untouched at `results/evidence/environment/fresh_gpu_smoke.{json,log}`.
- Original GPU JSON SHA256:
  `d7fd4b231526025c145fc89e02c2d33ba56f1aac76cd195cbd036b71fd8df49c`.
- Original GPU log SHA256:
  `93baa93fa4b3049ef4daf3da035eec1230c9c53017abc0cf7fb5e24bfa7bc029`.

These CPU probes verify host compilation only. GPU linking, compiled Triton
kernels and SCA numerical checks require the retry below. Archive the final
preflight JSON/log and test log on work together with a successful GPU receipt.

## Exact proposed retry task

Suggested distinct task filename: `a_000_repro_gpu_smoke_gcc_1076947.sh`.
Publish only after root verifies the failed attempt's owning step is gone.
Do not overwrite or reuse the original task/output/archive paths. No submission
is required if an existing SCA GPU fleet can claim this420-second task.

```bash
#!/bin/bash
# SCA_MIN_SECONDS=420
set -euo pipefail
printf '%s\n' 'e02a2a7ba97b2571f43344518e2366ab84109acf9fc7eca4568953a7ede0e14d  /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/repro/environment/gpu_smoke.py' | sha256sum -c -
printf '%s\n' '29238d938c36c1ef6bc6c50f7676bafe6ce6b322c707bae0e078c8a8c5e8d4fa  /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/repro/environment/generation.lock' | sha256sum -c -
printf '%s\n' '09c201611917c5ad0d3dd10037d70ed69a8462191d972376c10a4beae7fa855f  /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention/sca/attn.py' | sha256sum -c -
cd /work/09032/saini_2/vista/Projects/Exploration/Scale_Covariant_Attention
uv run --no-project --python /work/09032/saini_2/vista/library/uv-envs/sca-repro-20261017/bin/python python repro/environment/gpu_smoke.py --cc /usr/bin/gcc --cxx /usr/bin/g++ --out /scratch/09032/saini_2/sca/repro_env_gpu_compiler_repair_1076947/gpu_retry.json --archive results/evidence/environment/fresh_gpu_smoke_gcc_1076947.json
```

Changed source paths: `repro/environment/gpu_smoke.py`,
`repro/tests/test_gpu_smoke.py`, `repro/environment/README.md`, and this note.
Source SHA256 at handoff:
`e02a2a7ba97b2571f43344518e2366ab84109acf9fc7eca4568953a7ede0e14d`.
No commit was created by the delegated worker.
