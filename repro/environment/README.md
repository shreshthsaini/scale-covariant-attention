# Fresh generation environment

This recipe was installed into a new, isolated uv environment on October 17,
2026, on Linux aarch64 with CPython 3.11.13. It passed 67 CPU tests and imports for
the FLUX.1 and FLUX.2 pipelines. Six command-line help checks and the AReA and
UltraImage dry runs passed. No model weights were loaded, no GPU was used, and
the campaign environments and frozen jobs were not changed. This is a tested
generation dependency recipe; the separate learned-metric environment has not
been rebuilt from scratch.

`generation.in` preserves the inherited Torch 2.11.0+cu128, torchvision
0.26.0+cu128, Triton 3.6.0, Diffusers 0.39.0, Transformers 5.19.0 and other explicit
generation dependencies. The inherited Hugging Face Hub 2.2.0 violates Diffusers'
`>=0.34.0,<2.0` constraint. Transformers requires `>=1.31.0,<3.0`; resolving their
intersection selected Hub 1.33.0. `generation.lock` pins all 71 selected packages
and their artifact hashes. `fresh_validation.json` lists every selected package,
14 changes from the inherited package freeze, and packages omitted from this
smaller environment. GPU numerical equivalence between these environments has
not been tested.

## Installation

Run on a compute node. Both the environment and wheel cache are several GB.
The installer refuses to change an existing environment, copies files from its
cache, and disallows source builds. A network connection is needed for Python
packages; model weights are not downloaded by these commands.

```bash
export UV_PYTHON_INSTALL_DIR=/work/09032/saini_2/vista/library/uv-python
bash repro/environment/install.sh \
  /work/09032/saini_2/vista/library/uv-envs/sca-repro-NEW \
  /scratch/09032/saini_2/sca/repro_env_cache_NEW \
  /path/to/existing/python3.11
```

Use CPython 3.11.13 for the tested configuration. The checked environment is
`/work/09032/saini_2/vista/library/uv-envs/sca-repro-20261017`; this account's
`uv-envs` directory resolves to scratch and is subject to its purge policy.
Restore the environment using this recipe if needed. The archived lock and
evidence files reside on work.

The installer's final `uv pip check` returns a nonzero status for one preserved
upstream packaging issue: `nvidia-cusparselt-cu12==0.7.1` declares wheel tag
`py3-none-manylinux2014_sbsa`. uv reports that tag as a platform mismatch on
aarch64. The installed library's ELF header identifies AArch64 (machine 183),
and Torch imports successfully. `cusparselt_platform.json` preserves this check;
`pip_check.log` preserves the warning. No wheel metadata or binaries were edited.
Active package version requirements pass the separate metadata check. This does
not establish that cuSPARSELt GPU kernels run correctly. Investigate any additional
installation or dependency error rather than treating it as this known warning.

## Bounded validation

From the repository root, set `SCA_FRESH_PY` to the new environment's interpreter:

```bash
export SCA_FRESH_PY=/work/09032/saini_2/vista/library/uv-envs/sca-repro-20261017/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1
"$SCA_FRESH_PY" repro/environment/check_install.py /scratch/09032/saini_2/sca/fresh_import_check.json
PYTHONPATH="$PWD/repro/generation_snapshot" "$SCA_FRESH_PY" -m pytest -q \
  tests/test_sca.py tests/test_flux2.py \
  repro/generation_snapshot/tests/test_rope.py \
  repro/generation_snapshot/tests/test_runtime.py repro/tests/test_generation_package.py
```

`cpu_tests.log` records 67 passes in 20.06 seconds. The single warning concerns
deliberately unfused CPU FlexAttention in a tiny test. Run evaluator-snapshot
tests in a separate process: both historical source snapshots use the `crope`
package name and should not share one Python import namespace.

`cli_validation.json` records all eight checked commands and their return codes.
The baseline dry runs used existing source checkouts and read cached scheduler
configuration, without loading weights or creating generated images. Restore
the pinned baseline sources as described in the parent reproduction guide first.
UltraImage used the existing exact Haar level-1 shim because PyWavelets was absent,
matching the inherited configuration.

The inherited environment was missing 96 zero-byte Torch/Triton Python modules
listed in package RECORD files, consistent with scratch purge. All are present
in the fresh installation; `sca.envfix` finds an empty repair table. No shared
environment was repaired or overwritten. `inherited_audit.json` and
`fresh_validation.json` preserve the before/after evidence.

`install.log` records the fresh wheel-only installation. No GPU benchmark,
learned-image rescoring, full-model generation, or other operating-system or
Python-version compatibility claim follows from these CPU checks.

## Prepared GPU smoke check

`gpu_smoke.py` is prepared for a slot inside an existing GPU allocation. It was
not executed on GPU while preparing this recipe. It uses no model weights and
sets a 280-second worker deadline, followed by bounded process cleanup. It checks
bf16 CUDA matrix multiplication and SDPA against float32 references, then a tiny
random FLUX transformer: stock versus full attention, stock versus SCA with a
native-grid window covering the whole image, and sparse fused FlexAttention
versus the gather implementation. The fixed tolerance is absolute 0.02 plus
relative 0.02 using `torch.allclose`; each comparison records its actual error.
The SCA checks execute compiled FlexAttention kernels, using one compiler worker.
They do not directly exercise cuSPARSELt sparse kernels or reproduce a paper result.

Root may run this command within an existing scheduled GPU worker:

```bash
"$SCA_FRESH_PY" repro/environment/gpu_smoke.py \
  --out /scratch/09032/saini_2/sca/repro_env_gpu_20261017/smoke.json \
  --archive results/evidence/environment/fresh_gpu_smoke.json
```

The script refuses an existing output/archive path. Partial progress is written
atomically after each stage, so a timeout or failure remains inspectable. The final
JSON and log are copied to work only when `--archive` is supplied. The two tests in
`repro/tests/test_gpu_smoke.py` check failure preservation and timeout cleanup on
CPU; they establish no CUDA result.
