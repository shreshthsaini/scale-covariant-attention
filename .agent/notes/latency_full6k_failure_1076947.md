# Full 6K latency assertion and bounded repair

Reviewed 2026-10-17 06:30-06:40 CDT on CPU allocation 1076947. The full-attention latency attempt failed with exit code 1 before completing its warm-up. It was not a timeout and produced no completed image. The dispatcher moved the task into `failed` and immediately started productive AReA work on the same worker. A blind retry would repeat the assertion.

## Cause and scope

The historical `code/2d11740/scripts/sample.py:56` created `SCAContext(mode="full")` without the known image grid. Its transformer pre-hook still calls `set_inputs`, which inferred dimensions from bfloat16 image positions. At 6144 pixels the actual packed grid is 384 by 384 (147456 tokens), but bf16 rounds position 383 to 384. The fallback inferred 385 by 385, then failed `sca/attn.py:447`: `image ids must form a full grid`.

The minimal repair passes `known_grid=grid` in the full branch, as the SCA branch already does. It does not change image IDs, RoPE coordinates, precision, attention, or the sampling protocol. Full attention continues through its existing SDPA path. Incorrect token counts still raise the assertion.

Mutable `scripts/sample.py` now contains the one-line repair. The historical frozen source remains untouched. Mutable sampling also includes separate Krea changes, so it must not replace the historical sample wholesale. The historical `sca/attn.py` and mutable counterpart are byte-identical (SHA256 `09c201611917c5ad0d3dd10037d70ed69a8462191d972376c10a4beae7fa855f`). Historical and mutable `sca/runtime.py` differ because of other existing work.

## Validation

With OMP, MKL and OpenBLAS each bounded to four threads, the managed `sca` environment passed:

- `tests/test_sample_grid.py tests/test_sca.py`: 22 tests in 21.89 seconds. The only warning was the existing unfused CPU FlexAttention warning.
- `tests/test_sample_grid.py` against an isolated historical sample with exactly the one-line repair: four tests in 7.37 seconds. The historical source was not edited.

The regression reproduces the old bf16 failure for 384 by 384, 128 by 384, and 512 by 512 grids. It then verifies the intended dimensions, exact preservation of every bf16 coordinate, and rejection of an incorrect token count. The tiny random transformer comparison verifies that explicit dimensions preserve the stock full-attention output within the existing absolute tolerance of 1e-5.

Historical sample SHA256 before repair: `8446dd11f9b574b8f601eec5820780711eae20e6a43264a6f581011187b40578`.

Historical sample SHA256 after the single replacement: `75e0769b1dad082a0c5e937d837e83cdc9cc0b0c4058fc11c6098f1e615c59f1`.

The isolated test source is `/scratch/09032/saini_2/sca/full6k_grid_fix_validation_1076947/scripts/sample.py`. Its test log and JUnit receipt are archived under `results/evidence/latency_full6k_failure_1076947/`.

## Ownership and preserved evidence

The immutable failed attempt was `/scratch/09032/saini_2/sca/logs/attempts/a_y003_latency_6144x6144_full.1792236599445480025.c610-092-g0-j1075814-p1379015-b8963cdb.log`, with adjacent JSON metadata. Start was 06:29:59 CDT; terminal exit code 1 was recorded at 06:31:42. The same worker then claimed `i024_area_6144x6144.sh` at 06:31:42.

Copies of the original failed task, immutable log, attempt metadata, and worker event snapshot are preserved under `results/evidence/latency_full6k_failure_1076947/`. `receipt.json` contains their hashes. At review, `/scratch/09032/saini_2/sca/out/latency_curve/6144x6144/full` did not exist. Any output that appears before a retry must still be preserved.

The four GPU telemetry files at 06:33:59 CDT reported 99-100 percent utilization, with about 49-89 GiB occupied, on nodes c610-042, c610-051, c610-092 and c610-101. Initial low utilization was model loading. No cancellation, co-location, or other utilization repair was indicated.

## Exact recovery proposal for the main operator

Root will create `/scratch/09032/saini_2/sca/code/2d11740_full_gridfix_1076947` from the original `2d11740` freeze, apply only this replacement in its `scripts/sample.py`, and record a complete before/after manifest:

```python
old = 'return SCAContext(mode="full"), {}'
new = 'return SCAContext(mode="full", known_grid=grid), {}'
assert source.count(old) == 1
source = source.replace(old, new)
```

Verify the resulting sample has the historical patched hash above and every other frozen file is unchanged. The prepared candidate task is `results/evidence/latency_full6k_failure_1076947/a_y003_latency_6144x6144_full_gridfix_1076947.sh`, SHA256 `65c27ee22b038f7c4c6f9aa942f42c939028c9866275e55916e4eda06484c04d`. It uses that new freeze and retains `SCA_MIN_SECONDS=4800`, the same four prompts, seed, guidance and one-process warm-up plus three timed images.

Keep the hash-checked candidate in `taskq/hold_review` until root verifies that owning step `1075814.0` and its job are gone. Immediately before release, inspect live ownership and task receipts, with scheduler polls spaced at least ten seconds, and verify no running, pending or completed duplicate. Preserve the original failed task in recovered history with its terminal receipts. Then release the uniquely named candidate through the normal spool. This subagent made no taskq changes and no scheduler submissions.

The retry task reserves a fresh history directory with `mktemp -d` and moves any prior latency output into its `attempt` child. This preserves partial images and avoids mixing a prior warm-up or timed sample into the new four-image process. Do not use resume skips to complete latency trials.

If no eligible worker has at least 4800 seconds remaining, the task needs a sufficiently long existing or replacement fleet. The assertion is a software failure, while the missing latency result remains pending scientific evidence. No GPU success is claimed by this CPU validation.
