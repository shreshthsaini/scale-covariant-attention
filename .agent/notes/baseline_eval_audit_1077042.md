# Baseline evaluation audit, allocation1077042

Completed a bounded audit of `repro/baseline_evaluation_20261017/` on the
current CPU compute node. No scheduler, frozen generation, task spool, controller,
manuscript or scratch outputs were changed. Root owns commit and checkpoint.

## Fixed findings

- The CPU distribution stage overwrites `base/paired_deltas.csv`. It now preserves
  `scored_paired_deltas.csv` and binds its hash to `GPU_SCORES_COMPLETE.json`, as
  already done for metrics, per-image scores and summaries. CPU reruns preserve
  all four GPU report snapshots.
- Reference manifests previously bound filenames and weights, but reference image
  bytes were not in the input receipt. Preparation now hashes each distinct
  resolved image once, retains every reference row, and each scoring barrier
  verifies the resulting identities and bytes. This detects replacement even with
  the same size and timestamp. The helper is frozen with evaluator sources.
- Shell limits were4threads, but the evaluator CLI default8 overrides them.
  Both GPU evaluator calls and the CPU distribution call now pass `--threads 4`;
  generated shell headers also bound MKL and NumExpr to4.
- New input acceptance now verifies PNG CRCs in addition to recorded SHA256,
  dimensions and exact sidecar identity.

## Verification

`repro/baseline_evaluation_20261017/audit_validation_1077042.json` records:

-13 focused tests passed in1.86seconds using the managed `sca` Python via uv,
  with OMP/OpenBLAS/MKL threads4. Tests include exact12/46 pair coverage,
  report preservation across CPU replacement, reference-byte tampering at the
  actual input barrier and same-size/same-timestamp replacement.
-39 deployed frozen generation files retain their recorded hashes.
- All6 historical controls have46 paired IDs and matching sidecar prompt,
  method, seed0, resolution4096x4096 and complete flag. The first image for each
  method also passed the full byte SHA256 and PNG CRC/dimension checks. Remaining
  control46 PNG hashes are checked by actual preparation, not claimed here.
- Full195 reference rows resolve to168 image identities. All distinct reference
  bytes were hashed, preserving the repeated row weights.
- Both missing smoke costs remain null, with explicit `smoke_not_complete`.

## Promotion helper review

Read `repro/baseline_promotion_1077042/prepare.py` in bounded chunks and its tests.
No substantive defect found in the reviewed partial-output, timing or fit logic.
The frozen runner emits matching `seconds`/`wall_seconds` and the expected finite
checks. Partial future outputs close promotion rather than being overwritten.
Observed images are excluded from contiguous ranges; each range fits its declared
budget using twice the maximum observed pipeline time plus explicit startup.
Dispatcher cleanup120seconds is separately disclosed. No quality ranking enters
promotion. The frozen manifest, config construction and output root match the
evaluation helper, so no root-override change is required.

## Limits and next action

Actual12/46 new-baseline generation, GPU scoring and1000draw distribution
intervals remain unrun. Fixture and historical-control checks establish neither
fresh GPU fidelity nor GPU precision parity. When each stage has exact validated
coverage, prepare a new workspace, run `score_gpu.sh` in an existing suitable
allocation, then `intervals_cpu.sh` on CPU. Never reuse an old prepared workspace
with newly changed evaluator files or silently import caches.
