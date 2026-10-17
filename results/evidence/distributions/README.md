# Cached-feature distribution audit

All 152 historical FID, KID, patch-FID, and patch-KID point estimates in 17 archived
evaluation groups were reproduced exactly from their cached Inception features.
This checks feature assembly and numerical statistics. Image generation and Inception
encoder inference were not rerun. HRGen has no reference-image distribution evaluation.

The joined float32 banks preserve prompt IDs, seeds, and crop grouping. Each experiment's
`*_banks.json` records the source cache paths and hashes, input image signatures, scorer
provenance, and hashes of the joined banks. Identical banks are stored once. The full
archive is about 136 MB, without generated images or model weights.

## Reference crop assembly

The official Aesthetic-4K reference pool contains 195 rows but 168 distinct image
contents. Different upstream filenames can resolve to the same Hugging Face blob.
The original evaluator caches features by prompt ID and image signature, then assembles
its feature bank by resolved image path. For repeated reference images, the last
prompt's crop features replace the earlier prompts' crops. The original saved numbers
are reproducible under that historical policy.

The corrected `rows` policy, also called `reference_prompt_row_v2`, retains all 195
reference rows and their original weights while using each row's intended cached crops.
Whole-image FID and KID are unchanged. Corrected main-comparison patch-FID values are:

| Method | Historical path assembly | Corrected row assembly |
|---|---:|---:|
| AReA | 110.184895 | 103.411062 |
| SCA | 106.472475 | 99.815615 |
| UltraImage | 117.527434 | 110.563269 |

The historical artifacts remain unchanged. Corrected values and sensitivity results
live under `checks/`; they are not automatic replacements for manuscript numbers.
The separately labeled `unique` policy reduces the reference pool to one row per image
identity. It is a different reference distribution and must be reported as a sensitivity.

## Offline verification and bootstrap

The checker uses only WORK files and the preserved evaluator snapshot:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python \
  scripts/check_archived_distributions.py --policy rows
```

Use `--experiments final4k_nb_base --metrics fid kid --draws 1000` for paired intervals.
The original 1,000-draw whole-image FID and KID paired intervals were reproduced exactly.
The corrected main patch intervals use `--policy rows --metrics patch_fid patch_kid`.
With `--reference-clusters images`, the bootstrap resamples the 168 distinct reference
image identities, keeping all rows belonging to each sampled image together. The point
estimate still uses the same 195-row weighted distribution. This separates correcting
the crop lookup from checking uncertainty under repeated image contents.

Each check writes a CSV named by experiment, feature policy, reference cluster policy,
metric, and draw count. Concurrent checks with different names do not overwrite each
other. Original per-row bootstrap results and image-cluster sensitivities must retain
their labels. `historical_ci_max_abs_error` compares a new interval against the original;
nonzero differences are expected for corrected crops or a changed cluster policy.

To rebuild the joined archive while the scratch caches still exist, run
`scripts/verify_distribution_evidence.py` with the same four-thread environment. Its
default `legacy` mode verifies the historical numbers and preserves both the historical
assembled banks and the raw per-reference-row banks needed by the offline checker.
