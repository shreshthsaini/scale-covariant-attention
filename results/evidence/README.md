# Archived evidence

This archive preserves the paper's numeric records, generation metadata, calibration
tables, and prompt manifests on WORK. Images and feature caches remain on scratch.
`manifest.json` records source paths and SHA256 hashes. `generation_sidecars.jsonl.gz`
contains the complete sidecars for 3,797 unique evaluated image paths, including image
hashes, settings, runtimes, and implementation fingerprints.

Rebuild the archive and check the archived numbers from the repository root on a compute
node using the existing `sca` uv environment:

```sh
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python scripts/archive_evidence.py
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python scripts/audit_evidence.py
```

The archive helper also refreshes E2 and E9 when both evaluator records report completion,
all requested metrics are available, and every expected method has the exact expected
prompt IDs: 100 at 6K, 12 at 8K, and 100 on Krea. Partial results are skipped. It creates
flat summaries and paired-delta CSVs with the same experiment prefixes in `results/`.

The October 17 audit rederived all 762 scalar means exactly, including the completed
12-prompt 8K evaluation, verified 195 archive hashes, found equal prompt sets
between methods within every evaluation, and reproduced all 13 per-head calibration
fits and relative squared errors from seven archived tables. The original 32-prompt
calibration split-half correlations were also rederived: 0.9885567 at scale 2 and
0.9863842 at scale 4. The two half-statistic tensors are preserved in
`calib/k1_1024_split_half_stats.npz`; source part hashes and results are in
`calibration_split_half_audit.json`.

## Coverage and interpretation

| Experiment | Paired prompts | Run seeds |
|---|---:|---|
| E1 baseline comparisons | 191 | 0 |
| E4 HRGen | 200 | 0 |
| E5 seed checks | 46 each | 1, 2 separately |
| E6 calibration budget | 46 | 0 |
| E7 gate thresholds | 46 | 0 |
| E8 FLUX.2 | 50 | 0 |
| Step and guidance budget | 46 | 0 |

Every listed set excludes the four development prompt IDs. HRGen is the exact first
200 prompts in the vendored AReA HRGen-1K file. It is a fixed subset, not the full benchmark.
The FLUX.2 subset is a separate 50-prompt set. Its protocol uses guidance 4.0 and native
scheduler shifting; the FLUX.1 shared protocol uses guidance 4.5 and max-shift scheduling.
UltraImage uses its two-stage 60-step procedure and guidance 3.5 then 7.0.

The original evaluator summaries use 1,000 bootstrap draws. Flat files produced by
`paired_bootstrap.py` use 2,000 draws, seed 0, and report separate resolution/run-seed
groups. Missing pairs, duplicate measurements, and nonfinite values now cause an error.
The script resamples prompts, not image crops. CLIP is 100 times nonnegative cosine;
HPS values are raw scores rather than values multiplied by 100. KID files store raw
units. Native MUSIQ and the within-image average over four fixed 1024-pixel crops are
separate measurements.

The budget result has a quality tradeoff: doubling SCA steps improves several preference
metrics and crop MUSIQ, while patch FID and patch KID worsen against 28-step SCA. The
56-step SCA preference differences against UltraImage are not significant on these
46 prompts; this does not establish equivalence. Budget and HRGen runtimes came from
campaign workers and should be distinguished from isolated latency measurements.

## Reproduction limits

The scalar and calibration audit uses saved sufficient statistics. The subsequent
cached-feature audit in `distributions/` rederived all 160 FID/KID and patch-FID/KID
point estimates across 18 groups and preserved joined feature banks for offline checks.
It also identified and isolated a reference crop lookup issue for repeated image paths;
historical and corrected row-weighted results remain separate. Inception Score was
not recomputed. Absolute FID percentile bootstrap intervals may exclude
the original point estimate because empirical FID has finite-sample bias; use paired
delta intervals for comparative claims.

Historical evaluation tasks used the live constancy_rope evaluator rather than a pinned
copy. `evaluator_sources.json` hashes the sources at archival time, and
`environment_versions.json` records the current sca and sca-metrics package inventories
without importing the models. Neither file establishes the historical evaluator revision.
Stored metric metadata retains the versions and definitions reported during evaluation.
