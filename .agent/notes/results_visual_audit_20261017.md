# Results visual audit, 17 October 2026

Completed on CPU workspace i618-051. No new generation, scoring, GPU allocation,
or SSH was used. Root owns manuscript integration and project state.

## Verified latency extension

`scripts/audit_latency.py --allow-partial` now verifies 13 groups. Full attention
at 6K remains missing and must not be inferred from another resolution or its
failed attempt. Each group has one excluded warm-up and three timed prompts in
one process. The auditor checks manifest order, no skipped/resumed output,
28 steps, guidance 4.5, seed label 0, sidecar/log agreement within 0.0006 seconds,
and `first_call=false` for each timed sample. The two new AReA task scripts
invoke the frozen `2d11740` wrapper once, preserve previous attempts, and use
`set -e`. GPU preambles show an idle GH200 120GB without an existing compute PID.
No CPU offload is recorded. The idle check is before the process, not continuous
telemetry; the timing evidence is three prompts, not independent hardware runs.

| Resolution | AReA median (s) | Observed range (s) | SCA median (s) | AReA/SCA |
|---|---:|---:|---:|---:|
| 4096 square | 202.437766349 | 192.7143 to 204.2793 | 71.064959488 | 2.848629871 |
| 6144 square | 482.284412914 | 426.430220487 to 510.913265883 | 161.558499005 | 2.985199887 |
| 8192 square | 951.987426512 | 934.794426899 to 982.248244258 | 287.639789863 | 3.309651377 |

Excluded AReA warm-ups: 508.287086176 seconds at 6K and 967.559707228 at 8K.
New scripts, raw logs, all eight sidecars, and audit provenance are archived in
`results/evidence/latency/`. Canonical tables are `results/latency_curve.csv` and
`results/latency_curve_summary.csv`. Whiskers are observed minimum/maximum,
not confidence intervals. No shared-campaign timing enters these figures.

New log SHA256 values:

- 6K: `fd30f683924cc29da9d06418b47aa5222efeebc2ba8db1c6504ff9892a4a0e5d`
- 8K: `8aa7ae74876156adbea72ad77396ddf9458bea684b54c9eed68e903a60c7fc81`

## Paired baseline quality figure

`paper/figures/src/fig_baseline_deltas.py` reconstructs every point directly
from archived per-image CSVs. It requires exact 191-prompt coverage at 4096 square
and seed label 0, unique rows, finite metric coverage, matching base/extra image
paths, and matching repeated SCA image paths/scores across evaluation groups.
No historical patch-FID/KID summary is used.

`results/baseline_quality_deltas.csv` preserves full-precision means and differences,
plus canonical interval endpoints at their archived precision. `paper/figures/baseline_deltas_provenance.json` records all 12
source hashes, exact ordered prompt IDs, and hashes of the bootstrap index arrays.
The figure reproduces the canonical 2000-draw, seed-0 paired prompt bootstrap,
advancing the RNG through all ten metrics in alphabetical order as in
`scripts/paired_bootstrap.py`. The 16 existing intervals are independently
reproduced, then read from `results/e1_<baseline>_vs_sca.csv` with their signs
reversed. Four full-attention comparisons are newly computed using the same
procedure and independently agree with the existing helper at its CSV precision. Intervals are two-sided percentile 95%, without multiplicity
correction. They estimate prompt-sampling uncertainty for one generation seed,
not seed variability. An initial preview reused the first draw matrix for each metric and had slightly
different endpoints. That preview was replaced; final endpoints exactly reuse
the canonical intervals, so manuscript and figure have no competing estimates.

| Baseline | ImageReward delta [95% CI] | PickScore delta [95% CI] | HPSv2.1 x100 delta [95% CI] | Crop MUSIQ delta [95% CI] |
|---|---|---|---|---|
| Full | 3.4071 [3.2979, 3.5057] | 5.1433 [4.9389, 5.3531] | 22.0930 [21.2802, 22.8961] | 27.9860 [26.2996, 29.5251] |
| AReA | 0.3120 [0.2339, 0.3859] | 0.7345 [0.6190, 0.8528] | 2.5009 [2.1396, 2.9055] | 10.8625 [9.4926, 12.3342] |
| DyPE | 1.3338 [1.1991, 1.4776] | 2.6007 [2.4124, 2.7905] | 9.1819 [8.5030, 9.9540] | 10.8699 [9.0490, 12.7218] |
| SEGA | 0.6773 [0.5616, 0.8026] | 1.4492 [1.3050, 1.5990] | 5.3395 [4.8124, 5.8835] | 10.8473 [9.1661, 12.5026] |
| UltraImage | -0.0468 [-0.1113, 0.0157] | -0.2354 [-0.3324, -0.1423] | -1.0964 [-1.3997, -0.8018] | 19.4166 [18.1628, 20.6986] |

All differences are SCA minus baseline. The figure intentionally preserves
UltraImage's resolved PickScore/HPSv2.1 advantages and unresolved ImageReward
difference. UltraImage is separated by a dashed row boundary and diamond marker
because it uses a two-stage recipe with 30 steps per stage and different guidance,
not matched compute or identical latent trajectories. DyPE/SEGA use FLUX.1-dev;
their preferred Krea checkpoint is not represented by these results.

## Figure validation and integration

- `paper/figures/latency.pdf` and `.svg`: 3.25 x 2.34 inches; same SCA blue,
  AReA orange, full-attention gray as existing figures. Updated caption at
  `paper/figures/snippets/latency.tex`.
- `paper/figures/baseline_deltas.pdf` and `.svg`: 5.5 x 4.15 inches, four paired
  difference panels with numeric point labels, zero reference, and open unresolved
  marker. Caption at `paper/figures/snippets/baseline_deltas.tex`, label
  `fig:baseline_quality`.
- Both style audits are clean. PDFs contain zero raster image objects. Inspected
  PDF rendering at 150 dpi at the intended printed dimensions. Font sizes remain
  6.5 points or larger. All labels and whiskers fit without overlap.
- Regenerate with the existing uv-managed `csd-figs/bin/python` interpreter:
  `paper/figures/src/fig_latency.py` and
  `paper/figures/src/fig_baseline_deltas.py`.

Root can include the quality snippet directly; no `main.tex` edits were made by
this agent. The quality plot deliberately uses no timing coordinate because
clean matched timing is unavailable for DyPE, SEGA, and UltraImage.
