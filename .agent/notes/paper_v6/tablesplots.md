# Paper v6: tables and plots (role `tablesplots`), 2026-11-10

Status: IN PROGRESS. This file is rewritten as work lands; the sections marked FINAL are settled.

## 1. Holm result (FINAL; text owner please read now)
Module: paper/figures/src/gen_holm.py. It re-runs scripts/paired_bootstrap.py on the archived per-image files
(2000 prompt draws, seed 0, one generator shared by the metrics in sorted order) and reproduces all 16 archived paired
CSVs digit for digit (the run fails otherwise). Two-sided bootstrap p-values from the same draws, Holm step-down over
the ten per-image metrics of each pair at 5%; a paired sign-flip test (2000 sign vectors, seed 0) as a check.

| pair (n) | unadjusted better/worse | Holm, bootstrap | Holm, sign-flip |
|---|---|---|---|
| progressive SCA - AReA (195) | 10/0 | 10/0 | 10/0 |
| progressive SCA - HRDiT (195) | 3/1 (HPSv2, MUSIQ, ARNIQA / CLIP-IQA) | 2/0 (HPSv2, ARNIQA) | 2/0 |
| single-pass SCA - AReA (195) | 10/0 | 10/0 | 10/0 |
| progressive - UltraImage / DyPE / SEGA / full (195) | 9/0, 10/0, 10/0, 10/0 | same | same |
| progressive - single-pass (195) | 7/2 | 7/2 | 6/2 (ImageReward drops) |
| progressive - AReA, seeds 1 and 2 (195), HRGen-1K (1000) | 10/0 each | 10/0 each | 10/0 each |
| single-pass - AReA, seeds 1, 2 (195), HRGen-1K (1000) | 10/0 each | 10/0 each | 10/0 each |
| single-pass - AReA, FLUX.1-Krea (195) | 7/0 | 7/0 | 7/0 |
| w/o schedule - full attention, FLUX.2-dev (50) | 9/0 | 9/0 | 9/0 |

What must weaken in the prose:
- v5.1 Section 4.1, HRDiT paragraph: "Progressive SCA is better on HPSv2, ARNIQA, and MUSIQ and worse on CLIP-IQA".
  After Holm only HPSv2 and ARNIQA remain (MUSIQ p = 0.030, Holm 0.24; CLIP-IQA p = 0.035, Holm 0.24). Say: "After
  Holm correction over the ten metrics it remains better on HPSv2 and ARNIQA and is worse on none; MUSIQ (better) and
  CLIP-IQA (worse) do not survive the correction."
- Progressive over single-pass ImageReward (+0.07 [0.01, 0.13]): unadjusted p = 0.018, Holm 0.036 (bootstrap) but
  0.059 (sign-flip). Keep the interval and avoid calling it a robust gain.
- Everything else survives unchanged: "all ten per-image metrics" against AReA at three seeds and on HRGen-1K, both
  modes; UltraImage nine; FLUX.1-Krea seven; FLUX.2-dev nine.
Macros (in headline.tex, list below once written).

## 2. Planned snippet changes (being implemented)
- main_tables/tab_main_compact.tex (Table 1): full text width (tabular* to \linewidth), adds a speed-ratio column
  (time over progressive SCA's, same kind) and a "record" column (of the ten per-image metrics, how many progressive
  SCA is significantly better / worse than the row on, Holm-adjusted); campaign-only times printed with a mark
  instead of "--"; best bold and second underlined per group.
- main_tables/tab_ablation_compact_prog.tex (Table 2): becomes FULL WIDTH (it no longer fits a 0.48 minipage); each
  group opens with a shaded reference row giving that reference's absolute values (v5.1 showed single-pass SCA's
  values above deltas taken against SCA without the schedule).
- main_tables/tab_generality.tex (Table 3): fits 0.48 text width at 8 pt, one line per setting, no interval rows
  (intervals move to an appendix table), plus n and a ten-metric record column. Suggested placement: Table 3 and
  Figure 7 (scaling) side by side in Section 4.3.
