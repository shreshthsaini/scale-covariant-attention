# Independent source audit of the paper revamp

17 October 2026. Read-only audit of `paper/main.tex`; no manuscript edits or
experiments. Final inspected main-source SHA256:
`ad09a1e1de0c56eab315960d7e285c99dc3d97566961b070c8530701eee42532`.
Root was editing concurrently. The two material consistency findings below
were communicated and are fixed in this inspected revision.

## Verified

- All 30 numeric cells in the new six-method, five-metric main table match their
  archived source values at the displayed precision. ImageReward, HPSv2.1,
  CLIP, and crop MUSIQ means were independently recomputed from the 191 per-image
  rows for each method. Whole-image FID points match the archived summaries;
  these points are unchanged by reference-crop correction. The SCA boldface
  correctly denotes the best mean within the shared-protocol block only.
- Abstract ImageReward 0.92 to 1.23, HPSv2.1 27.2 to 29.7, and 4K latency
  202 to 71 seconds are correctly rounded. AReA/SCA median ratios are
  2.9851998866 at 6K and 3.3096513767 at 8K. Equivalent time reductions are
  66.5014057% and 69.7853373%. The abstract's 2.99x and 3.31x are supported.
  The timing table correctly leaves full attention at 6K pending and 8K not run.
- The broad scalar claims are supported: all ten canonical per-image intervals
  favor SCA against AReA, DyPE, and SEGA. All ten archived full-attention versus
  AReA scalar intervals favor AReA. These statements do not extend to every
  distributional metric or to the separate UltraImage recipe.
- The new paired quality plot reuses the canonical 2000-draw scalar intervals,
  preserving UltraImage's PickScore/HPSv2.1 advantages, unresolved ImageReward
  difference, and SCA's crop-MUSIQ advantage. Its caption explicitly distinguishes
  prompt uncertainty from seed variability and states the separate UltraImage
  budget and DyPE/SEGA checkpoint caveat.
- Teaser provenance uses the same evaluation prompt, seed label 0, and derived
  seed 2256563941301075758 for all six sources. The prompt belongs to the 191-row
  held-out evaluation set. Full/DyPE/SEGA/AReA/SCA sidecar configurations record
  28 steps and guidance 4.5; UltraImage records 60 total steps and guidance 7.
  The two-stage guidance schedule is described elsewhere in the paper.
  All 12 displayed JPEG hashes match the provenance manifest. Crop coordinates
  are uniformly (1600,1280,2368,2048), producing 768-square crops, and match the
  earlier portrait crop in `fig_qual.py`. Curated selection is stated explicitly.
- Expanded actual snippet branches and inputs: 84 active labels, no duplicate
  labels, no undefined concrete references, and no missing concrete graphics.
  Macro parameters such as `#1` and `##1` were excluded from this literal-file
  check. `fig:baseline_quality`, `fig:teaser`, `tab:main`, and `tab:all_metrics`
  each resolve to the intended new figure/table. This does not replace compilation.

## Findings fixed during this audit

1. The prior Hardware and Compute paragraphs claimed GH200 96GB, contradicting
   archived latency logs identifying GH200 120GB. Root removed the blanket
   historical-memory claim and now explicitly scopes 120GB to verified timing
   devices. This is supported by the logs without guessing historical hardware.
2. The initial main-text AReA intervals used historical 1000-draw estimates,
   while the new plot used canonical 2000-draw intervals. Root aligned the
   displayed scalar claims to the canonical values: ImageReward [0.23,0.39],
   HPSv2.1 x100 [2.14,2.91], PickScore [0.62,0.85], CLIP [1.10,1.81], and
   crop MUSIQ [9.5,12.3]. The revised paragraph states 2000 scalar draws and
   separately identifies the 1000-draw reference-image-cluster FID interval.

## Minor caption cleanup and remaining check

- The unchanged older hero caption says "All images are SCA outputs" before
  describing its full-attention and AReA comparison tiles. For literal accuracy,
  change this to "Unless labeled otherwise, images are SCA outputs". The new
  six-method teaser caption has no corresponding attribution problem.
- The abstract's "preserves more crop detail" is supported by crop MUSIQ and
  selected qualitative examples, not paired ground-truth detail reconstruction.
  "Scores higher on crop MUSIQ" would be the narrower quantitative wording;
  current results and limitations already make the mixed system-level result clear.
- Final PDF layout, caption wrapping, float placement, and compiled references
  remain for root's build. The old PDF was not used to assess this source revision.
  The two new evidence plots were separately inspected at their printed sizes.

No unresolved numeric error was found in the revised headline claims or main
table. The inherited wide appendix table was not subjected to a redundant full
audit; its unchanged values retain the earlier evidence checks.
