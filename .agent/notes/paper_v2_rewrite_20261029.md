# Paper rewrite around sca_v2 (2026-10-29, paper owner; Opus subagent of the SCA lead)

## Operator facts the paper states (verified against sca/attn.py by agent C)
- Opening sigma > 0.89 global only; windows 128/112/96/80 on the four steps after the gate (0-based steps 8-11, the
  paper says "the four steps after the opening"), then n0 = 64. Weight pi_lh(sigma; r), r = max(H,W)/w, log-linear
  between the scale-2 and scale-4 tables (Eq. sched; lambda = clip(log2 r - 1, 0, 1)); the native phase uses the
  nearest calibrated scale in log space (always 4 at evaluated sizes; would differ from interpolation at 3072^2).
- Bridge (Eq. bridge): 8x8 cells x 16 tokens = 1,024, farthest from the mean image key summed over heads, per block
  and step, TRUE positions, in the LOCAL softmax (not the global branch as the lead's brief said). Keys form a
  multiset: a bridge token inside the window enters twice. Calibration tables were fitted without the bridge.
- Prop. 1 now covers only the global operator and window keys at w = n0; bridge keys and wide windows leave the range.
- Progressive sca_v2: same flags; stages re-enter at sigma .81/.61, so only the native window and the bridge run.
- The schedule sizes were chosen by hand on the 46-prompt subset, which is part of the 195 evaluation prompts
  (disclosed in Data and seeds and Limitations).
- Cost (agent C, figures/src/alg_drafts/cost_v2.txt): local keys 17,920/14,080/10,752/7,936/5,632 for w=128..64;
  pairs over 28 steps: full 1.2215e11, published 1.4562e10 (8.39x fewer), sca_v2 1.7749e10 (6.88x). FLOPs/image
  sca_v2: 6.24e15 at 2048x4096 (3.5x vs full), 1.24e16 at 4096^2 (6.9x), 9.34e15 at 2048x6144 (5.2x). Table D.1
  (tab:flops) still shows the published operator only: add an sca_v2 row from cost_v2.txt.

## Number pipeline (one command: paper/figures/src/swap_all.sh)
- gen_main_tables.py (agent A1): Tables 1-4 + headline.tex macros; prints landed/pending per run. TABLE1_MARKS=False.
  New macros: \hlDupV \hlDupPub \hlDupArea \hlDupFullAttn \hlDupN \hlDupFsPub \hlDupFsV \hlDupFsArea \hlDupFsN
  \hlDupGateNinetyThree/EightyNine/EightyFive \hlDupGlobalOnly \hlTimeV \hlTimePub \hlTimeOverhead \hlFsArniqa
  \hlFsMusiqc \hlFsIrAreaPub \hlFsIrAreaV (+Lo/Hi), \hlIrScaAbs \hlFidScaAbs \hlWideSixWins, \pendTab{Main,Panel,Abl,Gen}.
- gen_app_tables.py (agent A2): 11 appendix tables -> snippets/app_tables/*.tex; caption notes \pendApp*; preamble
  needs \input{figures/snippets/app_tables/pending_notes.tex}. main.tex body ranges to replace: run
  `gen_app_tables.py --where` (line numbers moved after A2 reported). Captions of the published-only tables (app_ps2,
  app_abl, app_abl_band(2), ablation_band, phase2_sensitivity, phase2_budget, phase2_distribution, flops) must say
  "without the schedule".
- swap_all.sh: runs collect_latency_v2.py AFTER gen_main_tables.py; reorder so the collector runs first.
- fig_dup.py (agent B): regenerate after a4k195_main_sca_v2 lands (panel b gets sca_v2's 195-prompt bar).
  gen_dupfix_table.py: Table tab:dupfix, rows for dup_i_160/bridge32/ext75 appear when their DONE exists.
- Gaps: z98x evals compute no reference-cluster bootstrap for corrected FID/KID (patch FID/KID lose intervals, FID
  falls back to eval.py intervals). Fix = clone taskq/hold_cpu_finalize/a4k195_main_bootstrap.sh for the sca_v2 run
  and run it on CPU (not through the spool), or say in captions which intervals resample references.
- \def\pend keeps \up/\dn coloured inside grey cells; A2 suggests {\color{pendgrey}\let\up\textbf\let\dn\textit#1}.

## Status macros in main.tex preamble that must be edited BY HAND when runs land
\pendImgTeaser \pendImgQual (teaser, zoom4k images: regenerate fig_hero.py / fig_zoom4k.py from
out/dupfix/sca_v2/4096x4096 and the 2048x6144 run, same prompt ids; then set empty), \pendFigPaired \pendFigLatency
(set empty once paired_area / latency PDFs read sca_v2), \hlDupAbstract \hlDupIntro \hlDupResults \hlDupLimit
(rewrite with the 195-prompt rate \hlDupV of 95 prompts once m4k lands), \hlLatExpSca \hlLatEightSca
\hlLatEightRatio (latency_curve_v2), \hlWorkedPairsRatio (6.9, from cost_v2.txt; fine). Galleries (App. R),
failure figure, progressive figure crops and Figure 4 thumbnail still show the published operator.

## Prose claims to re-check when m4k lands (Section 4.1 and captions)
Table 1 Key result "best on every quality metric shown"; "Every shared-sampler baseline trails SCA on all ten
per-image metrics"; UltraImage "significantly better on HPSv2.1 and PickScore and worse on CLIP and both MUSIQ";
"\hlUnresolvedAreaWord are unresolved"; Krea sentence (kr); progressive paragraph (pg); "Gains hold" (s1, s2, hd);
abstract/intro speed and wins macros; Limitations Krea sentence.

## Page budget
Main text must end on p. 10 (Reproducibility statement starts p. 11). After moving Table 5 (tab:related) to
App. app:related and trimming, the conclusion still ended on p. 11 at y = 444 (about 0.55 page over) at 23:40.
