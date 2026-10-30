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

## Swap procedure (one agent per landing batch; main.tex single writer = that agent)
1. `bash paper/figures/src/swap_all.sh` (runs collect_latency_v2.py, gen_main_tables.py, gen_app_tables.py,
   fig_paired_area.py, fig_latency.py, fig_dup.py) and `gen_dupfix_table.py`; read the landed/pending status lines.
2. If a4k195_main_sca_v2 landed: corrected FID/KID bootstrap. The z980 eval has no reference-cluster bootstrap.
   Inspect /scratch/09032/saini_2/sca/taskq/hold_cpu_finalize/a4k195_main_bootstrap.sh (published run); if it is
   CPU only, run an sca_v2 clone directly (nice, background, outputs under eval/a4k195_main_sca_v2/ and the CSVs the
   generators read; NOT through the spool), then rerun step 1. Otherwise state in the Table 1/J captions which FID
   intervals resample references.
3. Hand edits in main.tex preamble (see "Status macros" above): rewrite \hlDupAbstract \hlDupIntro \hlDupResults
   \hlDupLimit with the 195-prompt numbers (\hlDupV of \hlDupN prompts vs \hlDupPub without the schedule, \hlDupArea
   AReA), keeping the 46-prompt numbers only where the text is about the ablation set; set \pendFigPaired /
   \pendFigLatency empty once their PDFs read sca_v2; latency macros once latency_curve_v2 lands.
4. Images: regenerate figures that show SCA images with the sca_v2 outputs of the same prompt ids and seeds:
   zoom4k (fig_zoom4k.py; MUSIQ_c from the sca_v2 per-image CSV), teaser (fig_hero.py; its 2048x6144 image needs
   a4k195_2048x6144_sca_v2), random galleries of sizes that have sca_v2 runs (fig_gallery_large.py; keep the frozen
   random draws), Figure 4 thumbnail if fig_heads_view.py reads an image path, the progressive figure after
   prog195_sca_v2. Then set \pendImgTeaser / \pendImgQual empty. Captions that say "curated" stay curated; if the
   curated prompt now shows a defect in sca_v2, say so in the caption rather than swapping prompts silently.
5. Claims check (list above) against the new tables; fix prose and Key results; keep every claim tied to a number.
6. quick_build.sh, then build_delivery.py (0 overfull, 0 undefined, Section 6 ends p. 10), git add -A paper .agent,
   plain one-line commit, no trailers, git push -q origin main.

## Swap S1 (2026-10-30, 01:37 to ~03:00 CDT): m4k, m24, m26, s1, s2, pg folded in
- Landed and folded: a4k195_main/2048x4096/2048x6144/seed1/seed2_sca_v2, prog195_sca_v2. Still grey: krea195,
  hrgen1k_dev, hrgen1k_krea, latency_curve_v2 (clean time, speed ratio, latency figure/table, \hlLatExp*).
- Bootstrap clones (CPU, nice, not the spool): /scratch/09032/saini_2/sca/eval/<run>/cpu_bootstrap_sca_v2.sh for
  main, 2048x4096, 2048x6144 -> results/<run>/corrected_fidp_ci/ + SHA256SUMS. Effect at 4096^2: wins vs AReA
  11 -> 12 of 15 (patch KID becomes significant), FID -5.6 [-10.6, -0.4], unresolved: IS, KID, patch FID.
  2048x6144: 7 of 15 wins, losses aesthetic + patch FID; 2048x4096: 3 wins, 3 losses (aesthetic, CLIP, CLIP-IQA).
- Duplication (people/animals, 95 prompts): sca_v2 26% (25/95, Wilson [18.5, 36.0]); seeds 1/2: 28%/26%;
  2048x4096 24%, 2048x6144 33%; progressive sca_v2 6% (published 8%). 46-prompt tuning subset gave 15% (4/26).
  Prose now says "halves" everywhere (abstract, bullet 2, 4.2 lead, limitations, conclusion, app. dupfix/failures).
- Progressive sca_v2 vs HRDiT: only CLIP-IQA significantly worse (-0.9 [-1.8, -0.1] x100); HPSv2, MUSIQ, ARNIQA better.
  app_tab_phase2.py prog_deltas caption lead is now data-driven; gen_main_tables.py progressive check accepts losses
  named in main.tex ("Only CLIP-IQA ...").
- Generator edits: v2swap/scav2_runs NOTE shortened; fig_dup.py takes Progressive SCA from results/prog195_sca_v2.
- Page budget: Section 6 ends p. 10 at y=731 (text bottom 732): zero slack. Any added main-text line spills.

## Swap S2 (2026-10-30, 03:25 to ~04:10 CDT): krea195 and latency_curve_v2 folded in
- collect_latency_v2.py looked for logs/task_x_e_lat_<HxW>.log; the tasks ran as x_d96z_lat_<HxW> (taskq/done). It
  now accepts both names. All five sizes pass (warm-up then three timed, no skip, idle GH200, sidecars = log).
  sca_v2 / sca_nb medians (s): 19.8/18.2, 44.3/39.9, 77.1/71.1, 174.7/161.0, 309.7/285.8. The re-timed sca_nb is
  within 0.8% of the first sweep at every size, so sca_v2 is comparable with the first sweep's AReA and full rows.
- Clean ratio vs AReA at 4096^2: 202.4/77.1 = 2.63 (was 2.85); 8192^2: 952.0/309.7 = 3.07 (was 3.31). Exponent
  (lstsq of log median s on log N, 5 sizes, the published protocol, which reproduces 0.997 -> 1.00 for sca_nb):
  sca_v2 0.992 -> 0.99. Preamble: \hlLatExpSca 0.99, \hlLatEightSca 309.7, \hlLatEightRatio 3.07, \pendFigLatency {}.
- \hlTimeOverhead now comes from the matched pair (gen_main_tables.py duplication_macros): 77.10/71.13 - 1 = 8.4%
  -> 8% (8 to 11% across sizes). The old 9% compared 46-image medians from two campaigns (out/main sca_nb vs
  out/dup46 dup_i_shrink). Section 4.3 and Limitations say "timed back to back"; App. D paragraph rewritten (two
  sweeps, per-image values, 63.8%/67.5% reductions, 4.4x full at 6144^2, non-square campaign means 41.1/60.5 s).
- Precision: 4.3 lead "faster than every baseline we timed at every size" (sca_nb, our ablation, is faster);
  conclusion "method we compared"; Table F.1 lead "runs \hlSpeedArea x faster" (0.38 of AReA's time, not a third).
- Krea: CPU clone eval/krea195_sca_v2/cpu_bootstrap_sca_v2.sh (reference sca_v2_krea) -> results/krea195_sca_v2/
  corrected_fidp_ci + SHA256SUMS. sca_v2_krea minus AReA: 7 of 10 per-image wins (IR +0.020 [-0.021, 0.060], CLIP,
  CLIP-IQA unresolved), IS win, FID +1.0 [-3.6, 6.0], KID, patch KID unresolved, patch FID +3.4 [1.6, 7.1] LOSS
  (published operator +8.1; sca_v2 vs sca_krea patch FID -4.7 [-7.7, -2.3]). Table K: "wins eight metrics and loses
  patch FID, in a third of the time" (78.4 vs 239.3 s campaign). Prose: 4.5, Limitations, App. H, App. K ("SCA
  does not" trail on ImageReward).
- baseline_deltas.tex: review 2 had hand-labelled the caption "SCA without the schedule"; the figure plots sca_v2
  (provenance: landed a4k195_main_sca_v2) and the generator restored "SCA beats ...". Keep the generated caption.
- Still grey: hrgen1k_dev, hrgen1k_krea (Table 4 HRGen row, tab:published_compare HRGen row, phase2_general,
  published_hrgen, 4.5 "Gains hold ... on 1000 HRGen-1K prompts"). Krea gallery (fig_phase2.py krea_complete)
  still shows the operator without the schedule, labelled so; regenerate from out/dupfix_krea/sca_v2_krea if wanted.
