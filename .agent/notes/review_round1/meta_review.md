# Meta-review, round 1 (revision author, 2026-11-02)

Inputs: reviewer1_method.md, reviewer2_experiments.md, reviewer3_presentation.md, results/, duplication_fix_20261027.md.
Constraints: no GPU, no job submission, no measured number invented or altered, every number from results/*.csv,
main text at most 10 pages. Figure agent owns figures/src/method.tex (Figure 3), Figure D.1 (method_zoom) and the
calibration figures (Appendix B); it lands them separately. Deferred GPU work is specified in
/scratch/09032/saini_2/sca/taskq/hold_review_round/README.md (held, never pending).

## Lead decisions (binding)
1. Framing: per-head calibrated receptive-field mixing, measured once at 1024^2 and transferred. FLUX.1-dev 4096^2:
   equal-or-better than every single-pass method on 12 of 15 metrics at 2.6x AReA's speed. Krea (AReA's checkpoint):
   quality on par (ImageReward unresolved, patch FID lost) at a third of the time. Duplication is a stated weakness:
   26% vs AReA 4% (people/animal prompts, FLUX.1-dev), about 30% vs 2% on Krea. Abstract, intro, bullets, conclusion
   and a three-checkpoint block (FLUX.1-dev, Krea, FLUX.2) say so. No general "better than AReA"; the duplication
   number sits next to every "12 of 15".
2. Duplication mechanism: window width at high noise, not gradualness (Table P.1: linear pi ramp with native windows
   5/26, one 128-window phase 1/26). Intro bullet and Section 2.4 rewritten around window width versus subject scale;
   the four-step shrink is a cheap width schedule (hand-set sizes, calibrated weights); AReA keeps windows wide for 20
   steps at 2.6x our cost; its remaining advantage is an open question; the controlled swap is deferred.
3. Hand-set vs calibrated table in Appendix F; the sigma bins are tied to the 28-step sampler; Table 2 budget rows run
   without the schedule and are marked.
4. Calibrated vs deployed operator gap (calibration without the bridge, emulated key counts) stated as a limitation,
   with the deferred experiment.
5. "Scale covariance" toned down to what is shown (pi approximately invariant at fixed ratio across ratios 2 and 4 and
   two resolutions); Table J.1's AReA entry justified from the code.
6. Operator consistency: every table and figure says which operator; previous-operator tables (6144^2, 8192^2,
   FLUX.2, sensitivity, L.5, Appendix M) labelled "SCA without the schedule" in row names and captions, sentences too.
   Fill the two pending Krea cells from results/krea195_sca_v2 and results/hrgen1k_krea_sca_v2 (both CSVs exist:
   sca_v2_krea rows present in base_summary.csv and extra_summary.csv of each).
7. Detector validation stated exactly (labelled images are SCA images only; per-method validation pending); use
   results/dup_audit_20261027/validation_by_method.csv if it appears.
8. Timing: clean and campaign numbers never share one bold; sweep difference stated (AReA and full attention from
   sweep 1, SCA from sweep 2).
9. Tables: bold best per group, grey context rows for multi-stage methods, one significance convention for Tables 1
   to 4, intervals in Table 4, Dup. header says what it prints, patch FID not bolded against other papers, KID beside
   FID with "FID at n=195 is bias-dominated", "12 of 15" with the family count.
10. Writing and figures per reviewer 3 (listed under R3 below).

## Dispositions
Legend: FIX = fix now (this pass); GPU = needs GPU, deferred (task id in hold_review_round); CPU-D = CPU-only but
deferred (task id, reason); FIG = figure agent; REJ = rejected (reason).

### Reviewer 1 (method)
| # | Item | Disposition |
|---|---|---|
| 1 | Bullet 2 and Section 2.4 overreach; width not gradualness | FIX (decision 2). Bullet and Section 2.4 rewritten around window width at high noise, Table P.1 rows cited |
| 2 | Why AReA does not duplicate; controlled swap | FIX text (open question, AReA wide for 20 steps at 2.6x cost); GPU RR01 (SCA with AReA's 20-step shrink and bridge ratio 0.5), RR02 (AReA with SCA's 4-step shrink) |
| 3 | "Both parts needed": no schedule-alone row, 46-prompt counts within one prompt | FIX text: claim limited to what Table 3 shows, McNemar p from results/dup_audit_20261027/review_round1_cpu/pairs.csv (shrink+bridge 4/26 vs bridge 5/26, p = 1.0); GPU RR03 (schedule alone and bridge alone on 195 prompts, McNemar) |
| 4 | Schedule defined by sigma bins that hold one step at 28 steps | FIX text (decision 3) in Section 2.3 and Table F.1; Table 2 budget rows marked "without the schedule"; GPU RR04 (42 and 56 steps with a step-indexed schedule) |
| 5 | Calibrated operator is not the deployed one | FIX limitation (decision 4); GPU RR05 (deployed-operator output error vs full attention at 4096^2 on 8 prompts; recalibration with the bridge) |
| 6 | "Scale covariance" overstated | FIX (decision 5): defined in one sentence with its tested domain (ratios 2 and 4, 1024^2 and 1280^2) |
| 7 | Table J.1 AReA O(N^2)* | FIX: entry kept quadratic and justified from the code (pooled steps at f = 2 attend to N/4 keys; the first window rung is the largest window whose cost stays below full attention), measured exponent 1.12 vs 0.99 in the caption |
| 8 | 12 of 15 bullet omits duplication; Krea beside dev | FIX (decision 1) |
| 9 | Lemma 1 is motivation; overlapping sets double-count | FIX: stated as motivation; double count stated (bridge and window keys not deduplicated) |
| 10 | Prop 1 scope | FIX: statement scoped to window keys in the native phase |
| 11 | Prop 3 is a scalar quadratic | FIX: scope (bin-constant tables) stated in the statement; kept in Appendix A only |
| 12 | Fig 5d "r = 0.77" | FIX: rho in caption and figure label |
| 13 | Hand-set vs calibrated column | FIX (decision 3): Table F.2; GPU RR06 (sensitivity to bin count and tile t) |
| 14 | "thus" for HRDiT's lead | FIX: "largely" |
| 15 | Curated teaser | FIX: Figure 1 drops the top row duplicated in Figure T.1; caption says curated and points to the uncurated gallery (Appendix T) |
| Q1 | AReA's 4% | Open question in text; GPU RR01, RR02 |
| Q2 | pi with the bridge, or emulated r = 4 at 2048^2 | GPU RR05 |
| Q3 | Tied to 28 steps and mu = 1.15 | FIX text; GPU RR04 |
| Q4 | Two-step schedule | GPU RR07 (two wide steps 128, 96 with matched weights, 46 prompts) |
| Q5 | Per-query weight | REJ as a paper change: a per-query softmax-mass mix (LSE mixing) was piloted and rejected (duplication_fix_20261027.md, dup_lse1: 6/9 duplicated, worse PickScore, 15 prompts); answered in this note only |

### Reviewer 2 (experiments)
| # | Item | Disposition |
|---|---|---|
| Cov a | Non-square sizes carry AReA alone | GPU RR08 (all single-pass baselines at 2048x4096 and 2048x6144, 195 prompts); FIX text: Section 4.3 quotes the non-square duplication rates beside AReA's |
| Cov b | 6144^2, 8192^2, FLUX.2, L.5 on the retired operator | FIX labels (decision 6); GPU RR09 (sca_v2 at 6144^2, 100 prompts), RR10 (8192^2, 12), RR11 (FLUX.2, 50) |
| Cov c | Clean latency for 3 of 8 methods | GPU RR12 (one back-to-back sweep of all eight methods at 4096^2) |
| Fill | Two pending Krea cells | FIX (CSVs present) |
| P1 | Operator mismatch | FIX labels in row names and captions; Appendix M text corrected (Prog. SCA in Table 2 is sca_v2) |
| P2 | Detector validated on SCA images only | FIX statement (decision 7) and Wilson intervals in Table 1's Dup. column from rates_a4k195_people.csv; CPU-D RR13 (label 20 images per method; needs a human or VLM labelling pass) |
| P3 | Timing sweeps differ | FIX presentation (decision 8); GPU RR12 |
| P4 | Krea coverage for DyPE/SEGA/SigMa/HRDiT/UltraImage | GPU RR14 (DyPE and SEGA on the 95 missing Krea prompts; SigMa on Krea) |
| P5 | FID at n=195 bias | FIX: KID beside FID, bias sentence in Section 4 and captions |
| P6 | Patch FID own rule | FIX: dagger, not bolded against other papers' numbers |
| P7 | Interval recipes differ | FIX: each caption names its recipe; CPU-D RR15 (one recipe re-derived from per-image CSVs: changes many intervals, needs its own check pass) |
| P8 | 12 of 15 counts correlated metrics | FIX: wins by family (preference, no-reference, distribution) next to 12 of 15; CPU-D RR16 (Holm-adjusted count from per-image bootstrap) |
| P9 | 149 prompts outside the 46 subset | CPU-D RR17 (one line from per-image CSVs) |
| P10 | fp32 remark | FIX: moved into Table N.1 caption |
| S3 Table 1 | Bold across groups; Time mixes clean and campaign; Dup. header | FIX (decisions 8, 9) |
| S3 Table 2 vs 1 | Significance conventions differ | FIX: one convention for Tables 1 to 4 |
| S3 Table 4 | No intervals, no Dup., MUSIQ_c scale | FIX: intervals from the paired CSVs, Dup. from the Krea and seed rates, scales stated |
| S3 non-square | Summary per size with Dup. | FIX as prose in Section 4.3 (page budget) |
| S3 H.1 | Ranks published and our-run rows together | FIX: rank within block |
| S3 Fig 8 | Per-size campaign means table | REJ for this round: campaign means are not comparable across sweeps (P3); RR12 replaces them |
| S4 | Most damaging question | FIX: three-checkpoint block in Section 4.5; GPU RR14 completes Krea |

### Reviewer 3 (presentation)
| # | Item | Disposition |
|---|---|---|
| Fig 3 | Redraw method figure | FIG (figure agent) |
| Fig 1 | Top row duplicates Figure T.1; insets amber | FIX: top row dropped, caption to lead plus two sentences; inset recolouring with the strip regeneration |
| Fig 2 | Bold (a), (b) | FIX |
| Fig 4 | (a) cluttered; red frames vs crop red | FIX: (a) shows the gate only, schedule separated; (c) frames thinned |
| Fig 5 | (b) thumbnails small; subscripts; sigma_g | FIX: (b) enlarged, text at 6 pt or more |
| Fig 6 | Full-attention column | FIX: replaced by HRDiT (or dropped if the crops are missing), window 1 pt |
| Fig 7 | n = 195 | FIX |
| D.2, B.3/B.4 layout | Half-blank pages 17 and 21 | FIX float placement (B figures themselves: FIG) |
| Tables 1, 2 headers | Bare arrows over split names | FIX: one header line with arrows |
| Table 3 | 7 pt, n/a, Err. column | FIX: Err. moves to Table K.5, n/a removed |
| Table 4 | Intervals | FIX (decision 9) |
| M.1 | One grey diamond for two methods | FIX marker |
| Writing | Ten sentences over 35 words, captions, comparators, ARNIQA x100, abstract numbers, intro order | FIX (decision 10) |
| Algorithms | Type about 6 pt | FIX: at least 6 pt |
| Page 7 | Stretched paragraphs | FIX layout |

## Deferred GPU and CPU tasks (specs in taskq/hold_review_round/README.md)
RR01 SCA with AReA's 20-step shrink and bridge ratio; RR02 AReA with SCA's 4-step shrink; RR03 schedule alone and
bridge alone on 195; RR04 schedule at 42 and 56 steps; RR05 deployed-operator error and recalibration with the
bridge; RR06 bin count and tile sensitivity; RR07 two-step schedule; RR08 non-square baselines; RR09 to RR11 sca_v2
at 6144^2, 8192^2, FLUX.2; RR12 one latency sweep of all methods; RR13 per-method detector validation; RR14 Krea
baselines; RR15 one interval recipe; RR16 Holm count; RR17 the 149 prompts outside the subset.

## Status after the first revision pass (2026-11-02 03:56 CST)
Applied: decisions 1 to 7 in full; decisions 8 and 9 for Table 1 (per-group ranking, grey context rows, unranked campaign
time, sweep note, Dup. percent, FID_p dagger, KID and FID bias in the caption); decision 10 in part (abstract numbers,
intro order, Figure 1 top row dropped, Figure 7 n, ARNIQA x100, page 7 spacing). R1 items 1, 3 (text), 4 (text), 5, 6, 7,
8, 9, 10, 11, 13, 14, 15 done; R1.12 done in the text (figure label pending). R2 P2 now uses validation_by_method.csv.
Still FIX and not yet applied: Tables 2 to 4 significance convention, Table 4 intervals and Dup., Table H.1 ranking within
block, operator labels in appendix row names (L.1, N.1, L.3 to L.5), long sentences, figure captions (4, 5, A.1, D.1, B.1),
Figures 4a, 5b, 6, Table 3 Err. column, Figure M.1 markers, algorithm type size, pages 17 and 21, Table N.1 fp32 note.
Not available in results/: AReA's duplication rate on Krea (the "about 2%" in the brief), so the paper quotes only SCA's
29% (28 of 95) on Krea.
