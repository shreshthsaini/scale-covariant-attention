# Paper v6: main text (role `text`), 2026-11-10

Status: DONE, including the lead-review pass of 2026-11-10 ~11:40 (Section 8) and the "more visual results" round
(~13:20 to 14:10, Section 9). Owner of paper/main.tex only (no git command run, no other file edited).
Last verified builds: /scratch/09032/saini_2/sca/paper_builds/v6_text_46/ (main.tex, 67 pp) and v6_text_s46/
(main_short.tex, 31 pp); main.pdf, logs, aux, p01..p10.png at 110 dpi.
Tools (session scratchpad .../scratchpad/text_v6/): qb.sh (build into v6_text_<n>, page report, falls back to a
mirror with the last appendix that compiled when the live appendix is mid-edit), logcheck.py (overfull boxes by page,
undefined refs, placeholders), pages.py (headings, floats, fill per page), textcheck.py (forbidden words on the PDF),
srclen.py (sentence lengths on the LaTeX source).

## 0. Requests to other owners and decisions for the lead
- lead, decision 1: the ICLR 2027 template marks an "AI use statement" as REQUIRED (outside the page limit). v5.1 and v6
  have none. It needs the authors' own disclosure, so I did not write it.
- lead, decision 2: figure numbering in Section 4 differs from the brief's table. Gains (gains.pdf, fig:paired_area)
  is Figure 5 and the qualitative grid (qual_main.pdf, fig:qual_prog) is Figure 6. With the qualitative figure first,
  the gains and generality float landed two pages after 4.1 cited it. Stems and labels are unchanged.
- lead, decision 3: Sections 4.2 and 4.3 swapped. 4.2 is now "Generality, other sizes, and latency" and 4.3 "The
  measured reach carries the gain"; labels sec:generality, sec:sizes, sec:ablations unchanged. This keeps the
  generality table and Figure 7 next to their text and ends the experiments on why the method works.
- lead, layout item (RESOLVED, Section 8): every float is now on the page of its first citation or one page before it,
  with Table 3 beside Section 4.3 and Figure 7 wrapped in Section 4.2. Sections 5 and 6 below describe the earlier layout.
- appendix: labels of the main figures. Figure 1 carries fig:prog_hero AND fig:speed (the hero holds the speed
  panel); Figure 4 = fig:emulation (new); Figure 5 = fig:paired_area (gains.pdf); Figure 6 = fig:qual_prog; Figure 7 =
  fig:scaling (new; the time-against-size plot that v5.1 called fig:speed b). Point latency text at fig:scaling.
  Please also state in the appendix what the main text no longer states (Section 4 below).
- appendix: the main-text pseudocode names the ratio r; the full code in Appendix C.2 names it s_c. Consider "s_c
  (the ratio r of the main text)" in its first comment.
- tablesplots: \progAblResult is no longer printed. Its first sentence has 48 words, and "window only loses on eight"
  is the unadjusted count while Table 3 prints the Holm record 6/1. The main text writes that sentence from
  \holmProgAblSharedWorseWord. If wanted back, make it at most two sentences of at most 30 words with Holm counts.
- tablesplots (optional): \hlProgUltraMusiqc{,Lo,Hi} (visuals' request; results/prog195_vs_baselines/
  sca_v2_prog_vs_ultraimage.csv, musiq_crop1024). The Figure 6 caption prints "and UltraImage's by ..." automatically
  once the macro exists (\ifdefined), at a cost of about one caption line.
- visuals: Figure 1 follows your caption ("showcase prompt" became "purpose-written prompt": readers do not know our
  set name). Figure 6 changed three times during this round; the caption matches the file of 11:33 (content agreement
  among the top tenth of windows by detail, per its provenance JSON, newer than visuals.md). In that draw AReA's crop
  scores best in two of three rows, so the caption makes no per-row ranking claim: its bold lead states the times
  (macros), and the body gives the rule, the per-crop MUSIQ, and the 195-prompt MUSIQ_c intervals (+3.7 [2.0, 5.4] over
  AReA; indistinguishable from HRDiT, +0.6 [-0.7, 1.8]). Please update visuals.md to the 11:33 design, and tell the
  text owner (or the lead) if the figure changes again. The hero at 3.29 in is accepted; the Introduction still
  starts on page 1.
- methodfigs: Figures 2, 3 (round-2 candidate C), 4 use your final captions, lightly cut; Figure 2's bold lead is
  "Each head keeps its reach across denoising steps." (states the finding without the "not X" contrast).

## 1. Title options (adopted: 1)
1. Reach Is a Property of the Head: Scale-Covariant Attention for High-Resolution Diffusion Transformers (lead's
   default; states the finding, names the method; the abstract defines "reach" in its second sentence).
2. Measure Once, Generate at Any Size: Per-Head Reach for Training-Free High-Resolution Diffusion Transformers
   (claims the use but hides the scientific finding).
3. Heads, Not Steps, Set the Reach: Scale-Covariant Attention for 4K Diffusion Transformers (sharper, but a binary
   contrast, and "4K" undersells the 8192^2 results).

## 2. What changed, section by section
- Preamble: every v5.1 switch and input kept (\resprogabl on, \resra off, \resfluxtwo/\resqwen/\resideogram off,
  \ifshowcase off, \snippet, headline.tex, speed_numbers.tex, attn_focus_numbers.tex, pending_notes.tex). New:
  \ifvsix{stem}{v6}{v5.1} and \vsixinc{stem}: each main figure uses paper/figures/v6/<stem>.pdf when it exists (at
  natural size, scaled down only if wider than the column), else its v5.1 file. Captions that depend on the design
  switch with it. \holmref points to Table tab:holm when that label exists. A listing style scaalg (7.5 pt lmtt) is
  added for the main-text algorithms; scacode is kept for the appendix. \reqnum + \ifscafinal stay as the placeholder
  mechanism, but no placeholder remains: all 47 requested macros are generated (tablesplots). The hand-typed v5.1
  defaults (hlLatExpSca, hlLatEightSca, hlLatEightRatio, hlWorkedLocalKeys, hlWorkedPairsRatio) were removed from
  main.tex because headline.tex now defines them.
- Title: a claim (Section 1 above).
- Abstract (162 words, 9 sentences): context; the gap (every method must set reach at an unseen size; current methods
  share one schedule or search per size); "We show" (97.5% head, 2.5% step); "We introduce"; the closed-form fit at
  1024^2 "where full attention still works"; headline numbers (49.2 s, 4.1x, all ten metrics at three seeds); HRDiT
  "statistically indistinguishable ... in 46% of HRDiT's time"; URL. "Matches" is used nowhere.
- Introduction: P1 ends on the question; P2 states the observation as a finding with the first priority claim (Figure
  2 at the top of page 2); P3 "We introduce" with the second priority claim; P4 numbers with an interval; three "We"
  bullets in one unbreakable block on page 2.
- Section 2: intuition sentence, then the exact softmax split (Lemma 1) inline, the measurement, the variance split.
- Section 3: 3.1 opens with the intuition and the mix O = pi O^loc + (1 - pi) O^glob, then Setting (Eq. 1), Near view
  (Eqs. 2, 3), Far view (Eq. 4). 3.2 reach emulation (Eq. 5, Proposition 1, "One table serves every size"). 3.3
  progressive SCA and single-pass mode (Eqs. 6, 7; Eq. 6 is now one display line). Every label, equation and
  proposition of v5.1 is kept, and every display has a "where" line.
- Algorithms 1 and 2: 13-line pseudocode at 7.5 pt (was 5 pt code), side by side at the top of page 5, Input/Output
  lines, comments that point to Eqs. 1 to 7, Prop. 1, Appendix B (bins), C.2 (full code), D (kernel).
- Section 4: run-ins Data, Models and baselines, Metrics and timing ("beats" = the 95% interval excludes zero; Holm
  sentence). 4.1: four finding paragraphs whose bold leads are claims (AReA; HRDiT, weakened per Holm; single pass;
  patch-fidelity trade-off). 4.2: generality, time, aspect ratio. 4.3: output error, images (Holm counts), views
  and opening.
- Floats: Figure 1 hero (v6) under the abstract; Figure 2 (v6) top of page 2; Figure 3 (v6, full width) top of page 3;
  Figure 4 (v6, new) top of page 4; Algorithms top of page 5; Table 1 (full width) top of page 6; Figure 5 gains (0.48)
  beside Table 2 generality (0.48) in one float, top of page 7; Figure 6 qualitative (full width) top of page 8;
  Figure 7 with a side caption, foot of page 8; Table 3 ablations (full width) top of page 9. No wrapfigure: the gains
  wrap broke across pages twice (caption below the margin, 4.2 heading squeezed), so it was replaced.
- Related work tightened. Limitations stay concrete: patch FID, coarser weights' lower patch FID in the progressive
  sampler, the verbatim single-pass sentence, hand-set parts and where they were chosen, HRDiT-sampler dependence, the
  bridge left out of emulation, automatic metrics. v5.1's "intervals ignore multiple comparisons" is gone because Holm
  results now exist and Section 4 states them. Conclusion: the spine sentence and three headline numbers.
- Reproducibility statement (page 10): maps protocol, metrics, intervals (app:intervals), timing (app:timing),
  emulation, prompts, algorithms, progressive protocol, hyperparameters, kernel, baselines, compute, and Table
  tab:sources to their sections.

## 3. Claims and evidence (abstract, introduction, conclusion)
Every number prints from a generated macro. Times are point estimates (clean batch-1 medians of three prompts on one
GH200) and ratios divide clean by clean. Holm (tablesplots, gen_holm.py, 2000 draws, seed 0) leaves every "all ten"
claim below at 10/0.
| claim (where) | macro / source cell | value | interval or note |
|---|---|---|---|
| head explains 97.5% of reach variance, step 2.5% (abstract; intro P2; Sec 2) | \hlVarHead, \hlVarStep; Table B.1 tab:variance_split, FLUX.1-dev 1024^2, within 16 tokens, after the gate | 97.5, 2.5 | descriptive decomposition over 32 prompts, no interval |
| shared curve explains 0.7% (intro P2; Sec 2) | \hlVarAvg; Table B.1 "Avg." prints 0.65 | 0.7 | one-decimal rounding of 0.65; Figure 2c prints 0.7 |
| 97.5 to 99.1% in three FLUX models (bullet 1) | \hlVarHead, \hlVarHeadFluxTwo (Krea \hlVarHeadKrea 97.6) | | descriptive |
| 49.2 s per 4096^2 image (abstract; intro P4) | \hlSecProg; results/prog_latency/latency_summary.csv | 49.2 | point estimate |
| 4.1x faster than AReA (abstract; intro P4; bullet 3; conclusion) | \hlSpeedProgArea = \hlSecAreaThree / \hlSecProg = 202.4 / 49.2 | 4.1 | clean over clean |
| 2.2x faster than HRDiT (intro P4) | \hlSpeedProgHrdit = 107.0 / 49.2 | 2.2 | clean over clean |
| beats AReA on all ten per-image metrics at three seeds (abstract; intro P4; bullet 3; conclusion) | \hlProgWinsArea, \hlProgSeedOneWinsArea, \hlProgSeedTwoWinsArea = 10; \holmProgArea{,SeedOne,SeedTwo}Better = 10; Table 2 rows Seed 0-2 | 10/0 each | e.g. IR +0.40 [0.32, 0.49], +0.37 [0.29, 0.46], +0.35 [0.27, 0.45] |
| and on HRGen-1K (intro P4) | \hlProgHrgenWinsArea = 10, Holm 10/0 | IR +0.90 [0.84, 0.96], n = 1000 | |
| ImageReward 0.92 to 1.32 (intro P4) | \hlAreaIrAbs, \hlProgIrAbs; Table 1 IR | +0.40 [0.32, 0.49] | \hlProgAreaIr/Lo/Hi, n = 195, seed 0 |
| statistically indistinguishable from HRDiT on three preference scores (abstract; intro P4; conclusion) | \hlProgHrditIr, \hlProgHrditHps, \hlProgHrditPick with Lo/Hi | -0.02 [-0.09, 0.03]; +0.10 [-0.14, 0.35]; -0.05 [-0.14, 0.04] | all include zero; "matches" not used |
| in 46% of HRDiT's time (abstract; intro P4; conclusion) | \hlProgHrditPct = 49.2 / 107.0 | 46 | clean over clean |
| table repeats on disjoint prompt halves, correlation 0.989 (intro P3) | \hlEmuSplitHalfTwo (ratio 2; \hlEmuSplitHalfFour 0.986) | 0.989 | Pearson over all cells; Figure 4c |
| single pass 2.63x faster than AReA (intro P4) | \hlSpeedArea = 202.4 / 77.1 | 2.63 | clean over clean; 2.1x to 3.1x across sizes (\hlSpeedScaArea{Lo,Hi}, Figure 7) |
| single pass keeps the most crop-level detail (intro P4) | Table 1, M_c column: 60.1 is the highest | point-estimate ranking | against progressive SCA -6.0 [-7.3, -4.6] (\hlProgScaMusiqc) |
| progressive schedule costs patch fidelity, where HRDiT and single pass lead (intro P4) | \hlProgHrditFidP, \hlProgScaFidP | +4.6 [0.6, 7.7]; +9.2 [5.1, 13.2] | set-level bootstrap with reference clusters |
| per-head weights beat per-layer and shared ones (bullet 3) | \hlEmuErr{Head,Layer,Shared}Four; Holm records Table 3 | 0.200 vs 0.223, 0.230; 9/0, 5/0 single pass; 10/0, 10/0 progressive | output error is a point estimate over 32 prompts |
| 16x tokens, about 256x cost (intro P1) | arithmetic (4096^2 / 1024^2 = 16; 16^2 = 256) | | |
Holm weakening applied in 4.1: against HRDiT, progressive SCA "remains better on HPSv2 (+0.38 [0.24, 0.51]) and ARNIQA
(+2.9 [2.3, 3.4]) and is worse on none; \holmProgHrditDropped (MUSIQ and CLIP-IQA) differ only before the
correction". The progressive-over-single-pass ImageReward gain (+0.07 [0.01, 0.13]) is stated with its interval and
not called robust.

## 4. Moved out of the main text (appendix: please make sure the appendix states these)
- The window start a(tau) = clip(t tau + floor(t/2) - floor(w/2), 0, H - w) per axis and the tile index
  tau = floor(p_i / t). The main text says "the w x w block centred on its tile and clamped to the grid (Appendix C)".
  Proposition 2's proof defines a' itself, so nothing breaks.
- The latent x_sigma = (1 - sigma) x_0 + sigma epsilon (the main text introduces sigma as "noise level sigma").
- Emulation details: "at every block and step both emulated views see the same queries, keys, and values"; "with r^2
  times fewer keys per view" (Limitations keeps "16 times fewer keys per view than the target"); the pooling of
  relative errors "over blocks, heads, and steps, on 32 prompts at 1024^2 and 16 at 1280^2" (the main text keeps the
  formula sum J / sum ||O_full||^2).
- Intervals: "1000 or 2000 draws; set-level intervals also resample the 168 reference-image clusters". v5.1 said
  "2000 draws" for every interval, which the appendix contradicts (original summaries use 1000); the main text now
  cites app:intervals instead of stating a draw count.
- Protocol: seeds "hash the prompt id and a seed label"; "campaign means on shared nodes are never ranked" (Table 1
  marks campaign means with a double dagger and does not rank them); HRDiT's stage details (decode, sharpen,
  re-encode; guidance toward the low frequencies of the previous stage); AReA's sampler settings beyond "28 Euler steps
  at guidance 4.5" (shift mu = 1.15, bf16, tiled VAE decoding); progressive guidance 3.5 / 4.5 / 6.0 per stage
  (Table tab:hparams).
- 4.3 of v5.1, "Without the schedule it wins 14 of 15 at 6144^2" and the 12-prompt 8192^2 sentence: now only
  Tables G.3 and Appendix G.4 (macros \hlSixKWins and \hlEightKFavWord stay generated and unused in main).
- v5.1's "Most of the head's share lies between heads of the same block" is dropped: no table reports that split.

## 5. Layout and checks (build v6_text_27)
- 61 pages. Main text pages 1 to 9; page 9 is 95% full (about three lines spare). The reproducibility statement starts
  page 10, then references and the appendix.
- Page 1: title, abstract, Figure 1, Introduction heading and two lines. Page 2: Figure 2, Introduction, the three
  bullets, Section 2 heading. Contribution bullets are on one page; the method figure is full width.
- Float placement against first citation (placed/cited): Alg 1 p5/p4, Alg 2 p5/p5, Fig 1 p1/p2, Fig 2 p2/p2, Fig 3 p3/p2,
  Fig 4 p4/p5 (one page before), Table 1 p6/p6, Fig 5 p7/p6, Table 2 p7/p6, Fig 6 p8/p7, Fig 7 p8/p7, Table 3 p9/p7
  (the one miss, Section 0).
- Overfull boxes: 0 on pages 1 to 10 and 0 in the appendix. Undefined references and citations: none. Placeholders:
  none.
- Switch tests (mirror copies, live tree untouched): \resprogabl off builds clean, main text ends at 85% of page 9;
  all three newer-model switches on builds clean, main text ends at 97% of page 9 (with their default one-line
  sentences; real \fluxTwoResult-style sentences of more than one line would need a trim).
- Sentence lengths (srclen.py on the source; inline math = 1 word, citations removed): body 166 sentences, median 20,
  mean 19.5. Every real sentence has at most 35 words; the longest are the conclusion's second sentence (34) and four
  sentences of 35 around displays. The script's max of 39 counts the switched-off "\ifanynewmodel" clause, and its
  36-37 entries merge a heading into a sentence. The PDF-based count (textcheck.py) reports a median of 22 because it
  counts equation glyphs as words.
- Forbidden words (PDF, pages 1 to 10): none of calibration/calibrate/calibrated, curated/uncurated/curation,
  "random", license, warrant, plainly, em dash, section symbol. Source main.tex is clean of the same words, comments
  included. Appendix pages 26, 50 and 53 contain "harness" and "intricate" (inside quoted prompts, appendix owner).
- Abstract: 162 words.

## 6. Self-check against the gold-standard rubric (text items; 0 below, 1 meets, 2 above)
Abstract with number and URL 1; hero under abstract with walking caption 1; question by end of P1 1; motivating
figure beside P2 1 (top of the page, not a wrap); "We introduce" paragraph defines the method 1; numbers in P4 2
(interval and n); bullets 1; median sentence under 22 and none over 35 1; every display introduced and followed by
"where" 1; boxed algorithms side by side with Input/Output and pointers 1; setup run-ins 1 (no compute hours in the
main text; Appendix I.1 and tab:sources hold them); finding paragraphs with claim leads and before/after numbers 2
(intervals and Holm); confounds named (HRDiT precision and runner, development prompts) 1; limitations concrete 1;
reproducibility statement mapped 1; floats within one page 0 for Table 3, 1 otherwise; captions bold-lead, above for
tables 1; no em dash or section symbol 1.

## 7. Could not verify
- The figure files kept changing during this round (Figure 3 round 2 at 11:29; Figure 6 at 11:23 and again at 11:33).
  The captions match those files. A later change of any v6 figure's height, panels or draw needs a caption check and a
  page check (page 9 has about three lines spare).
- The selection rules in the Figure 1 and Figure 6 captions are taken from the visuals notes and provenance JSON; I
  did not re-run their selection scripts.
- I did not build with \scafinaltrue (no placeholder remains, so it changes nothing).

## 8. Lead-review pass (11:40 to 12:00)
- Float layout, final (build v6_text_33; placed / first cited): Table 1 p5/p6 (its source now sits in Section 3.3 and
  the Models paragraph cites it); Figure 5 gains + Table 2 generality p6/p7 (source opens Section 4); Figure 6
  qualitative p7/p7 (source opens 4.1); Table 3 ablations p8/p8 (4.3 now opens with "Per-head weights beat coarser
  ones in images (Table 3)", then "Output error shows the same order", then views and opening); Figure 7 scaling p8/p8
  as a wrapfigure beside the two 4.2 paragraphs (time paragraph first). No float inside Related Work. Every float is on
  the page of its first citation or one page before it.
- Figure 7 wrap width: natural 0.48 text width, not 0.42. At 0.42 its smallest labels would fall to 5.3 pt (it is
  drawn at 2.64 in with 6.04 pt math scripts). If the lead wants 0.42, tablesplots should redraw it at 2.31 in.
- Page 9: 90% full; main text ends on page 9; the reproducibility statement starts page 10.
- Figure 6 (percentile version, qual_main.pdf of 11:38, 5.5 x 4.2 in; rule from qual_main_provenance.json). Caption:
  "Progressive SCA's gain over AReA is in the whole image, in a quarter of AReA's time, and it is level with HRDiT.
  Rows: the evaluation prompts at the 25th, 50th, and 75th percentile of the paired ImageReward difference,
  progressive SCA minus AReA (seed 0, n = 195, mean +0.40). Numbers: each image's ImageReward, best bold; insets:
  each image's 512 x 512 window of highest edge energy at native pixels; heads: seconds per image (double dagger:
  campaign mean). Over all prompts, ImageReward differs from HRDiT's by -0.02 [-0.09, 0.03]. Figures L.3 and L.4 show
  more." The citing sentence ends the first 4.1 paragraph: "Figure 6 shows that the gain lies in the whole image, on
  the prompts at the quartiles of the ImageReward difference."
- Crop detail: the HRDiT paragraph now says "Crop detail is statistically indistinguishable from HRDiT's (MUSIQ_c
  +0.6 [-0.7, 1.8]; \cropsref)". \cropsref prints Figure~\ref{fig:crops_matched} once that label exists and
  "Appendix L" (app:galleries) until then. APPENDIX OWNER: please give the matched-crop figure the label
  fig:crops_matched. This is the only main-text citation of it.
- Bullets: one quotable sentence each, at most two lines, each starting with "We"; the last carries 49.2 s, 4.1x and
  all ten metrics at three seeds (macros \hlSecProg, \hlSpeedProgArea). The lead-in "We make three contributions."
  was dropped so that Section 2 still starts on page 2.
- Figure 2 caption (c) now reads "a schedule shared by all heads can follow only the head average", matching the
  "shared schedule 0.7%" label of the fixed panel (c).
- AI use statement: a commented marker sits above the reproducibility statement in main.tex ("AI USE STATEMENT GOES
  HERE ..."); the lead is raising the disclosure with the user.
- Checks on v6_text_33: 0 overfull boxes (main and appendix), no undefined references or citations, abstract 162
  words, no forbidden words in the main text, source-based sentence median 20 with every real sentence at most 35
  words.

## 9. Round "more visual results" (2026-11-10, ~13:20 to 14:10): Figure 8 and the room for it
Page limit. The ICLR 2027 instructions (template text, Exploration/semtwin-twinflow-2/paper/iclr2027_conference.tex,
line ~131): "There will be a strict upper limit of 9 pages for the main text of the initial submission, with unlimited
additional pages for citations. This limit will be expanded to 10 pages for rebuttal/camera ready."
paper/iclr2027_conference.sty (identical to the TABES and semtwin copies) states no limit; TABES-iclr2027/README.txt:
"The main text is 9 pages". So the main text stays within 9 pages. Item (e) was not needed: both algorithms stay.

New figure: Figure 8, figures/v6/sizes_models.pdf (13:33, 5.50 x 2.40 in, layout A, sizes_models_provenance.json),
label fig:sizes_models, [t] float sourced after Section 4.2, lands at the top of page 8; cited once, in 4.2 ("Figure 8
shows samples at other sizes and on both checkpoints; at 2048 x 6144 ... but at 2048 x 4096 it wins three and loses
three"). Not cited from the abstract or introduction: neither discusses other sizes, so no sentence there reads
naturally with it. \IfFileExists keeps a 5.5 x 2.4 in grey block if the file is ever missing. Caption:
  One reach table per checkpoint serves every size and checkpoint shown. Single-pass SCA on FLUX.1-dev and
  FLUX.1-Krea-dev, SCA without the schedule on FLUX.2-dev; one prompt per setting naming no person, drawn with seed 0;
  insets: native 256 x 256 windows of highest edge energy. Settings left out for artifacts: 6144 x 4096, 6144^2, and
  8192^2 (Figures G.4 and G.6). Appendices G and L show more.
  (Short edition: no gallery pointer, "Appendix G shows more."; macros \sizesArtifactRef and \sizesMoreRef.)
- visuals: your proposed clause "each from a different subject category" does not hold for the final draw (the
  provenance gives category "landscape" for both the 2048 x 6144 and the FLUX.2-dev tile: the "while one remains"
  fallback applied), so the caption gives the seed and the person rule only; the full rule is in the provenance. Your
  caption also left the 6144 x 4096 drop unstated; ours names all three dropped settings. If the figure is redrawn,
  tell me: the tile description and the dropped list are written from sizes_models_provenance.json of 13:33.

Pointers to the new appendix pages (labels from visuals.md; guard \whenlabel{label}{text}, so a missing label prints
nothing; main.tex cannot use \iflabel, which appendix.tex defines after the main text):
- 4.1, first paragraph: seeds 1 and 2 "(...; Figure~\ref{fig:seeds12})", HRGen-1K "(...; Table 2, Figure~\ref{fig:hrgen_samples})",
  and "Figure 6 shows whole images at the quartiles of the per-prompt difference and Figures~\ref{fig:deciles_a}
  and~\ref{fig:deciles_b} at every decile". Both editions carry these pages (full E.1, E.2, G.1, G.2; short E.1 to E.4).
- Figure 1 caption: "Figures L.3 and L.5 show more" (gallery_prog16, mosaic); short edition "Figures G.1 and G.3".
  The mosaic is 16 more progressive SCA showcase images, Figure 1's own kind, so it is cited from Figure 1.

What I tightened (all prose restating a table or figure; no result, interval, bullet, or the single-pass limitation
sentence was dropped; no figure shrunk):
- Intro P2: prompt count and resolution (Section 2 gives them). Section 2: the lead sentence before "Reach."
  (restated Figure 2's title); "because it barely moves with the step" (restates the 2.5%).
- 3.1: AReA citation repeated in "(AReA's rule)". 3.2: rho = 0.77 and "below every coarser rule" (Figure 4a, c print
  them). 3.3: "where lambda is the weight of the ratio-4 table" (Eq. 7 shows it), "of 28", "native", "right".
- Section 4: Data (seed labels, development prompts) and timing sentences shorter; "a point estimate" dropped.
- 4.1: "also beats DyPE, SEGA, full attention ... UltraImage on nine" (Table 1's record column).
- 4.3: "Output error shows the same order" folded into the per-head paragraph (the four errors are in Figure 4b);
  the mean-pooling clause (Table 3 row); "MUSIQ_c changes by +1.7 and no other column" became "only MUSIQ_c moves
  significantly (+1.7)".
- Related Work: one paragraph; the HRDiT and UltraImage descriptions point to Section 1 instead of repeating it.
- Captions: Table 1 (one line), Table 3 ("coloured as in Section 4", "each group's", the full-metric pointer),
  Figure 5 ((a)/(b) restated the panel titles), Figure 6 (one line).
- Conclusion: "Code: <url>" (the abstract carries it; one fewer place to anonymize).
- Fixed: "exceeds HRDiT's ... and, at two of three seeds, than AReA's" lost its stray "than".
Added: Limitations "At 6144^2 and 8192^2 its samples show block and stripe artifacts." (SCA without the schedule;
Appendix G galleries), since Figure 8 leaves those sizes out. Three sentences over 35 words were split.

Layout (both editions; full has every new pointer printed, so it is the longer case):
- Figure 7 (scaling, natural 0.48 width) now wraps beside the 4.1 HRDiT paragraph on page 7, under Figure 6, with
  \needspace{20\baselineskip} before it. Beside the 4.2 time paragraph it began at the foot of page 7 and wrapfig cut
  it across the break (build 40). The 4.1 single-pass paragraph cites it first ("2.63x faster than AReA (Figure 7)",
  the plot's bottom row); 4.2 cites it again. The end \vspace is -24pt so the next paragraph starts at full width.
- Float page / first citation: Fig 1 p1/p2, Fig 2 p2/p2, Fig 3 p3/p2, Fig 4 p4/p4 (new cite "(Figure 4b)" at the end
  of 3.2), Alg 1 p5/p4, Alg 2 p5/p5, Table 1 p5/p6, Fig 5 + Table 2 p6/p6, Fig 6 p7/p6, Fig 7 p7/p7, Fig 8 p8/p8,
  Table 3 p9/p8. Section 4.1 now starts on page 6.
- Main text ends on page 9: full edition 95% (3 lines free), short edition 97%; reproducibility statement on page 10.
  0 overfull boxes, 0 undefined references or citations in both; abstract 162 words; no forbidden word in pages 1 to 10.
- Robustness: wraps cannot cross a page; the \needspace guard moves the HRDiT paragraph to the next page if it ever
  starts with fewer than 20 lines left (it starts with 23 free lines now).
