# Paper v6: tables and plots (role `tablesplots`), 2026-11-10

Status: DONE. Main Tables 1 to 3, the Holm analysis and its macros, the 47 requested macros, Figures 6 and 7, three
appendix result plots, and every generated appendix table restyled (plus four new generated tables that replace
typed ones). Generators, run from paper/ with the sca env python: gen_main_tables.py prints "checks passed";
gen_app_tables.py exits 0 with its two old informational warnings (krea claim regenerated; prog CLIP-IQA note).
Full build (tectonic, 61 pages, exit 0, no undefined references): /scratch/09032/saini_2/sca/paper_builds/
v6_tablesplots_4/main.pdf; renders read: pages 6, 7, 9 (Tables 1 to 3, Figures 5/6 and 7 as numbered there), 30, 32,
34, 35, 43 (appendix tables).

## 0a. Lead review fixes (2026-11-10, done)
- Panel tags: one style paper-wide, the whole tag bold, "(a) Against AReA" / "(b) Against HRDiT" in gains.pdf
  (v6_plotkit.panel_tag / panel_tags, which also keep a row of tags clear of the canvas edge and of each other);
  latency.pdf now "(a) Time per image" / "(b) Speed-up of single-pass SCA"; baseline_deltas.pdf "(a) ImageReward" to
  "(d) MUSIQ_c" bold. Candidate gains_B renamed in the same words. All audits clean; rerun byte-stable.
- gen_all prints ImageReward to three decimals (as phase2_general and prog_all). gen_all now holds every number of
  phase2_general (multiset check: nothing of phase2_general missing), so the appendix owner can retire
  tab:phase2_general (put \label{tab:phase2_general} beside tab:gen_all so the main-text reference still resolves).
- scafig.py: CROP = "#E4572E" (crop boxes, visuals' request); sca_colors.tex regenerated with
  \definecolor{scaCrop}{HTML}{E4572E} (the only new line). The colour name does not collide with the \scaCrop macro
  of the gallery snippets (colours live in their own namespace).
- tab:flops is typed inline in appendix.tex (no snippet exists, so there is none of mine to comment). Its timing rows
  ("One layer, bf16 (ms)": 7.1 / 87.3 / 49.6 full attention, 5.2 / 18.7 / 14.0 SCA) have NO archived source under
  results/; the only record is HANDOVER.md Section 4 ("Per-layer attention at 4096^2 is 18.7 ms for SCA vs 87.3 ms").
  REQUEST to the appendix owner: (1) a comment above the float: "% Timing rows: measured once on one GH200 during
  development; no archived timing file under results/ (HANDOVER Section 4); not regenerable by a script.";
  (2) in the caption replace "Bottom: measured time of one attention layer ..., minimum over interleaved runs" by
  "Bottom: time of one attention layer ..., measured once on one GH200 (minimum over interleaved runs; no archived
  timing file)"; (3) its lead "6.9 times" can be \hlWorkedPairsRatio (same cost model, checked against 8.56e16 and
  1.24e16). The FLOPs rows are analytic and reproduced by gen_main_tables.request_macros at 4096^2.
- Legacy generators: gen_late_tables.py and gen_prog_tables.py now open with a "DO NOT RUN" banner and refuse to run
  unless SCA_RUN_LEGACY=1 (tested: both exit 1 with the message; importing them, as gen_main_tables.py and
  app_tab_phase2.py do, still works).
- Also: \hlProgUltraMusiqc{,Lo,Hi} (+12.4 [11.5, 13.2]) generated, the text owner's optional request.
- Verified: gen_main_tables.py "checks passed", gen_app_tables.py exit 0, strict tectonic build of main.tex exit 0
  (61 pages, no undefined references; three 2 pt overfull lines are in main.tex's algorithm listings, not tables):
  /scratch/09032/saini_2/sca/paper_builds/v6_tablesplots_5/main.pdf, page 6 read.

## 0. Actions for other owners
- text: done on your side as far as tables go: Tables 1 to 3, Figure 6 caption (standardized differences, two
  panels) and the Holm wording are in the build. All 47 \reqnum macros are generated and equal their placeholders.
  Labels: Figure 6 keeps fig:paired_area; Figure 7 is fig:scaling (main.tex).
- appendix: all generated tables are wired (holm, gen_all, app_abl, app_ps2, phase2_sensitivity, phase2_budget,
  v6_baseline_deltas, v6 latency). Remaining:
  (1) appendix.tex line ~779 "Against HRDiT. Progressive SCA is significantly better on HPSv2, MUSIQ, and ARNIQA,
      worse only on CLIP-IQA" is the unadjusted reading: add "before correction; after Holm correction over the ten
      metrics HPSv2 and ARNIQA remain and no metric is worse (Table~\ref{tab:holm})" or use \holmProgHrditKept.
  (2) phase2_general (tab:phase2_general) duplicates gen_all cell for cell (checked mechanically by the worker; gen_all
      prints ImageReward to 2 decimals, phase2_general to 3). Either drop it and put \label{tab:phase2_general}
      beside tab:gen_all, or keep it. phase2_baselines is likewise contained in panel46_deltas (labels already shared).
  (3) Proposed captions for the "body" tables are in Section 7 below (all_metrics, app_4k, app_wide, latency,
      phase2_general, panel46_deltas, phase2_baselines); each claim is checked by gen_app_tables.py.
  (4) Sizes grew where 7 pt replaced \resizebox: krea_6144 3.5 -> 6.7 in total; panel46_deltas 6.7 -> 7.2 in body
      (8.8 with its caption: give it a float page); phase2_baselines and phase2_general 1.5 -> 3.4 in.
- methodfigs: (1) rel_err belongs to you; a v6-style option sits on scratch:
  /scratch/09032/saini_2/sca/paper_builds/v6_tablesplots_figs/rel_err_option/ (v6_rel_err.py, rel_err.pdf). (2) One
  panel-tag style across owners: hero, gains, scaling and the appendix plots use a bold "(a)" with a regular name and
  sentence-case axis titles with arrows; emulation.pdf bolds the whole tag and uses lowercase axis titles. Please
  match, or the lead picks one.
- lead: legacy generators NOT to run: gen_late_tables.py rewrites paper/main.tex unconditionally (v4 era; its
  snippets are now written by gen_app_tables.py); gen_prog_tables.py writes the unused snippets/prog_deltas.tex and its
  tab_panel46 cross-check has failed since v5 (it expects pre-v5 cell marks). Untouched by me.

## 1. Holm result (FINAL)
Module paper/figures/src/gen_holm.py re-runs scripts/paired_bootstrap.py on the archived per-image files (2000 prompt
draws, seed 0, one generator shared by the metrics in sorted order) and reproduces all 25 archived paired CSVs it uses
digit for digit (the run fails otherwise). Two-sided bootstrap p-values from the same draws, p = min(1, 2 min(1 + #{b
<= 0}, 1 + #{b >= 0}) / 2001); Holm step-down over the ten per-image metrics of each pair at 5%; a paired sign-flip
test (2000 sign vectors, seed 0) as a check. Smallest resolvable p = 0.001.

| pair (n) | unadjusted better/worse | Holm, bootstrap | Holm, sign-flip |
|---|---|---|---|
| progressive SCA - AReA (195) | 10/0 | 10/0 | 10/0 |
| progressive SCA - HRDiT (195) | 3/1 (HPSv2, MUSIQ, ARNIQA / CLIP-IQA) | 2/0 (HPSv2, ARNIQA) | 2/0 |
| single-pass SCA - AReA (195) | 10/0 | 10/0 | 10/0 |
| progressive - UltraImage / DyPE / SEGA / full (195) | 9/0, 10/0, 10/0, 10/0 | same | same |
| progressive - single-pass (195) | 7/2 | 7/2 | 6/2 (ImageReward drops) |
| progressive - AReA, seeds 1, 2 (195), HRGen-1K (1000) | 10/0 each | 10/0 each | 10/0 each |
| single-pass - AReA, seeds 1, 2 (195), HRGen-1K (1000) | 10/0 each | 10/0 each | 10/0 each |
| single-pass - AReA, FLUX.1-Krea (195) | 7/0 | 7/0 | 7/0 |
| w/o schedule - full attention, FLUX.2-dev (50) | 9/0 | 9/0 | 9/0 |
| per-head over shared / per-layer, single pass (46) | 9/0, 5/0 | 9/0, 5/0 | 7/0, 5/0 |
| per-head over shared / per-layer / window only, progressive (195) | 10/0, 10/0, 8/1 | 10/0, 10/0, 6/1 | same |
| per-head over no opening / local / global / mean pooling (46) | 6/2, 8/0, 10/0, 9/0 | 6/2, 7/0, 10/0, 8/0 | same |

What must weaken in the prose:
- HRDiT paragraph (v5.1: "better on HPSv2, ARNIQA, and MUSIQ and worse on CLIP-IQA"). After Holm only HPSv2 and
  ARNIQA remain; MUSIQ (p = \holmProgHrditMusiqP, Holm \holmProgHrditMusiqPHolm) and CLIP-IQA (p =
  \holmProgHrditClipiqaP, Holm \holmProgHrditClipiqaPHolm) do not. Suggested: "After Holm correction over the ten
  metrics it remains better on \holmProgHrditKept{} and is worse on none; \holmProgHrditDropped{} differ only before
  the correction (Table~\ref{tab:holm})."
- Progressive over single-pass ImageReward (+0.07 [0.01, 0.13]): p = \holmProgScaIrP; Holm \holmProgScaIrPHolm
  (bootstrap), \holmProgScaIrPHolmFlip (sign-flip). Keep the interval; do not call it a robust gain.
- Everything else survives: "all ten per-image metrics" against AReA, both modes, three seeds, HRGen-1K; UltraImage
  nine; FLUX.1-Krea seven; FLUX.2-dev nine; inside the progressive sampler per-head weights win all ten against
  shared and per-layer weights.

## 2. Macros (headline.tex, all \newcommand; generated by gen_main_tables.py)
- Holm counts, for every pair: \holm<Pair>Better, \holm<Pair>Worse (digits), ...BetterWord / ...WorseWord (words),
  ...BetterRaw / ...WorseRaw (unadjusted). <Pair> in {ProgArea, ProgHrdit, ScaArea, ProgUltra, ProgDype, ProgSega,
  ProgFull, ProgSca, ProgAreaSeedOne, ProgAreaSeedTwo, ProgAreaHrgen, ScaAreaSeedOne, ScaAreaSeedTwo, ScaAreaHrgen,
  ScaAreaKrea, FluxTwo, AblShared, AblLayer, ProgAblShared, ProgAblLayer, ProgAblWindow, AblNoOpening, AblLocal,
  AblGlobal, AblMean}. For the ablation pairs "Better" means the variant is better (so per-head wins = ...Worse).
- \holmProgHrditKept (HPSv2 and ARNIQA), \holmProgHrditDropped (MUSIQ and CLIP-IQA), \holmProgHrditMusiqP,
  \holmProgHrditMusiqPHolm, \holmProgHrditClipiqaP, \holmProgHrditClipiqaPHolm, \holmProgScaIrP,
  \holmProgScaIrPHolm, \holmProgScaIrPHolmFlip.
- Latency: \hlLatExpSca (0.99), \hlLatExpArea (1.12, was typed in main.tex and appendix.tex), \hlLatEightSca,
  \hlLatEightArea (952.0, was typed), \hlLatEightRatio.
- The 47 \reqnum requests of main.tex (hlEmu*, hlAbl*, hlKrea*, hlWide*, hlSixKWins, hlEightKFavWord, hlWorked*,
  hlFidAreaAbs, hlAreaReproGap, hlFluxTwoWinsWord, hlLat*): all generated from results/ (sources in headline.tex
  comments); the generator checks each equals the placeholder value. hlWorkedPairsRatio / LocalKeys come from the
  analytic cost model of Table tab:flops (method constants), checked against its printed 8.56e16 / 1.24e16.

## 3. Tables (main text)
Shared style file: main_tables/tabstyle.tex (written by gen_main_tables.py; every v6 table snippet inputs it once):
colour scaRowTint #EAF4F1 (pale tint of SCA green #13865F), scaCtx #6E7781 (interval grey), \scaRowOurs (shade),
\camp (campaign-mean mark), \scaCI, \scaFitTab{colspec}{ncols}{rows} (sets a booktabs tabular at \linewidth by
spreading spare width over the cell paddings; shaded rows stay seamless, unlike tabular*).

### Table 1, main_tables/tab_main_compact.tex (tab:main): full text width at \small (9 pt)
Columns: time per image (clean batch-1 seconds; a method with no clean time prints its campaign mean with a double
dagger) and "vs. ours" (that time over progressive SCA's time of the same kind; equals \hlSpeedProgArea,
\hlSpeedProgHrdit, \hlSpeedProgFull, \hlSpeedProg{Dype,Sega,Ultra}Camp, checked); IR, HPSv2.1, PickScore; MUSIQ_c;
FID and patch FID (FID is new); "Ten metrics +/-" (Holm-adjusted record of progressive SCA against the row). No "--"
cells. Best bold, second underlined within each group; both our rows shaded; group heads italic.
Proposed caption:
\caption{\textbf{Progressive \sca{} is the fastest method at $4096^2$ and beats every single-pass baseline on all ten
per-image metrics.} FLUX.1-dev, \nmain{} Aesthetic-4K prompts, seed 0. Time: clean batch-1 median on one idle GH200;
vs.\ ours: that time over progressive \sca{}'s; $^\ddagger$campaign mean on shared nodes, the only time for that
method, over progressive \sca{}'s campaign mean (\hlSecProgCamp\,s). IR ImageReward, HPSv2.1 $\times100$, Pick
PickScore, M$_\text{c}$ MUSIQ on four $1024^2$ crops, FID$_\text{p}$ patch FID$^\dagger$. Ten metrics: on how many of
the ten per-image metrics progressive \sca{} is significantly better\,/\,worse than the row (paired bootstrap over
prompts, 2000 draws, Holm-adjusted). Bold: best, underline: second per group; shaded: ours.
Table~\ref{tab:all_metrics} lists every metric. Key result: progressive \sca{} needs \hlSecProg\,s per image,
$\hlSpeedProgArea\times$ less than AReA and $\hlSpeedProgHrdit\times$ less than HRDiT, and HRDiT is significantly
better on none of the ten metrics.}

### Table 2, main_tables/tab_ablation_compact_prog.tex (tab:ablation): FULL WIDTH at \small
(tab_ablation_compact.tex is the same table without the progressive group.) Three groups: reach weights in single pass
(46 prompts), reach weights in progressive SCA (195), views (46). Each group opens with its shaded per-head reference
in absolute values. FIX of v5.1: its shaded row printed single-pass SCA (1.17 / 30.07 / 23.02 / 60.7) above deltas
taken against SCA without the schedule; v6 prints the true reference (1.12 / 29.98 / 22.96 / 62.5). New columns:
patch FID (progressive rows with cluster-bootstrap intervals, single-pass rows point estimates) and the Holm record.
Proposed caption:
\caption{\textbf{Per-head weights beat shared and per-layer ones in both samplers.} FLUX.1-dev at $4096^2$. Each group
opens with its shaded per-head reference in absolute values; the rows below give the variant minus that reference
(views: minus the first shaded row). \dn{Orange} / \up{blue}: the 95\% paired interval excludes zero against / in
favour of the variant (FID$_\text{p}$: lower is better; the \nabl-prompt values are point estimates). $+$\,/\,$-$: of
the ten per-image metrics, those on which per-head weights are significantly better\,/\,worse after Holm correction.
Tables~\ref{tab:app_abl} and~\ref{tab:prog_abl_all} give every metric with its interval. Key result: inside the
progressive sampler per-head weights win all ten per-image metrics against shared and per-layer weights, which have
lower patch FID.}

### Table 3, main_tables/tab_generality.tex (tab:generality): 0.48 text width at \footnotesize (8 pt)
One line per setting: n, paired differences in IR, HPS (HPSv2.1 x100), PickScore (colour = interval), Holm record.
Single-pass seed 0 row added. Intervals moved to the new appendix table tab:gen_all (every number kept).
Proposed caption:
\caption{\textbf{The gains hold across seeds, prompt sets, and checkpoints.} Paired differences at $4096^2$ (2000
prompt draws); HPS is HPSv2.1$\times100$. FLUX.1-Krea-dev is AReA's checkpoint; FLUX.2-dev pairs \sca{} without the
schedule with full attention. \up{Blue}: the 95\% interval excludes zero in favour of \sca{}. Holm $+$\,/\,$-$:
metrics of ten significantly better\,/\,worse after Holm correction. Table~\ref{tab:gen_all} gives all ten metrics with
intervals. Key result: no setting is significantly worse on any metric.}

## 4. Appendix tables (FINAL)
| snippet | label | kind | change | size |
|---|---|---|---|---|
| app_tables/holm.tex | tab:holm | float (new) | (a) three headline pairs x ten metrics: delta, p, p_Holm; (b) record of all 25 pairs | about 0.75 page |
| app_tables/gen_all.tex | tab:gen_all | float (new) | expands Table 3: 10 settings x 10 metrics, two panels, grey intervals below | about 0.6 page |
| app_tables/app_abl.tex | tab:app_abl | float (new, replaces the typed one) | expands Table 2 single-pass rows: (a) preference, (b) image quality, (c) patch FID and output error | about 0.55 page |
| app_tables/prog_all.tex | tab:prog_all | float (restyled) | transposed: 7 baselines as rows, three panels (preference, image quality, set level), grey intervals; no \resizebox | about 0.6 page (taller than v5.1, now 7 pt) |
| app_tables/prog_abl_all.tex | tab:prog_abl_all | float (restyled) | same layout, 3 variants | about 0.35 page |
| main_tables/tab_panel46.tex | tab:panel46 | body (restyled) | one time column (campaign mean, every row has one), grouped heads; caption in appendix.tex should drop "clean" | same height |
| app_tables/variance_split.tex | tab:variance_split | float | unchanged | |
| app_tables/app_ps2.tex | tab:app_ps2 | float (new, replaces the typed one) | 2 rows (ratio-2 table, AReA) x ten metrics, two panels | about 0.25 page |
| app_tables/phase2_sensitivity.tex | tab:phase2_sensitivity | float (new, replaces the typed one) | 5 rows (4/8/16-prompt tables, two gates) | about 0.4 page |
| app_tables/phase2_budget.tex | tab:phase2_budget | float (new, replaces the typed one) | 3 rows (56 vs 28 steps, guidance 7 vs 4.5, 56 steps vs UltraImage) | about 0.3 page |
Captions of float snippets are generated; each states what it expands or supports and ends with "Key result:"; each
claim in a caption is checked against the cells by the generator.
Restyled in app_tab_main.py and app_tab_phase2.py (gen_app_tables.py), same labels, all at 7 pt with tabstyle.tex,
no \resizebox; body = caption in appendix.tex, float = caption in the snippet (sizes v5.1 -> v6, height x width):
| snippet | kind | change | size |
|---|---|---|---|
| all_metrics | body | heads grouped Time / Preference / Image quality / Set level; ours shaded; second best underlined in the 4096^2 block | 2.96 x 5.5 (at about 6 pt) -> 2.85 x 5.5 in |
| app_4k | body | one row per metric under family heads; each delta with its grey interval | 2.41 x 4.9 -> 2.88 x 5.5 in |
| app_wide | body | same layout, two size groups | 2.59 x 5.15 -> 3.01 x 5.5 in |
| latency | body | same cells, "not run" in grey | 1.23 x 3.55 (8 pt) -> 1.31 x 3.10 in |
| published_compare | float | single-line heads with arrows, ours shaded; new caption (lead: six of seven published scores reproduced within 0.28) | 4.47 -> 4.05 in |
| published_hrgen | float | our rows split by checkpoint | 2.98 -> 3.18 in |
| krea_6144 | float | comparisons as rows in three panels (it did not fit side by side at 7 pt) | 3.47 -> 6.71 in |
| phase2_baselines | body | baselines as rows, two panels (label shared with panel46_deltas) | 1.50 -> 3.38 in |
| phase2_general | body | gen_all layout, two panels (duplicates gen_all) | 1.54 -> 3.38 in |
| panel46_deltas | body | three panels, baselines as rows | 6.66 x 5.24 -> 7.23 x 5.5 in |
| prog_deltas | float | one row per metric; caption now states the Holm calls (read from gen_holm.pair) | 2.41 -> 3.15 in |
Worker's numeric check against tag paper-v5.1, all 11 tables: equal multisets of decimal values, equal \up / \dn /
\textbf marks, every (mark, estimate, lo, hi) tuple preserved; intended differences only the 13 new \underline cells of
all_metrics and repeated panel labels. Script and output: /scratch/09032/saini_2/sca/paper_builds/
v6_tablesplots_app/numcheck.py, numcheck_out.txt.
Still typed by hand in appendix.tex: tab:flops (analytic cost; its 4096^2 entries are reproduced by the cost model in
gen_main_tables.request_macros, so a generated table can follow on request), tab:handset, tab:hparams,
tab:protocol, tab:related (configuration tables, no results).

## 5. Number audit (no number changed)
Script: /scratch/.../scratchpad/tp/numdiff.py compares the multiset of numbers in table rows, old snapshot
(/scratch/09032/saini_2/sca/paper_builds/v6_tablesplots_base/snippets_orig, = v5.1) against v6:
tab_main_compact: every v5.1 number present; tab_ablation_compact_prog: every v5.1 number present except the four
values of the mislabelled shaded row (intended fix above); tab_generality: every v5.1 number present in
tab_generality + gen_all; prog_all, prog_abl_all: identical numbers and identical \up/\dn sets; app_abl, app_ps2,
phase2_sensitivity, phase2_budget: every number of the typed v5.1 tables present. One formatting rule: an interval
end of a SIGNIFICANT interval that would print as zero gets more decimals (as gen_tables.py did: [0.0002, 0.062]);
this refines three v5.1 ends that printed as 0.0 (16-prompt MUSIQ lower end, 56 steps vs UltraImage CLIP-IQA lower
end, window-only ARNIQA upper end in prog_abl_all). No value changed.

## 6. Figures 6 and 7 and the appendix result plots (FINAL)
Scripts in paper/figures/src (new, mine): v6_plotkit.py (widths, panel tags, save with the orx audit), v6_gains.py,
v6_scaling.py, v6_latency.py, v6_quality_time.py, v6_baseline_deltas.py. Each reads results/ (gen_holm.pair for the
gains), asserts every plotted value against the printed tables and headline macros, and reruns byte-identical
(checked after the scafig.py change below). Audits clean; smallest text 6.04 pt (math scripts), labels 7 pt or more.
Two candidates for each main figure; losers in paper/figures/v6/candidates/. PNG renders (300 dpi):
/scratch/09032/saini_2/sca/paper_builds/v6_tablesplots_figs/final/.

### Figure 6, v6/gains.pdf, 0.48 wrap (2.64 x 2.42 in), label fig:paired_area
Claim: both modes beat AReA on all ten per-image metrics after Holm; progressive SCA is never significantly worse
than HRDiT and is better on HPSv2 and ARNIQA. Normalisation: standardized paired effect d_z (difference / SD of the
195 per-prompt differences; the interval ends divided by the same SD), because percent of AReA's mean put ImageReward
at +43% and squeezed the rest. Marks: star progressive, circle single pass (as the hero); green = interval excludes
zero in SCA's favour, orange = against, grey = includes zero; filled = significant after Holm, open = before only.
Caption (fits a 0.48 wrap):
\caption{\textbf{Both modes beat AReA on all \holmProgAreaBetterWord{} per-image metrics; progressive \sca{} is worse
than HRDiT on \holmProgHrditWorseWord{}.} Paired differences at $4096^2$, $n=\nmain$, divided by the standard
deviation of the per-prompt differences, with 95\% bootstrap intervals. (a)~Progressive (star) and single-pass
(circle) \sca{} minus AReA; (b)~progressive \sca{} minus HRDiT. Filled: significant after Holm correction; open:
before it only. Table~\ref{tab:holm} lists the values.}
| rubric (0/1/2) | A wrap (kept) | B full width 5.5 x 1.85 |
|---|---|---|
| one claim at a glance | 2 | 2 |
| labels >= 7 pt, nothing < 6 | 2 | 2 |
| <= 4 hues, consistent meanings | 2 | 2 |
| no title, frameless legend, no top/right spines, light grid | 2 | 2 |
| direction stated | 2 | 2 |
| uncertainty shown, n in caption | 2 | 2 |
| layout density | 2 | 1 (three panels repeat one message, over the 1.5 in budget) |
| words <= 30 | 2 (22) | 2 (21) |
| mean | 2.0 | 1.9 |

### Figure 7, v6/scaling.pdf, 0.48 (2.64 x 2.19 in), beside Table 3; label: text owner's choice (fig:scaling)
Claim: single-pass SCA's time grows linearly with pixel count (b = \hlLatExpSca against \hlLatExpArea for AReA), so
its lead over AReA widens from \hlSpeedScaAreaLo x to \hlSpeedScaAreaHi x; progressive SCA is faster still at 4096^2.
Caption:
\caption{\textbf{Single-pass \sca{}'s time grows linearly with pixel count, so its lead over AReA widens from
$\hlSpeedScaAreaLo\times$ to $\hlSpeedScaAreaHi\times$.} Clean batch-1 seconds per image on one GH200 against image
side, log-log; median of three timed images after a warm-up, whiskers min to max. Labels give $b$ of seconds
$\propto N^b$ ($N$ image tokens, least squares over all sizes); bottom row: AReA's time over single-pass \sca{}'s.
Progressive \sca{} and HRDiT were timed at $4096^2$ only, full attention not at $8192^2$. Table~\ref{tab:latency}
lists every time.}
| rubric (0/1/2) | A one panel + ratio row (kept) | B panel + speed-up bars |
|---|---|---|
| one claim, labels, hues, chrome, direction, uncertainty (whiskers, n = 3) | 2 each | 2 each |
| layout density | 2 | 1 (bars repeat the five ratios, +0.28 in) |
| words <= 30 | 2 (16) | 2 (15) |
| mean | 2.0 | 1.9 |
(The figure prints full attention's exponent 1.60 directly; it has no macro and the text does not quote it.)

### Appendix plots
- v6/latency.pdf (fig:latency_full), 5.5 x 1.95 in: (a) seconds log-log with exponents, (b) speed-up over AReA
  (2.07 to 3.07) and over full attention (to 4.36), parity line. Caption: \caption{\textbf{Single-pass \sca{} is the
  fastest method at every size; its lead over AReA grows to $\hlLatEightRatio\times$ at $8192^2$.} (a)~Median seconds
  per image of three timed images after one excluded warm-up (batch 1, one GH200, 28 steps including text encoding
  and VAE decoding), whiskers min to max; labels give $b$ of seconds $\propto N^b$. (b)~Each baseline's median time
  over single-pass \sca{}'s; dotted: parity. Full attention was not run at $8192^2$. Table~\ref{tab:latency} lists the
  times.}
- v6/quality_time.pdf (fig:quality_time), 0.5 width, 2.75 x 1.95 in. Caption: \caption{\textbf{At
  \hlCampProgHrditPct\% of HRDiT's time, progressive \sca{} is indistinguishable from HRDiT on ImageReward
  (difference $\hlProgHrditIr$ $[\hlProgHrditIrLo,\hlProgHrditIrHi]$).} ImageReward against campaign seconds per image
  on shared nodes, $n=\nmain$ prompts at $4096^2$. Star: progressive \sca{}; filled circle: single-pass \sca{}; open
  circles: \sca{} without the schedule at 28, 42 and 56 steps and at guidance 7; the dashed guide carries the ratio of
  campaign times. Full attention, DyPE and SigMa fall below the range (Table~\ref{tab:panel46}).}
- v6/baseline_deltas.pdf (fig:baseline_quality), 5.5 x 1.50 in, with snippets/v6_baseline_deltas.tex (caption
  inside; the old baseline_deltas.tex is untouched).

### Shared style change (announce)
scafig.py: added NS = "#828E9A" (not significant, the existing neutral grey) and METHOD_COLOR / METHOD_NAME entries
for HRDiT ("#56626F", the darkest neutral, as fig_speed.py drew it). No existing hex value changed; every v6 plot
reran byte-identical.

## 7. Proposed captions of the body tables (appendix owner pastes them; source of truth)
Written by the table worker; every claim is checked by gen_app_tables.py. Copy of
/scratch/09032/saini_2/sca/paper_builds/v6_tablesplots_app/captions.tex (scratch, purge-eligible):
```latex
% Proposed v6 captions for the tabular snippets (appendix.tex owns the floats). One \newcommand per table.
\newcommand{\capAllMetrics}{\caption{\textbf{Both \sca{} modes beat AReA on all ten per-image metrics at $4096^2$; the non-square sizes are mixed.} Expands Table~\ref{tab:main} to every metric and adds the non-square sizes and $6144^2$. FLUX.1-dev, \nmain{} prompts (100 at $6144^2$, where the row is \sca{} without the schedule); HPS, CLIP-IQA, and ARNIQA $\times100$; AReA runs its released code. \up{Blue} / \dn{orange}: the 95\% paired percentile-bootstrap interval against AReA excludes zero in favour of / against the row. Progressive \sca{} is paired over 2000 prompt draws, and for FID and FID$_\text{p}$ over 1000 draws that also resample reference-image clusters. Full attention's FID and FID$_\text{p}$ are unmarked; Table~\ref{tab:panel46_deltas} pairs DyPE, SEGA, UltraImage, and HRDiT with single-pass \sca{}. \textbf{Bold} / \underline{underline}: best / second best per size (bold only in the two-row blocks); shaded: ours. Time: batch-1 median of three on an idle GH200 (HRDiT and progressive \sca{}: run T3, Appendix~\ref{app:timing}); $^\ddagger$campaign mean on shared nodes; $^\circ$UltraImage's two-stage defaults. Key result: at $4096^2$ progressive \sca{} is the fastest method (49.2\,s) and best on CLIP, both HPS scores, and ARNIQA; single-pass \sca{} is best on patch FID and both MUSIQ scores.\pendAppAllMetrics}}
\newcommand{\capFourK}{\caption{\textbf{Single-pass \sca{} beats AReA on all ten per-image metrics, FID, and patch KID at $4096^2$.} Expands the single-pass row of Table~\ref{tab:main}. FLUX.1-dev, \nmain{} prompts. $\Delta$: the method minus AReA, with its 95\% paired percentile-bootstrap interval in grey. Intervals use 1000 prompt draws for single-pass \sca{} (FID, KID, and their patch versions also resample reference-image clusters) and 2000 for full attention's per-image metrics; full attention's set-level differences and campaign time have none. \up{Blue} / \dn{orange}: significantly better / worse than AReA; for FID, KID, and time lower is better. $^\ddagger$Campaign mean (clean latency: Table~\ref{tab:latency}). Table~\ref{tab:prog_all} pairs progressive \sca{} with every baseline. Key result: KID, patch FID, and Inception score are unresolved; every other metric favours single-pass \sca{}.\pendAppFourK}}
\newcommand{\capWide}{\caption{\textbf{At $2048\times6144$ single-pass \sca{} wins seven of 15 metrics; at $2048\times4096$ the result is mixed.} Expands the aspect-ratio paragraph of Section~\ref{sec:sizes}. FLUX.1-dev, \nmain{} prompts each; AReA runs its released code. $\Delta$: single-pass \sca{} minus AReA, with its 95\% paired percentile-bootstrap interval in grey over 1000 prompt draws; for FID, KID, and their patch versions the draws also resample reference-image clusters. \up{Blue} / \dn{orange}: significantly better / worse than AReA; for FID, KID, and time lower is better. $^\ddagger$Campaign mean on shared nodes. Key result: at $2048\times6144$ single-pass \sca{} wins seven of 15 metrics and loses two, at $2048\times4096$ it wins three and loses three, and it is faster at both.\pendAppWide}}
\newcommand{\capLatency}{\caption{\textbf{Single-pass \sca{} is the fastest method at every size.} Expands the timing of Section~\ref{sec:sizes} and Figure~\ref{fig:scaling}. Seconds per image of full attention, AReA, and single-pass \sca{}: batch 1 on an idle GH200, 28 steps with text encoding and VAE decoding, median of three timed images after an excluded warm-up (Appendix~\ref{app:timing}). $N$: image tokens; full attention was not run at $8192^2$; bold: fastest per size. Key result: single-pass \sca{} is 2.6 times faster than AReA at $4096^2$.\pendAppLatency}}
\newcommand{\capGeneral}{\caption{\textbf{The single-pass gains hold on HRGen-1K, at two more seeds, and on FLUX.2-dev.} Expands the single-pass rows of Table~\ref{tab:generality}. Each cell is the paired difference at $4096^2$ (HPS, CLIP-IQA, ARNIQA $\times100$) with its 95\% percentile-bootstrap interval below it in grey (2000 prompt draws, seed 0). Rows: single-pass \sca{} minus AReA on HRGen-1K (\nhrgen{} prompts) and at seeds 1 and 2 (\nseed{} prompts each); \sca{} without the schedule minus full attention on FLUX.2-dev (50 prompts). \up{Blue}: significantly better than the reference. Key result: every per-image difference favours \sca{}, and only FLUX.2-dev's CLIP score interval includes zero.\pendAppPhaseGeneral}}
\newcommand{\capPanel}{\caption{\textbf{Every shared-sampler baseline trails single-pass \sca{} on all ten per-image metrics; HRDiT and UltraImage lead on several preference metrics.} Expands the baseline rows of Table~\ref{tab:main}. FLUX.1-dev at $4096^2$, \npanel{} prompts, seed 0: each baseline minus single-pass \sca{} in (a) and (b), and minus \sca{} without the schedule in (c), since set-level intervals across runs do not exist. Grey: the 95\% paired percentile-bootstrap interval, over 2000 prompt draws per image and 1000 draws that also resample reference-image clusters for FID, KID, and their patch versions. HPS, CLIP-IQA, and ARNIQA $\times100$; KID $\times10^3$; time: campaign mean in seconds. \up{Blue} / \dn{orange}: baseline significantly better / worse than \sca{}; in (c) lower is better. SigMa uses its dev-matched profile, UltraImage and HRDiT their own recipes (Appendix~\ref{app:additional_protocols}). Full attention's FID$_\text{p}$ lies outside its percentile interval, as a biased estimate can. Key result: UltraImage and HRDiT are significantly better on Aesthetic, PickScore, and HPSv2.1, worse on both MUSIQ scores, and higher in patch FID and patch KID.\pendAppPanelDeltas}}
\newcommand{\capBaselines}{\caption{\textbf{DyPE and SEGA trail single-pass \sca{} on every per-image metric; UltraImage and HRDiT are mixed.} A subset of Table~\ref{tab:panel46_deltas}. FLUX.1-dev at $4096^2$, \nmain{} prompts: baseline minus single-pass \sca{}, with 95\% paired percentile-bootstrap intervals over 2000 prompt draws below each difference in grey. \up{Blue} / \dn{orange}: baseline significantly better / worse than \sca{}. Key result: UltraImage is significantly better on HPSv2.1 and PickScore and worse on CLIP score and both MUSIQ scores.\pendAppPhaseBaselines}}
```
