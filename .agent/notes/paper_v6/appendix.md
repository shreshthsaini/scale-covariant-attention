# Paper v6: appendix (role `appendix`), 2026-11-10

Status: DONE for this round. Owner of paper/appendix.tex only (no other file edited). Last verified build:
/scratch/09032/saini_2/sca/paper_builds/v6_appendix_24/main.pdf (61 pp; appendix pp. 13 to 61 = 49 pp; v5.1 had 43).
Renders: v6_appendix_22/pages/ and v6_appendix_23/pages/ (100 dpi pages and contact sheets); switch builds in
v6_appendix_switches/. Tools (scratch, purge-eligible):
/scratch/09032/saini_2/tmp/tmp/claude-883311/-work-09032-saini-2-vista-Projects-Exploration/4c03b520-8e32-49ed-8c50-e539347fc179/scratchpad/
(render.py, page_fill.py, pdf_numbers.py, table_numbers.py, check_lost.py, sentences.py, switch_tests.sh).

## 1. New order (page numbers of build 24)
| | section (label) | page | what it holds |
|---|---|---|---|
| | Contents block | 13 | two-column list of sections and subsections with page numbers |
| A | Proofs (app:proofs) | 13 | Lemma 1, Props. 2 and 3, Remark 1, proof of Prop. 1 (text unchanged) |
| B | Reach emulation (app:emulation) | 14 | B.1 what one pass records (statistics, geometry, noise bins), B.2 the head sets the reach (Table B.1 variance split, Fig. B.1, attention mass), B.3 the reach table (Figs. B.2 to B.4), B.4 output error of each rule (Figs. B.5, B.6, no global phase) |
| C | Implementation, pseudocode, hyperparameters (app:impl) | 18 | C.1 operators in detail (Fig. C.1 zoom, design choices, window placement), C.2 pseudocode (Algs. 3 to 5, 8 pt, no wrapped lines), C.3 kernel and numerics, C.4 attention cost (Table C.1), C.5 hyperparameters (Tables C.2, C.3 with grey section bands) |
| D | Evaluation protocol and baselines (app:metrics) | 22 | D.1 protocol and prompts (Table D.1), D.2 metrics, D.3 intervals and table conventions, D.4 timing (runs T1, T2, T3), D.5 sample figures (selection, crop rules, configurations), D.6 how methods change attention (Table D.2), D.7 AReA (Tables D.3, D.4), D.8 other baselines and FLUX.2-dev |
| E | Extended results (app:tables) | 28 | E.1 every metric of Table 1, E.2 headline comparisons, E.3 Holm, E.4 every baseline vs single pass (+ Figs. E.1 plot, E.2 baseline samples), E.5 ablations (Tables E.6, E.7, Fig. E.3), E.6 latency (Table E.8, Fig. E.4) |
| F | Progressive SCA (app:prog) | 37 | F.1 protocol, F.2 against HRDiT and single pass (Table F.1, Figs. F.1, F.2) |
| G | Generality (app:generality) | 40 | G.1 seeds, HRGen-1K, FLUX.2-dev (Tables G.1, G.2, Fig. G.1), G.2 6144^2 and Krea (Table G.3, Figs. G.2, G.3), G.3 non-square (Table G.4), G.4 8192^2 (Table G.5, Fig. G.4), [G.5 newer models, only with a model switch on] |
| H | Sensitivity and budgets (app:budget) | 46 | H.1 emulation budget and gate, H.2 table at non-square sizes, H.3 steps, guidance, SigMa (Tables H.1 to H.4, Fig. H.1) |
| I | Compute and reproducibility (app:compute) | 49 | I.1 hardware and compute, I.2 release and sources (Table I.1: every script, timing file and result folder) |
| J | Limitations in detail (app:limits) | 50 | J.1 failure cases and scope (Fig. J.1, eight concrete limits), J.2 broader impact |
| K | Emulation prompts (app:prompts) | 52 | the 48 prompts, two columns, 8 pt |
| L | Sample pages (app:galleries) | 53 | Figs. L.1 to L.9, one per page, filled |
Every section opens with what it holds and which main-text item it expands; every table says which main table it
expands ("Expands Table 1 ...", "Table E.6 expands Table 2"). Floats are numbered by section (Table E.2).

## 2. What moved, merged or changed (no result removed)
- Order follows the reader: proofs, emulation, implementation + full pseudocode, protocol + baselines, extended
  results, progressive, generality, sensitivity, compute, limitations, prompts, samples. v5.1 sections were merged:
  "Implementation details and cost" + "Algorithms in full" + "Hyperparameters" -> C; "Evaluation protocol" +
  "Reproducing AReA" + "Other baselines" + "Mechanisms of related methods" -> D; "Extended results" + "Additional
  paired intervals" -> E, F, G; "SigMa, budgets, guidance" + "Reach table at non-square sizes" -> H; "Compute" +
  "Broader impact" + "Failure cases" -> I, J; "Exploratory 8192^2" -> G.4.
- The v5.1 end-to-end latency paragraph (one 330-word paragraph) is split: protocol in D.4 (runs T1, T2, T3 instead
  of "first/second/third sweep"), times in E.6. File paths of results moved from prose into Table I.1 (I.2).
- "Emulation budget" paragraph of v5.1 B became one sentence in B.4 pointing to H.1, where Table H.1 sits.
- Smaller sample figures now sit beside the results they illustrate (each within its section): baselines (E.2) and
  ablation grid (E.3) in E; progressive and crops (F.1, F.2) in F; FLUX.2-dev, 6144^2, Krea, 8192^2 (G.1 to G.4) in G;
  failures (J.1) in J. The full-page samples stay in L. The selection and crop rules moved from L to D.5.
- Text that the main text no longer states is now in the appendix (text.md Section 4): window start a(tau) and tile
  index (C.1 "Window placement"), latent x_sigma (B.1), same q/k/v and r^2 fewer keys per view (B.1), relative-error
  pooling (B.4), seeds hash prompt id and seed label (D.1), campaign means never ranked (D.4), progressive guidance
  3.5/4.5/6.0 (D.5, F.1). Not added: "most of the head's share lies between heads of one block" (no number reports it).
- Holm results added where the prose made unadjusted claims: E.3 (counts, \holm* macros), F.2 (HRDiT: before vs
  after correction; ImageReward gain over single pass survives bootstrap Holm but not sign-flip).
- Hand-typed v5.1 tables now come from generators (tablesplots): tab:app_abl, tab:phase2_sensitivity, tab:app_ps2,
  tab:phase2_budget (\apxfloat, a labelled placeholder if a snippet is ever missing). Remaining typed tables are
  configuration tables only (flops, handset, hparams, protocol, related, sources).
- Captions: every caption I own leads with a bold takeaway; the six body-table captions are tablesplots' checked
  proposals (Section 7 of tablesplots.md); figure captions adopt methodfigs' corrections (near/far, ratio not scale,
  sigma direction, panel orders of pi_heatmap and reach_tables, head 24 in the zoom) and the v6 visuals snippets.
- Pseudocode: the emulated ratio is `r` (was `s_c`), matching the main text's Algorithm 1.
- Layout: \FloatBarrier per section is replaced by \apxbarrier, a barrier that flushes only when the waiting floats
  exceed 0.9 page; combined with placing floats early in their subsections, every float lands within its own
  section's page span (checked from main.aux) and no page is a stranded paragraph. Page fill of every appendix page
  is 0.69 or more (at most 31% white), measured with page_fill.py. The prompt list uses 8/8.8 pt so that L opens with
  its intro and Figure L.1 on one page.

## 3. Proof that nothing was lost
- Labels: all 83 labels of the v5.1 appendix and of every snippet it input are defined in the v6 build (aux check);
  the 6 switch labels (app:newer, fig:showcase_*, ideogram) are defined when their switches are on. Retired: none.
- References: 0 undefined references and 0 overfull boxes in the whole paper (build 24). All main.tex refs into the
  appendix resolve.
- Numbers, per table (table_numbers.py, git tag paper-v5.1 vs now, multiset of decimal numbers, caption excluded):
  2514 numbers in the v5.1 appendix tables; every data value is present now. The 54 "missing" tokens are layout
  parameters (tabcolsep 1.6/2.4/2.5/2.6/3.1/3.2 pt, row skip 0.5 pt, column widths 0.52, 57.5), the column name
  "HPSv2.1", reference values that v5.1 printed twice (the absolute column of tab:app_abl appeared in both panels,
  13 values, each still printed once), three interval ends that printed as 0.0 and now carry more decimals (tablesplots
  rule), and the clean-time column of tab:panel46 (77.1, still in Tables 1, E.1, E.8).
- Numbers, whole appendix text (pdf_numbers.py on the two PDFs): 4808 numbers in v5.1, 5749 now. Every shortfall
  checked by context: page numbers of the old contents block, figure axis ticks and glyph-encoded labels of redrawn
  plots, "Aesthetic-4K"/"scale-4" tokens, ranges written "0 to 255" instead of "0--255", duplicates (above), and the
  items in Section 6 below that other owners changed.
- Every prose fact of v5.1 is present (checked sentence by sentence while rewriting; two dropped facts found by the
  number check were restored: "Table G.3 uses 1000 draws" and "every set-level interval on a 195-prompt set resamples
  reference clusters").
- Switches (switch_tests.sh, scratch copies): \resprogabl off, all newer models + \showcase on, and v6 visuals snippets
  absent (v5.1 sample figures and fallback texts): each builds with 0 undefined references and 0 appendix overfulls.
- Forbidden words in the appendix PDF text: calibrat*, curat*, random, em dash, section sign, license, warrant,
  plainly, slop words, "sweep", "v0": 0 hits. "match" occurs only as protocol matching (no equivalence claim).
- Sentences: median 18 words; the long ones the audit flagged were split.

## 4. Labels added (all new; none retired)
Sections: app:emu_record, app:emu_head, app:emu_table, app:emu_error, app:zoom, app:code (kept), app:kernel,
app:cost, app:protocol, app:metricdefs, app:intervals, app:timing, app:samples, app:allmetrics, app:pairs, app:holm
(only when holm.tex exists), app:abl, app:latency, app:prog_protocol, app:prog_results, app:generality, app:gen_seeds,
app:gen_krea, app:wide, app:sens, app:budgets, app:hardware, app:release, app:limits. Table: tab:sources.
Snippet labels now included: tab:holm, tab:gen_all (tablesplots), and the v6 visuals snippets (same labels as v5.1).

## 5. Requests to other owners
- text: (1) Figure 1's proposed caption (visuals) cites fig:showcase_models; that figure exists only with
  \showcasetrue (as in v5.1), so guard the reference or drop it. (2) The appendix uses fig:scaling and fig:emulation
  through a guard (\iflabel); if you rename them, nothing breaks, but tell me. (3) Reproducibility statement: point to
  D.4 (timing), D.5 (sample-figure rules), I.2 / Table I.1 (every script and result file).
- tablesplots: tab:phase2_general repeats tab:gen_all cell for cell except the third decimal of ImageReward; see the
  cut list. If you print ImageReward to three decimals in gen_all, phase2_general can go without losing a digit.
- visuals: (1) all_methods caption says "Every 4096^2 method"; SigMa (Table H.3) is not shown, so "eight methods" is
  exact. (2) The Krea gallery's DyPE/SEGA images for m4850138611731379908 come from eval/review1_krea195_dype_sega,
  while Table G.3 reports DyPE and SEGA on the 100 prompts of rows 4 to 103 only; the image is fine, but the lead may
  want to know a 195-prompt Krea DyPE/SEGA run exists. Neither shown SEGA image is one of the seven regenerated ones
  (checked against results/evidence/krea_baseline_provenance/repair_preparation.json); the text says so.
- methodfigs: the panel orders of pi_heatmap and reach_tables changed; the captions now switch on the v6 file, so do
  not reverse the order again without telling me.

## 6. Changes by other owners that remove v5.1 content (lead's decision; not mine to restore)
- zoom4k.pdf (v5.1 fig:qual_main) printed MUSIQ_c under each crop (55.7, 55.3, 57.4, 36.3, 51.2, 49.1, 70.6, 54.4,
  45.2, 48.6, 65.3, 48.5); crops4k.pdf (v6) does not. The v5.1 claim "progressive SCA is softer in every crop" is no
  longer made (visuals: not supported by the new draw).
- v5.1 gallery selection facts no longer describe any figure: "167 of the 195 prompts" (gallery_prog), "Three of 191"
  (gallery_baselines), "320 x 320" and "512 x 512" windows, "16 prompts excluded when the pool was fixed". The D.5 text
  states the v6 rules instead (the v5.1 text returns automatically if the v6 snippets are removed).
- The v5.1 failures figure showed four hand-picked local errors; v6 shows the four largest ImageReward deficits.

## 7. Candidates for cutting (not cut; your decision)
- tab:phase2_general (G.2, about 0.4 page): duplicates tab:gen_all except ImageReward's third decimal.
- tab:phase2_baselines is only a second label on tab:panel46_deltas (kept because main.tex cites it).
- Table D.4 (published_hrgen) repeats the HRGen block of Table D.3 with more rows; D.3 could drop its HRGen block.
- K (prompts) could move into B as a float page, which would remove the one layout dependency in Section 8.

## 8. Not verified / caveats
- Page breaks will move when other owners change float sizes. The fragile spot is the K/L boundary: L opens with
  Figure L.1 below its intro only if the prompt list ends on its own page. If a later build shows a page under 40%
  full there, adjust the prompt list leading (\fontsize{8}{8.8}) or the K intro. Check with page_fill.py.
- The appendix is 49 pages, 6 more than v5.1: the new Holm and Table-3 tables, the restyled 7 pt tables (no
  \resizebox), four-prompt wide galleries, Table I.1, D.5 and the contents block. Shorter needs cuts from Section 7.
- I did not inspect every image tile at 1:1; the galleries were read at page scale.
- Float-to-reference distance: in E and G the text is much shorter than the tables (about 1.7 vs 7.5 pages in E), so
  some tables sit two to four pages after the paragraph that cites them, always inside their own section.
