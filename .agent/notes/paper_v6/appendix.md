# Paper v6: appendix (role `appendix`), 2026-11-10

Status: DONE, round 2 (lead review of build 24) applied. Owner of paper/appendix.tex only. Last verified build:
/scratch/09032/saini_2/sca/paper_builds/v6_appendix_27/main.pdf (62 pp; appendix pp. 13 to 62 = 50 pp; v5.1 had 43).
Renders: v6_appendix_26/pages/ (round 2, pages 21 to 62 at 100 dpi and contact sheets; build 27 differs by three
words), v6_appendix_22/ and v6_appendix_23/pages/ (round 1); switch builds in v6_appendix_switches/. Tools (scratch, purge-eligible):
/scratch/09032/saini_2/tmp/tmp/claude-883311/-work-09032-saini-2-vista-Projects-Exploration/4c03b520-8e32-49ed-8c50-e539347fc179/scratchpad/
(render.py, page_fill.py, pdf_numbers.py, table_numbers.py, check_lost.py, sentences.py, switch_tests.sh).

## 0. Round 2 (lead review of build 24), what changed
1. Sample captions no longer claim more than the page shows. The v6 snippets of the visuals owner keep their
   selection and crop clauses; the appendix owns the claim. \apxcap{n}{lead}{close} + \vsnippetcap{placement}{stem}
   replace the bold lead of a snippet's n-th caption and add a measured closing sentence (macros only, no number
   typed). Leads that asserted detail or sharpness ("keeps the native detail that AReA and UltraImage blur", "is
   sharper than AReA", "keeps finer texture", "adds native detail", "AReA's crops carry a fine hatch", "carries SCA to
   that checkpoint", "carries native detail to FLUX.2-dev") are now descriptive, and the body gives the 195-prompt
   statement with its interval and table: e.g. L.3 "Over all 195 prompts, progressive SCA's MUSIQ over four 1024^2
   crops exceeds AReA's by +3.7 [2.0, 5.4] and UltraImage's by +12.4 [11.5, 13.2], and is indistinguishable from
   HRDiT's (+0.6 [-0.7, 1.8]; Table E.2)". Leads kept because the page shows them: L.2 (full attention loses the scene,
   DyPE/SEGA lattice), L.8 (both keep the layout), E.3 (no-opening subjects shrink; v5.1 wording the lead approved),
   F.2 (progressive keeps HRDiT's large subject, single pass draws it smaller), G.2 and G.4 (artifacts of both
   methods, now named for both), J.1 (largest deficits). The v5.1 fallback snippets keep their captions.
2. crops_matched (fig:crops_matched, new label) is Figure F.1, beside the HRDiT comparison of F.2. Lead: "Where the
   four methods draw the same content, progressive SCA's crop detail is level with HRDiT's." Body: the printed
   per-crop MUSIQ (UltraImage lowest in every row, AReA highest in two) and the 195-prompt intervals. The F.2
   "Against HRDiT" paragraph cites it once. D.5 states its window rule. No appendix sentence described main Figure 6's
   content, so nothing else changed for its new design.
3. fig:scaling stays in the main text (text.md, 11:44: Figure 7 wrapped in Section 4.2), so nothing moved to E.6.
   Latency text points at fig:scaling; fig:speed and fig:emulation are only used through guards. The text owner's
   reproducibility statement already points at D.3, D.4 and Table I.1.
4. tab:phase2_general retired: gen_all now prints ImageReward to three decimals and holds all 121 numbers of
   phase2_general (multiset check: 0 missing; tablesplots.md 11:43 says the same). The label is an alias of
   tab:gen_all (\apxalias, set inside the gen_all float; aux shows both as G.1, p. 42). The HRGen block of Table D.3
   is kept.
5. E: a \FloatBarrier before each subsection, so each subsection's text sits next to its own tables (same page count;
   lowest page fill 0.52). Tried the same in F and G: it added two pages of half-empty float pages, so not adopted.
6. tab:flops: the comment above the float states the timing rows were measured once with no archived file; the caption
   says "measured once on one GH200 (minimum over interleaved runs; no archived timing file) ... these per-layer times
   are indicative"; the lead uses \hlWorkedPairsRatio.
Also: "harness" -> "runner" (three places); the J.1 8192 sentence names the artifacts of both methods as G.4 does.

## 1. New order (page numbers of build 27)
| | section (label) | page | what it holds |
|---|---|---|---|
| | Contents block | 13 | two-column list of sections and subsections with page numbers |
| A | Proofs (app:proofs) | 13 | Lemma 1, Props. 2 and 3, Remark 1, proof of Prop. 1 (text unchanged) |
| B | Reach emulation (app:emulation) | 14 | B.1 what one pass records (statistics, geometry, noise bins), B.2 the head sets the reach (Table B.1 variance split, Fig. B.1, attention mass), B.3 the reach table (Figs. B.2 to B.4), B.4 output error of each rule (Figs. B.5, B.6, no global phase) |
| C | Implementation, pseudocode, hyperparameters (app:impl) | 18 | C.1 operators in detail (Fig. C.1 zoom, design choices, window placement), C.2 pseudocode (Algs. 3 to 5, 8 pt, no wrapped lines), C.3 kernel and numerics, C.4 attention cost (Table C.1), C.5 hyperparameters (Tables C.2, C.3 with grey section bands) |
| D | Evaluation protocol and baselines (app:metrics) | 22 | D.1 protocol and prompts (Table D.1), D.2 metrics, D.3 intervals and table conventions, D.4 timing (runs T1, T2, T3), D.5 sample figures (selection, crop rules, configurations), D.6 how methods change attention (Table D.2), D.7 AReA (Tables D.3, D.4), D.8 other baselines and FLUX.2-dev |
| E | Extended results (app:tables) | 28 | E.1 every metric of Table 1, E.2 headline comparisons, E.3 Holm, E.4 every baseline vs single pass (+ Figs. E.1 plot, E.2 baseline samples), E.5 ablations (Tables E.6, E.7, Fig. E.3), E.6 latency (Table E.8, Fig. E.4) |
| F | Progressive SCA (app:prog) | 38 | F.1 protocol, F.2 against HRDiT and single pass (Table F.1; Fig. F.1 matched crops, F.2 progressive vs single pass, F.3 crop by crop) |
| G | Generality (app:generality) | 41 | G.1 seeds, HRGen-1K, FLUX.2-dev (Table G.1, Fig. G.1), G.2 6144^2 and Krea (Table G.3, Figs. G.2, G.3), G.3 non-square (Table G.4), G.4 8192^2 (Table G.5, Fig. G.4), [G.5 newer models, only with a model switch on] |
| H | Sensitivity and budgets (app:budget) | 47 | H.1 emulation budget and gate, H.2 table at non-square sizes, H.3 steps, guidance, SigMa (Tables H.1 to H.4, Fig. H.1) |
| I | Compute and reproducibility (app:compute) | 50 | I.1 hardware and compute, I.2 release and sources (Table I.1: every script, timing file and result folder) |
| J | Limitations in detail (app:limits) | 51 | J.1 failure cases and scope (Fig. J.1, eight concrete limits), J.2 broader impact |
| K | Emulation prompts (app:prompts) | 53 | the 48 prompts, two columns, 8 pt |
| L | Sample pages (app:galleries) | 54 | Figs. L.1 to L.9, one per page, filled |
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
app:gen_krea, app:wide, app:sens, app:budgets, app:hardware, app:release, app:limits. Table: tab:sources. Figure (round 2): fig:crops_matched (snippet of the
visuals owner). Retired: tab:phase2_general (alias of tab:gen_all, so it still resolves).
Snippet labels now included: tab:holm, tab:gen_all (tablesplots), and the v6 visuals snippets (same labels as v5.1).

## 5. Requests to other owners
- text: (1) Figure 1's proposed caption (visuals) cites fig:showcase_models; that figure exists only with
  \showcasetrue (as in v5.1), so guard the reference or drop it. (2) The appendix uses fig:scaling and fig:emulation
  through a guard (\iflabel); if you rename them, nothing breaks, but tell me. (3) Reproducibility statement: point to
  D.4 (timing), D.5 (sample-figure rules), I.2 / Table I.1 (every script and result file).
- tablesplots: (optional) macros \hlScaAreaMusiqc{,Lo,Hi} (single-pass minus AReA, MUSIQ_c, 195 prompts; +9.7
  [8.2, 11.1] in Table E.3). Figure L.6's caption prints the interval automatically once they exist (\ifdefined);
  until then it says "both MUSIQ scores among them (Table E.3)".
- visuals: the leads of your snippets are replaced in the appendix (see Section 0, item 1); please keep each caption
  starting with \textbf{...} so the replacement finds it.
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
- (done) tab:phase2_general retired as an alias of tab:gen_all.
- tab:phase2_baselines is only a second label on tab:panel46_deltas (kept because main.tex cites it).
- Table D.3's HRGen block: kept (lead's decision).
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

## 9. Short edition (2026-11-10, coordinator request; user: "remove unnecessary tables, keep figures and plots";
## "in tables don't write CI")
Status: DONE. paper/main_short.pdf = 26 pages (main text 1 to 9, reproducibility + references 10 to 12, appendix
13 to 26), 11.6 MiB, 0 overfull, 0 undefined. The full edition is unchanged: main.tex builds 62 pages whose
extracted text equals paper-v6's on every page (PyMuPDF), 0 overfull, 0 undefined.

Build: `tectonic main_short.tex` (short) and `tectonic main.tex` (full), both in paper/. Delivery:
`python paper/build_delivery.py --main main_short.tex --stem short` writes main_short.pdf, sca_overleaf_short.zip
(53 sources, only what the short build reads, plus README.txt naming main_short.tex as main document) and
delivery_receipt_short.json; its standalone ZIP build gives the same 26 pages and identical text. The default call
(no options) is unchanged: tested in a scratch copy of paper/, same 96 ZIP entries as v6, 62 pages, identical text.

Mechanism: main.tex declares `\ifshortversion` (false) unless main_short.tex declared it true first;
`\fullonly{x}` prints x only in the full edition; `\shortalt{full}{short}` picks per edition. One appendix source
with `\ifshortversion ... \else ... \fi` blocks. main_short.tex defines `\fullversionpages` (62), used twice;
build_delivery.py --main main_short.tex stops if it differs from primary_pages in delivery_receipt.json (whose
sha256 must match main.pdf). Floats used by both editions are macros defined before the contents:
\apxablationgrid, \apxcropsmatched, \apxqualitytime, \apximpacttext, \apxprotocoltable (full only now).
Main-text pointers to dropped floats read "full version" (at most one per paragraph, checked on the source).

Generated variant (announced): app_tables/holm_short.tex = panel (a) of holm.tex alone, with its own caption,
written by gen_main_tables.py (new table_holm_short(), added to main()). The generator was rerun: every other
output byte-identical to before, checks passed. The short edition inputs it in place of holm.tex.

Kept: A proofs; B with all six plots (B.1 now "Noise bins"; Statistics and Geometry paragraphs full only);
C.1 operators figure + window placement (pseudocode, kernel, cost, both hyperparameter tables full only);
D "Evaluation protocol in brief" (baselines and how run, intervals and Holm, timing; data and metrics point to
Section 4; protocol table full only); E: Table E.1 (all metrics, no intervals), Table E.2 (Holm panel (a): deltas,
p, p_Holm; no intervals), Figures E.1 crops matched, E.2 baseline deltas, E.3 latency, E.4 quality-time;
F limitations + broader impact, one paragraph each; G samples: G.1 progressive gallery, G.2 all methods,
G.3 aspect ratios, G.4 single-pass vs AReA, G.5 ablation grid (+ newer-model figures only when switched on).
Dropped (full version only): 24 of the 26 appendix tables, namely B.1 (variance split), C.1 to C.3 (flops,
hand-set, hyperparameters), D.1 to D.4 (protocol, related settings, published comparisons), E.2, E.3, E.5 to E.8,
F.1, G.1 to G.4, H.1 to H.4, I.1 (sources); appendices F to I and K (prompts); Algorithms 3 to 5; the duplicate
galleries (prog16, 4096 page 2, 2048x4096, 2048x6144, baselines); the clean-time paragraphs.

No-interval check (short): no table cell prints [lo, hi] (PyMuPDF scan of lines set at <= 8.1 pt); intervals
remain only in prose and figure captions. Full-edition tables that print intervals in cells: E.2 prog_all,
E.3 app_4k, E.5 panel46_deltas, E.6 app_abl, E.7 prog_abl_all, F.1 prog_deltas, G.1 gen_all, G.2 krea_6144,
G.3 app_wide, H.1 phase2_sensitivity, H.2 app_ps2, H.4 phase2_budget; E.4 holm panel (b) has a column headed
"95% interval" that holds win/loss counts, not intervals.

Found, not fixed (would change the full edition): holm.tex panel (b) labels its groups "Table 3: SCA against AReA,
other seeds..." and "Table 2: per-head weights...", but tab:generality is Table 2 and tab:ablation Table 3; its
caption says "Tables 1 to 2" (\ref{tab:generality}) where it means 1 to 3. Source: HOLM_GROUPS and the caption
in gen_main_tables.py table_holm().

Switch builds of the short edition (scratch v6_short_7/switch, v6_short_8/switch): progressive ablation off 26 pp;
newer models + showcase on 28 pp (one showcase page, one page with the FLUX.2 and Qwen figures), 0 undefined;
v6 visual snippets absent 25 pp, 0 undefined (crop-figure sentences guarded with \ifvsnip).
Renders of the delivered PDF: /scratch/09032/saini_2/sca/paper_builds/v6_short_final/pages/ (110 dpi + sheets).
Page 12 (end of references) is about half empty in both editions because the appendix starts on a new page.
