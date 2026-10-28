# State: AReA presentation pass (paper owner agent), 2026-10-27 23:20 CT

## Goal
Make the SCA paper better written and presented than AReA (2610.17227) on every axis, within the gold bar.
Deliver: .agent/notes/area_vs_sca_presentation.md (0/1/2 scores + actions), apply actions, 2 render-review
cycles, <=10 main pages, delivery build + fresh unzip, latest_delivery.md, HANDOVER Section 19, commit+push
(plain message, NO trailers). Final report <250 words via SubagentHandback.

## Done in main.tex (uncommitted)
- Intro para 1: new hook (full attention flat field; windows from step 1 draw five Moai) + question at end of para 1.
  Para 2: removed duplicate Moai sentences; OLD QUESTION SENTENCE STILL AT END OF PARA 2 ("We ask whether a head's
  local/global output mixture...") -> replace with a verified gap sentence after relwork agent reports.
- Method: new "Worked example" paragraph after Design choices (256x256 grid, 4,608 keys/branch vs 66,048, 6.06e8 vs
  4.36e9 pairs, 7.2x/8.4x, 18.7 vs 87.3 ms).
- Limitations: leads with late duplication (5 of 11 uncurated, last 20 steps, fix in progress, not used here).
- Captions tightened: Fig 1, 2, 3, 5, 6, Tables 1-3; HRDiT cold-run detail moved to Latency paragraph.
- Figure quality_time moved from main text to Appendix app:budget (figure [!htbp], 0.5 width).
- Build after these (before moving quality_time): main text spilled ~10 lines onto p11 -> must cut more.

## Subagents launched (results arrive as notifications)
- AReA digest (presentation facts of 2610.17227 + project page).
- gen_main_tables.py: Tables 1-4 tabular bodies -> paper/figures/snippets/main_tables/*.tex + headline.tex macros.
  Then replace hand-typed tabulars in main.tex with \input lines (captions stay in main.tex).
- Related-work mechanism facts (AReA, DyPE, SEGA, UltraImage, HRDiT, ResDiT, FreeScale, SCA) with evidence.
- heads_view figure: paper/figures/heads_view.pdf (+src/fig_heads_view.py, results/heads_view/*.csv): (a) two key
  sets on real 4096^2 image, (b) measured reach of 3 heads (calib mass_mean), (c) pi(sigma), (d) pi vs near-field
  share scatter (r=0.771 s2/r16 at bin 8; 0.610 s4/r8). Replaces Figure 4 pi_heads (old figure -> Appendix B for
  split-half r=0.989; update Section 3 refs to panels).

## Update 23:55
- Comparison note written: .agent/notes/area_vs_sca_presentation.md (22 items; open: per-step figure needs traces).
- Related-work table tab:related added (8.7 pt overfull to fix); intro para-2 gap sentence; HRDiT "released code" wording;
  \cmark/\xmark macros. Table agent done: gen_main_tables.py + snippets/main_tables/*.tex + headline.tex (0 mismatches).
- Integration, page fit, reviews, delivery, HANDOVER 19, commit handed to an integrator subagent (single main.tex writer).
- Do NOT commit sca/attn.py, scripts/*, web/, tests/test_dedup.py, results/dup_audit_20261027, duplication_fix note.

## Pending / next
- Per-step figure: needs traces; /scratch/09032/saini_2/sca/web_assets has only method_video/ (no sample_trace.py yet,
  no GPU held). If still absent at the end, record as open action and ask coordinator for trace runs.
- Related-work table (house style, check marks) + shorter related-work prose.
- Page fit <=10, two render-review cycles (delegate reviewers), delivery build, notes, HANDOVER 19, commit, push.

## Tools (scratchpad /scratch/09032/saini_2/tmp/tmp/claude-883311/-work-09032-saini-2-vista-Projects-Exploration/4c03b520-8e32-49ed-8c50-e539347fc179/scratchpad)
qb.sh (quick build -> /scratch/09032/saini_2/sca/paper_builds/quick_area, prints pages/overfull/main-end),
pinfo.py PDF pages N (text blocks), rep.py TARGET EDITS (exact replacements, @@OLD/@@NEW/@@END), sheet.py.
Node i618-062 has 144 cores (OMP_NUM_THREADS=1 makes nproc say 1). Context ~190K at 23:20: be terse, delegate.

## Integrator (single main.tex writer), 2026-10-28
- Task 1 done: tab:related cost column as $O(N^2)$/$O(N)$, tabcolsep 2.8pt, 0 overfull; related-work prose tightened, all 27 citations kept, prose now cites tab:related.
- Task 2 done: four tabulars -> \input{figures/snippets/main_tables/*.tex}, headline.tex in preamble, prose literals -> \hl macros.
  PDF text and word positions identical on all 50 pages (pdfcmp.py, before/after in scratchpad t2_*.pdf).
  gen_main_tables.py: Table 4 (tab:generality) marks \dn{} where the CSV interval is below zero (only sigma_g=0.85 HPS, M_c);
  caption now "orange if significantly worse than the reference". apply_krea_hrgen.py: tab:generality patches dropped
  (rerun leaves main.tex byte-identical). gen_prog_tables.py check_main reads tab_panel46.tex (passes; prog_deltas unchanged).
- Next: task 3 (heads_view figure is final per figure agent), then task 4 page fit (spill was ~25 lines on p11).
- Task 3 done: heads_view.pdf in Section 3 as fig:heads (caption lead + 2 sentences, r=0.77 s2/r16/sigma 0.62 from summary.csv);
  old pi_heads figure moved to Appendix B after \snippet{calib}{} and named in its opening sentence; Section 2.3 and 3 cite fig:heads(c,d);
  opacity cap note in Appendix B "Attention mass." paragraph.
- Task 4 done (quick build): main text ends p10 (y663), repro p11. Prose tightened (setup, Sec 3, Argument 1, related, limitations).
  Structural: Table published_compare moved to Appendix H (only way to recover ~21 lines; claims stay, refs auto-update);
  Figure 6 (paired_area) wrap anchored at the Metrics paragraph; qual figure source moved before Models paragraph (now Figure 5).
  All floats within one page of first ref (firstref.py), no overflow, sentences <=35 (4 flags are sentlen artifacts).
- Next: review cycle 1 (fresh Opus reviewer, no edits), apply, cycle 2, delivery, notes, HANDOVER 19, commit.
- Review 1 applied (2026-10-28): Table 2 caption (no underline claim), Table 5 offsets scoped to image keys (Remark 1),
  appendix openings B/D, compute appendix times now 72/264/196 s from a4k195_main (was 191-prompt run), band-guard run
  labelled, Krea IR "edge of significance" in App I, HPSv2 added to per-layer losses, 4.4 bold lead, header arrows
  (row 1) in Tables 1-2 via gen_main_tables.py. Left open: figure subscripts <6pt (Figs 4, 6, B.2), \codeurl anonymity,
  J.1 lacks HRDiT/SigMa rows, Table H.1 caption ranking (generated by build.py), R.4/R.5 WxH, F.1 1280 prompt ids.
- Done 2026-10-28: reviews 1-2 applied, delivery delivery_3ip7uv4h (51 pp, main ends p10), latest_delivery.md, HANDOVER 19 (mean 1.09, two 0s: sub-6pt subscripts), committed.
