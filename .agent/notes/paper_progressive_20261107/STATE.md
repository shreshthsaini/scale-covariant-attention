# Paper owner state: progressive-SCA rewrite (2026-11-07, paper/ and web/ owner)

## Goal
Rewrite paper/ and web/ with progressive SCA (HRDiT's three-stage schedule + sca_v2 attention) as the headline
configuration and single-pass SCA as the fast mode; framing per positioning_20261105/recommendation.md; no
duplication content anywhere except the one limitations sentence the user dictated. Two render-review cycles,
<= 10 main pages, 0 overfull, 0 undefined, delivery build with fresh unzip, latest_delivery.md, HANDOVER Section 31,
web rebuilt, commits after each stage.

## Status and next action
- DONE (commits 39b9000, 067afd0, pushed): main.tex/appendix.tex split; main text rewritten (title, abstract, intro,
  bullets, method incl. progressive subsection + Algorithm 3, Section 3 variance-split lead, experiments, related
  work, limitations with the dictated sentence, conclusion); gen_main_tables.py restructured (Table 1 progressive
  headline, colours vs progressive, no Dup., generality with pending progressive rows + PickScore, appendix tables
  prog_all, variance_split, tab_panel46 without multi-stage rows); figures concept.pdf, paired_area.pdf (both modes vs
  AReA), method.pdf lane (d); clean latency sweep 3 folded in. Build: 55 pp, main text ends p10, 0 overfull.
- WAITING: appendix agent (appendix.tex: remove duplication, app:prog, tab:panel46/fig:teaser floats,
  variance_split + prog_all inputs, all_metrics prog row); figure agent (hero_prog.pdf, zoom4k with prog column,
  gallery_prog snippet); web agent (web/). Undefined refs until appendix lands: fig:dup, sec:dup, sec:rootcause,
  fig:teaser, tab:panel46, tab:prog_all, tab:variance_split.
- NEXT: integrate agents; switch hero fallback off; point hero caption to fig:gallery_prog; update zoom caption;
  render-review cycle 1 (main by me, appendix by 2 reviewer subagents), fixes, cycle 2; delivery build; notes;
  HANDOVER 31; web commit; remove legacy hlDup* macros once no source uses them.

## Results status (watcher: scratchpad watch_results.sh, background)
- LANDED: prog195_vs_baselines per-image pairs; prog195_pairs/cfid_vs_{area,sca_v2}; prog_latency (sweep 3).
- PENDING: prog195_vs_baselines/cfid (set-level vs other baselines: Table 1 FID colours, prog_all), prog195_s1/s2
  (Table 3 seeds), HRGen-1K prog (Table 3), Krea: unsupported (single-pass only).

## Decisions
- Table 1 colours every row against progressive SCA; ranks within group; HRDiT and progressive clean times from sweep 3.
- Speed claims use clean sweep 3 (4.1x AReA, 46% of HRDiT); campaign means only in Table 1's camp. column.
- Commit only my files while agents work; no autostash pulls (agents edit the shared tree).
