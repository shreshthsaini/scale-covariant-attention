# Paper v6: appendix (role `appendix`), 2026-11-10

Status: IN PROGRESS (first full rewrite builds; integrating the other owners' figures and tables as they land).
Owner of paper/appendix.tex only. Builds: /scratch/09032/saini_2/sca/paper_builds/v6_appendix_<n>/.

## Requests to other owners (read these)
- text: every v5.1 appendix label is kept, so all your \ref targets still resolve. New subsection labels you can
  cite: app:emu_record, app:emu_head, app:emu_table, app:emu_error, app:zoom, app:kernel, app:cost, app:protocol,
  app:metricdefs, app:intervals, app:timing, app:allmetrics, app:pairs, app:holm, app:abl, app:latency,
  app:prog_protocol, app:prog_results, app:generality, app:gen_seeds, app:gen_krea, app:wide, app:sens, app:budgets,
  app:hardware, app:release, app:limits; table tab:sources (where each item lives in the release).
  - Do not reuse these appendix labels for new main figures: fig:qual_main (zoom4k.pdf, single-pass detail),
    fig:hero (three-aspect-ratio showcase), fig:teaser (six methods on one prompt).
  - Please tell me (in text.md) the labels of the new Figures 6 (gains.pdf) and 7 (scaling.pdf). The appendix
    currently says "expands Figure~\ref{fig:paired_area}" and "Figure~\ref{fig:speed}"; I will switch.
  - Algorithm 5 (alg:prog) and the full Algorithms 3 and 4 live in Appendix C.2 (app:code), as before.
  - Reproducibility statement: Appendix I.2 (app:release, Table tab:sources) now maps every script, table and
    timing file; Appendix D.4 (app:timing) holds the timing protocol; D.3 (app:intervals) the table conventions.
- tablesplots: holm.tex (tab:holm) is wired as Appendix E.3; gen_all.tex (tab:gen_all) as the first table of
  Appendix G. The four interval tables that v5.1 pasted from gen_tables.py stay inline in appendix.tex, cell for
  cell (tab:app_abl, tab:phase2_sensitivity, tab:phase2_budget, tab:app_ps2); if you generate them as
  app_tables/<stem>.tex with the same cells, tell me the stems and I switch to \snippet.

## Not yet done (being worked)
- Integrate methodfigs, visuals and text notes once they exist; final layout pass; full verification.
