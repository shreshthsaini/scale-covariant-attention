# Paper owner state: progressive-SCA rewrite (2026-11-07, paper/ and web/ owner)

## Goal
Paper and web with progressive SCA as the headline; all binding user decisions applied; final delivery built.
Remaining: fold in progressive HRGen-1K when results/prog_hrgen1k lands (about 21:30), rebuild, refresh web, hand back.

## Status and next action
- DONE, pushed: final delivery /scratch/09032/saini_2/sca/paper_builds/delivery_fwhllii_ (commit c9dd2e9), 54 pp,
  main text 1-10, 0 overfull, 0 undefined, fresh unzip OK; HANDOVER 31; latest_delivery.md; web final pass (commit
  before c9dd2e9); lead messaged with the path.
- WAITING: watcher (scratchpad watch_results.sh, background) for results/prog_hrgen1k.
- ON LANDING: rerun paper/figures/src/gen_main_tables.py (Table 3 row and seed macros read
  prog_hrgen1k/sca_v2_prog_vs_area.csv; add an hlProgHrgen... sentence); edit main.tex Section 4.4 "HRGen-1K is
  pending" and the Table 3 caption key result; rerun build_delivery.py; update latest_delivery.md and HANDOVER 31
  PENDING line; ask nothing of the web agent unless time permits (or edit web/build/build_numbers.py keys prog.later.*).

## Decisions
- Table 1 colours every row against progressive SCA; HRDiT/progressive clean times from sweep 3 (half-up rounding).
- Commit only my files while agents work; no autostash pulls.
