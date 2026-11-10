# SCA live state, 2026-11-10 04:00 CST (paper owner; HANDOVER.md Sections 31 to 42 are the log)

## Goal
Submission-ready SCA paper that sells: measurement-first story (observation, two views, reach emulation, use,
evidence), progressive SCA as the headline, single-pass SCA as the fast mode. Wording rules in force: no
calibration/calibrate/calibrated, no curated/uncurated, no "random" as selection, no em dashes or section symbols.
Main text at most 10 pages (now 9).

## Status and next action
- Paper v5.1 delivered (HANDOVER Section 42; hashes in .agent/notes/latest_delivery.md, build delivery_5revguaf):
  55 pp, main text pages 1 to 9, 0 overfull, 0 undefined, fresh unzip builds to the same pages.
- \resprogabl is ON: Table 2 progressive group, Section 4.2 sentences (\progAblResult), appendix Table L.3
  (tab:prog_abl_all). Result: shared and per-layer weights lose to per-head weights on 10 of 10 per-image metrics
  (IR -0.014 and -0.012), window only on 8; patch FID is lower with shared (-7.3 [-10.3, -5.2]) and per-layer
  (-6.4 [-8.8, -4.8]) and higher with window only (+6.0 [3.9, 9.9]).
- \resra stays OFF for good: the reach-annealed screen met its kill criterion (HANDOVER 40, results/ra_screen).
- NEXT: nothing in flight. Before submitting, anonymize the code URL macro.

## Map
| file | read it when |
|---|---|
| .agent/notes/paper_v5_20261110.md | editing the paper: what moved, figure critiques, switch recipes, layout facts |
| .agent/notes/paper_v4_20261108.md | touching the speed or attention-focus data, the newer-model switches |
| .agent/notes/latest_delivery.md | you need the current PDF/ZIP hashes or build directory |
| paper/figures/src/gen_main_tables.py | any table number or headline macro (nothing is typed by hand) |
| paper/figures/src/fig_observation.py, fig_method_v4.py, fig_speed.py | the three redesigned figures (+ web exports) |
| HANDOVER.md Section 40 | the method owner's reach-annealed hand-off and progressive ablation runs |

## Decisions
- Priority claims stop at: first variance decomposition of reach across heads and steps; first per-head reach measured
  once at the training resolution and reused at any size (UltraImage noted stable head roles first).
- Table 1 keeps only the columns that carry the argument (time, IR, HPS, Pick, M_c, FID_p); bold best per group only;
  every metric and every interval stays in the appendix tables.
- Captions at most three lines with one takeaway; numbers repeat in captions only where a figure must stand alone.
- Speed ratios divide two times of the same kind; the figure draws only the clean-over-clean arrows (AReA, HRDiT).
- Web exports use new stems (observation_*, method_v4_*, speed_v5_*); the web owner decides when the page switches.

## Open questions
- None blocking the paper.
