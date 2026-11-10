# SCA live state, 2026-11-10 afternoon (paper owner; HANDOVER.md Sections 31 to 42 are the log)

## Goal
Submission-ready SCA paper that sells: measurement-first story (observation, two views, reach emulation, use,
evidence), progressive SCA as the headline, single-pass SCA as the fast mode. Wording rules in force: no
calibration/calibrate/calibrated, no curated/uncurated, no "random" as selection, no em dashes or section symbols.
Main text at most 10 pages (now 9).

## Status and next action
- Paper v6 delivered (tag paper-v6; HANDOVER Section 43; hashes in .agent/notes/latest_delivery.md, build
  delivery_i6ibrtrb): 62 pp, main text pages 1 to 9, 0 overfull, 0 undefined, fresh unzip builds to the same pages.
- v6 is the full presentation revamp the user asked for on 2026-11-10; no result changed. Lead brief and the five
  owners' notes (text, appendix, methodfigs, visuals, tablesplots): .agent/notes/paper_v6/. Figures live in
  paper/figures/v6/ with scripts paper/figures/src/v6_*.py; tables come from gen_main_tables.py, gen_app_tables.py,
  gen_holm.py (never run gen_late_tables.py or gen_prog_tables.py).
- Short review edition also delivered: paper/main_short.pdf, 26 pp, no intervals in tables; build with
  `python paper/build_delivery.py --main main_short.tex --stem short` (HANDOVER 43, appendix.md Section 9). Asked the
  user whether the full edition's appendix interval tables should lose their intervals too; no answer yet.
- Open for the user: AI use statement (marker in main.tex above the reproducibility statement), anonymous code URL.
- NEXT: wait for the user's reaction to v6. To change a figure, edit its v6_ script and rerun it; to change text,
  edit main.tex / appendix.tex; then paper/build_delivery.py, update latest_delivery.md, commit, push, sync.
- GitHub: push after every change (user, 2026-11-10: "keep pushing all to github so that other agent can continue").
  For long multi-worker work run `bash scripts/wip_push.sh --loop 15` (remove
  /scratch/09032/saini_2/sca/paper_builds/wip_push.stop first).
- \resra stays OFF for good (HANDOVER 40).

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
