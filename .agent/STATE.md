# SCA live state, 2026-11-10 10:25 CST (paper owner; HANDOVER.md Sections 31 to 42 are the log)

## Goal
Submission-ready SCA paper that sells: measurement-first story (observation, two views, reach emulation, use,
evidence), progressive SCA as the headline, single-pass SCA as the fast mode. Wording rules in force: no
calibration/calibrate/calibrated, no curated/uncurated, no "random" as selection, no em dashes or section symbols.
Main text at most 10 pages (now 9).

## Status and next action
- v5.1 is the last delivered paper (tag paper-v5.1, HANDOVER 42, .agent/notes/latest_delivery.md).
- IN FLIGHT since 2026-11-10 10:20: version 6, a full presentation revamp the user asked for ("best paper at ICLR":
  all text, tables, figures, method and qualitative design). Lead brief: .agent/notes/paper_v6/BRIEF.md (story spine,
  figure stems under paper/figures/v6/, design system, hard rules, one writer per file).
- Five Opus workers (background agents of the lead session), each with a notes file in .agent/notes/paper_v6/:
  text (main.tex), appendix (appendix.tex), methodfigs (observation, method, emulation figures, appendix zoom),
  visuals (hero, qualitative, all sample pages), tablesplots (generators, Tables 1 to 3, gains and scaling plots,
  Holm-adjusted significance). Workers do not commit; the lead commits.
- NEXT: when workers report, read their notes, have text and appendix integrate the new figures and captions,
  build, review every page against the gold-standard rubric (lead reads the renders), send corrections, repeat until
  no rubric item scores 0. Then paper/build_delivery.py, latest_delivery.md, HANDOVER Section 43, commit, push,
  sync, and send the user a reading PDF under 30 MiB (PyMuPDF rewrite_images, see share_20261110).
- GitHub checkpoints: `bash scripts/wip_push.sh --loop 15` runs in the lead session (commits paper sources, figures,
  notes and the handover to origin/main every 15 minutes; stop with `touch
  /scratch/09032/saini_2/sca/paper_builds/wip_push.stop`). User, 2026-11-10: keep pushing everything so another agent
  can continue. A new agent starts from origin/main, this file, and .agent/notes/paper_v6/.
- If the session restarts mid-way: the workers die with it. Read each notes file for progress, check
  paper/figures/v6/ and git status, and relaunch only the unfinished roles with the same brief.
- \resra stays OFF for good (HANDOVER 40). Before submitting, anonymize the code URL macro.

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
