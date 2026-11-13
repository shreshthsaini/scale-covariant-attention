# SCA live state, 2026-11-10 afternoon (paper owner; HANDOVER.md Sections 31 to 42 are the log)

## Goal
Submission-ready SCA paper that sells: measurement-first story (observation, two views, reach emulation, use,
evidence), progressive SCA as the headline, single-pass SCA as the fast mode. Wording rules in force: no
calibration/calibrate/calibrated, no curated/uncurated, no "random" as selection, no em dashes or section symbols.
Main text at most 10 pages (now 9).

## Status and next action
- Paper: v6.2 delivered in two editions (tag paper-v6.2; .agent/notes/latest_delivery.md). Nothing in flight there.
  Open for the user: AI use statement, anonymous code URL.
- IN FLIGHT since 2026-11-13 16:15: project page v5. User: "check the AReA webpage... our webpage should beat that
  in style and quality and examples and video and gif etc." Lead brief with the audit of both pages, acceptance
  list, page plan, component and video lists, shared contract: .agent/notes/web_v5/BRIEF.md. Captures of both pages:
  /scratch/09032/saini_2/sca/web_builds/v6_audit/.
- Three Opus workers (background agents of the lead session), notes in .agent/notes/web_v5/: webdesign (index.html,
  site.css, site.js, sub-pages, numbers in line with paper v6.2), webinteractive (components percentiles, heads,
  emulate, table, speed under assets/js/c_*.js with demos in web/dev/), webmedia (videos and loops under
  assets/video/v5/ with manifest.json). Workers do not commit or publish.
- User, 2026-11-13: "show in claud artifact first". Do NOT run publish_pages.sh until the user has seen the page as
  a Claude Artifact and agreed. webdesign builds the artifact bundle (web/build/v5_artifact_bundle.py ->
  /scratch/09032/saini_2/sca/web_builds/v5_artifact/bundle/, page.html + files.json chunks of at most 240 files /
  55 MB); the lead publishes it with the Artifact tool (first chunk with the page, then further chunks to the same
  url). webmedia is finished (18 videos in web/assets/video/v5/, manifest.json).
- NEXT: when they report, the lead reads screenshots, sends corrections, has webdesign integrate components and
  videos, runs web/build/check_page.py, then publishes with `bash web/build/publish_pages.sh` and checks the live
  URL https://shreshthsaini.github.io/scale-covariant-attention/. Then HANDOVER Section 44, commit, push, sync.
- The controller node's allocation ends about 22:10 on 2026-11-13. If the session dies: workers die with it; read
  the notes files, check web/ and git status, relaunch only unfinished roles with the same brief.
- GitHub: `bash scripts/wip_push.sh --loop 15` runs in the lead session (now also covers web/; stop with
  `touch /scratch/09032/saini_2/sca/paper_builds/wip_push.stop`).
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
