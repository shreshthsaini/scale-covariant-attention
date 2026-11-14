# SCA live state, 2026-11-10 afternoon (paper owner; HANDOVER.md Sections 31 to 42 are the log)

## Goal
Submission-ready SCA paper that sells: measurement-first story (observation, two views, reach emulation, use,
evidence), progressive SCA as the headline, single-pass SCA as the fast mode. Wording rules in force: no
calibration/calibrate/calibrated, no curated/uncurated, no "random" as selection, no em dashes or section symbols.
Main text at most 10 pages (now 9).

## Status and next action
- Paper: v6.2 delivered in two editions (tag paper-v6.2; .agent/notes/latest_delivery.md). Nothing in flight there.
  Open for the user: AI use statement, anonymous code URL.
- Project page v5 is BUILT in web/ and NOT published live. User, 2026-11-13: "show in claud artifact first": do not
  run web/build/publish_pages.sh until the user has seen the Artifact and agreed. Brief and owners' notes:
  .agent/notes/web_v5/ (BRIEF, design, webdesign, webinteractive, webmedia). Rounds 1 and 2 are superseded.
- Round 3 DONE (2026-11-13 20:10): "One patch, three sizes" motion graphic (web/build/v5_patch3.py; ten files
  assets/video/v5/patch3_<prompt>[_9x16].mp4, five traced prompts) leads "How an image forms" with a five-example
  switcher, the pick-a-patch explorer below it; formation_* carousel moved to gallery.html. Main page 8,515 px,
  964 visible words. CURRENT ARTIFACT (new URL, the first one hit its file limit and is now stale):
  https://claude.ai/artifact/9kjLBJTwgYmvVNhTWrLgad, 151 files from
  /scratch/09032/saini_2/sca/web_builds/v5_artifact/bundle_r3_final (lead's publish copy in the session scratchpad,
  sca_page_r3). Old Artifact (rounds 1 and 2): https://claude.ai/artifact/2bjKpeUBiJMNKzkTmkftH6.
  The session restarted at 19:36 on controller job 1142854 (about two days left).
- Round 4 in flight since 2026-11-14 12:15 (user: "continue"). webmedia: the patch3 reading strokes were nearly
  invisible in the encoded videos; re-render all ten with visible flowing strokes, two sub-beats per stage tying the
  orange/blue flows to the near- and far-looking heads' weights, a 2x magnifier at 2K. webdesign: rule-based
  gallery set (seeded draw of 15, seed 0), person rule on review/ single-pass panels, v5 styling on sub-pages,
  bundle_r4 with changed.json against the published copy; then the lead republishes the same Artifact URL.
- NEXT: wait for the user's reaction to the Artifact; apply their changes (designer notes say how to rebuild), rebuild
  the bundle, republish the Artifact to the same URL; publish live only on the user's word; then HANDOVER 44,
  commit, push, sync.
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
