# SCA live state, 2026-11-02 03:40 CST (review round 1 revision; Opus revision author, subagent of the SCA lead)

## Goal
Apply every "FIX" item of .agent/notes/review_round1/meta_review.md to paper/main.tex and the scripted tables,
build (paper/build_delivery.py), two render-review cycles, delivery with fresh-unzip check, update
.agent/notes/latest_delivery.md and HANDOVER.md Section 24 (rubric self-score), commit, pull --rebase, push.

## Status and next action
- WARNING: controller job 1111046 on i617-044 ends 2026-11-02 04:12; this session dies then. Files on /work persist.
- Done: meta_review.md (all dispositions); deferred specs in /scratch/09032/saini_2/sca/taskq/hold_review_round/README.md;
  decisions 1 to 7 applied, decisions 8 and 9 for Table 1, decision 10 in part; delivery delivery_2duo3frk (55 pp, main 1-10).
  Open FIX items are listed under "NOT YET DONE" in HANDOVER.md Section 24 (start there).
- In progress: main.tex edits in this order: (1) abstract/intro/bullets/conclusion framing (decision 1),
  (2) Section 2.3/2.4 duplication mechanism + 28-step tie (decisions 2, 3), (3) table generators (Krea cells,
  Table 1 groups/Time/Dup header, Table 4 intervals), (4) labels "SCA without the schedule", (5) appendix F.2
  hand-set table, J.1 AReA note, limitations (decision 4), (6) sentences over 35 words, captions, figures 1/4/5/6.
- Check `git log origin/main` and `git diff --stat` to see which steps landed before continuing.

## Map
| note | read it when |
|---|---|
| .agent/notes/review_round1/meta_review.md | dispositions (the work list) |
| .agent/notes/paper_v2_rewrite_20261029.md | macro system, swap_all.sh procedure |
| .agent/notes/state_20261029_paper_v2.md | previous STATE |
| .agent/notes/duplication_fix_20261027.md | sca_v2 definition, 46-prompt evidence |
| results/dup_audit_20261027/review_round1_cpu/pairs.csv | McNemar p for dup comparisons (46, 195) |

## Decisions
- Figure agent owns figures/src/method.tex, method_zoom (D.1), Appendix B calibration figures; commit only own files.
- No GPU, no sbatch; never alter a measured number.
