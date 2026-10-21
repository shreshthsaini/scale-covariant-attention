# SCA live state, 2026-10-20 21:40 CDT (final paper pass, Opus subagent of the SCA coordinator)

## Goal
Final paper pass: (A) curate qualitative figure samples (clean SCA, fair comparisons, captions say curated/random),
(B) fold Krea-195 and HRGen-1K (1000) into Tables 2/4 and appendix, (C) render review, delivery build, commit, push.

## Status and next action
- DONE 2026-10-20 23:00: Parts A, B, C. Delivery /scratch/09032/saini_2/sca/paper_builds/delivery_9un4nd1w (49 pp.,
  main 1-10, 0 overfull, 0 undefined, standalone pass); notes/latest_delivery.md; HANDOVER Section 17.
- Open (HANDOVER 17): p27 Table J.2 alone (61%); \codeurl public; calibration prompt count 48 vs 32+16 wording;
  2.85x/2.99x vs 2.84/2.98 rounding in Figures 8/D.2; R.7 column names. All evals done; create taskq/STOP when no
  more GPU work is planned.
- NEXT: nothing in flight from this pass.

## Map
| note | read it when |
|---|---|
| HANDOVER.md Sections 13-16 | project state, protocol, rubric |
| .agent/notes/latest_delivery.md | current PDF/ZIP hashes and build dir |
| .agent/notes/eval_protocol_survey.md Section 5 | the 13 protocol edits to main.tex |
| .agent/notes/figure_curation_20261020.md | which prompt ids each qualitative figure uses and why |
| .agent/notes/state_20261018_snapshot.md | the old (stale) STATE.md, history only |

## Decisions
- Random galleries keep their random draws (user: must stay random and say so); only curated figures get swaps.
- Subagents never edit main.tex or commit; the lead applies caption edits, so main.tex has one writer.

## Open questions
- None yet.
