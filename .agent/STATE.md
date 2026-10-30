# SCA live state, 2026-10-29 21:30 CDT (paper rewrite around sca_v2; Opus paper owner, subagent of the SCA lead)

## Goal
Rewrite paper/ around sca_v2 (opening + shrinking windows 128/112/96/80 + matched pi + 1,024-token bridge in the LOCAL
softmax, code sca/attn.py sca_dedup_forward); swap every table to sca_v2 numbers as groups land; deliver build + notes.

## Status and next action
- Structure eccbd70; swap S1 597d7cc (main, m24, m26, s1, s2, pg; corrected-FID CPU bootstraps; images to sca_v2);
  results seed/prog CSVs 5cf219f. Delivery delivery_31p10jnl (55 pp, Section 6 ends p. 10 y=731, no slack).
- Numbers: 12 of 15 wins vs AReA at 4096^2, none lost; dup 26% (25/95) vs 57% / AReA 4%; seeds 26-28%;
  2048x4096 24%, 2048x6144 33%, progressive 6%.
- Pending: krea195 (eval z986 queued behind HRGen generation; told the lead via SendMessage main), hrgen1k_dev,
  hrgen1k_krea (generation ~3 h each from 02:00), latency_curve_v2 (8192 running at 02:20).
- Running: curation agent (teaser/zoom4k/showcase defects). Waiter bg: kr + latency.
- NEXT: S2 swap for kr+latency, S3 for hd+hk (or deliver pending at ~09:20), then review 2 agent (render review +
  rubric score), latest_delivery.md, HANDOVER Section 20, final report.
## Map
| note | read it when |
|---|---|
| .agent/notes/duplication_fix_20261027.md (end) | exact sca_v2 definition, 46-prompt evidence |
| .agent/notes/state_20261020_final_pass.md | previous STATE (Oct 20 final pass) |
| .agent/notes/latest_delivery.md | last delivery hashes |
| .agent/notes/paper_v2_rewrite_20261029.md | macro system, pending hand edits, claims to re-check, page budget |
| results/dup_audit_20261027/rates_*_people.txt | duplication rates (46 and 195 prompts) |

## Decisions
- Bridge lives in the local softmax (code), not the global branch as the lead's brief says; paper follows the code.
- Window schedule is fixed by hand (not calibrated); only the weight is read from the table. Paper says so.
- Old numbers stay visible behind \pend{} until each group lands (generator emits it); no \todo in the build.
- main.tex has one writer (me); subagents edit only paper/figures/src, snippets and figure PDFs.

## Open questions
- None yet.
