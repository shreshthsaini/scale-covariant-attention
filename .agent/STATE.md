# SCA live state, 2026-10-29 21:30 CDT (paper rewrite around sca_v2; Opus paper owner, subagent of the SCA lead)

## Goal
Rewrite paper/ around sca_v2 (opening + shrinking windows 128/112/96/80 + matched pi + 1,024-token bridge in the LOCAL
softmax, code sca/attn.py sca_dedup_forward); swap every table to sca_v2 numbers as groups land; deliver build + notes.

## Status and next action
- Commits: structure eccbd70; S1 swap 597d7cc; results 5cf219f; curation 548520f; review 2 64a0a9d
  (delivery_d2_auubg, 54 pp, Section 6 ends p. 10 y=731, rubric mean 1.23 no 0, AReA: SCA leads 19 of 22).
- Numbers: 12 of 15 wins vs AReA at 4096^2, none lost; dup 26% (25/95) vs 57% / AReA 4%; tuning subset 15%.
- 03:30 S2 swap running (krea195 + latency_curve_v2 landed). Waiter bg for hrgen1k_dev DONE.
- Pending: hrgen1k_dev (generation ~05:30, eval after), hrgen1k_krea (generation ~09:00): deliver at ~09:20 with
  pending marks if hk has not landed.
- NEXT: S3 for hd (and hk if landed), final light check, latest_delivery.md, HANDOVER Section 20, report.
## Map
| note | read it when |
|---|---|
| .agent/notes/duplication_fix_20261027.md (end) | exact sca_v2 definition, 46-prompt evidence |
| .agent/notes/state_20261020_final_pass.md | previous STATE (Oct 20 final pass) |
| .agent/notes/latest_delivery.md | last delivery hashes |
| .agent/notes/rubric_20261030.md | review 2 scorecard (22 items) and AReA verdict |
| .agent/notes/paper_v2_rewrite_20261029.md | macro system, pending hand edits, claims to re-check, page budget |
| results/dup_audit_20261027/rates_*_people.txt | duplication rates (46 and 195 prompts) |

## Decisions
- Bridge lives in the local softmax (code), not the global branch as the lead's brief says; paper follows the code.
- Window schedule is fixed by hand (not calibrated); only the weight is read from the table. Paper says so.
- Old numbers stay visible behind \pend{} until each group lands (generator emits it); no \todo in the build.
- main.tex has one writer (me); subagents edit only paper/figures/src, snippets and figure PDFs.

## Open questions
- None yet.
