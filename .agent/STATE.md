# SCA live state, 2026-10-29 21:30 CDT (paper rewrite around sca_v2; Opus paper owner, subagent of the SCA lead)

## Goal
Rewrite paper/ around sca_v2 (opening + shrinking windows 128/112/96/80 + matched pi + 1,024-token bridge in the LOCAL
softmax, code sca/attn.py sca_dedup_forward); swap every table to sca_v2 numbers as groups land; deliver build + notes.

## Status and next action
- Commits: structure eccbd70; S1 597d7cc; results 5cf219f; curation 548520f; review 2 64a0a9d; S2 58eb7e1
  (delivery_g2pm1d_s, 53 pp, Section 6 ends p. 10, one line spare).
- Numbers: vs AReA at 4096^2 12 of 15 wins, none lost, 2.63x faster (77.1 vs 202.4 s), 3.07x at 8192^2, b = 0.99,
  overhead 8% back to back; Krea 8 of 15 wins, loses patch FID only; dup 26% vs 57% / AReA 4% (subset 15%).
  Rubric (review 2): mean 1.23, no 0; AReA verdict: SCA leads 19 of 22.
- 03:55 gallery agent running (Krea gallery to sca_v2, caption trims). hd generation ~05:15, eval z987 may sort
  behind hk generation (x_d98) unless the lead prefixes it a000_; hk generation ~09:00.
- NEXT: S3 when hd lands (or at ~09:20 with hd/hk pending): swap, final check, delivery, latest_delivery.md,
  HANDOVER Section 20, report.
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
