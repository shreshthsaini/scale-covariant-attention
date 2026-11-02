# SCA live state, 2026-10-29 21:30 CDT (paper rewrite around sca_v2; Opus paper owner, subagent of the SCA lead)

## Goal
Rewrite paper/ around sca_v2 (opening + shrinking windows 128/112/96/80 + matched pi + 1,024-token bridge in the LOCAL
softmax, code sca/attn.py sca_dedup_forward); swap every table to sca_v2 numbers as groups land; deliver build + notes.

## Status and next action
- DELIVERED 2026-10-30 09:20: commit a51e9ae pushed; build delivery_32xjiemn (53 pp, main 1-10, 0 overfull,
  0 undefined, fresh unzip ok). latest_delivery.md and HANDOVER 20.1 updated. No background tasks left.
- Pending: hrgen1k_dev_sca_v2, hrgen1k_krea_sca_v2 (grey cells). NEXT (any agent): when their eval DONE exists,
  follow the Swap procedure in .agent/notes/paper_v2_rewrite_20261029.md (swap_all.sh + HRGen prose by hand).
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
