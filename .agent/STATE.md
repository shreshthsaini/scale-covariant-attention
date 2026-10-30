# SCA live state, 2026-10-29 21:30 CDT (paper rewrite around sca_v2; Opus paper owner, subagent of the SCA lead)

## Goal
Rewrite paper/ around sca_v2 (opening + shrinking windows 128/112/96/80 + matched pi + 1,024-token bridge in the LOCAL
softmax, code sca/attn.py sca_dedup_forward); swap every table to sca_v2 numbers as groups land; deliver build + notes.

## Status and next action
- Structure stage DONE: commit eccbd70 pushed (delivery_lxyo4jjv, 55 pp, Section 6 ends p. 10).
- 01:36 m4k + m24 + m26 landed. Full-set duplication for sca_v2: 25/95 = 26% [18.5, 36.0] (published 57%, AReA 4%);
  46-prompt tuning subset said 15%. vs AReA at 4096^2: 11 of 15 wins, none lost, IR +0.33 [0.25, 0.41],
  FID -5.6 [-9.9, -0.8] (no reference bootstrap yet). 2048x6144: seven of 15 wins. Speed ratio waits for latency.
- Swap agent S1 running (m4k, m24, m26, plus s1/s2/pg if they land): swap procedure in the notes file, commit+push.
- NEXT: re-arm waiter for s1 s2 pg kr hd hk latency; spawn S2 for the remaining batch; final render review 2 +
  delivery + latest_delivery.md + HANDOVER Section 20 + report (12 h mark about 09:20 CDT Oct 30).
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
