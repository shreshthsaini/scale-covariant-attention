# SCA live state, 2026-10-29 21:30 CDT (paper rewrite around sca_v2; Opus paper owner, subagent of the SCA lead)

## Goal
Rewrite paper/ around sca_v2 (opening + shrinking windows 128/112/96/80 + matched pi + 1,024-token bridge in the LOCAL
softmax, code sca/attn.py sca_dedup_forward); swap every table to sca_v2 numbers as groups land; deliver build + notes.

## Status and next action
- 23:50 structure committed and pushed (5351655): main text rewritten, agents A1 A2 B C delivered (all reports
  summarized in .agent/notes/paper_v2_rewrite_20261029.md). Main text still ends p. 11 (~0.55 page over).
- Waiter bg task (scratchpad/wait_groups.sh, seen file seen_groups.txt) exits when a group lands; re-arm after each.
- I1 done (Oct 29 late): A2 app tables wired, \pend robust + neutral marks, swap_all order fixed (collect_latency first;
  fig_paired_area unwraps \pend), tab:flops has an SCA row, published-only captions say 'without the schedule',
  Sections 1-6 end p10 (y~698), render review 1 done; 55 pages; dup_i_160/bridge32/ext75 rows in Table 3 + tab:dupfix.
- Open: Tables K.1/K.2 alone on pages; Figure L.1/M.2 snippet captions need 'without the schedule' in their generators;
  grey values in abstract/intro have no caption note; Table K.1 title claim rests on grey values;
  results/.../decision_round5.txt untracked (not committed). NEXT: per landing, swap agent.
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
