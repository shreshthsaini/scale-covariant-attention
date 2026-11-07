# SCA live state, 2026-11-07 13:40 CST (round 7 method owner, afternoon; HANDOVER.md Section 30 is the log)

## Goal
A single-pass SCA change that meets the relaxed bar: human-corrected person/animal duplication at or below AReA's on
the same prompts, no significant loss vs sca_v2 on IR, HPSv2.1 and PickScore, nothing else below AReA, faster than
AReA. Decisions are made on the 46 and the 195; copyprone21 (21 copy-prone prompts) is the screen.

## Status and next action
- Closed today: the wide-window family (hyb0 17/21 by hand, core losses on the 46), f13 (q13i glitches), the window
  prompt-content route (tcg: dash strokes, lost diver gear), and the extended pool-2 opening, which is still pending
  (g2x*). Progressive SCA fails the relaxed bar on patch FID/KID against AReA, and its census duplication is 10.5%
  against AReA's 5.3%.
- Best clean lever: --glob-count 1 (count-corrected global branch, code d440b71/86a4f2f). gc1 on copyprone21 has 11
  detector flags (sca_v2 21) at no cost. Blind labels pQ32 are running, and its 46-prompt eval is running
  (r7_decide waiter).
- Next: label count pQ32; r7_decide gc1 46; triage gc1a, gch, gc2, gc1pb, gc1pb55 (r7/cpu_det/auto_triage.log);
  then g2x2/g2x4/g2x8, fg (c21 + 46), the area_scaOpen and area_scaText rungs (label them), prog14 (46).

## In flight (taskq; 8 GPUs, SCA21 to about 16:58, SCA22 to about 23:10)
- Spool head: gc2, gc1pb, gc1pb55, area_scaOpen, g2x2 c21 + 46 (eval K0b37), g2x4, g2x8, fg c21, area_scaText, fg 46
  (eval a0_K0b4490eval_r7_fg), prog14 46 (eval a0_K0b4890eval), q13n eval, ladder rest, then the review round.
- Background on i615-091: auto_triage.sh PIDs 2818058 and 2819325. Kill them at hand-back after checking with ps -o.
- Staged, not enqueued: taskq/hold_prog_headline (progressive seeds 1 and 2), hold_r7_tc46 (tcg 46),
  hold_r7_q13i46.

## Map
| file | what |
|---|---|
| HANDOVER.md Section 30 | decisions log (every decision under 200 words) |
| .agent/notes/duplication_fix_round7.md "Afternoon" | details |
| results/dup_audit_20261027/copyprone21_counts.txt | hand counts per variant (scripts/r7_c21_table.py) |
| results/prog195_pairs/ | progressive SCA vs AReA/sca_v2 (paired, set-level, decide.txt) |
| /scratch/09032/saini_2/sca/r7/{cpu_det_c21.sh,auto_triage.sh,preview/make_preview.py,label_people.sh} | tools |

## Decisions
- Screens are triaged with the CPU detector first. Blind labels go only to screens with about 6 or fewer flags, plus
  the ladder rungs and the best clean candidate. The detector undercounts wide-window variants by about 5.
- The window branch's prompt content is not a clean lever (tv5, q13n, tcg); the global branch's prompt read is.
- Plain commit messages, no trailers; pull --rebase before push; no em dashes; eval tags in the 0a0 style, since the
  spool sorts under a collation that ignores underscores.
