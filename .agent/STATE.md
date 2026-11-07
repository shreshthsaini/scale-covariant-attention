# SCA live state, 2026-11-07 15:10 CST (round 7 method owner, afternoon; HANDOVER.md Section 30 is the log)

## Goal
A single-pass SCA change that meets the relaxed bar: human-corrected person/animal duplication at or below AReA's on
the same prompts, no significant loss vs sca_v2 on IR, HPSv2.1 and PickScore, nothing else below AReA, faster than
AReA. Decisions are made on the 46 and the 195; copyprone21 is the screen (its bias is about 1 prompt, s1 = 18/21).

## Status and next action
- Closed by measurement:
  - wide windows: hyb0 at 17/21, with core losses on the 46.
  - f13: q13i glitches.
  - window prompt content: tcg dash strokes.
  - count-corrected global branch: gc1, gch, gcm and gc2 all lose PickScore 0.12 to 0.22 on the 46, with dash
    strokes at G >= 1.
  - extended pool-2 opening: g2x2 15.4% census with no core loss; g2x4 PickScore -0.09; g2x8 7.7% census but 10
    losses.
- Progressive SCA fails the relaxed bar on patch FID/KID against AReA, with census duplication 10.5% against 5.3%.
- Still running and unread: area_scaOpen and area_scaText (ladder rungs; label them as pQ35/pQ36 with
  r7/label_people.sh), fg on copyprone21 (a0_K0b390*) and the 46 (a0_K0b440*, eval a0_K0b4490eval_r7_fg), and prog14
  on the 46 (a0_K0b48*0, eval a0_K0b4890eval_r7_prog14).

## In flight (taskq; SCA21 to about 16:58, SCA22 to about 23:10)
- The spool after the r7 items: q13n eval, the ladder rest (scaB, area_fixed, scaE, scaC), r6_ffl t013, then the
  review round r1_*.
- Background on i615-091: auto_triage.sh PID 2818058 (exits by itself after triaging fg and the area rungs, or at
  8 h). Check it with ps -o and kill it if it is still alive at hand-back.
- Staged, not enqueued: taskq/hold_prog_headline (progressive seeds 1 and 2), hold_r7_tc46, hold_r7_q13i46.
  Parked: hold_r7_parked (hyb3, hyb4, q13 family, tc, gctv, gc1pb, gc1pb55, gcm2).

## Map
| file | what |
|---|---|
| HANDOVER.md Section 30 | decisions log |
| .agent/notes/duplication_fix_round7.md "Afternoon" | details and verdict tables |
| results/dup_audit_20261027/round7_frontier_46_afternoon.csv | r7_decide rows (hyb0, gc1, gcm, gch, g2x2, g2x4, g2x8) |
| results/dup_audit_20261027/copyprone21_counts.txt | hand counts per variant (scripts/r7_c21_table.py) |
| results/prog195_pairs/ | progressive SCA vs AReA/sca_v2 at 195 (paired, set-level, decide.txt) |
| /scratch/09032/saini_2/sca/r7/{cpu_det_c21.sh,auto_triage.sh,preview/make_preview.py} | triage and preview tools |

## Decisions
- Triage with the CPU detector, then blind labels only for promising screens, the ladder rungs, and 46-prompt
  candidates that keep the core metrics.
- New operator flags, all default off and tested (tests/test_globcount.py): --glob-count, --glob-count-lo,
  --glob-count-map, --qmix tc. Frozen trees: code/d440b71, a6f93ee, 86a4f2f, 023918f.
- Plain commit messages, no trailers; no em dashes; eval tags in the 0a0 style.
