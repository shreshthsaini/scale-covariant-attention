# SCA live state, 2026-11-07 17:25 CST (method owner hand-back; HANDOVER.md Section 30 is the log)

## Goal
User decision (lead relay, 15:40): progressive SCA (sca_v2_prog) is the headline configuration and the project is
finishing for submission. The duplication work (single-pass variants) is paused in taskq/hold_paused_20261107.

## Status and next action
- Done and pushed:
  - (1) results/prog195_vs_baselines: paired per-image against 7 baselines, detector rates, corrected set-level
    with intervals, IS table.
  - (4) results/prog_latency: 49.15 s, against HRDiT's 106.97 s and AReA's 202.36 s.
  - (2) seed 1: results/prog195_s1. Its set-level intervals come from eval/prog195_s1/cpu_ci.sh on CPU, which writes
    results/prog195_s1/corrected_fidp_ci when it finishes; commit them.
- Done since: (2) seed 2 (results/prog195_s2 with corrected_fidp_ci), and seed 1's intervals.
- Running: (3) HRGen-1K, 40 generation tasks 0a2_pghk_* on SCA22 (about 22:55 end), then the self-releasing
  0a0_eval_prog_hrgen1k writes results/prog_hrgen1k. Commit it when eval/prog_hrgen1k/DONE exists (about
  21:30). If SCA22 ends first, the worker requeues the eval for the next allocation.
- Krea is not run: scripts/sample_progressive.py supports --model dev only.

## In flight
- taskq: the 0a* progressive tasks lead the spool; the review round (r1_/r2_/r3_) follows.
- Nothing of the method owner's is left running on i615-091.

## Map
| file | what |
|---|---|
| HANDOVER.md Section 30 | decisions and landings |
| slurm/prog_headline_tasks.py | generator of the progressive submission tasks |
| results/prog195_vs_baselines, prog_latency, prog195_s1 (s2 and prog_hrgen1k pending) | submission results |
| .agent/notes/duplication_fix_round7.md | round 7 log and verdict tables |
| taskq/hold_paused_20261107 | paused duplication tasks (README) |

## Decisions
- Item (1) is computed on CPU from cached evaluations (one evaluator; AReA's values match across evals) rather than
  by a GPU re-run.
- Set-level intervals use 1000 draws (the paper's existing protocol). Per-image paired bootstraps use 2000 draws,
  seed 0.
