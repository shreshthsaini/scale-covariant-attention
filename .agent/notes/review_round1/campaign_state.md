# Review-round 1 GPU campaign: live state (campaign owner, rewritten at each checkpoint)

## Goal
Run the five held items (README /scratch/09032/saini_2/sca/taskq/hold_review_round/README.md), land each evaluation and
CPU bootstrap, copy CSVs to results/review1_<name>/ with SHA256SUMS (the eval tasks do this), write
.agent/notes/review_round1/results_item<N>.md, commit+push (only results/, .agent/notes, HANDOVER additions; fetch, push
when origin has not moved, since pull --rebase refuses while the paper agent has unstaged edits), then run
repro/migration/sync_incremental.sh. Finally HANDOVER Section 26 and a report under 300 words.

## Status (2026-11-06 00:25 CST)
- Fleets SCA21 job 1123398 and SCA22 job 1123443 (method agent; 4 gh, 10 h each, same sbatch and queue) PENDING.
  NO MORE SUBMISSIONS (lead, 2026-11-05 00:50): if work cannot fit, tell the lead the remaining GPU-h. a6_ tasks (method
  agent) sort first; never touch them. Controller i615-133 job 1123393 ends 2026-11-06 23:31.
- 140 tasks in taskq/pending; 8 held (evals and CPU boots release themselves). Order 5, 1, (3b gen + 4), 2, rest of 3.
- Controller now i615-091 (job 1125760, ends 11-08 04:59): release loop v2 PID 1692172, CPU workers 1692173, 1692174.
  Old i615-133 processes (632229, 2733389, 2733391) stopped. Commits 9a1bb81, 04aace7.
- Short fleets SCA23 1124593 (gh-dev 4x2 h, est. start 18:20) and SCA24 1124594/5 (gh 1x3 h) share the queue. Walltime
  guards on r4 (now b0_r4_latency_sweep.sh, 23400 s), r0_k5 (7200), r1_z (6300), r2_z (2700), r3_z_6144 (6300),
  r3_z_flux2 (3600): they park in hold and exit 3 when their job is short; failed/ entries for them are bounces.
- Lead told about the item 4 / 8192 rename (SendMessage to main); revert names if the lead says "keep README order".
- 00:25 11-06: 0 of 140 tasks run; SCA21/22 pending 25 h; short fleets ran only a6_ (64 done, mean 580 s, 33 pending).
  Lead informed (remaining 33.5 GPU-h; options a: wait, b: prefix r1_g with a5_ to interleave, c: new controller for CPU).
  Controller ends 23:31 today; 8192 CPU eval must start by about 11:00 to fit there.

## Items: done / open
- 5 krea195_dype_sega: open. 1 a4k195_sched_bridge: open. 2 abl46_pair: open. 3 6144, 8192 (CPU eval), flux2: open.
  4 latency: open.

## Decisions
- Pre-approved second fleet already used by the method agent (SCA22). Do not submit; report remaining GPU-h instead.
- Never create taskq/STOP while work remains; taskq_cpu/STOP stops the CPU workers at the end.

## Tools
status: bash /scratch/09032/saini_2/sca/dispatch/review_round1/status.sh; watcher: watch.sh <hours> <jobids> [noidle];
summary of a results dir: python -I <scratchpad>/summarize_item.py <dir> (copy kept in campaign_scripts/).

## Lead decisions 2026-11-06 00:30
- GPU order (a): a6_ stays first, item 1 right after it; no renames.
- Controller (c): new 48 h controller job 1125760 (gg, cntrl-cmd, est. start 07:42). When RUNNING: one
  `ssh <node> bash /scratch/09032/saini_2/sca/dispatch/review_round1/start_on_controller.sh 1125760`, then SIGTERM the old
  release loop 632229 and CPU workers 2733389, 2733391 on i615-133 (verify with ps first), record new PIDs in HANDOVER.
  Leave nothing on i615-133 that dies at 23:31 without a successor. Never submit.
