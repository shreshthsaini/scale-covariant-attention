# Review-round 1 GPU campaign: live state (campaign owner, rewritten at each checkpoint)

## Goal
Run the five held items (README /scratch/09032/saini_2/sca/taskq/hold_review_round/README.md), land each evaluation and
CPU bootstrap, copy CSVs to results/review1_<name>/ with SHA256SUMS (the eval tasks do this), write
.agent/notes/review_round1/results_item<N>.md, commit+push (only results/, .agent/notes, HANDOVER additions; fetch, push
when origin has not moved, since pull --rebase refuses while the paper agent has unstaged edits), then run
repro/migration/sync_incremental.sh. Finally HANDOVER Section 26 and a report under 300 words.

## Status (2026-11-04 23:58 CST)
- Fleets SCA21 job 1123398 and SCA22 job 1123443 (method agent; 4 gh, 10 h each, same sbatch and queue) PENDING.
  NO MORE SUBMISSIONS (lead, 2026-11-05 00:50): if work cannot fit, tell the lead the remaining GPU-h. a6_ tasks (method
  agent) sort first; never touch them. Controller i615-133 job 1123393 ends 2026-11-06 23:31.
- 140 tasks in taskq/pending; 8 held (evals and CPU boots release themselves). Order 5, 1, (3b gen + 4), 2, rest of 3.
- Release loop PID 2733390; CPU workers PIDs 2733389, 2733391 (nice 19). Setup commit 9a1bb81 pushed.
- Lead told about the item 4 / 8192 rename (SendMessage to main); revert names if the lead says "keep README order".

## Items: done / open
- 5 krea195_dype_sega: open. 1 a4k195_sched_bridge: open. 2 abl46_pair: open. 3 6144, 8192 (CPU eval), flux2: open.
  4 latency: open.

## Decisions
- Pre-approved second fleet already used by the method agent (SCA22). Do not submit; report remaining GPU-h instead.
- Never create taskq/STOP while work remains; taskq_cpu/STOP stops the CPU workers at the end.

## Tools
status: bash /scratch/09032/saini_2/sca/dispatch/review_round1/status.sh; watcher: watch.sh <hours> <jobids> [noidle];
summary of a results dir: python -I <scratchpad>/summarize_item.py <dir> (copy kept in campaign_scripts/).
