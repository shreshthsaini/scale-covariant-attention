# Unattended project continuation

The user requested continuation while away. `continuation.sbatch` runs one CPU
controller for the entire SCA campaign, including environment validation, both
CPU bootstraps, paper integration and delivery. This is a working batch pipeline;
it does not hold a node solely to poll a queue.

`continuation.py` starts noninteractive Codex with the existing authenticated
launcher, account configuration, hooks and rules. The controller uses GPT-6.1 Sol
at high effort with a 200K context window and a 170K compaction threshold. Each
completed work unit checkpoints on disk before another fresh-context unit.
The interface follows [OpenAI's noninteractive documentation](https://developers.openai.com/codex/noninteractive).

Before starting agent work, the controller submits a successor dependent on its
own job ending, including timeout or node failure. If useful CPU work is exhausted,
it changes the successor to wake when a pending SCA GPU job starts or ends, or
when a running GPU job ends, then exits. No watcher runs on a login node. Slurm
dependencies consume no nodes while pending. A single-owner lock prevents two
controllers from writing concurrently during the handoff.

Every controller submission checks the live balance on login2 and retains the
96% burn boundary plus a conservative four-hour CPU reservation. Auth and budget
failures remain external limits. Three consecutive failed worker starts or three
invalid agent turns stop the chain with recorded evidence. A concrete blocker
also stops it; a failed experiment by itself does not qualify. Successful project
completion requires a hashed evidence report, final CPU bootstrap markers, clean
latency groups, successful fresh GPU smoke, no remaining queued/running/failed
tasks, and a current independent paper source build. These mechanical gates do
not replace the agent's scientific and visual review.

Runtime state lives at `/scratch/09032/saini_2/sca/continuation/`:

- `status.json`: current worker, successor, latest result and dependency.
- `submissions/<job>.json`: budget and submission receipts.
- `job_<job>/turn_*.events.jsonl`: full agent events and adjacent final outputs.
- `slurm_<job>.log`: scheduler output; `failure_<job>.json` records controller errors.
- `STOP`: an optional operator-created file that stops new work at the next unit
  boundary. It does not interrupt an active experiment. Remove it only to resume
  explicitly requested continuation; never delete unrelated scratch artifacts.

The frozen deployment is
`/scratch/09032/saini_2/sca/dispatch/20261017_continuation/`, with SHA256 checks.
After resolving a blocker, submit its `continuation.py submit` with the managed
sca Python on a compute node. This checks budget before submitting. Inspect live
SCAC jobs and the owner state first to avoid duplicate controllers.

The worker maintains `HANDOVER.md`, `.agent/STATE.md`, and
`.agent/notes/latest_delivery.md`. Latest user artifacts remain `paper/main.pdf`
and `paper/sca_overleaf.zip`. No public repository or external messaging is
authorized by this continuation request.
