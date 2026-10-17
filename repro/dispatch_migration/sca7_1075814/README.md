# Pending SCA7 compatibility route

This is a reviewed activation proposal for existing job1075814. It has not been
applied by this preparation step. No job is submitted, cancelled or resized.
The saved batch retains four nodes and one launcher task per node.

`node_launch_before.sh` preserves the exact mutable launcher before migration.
`node_launch_proposed.sh` adds a branch for job1075814, then retains the original
body verbatim for every other job. `node_launch.patch` is the reviewable diff.
`receipt.json` records hashes and the frozen bridge path. The bridge verifies
the three frozen v2 dispatcher hashes before running them.

The bridge accepts a future numeric `SLURM_JOB_END_TIME`. Otherwise rank zero
queries `scontrol show job 1075814 -o` once and atomically publishes its actual
EndTime. Peers read that file for at most45 seconds, without scheduler queries.
Receipts are separated by job, restart count and step under
`/scratch/09032/saini_2/sca/dispatch/sca7_deadlines`. Invalid topology, expired
deadlines and missing coordinator receipts fail closed. There is no fallback to
the requested four-hour duration. The v2 supervisor reaps its telemetry process.

Activation belongs to the main operator after confirming the job remains
pending and the live launcher still has the archived SHA256. Apply the supplied
patch to `slurm/node_launch.sh`, verify it equals `node_launch_proposed.sh`, and
record activation separately. Do not replace the queued batch or resubmit it.
This preserves queue priority. If the allocation has already started, review
its actual step state before any change to its entrypoint.

Validation used no SSH, scheduler query, GPU or live task queue:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-project --python /work/09032/saini_2/vista/library/uv-envs/sca/bin/python python -m pytest -q tests/test_sca7_bridge.py
bash -n repro/dispatch_migration/sca7_1075814/node_launch_proposed.sh
```

The14 focused tests cover four-rank coordination, zero-query environment
deadlines, one-query fallback, shared failures, bounded peer waits, topology and
receipt validation, timestamp parsing and the final preserved Slurm environment.
This is CPU validation; successful launch still needs allocation-time evidence.
