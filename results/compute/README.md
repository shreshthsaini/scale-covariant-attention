# Attributable compute accounting

Refresh on a compute node:

```sh
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python scripts/account_compute.py
```

`summary.json` gives the timestamped accounting snapshot. The defensible resource
lower bound is `attributable_resource_hours_lower_bound`: completed SCA task intervals
unioned by physical GH200 node. Advancing each start by one second allows for the
logs' one-second timestamp resolution. Intervals include model loading, calibration,
evaluation, failed attempts, and waits inside tasks. They measure occupied worker
time rather than GPU kernel activity or exclusive GPU utilization.

Worker logs are archived by content hash. Refreshes retain old snapshots, deduplicate
their events, and count repeated attempts at different times separately. A start with
no matching end contributes no time. `unpaired_events.json` preserves these censored
attempts; the summary distinguishes starts still present in the running spool.
Completed spool tasks missing a logged successful interval are listed explicitly.

The image-sidecar subtotal is the sum of completed batch-one generation durations.
Resolved paths and identical image hashes are deduplicated. External project outputs,
including linked Krea baseline images, are excluded. This subtotal overlaps worker
time and omits model loading and file I/O where stated by the original sidecars.
Sidecars lack host placement and start/end timestamps, so their sum cannot establish
GPU-node hours independently. It is never added to the worker subtotal.

Dedicated SCA scheduler node-hours are shown separately from the supplied accounting
snapshot. They overlap worker spans and may include idle allocation time. The full
MF9, CF2, or other shared parent allocations are never charged to this project.
Refresh the scheduler snapshot separately after the final jobs finish; the helper
uses the lexicographically latest `sca_scheduler_accounting_*.psv`, or an explicit
`--scheduler` argument.

The raw log snapshots, compact generation metadata, and source hashes are in `raw/`
and `source_manifest.json`. `closed_task_spans.csv` and `generation_seconds.csv` permit
independent aggregation. Missing and overwritten historical logs, manual jobs, and
ongoing work mean this snapshot does not establish the complete campaign cost.
