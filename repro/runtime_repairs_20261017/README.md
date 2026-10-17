# Gated runtime repair release

The two replacements in `manifest.json` remain in `taskq/hold_review` while job
1075814 or any of its steps is present. The originals remain in `taskq/failed`.
This helper performs one bounded scheduler check per invocation. It never submits
jobs, starts a watcher, or changes the controller.

Run from the repository root on an allocated CPU node. The default is a dry run:

```bash
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python \
  repro/runtime_repairs_20261017/release.py \
  --taskq /scratch/09032/saini_2/sca/taskq \
  --receipt /scratch/09032/saini_2/sca/runtime_repairs_20261017/release_dryrun_001.json
```

After the owning job has ended, release with a new receipt filename:

```bash
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python \
  repro/runtime_repairs_20261017/release.py \
  --taskq /scratch/09032/saini_2/sca/taskq \
  --receipt /scratch/09032/saini_2/sca/runtime_repairs_20261017/release_apply_001.json \
  --apply
```

Every invocation checks account-wide `squeue` job rows, accounting records for
1075814, and account-wide `squeue --steps` identifiers. The root accounting row
must exist, all related accounting states must be terminal, and no related job
or step may still be active. The steps query uses `%i` because Slurm does not
support `%T` in its step format. Unrelated jobs and steps are ignored. Scheduler
errors, stderr warnings, unknown or truncated states, missing evidence, and hash
mismatches refuse release with exit code 2 and a receipt. An existing receipt
is never overwritten. Keep at least 10 seconds between scheduler invocations.

Before any task move, the helper verifies the manifest's exact two filename
pairs, its repository-relative terminal-failure evidence hash, both original
hashes, both replacement hashes, and the queue states for both repairs. It uses
an exclusive lock for application. Originals move to
`taskq/recovered/runtime_repairs_1076947/`; replacements move atomically from
`hold_review` to `pending` without overwriting another task. Both directories
must be on the same filesystem. No queue files change during a dry run.

Reruns recognize matching replacements already in `pending`, `running`, or
`done`, including worker filename suffixes. Interrupted hard-link publication
or archival can be completed safely. Independent duplicates, replacement
failures, quarantine entries, and unexpected queue states refuse release.
Each retry needs a new receipt filename. Inspect any refusal before retrying;
the helper has no polling or automatic retry loop.

The tests use temporary queues and mocked scheduler responses only:

```bash
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python \
  -m unittest discover -s repro/runtime_repairs_20261017 -p test_release.py -v
```

No application to the live queue was performed during helper development.
