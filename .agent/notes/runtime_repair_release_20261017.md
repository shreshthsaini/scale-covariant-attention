# Runtime repair release helper checkpoint

Completed a one-shot, default dry-run helper for the two frozen repairs in
`repro/runtime_repairs_20261017/manifest.json`. Root owns live deployment.
No scheduler query or real taskqueue mutation was performed by this subagent.
No owned processes remain running.

Files: `release.py`, `test_release.py`, and `README.md` in that directory.
Receipt and complete test log:
`results/evidence/runtime_repairs_1076947/release_helper_validation.json` and
`release_helper_tests.log` beside it. All 18 temporary-queue tests pass.
The validation receipt hashes helper, tests, documentation, manifest, and log.

The scheduler gate uses account-wide user squeue queries and filters exact job
1075814 or its step/heterogeneous/array prefixes. Step format is `%i`, because
Slurm rejects `%T` for `squeue --steps`. A root sacct row is mandatory and all
related records must be terminal. Scheduler errors and unknown/truncated states
refuse application. There is no scheduler polling loop.

Application verifies both repairs before mutations, archives each original
without overwriting, then publishes the matching held task atomically. It is
idempotent after publication and handles interrupted archival or publication
hard links, including an immediate dispatcher claim. Quarantine, independent
duplicates, failed replacements, and unexpected states refuse application.

Next: root runs one real dry run with a new scratch receipt. It should refuse
while owner 1075814 is running. Keep originals failed and replacements held.
Only use `--apply` after the helper itself confirms the owning job and all
related steps are gone. Commands and recovery details are in the README.
