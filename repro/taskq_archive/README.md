# Task spool archive (2026-10-31)

Every SCA run on Vista went through the file spool `/scratch/09032/saini_2/sca/taskq` (workers take the alphabetically
first script in `pending/`, move it to `running/`, then `done/` or `failed/`). The task scripts are the exact command
lines, environment variables, frozen code tags (`/scratch/09032/saini_2/sca/code/<tag>`) and output paths of each run.

- `taskq_done_20261031.tar.xz`: all 2,463 finished task scripts (`taskq/done/*.sh`). `taskq_done_index.txt` lists them.
- `taskq_state_20261031.tar.xz`: the rest of the spool at migration time: `pending/` (one task:
  `a000_z988_eval_hrgen1k_krea_sca_v2.sh`, the HRGen-1K Krea evaluation of sca_v2), the `hold_*` queues
  (staged reruns never released), `stale/`, `recovered/`, the deployment JSON records and the `STOP.until_*` markers.
- `SHA256SUMS`: hashes of both archives.

Extract with `tar -xJf taskq_done_20261031.tar.xz -C <new scratch>/sca` (creates `taskq/done/`). The scripts use Vista
absolute paths; `repro/migration/relocate_paths.sh` rewrites them for a new machine (see `MIGRATION.md`).
