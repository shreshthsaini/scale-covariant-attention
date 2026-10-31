# Calibration tables (copy of /scratch/09032/saini_2/sca/calib)

The calibrated per-head tables are SCA's only learned state. `*.npz` files are stored with Git LFS
(`git lfs pull` after cloning). The same files are in the private Hugging Face model repo `shreshthsaini/sca-calibration`,
whose model card describes each table.

- `k1_1024.npz` (SHA256 `7c54b727b75f4680264a98082db814974282108c58470804a2348025dfde0a47`): the FLUX.1-dev table every
  reported FLUX.1-dev result uses (sidecar field `config.sca.pi_table_sha256`). `k1_1024.json` is its report.
- `*_parts/`: per-prompt calibration statistics that `scripts/calibrate.py` merges into a table.
  `k1_1024_n{4,8,16}_parts/` were symlinks to subsets of `k1_1024_parts/`; `PART_LINKS.txt` lists them.
- The code reads `/scratch/09032/saini_2/sca/calib/<name>.npz` (flag `--pi-table`); on a new machine copy or link this
  directory to `<SCA_SCRATCH>/sca/calib` (see `MIGRATION.md`).
