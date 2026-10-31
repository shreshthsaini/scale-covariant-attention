# Evaluation sidecars (copy of /scratch/09032/saini_2/sca/eval, text files only)

One directory per evaluation group, copied on 2026-10-31 with the layout of `/scratch/09032/saini_2/sca/eval/<group>/`:
per-image metric tables (`*.csv`, `extra/extra_cache.jsonl`), summaries, paired deltas, bootstrap outputs, run
configuration JSON, the evaluation task and CPU-bootstrap scripts (`*.sh`, `*.py`), logs and the `DONE`,
`BOOTSTRAP_DONE`, `CPU_BOOTSTRAP_COMPLETE` and `READY_FOR_CPU_BOOTSTRAP` markers. 1,507 files, 308 MB.

Not included here: the 88,755 `*.npz` feature files (Inception, crop and CLIP features used by FID, KID, patch FID and
IS) and the 15,373 symlinks that point evaluation roots at image sidecars in `out/`. The complete tree, features and
links included, is in the private Hugging Face dataset `shreshthsaini/sca-eval-outputs` as one tar archive per group
(`eval/<group>.tar`); see `MIGRATION.md` for the download and extraction commands.

`SHA256SUMS` covers every file in this directory. The paper's tables read the archived copies under `results/<run>/`;
these sidecars are the per-image inputs behind them (for example `a4k195_main_sca_v2/base/` and `extra/`, which the
project page notes were not archived before).
