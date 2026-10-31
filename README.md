# Scale-Covariant Attention (SCA)

Training-free high-resolution sampling for diffusion transformers (FLUX.1-dev, FLUX.1-Krea-dev, FLUX.2-dev).
Private repository; paper under review. Start here if you are a fresh agent picking the project up.

## Read first, in this order

1. `MIGRATION.md`: the project left TACC Vista on 2026-10-31. What lives where (GitHub, private Hugging Face
   repos, public downloads), restore steps, path layout, work in flight.
   `HANDOVER.md`: authoritative state. Sections 13 to 20 are the latest (gold-standard rubric scores, the
   191 to 195 prompt extension, the standard evaluation protocol, the sca_v2 rewrite, open items).
2. `.agent/STATE.md` and `.agent/notes/latest_delivery.md` (current PDF and zip hashes, build directory).
3. `.agent/notes/eval_protocol_survey.md`: what every related paper benchmarks on, with sources, and the
   13 paper edits that followed from it. `.agent/notes/paper_presentation_20261020.md`: the figure and layout
   overhaul notes.
4. `paper/figures/REFERENCE_STYLE.md` and the `gold-standard-paper` skill (`~/.claude/skills/gold-standard-paper`):
   the design bar every figure, table and page must meet.

## Layout

- `sca/`: the method (`attn.py` SCA processor, `calib.py` calibration, `runtime.py`, `flux2.py`).
- `scripts/`: sampling, calibration, baselines (`run_area.py`, `run_ultraimage.py`), evaluation,
  paired bootstrap, spool helpers (`enqueue.sh`, `requeue_missing.py`, `restore_empty_files.py`).
- `slurm/`: fleet jobs (`fleet_SCA13.sbatch`), `node_launch.sh`, `worker_loop.sh`, guest workers
  (`guest_worker_shared.sh` claims idle GPUs only; `guest_worker_cotenant.sh` shares busy GPUs),
  `occupy_all.sh` (keeps a worker on every held node), `release_evals.sh` (releases held evaluations
  when their generation group drains, then runs the CPU bootstraps).
- `results/`: every number in the paper as CSV with SHA256SUMS (`a4k195_*` are the 195-prompt headline
  sets, `published_compare/` the AReA published-number comparison, `compute/` the GPU-hour audit).
- `paper/`: `main.tex`, `references.bib`, `figures/` (scripts `fig_*.py`, TikZ source `figures/src/method.tex`,
  snippets), `build_delivery.py` (builds `main.pdf` and the self-contained `sca_overleaf.zip`, verifies a
  fresh-unzip build). The committed `paper/main.pdf` and `paper/sca_overleaf.zip` are the latest delivery.
- `repro/`: baseline promotion records and spool task archive (`repro/taskq_control/` holds the evaluation
  release order).
- `tests/`: unit tests for the kernels and the spool tooling.
- `calib/`: calibration tables (`k1_1024.npz` is the FLUX.1-dev table every result uses; npz in Git LFS).
- `third_party/`: pinned AReA, UltraImage, DyPE and SEGA sources; README lists all six baseline revisions.
- `MIGRATION.md`, `repro/migration/`: the 2026-10-31 move off Vista and the scripts that built and checked it.

## Where the heavy data lives

The project moved off TACC Vista on 2026-10-31; `MIGRATION.md` is the map and the restore recipe.

- This repository: code, paper, web, notes, every reported number (`results/`, with per-image metric files in
  `results/eval_sidecars/`), calibration tables (`calib/`, Git LFS), pinned baseline sources (`third_party/`), every
  task script (`repro/taskq_archive/`), environment locks and model revisions (`repro/`).
- Private Hugging Face repos (`shreshthsaini/`): `sca-calibration` (model), `sca-eval-outputs` (evaluation tree with
  feature banks, logs, prompts), `sca-generated-images` (headline and sca_v2 image sets as tar archives),
  `sca-frozen-code` (frozen code copies, web assets, trace renders). `results/migration_manifest.csv` lists every
  archive with its SHA256.
- Vista paths the code still uses (`MIGRATION.md` Section 2 maps them to a new machine):
  `/scratch/09032/saini_2/sca/{out,eval,calib,code,taskq,logs}`; environments under
  `/work/09032/saini_2/vista/library/uv-envs/{sca,sca-metrics}` (uv only; torch 2.11 cu128, diffusers 0.39).

## Restart recipe

1. Restore code, data and environments with `MIGRATION.md` Section 3, and point the paths with Section 2.
2. Finish the work in flight (`MIGRATION.md` Section 4): the HRGen-1K Krea evaluation of sca_v2, and the HRGen-1K
   swap into the paper and page if it is not on GitHub.
3. Generation and evaluation run through the file spool (`taskq/pending`, alphabetical order; `taskq/STOP` stops every
   worker): `slurm/worker_loop.sh` is the worker; the `slurm/fleet_*.sbatch` files are Vista-specific examples.
4. When new `results/*` CSVs land: regenerate tables (`paper/figures/gen_tables.py`, `paper/figures/src/swap_all.sh`),
   edit `paper/main.tex`, then `uv run --no-project --python $SCA_ENVS/sca/bin/python python paper/build_delivery.py`.
5. Commit and push after every meaningful change (no attribution trailers).

## Rules that apply to all work here

Login nodes are for editing, git and scheduler commands only; keep under 40 processes and at most 3 ssh
connections. Never delete anything on scratch. Python through uv only. No em dashes or section symbols in
prose. Every reported number must trace to a CSV in `results/`.
