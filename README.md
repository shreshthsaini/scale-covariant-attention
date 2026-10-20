# Scale-Covariant Attention (SCA)

Training-free high-resolution sampling for diffusion transformers (FLUX.1-dev, FLUX.1-Krea-dev, FLUX.2-dev).
Private repository; paper under review. Start here if you are a fresh agent picking the project up.

## Read first, in this order

1. `HANDOVER.md`: authoritative state. Sections 13 to 16 are the latest (gold-standard rubric scores, the
   191 to 195 prompt extension, the standard evaluation protocol, open items).
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

## Where the heavy data lives (TACC Vista, scratch, purged after about 10 idle days)

- `/scratch/09032/saini_2/sca/out/<campaign>/<method>/<HxW>/`: generated PNGs with JSON sidecars.
- `/scratch/09032/saini_2/sca/eval/<group>/`: metric outputs; `DONE` and `BOOTSTRAP_DONE` markers.
- `/scratch/09032/saini_2/sca/taskq/{pending,running,done,failed,hold_*}`: the resumable task spool.
  Workers take the alphabetically first pending task. Create `taskq/STOP` to make every worker exit.
- `/scratch/09032/saini_2/sca/code/<tag>/`: frozen code copies that the task scripts run.
- `/scratch/09032/saini_2/sca/calib/k1_1024.npz`: the calibrated per-head table used at inference.
- Environment: `/work/09032/saini_2/vista/library/uv-envs/sca` (uv only; torch 2.11 cu128, diffusers 0.39).

## Restart recipe

1. `squeue -u $USER` and `ls /scratch/09032/saini_2/sca/taskq/*/ | head`: see what is still running.
2. If the spool is not empty and no workers run: submit `slurm/fleet_SCA13.sbatch` from a login node
   (`ssh login1 'sbatch /abs/path/fleet_SCA13.sbatch'`) or start guest workers on held nodes.
3. `nohup bash slurm/release_evals.sh > /scratch/09032/saini_2/sca/logs/release_evals.log 2>&1 &` if the
   held evaluations (`taskq/hold_eval`) still need releasing.
4. When new `results/*` CSVs land: regenerate tables (`paper/figures/gen_tables.py` and friends), edit
   `paper/main.tex`, then `uv run --no-project --python /work/09032/saini_2/vista/library/uv-envs/sca/bin/python python paper/build_delivery.py`.
5. Commit and push after every meaningful change (no attribution trailers).

## Rules that apply to all work here

Login nodes are for editing, git and scheduler commands only; keep under 40 processes and at most 3 ssh
connections. Never delete anything on scratch. Python through uv only. No em dashes or section symbols in
prose. Every reported number must trace to a CSV in `results/`.
