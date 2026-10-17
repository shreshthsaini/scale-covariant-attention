# Scale-Covariant Attention (SCA): full handover

Last updated 2026-10-17 10:39 PDT. Read the first 85 lines, then `.agent/STATE.md`.
This current section supersedes historical claims below it.

## 1. Current status and corrections

Codex owns completion of experiments, evaluation, paper, figures and reproducibility.
The user explicitly expanded the task: finish the entire paper, redesign diagrams,
make measured multi-baseline gains clear, and add recent relevant baselines.
Do not manufacture wins or discard unfavorable results. GPU queuing alone is not
completion. All root/subagent ownership is released to continuation1077042.

The major paper revamp is committed as 4011238. Current draft: 38 pages, 473-file
Overleaf ZIP; independent build passes, all 38 page renders identical at 72 dpi,
changed pages inspected at 144 dpi, no overfull boxes or undefined references.
New main diagram: native calibration, actual coefficient inset, attention tensors,
head/noise mixing and global opening; PDF/SVG/source/provenance preserved.
New six-method portrait uses authentic outputs and identical crop coordinates.
New paired quality figure exposes all five existing baselines, uncertainty and losses.
Compact main and ablation tables keep full metrics in the appendix. Ablation data
font increased from 6.96 to 8.97 pt. All 30 main cells independently verified;
42 retained ablation values match the earlier audited table exactly.
Receipts: results/evidence/{paper_revamp,presentation_review}_20261017.json.
Current links/hashes: .agent/notes/latest_delivery.md and paper/delivery_receipt.json.
This is a verified draft, not project completion. No git remote; commit locally.

New latency audit verified warm-up plus three same-process trials: AReA 6K482.2844s,
8K951.9874s; SCA 6K161.5585s,8K287.6398s. Speedups 2.9852x and 3.3097x,
time reductions 66.5% and 69.8%. All seven timing logs identify GH200 120GB.
Thirteen clean groups now audited; only full-attention 6K remains pending.
CSV summaries, raw logs, figure and manuscript now contain the verified values.

Baseline expansion is required next work, not an optional future suggestion:
repro/baseline_expansion_20261017/ and .agent/notes/baseline_audit_20261017.md.
HRDiT/SigMa official source pins, isolated dependency overlay, paired46 prompt list,
strict resume provenance and18CPUchecks pass. CPU tests are not GPU results.
Both one-image smoke tasks are frozen and enqueued; do not duplicate them.
After finite/fidelity/runtime checks, prepare pilot12, then46; consider full191 within
verified SU headroom. Promotion must not depend on SCA winning. Preserve failures.
HRDiT has a separate57-step progressive budget. SigMa paper dev/guidance3.5 differs
from released Krea/guidance4.5; released mu already equals max_shift. Do not count
identical schedule aliases as separate controls. No official ResDiT code verified.
New baselines must be evaluated/integrated or their concrete unresolved limitations
recorded before claiming whole-project completion. Read the audit and README.
Validated scoring handoff:repro/baseline_evaluation_20261017/README.md,9tests.
Exact12/46 eight-method pairing, GPU score stage, cache-only CPU bootstrap;
missing outputs close the barrier. No fresh baseline generation/scoring has run.

Durable CPU controller1076947 finished its first unit. Successor1077042 is PENDING,
dependency satisfied; start estimates can move. No new CPU watcher is needed.
Do not start another watcher, change its dependency, or edit the frozen controller.
Root CPU allocation1073260 ends10:45PDT. Follow ownership note for handoff.
SCA7/SCA9 completed, adding24 tasks. No SCA GPU allocation currently running.
SCA10 job1077447 PENDING:2-8nodes,4hours,min2hours. SCA11 job1077473 PENDING:
2-4nodes,2hours on gh-dev. Both await scheduler priority.
Both use unchanged frozen dispatch v2. Queue60pending,0running,0failed,894done.
Receipt: continuation/baseline_smoke_deployment_20261017.json (under evidence).
Latest submission balance7908/40000SUs. Before every new submission check login2
balance and reservations, retaining1600SUs minimum. At most3 SSHs; guard first.
MF9 guests remain yielded. Preserve foreign work, original logs and all scratch.

Root released the two owner-gated software repairs after owning job/steps ended:
a_000_repro_gpu_smoke_gcc_1076947.sh selects/probes GNU compiler before Torch;
a_y003_latency_6144x6144_full_gridfix_1076947.sh supplies true384x384 grid metadata.
Original failures retained; bf16 coordinates unchanged.18 release tests passed.
Do not duplicate retries. Full6K requires4800s admission plus dispatcher cleanup.
After successful GCC smoke, review/promote its distinct receipt to canonical
results/evidence/environment/fresh_gpu_smoke.json, preserving original failure.

Generation coverage:6K AReA56/100,SCA64/100;Krea AReA0/100,SCA20/100.
Krea overlay hash audit passed:100DyPE+100SEGA,193originals+7replacements.
All7regenerated PNGs are byte-identical to originals;200original sidecars preserved.
Receipt:results/evidence/krea_overlay_complete/receipt.json;207PNGs hashed/CRC checked.
Overlay: /scratch/09032/saini_2/sca/out/krea_baseline_overlay_20261017.
Strict footers still require all4E9methods at exact100 coverage before GPU eval.
Neither final CPU bootstrap READY marker exists at this handoff. When GPU eval
writes eval/{e2_6144,e9_krea}_base/READY_FOR_CPU_BOOTSTRAP, run the corresponding
hold_cpu_finalize script on CPU,1000reference-image-cluster draws, no completed reruns.
Then archive_evidence.py,gen_final_tables.py,complete6K/Krea fig_phase2.py galleries,
and reconcile abstract/main tables/interpretation/appendix/limitations. Rebuild delivery.

All three restored metric envs pass bounded CPU parity exactly;39assets12.08GB,
0downloads,42focused/71combined checks. Proof: metric_environment/validation_summary_1076947.json.
GPU metric parity/native4K MUSIQ/full benchmark reruns are not established by CPU checks.
E1/E4/E5/E6/E7/E8,budgetcontrols and exploratory12paired8K metrics/gallery are complete.
Preserve corrected195reference rows/168identities; patchFID/KID gain unresolved.
8K has six resolved SCA gains/four unresolved; MUSIQ means favor AReA; grid failures.
8K CPUfloat32 HPS differs from main GPU precision. Calibration runtime remains unknown.
Novelty is native emulation plus fitted/transferred per-head/noise mixing; ingredients
are established. Exact-coordinate theorem does not remove bf16 drift;0.015625 is
an observed GPU absolute difference, not a proven ULP bound.

## 2. Goal and user requirements

- The user wants a strong, novel training-free method paper that beats AReA, not an analysis or
  failure paper. Reference paper: AReA, arXiv 2610.17227, code github.com/KooroshRH/AReA (commit
  9e38ec2). AReA is training-free high-res generation for DiTs: a pooled-KV global phase, then
  shrinking windows.
- User decisions:
  - **Compare against AReA's released code under one shared protocol:** FLUX.1-dev, guidance 4.5,
    28 steps, mu = max_shift. Do **not** rerun AReA with its Krea checkpoint or its own guidance.
  - Keep SCA as the method and improve it until it clearly beats AReA.
  - Using idle nodes from the user's other jobs is approved ("use any idle node").
- Account rules are in `~/.claude/CLAUDE.md`: no em dashes, no section symbol, uv only, login-node
  limits, and no Claude attribution trailers on commits.

## 3. Final method (what the paper describes)

Applied to FLUX.1-dev joint attention (57 blocks, 24 heads, native grid n0 = 64x64 tokens at
1024^2). Code: `sca/attn.py`, SCAProcessor in mode `sca`.

- **Global branch O_glob.**
  - Image K/V are subsampled at block centres by the scale factor s per axis, onto a grid of
    about native size.
  - Query and key RoPE coordinates are block indices (coarse grid, AReA Phase-I geometry with
    fixed f = s).
  - Text tokens are always included.
- **Local branch O_loc.**
  - An exact native-extent sliding window: 64x64 image keys centred on 8x8 query tiles and
    clamped at borders, plus text keys, with the original RoPE.
  - Implemented as a FlexAttention block mask over tile-major token order. It equals a
    gather-based reference implementation to 1 bf16 ulp.
- **Gate.** For sigma > 0.89 (the first 8 of 28 steps under mu = max_shift) every head uses
  O_glob only, and the local branch is skipped.
- **Mix.** After the gate, image queries get `pi_h(sigma) * O_loc + (1 - pi_h(sigma)) * O_glob`
  per head. Text queries get O_glob.
- **Calibration of pi_h(sigma).**
  - Run FLUX at an in-distribution resolution (1024^2, 32 prompts; also 1280^2, 16 prompts) with
    full attention.
  - Emulate scale s (window n/s, pooling s) and solve closed-form least squares per
    (layer, head, sigma bin) against the model's own full-attention output.
  - Table: `/scratch/09032/saini_2/sca/calib/k1_1024.npz`, using the scale-4 entry at inference.
  - Cost: the historical 1.5 GPU-hour estimate is unsupported. Surviving partial logs cannot establish total calibration runtime; see `results/compute/claims_audit/`.
- **Band guard (optional, not in the final method).** It replaces the low band (avg-pool 16,
  bilinear up) of each head output with O_glob's. It fixed duplication on hand-picked prompts but
  lowered metrics and cost time. It is in the appendix.
- **Final CLI** (code commit 2d11740, frozen copy `/scratch/09032/saini_2/sca/code/2d11740`):
  `scripts/sample.py --method sca --pi-table calib/k1_1024.npz --gate-sigma 0.89 --label sca_nb`.
  There is no `--band`.

## 4. Key results (all numbers archived in `results/`; paired bootstrap 95% CIs)

Main comparison: SCA final (`sca_nb`) vs AReA (`area`, released code), 191 Aesthetic-4K prompts
(195 minus 4 development prompts, listed in `/scratch/09032/saini_2/sca/dev4.jsonl`), seed 0.

**4096^2** (`results/final4k_nb_*`): SCA is significantly better on every preference, alignment
and no-reference quality metric.

| Metric | SCA | AReA |
|---|---|---|
| ImageReward | 1.229 | 0.917 |
| HPSv2.1 | 0.297 | 0.272 |
| HPSv2.0 | 0.298 | 0.286 |
| PickScore | 23.09 | 22.35 |
| CLIP | 29.11 | 27.67 |
| Aesthetic | 6.45 | 6.28 |
| MUSIQ | 45.3 | 39.4 |
| MUSIQ crop | 61.2 | 50.4 |
| CLIP-IQA | 0.729 | 0.675 |
| ARNIQA | 0.712 | 0.673 |
| FID | 138.4 | 146.0 (CI -12.5 to -3.5) |
| patch KID | | significantly better for SCA |
| patch FID, KID, IS | | not significant |

Full attention at 4096^2 collapses (`results/final4k_full_*`): IR -2.18, FID 312.7.

**2048x4096** (`results/final_2048x4096_*`, scale 2 by 4): SCA roughly ties AReA.
- Not significant: IR 1.130 vs 1.133, HPSv2.1, PickScore, CLIP.
- SCA significantly better: MUSIQ +4.1, MUSIQ crop +4.3, HPSv2 +0.002.
- SCA significantly worse: aesthetic -0.21, CLIP-IQA -0.013, patch FID +5.1.
- Speed: 2.9x faster (shared-node mean, 38 vs 112 s).

**2048x6144** (`results/final_2048x6144_*`, scale 2 by 6): SCA clearly better.
- IR 0.664 vs 0.390, FID 143.8 vs 163.6.
- Also significantly better: KID, IS, PickScore, MUSIQ, CLIP-IQA, ARNIQA, HPSv2.
- Aesthetic -0.15 (significantly worse); patch FID, HPSv2.1 and CLIP not significant.
- Speed: 2.7x faster.
- Takeaway: the advantage grows with the scale factor.

**Latency**, clean batch 1 on an idle GH200, median of 3 images
(`results/latency_4096_batch1.csv`):

| Method | Seconds per 4096^2 image |
|---|---|
| Full attention | 184.6 |
| AReA released code | 212.6 |
| SCA final | 71.0 |
| SCA + band guard | 88.1 |

SCA final is 2.6x faster than full attention and 3.0x faster than AReA. Per-layer attention at
4096^2 is 18.7 ms for SCA vs 87.3 ms for full SDPA.

**Calibration** (`calib/k1_1024.json`, `k1_1280b.json`): the per-head fit beats per-layer,
uniform-per-sigma and the best hard switch.
- Relative error at s=2: 0.057 vs 0.071 / 0.077 / 0.092. At s=4: 0.200 vs 0.223 / 0.230 / 0.261.
- Split-half correlation 0.99.
- Transfer: s2 to s4 gives 0.205 vs 0.200 own; 1024 to 1280 gives 0.0589 vs 0.0585 own.
- About 50% of cells have pi in [0.2, 0.8]; the std of pi over heads is about 0.24.

**Ablations** in the final no-band setting, 46 prompts, reference sca_nb
(`results/ablnb_*_deltas.csv`, `results/localnb_paired_deltas.csv`):

| Ablation | Effect vs SCA final |
|---|---|
| pi=1 (local only) | IR -0.357, HPSv2.1 -0.029, CLIP -1.93, PickScore -1.20; all significant |
| uniform pi | IR -0.090, HPSv2.1 -0.018, PickScore -0.38, CLIP-IQA -0.077, ARNIQA -0.065, MUSIQ crop -10.4; significant |
| per-layer pi | HPSv2.1, PickScore, ARNIQA and MUSIQ crop significantly worse; IR not significant |
| pi=0 (global only) | IR -0.395 |
| mean pooling | IR -1.52 |
| no gate (v0) | IR -1.33 |

Patch FID: sca_nb 170.2, uniform 169.1, local 194.3, global 238.9. So the gate and the per-head
mix are both essential.

The earlier band-guard ablations (`results/abl_*`, `route_*`) are in appendix Table 4. Hard
per-head routing (`--route`) is no better than the soft mix. The scale-2 pi table at 2048x4096
(`results/ps2_*`) changes metrics negligibly; the default stays with the scale-4 table.

## 5. Findings and design history (why the method looks like this)

1. **v0** (pure per-head mix with a native 64 window, no gate) gave subject duplication at 4K: 5
   Moai statues, a sea of tiny boats. A window of 16 made it worse, because objects form at the
   window scale.
2. **Diagnosis.** A W-token window has no information below about 1/W spatial frequency, so its
   low band is hallucinated. Per-token L2 calibration cannot see this.
3. **Tried:**
   - the band guard: fixed boats but not the people prompts;
   - the gate alone: good, but tiny extra figures;
   - gate + band;
   - a gate at 0.80: 4x4 block mosaic from coarse query positions;
   - an adaptive band: same as band 16.
4. **Metrics then showed the band guard hurts** (HPS, PickScore, CLIP-IQA, MUSIQ), and that pi=1
   with the band looked as good as the mix.
5. **Without the band, pi=1 is far worse** (IR -0.36): the band was injecting the global low band.
   The final method is therefore gate + per-head mix, with no band.
6. **Known limitations:**
   - an articulated-subject prompt (underwater surfer) shows limb ghosting in band variants;
   - the gate threshold 0.89 equals AReA's sigma*;
   - only FLUX.1-dev was tested;
   - under our protocol AReA scores IR 0.92 at 4K vs 1.32 in its own paper.

## 6. Code map (repo = this directory)

- `sca/attn.py`: everything in the attention path.
  - SCAProcessor (modes `full`, `sca`, `calib`), SCAContext fields: gate_sigma, band, route,
    known_grid, local_impl flex|gather, pool center|mean.
  - global_branch, local_branch (gather), local_branch_flex, band_low and band_mix, PiTable
    (head|layer|uniform|switch|const), install(), attention_flops().
- `sca/calib.py`: calibration statistics, merge and report.
- `sca/runtime.py`: FLUX loader, seeding, file naming, sidecars, fingerprint.
- `sca/envfix.py`: serves zero-byte files deleted from the env by the scratch purge, so
  torch.compile works.
- `scripts/sample.py`: generation, methods full|sca|sca_uniform|sca_layer|sca_switch. Flags
  include `--gate-sigma`, `--band`, `--route`, `--pi-scale`, `--pi-const`, `--local-impl`,
  `--label`. `--res` is HxW.
- `scripts/calibrate.py`: calibration runs.
- `scripts/run_area.py`: AReA official wrapper with the same CLI and output format.
- `scripts/eval_extra.py`: IR, aesthetic, HPSv2/2.1, CLIP-IQA, ARNIQA, IS. Runs in the
  sca-metrics env.
- `scripts/paired_bootstrap.py`: paired CIs from per-image CSVs.
- `scripts/enqueue.sh`: spool task generator. Env vars: CODE (frozen code dir), PRF (prompts
  file), OUTROOT, LABEL.
- `scripts/bench_attn.py`, `scripts/bench_step.py`: kernel and step timing.
- `tests/test_sca.py`: 18 CPU tests (exactness of both branches, gate, band, flex vs gather,
  calibration).
- `slurm/`:
  - `fleet*.sbatch` (fleet jobs), `node_launch.sh` (telemetry plus one worker per GPU),
    `worker_loop.sh` (spool worker);
  - `guest_worker.sh`: worker for idle nodes of other jobs. It takes a task only when no foreign
    GPU process is present, and exits when one appears.
- Base evaluation reuses `/work/09032/saini_2/vista/Projects/Exploration/constancy_rope/scripts/eval.py`
  (CLIP, PickScore, MUSIQ, FID, KID, patch FID/KID). Read-only; owned by the constancy_rope
  project.
- `paper/`: main.tex, references.bib, ICLR 2026 style, `figures/` (qual/, failure/,
  method.pdf, pi_heatmap, pi_sigma, rel_err), `figures/src/` (figure scripts,
  fig_qualitative.py).
- Commit history: `git log --oneline`. Key commits:
  - 3746705: initial code;
  - b7cb957: flex kernel;
  - c1ba80b: gate/band;
  - 213a3a1: fused band;
  - 8ee220d: bf16 grid fix;
  - 2d11740: routing, the **final campaign code**;
  - f1d0829: paper complete.

## 7. Environments and data

- **Generation env:** `/work/09032/saini_2/vista/library/uv-envs/sca` (torch 2.11 cu128,
  diffusers 0.39). Damaged by the scratch purge (zero-byte files); `sca/envfix.py` works around
  it. A proper fix is to reinstall torch and triton on a compute node.
- **Metrics env:** `/work/09032/saini_2/vista/library/uv-envs/sca-metrics` (transformers 4.45,
  image-reward, hpsv2, pyiqa, torch-fidelity).
- **Env vars for runs:** `HF_HOME=/scratch/09032/saini_2/tmp/hf`,
  `TORCH_HOME=/scratch/09032/saini_2/tmp/torch`, `HF_HUB_OFFLINE=1`,
  `TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton`.
- **Model:** the FLUX.1-dev snapshot is cached under HF_HOME.
- **Prompts and references:** `/scratch/09032/saini_2/constancy_rope/prompts/aesthetic4k_eval.jsonl`
  and `aesthetic4k_refs.jsonl`. Calibration prompts: `scripts/calib_prompts.jsonl` (48, no
  overlap with Aesthetic-4K).
- **AReA checkout:** `/scratch/09032/saini_2/sca/third_party/AReA`. Defaults are documented in
  `.agent/notes/area_code.md`.

## 8. Scratch layout (`/scratch/09032/saini_2/sca`; purge risk after about 10 idle days)

- `out/main/{sca_nb,area,full,sca}/<HxW>/`: images with JSON sidecars. sca = the earlier
  band-guard config.
- `out/abl/<label>/4096x4096` and `out/abl/abl_ps2/2048x4096`: ablations.
- `out/dev/`: 4-prompt development variants.
- `out/latency*/`: batch-1 timing runs.
- `eval/<name>_{extra,base}/`: per-image and summary CSVs (archived copies in `results/`).
- `calib/`: calibration tables and reports.
- `code/<commit>/`: frozen code used by spool tasks (never edit; the resume fingerprint includes
  an implementation hash).
- `taskq/{pending,running,done,failed,stale}/`: the spool. `STOP` makes workers exit.
- `logs/`: task logs (task_<name>.log), guest_<node>.log, fleet logs.
- `run_missing.sh`, `requeue_missing.py`: recovery helpers (Section 9).
- Everything paper-critical is already in `results/` and `paper/` on /work. Images live only on
  scratch.

## 9. How to run things (recipes)

- **Generate:**
  `cd /scratch/09032/saini_2/sca/code/2d11740 && <sca env python> scripts/sample.py --method sca --res 4096x4096 --prompts <jsonl> --start 0 --end 5 --seed 0 --guidance 4.5 --out <dir> --pi-table /scratch/09032/saini_2/sca/calib/k1_1024.npz --gate-sigma 0.89 --label sca_nb`
- **Spool campaign:**
  1. Enqueue: `CODE=2d11740 LABEL=x [PRF=prompts] [OUTROOT=...] bash scripts/enqueue.sh <method> <HxW> 5 <sampler args>`.
  2. Task names sort as priority (a < b < ... < z).
  3. Submit a fleet: `ssh login1 'sbatch <repo>/slurm/fleet.sbatch'`. Check
     `/usr/local/etc/taccinfo` first. Remove `taskq/STOP` before launching workers.
- **Guest workers on idle nodes:**
  1. Find nodes: `bash ~/.claude/tools/slurm-idle.sh`, FREE-IDLE list.
  2. Launch: `ssh <node> "setsid nohup bash <repo>/slurm/guest_worker.sh > <scratch>/logs/guest_<node>.log 2>&1 < /dev/null &"`.
  3. When a host job ends, its task stays in `taskq/running/*-guest`. Requeue it with
     `python -I /scratch/09032/saini_2/sca/requeue_missing.py <running files>`, which rewrites the
     task to sample only missing prompt ids. Evaluation tasks can just be moved back to pending
     (they are cached and rebuild their inputs).
- **Evaluation** (template: `taskq/done/b999_eval_final4k_nb.sh`):
  1. Symlink common prompt ids (excluding dev4) into `out/<root>/<method>/<res>/`.
  2. Run `eval_extra.py` (sca-metrics env) and constancy_rope `eval.py` (sca env) with
     `--reference-method <label>`. The method name comes from the sidecar label, e.g. `sca_nb`,
     not the directory name.
  3. If the deltas come out empty, run `scripts/paired_bootstrap.py` on the per-image CSVs.
- **Tests:** `<sca env python> -m pytest -q tests`.
- **Paper:** `cd paper && tectonic main.tex`. Figure scripts are in `paper/figures/src`.

## 10. Bugs found and fixed (do not reintroduce)

- **Resume fingerprint unstable.** diffusers `_use_default_values` is a set-ordered list, so
  reruns of partly-done chunks failed with "different settings". Fixed in c579425 (sorted). Frozen
  older code still has the bug, so use `requeue_missing.py`.
- **FLUX img_ids are bf16.** Positions above 256 tokens per axis are rounded, so SCA's grid
  inference failed at 2048x6144. Fixed in 8ee220d (the sampler passes `known_grid`). Note: base
  FLUX RoPE positions are rounded at width 6144 for every method.
- **Live code edits break running campaigns.** Spool tasks always run from frozen copies in
  `code/<commit>`.
- **Flex block-size crash** on GH200 (Q_BLOCK_SIZE 64). Fixed in b7cb957 via kernel options.
- **Slurm kills guests without a log line** when a host job ends. Detect orphans by checking
  `squeue -w <node>`.

## 11. Open items / next steps (ranked)

1. Ask the user whether to create a private GitHub repo and push. No remote exists.
2. A second model (Qwen-Image or FLUX.2). This needs SCA ported to its attention processor and a
   new calibration table.
3. RoPE-based baselines (DyPE, SEGA, UltraImage), with code under
   `/scratch/09032/saini_2/constancy_rope/third_party`, and HRDiT (github.com/zylwithxy/HRDiT,
   needs its own env with transformers 4.48).
4. Optional speedups: fuse the mix into the flex kernel; skip the local branch for heads with
   pi near 0.
5. Reinstall torch and triton in the `sca` env instead of relying on envfix.
6. The paper's no-ai-slop pass and a final read; the figure scale and crops can be polished.

## 12. PHASE 2 (started 2026-10-16 11:30): bulletproof experiments + presentation upgrade

User request: complete the paper with many visual examples, a hero figure, polished diagrams, clear
contributions and method (with math), comparison tables, qualitative results, a much larger appendix
with proofs and extra evidence, more experiments. Keep this file current; it is the crash-recovery doc.

Experiments (all at seed 0, Aesthetic-4K ids excl dev4 unless noted; spool prefixes in parentheses):
- E1 baselines at 4096^2, 191 prompts: DyPE, SEGA (official), UltraImage (crope harness / third_party).
- E2 higher resolution: 6144^2 metrics (SCA, AReA; 100 prompts) and 8192^2 qualitative + latency.
- E3 latency curve: 2048^2, 3072^2, 4096^2, 6144^2, 8192^2 x {full, AReA, SCA}, batch 1, idle node.
- E4 HRGen-1K (AReA's second prompt set, no references): SCA vs AReA at 4096^2, 200 prompts.
- E5 seeds 1 and 2 on 50 prompts at 4096^2 (SCA, AReA) for variance.
- E6 calibration budget: pi tables from 4 / 8 / 16 prompts (merge subsets of calib/k1_1024_parts with
  --report-only), generate 46 prompts at 4K each.
- E7 gate sensitivity: sigma* in {0.85, 0.93} at 4K, 46 prompts.
- E8 second model FLUX.2-dev (cached): port SCA, calibrate, SCA vs full attention at 4K (50 prompts).
- E9 checkpoint generality FLUX.1-Krea-dev (AReA's code default): recalibrate, SCA vs AReA, 100 prompts.
Presentation: hero figure (page 1), TikZ method diagram, calibration figures, many qualitative grids
(main + appendix galleries per resolution and per baseline), math (formal method, closed-form
calibration, propositions on relative-position range and cost, proofs in appendix).
Owners: experiments-engineer agent (E1-E7, E9 task scripts), FLUX.2 port agent (E8), figures agent
(paper/figures only, no main.tex edits; writes paper/figures/snippets.tex), writing agent (owns main.tex).
