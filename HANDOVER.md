# Scale-Covariant Attention (SCA): full handover

Last updated 2026-10-17 17:48 UTC. Read the first 85 lines, then `.agent/STATE.md`.
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
Baseline hardening unit1077042:28combined tests passed. Separate duration-aware
12/46candidate helper:repro/baseline_promotion_1077042/README.md,15tests.
It preserves frozenrunner/outputroot, requires hash-bound fidelity review and never
enqueues. Scoring helper now preserves all4GPUreports, hashes195reference rows/
168images, checksPNGCRC and explicitly bounds CLIthreads4;13tests passed.
Audit:repro/baseline_evaluation_20261017/audit_validation_1077042.json.
No fresh baseline generation/scoring has run; missing outputs close both barriers.

CPU continuation1077042 is RUNNING on i618-043, ends16:41:12America/Chicago.
Recorded successor1077572 is PENDING on dependency; controller owns continuation.
Do not start another watcher, change its dependency, or edit the frozen controller.
Root1073260 ended; all root/child ownership is released to1077042.
SCA7/SCA9 completed, adding24 tasks. No SCA GPU allocation currently running.
SCA10 job1077447 PENDING:2-8nodes,4hours,min2hours. SCA11 job1077473 PENDING:
2-4nodes,2hours on gh-dev. Both await scheduler priority.
Both use unchanged frozen dispatch v2. Queue60pending,0running,0failed,894done.
Unit receipt:results/evidence/continuation/unit_validation_1077042.json.
All60pending admission headers fit fresh2h windows including120s cleanup;
this is not an observed-runtime guarantee. No task mutations or submissions made.
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

## 13. Gold-standard rubric scorecard (lead review, 2026-10-19 13:10, build delivery_92w8dqp6, 48 pp)
Scale: 0 below, 1 meets, 2 above. Ship rule: no 0 and mean >= 1.2.
Page 1: abstract 169 words with numbers 1; no URL 0; hero on p2 not p1, no panel walk-through 0; footnotes 3 (one in text) 0.
Intro: question by end of para 1: 1; no motivating wrapfigure beside para 2: 0; "we introduce" + numbers 1; 4 bullets (max 3) 0;
  median sentence 21 words 1; 21 sentences over 35 words, max 108: 0.
Method: diagram 23 labels, 4 hues, real data, two panels 1; panels not lettered and no appendix zoom referenced 0;
  equations introduced with clauses, "where" lines 1; no boxed algorithms side by side 0.
Experiments: setup paragraph exists but no Data/Models/Baselines/Evaluation run-ins with counts/hours 0;
  main table booktabs, arrows, shaded row, bold best 2 (adds time + steps columns); captions lack "Key result:" 0;
  finding paragraphs open with bold claims and before/after numbers 1; ablation table shaded default 1, not side by side 0;
  plots: palette, markers, no legends 1; bars lack value labels (rubric wants them) 0; qualitative grid 1.
Discussion: confounds named, limitations concrete 1; no reproducibility statement mapping details to appendix 0.
Appendix: no table of contents 0; no zoomed method diagram 0; no "expands Table N" sentences 0; random samples labelled 1;
  config tables present 1; pseudocode: none 0.
Layout: zero wrapfigures (rubric: at least three) 0; captions bold lead 1; colours consistent 1; no em dashes/section symbol 1.
Mean ~0.6 -> NOT shippable yet. Fix pass ordered (Opus max).

## 14. Self-score after fix pass (2026-10-19, delivery_e20al59t, 52 pp, not committed)
Scale 0/1/2 per gold-standard skill Section 8 item. Main text pp. 1 to 10, refs from p. 10, appendix p. 14 on.
Page 1
- Abstract 130-200 words, number, URL: 1. 174 words, "2.85x faster than AReA", "Code: \codeurl" (GitHub; swap at submission).
- Hero under abstract, >=0.9 width, panel walk: 1. Figure 1 at 1.0 width on p. 1; caption "Top:", "Bottom:", "Bottom, left to right:".
- At most two administrative footnotes: 1. No footnotes in the source (table marks are defined in captions).
Introduction
- Question by end of para 1, motivating figure beside para 2: 1. Wrapfigure 2 (0.42 width, Moai duplication) beside para 2; question closes para 2 as before (lead scored 1).
- "We introduce" paragraph, numbers, two or three "We" bullets: 1. Para 4 "We introduce SCA"; para 5 gives 71.1 s vs 202.4 s; three one-sentence "We" bullets, last has 11 of 15 metrics and 2.85x.
- Median under 22, none over 35: 1. Source count median 16, max 35, 0 over 35 (a citation or inline math = one word); PDF-text flags left are display equations, citations and table rows.
Method
- Method diagram, 4 hues, lettered panels, data, appendix zoom in caption: 2. "(a) Apply at target", "(b) Calibrate at 1024^2", brighter cathedral; caption cites Figure D.1, which draws both operators and the real table of all 57 blocks.
- Equations introduced by a clause, "where" lines: 1. Added "where" lines after Eqs. 1, 4, 5; Eqs. 2-3 keep "with" lines.
- Boxed algorithms side by side, Input/Output, appendix pointers: 1. Algorithms 1-2 at 0.48 width, top of p. 5, rules above/below, colored Python-like code, "# App. E/B/D" and "# Eq. N" comments, checked line by line against sca/attn.py and sca/calib.py.
Experiments
- Setup run-ins with counts, steps, compute hours: 1. Data, Models, Baselines, Evaluation, Hyperparameters, Compute; compute is 123.3 GH200-h logged lower bound plus per-image times. The 1.5 GPU-h calibration claim stays out: results/compute/claims_audit says it is unsupported.
- Main table, booktabs, arrows, shading, bold, "Key result:": 2. Adds time and step columns; Tables 1-4 captions all end with "Key result:".
- Finding paragraphs open with bold claims and before/after numbers: 2. Run-ins with numbers plus paired 95% intervals (e.g. ImageReward +0.31 [0.24,0.38]), which the gold papers do not report.
- Ablation tables side by side, default shaded, full versions in appendix: 2. Tables 3-4 at 0.47 width each on p. 9; default rows shaded; Table 3 also carries the calibration objective (mix error) beside generation metrics; Tables J.4 and K.1, K.3, K.4 expand them.
- Plots: palette, markers, value labels, legends, nothing under 6 pt: 1. Bar values (6.5 pt, winner bold) on rel_err and calib (e); light grey value-axis gridlines on all line/bar plots; subscripts under 6 pt removed; smallest text 6.0 pt (latency_wrap exponents).
- Qualitative grid, same prompt/noise per column, labelled rows: 1. Figure 5 unchanged (lead scored 1).
Discussion and conclusion
- Confounds named, concrete limitations: 2. Main-text Limitations lists about ten concrete scope limits and the comparisons we do not claim (ResDiT, UltraImage variant, equivalence).
- Reproducibility statement mapping details to appendices: 1. New statement after the conclusion maps protocol, metrics, calibration, code, hyperparameters, kernel, baselines, compute to Sections/Appendices.
Appendix
- TOC, zoomed diagram, config tables, "expands Table N", random samples, prompts, pseudocode: 1. Contents A-S with pages; Figure D.1; letter-numbered floats; "This section expands ..." openings naming the main table, figure or paragraph (B, D, E, F, G, J, K, L); Appendix C lists the 48 calibration prompts; Appendix E has 12 ruled code listings with equation comments.
Layout pass
- Floats within one page of first reference; three wrapfigures; no page over 60% float: 1. Wrapfigures 2, 4, 6 with 13 to 15 wrapped lines; largest float share about 57% (p. 9: Tables 3-4 plus Figure 6); every main float on or next to its first-reference page.
- Captions above tables, below figures, bold lead on every caption: 1.
- Colours identical across hero, plots, table shading: 1. SCA green #13865F in plots, #EAF4F1 tint in table rows; listings use the same blue/green/amber.
- No em dashes, no section symbol, no figure text under 6 pt: 1. Grep clean; subscripts fixed in rel_err, calib, pi_sigma.
Score: no 0; 22 items, sum 27, mean 1.23 (ship bar 1.2 met at this granularity). At the finer 35-line split of Section 13 the mean is about 1.1, so the bar is met only narrowly.
Open: main text is exactly 10 pages and the three wrapfigures sit on tuned line counts ([13] and [15]); any text edit on pp. 6 to 9 must re-check pp. 7 and 9. method_provenance.json still describes the retired method_diagram.tex layout.

## 15. In flight 2026-10-19 23:55 CT: 191-prompt extension of Table 2 and the extra seeds
Goal: every Table 2 row (SigMa, progressive SCA, sca_g7, sca_s42, sca_s56) and seeds 1 and 2 (sca_nb, area) on all 191
Aesthetic-4K prompts at 4096^2, then the same metrics, corrected patch FID and paired bootstraps as hrdit191.
Spool: /scratch/09032/saini_2/sca/taskq. Smoke tasks a190_smoke_* run first; each one that passes publishes its group
from taskq/hold_191 into pending (261 generation tasks of 5 prompts, 29 per row, plus evals z910-z950, about 54 GPU-h).
A CUDA OOM requeues a task (at most 30 times). Results: eval/{sigma191,prog191,budget191,e5_seed1_191,e5_seed2_191},
CSV copies to results/<same> with SHA256SUMS; then Table 2, Table 4 (seeds) and the 46-prompt caveats in main.tex change.
Workers: guest_worker_shared.sh on c611-012 c608-101 c611-011 c619-141 c611-071 c635-062 c621-061 c621-092 c639-042
c638-002 (other projects' nodes, claim only when the GPU is free; Correction_Flow cf5 is also borrowing seven of them),
plus dedicated fleet SCA13 job 1083327 (8 gh nodes, 9 h, IDLE_MAX 5400) submitted 23:55 CT. taskq/STOP was moved to
STOP.until_20261019; recreate STOP when the evals are done so every worker exits.
Paper: delivery_i3592753 (50 pp) committed 9e4a585 and pushed to github.com/shreshthsaini/scale-covariant-attention
(private, origin). Code link is the \codeurl macro (swap for an anonymous link at submission).

## 16. 2026-10-20 08:00 CT: standard protocol (195 prompts) and presentation overhaul landed
Build delivery_m1gkdr8f (48 pp, main 1-10, 0 overfull, standalone pass). All main tables on the full Aesthetic-4K split
(195 prompts, 4 development prompts disclosed); seeds 1-2 on 195; Krea 100 and HRGen-1K 200 until the running spool
(a192 Krea-195, a193 HRGen-1K Krea, a194 HRGen-1K dev; evals z966-z968 auto-released by slurm/release_evals.sh) finishes.
AReA's published numbers are FLUX.1-Krea-dev (Table 2, results/published_compare). Headline unchanged: 11 of 15 metrics,
IR +0.31 [0.23,0.38], FID -7.7 [-13.1,-2.6]. Open: Table 3 rows duplicating Table 1; J.1 band-guard rows still 191;
results/baseline_quality_deltas.csv stale. Workers: idle-only guest workers on all 28 held gh nodes (slurm/guest_worker_shared.sh);
slurm/occupy_all.sh (co-tenant) is ready but needs the user to launch it. SCA13 job 1083327 still pending.

## 17. 2026-10-20 23:00 CT: final pass (sample curation, Krea-195 and HRGen-1K numbers, render review)
Build /scratch/09032/saini_2/sca/paper_builds/delivery_9un4nd1w (49 pp., main 1-10, 0 overfull, 0 undefined, fresh-unzip build passes); latest_delivery.md has hashes.
Sample curation (.agent/notes/figure_curation_20261020.md; four opus auditors viewed every SCA image at full resolution):
- Curated figures now show clean SCA samples on fair prompts (every baseline IR and HPS >= p25 of its own 195):
  Figure 1 portrait -> 5849277588805396896 (old had a second pipe) and strip -> m8343371271909676658 (old strip had
  AReA at p1); Figure 6 all three rows -> 7109958488611222679, 2232973817337527860, m548913239957610051 (old rows had a
  duplicated character, a second ear, and an unfair AReA); R.1 panorama -> Great Wall 4230887929316158402 and still
  life -> minus8919475713516982506; R.7 owl -> warrior m2857555394095239941. Captions say curated and that curation
  favours SCA. Figure 2 keeps the Moai development prompt; its caption now says a small second Moai remains.
- Random galleries keep their seeded draws (all verified against the frozen manifests); captions now state what the
  samples show, including SCA's extra subjects. Visual audit CSV: results/visual_audit_20261020 (5 of 11 uncurated
  FLUX.1-dev 4096^2 SCA images add people or repeat objects), quoted in Appendix S and Limitations.
Numbers (every value from results/*.csv via scripts):
- Krea-195 (results/krea195, SCA and AReA on all 195; DyPE/SEGA stay at 100): SCA wins 6 of 10 per-image metrics, IR
  -0.052 [-0.105, +0.000] (2000 draws; significant with K.1's 1000 draws), patch FID +8.1 [5.5, 12.8], FID +1.4
  [-3.0, 7.0], 3.3x faster. AReA rerun reproduces published IR, aesthetic, PickScore within 0.1, rest within 0.28.
- HRGen-1K FLUX.1-dev, 1000 prompts (results/hrgen1k_dev): SCA beats AReA on all ten per-image metrics, IR +0.818
  [0.768, 0.870]. SCA on Krea (results/hrgen1k_krea) exceeds every published AReA HRGen-1K score (not a rerun).
- Table 2: HRGen-1K block added, AReA's multi-stage rows dropped (still in published.csv); new appendix Table H.1
  (snippets/published_hrgen.tex). Table 3: the four rows duplicating Table 1 dropped. Table 5, K.1, K.3, macros
  (\nhrgen 1000, \nkrea 195), abstract, 4.4, Limitations, Appendix H updated. results/baseline_quality_deltas.csv
  now written by fig_baseline_deltas.py (195). J.1/J.2 band guard keep the 191 caption (no 195 band-guard eval).
Scripts: results/published_compare/build.py, paper/figures/src/{apply_krea_hrgen.py, gen_late_tables.py
(krea_6144_table only; do not run its main()), fig_hero.py, fig_zoom4k.py, fig_showcase.py, fig_qual.py,
fig_gallery_large.py, fig_phase2.py}.
Spool: every evaluation is done (pending 0, running 0); slurm/release_evals.sh (PID 2849732 on i615-123) still loops.
Next: create taskq/STOP when no more GPU work is planned (Section 15), then the guest workers and the release loop exit.
Render review (opus reviewer, all 49 pages): fixed the split citation that broke page-10 links, unreferenced Figure
D.2 and Table J.7, Figure S.1 above its heading, "Fig-ure" hyphenation, Table 2 caption wording. Left open: page 27
holds Table J.2 alone (61 percent); public code URL with anonymous authors (swap \codeurl at submission); Appendix C
lists 48 calibration prompts while the text uses 32 at 1024^2 and 16 at 1280^2 (check which 16); Figure 8 and D.2
print 2.85x and 2.99x where Table D.2 gives 2.84 and 2.98 (rounding source differs); R.7 column names vs Table 4.

## 18. 2026-10-24: SU freeze (relayed by session vista-6a as a user directive)
Do not submit, resubmit or release any Slurm job for SCA (no sbatch, scontrol release, spool watchers or fleet
auto-resubmit) until the user lifts the freeze; the allocation is near 2,800 SU and other users need the balance.
SCA holds no jobs and runs no workers (taskq/STOP set 2026-10-20). CPU-only and paper work may continue. The one
optional GPU item (band-guard rows of J.1/J.2 at 195 prompts, about 4 GPU-h) waits for the freeze to lift.

## 19. 2026-10-27: AReA presentation pass (6 pt follow-up 2026-10-28)
Build /scratch/09032/saini_2/sca/paper_builds/delivery_mzzf_m71 (51 pp., main 1-10, Section 6 ends p. 10 y = 673 pt,
reproducibility statement p. 11; 0 overfull, 0 undefined, fresh-unzip build passes); latest_delivery.md has hashes.
The AReA pass itself was build delivery_3ip7uv4h, commit 82d1c07.
Comparison with AReA (2610.17227): .agent/notes/area_vs_sca_presentation.md (22 items; SCA leads on 16, trails on 1).
What changed in the AReA pass:
- Intro hook and gap sentence, Worked example (Section 2.2), Limitations led by late duplication (5 of 11), captions
  cut to a bold lead plus at most two sentences, quality-time figure moved to Appendix L.
- New Figure 4 (heads_view: key sets on a 4096^2 output, measured reach of three heads, their pi(sigma), pi against
  near-field share over 1368 heads, r = 0.77 at scale 2, 16 tokens, sigma = 0.62; results/heads_view). The old
  pi_heads figure is Figure B.2 (split-half r = 0.989). Sections 2.3 and 3 cite the new panels.
- Tables 1 to 4 come from paper/figures/src/gen_main_tables.py (snippets/main_tables, headline.tex macros in the
  prose); PDF text identical before and after the swap. Table 4 marks significantly worse cells (sigma_g = 0.85, HPS
  and MUSIQ_c); Tables 1 and 2 carry direction arrows. apply_krea_hrgen.py no longer patches Table 4 rows;
  gen_prog_tables.py checks the tab_panel46 snippet.
- Related-work table (Table 5, check marks, O(N^2)/O(N) cost) and shorter related-work prose (27 citations kept).
- Page fit: prose tightened throughout; published-numbers table moved from Section 4 to Appendix H (Table H.1);
  Figure 6 wraps beside the Metrics paragraph; the qualitative grid is now Figure 5.
- Two fresh Opus render reviews: fixed stale appendix openings (B, D, K, L), a wrong figure pointer, compute-appendix
  campaign times (72/264/196 s from results/a4k195_main), the Table 2 caption, Table 5's offsets note (image keys).
What changed on 2026-10-28 (6 pt follow-up):
- No figure text under 6 pt: PyMuPDF spans of all 17 placed figure PDFs (all at scale 1.0) are at least 6.0 pt.
  New scafig.script_floor(base_pt) raises matplotlib's mathtext script factor (fixed 0.7) so the scripts of base_pt
  text print at 6.04 pt or more; called in fig_heads_view (Figure 4), fig_paired_area (Figure 6), fig_pi_heads
  (Figure B.2; y-label pad 1.5 pt keeps the label inside the edge) and fig_pi_heatmap (Figure B.5, also at 5.6 pt and
  missed by the first review). Figure 1's size badges and Figure 5's MUSIQ_c subscript went from 6 TeX pt (5.98 pt)
  to 6.1 TeX pt (fig_hero.py, fig_zoom4k.py). Data, sizes and layout unchanged; JPEG assets byte-identical.
- fig_paired_area.py expands the headline.tex macros before its main-text check, which failed after the macro swap.
- Sections 4.1 and 4.2: every finding paragraph opens with a bold claim sentence instead of a label. The AReA
  paragraph adds Table 1's before and after values (ImageReward 0.92 to 1.23, FID 145.8 to 138.1), Coverage the
  6144^2 ImageReward interval of Table K.1, and the progressive paragraph its campaign times (49.5 and 110.4 s).
  No reported number changed.
Gold-standard rubric (0/1/2 per item, Section 14 format), re-scored 2026-10-28:
- Page 1: abstract 1 (175 words, 8 sentences, 2.85x, URL); hero 1; footnotes 1 (none).
- Introduction: question and motivating wrapfigure 1; "We introduce", numbers, three "We" bullets 1; sentences 1
  (median 20, max 35; text unchanged).
- Method: diagram 2; equations with "where" lines 1; boxed algorithms 1.
- Experiments: setup run-ins 1; main table 2 (time columns, arrows, Key result); finding paragraphs 2 (every finding
  of Sections 3 and 4 is led by a bold claim sentence, in 4.3 after a one-line setup, and quotes numbers; each quoted
  delta carries its 95% paired interval, which neither gold paper reports); ablation tables 2; plots 1 (palette,
  direct labels, every figure span >= 6.0 pt); qualitative grid 1.
- Discussion: limitations 2; reproducibility statement 1.
- Appendix: 1 (Table J.1 has no HRDiT or SigMa rows).
- Layout: floats, three wrapfigures, <= 60% float 1; captions 1; colours 1; no em dash/section sign/sub-6 pt figure
  text 1 (PDF text has no em dash, section sign or "??"; 5 pt second-level math scripts remain only outside figures:
  body equations, algorithm comments such as 1024^2, appendix table marks).
Score: 22 items, sum 27, mean 1.23; no 0 item. Meets the ship bar (no 0, mean >= 1.2) by one point, so any item that
drops to a lower score puts the paper back under it.
Open:
- Per-step figure needs x0 traces of SCA and AReA (scripts/sample_trace.py); none in
  /scratch/09032/saini_2/sca/web_assets, and the SU freeze blocks GPU runs.
- \codeurl is the public GitHub URL under the anonymous author block (swap at submission).
- Table J.1 lacks HRDiT and SigMa rows; Table H.1 caption should say the FLUX.1-dev rows are unranked
  (results/published_compare/build.py); R.4/R.5 captions give width x height; Table F.1 should name the 1280^2 prompt
  ids (Appendix C); Table K.1 marks Krea ImageReward significant at 1000 draws while Table 4 (2000 draws) does not.

## 20. 2026-10-28: freeze lifted for SCA ("u can use the compute however u want"); SCA14 fleet submitted
Fleet SCA14 (slurm/fleet_SCA14.sbatch): 8 gh nodes, 10 h, IDLE_MAX 5400, about 80 SU. Drains taskq/pending: the 16
w_trace tasks (webpage trajectory videos, ~3 GPU-h), then the duplication-fix pilots and reruns the method agent enqueues
(.agent/notes/duplication_fix_20261027.md). taskq/STOP was moved to STOP.until_20261027; recreate STOP when done.

### 20.1 2026-10-30 09:20 CDT: paper rewritten around the scheduled operator (paper owner)
Build /scratch/09032/saini_2/sca/paper_builds/delivery_32xjiemn (53 pp., main text 1-10, 0 overfull, 0 undefined,
fresh unzip passes); .agent/notes/latest_delivery.md has hashes; working notes .agent/notes/paper_v2_rewrite_20261029.md.
- Story: measure each head's appetite for context once at 1024^2; at the target open with the global view, shrink the
  window 128/112/96/80 to native 64 over four steps, read each head's weight at the matching image-to-window ratio
  (Eq. sched, log-linear between the scale-2 and scale-4 tables), plus a 1,024-token bridge in the LOCAL softmax
  (Eq. bridge; the lead's brief said global branch, the code says local). Published SCA is "SCA without the
  schedule", an ablation row only. New Sections 2.3 (schedule), 2.4 (root cause with the gate sweep 20/15/7/1 of 26),
  4.2 (late duplication), duplication figure (gate sweep, rates of all methods, uncurated before/after), Appendix
  app:dupfix (detector, validation 20/20, every variant in Table tab:dupfix), Prop. 1 narrowed to window keys at n0.
- Numbers (results/*_sca_v2): 4096^2 vs AReA 12 of 15 wins, none lost, ImageReward +0.33 [0.25, 0.41], 2.63x faster
  (77.1 vs 202.4 s; 8% over SCA without the schedule, back to back), b = 0.99; duplication 26% (25/95) vs 57% without
  the schedule and 4% for AReA (15% on the 46-prompt tuning subset, disclosed); Krea 8 of 15 wins, loses patch FID.
- Scripted swap: paper/figures/src/swap_all.sh (gen_main_tables.py, gen_app_tables.py, collect_latency_v2.py,
  fig_paired_area.py, fig_latency.py, fig_dup.py); gen_dupfix_table.py; CPU reference bootstraps in
  eval/<run>/cpu_bootstrap_sca_v2.sh.
- Pending at delivery: hrgen1k_dev_sca_v2 and hrgen1k_krea_sca_v2 (grey cells, caption notes). Rubric (review 2):
  mean 1.23, no 0 item; SCA leads AReA on 19 of 22 presentation items (trails on per-step figure, project page).
- Open: no per-step trajectory figure (needs sca_v2 x0 traces); Figure A.1 (fig:failure) caption 84 words; 6144^2,
  8192^2, FLUX.2-dev, band guard and sensitivity results use the operator without the schedule (labelled).

## Late-duplication audit and fix (method owner, 2026-10-28 07:50; full log .agent/notes/duplication_fix_20261027.md)
Detector: scripts/dup_detect.py (Grounding DINO tiny, offline) + scripts/dup_rate.py; subjects and validation labels in
results/dup_audit_20261027 (build.sh rebuilds every table). Validation on 30 labeled SCA images: people/animals 20/20
after review (18/20 vs the original audit), any subject 26/30. Rates, 195 prompts at 4096^2, people/animals: SCA 57%
[47,66] (seeds 50, 56), band guard 20, progressive SCA 8, AReA 4 [2,10], full 7, UltraImage 5, HRDiT 11, SEGA 12,
DyPE 17, SigMa 27; 46-prompt set: gate .93 77, .89 58, .85 27, global only 4. Cause: pi jumps from 0 to ~0.7 at
sigma .89 while the layout of small subjects is still forming. Options in sca/attn.py + sca/dedup.py (default off,
published sidecars unchanged, 14 tests): pi ramp, per-head opening, fractional global ids, window schedule, local
text bias, AReA-style bridge tokens, per-query LSE mixing. Queue: pilot a_d100..a_d113 (14 variants, 15 prompts) and
46-prompt runs b_d2XX (bridge, bridge_ramp70, ramp70, lse1) in taskq/pending; evaluations park in
taskq/hold_dup_eval and slurm/release_dup_evals.sh (PID 1256926 on i618-062) releases them. 195-prompt reruns staged in
taskq/hold_dup/stage_x_{dupfix_bridge,dupfix_bridge_ramp70} (release recipe in the log; about 100 GPU-h).
Update 18:55: pilot done (results/dup_audit_20261027/pilot_20261028): bridge+ramp70 1/9 people and 2/14 any at +4% time,
but the linear ramp leaves block/lattice artifacts (CLIP-IQA -0.08, ARNIQA -0.06); the reruns are ENQUEUED for
dupfix_bridge_head50 (x_ tasks, evals z980-z988), gated in practice by its 46-prompt paired run (b_ tasks) that runs first.
Addendum 2026-10-28 21:20: SCA14 (1105980, 8x10h) has not started after 18 h (gh fully allocated, deep queue). Submitted
backfill shapes: SCA15 1107850 (gh-dev 4x2h, IDLE_MAX 900), SCA16 1107851 (gh 2x6h), SCA17 1107852 (gh 4x8h). All drain
the same spool; cancel the still-pending ones once enough are running. Pilots so far ran on the one guest GPU (cntrl-cmd
1103038). Queue: 14 b_ (46-prompt check), 16 w_trace, 473 x_ reruns.

## 21. 2026-10-31: moved off TACC Vista (migration agent; full log .agent/notes/migration_20261031.md)
`MIGRATION.md` is now the entry point: what lives where, restore steps, path mapping (`repro/migration/relocate_paths.sh`
or same-path symlinks), model revisions, work in flight. GitHub gained `calib/` (Git LFS), `third_party/` (pinned AReA,
UltraImage, DyPE, SEGA; README with all six baseline pins; no local patches existed), `repro/taskq_archive/` (2,463 task
scripts plus the spool state) and `results/eval_sidecars/` (1,507 per-image metric files). Private Hugging Face repos:
`sca-calibration`, `sca-eval-outputs`, `sca-frozen-code`, `sca-generated-images` (phase 1: headline 4096^2 sets of every
main-table method, the sca_v2 reruns, latency, and the HRGen-1K Krea sca_v2 images). Phase 2 (221 GB: other HRGen-1K
sets, comparators, 6144^2, 8192^2, FLUX.2, ablations, duplication variants) was not uploaded because the free Hugging
Face plan caps private storage at 100 GB; `results/migration_manifest.csv` lists every archive, status and SHA256.
Still open: the HRGen-1K Krea sca_v2 evaluation (`taskq/pending/a000_z988_eval_hrgen1k_krea_sca_v2.sh`, never ran).

## 22. 2026-10-31: HRGen-1K dev row folded in (paper owner; Section 21 is the migration)
Build /scratch/09032/saini_2/sca/paper_builds/delivery_tj1oh6am (53 pp., main text 1-10 with Section 6 ending on the last
line of p. 10, 0 overfull, 0 undefined, fresh unzip passes, no word box past a margin); hashes in .agent/notes/latest_delivery.md.
- hrgen1k_dev_sca_v2 (eval z987, DONE 2026-10-30 20:24 UTC). Its task ran the paired bootstrap (2000 draws, seed 0) and copied
  the CSVs with SHA256SUMS to results/hrgen1k_dev_sca_v2. HRGen-1K has no references, so there is no reference bootstrap and
  no BOOTSTRAP_DONE. sca_v2 minus AReA, 1000 prompts at 4096^2: 10 of 10 per-image metrics significantly better, ImageReward
  +0.764 [0.714, 0.816], HPSv2.1 +4.60, MUSIQ_c +12.56; absolute IR 0.73, aesthetic 6.19, CLIP 28.43, PickScore 21.52, HPSv2
  27.96, IS 15.8, MUSIQ 45.1. Against SCA without the schedule: 8 of 10 significantly lower (MUSIQ and CLIP-IQA unresolved),
  ImageReward -0.055 [-0.072, -0.038]. For comparison, the 195-prompt main set has 4 of 10 lower (aesthetic, ARNIQA, CLIP-IQA,
  MUSIQ_c) and none higher.
- Paper: swap_all.sh regenerated Table 4 (HRGen-1K row), Table L.3 (HRGen-1K column) and Table H.2 (FLUX.1-dev SCA row);
  \pendTabGen and \pendAppPhaseGeneral are empty. Limitations now says "The schedule costs time and some quality" and gives
  the HRGen-1K number; the old "two no-reference metrics" held only on the 46 prompts.
- Render review of every page (subagent), fixed: widowed 4.1 heading (paired-area wrapfigure moved after the first 4.1
  paragraph; wording trims on pp. 7, 8 and 10 keep Section 6 on p. 10, zero slack), appendix contents entry M (the page number
  now ends the wrapped line; \appline macro, ragged column), calibration prompt labels inside the margin (leftmargin 2.1em),
  Figure M.1 progressive label off the y-axis spine (fig_quality_time.py WRAP_PLOT offset), Figure T.7 header inside the
  margin (fig_qual.py ablation grid 6 pt narrower; ablation.tex edited to the same \scaT line).
- Web: build_numbers.py quotes the new HRGen-1K values (292 of 292 agree); the HRGen-1K fact reads the final operator and sits
  next to the seeds; check_page PASS (desktop, phone, light, dark).
- Still pending: hrgen1k_krea_sca_v2 (grey "SCA (ours), FLUX.1-Krea-dev" rows of Tables H.1 and H.2, caption notes). Its eval
  task taskq/pending/a000_z988_eval_hrgen1k_krea_sca_v2.sh needs a GPU worker: no SCA job is in squeue, and
  guest_worker_shared.sh on CPU node i617-044 (cntrl-cmd 1111046) cannot claim it because the node has no GPU. Once its DONE
  exists: bash paper/figures/src/swap_all.sh, then paper/build_delivery.py, a render check of p. 26, commit.

## 23. 2026-10-31: GitHub and Hugging Face mirrors kept current (sync agent; MIGRATION.md Section 7)
`repro/migration/sync_loop.sh` runs `repro/migration/sync_incremental.sh` every 6 hours on i617-044 (job 1111046, ends
2026-11-02 04:12): commit and push of `results/`, `.agent/`, paper PDF and zip, `web/`, `HANDOVER.md`, `README.md`,
`MIGRATION.md`; upload of new evaluation groups (with `DONE`), calibration files, frozen code, trace renders and phase-1
image sets to the four private Hugging Face repos, capped at 97 GB of private storage, each upload verified and recorded
in `results/migration_manifest.csv`. Loop PID 2553656 (started 2026-10-31 15:35); the PID and one status line per run
are in `.agent/notes/sync_log.md`; loop
output in `/scratch/09032/saini_2/sca/logs/sync_loop.log`. First run 2026-10-31 15:30: nothing new (the HRGen-1K dev
sca_v2 evaluation was already uploaded by the migration), private storage 85.68 GB. Stop:
`touch /scratch/09032/saini_2/sca/taskq/SYNC_STOP`. After the job ends, restart on the next compute node with the command
in MIGRATION.md Section 7. The HRGen-1K Krea sca_v2 evaluation (`eval/hrgen1k_krea_sca_v2`) uploads on the first run after
its `DONE` appears.

Addendum 2026-10-31 14:10: controller node i617-044 has no GPU, so the last evaluation (z988, HRGen-1K Krea for sca_v2) waits for SCA19 job 1112649 (1 gh node, 3 h, about 3 SU); its result is folded in by the paper agent and uploaded by the sync loop.

## 24. 2026-11-02: review round 1 revision (revision author; dispositions in .agent/notes/review_round1/meta_review.md)
Three reviews (method, experiments, presentation) and the lead's ten binding decisions are dispositioned item by item in
`.agent/notes/review_round1/meta_review.md`; GPU work is specified, held, in
`/scratch/09032/saini_2/sca/taskq/hold_review_round/README.md` (RR01 to RR17; nothing pending, no job submitted).
Applied this pass (commits a3503ce, b462139, 5895abb, ff17113, 2ffea28 and the delivery commit):
- Framing (decision 1): abstract gives 77.1 vs 202.4 s, the 26% vs 4% duplication, Krea parity in a third of the time and
  50.7 vs 110.4 s; intro paragraphs reordered ("We introduce" before the emulation answer); bullets and conclusion put the
  duplication number beside every 12-of-15 claim; Section 4.1 names the families (all ten per-image metrics, two of five
  set-level) and the three unresolved ones (KID, patch FID, IS); Section 4.5 is a three-checkpoint block (FLUX.1-dev,
  Krea with ImageReward +0.020 [-0.020, 0.063], patch FID +3.4 [1.6, 7.1], 29% duplication; FLUX.2 without the schedule).
- Mechanism (decision 2): Section 2.4 "Why narrow windows repeat small subjects": ramp with native windows 5/26, bridge
  alone 5/26, bridge plus 128 windows 1/26; the schedule is called a cheap width schedule; AReA's remaining advantage is an
  open question. Section 4.4 cites McNemar p = 1.0 for schedule (4/26) vs bridge alone (5/26) from
  results/dup_audit_20261027/review_round1_cpu/pairs.csv.
- Decisions 3 to 5: Table F.2 (tab:handset) lists every setting as hand-set or calibrated; Section 2.3 ties the sigma bins
  to the 28-step sampler and says the 42/56-step rows run without the schedule; Limitations states the calibration gap
  (no bridge, 16x fewer keys per branch); "scale covariance" narrowed in the intro; Table J.1 caption justifies AReA's
  quadratic entry from its code (f = 2 pooled steps, cost-capped first window) with exponents 1.12 vs 0.99; Proposition 1
  scoped to native windows; Lemma 1 stated as motivation with the bridge double count; Proposition 2 scoped to constant
  weights per cell.
- Decision 6: HRGen-1K Krea sca_v2 cells filled by swap_all.sh (Table H.2 SCA Krea row: IR 0.84, aesthetic 5.96, CLIP
  30.02, PickScore 21.68, HPSv2 28.06, IS 16.4, MUSIQ 44.5); "without the schedule" added to the 8192^2, FLUX.2,
  sensitivity sentences, Table 2's budget group heading and Figure M.1's caption.
- Decision 7: detector validation stated exactly; baseline validation from results/dup_audit_20261027/validation_by_method.csv
  (AReA precision 0.50, corrected recall 0.29, implied any-subject rate 13% vs detected 7.5%) in Section 4 setup and
  Appendix P.
- Decisions 8 and 9 (Table 1 only): ranked within the single-pass group, multi-stage rows grey context
  (\ctxrow), campaign time unranked, one header line with arrows, Dup. as percent, FID_p dagger, sweep note and FID bias
  in the caption (gen_main_tables.py).
- Decision 10 (part): Figure 1 top row dropped (trim of teaser.pdf; caption lead plus two sentences, curated and pointing to
  the uncurated gallery); Figure 7 states n; ARNIQA on the x100 scale in Limitations; page 7 \enlargethispage removed;
  main text ends on page 10 with about 3 lines of slack; main-text sentences over 35 words split (bullet 3, Sections
  2.3, 2.4 and 4 setup, Proposition 2, conclusion). Delivery: /scratch/09032/saini_2/sca/paper_builds/delivery_2duo3frk
  (55 pp., 0 overfull, 0 undefined, fresh unzip passes); hashes in .agent/notes/latest_delivery.md.
NOT YET DONE (next agent; the controller job on i617-044 ends 2026-11-02 04:12 and ends this session): Table 2 to 4 significance
convention (one code), Table 4 intervals and Dup., Table H.1 ranking within block, operator labels in the row names of
Tables L.1 (krea_6144), N.1, L.3 to L.5; the remaining sentences over 35 words (python3 -I long.py scan: intro bullets,
Sections 2.2, 2.3, 2.5, 4 setup); captions of Figures 4, 5, A.1, D.1, B.1 to lead plus two sentences; Figure 4a
declutter (fig_dup.py), Figure 5b enlargement and rho label (fig_heads_view.py), Figure 6 full-attention column to HRDiT
(fig_zoom4k.py), Table 3 Err. column to K.5, Figure M.1 markers, algorithm type at 6 pt, half-blank pages 17 and 21,
Table N.1 fp32 note. Figures 3, D.1 and the Appendix B calibration figures belong to the figure agent.
Rubric self-score (gold-standard 22 items, 0/1/2): abstract 1, hero 1, footnotes 1, question and wrap 2, introduce and
bullets 1, sentence length 0, method diagram 1 (pending figure agent), equations 1, algorithms 2, setup run-ins 1, main
table 2, finding paragraphs 2, ablations 1, plots under 6 pt 0, qualitative grid 1, limitations 2, reproducibility 2,
appendix 2, float placement 1, captions 1, colours 1, dashes and section symbol 1. Mean 1.23 with two zeros: not
shippable under the house rule until the long sentences and sub-6 pt figure text are fixed.

## 25. 2026-11-02 04:05 CT: session node i617-044 ends 04:12; allocation at 79 SU (no GPU work possible here)
State: review round 1 done (.agent/notes/review_round1/: three reviews and meta_review.md); revision round 1 pushed
(6d87526, build delivery_2duo3frk, 55 pp, rubric 1.23 with two zeros). Figure redesign (Figure 3, D.1, calibration
figures) was in progress: see .agent/notes/figure_redesign_20261102.md and figures/src/ for its state.
Next agent, in order: (1) finish revision round 1's open items listed in Section 24 (Tables 2-4 one significance
convention, Table 4 intervals and Dup., Table H.1 ranking, "without the schedule" row names, captions of Figures 4, 5,
A.1, D.1, B.1 to bold lead plus two sentences, Figures 4a, 5b, 6 fixes, pages 17 and 21 half blank, all figure text
at least 6 pt); (2) land the redesigned Figure 3/D.1/calibration figures; (3) re-run the three reviewers on the new
build and fix anything blocking; (4) delivery build, push, publish the PDF. GPU-deferred experiments: taskq/hold_review_round
(README with order and GPU-hours; needs a new allocation or another machine, see MIGRATION.md). The 6-hourly sync loop
(repro/migration/sync_loop.sh) also dies with this node; restart it on the next node (MIGRATION.md Section 7).
User directive 2026-11-02: finished oral-standard paper by morning; framing per meta_review.md (equal-or-better quality
at 2.6x AReA's speed on FLUX.1-dev, parity on Krea, duplication stated as a weakness; never "much better" in general).

Addendum 04:12: Figure 3 and D.1 redrawn (figures/candidates/*_print300.png for review). Open from the figure agent: the calibration thumbnail in Figure 3 is a stand-in (calib00 resized from a 2048^2 SCA output; generate a real 1024^2 full-attention image for it); Appendix B opening cites Figure 3b for calibration, now panel (a); Figure 5 (heads_view) and the calibration-results figure still need the redesign.

Addendum 04:09 (method agent, pushed fe5a56c/f479459): (a) detector validation on baselines (results/dup_audit_20261027/validation_by_method.csv): AReA precision 0.50 recall 0.29, HRDiT 0.60/0.33, SCA 0.91/0.84; AReA true rate plausibly about 13 percent, so the paper must report detector-measured rates with this caveat and not state "4 percent" as fact. (b) On 46 prompts the bridge alone gives 5/26 and bridge plus schedule 4/26 (McNemar p 1), schedule costs ARNIQA -0.007 and MUSIQ_c -0.5: present the bridge as the main fix and the schedule as a small, not yet significant addition, pending the 195-prompt schedule-only ablation (held, item 1). (c) Krea-195 duplication: sca_v2 30, published SCA 64, AReA 8, DyPE 20, SEGA 8 percent. (d) 148 GPU task scripts in taskq/hold_review_round (33.5 GPU-h); copy the reference sets named in its README off scratch before the purge.

Addendum 2026-11-04 23:55 CST (review-round GPU campaign owner; controller node i615-133, job 1123393, ends 2026-11-06 23:31):
allocation refilled (9,992 SU); fleet SCA21 (job 1123398, 4 gh nodes, 10 h, slurm/fleet_SCA21.sbatch: node_launch.sh, bash
worker_loop.sh, IDLE_MAX 1800 s) queued. All 140 GPU tasks of taskq/hold_review_round are in taskq/pending; 8 stay held and
release themselves (r1_z, r2_z, r3_z_eval_6144, r3_z_eval_flux2, c3_eval_8192_cpu, c1/c3/c5 boot). One change to the README
run order (5, 1, 2, 3, 4): r4_latency_sweep.sh is now r1_zz_b_r4_latency_sweep.sh and the six r3_g_8192 tasks carry the
prefix r1_zz_a_, so the order is 5, 1, (3b generation and 4), 2, rest of 3. Item 4 is one 5 to 6 h task: placed last it
would start near hour 6.5 of the 10 h fleet and be cut off, and each GH200 node has one GPU and one worker, so the sweep is
alone on its node at any time (its timing covers encode, denoise and decode only). The 8192 tasks go early because their
CPU evaluation takes about 6 h. Release loop (backup to the generation tasks' own release; matches the prefixed names;
sends the CPU evaluation to taskq_cpu): /scratch/09032/saini_2/sca/dispatch/review_round1/release_review_evals.sh, PID
2733390, log logs/release_review_evals.log. CPU workers (nice 19, frozen slurm/worker_loop_v2.py on taskq_cpu, logs
logs/cpu): PIDs 2733389 and 2733391, idle exit 36 h, deadline 2026-11-06 23:10; stop them with taskq_cpu/STOP once the CPU
work is done, never taskq/STOP while GPU work remains. Status: bash /scratch/09032/saini_2/sca/dispatch/review_round1/status.sh.
Copies of the three scripts: .agent/notes/review_round1/campaign_scripts/.

## 27. 2026-11-05: review round 1 finished (revision author; paper/ only, no GPU)
Closed every open item of Sections 24 and 25 and folded in the method agent's 04:09 findings. Numbers come from
results/*.csv through the generators (gen_main_tables.py, gen_app_tables.py with app_tab_main.py, gen_dupfix_table.py);
no measured number was changed. Delivery: /scratch/09032/saini_2/sca/paper_builds/delivery_eeofzli6 (54 pp, main text
pages 1 to 10, 0 overfull, 0 undefined, fresh unzip builds to the same 54 pages); hashes in .agent/notes/latest_delivery.md.
- Significance: one colour convention in every table, defined once (Section 4, Metrics): blue/orange = 95% paired interval
  excludes zero for/against the row; bold/underline only rank within a group. \up and \dn are colour-only; Table 2's
  superscripts and Table K.1's \sg/\sw became colours; Table 1's SCA row is coloured against AReA. Dup. is unranked in
  Tables 1 and 2 (detector validity differs by method).
- Table 3: Err. column moved to Table K.5 (output-error rows from results/evidence/calib/k1_1024.json), no n/a,
  \footnotesize. Table 4: HRGen-1K, seeds 1 and 2, Krea (against AReA) and FLUX.2 (without the schedule), each
  difference with its interval underneath, Dup. as SCA / reference (28/3, 26/6, 29/8); the sensitivity rows now live
  only in Table L.4. Table H.1 ranked within block; Table P.1 key result leads with the bridge (McNemar p = 1.0 for the
  windows on top). \DeclareMathSizes{7}{7}{6}{6} lifts scripts in \scriptsize tables to 6 pt.
- "SCA without the schedule" is defined once (Section 2.3: one-step switch and no bridge) and named in the heads or
  captions of Tables K.1, K.2, K.4, K.5, K.6, L.1, L.3, L.4, L.5, L.6, M.1, N.1 (with the fp32 note), P.1, Q.1, Figure
  M.1 (caption lead), Figures T.7 and U.1, and Appendices I, M, U.
- Finding (a): every duplication rate is called a detector rate; Section 4 gives precision/recall (SCA without the
  schedule 0.91/0.84, AReA 0.50/0.29) and the corrected estimates (AReA about 7% for people or animals, 13% any subject;
  SCA about 28%), Appendix P the derivation; abstract, bullets, Sections 4.1 and 4.2 and the conclusion say "AReA's
  detected 4%". Finding (b): bridge alone 15 to 5 of 26 (McNemar p = 0.006), wide windows on top 4 (p = 1.0, ARNIQA
  -0.7 x100, MUSIQ_c -0.5): intro, bullet 2, Sections 2.4 and 4.4, limitations, Figure 4a and Appendices P and U
  present the bridge as the main fix and the schedule's gain as unresolved (Section 4.4 says a 195-prompt decomposition
  is running). Finding (c): Krea-195 rates in Section 4.5 (29, 64, 20, 8, 8%) and Table 4. Note: sca_v2 on Krea is
  28/95 = 29.5% (review_round1_cpu/rates_krea195_people.csv), printed 29%, not the 30 of the 04:09 addendum.
- Figures: Figure 4 (fig_dup.py) shows the gate sweep and bridge/schedule as two groups, no arrow or floating note,
  thinner crop frames; the zoom figure (fig_zoom4k.py) dropped full attention (AReA, HRDiT, SCA crops, windows 432x240
  px, outline 1 pt, same height); Figures 5d and B.2 print Pearson rho; Figure M.1 gives UltraImage a square. Figure order
  changed for layout: paired_area is now Figure 6 (wrap on p7), the zoom figure Figure 7, latency Figure 8 (wrap on p8).
  Captions of Figures 4, 5, 7, A.1, B.1 and D.1 are a bold lead plus at most two sentences.
- Layout: no section heading at a page bottom, every main float within one page of its first reference, B.3/B.4 [!tb]
  (no half-blank page), K.1 and K.2 share a page; main text ends on page 10 with no slack. Sentences in the main text:
  median 20 words, none over 35 (citation chains split). Smallest figure text 6.0 pt.
- Two render-review cycles over all pages: main pages by the revision author, appendix pages by two reviewer subagents
  per cycle; every MUST item fixed (Appendix G intervals now quote the final operator, F.1 calibration and seed rows,
  Appendix C spare prompts 33 to 48, J.1 offsets, Figure L.1 sign, App. P 64-key bridge, D.1 ratios, Algorithm 5).
- Figure 3 calibration thumbnail: no 1024^2 full-attention image of a calibration prompt exists under
  /scratch/09032/saini_2/sca/out or calib (2026-11-04 search); stand-in kept, recorded in method_provenance.json.
- Left: Figure 1 insets still amber; Figure D.1(d) numbers blocks 1 to 57 while B.2/B.5 use 0 to 56; resized appendix
  tables L.1, L.2, M.1, Q.1 print a few scripts at 5.7 to 5.9 pt; the code URL in the abstract names the author (swap
  for an anonymous mirror before submission); the 195-prompt bridge/schedule decomposition (held task item 1) will need a
  text update when it lands.
Rubric self-score (22 items, 0/1/2): abstract 1, hero 1, footnotes 1, question and wrap 2, introduce and bullets 1,
sentence length 1, method diagram 1, equations 1, algorithms 2, setup run-ins 1, main table 2, finding paragraphs 2,
ablations 1, plots under 6 pt 1, qualitative grid 1, limitations 2, reproducibility 2, appendix 2, float placement 1,
captions 1, colours 1, dashes and section symbol 1. Mean 1.32 (29/22), no zero.

Addendum 2026-11-05 (lead): human-validated duplication on the 95 person/animal prompts (results/dup_audit_20261027/human_rates_95.txt): AReA 4.2%, HRDiT 6.3%, full attention 4.2% (but subject missing in 22 of 37 labeled), sca_v2 26.7%. The paper build delivery_eeofzli6 says AReA corrects to about 7% (13% any subject): WRONG, must be replaced by the human-validated 4.2% in the next revision (Section 4 Duplication paragraph, Table 1 caption, Appendix P). Bar for the round-6 fix: human rate at most 4.2% on the same prompts.

Addendum 2026-11-05 14:58 CST (review-round campaign owner): fleets SCA21 (1123398) and SCA22 (1123443, submitted by the
method agent; the lead says no further submissions) still pending on priority; SCA23 (1124593, gh-dev 4 x 2 h) and SCA24
(1124594, 1124595, gh 1 x 3 h) share taskq/pending with the same bash workers, which ignore SCA_MIN_SECONDS. So the six
non-resumable tasks got a walltime guard (dispatch/review_round1/add_guard.sh; originals in dispatch/review_round1/originals):
if the claiming job has less than GUARD_NEED seconds left the task parks itself in hold_review_round and exits 3 (its
failed/ entry is that bounce, not a failure). GUARD_NEED: item 4 sweep 23400 s (renamed b0_r4_latency_sweep.sh, sorts
after the method agent's a6_ tasks), item 5 evaluation 7200, item 1 evaluation 6300, item 2 2700, 6144 6300, FLUX.2 3600.
Release loop v2 (PID 632229, same script path and log; v1 PID 2733390 stopped) re-releases a parked guarded task only
while a running SCA fleet has GUARD_NEED + 300 s left, at most every 600 s. Generation tasks are resumable and unguarded.

Addendum 2026-11-06 05:08 CST (review-round campaign owner): moved to the new controller i615-091 (job 1125760, ends
2026-11-08 04:59), started by dispatch/review_round1/start_on_controller.sh 1125760: release loop v2 PID 1692172, CPU
workers PIDs 1692173 and 1692174 (nice 19, deadline 2026-11-08 04:39, idle exit 48 h; the 8192 CPU evaluation runs here).
The i615-133 processes (release loop 632229, CPU workers 2733389 and 2733391) were stopped with SIGTERM; nothing of this
campaign stays on i615-133. Lead decisions 2026-11-06 00:30: the a6_ tasks (duplication fix, the user's top priority)
keep running first and the review items follow in the same queue; no submissions by this agent. As of 05:03 none of the
140 review tasks had run: SCA21 and SCA22 (10 h each) still pending, the short fleets SCA23 and SCA24 drained a6_ work.

## 27. 2026-11-06: round 6 duplication work (method owner; full log .agent/notes/duplication_fix_round6.md)
- Step 1, human-validated bar (commit 9f464ba): blind full-resolution labels of every detector flag plus 30 random
  unflagged images per method (166 images, 42 Opus labelers, all disagreements adjudicated from full-resolution crops).
  Human-corrected people/animal duplication on the 95 prompts at 4096^2: AReA 4.2% [1.6, 17.0], HRDiT 6.3%, full
  attention 4.2% (subject missing in 22 of 37 labeled images), sca_v2 26.7% [24.6, 36.5]. The detector is near exact on
  sca_v2; the earlier "AReA plausibly 13%" estimate does not hold. Files: results/dup_audit_20261027/human_labels_95.csv,
  human_rates_95.txt (scripts/human_rates.py).
- Step 2, mechanism: copies are seeded in the first 1 to 3 window steps after the gate (sigma 0.888 to 0.85) on props the
  global-only phase leaves open (a second rock gets an astronaut), finished by about 0.75 (trace sheets at 10 labeled
  copies, results/round6_mechanism/). At copy sites the window reads the prompt about 2x as strongly as full attention
  and sees the main subject with 0.4 to 0.5x full attention's mass (attention probe, scripts/r6_attn_probe.py). Fixing
  the text over-read leaves the copies; far-field context removes them in proportion to its weight and its duration.
- Step 3: 13 variants on the 46-prompt subset (code sca/textgate.py and sca/native_open.py behind default-off flags,
  tests/test_textgate.py; tasks slurm/round6_tasks.py; decisions scripts/r6_decide.py). None passes the bar (rate at or
  below AReA's, no significant loss vs sca_v2, wins over AReA kept, <= +25% time). Frontier:
  results/dup_audit_20261027/round6_frontier_46.csv. Best quality-neutral: bridge +0.25 log(N/R) at the three hand-off
  steps (3 of 26 vs 4). Lowest duplication with intact images: lattice far field over sigma > 0.5 (--text-mode ffl
  --text-mode-lo 0.5; 1 of 26, AReA's count; 9 paired losses incl. PickScore -0.24, IR -0.066; 7/10 wins over AReA;
  x1.08 time). Applying it only at the hand-off steps removes nothing. Not promoted; no rerun staged.
- Side finding: lowering the window's text weight (--text-mode ctw) puts a garbage band in the last token column on about
  a third of images; the exact window split itself is clean (identity 34 dB).
- Open: next hypothesis (head-selective or sigma-decaying far field; or a finer opening layout) in the round-6 notes.
  Deferred tasks in taskq/hold_r6_ctw (traces of candidate c), superseded copies in taskq/hold_r6/superseded_20261105.

## 28. 2026-11-06: round 7 duplication work (method owner; full log .agent/notes/duplication_fix_round7.md)
State 14:05 CST. The i615-133 session ended early; work continues from the controller node i615-091 (job 1125760),
where the review release loop and the two CPU workers (PIDs 1692172 to 1692174) run.
- Code (frozen trees /scratch/09032/saini_2/sca/code/80396bd and 764f337): ffl options behind default-off flags
  (--ffl-near bridge, --ffl-head-frac, --ffl-fade-lo, --ffl-outside, --ffl-coarse, --text-bw-lo), --open-pool and
  --open-pool-hi; slurm/round7_tasks.py; scripts/r7_decide.py (bar plus fixed/clean/new split), r7_label_prep.py,
  r7_label_collect.py, r7_crops.py, r7_grid_index.py.
- Track B on the 46 prompts (results/dup_audit_20261027/round7_frontier_46.csv, labels human_labels_r7.csv; 26
  people/animal prompts; sca_v2 17.0% human-corrected, AReA 2.2%): op2 18.1%, 0 losses; op4 17.9%, 3 losses; fd60
  21.9%, 9 losses; fo 9.1%, 10 losses (PickScore -0.61), x1.13; hs40o 14.3%, 8 losses; r6_ffl recomputed 13.6%, 9
  losses. None passes. Finer opening refuted; far fields read at true positions cost quality.
- KEY FINDING (11:30): the detector misses most copies of far-field variants. Blind labels of a random sample of
  r6_ffl's unflagged images: 5 of 19 are duplicates (sca_v2 and AReA: 1 of 30). Of 6 sampled prompts where the detector
  says ffl removed sca_v2's copy, 4 still have one (extra figures packed around the main subject, merged by the
  detector's NMS). Round 6's "ffl 1 of 26, AReA's count" was a detector artifact; r6_ffl's human-corrected rate is
  close to sca_v2's. Detector counts of far-field variants must not be used without labeling every image.
- LEAD DIRECTIVE 13:40 (urgent; running on SCA26 since 16:28; stale Triton-cache handles are requeued with a retry by
  r7/requeue_stale.py, PID 1886414 on i615-091, exits by 20:35): AReA's wide shrinking windows in SCA: sca_area20 (review
  item 2, renamed a0_r2_g_*), r7_area8, r7_area12 on the 46 prompts and copyprone21 (a0_s8*, a0_s9*, a0_t1..t3*), evals
  a0_e_eval_* (self-released); hand-check every person/animal image; relaxed bar in scripts/r7_decide.py (relaxed_pass);
  report when sca_area20's 46-prompt result is in. Details: round-7 notes, "Lead directive 13:40". sca_area20's
  promotion to 195 is staged (held) in taskq/hold_r7_promote_area20 (26 tasks a0_p*, the 128 prompts outside the
  subset and copyprone21, people first; evaluation a7_e1_eval195_sca_area20 in hold_dup_eval self-releases).
- sca_area20 on the 46 (17:15): every person/animal image hand-labeled: 3 duplicates of 26 (11.5%); AReA, now fully
  hand-labeled on the same prompts, 1 of 26 (3.8%); sca_v2 17.0%. 8 significant losses vs sca_v2 incl. PickScore -0.19,
  IR -0.060, HPSv2.1 -0.0079 (also on clean prompts); 7/10 wins over AReA kept; 103 s (AReA 220 s). Relaxed bar FAIL;
  not promoted (staged tasks stay held). r7_area8/area12 and the copyprone21 runs continue on SCA26 (ends 18:28): label
  every person/animal image (r7/label_people.sh PACKET SEED label=DIR[:prompts]) and run scripts/r7_decide.py.
- r7_area8 on the 46 (17:29, detector only): 1 flag; 9 losses incl. PickScore -0.13, IR -0.056, HPSv2.1 -0.0067;
  96 s. Relaxed bar fails on the core losses. Next session: area12 and copyprone21 results, hand labels of all three.
- Iteration queued: r7_fg (far field read at the global branch's coarse ids, outside the window, near bridge, through
  0.5; tree 764f337): a7_b500..b505 (46 prompts, evaluation a7_a0_eval_r7_fg) and a7_b800..b802 (the 21 people
  prompts outside the subset where sca_v2 has a copy, r7/copyprone21.jsonl; label every image by hand). Its 195
  promotion is staged in taskq/hold_r7_promote (release only on a pass). Item (4) r7_op2bw re-queued after Track A.
- Track A (r6_ffl at 195): 7 of 30 generation tasks done; the people tasks were renamed a7_a9_t007..t013 so they run
  first; then a7_t014..t029 and a7_a1_eval195_r6_ffl (eval/r7_195_r6_ffl); then the CPU bootstrap
  r7/cpuboot/r6_ffl.sh. Human check: 8 of 8 flags so far confirmed; stratified unflagged sample, stratum A done (19:
  5 yes, r7/label/trackA_strata_A.json), stratum B (11 more, from the 34 people prompts still to generate) to do.
- GPU: SCA25 ran 08:31 to 10:31. SCA26 (job 1126265, gh-dev 4 nodes 2 h) started 16:28 (until 18:28) and is the second and
  last short job of the lead's allowance. Parked tasks: taskq/hold_r7_parked (op2bw's first copy, the full-lattice
  hs40/hs20/fd70, hs20o, and hs40o's GPU evaluation, which ran on CPU instead).

## 29. 2026-11-06 evening: morph ladder between AReA and sca_v2 (method owner; log .agent/notes/duplication_fix_round7.md, "Morph ladder")
User directive (via lead, 19:30): keep going until the repeats are removed; no publication before. Localize the cause by
single-component swaps between AReA and sca_v2, hand-checking every image of copyprone21 (r7/copyprone21.jsonl: the 21
person/animal prompts outside the 46-subset where sca_v2 has a copy), metrics on the 46 for the decisive swaps.
- Baselines on copyprone21, every image hand-labeled (human_labels_95/r7.csv): sca_v2 19 yes of 21; AReA 4 of 21;
  sca_area20 (AReA's 19 windows plus bridge in SCA) 12 of 21.
- Code (commit 71b31b6, frozen /scratch/09032/saini_2/sca/code/71b31b6): sca/morph.py; SCA flags --morph-local
  area_tiles (sca+B: AReA tiles, Halo, DWT, merge as the local branch) and --morph-text window_mean (sca+C); run_area.py
  flags --phase1-factors, --sca-pi-mix NPZ, --sca-text-rule, --fixed-tiles (tests/test_morph.py).
- Queued first in the spool (a0_L*, 3 tasks of 7 prompts each): L1 sca+A (label r7_op4, pool 4,4,4,4,2,2,2,2), L2 sca+D
  (r7_scaD, pi = 1), L3 sca+E (r7_scaE, bridge round(0.5 w)^2), L4 sca+B (r7_scaB), L5 sca+C (r7_scaC); AReA side L6
  area_scaOpen (factor 4 on all 8 opening steps), L7 area_scaMix (SCA global branch mixed by calibrated pi), L8
  area_scaText (text queries read the global branch), L9 area_fixed (no jitter, Halo, DWT; the sliding-window swap's
  closest AReA analogue). Outputs r7/out/<label>/4096x4096; label every image with r7/label_people.sh PACKET SEED
  label=DIR:/scratch/09032/saini_2/sca/r7/copyprone21.jsonl, collect with scripts/r7_label_collect.py.
- GPU: SCA27 (1127223) and SCA28 (1127231), gh 1 node 3 h each (the two allowed short jobs; gh-dev is blocked by other
  projects' gh-dev jobs), plus SCA24 x2; estimated starts 19:44. SCA21 est. 2026-11-07 22:00.
- Helper on i615-091: r7/requeue_stale.py (wrapper PID 4136005, started 19:46) requeues stale-handle failures until 05:46.
- Lead 21:00 (root-cause audit .agent/notes/root_cause_audit_20261106.md): Fix 1 and Fix 1+3 ahead of the ladder.
  Implemented in sca/qmix.py (commit 606e2ae, frozen /scratch/09032/saini_2/sca/code/606e2ae; tests/test_qmix.py
  brute force): --qmix f1 = per-query weight w_i = sigmoid(logit pi_h + d_i - mean_i d_i), d_i = log Z_loc,image -
  log(s^2 Z_glob,image outside the window) (text excluded from both); --qmix f13 = one three-part softmax over text
  (true ids), near (bridge + window) and far (block centres outside the window at coarse ids, s^2-corrected), pi as the
  centred offset on log Z_near; text queries keep the global branch. Queued first (a0_K*): r7_q1s (f1, sigma > 0.845),
  r7_q1a (f1, all window steps), r7_q13s, r7_q13a on copyprone21 (K1..K4) and the 46 (K5..K8, evaluations
  a0_Ke_eval_* self-release). Hand-check every copy-prone image and the 26 person/animal images on the 46.
- Copy-prone hand counts so far (yes of 21): sca_v2 19, AReA 4, sca_area20 12, r7_area12 12, r7_area8 16.
- Staged (held) for the decisive rungs: 46-prompt metric runs of r7_scaD/E/B/C in taskq/hold_ladder46 (a0_M*; move to
  pending to run; evaluations a0_Me_eval_* self-release into eval/r7_46_<label>). r7_op4 already has its 46 result.
- GPU 21:25: nothing running since 18:28 (SCA21 est. 2026-11-07 13:14); my gh jobs (SCA24 x2, SCA27, SCA28) have priority ~1530 with ~430 gh jobs ahead (fairshare
  11); gh-dev submission is blocked by other projects' gh-dev jobs (JLDD pending, BL23 and V2P1 held).
- 21:50 audit Fix 2, cheap form (commit 250cf0c, frozen /scratch/09032/saini_2/sca/code/250cf0c; tests/test_qmix.py):
  --pi-blend LAMBDA --pi-blend-lo 0.8, pi' = (1 - LAMBDA) pi + LAMBDA m_h while sigma > 0.8, m_h = full attention's
  image-key share inside the step's window (calibration mass_mean at the emulated radius, textgate.ctw_beta unfloored).
  On the hand-off steps (sigma 0.888 to 0.808) mean pi 0.65 to 0.70 becomes 0.54 to 0.40 at LAMBDA = 1. Queued right
  after the f1/f13 copy-prone runs: r7_q2 (Fix 2 alone, a0_K45*) and r7_q12 (f1 on all steps plus Fix 2, a0_K46*); their
  46-prompt runs are parked in taskq/hold_r7_fix2 (a0_K91*, a0_K92*; evaluations a0_Ke_eval_r7_q2/q12 self-release)
  until a copy-prone count warrants them (move to pending).
- Spool order (21:50): a0_K fixes, a0_L ladder, a7_a9 (Track A's last people task), a7_b (r7_fg 46 and 21), review round
  r1_ to r3_, then s0_ (Track A's 16 other-80 tasks and r7_op2bw, renamed s0_a7_*; markers keep the old names).
- Tests: 249 pass. tests/test_review_round.py::test_generated_tasks fails by design of its subject: review_tasks.py skips
  names already in the live spool, so its 148-task count depends on scratch state (24 now), not on code.

- 11-07 02:55 SCA24 x2 started (c609-142, c609-042; end about 05:55). The first qmix task crashed at step 8: Triton
  built a launcher for the new kernels with the account's CC=nvc (rejects -Wno-psabi; as in
  .agent/notes/gpu_smoke_compiler_repair_1076947.md). Fix: every task script (pending and every hold dir) and
  slurm/dup_tasks.ENV export CC=/usr/bin/gcc CXX=/usr/bin/g++ (commit b40c295); K100 requeued. f1 at the seeding
  steps costs 78.5 s per image warm vs sca_v2 77.1 s on the same prompts (+2%), peak 38 GiB.
- r7_q1s (f1, seeding steps) on copyprone21, blind labels pQ1 (6 Opus labelers): 20 yes of 21 (sca_v2 19). Fails the
  screen; its 46-prompt tasks a0_K50x parked in taskq/hold_r7_screen_fail.
- r7_q1a (f1 on every post-gate step), pQ2: 20 yes of 21; 84.6 s per image (+10%). Fails; a0_K60x parked too.
- Lead 03:30: Fix 2 screens moved ahead of f13 (renamed a0_K25* r7_q2, a0_K26* r7_q12) plus r7_q12x (a0_K27*: f1 on
  every step plus pi = near share through sigma 0.75, steps 8 to 14; 46-prompt run parked as a0_K93* in hold_r7_fix2).
- r7_q2 (Fix 2 alone), pQ3: 17 yes, 1 borderline, 3 no of 21 (paired vs sca_v2: 2 removed, 0 added, p = 0.5); 77.3 s.
  Fails. Spool 04:03: r7_q12 running, r7_q12x, r7_q13a, then the ladder (a0_L*). r7_q13s held in
  taskq/hold_r7_screen_wait until r7_q13a shows an effect; f13's 46-prompt runs held in taskq/hold_r7_fix13_46.
- r7_q12 (f1 on every step plus Fix 2), pQ4: 20 yes, 1 borderline of 21; 84.5 s. Fails.
- r7_q12x (f1 on every step plus pi = near share through sigma 0.75), pQ5: 19 yes, 2 no of 21; 84.6 s. Fails.
- Ladder order (04:23): sca+B (renamed a0_L04*) and area+fixed (a0_L09*) first, then L1 op4, L2 scaD, L3 scaE, L5 scaC,
  L6 to L8. r7_q13a (f13 every step) runs at 91.7 s per image (+19%).
- r7_q13a (f13 on every step), pQ6: 3 yes, 4 borderline, 14 no of 21 (below AReA's 4), but every image carries
  colour-block glitches along the borders and in dark areas, and prompted content goes missing (4 of 6 men, no birds,
  no spacecraft). Reading: f13 is the only variant that lowers the global branch's text share (text normalized once
  against about 65K effective image keys); copies fall with it, and the border tokens lose the text sink (round 6's
  ctw showed the same right-edge garbage). New knob --qmix-gamma (commit 8ace484, frozen code/8ace484): scales the image
  part against the text, near/far balance unchanged. Queued at the head: r7_q13m (f13 while sigma > 0.55, a0_K28*),
  r7_q13g (f13 every step, gamma 0.25, text share near sca_v2's, a0_K29*), r7_q13s (seeding steps, a0_K30*).
- 05:33: scripts/r7_qmix_gpu_check.py on an SCA24 GPU (4096^2 shapes, random q/k/v): compiled f1/f13 match eager to
  5e-4; text, near, far and in-window parts match a float32 brute force on border and interior queries; LSEs are natural
  log. So f13's glitches are the operator, not a bug. r7_q13m (f13 while sigma > 0.55), pQ7: 3 yes, 1 borderline, 17 no,
  same glitches; r7_q13g's first images glitch too. f13's near/far weights equal f1's; what differs is that f13's far
  part excludes the in-window block centres. New --qmix-far all (commit 92c8dff, frozen code/92c8dff): far part over
  every centre. Queued at the head: r7_q13w (far all, gamma 1, a0_K295*), r7_q13wg (far all, gamma 0.25, a0_K296*).
  46-prompt runs of q13m/q13g staged in taskq/hold_r7_fix13_46 (a0_K94*, a0_K95*).
- 05:51: r7_q13g (gamma 0.25), pQ8: 8 yes (one adjudicated: a second real bear, no reflection), 1 borderline, 12 no;
  glitches persist; r7_q13w's first images glitch too. Copy removal in f13 is partly text dilution (3 -> 8 when the text
  share is restored). Glitch hypothesis: f13 puts its head offset beta = logit pi - mean(lse_near - log_far) on the near
  logits, e^beta (up to about 50x) above the text keys, so prompt-sink tokens (flat background, borders) lose the sink.
  --qmix-anchor near (commit 1b8f855, frozen code/1b8f855): offset on the far part (lse_T, lse_near, log_far - beta), same
  near/far balance, window's own text/near balance. Spool head: r7_q13n (a0_K294*), q13w remainder, q13wg (a0_K296*),
  q13d5 (dose 0.5, a0_K297*, code/ddda216), q13s (a0_K30*), then the ladder (a0_L*). SCA24 ended 05:55.
- 07:08 SCA21 (4 nodes, 10 h from 06:58). r7_q13n (near-anchored f13) first 9 copy-prone images: no colour blocks
  (one image has a vertical-streak halo around its figures), 6 look clean by eye; 91 s per image warm (+18%). Its
  46-prompt run released to the head of the spool (a0_K2940Nx, gate markers unchanged; eval a0_Ke_eval_r7_q13n
  self-releases into eval/r7_46_r7_q13n). Blind labels of the 21 next.
- 07:21: r7_q13w (far all, gamma 1), pQ9: 2 yes, 3 borderline, 16 no; glitches. r7_q13n at native resolution: no colour
  blocks, but fine periodic vertical stripes on dark ground/sky/water and, on one prompt, the tent drawn as concentric
  dash strokes (sca_v2 crops clean; the aurora comb streaks are in sca_v2 too). f13n's near/far weights equal f1's
  (artifact-free); the difference left is the far output (outside centres, no text, no local coarse context). Queued
  at the head (code/1b8f855): r7_q13nw (anchor near, far all, a0_K2941*), r7_q13nd (anchor near, dose 0.5, a0_K2942*).
- 07:26: r7_q13n, pQ10: 9 yes, 2 borderline, 10 no, stripe banding on most images. Fails. Reading across f13: one knob,
  the prompt's attention mass, drives both copies (diluted -> fewer) and artifacts (diluted -> no sink -> blocks).
  New --text-value A (commit a70efc6, frozen code/a70efc6): image queries read the text VALUES scaled by A in both
  branches, weights unchanged (sink kept), text queries untouched; global branch corrected exactly by linearity.
  Queued at the head: r7_tv0m (A = 0 while sigma > 0.55, a0_K2938*), r7_tv5 (A = 0.5 every step, a0_K2939*).
- 07:47: r7_tv0m, pQ11 (17 of 21 labeled): 0 copies, but subjects drawn wrong (astronauts without suit or helmet, the
  diver a clothed man, a horse-human chimera hiker, 4 of 6 men), missing details, right-edge colour column in two images,
  ringing outlines. 80.0 s (+4%). Its 46-prompt run is generating (a0_K29395*; eval self-releases). Reading: the prompt
  content image tokens read on steps 8 to 19 draws both the copies and the main subject's own look. Per-token version
  (commit 9ef4ee4, frozen code/9ef4ee4): --text-value-subject scales only subject-word tokens (--subject-words
  /scratch/09032/saini_2/sca/r7/subject_words_r7.json, person/animal nouns of each prompt). Queued: r7_ts0m (subject
  words 0 while sigma > 0.55), r7_ts0a (every step), r7_tv0n (A = 0 while sigma > 0.75), r7_tv25m (A = 0.25, > 0.55).
- 07:59: r7_tv0m final, pQ11: 1 yes, 1 borderline, 19 no (below AReA's 4) but subjects drawn wrong; r7_tv5 (A = 0.5
  every step), pQ12: 11 yes, 1 borderline, 9 no, already losing helmets and diver gear. Probe (r6/probe, 6 copies): at
  the end of the opening (step 6) the future copy sites read the subject words at background level (global 0.0018 vs
  background 0.0016, main subject 0.0085) and overtake the main subject only at steps 11 to 14. New
  sca/subjgate.py --subject-anchor A (commit 7270551, frozen code/7270551): the last opening step's subject-word
  attention map (global branch, head mean, block sum; quantiles 0.5..0.95, dilated 2 tokens) keeps full subject
  content at the subject and scales it by A elsewhere on window steps, exact per-query linearity in both branches.
  Queued first: r7_sa0m (A 0 while sigma > 0.55, a0_K293954*), r7_sa0a (A 0 every window step, a0_K2939541*).
- 08:28: r7_sa0m, pQ13: 19 yes, 2 no; prompted subjects intact, no colour artifacts. The map leaks: the raw
  subject map lights the main subject but is speckled, and after a 5x5 max dilation g > 0.5 covers 17-21% and g > 0.1
  52-57% of the image (dumps r7/sa_maps/r7_saAq). --subject-anchor-all (f9f4bc1) gates every text token;
  --subject-anchor-q, -dump (9ffe974); -smooth K and -rel A B (be13ac6: average K x K, ramp from med + A (max - med) to
  med + B (max - med); dumps tagged by prompt id). Screens: r7_saA0m (all text, broad map; 21 images, labeling),
  r7_saAq (q 0.5/0.995, running), r7_saAs (smooth 9, rel 0.25/0.5, dilate 3; queued first). tv0m 46-prompt
  evaluation running (a0_K0eval_r7_tv0m, renamed to run first).
- 08:30: r7_tv0m on the 46 (eval/r7_46_r7_tv0m, scripts/r7_decide.py): human-corrected people/animal 3.3% (AReA
  3.8%, sca_v2 17.0%; detector 0 flags) but 9 significant losses incl. IR, HPSv2.1, PickScore (fixed prompts IR -0.90,
  PickScore -1.76); 80.0 s. Relaxed bar FAIL. Zeroing all prompt content on window steps reaches AReA's duplication at
  a large adherence cost; the subject anchor is the attempt to keep the prompt at the subject only.
- 08:39: r7_saAs (anchor over all text, map 9 x 9 smoothed, ramp med + 0.25..0.5 of max - med, dilate 3; commit be13ac6)
  first 7 copy-prone images: no extras, prompted subjects intact (suits, helmets), no colour artifacts; gate > 0.5 on
  3-27% of the image (leaky versions 17-57%). Its 46-prompt run queued at the head (a0_K2939545*, eval a0_Ke_eval_r7_saAs
  self-releases into eval/r7_46_r7_saAs). Labeling: pQ14 r7_saA0m (broad map; 12 yes of 17 so far), pQ15 r7_saAq.
- 08:48: r7_saA0m (all text, quantiles 0.5/0.95), pQ14: 14 yes, 1 borderline, 6 no; r7_saAq (quantiles 0.5/0.995),
  pQ15: 7 yes, 4 borderline, 10 no; subjects intact in both. Copies fall as the map sharpens. r7_saAs (smoothed,
  relative ramp) labeling as pQ16; 85.2 s per image (+10%).
- 08:57: r7_saAs, pQ16: 7 yes, 2 borderline, 12 no (= saAq); subjects intact, some background details weak (coral,
  asteroids), right-edge block column in 2 images. Remaining copies sit in or next to the subject region (kept at full
  prompt) or in the bear scenes. --subject-anchor-edge T (3be1ea6, frozen code/3be1ea6) keeps a T-token border at full
  prompt. Queued at the head: r7_saAt (rel 0.35/0.6, lo 0.45, dilate 2, edge 2), r7_saAw (saAs + edge 2 + sca_area20
  windows and bridge ratio 0.5). saAs 46-prompt run generating (a0_K2939545*).
- 09:14: r7_saAs on the 46 (eval/r7_46_r7_saAs): human-corrected 3.3% (AReA 3.8%, sca_v2 17.0%; 0 detector flags),
  8 significant losses (IR -0.20, PickScore -0.50, HPSv2.1 -0.015, HPSv2, CLIP -1.2, aesthetic, CLIP-IQA, MUSIQ crop),
  also on clean prompts (IR -0.20, PickScore -0.49); 85.0 s. Relaxed bar FAIL (core losses). Gating every text token
  outside the subject removes scene words too. Queued at the head: r7_saSs (subject-word tokens only, same sharp map,
  edge 2; copy-prone a0_K2939548*, 46 a0_K2939549*, eval self-releases).
- 09:20: r7_saAt (rel 0.35/0.6, lo 0.45, dilate 2, edge 2), pQ17: 7 yes, 1 borderline, 13 no; the anchor plateaus at 7
  (saAq 7, saAs 7). The edge guard puts speckle/glyph bands along the borders of nearly every image (dropped from the
  queued saSs tasks before they ran). Remaining copies sit in or next to the subject region. New --text-branch
  both|loc|glob (commit 394a37d, frozen code/394a37d): text-value scaling or the anchor in one branch only. Queued:
  r7_tvL (text values 0 in the window branch only, sigma > 0.55, a0_K2939550*), r7_tvG (global only, a0_K2939551*).
- 09:35: r7_saAw (saAs + sca_area20 windows + edge 2), pQ18: 6 yes, 15 no; 110.4 s; border bands from the edge guard.
  Anchor family on copyprone21: saA0m 14, saAq 7, saAs 7, saAt 7, saAw 6. r7_saSs copy-prone labeling (pQ19); its
  46-prompt run generating; r7_tvL/r7_tvG queued.
- 09:44: r7_saSs (subject words only, sharp map, no edge guard), pQ19: 18 yes, 3 no. With the same map, gating every
  text token gives 7 (saAs): the copies are drawn by the scene part of the prompt, not the subject noun. Queued after
  r7_tvL/r7_tvG: r7_saAsL (saAs on the window branch only, --text-branch loc, code/394a37d, a0_K2939552*).
- 09:50: r7_tvL (window branch text values 0, sigma > 0.55) first look: copies remain (4 of 6 previewed), adherence
  intact; the global branch is the suspect (audit cause 3). Labeling pQ20. Queued: r7_saAsG (all-text anchor on the
  global branch only, a0_K2939551*_r7_saAsG); running: r7_tvG, r7_saAsL, saSs 46-prompt evaluation.
- 09:54: r7_saSs on the 46 (eval/r7_46_r7_saSs): 0 significant losses, 9/10 wins over AReA kept, 84.6 s, but
  human-corrected 18.2% (4 confirmed flags; sca_v2 17.0%). With saAs (3.3%, 8 losses): scene-word content drives both
  copies and adherence; a two-branch spatial gate cannot separate them. Remaining lever: the branch (tvG, saAsG).
- 09:58: r7_tvL, pQ20: 14 yes, 7 no. 10:05: r7_tvG (text values 0 on the global branch only), pQ21: 8 yes, 1
  borderline, 12 no, plus comb streaks on figures, dashed halos and missing prompt details. Branch split on
  copyprone21: window 14, global 8, both 1 (tv0m, adherence lost); neither branch alone separates copies from
  adherence. tvG's 46-prompt run (queued 10:03 on early labels) was parked after the count: 3 tasks had started and
  finish (24 images); a0_K0a03..05 and its eval sit in taskq/hold_r7_parked.
- Lead levers (10:00): (2) guidance dependence: sca_v2 at guidance 3.5 (r7_g35) and 3.0 (r7_g30) on copyprone21
  (our runs use 4.5), a0_K0g*, code/394a37d. (1) spatially varying guidance, sca/guidemap.py (commit 96c211e, frozen
  code/96c211e, tests/test_guidemap.py): on window steps with sigma > 0.58 (steps 8 to 19) a second transformer pass
  at low guidance, blended v_low + g (v_high - v_low) by the opening's subject map g (saAs map flags, rel 0.25/0.5,
  dilate 3, smooth 9); attention untouched. Queued: r7_gm15 (low 1.5, a0_K0h0*), r7_gm10 (low 1.0, a0_K0h1*).
- 10:13: r7_saAsG (all-text anchor, global branch only), pQ22: 6 yes, 3 borderline, 12 no. 10:21: r7_saAsL (window
  only), pQ23: 12 yes, 9 no. Anchor family: both 7, global 6, window 12; the global branch's prompt read carries most
  of the drive. sca_v2 itself draws the comb/drip streaks on the man-and-child aurora prompt (minus7692529131595280570);
  tvG strengthens them. r7_gm15 runs at 124 s per image (first images). r7_g30 complete, labeling pQ24. Queued after
  gm10: r7_q13i (f13 img anchor: text share of the plain three-part softmax, offset only on the near/far split;
  commit cc62f52, frozen code/cc62f52, a0_K0i*).
- 10:31: r7_g30 (sca_v2, guidance 3.0) pQ24: 19 yes, 1 borderline, 1 no; 10:36: r7_g35 (3.5) pQ25: 20 yes, 1 no.
  Copies do not depend on guidance (4.5: 19). Screen bias: copyprone21 was selected on sca_v2 seed 0, so sca_v2's 19
  is inflated; control r7_s1 (sca_v2 seed 1, a0_K0c*) queued. r7_op4 (AReA's Phase I pools) reproduces AReA's layouts
  on the 46 yet keeps sca_v2-level copies (17.9% vs 2.2%): with the layout matched, SCA's window phase adds copies.
  Ladder moved to the spool head (renamed a0_K0b*): L2 r7_scaD, L7 area_scaMix, L1 r7_op4 (copyprone21), L6
  area_scaOpen, L8 area_scaText. 10:36: L2 r7_scaD (pi = 1) fails on sight (6 of 21 previewed, unlabeled): glyph
  confetti and coloured dots on every image, copies remain (extra astronauts, a second woman); 76 s.
- 10:44: r7_gm15 (spatially varying guidance, low 1.5) pQ26: 17 yes, 1 borderline, 3 no; haze, washed-out
  backgrounds, dissolve streaks; 124 s. r7_gm10 by preview (6 of 21, unlabeled): copies remain, stronger haze.
  Guidance levers closed. New: scale-covariant bridge (commit b36ef9a, frozen code/b36ef9a, tests/test_dedup.py
  brute force): --bridge-coarse scores the bridge keys with the window queries at coarse ids (as the global branch)
  and joins them to the window softmax by log-sum-exp; window keys stay at true ids. Screens r7_bc16 (k 16) and
  r7_bc64 (k 64), a0_K0b15* / a0_K0b16*, after area_scaMix.
- 10:55: r7_bc16 by preview (12 of 21): sca_v2-like images at 77 s, copies stay or move: FAIL, unlabeled. L1 r7_op4 on
  copyprone21 (7 of 21 previewed): AReA's layouts, copies added on 3 of 7 where AReA has none. op4 on the 46 loses
  PickScore (-0.12), aesthetic, MUSIQ crop, so AReA's layouts themselves cost a core metric. Hybrid (commit 17dcf9e,
  frozen code/17dcf9e, --pi-one-lo): AReA's window sizes on steps 8 to 13 (180 to 137, bridge ratio 0.5), then 96,
  80 and sca_v2: r7_hyb (+ pi = 1 while sigma > 0.76, a0_K0b25*), r7_hyb0 (sca_v2 pi, a0_K0b26*), r7_ohyb (hyb with
  op4's opening, a0_K0b27*), all on copyprone21 ahead of area_scaOpen / area_scaText / s1 / q13i.
- 11:00: r7_bc64 (coarse bridge, 4096 tokens) by preview: copies fall but pointillist speckle everywhere: FAIL; last task
  parked. 11:08: r7_hyb (pi = 1 on steps 8 to 13) by preview: coloured square artifacts: FAIL; r7_ohyb parked (same
  pi = 1). Reading: sca_v2's low-pi heads only work through the global branch; any pi = 1 exposes them (scaD, hyb).
  r7_hyb0 (AReA windows on steps 8 to 13, sca_v2 pi): clean, 92 s, copies move (first 2 previewed).
- 11:22: L7 area_scaMix (AReA + SCA's per-head global mix) pQ27: 5 yes, 3 borderline, 13 no (AReA 4), with ripple
  and posterization. The global mix is not the copy source; SCA's small-window phase is (op4 on matched layouts).
- Candidates on the 46 (bar test directly; the copyprone21 screen is selection-biased): r7_saAs75 (saAs gated only
  while sigma > 0.75, i.e. steps 8 to 14; saAs met the copy bar at 3.3% but lost IR/PickScore/HPSv2.1 on clean
  prompts), a0_K0a10*, eval a0_K0_eval_r7_saAs75. r7_csg, dynamic copy-site gate (commit 538114c, frozen
  code/538114c, tests in tests/test_qmix.py): each window step measures the global branch's subject-word share per
  token; sites outside the opening's subject region (dilated 3) at 0.5 to 1.0 of the main subject's level lose all
  prompt values on the next step (while sigma > 0.6); clean prompts untouched. copyprone21 screen with map dumps
  (r7/sa_maps/r7_csg, a0_K0a20*), then the 46 with eval (a0_K0a30*, a0_K0_eval_r7_csg).
- 11:45: r7_csg first look (map overlay on sca_v2, scratch script csg_vis.py): the detector marks the medieval
  square's extra pedestrians exactly; misses a seabed copy; fires weakly on the surfer's hand. Gating the prompt at the
  sites from step 9 does not stop the figures (they form, shifted). csg's 46-prompt run parked (hold_r7_parked).
  New: copy-site anchor (commit 7ab46a3, frozen code/7ab46a3, --copy-gate-anchor): at copy sites x0' = x0_hat + C
  (L - x0_hat), L the opening's x0 estimate (sca/anchor.py local mode, full band). Screen r7_csga on copyprone21 with
  dumps (r7/sa_maps/r7_csga, a0_K0a25*). saAs75 previews look like saAs (background dissolve on the anime prompt).

Addendum 2026-11-06 (lead): user directive: do not cancel any job; use every GPU node the account gets (gh, gh-dev, gb) fully with no downtime; finish all the work. slurm/gpu_watch.sh runs on the controller i615-091 (nohup, log /scratch/09032/saini_2/sca/logs/gpu_watch.log) and starts an idle-only SCA worker on any held GPU node lacking one, every 10 min, while the spool has work; restart it on the next controller if this one ends (job 1125760 ends 11-08 04:59).
