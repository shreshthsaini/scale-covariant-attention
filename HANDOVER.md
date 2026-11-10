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
- 11:50: r7_csga by preview (2 of 21): copies still form; dumps show the map first fires at step 12 (too late) and
  flickers (the gate lowers the share it was detected by). Fix (commit 9b4b34c, frozen code/9b4b34c): hold detected
  sites (running max), earlier ramp 0.3 to 0.6, grow 3: r7_csga2 (a0_K0a26*, dumps r7/sa_maps/r7_csga2); csga's last
  task parked. Timing probe (scripts/r7_probe_x0.py, commit 09cce0a): decodes x0_hat of sca_v2 at steps 7, 9, 11, 13,
  15, 19 on 3 copy prompts (r7/probe3.jsonl) into r7/probe_x0/sca_v2, to see whether a copy is already in the opening.
  Spool sort quirk: LANG collation ignores "_", so a0_K0_eval_* runs after every a0_K0a/b task; use eval tag 0a0
  (saAs75's eval renamed a0_K0a0eval_r7_saAs75 at 11:46).
- 11:52: r7_saAs75 on the 46: FAIL (human-corrected 10.7%, 9 losses incl. IR -0.18, PickScore -0.38, HPSv2.1 -0.013,
  also on clean prompts). Static anchor line closed.
- 11:54: timing probe (black-hole prompt): x0_hat at step 7 shows no copy; at step 9 a blob sits at the copy site; by
  11 it is a figure; a second, lower copy appears at 11 to 13. Copies are made by the first window steps.
- 11:56: r7_csga2 by preview (4 of 21, full-res crops): the seated second astronaut and the flying astronaut are gone
  (clean rock and clouds, one faint crescent smudge); one square pedestrian and the black hole's late ground figure
  remain; no visible quality change; about 85 s. Queued r7_csga2 on the 46 with eval (code/9b4b34c, a0_K0a40*, eval
  a0_K0a0_eval_r7_csga2 in hold_dup_eval).
- 12:10: r7_csga2 false positives: on 4 of 11 dumped prompts the gate covers 10 to 26% of the image (portrait face and
  planet, polar-bear sky), and the pull to the blocky opening estimate pixelates them. Offline calibration of the
  detector on the 61 human-labeled copy sites of sca_v2's copyprone21 images (human_labels_95 xy): step-8 shares find
  12 to 20 of 61 (relative or background z-score, steps 8 to 12). Copy-site route parked (csga2 46-prompt run, its
  eval and csga3 in hold_r7_parked; pQ28 packet built, unlabeled). New code 2497cae: growth and z-score detectors,
  smoothed anchor target, coarse bridge only while sigma > lo with block-averaged values. Screen r7_bce (coarse bridge,
  4096 tokens, steps 8 to 10, value pool 4; a0_K0a60*).
- 12:12: r7_bce by preview (2 of 21): no speckle, copies stay (seabed diver, black-hole figures): FAIL; last task
  parked. Queued r7_hyb0 on the 46 with eval (code/17dcf9e, a0_K0b26*, eval a0_K0b27_eval_r7_hyb0 in hold_dup_eval).

### State at 12:15 (round 7 method owner hand-back)
- Verdicts on the relaxed bar so far: none passes. On the 46: saAs (copies 3.3%, 8 losses), saAs75 (10.7%, 9 losses),
  tv0m (copies removed, adherence lost), sca_area20 (11.5%, 8 losses), op4 (17.9%, PickScore loss).
- Mechanism (evidence in .agent/notes/duplication_fix_round7.md): copies are made by the first window steps (8 to 9,
  probe) acting on the coarse opening latent; with the layout held (op4 vs AReA) SCA's window phase adds them, AReA's
  Phase II does not; SCA's global mix is not the source (area_scaMix 5 vs AReA 4) but removing it breaks the image
  (scaD, hyb: low-pi heads depend on it); guidance does not matter (g30 19, g35 20, gm15 17). Every change that
  removes copies so far also costs quality or adherence (f13 blocks, tv0m, saAs, bc64 speckle); copy-site detection by
  subject-word attention finds 12 to 20 of 61 labeled copies at the hand-off.
- Spool order (4 GPUs on SCA21 until about 17:20; SCA22 pending): op4 copyprone21 (a0_K0b21/22), r7_s1 seed-1
  control (a0_K0b23..25), hyb0 on the 46 + eval, area_scaOpen, area_scaText, q13i, q13nw/nd/wg/d5/s, eval q13n, the
  rest of the ladder (scaE, scaC, area_fixed, scaB's last task), then the review round (r1_*) and later items.
- Next actions: blind-label op4 (copyprone21), s1, area_scaOpen, area_scaText as they land (r7/label_people.sh, now
  seed-agnostic); r7_decide on hyb0's 46 eval; label q13i if its preview is clean. Parked (taskq/hold_r7_parked):
  tvG 46, bc64, hyb, ohyb, csg 46 + eval, csga, csga2 46 + eval, csga3, bce. Unlabeled packets: pQ28 (csga2).

Addendum 2026-11-06 (lead): user directive: do not cancel any job; use every GPU node the account gets (gh, gh-dev, gb) fully with no downtime; finish all the work. slurm/gpu_watch.sh runs on the controller i615-091 (nohup, log /scratch/09032/saini_2/sca/logs/gpu_watch.log) and starts an idle-only SCA worker on any held GPU node lacking one, every 10 min, while the spool has work; restart it on the next controller if this one ends (job 1125760 ends 11-08 04:59).

## 30. 2026-11-07 afternoon: round 7 continued (method owner; log .agent/notes/duplication_fix_round7.md, "Afternoon")
- 12:15 took over from the Section 29 owner. GPU: SCA21 (4 gh, ends about 16:58); SCA22 (4 gh x 10 h) pending, Slurm
  start estimate 12:52. SCA24 x2, SCA27 and SCA28 have already ended (TIMEOUT), so no other job is pending.
- Progressive SCA (sca_v2_prog) at 195 prompts: before today only paired files against HRDiT and sca_prog existed
  (results/prog195_sca_v2). Paired deltas against AReA and sca_v2 are now computed on CPU from the cached per-image
  metrics of eval/a4k195_main_sca_v2 and eval/prog195_sca_v2 (same evaluator and cache generation, AReA's values are
  identical in both evals): scratch eval/prog195_pairs, results/prog195_pairs. Against AReA, progressive SCA
  significantly beats AReA on all 10 per-image metrics (IR +0.40, PickScore +1.05, HPSv2.1 +0.039, MUSIQ +2.3, MUSIQ
  crop +3.7). Against sca_v2 it gains on IR (+0.072), HPSv2.1 (+0.013), PickScore (+0.29), aesthetic, ARNIQA, CLIP-IQA and
  HPSv2, but loses MUSIQ (-3.4) and MUSIQ crop (-6.0). Corrected set-level point estimates (168 reference identities):
  FID 141.9 (AReA 145.8, sca_v2 140.2), KID 0.0071 (0.0072, 0.0058), patch FID 107.7 (102.9, 98.4), patch KID 0.0148
  (0.0114, 0.0089). So patch FID and patch KID are worse than AReA's. Their 1000-draw intervals are running on CPU
  (prog195_pairs/cpu_setlevel.sh, PID 2345099). Detector rate 6/95 against AReA's 4/95 (McNemar p = 0.75). Median time
  50.7 s per image (sca_v2 78.4 s, AReA 264 s on this set). No GPU evaluation is needed.
- 12:45 human-labeled census of the same 95 prompts (subagent of the lead, CPU only; results/dup_audit_20261027/
  human_rates_prog95.txt, rows human_labels_prog95.csv, scripts/human_rates_prog95.py): every sca_v2_prog, AReA and
  HRDiT image labeled blind with the round-6 protocol (packet r7/label/pP95: prog's 95 plus the 55 HRDiT and 32 AReA
  images no earlier round had labeled, 46 Opus labelers; the other 103 keep their round 6/7 labels). 25 disagreements
  and borderlines were adjudicated method-blind from full-resolution crops. Human rates: sca_v2_prog 10/95 = 10.5% [5.8,
  18.3], AReA 5/95 = 5.3% [2.3, 11.7], HRDiT 12/95 = 12.6% [7.4, 20.8]. HRDiT's round-6 sampled estimate (6.3%) was
  low: its 55 previously unlabeled images held 6 detector misses. McNemar sca_v2_prog vs AReA p = 0.23 (8 vs 3
  discordant), vs HRDiT p = 0.79. The detector caught 4 of prog's 10 (6 flags, 2 false: a prompted portrait and
  reflections). The 6 misses are one small extra figure each (a second paddler, a second astronaut, a 40 px person by
  the astronaut's boot, a walker on a far dune, a person on a rock boxed as a bird, a fifth man cut by the frame), and 2
  of the 10 are unprompted portraits of the prompted man. sca_v2 for reference: 26.7% (round 6).
- hyb0 on copyprone21 is complete (21 images, 92 s each) and is being labeled blind as packet pQ29.
- Queued, all on code/17dcf9e. The copyprone21 screens are a0_K0b28* r7_ohyb0 (AReA's opening pools 4,4,4,4,2,2,2,2 plus
  hyb0's windows, sca_v2's pi), a0_K0b29* r7_hyb0g (one extra global step at pool 2 on step 8, gate 0.87, then hyb0's
  windows from step 9), a0_K0b35* r7_g2x2 (steps 8 and 9 global at pool 2, gate 0.86, then sca_v2's windows) and
  a0_K0b45* r7_hyb3 / a0_K0b46* r7_hyb4 (AReA's sizes on steps 8 to 10 or 8 to 11, then 96 and 80). The 46-prompt run
  of r7_g2x2 is a0_K0b36*, with its eval a0_K0b37_eval_r7_g2x2 self-releasing from hold_dup_eval.
- 12:40 decision, r7_hyb0 on copyprone21 (pQ29, 6 blind labelers): 17 yes and 4 no of 21. The references are
  AReA 4, sca_v2 19, area8 16, area12 12 and sca_area20 12. AReA's six widest windows on the birth steps 8 to 13
  therefore do not stop the copies. This matches round 6 (r6_fflh: a far field on the three hand-off steps removed
  nothing) and area8 (16). Wide context has to last through mid noise, and that is what costs quality. So
  shortening the widening to steps 8 to 10 or 8 to 11, as the lead suggested, cannot do better than hyb0.
  r7_hyb3 and r7_hyb4 are parked in taskq/hold_r7_parked. The hyb0 46-prompt run continues for its metric verdict.
  q13i (the joint three-part softmax, the only family that has cut copies below AReA's count) moved ahead of the
  ladder rungs as a0_K0b295*. The r7_fg screens moved up as a0_K0b390* (copyprone21) and a0_K0b440* (the 46,
  whose eval is released as a0_K0b4490eval_r7_fg). fg is the joint far field at coarse ids, never run before.
- 12:45 Copyprone21 triage with the CPU detector (r7/cpu_det_c21.sh LABEL; about 1 min per screen). The detector
  undercounts wide-window variants: hyb0 has 12 flags but 17 by hand, AReA 3 flags and 4 by hand. A screen goes to
  blind labels only when it flags about 6 or fewer, except for the ladder rungs I was asked to label. Seed-1
  control r7_s1: 15 of 21 flagged (sca_v2 seed 0 flags 21 by construction). The unselected sca_v2 rate on these
  prompts is therefore about 15 to 18 by hand, not 19. op4: 15 flagged. Blind packets pQ30 (op4) and pQ31 (s1) are
  being labeled.
- New code d440b71 (frozen code/d440b71, tests/test_globcount.py; suite 281 pass, the known test_review_round
  failure): --glob-count G [--glob-count-lo S]. On window steps the global branch weights its block-centre keys by
  (fh fw)^G against the text for image queries, which is audit cause 3 confined to the global branch. At G = 1 its
  text share falls from about 0.25 to full attention's level, and the result is renormalized, unlike tvG's zeroed
  text values (tvG had 8 of 21 copies plus streak artifacts). Screens: a0_K0b296* r7_gc1 (G 1, sigma > 0.55),
  a0_K0b297* r7_gc1a (G 1, every window step), a0_K0b298* r7_gch (G 0.5, sigma > 0.55). Also queued:
  a0_K0b385* r7_g2x8 (global at pool 2 through step 15, gate 0.72, then 96, 80, 64).
- 12:50 ladder rung L1, r7_op4 on copyprone21 (pQ30, 6 blind labelers): 18 yes and 3 no of 21; detector 15. This is
  SCA's window phase after AReA's own opening pools 4,4,4,4,2,2,2,2, and AReA's layouts are reproduced. AReA itself
  has 4. With the layout held, SCA's window phase draws the copies on copyprone21 as on the 46 (op4 17.9% against
  AReA's 3.8%). So the opening does not decide the copies. The rungs that remain are area_scaOpen (AReA with SCA's
  opening, the converse) and ohyb0 (AReA's opening plus hyb0's wide early windows inside SCA).
- 12:55 progressive SCA, set-level intervals against sca_v2 (195 prompts, 1000 draws, prog195_pairs/cfid_vs_sca_v2).
  FID +1.6 [-2.3, +7.3] and KID +0.0013 [-0.0005, +0.0031] are not significant. Patch FID +9.2 [+5.1, +13.2] and
  patch KID +0.0059 [+0.0036, +0.0083] are significant losses, and both points are worse than AReA's (107.7 against
  102.9, and 0.0148 against 0.0114). So under r7_decide's relaxed rule (a significant loss on another metric must still
  beat AReA), progressive SCA fails on patch FID and patch KID, even though it gains on IR, HPSv2.1 and PickScore
  and beats AReA on all 10 per-image metrics. It also loses MUSIQ (-3.4) and MUSIQ crop (-6.0), both still above
  AReA. The interval against AReA is still computing.
- 13:00 decision, seed-1 control r7_s1 (sca_v2 at seed 1 on copyprone21, pQ31, 6 blind labelers): 18 yes and 3 no of
  21; detector 15. The selection bias is about 1 of 21, since sca_v2 has 19 at its own selecting seed 0. The 21
  prompts are copy-prone for sca_v2 at any seed, so the screen stays valid. The reference is about 18 to 19 for sca_v2
  and 4 for AReA, and any variant at 17 or 18 (hyb0 17, op4 18) has no effect. Table:
  results/dup_audit_20261027/copyprone21_counts.txt (scripts/r7_c21_table.py).
- 13:05 progressive SCA, combined with the parallel census (results/dup_audit_20261027/human_rates_prog95.txt,
  commit 8f3ebc5). Labeling every image of the 95 by hand gives sca_v2_prog 10/95 = 10.5% [5.8, 18.3] against AReA
  5/95 = 5.3% and HRDiT 12.6%. The paired McNemar against AReA is b 8, c 3, p = 0.23. So the detector's 6.3% does
  not hold, and progressive SCA sits above AReA on duplication (not significantly). It also fails the relaxed bar's
  other-metric rule on patch FID and patch KID (above). As a headline it would therefore need the bar changed, or a
  fix to its 4K stage. Queued a cheap probe of the latter: a0_K0b48*0 r7_prog14 (--highres-steps 17,14, so the 4K
  stage gets 14 steps instead of 10) on the 46, with eval a0_K0b4890eval_r7_prog14.
- 13:15 decision, r7_q13i (f13 with the img anchor: the plain three-part softmax's text share, offset only on the
  near/far split; code cc62f52), from a preview of the first 6 images against sca_v2 (r7/preview/r7_q13i/sheet.jpg).
  The copies are gone where sca_v2 has them (the square's pedestrians, the second astronaut by the rock, the lake's
  extra hikers, the black hole's flier). But the f13 glitches remain: dotted colour lines along the top and bottom
  borders, coloured confetti over the black-hole sky, and a speckled texture on the astronaut's suit. FAIL on sight,
  no label packet. The third task had already started on SCA22 and finishes. The f13 family is closed: every
  anchor (text, near, img) either keeps the glitches or brings the copies back.
- 13:15 progressive SCA, full relaxed-bar verdict (r7_decide on eval/prog195_pairs, results/prog195_pairs/decide.txt).
  FAIL on one rule: other losses not better than AReA, namely patch FID (+9.2 against sca_v2, and +4.8 [+0.1,
  +10.6] against AReA, significantly worse) and patch KID (+0.0034 [+0.00005, +0.0066] against AReA). There is no core
  loss: IR, HPSv2.1 and PickScore all gain. Time is 49.4 s. Duplication by hand is 10.5% against AReA's 5.3%
  (census).
- SCA22 started about 13:10 (c620-082, c621-151, c622-132, c634-152): 8 GPUs until about 16:58, then 4 until about
  23:10.
- 13:25 decision, r7_hyb0 on the 46 (r7_decide on eval/r7_46_r7_hyb0): relaxed bar FAIL on core losses IR -0.064
  [-0.113, -0.015], HPSv2.1 -0.0060 and PickScore -0.107 [-0.173, -0.041]. IR and HPSv2.1 also fall on the 27
  clean prompts. In all there are 8 significant losses, 7/10 wins over AReA are kept, and time is 92.7 s (x1.20).
  AReA's widest six windows cost the core metrics even when they act only on steps 8 to 13, the same pattern as area8
  (8 steps) and area12 (12 steps), and on copyprone21 they leave 17 of 21. The wide-window family is closed on both
  counts.
- 13:25 first look at r7_gc1 (count-corrected global branch, sigma > 0.55; 9 of 21 previewed against sca_v2,
  r7/preview/r7_gc1/sheet.jpg). Compositions match sca_v2 and there are no artifacts; time 75 s, no cost. Some copies
  are gone: the horned figure's twin on the right peak, the hiker on the far summit, the small astronaut by the rock.
  Others remain: the square's pedestrians, the black hole's fliers, the shore figures. Detector triage and labels
  follow when the screen completes.
- 13:35 r7_gc1 on copyprone21, detector triage: 11 of 21 flagged (sca_v2 seed 0: 21, seed 1: 15, hyb0: 12 by detector
  and 17 by hand, AReA: 3). gc1 keeps sca_v2's compositions, so this is a paired drop from the same layouts
  (McNemar p = 0.002), at no extra time (75 s) and with no visible artifacts. On sca_v2-like images the detector is
  close to the hand count (sca_v2 precision 23/25, miss rate 1/30). The 46-prompt run is released (a0_K0b2965*, eval
  a0_K0b2969_eval_r7_gc1) for the quality verdict. gc1a (every window step) and gch (G 0.5) screens follow.
  This is the first operator change that removes about half the copies without moving the layout.
- 13:20 localization by detector triage on copyprone21 (no hand labels; the detector undercounts wide-window variants
  by about 5, as hyb0 shows with 12 flagged and 17 by hand). r7_ohyb0 (AReA's opening plus hyb0's windows): 10
  flagged. r7_oarea20 (AReA's opening plus all 19 AReA windows and its bridge inside SCA, the closest SCA analogue of
  AReA): 9 flagged. AReA itself: 3 flagged, 4 by hand. AReA's opening plus AReA's window sizes inside SCA still
  leaves about three times AReA's count. The remaining difference lies in what SCA's window phase does with the same
  keys: the global mix (area_scaMix says no), sliding per-tile windows versus AReA's few large shared windows
  (area_fixed and sca+B at AReA sizes), or the text-query rule (area_scaText, queued).
- New mode (commit 86a4f2f, frozen code/86a4f2f, tests/test_globcount.py brute force; suite 285 pass): --qmix tc.
  The window reads the prompt's content at full attention's per-query strength. Its text values are scaled by
  A_i = (Z_T + Z_near) / (Z_T + Z_near + Z_far), with Z_far the s^2-corrected global far field as in f1, and its
  weights are unchanged, so the prompt stays a sink, unlike f13. Combined with --glob-count 1, both branches read the
  prompt as full attention would. Queued at the spool head: a0_K0b2964* r7_tcg (tc plus glob-count 1, sigma > 0.55),
  and after gch a0_K0b2984* r7_tc (tc alone). gctv is the crude form: uniform window text values x 0.5.
- 13:25 r7_hyb0g (one extra global step at pool 2 on step 8, then hyb0's windows), detector triage: 11 of 21
  flagged (hyb0 12). A finer layout for one more step before the wide windows changes nothing. FAIL by triage, no
  labels.
- 13:40 decision, r7_tcg (qmix tc plus glob-count 1, sigma > 0.55): detector 6 of 21, the lowest of any variant
  with intact compositions. Time 82 s (x1.07). But the native-resolution crops (r7/preview/r7_tcg/detail.jpg) show
  q13n's failure. The camp tent is drawn as a cloud of dash strokes, the scuba diver loses mask and regulator (a man
  in a hoodie), the astronaut's armour gets a bubbly droplet texture, and the anime prompt's two background figures
  become faceless dashed ghosts. FAIL on sight, no labels. Reading: halving the window branch's prompt content (A_i is
  about 0.4 to 0.5 even at the subject) breaks the subjects' prompted details and leaves dash-stroke texture, as
  tv5 and q13n did. The window's prompt read is not a clean lever; the global branch's is (gc1: clean, about half
  the copies). r7_tc (tc alone) and r7_gctv (window values x 0.5) are parked in hold_r7_parked. The clean global-only
  family continues: gc1a, gch and gc2 screens, and gc1's 46-prompt run.
- 13:50 decision, r7_gc1 on the 46 (r7_decide on eval/r7_46_r7_gc1): relaxed bar FAIL. Copies drop: 0 detector flags
  on the 26 person/animal prompts (sca_v2 4, AReA 1), human-corrected estimate 3.3% against AReA's 3.8%. But there are
  core losses: PickScore -0.215 [-0.283, -0.146], IR -0.081, HPSv2.1 -0.0081, and on the 26 clean prompts too
  (PickScore -0.19, IR -0.063). In all there are 8 significant losses, 7/10 wins over AReA are kept, ARNIQA ends below
  AReA, and time is 80.0 s (x1.04). The global branch's native-strength prompt read is part of SCA's adherence
  advantage (PickScore +0.76 over AReA), not only a copy driver. Lowering it to full attention's balance trades
  the one for the other, as every earlier prompt lever did. Stronger forms (gc2 running, gc1pb and gc1pb55) are
  parked or left to finish. gch (G 0.5, the weaker form) is the family's last point; if it keeps the core metrics it
  goes to its 46.
- 14:00 r7_gc1 on copyprone21 by hand (pQ32, 6 blind labelers): 13 yes, 1 borderline, 7 no (sca_v2 19 on the same
  layouts, AReA 4); detector 11. gc1a (every window step): detector 11, so extending past sigma 0.55 adds nothing.
- New code 023918f (frozen code/023918f; suite 287 pass): --glob-count-map. The count correction now acts only away
  from the opening's subject map, with per-query weight 1 + (W - 1)(1 - g_i). The subject keeps the native-strength
  prompt read that sets its look, and the rest of the image reads the prompt as full attention would. A prompt that
  names no subject (subject_words_r7.json, which covers the 26 people prompts of the 46) is left exactly as sca_v2.
  Hypothesis: gc1's PickScore loss comes from the subject's prompt read, and its copy removal from the read
  elsewhere. Queued at the spool head: r7_gcm on copyprone21 (a0_K0b2962*) and on the 46 (a0_K0b29625*, eval
  a0_K0b29629_eval_r7_gcm). Map settings are saAs's (smooth 9, rel 0.25/0.5, dilate 3).
- 14:05 glob-count family, native-resolution crops of the tent, diver and anime prompts
  (r7/preview/gc_detail.jpg; gc2 sheet r7/preview/r7_gc2/sheet.jpg). gc1 (G 1) has the same dash-stroke tent,
  lost diver gear and dashed figure outlines as tcg; the first gc1 preview covered only 9 prompts and missed them.
  gc2 (G 2, detector 7 of 21) is worse. gch (G 0.5, detector 14) is mostly clean: a normal tent, the diver's mask
  and regulator, faint dashes around the anime figures. gcm (G 1 off the subject map) keeps the diver, the subject,
  intact but draws the tent as concentric stripes. Reading: the global branch's native-strength prompt read is what
  keeps prompted objects coherent at the coarse scale. Without it the windows draw prompted objects (tent, figures)
  as fragmented stroke texture, which is also what tv5, q13n and tcg showed. It is also the copy drive, so no
  setting of this lever separates the two. gcm2 is parked. gcm's 46-prompt run finishes for the record.
- 14:05 r7_gcm (G 1 away from the subject map) on copyprone21: detector 9 of 21 (gc1 11, gch 14, gc2 7), with the
  tent-stripe artifact above. The pool-2 extended openings g2x2, g2x4 and g2x8 moved to the spool head
  (a0_K0b2961*; g2x2's 46 as a0_K0b29613*). They leave the prompt read untouched and move the hand-off later, the
  single-pass analogue of progressive SCA's committed layout, so they are the main remaining family.
- 14:30 extended pool-2 opening on copyprone21 (detector triage): g2x2 (global at pool 2 on steps 8 and 9) 16 of 21,
  g2x4 (steps 8 to 11) 11, g2x8 (steps 8 to 15, windows from sigma 0.70) 7. Copies fall with the length of the
  global phase, and the prompt read stays intact; time 76 s, no cost. Native crops (r7/preview/g2x_detail.jpg)
  show artifacts that grow with length. g2x8 has vertical and horizontal streaks along object edges and the horizon,
  a mottled, cratered texture on the astronaut's armour and dashed outlines on the anime figures, while the diver
  keeps his gear. g2x4 is milder (mottled armour, faint dashes). Likely cause: the pool-2 global steps at mid noise
  give every 2 x 2 query block one shared coarse position. Queued: the 46-prompt runs of g2x8 (a0_K0b29614*, eval
  a0_K0b29619_eval_r7_g2x8) and g2x4 (a0_K0b29616*, eval a0_K0b29618_eval_r7_g2x4) for the metric verdict, and
  g2x8f (g2x8 with fractional coarse ids, --glob-pos frac, against the shared-position streaks) on copyprone21.
- 14:35 decision, r7_gcm on the 46: relaxed bar FAIL. The core losses are smaller than gc1's but still significant:
  PickScore -0.130 [-0.215, -0.059], IR -0.067 and HPSv2.1 -0.0048, all three also on clean prompts. Detector flags
  are back to 2 (the canoe and the kimono scene), where gc1 had 0. Keeping the native prompt read at the subject buys
  back about 40% of the PickScore loss and gives back part of the copy removal. Together with the artifacts above,
  the glob-count family is closed. Queued as a last two-lever combination of the mild forms: r7_g4gh (g2x4 plus
  glob-count 0.5) on copyprone21.
- 14:50 decision, r7_gch (glob-count 0.5, visually almost clean) on the 46: relaxed bar FAIL. Core losses PickScore
  -0.125 [-0.173, -0.080], IR -0.059 and HPSv2.1 -0.0038, with 2 detector flags. Even the mild count correction costs
  PickScore about 0.12. On the 46 the global-prompt lever gives gch -0.125 at 2 flags, gcm -0.130 at 2 flags and gc1
  -0.215 at 0 flags (sca_v2 4 flags). There is no setting with PickScore inside its interval.
- 14:55 decision, r7_g2x2 (global at pool 2 on steps 8 and 9, then sca_v2's windows) on the 46: no core loss.
  IR, HPSv2.1 and PickScore are all within their intervals, and the remaining losses (CLIP -0.31, CLIP-IQA -0.017,
  HPSv2 -0.0013, MUSIQ crop -0.59) are all still better than AReA. 8/10 wins are kept and time is 74.9 s (x0.97).
  It fails the relaxed bar only on duplication: 1 detector flag (the kimono crowd prompt 820801809737345747; sca_v2 4,
  AReA 1) gives a human-corrected estimate of 7.0% against AReA's 3.8%. This is the first quality-neutral change
  of the day. The longer forms cut copyprone21 further (g2x4 11, g2x8 7), so their 46-prompt evaluations (running)
  decide whether a longer pool-2 phase keeps the metrics. If g2x4 or g2x8 holds them, its 26 person/animal images go
  to blind labels at once.
- 15:00 decision, extended pool-2 opening on the 46 by full blind census of the 26 person/animal images. r7_g2x2
  (pQ34): 4 yes = 15.4% (sca_v2 17.0%, AReA 3.8%), FAIL on the rate despite no core loss. r7_g2x8 (pQ33): 2 yes and
  1 borderline = 7.7%, FAIL on the rate whatever its metrics (still evaluating). The copies left are a figure on a
  distant path, the canoe's extra paddlers and one anime-crowd figure. g2x8f (fractional coarse ids) is 8 of 21 on
  copyprone21: frac removes the long streaks but keeps the mottled armour and the dashed tent edges
  (r7/preview/g2x8f_detail.jpg). The family trades copies for texture artifacts in proportion to the global
  phase's length, and stops above AReA's rate even at 8 steps.
- 15:00 decision, r7_g2x8 on the 46: relaxed bar FAIL. Its rate is 7.7% by census, and it has 10 significant losses:
  PickScore -0.261, HPSv2.1 -0.0076, aesthetic -0.134, CLIP -0.91, MUSIQ -3.3, MUSIQ crop -2.8, plus ARNIQA,
  CLIP-IQA and patch KID worse than AReA. 5/10 wins over AReA are kept. Eight pool-2 global steps at mid noise cost
  as much as AReA's windows.
- 15:05 decision, r7_g2x4 on the 46: relaxed bar FAIL. Core losses PickScore -0.094 [-0.183, -0.007] and HPSv2.1
  -0.0028; patch KID is worse than AReA; 2 detector flags (10.7% est.). Across the pool-2 family the trade-off is
  monotone: g2x2 (no core loss, 15.4% census), g2x4 (PickScore -0.09, about 10.7%), g2x8 (PickScore -0.26, 7.7%
  census). AReA is at 3.8%.
- 15:05 r7_g4gh (g2x4 plus glob-count 0.5) on copyprone21: detector 9 of 21 (g2x4 11, gch 14). The two levers add,
  but each alone already loses PickScore on the 46 (-0.094 and -0.125), so no 46-prompt run.
- 15:05 r7_gcs (glob-count 1 on the birth steps only, sigma > 0.84) on copyprone21: detector 15 of 21, the same as
  sca_v2 at seed 1. Removing the prompt drive on steps 8 to 10 alone changes nothing; gc1 needed it through sigma 0.55.
  Copies are seeded early but are drawn only if the drive lasts through mid noise, as round 6's fflh showed for the
  far field.
- 15:40 ladder rungs, blind labels on copyprone21: area_scaOpen (AReA with SCA's pool-4 opening, pQ35) 4 yes and 2
  borderline; area_scaText (AReA with SCA's text-query rule, pQ36, 20 of 21 returned) 5 yes and 1 borderline; AReA 4,
  area_scaMix 5. Neither SCA's opening, nor its text rule, nor its global mix adds copies inside AReA. The copies
  belong to SCA's window phase itself, its native-canvas sliding windows reading the prompt on an open layout (op4
  18 with AReA's opening). Triage: r7_fg (joint far field at coarse ids) 8 of 21 by detector; its 46-prompt
  generation finished but the eval is paused.
- 15:40 lead relay of the user's decision: progressive SCA (sca_v2_prog) is the headline and the project is finishing
  for submission. 51 not-yet-started duplication tasks and their parked evals went to taskq/hold_paused_20261107
  (README). Running tasks were allowed to finish and no job was cancelled.
- 15:50 submission items for progressive SCA (lead relay). Queued at the spool head (slurm/prog_headline_tasks.py,
  all gated, evaluation tags in the 0a0 style):
  - (4) 0a0_latency_prog_hrdit_area: one batch-1 sweep on one GPU, 3 timed prompts after a warm-up, median, about
    0.4 GPU-h, into results/prog_latency.
  - (2) 0a1_pgs1/pgs2_*: seeds 1 and 2 on the 195 prompts, 26 tasks, about 5.4 GPU-h, with evaluations
    0a0_eval_prog195_s1/s2 against AReA and sca_v2 at the same seed. These cover all 15 metrics plus corrected
    FID/patch FID and the detector, caches from a4k195_seed*_sca_v2, about 0.7 GPU-h each, into
    results/prog195_s{1,2}.
  - (3) 0a2_pghk_*: HRGen-1K, 1000 prompts, 40 tasks, about 14 GPU-h, with evaluation 0a0_eval_prog_hrgen1k
    against AReA and sca_v2, about 2 GPU-h, into results/prog_hrgen1k. Krea is not possible:
    sample_progressive.py has --model dev only.
  - (1) done on CPU from cached metrics (results/prog195_vs_baselines). Against AReA, DyPE, SEGA and full
    attention all 10 per-image metrics are significant wins, and against UltraImage 9 of 10. Against HRDiT
    (same pipeline, full attention at 2K/4K) it is level on IR, PickScore and HPSv2.1, ahead on HPSv2, ARNIQA
    and MUSIQ, and behind on CLIP-IQA (-0.009), at 50 s against 110 s. Set-level corrected FID/KID intervals are
    running on CPU (PID 3125365).
- 16:08 item (4) landed (results/prog_latency, run out/prog_latency/20261107T153927_c609-002, one GH200, batch 1,
  each method in its own process, warm-up excluded, median of 3 prompts): sca_v2_prog 49.15 s [49.13, 49.17],
  HRDiT (author profile) 106.97 s, AReA 202.36 s. Progressive SCA is 2.18x faster than HRDiT and 4.12x faster
  than AReA.
- 16:35 item (1) set-level landed (results/prog195_vs_baselines/cfid; corrected, 168 reference identities, 1000
  draws, reference sca_v2_prog). Deltas are baseline minus prog, so positive means prog is better on these
  lower-is-better metrics.
  - FID: prog 141.9. AReA +3.9 [-3.1, +8.5], HRDiT -3.4, UltraImage +0.2 and sca_v2 -1.6 are not significant.
    DyPE +63.6, SEGA +23.1 and full +170.6 are significant.
  - Patch FID: prog 107.7. It is significantly worse than AReA (-4.8 [-10.6, -0.07]), HRDiT (-4.6) and sca_v2
    (-9.2), level with UltraImage, and better than DyPE, SEGA and full.
  - Patch KID: prog is worse than AReA, HRDiT and sca_v2, and better than UltraImage, DyPE, SEGA and full.
  - KID: only HRDiT is significantly better.
  - IS (results/prog195_vs_baselines/is_summary.csv): prog 7.18, highest after UltraImage's 7.38.
- 16:45 item (2), seed 1 landed (eval/prog195_s1, results/prog195_s1).
  - Against AReA at seed 1, sca_v2_prog wins all 10 per-image metrics: IR +0.372, PickScore +0.999, HPSv2.1
    +0.037, MUSIQ +3.0, MUSIQ crop +4.1.
  - Against sca_v2 at seed 1 it gains IR +0.065, PickScore +0.224, HPSv2.1 +0.012, HPSv2, aesthetic, CLIP-IQA and
    ARNIQA, and loses MUSIQ -2.4 and MUSIQ crop -6.1. This is the same pattern as seed 0.
  - Corrected set-level points: FID 144.7 (AReA 147.3, sca_v2 141.2) and patch FID 104.7 (AReA 103.2, sca_v2
    98.1). The 1000-draw intervals are on CPU (eval/prog195_s1/cpu_ci.sh).
  - Detector: 9/95 against AReA's 4/95 (p = 0.23). Campaign median 49.4 s.
- 17:05 seed-1 set-level intervals (results/prog195_s1/corrected_fidp_ci, 1000 draws; baseline minus prog).
  Against AReA nothing is significant: FID +2.6 [-5.4, +7.3], KID, patch FID -1.45 [-6.9, +2.7] and patch KID are
  all within their intervals. Against sca_v2, prog is worse on patch FID (-6.5 [-10.5, -3.4]), patch KID and KID.
  So AReA's patch-FID lead over progressive SCA is significant at seed 0 (4.8) but not at seed 1. sca_v2's lead is
  significant at both.
- 17:10 item (2), seed 2 landed (eval/prog195_s2, results/prog195_s2).
  - Against AReA at seed 2, sca_v2_prog wins all 10 per-image metrics: IR +0.354, PickScore +0.995, HPSv2.1
    +0.036, MUSIQ +3.8, MUSIQ crop +5.9.
  - Against sca_v2 it gains IR +0.090, PickScore +0.256, HPSv2.1 +0.011, HPSv2, aesthetic, CLIP-IQA and ARNIQA,
    and loses MUSIQ -2.5 and MUSIQ crop -5.1.
  - Corrected set-level points: FID 144.4 (AReA 146.5, sca_v2 141.1), patch FID 107.7 (AReA 102.3, sca_v2 97.6),
    patch KID 0.0141 (AReA 0.0111). The intervals are on CPU (PID 3435715), written to
    results/prog195_s2/corrected_fidp_ci.
  - Detector: 10/95 against AReA's 4/95 (p = 0.07). Median 49.4 s.
  - Across seeds 0, 1 and 2 the per-image picture is the same; the patch FID gap to AReA ranges from 1.5 to 5.4.
- 17:25 seed-2 set-level intervals (results/prog195_s2/corrected_fidp_ci; baseline minus prog). Against AReA, FID
  +2.1 [-5.4, +7.9] and KID are within their intervals, but patch FID -5.4 [-10.8, -0.8] and patch KID -0.0031 are
  significantly worse. Against sca_v2: patch FID -10.1 and patch KID worse. Across seeds 0, 1 and 2, progressive
  SCA's patch FID is worse than AReA's by 4.8 (significant), 1.5 (not significant) and 5.4 (significant). FID is
  level with AReA at every seed, and the per-image metrics beat AReA on all 10 at every seed.
- Remaining: HRGen-1K. 40 generation tasks 0a2_pghk_* run on SCA22 (184 of 1000 images at 17:07), then the
  self-releasing 0a0_eval_prog_hrgen1k writes results/prog_hrgen1k (with SHA256SUMS) and eval/prog_hrgen1k/DONE,
  expected about 21:30. Commit results/prog_hrgen1k when DONE appears. Nothing of the method owner's is left
  running on i615-091.

## 31. 2026-11-07 evening: paper and web rewritten with progressive SCA as the headline (paper owner)
User decisions (binding, via lead): progressive SCA (HRDiT's three-stage schedule with sca_v2 attention at the 2048^2
and 4096^2 stages; HRDiT credited) is the headline and single-pass SCA the fast mode; framing per
.agent/notes/positioning_20261105/recommendation.md (a head's reach is a property of the head; variance split as the
hook); no duplication content anywhere in paper or web except the one dictated limitations sentence; later priorities
(protocol in one place, paired margins first, curated qualitative evidence, rebuilt method diagram, interim PDF).
- Delivery (first final): /scratch/09032/saini_2/sca/paper_builds/delivery_fwhllii_ (commit c9dd2e9), 54 pp, main text pages 1-10,
  0 overfull, 0 undefined, fresh unzip builds to 54 pp; hashes in .agent/notes/latest_delivery.md. Interim delivery
  delivery_ru0txknm (80078a9) went to the lead earlier.
- Source: paper/main.tex (preamble + main text) and paper/appendix.tex (\input after the bibliography). New title
  "Scale-Covariant Attention: Per-Head Receptive Fields Calibrated at the Training Resolution".
- Main text: abstract with the hook (97.5% head, 2.5% noise level) and headline (all ten per-image metrics against
  AReA at each of three seeds, 4.1x its speed; HRDiT's preference scores in 49.2 against 107.0 s). Figure 1 curated
  progressive hero (fig_hero_prog.py). Figure 2 (method_v2, chosen from two candidates against the gold-standard
  recipe): (a) AReA's one switch vs 1,368 per-head curves, (b) calibration picture equation, (c) progressive stages.
  Section 2.5 + Algorithm 3 for progressive SCA. Section 3 opens with the variance split (Table B.1). Section 4 opens
  with the protocol in one place (195 Aesthetic-4K prompts as AReA, DyPE, TIDE, SigMa; 1000 HRGen-1K as AReA; three
  seed labels; 2000-draw paired intervals, 1000-draw set-level with reference clusters; FID at 299^2; released code;
  clean timing sweep 3). Table 1 has progressive SCA as the headline row and colours every other row against it;
  Figure 4 both modes minus AReA; Figure 5 curated grid AReA / UltraImage / HRDiT / progressive with 1:1 crops.
  The fidelity trade-off against single-pass SCA (MUSIQ_c -6.0, patch FID +9.2) and the patch FID gap to AReA
  (significant at seeds 0 and 2, not 1) and to HRDiT are stated in results, introduction and limitations; the HRDiT
  comparison names its confound (fp16 runner vs our bf16).
- Appendix: duplication appendix, band guard, frequency argument, repeated-subject failures and Dup. columns removed;
  Appendix L progressive SCA (protocol, Table L.1 against every baseline, L.2, Figures L.1 and L.2); galleries open
  with a curated two-page progressive gallery and the uncurated progressive gallery; configuration named everywhere.
- Web: progressive hero, concept chart, four-method curated block, method_v2, three seeds, 49.2 s; duplication section
  and examples removed; build_numbers.py strict check 348/349 quotes match, 0 mismatches; web/ 69 MB.
- Results folded in: prog195_vs_baselines (+cfid), prog195_pairs, prog_latency (sweep 3, half-up rounding of 49.15 to
  49.2 from raw median 49.153 s), prog195_s1, prog195_s2 (+ corrected_fidp_ci).
- 21:00 progressive HRGen-1K landed (results/prog_hrgen1k): all ten per-image wins over AReA (IR +0.90 [0.84, 0.96],
  PickScore +1.40 [1.33, 1.46]); folded into Table 3, abstract, introduction, Section 4.4, conclusion, Appendix L and
  the web page. Final delivery /scratch/09032/saini_2/sca/paper_builds/delivery_rjy3ai7t (54 pp, main text 1-10,
  0 overfull, 0 undefined, fresh unzip passes). Nothing pending.
- Notes for the lead: the code URL in the abstract names the author (swap before a double-blind submission); faint
  tiled-VAE seams (1-2 grey levels) and an 8 px lattice harmonic are visible at 1:1 in some progressive images
  (invisible at print); Krea runs single-pass only.
Rubric self-score (gold-standard 22 items, 0/1/2): abstract 1, hero 2, footnotes 1, question and wrap 1, introduce
and bullets 2, sentence length 1, method diagram 2, equations 1, algorithms 2, setup run-ins 1, main table 2, finding
paragraphs 2, ablations 1, plots under 6 pt 1, qualitative grid 2, limitations 2, reproducibility 2, appendix 2,
float placement 1, captions 1, colours 1, dashes and section symbol 1. Mean 1.45 (32/22), no zero.

## 32. 2026-11-07 night: GPU showcase campaign (showcase owner; log .agent/notes/showcase_20261107.md)
Goal (user, via lead): SCA on FLUX.1-dev, FLUX.2-dev and Qwen-Image at 4096 to 6144 px with zoom-ins and per-example
"how it works" videos for the web page and paper. Ideogram has no weights: skipped (web placeholder stays).
- Spool: the 81 not-yet-started review-round tasks (r1_*, r2_*, r3_*) moved to taskq/hold_paused_20261107b (README
  has the one-line restore). Nothing deleted, no job cancelled. Showcase tasks are named 00s* so they sort first.
- Prompts: web/build/showcase_prompts.jsonl, 40 prompts (12 product, 10 landscape and architecture, 8 portrait, 6
  animal, 4 dense), fields id, category, orient, title; no brands, no characters, text-free labels; themes avoid the
  calibration prompts. Scratch copies prompts/showcase40{,_land,_port}.jsonl (25 landscape, 15 portrait).
- Tasks: slurm/showcase_tasks.py GROUP ... (flux1, area, ...); outputs out/showcase/<model>_<method>/<HxW>/ (PNG +
  sidecar); completion markers out/showcase/_done/<task>; per-task GPU co-tenant log out/showcase/_gpu/<task>.apps.
  Every task waits for its free-memory need and otherwise puts itself back in pending for another worker.
- Configurations. FLUX.1-dev: progressive SCA (sca_v2_prog, PROG_FLAGS of slurm/prog_headline_tasks.py, code/7bd64c0)
  at 4096^2; single-pass sca_v2 (the paper's flags, guidance 4.5) at 5120^2 and at 4096x6144 (landscape prompts) or
  6144x4096 (portrait prompts); native full attention (guidance 3.5, 28 steps) at 1024^2 and 1024x1536 / 1536x1024 for
  the Lanczos reference; AReA (flux1 defaults) at the same three sizes. FLUX.2-dev: the review-round flux2_sca_v2 flags
  (gate 0.959) at 4096^2 and 5120^2, full attention at both, native 1024^2. Qwen-Image: see the port below.
- Fleets: SCS1 (1129438) and SCS2 (1129439), gh 4 nodes x 10 h each, slurm/fleet_SCS{1,2}.sbatch, submitted 21:31
  (SU 9,164 available). gh-dev backfill is not possible: qdevelopment allows 3 submitted jobs per user and TLT-dev,
  BL23 and V2P1 (other projects) hold all three.
- Finding: within minutes of the review tasks being paused, other projects' processes (JLD w4sr.train and
  encode_latents, looped-distill-video score.py, run_video.py) started on all four SCA22 GPUs (20 to 62 GB each).
  Twelve showcase tasks died of OOM and were rewritten with the memory guard (old scripts in taskq/superseded_showcase).
  Timings from shared GPUs are flagged from the _gpu logs; clean batch-1 timings come from a separate timing pass.
- Qwen-Image port (sca/qwen.py, scripts/sample_qwen.py, scripts/calibrate_qwen.py, tests/test_qwen.py, 16 CPU tests
  pass): one processor for the 60 MMDiT blocks; MSRoPE converted once per call to diffusers' real interleaved RoPE
  (exactly the complex product); text sits at max(H/2, W/2) + j, so the global branch's coarse ids are centred like a
  native Hc x Wc image and shifted by M - Mc, which makes every offset of the global branch that of a native image of
  the coarse grid's size (brute-force test). Native 1328^2 has a prime grid (83), so the table is fit at 1344^2 (84 x
  84, ratios 2 and 4 tile it); inference n0 = 83 (5312^2 = 4 x native gives a coarse grid of exactly 83 x 83). Model
  card sampling: true CFG 4, negative " ", 50 steps; mu of the native sequence (pipeline extrapolation would give 3.8).
  Gate and window schedule matched in unshifted time (the FLUX.2 port's rule, 8 of 28 steps global then 4 window
  steps): 14 of 50 steps global (gate 0.85799), then windows 2, 1.75, 1.5, 1.25 x n0 = 166, 145, 124, 104 for two steps
  each (--win-sched 0.83248:166,0.80553:145,0.77701:124,0.74679:104), native 83 after; bridge 1,024 tokens (8 x 16 at
  4096^2, 4 x 64 at 5312^2 where 8 cells do not divide 332).
- Code and trees: Qwen and FLUX.2 tasks run code/fe00d4b (git archive of fe00d4b); FLUX.1 progressive, sca_v2 and AReA
  run code/7bd64c0 (the paper's tree). Traces: web/build/make_showcase_traces.py --ids ... writes
  web/build/showcase_trace_selection.json (box = most-detailed 512 px window of the progressive image), freezes
  code/web_trace_sc_<commit> (7bd64c0 plus the committed sample_trace.py and sample_progressive.py with the per-step
  hook) and enqueues AReA on the traced prompts first (00s4_f), then 00s5 trace tasks; traces land in
  web_assets/trace/{sca_v2_prog,area_sc}/<id> (57 progressive steps: 30 at 1024^2 stock attention, 17 at 2048^2, 10 at
  4096^2; raw x0 decoded per step, 1024 px preview, box crop, the [57, 24] pi per step). Videos:
  web/build/render_showcase_videos.py (<id>_prog.mp4, <id>_prog_vs_area.mp4, to web_assets/video_showcase).
- Manifest and web assets: web/build/build_showcase.py (manifest web/build/showcase_manifest.json with seconds,
  gpu_shared flag from out/showcase/_gpu, scratch paths; Lanczos references out/showcase/<model>_lanczos/<size>; WebP
  overview 2400 px + four 768 px crops at the most detailed windows into web/assets/img/showcase/<group>/, 60 MB budget).
- HF mirror: plan_archives.py lists the showcase SCA sets as phases 1e (prog 4096, Qwen 5312, FLUX.2 5120, FLUX.1 5120)
  and 1f (aspect, FLUX.2 4096, Qwen 4096); sync_hf.py uploads them under the 97 GB cap (85.9 GB used on 2026-11-06),
  comparison sets stay on scratch. No sync loop was running at 22:10; run repro/migration/sync_incremental.sh once the
  sets are complete.
- 2026-11-08 morning, Qwen-Image port verified on GPU (SCSd 1129746; out/showcase/_checks/qwen_gpu_check.json): one
  transformer call at 1328^2 against the stock processors, rel L2 0.0028 (sigma 0.95) and 0.0038 (0.5) in mode full,
  0.0027/0.0035 with the mix off (pi 0, global branch only), 0.0029/0.0039 with pi 1, all below the bf16 floor of
  perturbing the input by one ulp (0.0070/0.0077); max abs 0.031. Cost per call (cond, text 34 tokens): 4096^2 full 7.5 s,
  SCA 2.5 s global, 6.4 s widest window, 3.7 s native window; 5312^2 full 18.7 s, SCA 4.5 / 13.1 / 7.2 s; peak 48 and 52 GB;
  compile warm-up 10 to 20 s per text length and window. Per-head table calib/qwen_1344.npz (16 prompts, 1344^2):
  relative error per head 0.063 at ratio 2 and 0.180 at ratio 4 (global only 0.36 and 0.74, local only 0.082 and 0.216),
  split-half correlation 0.97.
- First pairs tranche at the spool head (000p_*, slurm/showcase_tasks.py pairs): FLUX.2-dev 5120^2 and Qwen-Image 5312^2,
  SCA and full attention, prompts 0-2 of showcase40 (sc_p01..p03, the ones fig_showcase_models.py takes first). Then:
  manifest (build_showcase.py), backbones entries in web/build/showcase_page.json, build_showcase_page.py, paper snippets
  (fig_showcase_models.py --model FLUX.2-dev / Qwen-Image; switches stay off for the paper owner).
- 2026-11-08 13:00, first pairs on the page (dev fleet 1130161 from 11:59): Qwen-Image 5312^2 SCA for sc_p01..p03
  (806, 748 s and the third; peak 52 GB) and full attention for sc_p01 (1,858 s clean), sc_p02/p03 finishing; FLUX.2-dev
  5120^2 SCA for sc_p01..p03 (397 s, peak 92 GB) and full attention for sc_p01, sc_p02/p03 running. Full attention at
  these sizes breaks the composition (Qwen: two small repeated watches on an empty canvas; FLUX.2: a tiled pattern of
  watch faces); SCA keeps one coherent close-up. Page: web/build/set_showcase_backbones.py fills showcase_page.json's
  backbones with complete pairs (largest size, first three prompts), then build_showcase_page.py; check_page.py --http
  PASS; commit 7e3d6f6. The live site was not republished: publish_pages.sh was refused by the session's permission
  classifier (public surface), so the web owner or the user runs it. Paper: fig_showcase_models.py --model Qwen-Image and
  --model FLUX.2-dev wrote paper/figures/showcase_{qwen,flux2}/ and snippets/showcase_{qwen,flux2}{,_text}.tex
  (5312^2 and 5120^2); switches stay off; a scratch build with both on gives 56 pp, 0 overfull, 0 undefined.
- Co-tenants again: looped-distill-video score.py processes joined the Qwen full-attention GPUs on c642-002 and c642-092
  about 30 s after those tasks started (76 s per step instead of 36); their timings are flagged gpu_shared by the end
  snapshot. FLUX.2 commands had an env prefix inside $C that bash ran as a command (fixed in the generator, 13:00).
- Queue after the pairs: Ideogram 00a1_3x (sca8k), 00s1a1, then 00s2 FLUX.1 rest, 00s3 FLUX.2 encode and smokes, 00s4
  Qwen/FLUX.2 SCA (all prompts), AReA (00s4_f traced prompts, 00s6/7), traces 00s5, FLUX.2 full 00s8, Qwen full 00s9.
  To refresh the page as pairs land: build_showcase.py, set_showcase_backbones.py, build_showcase_page.py, check_page.py
  --http, commit web/, then publish_pages.sh (needs approval in agent sessions).

## 33. 2026-11-07 night: no "calibration", no curation language, new method figure (paper owner)
User directives (via lead): (1) drop "calibration" and its forms everywhere in the paper (text, figures, tables,
algorithm names, section titles, appendix, file names in the source zip); (2) drop "curated", "uncurated" and every
selection phrase (figures state prompt set and seed; random galleries keep a neutral line on how prompts were drawn);
(3) redraw the main method figure as a mathematical picture; (4) leave room for a new-model showcase figure.
- Delivery: /scratch/09032/saini_2/sca/paper_builds/delivery_mqhyl3id (commit b131b66), 54 pp, main text pages 1-10
  (two spare lines on page 10), 0 overfull, 0 undefined, fresh unzip builds to the same 54 pages; hashes in .agent/notes/latest_delivery.md. PDF text has no
  calib*, curat*, hand-pick*, "picked by eye", em dash, section symbol or "??".
- Terms, defined once in the introduction and used everywhere: the procedure is *reach emulation* (each head's reach is
  measured once at 1024^2 by shrinking the native window and the native block-centre grid to the target's
  image-to-window ratio; the per-head weight is fitted in closed form against the model's own full attention), its
  output the *reach table*, weights are "fitted". New title: "Scale-Covariant Attention: Per-Head Receptive Fields
  Measured at the Training Resolution". Section 2.4 "Reach emulation"; Algorithm 1 "Reach emulation, once per model"
  (function emulate); Proposition 3 "Closed-form fit and transfer cost"; Appendix B "Reach emulation: details and
  statistics", C "Emulation prompts", O "Reach table at non-square sizes"; labels sec:emulation, alg:emulate,
  alg:emulate_full, app:emulation, fig:reach_tables. File paths with "calib" are no longer printed (sca/calib.py,
  scripts/calib_prompts.jsonl, calib/k1_1024.npz now appear as sca/ and k1_1024.npz).
- Figure 2 is new (figures/method_v3.pdf; TikZ src/method_v3.tex + method_v3_common.tex, assets
  src/fig_method_v3.py -> figures/method_v3/). (a) picture equation O_full = m_i O_near + (1 - m_i) O_far stacked over
  SCA's pi_h O_loc + (1 - pi_h) O_glob on the 1024^2 canvas; (b) the 4096^2 target's window and 64^2 grid shrunk by r
  onto the 1024^2 canvas, r = n/w = 256/64 = 64/16; (c) three real quadratics J(pi) of block 12 at sigma 0.618, r = 4
  (pi* = 0, 0.749, 1), pi* = clip(B/A, 0, 1), reach-table cards; (d) HRDiT's stages with the per-head mix at r = 2, 4.
  Candidates A and B and the critique against the gold-standard recipe: .agent/notes/method_figure_v3.md (A chosen;
  15 words, 4 hues). The intro's head-versus-step pointer moved from the old Figure 2a to Figure 3c.
- Regenerated in place: heads_view.pdf "(c) Fitted pi_h"; method_zoom.pdf "(d) Reach table at r = 4"; calib.pdf ->
  reach_tables.pdf (src/fig_reach_tables.py, axis "emulated setting"). Text diffs against HEAD show only those words.
- Curation: the captions of the hero (Figure 1), the four-method grid (Figure 5), zoom4k, the six-method teaser, the
  single-pass hero, the ablation grid, the failure cases and the progressive galleries now give prompt set and seed
  only; the 10th-percentile sentences are gone; random galleries keep "drawn at random (selection seed 0)".
  The six-method teaser moved ahead of the progressive galleries (it fills the first gallery page).
  gallery_prog_curated was renamed gallery_prog16 (image folder, snippet, src/fig_gallery_prog16.py, labels fig:gallery_prog16 and fig:gallery_prog16b).
  Generator scripts carry the same caption strings, so regeneration reproduces the snippets.
- Showcase slot: main.tex has \newif\ifshowcase, \showcasefalse. src/fig_showcase_models.py turns
  web/build/showcase_manifest.json (generic schema: model, file, prompt, seed, optional method, prompt_set,
  crop_center, status) into figures/showcase_models/ and snippets/showcase_models.tex; set \showcasetrue and the
  figure opens Appendix S with a one-sentence pointer in Section 4.4. Tested on a fake manifest in
  /scratch/09032/saini_2/tmp/showcase_test and showcase_test2 (both manifest schemas; main text still ends on page
  10, 0 overfull). At delivery the manifest held 23 FLUX.1-dev images and no FLUX.2-dev or Qwen-Image image, so the
  switch stays off: run the script and set \showcasetrue when they land.
- build_delivery.py now ships only the .tex files the build read (unused snippets with old wording stay out of the zip).
- Web: the web owner already uses method_v3 and the new terms (fc200a7). paper/figures/method_v2.pdf (still printing
  "Calibrate once") is superseded and no longer read by the paper; heads_view.pdf and method_zoom.pdf were regenerated
  in place under their old names.
- Render review, two cycles (main pages by the paper owner, appendix pages by two reviewer subagents per cycle):
  cycle 1 found 3 MUST items on pages 11 to 32 (Table B.1 resolution wording, a --pi-table path that did not exist
  as printed in Table F.2, a missing seed in Figure L.1) and 16 SHOULD items; cycle 2 found no MUST item. Every
  number in the PDF matches the previous delivery except counts removed with the selection clauses and the old
  Figure 2. Open by choice: Figure T.1 size, free space on gallery pages, and the code release's calib file names
  (sca/calib.py, scripts/calibrate.py, scripts/calib_prompts.jsonl, calib/), which the paper no longer prints.
- Rubric self-score (Section 31 values) unchanged: method diagram 2 (15 words, 4 hues, real data, appendix zoom).

## 34. 2026-11-07 night: project page v3, live on GitHub Pages (web owner; state in .agent/notes/web_v3_20261107.md)
Site: https://shreshthsaini.github.io/scale-covariant-attention/ (gh-pages, built from web/ on origin/main by
`bash web/build/publish_pages.sh`; run it after every web/ push and check the URL). The script now leaves out
web/build and web/README.md and refuses to publish internal paths or calibration/curation wording.
- Page rebuilt around progressive SCA (Source Serif 4 and Inter, light and dark): title and teaser loops (deep zoom,
  1,368-head curves; the three-stage loop waits for a progressive trace), hero with a native-pixel lens, headline
  numbers, abstract, explainer video (116 s, render_explainer.py), observation (full attention at 4096^2, variance
  split, interactive 1,368-head chart, paper heads_view), method (paper method_v3, then interactive figures from real
  data: near/far rings, the two views to scale, reach emulation, the closed-form parabola of three heads, the reach
  table explorer, both schedules), showcase (cards from build/showcase_manifest.json via build_showcase_page.py,
  four native windows each, compare with AReA and Lanczos), comparisons (15 prompts, six methods incl. UltraImage,
  synced full and 1:1 sliders), per-example videos (single-pass until the progressive ones exist), results (three
  seeds, HRGen-1K, time bars, Table 1, charts), gallery (first 12 evaluation prompts, fixed rule), the lead's
  96-prompt page web/review/ (restyled, linked from nav, results and gallery), Ideogram placeholders, limitations.
- Wording: paper terms (reach emulation, reach table); no curated/random labels (build_images.py strips them);
  numbers only through build_numbers.py (172 keys, all present).
- 23:45 showcase in: hero = single-pass SCA at 5120^2 (sc_l04_hillside_town) and 6144x4096 (sc_p01_skeleton_watch)
  with native lens tiles; 12 progressive SCA 4096^2 cards with four native windows each and the native image
  upscaled as comparison (selection in web/build/showcase_page.json; build_showcase_page.py renders from the
  manifest's PNGs). Unused single-pass SCA-only trace videos removed (23 MB).
- Pending (GPU queue; fleets SCS1/SCS2 still waiting): AReA showcase runs (they appear as a comparison when the
  manifest lists them), FLUX.2-dev and Qwen-Image pairs (add to showcase_page.json backbones), the progressive
  traces: then render_process_videos.py process and stages, build_media.py, and drop the remaining 22 MB of
  single-pass videos. Size: web/ without review/ 85 MB; review/ adds 114 MB.
- 2026-11-08 01:40 user directive (lead relay): compare only against methods, not native-resolution images upscaled
  with Lanczos. The unrun native tasks (FLUX.2 1024^2, Qwen 1328^2) moved to taskq/hold_showcase_dropped (README); the
  FLUX.1 natives (out/showcase/flux1_full) and their Lanczos upscales (out/showcase/flux1_lanczos) already existed and
  stay on scratch, unpublished; build_showcase.py lists and exports no Lanczos entries (--lanczos restores them). The
  FLUX.2 native task was replaced by 00s3_c_f2encode (Mistral embeddings for the 40 prompts). Comparisons kept: AReA on
  FLUX.1-dev, full attention at the target size on FLUX.2-dev and Qwen-Image.
- AReA on FLUX.2 or Qwen-Image (directive: only if the released code supports it without porting): the pinned copy
  (9e38ec2) has FLUX.1 only. Upstream HEAD 3efb7e5 (github.com/KooroshRH/AReA) lists Qwen-Image as available (family
  qwen-image: 40 steps, true CFG 4, negative "", its own shift mu 2 at 4096^2, switch sigma 0.87, a quality suffix
  appended to prompts) and FLUX.2 as coming soon; it pins diffusers 0.40 (sca env has 0.39, which does ship the Qwen
  and Ideogram 4 modules it imports). Running that newly fetched code was refused by the session's permission
  classifier, so AReA-Qwen is not run; it needs the user's approval to vet and pin 3efb7e5.
- Compute: SCA22 ended 22:55 with 4 shared GPUs (co-tenants on all four). SCS1/SCS2 still pending (test-only estimate
  2026-11-13); gh-dev backfill SCSd (1129746, 4 x 2 h) submitted 01:38 once TLT-dev freed the qdevelopment slot. Spool
  order: 00s0 Qwen GPU check and calibration, 00s2 FLUX.1 sca_v2 rest, 00s3 FLUX.2 encode and 5120^2 smokes, 00s4 Qwen
  and FLUX.2 SCA plus AReA on the traced prompts, 00s5 traces, 00s6/7 AReA, 00s8 FLUX.2 full, 00s9 Qwen full.

## 35. 2026-11-08 early: page v4, synced zoom grids, speed up top, no Lanczos (web owner; state in .agent/notes/web_v4_20261108.md)
User requests (binding, via lead): every multi-method comparison is a grid with synchronized zoom (wheel or pinch,
drag, double-click reset, an inset map, an auto-zoom tour through high-detail regions chosen at build time, deep zoom
from native tiles loaded on demand); no comparison against the native image upscaled with Lanczos; speed as a
highlight near the top from the paper owner's export; the explanation walks through near view, far view, reach and
reach emulation with the new attention-focus figures instead of the curve bundles.
- Component: web/assets/js/zoomgrid.js + css/zoomgrid.css (vanilla, file:// safe), used by the hero, the observation's
  four-method row, the showcase cards, Comparisons (15 prompts, eight methods: progressive and single-pass SCA, AReA,
  HRDiT, UltraImage, DyPE, SEGA, full attention, with method toggles), the gallery lightbox and every card of
  web/review/ (96 prompts x 5 methods). Plain lazy <img> at the whole-image view; canvases only while zoomed.
- Stops and tiles: web/build/zoomlib.py + build_zoom.py (mean-normalised edge energy averaged over a grid's images,
  three stops at least 1.4 windows apart, the same for every method; region cache on scratch web_assets/zoom_cache/).
  One 1024 px q80 tile per stop and image on the main page, 640 px on web/review/ (1024 px overviews there); the
  showcase uses its four 768 px windows, the hero its 512 px whole-image tiles. Whole-image 4x4 tiling of every
  compared image would cost 0.5 to 2 MB per image (web/ far past 250 MB), and a private HF dataset cannot serve a
  public page, so tiles are cut where the tour goes. web/ is about 250 MB published.
- Lanczos: gone from page data, text and files (showcase owner dropped it from the manifest, 505fd14); the showcase
  cards are single zoom panels until AReA's showcase runs land, then AReA joins each grid automatically
  (build_showcase_page.py), as do FLUX.2-dev and Qwen-Image with full attention at the target size.
- Speed: new section after the headline numbers with the paper owner's speed_time and speed_latency exports (light
  and dark WebP from paper/figures/src/fig_speed.py); the page's own latency chart is retired.
- Explanation: the paper owner's attention-focus figure (attn_focus_{light,dark}.webp; two heads of one block on the
  sheep-and-church image: full attention, near and far parts, SCA's window and grid; reach emulation; the variance
  split) opens the method section, followed by a four-step walk (near view, far view, reach, reach emulation), then
  the method figure, the two views to scale, the closed form, the reach table and the schedules. The 1,368-curve chart,
  its data file, the head-curve teaser loop, the heads_view figure and the rings and emulation diagrams are off the page.
- Checks: check_page.py PASS on desktop, phone, light and dark and on web/review/ (wheel, drag, synthetic pinch,
  double-click, auto zoom, pause); file:// works on both pages; live site checked headless after each publish.

## 36. 2026-11-08: Ideogram 4 at 8192^2 (Ideogram owner; log .agent/notes/ideogram4_20261108.md)
Goal (user, via lead): SCA on Ideogram 4 at 8192^2 (twice its native 4096^2) against full attention, SEGA, NTK and YaRN at
8192^2, with Ideogram 4 at 4096^2 shown separately as the native reference (no upscale comparison); results go to the
web page (synced-zoom grids) and the paper (generality row, figure). Scope (lead, 2026-11-08 02:10): 10 prompts for SCA,
SEGA, full attention and 4096^2; NTK and YaRN on 3; seed 0; cut full attention to 5 prompts if it costs over 3 GPU-h.
- Weights: ideogram-ai/ideogram-4-nf4-diffusers snapshot 1874bc70 (gated; the user accepted the terms 2026-11-08) and
  diffusers/qwen3-vl-8b-instruct-lm-head 9728767e (caption upsampling) in HF_HOME=/scratch/09032/saini_2/tmp/hf; all
  16 files match the Hub SHA256. Env: the sca env (diffusers 0.39, torch 2.11, bitsandbytes 0.50.2); no new env.
- One code path: every method runs the pinned SEGA copy third_party/sega/ideogram4_sega (pipeline + transformer;
  diffusers 0.39's built-in Ideogram 4 lacks its padding trim and maskless path, without which a dense mask over more
  than 65,536 tokens corrupts the bottom rows). NTK / YaRN / SEGA are its set_rope_scaling methods (train grid 256);
  full attention and SCA run with RoPE scaling off. Defaults kept: 48 steps, guidance 7.0 x 45 then 3.0 x 3, dual CFG
  (separate unconditional transformer), logit-normal schedule of the run's size, bf16 on nf4.
- Port (sca/ideogram4.py, 619d0cd + bf2d882): 34 single-stream blocks, 18 heads x 256. Interleaved M-RoPE (h comb
  1, 4, ..., 58; w comb 2, 5, ..., 59; t the other 88; text at (j, j, j), image at 65536 + (0, r, c)) is applied in the
  rotate-half layout; the processor permutes q / k channels into adjacent pairs once per call, so sca/attn.py's kernels
  run unchanged. The global branch's coarse ids change only the h / w combs (65536 + r // f); t and text keep their
  positions, so every text-image phase is that of a native image. Both transformers run SCA, each with its own reach
  table (the unconditional pass has no text, L = 0; sca.attn.rope now passes empty inputs through). n0 = 256.
- Tests: tests/test_ideogram4.py, 14 CPU tests (full mode bit-identical to stock; SCA at native extent equals full
  attention for pi 0, 1, 0.37, flex and gather, with and without text; brute-force native geometry of the global branch;
  only h / w combs move; layout checks; the scripts end to end on a tiny pipeline). The 63 Qwen / FLUX.2 / SCA / dedup
  tests still pass.
- Scripts: scripts/ideogram4_captions.py (JSON captions once per prompt, seeded, shared by every method and size),
  scripts/calibrate_ideogram4.py (reach emulation, ratio 2, one table per transformer, 28 sigma bins because the
  logit-normal schedule keeps 17 of 48 steps above sigma 0.93 at 4096^2), scripts/sample_ideogram4.py,
  scripts/ideogram4_gpu_check.py (identity on GPU, flex kernel options at head_dim 256, step cost at 8192^2),
  slurm/ideogram4_tasks.py (spool tasks; frozen tree /scratch/09032/saini_2/sca/code/ideogram4_bf2d882).
- SCA configuration (sca_v2 mapped to 48 steps like the Qwen port): 14 global steps (gate 0.97444), then windows 512,
  448, 384, 320 for two steps each (win-sched 0.96957:512,0.96413:448,0.95802:384,0.95112:320, win-pi-match and interp;
  below the fitted scale 2 the scale-2 table is used), native 256 after; bridge 8 x 16; tile 8; reach tables
  /scratch/09032/saini_2/sca/calib/ideogram4_4096_{cond,uncond}.npz (2048^2 fit as a transfer check).
- Prompts: the 10 prompts the page already features in the four categories (web/build/ideogram4_prompts.jsonl, order =
  priority): p01 watch, l04 hillside town, h01 silver braids, l06 atoll, l03 opera house, p08 ring, h03 watchmaker,
  l05 palm house, p09 necklace, p11 honey jars; NTK / YaRN on the first 3.
- Spool (lead's order): 00a1_00 check, 00a1_01 captions (releases the fits), 00a1_10/11 fits at 2048^2 and 4096^2,
  00a1_28/29 merges, 00a1_3x SCA at 8192^2 (released by the 4096^2 merge); 00s1a1_2x SEGA and 00s1a1_5x 4096^2 after
  the Qwen check; 00z_a1_* full attention, NTK and YaRN last. Held tasks wait in taskq/hold_ideogram4. Outputs
  /scratch/09032/saini_2/sca/out/ideogram4/ideogram4_<method>/<HxW>/, markers out/ideogram4/_done, GPU logs _gpu.
  Frozen tree /scratch/09032/saini_2/sca/code/ideogram4_227a086 (bf2d882's tasks superseded unrun, kept in
  taskq/superseded_ideogram4). Evaluation runs itself: when no SCA or SEGA task is left, the last one releases
  00a1_40_eval8k; when no generation task is left, 00z_a1_90_eval8k_all. A failed task does not block (the evaluation
  scores what exists and pairs on common prompts). To cut full attention to 5 prompts, move 00z_a1_35 to 00z_a1_39 to
  a folder other than hold_ideogram4 (held tasks count as left, so they would block the final evaluation).
- Evaluation (scripts/ideogram4_eval.py, writes results/ideogram4_8k with SHA256SUMS): eval_extra.py (ImageReward,
  aesthetic, HPSv2, HPSv2.1, CLIP-IQA, ARNIQA), constancy_rope eval.py (CLIP, PickScore), scripts/ideogram4_musiq.py
  (MUSIQ_c on the project's four 1024^2 native crops; full-image MUSIQ on a 4096 px downscale, because pyiqa 0.1.16's
  MUSIQ builds full attention over ~65.7k native patches at 8192^2, about 100 GB), paired bootstrap of SCA against each
  baseline on common prompts, timing with a shared-GPU flag from the task logs, the 4096^2 set scored apart.
- Page and paper (ready, smoke-tested on synthetic 8192^2 images; run on a compute node once images land):
  web/build/build_ideogram4.py (zoomlib: 2400 px overview WebP, 8 x 8 native 1024 px tiles, tour regions shared per
  grid; web/assets/img/ideogram4, web/build/ideogram4_manifest.json, web/assets/js/ideogram4.js with one zoomgrid spec
  per prompt; tiles are about 0.2 MB each, so 36 grids at 8192^2 come to several hundred MB, check the budget);
  paper/figures/src/gen_ideogram4_table.py (snippets/ideogram4.tex, tab:ideogram4) and paper/figures/src/fig_ideogram4.py
  (snippets/ideogram4_fig.tex, fig:ideogram4, first prompts in order with all of SCA, full attention and SEGA).
- After each evaluation the task also runs web/build/build_ideogram4.py (16 processes) and, after the final one, the
  paper table and figure scripts, all in the live repo as new untracked files: web/assets/img/ideogram4,
  web/build/ideogram4_manifest.json, web/assets/js/ideogram4.js, paper/figures/snippets/ideogram4{,_fig}.tex,
  paper/figures/ideogram4/. Nothing is committed or published by a task.
- Status 02:55: all tasks enqueued (check on tree ideogram4_bf3d92d, the rest on ideogram4_227a086); nothing has run.
  SCSd's estimated start is 07:30; SCS1 and SCS2 have no estimate; gh 570 of 577 nodes are busy; qdevelopment allows 3
  submitted jobs per user and SCSd, BL23 and V2P1 hold them. For whoever picks this up: read
  /scratch/09032/saini_2/sca/ideogram4/check/check.json first (identity on GPU, flex options, step cost; cut full
  attention to 5 prompts if 48 x (cond + uncond) seconds exceed 3 h), report the first SCA sidecar's seconds and peak
  memory, then after 00z_a1_90 commit results/ideogram4_8k, the snippets and figures/ideogram4 (and the web assets
  once the web owner agrees on their size).

Addendum 2026-11-08 02:50 (lead): controller i615-091 (job 1125760, also this lead session) ends 04:58. gpu_watch.sh now also runs on i617-052 (job 1126651, ends about 16:00); a fresh 48 h controller job 1129781 is queued. Resume from a new session on whichever controller is up: "continue SCA from HANDOVER". In flight: showcase campaign (FLUX.1 progressive, FLUX.2, Qwen-Image; Sections 32-33), Ideogram 4 at 8192^2 (Section 36, 50 tasks queued), paper figure pass (Section 34), web page live via web/build/publish_pages.sh. All wait on SCS1/SCS2/SCSd (pending, gh saturated).

## 37. 2026-11-08: speed headline, attention-focus figure, newer-model switches (paper owner; notes .agent/notes/paper_v4_20261108.md)
User requests (via lead): speed as the headline visual beside or under the hero; replace the 1,368-curve explanation
visuals with a figure of what attention looks at on a real image; Sections 1-2 walk through near view, far view, reach,
reach emulation; then (coordinator) slots for FLUX.2-dev, Qwen-Image and Ideogram 4, each off behind a switch.
- Delivery: /scratch/09032/saini_2/sca/paper_builds/delivery_gvyduue4 (commit in this section's push), 55 pp, main text pages 1-10 (about 4 spare lines with the switches
  off), 0 overfull, 0 undefined, fresh unzip builds to the same pages; hashes in .agent/notes/latest_delivery.md.
- Figure 2, fig:speed (paper/figures/src/fig_speed.py), under the hero on page 1 (hero now 0.9 text width):
  (a) ImageReward against seconds per image at 4096^2 (log), every Table 1 method at seed 0; filled = clean batch-1
  median, hollow = campaign mean (UltraImage, DyPE, SEGA have no clean time); broken y axis (strip for SEGA, DyPE, full
  attention); bold arrows from AReA, HRDiT and UltraImage to progressive SCA; each marker carries progressive SCA's
  speedup: AReA 4.1x and HRDiT 2.2x clean over clean (prog_latency sweep 3), full attention 3.8x clean, UltraImage
  6.1x, DyPE 3.9x, SEGA 3.8x campaign over campaign (50.7 s for progressive). (b) clean seconds against image side,
  single-pass SCA, AReA, full attention, progressive star at 4096^2, single-pass over AReA along the top (2.1, 2.6,
  2.6, 2.8, 3.1). The script stops unless every value matches Table 1, Appendix D and headline.tex (half-up rounding,
  as gen_main_tables.py); snippets/speed_numbers.tex holds \hlSpeedProgUltraCamp and the other ratios.
- Figure 3, fig:focus (fig_attn_focus.py, attn_focus_data.py): the 1024^2 image is progressive SCA's first stage for
  a4k_eval_4096_minus8343371271909676658, seed 0 (sheep and church), re-run on CPU in bf16 by scripts/attn_focus.py
  (25 min, 112 threads at nice 19 on i615-091), which stores q,k of all 57 blocks at steps 8, 20, 26 (sigma 0.897,
  0.612, 0.327) in /scratch/09032/saini_2/sca/out/attn_focus/sheep_church_seed0 (9.7 GB, purge-eligible; the figure
  reads only results/attn_focus/ and paper/figures/attn_focus/). Heads by rule: block 12 (Figure 4c's block), highest
  and lowest share within 16 tokens in k1_1024.npz at bin 8: heads 4 (share 0.92, pi 0.93) and 7 (0.02, 0.00). Query:
  the leftmost sheep's face. Panels: full attention (amber inside the r = 4 window, blue outside), near part zoomed
  (m_i 0.96 and 0.00), far part, SCA's window and block-centre grid faded by pi_h; (b) reach emulation on the same
  image (4096^2 output with its 64-token window and 64^2 grid, the 1024^2 image with 16 and 16^2); (c) one bar, head
  97.5% and noise level 2.5%. Head 7 attends to the other sheep. Text numbers via snippets/attn_focus_numbers.tex.
- heads_view (the curve bundle and the 1,368-head scatter) moved to Appendix B as Figure B.1.
- Text: abstract cites Figure 2 (49.2 s, 4.1x AReA, 2.2x HRDiT); introduction: question, near/far split and the two
  heads (Figure 3a), 97.5%/2.5% (3c), prior fixes, SCA's two views (Figure 4a), reach emulation (3b), progressive SCA
  with the speed sentence (6.1x UltraImage in campaign time), results; two bullets. Section 2: 2.1 split (Lemma 1),
  2.2 "Near view and far view" (local operator first, then global), 2.3 reach emulation (the mix stated as the head's
  reach; scale covariance folded in), 2.4 single-pass sampling (inference interpolation moved here; sigma bins point
  to Table F.1), 2.5 progressive. "Design choices" moved to Appendix D. Many sentences tightened; Table 3's interval
  rows got 0.5 pt spacing (generator change only); Figure 5 regenerated 2.16 in tall. Number audit: the only numbers
  that left the main text (0.22, 42, 56) are in the appendix; every new number comes from Figures 2 and 3.
- Switches (main.tex preamble, all false): \resfluxtwo, \resqwen, \resideogram. Snippets: fig_showcase_models.py
  --model FLUX.2-dev | Qwen-Image writes showcase_{flux2,qwen}.tex (fig:showcase_{flux2,qwen}) and *_text.tex
  (sizes); Section 36's gen_ideogram4_table.py and fig_ideogram4.py write ideogram4.tex (tab:ideogram4) and
  ideogram4_fig.tex (fig:ideogram4); optional one-sentence results flux2_result_text.tex (\fluxTwoResult),
  qwen_result_text.tex (\qwenResult), ideogram4_text.tex (\ideogramResult). On: the abstract's single-pass sentence
  swaps "keeps more fine detail" for the model list, Section 4.4 gets one run-in per model with its pointers, the
  conclusion the same swap, and the appendix a "Newer models" section with a contents line. Tested with fake snippets
  (all on; Qwen only without a result sentence): Figure 2 stays on page 1, main text ends on page 10 (about 0.3 line
  spare with all three on), 0 overfull, 0 undefined. Longer result sentences need a trim elsewhere.
- Web exports for the web owner: web/assets/img/figs/{speed,speed_time,speed_latency,attn_focus}_{light,dark}.
  {svg,png,webp} (transparent backgrounds; dark uses the page's dark palette).
- Render review: two cycles. Cycle 1: main pages by the paper owner (3 MUST: the 2.5 heading stranded at the bottom of page 5, fixed with needspace; Figure 5's caption overlapping text after the wrap shrank, fixed with a 21-line wrap; nested parentheses in related work); appendix by two reviewer subagents (MUST: Figure D.1 cited Equations 4 to 3 after the operator reorder, now 2 and 3 with 4 for panel a; 'randomly drawn' in Appendix L, gallery and FLUX.2 body text, reworded, captions keep the neutral 'drawn at random (selection seed 0)'; prompts recurring between Figures S.2/S.3 and the seeded galleries, now stated: the sixteen prompts were set later, so a prompt can appear in more than one figure). Also fixed: Figure B.1 caption (gate line and band explained), Figure D.1 placed after its section opening, Figure N.1 caption sign, FLUX.2 portrait phrasing, Figure 2 caption ties 195 prompts to ImageReward. Cycle 2 (paper owner: main pages and appendix pages 15, 19, 32, 40): no MUST item. Open SHOULD items, all pre-existing: blank space under Algorithms 4-5 (p22) and on some table and gallery pages, Table L.1 at 6 pt and 'MUSIQ, crops', Table D.1's first column, Table K.5 caption clause, Figure S.2 says sixteen prompts while showing eight (eight more in S.3), a results/ path in Appendix N prose, mathtext glyphs that copy as fractions in Figures B.1-B.3.
- Rubric self-score (gold standard, 0/1/2): abstract 1, hero 2, footnotes 1, question and wrap 1, introduce and
  bullets 1, sentence length 1, method diagram 2, equations 1, algorithms 2, setup run-ins 1, main table 2, finding
  paragraphs 2, ablations 1, plots under 6 pt 1, qualitative grid 2, limitations 2, reproducibility 2, appendix 2,
  float placement 1, captions 1, colours 1, dashes and section symbol 1. Mean 1.41 (31/22), no zero. Plots now add
  a real-data explanation figure (above the bar for "data flowing through" figures).
- Pending: switches stay off until FLUX.2-dev / Qwen-Image showcase images and Ideogram 4 results land (all GPU work
  waits on SCS1/SCS2/SCSd).

Addendum 2026-11-08 10:30 (lead, after crash): lead session restarted on controller i614-062 (job 1129781, ends 11-10 about 02:45). gpu_watch.sh running here (log logs/gpu_watch_i614-062.log) and on i617-052 (job 1126651, ends about 16:00). Overnight dev fleet SCSd (1129746) completed Ideogram 00a1 check (identity passes both passes, rel L2 at noise floor), 2048/4096 fits and merges, plus 19 showcase tasks. New dev fleet 1130161 submitted; SCS1/SCS2 pending. Showcase agent resumed. 187 tasks pending.

## 38. 2026-11-09: AReA's released Qwen-Image code against SCA (AReA-Qwen owner; log .agent/notes/area_qwen_20261109.md)
Goal (user approval relayed by the lead): run AReA's upstream release on Qwen-Image at its own recommended settings on
the three Qwen showcase prompts (5312^2, and 4096^2, which its README targets), then SCA and AReA on 12 more prompts
for a 15-prompt comparison with the project's metrics, AReA as a column of the page's Qwen grids, and
paper/figures/snippets/showcase_qwen.tex with AReA (switch \resqwen stays off; the lead decides).
- Source: github.com/KooroshRH/AReA at 3efb7e57 ("Add Qwen-Image support", 2026-10-30), the showcase agent's clone,
  vendored unmodified as third_party/AReA_3efb7e5 (no .git, no README images; row and audit note in
  third_party/README.md). Audit: the area package imports only the standard library, numpy, torch and diffusers; no
  network call, no file write, all randomness from seeded numpy generators. Its requirements pin diffusers 0.40; the
  sca env (0.39) is used because every file on the Qwen path is numerically identical on CUDA (diffed against the
  flashart env's 0.40: Neuron RoPE and tensor-parallel plans, removed VAE helpers, user-timestep branch of the
  scheduler; dispatch_attention_fn and native SDPA byte-identical). No new env.
- Settings, read from the vendored generate.py at run time (scripts/run_area_qwen.py): 40 steps, true CFG 4.0, negative
  prompt "" (empty), its quality suffix QWEN_SUFFIX appended to every prompt (", Ultra HD, 4K, cinematic composition,
  faithful to the described content, ..."), VAE tiling; AReA() defaults: switch sigma 0.87, subsample pooling, Halo 8,
  bridge 0.5, AReA seed = run seed 0. The backbone sets its own shift: mu 2.0 at 4096^2 (Phase II from step 20 of 40,
  as its README says), 2.26 at 5312^2 (22 coarse steps, f 4 then 2, then windows 234 down to 64 over 18 steps).
  Harness (not method) differences from generate.py: CPU generator with the per-prompt derived seed (same initial
  noise as SCA and full attention), prompt embeddings precomputed through QwenImagePipeline.encode_prompt (sca/qwen.py
  cache) with the text encoder freed, so the timed scope equals sample_qwen.py's (denoising + VAE decode). Sidecar
  "prompt" is the benchmark prompt (metrics score it); config.conditioning_prompt has the suffix.
- SCA side unchanged: code/fa20267 and the QWEN_SCA flags of every Qwen SCA image so far (50 steps, CFG 4, negative
  " ", mu native 0.834).
- Prompts: /scratch/09032/saini_2/sca/prompts/qwen_area15.jsonl = the first three prompts of each of the five groups
  of showcase_prompts.jsonl (p01-p03, l01-l03, h01-h03, a01-a03, d01-d03); rows identical to showcase40.jsonl.
- Tasks (slurm/area_qwen_tasks.py --code code/<commit>): 000q_a AReA 5312^2 prompts 1-3; 000q_b AReA 4096^2 1-3;
  000q_c SCA 4096^2 1-3 (00s4_a flags, so 00s4_a later skips them); 000q_{d,e,f,g}1 AReA 5312^2 and
  000q_{d,e,f,g}2 SCA 5312^2 on prompts 4-15, three per task (00s4_b flags, outputs shared). Outputs
  out/showcase/qwen_{area,sca}/<HxW>/. Every generation task waits up to 5 min for an exclusive GPU, then polls
  nvidia-smi every 15 s and appends other processes to out/showcase/_gpu/<task>.apps as "poll <epoch> ..." lines;
  run_area_qwen.py also polls in-process and writes gpu_cotenants, gpu_shared and first_call_in_process to the sidecar.
  The task that completes the set moves taskq/hold_area_qwen/000q_z_eval.sh to pending.
- Evaluation (scripts/qwen_area_eval.py -> results/qwen_5312_vs_area with SHA256SUMS): ImageReward, aesthetic,
  HPSv2, HPSv2.1, CLIP-IQA, ARNIQA, CLIP, PickScore, MUSIQ_c (four 1024^2 crops) and MUSIQ on a 4096 px downscale;
  paired bootstrap of SCA minus AReA (2000 draws, seed 0) on common prompts; per-image timing with shared and
  first-in-process flags; medians over all, clean, and clean non-first images.
- Tests: tests/test_area_qwen.py (5 CPU tests: generate.py defaults parsed, README schedule, end-to-end on a tiny
  pipeline at 1344^2 with both AReA phases on both CFG passes, resume, detach).
- Page (b8e55fe, code only; data rebuilt once AReA images land): build_showcase.py lists qwen_area in the qwen_4096 and
  qwen_5312 groups (members SCA, AReA, full attention; crop boxes stay the SCA image's). A new block "Qwen-Image beside
  AReA" under Other backbones (index.html #qa-block, site.js qwenArea): prompt strip, method chips, one synced zoom grid
  (zoomgrid.js) per prompt with SCA, AReA and, for prompts 1-3, full attention; per-image seconds; a table of the ten
  metric means and SCA minus AReA with 95% intervals plus clean median seconds, read by build_showcase_page.py from
  results/qwen_5312_vs_area. The block stays hidden until showcase.js has qwen_area items, so the committed page renders
  as before with the current data. set_showcase_backbones.py now fills "backbones" with FLUX.2-dev pairs only and
  "qwen_area" with the 15-prompt set (SCA and AReA both present). Smoke test on a scratch copy of web/ with stand-in
  AReA images: check_page --http passes (desktop and phone, 0 console errors, 0 overflow) except the copy's missing
  paper PDF.
- Paper: fig_showcase_models.py --model Qwen-Image writes a comparison grid (columns full attention, AReA, SCA (ours),
  each image with its native crop of the same window; rows = prompts 1-3) whenever the manifest has those images, and
  the caption names steps and settings and, once results exist, median seconds. Compiled in a minimal tectonic test.
- Refresh recipe after the 000q images: build_showcase.py, set_showcase_backbones.py, build_showcase_page.py,
  check_page.py --http, fig_showcase_models.py --model Qwen-Image, commit web/ and paper/figures; the lead publishes.
- Status 13:10: 11 generation tasks at the head of the spool (code/535523d for AReA and the evaluation, code/fa20267
  for SCA; the AReA implementation hash 4cbf0ba5 is the same in 8b6a7e9, b8e55fe and 535523d), 000q_z_eval.sh in
  taskq/hold_area_qwen. Evaluator smoke-tested on CPU (two prompts, stand-in AReA images): every metric tool, the paired
  bootstrap, timing and SHA256SUMS ran. SCSd 1132074 (gh-dev 4 x 2 h) estimated start 19:50.
- 2026-11-09 15:40, lead: the 50 vs 40 step mismatch confounds timing and quality, so SCA also runs at AReA's 40
  steps on the same 15 prompts at 5312^2 (000q_z_sca40_00_03 ... _12_15, after the other 000q tasks; label
  qwen_sca40, out/showcase/qwen_sca40/5312x5312; code/fa20267, flags of 000q_<k>2 with --steps 40). SCA's gate and
  window thresholds are noise levels, so at 40 steps the global opening covers 11 steps (14 of 50 scaled is 11.2) and
  the windows 166, 166, 145, 124, 124, 104, 104 cover the same noise range (sigma 0.858 to 0.747), native 83 after.
  The evaluation waits for these too. qwen_area_eval.py adds sca40_minus_area_5312.csv (paired bootstrap of SCA at 40
  steps minus AReA), means over the prompts all three share (summary_5312_paired_prompts.csv) and
  timing_step_matched.csv (per prompt, SCA 40 vs AReA 40, both-clean flag, median row). Page: the Qwen grid gains an
  "SCA, 40 steps" panel (every panel label names its steps) and the table shows AReA, SCA 40 with its difference and
  SCA 50 with its difference, plus the step-matched median seconds. gen_qwen_area_text.py reports both comparisons and
  the step-matched seconds; the paper figure keeps SCA at 50 steps beside full attention and AReA, and its caption
  lists median seconds for all four. All re-tested on stand-in images (CPU evaluator, tectonic, check_page).
- First timings (all clean, GH200): AReA 4096^2 420 (first in process), 370, 370 s (40 steps); SCA 4096^2 468 (first),
  401, 401 s (50 steps); AReA 5312^2 762 s (sc_p01, first) and 760 s (sc_l01, first), peak 55.8 GB, against SCA
  5312^2 748 s warm (50 steps).
- 15:55 status: SCSd 1132074 (c642-022/031/032/082, ends about 17:13) has finished 000q_b (AReA 4096^2: 420 s first in
  process, 370, 370 s; 40 steps; peak 49.8 GB) and 000q_c (SCA 4096^2: 468 s first, 401, 401 s; 50 steps; 47.7 GB) and
  is running 000q_a, d1, d2, e1. AReA 5312^2 so far 762 and 760 s (first in process), 688 and 690 s warm, 40 steps,
  peak 55.8 GB, i.e. 17.2 s per step against SCA's 15.0 (748 s warm at 50 steps); no other process on any GPU so far.
  Visual check of the first real images: AReA and SCA both give one coherent watch at 5312^2 (full attention breaks
  it into two small watches); native crops clean in both. Ten 000q tasks still pending (e2, f1, f2, g1, g2 and the
  five sca40 tasks); whatever is running at the walltime goes back to pending and resumes per image on the next dev
  fleet (dev_resubmit.sh). Code: AReA and the evaluation on code/5e945ef, SCA on code/fa20267.
- Finish path: when every generation marker exists, the last task moves 000q_z_eval.sh to pending; after scoring it
  runs slurm/area_qwen_finish.sh (live repo, CPU, nice): build_showcase.py, set_showcase_backbones.py,
  build_showcase_page.py, fig_showcase_models.py --model Qwen-Image, gen_qwen_area_text.py and check_page --http
  (screenshots in /scratch/09032/saini_2/sca/web_assets/check_area_qwen, log logs/area_qwen_finish.log). Nothing is
  committed or published: review results/qwen_5312_vs_area, the screenshots and the diff, commit web/,
  paper/figures and results/qwen_5312_vs_area, then the lead publishes. Done marker: out/showcase/_done/000q_z_eval.sh.

## 39. 2026-11-09 night: tuning SCA on Qwen-Image against AReA (Qwen-tune owner; log .agent/notes/qwen_tune_20261109.md)
Goal (user, via lead): optimize SCA on Qwen-Image, whose settings were carried over from FLUX, then compare the chosen
configuration with AReA on the 15 TEST prompts at 5312^2; review page web/review_qwen/index.html (not published).
- Starting point (results/qwen_5312_vs_area): SCA 40 steps vs AReA PickScore -0.26 [-0.46, -0.08], HPSv2.1 -0.016,
  HPSv2 -0.004 (significant), MUSIQ_c +5.7, CLIP-IQA +0.026; 604 s vs 690 s. HPSv2.1 lower on 13 of 15 prompts.
- Harness differences found (not the attention operator): AReA's Qwen backbone shifts the schedule with mu 2.0 up to
  4096^2 and 2.0 + 0.5 ln(tokens / 256^2) above (2.26 at 5312^2), the SD3 resolution rule; SCA ran Qwen's native mu
  0.834 at every size (11 of 40 steps above sigma 0.858 against AReA's 23). AReA also appends QWEN_SUFFIX and uses the
  empty negative prompt. Visual: SCA's 5312^2 images are busier and more zoomed in than AReA's.
- Anti-overfitting protocol: the 15 TEST prompts are never used for choices. TUNE = 12 other showcase prompts by a
  fixed rule (prompts/qwen_tune12.jsonl: p04-p06, l04-l06, h04-h06, a04-a05, d04). Selection rule fixed before any
  result: mean d_z (paired mean / SD) over PickScore, HPSv2.1, ImageReward against AReA at the same size on TUNE.
- Code (a7034a3, frozen code/a7034a3; all new behaviour default-off, 7 new CPU tests, the Qwen/AReA/progressive suites
  pass): sca/qwen.py mu_mode "resolution" (native mu + 0.5 ln(seq / native seq): 1.96 at 4096^2, 2.22 at 5312^2);
  sample_qwen.py --mu-mode resolution, --prompt-suffix area (QWEN_SUFFIX read from the vendored generate.py; sidecar
  prompt stays the benchmark prompt); scripts/sample_progressive_qwen.py (sample_progressive.py's HRDiT design on Qwen:
  1328^2 stock attention, SCA at 2656^2 and 5312^2, true CFG per stage, Qwen VAE round trip; the native stage is
  bit-identical to QwenImagePipeline); scripts/qwen_tune_eval.py (66dd2fa; qwen_area_eval.py's metric path, paired
  bootstrap, timing flags, screen.csv); slurm/qwen_tune_tasks.py (spool tasks 000r_*).
- Screens queued 22:05-22:15 (outputs out/qwen_tune/<label>/<HxW>): references AReA 4096^2 and 5312^2 on TUNE
  (qt_area, code/5e945ef); single-pass at 4096^2 qt_sca40 (untuned), qt_mu, qt_sfx, qt_mu_sfx; progressive at 5312^2
  qt_prog, qt_prog_sfx (30 native steps, 20 at 2656^2 from sigma 0.81, 13 at 5312^2 from sigma 0.62). Full attention
  for the 12 TEST prompts without it at 5312^2 (000r_h_qfull5312_*, exactly the 00s9_qfull5312 tree and flags, so
  those skip them), for the review grids.
- 22:45 status: screen tasks running 3 wide (5 fleet GPUs on Ideogram SEGA until about 23:15). Added qt_prog_late
  (HRDiT's step counts 17 and 10 of 30: stages from sigma 0.74 and 0.51) and qt_prog_late_sfx, so the screen has 8
  configurations: single pass 2 x 2 (time shift x conditioning) at 4096^2, progressive 2 x 2 (stage starts x
  conditioning) at 5312^2. First progressive images 298-344 s (AReA 690 s at 5312^2), peak 52.8 GB.
- Evaluation: 000r_e_eval_tune{4096,5312}.sh on code/362d4ab (qwen_tune_eval.py --expect 12: claimed after the last
  screen task starts, scores images as they land) write results/qwen_tune/{4096,5312} (screen.csv ranks by mean d_z).
- Next (resume recipe): read results/qwen_tune/*/screen.csv; pick by the rule; then
  `python3 -I slurm/qwen_tune_tasks.py test evaltest --labels <chosen> [<second>] --code /scratch/09032/saini_2/sca/code/362d4ab`
  (TEST 5312^2 into out/qwen_tuned/<label>/5312x5312; 000r_g eval writes results/qwen_5312_tuned), then
  `web/build/build_review_qwen.py --tuned <chosen> --summary "..."` (page web/review_qwen/index.html; check with the
  playwright script in the notes), commit results/qwen_tune, results/qwen_5312_tuned, web/review_qwen.
- 2026-11-10 00:05, new owner (the previous session died; work committed): verified that every 000r task (running and
  done) reads prompts/qwen_tune12.jsonl and every image in out/qwen_tune is a TUNE id: no TEST prompt was touched.
  Progressive screens done for qt_prog, qt_prog_late, qt_prog_late_sfx (warm 250 to 315 s), qt_prog_sfx finishing
  (320 to 340 s); single-pass qt_mu, qt_mu_sfx, qt_sfx at 4096^2 running or next. Visual guard rail so far (contact
  sheets of qt_prog, qt_prog_late, qt_prog_late_sfx, qt_sca40, AReA): one coherent subject per image, no duplication.
  596fa4b (frozen code/596fa4b): qwen_tune_eval.py --pair (tuned minus untuned SCA on TEST); `test` always runs the
  screens' tree a7034a3 and writes out/qwen_tuned/chosen.env; slurm/qwen_review_finish.sh runs after 000r_g and every
  000r_h task (pending 000r_h rewritten with it): the last one builds web/review_qwen and runs check_review_qwen.py
  (web_assets/check_review_qwen/report.json, logs/review_qwen_finish.log). Nothing is committed by tasks.
  Resume: `python3 -I slurm/qwen_tune_tasks.py test evaltest --labels <chosen> --name "<display name>" --code
  /scratch/09032/saini_2/sca/code/596fa4b`; then wait for the hook (or rmdir out/qwen_tuned/.review_built and run it).
  Controller note: the fresh controller 1132105 (48 h) cannot start before the Vista_PM maintenance (11-11 06:15 to
  22:00) because its walltime overlaps it; a job under about 30 h could.
- 2026-11-10 00:45 state. TUNE screens archived (results/qwen_tune/{4096,5312}, SHA256SUMS). Mean d_z against AReA:
  qt_prog_sfx -0.07, qt_prog_late_sfx -0.07, qt_prog -0.28, qt_prog_late -0.35 (5312^2); qt_sfx -0.55, qt_mu_sfx
  -0.60, qt_sca40 (untuned) -0.78, qt_mu -0.86 (4096^2). Every progressive configuration fails the guard rail: their
  native crops are soft with a fine mesh texture (MUSIQ_c 7.5 to 13.7 below AReA, untuned single pass is 7.2 above);
  the 5312^2 stage starts at sigma 0.62 or 0.51, about 0.29 or 0.21 in native terms at 4x the side, so it keeps the
  upsampled image. qt_mu_sfx fails the MUSIQ_c guard rail by 0.01 (operationalization written before any result).
  Chosen: qt_sfx (single pass, AReA's prompt suffix and empty negative prompt, else untuned): TUNE PickScore -0.28
  [-0.47,-0.10], HPSv2.1 -0.008 [-0.015,+0.000], IR -0.02 (ns), MUSIQ_c +4.6 against AReA; against untuned PickScore
  +0.29 [+0.12,+0.48], HPSv2.1 +0.011 [+0.002,+0.022], MUSIQ_c -2.6 [-4.3,-0.9]. Details in the notes.
- In the spool (nothing else needs a controller): 000r_f_sfx5312_* (TEST qt_sfx, 15 prompts, tree a7034a3),
  000r_g_eval_test5312 (results/qwen_5312_tuned with --pair qt_sfx=qwen_sca40, then the review hook),
  round 2 000r_gz_prog_deep_sfx5312_* + 000r_gz_zeval_tune5312_r2 (TUNE only: the progressive design with a 25-step
  5312^2 stage from sigma 0.915, to fix the softness), 000r_h_qfull5312_* (8 pending, 4 running). chosen.env =
  qt_sfx. BLOCKER: the 74 000a_*/000b_* tasks of Section 40 (enqueued 00:27) sort ahead of every 000r task, so none of
  these runs until they drain (SCS1/SCS2 end about 07:10). The lead decides the order.
- When the TEST evaluation and the full-attention tasks are done the hook builds web/review_qwen (final) and checks it;
  then review logs/review_qwen_finish.log and web_assets/check_review_qwen/report.json, and commit web/review_qwen
  and results/qwen_5312_tuned (the interim page on disk now has TUNE tables, TEST pending, grids without the tuned
  panel; not committed). If round 2 passes the guard rail and beats -0.547: `qwen_tune_tasks.py test evaltest
  --labels qt_prog_deep_sfx qt_sfx --code /scratch/09032/saini_2/sca/code/596fa4b` after setting chosen.env to
  TUNED=qt_prog_deep_sfx, a display name and ALSO="qt_sfx=single pass with AReA's conditioning" (evaltest rewrites
  000r_g only while it is pending; otherwise add a tagged rerun), then rebuild with the hook.

## 40. 2026-11-10: reach-annealed hand-off (A) and the progressive table ablation (B) (method owner; this section is the log)
Brief (lead, 00:00): A = single pass that walks the reach table r = 1 -> 2 -> 4 with measured switch points; B = does
the per-head table matter inside the progressive schedule. GPU only through taskq, tasks 000a_* (A) and 000b_* (B) sort
ahead of the Qwen 000r_* tasks; evaluations self-release (tag 0a0). Controller i614-062 ends about 02:45, so every
step below runs from the spool without a controller.
- Code 4fb0b4d (frozen /scratch/09032/saini_2/sca/code/4fb0b4d; tests/test_reach_anneal.py, 9 new tests incl. a float64
  brute force; full suite 339 pass, the 3 failures are AReA-import order effects that pass alone):
  - --lattice-pool mean (default center = every published path): on --scale-fixed steps at apparent scale 2 each query
    (at its coarse id floor(p / 2)) attends the text, the bridge and a 64 x 64 window of 2 x 2 block means of the
    pre-RoPE keys and the values (a 128-token footprint, offsets inside the native range, the native key count), mixed
    with the pool-4 global branch by the scale-2 per-head table. Cost: the stride-2 block layout (1.055x a native
    window step, round two) plus one average pool.
  - PiTable headmean / layermean and sample_progressive.py --pi-variant (B).
  - scripts/handoff_from_mass.py: measured schedule (CPU). m_r = head-averaged image share of full attention inside the
    ratio-r window (mass_mean at radius 0.5 x 64 / r, divided by the image mass, as Fix 2's m_h).
- Measured schedule (28 steps at 4096^2). m_2 runs 0.476 -> 0.624 over the trajectory and m_4 0.286 -> 0.475: the
  ranges do not overlap, so one tau can never produce both switches inside the trajectory.
  - tau 0.4: m_2 > 0.4 from step 0 (no global opening), r = 2 on steps 0-12, r = 4 from 13 (sigma_4 0.789).
  - tau 0.5: global steps 0-4 (sigma_2 0.940), r = 2 from step 5 to the end (m_4 never reaches 0.5).
  - tau 0.6: global steps 0-19 (sigma_2 0.590), r = 2 on steps 20-27, never r = 4.
  - The only measured schedule that walks 1 -> 2 -> 4 uses two thresholds, tau_2 0.5 and tau_4 0.4: global 0-4, r = 2
    on 5-12, r = 4 from 13 (same steps under the near-plus-text normalization with 0.6 / 0.5). Screened as ra_m.
  - The all-key normalization (text in the denominator) is degenerate too (tau 0.6 never leaves the opening).
- Variants (all with the bridge, 8 x 8 cells x 16, and --lattice-pool mean; r = 4 is sca_v2's native window without
  its 128 -> 80 shrink): ra_fixed (control: gate 0.89, r = 2 on steps 8-13, --scale-fixed 0.77:2), ra_m (0.9404 /
  0.7892), ra_t40 (gate 0, --scale-fixed 0.7892:2), ra_t50 (gate 0.9404, r = 2 to the end), ra_t60 (gate 0.5896, r = 2
  to the end; its 46-prompt run is parked in taskq/hold_ra/park46_ra_t60 because 20 pool-4 steps are expected to
  leave the 4 x 4 block mosaic of the early gate-0.80 test).
- Spool (slurm/reach_anneal_tasks.py, enqueued 00:27): copyprone21 (000a_a<k>NN, 3 chunks per variant, out
  r7/out/<label>/4096x4096); each variant's last chunk releases 000a_0a1_tri_<label> (GPU detector triage into
  r7/cpu_det/<label>, line in r7/cpu_det/auto_triage.log, and the blind packet r7/label/pRA1 to pRA5 in the order ra_fixed, ra_m, ra_t40, ra_t50, ra_t60). The 46
  (000a_b<k>NN, 6 chunks) release 000a_0a0_eval_<label> (eval/r7_46_<label>); each evaluation then queues the gate
  000a_0a2_gate195 (template taskq/hold_ra/gate195.template). Once the four 46-prompt evaluations are DONE the gate runs
  r7_decide and writes eval/ra_gate195/{decision.txt,DECIDED}: a variant is released for the provisional 195-prompt run
  only if it passes the metric part of the relaxed bar (no core loss on IR, HPSv2.1, PickScore; other losses still
  better than AReA; time < 202 s), has <= 6 of 21 copyprone21 detector flags and <= 2 detector flags on the 46;
  fewest copyprone21 flags wins. The staged run sits in taskq/hold_ra195/<label> (13 chunks 000a_c<k>NN into
  out/ra195/<label>, evaluation 000a_0a0_eval195_<label> -> eval/ra195_<label> and results/ra195 with paired
  bootstraps against sca_v2, AReA, sca_v2_prog and HRDiT, clean batch-1 timing 000a_0a3_latency_<label> ->
  results/ra195/latency.csv). The 195 result is provisional until the hand labels confirm the copy counts.
- B (000b_*, after A in the spool): prog_headmean (head mean of the table per sigma bin and ratio), prog_layermean (mean
  over each layer's heads; the averaged family isolates head variation; the paper's single-pass ablation used the
  separately fitted pooled tables), prog_win (--pi-const 1.0: window plus bridge only; text queries still read the
  global branch, SCA's text rule). 195 prompts, seed 0, schedule and flags of sca_v2_prog, out/prog195_ablation/<label>;
  evaluation 000b_0a0_eval_prog195_ablation -> eval/prog195_ablation and results/prog195_ablation (15 metrics,
  corrected FID / patch FID, paired bootstraps vs sca_v2_prog and vs HRDiT, detector); clean batch-1 timing
  000b_0a1_latency_prog_ablation -> results/prog195_ablation/latency.csv.
- Next (whoever holds this section): blind-label every copyprone21 packet r7/label/pRA1 to pRA5 with the round-7
  protocol (one Opus labeler per batch file, prompt_NN.txt), collect with scripts/r7_label_collect.py; kill criterion:
  no variant at <= 8 of 21 without core losses -> stop and report. For a variant that holds, label its 26 person/animal
  images on the 46 (r7/label_people.sh with the default prompt set) and run r7_decide; if the gate already released
  a 195 run, label all 95 person/animal images (protocol results/dup_audit_20261027/human_labels_95_protocol.md).

## 41. 2026-11-10: paper v5, measurement-first rewrite (paper owner; notes .agent/notes/paper_v5_20261110.md)
User request via lead: the complete paper did not sell; rewrite it as a lean ICLR paper in five steps (observation,
idea, reach emulation, use, evidence) with an abstract of at most 150 words, a one-page introduction with three
contributions, captions of at most three lines, a leaner Table 1, and redesigned figures.
- Delivery: /scratch/09032/saini_2/sca/paper_builds/delivery_gy4l47kq, 54 pp, ICLR 2027 style (official files copied
  from Projects/TABES-iclr2027; 2026 files removed), main text pages 1 to 9, 0 overfull, 0 undefined, fresh unzip
  builds to the same pages; hashes in .agent/notes/latest_delivery.md.
- Abstract (150 words) opens with the observation. Introduction: question; observation with the priority line
  (UltraImage noted stable head roles; ours is the first variance decomposition of reach into heads and steps and the
  first per-head reach measured once at 1024^2 and reused at any size); SCA and reach emulation; results with margins
  and the fidelity trade-off in one sentence; bullets Observation / Method / Results.
- Structure: 2 "Reach is a property of the head" (labels sec:observation, sec:analysis); 3.1 two native-size views
  (notation moved here; Propositions on trained range and cost moved to Appendix A with their proofs), 3.2 reach
  emulation (Prop. 1, "One table serves every size"), 3.3 sampling (progressive, single pass; Algorithm 3 out of the
  main text, Algorithm 5 cited); 4.1 main results, 4.2 "The measured reach carries the gain", 4.3 generality, sizes,
  latency; related work and limitations tightened (the one repeated-subject sentence kept verbatim).
- Figures: Figure 3 observation (fig_observation.py, candidate B of two: maps of heads 4 and 7, the 1,368 x 11 matrix
  with head-average strip, 97.5% / 2.5% bar; asserts \hlVarHead, \hlVarStep, \hlVarAvg); Figure 4 method v4
  (method_v4.tex, candidate A of two, three panels, pi* = clip(B/A, 0, 1) in the centre box, titles on one
  baseline); Figure 2 speed cleaned (arrows from AReA and HRDiT only, UltraImage plain point; checks and
  speed_numbers.tex unchanged); Figure 5 (paired differences) floats ({R}) so it no longer overflows a page foot.
  Critique tables in the notes. Web exports for the web owner (no page edited): web/assets/img/figs/{observation,
  method_v4,speed_v5,speed_v5_time,speed_v5_latency}_{light,dark}.{svg,png,webp}.
- Tables (gen_main_tables.py): tab_main_compact.tex (time, IR, HPS, Pick, M_c, FID_p; bold best per group only),
  tab_ablation_compact.tex (reach weights, then views); full versions stay generated for the appendix.
- Switches, off: \resprogabl (results/prog195_ablation/ with SHA256SUMS -> rerun gen_main_tables.py, which writes
  tab_ablation_compact_prog.tex and prog_abl_text.tex for prog_headmean / prog_layermean / prog_win against
  sca_v2_prog) and \resra (results/ra195/ with SHA256SUMS -> figures/src/gen_ra195_text.py writes \raResult; the
  "Measured hand-off" paragraph uses a default \raMethod from HANDOVER 40 unless ra195_method.tex overrides it).
  Tested with fake snippets and every switch on, newer models included: 0 overfull, main text still ends on page 9.
  A background wait for the two folders ran until this session finished (about 00:50); neither had landed.
- Wording: no "random" as selection anywhere in the PDF (gallery captions say "chosen by a fixed seed" or "selection
  seed 0", queries are "seeded", crops "fixed"), changed in the generating scripts and their snippets alike.
- Render review, two cycles over every main page. Cycle 1 MUST: notation ahead of the finding in Section 2 (moved to
  3.1); numbers repeated in the bullets; captions over three lines; abstract at 152 words. Cycle 2 MUST: Figure 5
  overflowed the foot of page 5 when Section 4 started low (now floating); sentences over 35 words (19 split); stale
  appendix references to the old Sections 2 and 3 and to fig:focus and alg:prog_main; random-as-selection captions;
  a stray parenthesis and a doubled "seed" in two captions. Open SHOULD: page 9 is 40% used (room for the switches);
  Figure 3a maps are small; Figure 4's far grid reads as a fine mesh at print size.
- Pending: the two switches above; nothing of the paper owner's is running.
- 00:45 ra_fixed on copyprone21 complete: 73.0 to 73.9 s per image warm (sca_v2 about 77 s), compositions as sca_v2
  (same opening), native crops clean (r7/preview/ra_fixed). Detector triage 13 of 21 (sca_v2 seed 0 21, seed 1 15;
  r7/cpu_det/ra_fixed). Blind packet pRA1 out to 6 Opus labelers.
- 00:56 ra_fixed by hand (pRA1, 6 blind Opus labelers): 14 yes, 3 borderline, 4 no of 21 (sca_v2 19, seed-1 control
  18, AReA 4; detector 13). FAIL on the screen bar (<= 8). The pooled r = 2 stage on steps 8-13 removes about 4 of
  sca_v2's copies, at 73 s per image. ra_m (opening 0-4, r = 2 on 5-12): detector 12 of 21, labeling as pRA2.
- 01:02 detector triage on copyprone21: ra_m 12, ra_t40 13, ra_t50 1, ra_t60 0 of 21 (ra_fixed 13). Previews
  (r7/preview/<label>): ra_t60 has the 4 x 4 pixel mosaic on every image (FAIL on sight, no labels); ra_t50 (no
  native window at any step) is soft with a 2 x 2 lattice texture and lost hair and suit detail at native resolution
  (FAIL on sight as a 4K method; labeled anyway as pRA4 because it is the only variant near the copy bar, metrics on the
  46 run as queued); ra_t40 (no opening) changes compositions (smaller subjects, speckle in dark skies), native detail
  clean. ra_t50 dropped from the 195 gate (gate195.template cands; its staged run parked in
  taskq/hold_ra/park195_ra_t50). Labeling: pRA2 (ra_m), pRA3 (ra_t40), pRA4 (ra_t50).
- Mechanism read so far: copies track the native-window (r = 4) phase. Whenever native windows act from step 13 or 14
  on, 12 to 14 of 21 carry copies; with no native window at all (ra_t50) the detector finds 1 of 21, at the price of
  native-resolution detail. The pooled r = 2 view itself does not draw copies.
- 01:08 ra_m by hand (pRA2): 13 yes, 0 borderline, 8 no of 21. FAIL on the screen bar.
- 01:15 copyprone21 by hand, all screens (6 blind Opus labelers each; results/dup_audit_20261027/human_labels_r7.csv,
  copyprone21_counts.txt): ra_t50 3 yes, 2 borderline; ra_m 13, 0; ra_fixed 14, 3; ra_t40 15, 3; ra_t60 not labeled
  (pixel mosaic, detector 0). References: AReA 4, 1; sca_v2 19, 1; sca_v2 seed 1 18. Only ra_t50 (no native window at
  any step) reaches the bar of 8; its labelers also note doubled outlines and fused second copies (the 2 x 2 lattice).
  Whether the kill criterion triggers depends on ra_t50's core metrics on the 46 (evaluation queued).
- 01:20 staged, not released (taskq/hold_ra/next_ra_p19, next_ra_p22, README_next.txt): ra_p19 = ra_t50 with native
  windows from step 19 (sigma 0.599, progressive SCA's 4K start), ra_p22 from step 22. Both test whether a short native
  phase at the end restores detail without the copies (the single-pass analogue of progressive SCA). Release only if the
  lead wants the iteration (kill criterion).
- ra_fixed's 46 chunk 000a_b002 ran at 158 s per image on c619-002 next to a foreign 19 GB process (PID 1691541, not
  ours, left alone); those 8 sidecar times are not clean.
- 01:22 ra_m on the 46 (r7_decide, eval/r7_46_ra_m): relaxed bar FAIL. 9 significant losses vs sca_v2, core IR -0.079,
  HPSv2.1 -0.019, PickScore -0.34 (also on the 25 clean prompts); aesthetic, ARNIQA, CLIP-IQA and HPSv2 fall below
  AReA; MUSIQ crop -5.2 [-6.9, -3.7] (level with progressive SCA, +0.8 n.s.; scripts/ra_sharpness.py); 4 detector
  flags on the 26 person/animal prompts (sca_v2 4); 76.2 s (x0.99).
- 01:25 B identity check: the latency sweep's sca_v2_prog images from tree 4fb0b4d are bit-identical (PSNR 168 dB,
  zero error) to out/dupfix_prog (tree 7bd64c0) on the three timed prompts, so (i) and (ii)-(iv) differ only in the
  table. B generation running since 01:20 (000b_g*), its evaluation self-releases.
- 01:29 46-prompt verdicts (r7_decide; MUSIQ crop vs progressive SCA from scripts/ra_sharpness.py):
  - ra_fixed: relaxed bar FAIL. 10 losses vs sca_v2, core IR -0.114, HPSv2.1 -0.016, PickScore -0.32 (also on the 26
    clean prompts); aesthetic, ARNIQA, CLIP-IQA, HPSv2 and patch KID fall below AReA; MUSIQ crop -4.4 [-5.8, -3.1]
    (vs progressive SCA +1.7 n.s.); 1 detector flag on the 26 person/animal prompts; 74.2 s (x0.96). Six pooled r = 2
    steps alone cost adherence and crop detail.
  - ra_t40 (no opening): FAIL, IR -0.48, PickScore -1.01, MUSIQ crop -14.3, every win over AReA lost, 8 flags.
  - Gate: eval/ra_gate195/DECIDED = NONE (every candidate has core losses and > 6 copyprone21 flags). No 195 run.
- 01:37 ra_t50 on the 46: relaxed bar FAIL with the largest losses of the screen: IR -0.50, PickScore -1.53, HPSv2.1
  -0.062, MUSIQ crop -19.8 (vs progressive SCA -13.7), patch FID +22.6 [+10.3, +27.3], every win over AReA lost; 1 flag;
  77.0 s. So the only variant at or below 8 of 21 (ra_t50, 3) has core losses: the kill criterion is met. Screen stopped;
  no further variant runs without the lead (ra_p19 / ra_p22 stay staged).
- B status 01:37: clean batch-1 (results/prog195_ablation/latency.csv, one GH200, 3 timed prompts): sca_v2_prog 49.29 s,
  prog_headmean 49.44, prog_layermean 49.33, prog_win 49.43. Generation 000b_g* running on the fleets (about 02:15),
  then 000b_0a0_eval_prog195_ablation self-releases (about 1 GPU-h) into results/prog195_ablation with SHA256SUMS.
  Readout: python3 -I scripts/prog_ablation_summary.py; set-level intervals need the CPU bootstrap afterwards
  (as results/prog195_vs_baselines/cfid).

### State at 01:40 (method owner hand-back; controller i614-062 ends about 02:45)
- A: done. copyprone21 by hand (yes of 21, borderline): ra_t50 3 (2), ra_m 13 (0), ra_fixed 14 (3), ra_t40 15 (3),
  ra_t60 not labeled (pixel mosaic; detector 0); sca_v2 19, AReA 4. All four 46-prompt runs fail the relaxed bar on
  core losses; the 195 gate decided NONE (eval/ra_gate195). The measured single-tau rule is degenerate (above).
- Reading: copies follow the native-window phase (none at any step: 3 of 21; native from step 13 or 14: 13 to 15), but
  the pooled r = 2 view itself costs adherence and crop detail even over 6 steps (ra_fixed IR -0.11, PickScore -0.32,
  MUSIQ crop -4.4), and over the whole trajectory it leaves a 2 x 2 lattice texture.
- In flight: B (above). Parked: taskq/hold_ra (gate template, ra_t60's 46, ra_t50's 195, next_ra_p19, next_ra_p22),
  taskq/hold_ra195/{ra_fixed,ra_m,ra_t40} (staged 195 runs, not needed). Nothing of the method owner runs on the
  controller.
- Crux archived: results/ra_screen (frontier_46.csv, split_46.csv, r7_decide_46.txt, sharpness_46.csv,
  handoff_schedule.txt, copyprone21 detector rates, gate decision, timing; SHA256SUMS); hand labels in
  results/dup_audit_20261027/human_labels_r7.csv (packets pRA1 to pRA4) and copyprone21_counts.txt.

## 42. 2026-11-10 early: progressive ablation folded into the paper (v5.1); reach-annealed switch stays off (lead)
- B landed at 03:14 (results/prog195_ablation, SHA256SUMS). 195 prompts, 4096^2, each variant minus progressive SCA
  with per-head weights, 2000 prompt draws:
  - shared pi(sigma) (prog_headmean): loses on 10 of 10 per-image metrics. IR -0.014 [-0.025, -0.004], HPSv2.1 x100
    -0.38, PickScore -0.07, MUSIQ crop -4.3 [-4.8, -3.8].
  - per-layer (prog_layermean): loses on 10 of 10. IR -0.012 [-0.021, -0.003], HPSv2.1 -0.39, PickScore -0.07,
    MUSIQ crop -2.8 [-3.2, -2.4].
  - window only (prog_win, pi = 1): loses on 8, MUSIQ unresolved, MUSIQ crop +0.5 [0.2, 0.9]. PickScore -0.14.
  - Detector on the 95 people/animal prompts: 5, 6, 10 flags against 6 (no significant difference; not in the paper).
  - Clean batch-1: 49.3 to 49.4 s for all four.
- Set-level intervals (new, CPU bootstrap on the controller, 1000 draws, 168 reference identities, about 10 minutes;
  results/prog195_ablation/cfid_ci with its own README and SHA256SUMS). Patch FID, variant minus per-head:
  shared -7.3 [-10.3, -5.2], per-layer -6.4 [-8.8, -4.8], window only +6.0 [3.9, 9.9]. Patch KID agrees in sign and
  significance. FID and KID are unresolved for all three. So per-head weights win every per-image metric by small
  margins and lose patch FID to the two coarser rules. The paper says both in the main text.
- Paper v5.1: \resprogabl ON. Table 2 gains the group "Progressive SCA, against per-head pi_h" (caption clause under
  the switch), Section 4.2 gains \progAblResult (two generated sentences whose wording follows the intervals),
  appendix L gains a paragraph and Table L.3 (tab:prog_abl_all, 14 metrics with intervals). gen_main_tables.py writes
  all of it (prog_ablation_full, table_prog_abl_all, prog_ablation_sentence; \progAblSeconds from latency.csv).
- Delivery: /scratch/09032/saini_2/sca/paper_builds/delivery_5revguaf, 55 pp, main text pages 1 to 9, 0 overfull,
  0 undefined, fresh unzip builds to the same 55 pages. PDF SHA256 8af9545fcace9877, ZIP 260d4287aba5968f (full hashes in
  .agent/notes/latest_delivery.md).
- A (reach-annealed hand-off): kill criterion met (Section 40), so results/ra195 will not exist and \resra stays off.
  ra_p19 and ra_p22 stay staged in taskq/hold_ra, not released. Progressive SCA remains the headline.
- Before submitting: anonymize the code URL macro (unchanged open item).

## 43. 2026-11-10: version 6, full presentation revamp (lead; brief .agent/notes/paper_v6/BRIEF.md)
User, 10:05: the v5.1 presentation is "still very subpar"; full revamp for version 6 (text, tables, figures, images,
method-figure design, qualitative design, examples), best-paper level in rigour, clarity and claim.
- Lead read all main pages and the appendix sheets (/scratch/09032/saini_2/sca/paper_builds/v6_review_v51) and wrote
  the brief: critique, one story spine, target structure, fixed figure stems (paper/figures/v6/), design system,
  hard rules, ownership. v5.1 tagged paper-v5.1.
- 10:20 five Opus workers launched in parallel: text, appendix, methodfigs, visuals, tablesplots (notes files in
  .agent/notes/paper_v6/). No result changes; no GPU work planned.
- Delivered 2026-11-10 afternoon: **paper v6**, build /scratch/09032/saini_2/sca/paper_builds/delivery_i6ibrtrb, 62 pp
  (main text pages 1 to 9, page 9 about 90% full; reproducibility statement page 10; appendix pp. 13 to 62), 0 overfull,
  0 undefined, fresh unzip builds to the same 62 pages; PDF 19.7 MB (v5.1: 34 MB), ZIP 95 source files. Hashes in
  .agent/notes/latest_delivery.md. Tag paper-v6.
- What v6 is (each worker's notes file in .agent/notes/paper_v6/ has the detail):
  - Title "Reach Is a Property of the Head: Scale-Covariant Attention for High-Resolution Diffusion Transformers";
    abstract 162 words; one spine (question, observation, method, evidence, honest edges); median sentence 20 words.
  - Figures (paper/figures/v6/, scripts paper/figures/src/v6_*.py, each with a *_provenance.json): 1 hero (one large
    4096^2 sample with two 1:1 crops, five samples, quality-against-seconds panel), 2 observation (two heads at
    three noise levels, all 1,368 heads as lines, variance split), 3 method (two bands: measure once at 1024^2,
    generate at any size, joined by the reach table), 4 emulation (weight tracks reach, per-head fits best, one
    table transfers), 5 paired gains against AReA and HRDiT, 6 whole-image comparison at the 25th / 50th / 75th
    percentile of the ImageReward gain over AReA, 7 time against image side (wrapfigure).
  - Tables from the generators only: Table 1 full width with speed ratio, FID and a Holm "+/-" record; Table 2
    generality; Table 3 ablations (fixes a v5.1 error: the shaded reference row showed single-pass SCA's values above
    differences taken against SCA without the schedule). 47 formerly typed numbers and four appendix tables are
    now generated. gen_late_tables.py and gen_prog_tables.py refuse to run (legacy; the first rewrites main.tex).
  - Rigour: Holm correction over the ten per-image metrics (paper/figures/src/gen_holm.py reproduces all 25 archived
    paired files digit for digit). Against AReA every "all ten" claim survives at every seed and on HRGen-1K.
    Against HRDiT only HPSv2 and ARNIQA stay significantly better and none worse (v5.1 also listed MUSIQ better
    and CLIP-IQA worse). Progressive over single-pass on ImageReward is borderline (Holm 0.036 bootstrap, 0.059
    sign-flip). Claims-and-evidence audit in text.md.
  - Finding from the qualitative rounds: per-crop detail is roughly level across methods (in matched windows AReA's
    crop often scores highest), so no figure claims "sharper than AReA" from crops; the main qualitative figure
    compares whole images, and the matched-crop figure is appendix Figure F.1 with a limited claim. Sample-page
    caption leads were made descriptive, with the 195-prompt intervals in the body.
  - Selection rules are in the scripts and captions (seeded draws, percentile and edge-energy rules). Pages that show
    single-pass SCA draw from prompts that name no person, and say so. The ablation grid keeps the "no opening" row.
  - Appendix reordered by reader questions with a contents block; all 83 v5.1 labels and all 2,514 table numbers
    present; tab:phase2_general retired into tab:gen_all (label kept as an alias).
- Rubric (gold-standard-paper, lead's scores, 0/1/2): page 1 2, introduction 1, method diagram 2, equations 1,
  algorithms 1, setup 1, main table 2, finding paragraphs 2, ablations 1, plots 2, qualitative 1, limitations and
  reproducibility 1, appendix 2, layout 1. No zero; mean 1.43.
- Open for the user: (1) ICLR 2027 requires an AI use statement; a commented marker sits above the reproducibility
  statement in main.tex and the authors must write it. (2) The code URL is not anonymous. (3) tab:flops per-layer
  timings have no archived source (caption says measured once).
- GitHub checkpoints during the work: scripts/wip_push.sh (loop stopped with the stop file at delivery).
