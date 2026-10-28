# Late-duplication fix, running log (started 2026-10-27, method owner, Opus subagent)

Directive: SCA adds small extra people, animals or objects in the last 20 of 28 steps (5 of 11 uncurated
FLUX.1-dev 4096^2 images, results/visual_audit_20261020); AReA and the other baselines do not. Remove it
without losing the speed and metric wins. Constraints: SU freeze (no sbatch, no scontrol release); GPU only
on held, idle GPUs through the spool with guest_worker_shared.sh; Python via the sca uv env only.

## Environment at start (2026-10-27 23:00 CDT)
- Operator host i618-062 (gg CPU node, job 1103029, 144 cores, load ~1). Held GPU nodes: none running
  (one gh job 1103038 pending). So Stage 3 GPU runs wait for an idle held GPU.
- Detector candidates in the offline HF cache: IDEA-Research/grounding-dino-tiny, openai CLIP ViT-L/14,
  laion CLIP ViT-H-14, google siglip-so400m, Qwen2/2.5-VL (2B, 3B, 7B), Qwen3-VL-4B, Phi-3.5-vision.
  No OWL-ViT/OWLv2. transformers 5.19 in the sca env has GroundingDinoForObjectDetection.

## Stage 1 (understand)
### Own evidence (CPU, no GPU)
- Schedule (28 steps, max shift, exp(mu)=3.16): sigma_k for k=8..27 = .888 .870 .850 .830 .808 .785 .760 .732
  .703 .671 .637 .599 .558 .513 .463 .407 .345 .275 .196 .105. Mid noise [0.5, 0.89] = steps 8-21 (14 steps).
- The calibrated table is almost flat in sigma (calib/k1_1024.npz, scale 4): mean pi .691 at sigma .896,
  .720 at .687, .735 at .536, .770 at .105; median .73-.83; 38-57% of heads above 0.8, 6-8% below 0.3.
  So at step 8 (sigma .888, signal fraction 11%) the local branch jumps from weight 0 to about 0.7 in one
  step and stays there. AReA at the same step uses 180-token windows plus (0.5 w)^2 global bridge tokens and
  shrinks to 64 only at step 27 (.agent/notes/area_code.md); its last four opening steps already use
  pooling factor 2 (128x128 global keys), SCA keeps factor 4.
- Mechanism: a 64x64 local window with native positions and all 512 text keys is exactly the model's
  native 1024^2 canvas, and the trained prior says a native canvas contains the prompt's subject. With
  pi ~0.7 from sigma .89 on, every window is pushed to draw the subject while medium-scale structure
  (objects of 6-25 tokens = 100-400 px) is still forming; the global branch (weight ~0.3, one key per 4x4
  block) is the only path that tells a window the subject already exists elsewhere. Predicted size of a
  copy scales with the window (paper: a 16-token window gives smaller, more numerous copies).

## Stage 2 (measure)

## Stage 3 (fix)
### Detector (scripts/dup_detect.py, scripts/dup_rate.py)
- Grounding DINO tiny (offline snapshot path, transformers 5.19, CPU, nice 19, 2 shards x 12 threads on the
  operator host gg i618-062; about 5 s per image per shard). One forward per scale (longest side 800 and 1333)
  with all subject queries of the prompt joined ("person . bear ."); per-query box score = max token
  probability over the query's tokens; raw boxes >= 0.2 stored in /scratch/09032/saini_2/sca/eval/dup_detect_20261027.
- dup_rate.py: boxes >= thr from both scales, NMS IoU 0.5, containment merge (>= 0.9 inside and >= 0.3 of the
  area), boxes over 90% of the image dropped; flag = any subject count > implied count.
- Subjects: results/dup_audit_20261027/subjects_a4k195.json, written blind to images by an Opus subagent from
  the prompt text (person query for every human, implied counts, null for crowds; reflections doubled).
  147 of 195 prompts have a numeric count, 97 solo, 31 have no countable subject.
- Validation labels: results/dup_audit_20261027/validation_labels.csv: the 11 audited images (5 pos) plus 23
  SCA images judged by the figure-curation auditors (13 pos, 10 neg); 4 negatives have no countable subject.
- First look (19 scored, thr 0.35): 16/19 agree. Inspected all 3 disagreements at full resolution:
  * 997293625843621481 ice climber (curation label clean, "small hook at top edge"): the detector is right,
    three small extra climbers walk along the top edge (crop viewed); label was an audit miss.
  * 1516768667427572324 artist's corner (audit label no): a second, smaller framed portrait of the same man
    with glasses sits lower left (one of the "empty" frames); the detector's person 2/1 is a real copy; its
    suitcase 2/1 is a false positive on a tin box.
  * 3614061571823823193 Moai (label yes, small second Moai on the far shore): missed (statue 1/1).

### Candidate fixes (ranked by expected effect per cost; implemented flags marked [impl])
Cost model at 4096^2 on a GH200: 70.6 s per image; attention 18.7 ms per layer, about half of it the local
branch, so a local window of side w costs about (512 + w^2)/(512 + 4096) of the published local branch.
1. [impl] Soft opening phase, `--pi-ramp SIGMA_R` (ramp_sigma): pi(sigma) * r^gamma with
   r = clip((0.89 - sigma)/(0.89 - SIGMA_R), 0, 1). Gives the AReA-like gradual hand-over from the
   whole-image canvas to native windows over steps 8-19 instead of one step. Cost 0 s (local branch already
   computed). Risk: more global weight at mid noise softens texture (global-only mosaic).
2. [impl] Per-head opening, `--pi-ramp-mode head`: head h opens at sigma_r + (0.89 - sigma_r) * pi_h, so
   heads that prefer the global branch (small pi) stay global longer and local heads open at the gate.
   Cost 0 s.
3. [impl] Fractional global positions, `--glob-pos frac`: global-branch ids p/s instead of floor(p/s), so the
   16 queries of a block no longer share one position (the cause of the 4x4 mosaic when the opening phase is
   long; paper Section 2.3). Enables a longer or softer opening phase (e.g. gate 0.80) at 0 s. Offsets stay
   below n0 (Prop. 1 holds with real-valued ids).
4. [impl] Wider mid-noise windows, `--win-sched 0.80:128,0.70:96`: AReA-like shrink 128 -> 96 -> 64 over
   steps 8-14. Local cost x3.6 for 4 steps and x2.1 for 3 steps: about +5 to +6 s (8%). Offsets exceed the
   trained range while wide (as in AReA).
5. [impl] Text-key bias in the local branch, `--text-local -2.77 [--text-local-lo 0.5]`: a native window
   gets the full prompt drive of a native image (512 text keys against 4096 image keys), so every window is
   pushed to draw the subject; full attention at 4096^2 dilutes text mass by up to 16x. Bias log(1/16)
   restores that dilution in the local branch only (global branch and text queries unchanged). Cost about
   0 s (score_mod). Risk: weaker prompt detail late, hence the mid-noise-only variant.
6. Global block-centre phase dithering across layers and steps (every token is a key somewhere); cost 0 s,
   expected effect small (does not change who decides the subject count).
7. LSE (joint-softmax) mixing: weight O_loc and O_glob per query by their softmax masses
   (sigmoid(lse_loc - lse_glob - log c)), as full attention does (Lemma split), instead of one pi per head.
   Lets a query in an empty window follow far subject tokens. Needs return_lse from both branches;
   about 30 lines, cost about +2%.
8. Latent low-band anchoring to the opening-phase x0 estimate (HRDiT structure guidance / FreeScale / ResDiT
   style) in steps 8-14: fixes layout above the cutoff, but copies of 100-400 px are mid-frequency, so
   expected effect on small copies is limited; cost about 0 s; needs a step callback.
9. Per-head shrinking windows w_h(sigma) with pi recalibrated at each emulated ratio: closest to AReA but
   needs per-head block masks and new calibration passes (GPU); deferred.

### Literature agent report (2026-10-27 23:40; AReA PDF read in full, code 9e38ec2 checked)
- AReA 4096^2: Phase I pooling f = 4,4,4,4,2,2,2,2 (steps 1-8); Phase II windows 180 ... 64 (code counts text;
  paper lists 181 ...); 2 windows per axis steps 9-15, 3 at 16-22, 4 at 23-28; nothing per head; one layout
  per step for all 57 blocks. Jitter cannot move at steps 9-15 (two pinned windows) or step 28.
- Global-KV bridge (App. E): R = (0.5 w)^2 real tokens outside the window, ranked by ||k_i - kbar||^2 of
  unrotated keys summed over heads, 8x8 cell quotas, keys at TRUE RoPE positions, one shared softmax with the
  window ("each query sets its own balance"), about 25% of a window's keys; cost cap removes it at step 9.
- AReA Table 3 ablations (H100; full 151 s, IR 1.32, FID 148.5): no bridge 133 s, FID 161.3, CLIP -0.60 and
  "duplicated and repeated subjects" (Fig. 13), the ONLY piece tied to duplication (+18 s); fixed w=64 with
  bridge: fragmentation, not duplicates (IR 1.26); flat f=4: subjects fuse (IR 1.27); flat f=2: scattered and
  repeated subjects; no Halo: seams; no DWT: slower only.
- Others: FreeScale fuses global and local bands every step (says local attention "can bring unexpected small
  object repetition"); ResDiT global-only steps 1-10, fusion 11-20, local only 21-35 (averaging "led to
  repeated generation artifacts"); HRDiT keeps every 32nd block visible to all heads; TIDE adds +ln(token
  ratio) = +2.77 to text logits for dilution (SCA has native key counts, so its formula gives 0; my
  text_local is the opposite sign, motivated by AccDiffusion: same prompt per patch repeats objects).
  RoPE fixes (DyPE, SEGA, UltraImage) do not apply: SCA stays in range, and AReA without bridge duplicates.
- Calibration file: native full attention puts 41-53% of its mass outside the emulated window (37-29%
  beyond radius 16), but the scale-4 pi gives the global branch only 0.23-0.31.
- Verdicts: (a) abrupt hand-off supported, likely main cause (SCA local-dominant after 29% of steps; AReA,
  ResDiT, FreeScale keep wide or global terms to 57-100%); (b) the subsample itself is fine (AReA's best
  layout), but s=4 throughout matches AReA's flat-f=4 failure, and the 4x4 shared query positions cannot
  place small objects; (c) supported (each 64-token neighbourhood is a native canvas); added (d) a fixed
  per-head mix instead of a per-query shared softmax; (e) same prompt in every window (not specific to SCA).
- Ported ranking: 1 bridge inside the local softmax (1024 tokens = 16 per 8x8 cell, true positions, about
  +2.4 s); 2 gradual hand-off (pi ramp, free; or window 96 -> 64); 3 spectral fusion steps 9-16; 4 per-query
  LSE merge with per-head offset matched to pi (< 0.5 s); 5 pooling f=2 at the end of the opening phase.

Decision: implement two more options before the pilot: AReA's bridge (top evidence) and the per-query LSE mix
(in-range alternative to the bridge), both in a separate compiled forward so the published path is untouched.

## Stage 3 (fix): state 2026-10-27 23:55
- Code: commits a8dd72c and 792259c pushed (origin main). Options (all default off; published sidecar block
  unchanged, test_sample_info_unchanged_by_default): --pi-ramp/--pi-ramp-mode/--pi-ramp-gamma,
  --glob-pos frac, --win-sched, --text-local/--text-local-lo, --bridge-cells/--bridge-k/--bridge-lo,
  --lse-tau. Bridge and LSE run in sca_dedup_forward (separate compile); others reuse the published fused
  forward. tests/test_dedup.py 14 tests; full suite 154 passed.
- Frozen code: /scratch/09032/saini_2/sca/code/792259c (git archive of sca scripts slurm tests results/dup_audit).
- Pilot prompts: /scratch/09032/saini_2/sca/pilot15_dup.jsonl (dev4 + 11 audited).
- Pilot tasks HELD (not pending): /scratch/09032/saini_2/sca/taskq/hold_dup/d100..d109 (base, bridge, ramp70,
  lse1, bridge_ramp70, head50, bridge_mid, txtmid, win128, frac_ramp60) + d199_eval_dup_pilot.sh, generated
  by slurm/dup_tasks.py --code .../792259c. Outputs out/dup/<label>/4096x4096; eval eval/dup_pilot.
- GPUs: none held (gh job 1103038 pending). Release recipe when a held gh node is running and idle:
  rm taskq/STOP; mv taskq/hold_dup/d1*.sh taskq/pending/; start guest workers with the exact command from
  the brief; recreate taskq/STOP when done. Watcher: background squeue loop (120 s).
- Detection of existing images running (2 CPU shards, bg task), output eval/dup_detect_20261027.

### Detector validation (all 30 eligible labeled SCA images scored; thr 0.35 fixed a priori, = GDINO default)
- Inspected every disagreement at full resolution. Detector right, audit missed it: ice climber (3 small
  climbers), artist's corner (second portrait of the man), rolling hills (3 trees for "a solitary tree").
  Detector false positives: brimmed hat read as a second umbrella (minus5673083787255800687), a twin-channel
  waterfall (minus548913239957610051, natural). Misses: steeple drawn as fused spires (one box), the small far
  Moai. Reviewed labels in validation_labels.csv column dup_reviewed (only the 3 inspected flips).
- Agreement, any subject: audit labels 7/11 on the audited 11, 23/30 overall; reviewed 9/11, 26/30
  (precision 19/21, recall 19/21). People and animals only (--kinds person,animal): audit 6/7, 18/20;
  reviewed 7/7, 20/20. thr 0.3-0.4 give identical results.
- So the table reports two rates: people/animals (primary, near-perfect on the labels) and any subject.

### Interim rates (2026-10-28 00:43, people/animals, thr 0.35, 95 prompts scored so far of about 120 eligible)
sca_nb 57%, sca_s42 57%, sca_s56 55%, sca_g7 49%, sigma 27%, sca_band (band guard) 20%, dype 17%, sega 12%,
hrdit 11%, sca_prog 8%, full 7%, ultraimage 5% (area 7.5% on any-subject earlier). Key evidence: progressive SCA
(1024^2 stage sets the layout, SCA at 2K/4K entering at sigma .81/.61 with the same pi table) duplicates as
rarely as AReA, so the SCA operator itself is fine once the layout exists; the copies form at 4096^2 between
the gate and about sigma .6, when 64-token windows (1/4 of the side) dominate an unformed layout. More steps do
not help (s42, s56); the band guard (global low band) cuts it by two thirds.

## 2026-10-28 01:15: SU freeze lifted (lead relay); pilot enqueued
- Lead: fleet SCA14 (8 GH200 nodes, 10 h, job 1105980) submitted, taskq/STOP removed, 16 w_trace tasks pending.
- Pilot: 14 generation tasks taskq/pending/a_d100..a_d113 (base, bridge, ramp70, lse1, bridge_ramp70, bridge32,
  bridge_lse1, head50, ramp60, frac_ramp60, bridge_mid, txtmid, win128, frac_g80), code freeze
  /scratch/09032/saini_2/sca/code/bd42387 (commit bd42387), outputs out/dup/<label>/4096x4096.
- Evaluations never sleep: they park in taskq/hold_dup_eval and slurm/release_dup_evals.sh (PID 1256926 on
  i618-062, log logs/release_dup_evals.log, 120 s loop, exits when hold is empty or STOP) moves each to pending
  when its RELEASE_AFTER group has drained. Pilot eval: a_d199_eval_dup_pilot.sh -> eval/dup_pilot
  (dup_rates.txt, dup_rates_people.txt, extra/, base/, timing printout in its task log).
- Old held copies moved to taskq/hold_dup/superseded_20261028 (not deleted).
- Next: when pilot numbers land, pick the winner, then dup_tasks.py --stage abl46 --prefix b_ (46-prompt paired
  run) and --stage rerun195 --prefix x_ (473 generation tasks, evals z980-z988 in hold_dup_eval), restart the
  release loop if it exited.

### Stage 2 final tables (2026-10-28 01:45; results/dup_audit_20261027/rates_*.csv, build.sh rebuilds them)
Detection finished: 3476 images in 8560 s (2 CPU shards x 12 threads). thr 0.35, Wilson 95% intervals, paired
McNemar vs sca_nb. People/animals rate over the 95 prompts that imply a person or animal count; any-subject rate
over the 147 prompts with any numeric count.
| method (4096^2, 195-prompt set) | people/animals | any subject |
|---|---|---|
| SCA (sca_nb) seed 0 / 1 / 2 | 56.8% [46.8,66.3] / 49.5 / 55.8 | 55.8% [47.7,63.6] / 52.4 / 57.1 |
| SCA 42 / 56 steps, guidance 7 | 56.8 / 54.7 / 49.5 | 58.5 / 57.8 / 50.3 |
| SCA + band guard | 20.0 [13.2,29.1] | 24.5 |
| progressive SCA | 8.4 [4.3,15.7] | 15.0 |
| AReA seed 0 / 1 / 2 | 4.2 [1.6,10.3] / 3.2 / 6.3 | 7.5 [4.2,12.9] / 13.6 / 13.6 |
| full attention | 7.4 | 6.8 |
| UltraImage | 5.3 | 15.0 |
| HRDiT | 10.5 | 14.3 |
| SEGA | 11.6 | 17.0 |
| DyPE | 16.8 | 25.2 |
| SigMa | 27.4 | 28.6 |
46-prompt ablation set (26 / 34 eligible): sca_nb 57.7 / 61.8; gate .93: 76.9 / 73.5; gate .85: 26.9 / 26.5
(p .008 / .002); no gate (v0) 100 / 100; local only 96 / 85; global only 3.8 / 14.7; shared pi 69 / 62; per-layer
77 / 71; mean pooling 96 / 91; band guard 11.5 / 23.5; with the guard: local only 27, shared 19, per-layer 11.5,
mean 42, routing .3/.5 19.
Reading: duplication falls monotonically with the global weight at mid noise (gate .93 > .89 > .85 > global only),
the per-head table beats shared or per-layer weights, and a global low band (guard) removes most of it. But gate
.85 costs aesthetic -0.19, ARNIQA -0.04, CLIP-IQA -0.03, HPSv2.1 -0.009, MUSIQ -1.9, MUSIQ_c -2.0 (all
significant, results/e7_sca_g85_deltas.csv) and the guard costs HPSv2.1 -1.17 etc.; the pilot variants try to
keep the global context without the hard switch.
Other sizes (eval/dup_detect_20261027_sizes, 656 images): people/animals SCA vs AReA 43.2% vs 17.9% at 2048x4096,
53.7% vs 6.3% at 2048x6144; any subject 46.3% vs 22.4% and 55.8% vs 15.0% (all paired p < 1e-4).

## 2026-10-28 07:45: still no GPU (SCA14 pending 6.5 h, no start estimate)
- Enqueued 46-prompt paired runs (abl50 incl. dev4, chunks of 5) for four candidates behind the pilot, prefix b_:
  taskq/pending/b_d2XX_{dup_bridge,dup_bridge_ramp70,dup_ramp70,dup_lse1}.sh (40 tasks, out/dup46/<label>), evals
  z97_eval_dup46_<label>.sh parked in taskq/hold_dup_eval (detector + E7 metric protocol + paired bootstrap vs
  sca_nb, eval/dup46_<label>). About 4.5 GPU-h + 4 evals.
- Staged (NOT enqueued) the 195-prompt headline reruns for the two leading candidates: taskq/hold_dup/stage_x_<label>
  for dupfix_bridge and dupfix_bridge_ramp70 (each 473 generation tasks x_d90..x_d98 over rows m4k, m24, m26, s1,
  s2, pg, kr, hd, hk, plus 9 evals z980..z988 in stage_x_<label>/evals). Release for the winner L:
    mv taskq/hold_dup/stage_x_L/x_*.sh taskq/pending/ && mv taskq/hold_dup/stage_x_L/evals/z98*.sh taskq/hold_dup_eval/
  and make sure slurm/release_dup_evals.sh is running. Estimate about 100 GPU-h (65 h generation incl. 41 h
  HRGen-1K, 24 h load/compile at about 3 min per task, 11 h evaluation).
- Pilot summary when eval/dup_pilot/DONE exists: python scripts/dup_pilot_summary.py /scratch/09032/saini_2/sca/eval/dup_pilot
  (and eval/dup46_<label> with OUT_ROOT out/dup46).
