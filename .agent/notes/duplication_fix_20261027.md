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

## 2026-10-28 18:35: pilot numbers (eval/dup_pilot; results/dup_audit_20261027/pilot_20261028)
15 prompts (4 dev + 11 audited), seed 0, 4096^2, code bd42387, run on the guest GPU of job 1103038 (c622-021).
Duplicates people/animals (of 9) and any subject (of 14); metric deltas vs sca_nb on the 11 non-dev prompts
(* = 95% bootstrap interval excludes 0); median s/image vs dup_base 70.8 s.
- dup_base 6/9, 9/14, 70.8 s: reproduces the published images (all deltas ~1e-7).
- dup_bridge_ramp70 1/9, 2/14, +4%: IR +0.10, HPSv2.1 -0.005, PickScore -0.13, aesthetic +0.02 (n.s.);
  CLIP-IQA -0.079*, ARNIQA -0.057*. Reference rows on the same prompts: AReA 0/9, 1/14; full 0/9, 0/14;
  progressive SCA 1/9, 2/14; HRDiT 1/9, 1/14.
- dup_head50 2/9, 4/14, +0%: IR +0.16, PickScore +0.18 (n.s.), CLIP-IQA -0.029*.
- dup_bridge32 2/9, 6/14, +7%; dup_bridge 3/9, 7/14, +4% (PickScore +0.19*); dup_bridge_mid 3/9, 7/14, +3%.
- dup_ramp70 3/9, 3/14 (CLIP-IQA -0.066*, ARNIQA -0.051*); dup_ramp60 4/9, 4/14 (CLIP-IQA -0.107*, ARNIQA -0.086*).
- dup_frac_ramp60 3/9, 4/14 (MUSIQ -2.85*); dup_frac_g80 4/9, 6/14, -3%.
- dup_txtmid 3/9, 8/14 but IR -0.33*, HPSv2.1 -0.010*, PickScore -0.30*: rejected.
- dup_win128 3/9, 6/14, +14%: too slow. dup_lse1 6/9, 10/14 and worse PickScore: rejected (46-prompt run withdrawn
  to taskq/hold_dup/withdrawn_lse1_20261028). dup_bridge_lse1 4/9, 7/14.
Reading: the bridge (AReA's mechanism) and the gradual hand-off act on different copies (bridge: people;
ramp: objects too) and combine to AReA's level at +4% time; the ramp's extra global weight costs the no-reference
sharpness metrics, which on 195 prompts SCA leads AReA by only +0.053 CLIP-IQA and +0.039 ARNIQA.
Decision: winner dup_bridge_ramp70 (meets the stated criteria on the pilot: duplication at AReA's level, no
significant IR/HPSv2.1/PickScore loss, +4% time; FID pending). Its 195-prompt reruns are ENQUEUED (taskq/pending
x_d90..x_d98, 473 tasks, label dupfix_bridge_ramp70, evals z980-z988 in hold_dup_eval). 46-prompt paired runs
queued: bridge, bridge_ramp70, ramp70, and bridge_head50 (added; tests whether per-head opening keeps
CLIP-IQA/ARNIQA at equal duplication). If bridge_head50 wins at 46 prompts, swap the reruns: move x_*bridge_ramp70*
back to taskq/hold_dup/stage_x_dup_bridge_ramp70 and stage dupfix_bridge_head50 with dup_tasks.py --stage rerun195.

## 2026-10-28 18:55: visual check changes the winner
- Viewed the astronaut prompt (minus1071233499893339092): base draws about 7 astronauts; bridge_ramp70 draws one,
  but its spires carry a vertical lattice; ramp70 alone shows 4x4-block streaks at the spire edge (the global
  branch's shared query positions during the ramp), which is what CLIP-IQA/ARNIQA penalised; bridge alone is clean
  but keeps 2 tiny astronauts; head50 is clean with at most one tiny figure.
- So the linear ramp is rejected for the reruns. Reruns now ENQUEUED for dupfix_bridge_head50 (bridge 8x8x16 +
  per-head opening, ramp 0.5): taskq/pending x_d90..x_d98 (473), evals z980-z988 (label dupfix_bridge_head50) in
  hold_dup_eval. bridge_ramp70 reruns moved back to taskq/hold_dup/stage_x_dup_bridge_ramp70 (none had started).
- bridge_head50 has no 15-prompt pilot; its 46-prompt paired run (b_d2XX_dup_bridge_head50, eval
  eval/dup46_dup_bridge_head50) runs before the reruns and is the check: duplication at AReA's level, no significant
  IR/HPSv2.1/PickScore/patch-FID loss, CLIP-IQA/ARNIQA within SCA's margin over AReA (+0.053/+0.039). If it fails,
  withdraw: mv taskq/pending/x_*dupfix_bridge_head50* and hold_dup_eval/z98*dupfix_bridge_head50* to
  taskq/hold_dup/stage_x_dup_bridge_head50/ (evals/ for z98*), then stage the next candidate with
  slurm/dup_tasks.py --stage rerun195 --prefix x_.
- Release loop PID 1256926 (i618-062) must stay up until the z98x evals are released; restart with
  nohup bash slurm/release_dup_evals.sh >> /scratch/09032/saini_2/sca/logs/release_dup_evals.log 2>&1 &.
  Do not recreate taskq/STOP while the fleet should drain the queue (it also stops the release loop).

## Chosen operator, exact definition (candidate: dupfix_bridge_head50; flags --bridge-cells 8 --bridge-k 16
## --pi-ramp 0.5 --pi-ramp-mode head; code sca/attn.py sca_dedup_forward, bridge_index, SCAContext.ramp)
Notation as in paper Section 2. Grid H x W tokens (256 x 256 at 4096^2), text set T, sigma_g = 0.89, sigma_r = 0.5.
1. Global operator: unchanged (Eq. glob): text keys plus one block-centre key per s x s block, coarse ids floor(p/s).
2. Bridge set, recomputed in every block l and step: partition the grid into 8 x 8 cells (32 x 32 tokens at 4096^2);
   score each image token j by sum_h ||k_{j,h} - kbar_h||^2, where k_{j,h} is the pre-RoPE (post-RMSNorm) key of head h
   and kbar_h its mean over image tokens (AReA Eq. 17); in each cell keep the 16 highest-scoring tokens:
   B_l(sigma), |B| = 1024, the same set for all heads of block l.
3. Local operator: K^loc_i = {(k_j, v_j, 0)}_{j in T} U {(k_j, v_j, p_j)}_{j in B} U {(k_j, v_j, p_j)}_{p_j in Win(i)},
   one softmax, RoPE at true positions for bridge keys (their offsets to the query reach H - 1, so Prop. 1 holds for
   the window keys only; bridge keys are 1024 of 5,632 image-or-text keys per query, 18%). A token in both B and the
   window appears twice (not deduplicated). Tile t = 8, window w = 64 as before.
4. Per-head opening: for image queries of head h in block l at noise sigma,
     O_i = O^glob_i                                   if sigma > sigma_g,
     O_i = pi'_lh O^loc_i + (1 - pi'_lh) O^glob_i     otherwise, with
     pi'_lh(sigma) = pi_lh(sigma) if sigma < sigma_r + (sigma_g - sigma_r) pi_lh(sigma), else 0.
   So a head with table weight pi opens at sigma_open = 0.5 + 0.39 pi: pi 0.9 -> 0.85 (step 11), median 0.73 ->
   0.79 (step 13), 0.5 -> 0.70 (step 16), 0.3 -> 0.62 (step 18); heads that prefer the global branch stay global
   longest. Text queries take O^glob. pi table unchanged (calib/k1_1024.npz, scale-4 entry at 4096^2).
5. Cost: bridge selection (one fused reduction and a per-cell top-k per block and step) plus 1,024 more local keys
   (4,608 -> 5,632); pilot bridge alone 73.5 s vs 70.8 s (+4%); per-head opening costs nothing.
Decision rule from the lead (scripts/dup_decide.py): people/animal rate <= 10% on the 46 prompts with a Wilson
interval overlapping AReA's; no significant IR, HPSv2.1 or PickScore loss vs sca_nb; CLIP-IQA and ARNIQA stay ahead
of AReA (paper's 195-prompt lead +0.053 / +0.039 plus the variant's paired delta; the same-prompt 46 lead is only
+0.029 / +0.022, printed as a sensitivity check); median time <= +10%.
Ordering fix 18:40: the 46-prompt evaluations were renamed b_z97_eval_dup46_<label>.sh so that, once released, they sort
before the w_ and x_ tasks (z97 would have sorted after all 473 x_ tasks).

## 2026-10-28 19:00: user directive (binding, via lead) and round two
- Directive: the fixed method must reach AReA's duplication level AND meet or exceed every current SCA number (no
  significant paired loss on any metric vs published SCA); CLIP-IQA/ARNIQA losses are not acceptable; prefer one
  simple mechanism over stacked patches. Strict rule applied with scripts/dup_decide.py (STRICT line: every per-image
  metric in the paired bootstrap plus patch FID/KID, duplication <= 10% overlapping AReA's interval, time <= +10%).
- Done: all 473 x_ rerun tasks withdrawn before any started (taskq/hold_dup/stage_x_dup_bridge_head50, evals in
  evals/); nothing reruns until a variant passes strictly. 46-prompt evals renamed b_z97_* (sort before w_/x_).
- Round two, candidate 2 (free): per-head opening, slower ramp sigma_r = 0.4: pilot enqueued as a3_d100_dup_head40.sh,
  eval a3_d199_eval_dup_pilot2.sh -> eval/dup_pilot2.
- Round two, candidate 1 (user's): per-head apparent scale. Each head attends a native n0 x n0 canvas taken from the
  stride-(s/a) lattice around its query tile (coarse ids floor(p/d), so offsets stay below n0 and the cost never
  exceeds the published local branch); its apparent scale a grows 1 -> 2 -> 4 (whole image -> half image at stride 2
  -> native window) on a per-head timetable sigma_end_h = sigma_lo + (0.89 - sigma_lo) pi_lh; mix weight from the
  table at scale a (a = 1 is the global branch itself, a = 4 the published operator). Global-only phase, published
  SCA and per-head opening are special cases; the wide early windows cost nothing extra because the key count is
  fixed, so no 2 n0 bound is needed. Implementation delegated to an Opus subagent (tests, freeze, pilot a4_ dup_ss50
  and dup_ss40, eval/dup_pilot3).
- 19:40 apparent-scale operator landed (subagent): commit f17451c, freeze code/f17451c, 177 tests pass (23 new in
  tests/test_scale_sched.py). Flex path: one call per stride over all heads, heads outside the group masked off by
  zeroing their block counts (one compiled graph); stride-2 windows do 1.055x block work, step total peaks ~1.02x
  published (steps 13-19); a = 1 heads do no local work. About 13% of head-layers (pi near 1) still jump from a = 1
  to 4 in one step near the gate. Pilot enqueued: a4_d100_dup_ss50, a4_d101_dup_ss40 (eval a4_d199 -> eval/dup_pilot3).
- 22:00 head40 pilot (eval/dup_pilot2): 1/9 people, 2/14 any (AReA level), +0% time; IR +0.14, PickScore +0.18,
  HPSv2.1 +0.002, CLIP +1.02, ARNIQA +0.007, MUSIQ_c +0.12 (n.s.), MUSIQ -2.8 (n.s.), CLIP-IQA -0.031 (significant
  on 11 prompts). Queued its 46-prompt paired run: c_d2XX_dup_head40 (10 tasks, code f17451c), eval
  c_z97_eval_dup46_dup_head40.sh -> eval/dup46_dup_head40 (sorts after the b_ work, before w_).
- 2026-10-29 05:35: the gg control job ended ~21:45, killing the release loop (PID 1256926) and my watchers on
  i618-062; all generation (b_ 40, c_ 10, a3_, a4_, w_ 16) had finished, six evaluations were stuck in hold_dup_eval.
  Moved all six to pending for fleet SCA15 (4 gh-dev nodes): a4_d199 (apparent-scale pilot -> eval/dup_pilot3),
  b_z97 x4, c_z97 (head40). Operator host is now c622-021 (cntrl-cmd gh). No release loop is needed for these.
- 06:10 apparent-scale pilot (eval/dup_pilot3): ss40 3/9 people, 6/14 any, +1% time, IR +0.136*, HPSv2 +0.0027*,
  ARNIQA +0.012*, PickScore +0.27, CLIP-IQA -0.021*; ss50 3/9, 7/14, +1%, IR +0.144*, HPSv2 +0.0033*, CLIP-IQA
  -0.018*. Better preference scores than any variant so far, but duplication only halves: the stride-2 half-image
  canvas at mid noise is itself a native canvas covering a quarter of the image, so it can redraw the subject one
  level up. Per-head opening (head40: global until each head's own sigma) removes more (1/9, 2/14).
- Added to scripts/dup_decide.py: deltas stratified by detector flags (prompts the variant fixed, clean in both,
  other) to test whether CLIP-IQA/ARNIQA losses come from removing the sharp duplicates themselves (the paper already
  notes v0 scores higher on both).

## 2026-10-29 05:55: 46-prompt decision (strict rule): NO VARIANT PASSES; x_ reruns stay held
People/animals duplicated (of 26 eligible; sca_nb 15/26 = 58%, AReA 1/26 = 4%), time, significant paired losses vs
published SCA (95% bootstrap; results/dup_audit_20261027/dup46_20261029/decision.txt has every delta):
- bridge 5/26 (19%), +4%: only MUSIQ_c -1.26 [-1.84,-0.74]; everything else n.s. (nearly metric-neutral).
- head40 5/26 (19%), +0%: aesthetic -0.067, CLIP-IQA -0.020, HPSv2 -0.0029, HPSv2.1 -0.0067, MUSIQ -1.5.
- bridge_head50 4/26 (15%), +4%: aesthetic -0.072, CLIP-IQA -0.022, HPSv2 -0.0019, HPSv2.1 -0.0039, MUSIQ_c -1.05.
- bridge_ramp70 3/26 (12%), +4%: losses on 10 of 12 metrics (CLIP-IQA -0.068, ARNIQA -0.070, PickScore -0.22, ...).
- ramp70 5/26 (19%), +0%: losses on 9 metrics. The linear ramp is out.
Stratified CLIP-IQA deltas: bridge -0.012 on prompts it fixed, +0.006 on prompts clean in both (its small loss comes
from removing the duplicates themselves); head40 -0.017 fixed, -0.008 clean (partly genuine softening).
Round two pilots (15 prompts): apparent scale ss40 3/9, 6/14, +1%, IR +0.136*, HPSv2 +0.0027*, ARNIQA +0.012*,
CLIP-IQA -0.021*; ss50 3/9, 7/14, IR +0.144*, CLIP-IQA -0.018*; head40 1/9, 2/14, CLIP-IQA -0.031*.
Reading: the 15-prompt pilot overstated every effect (head40 1/9 there, 19% here). The bridge is the only mechanism
that is close to neutral on quality; nothing so far reaches AReA's 4% without a quality cost.
- 06:05 to keep SCA15 busy (pending was empty, w_trace done), queued two 46-prompt runs from the lead's candidate
  list using existing flags (code f17451c): c_d2XX_dup_bridge_win128 (bridge 16/cell + 2 n0 windows while sigma >
  0.80, the "bridge with a shorter 2x phase") and c_d2XX_dup_bridge32 (bridge 32/cell). Evals c_z97_eval_dup46_* in
  hold_dup_eval, release loop restarted: PID 3210493 on c622-021 (log logs/release_dup_evals.log). x_ still held.

## 2026-10-29 06:00: lead decision, round two on 46 prompts directly
Reading (lead): every fix that keeps heads global longer softens detail; only enriching the global branch (bridge)
is neutral. Anchor the subject by something other than delaying the local branch. Order, each about 1 GPU-h:
(A) latent low-band anchor: x0 prediction of the last global-only step is the layout anchor L; later steps replace
the block-scale low band of x0_hat by L's with weight 1 at the gate decaying to 0 at sigma 0.3 (linear, cosine;
refresh-every-4 variant with an extra global-only forward); (B) bridge 64/cell until sigma 0.6 then 16; (C) A +
bridge 16; (D) A + the better of bridge_win128 / bridge32 (both running now as c_ tasks). Strict rule unchanged.
If A alone passes: third component, "the opening phase decides the layout; the low band of that decision is held
while detail is filled in". Implementation (A, B, self-releasing evaluations) delegated to an Opus subagent; tasks
prefix d_, evals d_z97_*. The c_ evals still use the release loop PID 3210493 on c622-021 (control job with 31 h
left, outlives the fleets); new d_ evals release themselves from the last generation task's marker.
- 06:20 round two implemented (subagent): commit 60fb75f, freeze code/60fb75f, 192 tests (15 new in
  tests/test_anchor.py). Anchor: L = x_t - sigma v at the last global-only step (step 7 of 28); for sigma in (0.3,
  0.89] (steps 8-24) v' = v - (w/sigma) LP(L - x0_hat), LP = 4x4-token average over all 64 packed channels then
  bilinear upsample, w = u (linear) or sin^2(pi u/2) (cosine), u = clip((sigma - 0.3)/0.59, 0, 1); refresh K=4 adds
  global-only forwards at steps 11, 15, 19, 23 (about +12-14% time, likely over the limit). Bridge k64mid: 4096
  bridge tokens at steps 8-18, then 1024. Enqueued 50 tasks d_d2XX_{dup_anchor_lin, dup_anchor_cos,
  dup_anchor_lin_r4, dup_bridge_k64mid, dup_anchor_lin_bridge}; evals d_z97_* release themselves when the 10th
  generation marker appears (eval/dup46_<label>/gen_done). x_ still held.
- 06:45 c_ results (46 prompts, strict): bridge_win128 (bridge 16/cell + dense 128 windows while sigma > 0.80, i.e.
  steps 8-12) 1/26 people = 3.8% = AReA's 1/26; time 82.96 s (+17%, fails <= +10%); significant losses ARNIQA -0.0115,
  HPSv2.1 -0.0025, MUSIQ_c -2.49; all else n.s. bridge32: 5/26 (19%), +7%, only MUSIQ_c -1.48. So AReA's recipe
  (wide windows at the first post-gate steps plus a global bridge in the same softmax) removes duplication; the
  dense early windows cost the time.
- Enqueued (D) and two cheaper versions, prefix e_ (after d_), code 60fb75f, self-releasing evals:
  dup_anchor_lin_bridge_win128 (anchor + bridge + 128 windows sigma > 0.80), dup_bridge_win96 (96 windows sigma >
  0.80, ~2.25x local for 5 steps), dup_bridge_win128s (128 windows only while sigma > 0.85, 3 steps).
- 07:05 lead: time limit relaxed to +20%; each significant loss is split into prompts the variant fixed vs prompts
  clean in both; a loss on clean-in-both prompts is real softening (not acceptable). scripts/dup_decide.py now prints
  per-stratum paired means with 95% bootstrap intervals and flags real softening (clean_both interval below 0).
  bridge_win128: fixed prompts (n=16) all n.s. except MUSIQ_c -2.2; clean in both (n=13) IR -0.097 [-0.17,-0.03],
  HPSv2 -0.0038, HPSv2.1 -0.0049, CLIP -0.61, ARNIQA -0.025, MUSIQ_c -2.1 all significant: real softening on images
  that had no duplicates, so it fails the lead's rule despite reaching AReA's duplication rate.
