# Late duplication, round 6 (method owner, started 2026-11-05 00:30 CST)

User directive (binding, 2026-11-05): "we have to remove that issue and also make sure our method is SOTA and novel."
Issue: small repeated copies of the subject at 4096^2. Earlier rounds: .agent/notes/duplication_fix_20261027.md.
Live state of this round: .agent/notes/round6_state.md.

## Step 1: the human-validated bar (done 2026-11-05 01:15, commit 9f464ba)

Sample: every detector-flagged image (people/animal queries, thr 0.35) of AReA (4), HRDiT (10), full attention (7) and
sca_v2 (25) on the 95 people/animal prompts at 4096^2, plus 30 random unflagged images per method (seed 20261105):
166 images, shuffled across methods (seed 7), anonymous ids. 42 blind Opus labelers (4 images each) viewed the
overview and all 16 native 1024 px tiles of each image, with the prompt, the person/animal subjects and their implied
counts (subjects_a4k195.json), and wrote counts, a yes / borderline / no label, the location and size of each extra
figure and whether it repeats the subject's look (protocol: results/dup_audit_20261027/human_labels_95_protocol.md).
Adjudication (me, from full-resolution crops of every detector box and every labeler candidate): all 21 images where
the blind label disagreed with the detector or was borderline, and the 4 agreed yes labels of medium confidence
(human_labels_95_adjudication.json). Outcome of adjudication: 2 confirmed detector misses (AReA: two tiny people above
a roofline; sca_v2: three small children in a background field under group boxes), 11 detector false flags (body
plus head boxes of one figure, a dog read as a person, reflections, prompted portraits, figurines), 7 kept borderline,
1 yes downgraded to borderline (an arm on a glass door).

Human-corrected rate = (confirmed flags + U x k / 30) / 95, U the method's unflagged prompts, k its duplicates among
the 30 sampled; 95% interval = Wilson interval of k / 30 carried through (results/dup_audit_20261027/human_rates_95.txt):

| method | detector | precision | unflagged dup | human, primary (yes) | with borderline |
|---|---|---|---|---|---|
| AReA | 4.2% | 1/4 | 1/30 | 4.2% [1.6, 17.0] | 7.4% |
| HRDiT | 10.5% | 6/10 | 0/30 | 6.3% [6.3, 16.5] | 9.3% |
| full attention | 7.4% | 4/7 | 0/30 | 4.2% [4.2, 14.7] | 11.4% |
| sca_v2 | 26.3% | 23/25 | 1/30 | 26.7% [24.6, 36.5] | 27.7% |

- The detector is near exact on sca_v2; on the baselines its false flags and misses roughly cancel. The earlier estimate
  "AReA plausibly 13%" (10 flagged + 10 unflagged sample) does not hold: AReA's human rate equals its detector rate.
- Full attention at 4096^2 fails outright on many prompts: the subject is missing in 22 of 37 labeled images (flat
  gradient bands), so its low rate is not a quality signal.
- sca_v2's copies: 95 extra figures in 24 of 95 images, median height 135 px (about 8 tokens), a third repeating the
  subject's look (same clothing, pose or species), the rest generic small figures (pedestrians, distant hikers).
- Bar for a fix: human-corrected rate at or below AReA's on the same prompts, no significant paired loss versus sca_v2
  on any of the 15 metrics, every current win over AReA kept, at most +25% time.

## Mechanism evidence so far
- Copies are spread over the whole image: published SCA's extra boxes fall in the top-left 1024 px cell (where the
  window's text-image RoPE offsets stay in the trained range) 5.5% of the time vs 6.25% uniform, and most often in the
  left column. Out-of-range text offsets in the window branch are not the cause.
- Step 2 GPU measurements queued: x0 traces at 10 labeled copies (sca_v2 vs AReA, crop on the copy), and an attention
  probe (scripts/r6_attn_probe.py) at 6 copies: text, subject-word and main-subject mass of full attention, the window
  branch and the global branch at the copy, the main subject and background, per head and step.

## Step 3 candidates (round 1, 46-prompt subset, all on top of sca_v2; code sca/textgate.py, commit 09492d5)
The window softmax is split exactly into its text part and its image part (bridge plus window, flex with log-sum-exp)
and merged as one softmax; a gamma-0 split run is the GPU identity check (a6_a000_smoke, PSNR >= 24 dB vs sca_v2).
- r6_ctw (candidate c, calibrated text weight): text logits in the window get beta_lh(sigma, r) = log(near-field
  share), the fraction of full attention's image mass inside the emulated window at 1024^2 (calibration masses, radius
  n0 w / (2 max(H, W)), log-linear between radii 8 and 16). All post-gate steps. Typical beta at window 64: mean -1.1,
  p10 -2.2; 30 to 46% of heads keep less than 0.3 of their mass inside the window.
- r6_gtg (candidate a, global text gate): window logit on text token j at query i gets gamma min(0, log(m_ij / mean_i
  m_ij)), m = global branch's attention probability on j at i (head mean), floor -6, gamma 1, sigma > 0.5.
- r6_gtx (candidate b, global text read-out): O_loc = a_g O_T(coarse ids) + (1 - a_g) O_bridge+window, a_g the global
  branch's text mass; sigma > 0.5.
- r6_bw (own, context side): bridge keys get + log(N / R) = log 64, the tokens each stands for; sigma > 0.5.
- Candidate d (native opening, --native-open K, sca/native_open.py, commit 8c0cd0c) is implemented, held for round 2.

## Prior art for the round-1 candidates (literature subagent, 2026-11-05 01:40; citations as reported, to verify before the paper)
- Global text gate (gtg): closest AccDiffusion (Lin et al., ECCV 2024, arXiv 2407.10738): binarizes each word's
  low-resolution cross-attention map at the word's image mean, keeps the word in a patch prompt only if the mask covers
  more than 30% of the patch; its diagnosis ("small object repetition is the adversarial outcome of an identical text
  prompt on all patches") is the same as ours. New in gtg: a continuous one-sided log bias per query and word, read
  online from the same layer's global branch inside the joint softmax (no second pass, no prompt editing). Layout or
  region methods (DenseDiffusion 2308.12964, Regional Prompting for FLUX 2411.02395, Prompt-to-Prompt, FAM 2411.18552,
  HiPrompt 2409.02919, ResMaster 2406.16476) need user masks, MLLM prompts or another pass.
- Calibrated text weight (ctw): closest TIDE (arXiv 2603.08928, 2026): one constant +ln(pixel ratio) on text logits
  under full attention (raises text weight; opposite sign to ctw); I-Max (2410.07536) repeats text tokens; Jin et al.
  (NeurIPS 2023, 2306.08645) scale logits by sqrt(log_T N); Re-ttention (NeurIPS 2025, 2505.22918) measures the in-mask
  share of full-attention mass from cached full steps and rescales sparse attention. New in ctw: an offline per-head,
  per-noise table from native-resolution emulation, applied only to text logits inside windows, lowering a window's
  text share to the full-attention value. Window methods with all text keys add no compensation (CLEAR 2412.16112;
  TVSDA 2611.01696 reports repetition from local attention).
- Global text read-out (gtx): closest FreeSwim (arXiv 2511.14712): replaces the window branch's text cross-attention
  output with the full branch's in Wan2.1 (separate cross-attention) and shows duplicates receding as the swap weight
  grows. Only the joint-softmax form (global text mass and text output, image keys renormalized) is new.
- Weighted bridge (bw): standard importance weighting of sampled keys (sampled softmax, Blanc and Rendle 1712.00527;
  HyperAttention 2310.05869; ToMe log s 2210.09461); AReA's bridge has no count correction and is a deterministic
  top-k, so log(N/R) is a heuristic there. Novelty small.
- How baselines treat duplicates: AReA names it (a window cannot see a copy elsewhere; bridge removal brings duplicates
  back); DyPE and UltraImage address periodic RoPE repetition, not subject copies; HRDiT addresses spatial disorder;
  FreeScale and ScaleCrafter blame receptive field (FreeScale notes small-object repetition in local-patch methods).

## Round 1 results (46-prompt subset, 26 people/animal prompts; scripts/r6_decide.py; eval/r6_46_<label>)
- GPU path checks: the exact window split with gamma 0 reproduces sca_v2 at 34.4 / 32.0 dB PSNR (smoke2, two prompts;
  differences are bf16 trajectory noise); gtg first failed to compile (a tensor computed in the same graph cannot be a
  captured flex buffer), fixed by a graph break (commit 049d062). Two transient task failures: stale Lustre file handles
  in the shared Triton cache (requeued); a per-node cache made Triton rebuild its CUDA stub with nvc and fail (reverted).
- r6_ctw (candidate c, calibrated text weight, all post-gate steps): FAIL. Detector people/animal 4/26 (sca_v2 4/26,
  AReA 1/26), any subject 7/34 (sca_v2 6/34); human check of its 4 flags: 3 yes, 1 borderline; 3 sampled unflagged:
  0 yes; human-corrected 14% vs sca_v2 17% vs AReA 2.2%. Significant losses vs sca_v2: IR -0.057, PickScore -0.098,
  HPSv2.1 -0.0029, HPSv2 -0.0018, aesthetic -0.034, CLIP-IQA -0.011, patch FID +4.2, patch KID +0.0021 (also on
  prompts clean in both: HPSv2, HPSv2.1, PickScore, CLIP-IQA); gains MUSIQ, MUSIQ_c; win over AReA lost on CLIP-IQA;
  time 80.2 vs 76.9 s (+4%). It also leaves a band of colored garbage in the last token column (right edge, 16 to 24 px)
  on about a third of its images (8 of 37); the gamma-0 split is clean on the same prompts and ctw restricted to
  sigma > 0.5 still shows it on 3 of 5, so lowering the window's text weight at mid noise destabilizes the last column.
  Reading: the window's prompt over-read (2x median) is real but is not what seeds the copies; restoring the native
  text balance only weakens prompt adherence. The copies need context (the window must see that the subject exists
  elsewhere), not a weaker prompt.

## Step 2 statement (reported 2026-11-06 00:45)
- When: x0 traces of sca_v2 and AReA at 10 human-labeled copies (results/round6_mechanism/trace_sheets: crops on the
  copy, steps 6 to 21, sca_v2 above AReA) and the web traces of published SCA: every copy is seeded in the first 1 to 3
  window steps after the gate (sigma 0.888 to 0.85), is a recognizable figure by 0.85 and finished by about 0.75. It
  grows on a structure the global-only phase left open (a second rock gets an astronaut for "astronaut sits on a rock";
  a snowy ridge path gets a climber). AReA's coarse layout at the same place is plain sand or slope.
- Why (probe, results/round6_mechanism/probe_*.csv): at the copy sites in those steps the window reads the prompt about
  2x as strongly as full attention (text mass 0.06 to 0.10 vs 0.035 to 0.05) and sees the main subject with 0.4 to 0.5x
  full attention's mass (0.035 to 0.045 vs 0.08 to 0.10). Interventions: removing the over-read (c; measured over-read
  ~1.0) leaves the copies; far-field mass reduces them in proportion to its weight (bridge 57% -> 19%; calibrated share
  ffc 4 -> 3 of 26; uncalibrated weighted bridge 0 of 26 with destroyed images; wide windows at the hand-off reach AReA's
  level but soften). The window completes the prompt's composition on whatever prop the coarse layout offers; the lever
  is what it sees of the existing subject, not how strongly it reads the prompt.
- Round-1/2 human-checked duplicates on the 26 people/animal prompts (detector flags confirmed): sca_v2 4, ctw 3,
  gtg 4, ffc 3, native opening K=12 2 (2 of 4 flags false); detector only: gtx 8, K=8 7, bw 0 (images destroyed).

## Round 2 (2026-11-06)
- r6_gtg (a): detector 4/26 (the same prompts as sca_v2), 4 human-confirmed; no significant loss, all 10 wins over
  AReA kept, +12% time. Fails on duplication: the global branch's subject-word attention is as high at copy sites as at
  the main subject, so the gate has nothing to act on.
- r6_ffc (far-field calibrated share, sigma > 0.5): detector 3/26, 3 human-confirmed, human-corrected 14.1% (sca_v2
  17.0%, AReA 2.2%); significant losses PickScore -0.27, IR -0.071, CLIP -0.65, HPSv2.1 -0.0063, HPSv2 -0.0024,
  aesthetic -0.049, MUSIQ_c -1.55 (also on prompts clean in both: IR, HPSv2.1, PickScore, CLIP); 8/10 wins over AReA
  kept (aesthetic and CLIP-IQA lost); +3.4% time. FAIL: the far field buys copies down with prompt adherence.
- CPU-detector reads (GPU evaluations queued): gtx 8/26, native opening K=12 4/26 (2 human-confirmed), K=8 7/26.
- Lead's round 2 (queued after the pending evaluations, code 8937ed1): weighted bridge gamma log(N/R) with gamma 0.25,
  0.5, 0.75 only for sigma in (0.84, 0.89] (steps 8 to 10, the seeding steps), plain sca_v2 elsewhere (r6_bwh25/50/75);
  subject bridge (bridge tokens = block centres most attended by the subject words in the global branch, gamma 0.5,
  same steps; r6_sbh50; subject words r6/subject_words.json). Then r6_ffl (lattice far field).
- Round 2 results (46 prompts; detector flags on the 26 people/animal prompts, sca_v2 4, AReA 1):
  - r6_bwh25 (bridge + 0.25 log(N/R) at sigma 0.888, 0.870, 0.850 only): 3 flags (a subset of sca_v2's; fixes the canoe
    prompt), no significant loss, 10/10 wins over AReA kept, no clean-prompt softening, time x1.004. Quality-neutral,
    duplication reduction not significant (4 -> 3).
  - r6_bwh50 / r6_bwh75: still 3 flags; 10 / 11 significant losses (IR -0.16 / -0.43, PickScore -0.69 / -1.51, patch
    FID +20.5 / +51.3), 4/10 and 2/10 wins kept. Duplication saturates at 3 of 26 while quality collapses.
  - r6_sbh50 (subject-attended bridge, gamma 0.5, hand-off steps): 4 flags (one prompt swapped), 10 significant losses,
    6/10 wins kept. Steering the bridge onto the subject's tokens does not stop copies.
  - r6_nop12 (native opening K=12): 4 flags, 2 human-confirmed (10.1% human-corrected), 0.62x time, but softer and a
    different composition: MUSIQ -9.9, CLIP-IQA -0.096, ARNIQA -0.033, patch FID +9.1, PickScore -0.18 (6/10 wins kept).
    r6_nop8: 7 flags (worse), 0.80x time, 4 losses. r6_gtx: 8 flags (worse), only CLIP -0.60.
