# Late duplication, round 7 (method owner, started 2026-11-06 06:05 CST)

Brief (lead, 2026-11-06): two tracks through the shared spool (tasks a7_, before the review-round r1_ tasks).
Track A, the fallback: promote the lattice far field r6_ffl (as defined in round 6) to the 95 people/animal prompts and
all 195 prompts at 4096^2, full metric set, detector plus human check of every flag, paired bootstraps versus sca_v2 and
AReA, time. Track B, the fix without loss, 46 prompts first: (1) far field only in long-range heads, (2) far share fading
with sigma, (3) a finer layout before the hand-off, (4) the best of these plus the free hand-off bridge; iterate once on
the strongest; promote a passing variant to 95 and 195 and stage (not enqueue) the headline rerun set.
Bar (unchanged): human-corrected people/animal duplication at or below AReA's on the same prompts, no significant paired
loss versus sca_v2 on any metric, all 10 wins over AReA kept, time at most +25%. Round 6: duplication_fix_round6.md.

## Operator options added in round 7 (commits 30b3e8f, 80396bd; frozen tree /scratch/09032/saini_2/sca/code/80396bd)
All default off; sca_v2 and r6_ffl paths are unchanged (tests/test_textgate.py, 28 tests; the full suite passes except
tests/test_review_round.py::test_generated_tasks, which already fails on the clean tree before these changes).
- --ffl-near bridge: the near part of ffl is sca_v2's own window plus bridge (layout prefix R) instead of the window
  alone, so a head or step with far weight 0 is exactly sca_v2 (identity test against sca_dedup_forward). Every Track B
  ffl variant uses it: the bridge is sca_v2's established, quality-neutral far-field component, and with it the fade and
  the head set end exactly in sca_v2 instead of in a bridge-free window.
- --ffl-head-frac F: the round(F x 1368) heads with the smallest calibrated pi at the hand-off (target-scale pi table at
  sigma = gate 0.89) take the far field at their calibrated share; the others keep sca_v2 (far weight 0, near partition
  for the text balance). F = 0.2: 274 heads, pi < 0.509; F = 0.4: 547 heads, pi < 0.674. These heads have far share
  0.68 (0.2) and 0.60 (0.4) at the 128 window vs 0.40 / 0.36 for the rest, but their local branch carries only pi of
  the output: the mean pi x far of the selected heads is 0.197 (0.2) vs 0.306 for the others, so the 20% set receives
  about 14% of the ffl dose and the 40% set about 36%.
- Head set and fade enter as one per-head far weight w = [head in set] x fade(sigma) (textgate lat_w): the image part
  is (1 - w f) near + w f far, and the lattice keys enter the window's partition (the text-to-image balance) with
  weight w (logits + log w). So w = 0 is the near part with its own partition, exactly sca_v2 for near "bridge", and
  w = 1 is round-6 ffl; a fading variant ends continuously in sca_v2.
- --ffl-fade-lo LO: fade(sigma) = clip((sigma - LO) / (0.89 - LO), 0, 1), with --text-mode-lo LO (sca_v2 below).
  Mean far share of layer 10 at LO 0.6: 0.70 (sigma 0.888), 0.54 (0.83), 0.37 (0.76), 0.23 (0.703), 0.08 (0.637).
- --ffl-outside: lattice keys inside the query's own window are left out of the far field (BlockMask over queries x
  4,096 lattice keys). Reason: the lattice samples the near field too (every 4th token inside the window), so in r6_ffl
  roughly (1 - f) of the lattice's softmax sits inside the window: the effective far weight is about f^2 and the rest
  is a sparse second copy of the near field. Outside-only gives the far field the full weight f.
- --open-pool 2 --open-pool-hi 0.93: global-only steps with 0.89 < sigma < 0.93 (steps 6 and 7, sigma 0.921 and 0.905)
  pool by 2 (a 128 x 128 global grid at 4096^2, coarse ids floor(p / 2)) instead of 4.
- --text-bw-lo S: text mode bw applies only while sigma > S (for ffl+bw: far field through --text-mode-lo, hand-off
  bridge only at sigma > 0.84). ffl+bw needs --ffl-near bridge.

## Far-field dose of each variant relative to r6_ffl (sum over steps 8 to 21 of the mean over layers and heads of
pi x weight x calibrated far share, actual window schedule; computed on CPU from the calibration)
hs20 0.15, hs40 0.38, fd60 0.45 (0.99 / 0.93 / 0.86 of ffl's at the three seeding steps), fd70 0.32, fflh (round 6,
three steps only) 0.17. Round 6 reached 1 of 26 only at dose 1 (ffl) and removed nothing at 0.17 (fflh), so the head
sets and fades are expected to sit between; r7_fo (outside-window far field at full dose) tests whether the cost of
the dose can be lowered instead.

## Tasks (round7_tasks.py; smoke releases taskq/hold_r7; evaluations self-release from taskq/hold_dup_eval)
- a7_a000_smoke (tree 80396bd): r7_id0 (ffl near bridge, empty head set; must match sca_v2, PSNR >= 24 dB on 2 prompts) plus one
  prompt each of r7_op2, r7_fd60, r7_hs40, r7_fo, r7_fd60bw.
- Track B wave 1 (46 prompts, chunk 8, eval/r7_46_<label>), spool order: r7_op2 (b1: pool 2 on steps 6 and 7),
  r7_op4 (b2: pool 2 on steps 4 to 7, AReA's own Phase I schedule 4,4,4,4,2,2,2,2 at 4096^2, area_code.md), r7_fd60
  (b3), r7_fo (b4: ffl near bridge, outside-window far field, through 0.5), r7_op2bw (b5: op2 plus the hand-off bridge
  0.25 log(N/R) at sigma > 0.84; in wave 1 because the two act on disjoint steps), r7_hs40 (b6); then Track A; then
  r7_hs20 (u1) and r7_fd70 (u2), which run only if GPU time is left (doses 0.15 and 0.32).
- Track A: a7_t000..t029 r6_ffl on the 149 prompts it lacks, the 69 people/animal prompts first (t000..t013;
  r7/ffl_rest149_pf.jsonl = rest_people69 + rest_other80), frozen tree 5be811d, same out dir
  r6/out/r6_ffl, so the 46 round-6 images are reused with identical fingerprints), then a7_a1_eval195_r6_ffl
  (eval/r7_195_r6_ffl, results/r7_195_r6_ffl). Set-level CIs: CPU bootstrap afterwards on the operator node.
- GPU: fleet SCA25 (job 1125967, gh-dev, 4 nodes, 2 h) submitted 06:16; a second gh-dev job was refused
  (QOSMaxSubmitJobPerUserLimit: held BL23 and V2P1 gh-dev jobs occupy the slots). One short job left in the allowance.

## Results
- 08:31 fleet SCA25 started (c642-[002,082,091-092], until 10:31). 08:48 smoke PASS (tree 80396bd): ffl near bridge
  with an empty head set reproduces sca_v2 at 34.36 / 32.02 dB on the two identity prompts (round 6's split: 34.4 /
  32.0 dB); one prompt each of r7_op2, r7_fd60, r7_hs40, r7_fo and r7_fd60bw ran (first-image times with compilation
  100.5, 95.1, 97.3, 145.8, 104.9 s; the identity's second image, with the lattice computed at weight 0, 85.4 s). Visual
  check (r7/smoke/sheet.jpg): all intact; op2 and fo change the composition, fd60, hs40 and fd60bw stay close to
  sca_v2. 48 generation tasks released.
- r7_op2 (pool 2 on steps 6 and 7; eval 09:20): 4 of 26 people/animal flags, the same count as sca_v2 (fixes the
  canoe prompt 979216, adds 1516768667427572324, keeps 273390, 345747, 684465); no significant loss vs sca_v2; wins over
  AReA kept 8/10 (aesthetic and CLIP-IQA no longer significant); x1.025 time (78.9 vs 76.9 s). A finer layout on the
  last two opening steps does not remove the seeds. Consistent with the round-6 trace of the bear-on-rocks copy
  (results/round6_mechanism/trace_sheets/r6_4820545723762191815.jpg): AReA's x0 at step 8 shows the same thin spike that
  becomes sca_v2's copy; it dissolves over steps 9 and 10 under AReA's wide Phase II attention (180-token windows plus a
  bridge of round(0.5 w)^2 = 8,100 tokens, about 62% of the keys), not under its opening.
- r7_op4 (pool 2 on steps 4 to 7, AReA's own Phase I schedule; eval 09:35): 6 of 26 flags, worse than sca_v2 (adds
  5968998903196670637 and minus2679953061215100202, fixes none); 3 significant losses vs sca_v2 (aesthetic -0.088,
  MUSIQ crop -2.3, PickScore -0.12), ARNIQA gain; 9/10 wins kept; x1.049. Item (3) is refuted on the 46 prompts: a finer
  opening layout leaves the seeds (op2) or adds them (op4).
- Human check of the op2 and op4 flags (packet r7/label/p1, 3 blind Opus labelers, human_labels_r7.csv): op2 3 yes,
  1 borderline (572324: a framed picture that may show a face); op4 4 yes, 1 no (670637), 1 borderline (minus...100202:
  one dancer with merged limbs). Human-corrected: op2 14.3%, op4 17.9% (sca_v2 17.0%, AReA 2.2%).
- r7_fd60 (fade to 0 at 0.6, near bridge; eval 09:54): 3 of 26 flags (fixes 979216 and 684465, adds
  minus2679953061215100202); 9 significant losses (PickScore -0.18, IR -0.085, CLIP -0.45, HPSv2 -0.0034, HPSv2.1
  -0.0067, aesthetic -0.078, ARNIQA -0.014, CLIP-IQA -0.019, MUSIQ crop -1.16), also on the 26 prompts clean in both
  (IR, PickScore, HPS, aesthetic, ARNIQA, MUSIQ crop); 7/10 wins kept; x1.086. PickScore falls on 37 of 46 prompts.
- Finding (visual, results/round7_mechanism/): r6_ffl and fd60 images carry a periodic grid with the lattice's
  4-token (64 px) period: a halftone dot texture on fabric and backgrounds (fd60_vs_sca_v2_800687.jpg) and a diamond
  lattice in the stained glass (lattice_aliasing_207187.jpg). The far field of ffl includes the lattice keys inside the
  query's own window; a query near a lattice key attends to it more than a query between two, so the output is
  modulated with the lattice phase. r7_fo, whose far field excludes the in-window lattice keys, shows no grid on the
  same prompt. The quality cost measured for ffl in round 6 (9 losses) and for fd60 is then mostly this aliasing, not
  the far field itself.
- r7_fo (ffl near bridge, far field = lattice keys outside the query's window at true positions, through 0.5; eval
  10:17): 2 of 26 flags (fixes sca_v2's canoe 979216, kimono scene 345747 and three-men 684465; keeps the church
  273390, adds the dancer minus2679953061215100202, a tangle of fused bodies); but 10 significant losses, larger than
  r6_ffl's: PickScore -0.61, IR -0.17, CLIP -0.84, HPSv2 -0.0096, HPSv2.1 -0.020, aesthetic -0.14, ARNIQA -0.027,
  CLIP-IQA -0.049, MUSIQ -3.3, MUSIQ crop -4.0 (on the 25 clean prompts too); 3/10 wins kept; x1.13 time. Visually
  (results/round7_mechanism/ffl_vs_fo_800687.jpg) the grid is gone but surfaces are flattened, detail is lost and
  bright halos outline shapes. Reading: with true positions the far keys sit at query-key offsets up to 255 tokens,
  four times the native range (FLUX sees at most 63 at 1024^2); r6_ffl's lattice was dominated by its in-window keys
  (in range, but phase-aliased), fo's is entirely out of range. Both costs come from reading the far field at true
  positions. Iteration (queued, tree 764f337): r7_fg = fo with the far field read at the global branch's coarse ids
  (--ffl-coarse: queries at floor(p / 4), block keys at block ids, offsets within 63), i.e. the far field at the
  native resolution, where full attention's calibrated far share was measured.
- Parked (taskq/hold_r7_parked): r7_op2bw (the finer opening it combined is refuted), and the full-lattice r7_hs40,
  r7_hs20, r7_fd70 (they carry the aliasing); queued instead r7_hs40o and r7_hs20o (head sets with the outside far
  field; hs20o after Track A).
- Human check of the fd60 and fo flags (packet r7/label/p2, 2 labelers): fd60 3 yes (345747, 273390, the dancer);
  fo 1 yes (the dancer: a fused pile of about three bodies in mid-air plus a figure standing in the splash) and the
  church borderline (a cut-off figure at the doorway). Human-corrected on the 26: fd60 14.5%, fo 7.0% (AReA 2.2%).
- 10:33 SCA25 ended (r7_hs40o generation complete, its evaluation pending); fleet SCA26 submitted (job 1126265,
  gh-dev 4 nodes 2 h; balance 9,957 SU), the second and last short job of the allowance.
- Correction to the aliasing finding (10:45): the pattern is a token-period checkerboard (16 px, one latent token;
  results/round7_mechanism/zoom_sca_v2_vs_ffl_800687.png), not a 4-token lattice grid. Spectral index
  (scripts/r7_grid_index.py: log10 power at the 16 px period over its neighbourhood, central 2048 px; per image in
  token_checkerboard_index_46.csv): median over the 46 images sca_v2 1.67, r6_ffl 1.85, fd60 1.81, fo 1.59, hs40o 1.70,
  op2 1.67, AReA 2.00; on the illustration 800687 sca_v2 1.79 vs r6_ffl 3.58. So the checkerboard is strong on flat
  illustrated surfaces (anime fabric, stained glass) but not prevalent, and fo, without it, loses more: the measured
  cost of the far field is the mixing of a true-position far field at the calibrated share, with the checkerboard as
  a visible local symptom of its sparse, phase-sensitive keys. fg (coarse ids, offsets within 63, shared position per
  4 x 4 block) tests the in-range far field.
- r7_hs40o (40% head set, outside far field, true positions; evaluated on CPU on the operator node because no GPU was
  free, detector included): 3 of 26 flags (metrics pending).

## Frontier on the 46 prompts so far (26 people/animal prompts; sca_v2 4 flags 4 confirmed 17.0%, AReA 1 false flag 2.2%)
| variant | what | flags | confirmed | human-corrected | losses vs sca_v2 | wins over AReA | time |
|---|---|---|---|---|---|---|---|
| r6_ffl (round 6) | true-position lattice, window near, through 0.5 | 1 | 1 | 6.6% | 9 | 7/10 | x1.08 |
| r7_op2 | pool 2 on steps 6-7 | 4 | 3 | 14.3% | 0 | 8/10 | x1.025 |
| r7_op4 | pool 2 on steps 4-7 (AReA's Phase I) | 6 | 4 | 17.9% | 3 | 9/10 | x1.05 |
| r7_fd60 | lattice, bridge near, fade to 0 at 0.6 | 3 | 3 | 14.5% | 9 | 7/10 | x1.09 |
| r7_fo | outside-window lattice, true positions, through 0.5 | 2 | 1 (+1 borderline) | 6.9% | 10 | 3/10 | x1.13 |
| r7_hs40o | fo in the 40% lowest-pi heads | 3 | 3 | 14.5% | 8 | 7/10 | x1.13 |
(round7_frontier_46.csv holds the exact intervals.)
- Track A early read (11:05, CPU detector, r7/cpu_det/r6_ffl; 61 of the 95 people/animal prompts generated so far:
  the 26 of the subset plus 35 more): r6_ffl 8 flags, all a subset of sca_v2's 18 on the same prompts; AReA 4. On the
  35 prompts outside the subset ffl keeps 7 of sca_v2's 14 copies. The 46-prompt subset understates duplication
  (sca_v2 4 of 26 there, 21 of 69 elsewhere), so ffl's 1 of 26 there becomes roughly half of sca_v2's rate on the 95,
  about three times AReA's. The official read comes from the GPU evaluation (eval/r7_195_r6_ffl).
- Because the subset has only 4 sca_v2 copies, the iteration r7_fg also runs on the 21 people/animal prompts outside
  the subset where sca_v2 has a detector-flagged copy (r7/copyprone21.jsonl; a7_b800..b802, same out dir and tree, no
  metric evaluation): a paired count of copies removed, with r6_ffl's images from Track A as the reference. r7_hs20o
  parked (dose 0.15, lowest value); r7_op2bw re-queued after Track A as item (4) (op2 is the only no-loss point among
  items 1 to 3). SCA26 budget: fg 46 and 21, Track A 23 tasks plus the 195 evaluation, then op2bw.
- r7_hs40o (eval 11:06, CPU): 3 of 26 flags, 3 confirmed (the canoe, the church, the kimono scene: none of sca_v2's
  copies removed except 684465); 8 losses (PickScore -0.12, IR -0.044, CLIP -0.40, HPSv2 -0.0021, HPSv2.1 -0.0046,
  aesthetic -0.064, CLIP-IQA -0.012, MUSIQ crop -0.65); 7/10 wins; x1.13. Caveat: its extra metrics were computed on
  CPU while sca_v2's and AReA's came from the seeded GPU cache; CLIP, PickScore and MUSIQ of all three were recomputed
  on CPU, and those losses alone fail the bar.
- Track A human check (in progress): the 7 early r6_ffl flags outside round 6's labels are all confirmed (packet pA1;
  with 273390 from round 6, 8 of 8 flags confirmed on the 61 prompts so far). Miss rate by a stratified unflagged
  sample (proportional, 30 in all, seed 20261106): stratum A = the 61 prompts generated by 11:00 (53 unflagged; 3 of
  them labeled in round 6 are kept as known labels and left out of the pool; 19 sampled, packet pA2); stratum B = the
  34 people prompts generated later (11 to sample). Estimate: (confirmed flags + known unflagged dups + sum over strata
  of unlabeled unflagged x k_s / n_s) / 95, with the official flags from the GPU detector of eval/r7_195_r6_ffl.

## Finding (11:35): the detector misses most of the far field's copies; round 6's "ffl 1 of 26" was a detector artifact
Stratum-A unflagged sample of r6_ffl (19 blind labels, packet pA2): 5 yes, 1 borderline, 13 no (miss rate 26%,
Wilson [12, 49]; sca_v2 and AReA had 1 of 30 each in Step 1). Of the 10 prompts where the detector says ffl removed
sca_v2's copy (sca_v2 flagged, ffl not; 61 prompts so far), 6 were sampled: 4 still have a copy (979216 canoe,
918576, 345747 kimono scene, 660958 campers), 2 are truly clean (341041, 591135). The copies did not go away; they
moved: in the canoe image (results/round7_mechanism/ffl_canoe_detector_miss_979216.jpg) the detector counts 2
people where the canoe holds 6 (a blue-capped rower and two children packed between the two paddlers, a head behind
one of them); occluded figures overlapping the main subject fall to the detector's NMS and containment rules. With
far-field context the extra figures cluster around the main subject instead of appearing elsewhere as copies.
Rough human-corrected estimate on the 61 prompts: 8 confirmed flags + 10 x 4/6 + 43 x 1/13 = about 18 (29%), against
sca_v2's 18 flags + 43 x 1/30 = about 19.4 (32%) on the same prompts. So r6_ffl does not materially reduce human-
counted duplication. On the 46-subset at least 3 of 26 r6_ffl images are human-confirmed duplicates (273390, 979216,
345747), not 1. Consequences: (a) Track A will report a human-corrected rate with r6_ffl's own miss rate (stratified
sample), not the detector rate; (b) detector counts of far-field variants (fo, fd60, hs40o, fg) understate their
duplication; a candidate needs all its people/animal images labeled before any claim; (c) the round-6 frontier's
"lowest duplication" point does not hold.
- Spool reordered (11:30): Track A's people tasks (renamed a7_a9_t007..t013; their markers keep the original names, so
  the evaluation's gate is unchanged) run first on SCA26, then r7_fg (46 and 21), then Track A's other 80 prompts
  and its 195 evaluation, then op2bw. The 95-prompt human-corrected rate is the decisive Track A number.
- Packet p4 (7 images): every image where a wave-1 variant's detector says it removed one of sca_v2's 4 copies on the
  subset (op2 979216; fd60 979216, 684465; fo 979216, 345747, 684465; hs40o 684465), to test whether any variant
  removes copies rather than hiding them from the detector.
- Packet p4 result (11:36): of the 7 images where a variant's detector says it removed one of sca_v2's copies, 4 are
  still duplicates: the canoe 979216 keeps extra rowers under op2, fd60 and fo (7 real people under fd60), 684465
  keeps an extra man under fd60; truly removed: 684465 under fo and hs40o; 345747 under fo is borderline. Recomputed
  human-corrected rates on the 26 (unlabeled unflagged images still at sca_v2's 1/30 miss rate, so lower bounds for
  far-field variants, whose sampled miss rate is 26%): op2 18.1%, op4 17.9%, fd60 21.9%, fo 10.4%, hs40o 14.3%,
  r6_ffl 13.6% (was 6.6%); sca_v2 17.0%, AReA 2.2%. fo is the only variant clearly below sca_v2, and it carries 10
  quality losses.
- Reading for the next round (hypothesis, not tested here): copies need two things, the window's prompt over-read
  (about 2x full attention's text mass at copy sites, round 6 probe) and its context deficit (it sees the main subject
  at 0.4 to 0.5x full attention's mass). Round 6's ctw removed the over-read alone and left distant copies; the far
  field removes the deficit alone and the extra figures move next to the main subject (a crowd the detector merges).
  AReA avoids both at once: its first Phase II steps attend about 62% of all keys, so the text competes with nearly
  every image key and the subject is in view. A fix within SCA's cost would need both parts together, e.g. a far field
  read at coarse ids (fg) whose partition also restores full attention's text-to-image balance.
- fo unflagged sample (packet p5, 10 of its 21 unlabeled unflagged people images): 0 yes, 1 borderline (a fused second
  head on the owl). fo's human count on the 26: 2 yes (the dancer, the canoe's extra rowers), 3 borderline (church,
  kimono scene, owl), 11 unflagged unlabeled at a sampled miss rate of 0 of 10: about 2 of 26 (7.7%). So the
  outside-window far field does remove copies (unlike r6_ffl, whose copies mostly moved next to the subject), at the
  cost of 10 significant losses.

## Draft conclusion (13:35, before fg, Track A's last 34 people prompts and its 195 evaluation)
1. No variant meets the bar. The finer opening (op2, op4) leaves or adds seeds. Every far field read at true
   positions costs quality on clean prompts (PickScore -0.12 to -0.61), and the one that genuinely removes copies (fo,
   about 2 of 26 by hand) costs the most (10 losses, 3/10 wins over AReA kept).
2. The duplication detector undercounts far-field variants: ffl's extra figures crowd around the main subject and are
   merged by NMS (unflagged miss rate 5 of 19 vs 1 of 30 for sca_v2 and AReA). Round 6's "ffl 1 of 26" and the
   "duplication-free mode" premise of Track A do not hold; on the 61 people prompts so far r6_ffl's human-corrected
   rate (34.7% [22.8, 53.1]) is not below sca_v2's. Any future candidate must be judged by labels of every
   person/animal image, not by detector flags.
3. Reading: the copies need both the window's prompt over-read and its context deficit; AReA removes both by
   attending about 62% of the keys in its first Phase II steps (2.6x SCA's time).
Paper paragraph (limitations, proposed): "Late duplication is reduced but not solved. On the 95 person and animal
prompts, human-corrected duplication is 26.7% for SCA and 4.2% for AReA, whose first refinement steps attend most of
the image at 2.6 times SCA's cost. Giving SCA's windows a calibrated far field moves most extra figures next to the
main subject rather than removing them, and the one far field that does remove them costs ten of fifteen metrics. An
automatic detector undercounts such crowded figures, so we report rates corrected by blind human labels."

## Lead directive 13:40 (urgent): put AReA's wide shrinking windows into SCA
"AReA keeps windows wider than the subject's scale for most of the trajectory (Phase II windows shrink 180 to 68 over
steps 8 to 26, with its outlier bridge), while SCA drops to 64 within four steps." Run review item 2's sca_area20 first,
on the 46 prompts and copyprone21; add AReA's schedule truncated to its first 8 and 12 window steps with matched
weights; hand-check every image of all three. Relaxed bar: human-corrected repeats at or below AReA's; no significant
loss vs sca_v2 on IR, HPSv2.1, PickScore; every other metric within its interval or better than AReA; time below
AReA's 202 s. A pass on 46 is promoted to 95 and 195 at once (staged like hold_r7_promote) with an operator paragraph.
Report as soon as sca_area20's 46-prompt result is in (under 200 words).
Queued at the head of the spool (prefix a0_, tree f479459 = the review's, so sca_area20 images serve both):
- a0_r2_g_sca_area20_* (the review's 9 generation tasks, renamed; each also releases a0_e_eval_sca_area20 once all 46
  images exist; the review's pair evaluation still waits for area_sca4), out/review1/sca_area20.
  Operator: --win-sched 0.87858:180,...,0.15009:68 (AReA's 19 Phase II windows on steps 8 to 26, 64 on step 27)
  --win-pi-match --win-pi-interp --bridge-cells 8 --bridge-ratio 0.5 (bridge round(0.5 w)^2 tokens).
- r7_area8 (a0_s8*, eval a0_e_eval_r7_area8): AReA's first 8 windows 180, 170, 161, 153, 145, 137, 130, 123 on steps
  8 to 15, then 64; r7_area12 (a0_s9*): first 12 (to 99 on step 19), then 64; same weights and bridge rule.
- copyprone21 runs of all three: a0_t1* (sca_area20), a0_t2* (area8), a0_t3* (area12).
- scripts/r7_decide.py prints the relaxed-bar verdict next to the original bar (columns relaxed_pass, relaxed_fail).

## sca_area20 result on the 46 prompts (17:15; SCA26 since 16:28; eval/r7_46_sca_area20)
- Every one of its 26 person/animal images labeled blind (packets pS1 to pS3): 3 duplicates (345747 kimono scene: two
  pedestrians and three extra bears; 979216 canoe: four people in the canoe; minus2679953061215100202 dancer: a fused
  second dancer), 1 borderline (394876, a figure fragment in a mirror), 22 clean: 11.5%. The detector flagged 1.
- AReA's 17 previously unlabeled person/animal images on the same prompts labeled blind too (packet pR1): 1 duplicate
  (979216 canoe, an extra rower in a second boat), 3 borderline: AReA's fully hand-counted rate is 3.8% (was 2.2%
  estimated). sca_v2 17.0%.
- 8 significant losses vs sca_v2: PickScore -0.19 [-0.27, -0.11], IR -0.060 [-0.101, -0.015], HPSv2.1 -0.0079
  [-0.0104, -0.0055], HPSv2 -0.0041, CLIP -0.48, aesthetic -0.057, ARNIQA -0.0056, MUSIQ crop -1.67; on the 27 prompts
  clean in both: IR -0.066, PickScore -0.12, HPSv2.1 -0.0063, HPSv2 -0.0033, MUSIQ crop -1.28 (softening and weaker
  prompt adherence, not only on prompts whose copy was removed). Wins over AReA kept 7/10 (aesthetic, ARNIQA, CLIP-IQA
  lost). Time 103.4 s median (x1.34 sca_v2, AReA 219.6 s on these prompts).
- Relaxed bar: FAIL (rate 11.5% vs AReA 3.8%; core losses on IR, HPSv2.1 and PickScore). Not promoted; the staged
  195 tasks stay in taskq/hold_r7_promote_area20.
- Still running on SCA26 (until 18:28): r7_area8, r7_area12 (46), then the copyprone21 runs of all three; their
  evaluations self-release (a0_e_eval_r7_area8/12); label every person/animal image with r7/label_people.sh.
- r7_area8 (eval 17:29; detector only, its 26 person/animal images not yet hand-labeled): 1 flag; 9 significant losses
  vs sca_v2 incl. PickScore -0.13, IR -0.056, HPSv2.1 -0.0067 (core three again); 7/10 wins over AReA; 96.1 s (x1.25).
  Relaxed bar fails on the core losses whatever its hand count. r7_area12 and the copyprone21 runs still on SCA26.

## Morph ladder (user directive via lead, 19:30): localize the cause by single-component swaps
Readout: blind hand labels of every image on copyprone21 (r7/copyprone21.jsonl, the 21 person/animal prompts outside the
subset where sca_v2 has a detector-flagged copy); metrics on the 46 only for the decisive swaps. Baselines, every image
hand-labeled: sca_v2 19 yes of 21 (Step 1 labels), AReA 4 of 21 (9 Step-1 labels plus 12 new, packet pL1), sca_area20
12 of 21 (pL1). So AReA's Phase II windows and bridge inside SCA close about half of the gap; the rest lies elsewhere.
Rungs (code 71b31b6 = sca/morph.py, flags in scripts/sample.py and scripts/run_area.py; tests/test_morph.py):
| rung | from | swap | how |
|---|---|---|---|
| sca+A | sca_v2 | AReA's opening pool 4,4,4,4,2,2,2,2 | --open-pool 2 --open-pool-hi 0.96 (label r7_op4) |
| sca+B | sca_v2 | AReA's Phase II tiling: jittered base tiles of SCA's window side, DWT splits, jagged Halo 8, raised-cosine merge, per-window bridge of SCA's 1024 tokens | --morph-local area_tiles (AReA's windowed_attention) |
| sca+C | sca_v2 | text queries = mean over an even window partition (text, window, bridge keys) | --morph-text window_mean |
| sca+D | sca_v2 | no global branch after the opening | --pi-const 1.0 |
| sca+E | sca_v2 | bridge round(0.5 w)^2 | --bridge-ratio 0.5 |
| area+open | AReA | SCA's opening: factor 4 on all 8 steps | run_area.py --phase1-factors 4,4,4,4,4,4,4,4 |
| area+mix | AReA | SCA's global branch (pool 4, block centre, coarse ids) mixed into the image output with SCA's calibrated per-head pi at the window's emulated scale | --sca-pi-mix calib/k1_1024.npz |
| area+text | AReA | text queries read SCA's global branch | --sca-text-rule |
| area+fixed | AReA | no jitter, no Halo, no DWT splits (fixed partition; the closest AReA analogue of SCA's deterministic windows) | --fixed-tiles --halo 0 |
- Copy-prone counts so far (every image hand-labeled; yes of 21): sca_v2 19, AReA 4, sca_area20 (AReA's 19 windows plus
  bridge in SCA) 12, r7_area12 (first 12 windows) 12, r7_area8 (first 8) 16. Wide windows plus AReA's bridge for 12
  steps remove 7 of sca_v2's 19; widening further adds nothing; the remaining 8 above AReA come from another component.

## Fix 1 and Fix 1+3 from the root-cause audit (lead 21:00; sca/qmix.py, commit 606e2ae)
Exact definitions (tests/test_qmix.py checks both against brute force in float64):
- f1: image output w_i O_loc + (1 - w_i) O_glob (O_loc, O_glob = sca_v2's branches), w_i = sigmoid(logit pi_h + tau (d_i
  - mean_i d_i)), d_i = log Z_near,i - log Z_far,i; Z_near = bridge plus window keys at true ids (the window softmax's
  image partition, text excluded); Z_far = s^2 (exp(lse_glob,i) - Z_text,glob,i - Z_in,i), lse_glob the global branch's
  partition (coarse ids), Z_text,glob its text part, Z_in its block centres inside the query's window, s^2 = 16. tau 1.
- f13: O = (Z_T O_T + e^beta Z_near O_near + Z_far O_far) / (Z_T + e^beta Z_near + Z_far), T = text keys at the query's
  true ids, far = block centres outside the window at coarse ids with log s^2 added, beta_h = logit pi_h - mean_i
  (log Z_near - log Z_far). Text queries keep the global branch's output in both.
- Variants: r7_q1s / r7_q13s at the seeding steps only (sigma > 0.845: 0.888, 0.870, 0.850), r7_q1a / r7_q13a at every
  window step; copyprone21 and the 46.


## Fix 2, cheap form (21:50; commit 250cf0c)
- --pi-blend LAMBDA --pi-blend-lo SIGMA: pi' = (1 - LAMBDA) pi + LAMBDA m_h while sigma > SIGMA (default 0.8, the
  hand-off steps 8 to 12), m_h = full attention's image-key share inside the step's window: calibration mass_mean at
  the emulated radius 0.5 n0 w / max(grid), interpolated over populated bins (SCAContext.near_share_now). It acts
  through pi_now, so it feeds f1's logit pi_h as well.
- Real table, window schedule of sca_v2: sigma 0.888 (w 128) pi mean 0.654 vs m 0.544; 0.870 (112) 0.665 vs 0.506;
  0.850 (96) 0.677 vs 0.470; 0.830 (80) 0.690 vs 0.435; 0.808 (64) 0.704 vs 0.395. So LAMBDA = 1 lowers the window's
  weight by 0.11 to 0.31 on the five hand-off steps and leaves step 13 on (0.707).
- Queued (copy-prone screen, after the f1/f13 copy-prone runs): r7_q2 = Fix 2 alone (LAMBDA 1), r7_q12 = f1 on every
  window step plus Fix 2. 46-prompt runs parked in taskq/hold_r7_fix2 until a screen warrants them.

## Fix screens on copyprone21 (11-07, SCA24 x2)
- r7_q1s (f1 at sigma > 0.845): blind labels pQ1, 20 yes of 21 (only 11799097571596124 clean); sca_v2 19, AReA 4.
  Per-query redistribution with the head mean pinned at the calibrated pi (about 0.7) over the three seeding steps
  does not touch the copies (passers-by, figures on far ridges, a second bear on a cliff, extra astronauts and divers).
  Cost 78.5 s per image warm (+2% over sca_v2's 77.1 s on these prompts). 46-prompt tasks parked
  (taskq/hold_r7_screen_fail).
- r7_q1a (f1 on every post-gate step): pQ2, 20 yes of 21; 84.6 s per image (+10% over sca_v2). Fails; 46-prompt
  tasks parked. Reading: redistributing the window weight across queries while each head's mean stays at the
  calibrated pi (about 0.7) does not reach the copies; the audit's other measurement (the image wants 0.2 to 0.4 on
  the window's output at the hand-off) points at the head mean, which Fix 2 lowers.
- Lead 03:30: Fix 2 screens first (r7_q2, r7_q12), plus r7_q12x = f1 plus pi = near share through sigma 0.75.
- r7_q2 (Fix 2 alone, pi = near share on steps 8 to 12): pQ3, 17 yes, 1 borderline, 3 no; paired vs sca_v2 2 copies
  removed (788875264758341041, 8995582008874513707), none added (exact McNemar p = 0.5). 77.3 s. Lowering the head
  mean by 0.11 to 0.31 at the hand-off barely moves the copies: neither the per-query distribution (f1) nor the head
  mean (Fix 2) of the window weight is the main lever on copyprone21.
- r7_q12 (f1 on every step plus Fix 2): pQ4, 20 yes, 1 borderline of 21; 84.5 s. Fails. Weight changes on the
  window branch, per query or per head, do not reach the copies (round 5's txtmid and round 6's ctw, which lowered
  the window's text weight, also moved copies little), while wider windows (sca_area20) halved them.
- r7_q12x (lead: f1 on every step plus pi = near share through sigma 0.75, steps 8 to 14): pQ5, 19 yes, 2 no. Fails.
  Summary of the weight fixes on copyprone21 (yes of 21): q1s 20, q1a 20, q2 17, q12 20, q12x 19; sca_v2 19,
  sca_area20 12, AReA 4. Only window geometry (AReA's wide windows held to about step 19) has moved the count.
- r7_q13a (f13 on every step): pQ6, 3 yes, 4 borderline, 14 no: the first variant below AReA (4). But glitches on
  every image (colour-block bands one to two tokens wide along the left, right and top borders; scattered blocks in
  dark regions) and missing prompted content (4 of 6 men, no birds, no spacecraft); 91.8 s (+19%). f13 is the only
  variant that lowers the text share of the global branch: in one softmax the prompt competes with the near keys and
  the s^2-weighted far lattice (about 65K effective keys), so its share falls to about full attention's. Next:
  --qmix-gamma (8ace484) scales the image part against the text with the near/far balance fixed; gamma 0.25 puts the
  text share near sca_v2's mix. Screens queued: q13m (f13 while sigma > 0.55), q13g (gamma 0.25, every step), q13s.
- GPU check (scripts/r7_qmix_gpu_check.py, 05:30, c609-042): compiled f1/f13 = eager (5e-4); all parts = float32 brute
  force at border and interior queries. The glitches are the operator's.
- r7_q13m (f13 while sigma > 0.55): pQ7, 3 yes, 1 borderline, 17 no; glitches as in q13a, so they are not a late-step
  effect. r7_q13g (gamma 0.25) glitches as well in its first images, so not the text share either. f13's per-query
  near/far weights are exactly f1's (sigmoid(beta + lse_n - log_far), same centring); the parts differ: f13's far part
  excludes the in-window centres, so a query whose far weight dominates gets an update that ignores its neighbourhood.
  Test: --qmix-far all (92c8dff), screens r7_q13w (gamma 1) and r7_q13wg (gamma 0.25), a 2 x 2 with q13a and q13g.
- r7_q13g (gamma 0.25): pQ8, 8 yes (img013 adjudicated yes: a second real bear and no reflection, so the bound of 2
  reserved for a reflection is exceeded), 1 borderline, 12 no; glitches persist. r7_q13w (far part over every centre):
  first images glitch too. So the glitches are neither the text share nor the in-window exclusion.
- Hypothesis: f13's head offset beta = logit pi_h - mean_i(lse_near - log_far) sits on the near logits, so near keys
  outweigh the text keys by e^beta (beta up to about 4 when the far mass dominates); tokens that sink their mass into
  the prompt in the window softmax (flat background, borders, where the blocks appear) lose the sink. --qmix-anchor near
  (1b8f855) moves the offset to the far part: logits (lse_T, lse_near, log_far - beta); near/far balance unchanged,
  text/near balance as in the window softmax. Also new: --qmix-dose (ddda216), dose * f13 + (1 - dose) * sca_v2.
- Label rule (all packets): a second real animal of the prompted kind counts as an extra even when the note's bound
  reserves a slot for a reflection that is not drawn.
- r7_q13n (near-anchored f13): pQ10, 9 yes, 2 borderline, 10 no; colour blocks gone (sink restored) but fine periodic
  vertical stripes on dark ground, sky and water and dash-stroke textures (tent) on most images. f13 family on
  copyprone21: q13a 3, q13m 3, q13w 2 (all colour blocks), q13g 8 (blocks), q13n 9 (stripes). The prompt's attention
  mass is one knob for both: diluted, copies fall and the sink is lost (blocks); restored, copies return.
- New: --text-value A (a70efc6). Image queries read the text values scaled by A in both branches; their attention
  weights are unchanged, so the prompt keeps its sink mass while its content drive is attenuated; text queries are
  untouched. Global branch: O' = O - (1 - A) t O_T, exact by linearity (tests/test_qmix.py). Screens: r7_tv0m (A = 0
  while sigma > 0.55), r7_tv5 (A = 0.5 every post-gate step).
- r7_tv0m (A = 0 while sigma > 0.55; 17 of 21 labeled): 0 copies, but the subjects lose their prompted look (astronauts
  without suit, the diver a clothed man, a horse-human hiker), details go missing, and two images get the right-edge
  colour column seen with round 6's ctw. Copies are driven by the prompt content that image tokens read on steps 8 to
  19, and that content also draws the main subject. Next: per-token scaling of the subject-word tokens only (9ef4ee4,
  r7/subject_words_r7.json), a narrower sigma range (> 0.75) and a partial scale (0.25).
- r7_tv0m final: 1 yes, 1 borderline, 19 no; r7_tv5 (0.5 every step): 11 yes, 1 borderline, 9 no with adherence losses.
  Uniform attenuation has no middle ground: copies fall only when the subject breaks.
- Probe (r6/probe/sca_v2, 6 copies; subject-word mass by region and step): at step 6 (end of the opening) copy sites read
  the subject words at background level (global 0.0018, background 0.0016, main subject 0.0085; window 0.0005 vs
  0.0012 at the main subject); they overtake the main subject at steps 11 to 14 (global 0.0089 vs 0.0065). So the
  opening's subject-word map separates the main subject from the future copy sites.
- sca/subjgate.py (7270551): --subject-anchor A. Opening: per image token the subject-word share of the global branch
  (head mean, summed over blocks) of the last opening step. Map: quantile-normalized (0.5, 0.95), dilated 2 tokens.
  Window steps while sigma > lo: query i reads the subject-word values scaled by A + (1 - A) map_i in both branches,
  O_i' = O_i - (1 - s_i) w_i Os_i (exact). Screens: r7_sa0m (A 0, sigma > 0.55), r7_sa0a (A 0, every window step).
- r7_sa0m (subject anchor, subject words, A 0 while sigma > 0.55): pQ13, 19 yes, 2 no, subjects intact. The anchor map
  leaks: speckled raw map (single background tokens with high subject attention), and the quantile ramp plus a 5 x 5
  max dilation leave g > 0.1 on half the image. Fix (be13ac6): 9 x 9 average first, ramp relative to the maximum
  (med + 0.25 .. 0.5 of max - med), dilation 3. Screen r7_saAs gates every text token with that map.
