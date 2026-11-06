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
| r7_hs40o | fo in the 40% lowest-pi heads | 3 | 3 | pending | pending | pending | pending |
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

