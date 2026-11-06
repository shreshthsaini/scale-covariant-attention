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

## Tasks (round7_tasks.py; smoke releases taskq/hold_r7; evaluations self-release from taskq/hold_dup_eval)
- a7_a000_smoke (tree 80396bd): r7_id0 (ffl near bridge, empty head set; must match sca_v2, PSNR >= 24 dB on 2 prompts) plus one
  prompt each of r7_op2, r7_fd60, r7_hs40, r7_fo, r7_fd60bw.
- Track B wave 1 (46 prompts, chunk 8, eval/r7_46_<label>; order of the spool): r7_op2 (b1), r7_fd60 (b2), r7_hs40 (b3),
  r7_fd70 (b4), r7_hs20 (b5), r7_op2bw (b6: op2 plus the hand-off bridge 0.25 log(N/R), sigma > 0.84; run in wave 1
  because the two act on disjoint steps).
- Track A: a7_t000..t029 r6_ffl on the 149 prompts it lacks (r7/ffl_rest149.jsonl, frozen tree 5be811d, same out dir
  r6/out/r6_ffl, so the 46 round-6 images are reused with identical fingerprints), then a7_a1_eval195_r6_ffl
  (eval/r7_195_r6_ffl, results/r7_195_r6_ffl). Set-level CIs: CPU bootstrap afterwards on the operator node.
- GPU: fleet SCA25 (job 1125967, gh-dev, 4 nodes, 2 h) submitted 06:16; a second gh-dev job was refused
  (QOSMaxSubmitJobPerUserLimit: held BL23 and V2P1 gh-dev jobs occupy the slots). One short job left in the allowance.

## Results
(pending)
