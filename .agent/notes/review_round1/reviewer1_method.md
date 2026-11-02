# Reviewer 1 (method, AC-level): SCA, round 1

Read: paper/main.pdf (53 pp), main.tex, Appendices A, F, H, J, P, Q; compared with AReA (2610.17227) and the LILIE/MFlow bar (one clean idea, causal language only for ablations that ran).

## 1. Claimed contribution and framing

Claims: (i) each FLUX head's balance between a native sliding window and a block-centre global view can be fit in closed form at 1024^2 by emulating the target image-to-window ratio, and the weight is a property of that ratio ("scale covariance"); (ii) a 4-step shrinking-window handover with 1,024 bridge tokens halves late duplication caused by a one-step global-to-native switch; (iii) at 4096^2 on FLUX.1-dev it beats AReA on 12/15 metrics at 2.63x speed.

Evidence: (i) holds for ratios 2 and 4 and resolutions 1024/1280 only. (iii) holds on FLUX.1-dev; on FLUX.1-Krea-dev, the checkpoint AReA's code defaults to and its paper reports, the margin collapses to +0.02 ImageReward and a patch-FID loss (Table 4). (ii) overreaches: the paper cuts its own 57% to 26% while AReA, also with shrinking windows, sits at 4% and full attention at 7%. Sec 2.4 explains why SCA duplicates, not why AReA does not, and Table P.1 shows a single 128-window phase reaching 1/26, which undercuts "the one-step switch" as the cause. The honest limitations paragraph does not rescue an intro bullet that reads as solved.

## 2. Strongest and weakest points (oral track)

Strongest: (a) the emulation trick is simple and cheap, and the per-head > per-layer > shared ordering predicted by output error is confirmed by image metrics; (b) paired-interval evaluation on 15 metrics, three seeds, 1000 HRGen prompts, and a documented AReA reproduction (Table H.1); (c) Table P.1 and Appendix Q document dead ends with their costs.

Weakest: (a) duplication is the second headline, yet SCA is the worst single-pass method except SigMa on it and the mechanism that makes AReA 6x better is never isolated; (b) the window sizes, sigma boundaries, gate, tile, bridge size and bin count are hand-set, and the schedule is defined in sigma for exactly 28 steps (one step per window) and never tested at another step count; (c) the calibrated operator is not the deployed one (no bridge, 16x fewer keys per branch at emulated r=4) and the gap is unmeasured.

## 3. Actionable issues

1. **Blocking.** p.2 bullet 2 "We trace repeated subjects to a one-step switch ... and halve them." Table P.1: a gradual "linear pi ramp to sigma=0.7" with native windows gives 5/26, same as bridge-only; a single "128 windows (sigma>0.8)" phase gives 1/26. Width, not gradualness, moves the count. Rewrite the bullet and Sec 2.4: native windows narrower than the forming subject duplicate it at high noise; wider windows plus bridge reduce it; the 4-step schedule is a quality trade-off.

2. **Blocking.** Sec 2.4/4.2: no analysis of why AReA (4%) does not duplicate with the same window family. Add one controlled pair: SCA with AReA's 20-step shrink and bridge ratio, and AReA with SCA's 4-step shrink.

3. **Blocking.** Sec 4.4 "Both parts of the schedule are needed." Table 3 has "bridge only" but no "schedule without bridge" row, and the 46-prompt dup counts (4 vs 5 of 26) are within one prompt with no interval. Add schedule-alone and bridge-alone on all 195 prompts with McNemar p-values; the text admits the 46 subset understated dup by 11 points.

4. **Major.** Sec 2.3 and Table F.1: the schedule is specified by sigma intervals of width 0.02 that hold exactly one Euler step at 28 steps, mu=1.15. At 42/56 steps or another shift a window gets 0 or 2 steps. State this, or define the schedule by ratio or step index, and run the 42/56-step budget rows with the schedule (Table 2 runs them without it).

5. **Major.** Sec 2.5: calibration omits the bridge and emulates r=4 with 256 global keys, 256 window keys, and 2-token tiles against 4,096/4,096/8 at the target. Softmax entropy and the text-key share differ by an order of magnitude. Report the deployed operator's output error against full attention at 4096^2 on a few prompts, and a table recalibrated with the bridge, or state the gap as untested.

6. **Major.** Term "scale covariance" (abstract, p.2, p.6). Evidence is two ratios, two resolutions at r=2, and the native phase extrapolates the r=4 table to s=6 and 8. The claim is approximate invariance of pi at fixed ratio, not covariance of an operator. Define it in one sentence with its tested domain, or rename the concept.

7. **Major.** Table J.1 marks AReA O(N^2)* while SCA is O(N). AReA uses pooled keys and fixed windows; your own fit gives exponent 1.12 vs 0.99. Justify the entry from AReA's code (bridge ratio 0.5 of all tokens?) or correct it.

8. **Major.** Sec 4.1 and p.2 bullet 3: "beats AReA on 12 of 15 metrics ... loses none". Duplication is a measured outcome where SCA loses 26% vs 4%. Add it to the bullet and place the near-parity Krea result beside the dev headline, since AReA was tuned on Krea.

9. **Minor.** Lemma 1 motivates "every head mixes a near and a far field" but SCA's sets overlap (block centres inside the window, text in both, bridge tokens double-counted). Say the lemma is motivation only; state the double-count's effect or dedupe in the mask.

10. **Minor.** Prop 1 holds only in the native phase without bridge; wide steps and bridge keys reach offset 255, and bf16 ids round beyond 256. Put the scope in the statement.

11. **Minor.** Prop 3 is the projection of a scalar quadratic; fold it into the text. The bound applies only to bin-constant tables, as the proof notes.

12. **Minor.** Fig 5d caption "(r = 0.77)" reuses the ratio symbol for a correlation. Use rho.

13. **Minor.** Sec 2.3 "The schedule's four sizes were set by hand; only the weights are calibrated." Add to Table F.1 a hand-set column (gate, 4 sizes, 4 sigma bounds, t, bridge 8x8x16, ratios, 14 bins, interpolation rules) versus calibrated (pi only). Sensitivity varies only prompt count and gate; add bin count and t.

14. **Minor.** Sec 4.1 "HRDiT's lead thus comes from its schedule." One swap with a CLIP-IQA loss supports "largely", not "thus".

15. **Minor.** Fig 1 bottom and Fig 6 are curated where baselines score "unusually low"; an oral teaser should be uncurated or paired with the uncurated gallery.

## 4. Questions

1. Why does AReA reach 4% duplication with shrinking windows while SCA reaches 26%: shrink length, halo, bridge ratio, or the per-head weights themselves?
2. How much does pi change if calibrated with the bridge in the local branch, or at 2048^2 emulating r=4 with 32-token windows?
3. Is the schedule tied to 28 steps and mu=1.15? What happens at 42 or 56 steps when a 0.02 bin holds zero or two steps?
4. A single 128-window phase gives 1/26 duplicates at a quality cost; does a two-step schedule with matched weights recover the quality? Why four steps?
5. Lemma 1 gives a per-query mass m_i and the global softmax is a cheap proxy for it. Why a per-head constant rather than a per-query weight, and does the 0.77 correlation with attention mass mean a direct mass measurement could replace calibration?
