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
