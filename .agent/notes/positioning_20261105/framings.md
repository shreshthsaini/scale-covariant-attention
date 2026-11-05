# Three candidate framings for SCA (positioning agent, 2026-11-05)

Rules for every draft below. Numbers come from the paper or results/ only; the one exception is marked NEW and
needs archiving first (recommendation.md, step 0). Duplication is the human-validated rate on the 95 person or
animal prompts at 4096^2: SCA 26.7%, AReA 4.2% (results/dup_audit_20261027/human_rates_95.txt). It is a slot
[DUP]: replace both numbers if the round-6 operator lands. Krea parity sits beside the FLUX.1-dev headline, and
no draft says "better than AReA" in general (meta_review.md, decision 1).

Shared facts each framing can use (all in the paper): two views per head, an exact native-size window (64 tokens)
and a 64 x 64 grid of block-centre keys over the whole image; mix pi per head, noise level and image-to-window
ratio, fitted at 1024^2 with both views shrunk to the target's ratio; closed form clip(B/A, 0, 1); opening with
the global view for 8 of 28 steps; windows of 128, 112, 96 and 80 tokens on the next four steps; 1,024 bridge
tokens. Fit quality: split-half 0.989; ratio 2 explains ratio 4 with 2.4% more error; 1024^2 against 1280^2:
0.0589 against 0.0585; per-head 0.200 < per-layer 0.223 < shared 0.230 < best hard switch 0.261 at ratio 4.
Results: 12 of 15 metrics against AReA, none lost, 77.1 against 202.4 s; Krea on par in a third of the time.

---

## Framing A: Transfer by emulation ("fit where full attention is still right")

**Thesis.** How much each view of a training-free high-resolution attention should count can be fitted at the
training resolution, where the model's own full attention is still correct, by shrinking the views to the target's
image-to-window ratio; SCA fits each head's mix this way, in closed form and once for every target size, where AReA
sets the same quantity by hand. (The brief's broader version, "any free parameter", fails risk 1 below.)

**Title.** Scale-Covariant Attention: Fitting High-Resolution Attention at the Training Resolution

**Abstract (157 words).**
Diffusion transformers trained near one megapixel fail at 4096^2, where attention meets token counts and
offsets it never saw. Training-free fixes reshape attention by hand or per
target size, since full attention at the target is no reference. We introduce Scale-Covariant Attention (SCA), which finds that
reference at the training resolution. Each head mixes a native-size window with a native-size grid of the whole
image. Shrinking both views at 1024^2 to the target's image-to-window ratio lets the model's own full attention fit
each head's mix in closed form, once for every size. The fit repeats on disjoint prompts (correlation 0.99), and the
ratio-2 table explains ratio 4 with 2.4% more error. On FLUX.1-dev at 4096^2, SCA beats AReA on 12 of 15 metrics in
77.1 against 202.4 s, and matches it on FLUX.1-Krea-dev in a third of the time. It repeats a person or animal in
26.7% of prompts naming one, against 4.2%.

[DUP] slot: the last sentence's 26.7% and 4.2% (replace both if the round-6 operator lands).

**Contributions.**
- We fit how much each view of a training-free high-resolution attention counts, per head, at the training
  resolution, by emulating the target's image-to-window ratio against the model's own full attention, in one pass
  and closed form.
- We show the fit is stable and transfers: disjoint prompt halves agree at 0.99, the ratio-2 table explains
  ratio 4 with 2.4% more error, and tables fitted at 1024^2 and 1280^2 agree.
- We show that at 4096^2 on FLUX.1-dev SCA beats AReA on 12 of 15 metrics at 2.63 times the speed and matches it
  on Krea in a third of the time, while repeating subjects in [DUP] 26.7% of prompts against 4.2%.

**Figure that lands it: "Same ratio, trusted reference."** One panel, 0.42 text width, two canvases side by side
at true relative size. Left, the 4096^2 grid (256 tokens a side) with one query tile, its 64-token window in amber
and every fourth block-centre key in blue; above it a small purple dial "AReA: set by hand" and a red cross over a
flat-field thumbnail "full attention: no reference". Right, the 1024^2 grid (64 a side) with a 16-token window
and every fourth key, the same picture shrunk four times; above it a green check over a real 1024^2 image "full
attention: reference" and an arrow into a parabola J(pi) with its minimum marked. Between them "r = 4" with an
equals sign. Under 20 words. Caption lead: "**The target's geometry, rebuilt where full attention is right.**"
Risk: it overlaps Figure 3a, which already draws the calibration; it shows the recipe, not a finding.

**Risks a reviewer would raise.**
1. *"Any free parameter" is false for the objective we have.* Output error picks local attention from step 1 as
   the best hard switch (sigma_star = 1.0 in calib/k1_1024.json at both ratios), which v0 shows is catastrophic
   (ImageReward -1.33, all 26 prompts repeat). Scope the claim: the recipe sets how much each view counts once the
   layout is fixed; the opening stays hand-set, as in AReA.
2. *One fitted scalar.* Gate, window sizes, sigma bins and bridge size are hand-set (Table F.2). "A general recipe,
   shown for one weight per head." If the round-6 calibrated text weight lands, there are two.
3. *Emulation fidelity.* Calibration omits the bridge and sees 16 times fewer keys per branch at ratio 4; the
   deployed operator's error is untested (held task RR05).
4. *Prior art.* HRDiT fits per-head scopes with a profiling search per target size; MoA and DuoAttention fit
   per-head spans against dense attention in language models. The difference must be stated: the reference is full
   attention at the training resolution, reached by ratio emulation, one pass for every size.
5. *Trivial math.* The closed form is a scalar projection (reviewer 1, item 11). Sell the emulation, not the solver.
6. *"AReA is a hand-set point in the family"* holds for the weight axis only (phases = weights 0, then 1, for every
   head). AReA's window and pooling schedules are other axes that we also set by hand.

**Verdict.** The clearest novelty claim and the weakest memory hook: it is still about fitting, and nothing in it
tells the reader something new about the model.

---

## Framing B: Reach belongs to the head (recast from "two kinds of heads")

**Why the recast.** The data reject a binary taxonomy. Per-head weights form a continuum, skewed local: on the
scale-4 table 19% of heads stay above 0.9 in at least three quarters of the noise bins and 5% stay below 0.3
(evidence/pi_stats_out.txt). Head typing also has precedents in this exact setting (UltraImage's run-time head
pattern types and its stable local and global roles) and outside it (DuoAttention, Sparse VideoGen). And
"failure comes from treating both kinds the same" fails a direct check: AReA treats all heads alike and repeats
subjects about six times less (4.2% against 26.7%). What the data support is stronger and simpler: how far a head
attends is set by the head, far more than by the step.

**Thesis.** How far a head attends is a property of the head, not of the denoising step, so SCA gives each head its
own measured mix of a native window and a whole-image view where AReA gives every head one schedule.

**Title.** Heads, Not Steps: Per-Head Receptive Fields for Training-Free High-Resolution Diffusion Transformers

**Abstract (163 words).**
Training-free methods run diffusion transformers at 4096^2 by restricting attention, and AReA gives every
head one schedule: pooled keys, then windows shrinking to native size. Measured at the training resolution, where full
attention is still correct, reach is set by the head far more than by the step. We introduce Scale-Covariant Attention (SCA), which gives each head its own mix of a native-size window
and a native-size grid of the whole image. The mix is fitted once at 1024^2 against the model's own full attention,
with both views shrunk to the target's image-to-window ratio. It spreads across heads (standard deviation 0.24 to
0.26), repeats on disjoint prompts (correlation 0.99), and tracks each head's attention mass within 16 tokens
(rho = 0.77). On FLUX.1-dev at 4096^2, SCA beats AReA on 12 of 15 metrics in 77.1
against 202.4 s, and matches it on Krea in a third of the time. It repeats a person or animal in 26.7% of prompts
naming one, against 4.2%.

[DUP] slot: the last sentence's 26.7% and 4.2% (replace both if the round-6 operator lands).

Optional sentence once the variance split is archived (NEW): "After the first 8 steps, the head accounts for 97.5%
of the variance in how much attention a query keeps within 16 tokens, and the noise level for 2.5%."

**Contributions.**
- We show that the reach of attention heads, measured at the training resolution, is set by the head far more than
  by the denoising step, on FLUX.1-dev, FLUX.1-Krea-dev and FLUX.2-dev (needs the NEW split archived).
- We give each head its own mix of a native window and a whole-image view, fitted once at 1024^2 by emulating the
  target's image-to-window ratio; per-head mixing beats per-layer and shared rules in output error and in images, in
  the same order.
- We show that at 4096^2 on FLUX.1-dev SCA beats AReA on 12 of 15 metrics at 2.63 times the speed and matches it on
  Krea in a third of the time, while repeating subjects in [DUP] 26.7% of prompts against 4.2%.

**Figure that lands it: "One schedule, or one mix per head."** Mock at figs/concept_mock.png (script beside it,
data calib/k1_1024.npz). One panel, 0.42 text width. x: sampling step 1 to 28 at 4096^2, the first 8 shaded as the
global view both methods share. y: weight on the window, pi_h. AReA in our notation: a purple dashed step, 0 for
8 steps then 1, labelled "AReA: every head". SCA: 1,368 thin lines, one per head, the deployed weight at each step
(ratio read from the window schedule), coloured blue to amber by the head's mean weight; the median in SCA green.
The lines spread over the whole range and stay nearly flat after the gate; that is the claim, visible without the
caption. 9 words of text.

**Risks a reviewer would raise.**
1. *Head heterogeneity is known* (UltraImage, HRDiT, DuoAttention, Sparse VideoGen, MoA). The new parts are the
   measured comparison of the head and step axes and the link from measurement to parameter. The paper must say
   which part is new or a reviewer will say it for us.
2. *"Not the step" collides with our own opening and four-step shrink.* Scope every statement to "after the
   opening". Layout at high noise needs the step axis (v0; the gate sweep: 20, 15 and 7 of 26 prompts repeat for
   gates 0.93, 0.89 and 0.85), and output error cannot see it.
3. *A different statistic from AReA's.* AReA's distance to dominant keys shrinks 5.5x. Ours, mass share within r
   tokens, is the quantity that sets the output by Lemma 1, and its head average narrows too. Say both.
4. *The strongest numbers are NEW.* Until results/heads_view/variance_split.csv exists with SHA256SUMS, the paper
   can only use the existing spread-against-drift numbers.
5. *"Then why does AReA, which treats heads alike, repeat six times less?"* Duplication is a layout failure at high
   noise, on the step axis, where AReA keeps wide windows for 20 steps at 2.6 times our time. After the opening our
   mix helps (local attention alone repeats in 25 of 26 prompts, the per-head mix in 15), and the bridge makes most
   of the cut.
6. *The fitted weight is not the measured reach.* It tracks it (rho = 0.77) but sits more local: after the gate the
   global view gets a mean weight of 0.23 to 0.30, while 53 to 61% of the image attention lies outside the ratio-4
   emulated window (calib/k1_1024.npz, recomputed 2026-11-05). Write "tracks", never "equals".

**Verdict.** The most memorable line and a direct answer to AReA's Figure 3, measured the same way. Its novelty has
to lean on the emulation, because heads differing is not new on its own.

---

## Framing C: Native-geometry attention

**Thesis.** Every query should see the geometry the model was trained on, a native-size window and a native-size
grid of the whole image, and the only free choice, how much each head weights the two, is measured at the training
resolution.

**Title.** Native-Geometry Attention for Training-Free High-Resolution Diffusion Transformers

**Abstract (162 words).**
Diffusion transformers trained near one megapixel fail at 4096^2, where full attention meets 16 times more tokens
and offsets four times the trained range. We introduce Scale-Covariant Attention (SCA), which shows every query two
views at the training geometry: an exact native-size window around it, and a native-size grid of block-centre keys
over the whole image. Both keep FLUX.1's position encoding, at a cost linear in pixels. The only free choice is how much each head weights the two views. We fit it once at
1024^2, where full attention is still correct, by shrinking both views to the target's image-to-window ratio.
Bridge tokens, text offsets in the window and four wider windows after the opening still leave the
trained range. On FLUX.1-dev at 4096^2, SCA beats AReA on 12 of 15 metrics in 77.1 against 202.4 s, and matches it on
FLUX.1-Krea-dev in a third of the time. It repeats a person or animal in 26.7% of prompts naming one, against 4.2%.

[DUP] slot: the last sentence's 26.7% and 4.2% (replace both if the round-6 operator lands).

**Contributions.**
- We build attention for a diffusion transformer at 4096^2 from two views at the training geometry, an exact
  native-size window and a native-size grid of the whole image, at cost linear in pixels.
- We fit how much each head weights the two views at the training resolution, by emulating the target's
  image-to-window ratio, in closed form.
- We show that at 4096^2 on FLUX.1-dev SCA beats AReA on 12 of 15 metrics at 2.63 times the speed and matches it on
  Krea in a third of the time, while repeating subjects in [DUP] 26.7% of prompts against 4.2%.

**Figure that lands it: "How far outside the trained range each step reaches."** One panel, 0.42 text width.
x: sampling step 1 to 28 at 4096^2; y: largest image query-key offset in units of the trained range (64 tokens).
Full attention: 4 at every step (grey). AReA, from its released code: 1 for steps 1 to 4 (pooling 4), 2 for steps
5 to 8 (pooling 2), then 2.8 shrinking geometrically to 1 over 20 window steps (purple). SCA: 1 for 8 steps, then
2, 1.75, 1.5 and 1.25, then 1 (green); a note that 1,024 bridge keys and text-image offsets in the window branch
are excepted. The per-head weights are absent, which is this framing's problem.

**Risks a reviewer would raise.**
1. *Not true as stated for the deployed operator.* The 1,024 bridge keys keep their true positions (offsets up to
   255), text-image offsets in the window branch reach 255 (Remark 1), the four wide steps reach 127, positions
   round in bf16 beyond 256, and the window branch holds 5,632 keys against 4,608 in training.
2. *AReA sells the same property:* "no RoPE extrapolation", "keeps the model close to its training regime".
   A reviewer sees no new principle.
3. *Native geometry does not predict the failure that matters.* AReA leaves the trained range for more steps than
   SCA and repeats subjects less.
4. *Outcome-dependent.* It reads truer only if round 6 lands the global text read-out (text enters through the
   global view at native offsets).

**Verdict.** The weakest thesis. Keep it as a design property stated with its exceptions (Section 2, Design
choices; Table J.1), not as the idea.
