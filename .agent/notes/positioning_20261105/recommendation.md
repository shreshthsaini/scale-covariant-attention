# Recommendation: lead with the head, claim the emulation (positioning agent, 2026-11-05)

Apply after the round-6 method result lands. Nothing here touches main.tex yet. The LaTeX blocks are written
against build delivery_eeofzli6 and use its macros. New macros are listed in step 0.

## 1. The choice

**Framing B's observation as the hook and the picture, Framing A's emulation as the claimed contribution, and
Framing C demoted to a design property with its exceptions.** One-sentence thesis, in AReA's form:

> A head's reach is a property of the head, not of the denoising step, so it can be measured once at the training
> resolution, where full attention is still correct, and reused at any size with the same image-to-window ratio.

Side by side with AReA's: *"Attention narrows as denoising proceeds, so AReA's receptive field follows the step,
one schedule for every head."* Ours: *"Reach is set by the head, so SCA measures each head where full attention is
still right."*

Why this one:
1. **It answers AReA in kind.** Both papers measure vanilla FLUX attention at 1024^2. AReA averages over heads and
   finds a trend over steps. De-averaged, after the opening the head accounts for 97.5% of the variance in
   near-field attention mass and the noise level for 2.5%. The head-averaged curve that AReA follows accounts for
   0.65%. The pattern holds on Krea (97.6% head) and FLUX.2-dev (99.1% head). This makes ours an observation of
   the same kind as AReA's, measured the same way, and larger.
2. **The novelty sits where prior art is thin.** Heads differing is not new (UltraImage, HRDiT, DuoAttention,
   Sparse VideoGen). Measuring at the training resolution by ratio emulation, once for every size, is new.
   HRDiT searches per target size, UltraImage types heads per image at run time, and language-model methods fit
   against dense attention where it is valid. So the title and bullets claim the emulation, and the figure shows
   the observation.
3. **It survives every round-6 outcome.** Only the duplication sentences change (Section 4 below).
4. **It keeps the name honestly.** "Scale-covariant" becomes the one-sentence property that makes measuring at
   1024^2 possible: the weight depends on the head and the image-to-window ratio, not the absolute size. The test
   is 1024^2 against 1280^2 at ratio 2, plus ratio 2 against ratio 4. Repo, Hugging Face and code names stay.
5. **C stays a design property** (Section 2, Design choices; Table J.1), because the deployed operator leaves the
   trained range for bridge keys, text offsets and four wide steps, and AReA already sells "no RoPE extrapolation".

What this framing must never claim: that heads differing is new; that the step does not matter (it does, for
layout, before and just after the gate); that the fitted weight equals the measured reach (it tracks it,
rho = 0.77); that any parameter can be fitted this way (output error picks no opening, sigma_star = 1.0); or that
SCA is better than AReA in general.

## 2. Step 0, before any text edit (CPU, minutes; paper agent or method agent)

1. Move `.agent/notes/positioning_20261105/evidence/variance_split.py` to `scripts/variance_split.py`. Make it write
   `results/heads_view/variance_split.csv` with columns
   `checkpoint,res,sigma_range,quantity,radius,layer,head_within_layer,noise_within_head,head_avg_curve,n_heads,n_bins`
   for calib/{k1_1024,krea_1024,flux2_1024,k1_1280b}.npz and both sigma ranges. Add the file to
   results/heads_view/SHA256SUMS. Expected values are in evidence/variance_split_out.txt; the post-gate radius-16
   rows are FLUX.1-dev 0.166 / 0.809 / 0.025 / 0.0065, Krea 0.156 / 0.820 / 0.024 / 0.0079, FLUX.2-dev
   0.104 / 0.887 / 0.009 / 0.0004, and 1280^2 0.154 / 0.824 / 0.022 / 0.0046.
2. In gen_main_tables.py, emit to headline.tex:
   - all five with one decimal, from the FLUX.1-dev, sigma < 0.89, radius-16 rows unless named otherwise:
     `\hlVarHead` = 100 x (1 - noise_within_head) = 97.5 (unrounded 97.4993); `\hlVarStep` = 100 x
     noise_within_head = 2.5 (2.5007); `\hlVarAvg` = 100 x head_avg_curve = 0.7 (0.6517);
     `\hlVarHeadKrea` = 97.6 (97.5697); `\hlVarHeadFluxTwo` = 99.1 (99.1075). Head and step then sum to 100.
   - `\hlDupHumanSca` and `\hlDupHumanArea` from results/dup_audit_20261027/human_rates_95.csv, label `primary`,
     100 x human_rate, one decimal (26.7 and 4.2 today; replaced by the round-6 values if the operator changes).
3. Build `figures/concept.pdf` from `figs/concept_mock.py` (move it to paper/figures/src/fig_concept.py, read the
   curves from a new results/heads_view/concept_curves.csv, keep scafig, size 0.42 TEXT; the audit's width warning
   is expected for a wrapfigure).
If step 0 is skipped, use the fallback sentences marked FALLBACK below; they use only numbers already in the paper.

## 3. The edits, in paper order

### 3.1 Title (line 159)
```latex
\title{Scale-Covariant Attention:\\Per-Head Receptive Fields Calibrated at the Training Resolution}
```
Alternatives: "Calibrate at 1K, Generate at 4K: Per-Head Attention for Training-Free High-Resolution Diffusion
Transformers"; or, bolder and riskier (it invites "you also schedule steps"), "Heads, Not Steps: Per-Head Receptive
Fields for Training-Free High-Resolution Diffusion Transformers".

### 3.2 Abstract (line 171), outcome (iii) "no round-6 operator passes"; variants in Section 4
```latex
Diffusion transformers trained near one megapixel fail at $4096^2$: attention meets token counts and offsets it never saw, at 256 times the cost. Training-free fixes reshape attention by hand or per target size, since full attention at the target is no reference. At the training resolution, where full attention is valid, reach depends on the head far more than on the denoising step. We introduce Scale-Covariant Attention (\sca{}), which gives each head its own mix of a native-size window and a native-size grid of the whole image. Shrinking both views at $1024^2$ to the target's image-to-window ratio lets the model's own full attention fit each mix in closed form, once for every size. The fit repeats on disjoint prompts (correlation 0.99), and the ratio-2 table explains ratio 4 with 2.4\% more error. On FLUX.1-dev at $4096^2$, \sca{} beats AReA, which gives all heads one schedule, on \hlWinsArea{} of \hlMetricsArea{} metrics and loses none, in \hlSecSca{} against \hlSecArea\,s. On FLUX.1-Krea-dev, AReA's checkpoint, quality is on par in a third of the time. \sca{} repeats a person or animal in \hlDupHumanSca\% of prompts naming one, against \hlDupHumanArea\% for AReA. Code: \codeurl.
```
This drops the progressive-HRDiT sentence (it stays in Section 4.1 and Table 2) and the wrong "about 7%
corrected".

### 3.3 Introduction (lines 182 to 213): five paragraphs and three bullets, replacing the current text
Paragraph 1 (context, ending in the question):
```latex
Diffusion transformers~\citep{peebles2022dit} such as FLUX.1~\citep{flux2024} and Stable Diffusion~3~\citep{esser2024sd3} are trained near one megapixel. At $4096\times4096$ the image has 16 times as many tokens, full attention costs about 256 times as much, and every layer meets token counts and position offsets it never saw in training. Full attention then returns stripes or a flat field (Figure~\ref{fig:teaser}, left). Fixes that restrict attention must decide how far each head looks, and at the target nothing can check that decision, because full attention there is the failure. We ask whether it can be measured where full attention is still correct, at the training resolution.
```
Figure 2 (replaces the Moai wrapfigure; same size, same place):
```latex
\begin{wrapfigure}{r}{0.42\linewidth}
  \wraptop
  \centering
  \includegraphics[width=\linewidth]{figures/concept.pdf}
  \caption{\textbf{One schedule, or one measured mix per head.} In \sca{}'s notation AReA gives every head weight 0 on its windows for 8 steps, then 1. \sca{} gives each of FLUX.1-dev's 1,368 heads the weight calibrated at $1024^2$ (thin lines, coloured by mean; green: median), which tracks each head's measured reach (Figure~\ref{fig:heads}d).}
  \label{fig:concept}
  \wrapbot
\end{wrapfigure}
```
Paragraph 2 (how methods decide today, and what the measurement shows):
```latex
Some fixes edit the positional encoding~\citep{zhuo2024lumina,issachar2025dype,zhao2025ultraimage,rajabi2026sega}; others restrict the keys each query sees~\citep{zhang2023hidiffusion,ma2025resdit,xue2026hrdit,rajabi2026area}. AReA~\citep{rajabi2026area} finds that attention narrows during denoising, averaged over heads, and gives every head one schedule: pooled keys for 8 of 28 steps, then windows that shrink to native size. HRDiT searches per-head scopes for each target size, and UltraImage types heads at run time~\citep{xue2026hrdit,zhao2025ultraimage}. Measured head by head at $1024^2$, reach varies far more between heads than between steps (Figure~\ref{fig:concept}). After the first 8 steps, the head accounts for \hlVarHead\% of the variance in how much attention a query keeps within 16 tokens, and the step for \hlVarStep\%.
```
FALLBACK for the last sentence: "Each head's weight spreads over the whole range, with a standard deviation of 0.24
to 0.26 across heads, while the mean moves only from 0.69 to 0.77 across noise levels."

Paragraph 3 ("We introduce"):
```latex
We introduce Scale-Covariant Attention (\sca{}), which gives each head its own reach, calibrated once at the training resolution (Figure~\ref{fig:method}). Each head mixes two views that keep the training geometry: an exact native-size window around the query, and a native-size grid of block-centre keys over the whole image. We fit the mix by emulation. At $1024^2$, shrinking the window and the grid to the target's image-to-window ratio lets the model's own full attention score both views, and the best weight has a closed form. One pass serves every target size. We call this scale covariance: the weight depends on the head and the image-to-window ratio, not on the absolute size.
```
Paragraph 4 (what the fit cannot see, and what is set by hand):
```latex
Output error cannot see layout, so sampling opens with the global view for 8 of 28 steps, as in AReA. Windows from the first step draw five Moai where the prompt asks for one (Figure~\ref{fig:failure}a). The window then shrinks from 128 tokens to the native 64 over four steps and holds 1,024 bridge tokens sampled across the image. The gate, window sizes, and bridge are set by hand; only the weights are calibrated (Table~\ref{tab:handset}).
```
Paragraph 5 (what we found, with numbers):
```latex
The calibrated weights repeat on disjoint prompts (correlation 0.99) and track each head's attention mass within 16 tokens ($\rho=0.77$). They transfer: the ratio-2 table explains ratio 4 with 2.4\% more output error, and tables fitted at $1024^2$ and $1280^2$ agree. Per-head weights beat per-layer and shared ones in output error and in images, in the same order. At $4096^2$ on FLUX.1-dev, \sca{} beats AReA on \hlWinsArea{} of \hlMetricsArea{} metrics and loses none, at $\hlSpeedArea\times$ the speed. On FLUX.1-Krea-dev, AReA's checkpoint, quality is on par in a third of the time. \sca{} repeats subjects more: a person or animal in \hlDupHumanSca\% of the \hlDupN{} prompts naming one, against \hlDupHumanArea\% for AReA (Section~\ref{sec:dup}).
```
Bullets:
```latex
\begin{itemize}
  \item We show that the reach of attention heads, measured at the training resolution, varies far more between heads than between denoising steps, on FLUX.1-dev, FLUX.1-Krea-dev, and FLUX.2-dev (Section~\ref{sec:analysis}).
  \item We calibrate each head's mix of a native window and a whole-image view once at $1024^2$, in closed form, by emulating the target's image-to-window ratio against the model's own full attention (Section~\ref{sec:calib}).
  \item We show that \sca{} beats AReA on \hlWinsArea{} of \hlMetricsArea{} metrics at $\hlSpeedArea\times$ the speed on FLUX.1-dev at $4096^2$ and ties it on Krea, but repeats subjects more (\hlDupHumanSca\% against \hlDupHumanArea\%; Section~\ref{sec:main}).
\end{itemize}
```
FALLBACK bullet 1 (no archived split): "We show that each head's balance between near and far keys, measured at the training
resolution, differs widely between heads, repeats across prompts, and transfers across image sizes at a fixed
image-to-window ratio (Section~\ref{sec:analysis})."

The current paragraph 5 numbers (bridge 15 to 5 of 26, wide windows 4, 57% to 26%, 8% time) move out of the
introduction; they already appear in Sections 2.4 and 4.4.

### 3.4 Section 2 opening (insert after `\label{sec:method}`, line 224)
```latex
Each head gets two views of the image that keep the training geometry, and one weight for how much of its output comes from each. Only the weight is measured: it is fitted at $1024^2$ (Section~\ref{sec:calib}) and read at the target's image-to-window ratio (Section~\ref{sec:sched}).
```
And add a run-in paragraph at the end of Section 2.5, after the interpolation sentences (this answers reviewer 1,
item 6, with the tested domain):
```latex
\paragraph{Scale covariance.} Emulated ratio $r$ keeps two proportions of a target $r$ times larger, image side to window side and global keys to window keys, with $r^2$ times fewer keys per branch. We assume the weight depends on $r$, not on the absolute size, and test this in Section~\ref{sec:analysis}.
```

### 3.5 Section 3 (line 376): retitle and lead with the observation
`\section{What the heads want}` becomes `\section{Reach is set by the head}`. Insert as its first run-in, before
"Heads differ, and each head's weight depends on the ratio":
```latex
\paragraph{Reach varies by head, not by step.} For 256 queries per prompt we record the share of each head's image attention within $r$ tokens of the query. After the first 8 steps, the head accounts for \hlVarHead\% of the variance in this share at $r=16$, and the noise level for \hlVarStep\%. Most of the head's part lies between heads of the same block. The head-averaged curve accounts for \hlVarAvg\%, and it narrows as denoising proceeds, as AReA reports. FLUX.1-Krea-dev and FLUX.2-dev give \hlVarHeadKrea\% and \hlVarHeadFluxTwo\% for the head (Appendix~\ref{app:calib}).
```
Appendix B gains one small table (all four checkpoints and sizes, both sigma ranges, radii 8 and 16) from
variance_split.csv, with the caveat that AReA's statistic is distance to dominant keys and ours is mass share,
the quantity that sets the output by Lemma 1.

### 3.6 Related work (lines 492 and 494)
In "Training-free high-resolution sampling", replace the sentence beginning "Of the closest methods":
```latex
Of the closest methods (Table~\ref{tab:related}, Appendix~\ref{app:related}), AReA~\citep{rajabi2026area} sets one schedule for all heads by hand and adds bridge tokens against duplication; \sca{} borrows its opening, block-centre keys, and bridge selection rule. In \sca{}'s notation AReA's phases set every head's window weight to 0, then 1. On our two operators any switch shared by heads has relative output error of at least 0.261 at ratio 4, against 0.200 per head (Proposition~\ref{prop:opt}).
```
In "Head- and layer-specific attention", replace the paragraph's first three sentences with the first block and
its last sentence with the second:
```latex
HRDiT searches per-head scopes under a cost budget for each target size~\citep{xue2026hrdit}; UltraImage types heads by entropy at run time and noted stable local and global head roles before us~\citep{zhao2025ultraimage}. Sparse attention varies windows, spans, or key scales per layer or head in diffusion transformers~\citep{yuan2024ditfastattn,xi2025svg,zhang2025sta}, language models~\citep{fu2024moa,xiao2024duoattention}, and vision transformers~\citep{pan2022hilo}. These settings are often fitted against dense attention at the size where it runs. At $4096^2$ a diffusion transformer's dense attention is the failure, so \sca{} fits at the training resolution by emulating the target's image-to-window ratio, once for every size.
```
```latex
Head roles, two branches, a global opening, and shrinking windows have precedents; the calibration by emulation and the measured comparison of heads and steps are ours.
```
Keep the neighbourhood-attention sentence between them.

### 3.7 Limitations and conclusion (lines 499 to 501)
Limitations: in "Other limits", add after the first sentence:
```latex
The opening, window sizes, and bridge are set by hand because output error cannot see layout: fitted on output error alone, the best hard switch is local attention from the first step.
```
Conclusion, outcome (iii):
```latex
\paragraph{Conclusion.} A head's reach can be measured once at the training resolution, where full attention is still correct, by shrinking the window to the target's image-to-window ratio. Measured this way, reach varies far more between heads than between steps, so \sca{} gives each head its own calibrated mix where AReA gives all heads one schedule. On FLUX.1-dev at $4096^2$ it beats AReA on \hlWinsArea{} of \hlMetricsArea{} metrics at $\hlSpeedArea\times$ the speed, and it matches AReA's quality on Krea in a third of the time. It repeats subjects more often (\hlDupHumanSca\% against \hlDupHumanArea\%). Code: \codeurl.
```

### 3.8 The wrong AReA correction (independent of the framing, required by the 2026-11-05 addendum)
"about 7%" (and "13% any subject") for AReA must go at lines 171, 418, 454, 501 and 1359 and in the Table 1
caption produced by gen_main_tables.py. Use the human-validated rates with their intervals: SCA 26.7% [24.6, 36.5],
AReA 4.2% [1.6, 17.0], HRDiT 6.3%, full attention 4.2% (subject missing in 22 of 37 labelled images).

## 4. Round-6 variants (only the duplication sentences change)

**(i) The calibrated text weight (r6_ctw) passes the bar.** The second parameter is then measured the same way,
which is the strongest version of this framing. Abstract, last sentence before the code URL:
```latex
The same measurement sets how strongly each window hears the prompt, which cuts repeated people and animals to \hlDupHumanSca\% of the \hlDupN{} prompts naming one, against \hlDupHumanArea\% for AReA.
```
Bullet 2 adds "and each window's text weight from the share of the head's attention that falls inside it". The
conclusion's last sentence becomes "Measured the same way, each window's text weight removes the repeated subjects
(...% against ...%)." A two-line justification the method agent should check before it is printed (bridge keys aside): with
$s$ the head's image-attention share inside the window and text logits shifted by $\log s$, the window's softmax
gives the text $sZ_T/(sZ_T+Z_n)$ with $s=Z_n/(Z_n+Z_f)$, which equals full attention's $Z_T/(Z_T+Z_n+Z_f)$. So the
window hears the prompt exactly as loudly as full attention would. Without the shift it hears it about $1/s$ times
louder; the mean image share within 8 tokens at 1024^2 is 0.42 (column "mean" of evidence/variance_split_out.txt, all
bins), a factor of about 2.4 when text mass is small. This gives Section 2.4 a mechanism, not only a fix.

**(ii) Another candidate passes (global text read-out, global text gate, bridge count weight, native opening).** The
thesis is unchanged. The abstract's last sentence names the mechanism in one clause and gives the rates. For the
global text read-out, add one sentence to the design-choices paragraph: text now enters the window branch through
the global view at native offsets, which also narrows the exception list of framing C.

**(iii) Nothing passes.** Use the texts in Section 3 as written. Duplication stays a stated weakness, next to every
12-of-15 claim, with human-validated rates.

## 5. Page budget and checks

- The main text ends exactly on page 10 (HANDOVER Section 27). Figure 2 swaps one 0.42 wrapfigure for another,
  so it is page-neutral. Word counts below come from one counter that drops citations and counts each macro as one
  word, applied to both the current text and the blocks above.

  | Edit | Current | Proposed | Change |
  |---|---|---|---|
  | Abstract | 194 | 189 | -5 |
  | Introduction, paragraphs and bullets | 617 | 597 | -20 |
  | Section 2 opening | 0 | 46 | +46 |
  | Scale-covariance run-in (Section 2.5) | 0 | 50 | +50 |
  | Section 3 first run-in | 0 | 90 | +90 |
  | Related work (three replaced sentences) | about 105 | 190 | about +85 |
  | Limitations sentence | 0 | 32 | +32 |
  | Conclusion | 66 | 94 | +28 |
  | **Total before offsets** | | | **about +306** |

  Offsets, in order of preference:
  1. Section 4 setup, Duplication paragraph: the precision-over-recall correction ("Scaling by precision over
     recall puts AReA's rate near 7% ... wide uncertainty") goes, replaced by one sentence giving the
     human-validated rates: about -40. Section 4.2's "survives correction (about 28% against 7%...)": about -10.
  2. Section 3, "Heads differ, and each head's weight depends on the ratio": keep the standard deviation, rho = 0.77
     and the ratio-2 against ratio-4 means; drop "Half the (head, noise level) cells lie in [0.2, 0.8]" and the
     figure list: about -35.
  3. Related work: drop the last sentence of the AReA block (the 0.261 against 0.200 bound is already in Section 3,
     "Per-head beats coarser rules"): -28.
  4. Scale-covariance run-in: keep only its second sentence: -25.
  5. Limitations sentence: drop if still over: -32.
  6. Section 2 opening: keep only its second sentence: -24.
  Offsets 1 to 4 bring the net to about +170 words, and all six to about +115, roughly a quarter and a sixth of a
  page. The rest has to come from the paper agent's usual tightening of Sections 4.1 to 4.5, or from moving
  Table 4 (generality) to the appendix with a one-line pointer. The rebuild decides.
- Rebuild and check: main text on 10 pages; every float within one page of its first reference; no sentence over
  35 words (every block above is at most 33 when rendered); no em dash or section symbol; Figure 2 text at 6 pt
  or more.
- Rubric items this moves: "question and wrap" (the wrap now carries the idea, not a failure); "introduce and
  bullets" (bullet 1 is a finding about the model); "method diagram" (the concept panel is the missing picture of
  the idea, and Figure 3 stays the mechanism).
