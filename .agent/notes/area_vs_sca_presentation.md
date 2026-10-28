# AReA (arXiv 2610.17227v1) vs SCA: presentation comparison, 2026-10-27

Sources: AReA PDF (33 pp; renders and text in /scratch/09032/saini_2/tmp/area_digest/), project page
https://rajabi2001.github.io/area/, and paper/main.tex before this pass (commit 75baaa8, 49 pp).
Scale per item: 0 below the gold-standard bar, 1 meets it, 2 above it (gold-standard-paper skill, Section 8).
"SCA" is the score before this pass; "after" is the score once the action landed.

| # | Item | AReA | SCA | Evidence | Action | After |
|---|---|---|---|---|---|---|
| 1 | Abstract | 0 | 1 | AReA: 177 words, no number, no URL. SCA: 174 words, numbers, code URL. | none | 1 |
| 2 | Intro hook | 1 | 1 | AReA: generic first paragraph, the hook is the Fig. 2 failure strip on p. 2. SCA: generic first paragraph, question only at the end of paragraph 2. | Open paragraph 1 on the concrete failure pair (full attention gives a flat field; native windows from step 1 draw five Moai) and ask the question at its end. | 2 |
| 3 | Contributions | 0 | 1 | AReA: no contribution list. SCA: three "We" bullets, last with 11 of 15 metrics and 2.85x. | none | 1 |
| 4 | Method exposition | 1 | 1 | AReA: 3 equations, prose, cost only as O(N^2) and 1/f^2; every concrete size is in the appendix; notation contradictions (rho vs rho^2, sigma* set by steps). SCA: 5 equations with "where" lines, two propositions, but no concrete sizes in the main text. | Add a worked example at 4096^2: 256x256 grid, 4,608 keys per branch against 66,048, offsets 63 against 255, 6.06e8 against 4.36e9 pairs, 18.7 against 87.3 ms per layer. | 2 |
| 5 | Method diagram | 1 | 2 | AReA Fig. 4: real images in Phase II, but about 43 labels at 3.9 to 8.1 pt. SCA Fig. 3: two lettered panels, real thumbnail, appendix zoom (Fig. D.1). | none | 2 |
| 6 | Algorithms | 0 | 2 | AReA: no algorithm box. SCA: two boxed algorithms side by side plus appendix code. | none | 2 |
| 7 | Hero | 1 | 2 | AReA Fig. 1: 15-image mosaic over 4 models up to 8k, tags at 3.9 pt, 20-word caption, no comparison. SCA Fig. 1: curated outputs with native-pixel insets plus a six-method strip with 1:1 crops; caption walks the panels. AReA shows more models. | Tighten the caption to lead plus two sentences. | 2 |
| 8 | Qualitative grids | 1 | 2 | AReA Fig. 5: 4x5, no full-attention column, no crops. SCA Fig. 6: three curated prompts, every method's 1:1 crop with its MUSIQ_c; random galleries labelled random. | Tighten captions. | 2 |
| 9 | Per-step visualisation | 1 | 0 | AReA: Fig. 12 (appendix, full page) shows Phase I and II steps with DWT maps; the project page has step videos. SCA: none. | Needs x0 traces of SCA and AReA on one prompt (scripts/sample_trace.py, web agent). /scratch/09032/saini_2/sca/web_assets held no traces during this pass and the SU freeze blocks GPU runs. Request the trace run; the figure goes in Section 4.2 once traces exist. | 0 (open) |
| 10 | Attention / receptive-field visualisation | 1 | 0 | AReA Fig. 3: real attention maps and query-key distance at 1024^2 for vanilla FLUX.1, averaged over heads, never linked to AReA's parameters. SCA: none. | New Figure 4 from the calibration archive: both key sets drawn on a real 4096^2 output, the measured full-attention reach of three heads, their pi(sigma), and pi against near-field mass over all 1368 heads (Pearson r, results/heads_view). | 2 |
| 11 | Latency plot | 1 | 2 | AReA Fig. 6: text 3.0 to 5.4 pt; speedups disagree with Table 1 (2.00x vs 2.08x). SCA Fig. 8: 5 sizes to 8192^2, speedups in grey, exponents in text, per-layer timings in Table D.2. | none | 2 |
| 12 | Main table | 1 | 2 | AReA T1: broad (2 models, 4 sizes) but 5.6 pt font, no intervals, MUSIQ/CLIP-IQA/ARNIQA announced but absent. SCA Tables 1-2: time columns, multi-stage rows, significance marks, all metrics in Table J.1. | Script Tables 1-4 from results/*.csv (one command swaps numbers); tighten captions. | 2 (Tables 1 to 4 scripted; published-numbers table moved to Appendix H for page fit) |
| 13 | Statistics | 0 | 2 | AReA: no intervals, seeds or tests; T5 on 20 prompts. SCA: paired bootstrap intervals, three seeds, 195 and 1000 prompts. | none | 2 |
| 14 | Ablations | 1 | 2 | AReA: 8 variants, delta_max, rho and the 70% rule never swept; some effects within noise (Halo). SCA: 6 variants with significance and calibration error, sensitivity table, appendix band guard. | none | 2 |
| 15 | Related work | 1 | 1 | AReA: two run-in paragraphs plus a 1,460-word appendix, no mechanism table, no sparse-attention work. SCA: two paragraphs, sparse attention covered, no table. | Add a mechanism table (local view, global view, per-head, fitted offline, position range, cost, single pass) for FreeScale, DyPE, SEGA, UltraImage, HRDiT, ResDiT, AReA, SCA; shorten the prose. | 2 |
| 16 | Limitations | 0 | 1 | AReA: no limitations section and no failure cases shown. SCA: concrete limits, but the late-duplication rate sits mid-paragraph. | Lead with late duplication (5 of 11 uncurated 4096^2 samples), say that a fix is in progress and that no reported result uses it. | 2 |
| 17 | Reproducibility | 0 | 1 | AReA: no statement; step count, CFG and seeds are missing. SCA: statement maps each detail to an appendix; hyperparameter table; code listings. | none | 1 |
| 18 | Appendix | 1 | 2 | AReA: sections A to J, no contents page. SCA: contents page, zoomed diagram, code, "expands Table N" openings, random galleries, prompts. | Move the old Figure 4 (all heads, split-half r = 0.989) into Appendix B. | 2 |
| 19 | Captions | 1 | 1 | AReA: leads mix Title and sentence case; 20 to 136 words. SCA: bold leads, but several main captions run four to six sentences. | Every main caption becomes a bold lead plus at most two sentences. | 1 (lead plus at most two sentences everywhere, but Figures 1, 3 and 4 run 55 to 70 words, over the 50-word house cap) |
| 20 | Figure text size | 0 | 2 | AReA: 3.0 to 3.9 pt labels in Figs. 1 and 6. SCA: 6 pt minimum. | none | 0 (review found mathtext subscripts at 4.9 to 5.6 pt in Figures 4, 6 and B.2; open) |
| 21 | Internal consistency | 0 | 1 | AReA: Fig. 6 vs Table 1 speedups, 6144x2048 vs 4096x2048, duplicated sentences, "Base"/"Baseline"/"Full attention". SCA: Figure 8 and D.2 print 2.85x/2.99x where Table D.2 gives 2.84/2.98. | Check the speedup rounding against the CSVs when the tables are scripted. | 1 |
| 22 | Project page and code | 2 | 1 | AReA: page with pipeline, videos, demo, code. SCA: code URL in the abstract and conclusion; web/ is being built by another agent. | Web agent (out of scope here). | 1 |

## Verdict
Before this pass AReA led on two items (per-step and attention visualisation) and tied on five. After it, SCA leads
on 18 of 22 items and ties on three: abstract, contributions and reproducibility, where both meet the bar but
AReA scores lower. It still trails on the per-step figure, which needs traces from a GPU run, and on the project
page, which the web agent owns. AReA's broader backbone coverage (Qwen-Image, Ideogram) is an experimental gap,
not a presentation one.

Update 2026-10-28 (integrator): after the render reviews, item 20 drops to 0 (sub-6 pt subscripts in Figures 4, 6, B.2) and item 19 to 1 (three captions over 50 words); SCA leads on 16 of 22 items.
