# Reference figure style for the SCA paper

Compiled 2026-10-19 from the arXiv PDFs (renders: `/scratch/09032/saini_2/sca/review_style/`). Counts exclude ticks; sizes are relative to 10 pt body.

## (a) References

- **AReA**: AReA: Adaptive Receptive-Field Attention for Efficient High-Resolution Diffusion Transformers (Rajabi, Roohi, Lindell, Taati; Toronto, Vector). [arXiv 2610.17227](https://arxiv.org/abs/2610.17227), [project](https://rajabi2001.github.io/area/).
- **"Left loop diffusion transformer"**: LiFT: Loop Flow Transformers (Derakhshani et al.; Amsterdam, TNO), [arXiv 2610.05538](https://arxiv.org/abs/2610.05538). "LiFT" sounds like "left", and it loops a shared DiT core. Runner-up: Looped Diffusion Transformer (Chng et al.; SenseTime, Tsinghua, NTU), [arXiv 2609.40305](https://arxiv.org/abs/2609.40305). Rejected: ELT 2604.09168, LoopMDM 2605.26106.
- **"Budget picks"**: BudgetPix: Compute-Adaptive Tokenization for Pixel-Space Image Diffusion (Kara et al.; UIUC, Google), [arXiv 2610.12307](https://arxiv.org/abs/2610.12307). Rejected: BudCache 2606.13496, ReCache 2606.06060.
- **ByteDance**: Seedream 4.0: Toward Next-generation Multimodal Image Generation (ByteDance Seed), [arXiv 2509.20427](https://arxiv.org/abs/2509.20427).
- **DeepMind**: Unified Latents (UL): How to train your latents (Heek, Hoogeboom, Mensink, Salimans; Google DeepMind), [arXiv 2602.17270](https://arxiv.org/abs/2602.17270).

## (b) Observed conventions

**Teaser.** AReA uses a full-width mosaic of 16 images with about 1 pt white gutters. Each image carries one corner tag ("FLUX.1, 6k × 2k", about 0.4x body). Two hero tiles hold 2 to 3 insets with dotted green or red frames, matching source boxes and hairline leader lines. No numbers; 22-word caption. Seedream's hero is a full-page mosaic with no text. BudgetPix's only text is 3 block titles and 16 budget headers ("100%" to "10%"). Looped-DiT's dashboard (about 60 labels) is the exception.

**Method diagram.** The low-text diagrams draw space, not sentences. AReA draws token grids with pooled dots and windows over real intermediate images, along a σ 1 → 0 timeline with a σ* marker. Dashed containers colour the two phases, and the ablation figure reuses them. Labels are 0.55x body; math is single symbols (Q, K, V). UL uses black trapezoids for networks, 2 × 2 grey squares with one red accent for latents, thumbnails, two arrows and 8 short labels. BudgetPix's overview has 8 labels. Counter-examples: LiFT (about 45 labels, two thirds math), Looped-DiT (two equations).

**Results plots.** AReA's latency plot (Fig 6) has two small panels at 1.6:1, a log y axis with units in the ticks ("300 s") and two series: baseline orange, AReA navy, about 1.5 pt, small dots. The gap is shaded. Direct labels carry fitted exponents (N^1.77, N^1.24), and per-size speedups sit as small grey numbers inside the gap. Each panel has one bold headline ("3.2×", "faster at 6144²") and the model name in a corner. No grid, no legend, no top or right spine. Only Seedream and Looped-DiT print values on every bar.

**Qualitative grids.** None of the six puts crops or metrics in its main comparison grid. AReA uses 4 rows × 5 methods with 2 pt gutters and no frames. Method names appear once under the last row, rows carry rotated "model, size" labels, and prompts are tiny italics. LiFT and BudgetPix head columns with the compute setting; BudgetPix puts its own method last. Seedream and UL grids carry no text; the caption names the order.

**Tables.** All use booktabs. AReA adds one vertical rule between model groups, cmidrule-grouped headers, latency as the first column, ↑/↓ in headers, bold best, underlined second and a lightly shaded own row.

**Captions and type.** Captions open with a bold claim. Main-text captions run 22 to 85 words in AReA and 14 to 61 in UL. Figure text is 0.4 to 0.8x body, one family per figure.

## (c) House rules

1. The teaser is pictures: no metric rows, prompts, timing or footnotes, and one tag per image.
2. A method-diagram panel has at most 12 labels of at most 3 words each. A plot has at most 2 annotations beyond its axes and direct labels.
3. No equations inside diagrams; single symbols (σ, π, s) only.
4. Draw the mechanism spatially, as a token grid over a real thumbnail plus a denoising strip, in at most two panels.
5. Captions carry the explanation: a bold claim, then at most two sentences; at most 50 words in the main text, 35 in galleries.
6. Colours are fixed paper-wide: SCA green, AReA one contrasting hue, other baselines grey. Amber and blue mean only local and global. At most 3 saturated hues per figure.
7. Plots use direct labels and no legend boxes, with at most 4 series, one marker per method, 1 to 1.5 pt lines, no grid, and top and right spines off.
8. Each panel gets one bold speed ratio. Per-size ratios are small and grey, inside the gap. Measurement conditions go in the caption.
9. Never label every bar or dot; tables hold the numbers.
10. Comparison grids name each method once, with "SCA (ours)" last.
11. Crops appear only where detail is the claim, in one frame colour about 0.6 pt wide. The caption says "1:1 crop" once.
12. Figure text is 6 to 8 pt. Bold is only for "SCA (ours)" and panel letters.
13. Use panel letters only when the caption cites them; otherwise use a short corner title.
14. Tables use booktabs, with a vertical rule only between model groups. Best is bold, second is underlined, and our row is shaded in a pale tint of the SCA green used in the plots (amended in (e)).
15. Gutters are 1 to 2 pt white, and images have no borders.

## (d) Changes to our figures

**Fig 1 teaser (p2; 53 labels, 104 words).** Delete the Steps, Time and ImageReward rows, the clean and campaign footnote, the prompt column and "768² crop, 1:1". Either promote the Fig 13 mosaic, which already follows AReA, or keep only method names on the strip. Move timing to Fig 10 and the abstract; caption at most 35 words.

**Fig 2 method (p3; 76 labels, 46 math spans).** Remove both equation boxes, the legend block, "drawn to scale with s = 4, n0 = 8, t = 2", "emulated s_c = 4", the heatmap axis titles and the step ticks 14 and 21. Draw the grid over a real latent thumbnail.

**Fig 3 qualitative (p7; 3 labels).** Already close to the references. Enlarge the full-attention thumbnails or drop them, and move "auto-placed at maximal edge energy" to the appendix.

**Fig 4 scatter (p8; 25 labels).** Remove the "Below the plotted range" box and the "guidance 7", "42 steps" and "56 steps" labels. Aspect 3.4:1 → 2:1.

**Fig 10 latency (p18; 64 labels, 98-word caption).** Copy AReA Fig 6. Drop the UltraImage and HRDiT campaign markers, the marker legend, the top speedup row and the "not run by design" note. Put the ratios inside the gap and keep "3.31×".

**Fig 11 deltas (p27; 48 labels).** Delete the 20 per-dot values.

**Figs 6 to 9 (p15 to 16; 29 to 67 labels).** Move the σ panel titles and "(1 = pure noise)" into the captions. Drop the Fig 6 bar values and the in-bar "per-layer fit", and replace legends with direct labels.

**Galleries (Figs 12 to 24).** Drop the "Full image" and "1:1 crop" row labels from Figs 16, 17, 19, 20 and 22. In Fig 19, write "local only" and "global only" without the π= prefixes. Trim captions to 35 words (Fig 22 has 144) with the shared protocol in one paragraph. Keep Fig 13.

## (e) Additional references: 2610.23653 and 2610.23854

Added 2026-10-19 from the PDFs (renders: `/scratch/09032/saini_2/sca/review_style/{lilie,mflow}/`). Both are 10 pt single-column preprints dated 16 Oct 2026, no venue.

- **LILIE**: A Language-Informed Loss for Visual Generation (Agarwal, Chandrasegaran, Chen, Sargent, Niebles, Fei-Fei, Wu; Stanford). [arXiv 2610.23653](https://arxiv.org/abs/2610.23653), 29 pages.
- **MFlow**: Rethinking the design space of Vanilla Pixel Diffusion Transformers (Mukhopadhyay, Mai, Shrivastava, Mahapatra; Maryland, Adobe Research). [arXiv 2610.23854](https://arxiv.org/abs/2610.23854), 33 pages, "submission under review".

**Teaser.** MFlow Fig 1: 9-image mosaic, no text, hairline gutters, 12-word caption. LILIE Fig 1: 30-image mosaic plus one small bar panel (2 series, black and pink, 4 printed values, own value bold, 12 labels at 0.6x) and 4 prompt excerpts at 0.46x; 73-word caption. That single panel is the one defensible number in a teaser.

**Method diagram.** LILIE Fig 3 is the model: panels "(a) Encoder Pretraining" and "(b) Generative Model Training", 11 labels of at most 3 words, rounded boxes in 3 pastel fills plus grey, real thumbnails, math limited to L_CE and L_2, text 0.6 to 0.7x. Counter-examples: LILIE Fig 4 (about 90 indexed labels) and MFlow Fig 7 (3 nested panels, about 40 labels, one equation). MFlow Fig 3 is split: four schematic gain curves (about 28 labels) and four spatial strips "x0 - [m] (x) eps = u" over real cat thumbnails with a grey colour bar.

**Results plots.** LILIE: 3 colours fixed paper-wide (Pixel black, LILIE green, Pixel+LILIE pink), one marker per series, 1 to 1.5 pt lines, boxless legend above the axes, faint horizontal grid, no top or right spine, x ticks only at tested values, "gFD ↓" as y label, text 0.5 to 0.75x. Every bar is labelled (0.36x in Fig 5). MFlow: matplotlib defaults, all spines, grid, a legend inside every panel (13 entries in Fig 11), a grey band 0.5 to 0.7 with the word "corner" inside, panel titles coloured by method. Fig 12 pairs bold letters (a) to (d) with bold titles.

**Qualitative grids.** LILIE Fig 8: 5 by 10, 1 pt gutters, no frames, row labels at 0.77x, the default row boxed in pink with bold text, two 0.73x strips stating what varies. MFlow Fig 16: rotated page, 10 methods with sizes at 0.66x, own method last. Crops: LILIE Fig 2 uses two orange frames with leader lines to enlarged patches labelled "L2 = 1000"; MFlow Fig 9 inverts the order, crop as main tile and whole image as a black-framed inset, headed "w/o" and "w/ ConvRefiner".

**Tables.** Both booktabs, no vertical rules, arrows in headers, bold best; MFlow underlines second. LILIE shades its own rows in the same pink as its plot series and the Fig 8 frame, greys the baseline row, and uses green ticks and orange crosses; its Table 1 caption runs about 190 words. MFlow Table 6 shades two groups in two greys.

**Rules.** Rule 14 changes: the shaded row is a pale tint of the SCA green, so plot colour, grid highlight and table shading agree (LILIE Tables 1 to 3, Fig 8). Rules 1, 7, 9, 11 and 13 stand: MFlow's text-free teaser, AReA's direct labels and LILIE's 11-label diagram back them, and LILIE's 3.6 pt bar values and MFlow's unreadable 3 pt Fig 4 are what rule 12 prevents.

## (f) Checklist

1. Teaser: images only, at most one 2-series panel, no prompts or metric rows?
2. Diagram: at most 12 labels of at most 3 words, single symbols, real thumbnail, at most two panels?
3. Plot: at most 4 series, direct labels, no legend box, no grid, top and right spines off?
4. At most 2 annotations, one bold ratio per panel, per-size ratios small and grey?
5. Colours: SCA green, one AReA hue, grey baselines, at most 3 saturated hues?
6. Text 6 to 8 pt, one family, bold only for "SCA (ours)" and panel letters?
7. Grid: methods named once, "SCA (ours)" last, 1 to 2 pt gutters, no frames?
8. Crops only where detail is the claim, one frame colour, "1:1 crop" once in the caption?
9. Caption: bold claim plus at most two sentences, under 50 words (35 in galleries)?
10. Table: booktabs, bold best, underlined second, own row in the SCA green tint?
