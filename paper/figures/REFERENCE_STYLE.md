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
14. Tables use booktabs, with a vertical rule only between model groups. Best is bold, second is underlined, and our row is shaded.
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
