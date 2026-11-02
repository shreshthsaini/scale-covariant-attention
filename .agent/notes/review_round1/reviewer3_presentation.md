# Reviewer 3: presentation and figures, round 1

Source: paper/main.pdf (53 pages, 10 main) at 110 dpi, Figure 3 at 220 dpi, paper/main.tex. Bar: gold-standard-paper skill (LILIE, MFlow) with AReA as nearest competitor. Page numbers are PDF pages. No repo file edited.

## 1. Rubric (0 below, 1 meets, 2 above)

| Item | Score | Evidence |
|---|---|---|
| Abstract 130 to 200 words, number, URL | 1 | 194 words, 12/15, 2.63x, 57 to 26%, URL; "under half the time" lacks its numbers (p1). |
| Hero with bold-lead caption walking panels | 1 | Full width, "Top:/Bottom:"; 55 words; top row repeats Figure T.1 (p42). |
| Footnotes | 1 | None. |
| Question in paragraph 1, figure beside paragraph 2 | 2 | "We ask whether..." closes paragraph 1; Figure 2 wraps paragraph 2 (p2). |
| "We introduce", numbers, "We" bullets | 1 | Name arrives in paragraph 4, after "We answer by emulation"; three bullets (p2). |
| Median under 22 words, none over 35 | 0 | Median 22, mean 27; ten prose sentences over 35 (Section 3). |
| Method diagram | 1 | Figure 3 ticks the checklist yet shows no operator (Section 2). |
| Equations with "where" lines | 1 | Eq. 1, 5, 7 yes; Eq. 3, 4, 6 no (p3 to p4). |
| Boxed algorithms side by side | 2 | p4: I/O comments, equation and appendix pointers, colours; type about 6 pt. |
| Setup run-ins with counts and compute | 1 | Four run-ins, 123.3 GH200-hours (p6 to p7); no hyperparameter-selection run-in. |
| Main table design | 1 | booktabs, shading, bold, underline, "Key result:" (p6); header is a row of bare arrows over split names. |
| Finding paragraphs bold, before and after | 2 | "0.92 to 1.25, +0.33 [0.25, 0.41]" (p8). |
| Ablations side by side, default shaded | 1 | Tables 3 and 4 (p9); Table 3 about 7 pt, significance code differs from Table 2. |
| Plots: palette, labels, nothing under 6 pt | 0 | SCA green and AReA purple consistent; subscripts in Figures 3b and 5c and the Table 3 body under 6 pt. |
| Qualitative grid | 1 | Figure 6 (p7); the source window is nearly invisible. |
| Confounds and limitations | 2 | "curation favours SCA", "intervals ignore multiple comparisons", tuned-on-eval admitted (p10). |
| Reproducibility statement | 2 | Every detail mapped (p11). |
| Appendix | 2 | Contents, zoom D.1, config F.1, "adds all metrics", galleries with provenance. |
| Float placement, wraps, page load | 1 | All floats on first-reference page; three wraps; page 7 stretched, pages 17 and 21 half blank. |
| Caption placement and leads | 1 | Correct; several figure captions over lead plus two sentences. |
| Colours hero to tables | 1 | Shading matches SCA green; hero insets are amber, which Figures 3 and 5 reserve for "local". |
| No em dash, no section symbol | 1 | grep clean. |

Mean 1.18, two zeros. Not yet shippable under the house rule.

## 2. Figures and tables

**Figure 1 (p1).** The six-method strip earns the page; the top row duplicates Figure T.1 at thumbnail size. Drop it, run the strip on two prompts (church plus one person prompt, so duplication is shown honestly on page 1), recolour insets grey or SCA green.

**Figure 2 (p2).** Earns its place. Bold "(a)" and "(b)" at 8 pt.

**Figure 3 (p3). Why it is naive.** It shows artefacts, not the mechanism. Panel (a) is three photographs of one face with overlays: a pixelated face under blue dots (are the dots the kept keys, or is the pixelation the operator?), four nested orange squares of which only "128" is labelled, and dots called "bridge". Nothing names $O^{\mathrm{glob}}$, $O^{\mathrm{loc}}$ or $\pi$; there is no mixing node, so Eq. 5 has no picture in the main text (Figure D.1c, appendix, has the plus node the main figure lacks). The per-head weight over 28 steps, the paper's idea, is a 2 px blue-orange split inside a strip that reads as a border; the green "$\pi_h(r)$" box encloses four cells and shows nothing varying. Two faint grey wedges are the only link between strip and photos. Panel (b), calibration, shows two grid icons and two heatmap strips with no head-axis label, and $\sigma$ runs 1 to 0 while Figures 5c and B.1 run 0 to 1. Calibration is a least-squares fit against full attention; the panel shows only the output table, so the reader learns nothing about how the weight is obtained. LILIE Figure 4 draws tokens entering named blocks, MFlow Figure 3 draws picture equations, AReA Figure 4 draws Phase I and II as a pipeline; this figure draws none of the algebra.

**Best-paper version** (full width, 170 to 190 pt, two lettered panels, 30 words). (a) Calibrate at $1024^2$: one real $64\times64$ token grid; for one query, three small key maps, "full", "global" (block centres, blue), "local" (32-window, amber); arrows into a small box "$\pi^\star=\mathrm{clip}(B/A)$" with a tiny parabola $J(\pi)$; out comes the table as a heatmap with labelled head and $\sigma$ axes and a colourbar. (b) Sample at $4096^2$: a step axis with $\sigma$ values and the gate at step 8 as a dashed vertical; beneath it six key maps on the real $256\times256$ grid for one tile (opening, $w=128,112,96,80,64$) with window to scale, bridge dots, and $r$ printed under each; at right the operator as boxes $O^{\mathrm{glob}}$ (blue) and $O^{\mathrm{loc}}$ (amber) into a circled plus weighted $1-\pi_h$ and $\pi_h$, output $O_i$ (green); above the axis one head's $\pi_h(\sigma;r)$ as a curve, zero before the gate, replacing the colour strip. Four hues. Caption: lead, two sentences, pointer to D.1.

**Figure 4 (p5).** (a) is cluttered: categorical gate axis, an arrow between a grey and a green dot, a floating "7/10 metrics worse", a purple dashed AReA line; it conflates gate and schedule. (c) frames whole images in thick red while red elsewhere marks a crop window. Fold (a) into an inset of (b) or move it to Appendix P; mark each extra copy in (c) with a small red circle; caption to lead plus two sentences.

**Figure 5 (p6).** Earns its place. (b) thumbnails are about 45 pt with the mass a faint speck; subscripts under 6 pt. Enlarge (b) to the height of (a), overlay the mass as a heatmap, mark $\sigma_g$ on (c).

**Figure 6 (p7).** Good structure, per-crop MUSIQ$_c$. The full-attention column spends three cells on texture already shown in Figure 1; swap for UltraImage or DyPE. Source window to 1 pt.

**Figure 7 (p8).** Meets; add $n=195$. **Figure 8 (p9), D.2 (p21).** Above the bar; D.2 leaves half a page blank.

**Tables 1 (p6) and 2 (p8).** Strong content. Header is bare arrows above split names ("Image / Reward"); use one line of short names with arrows and a spanning "Time (s)$\downarrow$". Table 2 marks significance with $^+ \circ \ast$, Tables 3 and 4 with blue and orange; pick one. Table 1 caption sentence 2 is 54 words of semicolons.

**Table 3 (p9).** About 7 pt, an "n/a" cell, "Err." column belongs in K.5. **Table 4.** Fine; say "no intervals" in a word.

**Appendix.** B.1 and D.1 are the strongest. B.3 and B.4 leave half of page 17 blank. M.1 (p36) uses one grey diamond for both HRDiT and UltraImage.

## 3. Writing

**Abstract.** Good order and voice ("appetite for context"); two sentences of 33 and 35 words; give "under half the time" its numbers (50.7 against 110.4 s).

**Introduction.** 691 words, concrete hook (256 times, five Moai). Paragraph 3 answers before paragraph 4 introduces; swap them. Bullets correct. No slop words, binary contrasts or kickers found.

**Prose sentences over 35 words.** p2 lines 57 to 61 (47, citation chain); p2 64 to 67 (37); p3 125 to 127 (37); p3 Proposition 2 (39); p3 145 to 147 (51 across Eq. 4); p4 199 to 201 (43); p7 355 to 358 (43); p10 501 to 505 (83, citation chain); p10 511 to 513 (44); Table 4 caption sentence 1 (36).

**Captions not lead plus two sentences.** Figures 1, 4, 5 (70 words), A.1 (five sentences), D.1 (six, 130 words), B.1. Tables 1 to 4 run three sentences, allowed for tables.

**Numbers without comparison or scale.** "repeats subjects in 6% of prompts" (p4, against what); "standard deviation of 0.24 to 0.26" (p6, say the range is 0 to 1); "ARNIQA falls by 0.007" and "0.014" (p10) while Table K.1 reports ARNIQA $\times100$; "aesthetic score by 0.19" (p4, no scale); Table 4 deltas without intervals.

## 4. Top ten fixes

1. Redraw Figure 3 to the specification above: operator boxes and plus node, calibration fit shown, step axis with $\sigma$ and $r$, a $\pi_h$ curve in place of the colour strip, $\sigma$ direction matching Figures 5 and B.1.
2. Cut the ten sentences over 35 words; split both citation chains (p2, p10).
3. One significance code across Tables 2 to 4; replace the bare-arrow headers of Tables 1 and 2.
4. Lift all figure and table text to 6 pt or more (Figure 3b ticks, Figure 5 subscripts, Table 3 body, Algorithms 1 and 2).
5. Figure 1: remove the top row, add a person prompt to the strip, recolour insets.
6. Figure 4: simplify or relocate (a); consistent red circles in (c).
7. Captions of Figures 1, 4, 5, A.1, D.1 to lead plus two sentences; conventions once at the top of Section 4.
8. Put ARNIQA and aesthetic deltas on the table scales; add the comparators listed in Section 3 and 50.7 vs 110.4 s to the abstract.
9. Fix layout holes: page 7 stretched paragraphs (let Figure 6 float to the bottom or drop enlargethispage), pages 17 and 21 half blank.
10. Figure 6: swap the full-attention column for a competitive baseline and thicken the window; Figure 5b: enlarge and overlay the mass as a heatmap.
