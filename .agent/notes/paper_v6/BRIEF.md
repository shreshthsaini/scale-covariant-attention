# SCA paper v6: full presentation revamp (lead brief, 2026-11-10)

User request (verbatim sense): the v5.1 paper's presentation is "still very subpar". Do a full revamp for version 6:
all text, tables, figures, images, the design of the method figures, the design of the qualitative results, the
examples. Target: best paper at ICLR, with that rigour, clarity and strength of claim.

v5.1 is tagged `paper-v5.1` (PDF paper/main.pdf, 55 pp). Rendered pages for reference:
/scratch/09032/saini_2/sca/paper_builds/v6_review_v51/ (p01..p10.png, app_*.png contact sheets).
The science and every number stay. No result may change. What changes is how it is told and shown.

## 1. Lead's critique of v5.1 (what must not survive)
- Page 1: the title describes instead of claiming. The abstract opens mid-thought in our own jargon. Two figures
  crowd the page; the hero is seven unequal tiles with one tiny inset; the speed plot has a broken axis, hollow and
  filled markers, overlapping labels and two unrelated panels.
- Page 2: the observation figure sits before the Introduction heading; a third of it is legend and white space; the
  central fact of the paper (97.5% of reach variance is the head) is a 1 cm bar. The contribution bullets are split
  across two pages by the method figure.
- Page 3: the method figure is 1.3 in tall with three crammed panels and 5 pt math. It cannot be read alone.
- Pages 3 to 4: notation before intuition; page 4 has no figure; four stacked displays; "One table serves every
  size" is six numbers in a row with no picture.
- Page 5: both algorithms are 5 pt monospace code. Nobody will read them.
- Page 6: Table 1 is narrow and small with "--" cells. The paired-gain plot is a cramped wrapfigure.
- Page 7: the qualitative grid has 0.6 in thumbnails; the crops are too small to show the detail we claim.
- Page 8: two tables squeezed side by side at footnote size with interval rows.
- Page 9: 60% empty. More than a page of budget is unused while every figure is starved of space.
- Appendix: 45 pages; qualitative pages half empty with figures at 0.6 width; 6 pt tables; mixed plot styles.

## 2. The story of v6 (one spine, told three times: abstract, introduction, section heads)
One sentence: how far an attention head of a diffusion transformer looks is fixed by the head, not by the denoising
step, so it can be measured once at the training resolution, where full attention is still correct, and reused to
generate at any resolution.
1. Question. Every training-free high-resolution method must decide how far each head looks at a size the model
   never saw. They set it by hand, per step, shared across heads, or search it per target size.
2. Observation (the scientific finding). Reach is a property of the head: head identity explains 97.5% of the
   variance of the near share in FLUX.1-dev (97.6% Krea, 99.1% FLUX.2-dev); the curve shared by all heads, which is
   all a common schedule can follow, explains 0.7%.
3. Method. SCA gives each head two views that keep the training geometry at any size (a native-size window and a
   native-size grid over the whole image) and mixes them with one number, the head's reach. Reach emulation fits
   that number at 1024^2 against the model's own full attention, in closed form (Proposition 1). One table per
   checkpoint serves every size and sampler.
4. Evidence. Replacing only the attention of an existing sampler: 49.2 s per 4096^2 image, 4.1x faster than AReA
   and better on all ten per-image metrics at three seeds and on HRGen-1K; statistically indistinguishable from
   HRDiT on ImageReward, HPSv2.1 and PickScore in 46% of its time; single pass 2.63x faster than AReA with the most
   crop detail. Per-head weights beat shared and per-layer ones in output error and in images.
5. Honest edges, said once and directly: patch FID is higher than HRDiT's and single-pass SCA's; inside the
   progressive sampler per-head weights win all ten per-image metrics by small margins while coarser weights have
   lower patch FID; the opening, windows and bridge are set by hand.
Strength of claim comes from precision. Every claim carries its number and, where it is a comparison, its interval.
"Matches" is never used for an interval that includes zero; say "indistinguishable on X (difference d [lo, hi])".

## 3. Target structure (ICLR 2027, main text 9 pages, the ninth at least 85% full; check the style's limit)
- p1: title, abstract (130 to 170 words: context, gap, "We show", "We introduce", headline numbers, URL), Figure 1.
- 1 Introduction, about 1.2 pp: question by the end of paragraph 1; the observation with Figure 2 beside it; "We
  introduce"; numbers; three "We" bullets kept together on one page.
- 2 Reach is a property of the head, about 0.6 pp.
- 3 Scale-Covariant Attention, about 2.2 pp: Figure 3 (method) at full width at the top of the section; each
  subsection starts with one plain sentence of intuition, then the definition; Figure 4 (does the measurement hold)
  beside 3.2; two boxed algorithms side by side that a reader can actually read (at least 7 pt, about 12 lines
  each, pseudocode with Input/Output and appendix pointers, not a code dump; the full code moves to the appendix).
- 4 Experiments, about 3.6 pp: setup run-ins (Data, Models and baselines, Metrics, Timing); Table 1; main results;
  Figure 5 qualitative at full width; ablations; generality; sizes and latency.
- 5 Related work, 6 Limitations and conclusion, reproducibility statement.
- Appendix: contents, each section opens with what it contains and which main item it expands, every main table
  expanded, method zoom, full pseudocode, prompts, sample pages that fill their pages. Shorter and denser than 45 pp
  if nothing is lost; never drop a result.
Sentence targets: median 17 to 21 words, none over 35. Bold run-in lead on every experiment paragraph, each a claim.

## 4. Figure and table set of the main text (fixed stems, all under paper/figures/v6/)
| # | stem | owner | content | size (text width 5.5 in) |
|---|---|---|---|---|
| 1 | hero.pdf | visuals | 4096^2 progressive SCA samples with 1:1 insets that prove native detail, plus one compact panel that carries the headline (quality against seconds per image, ours against AReA and HRDiT) | 5.5 x about 3.0 in |
| 2 | observation.pdf | methodfigs | the finding: two heads' attention at one query, the 1,368 heads by noise level matrix, and the 97.5 / 2.5 / 0.7 variance split drawn large | 5.5 x about 1.9 in, or a 0.48 wrap |
| 3 | method.pdf | methodfigs | (a) two native-size views per head, (b) reach emulation at 1024^2 with pi* = clip(B/A, 0, 1), (c) one table, any size and sampler. A real image flows through. 15 to 30 words | 5.5 x about 2.3 in |
| 4 | emulation.pdf | methodfigs | does the measurement hold: fitted weight against measured near share; relative output error per-head / per-layer / shared / hard switch at ratios 2 and 4; split-half and cross-resolution transfer | 5.5 x about 1.6 in |
| 5 | qual_main.pdf | visuals | AReA, UltraImage, HRDiT, progressive SCA on the same prompts and seeds, large crops with boxes | 5.5 x about 3.6 in |
| 6 | gains.pdf | tablesplots | paired gains over AReA (and over HRDiT) on the ten per-image metrics with intervals | 0.48 wrap or 5.5 x 1.5 in |
| 7 | scaling.pdf | tablesplots | seconds per image against image side, all methods timed, with the fitted exponents | 0.48 wrap |
| T1 to T3 | gen_main_tables.py | tablesplots | main comparison, ablations, generality | see Section 6 |
Appendix figures keep their current stems unless the owner announces a new one in its notes file.
Each main figure: build two candidates, render both at final size, critique them on the gold-standard rubric
(0/1/2 per item), keep the better, record the scores in your notes file.

## 5. Design system (identical everywhere; read paper/figures/src/orx_figstyle.py, scafig.py, sca_colors.tex first)
- One method hue: the SCA green already in use. Near view orange, far view blue (as in the observation figure).
  AReA purple, HRDiT and the other baselines in the existing neutral set. Four hues at most per figure. If you
  change a hex value, change it in the shared style files, never locally.
- Sans text inside figures, 7 to 9 pt at final size, never under 6 pt. At most 30 words inside a figure (60 for
  the emulation figure). Math in the math font. No titles inside figures that repeat the caption.
- Plots: no top or right spines, light horizontal grid, legend without frame, values printed on bars, direction
  arrows on axis labels, no broken axes, no error bands unless they are the point of the plot.
- Diagrams: vector (TikZ or matplotlib vector), 8 pt grid, rounded blocks with 0.8 to 1 pt border and a 15 to 25%
  tint fill, 0.6 pt orthogonal arrows, lettered panels "(a) Name", real images embedded.
- Tables: booktabs, 8 to 9 pt in the main text, direction arrows in heads, our rows shaded in a pale tint of the
  method green, best bold and second underlined per group, baselines that are context only in gray. Caption above,
  bold lead, setup, what the marks mean, then "Key result:" and one sentence.
- Captions: bold lead that states the takeaway, then at most three lines; panels addressed by letter; last
  sentence points to the appendix item that expands it.
- Qualitative: same prompt and seed per column, methods labelled, red or method-coloured 1 pt box on the full
  image and the 1:1 crop beside or below it at a size where the detail is visible in print (crop at least 1.0 in).
  Prompts and images are chosen by a stated deterministic rule (seeded draw or a metric rule), and the caption
  states the rule. Never choose by eye and never say so.

## 6. Hard rules (binding; violations are rejected at review)
- Words never used in paper, captions, file names: calibration / calibrate / calibrated; curated / uncurated /
  curation; "random" for how something was selected (say "seeded draw" or "chosen by a fixed seed"); em dashes;
  the section symbol; "license", "warrant", "plainly" used rhetorically. No AI-slop phrasing (load the
  `no-ai-slop` skill for prose).
- Headline is progressive SCA; single-pass SCA is the fast mode. The repeated-subject issue is not discussed: the
  one neutral limitations sentence about single-pass mode in v5.1 is kept verbatim and nothing is added.
- Priority claims stop at: first decomposition of reach variance into heads and steps; first per-head reach
  measured once at the training resolution and reused at any size. UltraImage noted stable head roles first.
- No number is typed by hand. Tables and headline macros come from paper/figures/src/gen_main_tables.py and its
  siblings, reading results/. Figures read results/ or the caches the existing scripts use. If prose needs a new
  number, add a generated macro.
- Speed ratios divide two times of the same kind (clean over clean).
- Do not rename existing \label names. Announce every new label, stem and macro in your notes file.
- Environment: TACC Vista, aarch64, you are on a CPU compute node (144 cores). Python only through
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python (PyMuPDF for rendering PDFs lives in
  .../uv-envs/csd-figs/bin/python; run scripts with -I). Never pip or conda. LaTeX with `tectonic` (on PATH).
  Every build, render and intermediate goes under /scratch/09032/saini_2/sca/paper_builds/v6_<role>_*/, never
  $HOME, never /tmp. Only finished small artifacts go into the repo. Never delete anything on scratch.
- No GPU work and no sbatch. If you believe new images or runs are needed, write the exact request in your notes
  file and continue with what exists (image pools: /scratch/09032/saini_2/sca/out/, evaluation roots under
  /scratch/09032/saini_2/sca/eval/, selection helpers paper/figures/src/imgsel.py and data195.py).
- Do not run git add, commit, push, stash or checkout: several workers share this working tree and the lead
  commits. Do not touch files outside your ownership (Section 7); if you need a change there, ask in your notes.
- Do not kill processes you did not start. Never use pkill or killall.
- Verify by looking: render every figure and every page you changed to PNG and read it before you report.

## 7. Ownership (one writer per file)
| role | owns | notes file |
|---|---|---|
| text | paper/main.tex (all prose, structure, captions, algorithm boxes) | .agent/notes/paper_v6/text.md |
| appendix | paper/appendix.tex | .agent/notes/paper_v6/appendix.md |
| methodfigs | figures 2, 3, 4, the appendix method zoom and analysis plots; their scripts under paper/figures/src (new files prefixed v6_) | .agent/notes/paper_v6/methodfigs.md |
| visuals | figures 1 and 5, every qualitative and sample figure of the appendix; scripts prefixed v6_ | .agent/notes/paper_v6/visuals.md |
| tablesplots | gen_main_tables.py and the other gen_*.py table generators, figures 6 and 7, appendix result plots; shared style files (announce changes) | .agent/notes/paper_v6/tablesplots.md |
Figure owners write, in their notes file, for each figure: the stem, the intended width, a proposed caption, and
the one-sentence claim it supports. The text and appendix owners integrate from those notes. Until a v6 figure
exists, main.tex and appendix.tex fall back to the v5.1 figure with \IfFileExists so the paper always builds.

## 8. Rigour list (do these, they are cheap and they are what a best-paper committee checks)
- Every sentence with a comparison names its n, its metric and its interval or says "point estimate".
- Multiple comparisons: tablesplots computes Holm-adjusted significance over the ten per-image metrics for the
  headline pairs (progressive SCA against AReA and against HRDiT; single-pass against AReA) from the archived
  per-image files, reports it in the appendix, and tells the text owner whether any headline wording must weaken.
- A claims-and-evidence audit: the text owner lists every quantitative claim of the abstract, introduction and
  conclusion with the table cell or macro it comes from (notes file); the lead checks it.
- Limitations stay concrete (hand-set parts, HRDiT-sampler dependence, patch FID, automatic metrics).

## 9. Reference bar
Load the `gold-standard-paper` skill (measured recipe of the two reference papers; its rendered pages are under
/scratch/09032/saini_2/Looped_Distill/csl/refs/gold/pages/) and, for plots, the `orx-figures` skill. Where that
skill says samples are "random", our wording rule above wins. Project history worth reading before redesigning:
.agent/notes/paper_v5_20261110.md (what v5 changed, layout facts, switch recipes), .agent/notes/paper_v4_20261108.md,
HANDOVER.md Sections 41 and 42 (use grep and sed ranges; never cat a file over 200 lines).

## 10. Deliverable of each worker
Finished files in the repo paths you own, a notes file with what changed, the rubric scores, open requests to
other owners, and anything you could not verify. A short final report: what is done, what is not, where to look.
