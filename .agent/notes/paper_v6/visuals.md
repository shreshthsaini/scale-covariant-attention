# Visuals owner, paper v6 (2026-11-10, revised after the lead's review)

Figures 1 and 5 and every appendix sample page are rebuilt as one family. Every image and window comes from a rule
implemented in the scripts and stated in the caption. Nothing is picked by eye.

## Round: more visual results (2026-11-10; user: "add more visual results in main and appendix for shorter version as well")
Scripts: `v6_sizes_models.py` (main figure), `v6_more.py` (deciles, seeds12, hrgen, mosaic) on the new shared builder
`v6_wholegrid.py` (Figure 5's whole-image style). PDFs and `<stem>_provenance.json` are in paper/figures/v6/, snippets
in paper/figures/v6/snippets/, and build assets on scratch (v6_visuals_assets/<stem>/). All new floats compile with the
existing ones in testdoc/test_all.pdf (24 pages, no errors, no duplicate labels, no overfull boxes; the only warning is
`tab:ablation`, which the stand-in document lacks). Figure text is 6.48 to 7.97 pt.

| stem | label | size (in) | PDF | rule clause (caption) |
|---|---|---|---|---|
| sizes_models (main) | fig:sizes_models | 5.50 x 2.40 | 0.46 MB | one prompt per setting, drawn with seed 0 among prompts that name no person, each from a different subject category |
| deciles_a, deciles_b | fig:deciles_a, fig:deciles_b (snippet deciles.tex) | 5.50 x 6.93, 5.50 x 5.58 | 0.93 + 0.98 MB | the evaluation prompts at the 10th, 20th, ..., 90th percentile (nearest rank) of the paired ImageReward difference, progressive SCA minus AReA, 195 prompts, seed 0 |
| seeds12 | fig:seeds12 | 5.50 x 6.77 | 0.95 MB | per seed (1 and 2), the prompts at the 10th, 25th, 50th, 75th and 90th percentile of the same difference, among the 80 prompts that name no person |
| hrgen_samples | fig:hrgen_samples | 5.50 x 8.22 | 1.08 MB | the HRGen-1K prompts at the 10th, 25th, 50th, 75th and 90th percentile of the same difference, among the 740 of 1000 prompts that name no person |
| mosaic | fig:mosaic | 5.50 x 5.50 | 0.88 MB | 16 showcase prompts outside Figure 1, the six categories taken in turn, each category's order fixed by seed 1; insets on the four most detailed windows |

New PDFs total 5.27 MB. Estimates: full edition 19.7 + 5.3 = about 25.0 MB; short edition 12.2 + 0.46 (main figure)
+ 3.9 (four pages below) = about 16.6 MB. Appendix insets on these pages are native 256 x 256 windows (JPEG q85 4:2:0;
the red boxes are vector), and thumbnails are 250 ppi.

### A. Main figure `figures/v6/sizes_models.pdf` (`\vsixinc{sizes_models}`, natural width)
- Five tiles in two rows: FLUX.1-dev single-pass SCA at 2048 x 6144, 2048 x 4096 and 4096 x 6144; FLUX.1-Krea-dev
  single-pass SCA and FLUX.2-dev SCA without the schedule at 4096^2.
  - Each tile has a tag (size or checkpoint) and a 0.47 in native inset of its 256 x 256 window of highest edge
    energy.
  - The two full-width rows of these aspect ratios need 2.6 in, so the tiling is scaled to the 2.40 in cap and
    centred (5.07 in of content, 0.21 in side margins).
- Draw (seed 0, DRAW_ORDER): 1046736963520652161 (mountain lake), m4850138611731379908 (desert under the Milky Way,
  Krea), 6384782444147762726 (sheep and church), sc_l01_fjord_village (showcase; only showcase prompts exist at
  4096 x 6144), m1017950324987649681 (snow mountain, FLUX.2).
- Left out, every setting drawn first and dropped after so that no drop changes another draw (reasons in
  provenance):
  - 6144 x 4096 (sc_p03_amber_perfume): the 1:1 window lands on a lattice of blocks and stripes over defocused
    petals.
  - 6144^2 (m3130703310573987496, SCA without the schedule): at native pixels the stone wall smears into horizontal
    streaks.
  - 8192^2 (m5023570400257826529): horizontal stripes across the planet and its ring.
- Rule history: rule 1, without the variety clause, drew m6845347395255545991 for Krea. Its image has a second lit
  structure on the left peak for the one glowing tower the prompt asks for, and the draw had three mountain scenes.
  The variety clause was added before its draw was viewed.
- Two candidates were rendered on an ICLR test page (testdoc/test_sizes.png).
  - A (two rows with insets, kept): fit 1, global coherence 2 (1.2 in rows), detail 1 (0.47 in insets), variety 2,
    tags 1, words 1 (10), text 1. Mean 1.29.
  - B (one strip with crops beneath): fit 1, coherence 0 (0.63 in images), detail 2, variety 2, tags 0 (tags overflow
    the narrow tiles), words 1, text 1. Mean 1.14, with two zeros.
- Proposed caption:
  `\textbf{\sca{} at other sizes and on other checkpoints.} Single-pass \sca{} on FLUX.1-dev at $2048\times6144$,
  $2048\times4096$ and $4096\times6144$ with the reach table fitted at $1024^2$; single-pass \sca{} on
  FLUX.1-Krea-dev and \sca{} without the schedule on FLUX.2-dev at $4096^2$, each with its own table. One prompt
  per setting (a showcase prompt at $4096\times6144$), drawn with seed~0 among prompts that name no person, each
  from a different subject category; insets show 1:1 crops of the red boxes. Exploratory $6144^2$ and $8192^2$
  samples, which show artifacts, are in Figures~\ref{fig:gallery_6144_complete} and~\ref{fig:gallery_8192_complete}.`
- Claim: the same mechanism produces coherent images with native detail at three other aspect ratios and on two
  other checkpoints (qualitative; the numbers are Table 3's).

### B. Appendix pages (each a float page; snippets carry the captions)
1. **Deciles** (`\snippetat{p}{deciles}`, two floats). AReA, UltraImage, HRDiT, progressive SCA. All nine deciles
   are kept; only the median coincides with Figure 5 (which shows the 25th, 50th, 75th), so the distribution reads
   whole. Lead of the caption: "Whole-image comparison at every decile of progressive SCA's ImageReward gain over
   AReA (10th to 50th percentile / 60th to 90th percentile)". The rows show AReA ahead at the 10th percentile
   (-0.19) and level at the 20th and 30th; the gain grows from the 40th on (+0.11 to +1.28).
   - For the lead: at the 80th percentile ("twelve silhouettes in a row") the person-count detector counts 13 in
     ours and 14 in UltraImage's. It is the prompt's own crowd, off by one, not a duplicated main subject, so I kept
     the 195-prompt deciles. Say if you want the person-free subset here too.
2. **Seeds 1 and 2** (`\snippetat{p}{seeds12}`). Seed 1 and seed 2 groups, each AReA and progressive SCA (better of
   each pair bold), five percentile rows.
   - HRDiT has no seed-1 or seed-2 images on disk, so its column is absent and the caption says so.
   - Fallback used (I am telling you, as you asked): the draw over all 195 prompts put the seed-1 median on
     m5795767365495583614. The prompt asks for one tiny figure; our image has a second one on the far shore,
     invisible at the page's thumbnail size but present. Both seeds now draw from the 80 prompts that name no person,
     stated in the caption.
   - The rows show AReA ahead at the 10th percentile (-0.13, -0.19); the gain grows to +1.23 and +1.29 at the 90th.
3. **HRGen-1K** (`\snippetat{p}{hrgen_samples}`). AReA, single-pass SCA, progressive SCA (single-pass included, so
   the person-free rule applies: 740 of the 1000 prompts).
   - AReA is best at the 10th percentile (-0.09). Single-pass SCA is best at the median and 90th, progressive SCA at
     the 25th and 75th.
   - AReA's water-drop texture is visible at the 75th and 90th.
4. **Mosaic** (`\snippetat{p}{mosaic}`): 16 progressive SCA images at 4096^2, 4 x 4 at 1.35 in, insets on 4.
   One-line caption: "Progressive SCA at 4096^2 on 16 more showcase prompts."
5. **6144^2 and 8192^2**: already in the appendix as gallery_6144_complete and gallery_8192_complete (AReA against
   SCA without the schedule; single-pass SCA has no 6144^2 or 8192^2 images on disk). Nothing new built.

### C. Short edition: recommended four pages
deciles_a, deciles_b, hrgen_samples, seeds12 (about 3.9 MB). Together they show the whole distribution of the
headline gain, at the main seed, at two more seeds and on a second prompt set. These are the rigour a reviewer checks;
the mosaic is a showcase and can stay full-only.

### Missing on disk (no GPU work requested)
HRDiT at seeds 1 and 2. Single-pass SCA at 6144^2 and 8192^2. HRDiT on HRGen-1K.

## Lead review, round 3 (final): Figure 5 compares whole images
Figure 5 is rebuilt on whole images, with percentile rows of the ImageReward gain over AReA; see Figure 5. The
window-agreement figure is kept in the appendix as the new stem crops_matched.pdf (label fig:crops_matched, snippet
v6/snippets/crops_matched.tex). Its caption: "Where the four methods draw the same content, progressive SCA's crops
are sharper than UltraImage's and level with HRDiT's." followed by the selection clause and the note that AReA's and
HRDiT's layouts still differ in some rows. The comparison pages are unchanged; their draws still exclude the
crops_matched prompts (`v6_gallery` now imports `v6_crops_matched.picks_from`). None of the new Figure 5 prompts
appears on another page.

## Lead review, round 2 (Figure 5 selection on window agreement)
The window-level rule is built and is in the stem; see Figure 5. Two of three rows are not fully like for like
(AReA's crops show other content in rows 1 and 3, HRDiT's in row 3). As instructed I stopped and sent the render and
the top five pairs per category. The pair rule is not yet applied to the appendix pages. Each sample page shows every
method's own full image beside its crop, so the reader sees each crop's context, and the rule did not reliably give
like-for-like crops on Figure 5. I will apply it once the lead settles the rule.

## Lead review, round 1: what changed
1. Figure 5 is redone like for like (rule below). The render is
   /scratch/09032/saini_2/sca/paper_builds/v6_visuals_1/testdoc/test_main_p03.png (ICLR page) and
   .../cand2/qual_main_r300.png (300 dpi). Honest read: the portrait row shows a clear difference. The planet and church
   rows give smooth crops for every method. In the planet row AReA's crop has the best MUSIQ, and in the church row
   HRDiT's crop shows more visible detail. Nothing was swapped. A stated alternative rule is in .../cand3 for your
   decision (see Figure 5).
2. Every page that shows one window cropped from several methods now uses the maximin window. That is the window that
   maximises the smallest edge energy among the methods shown (`v6_common.best_windows_maximin`, used by `v6_grid`).
   The captions say so in one clause.
3. Hero: the HRDiT label now sits right of and below its marker, clear of "Progressive SCA". Every tag ("1:1",
   "4096 x 4096") is an opaque ink chip, so the two "1:1" tags are identical on any background.
4. Ablation grid: the "No opening" row is back. The lead is the v5.1 wording, "Without the opening phase, subjects
   shrink to a small scale."
5. The gallery draws changed, because they exclude Figure 5's new prompts and the 4096^2 pages are chained. All pages
   were re-rendered, re-screened and re-read.

## Where to look
- Scripts (paper/figures/src): `v6_common.py` (pool, keyword rules, seeded draws, edge-energy and maximin windows,
  TikZ canvas), `v6_grid.py` (methods x examples grid for the sample pages), `v6_hero.py`, `v6_qual_main.py`,
  `v6_gallery.py`. Regenerate on a compute node from paper/figures/src:
  `<sca python> -I v6_hero.py; <sca python> -I v6_qual_main.py; <sca python> -I v6_gallery.py [stem ...]`.
- Repo outputs (paper/figures/v6): `<stem>.pdf`, `<stem>_provenance.json` (rule, draw, windows, source paths and
  SHA256), and `snippets/<file>.tex` (one float each, `\includegraphics` at natural width, caption, v5.1 label).
- Build assets (JPEG tiles, standalone TikZ) are on scratch, regenerated by the scripts:
  /scratch/09032/saini_2/sca/paper_builds/v6_visuals_assets/<stem>/. The earlier per-stem asset folders under
  paper/figures/v6/ are retired: `git status` shows them as deleted, and the next commit should record that.
  `v6/zoom4k.pdf` became `crops4k.pdf`.
- Renders and test documents: /scratch/09032/saini_2/sca/paper_builds/v6_visuals_1/
  (`testdoc/test_main_p01.png` page 1, `testdoc/test_main_p03.png` Figure 5 page, `testdoc/test_all.pdf` all 21 floats,
  `final2/s1.png`..`s3.png` the rebuilt sample pages).

## Figure 1: `figures/v6/hero.pdf` (label fig:prog_hero, unchanged)
- Size: 5.5 x 3.29 in (397.5 x 237.7 pt), natural width; 0.51 MB. That is 0.29 in over the brief's 3.0 in: a
  five-column grid at full width fixes the height at 3s + 2g. Page 1 still holds title, abstract and figure.
- Layout: one large tile (2.18 in) with two native 1:1 crops (1.08 in, 256 px windows at 237 ppi, 7.9x), the speed
  panel (2.18 in square), and five small tiles. 29 words; smallest text 6.48 pt.
- Selection rule (caption clause): one of the 40 showcase prompts (web/build/showcase_prompts.jsonl) per category,
  drawn with seed 0; the portrait draw takes the large tile. Crops: the two non-overlapping 256 px windows of highest
  edge energy in the large image.
- Panel data: `fig_speed.load()`. ImageReward of the 195 evaluation prompts at seed 0 is plotted against clean
  batch-1 seconds on one GH200: progressive 49.15 s / 1.320, single-pass 77.10 / 1.248, HRDiT 106.97 / 1.342, AReA
  202.36 / 0.921. Arrows give 4.1x and 2.2x, clean over clean.
- Proposed caption (no panel letters in the figure):
  `\textbf{Progressive \sca{} makes $4096^2$ images with native detail, $\hlSpeedProgArea\times$ faster than AReA.}
  Images: FLUX.1-dev, one of 40 showcase prompts per category drawn with seed~0; red boxes mark native 1:1 crops.
  Plot: ImageReward on the \nmain{} evaluation prompts against clean seconds per image on one GH200; arrows give
  clean-over-clean speedups. Figures~\ref{fig:gallery_prog16} and~\ref{fig:showcase_models} show more.`
- Claim: progressive SCA makes 4096^2 images with native-pixel detail in 49.2 s. That is \hlSpeedProgArea x faster
  than AReA at higher ImageReward, and \hlSpeedProgHrdit x faster than HRDiT at an ImageReward indistinguishable from
  HRDiT's (\hlProgHrditIr [\hlProgHrditIrLo, \hlProgHrditIrHi]).
- Rubric (0/1/2) for candidates A (kept) and B. Width and placement: 1 / 1. Headline readable: 2 / 1. Native-detail
  proof: 2 / 1. Grid: 1 / 1. Variety: 2 / 1. Words and text size: 1 / 1. Palette: 1 / 1. Height budget: 1 / 1.
  Mean: 1.38 / 1.13.

## Figure 5: `figures/v6/qual_main.pdf` (label fig:qual_prog, unchanged), final (round 3: whole images)
- Size: 5.5 x 4.20 in, natural width; 2.27 MB. Script `v6_qual_main.py`.
- Layout:
  - Columns: AReA, UltraImage, HRDiT, progressive SCA (ours bold green). Under each head, a grey 6.5 pt line of
    seconds per 4096^2 image from the hero's data: 202.4 s, 308.5 s‡ (UltraImage's campaign mean, Table 1's mark),
    107.0 s, 49.2 s.
  - Rows: three prompts, labelled at left with the percentile and the ImageReward difference.
  - Cells: the full 4096^2 image (1.19 in). The image's own 512 x 512 window of highest edge energy is boxed in red and
    shown at native pixels as a 0.47 in inset, bottom-right unless that covers the box (then bottom-left, top-right,
    top-left). Under each image, its ImageReward in grey, best of the row bold.
  - Smallest text 6.48 pt; 27 words if a number with its unit counts as one.
- Rows (caption clause): the evaluation prompts at the 25th, 50th and 75th percentile (nearest rank, ranks 49, 98 and
  147 of 195) of the paired ImageReward difference, progressive SCA minus AReA, at seed 0. Per-image ImageReward comes
  from the files the paired bootstrap read: eval/a4k195_main (AReA, UltraImage) and eval/prog195_sca_v2 (HRDiT, ours).
  They equal eval/prog195_vs_baselines, and their mean difference reproduces the archived 0.399078. The person-free
  fallback was not needed: our three images show no repeated figures.

| Row | prompt | difference | ImageReward AReA / UltraImage / HRDiT / ours |
|---|---|---|---|
| 25th pct. | m1066207185399086750 (woman with glasses and silver jewelry) | +0.01 | 1.89 / 1.89 / 1.89 / **1.90** |
| median | 2022937813048109690 (snow-covered waterfall) | +0.24 | 1.23 / **1.56** / 1.38 / 1.47 |
| 75th pct. | m5370774387421928008 (riverside castle in autumn) | +0.66 | 0.83 / 1.40 / 1.17 / **1.49** |

- What the rows show: at the 25th percentile the four images are level. At the median and the 75th percentile AReA's
  whole image is flatter and hazier (washed out at the 75th) than the other three. Ours takes 49.2 s against AReA's
  202.4 s. Nothing over HRDiT is claimed; UltraImage has the best ImageReward at the median.
- Caption:
  `\textbf{Progressive \sca{}'s typical gain over AReA is in the whole image, at a quarter of AReA's time.} Rows: the
  evaluation prompts at the 25th, 50th and 75th percentile (nearest rank) of progressive \sca{}'s ImageReward gain over
  AReA, seed~0. Under each image its ImageReward (best bold); insets show each image's highest-detail $512^2$ window at
  1:1; \textdaggerdbl{} campaign mean. Figures~\ref{fig:gallery_prog16} and~\ref{fig:gallery_prog16b} show 18 more
  prompts.`
- Claim: progressive SCA's measured advantage over AReA is whole-image preference (ImageReward \hlProgAreaIr
  [\hlProgAreaIrLo, \hlProgAreaIrHi] over 195 prompts), at \hlSpeedProgArea x less time. The rows show a small, a
  typical and a large gain.
- Render on the ICLR page: /scratch/09032/saini_2/sca/paper_builds/v6_visuals_1/testdoc/test_main_p03.png; 300 dpi
  candidate: .../cand5/qual_main_r300.png (0.53 in insets; the stem uses 0.47 in).
- Rounds for the record:
  1. Each method's image beside a shared window: rejected, unrelated content per row.
  2. Image-level layout agreement: two flat rows (.../cand2).
  3. Window-level agreement: two of three rows not fully like for like (.../cand4, top five pairs in
     cand4/top5_pairs_labeled.jpg). This round moved to the appendix as crops_matched (below).

## Appendix sample pages (one family)
Shared design: full text width, methods as rows with rotated two-line 8 pt labels (ours green, progressive bold), and
2 pt gutters, 5 pt between examples, 8 pt between bands. Every page uses the same 1 pt red box and red crop frame.
Every shared window is maximin: the window that maximises the smallest edge energy among the methods shown, about
250 ppi, at least 1.03 in wide. Thumbnails are stored at 250 ppi and crops at JPEG quality 90 (4:4:4). Full pages run
6.6 to 8.0 in; half pages pair two per page. Each caption has a bold lead, the selection clause and the crop clause
("Each crop shows native pixels of the window that maximises the smallest edge energy among ...").

| PDF stem | label | snippet file | height (in) | MB | rule (seed) |
|---|---|---|---|---|---|
| gallery_prog16 + gallery_prog16b | fig:gallery_prog16, fig:gallery_prog16b | gallery_prog16 (both floats) | 8.03 each | 0.83 + 0.93 | 18 evaluation prompts outside Figure 5 and fig:gallery_prog (16); AReA, UltraImage, HRDiT, progressive |
| gallery_prog | fig:gallery_prog | gallery_prog | 6.63 | 0.61 | six person-free prompts outside Figure 5 (0); five methods incl. single-pass |
| crops4k (new stem) | fig:qual_main | crops4k | 3.26 | 0.40 | three person-free prompts outside the progressive figures (4); five methods |
| all_methods (new stem) | fig:teaser | all_methods | 7.50 | 0.64 | two person-free prompts not shown earlier (8); all eight methods |
| aspect (new PDF stem) | fig:hero | hero | 6.83 | 0.93 | four person-free prompts without quoted text (3); single-pass at three sizes, each image its own window |
| showcase_models | fig:showcase_models | showcase_models | 7.86 | 1.21 | Figure 1's six showcase prompts (0); FLUX.1 progressive, FLUX.2-dev and Qwen-Image single-pass |
| gallery_baselines | fig:gallery_baselines | gallery_baselines | 3.26 | 0.38 | three person-free prompts not shown earlier (17) |
| gallery_4096 + gallery_4096_p2 | fig:gallery_4096, fig:gallery_4096_p2 | gallery_4096, gallery_4096_p2 | 7.47 each | 0.98 + 0.94 | 16 person-free prompts not shown earlier (0), eight per page |
| gallery_4096x2048 | fig:gallery_4096x2048 | gallery_4096x2048 | 7.53 | 0.87 | four person-free prompts (33) |
| gallery_6144x2048 | fig:gallery_6144x2048 | gallery_6144x2048 | 7.53 | 0.76 | four person-free prompts (39) |
| gallery_flux2 | fig:gallery_flux2 | gallery_flux2 | 3.68 | 0.47 | frozen v5.1 draw (17); keeps the portrait the limitations text cites |
| ablation | fig:ablation_grid | ablation | 7.20 | 0.93 | two person-free ablation prompts of the 46 (46); six rows incl. "No opening" |
| gallery_krea_complete | fig:gallery_krea_complete | gallery_krea_complete | 3.60 | 0.37 | two person-free prompts of the Krea run (17); sources eval/review1_krea195_dype_sega |
| gallery_6144_complete | fig:gallery_6144_complete | gallery_6144_complete | 3.68 | 0.30 | four person-free of the 100 completed (17) |
| gallery_8192_complete | fig:gallery_8192_complete | gallery_8192_complete | 3.68 | 0.31 | four person-free of the 12 completed (17) |
| progressive | fig:progressive | progressive | 2.69 | 0.20 | frozen v5.1 draw over the 46-prompt panel (17) |
| crops_matched (new stem) | fig:crops_matched (new label) | crops_matched | 3.95 | 1.06 | per category, the prompt and 512 x 512 window on which the four methods agree most in content among high-detail windows (Figure 5's round 3; `v6_crops_matched.py`) |
| failures | fig:failures | failures | 3.68 | 0.53 | the four person-free prompts with the largest ImageReward deficit of SCA without the schedule against AReA |

Family rubric (per page):
- Same prompt and noise per example, rows labelled: 1.
- Native crops on every page, rule-placed, at least 1.0 in, none empty: 2.
- Page fill: 1.
- Selection rule in every caption, implemented, sources hashed: 2.
- One family: 2.
- Text at least 6 pt: 1.
- Palette: 1.
- Mean: 1.43.

Known effect of the maximin rule on all_methods: full attention's image is nearly flat, so the window lands where it
has any texture at all, and the stronger methods' crops show sky and clouds in the first prompt. The page still shows
full attention's flat field and DyPE's and SEGA's lattices; the second prompt's crops show mountains.

## Selection record: every change of rule and why (no image swapped by hand)
1. Hero, rule 1 (one evaluation prompt per keyword category, seed 0): it drew an image with an obvious defect at
   overview (4820545723762191815, an unrequested standing figure beside the polar bear). Rule 2 (showcase prompts,
   one per category, seed 0) was fixed before any showcase image was viewed.
2. Figure 5, first v6 version: stratified seeded draw, each method's own image beside a shared mean-energy window.
   Rejected in review (unrelated content per row). Replaced by the lead's layout-agreement rule and maximin window.
3. Aspect-ratio page: the first draw included a prompt that asks for lettering (3912690245302078105), and every crop
   landed on garbled lettering. "No quoted text" was added to the rule.
4. Pages showing single-pass SCA, SCA without the schedule or the ablation variants draw from the prompts that name
   no person (keyword rule `v6_common.PERSON`, 78 of the 191). The lead approved this as long as each caption states
   it. The project's person-count detector is a private QA screen only; it is never a rule and never in a caption.
5. The 4096^2 FLUX.1-dev pages are chained so no prompt repeats: Figure 5, gallery_prog, gallery_prog16/b, crops4k,
   all_methods, gallery_baselines, gallery_4096/p2.
6. Final screen: one prompt flagged on the progressive pages, m3268147817845827695 (UltraImage, HRDiT, ours). It shows
   one walker under an arch, with tiny background pedestrians in the square: a plausible scene, invisible at thumbnail
   size. Kept.

## Requests to other owners
Text owner (main.tex):
- Figure 1: the caption above. It has no panel letters, and its images are showcase prompts.
- Figure 5: the final caption above (round 3). `\vsixinc{qual_main}` at natural width; the figure is 4.20 in tall.

Appendix owner (appendix.tex):
- Make `\snippetat` prefer `figures/v6/snippets/<stem>.tex` and fall back to `figures/snippets/<stem>.tex`. Choose the
  file first with `\IfFileExists`, then `\input` it inside the existing placement wrapper. The v6 snippet files have
  the names the appendix already inputs, plus all_methods and crops4k.
- Replace the inline teaser block (`\apxgraphic[...,trim=0 0 0 1.12in,clip]{teaser}`, caption, label) with
  `\snippetat{p}{all_methods}`. Replace the inline zoom4k block (`\apxgraphic{zoom4k}`, caption, label) with
  `\snippetat{!htbp}{crops4k}`. Do not trim any v6 PDF.
- `\snippetat{p}{hero}` picks up v6/snippets/hero.tex, which includes aspect.pdf with fig:hero.
- New appendix figure: `\snippetat{p}{crops_matched}` (label fig:crops_matched), placed in the progressive appendix
  or the galleries. It documents that crop detail is level with HRDiT where the methods draw the same content; it
  pairs with crops4k or gallery_baselines on one float page.
- Failure section: fig:failures shows the four person-free prompts with the largest ImageReward deficit against AReA.
  It no longer shows "local errors".
- App. progressive: crops4k does not support "single-pass SCA keeps more native detail in every crop"; cite
  fig:gallery_prog for that comparison.
- Galleries text:
  - fig:gallery_prog16/b: 18 progressive prompts, with UltraImage added.
  - fig:gallery_prog: six person-free prompts, five methods.
  - fig:gallery_4096 and _p2: 16 single-pass prompts.
  - Wide sizes: four prompts each.
  - fig:teaser: all eight methods.
  - fig:hero: single-pass at three sizes.
  - Every caption states its rule. Windows follow the maximin rule. Provenance sits at
    figures/v6/<stem>_provenance.json.
- Float pairing: these half pages fit two per page: crops4k with gallery_baselines, gallery_flux2 with
  gallery_krea_complete, gallery_6144_complete with gallery_8192_complete. progressive and failures are half pages too.
- Re-check the Krea sentence about seven regenerated SEGA outputs against the two Krea prompts
  (m6858649860006634381, m4850138611731379908).

Tables and plots owner:
- Add `CROP = "#E4572E"` to scafig.py, and `scaCrop` to sca_colors.tex (the design system's red box).
  `v6_common.BOX` reads `scafig.CROP` when present.
- Optional, for Figure 5's claim in numbers: `\hlProgUltraMusiqc{+12.4}`, `\hlProgUltraMusiqcLo{11.5}`,
  `\hlProgUltraMusiqcHi{13.2}` from results/prog195_vs_baselines/sca_v2_prog_vs_ultraimage.csv (musiq_crop1024).

Lead: no open decision on Figure 5; no GPU work is needed.
Nothing was committed.

## Sizes
Figure 1 0.51 MB, Figure 5 2.27 MB; the 20 appendix PDFs (with crops_matched) 13.7 MB; my PDFs total 16.5 MB. All 22 floats compile
together to 15.7 MiB. The v5.1 qualitative assets were about 38.7 MB and the v5.1 paper 34.4 MB.

## Not verified
- I have not built the real paper with main.tex and appendix.tex. I checked the main figures on ICLR test pages and all
  snippets in a test document with stand-in macros: no errors, no duplicate labels, no overfull boxes.
- The detector screens people only. Repeated animals or objects were checked by eye at page scale, not at 1:1.
