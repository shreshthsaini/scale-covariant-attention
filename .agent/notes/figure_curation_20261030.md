# Qualitative figure curation, 2026-10-30 (final operator, sca_v2)

Scope: the curated figures that show the final operator: Figure 1 (fig:teaser, fig_hero.py), Figure 6
(fig:qual_main, fig_zoom4k.py) and Figure T.1 (fig:hero, fig_showcase.py). The swap of 2026-10-30 (597d7cc) put
sca_v2 images (out/dupfix/sca_v2/<HxW>, same prompt ids and seeds as the published operator) into them and
flagged two defects in the captions (\pendImgTeaser, \pendImgQual). This pass re-audited every curated sca_v2
image at native resolution and replaced each prompt whose image shows a defect a reader would see at print size.

## Protocol (figure_curation_20261020.md, now ranked on the final operator)
- Eligible: the 195 evaluation prompts minus the 4 development prompts and every prompt displayed in any figure
  (scan of every figure provenance JSON with pool keys dropped, plus the prompts of these three figures and the
  showcase exclusion list): 101 eligible prompts.
- Fair (the documented rule): every baseline the figure compares at that size has ImageReward and HPSv2.1 at or
  above the 25th percentile of its own 195 seed-0 images (percentile = floor of 100 x share at or below the
  value, as fig_zoom4k.percentiles). Figure 6: AReA and HRDiT (full attention unconstrained). SCA-only images
  (Figure 1 mosaic, showcase): AReA at that size, the only baseline evaluated there.
- Rank by SCA's ImageReward + MUSIQ_c percentiles on the final operator's per-image CSVs
  (eval/a4k195_main_sca_v2, a4k195_2048x6144_sca_v2, a4k195_2048x4096_sca_v2); the showcase also needs both in
  the top half (its 2026-10-20 rule).
- View: overview contact sheets first, then an independent auditor (Opus) reads every 1024 px native tile, the
  overview and the figure's window, and reports garbled text, streaks, seams, extra or repeated subjects,
  lattice, broken anatomy. Baseline images: z.source / the trial build load every shown method's PNG with a
  complete sidecar and matching hash.
- Tools and views (scratch, session 4c03b520): scratchpad/audit30/{tools,views,cands,rows,notes_*}.

## Audit of the 14 curated sca_v2 images (every native tile, overview and shown window)
Verdicts: DEFECT = visible at print size or inside a shown window; MINOR = only at 1:1. Coordinates are source px.
| figure, slot | prompt id | verdict | finding |
|---|---|---|---|
| Fig 1 mosaic, panorama 2048x6144 | m2742856837225126754 | DEFECT, replaced | garbled two-line lettering on a facade (x 4124-4202, y 1046-1077) and pseudo-signs inside the inset window (4064, 832, 384) |
| Fig 1 mosaic, cathedral | m8737279395568207187 | MINOR, kept | chandeliers on caps without visible chains, confetti-like lancet glass, tiny pseudo-text plaque; rose-window inset clean |
| Fig 1 mosaic, portrait | 5849277588805396896 | MINOR, kept | crumpled tuner keys, red striping on the headstock, bright rim around dark silhouettes; hands sound |
| Fig 1 strip, SCA | m8343371271909676658 | MINOR, kept | loose dark rod on the nave roof, 1.2 grey-level step in the sky; one church, about 40 sheep; belfry window clean |
| Fig 6 row 1 | 7109958488611222679 | CLEAN, kept | hair window clean |
| Fig 6 row 2 | 2232973817337527860 | DEFECT, replaced | ghost lines repeat the table top's back edge (x 1100-1750, y 3290-3430), white glow over the rim, streaks on the drape; curtain ribbing locked to the 16 px token grid |
| Fig 6 row 3 | m548913239957610051 | MINOR, kept | 2 px scanlines on the grass slopes (cancelled by the Lanczos thumbnail); window clean; one waterfall in two cascades |
| T.1 panorama | 4230887929316158402 | MINOR, kept | faint 16 px block texture in the dark forest; both windows clean |
| T.1 portrait | m1066207185399086750 | MINOR, kept | pseudo-script on the glasses arm (about 0.09 x 0.02 in printed), faint arc above a lens; earring inset clean |
| T.1 illustration | m6845347395255545991 | DEFECT, replaced | a second lit tower (pagoda with lit windows) on the left hill (x 1412-1499, y 2170-2310); the prompt asks for one glowing tower; visible in the embedded image |
| T.1 still life | m8919475713516982506 | CLEAN, kept | inset clean |
| T.1 ice climber | 997293625843621481 | CLEAN, kept | inset clean |
| T.1 interior 2048x4096 | 7414267858891695058 | DEFECT, replaced | a second dress form on a pole stand with a shelf through its torso (x 2280-2680, y 0-780), about 0.27 x 0.5 in printed; the prompt asks for one mannequin |
| T.1 river town 2048x4096 | m5370774387421928008 | MINOR, kept | pseudo-lettering on a boat hull and facade signs (10-15 px letters), a separate slender tower at the right edge |
The swap agent's FINAL_VIEW notes had called the illustration's left tower and the interior's dress form clean; the
independent audit counted both against the prompt.

## Replacements (percentiles: ImageReward/MUSIQ_c/HPSv2.1 within each method's own 195 seed-0 images)

### Figure 1 panorama (2048x6144): m2742856837225126754 -> 7095623346664983157
- Why: garbled lettering inside the inset window, the brightest detail of the inset. A text-free window existed
  on the same houses (3420, 740, 384), but the old prompt also fails the fairness rule at this size (AReA HPSv2.1
  p17), so it was replaced rather than re-windowed.
- Check: new SCA 66/87/35 (ImageReward 1.17, MUSIQ_c 71.1), AReA 56/96/47 (fair). Old SCA 70/91/87, AReA 27/98/17.
- Shortlist: 101 eligible, 58 fair, ranked by SCA ImageReward + MUSIQ_c percentile. Overview rejections:
  1638867997226496674 (two pairs of children for a boy and a girl), 1118979843958125111 (striped forearm, floating
  hand), m9219623328452982854 (second face), m3880872769316591135 (extra cloaked figures and trees),
  m293370523718655394 (second waterfall). Native audit: 7095623346664983157 ACCEPT (1:1 only: blocky texture on
  the right-hand snow ridge, gravel speckle, a small pale lump, a 6 px fragment above the ridge; the prompt's starry
  sky is 1 to 3 px dots, not a defect class, and the teaser prints no prompt); m1017950324987649681 also ACCEPT
  (backup, ranked second).
- Window (2100, 610, 384) on the snow ridges, picked from the auditor's proposals; inset on the right bank. Only
  SCA is shown at this size, so no baseline image is needed.

### Figure 6 row 2 (4096^2): 2232973817337527860 -> m7824758602266642611
- Why: ghost lines along the table top's back edge, white glow over its rim, streaks on the drape (borderline at
  print, plain at 2x zoom).
- Check: SCA 88/83/92; AReA 61/51/64 and HRDiT 71/27/71 (ImageReward and HPSv2.1 >= p25, asserted by
  fig_zoom4k.py); full attention 45/44/4 (unconstrained, as documented). Printed MUSIQ_c full/AReA/HRDiT/SCA
  30.9/51.2/49.1/70.6, SCA from eval/a4k195_main_sca_v2/base/per_image.csv (the swap agent's wiring, kept); the
  lily row printed 37.3/46.4/43.5/53.2. full, AReA (out/main) and HRDiT (panel) PNGs load with complete sidecars.
- Shortlist: 45 fair of 101, ranked. m947425340308551929 (top) REJECTED by the native audit: a see-through third
  bottle neck inside the tall bottle and pears floating inside both bottles, visible at print (my overview missed
  it). m7824758602266642611 ACCEPT: no copy of the character; 1:1 only: six digits on the left hand (about 0.4 mm
  per finger printed), pseudo-lettering on the plaque, book spines, jar and map, a melted chain tangle; the
  automatic window (eye) is clean. Overview rejections: m9039328566106297566 (neon pseudo-characters),
  m795146951737249951 (extra bass players and men), m5332089088153794307 (extra citrus slices),
  m5996982714209002611 (lattice on coat edges); 175243114551627459 and 5941097972571555184 skipped (repeat Figure
  1's sheep and church). Backup: 1118979843958125111 (views made, not audited).
- Re-admission: this prompt was rejected on 2026-10-20 for the published operator's image (two smaller copies of
  the character); the final operator's image has none (READMITTED_V2 in fig_zoom4k.py).

### Figure T.1 wide 1 (2048x4096): 7414267858891695058 -> m9196631274539277035
- Why: a second dress form with a shelf through its torso, visible at print; the prompt asks for one mannequin.
- Rule (showcase): SCA ImageReward and MUSIQ_c both in the top half of its 195 on a4k195_2048x4096_sca_v2, AReA
  ImageReward and HPSv2.1 >= its p25 (12 prompts). Old: SCA 11/29/33, AReA 7/54/3 (it was picked by eye).
- Check: new SCA 93/70/99, AReA 90/6/87 (fair; the rule does not constrain MUSIQ_c).
- Candidates in rank order: 3943607367684487957 boy on hay (94/98/97) REJECTED by the native audit (white
  pseudo-signature on the hay, egg-shaped ball wider than his torso, horn-like bat top, all visible at print);
  m3118670343091498341 (fused second wolf head) and 1139740195494332405 (two women, two owls) and
  1442138154946148100 (streaks around the figure) failed at overview; 8253776796901638375 skipped (repeats the
  showcase portrait). m9196631274539277035 dancer under water (93/70/99) and 5826245701465116957 cabin in snow
  (84/79/86) both passed the native audit (dancer clean: the band at the top is the dress and hair reflected under
  the surface; cabin minor at 1:1). They tie on the ranking sum (163); the dancer wins on ImageReward and HPSv2.1
  and adds variety to a row that already holds the river town.

### Figure T.1 square 2 (4096^2): m6845347395255545991 -> m7264311869050248176
- Why: a second lit tower on the left hill, visible in the embedded image (I checked the 802 px asset); the prompt
  asks for one glowing tower.
- Rule (showcase, strict): SCA ImageReward and MUSIQ_c both in its top half on a4k195_main_sca_v2, and every
  baseline evaluated at 4096^2 except full attention (AReA, HRDiT, DyPE, SEGA, UltraImage) with ImageReward and
  HPSv2.1 >= its p25: 7 prompts. The old image fails this rule (HRDiT ImageReward p8).
- Check: new SCA 89/52/100, AReA 86/92/100, HRDiT 84/62/95, DyPE 89/69/92, SEGA 82/47/98, UltraImage 77/45/96;
  old SCA 45/58/30.
- Candidates in rank order: m795146951737249951 (extra bass players) and m5996982714209002611 (lattice) failed at
  overview; m7264311869050248176 woman with owl ACCEPT (one woman, one owl, two wings; 1:1 only: six talons on one
  foot, a red rim on a curled finger); m5715104867979054073 skipped (repeats the showcase portrait);
  1139740195494332405 (an extra wing-like shape) and m3118670343091498341 (two wolves for three) failed at
  overview; 7883963688096379563 REJECTED by the native audit (a second miniature tree on its own islet).
- Window (2045, 1495, 384) on the owl's face (auditor's choice over my (2050, 1450)); inset docked bottom-right on
  grass and trunks, no leaders (they would cross the owl); badge top-left above the wing tip.

## Result
- Replaced 4 of 14: Figure 1 panorama, Figure 6 row 2, Figure T.1 square 2 and wide 1. Kept 10, all with flaws
  only at 1:1 (listed above). No caption note about a visible defect remains (\pendImgTeaser and \pendImgQual
  are empty). Figure sizes are unchanged (teaser 396 x 222.2 pt, zoom4k 396 x 219.9 pt, showcase 396 x 333.6 pt)
  and the smallest figure text is 6.08 pt.
- Residual 1:1-only flaws in the new images: river ridge texture and gravel speckle (Figure 1), six digits on the
  character's hand and pseudo-lettering on its plaque and books (Figure 6), six talons on the owl's foot (T.1).
- Process note: two of my overview passes missed defects that the native audits caught (the ghost bottle in
  m947425340308551929, the islet tree in 7883963688096379563), and the swap agent's view had passed the
  illustration's second tower and the interior's dress form. Overview checks are a pre-screen only.

## Captions
- Figure 1: \pendImgTeaser emptied; the caption now says "curation favours \sca{}" (line count unchanged, 4).
- Figure 6: \pendImgQual emptied; the caption already says curated and that curation favours SCA (3 lines).
- Figure T.1 (snippet from fig_showcase.py): "picked by eye for variety and clean results, so curation favours
  \sca{}".

