# Qualitative figure curation, 2026-10-20 (final paper pass)

User directive: "make sure in the paper you select the right samples to visualize and not bad ones."
Method: four opus auditors viewed every SCA image of every qualitative figure at full resolution (overview plus
1:1 crops); curated figures got clean replacements shortlisted by per-image ImageReward and MUSIQ_c
(eval/a4k195_main, a4k195_panel) and confirmed by eye; random galleries kept their seeded draws and got captions
that say what the samples show. Auditor notes and crops: scratchpad audit/a1..a4 (session 4c03b520, scratch).

## Curated figures (samples replaced where SCA had artifacts)
- Figure 1 (fig:teaser, fig_hero.py), curated:
  - panorama m2742856837225126754 (2048x6144) clean, kept; cathedral m8737279395568207187 clean, kept (its IR is
    p6 but every method scores p4-p7 on this prompt).
  - portrait 3657338173664010496 REJECTED (second miniature pipe on the headband, stray stem at the mouth);
    replaced by 5849277588805396896, man with hat and guitar (SCA IR 1.91, p92; MUSIQ_c 68.7, p73; HPS p90;
    headstock has too many tuners, a generic FLUX object error, not duplication).
  - six-method strip 1269964769307095670: SCA clean but comparison UNFAIR (AReA IR p1 and HPS p0, SEGA p2, DyPE
    p8); replaced by m8343371271909676658, sheep before a stone church (IR/MUSIQ_c/HPS percentiles: SCA 57/79/89,
    DyPE 37/54/53, SEGA 47/90/33, AReA 36/49/51, UltraImage 41/53/69). STRIP_BOX (1650,1300,768) on the church
    tower, where AReA and UltraImage also show church walls. Fairness rule: every baseline IR and HPS >= p25.
  - rejected candidates: polar bear 4820545723762191815 (five moons, extra bears), canoe (three waterfalls),
    veiled woman m7393293802729135153 (third hand), Japanese street m2670470656524755471 (speckle, extra
    figures), tree in water (second tree), tent (three people for two).
- Figure 2 (fig:motivation), development prompt Moai 3614061571823823193, kept by design: (a) v0 w64 draws five
  statues (claim holds); (b) SCA's big statue is clean but a small second Moai stands on the far shore, visible on
  page 2, so the caption now says so.
- Figure A.1 (fig:failure), development prompts by design: SCA boats clean; SCA bear has three partial extra women
  (over the bear's back, behind her shoulder, under her knees); caption (d) now says so.
- Figure 6 (fig:qual_main, fig_zoom4k.py), curated, all three rows replaced; MUSIQ_c now read from
  eval/a4k195_main and a4k195_panel (identical per PNG to hrdit191_base and final4k_*):
  - 2100482621711410240 REJECTED (smaller copy of the character at a desk in the background, floating hands);
    m2729538226706404616 REJECTED (second ear with an earring); m6394968997939197542 clean but UNFAIR (AReA HPS
    p7, IR p21; MUSIQ_c p2 for all methods).
  - new rows (IR/MUSIQ_c/HPS percentiles, SCA; AReA; HRDiT): 7109958488611222679 portrait 86/41/55; 90/64/63;
    76/53/60. m548913239957610051 waterfall 64/62/33; 71/34/38; 67/26/32. m3268147817845827695 medieval square
    was used first but its prompt asks for "a lone figure" and SCA adds about three pedestrians (one in the
    window), so a follow-up auditor (a5) replaced it with 2232973817337527860, pink lilies in a vase by a
    paneled door (percentiles IR/MUSIQ_c/HPSv2.1: SCA 87/31/86, AReA 83/38/95, HRDiT 88/15/25, full 89/71/22;
    background quirks only: a second flat door with a knob). No architecture or street candidate survived:
    alley (extra pedestrians, garbled sign), snowy church (second spire), neon street (SCA MUSIQ_c p47,
    invented characters), desk character (two smaller copies), balcony door (extra arm), stone window (two
    extra women), pitcher (second handle), gown on balcony (copy on a ledge), parasol couple (third hands),
    five children (eight or more), shutter window (echo outlines). Five more architecture prompts failed
    the fairness rule (now asserted in fig_zoom4k.py: AReA and HRDiT IR and HPSv2.1 >= p25). Printed
    MUSIQ_c (full/AReA/HRDiT/SCA): portrait 37.0/55.7/55.3/59.4, lilies 37.3/46.4/43.5/56.7, waterfall
    31.8/45.2/48.6/65.9.
  - SCA is bold in every row; selection partly favours SCA's MUSIQ_c, which the caption states.
- Figure R.1 (fig:hero, fig_showcase.py), curated:
  - panorama 7543519915112968451 REJECTED (gibberish restaurant signs, smeared right third incl. the inset);
    replaced by the Great Wall 4230887929316158402 (IR 1.343, rank 54/195; MUSIQ_c 70.37, rank 33). Insets
    (1530,1000,384) and (3330,1600,320). Rejected candidates: polar bear (several bears and moons), winter
    waterfall (soft right third).
  - still life minus947425340308551929 REJECTED (see-through third bottle, faded second bowl of pears, visible at
    figure size); replaced by blossoms with an apple minus8919475713516982506 (IR 1.397, rank 97; MUSIQ_c 68.40,
    rank 55), inset (1880,1900,384). Rejected candidate minus5332089088153794307 (floating second pitcher handle).
  - kept (flaws only at 1:1): portrait minus1066207185399086750 (scribble on glasses arm), mountains
    minus6845347395255545991 (extra small towers), ice climber 997293625843621481 (small hook at top edge);
    2048x4096 images 7414267858891695058 and minus5370774387421928008 clean.
- Figure R.7 (fig:ablation_grid, fig_qual.py ablation), prompts hand-picked (provenance says so); caption now
  says "curated":
  - owl 1139740195494332405 REJECTED (SCA draws a second woman holding a second owl at x 240-940, y 1430-2440);
    replaced by warrior m2857555394095239941 from the 46-prompt ablation subset (IR 1.709; MUSIQ_c 75.1, 5th of
    46), window (1280,576,512); its row shows all three captioned failures. Rejected candidates
    m7393293802729135153 (extra hand), 3943607367684487957 (third hand), m5685677173773420545 (streaks),
    3826098298648979216 (extra canoes). Artist row clean; sheep and Fantasyland rows minor only.
- Figure S.1 (fig:failures), curated failure cases by design: "foot for an arm" was wrong (it is an extra raised
  foot); caption rewritten and now says "Curated failures"; the gated images still hold small duplicates.

## Random galleries (draws verified, not swapped; captions corrected to match the samples)
- Draw checks: R.2/R.3 Random(0) 32 of a 175-prompt pool (195 minus 4 dev minus 16 curated/failure prompts);
  R.4 seed 33 and R.5 seed 39, 6 prompts each, first 2 shown; R.6, R.8-R.11 random.Random(17).sample over fixed
  look-free pools (FLUX.2 first 50, baselines all 191 non-dev, 6144 and Krea eval100, 8192 every 16th). Every
  draw recomputed, matches the frozen manifest, unchanged since first commit; source PNG hashes match.
- L.1 (fig:progressive): Random(17).sample(sorted 46 ids, 2) reproduces the shown pair; kept.
- Findings (SCA): R.2 dancer second airborne dancer, Milky Way three galaxy bands, red coat fused fingers;
  R.3 fused spires, extra hikers, extra astronauts; R.4 clean; R.5 extra climbers, extra trees; R.6 FLUX.2
  duplicates in three rows and a distorted portrait; R.8 dancer and hiker with extra people; R.9 stripes, block
  speckle, extra figures; R.10 SCA melts the lone fjord figure; R.11 SCA blocks and stripes at 8192^2. AReA
  blurs, speckles or grids instead, and keeps its layouts.
- Disclosure fix: the gallery pools exclude the 16 prompts of the curated and failure figures, not "those shown
  in the main text"; appendix text corrected.
