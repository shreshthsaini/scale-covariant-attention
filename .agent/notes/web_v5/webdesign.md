# Web v5: designer and integrator (role webdesign), 2026-11-13

## Round 3 (19:30): one patch, three sizes
- "How an image forms" leads with webmedia's patch3 loops (role patch3, five prompts) in a 72rem player (muted loop,
  plays in view, play/pause button, poster only under reduced motion, the 9:16 cut below 700 px when one exists) with
  a one-row example switcher (poster thumbnails); then "Try it: pick a patch and scrub the run." and the formation
  explorer. The formation video carousel moved to gallery.html ("Three runs, patch by patch").
- How it works: steps 2 and 3 now say "local window" and "global grid" once each.
- Bundle for a NEW Artifact: /scratch/09032/saini_2/sca/web_builds/v5_artifact/bundle_r3/ (complete, fresh), 140
  files, 41.1 MB, one chunk; no 9:16 cuts, no formation or process videos, no method loops (step 2 falls back to a
  poster), explorer with one run, no zoom tiles, served types only. Check PASS (1440x900, 400x844, light, dark).
- Section screenshots: /scratch/09032/saini_2/sca/web_builds/v5_r3_final/forms_{desktop,desktop_dark,phone,phone_dark}.png

## Round 4 (2026-11-14): close the go-live items
- gallery.html eight-method grid: a seeded draw replaces the partly hand-picked set. web/build/v5_gallery_draw.py ->
  web/build/gallery_draw.json. Rule: random.Random(0).sample(sorted(pool), 15) (Python's random module), pool = the
  195 Aesthetic-4K evaluation prompt ids minus the main page's figure prompts (the nine gain-explorer deciles and the
  How-it-works sheep prompt), 185 ids; every pick has all eight images (HRDiT from the images Table 1 scored,
  eval/prog195_sca_v2 per_image.csv). Picks (sid): m2014927182109364975, m2742856837225126754, 1316997283360365550,
  7289666028778764798, m4850138611731379908, m4450387145579100387, m2679953061215100202, 8065862701789418887,
  m4012780704267384036, 924453187493270635, m6394968997939197542, 6422600338162192082, m4844341235185759356,
  4230887929316158402, 7637161027288029693. build_zoom.py compare reads gallery_draw.json (overviews 1600 px, 3 stop
  tiles, thumbs; 63.7 MB new). Caption: "Fifteen evaluation prompts drawn with seed 0; single-pass SCA is hidden on
  prompts that name a person." The "earlier paper version" sentence is gone.
- review/: rebuilt on site.css (v5 top bar, sub-page head, raised cards as <article>; a <section> would take the
  sub-page section spacing), single-pass SCA hidden on prompts that name a person (site.js's keyword rule): 40 of the
  96 prompts, so 40 panels hidden; one line says so under the lead. check_page.py's review check scopes ".rv-card";
  publish_pages.sh rewrites the review page's paper link to ../paper.pdf.
- Sub-pages on the v5 system (site.css "round 4" block): left-aligned heads without kickers or eyebrows, one pill
  style for every toggle, plain meta lines, one card style, v5 type sizes; every section on the wide column like the
  main page (results' two mid-width sections and abstract/limits/protocol prose, exploratory's Ideogram section), table
  titles and note boxes left-aligned. All four sub-pages carry the main page's navigation (Result, How it works, How an
  image forms, Compare, Results, Gallery, Paper); the old "Idea" link pointed at a removed anchor (#observation).
  Sub-pages now preload the three main font files over http(s), as the main page does. Results keeps a layout shift
  of 0.09 on desktop: the font swap reflows the abstract while the page's data scripts run (under the 0.1 bar).
- Gallery carousel ("Each run, patch by patch", was "Three runs"): one item per run (5), the 16:9 cut, and below
  700 px that run's 9:16 cut, as in the main page's players.
- Publishing: web/build/publish_exclude.txt (read by publish_pages.sh as git pathspec excludes) leaves out 129 MB that
  no page loads: the old eight-method set (15 folders, 46.7 MB), assets/img/qual, the trace videos and the v4
  explainer (read only by fallbacks that never run while the v5 videos exist), the v5 process and tagged or 9:16 hero
  cuts, and the unlinked review_qwen page. assets/img/compare STAYS (exploratory's Ideogram cards use it). Verified
  two ways: the pathspecs against git ls-files, and a server that answers 404 for every excluded path while all five
  pages are browsed at 1440 and 390 px with the carousels and switchers clicked (0 requests failed). Site: 397 MB in
  3,959 files plus paper.pdf 25 MB.
- Artifact bundle: /scratch/09032/saini_2/sca/web_builds/v5_artifact/bundle_r4/ (fresh), 152 files, 71.4 MB, two
  chunks ([146 files, 51.4 MB], [5 files, 19.9 MB]); changed.json against the published copy sca_page_r3 lists 23
  paths (site.css, site.js, v5_assets.js and the 20 patch3 files: ten mp4 and ten webp posters, all re-rendered at
  12:30 to 12:35); removed.json empty; no new paths; page.html unchanged. Check PASS at 1440x900 and 400x844, light
  and dark. The 23 was recounted independently by hashing every file against sca_page_r3.
- Still open for the lead: git rm the 15 old web/assets/img/deep/compare folders (they are only excluded from the
  site); exploratory.html's Ideogram section is still a visible placeholder ("The Ideogram 4.0 images are not in yet").

## Round 3 (user via coordinator; finished on node i614-034 after the 19:35 restart)
- "How an image forms" leads with "One patch, three sizes" (webmedia patch3_<prompt>.mp4, role patch3): a 72rem
  (1152 px at 1440) 16:9 player in a frame, muted loop that plays while in view, a visible play/pause, poster only
  under reduced motion; the 9:16 cut (role patch3_9x16) below 700 px; a one-row switcher of five pills with round
  crops of each poster's image square. Then "Try it: pick a patch and scrub the run." and the formation explorer.
  site.js patch3(); the poster is data-src (kept out of the first screen).
- The formation_* carousel left the main page and sits on gallery.html ("Three runs, patch by patch").
- How it works: step 2 "a native-size local window ... a native-size global grid"; step 3 "mixes its local window
  and its global grid"; nothing else changed there.
- Registry: v5_assets.js keeps readouts only for the hero, drops provenance fields and is written compact (36 KB).
  Head pictures re-encoded at quality 74. First screen 2.97 MB.
- Check (check_page.py --http --sections): PASS on every page. Main page 8,515 px at 1440 (9,072 on a phone),
  964 visible words with the components' text (572 page copy), layout shift 0.002.
- Screenshots: section /scratch/09032/saini_2/sca/web_builds/v5_r3_final2/section/forms_{desktop,desktop_dark,phone,phone_dark}.png;
  full page /scratch/09032/saini_2/sca/web_builds/v5_r3_final2/shots/index_{desktop,phone}_{light,dark}_{first,full}.png;
  all pages and sections /scratch/09032/saini_2/sca/web_builds/v5_r3_final2/check/.
- Bundle for the new Artifact: /scratch/09032/saini_2/sca/web_builds/v5_artifact/bundle_r3_final/ (a fresh folder;
  bundle_r3 is left untouched because nothing on scratch is deleted). 152 files including page.html, 65.5 MB.
  files.json: chunk 0 = 147 files, 51.4 MB (every page file, all 16:9 videos, two 9:16 cuts); chunk 1 = 4 files,
  14.0 MB (the remaining 9:16 cuts). The 9:16 cuts are included because the page plays them below 700 px (five
  patch3 cuts and the speed race's cut). Explorer with one run, no zoom tiles, no formation_* videos, served types
  only, no directory entries. Check PASS at 1440x900 and 400x844, light and dark.

## Round 2 (user via coordinator, 18:40): subtraction and one centrepiece
- Main page now: hero; result (speed race + claim); How it works in three plain steps (1: two real attention
  pictures of block 12 heads 4 and 7 from paper/figures/src/data/v6_attn_block12.npz, web/build/v5_heads_pictures.py,
  plus one stat line 97.5%; 2: views {compact} with its own mix bars hidden, fallback the method_window poster;
  3: two plain bars from focus.{near,far}.{window,grid}_pct, new keys in build_numbers.py from \hlFocusNearPi and
  \hlFocusFarPi); How an image forms (formation explorer, then the formation videos as a smaller carousel; process
  videos if those are missing); Against the baselines (percentiles {compact}, rule printed once); Results (three macro
  numbers, table {compact}, honest edges as one paragraph, links); explainer; citation. No eyebrows, badges, rule
  chips, abstract, reach table, J curve, heat maps or paper figures on the main page.
- Moved to results.html: abstract (top) and "Method in detail" (heads, paper Figure 3, emulate, stages, the three
  method loops, method-zoom and emulation figures, the v4 widgets). Gains and speed were already there.
- gallery.html: the eight-method set is now described as "the comparison set of an earlier paper version plus a
  seeded draw of three from the other 183". FOR THE LEAD: that set is not fully rule-based (eight of its prompts were
  picked by hand for an earlier paper version); raise it with the user.
- publish_pages.sh also leaves out web/assets/img/ideogram4 (238 MB, used only by assets/js/ideogram4.js, which no
  page loads), assets/js/ideogram4.js and web/dev. Estimated published size about 365 to 375 MB plus paper.pdf.
- Artifact bundle round 2: /scratch/09032/saini_2/sca/web_builds/v5_artifact/bundle_r2/ (fresh folder). Served types
  only (no .vtt; the explainer keeps its burned-in captions), files only, chunk 0 = everything but zoom tiles (there
  are no tiles now: the percentile explorer's zoom stops at its overviews), formation keeps its first run (Silver
  braids, 49 files), changed.json and removed.json against the published copy in the scratchpad (sca_page).
- Round 2 final (check PASS on every page; bundle PASS): main page 8,291 px at 1440 wide (was 14,813), 950 visible
  words including the components' text (558 of page copy), first screen 2.99 MB, layout shift 0.002. Bundle_r2: 141
  files, 30.9 MB, one chunk; changed.json 75 paths, removed.json 368 paths. Screenshots:
  /scratch/09032/saini_2/sca/web_builds/v5_r2_final/shots/index_{desktop,phone}_{light,dark}_{first,full}.png,
  v5_r2_final/check/ (sections, sub-pages), v5_r2_final/vs_area_{desktop,phone}.jpg,
  /scratch/09032/saini_2/sca/web_builds/v5_artifact/shot_{1440,400}_{light,dark}.png (bundle).
- The first bundle folder (v5_artifact/bundle/) stays on scratch untouched; publish from bundle_r2/files.json.

## Final check of round 1 (18:10)
check_page.py --http --sections: RESULT PASS on index, gallery, results, exploratory (desktop and phone, light and
dark) and review/: no console errors, no broken files, no overflow, zoom checks pass, reduced motion (no video
plays), keyboard (first Tab reaches the skip link). Main page: 14,813 px at 1440 wide, 3,082 visible words of which
1,736 are page copy (the rest is the seven components' own text), first screen 2.94 MB, layout shift 0.002.
Screenshots: /scratch/09032/saini_2/sca/web_builds/v5_webdesign_final4/shots/index_{desktop,phone}_{light,dark}_{first,full}.png
(+ _full_s*.jpg segments); per-section and sub-page shots in v5_webdesign_final4/check/; side by side with AReA:
v5_webdesign_final4/vs_area_desktop.jpg and vs_area_phone.jpg. Artifact bundle rebuilt after the last change (PASS).

## State (18:00)
Main page rebuilt (web/index.html), new visual system (site.css), site.js split into boot + deferred rest(), three
sub-pages, every number from build_numbers.py (paper macros as tex.* keys). All seven components and the videos of
the other owners are wired; anything missing falls back to a static figure, a loop or a v4 widget. An Artifact
preview bundle of the main page is built and checked (below). Nothing published, no git.
- Before (lead's audit): 26,500 px, about 5,100 words. Now at 1440 wide: 14,764 px (hero 750, result 651,
  abstract 382, observation 1251, method 3421, see it 2250, sizes 1035, process 1089, results 2306, more 1482,
  footer 148). Words of visible text in <main>: about 1,700 of page copy; about 2,900 when the seven components'
  own labels and captions are counted.
- First screen (no scroll): 2.94 MB (hero loop 2.48 MB, poster, CSS, site.js, v5_assets.js, three preloaded fonts);
  the nine data scripts load after the first scroll, tap or key, or two seconds after load. Layout shift 0.002.

## Page (index.html), in order
hero (deep-zoom loop with a live readout from manifest.readout, "Measured once at 1024^2, reused at 4096^2, no
training", title, subtitle, "Anonymous authors, under review", Paper / arXiv soon / Code at release / Video / BibTeX,
three macro numbers; sized to content, at most about 82% of the viewport) -> 01 result (speed race video, 9:16 on
portrait screens; HTML race from numbers.js as fallback) -> abstract (two columns) -> 02 observation (97.5 / 2.5 /
0.7 beside the headline; heads component, fallback paper Figure 2) -> 03 method (step 1 views component, fallback
method_window loop then the v4 drawing; step 2 emulate, fallback method_emulation then the v4 parabola; step 3
stages, fallback method_stages then the v4 reach table; paper Figure 3 once) -> 04 see it, dark band (percentiles;
eight-method synced zoom grid on 15 prompts, single pass hidden on prompts that name a person) -> 05 sizes and
checkpoints (paper Figure 8, link to the twelve showcase images) -> 06 process (five progressive process videos) ->
07 results (four macro numbers, table component over the static Table 1, gains plot beside the speed component,
honest edges incl. the paper's one single-pass sentence) -> 08 explainer (84.5 s, controls, VTT) -> sub-page cards ->
BibTeX -> footer.

## Sub-pages (web root, same assets, no SCA_DEFER)
gallery.html: twelve showcase cards, push-in clips p25/p50/p75 (one per row so their labels read), eight methods on
15 prompts, first 12 prompts, single pass beyond 4096^2 (no person). results.html: seeds and HRGen card, time bars,
Table 1, gains, generality table (component), per-baseline deltas, facts (Holm wording), speed (quality-time + speed
component), method in detail (three method loops, paper method-zoom and emulation figures, v4 widgets),
limitations, protocol. exploratory.html: FLUX.2-dev and Qwen-Image beside the same model, Ideogram placeholders,
labelled exploratory. review/ untouched (lead's page).

## Artifact preview bundle (user request via the coordinator)
- Script web/build/v5_artifact_bundle.py; bundle /scratch/09032/saini_2/sca/web_builds/v5_artifact/bundle/
  (page.html + assets/...), manifest bundle/files.json ({"chunks": [...]}, page.html excluded).
- Last build: 449 files, 65.6 MB, 3 chunks (240 files 9.8 MB; 204 files 54.8 MB; 4 files 0.9 MB). Rebuilt in place,
  never deleted: files.json is the list (two _check_*.html wrappers beside page.html are not in it).
- page.html: no doctype/html/head/body; starts with <title>Scale-Covariant Attention</title> and
  <meta name="claude:viewport-fit-sides" content="cover">; paper, gallery, results, exploratory and 96-prompt links go
  to https://shreshthsaini.github.io/scale-covariant-attention/ (new tab, "on the live site" note). Trims: percentile
  tiles only at the 10th, 50th and 90th stops; eight-method grid overviews only; zoomgrid.js stops at the overview
  resolution for any grid without tiles. media.js and traces.js are empty bundle copies (v4 videos not carried).
- Check (127.0.0.1, minimal skeleton, other hosts blocked), 1440x900 and 400x844, light and dark: no console
  errors, no failed or external requests, no overflow, all seven components live, 7 videos. Screenshots:
  /scratch/09032/saini_2/sca/web_builds/v5_artifact/shot_{1440,400}_{light,dark}.png.

## Self-score (acceptance items, 0/1/2)
1 first screen 2 | 2 identity 2 | 3 length and rhythm 1 (14,764 px, copy about 1,700 words, about 2,900 with the
components' text) | 4 method in motion 2 | 5 examples 2 | 6 video and loops 2 | 7 results 2 | 8 craft 1 (published
site over 400 MB because of assets/img/ideogram4 and review/; file:// has a font-swap shift since preloads are
http-only). Still weak: the method section is long; sub-pages reuse the v4 widget styling under the new tokens;
phone hero image is small (16:10 box).

## Contract with the other owners
- Components: `<figure class="comp-figure">` holds `.comp-fallback` and `<div class="comp-mount" data-comp=NAME
  data-options=JSON>`; site.js loads c_NAME.css/js only when v5_assets.js lists them, mounts within 900 px of the
  viewport (observing the figure), adds .is-live; `.cap-fallback` captions hide when live; fallback images with
  data-src load only when the component is absent or fails.
- Videos: `<figure data-vrole=ROLE [data-vrole-portrait=ROLE2]>` with `.vslot-media`; loops are preload="none" and
  play while visible; captions from the manifest (typographic x, pi) unless the figcaption has data-keep.

## Rebuild
PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python; PW: PLAYWRIGHT_BROWSERS_PATH=/scratch/09032/saini_2/tmp/ms-playwright /work/09032/saini_2/vista/library/uv-envs/pwqa/bin/python
1. $PY web/build/build_numbers.py --fill web/index.html --fill web/gallery.html --fill web/results.html --fill web/exploratory.html
2. $PY web/build/v5_figs.py; PW web/build/v5_raster_figs.py (only when paper figures change)
3. $PY web/build/build_v5_assets.py (after any new component or video)
4. PW web/build/v5_shots.py --out /scratch/09032/saini_2/sca/web_builds/v5_webdesign_N/shots --pages index.html gallery.html results.html exploratory.html
5. PW web/build/check_page.py --http --sections --out DIR
6. PW web/build/v5_artifact_bundle.py --check (the Artifact preview; the lead publishes)
publish_pages.sh rewrites the paper link in every root HTML file (was index.html only). The sub-pages were generated
once from the v4 page by /scratch/09032/saini_2/sca/web_builds/v5_webdesign_2/make_subpages.py and are now edited by
hand (do not rerun it). v4 originals: /scratch/09032/saini_2/sca/web_builds/v5_webdesign_1/v4_backup/.
