# Web v5: designer and integrator (role webdesign), 2026-11-13

## State (17:40)
Main page rebuilt (web/index.html), new visual system (site.css), site.js split into boot + deferred rest, three
sub-pages, every number from build_numbers.py (paper macros as tex.* keys). All six components and all eighteen
videos of the other owners are wired; anything missing falls back to the static figure or the v4 widget.
- Before (lead's audit): 26,500 px, about 5,100 words. Now at 1440 wide: about 15,000 px; words of visible text in
  <main>: about 1,700 of page copy, about 2,800 including the components' own labels and captions.
- First screen (no scroll, 1.5 s): 3.17 MB desktop (hero loop 2.48 MB, poster, CSS, site.js, numbers.js, fonts);
  the eight below-the-fold data scripts load after the first scroll, tap or key, or two seconds after load.
- check_page.py: no console errors, no broken files, no horizontal overflow on index, gallery, results,
  exploratory (desktop and phone, light and dark) and review/; zoom checks pass; reduced motion: no video plays;
  keyboard: first Tab reaches the skip link.

## Page (index.html), in order
hero (deep-zoom loop, title, subtitle, "Anonymous authors, under review", Paper / arXiv soon / Code at release /
Video (shown when the explainer exists) / BibTeX, three numbers, zoom readout driven by manifest readout) ->
01 result (speed race video, 9:16 on portrait screens; HTML race from numbers.js as fallback) -> abstract (two
columns) -> 02 observation (97.5 / 2.5 / 0.7 beside the headline; heads component, fallback paper Figure 2) ->
03 method (step 1 views component, fallback method_window loop; step 2 emulate component, fallback
method_emulation loop then the v4 parabola; step 3 method_stages loop, fallback the v4 reach table; paper Figure 3
once) -> 04 see it, dark band (percentiles component; eight-method synced zoom grid, single pass hidden on prompts
that name a person) -> 05 sizes and checkpoints (paper Figure 8; link to the twelve showcase images) -> 06 process
(five progressive process videos, carousel) -> 07 results (four macro numbers, table component over the static
Table 1, gains plot beside the speed component, honest edges incl. the paper's one single-pass sentence) ->
08 explainer (84.5 s, controls, VTT) -> sub-page cards -> BibTeX.

## Sub-pages (web root, same assets, no SCA_DEFER)
gallery.html: full showcase (12), push-in clips p25/p50/p75 (one per row so their labels read), eight methods on 15
prompts, first 12 prompts, single pass beyond 4096^2 (no person). results.html: seeds and HRGen card, time bars,
Table 1, gains and per-baseline deltas, facts (Holm wording), speed (quality-time + speed component), method in
detail (paper method-zoom and emulation figures + v4 widgets), limitations, protocol. exploratory.html: FLUX.2-dev and
Qwen-Image beside the same model, Ideogram placeholders, labelled exploratory. review/ untouched (lead's page).

## Contract with the other owners
- Components: `<figure class="comp-figure">` holds `.comp-fallback` and `<div class="comp-mount" data-comp=NAME
  data-options=JSON>`; site.js loads c_NAME.css/js only when v5_assets.js lists them, mounts within 900 px of the
  viewport (observing the figure), adds .is-live on success; `.cap-fallback` captions hide when live.
- Videos: `<figure data-vrole=ROLE [data-vrole-portrait=ROLE2]>` with `.vslot-media`; loops are preload="none" and
  play while visible; captions from the manifest (typographic x, pi) unless the figcaption has data-keep.

## Artifact preview bundle (user request via the coordinator, 18:00)
- Script: web/build/v5_artifact_bundle.py (build with any python; `--check` with the pwqa env runs the browser check).
- Bundle: /scratch/09032/saini_2/sca/web_builds/v5_artifact/bundle/ (page.html + assets/...), manifest
  bundle/files.json ({"chunks": [...]}, page.html excluded). Last build: 447 files, 65.5 MB, 3 chunks
  (240 files 9.8 MB; 202 files 54.8 MB; 4 files 0.9 MB). Rebuilt in place, never deleted: files.json is the list.
- page.html: no doctype/html/head/body; starts with <title>Scale-Covariant Attention</title> and
  <meta name="claude:viewport-fit-sides" content="cover">; links to the paper, gallery, results, exploratory and the
  96-prompt review go to https://shreshthsaini.github.io/scale-covariant-attention/ (target _blank, "on the live
  site" note). Trims: percentile tiles only at the 10th, 50th, 90th stops; eight-method grid overviews only; no
  v4 teaser, explainer or trace videos (media.js and traces.js are empty bundle copies). zoomgrid.js stops at the
  overview resolution for any grid without tiles (same file on the site; no change there, every site grid has tiles).
- Check (bundle served on 127.0.0.1, wrapped in a minimal skeleton, other hosts blocked): 1440x900 and 400x844,
  light and dark: no console errors, no failed requests, no external requests, no overflow, all six components
  live, 7 videos. Screenshots: /scratch/09032/saini_2/sca/web_builds/v5_artifact/shot_{1440,400}_{light,dark}.png.
- Contract changes made on the real site too (one stylesheet): hero sized to content, min(82svh, 880px) on wide
  screens and a 16:10 media box plus copy on phones (next section peeks in); wrappers padded
  max(16px, var(--gutter), env(safe-area-inset-*)); tokens on :root, dark under the guarded media query and
  [data-theme="dark"] (already so); body background from --bg; localStorage only inside try/catch.

## Rebuild
PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python; PW: prefix PLAYWRIGHT_BROWSERS_PATH=/scratch/09032/saini_2/tmp/ms-playwright to /work/09032/saini_2/vista/library/uv-envs/pwqa/bin/python
1. $PY web/build/build_numbers.py --fill web/index.html --fill web/gallery.html --fill web/results.html --fill web/exploratory.html
2. $PY web/build/v5_figs.py; PW web/build/v5_raster_figs.py (only when paper figures change)
3. $PY web/build/build_v5_assets.py (after any new component or video)
4. PW web/build/v5_shots.py --out /scratch/09032/saini_2/sca/web_builds/v5_webdesign_N/shots --pages index.html gallery.html results.html exploratory.html
5. PW web/build/check_page.py --http --sections --out DIR
publish_pages.sh now rewrites the paper link in every root HTML file (was index.html only). The sub-pages were
generated once from the v4 page by /scratch/09032/saini_2/sca/web_builds/v5_webdesign_2/make_subpages.py and are
now edited by hand (do not rerun it). v4 originals: /scratch/09032/saini_2/sca/web_builds/v5_webdesign_1/v4_backup/.
