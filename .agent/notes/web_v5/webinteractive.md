# webinteractive notes (owner: webinteractive; components for the v5 page)

Contract: BRIEF.md Section 4 and Section 6, design.md "Components". Every component: assets/js/c_<name>.js,
assets/css/c_<name>.css, data in assets/data/<name>/ (data.js sets window.SCA_DATA.<name>, loaded by script tag so
file:// works), demo page web/dev/<name>.html (loads only that component plus web/dev/dev.css, a copy of the
design tokens, and dev.js, which mounts like site.js does), build script web/build/v5_build_<name>.py.
Check script for all of them: web/build/v5_check_components.py <names> --out DIR (console errors, overflow,
mount height against 80vh, scripted interactions, screenshots at 1440x900 and 390x844, light and dark).

Common behaviour (all components)
- Mount: SCAComp.<name>(el, options) returns {set?, destroy}. It renders a skeleton with the final geometry at
  once, then loads data.js when the mount is within 900 px of the viewport (IntersectionObserver), so it never
  shifts layout and costs nothing above the fold.
- Data failure: if data.js (or zoomgrid.js) fails to load, the component removes .is-live from its closest
  <figure> and empties itself when that figure has a .comp-fallback (the page's static image shows again);
  otherwise it shows assets/data/<name>/fallback.webp itself.
- Colours only from the design tokens (var(--x, fallback)); class names prefixed c-<name>-; no console output;
  prefers-reduced-motion: no animated transitions (zoom flights jump).
- The component needs no other script: c_percentiles.js loads zoomgrid.js and zoomgrid.css from its own folder
  if window.SCAZoom is missing.

## 1. percentiles: DONE (16:20)
Files: web/assets/js/c_percentiles.js, web/assets/css/c_percentiles.css, web/assets/data/percentiles/
(data.js 11 KB, fallback.webp 0.12 MB, img/p<pct>/<area|ultra|hrdit|prog>/{v.webp, <c>_<r>.webp}),
web/build/v5_build_percentiles.py, web/dev/percentiles.html. Data 24.9 MB in total (36 overviews 1024 px WebP q76,
3.4 MB; 576 native tiles 1024 px WebP q60, 21.5 MB, loaded only when a view needs them).
API: SCAComp.percentiles(el, {start: 50, lazy: true, base: "<url of assets/data/percentiles/>"}) ->
{set(p), destroy()}; start is the percentile shown first (10..90).
Mount on the page: <div class="comp-mount" data-comp="percentiles" data-options='{"start": 50}'></div>, fallback
sibling <img class="comp-fallback" src="assets/data/percentiles/fallback.webp" width="1944" height="480" alt="...">
(the median prompt's four whole images side by side, no text, so it suits both themes).
What it shows: a slider with nine stops over the nearest-rank deciles of the paired ImageReward difference,
progressive SCA minus AReA, 195 prompts, seed 0 (exactly the rule and rows of the paper's Figures deciles_a/b;
the build imports v6_more.py's functions and asserts every row against deciles_{a,b}_provenance.json and the mean
against results/prog195_vs_baselines/sca_v2_prog_vs_area.csv). Per stop: the stop and its difference (AReA ahead
at the 10th and 20th percentiles, shown in AReA purple with an "AReA ahead" pill), the prompt, a strip chart of
all 195 sorted differences (mean line, the stop's bar marked), the four whole images (AReA, UltraImage, HRDiT,
progressive SCA) with ImageReward (best bold, as the paper) and seconds per 4096^2 image (clean batch-1 medians
202.4 / 107.0 / 49.2 s; UltraImage 308.5 s campaign mean, marked with a double dagger), the selection rule.
Interaction: slider (drag, arrows, Home/End), click on the strip chart, swipe on the images; click or tap an image
and all four fly to native pixels at that point (SCAZoom.grid, tiles on demand, minimap), drag to pan, Auto zoom
tours three edge-energy stops, double-click or Esc or "Whole image" returns. Enter on a focused image zooms.
Checked: numbers equal the paper's provenance labels (diff -0.19, -0.02, +0.03, +0.11, +0.24, +0.38, +0.58, +0.82,
+1.28; bests area, area, prog, ultra, ultra, ultra, prog, hrdit, prog); no console errors; no overflow; height
712 px at 1440x900 (limit 720), 679 px at 390x844 (limit 675; the 2 x 2 images dominate).
Rebuild: nice -n 10 /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_build_percentiles.py
(about 10 s on 36 cores; rewrites only stale files; tour stops cached on scratch web_assets/zoom_cache/percentiles).
Screenshots: /scratch/09032/saini_2/sca/web_builds/v5_webinteractive_1/shots/percentiles_{desk,phone}_{light,dark}_*.png
Claims: only the paper's (rule, differences, ImageReward per image, seconds). It never says SCA wins every prompt.

## 2. heads: next
## 3. emulate: after heads
## 4. table, speed: last

## Requests to the designer
- Mount percentiles in the "See it" section, figure with a comp-fallback image as above. The component sets its own
  height (about 710 px desktop) and fills 100% width up to 1200 px.
