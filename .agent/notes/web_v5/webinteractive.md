# webinteractive notes (owner: webinteractive; components for the v5 page)

## Status
All seven components are finished and checked (demo pages and index.html, 1440x900 and 390x844, light and dark):
percentiles, heads, emulate, table, speed (the brief's list) and views, stages (extras: method steps 1 and 3).
The page mounts heads, views, emulate, percentiles, table and speed (webdesign.md, page order); stages waits for
the designer (request at the end). Rebuild everything: the seven web/build/v5_build_<name>.py scripts (sca env), then
web/build/build_v5_assets.py, then web/build/v5_check_components.py <names> --page index.html --out DIR (pwqa env).
Last full check: /scratch/09032/saini_2/sca/web_builds/v5_webinteractive_1/final_check.jsonl and final/*.png.

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

## 1. percentiles: DONE
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

## 2. heads: DONE
Files: web/assets/js/c_heads.js, web/assets/css/c_heads.css, web/assets/data/heads/ (data.js 0.47 MB, img/base.webp
39 KB, fallback.webp 59 KB), web/build/v5_build_heads.py, web/dev/heads.html.
API: SCAComp.heads(el, {view: "curves" | "matrix", head: [block, head] (one-based; default [12, 4]), sigma: 0 | 1 | 2
(map noise level, default 1), lazy, base}) -> {select(block, head), view(v), destroy()}.
Fallback image: assets/data/heads/fallback.webp (1152 x 576, the curves drawn on the media ground, no labels).
What it shows: all 1,368 near-share curves (57 blocks x 24 heads, 11 noise levels after the gate) on a canvas,
coloured by mean near share (far blue, neutral, near orange), heads 4 and 7 of block 12 heavy in near/far colours,
the head average in ink; hover or tap shows block, head, double- or single-stream, mean near share; click selects;
arrow keys on the plot step through heads by reach rank; "Sorted by reach" shows the same numbers as a matrix (rows =
heads by mean share, columns = noise bins, average strip under it: rows are flat, so reach belongs to the head).
Beside it: the variance split from the paper's loader (97.5 / 2.5 / 0.7, plus the head share in FLUX.1-Krea-dev
97.6 and FLUX.2-dev 99.1 = Table B.1 rows), drawn as a bar with the noise part magnified (as Figure 2c). Under it:
the real full-attention row of the selected block-12 head at the sheep query, noise level scrubber (sigma 0.90, 0.61,
0.33), near part (within 16 tokens) orange, far part blue, the dashed 16-token neighbourhood, the near share of
that map; 24 chips pick block 12's heads (chip underline = mean reach colour).
Source: paper/figures/src/v6_observation.py load() (the paper's own loader and checks: variance_split.csv,
headline macros \hlVarHead / \hlVarStep / \hlVarAvg), v6_attn_block12.npz; nothing typed.
Checked: split shows 97.5% / 2.5% / 0.7% (= headline.tex); head 4 of block 12 is rank 82 of 1,368 (the paper's
ranks[3] = 81, zero-based); no console errors, no overflow; height 608 px at 1440x900; 738 px at 390x844 (limit
675; notes collapse behind "About these numbers" on narrow screens).
Rebuild: /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_build_heads.py (seconds).
Screenshots: /scratch/09032/saini_2/sca/web_builds/v5_webinteractive_1/shots/heads_{desk,phone}_{light,dark}_*.png

## 3. emulate: DONE
Files: web/assets/js/c_emulate.js, web/assets/css/c_emulate.css, web/assets/data/emulate/ (data.js 0.27 MB,
img/base.webp, fallback.webp 16 KB), web/build/v5_build_emulate.py, web/dev/emulate.html.
API: SCAComp.emulate(el, {head: 1..24 (default 10, the method figure's head), pi: 0..1 (default 0.3), lazy, base})
-> {set(pi), head(h), destroy()}.
Fallback image: assets/data/emulate/fallback.webp (1152 x 480, the three featured heads' J curves on the media ground).
What it shows: block 12 at the sheep query, step 20 (sigma 0.61), r = 4: the window view (16 x 16 window keys, orange
heat), the grid view (16 x 16 block-centre keys, blue dots by attention), their mix at pi (pi x window + (1 - pi) x
grid, the effective attention of the mixed output), and the head's full attention (orange inside the window, blue
outside). Under them the head's real J(pi) = E - 2 pi B + pi^2 A (archived statistics, 32 prompts, 262,144 query
samples, noise bin sigma 0.62) for r = 4 (solid) and r = 2 (dashed), rescaled to [0, 1] as the paper's method
figure does, the dot at the current pi, pi* = clip(B/A, 0, 1) marked; readout: J(pi) / J(pi*) (big), pi* (and B/A
when it clips), the error relative to the full-attention output energy at pi and at pi*. Snap to pi* animates there.
Head buttons: near-reaching head 4 (pi* 0.93), in between head 10 (0.79), far-reaching head 7 (0.00: B/A = -0.19,
clipped), plus a select for all 24 heads.
Supported by data: maps at r = 4 only (the npz holds r = 4 operators); r = 2 appears only as its J curve and pi*.
Honest edge it shows as is: head 7's emulation error is large (211% of its output energy at pi*, r = 4); the toy
prints it.
Source: paper/figures/src/data/v6_attn_block12.npz and results/evidence/calib/k1_1024.npz through the constants and
rules of paper/figures/src/v6_method_assets.py (head-10 rule, window, bin 8); pi* asserted equal to the paper's
s2_pi_head / s4_pi_head tables.
Checked: pi* 0.93 / 0.79 / 0.00 at r = 4 and 0.95 / 0.83 / 0.06 at r = 2 (= the tables); no console errors, no
overflow; 684 px at 1440x900; 800 px at 390x844 (2 x 2 maps; notes collapse on narrow screens).
Rebuild: /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_build_emulate.py (seconds).
Screenshots: /scratch/09032/saini_2/sca/web_builds/v5_webinteractive_1/shots/emulate_{desk,phone}_{light,dark}_*.png

## 4a. table: DONE
Files: web/assets/js/c_table.js, web/assets/css/c_table.css, web/assets/data/table/data.js (8 KB),
web/build/v5_build_table.py, web/dev/table.html.
API: SCAComp.table(el, {tab: "main" | "gen", hide: ["time" | "pref" | "detail" | "set" | "holm"], lazy, base}) ->
{tab(t), destroy()}. On containers under 560 px the Detail and Set level groups start hidden (chips bring them back).
No fallback image: the page's own static Table 1 (or a link to results.html) is the fallback; on data failure the
component removes .is-live from its figure (or prints one line).
What it shows: tab 1 = paper Table 1 (tab_main_compact.tex) cell for cell: groups "Single pass, AReA's sampler" and
"Multi-stage, own recipes", columns s / vs. ours / IR / HPSv2.1 / Pick / M_c / FID / FID_p / +-; bold = best,
underline = second per group, double dagger = campaign mean, our rows shaded green with a green rule; the Holm
record drawn as ten cells (green = progressive SCA significantly better than the row, dark = worse, grey =
neither) with "10 / 0" beside it, "ref." for progressive SCA. Tab 2 = Table 3 (tab_generality.tex): paired
differences IR / HPS / Pick with n and the Holm record; significant cells (paper's \up) in accent ink.
Sorting: header click = best first (direction-aware), again = worst first, again = paper order; sorted rows carry a
"Single pass" / "Multi-stage" tag. Group chips hide or show column groups (at least one stays).
Source: the paper's generated snippets (gen_main_tables.py output) and the n macros of paper/main.tex; the build
asserts every time cell against fig_speed.load() (clean / campaign) and every IR cell against the mean per-image IR
of the 195 prompts (v6_common.pool195). Nothing typed.
Checked: paper order IR -2.18, -0.10, 0.56, 0.92, 1.25, 1.28, 1.34, 1.32; sort by time puts progressive SCA (49.2)
first and UltraImage (308.5, campaign) last; no console errors; no page overflow (the table scrolls inside its own
frame on phones, method column sticky); 660 px at 1440x900, 765 px at 390x844.
Rebuild: /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_build_table.py
Screenshots: /scratch/09032/saini_2/sca/web_builds/v5_webinteractive_1/shots/table_{desk,phone}_{light,dark}_*.png

## 4b. speed: DONE
Files: web/assets/js/c_speed.js, web/assets/css/c_speed.css, web/assets/data/speed/data.js (2 KB),
web/build/v5_build_speed.py, web/dev/speed.html.
API: SCAComp.speed(el, {side: 4096 (or null for no readout at first), lazy, base}) -> {show(side), destroy()}.
What it shows: the paper's Figure 7 data as an SVG log-log chart: single-pass SCA (accent line, hollow markers,
b = 0.99), AReA (purple, b = 1.12), full attention (neutral, b = 1.60, not run at 8192^2), whiskers = fastest and
slowest of three timed images; progressive SCA (accent star, 49.2 s) and HRDiT (107.0 s) at 4096^2. Hover, tap or
arrow keys pick a size: a readout lists every method's seconds there, single-pass SCA's speed-up over AReA (2.07,
2.58, 2.63, 2.76, 3.07) and at 4096^2 progressive SCA's 4.1x over AReA and 2.2x over HRDiT (sweep 3). A row of
ratio chips under the x axis (aligned to sizes on desktop, a strip on phones).
Source: paper/figures/src/v6_scaling.py load / derived / checks (the build runs the paper's checks: tab:latency,
headline macros, exponents, "fastest at every size"); nothing typed.
Checked: values above; no console errors; no overflow; 466 px at 1440x900, 610 px at 390x844.
Rebuild: /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_build_speed.py
Screenshots: /scratch/09032/saini_2/sca/web_builds/v5_webinteractive_1/shots/speed_{desk,phone}_{light,dark}_*.png

## 5. views (extra, method step 1): DONE
Not in the brief's list; built because Section 3 asks for the method in three steps and each step should be
playable: heads = the observation, views = step 1 (two native-size views), emulate = step 2.
Files: web/assets/js/c_views.js, web/assets/css/c_views.css, web/assets/data/views/ (data.js 1 KB, img/base.webp
0.63 MB = the 4096^2 picture at 2048 px, img/grid_r{1,2,4}.webp 0.1 to 0.23 MB, fallback.webp), web/build/v5_build_views.py,
web/dev/views.html. Data 1.5 MB.
API: SCAComp.views(el, {size: 1024 | 2048 | 4096 (default 4096), lazy, base}) -> {size(s), destroy()}.
Fallback image: assets/data/views/fallback.webp (1200 x 1200, the picture with the 4096^2 window box and grid dots).
What it shows: progressive SCA's 4096^2 output for the sheep prompt (the image of the paper's attention figures);
drag, click or tap moves the query (arrow keys: one tile). Orange box = the native-size window (n0 x n0 = 64 x 64
tokens around the query's 8 x 8 tile, clamped; sca/attn.py window_index), blue squares = the grid view's keys (one per
r x r block at kappa(b) = r b + floor(r/2); sca/attn.py pool_tokens(center)). Insets: the window's pixels (cropped
from the picture) and the grid view's keys laid side by side (the 16-px patch of every block-centre token). Counters:
keys in full attention at that size (65,536 at 4096^2, 16,384 at 2048^2, 4,096 at 1024^2) against 4,096 in each view.
Size buttons 1024^2 / 2048^2 / 4096^2 (r = 1, 2, 4); at 1024^2 both views are the whole image (= full attention).
Honest note printed: the smaller sizes reuse the 4096^2 picture resized; boxes and dots are exact for each size.
Geometry constants come from code (attn_focus_data.N0, TILE; v6_scaling.PATCH); nothing typed.
Checked: counters above; no console errors; no overflow; 675 px at 1440x900; 896 px at 390x844.
Rebuild: /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_build_views.py
Screenshots: /scratch/09032/saini_2/sca/web_builds/v5_webinteractive_1/shots/views_{desk,phone}_{light,dark}_*.png

## 6. stages (extra, method step 3): DONE
Files: web/assets/js/c_stages.js, web/assets/css/c_stages.css, web/assets/data/stages/data.js (11 KB),
web/build/v5_build_stages.py, web/dev/stages.html.
API: SCAComp.stages(el, {step: 0..56 (default 47, the first 4096^2 step), lazy, base}) -> {step(i), destroy()}.
No fallback image (the method figure or the method_stages loop is the natural fallback).
What it shows: the real progressive SCA run of the sheep prompt step by step (its sidecar: 30 steps at 1024^2 from
sigma 1, 17 at 2048^2 from 0.81, 10 at 4096^2 from 0.61): a timeline of sigma per step (three descents, each later
stage re-noised) with the mean weight over all 1,368 heads at every SCA step; a readout (step, size, sigma, "reads the
table at r = 2 / 4" or "the model's own attention, no table"); block 12's row of the one measured table (24 heads by
mean r = 4 weight x 13 populated noise bins, colour = pi from far blue to near orange) at the stage's r, a line at the
step's sigma, and a "now" column with every head's weight at that step. Play the run (380 ms per step; hidden under
reduced motion), slider, arrow keys, click on the timeline.
Source: the run's sidecar (/scratch/09032/saini_2/sca/out/dupfix_prog/sca_v2_prog/4096x4096/
a4k_eval_4096_minus8343371271909676658_0325f0a692_seed0.json), the table results/evidence/calib/k1_1024.npz (sha256
asserted equal to the run's pi_table_sha256), weights computed by sca.attn.PiTable (the sampler's own lookup) and the
mean over all heads asserted equal to the run's pi_trace at all 27 SCA steps; the stage settings asserted (window 64
at these noise levels: the hand-set wider windows act only above sigma 0.82). Nothing typed.
Checked: readouts above (step 48 = first 4096^2 step, sigma 0.61, mean pi 0.73); no console errors; no overflow;
642 px at 1440x900, 594 px at 390x844.
Rebuild: /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_build_stages.py (imports torch, ~30 s).
Screenshots: /scratch/09032/saini_2/sca/web_builds/v5_webinteractive_1/shots/stages_{desk,phone}_{light,dark}_*.png

## Integration on the page (checked with web/build/v5_check_components.py --page index.html)
heads, emulate, percentiles and table are mounted by the designer and go live (is-live) at 1440x900 and 390x844 in
both themes with no console errors from the components (the only request failures are the .mp4 files: Playwright's
Chromium has no H.264, so they are test artifacts, not page errors). I reran web/build/build_v5_assets.py (it says
"rerun after either owner adds files") so v5_assets.js lists all six components.
Data-failure path checked (dev pages with ?o_base=../nope/): percentiles, heads, emulate drop .is-live and the page's
fallback shows again; table and speed print one line. Reduced motion checked: zoom flights and Snap to pi* jump.

## Requests to the designer
- percentiles: mounted (thanks). At 1344 px wide it is 759 px tall (the four images grow with the width); at 1200 px
  it is 712 px. If the section can cap the figure at 1200 px it stays under 80vh at 1440x900.
- speed: please mount it in Results next to the table: <figure class="comp-figure"> with the paper's scaling figure
  (assets/img/v6/scaling.*) as .comp-fallback and <div class="comp-mount" data-comp="speed" data-options='{"side": 4096}'>.
  It replaces the retired speed plot; numbers come from the paper's Figure 7 loader.
- views: please mount it as method step 1 (two native-size views), before emulate:
  data-comp="views" data-options='{"size": 4096}', fallback <img class="comp-fallback"
  src="assets/data/views/fallback.webp" width="1200" height="1200" alt="A 4096 by 4096 image with the window view
  (orange box) and the grid view (blue dots) of one query.">. It can sit beside or replace the method_window loop.
- stages: please mount it as method step 3 (one table for any size and sampler stage), after emulate:
  data-comp="stages" data-options='{}'; fallback = the method_stages loop or paper Figure 3. With views (step 1),
  emulate (step 2) and stages (step 3) every method step is playable, and heads carries the observation.
- table: its own fallback is your static table; nothing else needed. On phones it starts with Detail and Set level
  hidden (chips bring them back) and scrolls inside its frame.
- emulate: the fallback slot you used (method_emulation video) is fine; fallback.webp exists too
  (assets/data/emulate/fallback.webp, 1152 x 480).
- Components read --bg-raised, --bg-sunk, --ink, --ink-soft, --ink-faint, --rule, --accent, --accent-ink,
  --accent-soft, --on-accent, --near, --far, --area, --hrdit, --neutral, --screen, --screen-ink, --radius-sm,
  --shadow, --font-sans, --font-serif, --ease. Canvas components re-read them on sca:themechange and on changes of
  data-theme on <html>, so .band-dark sections work too.
