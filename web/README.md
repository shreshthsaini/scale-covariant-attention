# SCA project page

A static page for the paper: `web/index.html`, plus a second page `web/review/index.html` (96 evaluation prompts,
five methods side by side; the lead's page, linked from the nav, the results and the gallery). No build step at view
time and no framework: open `web/index.html` directly (file:// works) or serve the folder, for example
`python -m http.server -d web 8000 --bind 127.0.0.1`. Images are WebP, videos H.264 MP4, fonts WOFF2 (Inter and
Source Serif 4, SIL OFL, licence texts in `assets/fonts/`). Bulky intermediates live on scratch under
`/scratch/09032/saini_2/sca/web_assets/`.

**What the page shows.** The headline is **progressive SCA** (`sca_v2_prog`): HRDiT's released three-stage schedule
(30 steps at 1024^2 with the model's own attention, then the last 17 steps at 2048^2 and the last 10 at 4096^2) with
SCA attention at the two upper stages, reading ratio 2 and ratio 4 from one reach table. **Single-pass SCA**
(`sca_v2`) is the fast mode. Every SCA image names its configuration. Sections, in order: title and teaser loop; hero
images (5120^2 and 6144x4096) as zoom grids; four headline numbers; **speed** (the paper's quality-against-time plot
with speed-up arrows and its latency-against-size plot, `assets/img/figs/speed_{time,latency}_{light,dark}.webp`,
exported by the paper owner from paper/figures/src/fig_speed.py); abstract; the explainer video; the observation (all
four methods at 4096^2 in one zoom grid and the variance split); the method (the paper owner's attention-focus figure,
`assets/img/figs/attn_focus_{light,dark}.webp` from paper/figures/src/fig_attn_focus.py, with a four-step walk through
near view, far view, reach and reach emulation; then the paper's method figure, the two views to scale, the
closed-form fit, the reach table and both schedules; the 1,368-curve visuals are retired); the showcase (cards from
build/showcase_manifest.json via build_showcase_page.py); comparisons at 4096^2 (15 prompts, eight methods in one
zoom grid: progressive and single-pass SCA, AReA, HRDiT, UltraImage, DyPE, SEGA, full attention); per-example videos;
results; the fixed-rule gallery (progressive SCA and AReA in a zoom grid); Ideogram placeholders; limitations;
citation. Native images upscaled with Lanczos are never shown (user directive 2026-11-08): comparisons are against
methods only, and for FLUX.2-dev and Qwen-Image against the same model sampled directly at the target size.

**Synced zoom grids** (`assets/js/zoomgrid.js`, `assets/css/zoomgrid.css`, shared with `review/`). Every multi-method
comparison is a grid of panels that share one view: scroll or pinch on any panel to zoom, drag to pan, double-click or
double-tap to reset; every panel shows the same region. A small map in the first panel shows where the view is (drag
it to move). **Auto zoom** flies (van Wijk and Nuij's smooth zoom path) through the grid's stops, holding at each at
about one native pixel per device pixel; any input stops it and leaves the view in place. The stops come from an
edge-energy map computed at build time (`build/zoomlib.py`: per image the mean-normalised |dx| + |dy| at half
resolution, averaged over the grid's images; the three windows with the most energy, at least 1.4 windows apart),
the same stops for every method. At the whole-image view each panel is a plain lazy `<img>`; canvases exist only
while a grid is zoomed, and native-pixel tiles load only when the view needs more pixels than the overview holds.
Tiles: one 1024 px WebP (quality 80) per stop and image on the main page (640 px on the 96-prompt page, whose
panels are about 250 px wide); the showcase uses its four 768 px windows as stops and tiles, the hero its whole
images in 512 px tiles. Whole-image 4 x 4 tiling of every compared image would take web/ far past 250 MB (0.5 to
2 MB per image), and a private Hugging Face dataset cannot serve tiles to a public page, so the tiles are cut where
the tour goes.

## Wording rules (user directives, 2026-11-07)

- Never write calibration, calibrate or calibrated in anything a visitor sees (text, captions, alt text, videos). The paper's
  terms: each head's reach is measured once at the training resolution by **reach emulation** (an emulation pass at
  1024^2 that shrinks both views to the target's image-to-window ratio), fit in closed form against the model's own
  full attention; the result is the **reach table**. Paper figures go on the page only when their text is clean
  (checked with PyMuPDF; `method_v3`, `heads_view`, `method_zoom` and `pi_heads` are clean since 2026-11-07 21:55).
- Never write curated or uncurated, and say nothing about how an example was picked: show each example with its prompt
  and settings. Never label a selected example as random. A set drawn by a fixed rule may say so neutrally (the
  gallery: "the first 12 prompts of the evaluation set"; the second page: "drawn once with a fixed seed").
  `build_images.py` strips selection labels from `images.js`; `check_page.py` fails on these words in the visible text.
- No discussion of repeated subjects. No em dashes, no section symbol, sentences under about 36 words.
- Every number comes from `results/` through `build/build_numbers.py`; never type one by hand.
- Placeholders stay placeholders until the real asset exists: no synthetic stand-ins for images, stages or videos.

## Layout

- `index.html`: the page. Numbers sit in `data-num="<key>"` elements filled from `assets/js/numbers.js`; Table 1's
  classes come from the `<key>.cls` entries. The Paper button links to `../paper/main.pdf`.
- `assets/css/site.css`, `assets/js/site.js`: styles and the vanilla JS (lens, sliders, charts, carousels, lightbox).
- Generated data (do not edit): `numbers.js`, `images.js`, `media.js` (which videos exist), `showcase.js`,
  `process.js`, `traces.js` (single-pass videos, shown until the progressive ones exist), `gallery.js`, `zoomsets.js`
  (the comparison and gallery grids: stops, overviews, tiles), `method_data.js`, `ideogram_manifest.js`.
- `assets/img/deep/<set>/<sid>/<method>/`: tour-stop tiles (`r<k>.webp`) and overviews (`v.webp`) of the zoom grids
  that had no overview on the page yet; `assets/img/figs/`: figures exported by the paper owner (light and dark).
- `assets/img/`: `hero/`, `compare/` (11 prompts, six methods), `qual/` (four prompts, four methods, 1600 px), `gallery/`
  (12 prompts), `method/` (paper figures), `scale/`, `showcase/` (build_showcase.py), `zoom/` (lens tiles),
  `teaser/`, `failures/` (kept for the record, not shown).
- `assets/video/`: `teaser/` (three loops), `method/` (explainer), `process/` (per-example videos), `trace/` (single-pass
  videos).
- `review/`: the second page (`index.html`, `items.json` from `build/review_gallery.py`, `items.js` and `img/z/` from
  `build/build_zoom.py review`).

## Regenerating every asset

Run from the repository root with `PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python`, CPU only, under
`nice -n 19`, ffmpeg at `-threads 4` (vendored build `/work/09032/saini_2/vista/library/ffmpeg-7.1-shared`).

1. **Numbers**: `$PY web/build/build_numbers.py --fill web/index.html` (also writes `build/numbers.json` and prints each
   value next to the value the paper quotes).
2. **Comparison and figure images**: `$PY web/build/build_images.py compare|qual|hero|method|scale|all` (see its
   docstring). `compare` adds UltraImage (`out/main/ultraimage`) to the 11 comparison prompts; `qual` re-encodes the
   paper's four-method grid at 1600 px; `method` rasterises `method_v3`, `method_zoom`, `pi_heads` and `heads_view`
   (set `SCA_PDF_PY` to an env with PyMuPDF, for example `uv-envs/csd-figs`).
3. **Method data**: `$PY web/build/build_method_data.py` (reach table at ratios 2 and 4, the three ring heads with their
   exact A, B, E, F sums, the rings). `build_concept.py` (the 1,368-curve chart) is retired from the page.
4. **Gallery**: `$PY web/build/build_gallery.py --n 12` (first 12 prompts of
   `results/evidence/prompts/aesthetic4k_eval.jsonl`, progressive SCA and AReA at seed 0, central 512 px crops).
5. **Showcase**: `build/build_showcase.py` (showcase owner; manifest, overview WebP and four
   768 px native crops per image), then `$PY web/build/build_showcase_page.py` (page side: zoom grids, whole-image tiles for the large
   images, `assets/js/showcase.js`; writes an empty `SCA_SHOWCASE` until the manifest exists).
6. **Videos**: `$PY web/build/render_explainer.py` (the explainer and `teaser/heads_loop.mp4`),
   `$PY web/build/render_process_videos.py` (per-example videos from the progressive and AReA traces under
   `/scratch/09032/saini_2/sca/web_assets/trace/`, `assets/js/process.js`, `teaser/zoom_loop.mp4` and
   `teaser/stages_loop.mp4`), then `$PY web/build/build_media.py` so the page picks them up.
7. **Zoom grids**: `$PY web/build/build_zoom.py compare gallery review --procs 12` (a few minutes; reuses the
   overviews of steps 2 and 4, caches the stops on scratch under `web_assets/zoom_cache/`, rewrites only stale
   files). Rerun `build_showcase_page.py` after the showcase manifest changes; it writes the showcase grids.
8. **Check**: `PLAYWRIGHT_BROWSERS_PATH=/scratch/09032/saini_2/tmp/ms-playwright
   /work/09032/saini_2/vista/library/uv-envs/pwqa/bin/python web/build/check_page.py --http --sections --out DIR`
   (desktop and phone, light and dark, and the 96-prompt page: missing files, broken images, console errors, overflow,
   forbidden words; wheel zoom, drag, a synthetic two-finger pinch, double-click reset and auto zoom on a grid;
   screenshots). Also open the page once over file:// (fonts load there; font preloads do not).

## Against a commercial model: adding the Ideogram 4.0 images

The page sets progressive SCA (FLUX.1-dev, 4096^2, one GPU, no training) beside Ideogram 4.0 on six prompts. Until the
images exist the section shows labelled placeholders; nothing is fabricated, and Ideogram's own published samples are
not used because their pages grant no permission to reuse them.

**How to obtain the images.** In the Ideogram app (ideogram.ai) or through the Ideogram API, select the Ideogram 4.0
model, paste each prompt below exactly as written (no "magic prompt" or prompt expansion, no style preset, no
negative prompt), choose a 1:1 aspect ratio at the largest size offered (2048 x 2048 in the API as of this writing;
2K in the app), and generate one image per prompt, keeping the first result. Save the full-resolution file as
`web/assets/ideogram/<prompt_id>.png` (or .jpg or .webp). For an image at 4096 x 4096, run Ideogram's own upscaler 2x
on that same image, save that instead, and set `"upscaled": "2x (Ideogram Upscale)"` in the manifest. Then run
`$PY web/build/build_ideogram.py`: it converts the files to WebP without resizing, moves the originals to
`/scratch/09032/saini_2/sca/web_assets/ideogram_originals/`, records each resolution and refreshes
`assets/js/ideogram_manifest.js` (the page loads an Ideogram file only once its resolution is recorded). Finally
fill `source`, `upscaled`, `generated_on` and a one-line `verdict` per prompt in `build/ideogram_manifest.json` (what
each image does better, judged at full size and in the crop), and rerun the script. The crop is 512 x 512 native
pixels centred on the same relative point as the SCA crop; set `crop_center` (`[x, y]` as fractions of width and
height) when the compositions differ. Ideogram's terms of service (ideogram.ai/legal/tos) assign the rights in your
own outputs to you; keep the rights note in the manifest accurate.

| prompt_id | prompt (use verbatim) |
|---|---|
| 2232973817337527860 | A bouquet of pink lilies is arranged in a clear vase filled with water, placed on a white side table next to a partially opened curtain and a closed door with vertical paneling, illuminated by soft light. |
| m8737279395568207187 | A grand interior with soaring arches, intricate stonework, and vibrant stained glass windows, illuminated by ornate chandeliers and soft light, creating an atmosphere of reverence and awe. |
| 5849277588805396896 | A smiling man with a gray beard and a hat plays an acoustic guitar, dressed in a stylish blazer over a dark shirt against a muted background. |
| m8343371271909676658 | A group of sheep stands in a snow-covered landscape in front of an ornate stone church with a spire, framed by colorful trees and quaint houses in the background. |
| 997293625843621481 | A climber, wearing a helmet and blue clothing, ascends through a narrow, icy crevice, using an ice axe for support amidst glacial blue ice formations. |
| m548913239957610051 | A cascading waterfall flows over rocky cliffs into a tranquil river, framed by a steep, conical mountain under a colorful sky at sunset. |
