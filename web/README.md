# SCA project page

A static page for the paper. No build step at view time and no framework: open `web/index.html` directly
(file:// works) or serve the folder, for example `python -m http.server -d web 8000 --bind 127.0.0.1`.
Images are WebP, videos H.264 MP4. Everything under `web/` stays under 60 MB; bulky intermediates live on
scratch under `/scratch/09032/saini_2/sca/web_assets/`.

## Layout

- `index.html`: the page. Numbers sit in elements with `data-num="<key>"` and are filled from the CSVs.
- `assets/css/site.css`, `assets/js/site.js`: styles (one accent, light and dark) and the vanilla JS for the
  teaser magnifier, the before/after sliders, the charts and the data-driven galleries.
- `assets/js/images.js`, `numbers.js`, `traces.js`, `ideogram_manifest.js`: generated data files (do not edit).
- `assets/img/`: teaser, comparison, scale strip, failure and method images (WebP).
- `assets/video/method/`: the method video, its poster and subtitles. `assets/video/trace/`: per-sample videos.
- `assets/ideogram/`: drop folder for the Ideogram 4.0 images (empty until they are added).
- `build/`: every script that produces the assets, plus the selection files they read.

## Regenerating every asset

Run from the repository root. `PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python`. CPU only unless
marked; run with `nice -n 19` on shared nodes and keep ffmpeg at `-threads 4`.

1. **Comparison prompts and crop windows**: `$PY web/build/select_compare.py` writes
   `build/compare_selection.json`: the 8 curated prompts (from `.agent/notes/figure_curation_20261020.md`), 3 random
   draws (seed 20261027 over the other 195 Aesthetic-4K prompts minus the 4 development prompts) and one shared
   512 x 512 crop per prompt (largest mean edge response over SCA, AReA and full attention, inside the central
   60 percent of each axis).
2. **Images**: `$PY web/build/build_images.py all` converts the generated PNGs on scratch into WebP (1600 px views,
   native 1:1 crops, the 6144 px native panorama for the teaser magnifier) and writes `assets/js/images.js`.
   Teaser, scale-strip and failure picks and their crop windows are listed in `build/page_images.json`. The method
   figures are rasterised from `paper/figures/method.pdf` and `method_zoom.pdf` with PyMuPDF (set `SCA_PDF_PY`
   to any env that has it).
3. **Numbers**: `$PY web/build/build_numbers.py --fill web/index.html` reads `results/` and writes
   `build/numbers.json` (value, display string, source CSV and row filter for every key) and
   `assets/js/numbers.js`, then rewrites every `data-num` element in the page. It prints each value next to the
   value the paper quotes and flags mismatches. Never type a number into the page by hand.
4. **Method video**: `$PY web/build/render_method_video.py` (matplotlib frames piped to the vendored ffmpeg at
   `/work/09032/saini_2/vista/library/ffmpeg-7.1-shared`) writes `assets/video/method/sca_method.mp4`, the SRT and
   WebVTT subtitles and the poster. It uses the real calibrated table `/scratch/09032/saini_2/sca/calib/k1_1024.npz`
   and real generated images, and reads its end-card numbers from `build/numbers.json`.
5. **Per-sample trajectory videos** (GPU): see below.
6. **Ideogram images**: `$PY web/build/build_ideogram.py` (see the next section).
7. **Check**: `PLAYWRIGHT_BROWSERS_PATH=/scratch/09032/saini_2/tmp/ms-playwright
   /work/09032/saini_2/vista/library/uv-envs/pwqa/bin/python web/build/check_page.py --sections` loads the page in
   headless Chromium (desktop and phone, light and dark), reports missing files, broken images, console errors
   and horizontal overflow, and saves screenshots to `/scratch/09032/saini_2/sca/web_assets/check/`.

## Per-sample trajectory videos (needs a GPU)

`scripts/sample_trace.py` re-runs a prompt at 4096^2 with the campaign seed and code path and saves, at every one
of the 28 steps, the decoded estimate of the clean image (tiled VAE decode, 1024^2 preview), a 1:1 crop at the
comparison window and the per-head weights in use. `web/build/make_trace_tasks.py` writes 16 spool tasks
(8 prompts, SCA and AReA) named `w_trace_*`. Because `taskq/STOP` was set under the SU freeze, they wait in
`/scratch/09032/saini_2/sca/taskq/hold_web/`; release them into `taskq/pending/` only when GPU work is allowed
again. When the traces exist, `$PY web/build/render_trace_videos.py` renders `assets/video/trace/<id>_sca.mp4` and
`<id>_vs_area.mp4` with posters and rewrites `assets/js/traces.js`; the page then swaps its placeholders for the
videos without other changes.

## Against a commercial model: adding the Ideogram 4.0 images

The page compares SCA (FLUX.1-dev, 4096^2, one GPU, no training) with Ideogram 4.0 on six prompts. Until the images
exist the section shows labelled placeholders; nothing is fabricated, and Ideogram's own published samples are not
used because their pages grant no permission to reuse them.

**How to obtain the images.** In the Ideogram app (ideogram.ai) or through the Ideogram API, select the Ideogram 4.0
model, paste each prompt below exactly as written (no "magic prompt" or prompt expansion, no style preset, no
negative prompt), choose a 1:1 aspect ratio at the largest size offered (2048 x 2048 in the API as of this writing;
2K in the app), and generate one image per prompt, taking the first result rather than the best of several. Save
the full-resolution file as `web/assets/ideogram/<prompt_id>.png` (or .jpg or .webp). If you also want an image at
4096 x 4096, run Ideogram's own upscaler 2x on that same image, save that instead, and set `"upscaled": "2x
(Ideogram Upscale)"` in the manifest. Then run `$PY web/build/build_ideogram.py`: it converts the files to WebP
without resizing, moves the originals to `/scratch/09032/saini_2/sca/web_assets/ideogram_originals/`, reads each
resolution and refreshes `assets/js/ideogram_manifest.js`. Finally fill `source`, `upscaled`, `generated_on` and a
one-line `verdict` per prompt in `build/ideogram_manifest.json` (what each image does better, judged at full size
and in the crop), and rerun the script. The crop is 512 x 512 native pixels of each image, centred on the same
relative point as the SCA crop; set `crop_center` (`[x, y]` as fractions of width and height) when the two
compositions differ. Ideogram's terms of service (ideogram.ai/legal/tos) assign the rights in your own outputs to
you; keep the licence note in the manifest accurate.

| prompt_id | prompt (use verbatim) |
|---|---|
| 2232973817337527860 | A bouquet of pink lilies is arranged in a clear vase filled with water, placed on a white side table next to a partially opened curtain and a closed door with vertical paneling, illuminated by soft light. |
| m8737279395568207187 | A grand interior with soaring arches, intricate stonework, and vibrant stained glass windows, illuminated by ornate chandeliers and soft light, creating an atmosphere of reverence and awe. |
| 5849277588805396896 | A smiling man with a gray beard and a hat plays an acoustic guitar, dressed in a stylish blazer over a dark shirt against a muted background. |
| m8343371271909676658 | A group of sheep stands in a snow-covered landscape in front of an ornate stone church with a spire, framed by colorful trees and quaint houses in the background. |
| 997293625843621481 | A climber, wearing a helmet and blue clothing, ascends through a narrow, icy crevice, using an ice axe for support amidst glacial blue ice formations. |
| m548913239957610051 | A cascading waterfall flows over rocky cliffs into a tranquil river, framed by a steep, conical mountain under a colorful sky at sunset. |

## Rules the page follows

- Curated images are labelled curated and were checked at native resolution; random draws are labelled random.
- Every number comes from `results/` through `build_numbers.py`; the failure rate comes from
  `results/visual_audit_20261020`.
- Placeholders stay placeholders until the real asset exists: no forward-noised or synthetic stand-ins.
- No em dashes or section symbols in the copy.
