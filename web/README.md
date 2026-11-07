# SCA project page

A static page for the paper. No build step at view time and no framework: open `web/index.html` directly
(file:// works) or serve the folder, for example `python -m http.server -d web 8000 --bind 127.0.0.1`.
Images are WebP, videos H.264 MP4. Everything under `web/` stays under about 70 MB (68 MB on 2026-11-07, 49 MB of
it video); bulky intermediates live on scratch under `/scratch/09032/saini_2/sca/web_assets/`.

**Configurations (paper rewrite of 2026-11-07).** The headline is **progressive SCA** (method `sca_v2_prog`, run
`results/prog195_sca_v2`): HRDiT's released three-stage schedule (30 steps at 1024^2 with the model's own attention,
then the last 17 steps at 2048^2 and the last 10 at 4096^2) with SCA attention at the two upper stages, reading
ratio 2 and ratio 4 from one calibration table. **Single-pass SCA** (`sca_v2`, the final operator: 28 steps, global
opening for 8, windows of 128, 112, 96, 80 tokens before the native 64, 1,024 bridge tokens, calibrated per-head
mix) is the fast mode. What exists only for the operator without the schedule (`sca_nb`) is labelled "previous
operator": 6144^2, FLUX.2-dev and the four curated local errors. FLUX.1-Krea-dev is single-pass only (the
progressive sampler does not run it). The hero shows the paper's curated progressive SCA images
(`paper/figures/hero_prog/provenance.json`); the side-by-side viewer and the comparison grid show progressive SCA
beside the baselines; the single-pass showcase (the former hero) sits in the 2K-to-8K section, labelled single-pass.
Every other SCA image names its configuration.

**What the page leaves out.** The page does not discuss repeated subjects: the duplication section, its charts,
before/after pairs, flagged failures, the Moai comparison and every duplication rate were removed on 2026-11-07,
and the method video was re-rendered without its Moai scenes.

## Layout

- `index.html`: the page. Numbers sit in elements with `data-num="<key>"` and are filled from the CSVs; Table 1's
  bold, underline and significance colours come from the `<key>.cls` entries. The Paper button links to
  `../paper/main.pdf`, the committed delivery PDF; when hosting `web/` on its own, copy the PDF next to the page and
  change that link.
- `assets/css/site.css`, `assets/js/site.js`: styles (one accent, light and dark) and the vanilla JS for the
  teaser magnifier, the before/after sliders, the concept chart, the other charts and the data-driven galleries.
- `assets/js/images.js`, `numbers.js`, `traces.js`, `concept.js`, `ideogram_manifest.js`: generated data files (do
  not edit).
- `assets/img/`: hero (progressive SCA, with one native-pixel image for the magnifier), teaser (the single-pass
  showcase), comparison, qual (the comparison grid), scale strip, local-error and method images (WebP).
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
   60 percent of each axis). `select_compare.py --add-final` then adds, without moving any window, the single-pass
   operator's image of each prompt and seed (`paths.sca_v2`); `--add-prog` adds progressive SCA (`paths.prog`,
   `out/dupfix_prog/sca_v2_prog`) and HRDiT (`paths.hrdit`, the image `eval/prog195_sca_v2` scored) with their
   SHA-256 sums. The windows were chosen on the previous operator's images, before the other images existed.
2. **Images**: `$PY web/build/build_images.py all` converts the generated PNGs on scratch into WebP (1600 px views,
   native 1:1 crops, the 6144 px native panorama for the teaser magnifier) and writes `assets/js/images.js`. The
   comparison viewer gets five methods per prompt (single-pass SCA, AReA, full attention, progressive SCA, HRDiT).
   `hero` re-encodes the paper's curated progressive hero (`paper/figures/hero_prog/provenance.json`, sources checked
   by SHA-256): the large image at 1600 px plus a full native WebP for the magnifier, six tiles at 800 px and the two
   native inset crops. `qual` re-encodes the paper's comparison grid (`paper/figures/qual_prog/provenance.json`: AReA,
   UltraImage, HRDiT and progressive SCA on four curated prompts, each image at 720 px plus a native crop of the
   row's shared window). The single-pass showcase (`teaser` job) has no native magnifier file, to keep `web/` under
   the size budget (`"native": false` in `page_images.json`).
   Teaser, scale-strip and local-error picks and their crop windows are listed in `build/page_images.json`
   (`"final": true` takes the final operator's image of the same prompt and seed; the scale strip reads the
   second timing sweep `out/latency_curve_v2/<size>/sca_v2`). The method figures are rasterised at 2400 px (twice the
   display width) from `paper/figures/method_v2.pdf` (the paper's method figure since 2026-11-07), `method_zoom.pdf`
   and `pi_heads.pdf` with PyMuPDF
   (set `SCA_PDF_PY` to any env that has it). Decoding the 4096^2 PNGs is CPU work: run it on a compute node that
   is not the operator host. `write_js` drops the retired `dup_pairs` and `failures_v2` records.
3. **Numbers**: `$PY web/build/build_numbers.py --fill web/index.html` reads `results/` and writes
   `build/numbers.json` (value, display string, source CSV and row filter for every key) and
   `assets/js/numbers.js`, then rewrites every `data-num` element in the page and its managed classes. It prints
   each value next to the value the paper quotes and flags mismatches (on 2026-11-07: 331 quoted, 330 agree, one
   explained label difference). Inputs beyond the evaluation CSVs:
   - progressive SCA: `results/prog195_sca_v2` (Table 1 row; FID and patch FID from `corrected_fidp`),
     `results/prog195_vs_baselines/sca_v2_prog_vs_<baseline>.csv` (per-image pairs, 2000 draws, keys
     `prog.vs.<b>.*`), `results/prog195_vs_baselines/cfid/paired_deltas.csv` (set-level pairs against every
     baseline, 1000 draws, 168 reference identities, keys `prog.set.<b>.*`; before it existed,
     `results/prog195_pairs/cfid_vs_{area,sca_v2}`), `results/prog_latency/latency_summary.csv` (clean sweep 3:
     progressive SCA, HRDiT and AReA back to back; Table 1's clean column for HRDiT and progressive SCA, keys
     `prog.clean.*`; each median is rounded half up from its decimal string, 49.15 to 49.2 as the paper prints it,
     and the ratios use the rounded values), the 195 progressive sidecars (`out/dupfix_prog/sca_v2_prog/4096x4096`, stage schedule, keys
     `prog.cfg.*`) and the sidecars of the HRDiT images `eval/prog195_sca_v2` scored (`main.hrdit.steps`);
   - the paper's paired-difference figure: `chart.pa.{prog,single}.*`, built as
     `paper/figures/src/fig_paired_prog.py` does (14 metrics, percent of AReA's mean);
   - the head-versus-noise variance split: `results/heads_view/variance_split.csv` (keys `heads.var.*`);
   - Table 1 classes: `main.<row>.<column>.cls` ("best"/"second" within a group, "sig-up"/"sig-dn" against
     progressive SCA, as `paper/figures/src/gen_main_tables.py` colours the paper's table);
   - two scratch inputs of the single-pass runs, as in `paper/figures/src/scav2_runs.py`: the eval.py paired deltas
     of the `*_sca_v2` runs (`/scratch/09032/saini_2/sca/eval/<run>/{base/paired_deltas,extra/extra_paired_deltas}.csv`)
     and the `sca_v2` generation sidecars (`out/dupfix/sca_v2/4096x4096`).
   - progressive SCA at seed labels 1 and 2 (`results/prog195_s1`, `prog195_s2`): per-image pairs and the patch FID
     gap from each run's `corrected_fidp_ci/paired_deltas.csv` (AReA minus progressive SCA, negated), keys
     `prog.seed<k>.*` and `prog.seeds.*`.
   Results the paper lists as pending print "pending" until their files exist: progressive SCA on HRGen-1K
   (`prog.later.hrgen.status`). When it lands, its keys appear in `numbers.json` and the page sentence that prints
   "pending" needs its numbers added. Never type a number into the page by hand.
4. **Concept chart data**: `$PY web/build/build_concept.py` turns `results/heads_view/concept_curves.csv` (the data
   of the paper's concept figure: each of the 1,368 heads' ratio-4 window weight on 80 noise levels) into
   `assets/js/concept.js`. Curves are stored as bytes for drawing only; the median and each head's mean are exact.
5. **Method video**: `nice -n 19 $PY web/build/render_method_video.py` (matplotlib frames piped to the vendored ffmpeg
   at `/work/09032/saini_2/vista/library/ffmpeg-7.1-shared`, `-threads 4`; a few minutes on one CPU) writes
   `assets/video/method/sca_method.mp4` (67.6 s, 1920 x 1080, 30 fps, H.264, silent, captions drawn in the frame),
   `sca_method.srt`, `sca_method.vtt` and the poster. Five scenes: full attention's failure at 4096^2, the two
   views of a head, calibration by emulation with the real scale-4 table filling in, the single-pass schedule at
   4096^2 (per-step weights from the scale-4 table and, on the four wide-window steps, the log-linear mix of the
   scale-2 and scale-4 tables, checked against the final image's `pi_trace`), and an end card whose speed lines
   (single pass; progressive SCA) come from `build/numbers.json`. The tables and images are real; token grids are
   schematic. Image cache: `/scratch/09032/saini_2/sca/web_assets/method_video/cache/`.
6. **Per-sample trajectory videos** (GPU): see below.
7. **Ideogram images**: `$PY web/build/build_ideogram.py` (see the next section).
8. **Check**: `PLAYWRIGHT_BROWSERS_PATH=/scratch/09032/saini_2/tmp/ms-playwright
   /work/09032/saini_2/vista/library/uv-envs/pwqa/bin/python web/build/check_page.py --http --sections --out DIR`
   loads the page in headless Chromium (desktop and phone, light and dark; `--http` serves on 127.0.0.1 only),
   reports missing files, broken images, console errors and horizontal overflow, and saves screenshots to `DIR`
   (default `/scratch/09032/saini_2/sca/web_assets/check/`).

## Per-sample trajectory videos (needs a GPU)

Status (2026-10-30): 16 of the 16 videos on the page show the single-pass final operator (traces of the
`zw2_trace_*` tasks); any other video is the previous operator's and the page labels it "previous operator". The
traces ran `scripts/sample_trace.py --method sca_v2` in the frozen copy
`/scratch/09032/saini_2/sca/code/web_trace2_7bd64c0` (git 7bd64c0, the code of the `sca_v2` campaign, plus this
`sample_trace.py` and `compare_selection.json`) and wrote `/scratch/09032/saini_2/sca/web_assets/trace/sca_v2/<sid>/`.
`nice -n 19 $PY web/build/render_trace_videos.py` (default `--sca-method sca_v2`) re-renders `<sid>_sca.mp4` and
`<sid>_vs_area.mp4` for every prompt whose trace is complete (AReA traces unchanged), records the operator in
`assets/video/trace/operators.json` and rewrites `assets/js/traces.js`; the page drops the label by itself.
Tasks: `$PY web/build/make_trace_tasks.py --code-tag web_trace2_7bd64c0 --methods sca_v2 --prefix zw2_trace --pending`
(add `--freeze --commit 7bd64c0 --campaign-code /scratch/09032/saini_2/sca/code/7bd64c0` for a new frozen copy).
No progressive-SCA traces exist yet.

Original notes (previous operator):

- `scripts/sample_trace.py` re-runs one prompt at 4096^2 with the campaign's derived seed, settings and code path
  (SCA as `sca_nb`, or AReA through its released code) and saves, at each of the 28 steps, the decoded estimate of
  the clean image (x0 from the transformer output, tiled VAE decode, 1024^2 WebP preview), a native 512 x 512 crop
  at the comparison window from `build/compare_selection.json`, and `trace.json` with sigma, timestep, phase and the
  57 x 24 weights the attention processor actually used. It also records whether the final image matches the
  campaign PNG (hash, pixel identity, PSNR). `--dry-run` checks prompts, seeds, boxes and paths without the model.
  Outputs: `/scratch/09032/saini_2/sca/web_assets/trace/<method>/<prompt_id>/`.
- `web/build/make_trace_tasks.py` writes the spool tasks for the 8 prompts, plus a README; `--freeze` rebuilds a
  frozen code copy (git archive of a commit plus `sample_trace.py`).
- When the traces exist: `nice -n 19 $PY web/build/render_trace_videos.py` renders `assets/video/trace/<id>_sca.mp4`
  (decoded preview, phase, sigma timeline, the 57 x 24 weight strip and the 1:1 crop, about 15 s) and
  `<id>_vs_area.mp4` (SCA and AReA side by side), each under 3 MB with a WebP poster, and rewrites
  `assets/js/traces.js`. The renderer refuses to write synthetic test traces into `web/`.

## Against a commercial model: adding the Ideogram 4.0 images

The page compares single-pass SCA (FLUX.1-dev, 4096^2, one GPU, no training) with Ideogram 4.0 on six prompts.
Until the images exist the section shows labelled placeholders; nothing is fabricated, and Ideogram's own published
samples are not used because their pages grant no permission to reuse them.

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
you; keep the rights note in the manifest accurate.

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
- Every number comes from `results/` through `build_numbers.py` (plus the scratch inputs named above). Results that
  have not landed print "pending".
- Every SCA image names its configuration: single-pass SCA, progressive SCA, or previous operator (SCA without the
  schedule).
- Placeholders stay placeholders until the real asset exists: no forward-noised or synthetic stand-ins.
- No em dashes or section symbols in the copy; sentences stay under 36 words.
