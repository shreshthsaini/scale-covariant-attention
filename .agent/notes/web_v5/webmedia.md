# webmedia (videos and loops of the v5 page): state

Owner: webmedia. Brief: .agent/notes/web_v5/BRIEF.md Section 5. Design contract: design.md ("Videos (webmedia)").
Outputs: web/assets/video/v5/ (MP4 + WebP poster each) and web/assets/video/v5/manifest.json (the designer reads it:
id, role, file, poster, width, height, seconds, fps, loop, bytes, alt, caption, sources, script, plus per-item
extras such as the hero readout track). Scripts: web/build/v5_*.py; shared helpers in web/build/v5_media.py.
Builds, frames, encode tests and contact sheets: /scratch/09032/saini_2/sca/web_builds/v5_webmedia_1/ (frames per
item under <item>/, sheets under sheets/, logs under logs/). Nothing heavy in the repo.

Every video: H.264 High, yuv420p, BT.709 tags, +faststart, no audio track, even size; loops at most 2.5 MB; drawn on
the darkroom ground #0B0D0C with the dark-theme tokens of design.md (media sits on --screen in both themes), text in
the page's faces (Inter SemiBold labels, Source Serif 4 SemiBold titles). Numbers are read from web/build/numbers.json
and checked against the paper's macros (paper/figures/snippets/main_tables/headline.tex); none is typed.

Rebuild any item from the repository root:
  PY=/work/09032/saini_2/vista/library/uv-envs/sca/bin/python
  $PY -I web/build/v5_<name>.py            (each script's docstring lists its options)
  $PY -I web/build/v5_media.py sheet <video> <out.png> [n]   (contact sheet of any video)

## Done

### 1. Hero deep-zoom loops (v5_hero.py): hero_16x9.mp4, hero_9x16.mp4
- 1600 x 900 and 900 x 1600 (design.md allows these when 1920 does not fit), 30 fps, 20.1 s, 2.48 MB each (two-pass
  960 kb/s; CRF 24 would be 7.6 MB). Roles hero16x9, hero9x16. Posters = first frame = the rest frame of the loop.
- Shows: three 4096^2 progressive SCA images, each whole on the ground (square, 92 percent of the short side, 10 px
  rounded corners), a 2.3 s eased zoom to native pixels (one video pixel per image pixel), a 1.4 s hold on the PNG's
  own pixels, a 1.6 s zoom back out, a 0.6 s dissolve to the next image. The last dissolve lands on the first frame,
  so the loop is seamless (seam sheet checked).
- No burned-in text (design.md). The manifest entry carries `readout`: label "4096 x 4096", native label "1:1 native
  pixels", per-image segments (t0, zoom_in, native [start, end], zoom_out, t1, window_px) and `zoom_curve`
  [[t, scale]] at 0.1 s (scale = video px per image px; 1.0 = native) for the page's tag and zoom indicator.
- Images (rule): the paper hero's draw (v6_hero.draw: one showcase prompt per category, Random("0:<category>")),
  first three of its order: sc_h04_blacksmith (portrait), sc_a03_owl (animal), sc_l10_ice_cave (landscape); the PNGs
  of hero_provenance.json, sidecar-checked. Three, not five: five images at this size need about 4.5 MB.
- Zoom target (rule): the video-sized native window of highest mean edge energy (v6_common.best_windows on
  detail_map, 1/16 border margin), per cut.
- Pixels: each frame resampled by PIL Lanczos from the full PNG with that frame's source box (filter widens with the
  downscale); native hold is an exact crop. Log-uniform zoom, smootherstep ease.
- Sheets: sheets/hero_16x9_sheet.png, hero_9x16_sheet.png, hero_*_seam.png, hero_16x9_quality.png (100 percent
  crops, source against encode). Encode tests: hero/enc_test/.

### 2. Speed race (v5_speedrace.py): speedrace_16x9.mp4, speedrace_9x16.mp4
- 1600 x 900 (0.31 MB) and a phone layout 900 x 1600 (0.42 MB), 13.8 s, 30 fps, roles speedrace, speedrace9x16.
  Poster = the final state (four, two, one, one images).
- Shows: one shared time axis, four lanes (progressive SCA solid green, single-pass SCA hollow green, HRDiT, AReA
  purple). Each image is one segment of its lane; it fills as time runs; at its end the image appears above it and
  the lane's counter goes up. Runs to AReA's first image (202.4 s), holds 2.6 s, cross-fades back to the empty start
  (seamless; seam sheet checked). Clock of real seconds and "real time, shown 20x faster" on screen.
- Numbers: numbers.json main.{prog,sca,hrdit,area}.time_clean (value drives motion, display is printed), asserted
  equal to \hlSecProg, \hlSecSca, \hlSecHrdit, \hlSecArea of the paper's headline.tex. Counts in the manifest.
- Images (rule): every lane works through the same queue, the paper's decile prompts of the ImageReward gain over AReA
  (v6_gallery deciles_a/b provenance) from the median outward, upper first: 50th (= Figure 5's median row), 60th,
  40th, 70th. Single-pass SCA only reaches positions 1 and 2, both prompts that name no person (asserted).
- Sheets: sheets/speedrace_16x9_sheet.png, speedrace_9x16_sheet.png, speedrace_*_seam.png.

## Next
3. Method loops. 4. Comparison clips. 5. Process videos. 6. Explainer.
