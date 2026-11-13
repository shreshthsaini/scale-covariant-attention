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

### 3. Method loops (v5_method.py [window|emulation|stages|all]): 1600 x 900, 30 fps, seamless
- method_window.mp4 (8.0 s, 0.74 MB, role method_window): the 4096^2 sheep-and-church image (the run behind the
  paper's method figure) twice; a query moves on a closed path; its 8 x 8 tile, the 64-token window around it
  (orange, clamped) and the 64 x 64 block-centre grid (blue, stride 4). Left block 12 head 4 (pi 0.93), right head 7
  (pi 0.00), pi from the reach table at r = 4, sigma bin 0.62 (k1_1024.npz); view opacity follows pi and 1 - pi; bars
  "window 93% / grid 7%". Geometry only on the 4096^2 image (no attention is captured at 4096^2); weights real.
- method_emulation.mp4 (9.8 s, 0.65 MB, role method_emulation): block 12 head 10 at 1024^2 (the paper's Figure 3
  head), query token (43, 11), step 20: real full attention (orange inside the r = 4 window, blue outside), the window
  shrinking 64 -> 32 -> 16 tokens and the grid thinning to stride 2 and 4, the real emulated near and far attention
  (local_r4, global_r4), then pi sweeping 0 -> 1 and settling at pi* = clip(B/A, 0, 1) = 0.79 on the real J(pi)
  (A, B, E from method_assets/provenance.json, asserted equal to the k1_1024 table cell).
- method_stages.mp4 (10.1 s, 1.58 MB, role method_stages): the recorded progressive run of sc_l03_opera_house (rule:
  first paper-hero draw with a trace); x0 estimate of each of the 57 steps drawn to scale (1024^2, 2048^2, 4096^2
  squares), the per-step pi of all 57 x 24 heads from the trace (grey "full attention at 1024 x 1024" in stage 1),
  a stage rail (30 / 17 / 10 steps; r = 2, r = 4).
- Loops b and c close with a full-frame cross-fade to frame 0 (xfade_loop); a is periodic. Sheets:
  sheets/method_*_sheet.png, method_*_seam.png; look frames method/look/.

### 4. Comparison clips (v5_compare.py [--only 25|50|75]): compare_p25.mp4, compare_p50.mp4, compare_p75.mp4
- 1600 x 900, 9.8 s, 30 fps, 2.24 / 2.49 / 2.48 MB, roles compare_1..3, seamless (cross-fade back to the whole
  images). Poster = first frame (three whole images).
- Shows: AReA | HRDiT | progressive SCA (488 px squares), each headed by name and clean seconds per image
  (numbers.json, asserted equal to the paper macros); whole images, a 6.2 s synchronized push-in (log-scale,
  smootherstep) to native pixels on one shared window, a 1.8 s native hold. Footer: the prompt's percentile and the
  scale ("whole image, 4096 x 4096", zoom factor, "1:1 native pixels"). No other text.
- Prompts (rule): the three rows of the paper's Figure 5 (qual_main_provenance.json: 25th, 50th, 75th percentile,
  nearest rank, of the paired ImageReward difference progressive SCA minus AReA, 195 prompts, seed 0); the PNGs of
  that file, sidecar-checked for same prompt and noise. Window (rule): the 488 x 488 native window of highest mean
  edge energy averaged over the three images (v6_common.best_windows over three detail maps).
- Bug caught by looking: the padded-source cache keyed by file stem made all three panels AReA; now keyed by the full
  path (sha1). Sheets: sheets/compare_p*_sheet.png, compare_p*_seam.png, compare_p25_p75_look.png.

### 5. Process videos (v5_process.py [--only <id>] [--look t,t]): process_<id>.mp4 x 5
- Progressive traces exist: web_assets/trace/sca_v2_prog/<id> (12 showcase prompts, 57 steps: x0 previews, 512 px
  native-window crops, per-step pi) and AReA traces of the same 12 (area_sc, 28 steps). All checked complete, not
  synthetic, final pixel-identical to the campaign PNG, same prompt/seed/box. (render_process_videos.py was written
  for these but never run on real traces; v5_process.py is a new layout.)
- 1600 x 900, 14.2 s, 30 fps, 1.9 to 2.4 MB each, roles process_1..5, seamless (cross-fade to frame 0). Poster = the
  hold on the finals.
- Shows: left the final 4096^2 progressive SCA image; middle its run (x0 of each of 57 steps, chip "1024^2 own
  attention" / "2048^2 table r = 2" / "4096^2 table r = 4", stage rail 30/17/10); right AReA's run (chip "coarse,
  pooled keys, f = 4/2" / "fine, windows of N", rail coarse/fine). Both runs: the trace's 512 px box in red and an
  inset of that window as decoded at the step (tag 1:1, 2x or 4x upsampled). Heads print clean seconds per image and
  "step k of n". One clock normalised: both runs spread over the same 11 s (stated on screen).
- Prompts (rule): traced showcase prompts, one per category by Random("0:<category>") over the category's traced ids,
  first five of the hero order: sc_h01_silver_braids, sc_a02_jumping_spider, sc_l01_fjord_village,
  sc_l03_opera_house, sc_d03_antique_shop (product would be sc_p08_sapphire_ring).
- Sheets: sheets/process_*_sheet.png, process_*_seam.png; look frames process/look/.

### 6. Explainer (v5_explainer.py [--look t,t]): explainer.mp4 + explainer.webp + explainer.vtt
- 1600 x 900, 84.5 s, 30 fps, 7.6 MB (CRF 21), role explainer, loop false (give it controls). 17 caption cues burned
  in (Inter 30 px, wrapped to two lines when long) and the same cues in explainer.vtt and in the manifest (`cues`).
- Scenes: title ("Reach Is a Property of the Head", subtitle, "Anonymous authors, under review"); problem (the run's
  native 1024^2 stage beside full attention run directly at 4096^2, same prompt and seed: 64 x 64 vs 256 x 256 tokens,
  offsets 63 vs 255); observation (near share of all 1,368 heads over sigma from v6_observation.load, heads 4 and 7
  and the average highlighted; variance split 97.5 / 2.5 / 0.7 from \hlVarHead, \hlVarStep, \hlVarAvg); the two
  views (method_window frames); band 1 "Measure once at 1024 x 1024" (method_emulation frames); band 2 "Generate at
  any size" (method_stages frames); speed (speedrace frames; 49.2 s, 4.1x, 2.2x; "finishes four"); claims ("10 of 10
  per-image metrics better than AReA, at each of three seeds"; "46% of HRDiT's time, statistically
  indistinguishable on ImageReward, HPSv2.1 and PickScore"); showcase (hero frames, blacksmith); end card.
- Numbers from headline.tex macros only; the script asserts the three HRDiT intervals include zero and that the ten
  wins hold at seeds 0, 1, 2 before it prints the claims. Reused scenes stop before each loop's closing tail.
- Replaces the 116 s v4 explainer (web/assets/video/method/sca_explainer.*, old title and claims; not deleted).
- Sheets: sheets/explainer_sheet.png (20 frames), explainer_boundaries.png (every scene cut); look frames explainer/look/.

### Extra: tagged hero fallback (v5_hero_tagged.py): hero_16x9_tagged.mp4, hero_9x16_tagged.mp4
- The clean hero frames plus a "4096 x 4096" chip (top left) and a zoom indicator (track whole -> 1:1, dot, label
  "2.5x" ... "1:1 native pixels" in green on the hold). 2.48 MB each, roles hero16x9_tagged / hero9x16_tagged. The
  lead's brief asks for the tag and indicator; design.md asks for no burned-in text: the clean loop is the default
  and the page can draw the readout from `readout`; the tagged pair is there if it does not.

### Totals (audit of 2026-11-13 17:05)
18 items, 41.2 MB with posters (web/assets/video/v5 is 40 MB on disk), all pass v5_manifest.py. All posters:
sheets/posters.png. Build order for a full rebuild: v5_hero.py, v5_hero_tagged.py, v5_speedrace.py, v5_method.py,
v5_compare.py, v5_process.py, v5_explainer.py (needs the frame folders of the first four), then v5_manifest.py.

Encodes: v5_media.encode_to_size writes every attempt to <build>/enc_tmp and copies only the file that fits into
web/assets/video/v5 (the repo is committed by a loop; checked that no oversized attempt reached the history).

### Audit (v5_manifest.py)
- Re-probes every item (codec, pix_fmt, size, no audio, moov first, byte limits, poster size = video size, wording
  rules), adds `source` (design.md's field name) and `order`, writes total_mb. Run after any rebuild.
- Phone check: sheets/phone_390.png (hero 9:16 tagged and the race phone cut at 390 px wide: legible).

## Requests to the designer (webdesign)
1. Hero: `<video muted autoplay loop playsinline preload="auto" poster="assets/video/v5/hero_16x9.webp">` with
   hero_16x9.mp4, and hero_9x16.mp4 below about 700 px width (source media query or swap in site.js). The image is a
   square centred on #0B0D0C at rest, so the --screen ground blends into the frame. The readout (tag, zoom indicator,
   "1:1 native pixels") can be driven from manifest `readout.segments` / `readout.zoom_curve` on timeupdate; if not,
   use the *_tagged files. First screen: poster 74 KB + video 2.48 MB.
2. Every other loop: muted, loop, playsinline, preload="none", src set when it scrolls into view (IntersectionObserver),
   paused when it leaves; prefers-reduced-motion: poster only, with a play control. Posters are the rest frames.
3. Show the 1600 x 900 loops at content width (1000 px or more on desktop): their smallest labels are 22 to 24 px at
   1600. On phones use speedrace_9x16.mp4; the method and compare loops are legible to about 600 px wide, below that
   consider a tap-to-enlarge or the static figure.
4. Explainer: controls, no autoplay, poster explainer.webp; captions are burned in, so add
   `<track kind="captions" src="assets/video/v5/explainer.vtt" srclang="en" label="English">` without `default`.
5. Captions under each video: manifest `caption` (each states its selection rule); alt text in `alt`.
6. The v4 videos (teaser/zoom_loop, method/sca_explainer with old claims, trace/*_vs_area single-pass) are superseded;
   take them off the main page (move or retire as the lead decides; nothing deleted).

## Not made / limits
- Hero at 1920 x 1080: a 20 s zoom loop does not fit 2.5 MB at 1920 with acceptable quality; 1600 x 900 at 2.48 MB
  (sheets/hero_16x9_quality.png). Three hero images, not five, for the same reason.
- No attention exists at 4096^2, so method_window draws the operator's exact geometry with real weights, not maps.
- Process videos use one normalised clock (both runs over 11 s); real seconds are printed, the speed race shows time.
