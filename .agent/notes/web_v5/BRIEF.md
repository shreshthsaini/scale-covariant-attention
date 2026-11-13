# SCA project page v5: beat the AReA page (lead brief, 2026-11-13)

User request: "check the AReA webpage... our webpage should beat that in style and quality and examples and video
and gif etc." Our page: https://shreshthsaini.github.io/scale-covariant-attention/ (source web/, published to
gh-pages by web/build/publish_pages.sh). AReA's: https://rajabi2001.github.io/area/.
Lead's captures of both (full-page screenshots cut into segments, structure dumps):
/scratch/09032/saini_2/sca/web_builds/v6_audit/ (area_seg0..3.jpg, sca_seg0..7.jpg, *_info.json, shot.py, full.py).
Look at them before designing. History: .agent/notes/web_v4_20261108.md, web_v3_20261107.md, HANDOVER Sections 34, 35.
The paper the page must agree with is v6.2 (tag paper-v6.2; paper/main.tex, .agent/notes/paper_v6/BRIEF.md Section 2
is the story spine, text.md has the claims audit).

## 1. Lead's audit
AReA's page (11,400 px tall, 1,500 words, 2 videos, 13 images): dark navy, one huge wordmark, a full-width mosaic of
15 images over four models with a magnifier lens (4x, 8x, 16x), three tagline pillars, an animated idea widget with a
draggable denoising marker, a narrated method video, a failure-mode strip with tags, two stat cards, the pipeline
figure, six mechanism cards, a 15-prompt comparison carousel with synced zoom, five "final image beside its run"
videos, an efficiency block with two big numbers and a chart, and a table with backbone and resolution toggles.
Links: paper, arXiv, code, demo. It is short, confident, and every screen has one idea and one picture.
Our page (26,500 px, 5,100 words, 175 images, 3 videos): more evidence and better tools (synced zoom grids, deep
zoom, eight methods, 96-prompt review, intervals), but it reads like a report. Specific failures:
- The first screen is a small title and one 330 px image. No drama, no headline number, no motion.
- It still shows the v4/v5 paper: old title and subtitle, the retired speed plot with a broken axis and hollow
  markers, the old attention-focus figure, the old paired-gain plot, an explainer video built on the old curves.
- Stale claims: "matches HRDiT's preference scores" (v6.2: statistically indistinguishable on ImageReward, HPSv2.1,
  PickScore; after Holm correction better only on HPSv2 and ARNIQA, worse on none); crop-detail wording that the
  paper retired (crop detail is level with HRDiT and only +3.7 MUSIQ_c over AReA; never claim "sharper than AReA").
- Five thousand words; long paragraphs of protocol; a 12-prompt gallery and an Ideogram section far down.
- The method is told with static figures and prose where AReA animates it.
- No GIF-like loops beyond one zoom loop; process videos are single-pass traces.

## 2. What "beats AReA" means here (acceptance list; the lead scores each 0/1/2 on screenshots)
1. First screen: a full-bleed hero that moves. Title, one-line claim, three headline numbers, link buttons, and a
   large auto-playing deep zoom into 4096^2 images down to native pixels. Must look better than AReA's first screen
   at 1440x900 and at 390x844.
2. Identity: one deliberate visual system, not a template and not a copy of AReA's navy. Light and dark both
   designed (keep the toggle); the SCA green as the single accent, near orange and far blue only where they mean
   near and far (as in the paper figures); real typography (the page already ships Source Serif 4 and Inter).
3. Length and rhythm: at most about 2,200 words on the main page and about 14,000 px; one idea and one visual per
   screen; protocol detail, the 96-prompt review, the full gallery and the long tables move to sub-pages or
   collapsed blocks. Nothing is deleted from the repo; it is re-homed.
4. Method told in motion: interactive components a visitor can play with (Section 4), each explaining one step.
5. Examples: more and better than AReA's 15-prompt carousel: the hero mosaic, the whole-image comparison across the
   distribution of gains, the eight-method synced zoom, sizes and checkpoints, showcase. Every selection by a stated
   rule printed in small type.
6. Video and GIF-like loops: Section 5. More of them than AReA and each one true to data.
7. Results: big numbers that are exactly the paper's, an interactive table, the speed story. Honest edges stated
   once (patch FID, hand-set parts, artifacts at 6144^2 and 8192^2, the one single-pass sentence).
8. Craft: no console errors, no layout shift, no horizontal scroll, keyboard and reduced-motion support, lazy
   loading, first screen under about 3 MB, total published site under 400 MB, works from file:// and on a phone.

## 3. Page plan (designer may improve it; keep the order of the argument)
1. Hero (full-bleed, moving) with title "Reach Is a Property of the Head", subtitle "Scale-Covariant Attention for
   High-Resolution Diffusion Transformers", "Anonymous authors, under review", buttons, three numbers.
2. The claim in one screen: 49.2 s per 4096^2 image, 4.1x faster than AReA, better on all ten per-image metrics at
   three seeds; statistically indistinguishable from HRDiT on three preference scores in 46% of its time. With the
   speed race loop.
3. The observation: reach belongs to the head (interactive heads explorer; 97.5 / 2.5 / 0.7).
4. The method in three steps: two native-size views, reach emulation at 1024^2 (interactive mix slider), one table
   for any size and sampler stage. The paper's method figure appears once, as the summary.
5. See it: whole-image comparison across the distribution of gains (percentile explorer), then the eight-method
   synced zoom grid.
6. Sizes and checkpoints; showcase mosaic with deep zoom.
7. How an image forms: process videos.
8. Results: numbers, interactive table, gains plot, honest edges.
9. Explainer video, citation, links to sub-pages (96-prompt review, full gallery, all tables).

## 4. Interactive components (owner: webinteractive). Each is a self-contained module: assets/js/c_<name>.js,
assets/css/c_<name>.css, data under assets/data/<name>/, mounted by `SCAComp.<name>(element, options)`, with a demo
page web/dev/<name>.html. No framework, no CDN, file:// safe, touch and keyboard support, respects
prefers-reduced-motion, reads colours from CSS variables the designer defines (Section 6).
- `percentiles`: a slider (and keys) over the nine deciles of the paired ImageReward gain of progressive SCA over
  AReA (n = 195, seed 0); at each stop the four whole images (AReA, UltraImage, HRDiT, progressive SCA) with their
  ImageReward and seconds per image, click to zoom all four in sync (reuse zoomgrid.js). Data and prompt choice come
  from the paper's figure script paper/figures/src/v6_gallery.py / v6_qual_main.py and their provenance files; reuse
  their rule exactly. This is the honest picture (AReA ahead at the 10th percentile) and it stays honest.
- `heads`: the 1,368 near-share curves over noise level (paper Figure 2b data), hover or tap a curve to highlight
  it; for the heads that have attention maps (block 12, paper/figures/src/data/v6_attn_block12.npz) show the head's
  attention at one query on the image at the scrubbed noise level; the variance split beside it.
- `emulate`: a slider for the weight pi between the shrunk window view and the shrunk grid view for one head at
  1024^2, showing the mixed attention against full attention and the point moving on the real J(pi) parabola, with
  the optimum pi* = clip(B/A, 0, 1) marked; a head switch (near-reaching, far-reaching).
- `table`: the main comparison as an interactive table (sort, highlight ours, toggle metric groups) from generated
  numbers, and `speed`: seconds against image side with hover values.
Report each component's API, data size and a screenshot.

## 5. Video and loops (owner: webmedia). H.264 MP4 (yuv420p, faststart) plus a WebP or JPEG poster each; loops are
muted, autoplay, playsinline, under about 2.5 MB each and seamless; longer videos have controls. Sources are real
outputs and real data only. Load the `vista-video-io` skill before encoding. Outputs web/assets/video/v5/, scripts
web/build/v5_*.py, a manifest web/assets/video/v5/manifest.json (file, poster, width, height, seconds, loop, alt
text, the data or images it was made from) that the designer reads.
1. Hero deep-zoom loop(s): from a whole 4096^2 progressive SCA image into native pixels and out, across three to
   five showcase images (selection by the paper hero's rule: v6_hero.py provenance), 16:9 and a 9:16 phone cut.
2. Speed race: progress bars filling at the measured clean times (49.2 s progressive SCA, 77.1 s single pass,
   107.0 s HRDiT, 202.4 s AReA) with each method's image appearing when its bar completes and a counter of images
   finished; time-compressed and labelled with the compression. Numbers read from web/build/numbers.json or the
   paper's generated macros, never typed.
3. Method loops, 6 to 10 s each: (a) one head's window and grid on the 4096^2 image as the query moves; (b) reach
   emulation: views shrinking by r and the weight settling at pi*; (c) the sampler stages 1024^2, 2048^2, 4096^2
   reading the table at r = 2 and r = 4. Drawn from real attention data where it exists (v6 figure scripts).
4. Comparison clips: for three prompts by the paper's quartile rule, a synchronized push-in on AReA, HRDiT and
   progressive SCA side by side with the seconds per image on screen.
5. Process videos: inventory /scratch/09032/saini_2/sca/web_assets/trace/ (area, area_sc, sca, sca_v2, sca_v2_prog)
   and web/build/render_process_videos.py; render "final image beside the run that made it" for progressive SCA if
   progressive traces exist (and beside AReA's run where both exist). If progressive traces do not exist, say so in
   your notes with the exact generation request (script, prompts, sizes); do not submit jobs.
6. Explainer: re-render the two-minute explainer (web/build/render_explainer.py) so its visuals and wording match
   paper v6.2 (title, variance split figure, two-band method, numbers); captions burned in and as a VTT file.
No sound is required. Total new video under about 45 MB.

## 6. Shared contract
- Designer (webdesign) owns web/index.html, web/assets/css/site.css, web/assets/js/site.js, the sub-pages, and
  defines CSS variables in :root and [data-theme="dark"]: --bg, --bg-raised, --ink, --ink-soft, --rule, --accent
  (SCA green), --near (orange), --far (blue), --area (purple), --radius, --font-serif, --font-sans, --font-mono.
  Announce them in .agent/notes/web_v5/design.md within the first half hour; the others read that file.
- Numbers: web/build/build_numbers.py writes numbers.json / numbers.js from results/ and the paper's generated
  macros. The designer brings it in line with paper v6.2 (Holm records, indistinguishable wording, crop detail) and
  every number on the page comes from it. No number is typed in HTML, JS or a video.
- Wording rules (binding, same as the paper): never calibration / calibrate; never curated / uncurated; never
  "random" for a selection (say seeded draw); no em dashes; "Anonymous authors, under review" stays; progressive SCA
  is the headline and single pass the fast mode; the repeated-subject issue is not discussed beyond the paper's one
  sentence; "matches" is never used for a difference whose interval includes zero; no claim of sharper crops than
  AReA. Pages that show single-pass images draw from prompts that name no person.
- Do not show Qwen-Image or Ideogram as headline evidence (their switches are off in the paper); the existing
  Ideogram and other-backbone blocks may stay on a sub-page, labelled as exploratory.
- Environment: TACC Vista, aarch64, CPU node with 144 cores, no GPU, no sbatch. Python only via
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python; headless Chromium via
  PLAYWRIGHT_BROWSERS_PATH=/scratch/09032/saini_2/tmp/ms-playwright
  /work/09032/saini_2/vista/library/uv-envs/pwqa/bin/python (see web/build/check_page.py); ffmpeg at /usr/bin/ffmpeg
  and /work/09032/saini_2/vista/library/ffmpeg-7.1-shared/bin. Builds, renders, frames and screenshots go under
  /scratch/09032/saini_2/sca/web_builds/v5_<role>_<n>/, never $HOME or /tmp; only finished assets go into web/.
  Never delete anything on scratch. A local preview server binds 127.0.0.1 only; no tunnels.
- No git commands that change state and no publishing: the lead commits (a loop pushes work in progress to main
  every 15 minutes) and the lead runs publish_pages.sh after review. Do not edit paper/ or results/.
- Verify by looking: screenshot what you build (desktop 1440x900 and phone 390x844, light and dark) and read the
  screenshots; watch your videos' frames (extract a contact sheet and read it).
- The controller node's allocation ends about 22:10 today. Work in finished increments, keep your notes file
  current (what is done, what is next, how to rebuild), so another agent can continue from the repo alone.
- Notes: .agent/notes/web_v5/<role>.md. Final message to the lead: what is done, what is not, where the screenshots
  are, and requests to the other owners.
