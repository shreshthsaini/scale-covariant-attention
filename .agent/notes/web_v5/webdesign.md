# Web v5: designer and integrator (role webdesign), 2026-11-13

## State (16:40)
- design.md written (tokens, fonts, component and video contract).
- Before: main page 26,500 px, about 5,100 words (lead's audit).
- In progress: build_numbers.py paper macros; v6 figure exports (SVG themed, PDFs rasterized by v5_raster_figs.py);
  new site.css / index.html / site.js; sub-pages.

## Plan
1. numbers: build_numbers.py adds tex.<macro> keys from paper/figures/snippets/**/*.tex (paper's own values).
2. figures: web/build/v5_figs.py (SVG to theme-aware inline SVG in assets/js/v6_figs.js);
   web/build/v5_raster_figs.py (PDF to WebP via Chromium and pdf.js; pwqa env).
3. page: index.html rebuilt (hero, claim, observation, method, see it, sizes and showcase, process, results, more).
4. sub-pages at web root: gallery.html, results.html, exploratory.html (re-homed v4 sections), review/ unchanged.
5. web/build/build_v5_assets.py writes assets/js/v5_assets.js (which c_*.js/css exist, v5 video manifest).
6. check_page.py covers the new pages.

## Rebuild
(filled in at the end)
