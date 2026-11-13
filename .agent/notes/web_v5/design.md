# Web v5 design contract (owner: webdesign; announced 2026-11-13 16:15, binding for webinteractive and webmedia)

## Direction
"Native pixels". An editorial gallery, not a report and not a dark-navy template. The images are the protagonists;
text is short, set large, and steps aside. Warm paper in light mode, a neutral darkroom (graphite with a trace of
green, never navy) in dark mode. One accent: the paper's SCA green. Near orange and far blue appear only where they
mean near and far; AReA purple only for AReA. Headlines and big numbers in Source Serif 4 (600, tight tracking);
text, labels and UI in Inter with tabular figures. Signature device: the reticle (four thin corner brackets in the
accent, like a camera focus box) marking where we zoom, used in the hero readout, on zoom targets and as the section
marker. Every screen: one kicker, one serif headline, at most two sentences, one visual. Media always sits on the
darkroom "screen" ground in both themes, so a video rendered once looks right in both.

## CSS variables (defined in web/assets/css/site.css on :root and [data-theme="dark"])
Components read these; never hard-code a colour that has a token. Values for both themes:

| token | light | dark | use |
|---|---|---|---|
| --bg | #F4F3EE | #0B0D0C | page ground |
| --bg-raised | #FFFFFF | #141816 | cards, panels, component surfaces |
| --bg-sunk | #EAE8E1 | #070908 | wells, image grounds, track backgrounds |
| --ink | #111513 | #ECEEE9 | text, strong lines |
| --ink-soft | #4F5853 | #A5AEA8 | secondary text (>= 4.5:1 on --bg) |
| --ink-faint | #7C847F | #727B76 | tertiary labels, axis ticks (large or non-essential text only) |
| --rule | #DCD9CF | #252B28 | hairlines, grid lines, borders |
| --accent | #13865F | #3DBB8C | SCA green: fills, markers, active states, our method |
| --accent-ink | #0B6B4A | #5CD2A4 | green text and links (contrast-safe) |
| --accent-soft | #DDEFE6 | #0F2A1F | tinted backgrounds, highlighted table rows |
| --on-accent | #FFFFFF | #04130D | text on --accent fills |
| --near | #E08A2B | #F0A04B | near view / local window / near share (paper orange) |
| --far | #2F6DB3 | #6FA3E0 | far view / global grid (paper blue) |
| --area | #7A4EA3 | #B592E3 | AReA only (paper purple) |
| --hrdit | #3D4541 | #C9D0CB | HRDiT (strong neutral) |
| --neutral | #8C9490 | #7F8984 | every other baseline, full attention |
| --screen | #0B0D0C | #0B0D0C | media ground in BOTH themes (videos, posters, hero) |
| --screen-ink | #ECEEE9 | #ECEEE9 | text on --screen |
| --radius | 12px | 12px | panels, cards, media frames |
| --radius-sm | 6px | 6px | chips, buttons, small thumbnails |
| --shadow | 0 1px 2px rgba(17,21,19,.06), 0 12px 32px rgba(17,21,19,.08) | 0 1px 2px rgba(0,0,0,.5), 0 14px 36px rgba(0,0,0,.45) | raised panels only |
| --font-serif | "Source Serif 4", "Iowan Old Style", Georgia, serif | same | headlines, big numbers |
| --font-sans | "Inter", system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif | same | everything else |
| --font-mono | ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace | same | code, coordinates only |
| --ease | cubic-bezier(.2,.7,.2,1) | same | all transitions (200 to 600 ms) |

Theme switching: the page sets data-theme="light" or "dark" on <html>; with no attribute it follows
prefers-color-scheme. Components must re-read colours on the `sca:themechange` event (dispatched on document) or use
CSS variables directly (preferred: canvas code reads getComputedStyle(element).getPropertyValue('--accent')).

Method colours in charts and components: progressive SCA = --accent, solid marker (star or filled circle);
single-pass SCA = --accent, hollow or dashed (the fast mode); AReA = --area; HRDiT = --hrdit; UltraImage, DyPE,
SEGA, full attention = --neutral (distinguish by marker or label, not by new hues).

## Type scale (for components that print text)
UI text 14px/1.45 Inter 400; labels 12px Inter 600 uppercase letter-spacing .08em in --ink-soft; numbers in
Inter with font-variant-numeric: tabular-nums; any big number inside a component: Source Serif 4 600. Minimum
text size 12px (11px only for axis ticks). Focus ring: 2px solid var(--accent), offset 2px.

## Components (webinteractive)
Mount: the page has `<div class="comp-mount" data-comp="<name>" data-options='{...}'>` inside a figure whose
fallback (static image) is a sibling with class comp-fallback. site.js loads assets/css/c_<name>.css and
assets/js/c_<name>.js only when they exist (it reads assets/js/v5_assets.js, written by web/build/build_v5_assets.py),
then calls SCAComp.<name>(mountEl, options). Return value: an object with destroy() if you can. On success the
page adds .is-live to the figure (fallback hidden). Throwing leaves the fallback visible.
Width: fill the mount (100%); the page gives at most 1200px on desktop and the full width minus 16px gutters at
390px. Set your own height; never taller than 80vh; no horizontal overflow at 360px. Transparent or --bg-raised
background, 1px --rule border at most, no outer shadow (the page frames you). Prefix every class with c-<name>-.
Respect prefers-reduced-motion (no autoplay animation; still images). No console output.

## Update 17:20 (designer)
- The hero is no longer 100vh: on wide screens it is sized to its content (at most about 82% of the viewport, the
  next section peeks in); on phones a 16:10 box shows the 16:9 loop (the 9:16 cut is kept but not used).
- Component fallback images may use data-src (loaded only when the component is absent or fails).
- An Artifact preview bundle of the main page is built by web/build/v5_artifact_bundle.py (see webdesign.md):
  components must load their data by relative path from their own script (as they do), nothing from other hosts.

## Videos (webmedia)
Ground --screen #0B0D0C; text #ECEEE9 (secondary #A5AEA8); SCA #3DBB8C, near #F0A04B, far #6FA3E0, AReA #B592E3,
HRDiT #C9D0CB, others #7F8984 (the dark-theme values, since media sits on the screen ground in both themes).
Fonts as TTF for renderers (same families as the page, SIL OFL texts beside them):
/scratch/09032/saini_2/sca/web_builds/v5_webdesign_1/fonts/ Inter-Regular.ttf, Inter-Medium.ttf,
Inter-SemiBold.ttf, SourceSerif4-Regular.ttf, SourceSerif4-SemiBold.ttf, SourceSerif4-Italic.ttf.
Labels Inter SemiBold, titles Source Serif 4 SemiBold. Keep text inside a 6% safe margin and at least 28 px tall at
1080p. Hero loops: no burned-in text at all (the page overlays title and readout); keep the subject central; the
page darkens the lower-left 45% with a gradient, so detail there is partly hidden. 16:9 at 1920x1080 and 9:16 at
1080x1920 (or 1600x900 / 900x1600 if size demands). Posters: WebP at the video's size, first frame (or the frame
the loop rests on), named <video>.webp. Manifest fields the page reads: file, poster, width, height, seconds, loop,
alt, source; add "role" (hero16x9, hero9x16, speedrace, method_window, method_emulation, method_stages,
compare_<n>, process_<n>, explainer) so the page can place each one without guessing.
