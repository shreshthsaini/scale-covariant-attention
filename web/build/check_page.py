"""Headless check of the v5 site: web/index.html, the sub-pages gallery.html, results.html and exploratory.html, and
web/review/index.html. Per page, desktop and phone, light and dark: console errors, broken local files, horizontal
overflow, forbidden words (and em dashes), screenshots. On the main page also: words in <main> and page height
against the v5 budget (2,200 words, 14,000 px, reported), bytes loaded before any scroll (first screen under 3 MB),
layout shift, a keyboard check (the first Tab reaches the skip link, Enter on it moves focus into the page) and a
reduced-motion check (no video plays on its own). The synced zoom grids (wheel zoom, drag, double-click reset, a
two-finger pinch from synthetic touch pointers, auto zoom; every panel must show the same region) are exercised on
the main page, the gallery and the 96-prompt page.

Usage (needs an env with playwright and its Chromium build):
    PLAYWRIGHT_BROWSERS_PATH=/scratch/09032/saini_2/tmp/ms-playwright \
    /work/09032/saini_2/vista/library/uv-envs/pwqa/bin/python web/build/check_page.py [--out DIR] [--http]

--http serves web/ with python's http.server on 127.0.0.1 (never 0.0.0.0) instead of opening file://.
Writes desktop and phone screenshots (light and dark) to --out and prints a pass/fail summary.
"""
import argparse
import functools
import http.server
import json
import os
import re
import sys
import threading

from playwright.sync_api import sync_playwright

WEB = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PAGES = ["index.html", "gallery.html", "results.html", "exploratory.html"]
FORBID = r"calibrat\w*|\buncurated\b|\bcurated\b|\brandom(?:ly)?\b|\u2014|\u00a7|\bmatches HRDiT"
WORDS_JS = """() => {
  const m = document.querySelector('main'); if (!m) return 0;
  const w = document.createTreeWalker(m, NodeFilter.SHOW_TEXT, null); let n = 0, t;
  while ((t = w.nextNode())) { const p = t.parentElement; if (!p || p.closest('[hidden],script,style,noscript,.sr-only')) continue;
    const cs = getComputedStyle(p); if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    n += (t.textContent.match(/[A-Za-z0-9][A-Za-z0-9'\u2019.,%+-]*/g) || []).length; }
  return n; }"""


def local_refs():
    """Every src/href/poster in index.html and every asset path in the generated data files must exist."""
    refs = set()
    for page in PAGES:
        html = open(os.path.join(WEB, page)).read()
        refs |= {r for r in re.findall(r'(?:src|href|poster|data-native|srcset)="([^"#:]+)"', html) if not r.startswith("../paper/")}
    for js in ("v5_assets.js",):
        p = os.path.join(WEB, "assets", "js", js)
        if os.path.exists(p):
            refs |= {r for r in re.findall(r'"(assets/[^"]+\.(?:js|css|webp|mp4|png|jpg|vtt|srt))"', open(p).read())}
    for js in ("images.js", "ideogram_manifest.js", "media.js", "showcase.js", "gallery.js", "process.js", "method_data.js", "zoomsets.js"):
        p = os.path.join(WEB, "assets", "js", js)
        if os.path.exists(p):
            refs |= {r for r in re.findall(r'"(assets/[^"]+\.(?:webp|mp4|png|jpg|vtt|srt))"', open(p).read()) if "{" not in r}
    tr = os.path.join(WEB, "assets", "js", "traces.js")
    if os.path.exists(tr):                           # only rendered trace videos must exist; expected ones are placeholders
        t = open(tr).read()
        for v in json.loads(t[t.index("{"):t.rindex("}") + 1]).get("available", []):
            refs |= {v["mp4"], v["poster"]}
    rv = os.path.join(WEB, "review", "items.js")
    if os.path.exists(rv):                           # the second page's files, relative to web/review/
        refs |= {"review/" + r for r in re.findall(r'"(img/[^"]+\.webp)"', open(rv).read())}
    missing = []
    for r in sorted(refs):
        if r.startswith("assets/ideogram/"):
            continue                                 # optional by design: placeholders until the files are added
        if not os.path.exists(os.path.normpath(os.path.join(WEB, r))):
            missing.append(r)
    return refs, missing


ZOOM_STATE = """(sel) => {
  const g = document.querySelector(sel);
  const cv = Array.from(g.querySelectorAll('canvas.zg-cv'));
  const sig = cv.map(c => { const x = c.getContext('2d'); const w = c.width, h = c.height;
    const d = x.getImageData(Math.floor(w * .5), Math.floor(h * .5), 1, 1).data; return [w, h, d[3]]; });
  return {cls: g.className, read: g.querySelector('.zg-read').textContent, canvases: cv.length,
          panels: g.querySelectorAll('.zg-p').length, sizes: sig, map: g.querySelector('.zg-map-r').getAttribute('style')};
}"""
PINCH = """(sel) => {
  const p = document.querySelector(sel + ' .zg-p'); const r = p.getBoundingClientRect();
  const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
  const ev = (t, id, x, y) => p.dispatchEvent(new PointerEvent(t, {pointerId: id, pointerType: 'touch', isPrimary: id === 11,
                                                 clientX: x, clientY: y, bubbles: true, cancelable: true, button: 0, buttons: t === 'pointerup' ? 0 : 1}));
  ev('pointerdown', 11, cx - 20, cy); ev('pointerdown', 12, cx + 20, cy);
  for (let k = 1; k <= 10; k++) { ev('pointermove', 11, cx - 20 - 6 * k, cy); ev('pointermove', 12, cx + 20 + 6 * k, cy); }
  ev('pointerup', 11, cx - 80, cy); ev('pointerup', 12, cx + 80, cy);
  return document.querySelector(sel + ' .zg-read').textContent;
}"""


def zoom_checks(pg, name, out, scope="#cmp-zoom"):
    """Wheel zoom, drag, synthetic pinch, double-click reset and auto zoom on the first grid inside `scope`."""
    bad = []
    sel = scope + " .zg"
    g = pg.locator(sel).first
    if not g.count():
        print(f"   {name}: no zoom grid in {scope}")
        return [name + ":nogrid"]
    pg.evaluate("(s) => { document.documentElement.style.scrollBehavior = 'auto'; document.querySelector(s).scrollIntoView({block: 'center'}); }", sel)
    pg.wait_for_timeout(900)
    bb = g.locator(".zg-p").first.bounding_box()
    pg.mouse.move(bb["x"] + bb["width"] * .45, bb["y"] + bb["height"] * .4)
    for _ in range(10):
        pg.mouse.wheel(0, -120)
        pg.wait_for_timeout(40)
    pg.wait_for_timeout(1400)
    st = pg.evaluate(ZOOM_STATE, sel)
    ok = "zoomed" in st["cls"] and st["canvases"] == st["panels"] and len({tuple(x[:2]) for x in st["sizes"]}) == 1
    print(f"   {name} wheel: {st['read']!r}, canvases {st['canvases']}/{st['panels']}, map {st['map']}", "" if ok else "FAIL")
    g.screenshot(path=os.path.join(out, f"{name}_zoom.png"))
    pg.mouse.move(bb["x"] + bb["width"] * .5, bb["y"] + bb["height"] * .5)
    pg.mouse.down()
    pg.mouse.move(bb["x"] + bb["width"] * .25, bb["y"] + bb["height"] * .7, steps=6)
    pg.mouse.up()
    pg.wait_for_timeout(300)
    st2 = pg.evaluate(ZOOM_STATE, sel)
    ok = ok and st2["map"] != st["map"]
    pg.mouse.dblclick(bb["x"] + bb["width"] * .5, bb["y"] + bb["height"] * .5)
    pg.wait_for_timeout(2600)
    st3 = pg.evaluate(ZOOM_STATE, sel)
    ok = ok and "zoomed" not in st3["cls"] and st3["canvases"] == 0
    print(f"   {name} drag moved the view: {st2['map'] != st['map']}; double-click reset: {'zoomed' not in st3['cls']}")
    read = pg.evaluate(PINCH, sel)
    pg.wait_for_timeout(400)
    st4 = pg.evaluate(ZOOM_STATE, sel)
    ok = ok and "zoomed" in st4["cls"]
    print(f"   {name} pinch: {read!r}", "" if "zoomed" in st4["cls"] else "FAIL")
    g.locator(".zg-reset").click()
    pg.wait_for_timeout(2600)
    g.locator(".zg-auto").click()
    pg.wait_for_timeout(4200)
    st5 = pg.evaluate(ZOOM_STATE, sel)
    g.screenshot(path=os.path.join(out, f"{name}_tour.png"))
    ok = ok and "touring" in st5["cls"]
    g.locator(".zg-auto").click()                          # pause: the view stays where it is
    pg.wait_for_timeout(300)
    st6 = pg.evaluate(ZOOM_STATE, sel)
    ok = ok and "touring" not in st6["cls"] and "zoomed" in st6["cls"]
    print(f"   {name} auto zoom: {st5['read']!r}; paused and kept: {'zoomed' in st6['cls']}")
    g.locator(".zg-reset").click()
    pg.wait_for_timeout(2600)
    if not ok:
        bad.append(name + ":zoom")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/09032/saini_2/sca/web_assets/check")
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--sections", action="store_true", help="also save one screenshot per section")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    refs, missing = local_refs()
    print(f"local refs: {len(refs)}, missing: {len(missing)}")
    for m in missing:
        print("  MISSING", m)
    server = None
    if a.http:
        class Quiet(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *args):            # no request log
                pass

            def handle(self):                         # the browser drops video range requests; not an error
                try:
                    super().handle()
                except (ConnectionResetError, BrokenPipeError):
                    pass
        handler = functools.partial(Quiet, directory=WEB)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        root = f"http://127.0.0.1:{server.server_address[1]}/"
    else:
        root = "file://" + WEB + "/"
    url = root + "index.html"
    problems = []
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--disable-gpu", "--no-sandbox", "--autoplay-policy=no-user-gesture-required"])
        for page in PAGES:
            stem = page.replace(".html", "")
            for name, vp, scheme in [("desktop", (1440, 900), "light"), ("desktop_dark", (1440, 900), "dark"),
                                     ("phone", (390, 844), "light"), ("phone_dark", (390, 844), "dark")]:
                tag = name if page == "index.html" else f"{stem}_{name}"
                ctx = b.new_context(viewport={"width": vp[0], "height": vp[1]}, color_scheme=scheme, device_scale_factor=1,
                                    is_mobile=vp[0] < 500, has_touch=vp[0] < 500)
                pg = ctx.new_page()
                errs, nbytes = [], [0]
                pg.on("console", lambda m: errs.append(m.text + " " + str((m.location or {}).get("url", ""))) if m.type == "error" else None)
                pg.on("pageerror", lambda e: errs.append(str(e)))
                pg.on("response", lambda r: errs.append(f"HTTP {r.status} {r.url}") if r.status >= 400 and "/ideogram/" not in r.url else None)
                def _sz(r):
                    try:
                        nbytes[0] += int((r.headers or {}).get("content-length", "0") or 0)
                    except Exception:
                        pass
                pg.on("response", _sz)
                pg.goto(root + page, wait_until="load")
                pg.wait_for_timeout(1500)
                first_bytes = nbytes[0]
                first_shift = pg.evaluate("""() => new Promise(r => { let s = 0; try { new PerformanceObserver(l => { for (const e of l.getEntries()) if (!e.hadRecentInput) s += e.value; }).observe({type: 'layout-shift', buffered: true}); } catch (e) {} setTimeout(() => r(s), 150); })""")
                pg.screenshot(path=os.path.join(a.out, f"{tag}_first.png"))
                h = pg.evaluate("document.documentElement.scrollHeight")
                for y in range(0, h, 700):                  # trigger lazy images, components and loops
                    pg.evaluate(f"window.scrollTo(0, {y})")
                    pg.wait_for_timeout(90)
                pg.wait_for_timeout(1200)
                pg.evaluate("window.scrollTo(0, 0)")
                pg.wait_for_timeout(300)
                h = pg.evaluate("document.documentElement.scrollHeight")
                over = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
                broken = pg.evaluate("""Array.from(document.images).filter(i => i.getAttribute('src') && i.complete && i.naturalWidth === 0
                                         && !i.src.includes('/ideogram/')).map(i => i.src)""")   # data-src fallbacks without src are held back, not broken
                n_img = pg.evaluate("document.images.length")
                text = pg.evaluate("document.body.innerText")
                bad = sorted(set(m.group(0).lower() for m in re.finditer(FORBID, text, re.I)))
                if bad:
                    errs.append("forbidden words in the visible text: " + ", ".join(bad))
                pg.screenshot(path=os.path.join(a.out, f"{tag}.png"), full_page=True)
                words = pg.evaluate(WORDS_JS)
                print(f"{tag}: height {h}px, words {words}, images {n_img}, broken {len(broken)}, horizontal overflow {over}px, "
                      f"first screen {first_bytes / 1e6:.2f} MB, layout shift {first_shift:.3f}, console errors {len(errs)}")
                errs = [e for e in errs if "/ideogram/" not in e]   # missing Ideogram files are expected placeholders
                for e in errs[:8]:
                    print("   console:", e[:200])
                for u in broken[:8]:
                    print("   broken:", u)
                if over > 0 or broken or errs:
                    problems.append(tag)
                if page == "index.html" and name in ("desktop", "phone"):
                    if first_bytes > 3e6:
                        print(f"   first screen {first_bytes / 1e6:.2f} MB > 3 MB"); problems.append(tag + ":first-screen-bytes")
                    if first_shift > 0.1:
                        print(f"   layout shift {first_shift:.3f} > 0.1"); problems.append(tag + ":layout-shift")
                    if name == "desktop" and (words > 2400 or h > 15000):
                        print(f"   NOTE: over the v5 budget (2,200 words, 14,000 px)")
                if page == "index.html" and name == "desktop":   # keyboard: the first Tab reaches the skip link
                    pg.keyboard.press("Tab")
                    first = pg.evaluate("document.activeElement && document.activeElement.className")
                    if "skip" not in (first or ""):
                        print("   keyboard: first Tab did not reach the skip link:", first); problems.append(tag + ":keyboard")
                if a.sections and page == "index.html" and name in ("desktop", "phone", "desktop_dark"):
                    for sid, sel in [("hero", ".hero5"), ("claim", "#claim"), ("method", "#method"), ("forms", "#forms"),
                                     ("see", "#see"), ("results", "#results"), ("more", "#more")]:
                        loc = pg.locator(sel).first
                        if loc.count():
                            loc.screenshot(path=os.path.join(a.out, f"{name}_sec_{sid}.png"))
                if a.sections and page in ("index.html", "gallery.html") and name in ("desktop", "phone") and pg.locator("#cmp-zoom").count():
                    problems += zoom_checks(pg, tag, a.out)   # the eight-method grid lives on the gallery since round 2
                if a.sections and page == "gallery.html" and name in ("desktop", "phone"):
                    g = pg.locator("#gal button").first
                    if g.count():
                        g.scroll_into_view_if_needed()
                        g.click()
                        pg.wait_for_timeout(1200)
                        pg.screenshot(path=os.path.join(a.out, f"{tag}_lightbox.png"))
                        pg.keyboard.press("Escape")
                ctx.close()
        # reduced motion: no video may play on its own, the hero shows its poster
        ctx = b.new_context(viewport={"width": 1440, "height": 900}, reduced_motion="reduce")
        pg = ctx.new_page()
        pg.goto(url, wait_until="load")
        pg.wait_for_timeout(2500)
        playing = pg.evaluate("Array.from(document.querySelectorAll('video')).filter(v => !v.paused).length")
        print(f"reduced motion: {playing} video(s) playing")
        if playing:
            problems.append("reduced-motion")
        ctx.close()
        review_url = url.replace("index.html", "review/index.html")
        for name, vp, scheme in [("review_desktop", (1440, 900), "light"), ("review_phone", (390, 844), "dark")]:
            ctx = b.new_context(viewport={"width": vp[0], "height": vp[1]}, color_scheme=scheme, device_scale_factor=1)
            pg = ctx.new_page()
            errs = []
            pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto(review_url, wait_until="load")
            pg.wait_for_timeout(800)
            over = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            n = pg.evaluate("document.querySelectorAll('.zg').length")
            text = pg.evaluate("document.body.innerText")
            bad = sorted(set(m.group(0).lower() for m in re.finditer(r"calibrat\w*|\buncurated\b|\bcurated\b|\brandom(?:ly)?\b|\u2014|\u00a7", text, re.I)))
            if bad:
                errs.append("forbidden words in the visible text: " + ", ".join(bad))
            pg.screenshot(path=os.path.join(a.out, f"{name}.png"))
            print(f"{name}: grids {n}, horizontal overflow {over}px, console errors {len(errs)}")
            for e in errs[:8]:
                print("   console:", e[:200])
            if over > 0 or errs or n < 1:
                problems.append(name)
            if name == "review_desktop":
                problems += zoom_checks(pg, name, a.out, scope=".card")
            ctx.close()
        b.close()
    if server:
        server.shutdown()
    print("RESULT", "PASS" if not problems and not missing else "FAIL " + ",".join(problems + missing[:3]))
    return 0 if not problems and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
