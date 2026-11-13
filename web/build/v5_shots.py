#!/usr/bin/env python3
"""Screenshots and quick metrics of the v5 pages (designer's check; the full check is check_page.py).

For each page, viewport (1440x900 desktop, 390x844 phone) and theme (light, dark): a first-screen shot after the
hero loop has started, then a slow scroll to the bottom (so lazy images, components and loops wake up) and a
full-page shot. Prints per page: height, words of visible text in <main>, console errors, failed requests, and
horizontal overflow. Full-page shots are also cut into 2,000 px tall JPEG segments for reading.

Usage (browser env):
  PLAYWRIGHT_BROWSERS_PATH=/scratch/09032/saini_2/tmp/ms-playwright \
  /work/09032/saini_2/vista/library/uv-envs/pwqa/bin/python web/build/v5_shots.py --out DIR [--pages index.html ...]
      [--only desktop|phone] [--themes light dark] [--no-full] [--file]
Serves web/ on 127.0.0.1 (never 0.0.0.0) unless --file.
"""
import argparse
import functools
import http.server
import json
import os
import threading
import time

from PIL import Image
from playwright.sync_api import sync_playwright

WEB = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
VIEWS = {"desktop": dict(width=1440, height=900, dpr=1, mobile=False), "phone": dict(width=390, height=844, dpr=2, mobile=True)}
WORDS_JS = """() => {
  const m = document.querySelector('main'); if (!m) return 0;
  const w = document.createTreeWalker(m, NodeFilter.SHOW_TEXT, null);
  let n = 0, t;
  while ((t = w.nextNode())) {
    const p = t.parentElement; if (!p) continue;
    const cs = getComputedStyle(p);
    if (cs.display === 'none' || cs.visibility === 'hidden' || p.closest('[hidden],script,style,noscript,.sr-only')) continue;
    if (window.__noComp && p.closest('.comp-mount')) continue;
    n += (t.textContent.match(/[A-Za-z0-9][A-Za-z0-9'’.,%+-]*/g) || []).length;
  }
  return n;
}"""


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a, **k):
        pass


def serve():
    h = functools.partial(Quiet, directory=os.path.dirname(WEB))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def segments(path, h=2000):
    im = Image.open(path).convert("RGB")
    out = []
    for i, y in enumerate(range(0, im.height, h)):
        p = path.replace(".png", f"_s{i}.jpg")
        im.crop((0, y, im.width, min(im.height, y + h))).save(p, quality=82)
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", nargs="+", default=["index.html"])
    ap.add_argument("--only", choices=list(VIEWS))
    ap.add_argument("--themes", nargs="+", default=["light", "dark"])
    ap.add_argument("--no-full", action="store_true")
    ap.add_argument("--file", action="store_true")
    ap.add_argument("--wait", type=float, default=2.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    srv = None if a.file else serve()
    base = ("file://" + os.path.dirname(WEB)) if a.file else f"http://127.0.0.1:{srv.server_address[1]}"
    report = []
    with sync_playwright() as p:
        br = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        for page_name in a.pages:
            for vname, v in VIEWS.items():
                if a.only and vname != a.only:
                    continue
                for theme in a.themes:
                    ctx = br.new_context(viewport={"width": v["width"], "height": v["height"]}, device_scale_factor=v["dpr"],
                                         is_mobile=v["mobile"], has_touch=v["mobile"], color_scheme=theme)
                    pg = ctx.new_page()
                    errs, fails = [], []
                    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
                    pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
                    pg.on("requestfailed", lambda r: fails.append("failed " + r.url) if not r.url.endswith((".mp4", ".webm")) else None)
                    pg.on("response", lambda r: fails.append(f"{r.status} {r.url}") if r.status >= 400 else None)
                    url = f"{base}/web/{page_name}"
                    pg.goto(url, wait_until="load")
                    pg.wait_for_timeout(int(a.wait * 1000))
                    stem = f"{page_name.replace('/', '_').replace('.html', '')}_{vname}_{theme}"
                    first = os.path.join(a.out, stem + "_first.png")
                    pg.screenshot(path=first)
                    cls = 0.0
                    try:
                        cls = pg.evaluate("""() => new Promise(r => { let s = 0; try { new PerformanceObserver(l => { for (const e of l.getEntries()) if (!e.hadRecentInput) s += e.value; }).observe({type: 'layout-shift', buffered: true}); } catch (e) {} setTimeout(() => r(s), 200); })""")
                    except Exception:
                        pass
                    full = None
                    if not a.no_full:
                        hgt = pg.evaluate("document.documentElement.scrollHeight")
                        y = 0
                        while y < hgt:
                            y += int(v["height"] * 0.8)
                            pg.evaluate(f"window.scrollTo(0, {y})")
                            pg.wait_for_timeout(260)
                            hgt = pg.evaluate("document.documentElement.scrollHeight")
                        pg.wait_for_timeout(1500)
                        pg.evaluate("window.scrollTo(0, 0)")
                        pg.wait_for_timeout(600)
                        full = os.path.join(a.out, stem + "_full.png")
                        pg.screenshot(path=full, full_page=True)
                    info = pg.evaluate("""() => ({h: document.documentElement.scrollHeight, sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth})""")
                    words = pg.evaluate(WORDS_JS)
                    pg.evaluate("window.__noComp = true")
                    words_own = pg.evaluate(WORDS_JS)
                    rec = dict(page=page_name, view=vname, theme=theme, height=info["h"], overflow=info["sw"] > info["cw"],
                               words=words, words_without_components=words_own, cls=round(cls, 4), console_errors=errs[:8], failed=sorted(set(fails))[:8],
                               first=first, full=full, segments=segments(full) if full else [])
                    report.append(rec)
                    print(json.dumps({k: rec[k] for k in ("page", "view", "theme", "height", "overflow", "words", "words_without_components", "cls", "console_errors", "failed")}))
                    ctx.close()
        br.close()
    if srv:
        srv.shutdown()
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
