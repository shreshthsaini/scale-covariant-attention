"""Headless check of web/review_qwen (or a preview copy): console errors, missing local files, horizontal overflow on
desktop and phone, a wheel zoom on the first grid (every panel must switch to its zoomed canvas), screenshots light and
dark. Usage: PLAYWRIGHT_BROWSERS_PATH=/scratch/09032/saini_2/tmp/ms-playwright
    /work/09032/saini_2/vista/library/uv-envs/pwqa/bin/python -I web/build/check_review_qwen.py web/review_qwen OUTDIR
Serves the folder on 127.0.0.1 only."""
import functools
import http.server
import json
import os
import re
import sys
import threading

from playwright.sync_api import sync_playwright

root = os.path.abspath(sys.argv[1])            # folder holding index.html
out = os.path.abspath(sys.argv[2])
os.makedirs(out, exist_ok=True)
html = open(os.path.join(root, "index.html")).read()
items = json.loads(re.search(r"window\.ITEMS = (\[.*?\]);</script>", html, re.S).group(1))
missing = [p for it in items for pn in it["panels"] for p in [pn["view"]] + [r[4] for r in pn.get("rt", [])]
           if not os.path.exists(os.path.join(root, p))]
class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


handler = functools.partial(Quiet, directory=root)
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_address[1]}/index.html"
report = {"grids": len(items), "missing_files": missing}
with sync_playwright() as p:
    b = p.chromium.launch()
    for name, vp, scheme in (("desktop", (1366, 900), "light"), ("phone", (390, 844), "light"), ("desktop_dark", (1366, 900), "dark")):
        ctx = b.new_context(viewport={"width": vp[0], "height": vp[1]}, color_scheme=scheme,
                            device_scale_factor=1 if name.startswith("desktop") else 2)
        pg = ctx.new_page()
        errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(url, wait_until="networkidle")
        pg.wait_for_timeout(800)
        over = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        n = pg.evaluate("document.querySelectorAll('.zg').length")
        pg.screenshot(path=os.path.join(out, f"{name}_top.png"))
        zoom = None
        if name == "desktop" and n:
            g = pg.locator(".zg").first
            g.scroll_into_view_if_needed()
            pg.wait_for_timeout(600)
            box = pg.locator(".zg .zg-img").first.bounding_box()
            pg.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
            for _ in range(8):
                pg.mouse.wheel(0, -240)
                pg.wait_for_timeout(120)
            pg.wait_for_timeout(1500)
            zoom = pg.evaluate("""() => { const g = document.querySelector('.zg');
                const c = [...g.querySelectorAll('canvas')].filter(x => x.offsetParent !== null).length;
                return {visible_canvases: c, panels: g.querySelectorAll('.zg-img').length}; }""")
            g.screenshot(path=os.path.join(out, "desktop_zoomed_grid.png"))
        pg.locator("#grids-h").scroll_into_view_if_needed()
        pg.wait_for_timeout(900)
        pg.screenshot(path=os.path.join(out, f"{name}_grids.png"))
        report[name] = {"console_errors": errs, "overflow_px": over, "grids_rendered": n, "zoom": zoom}
        ctx.close()
    b.close()
srv.shutdown()
print(json.dumps(report, indent=1))
