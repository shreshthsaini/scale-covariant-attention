"""Headless check of web/index.html: console errors, broken local files, horizontal overflow, screenshots.

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


def local_refs():
    """Every src/href/poster in index.html and every asset path in the generated data files must exist."""
    html = open(os.path.join(WEB, "index.html")).read()
    refs = set(re.findall(r'(?:src|href|poster|data-native)="([^"#:]+)"', html))
    for js in ("images.js", "ideogram_manifest.js"):
        p = os.path.join(WEB, "assets", "js", js)
        if os.path.exists(p):
            refs |= set(re.findall(r'"(assets/[^"]+\.(?:webp|mp4|png|jpg|vtt|srt))"', open(p).read()))
    tr = os.path.join(WEB, "assets", "js", "traces.js")
    if os.path.exists(tr):                           # only rendered trace videos must exist; expected ones are placeholders
        t = open(tr).read()
        for v in json.loads(t[t.index("{"):t.rindex("}") + 1]).get("available", []):
            refs |= {v["mp4"], v["poster"]}
    missing = []
    for r in sorted(refs):
        if r.startswith("assets/ideogram/"):
            continue                                 # optional by design: placeholders until the files are added
        if not os.path.exists(os.path.normpath(os.path.join(WEB, r))):
            missing.append(r)
    return refs, missing


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
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=WEB)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}/index.html"
    else:
        url = "file://" + os.path.join(WEB, "index.html")
    problems = []
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--disable-gpu", "--no-sandbox"])
        for name, vp, scheme in [("desktop", (1440, 900), "light"), ("desktop_dark", (1440, 900), "dark"),
                                 ("phone", (390, 844), "light"), ("phone_dark", (390, 844), "dark")]:
            ctx = b.new_context(viewport={"width": vp[0], "height": vp[1]}, color_scheme=scheme,
                                device_scale_factor=1)
            pg = ctx.new_page()
            errs = []
            pg.on("console", lambda m: errs.append(m.text + " " + str((m.location or {}).get("url", ""))) if m.type == "error" else None)
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto(url, wait_until="load")
            h = pg.evaluate("document.documentElement.scrollHeight")
            for y in range(0, h, 700):                  # trigger lazy images
                pg.evaluate(f"window.scrollTo(0, {y})")
                pg.wait_for_timeout(60)
            pg.wait_for_timeout(800)
            pg.evaluate("window.scrollTo(0, 0)")
            over = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            broken = pg.evaluate("""Array.from(document.images).filter(i => i.complete && i.naturalWidth === 0
                                     && !i.src.includes('/ideogram/')).map(i => i.src)""")
            n_img = pg.evaluate("document.images.length")
            pg.screenshot(path=os.path.join(a.out, f"{name}.png"), full_page=True)
            print(f"{name}: images {n_img}, broken {len(broken)}, horizontal overflow {over}px, console errors {len(errs)}")
            errs = [e for e in errs if "/ideogram/" not in e]   # missing Ideogram files are expected placeholders
            for e in errs[:8]:
                print("   console:", e[:200])
            for u in broken[:8]:
                print("   broken:", u)
            if over > 0 or broken or errs:
                problems.append(name)
            if a.sections and name in ("desktop", "phone"):
                for sid in ["top", "method", "compare", "traces", "results", "scale", "commercial", "failures"]:
                    loc = pg.locator("#" + sid).first
                    if loc.count():
                        loc.screenshot(path=os.path.join(a.out, f"{name}_sec_{sid}.png"))
            ctx.close()
        b.close()
    if server:
        server.shutdown()
    print("RESULT", "PASS" if not problems and not missing else "FAIL " + ",".join(problems + missing[:3]))
    return 0 if not problems and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
