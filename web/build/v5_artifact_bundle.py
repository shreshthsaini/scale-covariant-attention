#!/usr/bin/env python3
"""Artifact preview bundle of the main page (web/index.html), so the page can be looked at as a Claude Artifact
before anything goes to the live site. Only the main page: sub-pages link to the live site.

Builds /scratch/09032/saini_2/sca/web_builds/v5_artifact/bundle/:
  page.html     the page without <!DOCTYPE>, <html>, <head> or <body> (the host wraps it); it starts with
                <title>Scale-Covariant Attention</title> and <meta name="claude:viewport-fit-sides" content="cover">,
                then the stylesheet links, the content and the scripts
  assets/...    every file the page loads, at the same relative paths as on the site (CSS, JS, data, fonts, images,
                videos, posters, captions); nothing loads from another host
  files.json    {"chunks": [{relative path: absolute source path}, ...]} for the Artifact tool, page.html excluded,
                at most 240 files and 55 MB per chunk
What is trimmed for the Artifact limits (480 files, 230 MB, 15 MB per binary, 16 MB per text file): the
percentile explorer keeps native tiles for the 10th, 50th and 90th percentile stops only, and the eight-method grid
keeps its overviews without native tiles; where a grid has no tiles, zoomgrid.js stops at the overview's
resolution instead of requesting missing files. Videos: the roles the main page plays. Links to the paper, the
gallery, the results page, the exploratory page and the 96-prompt review point to the live site and say so.

The bundle is rebuilt in place (nothing on scratch is deleted); files.json is the authoritative list.

Usage, from the repository root:
  build only (any python 3):   /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_artifact_bundle.py
  build and check (browser):   PLAYWRIGHT_BROWSERS_PATH=/scratch/09032/saini_2/tmp/ms-playwright \
                               /work/09032/saini_2/vista/library/uv-envs/pwqa/bin/python web/build/v5_artifact_bundle.py --check
The check serves the bundle on 127.0.0.1, wraps page.html in a minimal skeleton (charset, viewport, a small reset),
blocks every request to another host, and at 1440x900 and 400x844 in light and dark reports console errors,
failed requests and horizontal overflow, and saves first-screen screenshots beside the bundle.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "web"
ROOT = Path("/scratch/09032/saini_2/sca/web_builds/v5_artifact")
OUT = ROOT / "bundle"
LIVE = "https://shreshthsaini.github.io/scale-covariant-attention/"
TITLE = "Scale-Covariant Attention"
ROLES = ["hero16x9", "speedrace", "speedrace9x16", "method_window", "method_emulation", "method_stages",
         "process_1", "process_2", "process_3", "process_4", "process_5", "explainer"]
TILE_STOPS = {10, 50, 90}           # percentile stops that keep their native tiles
DEFER_DROP = {"assets/js/showcase.js"}   # data the main page no longer uses
MAX_FILES, MAX_TOTAL, MAX_BIN, MAX_TEXT = 480, 230e6, 15e6, 16e6
CHUNK_FILES, CHUNK_BYTES = 240, 55e6
TEXT = {".html", ".css", ".js", ".json", ".vtt", ".svg", ".txt"}

files: dict[str, Path] = {}        # published relative path -> source written in the bundle


def put_file(rel: str, src: Path) -> None:
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != src.stat().st_size or dst.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dst)
    files[rel] = dst


def put_text(rel: str, text: str) -> None:
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.read_text(encoding="utf-8") != text:
        dst.write_text(text, encoding="utf-8")
    files[rel] = dst


def js_object(path: Path, var: str):
    t = path.read_text(encoding="utf-8")
    i = t.index(var) + len(var)
    i = t.index("=", i) + 1
    return json.loads(t[i:t.rindex(";")])


def copy_web(rel: str) -> None:
    if rel in files:            # already placed (possibly as a trimmed bundle copy): keep it
        return
    src = WEB / rel
    if not src.exists():
        raise SystemExit(f"missing source file {rel}")
    put_file(rel, src)


def css_refs(rel: str) -> None:
    """Copy a stylesheet and every url(...) it names (fonts, images), at the same relative paths."""
    copy_web(rel)
    base = (WEB / rel).parent
    for u in re.findall(r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)", (WEB / rel).read_text(encoding="utf-8")):
        if u.startswith(("data:", "http:", "https:", "#")):
            continue
        tgt = (base / u).resolve()
        if tgt.exists():
            copy_web(tgt.relative_to(WEB.resolve()).as_posix())


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = (WEB / "index.html").read_text(encoding="utf-8")
    head = src[src.index("<head>") + 6: src.index("</head>")]
    body = src[src.index("<body>") + 6: src.rindex("</body>")]
    desc = re.search(r'<meta name="description" content="([^"]*)">', head).group(1)
    theme_js = re.search(r"<script>(document\.documentElement\.classList\.add.*?)</script>", head).group(1)

    # ---------------------------------------------------------------- links: sub-pages and the paper on the live site
    note = '<span class="live-note">on the live site</span>'

    def live(m):
        pre, href, rest, inner = m.group(1), m.group(2), m.group(3), m.group(4)
        target = LIVE + ("paper.pdf" if href == "../paper/main.pdf" else href)
        if href != "../paper/main.pdf":
            inner = inner.replace('<span class="go">Open', '<span class="go">Open on the live site') if 'class="go"' in inner else inner + " " + note
        return f'<a {pre}href="{target}" target="_blank" rel="noopener"{rest}>{inner}</a>'
    body = re.sub(r'<a ([^>]*?)href="(\.\./paper/main\.pdf|gallery\.html|results\.html|exploratory\.html|review/index\.html)"([^>]*)>(.*?)</a>',
                  live, body, flags=re.S)
    if re.search(r'href="(?!https://|#)[^"]+\.html', body):
        raise SystemExit("a relative link to another page is left in the bundle page")

    # ---------------------------------------------------------------- deferred data scripts the bundle carries
    m = re.search(r"window\.SCA_DEFER = (\[.*?\]);", body)
    defer = [d for d in json.loads(m.group(1)) if d not in DEFER_DROP]
    body = body.replace(m.group(0), "window.SCA_DEFER = " + json.dumps(defer) + ";")
    links = re.findall(r'<link rel="stylesheet" href="([^"]+)">', head)
    preloads = re.findall(r'<link rel="preload" href="(assets/fonts/[^"]+)" as="font" type="font/woff2" crossorigin>', head)
    page = (f"<title>{TITLE}</title>\n"
            '<meta name="claude:viewport-fit-sides" content="cover">\n'
            f'<meta name="description" content="{desc}">\n'
            + "".join(f'<link rel="stylesheet" href="{h}">\n' for h in links)
            + "".join(f'<link rel="preload" href="{h}" as="font" type="font/woff2" crossorigin>\n' for h in preloads)
            + "<style>.live-note{font-size:.72em;font-weight:500;opacity:.7;margin-left:.35em;white-space:nowrap}</style>\n"
            + f"<script>{theme_js}</script>\n" + body.strip() + "\n")
    put_text("page.html", page)

    for h in links:
        css_refs(h)
    # ---------------------------------------------------------------- registry: only bundled components and videos
    reg = js_object(WEB / "assets/js/v5_assets.js", "window.SCA_V5")
    videos = {r: reg["videos"][r] for r in ROLES if r in reg["videos"]}
    for v in videos.values():
        for k in ("file", "poster", "vtt", "still"):
            if v.get(k):
                copy_web(v[k])
    comps = {}
    for name, c in sorted(reg["comps"].items()):
        comps[name] = c
        copy_web(c["js"])
        if c.get("css"):
            css_refs(c["css"])
        d = WEB / "assets/data" / name
        if not d.is_dir():
            continue
        if name == "percentiles":
            data = js_object(d / "data.js", "window.SCA_DATA.percentiles")
            for st in data["stops"]:
                for cell in st["cells"].values():
                    copy_web(f"assets/data/percentiles/{cell['view']}")
                    t = cell.get("tiles")
                    if not t:
                        continue
                    if st["p"] in TILE_STOPS:
                        for r in range(t["rows"]):
                            for c_ in range(t["cols"]):
                                copy_web("assets/data/percentiles/" + t["path"].replace("{c}", str(c_)).replace("{r}", str(r)))
                    else:
                        del cell["tiles"]
            head_js = (d / "data.js").read_text(encoding="utf-8").split("window.SCA_DATA.percentiles")[0]
            put_text("assets/data/percentiles/data.js", head_js + "window.SCA_DATA.percentiles = "
                     + json.dumps(data, ensure_ascii=True, separators=(",", ":")) + ";\n")
            if (d / "fallback.webp").exists():
                copy_web("assets/data/percentiles/fallback.webp")
        else:
            for f in sorted(d.rglob("*")):
                if f.is_file():
                    copy_web(f.relative_to(WEB).as_posix())
    put_text("assets/js/v5_assets.js", "// Bundle copy of assets/js/v5_assets.js: the components and videos this bundle carries.\n"
             "window.SCA_V5 = " + json.dumps({"comps": comps, "videos": videos}, indent=1, ensure_ascii=True) + ";\n")

    # ---------------------------------------------------------------- data scripts
    put_text("assets/js/media.js", "// Bundle: the v4 teaser and explainer are not carried; the v5 videos replace them.\nwindow.SCA_MEDIA = {};\n")
    put_text("assets/js/traces.js", "// Bundle: the single-pass trace videos are not carried; the v5 process videos replace them.\n"
             'window.SCA_TRACES = {"available": []};\n')
    zs = js_object(WEB / "assets/js/zoomsets.js", "window.SCA_ZOOMSETS")
    images = js_object(WEB / "assets/js/images.js", "window.SCA_IMAGES")
    thumb_of = {r.get("sid"): r.get("thumb") for r in (images.get("compare") or {}).get("items", []) if r.get("thumb")}
    comp_sets = zs.get("compare", [])
    keep_thumbs = []
    for z in comp_sets:
        for p in z["panels"]:
            p.pop("rt", None)
            p.pop("tiles", None)
            copy_web(p["view"])
        th = z.get("thumb") or thumb_of.get(z["sid"]) or z["panels"][0]["view"]
        copy_web(th)
        keep_thumbs.append({"sid": z["sid"], "thumb": th})
    put_text("assets/js/zoomsets.js", "// Bundle copy: the eight-method sets with overviews only (no native tiles).\n"
             "window.SCA_ZOOMSETS = " + json.dumps({"compare": comp_sets}, ensure_ascii=True, separators=(",", ":")) + ";\n")
    put_text("assets/js/images.js", "// Bundle copy: the comparison thumbnails only.\n"
             "window.SCA_IMAGES = " + json.dumps({"compare": {"items": keep_thumbs}}, ensure_ascii=True) + ";\n")
    for rel in ["assets/js/numbers.js", "assets/js/site.js", "assets/js/zoomgrid.js", "assets/js/process.js",
                "assets/js/method_data.js", "assets/js/v6_figs.js"]:
        copy_web(rel)
    # ---------------------------------------------------------------- every other file the page names
    for ref in re.findall(r'(?:src|data-src|poster|href)="(assets/[^"]+)"', page):
        copy_web(ref)
    for ref in re.findall(r'<script src="(assets/[^"]+)"', page):
        copy_web(ref)
    for d in defer:
        if d not in files:
            copy_web(d)


def verify() -> list[str]:
    """Every assets/... path named in a bundled text file exists in the bundle; limits hold."""
    problems = []
    for rel, p in list(files.items()):
        if p.suffix in {".js", ".css", ".html", ".json"}:
            t = p.read_text(encoding="utf-8", errors="replace")
            for ref in set(re.findall(r'["\'(](assets/[A-Za-z0-9_./-]+\.(?:webp|png|jpg|mp4|vtt|svg|js|css|woff2))', t)):
                if "{" in ref:
                    continue
                if ref not in files and not (OUT / ref).exists():
                    if rel in ("assets/js/numbers.js", "assets/js/method_data.js", "assets/js/v6_figs.js"):
                        continue          # data strings, not loads
                    problems.append(f"{rel} names {ref}, which is not in the bundle")
    n = len(files)
    total = sum(p.stat().st_size for p in files.values())
    if n > MAX_FILES:
        problems.append(f"{n} files > {MAX_FILES}")
    if total > MAX_TOTAL:
        problems.append(f"{total / 1e6:.1f} MB > {MAX_TOTAL / 1e6:.0f} MB")
    for rel, p in files.items():
        sz = p.stat().st_size
        if p.suffix in TEXT and sz > MAX_TEXT:
            problems.append(f"text file {rel} is {sz / 1e6:.1f} MB")
        if p.suffix not in TEXT and sz > MAX_BIN:
            problems.append(f"binary {rel} is {sz / 1e6:.1f} MB")
    if files["page.html"].stat().st_size > 1e6:
        problems.append("page.html is over 1 MB")
    head = files["page.html"].read_text(encoding="utf-8")[:8192]
    if not head.startswith(f"<title>{TITLE}</title>") or re.search(r"<!DOCTYPE|<html[\s>]|<head[\s>]|<body[\s>]", head, re.I):
        problems.append("page.html does not follow the page contract (title first; no doctype, html, head or body)")
    if re.search(r'(?:src|href)="(?!https://|#|assets/|data:)[^"]*"', files["page.html"].read_text(encoding="utf-8")):
        problems.append("page.html names a path outside the bundle")
    return problems


def write_manifest() -> dict:
    items = sorted((rel, p) for rel, p in files.items() if rel != "page.html")
    chunks, cur, cur_b = [], {}, 0
    for rel, p in items:
        sz = p.stat().st_size
        if cur and (len(cur) >= CHUNK_FILES or cur_b + sz > CHUNK_BYTES):
            chunks.append(cur)
            cur, cur_b = {}, 0
        cur[rel] = str(p)
        cur_b += sz
    if cur:
        chunks.append(cur)
    (OUT / "files.json").write_text(json.dumps({"chunks": chunks}, indent=1) + "\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in files.values())
    info = {"bundle": str(OUT), "files": len(files), "total_mb": round(total / 1e6, 1), "chunks": len(chunks),
            "chunk_sizes": [(len(c), round(sum(Path(v).stat().st_size for v in c.values()) / 1e6, 1)) for c in chunks]}
    print(json.dumps(info))
    return info


def check() -> int:
    import functools
    import http.server
    import threading
    from playwright.sync_api import sync_playwright

    wrap = ("<!doctype html>\n<html lang=\"en\"{theme}>\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            "<style>*,*::before,*::after{{box-sizing:border-box}}html,body{{margin:0;padding:0}}</style>\n"
            "</head>\n<body>\n{page}</body>\n</html>\n")
    page = files["page.html"].read_text(encoding="utf-8")
    for theme in ("light", "dark"):
        (OUT / f"_check_{theme}.html").write_text(wrap.format(theme=f' data-theme="{theme}"', page=page), encoding="utf-8")

    class Q(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def handle(self):
            try:
                super().handle()
            except (ConnectionResetError, BrokenPipeError):
                pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Q, directory=str(OUT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    bad = 0
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        for (vw, vh, tag) in [(1440, 900, "1440"), (400, 844, "400")]:
            for theme in ("light", "dark"):
                ctx = b.new_context(viewport={"width": vw, "height": vh}, color_scheme=theme,
                                    device_scale_factor=1 if vw > 500 else 2, is_mobile=vw < 500, has_touch=vw < 500)
                pg = ctx.new_page()
                errs, fails, blocked = [], [], []

                def route(r):
                    if r.request.url.startswith(f"http://127.0.0.1:{port}/"):
                        r.continue_()
                    else:
                        blocked.append(r.request.url)
                        r.abort()
                pg.route("**/*", route)
                pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
                pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
                pg.on("requestfailed", lambda r: fails.append("failed " + r.url) if not r.url.endswith((".mp4",)) else None)
                pg.on("response", lambda r: fails.append(f"HTTP {r.status} {r.url}") if r.status >= 400 else None)
                pg.goto(f"http://127.0.0.1:{port}/_check_{theme}.html", wait_until="load")
                pg.wait_for_timeout(2500)
                shot = ROOT / f"shot_{tag}_{theme}.png"
                pg.screenshot(path=str(shot))
                h = pg.evaluate("document.documentElement.scrollHeight")
                y = 0
                while y < h:
                    y += int(vh * .8)
                    pg.evaluate(f"window.scrollTo(0, {y})")
                    pg.wait_for_timeout(200)
                    h = pg.evaluate("document.documentElement.scrollHeight")
                pg.wait_for_timeout(1500)
                over = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
                live = pg.evaluate("Array.from(document.querySelectorAll('.comp-figure.is-live .comp-mount')).map(m => m.getAttribute('data-comp'))")
                vids = pg.evaluate("document.querySelectorAll('video').length")
                ok = not errs and not fails and not blocked and over <= 0
                bad += 0 if ok else 1
                print(json.dumps({"view": f"{vw}x{vh}", "theme": theme, "height": h, "overflow_px": over, "console_errors": errs[:6],
                                  "failed": sorted(set(fails))[:6], "blocked_external": sorted(set(blocked))[:6],
                                  "components_live": live, "videos": vids, "screenshot": str(shot)}))
                ctx.close()
        b.close()
    srv.shutdown()
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="also run the headless check (needs playwright)")
    a = ap.parse_args()
    build()
    problems = verify()
    write_manifest()
    for pr in problems:
        print("PROBLEM", pr)
    bad = check() if a.check else 0
    print("RESULT", "PASS" if not problems and not bad else "FAIL")
    raise SystemExit(1 if problems or bad else 0)


if __name__ == "__main__":
    main()
