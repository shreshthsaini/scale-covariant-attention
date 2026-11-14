#!/usr/bin/env python3
"""Artifact preview bundle of the main page (web/index.html), so the page can be looked at as a Claude Artifact
before anything goes to the live site. Only the main page: sub-pages link to the live site.

Builds /scratch/09032/saini_2/sca/web_builds/v5_artifact/bundle_r2/:
  page.html     the page without <!DOCTYPE>, <html>, <head> or <body> (the host wraps it); it starts with
                <title>Scale-Covariant Attention</title> and <meta name="claude:viewport-fit-sides" content="cover">,
                then the stylesheet links, the content and the scripts
  assets/...    every file the page loads, at the same relative paths as on the site; nothing loads from another host
  files.json    {"chunks": [{relative path: absolute source path}, ...]} for the Artifact tool, page.html excluded,
                at most 240 files and 55 MB per chunk; chunk 0 holds everything the page needs, zoom tiles come last
  changed.json  published paths that are new or whose bytes differ from the previous published bundle (--prev)
  removed.json  published paths of the previous bundle that this page no longer uses
The file set follows the page: the components it mounts (data-comp), the videos it plays (hero, speed race, the
step-2 loop, the formation or process carousel, the explainer), the data scripts it defers, the images it names.
Only types the Artifact host serves are carried (.vtt captions are left out and their track dropped). No native
zoom tiles: the percentile explorer zooms to its overviews and zoomgrid.js stops there. Links to the paper, the
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
import hashlib
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "web"
ROOT = Path("/scratch/09032/saini_2/sca/web_builds/v5_artifact")
OUT = ROOT / "bundle_r2"   # round 2 (18:45): a fresh folder, so no file of the first bundle lingers beside it
PREV = Path("/scratch/09032/saini_2/tmp/tmp/claude-883311/-work-09032-saini-2-vista-Projects-Exploration/"
            "4c03b520-8e32-49ed-8c50-e539347fc179/scratchpad/sca_page")
LIVE = "https://shreshthsaini.github.io/scale-covariant-attention/"
TITLE = "Scale-Covariant Attention"
FIXED_ROLES = ["hero16x9", "speedrace", "speedrace9x16", "method_window", "explainer"]
SERVED = {".html", ".css", ".js", ".mjs", ".wasm", ".json", ".map", ".txt", ".md", ".csv", ".xml", ".svg", ".png",
          ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".ico", ".woff", ".woff2", ".ttf", ".otf", ".mp3", ".wav",
          ".ogg", ".mp4", ".webm", ".pdf"}
MAX_FILES, MAX_TOTAL, MAX_BIN, MAX_TEXT = 480, 230e6, 15e6, 16e6
CHUNK_FILES, CHUNK_BYTES = 240, 55e6
TEXT = {".html", ".css", ".js", ".json", ".svg", ".txt"}
TILE = re.compile(r"/\d+_\d+\.webp$")

files: dict[str, Path] = {}        # published relative path -> file written in the bundle
skipped: set[str] = set()


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
    if not src.is_file():       # never a directory
        if src.exists():
            return
        raise SystemExit(f"missing source file {rel}")
    if src.suffix.lower() not in SERVED:
        skipped.add(rel)
        return
    put_file(rel, src)


def css_refs(rel: str) -> None:
    """Copy a stylesheet and every file url(...) it names (fonts, images), at the same relative paths."""
    copy_web(rel)
    base = (WEB / rel).parent
    for u in re.findall(r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)", (WEB / rel).read_text(encoding="utf-8")):
        if u.startswith(("data:", "http:", "https:", "#")):
            continue
        tgt = (base / u).resolve()
        if tgt.is_file():
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
            inner = inner + " " + note
        return f'<a {pre}href="{target}" target="_blank" rel="noopener"{rest}>{inner}</a>'
    body = re.sub(r'<a ([^>]*?)href="(\.\./paper/main\.pdf|gallery\.html|results\.html|exploratory\.html|review/index\.html)"([^>]*)>(.*?)</a>',
                  live, body, flags=re.S)
    if re.search(r'href="(?!https://|#)[^"]+\.html', body):
        raise SystemExit("a relative link to another page is left in the bundle page")

    m = re.search(r"window\.SCA_DEFER = (\[.*?\]);", body)
    defer = json.loads(m.group(1)) if m else []
    links = re.findall(r'<link rel="stylesheet" href="([^"]+)">', head)
    page = (f"<title>{TITLE}</title>\n"
            '<meta name="claude:viewport-fit-sides" content="cover">\n'
            f'<meta name="description" content="{desc}">\n'
            + "".join(f'<link rel="stylesheet" href="{h}">\n' for h in links)
            + "<style>.live-note{font-size:.72em;font-weight:500;opacity:.7;margin-left:.35em;white-space:nowrap}</style>\n"
            + f"<script>{theme_js}</script>\n" + body.strip() + "\n")
    put_text("page.html", page)
    for h in links:
        css_refs(h)

    # ---------------------------------------------------------------- components the page mounts; videos it plays
    reg = js_object(WEB / "assets/js/v5_assets.js", "window.SCA_V5")
    mounted = set(re.findall(r'data-comp="([a-z]+)"', page))
    allv = reg["videos"]
    carousel = [k for k in allv if k.startswith("formation")] or [k for k in allv if k.startswith("process")]
    videos = {}
    for r in FIXED_ROLES + sorted(carousel):
        if r in allv:
            v = dict(allv[r])
            v.pop("vtt", None)                       # the host does not serve .vtt; captions are burned in
            videos[r] = v
            for k in ("file", "poster", "still"):
                if v.get(k):
                    copy_web(v[k])
    comps = {}
    for name, c in sorted(reg["comps"].items()):
        if name not in mounted:
            continue
        comps[name] = c
        copy_web(c["js"])
        if c.get("css"):
            css_refs(c["css"])
        d = WEB / "assets/data" / name
        if not d.is_dir():
            continue
        if name == "percentiles":                     # overviews only: the zoom stops there
            data = js_object(d / "data.js", "window.SCA_DATA.percentiles")
            for st in data["stops"]:
                for cell in st["cells"].values():
                    copy_web(f"assets/data/percentiles/{cell['view']}")
                    cell.pop("tiles", None)
            head_js = (d / "data.js").read_text(encoding="utf-8").split("window.SCA_DATA.percentiles")[0]
            put_text("assets/data/percentiles/data.js", head_js + "window.SCA_DATA.percentiles = "
                     + json.dumps(data, ensure_ascii=True, separators=(",", ":")) + ";\n")
            if (d / "fallback.webp").is_file():
                copy_web("assets/data/percentiles/fallback.webp")
        else:
            for f in sorted(d.rglob("*")):
                if f.is_file() and not TILE.search(f.as_posix()):
                    copy_web(f.relative_to(WEB).as_posix())
    put_text("assets/js/v5_assets.js", "// Bundle copy of assets/js/v5_assets.js: the components and videos this page uses.\n"
             "window.SCA_V5 = " + json.dumps({"comps": comps, "videos": videos}, indent=1, ensure_ascii=True) + ";\n")
    put_text("assets/js/media.js", "// Bundle: the v4 teaser and explainer are not carried; the v5 videos replace them.\nwindow.SCA_MEDIA = {};\n")
    put_text("assets/js/traces.js", "// Bundle: the single-pass trace videos are not carried; the v5 videos replace them.\n"
             'window.SCA_TRACES = {"available": []};\n')
    for rel in ["assets/js/site.js"] + defer:
        copy_web(rel)
    for ref in re.findall(r'(?:src|data-src|poster|href)="(assets/[^"]+)"', page):
        copy_web(ref)
    for ref in re.findall(r'<script src="(assets/[^"]+)"', page):
        copy_web(ref)


def verify() -> list[str]:
    """Every assets/... path named in a bundled page, style or script exists in the bundle; limits hold."""
    problems = []
    for rel, p in list(files.items()):
        if p.suffix in {".js", ".css", ".html"}:
            t = p.read_text(encoding="utf-8", errors="replace")
            if p.suffix in {".js", ".css"}:
                t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)     # paths named in comments are not loads
            for ref in set(re.findall(r'["\'(](assets/[A-Za-z0-9_./-]+\.(?:webp|png|jpg|mp4|vtt|svg|js|css|woff2))', t)):
                if "{" in ref or ref in files:
                    continue
                if rel in ("assets/js/numbers.js",):
                    continue          # data strings, not loads
                problems.append(f"{rel} names {ref}, which is not in the bundle")
    n, total = len(files), sum(p.stat().st_size for p in files.values())
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
        if p.suffix.lower() not in SERVED:
            problems.append(f"{rel} has a type the host does not serve")
        if not p.is_file():
            problems.append(f"{rel} is not a file")
    if files["page.html"].stat().st_size > 1e6:
        problems.append("page.html is over 1 MB")
    head = files["page.html"].read_text(encoding="utf-8")[:8192]
    if not head.startswith(f"<title>{TITLE}</title>") or re.search(r"<!DOCTYPE|<html[\s>]|<head[\s>]|<body[\s>]", head, re.I):
        problems.append("page.html does not follow the page contract (title first; no doctype, html, head or body)")
    if re.search(r'(?:src|href)="(?!https://|#|assets/|data:)[^"]*"', files["page.html"].read_text(encoding="utf-8")):
        problems.append("page.html names a path outside the bundle")
    return problems


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_manifest(prev: Path) -> dict:
    items = sorted((rel, p) for rel, p in files.items() if rel != "page.html")
    items.sort(key=lambda x: (bool(TILE.search(x[0])), x[0]))     # chunk 0: everything but zoom tiles
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
    # ---------------------------------------------------------------- against the previous published bundle
    changed, removed = [], []
    if prev.is_dir():
        old = set()
        pf = prev / "files.json"
        if pf.is_file():
            for ch in json.loads(pf.read_text(encoding="utf-8")).get("chunks", []):
                old |= set(ch)
        old.add("page.html")
        for rel, p in sorted(files.items()):
            q = prev / rel
            if not q.is_file() or q.stat().st_size != p.stat().st_size or digest(q) != digest(p):
                changed.append(rel)
        removed = sorted(r for r in old if r not in files)
    else:
        changed = sorted(files)
    (OUT / "changed.json").write_text(json.dumps(changed, indent=1) + "\n", encoding="utf-8")
    (OUT / "removed.json").write_text(json.dumps(removed, indent=1) + "\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in files.values())
    info = {"bundle": str(OUT), "files": len(files), "total_mb": round(total / 1e6, 1), "chunks": len(chunks),
            "chunk_sizes": [(len(c), round(sum(Path(v).stat().st_size for v in c.values()) / 1e6, 1)) for c in chunks],
            "changed": len(changed), "removed": len(removed), "left_out_types": sorted(skipped)}
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
    allowed = set(files) | {"_check_light.html", "_check_dark.html"}
    bad = 0
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        for (vw, vh, tag) in [(1440, 900, "1440"), (400, 844, "400")]:
            for theme in ("light", "dark"):
                ctx = b.new_context(viewport={"width": vw, "height": vh}, color_scheme=theme,
                                    device_scale_factor=1 if vw > 500 else 2, is_mobile=vw < 500, has_touch=vw < 500)
                pg = ctx.new_page()
                errs, fails, blocked, unlisted = [], [], [], []

                def route(r):
                    u = r.request.url
                    if u.startswith(f"http://127.0.0.1:{port}/"):
                        rel = u.split(f":{port}/", 1)[1].split("?")[0].split("#")[0]
                        if rel and rel not in allowed:
                            unlisted.append(rel)     # a file the page asks for that the Artifact would not have
                        r.continue_()
                    else:
                        blocked.append(u)
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
                    pg.wait_for_timeout(220)
                    h = pg.evaluate("document.documentElement.scrollHeight")
                pg.wait_for_timeout(1500)
                over = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
                live = pg.evaluate("Array.from(document.querySelectorAll('.comp-figure.is-live .comp-mount')).map(m => m.getAttribute('data-comp'))")
                vids = pg.evaluate("document.querySelectorAll('video').length")
                ok = not errs and not fails and not blocked and not unlisted and over <= 0
                bad += 0 if ok else 1
                print(json.dumps({"view": f"{vw}x{vh}", "theme": theme, "height": h, "overflow_px": over, "console_errors": errs[:6],
                                  "failed": sorted(set(fails))[:6], "blocked_external": sorted(set(blocked))[:6],
                                  "unlisted": sorted(set(unlisted))[:6], "components_live": live, "videos": vids,
                                  "screenshot": str(shot)}))
                ctx.close()
        b.close()
    srv.shutdown()
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="also run the headless check (needs playwright)")
    ap.add_argument("--prev", default=str(PREV), help="the previous published bundle, for changed.json and removed.json")
    a = ap.parse_args()
    build()
    problems = verify()
    write_manifest(Path(a.prev))
    for pr in problems:
        print("PROBLEM", pr)
    bad = check() if a.check else 0
    print("RESULT", "PASS" if not problems and not bad else "FAIL")
    raise SystemExit(1 if problems or bad else 0)


if __name__ == "__main__":
    main()
