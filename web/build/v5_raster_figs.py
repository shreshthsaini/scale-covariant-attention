#!/usr/bin/env python
"""Rasterize v6 paper figure PDFs (page 1) to 2400 px wide WebP for the web page.

No native PDF renderer is available on Vista, so this uses headless Chromium
(Playwright) running pdf.js 3.11.174 from jsDelivr. A local http.server bound to
127.0.0.1 serves a scratch directory holding a small HTML page and copies of the
PDFs; the page renders each PDF onto a white canvas and returns a PNG data URL.
The PNG is saved to scratch and converted to WebP with Pillow.

Run from anywhere:
  PLAYWRIGHT_BROWSERS_PATH=/scratch/09032/saini_2/tmp/ms-playwright \
    /work/09032/saini_2/vista/library/uv-envs/pwqa/bin/python web/build/v5_raster_figs.py

Inputs : paper/figures/v6/<stem>.pdf (read only)
Outputs: web/assets/img/v6/<stem>.webp and web/assets/img/v6/figs.json
Scratch: /scratch/09032/saini_2/sca/web_builds/v5_webdesign_raster/{serve,png,tmp}
"""
from __future__ import annotations

import base64
import functools
import http.server
import io
import json
import os
import shutil
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "paper" / "figures" / "v6"
OUT_DIR = ROOT / "web" / "assets" / "img" / "v6"
SCRATCH = Path("/scratch/09032/saini_2/sca/web_builds/v5_webdesign_raster")
SERVE_DIR = SCRATCH / "serve"
PNG_DIR = SCRATCH / "png"
TMP_DIR = SCRATCH / "tmp"

STEMS = [
    "observation", "method", "method_zoom", "emulation", "qual_main",
    "sizes_models", "deciles_a", "deciles_b", "crops_matched", "progressive",
]
TARGET_W = 2400
QUALITY = 90
QUALITY_RETRY = 82
MAX_BYTES = 900 * 1000  # 900 KB
PDFJS = "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174"

# Keep Chromium's temp profile and caches off $HOME and /tmp.
for d in (SERVE_DIR, PNG_DIR, TMP_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TMP_DIR)
os.environ["XDG_CACHE_HOME"] = str(TMP_DIR / "xdg_cache")
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/scratch/09032/saini_2/tmp/ms-playwright")

from PIL import Image  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

HTML = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>raster</title>
<script src="{PDFJS}/build/pdf.min.js"></script>
</head><body>
<script>
pdfjsLib.GlobalWorkerOptions.workerSrc = "{PDFJS}/build/pdf.worker.min.js";
window.renderPdf = async function (url, targetW) {{
  const doc = await pdfjsLib.getDocument({{
    url: url,
    cMapUrl: "{PDFJS}/cmaps/",
    cMapPacked: true,
    standardFontDataUrl: "{PDFJS}/standard_fonts/",
  }}).promise;
  const page = await doc.getPage(1);
  const vp1 = page.getViewport({{ scale: 1 }});
  const scale = targetW / vp1.width;
  const vp = page.getViewport({{ scale: scale }});
  const canvas = document.createElement("canvas");
  canvas.width = targetW;
  canvas.height = Math.round(vp.height);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  await page.render({{ canvasContext: ctx, viewport: vp, background: "#ffffff" }}).promise;
  const out = {{
    dataUrl: canvas.toDataURL("image/png"),
    w: canvas.width, h: canvas.height,
    ptW: vp1.width, ptH: vp1.height, numPages: doc.numPages,
  }};
  page.cleanup();
  await doc.destroy();
  return out;
}};
window.rasterReady = true;
</script></body></html>
"""


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # silence per-request logging
        pass


def to_webp(png_path: Path, webp_path: Path) -> tuple[int, int, int, int]:
    with Image.open(png_path) as im:
        im.load()
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        q = QUALITY
        buf = io.BytesIO()
        im.save(buf, "WEBP", quality=q, method=6)
        if buf.tell() > MAX_BYTES:
            print(f"    {webp_path.name}: {buf.tell()} B at q{QUALITY} > {MAX_BYTES} B, retrying at q{QUALITY_RETRY}")
            q = QUALITY_RETRY
            buf = io.BytesIO()
            im.save(buf, "WEBP", quality=q, method=6)
    webp_path.write_bytes(buf.getvalue())
    return w, h, webp_path.stat().st_size, q


def main() -> int:
    (SERVE_DIR / "index.html").write_text(HTML)
    todo = []
    for stem in STEMS:
        src = PDF_DIR / f"{stem}.pdf"
        if not src.exists():
            print(f"skip {stem}: {src} does not exist")
            continue
        shutil.copyfile(src, SERVE_DIR / f"{stem}.pdf")
        todo.append(stem)

    handler = functools.partial(QuietHandler, directory=str(SERVE_DIR))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    print(f"serving {SERVE_DIR} at {base}")

    figs: dict[str, dict] = {}
    failed: list[str] = []
    retried: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--disable-gpu", "--no-sandbox"])
            try:
                page = browser.new_page()
                page.on("pageerror", lambda e: print(f"    pageerror: {e}"))
                page.on("console", lambda m: print(f"    console.{m.type}: {m.text}")
                        if m.type in ("error", "warning") else None)
                page.goto(f"{base}/index.html", wait_until="load", timeout=120_000)
                page.wait_for_function("window.rasterReady === true && !!window.pdfjsLib",
                                       timeout=120_000)
                for stem in todo:
                    try:
                        res = page.evaluate(
                            "([u, w]) => window.renderPdf(u, w)",
                            [f"{base}/{stem}.pdf", TARGET_W],
                        )
                        head, b64 = res["dataUrl"].split(",", 1)
                        assert head == "data:image/png;base64", head
                        png_path = PNG_DIR / f"{stem}.png"
                        png_path.write_bytes(base64.b64decode(b64))
                        w, h, nbytes, q = to_webp(png_path, OUT_DIR / f"{stem}.webp")
                        assert w == TARGET_W, (stem, w)
                        if q != QUALITY:
                            retried.append(stem)
                        figs[stem] = {
                            "file": f"assets/img/v6/{stem}.webp",
                            "width": w, "height": h, "bytes": nbytes,
                        }
                        print(f"ok   {stem}: {res['ptW']:.1f}x{res['ptH']:.1f} pt"
                              f" ({res['numPages']} page(s)) -> {w}x{h} px,"
                              f" webp q{q} {nbytes} B")
                    except Exception as e:  # keep going on per-figure failure
                        failed.append(stem)
                        print(f"FAIL {stem}: {type(e).__name__}: {e}")
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)

    (OUT_DIR / "figs.json").write_text(json.dumps(figs, indent=2) + "\n")
    print(f"wrote {OUT_DIR / 'figs.json'} ({len(figs)} figures)")
    if retried:
        print(f"retried at q{QUALITY_RETRY}: {', '.join(retried)}")
    if failed:
        print(f"failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
