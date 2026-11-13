"""Headless check of the interactive components' demo pages (web/dev/<name>.html): console errors, failed local
files, horizontal overflow, the mount's height against 80vh, and screenshots after scripted interactions, at
1440x900 and 390x844 (touch), light and dark.

Usage (repository root; an env with playwright and its Chromium build):
  PLAYWRIGHT_BROWSERS_PATH=/scratch/09032/saini_2/tmp/ms-playwright \
  /work/09032/saini_2/vista/library/uv-envs/pwqa/bin/python web/build/v5_check_components.py percentiles \
      [heads emulate table speed] --out /scratch/09032/saini_2/sca/web_builds/v5_webinteractive_1/shots
Pages open from file://. Prints one line per page and view and exits non-zero on any console error or overflow.
"""
import argparse
import json
import os
import sys

from playwright.sync_api import sync_playwright

WEB = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
VIEWS = {"desk": dict(viewport={"width": 1440, "height": 900}, device_scale_factor=1),
         "phone": dict(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True)}


def mount_box(page):
    return page.evaluate("""() => { const m = document.querySelector('.comp-mount'); const r = m.getBoundingClientRect();
      return {h: r.height, w: r.width, vh: innerHeight, sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth}; }""")


def shot(page, out, tag):
    fig = page.query_selector("figure")
    p = os.path.join(out, tag + ".png")
    (fig or page).screenshot(path=p)
    return p


def tap(page, x, y):
    page.touchscreen.tap(x, y)


def swipe(page, x0, y0, x1, y1, steps=8):
    """A one-finger horizontal swipe from synthetic touch pointer events (CDP)."""
    cdp = page.context.new_cdp_session(page)
    cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x0, "y": y0}]})
    for i in range(1, steps + 1):
        x = x0 + (x1 - x0) * i / steps
        y = y0 + (y1 - y0) * i / steps
        cdp.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": x, "y": y}]})
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})


# ------------------------------------------------------------------------------------------------ per component
def run_percentiles(page, view, out, tag, log):
    page.wait_for_selector(".c-percentiles .zg-p img", timeout=20000)
    page.wait_for_timeout(1200)
    log.append(("start", page.evaluate("() => document.querySelector('.c-percentiles-pct').textContent + ' ' + document.querySelector('.c-percentiles-diff-v').textContent")))
    shots = [shot(page, out, f"{tag}_0_median")]
    rng = page.query_selector(".c-percentiles-range")
    rng.focus()
    for _ in range(4):
        page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(1500)
    log.append(("after 4x left", page.evaluate("() => document.querySelector('.c-percentiles-pct').textContent + ' ' + document.querySelector('.c-percentiles-diff-v').textContent")))
    shots.append(shot(page, out, f"{tag}_1_p10"))
    # zoom on the SCA image at its upper third
    box = page.query_selector_all(".c-percentiles .zg-p")[3].bounding_box()
    x, y = box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.35
    if view == "phone":
        tap(page, x, y)
    else:
        page.mouse.click(x, y)
    page.wait_for_timeout(3500)
    log.append(("zoom readout", page.evaluate("() => (document.querySelector('.c-percentiles .zg-read')||{}).textContent")))
    shots.append(shot(page, out, f"{tag}_2_zoom"))
    page.keyboard.press("Escape") if view == "desk" else page.query_selector(".c-percentiles .zg-reset").click()
    page.wait_for_timeout(2000)
    if view == "phone":
        b = page.query_selector_all(".c-percentiles .zg-p")[0].bounding_box()
        swipe(page, b["x"] + b["width"] * 0.9, b["y"] + b["height"] * 0.5, b["x"] + b["width"] * 0.05, b["y"] + b["height"] * 0.52)
        page.wait_for_timeout(1500)
        log.append(("after swipe left", page.evaluate("() => document.querySelector('.c-percentiles-pct').textContent")))
    else:
        cb = page.query_selector(".c-percentiles-chart").bounding_box()
        page.mouse.click(cb["x"] + cb["width"] * 0.9, cb["y"] + cb["height"] * 0.5)
        page.wait_for_timeout(1500)
        log.append(("after chart click at 90%", page.evaluate("() => document.querySelector('.c-percentiles-pct').textContent + ' ' + document.querySelector('.c-percentiles-diff-v').textContent")))
    shots.append(shot(page, out, f"{tag}_3_after"))
    return shots


RUNNERS = {"percentiles": run_percentiles}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="desk,phone")
    ap.add_argument("--themes", default="light,dark")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bad = 0
    with sync_playwright() as pw:
        br = pw.chromium.launch()
        for name in a.names:
            url = "file://" + os.path.join(WEB, "dev", name + ".html")
            for view in a.views.split(","):
                for theme in a.themes.split(","):
                    ctx = br.new_context(color_scheme=theme, **VIEWS[view])
                    page = ctx.new_page()
                    errs = []
                    page.on("console", lambda m, errs=errs: errs.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning", "log") else None)
                    page.on("pageerror", lambda e, errs=errs: errs.append(f"pageerror: {e}"))
                    page.on("requestfailed", lambda r, errs=errs: errs.append(f"requestfailed: {r.url}"))
                    page.goto(url)
                    page.wait_for_timeout(300)
                    page.evaluate("() => document.querySelector('figure').scrollIntoView({block: 'start'})")
                    tag = f"{name}_{view}_{theme}"
                    log = []
                    try:
                        shots = RUNNERS[name](page, view, a.out, tag, log)
                    except Exception as e:  # noqa: BLE001
                        errs.append(f"runner: {e!r}")
                        shots = [shot(page, a.out, tag + "_fail")]
                    m = mount_box(page)
                    over = m["sw"] > m["cw"]
                    tall = m["h"] > 0.8 * m["vh"]
                    ok = not errs and not over
                    bad += not ok
                    print(json.dumps(dict(page=name, view=view, theme=theme, ok=ok, errors=errs[:6], overflow=over,
                                          mount_h=round(m["h"]), vh80=round(0.8 * m["vh"]), taller_than_80vh=tall,
                                          log=log, shots=shots)))
                    ctx.close()
        br.close()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
