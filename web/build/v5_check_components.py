"""Headless check of the interactive components' demo pages (web/dev/<name>.html): console errors, failed local
files, horizontal overflow, the mount's height against 80vh, and screenshots after scripted interactions, at
1440x900 and 390x844 (touch), light and dark.

Usage (repository root; an env with playwright and its Chromium build):
  PLAYWRIGHT_BROWSERS_PATH=/scratch/09032/saini_2/tmp/ms-playwright \
  /work/09032/saini_2/vista/library/uv-envs/pwqa/bin/python web/build/v5_check_components.py percentiles \
      [heads emulate table speed views stages] --out /scratch/09032/saini_2/sca/web_builds/v5_webinteractive_1/shots
      [--page index.html] [--shift]
Pages open from file://. Prints one line per page and view and exits non-zero on any console error or overflow.
--page also checks every component mounted on that page of web/ (live, height, screenshot of its figure);
--shift serves web/ on 127.0.0.1 with every data.js delayed 2 s and reports the skeleton's height against the
final height (a move above 12 px fails).
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


def run_heads(page, view, out, tag, log):
    page.wait_for_selector(".c-heads-chip", timeout=20000)
    page.wait_for_timeout(900)
    log.append(("read", page.evaluate("() => document.querySelector('.c-heads-read').textContent.slice(0, 90)")))
    log.append(("split", page.evaluate("() => Array.from(document.querySelectorAll('.c-heads-num-v')).map(e => e.textContent).join(' / ')")))
    shots = [shot(page, out, f"{tag}_0_default")]
    box = page.query_selector(".c-heads-ov").bounding_box()
    x, y = box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.3
    if view == "phone":
        tap(page, x, y)
    else:
        page.mouse.move(x, y)
        page.wait_for_timeout(300)
        page.mouse.click(x, y)
    page.wait_for_timeout(500)
    log.append(("after pick", page.evaluate("() => document.querySelector('.c-heads-read').textContent.slice(0, 90)")))
    shots.append(shot(page, out, f"{tag}_1_pick"))
    chip = page.query_selector_all(".c-heads-chip")[6]
    chip.click()
    page.query_selector_all(".c-heads-scrub-b")[0].click()
    page.wait_for_timeout(400)
    log.append(("map", page.evaluate("() => document.querySelector('.c-heads-map-t').textContent + ' | ' + document.querySelector('.c-heads-map-sv').textContent")))
    shots.append(shot(page, out, f"{tag}_2_head7"))
    page.query_selector_all(".c-heads-seg-b")[1].click()
    page.wait_for_timeout(400)
    ov = page.query_selector(".c-heads-ov")
    ov.focus()
    for _ in range(3):
        page.keyboard.press("ArrowDown")
    page.wait_for_timeout(300)
    log.append(("matrix + 3 down", page.evaluate("() => document.querySelector('.c-heads-read').textContent.slice(0, 90)")))
    shots.append(shot(page, out, f"{tag}_3_matrix"))
    return shots


def run_emulate(page, view, out, tag, log):
    page.wait_for_selector(".c-emulate-seg-b", timeout=20000)
    page.wait_for_timeout(900)
    rd = "() => document.querySelector('.c-emulate-big').textContent + ' | ' + document.querySelector('.c-emulate-star').textContent + ' | ' + document.querySelector('.c-emulate-small').textContent"
    log.append(("default", page.evaluate(rd)))
    shots = [shot(page, out, f"{tag}_0_default")]
    page.query_selector(".c-emulate-snap").click()
    page.wait_for_timeout(900)
    log.append(("snap", page.evaluate(rd)))
    shots.append(shot(page, out, f"{tag}_1_snap"))
    page.query_selector_all(".c-emulate-seg-b")[0].click()
    page.wait_for_timeout(400)
    log.append(("near head", page.evaluate(rd)))
    shots.append(shot(page, out, f"{tag}_2_near"))
    page.query_selector_all(".c-emulate-seg-b")[2].click()
    box = page.query_selector(".c-emulate-chartbox").bounding_box()
    x, y = box["x"] + box["width"] * 0.85, box["y"] + box["height"] * 0.5
    if view == "phone":
        tap(page, x, y)
    else:
        page.mouse.move(box["x"] + box["width"] * 0.5, y)
        page.mouse.down()
        page.mouse.move(x, y, steps=6)
        page.mouse.up()
    page.wait_for_timeout(400)
    log.append(("far head, pi ~0.85", page.evaluate(rd)))
    shots.append(shot(page, out, f"{tag}_3_far"))
    rng = page.query_selector(".c-emulate-range")
    rng.focus()
    page.keyboard.press("Home")
    page.wait_for_timeout(200)
    log.append(("Home", page.evaluate("() => document.querySelector('.c-emulate-range').value")))
    return shots


def run_table(page, view, out, tag, log):
    page.wait_for_selector(".c-table-tab", timeout=20000)
    page.wait_for_timeout(500)
    rows = "() => Array.from(document.querySelectorAll('.c-table-r')).map(r => r.querySelector('.c-table-nm').textContent + ':' + (r.querySelector('.c-table-v-ir') || {}).textContent).join(' | ')"
    log.append(("paper order", page.evaluate(rows)))
    shots = [shot(page, out, f"{tag}_0_main")]
    page.query_selector(".c-table-c-time .c-table-sort").click()
    page.wait_for_timeout(300)
    log.append(("sorted by time", page.evaluate("() => Array.from(document.querySelectorAll('.c-table-r .c-table-nm')).map(e => e.textContent).join(' | ')")))
    shots.append(shot(page, out, f"{tag}_1_sorted"))
    page.query_selector_all(".c-table-chip")[3].click()
    page.query_selector_all(".c-table-chip")[2].click()
    page.wait_for_timeout(300)
    log.append(("columns", page.evaluate("() => Array.from(document.querySelectorAll('.c-table-ch .c-table-sort')).map(e => e.textContent).join(' ')")))
    shots.append(shot(page, out, f"{tag}_2_toggled"))
    page.query_selector_all(".c-table-tab")[1].click()
    page.wait_for_timeout(300)
    log.append(("generality", page.evaluate(rows)))
    shots.append(shot(page, out, f"{tag}_3_gen"))
    return shots


def run_speed(page, view, out, tag, log):
    page.wait_for_selector(".c-speed-rc", timeout=20000)
    page.wait_for_timeout(500)
    log.append(("tip 4096", page.evaluate("() => document.querySelector('.c-speed-tip').textContent")))
    shots = [shot(page, out, f"{tag}_0_4096")]
    b = page.query_selector(".c-speed-plot").bounding_box()
    if view == "phone":
        tap(page, b["x"] + b["width"] * 0.62, b["y"] + b["height"] * 0.5)
    else:
        page.mouse.move(b["x"] + b["width"] * 0.7, b["y"] + b["height"] * 0.5)
    page.wait_for_timeout(300)
    log.append(("tip moved", page.evaluate("() => document.querySelector('.c-speed-tip-h').textContent")))
    shots.append(shot(page, out, f"{tag}_1_hover"))
    page.query_selector(".c-speed-plot").focus()
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(200)
    log.append(("keys", page.evaluate("() => document.querySelector('.c-speed-tip').textContent")))
    shots.append(shot(page, out, f"{tag}_2_keys"))
    return shots


def check_page(br, page_name, out, views, themes):
    """Every .comp-mount of a real page (e.g. index.html): scroll to it, wait until it is live, screenshot the figure."""
    bad = 0
    url = "file://" + os.path.join(WEB, page_name)
    for view in views:
        for theme in themes:
            ctx = br.new_context(color_scheme=theme, **VIEWS[view])
            page = ctx.new_page()
            errs = []
            page.on("console", lambda m, errs=errs: errs.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
            page.on("pageerror", lambda e, errs=errs: errs.append(f"pageerror: {e}"))
            page.on("requestfailed", lambda r, errs=errs: errs.append(f"requestfailed: {r.url}"))
            page.goto(url)
            page.wait_for_timeout(800)
            names = page.evaluate("() => Array.from(document.querySelectorAll('.comp-mount[data-comp]')).map(m => m.getAttribute('data-comp'))")
            res = {}
            for i, name in enumerate(names):
                page.evaluate("(i) => document.querySelectorAll('.comp-mount[data-comp]')[i].closest('figure').scrollIntoView({block: 'center'})", i)
                page.wait_for_timeout(2500)
                st = page.evaluate("""(i) => { const m = document.querySelectorAll('.comp-mount[data-comp]')[i], f = m.closest('figure'), r = m.getBoundingClientRect();
                    return {live: f.classList.contains('is-live'), h: Math.round(r.height), w: Math.round(r.width), kids: m.children.length}; }""", i)
                fig = page.query_selector_all(".comp-mount[data-comp]")[i].evaluate_handle("m => m.closest('figure')")
                path = os.path.join(out, f"page_{page_name.replace('.html', '')}_{name}_{view}_{theme}.png")
                try:
                    fig.as_element().screenshot(path=path)
                except Exception as e:  # noqa: BLE001
                    errs.append(f"shot {name}: {e!r}")
                res[name] = dict(st, shot=path)
            over = page.evaluate("() => document.documentElement.scrollWidth > document.documentElement.clientWidth")
            ok = not errs and not over and all(r["live"] for r in res.values())
            bad += not ok
            print(json.dumps(dict(page=page_name, view=view, theme=theme, ok=ok, errors=errs[:8], overflow=over, comps=res)))
            ctx.close()
    return bad


def run_views(page, view, out, tag, log):
    page.wait_for_selector(".c-views-seg-b", timeout=20000)
    page.wait_for_timeout(900)
    st = "() => Array.from(document.querySelectorAll('.c-views-st')).map(e => e.textContent).join(' | ')"
    log.append(("4096", page.evaluate(st)))
    shots = [shot(page, out, f"{tag}_0_4096")]
    b = page.query_selector(".c-views-main").bounding_box()
    x, y = b["x"] + b["width"] * 0.5, b["y"] + b["height"] * 0.3
    if view == "phone":
        tap(page, x, y)
    else:
        page.mouse.move(b["x"] + b["width"] * 0.2, b["y"] + b["height"] * 0.7)
        page.mouse.down()
        page.mouse.move(x, y, steps=8)
        page.mouse.up()
    page.wait_for_timeout(300)
    shots.append(shot(page, out, f"{tag}_1_moved"))
    page.query_selector_all(".c-views-seg-b")[1].click()
    page.wait_for_timeout(500)
    log.append(("2048", page.evaluate(st)))
    shots.append(shot(page, out, f"{tag}_2_2048"))
    page.query_selector_all(".c-views-seg-b")[0].click()
    page.wait_for_timeout(500)
    log.append(("1024", page.evaluate(st) + " || " + page.evaluate("() => document.querySelector('.c-views-st-m').textContent")))
    shots.append(shot(page, out, f"{tag}_3_1024"))
    return shots


def run_stages(page, view, out, tag, log):
    page.wait_for_selector(".c-stages-hm rect", timeout=20000)
    page.wait_for_timeout(500)
    rd = "() => document.querySelector('.c-stages-readout').textContent + ' | ' + document.querySelector('.c-stages-hmh').textContent.slice(0, 60)"
    log.append(("first 4096 step", page.evaluate(rd)))
    shots = [shot(page, out, f"{tag}_0_4096")]
    rng = page.query_selector(".c-stages-range")
    rng.focus()
    page.keyboard.press("Home")
    page.wait_for_timeout(300)
    log.append(("Home", page.evaluate(rd)))
    shots.append(shot(page, out, f"{tag}_1_native"))
    b = page.query_selector(".c-stages-tlbox").bounding_box()
    if view == "phone":
        tap(page, b["x"] + b["width"] * 0.66, b["y"] + b["height"] * 0.5)
    else:
        page.mouse.click(b["x"] + b["width"] * 0.66, b["y"] + b["height"] * 0.5)
    page.wait_for_timeout(300)
    log.append(("timeline 66%", page.evaluate(rd)))
    shots.append(shot(page, out, f"{tag}_2_2048"))
    page.query_selector(".c-stages-play").click()
    page.wait_for_timeout(1300)
    log.append(("playing", page.evaluate(rd)))
    page.query_selector(".c-stages-play").click()
    return shots


def check_shift(br, names, views):
    """Layout shift when the data arrives: serve web/ on 127.0.0.1, delay every assets/data/*/data.js by 2 s, and
    compare the mount's height before the data (the skeleton) with its height after."""
    import functools
    import http.server
    import threading
    import time

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=WEB))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    worst = 0
    try:
        for view in views:
            for name in names:
                ctx = br.new_context(**VIEWS[view])
                page = ctx.new_page()

                def slow(route):
                    time.sleep(2.0)
                    route.continue_()
                page.route("**/assets/data/*/data.js", slow)
                page.goto(f"http://127.0.0.1:{srv.server_address[1]}/dev/{name}.html")
                page.wait_for_timeout(1200)
                h0 = page.evaluate("() => document.querySelector('.comp-mount').getBoundingClientRect().height")
                page.wait_for_timeout(4500)
                h1 = page.evaluate("() => document.querySelector('.comp-mount').getBoundingClientRect().height")
                worst = max(worst, abs(h1 - h0))
                print(json.dumps(dict(shift=name, view=view, skeleton=round(h0), final=round(h1), moved=round(h1 - h0))))
                ctx.close()
    finally:
        srv.shutdown()
    return worst


def run_formation(page, view, out, tag, log):
    page.wait_for_selector(".c-formation-tab", timeout=20000)
    page.wait_for_timeout(1500)
    rd = "() => document.querySelector('.c-formation-pcap').textContent + ' | ' + document.querySelector('.c-formation-bars').textContent + ' | ' + document.querySelector('.c-formation-say').textContent"
    log.append(("final", page.evaluate(rd)))
    shots = [shot(page, out, f"{tag}_0_final")]
    rng = page.query_selector(".c-formation-range")
    rng.focus()
    page.keyboard.press("Home")
    page.wait_for_timeout(700)
    log.append(("first step", page.evaluate(rd)))
    shots.append(shot(page, out, f"{tag}_1_start"))
    for _ in range(9):
        page.keyboard.press("ArrowRight")
    page.wait_for_timeout(700)
    log.append(("stop 9", page.evaluate(rd)))
    page.query_selector_all(".c-formation-vt")[1].click()
    page.wait_for_timeout(500)
    shots.append(shot(page, out, f"{tag}_2_far"))
    b = page.query_selector(".c-formation-main").bounding_box()
    if view == "phone":
        tap(page, b["x"] + b["width"] * 0.3, b["y"] + b["height"] * 0.35)
    else:
        page.mouse.click(b["x"] + b["width"] * 0.3, b["y"] + b["height"] * 0.35)
    page.wait_for_timeout(300)
    page.query_selector_all(".c-formation-tab")[1].click()
    page.wait_for_timeout(300)
    rng.focus()
    page.keyboard.press("End")
    page.wait_for_timeout(1500)
    log.append(("tab 2, final, clicked patch", page.evaluate(rd)))
    shots.append(shot(page, out, f"{tag}_3_tab2"))
    return shots


def run_views_compact(page, view, out, tag, log):
    page.wait_for_selector(".c-views-seg-b", timeout=20000)
    page.wait_for_timeout(900)
    rd = "() => document.querySelector('.c-views-mix').textContent"
    log.append(("4096", page.evaluate(rd)))
    shots = [shot(page, out, f"{tag}_0_4096")]
    b = page.query_selector(".c-views-main").bounding_box()
    if view == "phone":
        tap(page, b["x"] + b["width"] * 0.5, b["y"] + b["height"] * 0.3)
    else:
        page.mouse.click(b["x"] + b["width"] * 0.5, b["y"] + b["height"] * 0.3)
    page.query_selector_all(".c-views-seg-b")[1].click()
    page.wait_for_timeout(400)
    log.append(("2048", page.evaluate(rd)))
    shots.append(shot(page, out, f"{tag}_1_2048"))
    page.query_selector_all(".c-views-seg-b")[0].click()
    page.wait_for_timeout(400)
    log.append(("1024", page.evaluate(rd)))
    shots.append(shot(page, out, f"{tag}_2_1024"))
    return shots


def run_table_compact(page, view, out, tag, log):
    page.wait_for_selector(".c-table-r", timeout=20000)
    page.wait_for_timeout(500)
    log.append(("header", page.evaluate("() => Array.from(document.querySelectorAll('.c-table-ch th')).map(e => e.textContent).join(' | ')")))
    log.append(("rows", page.evaluate("() => Array.from(document.querySelectorAll('.c-table-r')).map(r => Array.from(r.children).map(c => c.textContent).join(' ')).join(' || ')")))
    log.append(("tabs/chips visible", page.evaluate("() => [getComputedStyle(document.querySelector('.c-table-top')).display, document.querySelectorAll('.c-table-chip').length]")))
    return [shot(page, out, f"{tag}_0")]


RUNNERS = {"percentiles": run_percentiles, "heads": run_heads, "emulate": run_emulate, "table": run_table, "speed": run_speed, "views": run_views, "stages": run_stages, "formation": run_formation, "views_compact": run_views_compact, "table_compact": run_table_compact, "percentiles_compact": run_percentiles}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="desk,phone")
    ap.add_argument("--themes", default="light,dark")
    ap.add_argument("--page", default="", help="also check every component mounted on this page of web/ (e.g. index.html)")
    ap.add_argument("--shift", action="store_true", help="also measure the layout shift when each component's data arrives")
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
        if a.page:
            bad += check_page(br, a.page, a.out, a.views.split(","), a.themes.split(","))
        if a.shift:
            bad += check_shift(br, a.names, a.views.split(",")) > 12
        br.close()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
