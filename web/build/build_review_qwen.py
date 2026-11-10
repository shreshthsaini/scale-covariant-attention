#!/usr/bin/env python3
"""Review page of the Qwen-Image tuning campaign (HANDOVER Section 39): web/review_qwen/index.html.

One self-contained HTML file (CSS, zoomgrid.js, zoomgrid.css and all data inline; no fonts, no external requests) plus
WebP assets in web/review_qwen/img/ (overview per panel and native-pixel tiles at four tour stops; budget 60 MB).

Sections: the TUNE screens (results/qwen_tune/{4096,5312}: screen.csv and the paired bootstrap CSVs), the TEST
comparison of the chosen configuration (results/qwen_5312_tuned: means, differences to AReA with 95% intervals),
timing (timing_summary.csv), and one synced-zoom grid per TEST prompt with full attention, AReA, untuned SCA (40
steps) and tuned SCA at 5312^2. Four tour stops: web/build/zoomlib.py regions over the grid's edge maps (every panel of a
grid uses the same stops). Missing parts render as "pending", so the page can be previewed while runs finish.
Usage: build_review_qwen.py --tuned LABEL [--tuned-dir DIR] [--processes 8]
"""

import argparse
import csv
import html
import json
from multiprocessing import Pool
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zoomlib  # noqa: E402

P = Path(__file__).resolve().parents[2]
S = Path("/scratch/09032/saini_2/sca")
WEB = P / "web"
OUTDIR = WEB / "review_qwen"
RES = "5312x5312"
SPAN = 768                 # tour window and region tile side (native px)
OVERVIEW = 2000            # overview long side (px)
BUDGET = 60e6
METRICS = [("pickscore", "PickScore"), ("hpsv21", "HPSv2.1"), ("hpsv2", "HPSv2"), ("ir", "ImageReward"),
           ("aesthetic", "Aesthetic"), ("clip", "CLIP score"), ("clipiqa", "CLIP-IQA"), ("arniqa", "ARNIQA"),
           ("musiq_crop1024", "MUSIQ, 1024 px crops"), ("musiq_4096", "MUSIQ, 4096 px view")]
DESC = {  # what each screened configuration changes (plain words for the table)
    "qt_sca40": "Untuned: the FLUX-derived settings, 40 steps, native time shift (mu 0.834)",
    "qt_mu": "Time shift grown with resolution (mu 1.96 at 4096, 2.22 at 5312), else untuned",
    "qt_sfx": "AReA's conditioning (its prompt suffix, empty negative prompt), else untuned",
    "qt_mu_sfx": "Time shift grown with resolution and AReA's conditioning",
    "qt_prog": "Progressive: 1328 (stock attention, 30 steps), 2656 (SCA, 20 steps), 5312 (SCA, 13 steps)",
    "qt_prog_sfx": "Progressive with AReA's conditioning",
    "qt_prog_late": "Progressive with HRDiT's step counts: 2656 (17 steps from sigma 0.74), 5312 (10 from 0.51)",
    "qt_prog_late_sfx": "Progressive with HRDiT's step counts and AReA's conditioning",
}


def rows(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def f(x, d=3, sign=False):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "pending"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def ci_cell(row, d=3):
    if not row:
        return "pending"
    lo, hi = float(row["ci_low"]), float(row["ci_high"])
    sig = lo > 0 or hi < 0
    txt = f"{float(row['delta']):+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"
    return f"<b>{txt}</b>" if sig else txt


def paired(results, method, ref):
    return {r["metric"]: r for r in rows(results / f"{method}_minus_{ref}.csv")}


def sidecars(folder):
    out = {}
    for meta in Path(folder).glob("*.json"):
        try:
            d = json.loads(meta.read_text())
        except ValueError:
            continue
        if d.get("complete") and d.get("config", {}).get("res") == RES:
            out[d["config"]["prompt_id"]] = (d, str(meta)[:-5] + ".png")
    return out


# ----------------------------------------------------------------------------- assets

def _set_outdir(d):
    global OUTDIR
    OUTDIR = d


def grid_assets(job):
    """One TEST prompt: shared tour stops, then per panel an overview WebP and region tiles."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    pid, panels = job
    ims, maps = {}, []
    for key, png in panels:
        im = Image.open(png).convert("RGB")
        ims[key] = im
        maps.append(zoomlib.edge_map(im))
    size = next(iter(ims.values())).size
    regs = zoomlib.regions(maps, size, k=4, span=SPAN)
    out = {"regions": regs, "size": list(size), "panels": {}}
    for key, png in panels:
        d = OUTDIR / "img" / pid / key
        vs = zoomlib.save_view(ims[key], str(d / "v.webp"), long_side=OVERVIEW, q=80, src=png)
        rt = zoomlib.cut_regions(ims[key], regs, str(d), src=png, tile=SPAN, q=80)
        out["panels"][key] = {"view": f"img/{pid}/{key}/v.webp", "vs": vs,
                              "rt": [[x, y, w, h, f"img/{pid}/{key}/{n}"] for x, y, w, h, n in rt]}
    return pid, out


# ----------------------------------------------------------------------------- tables

def screen_table(tuned):
    out = []
    for res, fam in (("4096", "Single pass, 4096²"), ("5312", "Progressive, 5312²")):
        R = P / "results" / "qwen_tune" / res
        sc = rows(R / "screen.csv")
        for r in sc:
            pr = paired(R, r["method"], "qt_area")
            out.append({"family": fam, "label": r["method"], "desc": DESC.get(r["method"], r["method"]),
                        "n": r["n_prompts"], "pick": ci_cell(pr.get("pickscore")), "hps": ci_cell(pr.get("hpsv21")),
                        "ir": ci_cell(pr.get("ir")), "dz": f(r["mean_dz"], 2, True),
                        "musiq": f(r["musiq_crop1024_delta"], 1, True), "clipiqa": f(r["clipiqa_delta"], 3, True),
                        "sec": f(r["median_seconds"], 0), "area_sec": f(r["reference_median_seconds"], 0),
                        "chosen": r["method"] == tuned})
    return out


def test_table(tuned):
    R = P / "results" / "qwen_5312_tuned"
    summ = {r["method"]: r for r in rows(R / "summary.csv")}
    pu, pt = paired(R, "qwen_sca40", "qwen_area"), paired(R, tuned, "qwen_area")
    ptu = paired(R, tuned, "qwen_sca40")
    out = []
    for k, name in METRICS:
        d = 1 if k.startswith("musiq") or k == "clip" else 3
        out.append({"metric": name, "area": f(summ.get("qwen_area", {}).get(k), d),
                    "untuned": f(summ.get("qwen_sca40", {}).get(k), d), "tuned": f(summ.get(tuned, {}).get(k), d),
                    "d_untuned": ci_cell(pu.get(k), d), "d_tuned": ci_cell(pt.get(k), d),
                    "d_tu": ci_cell(ptu.get(k), d)})
    return out, {m: r.get("n") for m, r in summ.items()}


def timing_table(tuned):
    R = P / "results" / "qwen_5312_tuned"
    names = {"qwen_full": "Full attention", "qwen_area": "AReA", "qwen_sca40": "SCA, untuned (40 steps)",
             "qwen_sca": "SCA, untuned (50 steps)", tuned: "SCA, tuned"}
    out = []
    for r in rows(R / "timing_summary.csv"):
        if r["method"] in names:
            out.append({"name": names[r["method"]], "steps": r["steps"], "n": r["n"], "n_clean": r["n_clean"],
                        "sec": f(r["median_seconds_clean"], 0), "sec_all": f(r["median_seconds_all"], 0),
                        "mem": f(r["max_peak_memory_gb"], 1), "ours": r["method"] == tuned})
    order = ["Full attention", "AReA", "SCA, untuned (50 steps)", "SCA, untuned (40 steps)", "SCA, tuned"]
    return sorted(out, key=lambda r: order.index(r["name"]))


def auto_summary(tuned, tuned_name):
    """Lead paragraph from the result files (so the page's numbers come from the archived CSVs)."""
    n_cfg = sum(len(rows(P / "results" / "qwen_tune" / r / "screen.csv")) for r in ("4096", "5312"))
    R = P / "results" / "qwen_5312_tuned"
    pt, pu = paired(R, tuned, "qwen_area"), paired(R, "qwen_sca40", "qwen_area")
    tm = {r["method"]: r for r in rows(R / "timing_summary.csv")}
    if not pt:
        return "Results pending."

    def d(pr, k, dd=3):
        r = pr.get(k)
        return "pending" if not r else (f"{float(r['delta']):+.{dd}f} [{float(r['ci_low']):+.{dd}f}, "
                                        f"{float(r['ci_high']):+.{dd}f}]")

    def sec(m):
        r = tm.get(m)
        return "pending" if not r or r.get("median_seconds_clean") in ("", None) else f"{float(r['median_seconds_clean']):.0f} s"
    return (f"SCA's Qwen-Image settings had been carried over from FLUX. {n_cfg} configurations were screened against "
            f"AReA (its own defaults) on 12 TUNE prompts, and the selection rule fixed before any run chose "
            f"<b>{html.escape(tuned_name)}</b>. It then ran once on the 15 held-out TEST prompts at 5312². "
            f"Against AReA on TEST: PickScore {d(pt, 'pickscore', 2)}, HPSv2.1 {d(pt, 'hpsv21')}, ImageReward "
            f"{d(pt, 'ir', 2)}, MUSIQ on 1024 px crops {d(pt, 'musiq_crop1024', 1)} (untuned SCA: PickScore "
            f"{d(pu, 'pickscore', 2)}, HPSv2.1 {d(pu, 'hpsv21')}). Median time per image: {sec(tuned)} against "
            f"AReA's {sec('qwen_area')} and untuned SCA's {sec('qwen_sca40')} (40 steps).")


# ----------------------------------------------------------------------------- page

CSS = """
:root{--bg:#FCFCFB;--fg:#1C2024;--muted:#59636D;--faint:#8A939C;--card:#FFFFFF;--line:#E2E5E3;--line2:#D3D8D5;
--surface:#F4F5F4;--accent:#13865F;--accent-ink:#0E6B4C;--on-accent:#fff;--good:#0E6B4C;--sans:-apple-system,
BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;--serif:"Iowan Old Style",Palatino,Georgia,serif;
color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0D100F;--fg:#E7EAE8;--muted:#A3ACB3;
--faint:#77818A;--card:#151918;--line:#262C2A;--line2:#323A37;--surface:#151918;--accent:#3DBB8C;--accent-ink:#62D2A7;
--on-accent:#06140E;--good:#62D2A7;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0D100F;--fg:#E7EAE8;--muted:#A3ACB3;--faint:#77818A;--card:#151918;--line:#262C2A;
--line2:#323A37;--surface:#151918;--accent:#3DBB8C;--accent-ink:#62D2A7;--on-accent:#06140E;--good:#62D2A7;
color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:400 15px/1.55 var(--sans);
-webkit-font-smoothing:antialiased}
.top{position:sticky;top:0;z-index:6;background:var(--bg);border-bottom:1px solid var(--line)}
.top .in{max-width:1280px;margin:auto;padding:0 16px;height:52px;display:flex;align-items:center;gap:16px}
.brand{font:600 18px/1 var(--serif)}.nav{display:flex;gap:14px;margin-left:auto;font-size:14px}
.nav a{color:var(--muted);text-decoration:none}.nav a:hover{color:var(--fg)}
.theme{width:34px;height:34px;border-radius:8px;border:1px solid var(--line);background:transparent;color:var(--muted);
display:grid;place-items:center;cursor:pointer}.theme svg{width:17px;height:17px}
header{padding:34px 16px 8px;max-width:1280px;margin:auto}
.kicker{font:600 12px/1.3 var(--sans);letter-spacing:.12em;text-transform:uppercase;color:var(--accent-ink);margin:0 0 8px}
h1{font:600 clamp(26px,3.4vw,38px)/1.15 var(--serif);margin:0 0 10px}
h2{font:600 22px/1.25 var(--serif);margin:34px 0 8px}
.lead{color:var(--muted);max-width:60rem;margin:0 0 6px}.lead b{color:var(--fg)}
main{max-width:1280px;margin:auto;padding:4px 16px 48px}
.note{color:var(--muted);font-size:13.5px;max-width:62rem;margin:6px 0 10px}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{font-weight:600;color:var(--muted);background:var(--surface);position:sticky;top:0}
td.l,th.l{text-align:left;white-space:normal;min-width:14rem}
tr:last-child td{border-bottom:0}
tr.chosen td{background:color-mix(in srgb,var(--accent) 12%,transparent)}
tr.fam td{background:var(--surface);color:var(--muted);font-weight:600;text-align:left}
td b{color:var(--good);font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:16px 0}
.prompt{font-size:14px;margin:0 0 10px}.prompt b{color:var(--faint);font-weight:500;margin-right:6px}
.secs{font-size:12.5px;color:var(--muted);margin:8px 0 0}
.zg-btn{background:var(--card)}
"""

JS_THEME = ("document.getElementById('theme').onclick=function(){var r=document.documentElement,t=r.getAttribute("
            "'data-theme'),dark=t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches,n=dark?'light':'dark';"
            "r.setAttribute('data-theme',n);try{localStorage.setItem('sca-theme',n)}catch(e){}};")

JS_GRIDS = """(function(){
  var list=document.getElementById('grids'),items=window.ITEMS||[];
  items.forEach(function(it,n){
    var c=document.createElement('section');c.className='card';c.id='t'+(n+1);
    var p=document.createElement('p');p.className='prompt';var b=document.createElement('b');b.textContent='#'+(n+1);
    p.appendChild(b);p.appendChild(document.createTextNode(it.prompt));c.appendChild(p);
    c.appendChild(window.SCAZoom.grid({size:it.size,panels:it.panels,regions:it.regions,alt:it.prompt,minW:200,
      maxCols:4,label:'Four methods on TEST prompt '+(n+1)+'; zoom moves every panel'}).el);
    var s=document.createElement('p');s.className='secs';s.textContent=it.secs;c.appendChild(s);
    list.appendChild(c);
  });
})();"""


def table_html(head, body, cls=None):
    th = "".join(f'<th class="{c}">{h}</th>' if c else f"<th>{h}</th>" for h, c in head)
    return f'<div class="tw"><table{f" class={cls}" if cls else ""}><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def build_html(tuned, tuned_name, items, screen, test, ns, timing, summary_text):
    def tr(cells, cls=""):
        return f'<tr class="{cls}">' + "".join(cells) + "</tr>"
    sb, fam = [], None
    for r in screen:
        if r["family"] != fam:
            fam = r["family"]
            sb.append(f'<tr class="fam"><td colspan="11">{fam}, against AReA at the same size</td></tr>')
        sb.append(tr([f'<td class="l">{html.escape(r["desc"])}</td>', f"<td>{r['n']}</td>", f"<td>{r['pick']}</td>",
                      f"<td>{r['hps']}</td>", f"<td>{r['ir']}</td>", f"<td>{r['dz']}</td>", f"<td>{r['musiq']}</td>",
                      f"<td>{r['clipiqa']}</td>", f"<td>{r['sec']}</td>", f"<td>{r['area_sec']}</td>",
                      f"<td>{'chosen' if r['chosen'] else ''}</td>"], "chosen" if r["chosen"] else ""))
    screen_html = table_html([("Configuration", "l"), ("Prompts", None), ("PickScore", None), ("HPSv2.1", None),
                              ("ImageReward", None), ("Mean d<sub>z</sub>", None), ("MUSIQ crops", None),
                              ("CLIP-IQA", None), ("Seconds", None), ("AReA s", None), ("", None)],
                             "".join(sb) or tr(['<td class="l" colspan="11">pending</td>']))
    tb = "".join(tr([f'<td class="l">{r["metric"]}</td>', f"<td>{r['area']}</td>", f"<td>{r['untuned']}</td>",
                     f"<td>{r['tuned']}</td>", f"<td>{r['d_untuned']}</td>", f"<td>{r['d_tuned']}</td>",
                     f"<td>{r['d_tu']}</td>"]) for r in test)
    test_html = table_html([("Metric (mean over prompts)", "l"), ("AReA", None), ("SCA, untuned", None),
                            ("SCA, tuned", None), ("Untuned \u2212 AReA", None), ("Tuned \u2212 AReA", None),
                            ("Tuned \u2212 untuned", None)], tb)
    tm = "".join(tr([f'<td class="l">{r["name"]}</td>', f"<td>{r['steps']}</td>", f"<td>{r['n_clean']} of {r['n']}</td>",
                     f"<td>{r['sec']}</td>", f"<td>{r['sec_all']}</td>", f"<td>{r['mem']}</td>"],
                    "chosen" if r["ours"] else "") for r in timing)
    time_html = table_html([("Method", "l"), ("Steps", None), ("Images without a GPU co-tenant", None),
                            ("Median s (those images)", None), ("Median s (all)", None), ("Peak GB", None)],
                           tm or tr(['<td class="l" colspan="6">pending</td>']))
    zg_css = (WEB / "assets/css/zoomgrid.css").read_text()
    zg_js = (WEB / "assets/js/zoomgrid.js").read_text()
    items_js = "window.ITEMS = " + json.dumps(items, separators=(",", ":")) + ";"
    icon = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
            'stroke-linejoin="round" aria-hidden="true"><path d="M12 3a9 9 0 1 0 9 9 7 7 0 0 1-9-9z"/></svg>')
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SCA on Qwen-Image: tuning review</title>
<script>try{{var t=localStorage.getItem("sca-theme");if(t)document.documentElement.setAttribute("data-theme",t);}}catch(e){{}}</script>
<style>{CSS}
{zg_css}</style>
</head>
<body>
<div class="top"><div class="in"><span class="brand">SCA on Qwen-Image</span><nav class="nav"><a href="#screen">TUNE screens</a>
<a href="#test">TEST</a><a href="#timing">Timing</a><a href="#grids-h">Side by side</a></nav>
<button class="theme" id="theme" type="button" aria-label="Switch between light and dark theme">{icon}</button></div></div>
<header>
<p class="kicker">Review page, not published</p>
<h1>Tuning SCA on Qwen-Image against AReA</h1>
<p class="lead">{summary_text}</p>
</header>
<main>
<h2 id="screen">TUNE screens (12 prompts, used for every choice)</h2>
<p class="note">Each row is one configuration, scored by paired differences against AReA at its own defaults on the same
12 TUNE prompts and size (seed 0). Cells give the mean difference and its 95% bootstrap interval (2000 draws over
prompts); bold intervals exclude zero. Mean d<sub>z</sub> is the selection score fixed before any run: the mean, over
PickScore, HPSv2.1 and ImageReward, of the paired mean divided by the SD of the paired differences. MUSIQ crops scores
four 1024 px native crops. Seconds are medians per image on one GH200.</p>
{screen_html}
<h2 id="test">TEST (15 held-out prompts, 5312²)</h2>
<p class="note">The 15 prompts of the earlier comparison, untouched during tuning; {html.escape(tuned_name)} ran once on
them. Untuned SCA is the 40-step run of that comparison (step-matched to AReA). Prompts per method: {html.escape(json.dumps(ns))}.
The three right columns are paired differences over prompts: mean [95% bootstrap interval, 2000 draws]; bold intervals
exclude zero.</p>
{test_html}
<h2 id="timing">Timing at 5312²</h2>
<p class="note">Seconds per image cover denoising, any stage transitions and the VAE decode, with prompt embeddings
computed beforehand, on one GH200. An image counts as without a co-tenant when no other process held its GPU while it
was generated.</p>
{time_html}
<h2 id="grids-h">Side by side, 15 TEST prompts at 5312²</h2>
<p class="note">Full attention, AReA, untuned SCA (40 steps) and tuned SCA from the same prompt and seed. Scroll or
pinch on any panel to zoom and drag to pan; every panel follows, down to native pixels at four regions picked from
the edge energy of all four images. Double-click resets; Auto zoom visits the four regions.</p>
<div id="grids"><noscript><p>The zoom grids need JavaScript.</p></noscript></div>
</main>
<script>{items_js}</script>
<script>{zg_js}</script>
<script>{JS_GRIDS}
{JS_THEME}</script>
</body>
</html>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tuned", required=True, help="Label of the chosen configuration")
    ap.add_argument("--tuned-name", default="Tuned SCA")
    ap.add_argument("--tuned-dir", type=Path, help="Its TEST images (default out/qwen_tuned/<label>/5312x5312)")
    ap.add_argument("--summary", default="", help="Lead paragraph (HTML)")
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--outdir", type=Path, help="Write the page here instead of web/review_qwen (previews)")
    ap.add_argument("--limit", type=int, help="First N TEST prompts only (previews)")
    a = ap.parse_args(argv)
    if a.outdir:
        global OUTDIR
        OUTDIR = a.outdir
    tuned_dir = a.tuned_dir or S / "out" / "qwen_tuned" / a.tuned / RES
    prompts = [json.loads(x) for x in open(S / "prompts" / "qwen_area15.jsonl") if x.strip()][:a.limit]
    sh = S / "out" / "showcase"
    cols = [("full", "Full attention", sh / "qwen_full" / RES, False), ("area", "AReA", sh / "qwen_area" / RES, False),
            ("sca40", "SCA, untuned", sh / "qwen_sca40" / RES, True), ("tuned", "SCA, tuned", tuned_dir, True)]
    cars = {k: sidecars(d) for k, _, d, _ in cols}
    jobs = []
    for r in prompts:
        panels = [(k, cars[k][r["id"]][1]) for k, _, _, _ in cols if r["id"] in cars[k]]
        if len(panels) >= 2:
            jobs.append((r["id"], panels))
    with Pool(a.processes, initializer=_set_outdir, initargs=(OUTDIR,)) as pool:
        assets = dict(pool.map(grid_assets, jobs))
    items = []
    for r in prompts:
        if r["id"] not in assets:
            continue
        g = assets[r["id"]]
        panels, secs = [], []
        for k, label, _, ours in cols:
            if k in g["panels"]:
                panels.append({"m": k, "label": label, "ours": ours, **g["panels"][k]})
                d = cars[k][r["id"]][0]
                secs.append(f"{label}: {d['seconds']:.0f} s")
            else:
                secs.append(f"{label}: not generated")
        items.append({"id": r["id"], "prompt": r["prompt"], "size": g["size"], "regions": g["regions"],
                      "panels": panels, "secs": " · ".join(secs)})
    test, ns = test_table(a.tuned)
    page = build_html(a.tuned, a.tuned_name, items, screen_table(a.tuned), test, ns, timing_table(a.tuned),
                      a.summary or auto_summary(a.tuned, a.tuned_name))
    (OUTDIR / "index.html").write_text(page)
    total = zoomlib.dir_bytes(str(OUTDIR))
    print(f"wrote {OUTDIR / 'index.html'}: {len(items)} grids, {total / 1e6:.1f} MB in web/review_qwen", flush=True)
    if total > BUDGET:
        raise SystemExit(f"over budget: {total / 1e6:.1f} MB > {BUDGET / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
