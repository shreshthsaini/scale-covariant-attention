"""Data and images of the percentile explorer (SCAComp.percentiles, web/assets/js/c_percentiles.js).

Selection: exactly the rule of the paper's decile figures (paper/figures/src/v6_more.py, fig_deciles; Figures
deciles_a and deciles_b): the paired ImageReward difference, progressive SCA minus AReA, over the 195 evaluation
prompts at seed 0, sorted by (difference, short id); the 10th, 20th, ..., 90th percentile by nearest rank,
k = ceil(p * 195 / 100). The functions are imported from v6_more.py, not copied.
Checks: the mean difference equals the archived paired file (results/prog195_vs_baselines/sca_v2_prog_vs_area.csv);
every stop equals the paper's provenance (paper/figures/v6/deciles_{a,b}_provenance.json): percentile, nearest rank,
prompt key, difference, and each cell's image path and ImageReward; the four images of a stop share prompt and
initial noise (v6_common.same_prompt_and_noise, which also checks each PNG's sha256 against its sidecar).
Seconds per 4096^2 image: fig_speed.load() as the paper prints them (half-up to 0.1 s): clean batch-1 medians for
AReA, HRDiT and progressive SCA, the campaign mean for UltraImage (it has no clean time; marked, as in the paper).

Images, per stop and method: a 1024 px overview (WebP q76) and 1024 px native-pixel tiles (WebP q60, 4 x 4 at
4096^2) that SCAZoom.grid loads only when a view needs them; three tour stops per stop from zoomlib.regions over the
four images' edge maps. fallback.webp: the median stop's four images side by side (no text), the static picture the
page shows if the script or its data does not load.

Output (web/assets/data/percentiles/): data.js (window.SCA_DATA.percentiles), img/p<pct>/<method>/{v.webp,
<c>_<r>.webp}, fallback.webp. Files are rewritten only when missing or older than their PNG.

Run (repository root, CPU node, about two minutes with 36 processes):
  nice -n 10 /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_build_percentiles.py [--procs 36]
"""
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
REPO = os.path.dirname(WEB)
SRC = os.path.join(REPO, "paper", "figures", "src")
sys.path.insert(0, HERE)
sys.path.insert(0, SRC)

from PIL import Image  # noqa: E402

import fig_speed  # noqa: E402
import v6_common as vc  # noqa: E402
import v6_more  # noqa: E402
import zoomlib  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = os.path.join(WEB, "assets", "data", "percentiles")
PROV = [os.path.join(REPO, "paper", "figures", "v6", f"deciles_{s}_provenance.json") for s in ("a", "b")]
# (paper method id, short id used in paths and JS, label, ours)
METHODS = [("area", "area", "AReA", False), ("ultraimage", "ultra", "UltraImage", False),
           ("hrdit_dev_author", "hrdit", "HRDiT", False), ("sca_v2_prog", "prog", "Progressive SCA", True)]
SRC_RUN = {"area": "a4k195_main", "ultraimage": "a4k195_main", "hrdit_dev_author": "prog195_sca_v2",
           "sca_v2_prog": "prog195_sca_v2"}            # as v6_more.fig_deciles
VIEW_PX, VIEW_Q, TILE_Q = 1024, 76, 60


def rel(p):
    return os.path.relpath(p, OUT).replace(os.sep, "/")


def selection():
    data = {}
    for m, *_ in METHODS:
        data[m] = v6_more.per_image(SRC_RUN[m], "0", [m])[m]
    diff = {s: data["sca_v2_prog"][s][0] - data["area"][s][0] for s in data["area"]}
    n = len(diff)
    mean = sum(diff.values()) / n
    archived = v6_more.archived_mean("prog195_vs_baselines")
    assert n == 195 and abs(mean - archived) < 5e-6, (n, mean, archived)
    picks = v6_more.nearest_rank(diff, range(10, 100, 10))
    # the paper's provenance must agree stop by stop
    prov = []
    for p in PROV:
        prov += json.load(open(p))["rows"]
    assert len(prov) == len(picks) == 9
    for (pct, sid, d, k), row in zip(picks, prov):
        assert (row["percentile"], row["nearest_rank"], row["key"]) == (pct, k, sid), (row["percentile"], pct, sid)
        assert abs(row["difference"] - d) < 1e-9, (pct, row["difference"], d)
        for m, *_ in METHODS:
            assert row["cells"][m]["path"] == data[m][sid][1], (pct, m)
            assert abs(row["cells"][m]["value"] - data[m][sid][0]) < 1e-9, (pct, m)
    srt = sorted(diff, key=lambda s: (diff[s], s))
    return data, diff, picks, srt, mean


def seconds():
    d = fig_speed.load()
    out = {}
    for m, short, *_ in METHODS:
        if m in d["clean"]:
            out[short] = dict(value=fig_speed.r1(d["clean"][m]), kind="clean")
        else:
            out[short] = dict(value=fig_speed.r1(d["camp"][m]), kind="campaign")
    return out


def cell_job(job):
    """Overview and tiles of one image; returns the panel spec for SCAZoom.grid."""
    png, odir = job["png"], job["odir"]
    os.makedirs(odir, exist_ok=True)
    im = None

    def need():
        nonlocal im
        if im is None:
            im = Image.open(png).convert("RGB")
        return im
    view = os.path.join(odir, "v.webp")
    vs = zoomlib.save_view(need() if not zoomlib.fresh(view, png) else None, view, long_side=VIEW_PX, q=VIEW_Q, src=png)
    with Image.open(png) as probe:
        W, H = probe.size
    cols, rows = -(-W // zoomlib.TILE), -(-H // zoomlib.TILE)
    last = os.path.join(odir, f"{cols - 1}_{rows - 1}.webp")
    if not (zoomlib.fresh(last, png) and zoomlib.fresh(os.path.join(odir, "0_0.webp"), png)):
        zoomlib.cut_grid(need(), odir, src=png, q=TILE_Q)
    return dict(size=[W, H], view=rel(view), vs=vs,
                tiles=dict(path=rel(os.path.join(odir, "{c}_{r}.webp")), size=zoomlib.TILE, cols=cols, rows=rows))


def regions_job(job):
    pngs = job["pngs"]
    cache = os.path.join(job["cache"], f"p{job['pct']}.json")
    if os.path.exists(cache) and all(zoomlib.fresh(cache, p) for p in pngs):
        c = json.load(open(cache))
        if c.get("pngs") == pngs:
            return job["pct"], c["regions"]
    maps, size = [], None
    for p in pngs:
        with Image.open(p) as im:
            im = im.convert("RGB")
            size = im.size
            maps.append(zoomlib.edge_map(im))
    regs = zoomlib.regions(maps, size, k=3)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump({"pngs": pngs, "regions": regs}, open(cache, "w"))
    return job["pct"], regs


def fallback(paths, out, h=480, gap=8):
    ims = [Image.open(p).convert("RGB").resize((h, h), Image.LANCZOS) for p in paths]
    canvas = Image.new("RGB", (len(ims) * h + (len(ims) - 1) * gap, h), (128, 128, 128))
    for i, im in enumerate(ims):
        canvas.paste(im, (i * (h + gap), 0))
    canvas.save(out, "WEBP", quality=78, method=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=36)
    ap.add_argument("--cache", default="/scratch/09032/saini_2/sca/web_assets/zoom_cache/percentiles")
    a = ap.parse_args()
    data, diff, picks, srt, mean = selection()
    secs = seconds()
    prompts = {}
    for pct, sid, d, k in picks:
        cells = {m: data[m][sid][1] for m, *_ in METHODS}
        same = vc.same_prompt_and_noise(cells)          # prompt, seed and noise shared; sha256 checked
        prompts[sid] = same["prompt"]
    jobs, rjobs = [], []
    for pct, sid, d, k in picks:
        for m, short, *_ in METHODS:
            jobs.append(dict(pct=pct, m=short, png=data[m][sid][1], odir=os.path.join(OUT, "img", f"p{pct}", short)))
        rjobs.append(dict(pct=pct, pngs=[data[m][sid][1] for m, *_ in METHODS], cache=a.cache))
    with Pool(a.procs) as pool:
        specs = pool.map(cell_job, jobs, chunksize=1)
        regs = dict(pool.map(regions_job, rjobs, chunksize=1))
    spec = {(j["pct"], j["m"]): s for j, s in zip(jobs, specs)}
    stops = []
    for pct, sid, d, k in picks:
        cells, best = {}, None
        for m, short, *_ in METHODS:
            s = spec[(pct, short)]
            ir = data[m][sid][0]
            cells[short] = dict(ir=round(ir, 4), view=s["view"], vs=s["vs"], tiles=s["tiles"])
            if best is None or ir > data[best[0]][sid][0]:
                best = (m, short)
        size = spec[(pct, METHODS[0][1])]["size"]
        stops.append(dict(p=pct, rank=k, key=sid, diff=round(d, 4), prompt=prompts[sid], size=size,
                          regions=regs[pct], best=best[1], cells=cells))
    rec = dict(
        rule=("Nearest-rank deciles of the paired ImageReward difference, progressive SCA minus AReA, over the "
              "195 evaluation prompts at seed 0 (the paper's decile figures)."),
        n=len(diff), seed=0, mean=round(mean, 4),
        sorted=[round(diff[s], 4) for s in srt],
        methods=[dict(id=short, label=lab, ours=ours, secs=secs[short]["value"], secs_kind=secs[short]["kind"])
                 for m, short, lab, ours in METHODS],
        stops=stops,
        source=dict(script="web/build/v5_build_percentiles.py", paper="paper/figures/src/v6_more.py (fig_deciles)",
                    provenance=[os.path.relpath(p, REPO) for p in PROV],
                    archived="results/prog195_vs_baselines/sca_v2_prog_vs_area.csv",
                    seconds="paper/figures/src/fig_speed.py load(): clean batch-1 medians; UltraImage campaign mean"))
    with open(os.path.join(OUT, "data.js"), "w") as f:
        f.write("/* Generated by web/build/v5_build_percentiles.py; do not edit. */\n")
        f.write("window.SCA_DATA = window.SCA_DATA || {};\n")
        f.write("window.SCA_DATA.percentiles = " + json.dumps(rec, separators=(",", ":")) + ";\n")
    med = next(s for s in picks if s[0] == 50)
    fb = os.path.join(OUT, "fallback.webp")
    fallback([data[m][med[1]][1] for m, *_ in METHODS], fb)
    print("stops:", [(s["p"], s["rank"], s["key"], s["diff"], s["best"]) for s in stops])
    print("mean", round(mean, 4), "seconds", secs)
    print(f"data size {zoomlib.dir_bytes(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
