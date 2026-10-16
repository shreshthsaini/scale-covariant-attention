"""Qualitative figures for the SCA paper: image tiles (JPEG) plus a LaTeX snippet per figure.

Regenerate one figure:   <sca env python> paper/figures/src/fig_qual.py hero
All figures:             <sca env python> paper/figures/src/fig_qual.py all
Options: --methods area sca (gallery columns; any imgsel.ROOTS label, e.g. a new baseline once its
         ROOTS entry exists), --res 4096x4096, --n 32, --seed 0, --workers 16.

Every tile is either a downscaled full image (Lanczos, longest side <= 1600 px) or a native 1:1
crop (never resampled). Crop windows are given by hand in the specs below, or chosen
automatically as the window with the highest mean absolute Laplacian, averaged over all methods
shown in that row so no method is favoured. Output is deterministic: tiles go to
paper/figures/<figure>/ and the snippet to paper/figures/snippets/<figure>.tex. Sources live on
scratch (purge-eligible): /scratch/09032/saini_2/sca/out/...
"""

import argparse
import os
import random
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import imgsel

SNIP = os.path.join(imgsel.FIGDIR, "snippets")
COLORS = ["#E69F00", "#56B4E9", "#009E73", "#D55E00"]  # Okabe-Ito, matches orx_figstyle
GAP = 0.004  # gap between tiles, fraction of \linewidth
LABW = 0.022  # width reserved for a rotated row label
NAMES = {"sca": r"\textbf{SCA (ours)}", "area": "AReA", "full": "Full attention",
         "abl_v0": "No gate", "abl_local": r"$\pi=1$ (local only)", "abl_uniform": r"Uniform $\pi(\sigma)$",
         "abl_global": r"$\pi=0$ (global only)", "abl_mean": "Mean pooling", "abl_layer": r"Per-layer $\pi$",
         "dev_w16": "No gate, window 16", "dev_w32": "No gate, window 32",
         "dypes": "DyPE", "sega": "SEGA", "ultraimage": "UltraImage"}


def wxh(res):
    h, w = res.split("x")
    return rf"{w}$\times${h}"


# ---------------------------------------------------------------- image ops
def detail_map(img, f=8):
    g = img.convert("L").resize((img.width // f, img.height // f), Image.BOX)
    a = np.asarray(g.filter(ImageFilter.FIND_EDGES), dtype=np.float64)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 0
    return a


def auto_box(imgs, size, f=8):
    """Native box (x, y, size) maximising mean edge energy, averaged over imgs (all same size)."""
    a = np.mean([detail_map(i, f) for i in imgs], axis=0)
    k = size // f
    ii = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    m = max(1, a.shape[0] // 16)  # keep away from the border
    s[:m, :] = s[-m:, :] = -1
    s[:, :m] = s[:, -m:] = -1
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return (int(x * f), int(y * f), size)


def outlined(img, max_side, boxes):
    """Downscaled full image with each crop window outlined in its colour."""
    t = imgsel.fit(img, max_side)
    s = t.width / img.width
    lw = max(2, round(t.width / 220))
    d = ImageDraw.Draw(t)
    for (x, y, sz), c in boxes:
        d.rectangle([x * s, y * s, (x + sz) * s - 1, (y + sz) * s - 1], outline=c, width=lw)
    return t


def framed(img, color):
    t = img.copy()
    lw = max(3, round(t.width / 128))
    ImageDraw.Draw(t).rectangle([0, 0, t.width - 1, t.height - 1], outline=color, width=lw)
    return t


def make_tiles(job):
    """job: dict(out, label, res, sid, side, boxes=[(box, color)], crops=[(box, color, name)])."""
    img = imgsel.load(imgsel.find(job["label"], job["res"], job["sid"]))
    out = {}
    if job.get("side"):
        p = os.path.join(job["out"], f"{job['name']}.jpg")
        imgsel.save_jpeg(outlined(img, job["side"], job.get("boxes", [])), p)
        out["full"] = p
    for box, color, cname in job.get("crops", []):
        p = os.path.join(job["out"], f"{job['name']}_{cname}.jpg")
        imgsel.save_jpeg(framed(imgsel.crop(img, box), color), p, max_side=box[2])
        out[cname] = p
    return job["name"], out


def find_boxes(job):
    """job: dict(key, labels, res, sid, size) -> key, auto box over all labels."""
    imgs = [imgsel.load(imgsel.find(l, job["res"], job["sid"])) for l in job["labels"]]
    return job["key"], auto_box(imgs, job["size"])


# ---------------------------------------------------------------- latex layout
def rel(p):
    return os.path.relpath(p, imgsel.PAPER)


def tile(path, h, overlay=None):
    s = rf"\includegraphics[height={h:.4f}\scaW]{{{rel(path)}}}"
    if overlay:
        s += (r"\llap{\raisebox{0.5ex}{\colorbox{black}{\color{white}\tiny " + overlay
              + r"}}\hspace{0.4ex}}")
    return s


def row(items, label=None, total=0.999):
    """items: list of (aspect, path, overlay) or ('stack', [(path, overlay), ...]) (squares)."""
    asp = sum(0.5 if it[0] == "stack" else it[0] for it in items)
    avail = total - (LABW if label is not None else 0) - GAP * (len(items) - 1)
    h = avail / asp
    parts = []
    for it in items:
        if it[0] == "stack":
            hs = (h - GAP) / 2
            a, b = it[1]
            parts.append(r"\begin{minipage}[b]{%.4f\scaW}" % hs + tile(a[0], hs, a[1])
                         + r"\\[%.4f\scaW]" % GAP + tile(b[0], hs, b[1]) + r"\end{minipage}")
        else:
            parts.append(tile(it[1], h, it[2] if len(it) > 2 else None))
    lab = ""
    if label is not None:
        lab = (r"\makebox[%.4f\scaW][l]{\rotatebox{90}{\makebox[%.4f\scaW][c]{%s}}}"
               % (LABW, h, label))
    body = lab + (r"\hspace{%.4f\scaW}" % GAP).join(parts)
    return r"\makebox[\textwidth][l]{" + body + "}", h


def header(cells, h, label=False):
    """cells: list of (aspect, text); same geometry as a row of height h."""
    parts = [r"\makebox[%.4f\scaW][c]{%s}" % (a * h, t) for a, t in cells]
    pre = r"\hspace*{%.4f\scaW}" % LABW if label else ""
    return r"\makebox[\textwidth][l]{" + pre + (r"\hspace{%.4f\scaW}" % GAP).join(parts) + "}"


def snippet(name, body_lines, caption, label, env="figure*", comments=()):
    os.makedirs(SNIP, exist_ok=True)
    lines = [f"% Generated by paper/figures/src/fig_qual.py {name}; do not edit by hand."]
    lines += [f"% {c}" for c in comments]
    lines += [rf"\begin{{{env}}}[t]", r"\centering\sffamily\scriptsize\ifdefined\scaW\else\newlength{\scaW}\fi\setlength{\scaW}{\textwidth}",
              r"\setlength{\fboxsep}{1pt}\setlength{\lineskip}{0pt}\setlength{\lineskiplimit}{0pt}"]
    lines += [b + r"\\[%.4f\textwidth]" % GAP if i < len(body_lines) - 1 else b
              for i, b in enumerate(body_lines)]
    lines += [r"\rmfamily\normalsize", rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\end{{{env}}}", ""]
    p = os.path.join(SNIP, f"{name}.tex")
    with open(p, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", p)


def prompt_comments(pairs):
    out = []
    for label, res, sid in pairs:
        try:
            out.append(f"{sid} ({res}): {imgsel.prompt(imgsel.find(label, res, sid))[:150].replace(chr(0x2014), ' - ')}")
        except FileNotFoundError:
            out.append(f"{sid} ({res}): missing for {label}")
    return out


def run(jobs, workers):
    with Pool(workers) as pool:
        return dict(pool.map(make_tiles, jobs, chunksize=1))


def boxes_for(specs, workers):
    with Pool(workers) as pool:
        return dict(pool.map(find_boxes, specs, chunksize=1))


# ---------------------------------------------------------------- figures
HERO = dict(
    pano=("sca", "2048x6144", "m2742856837225126754", (3456, 128, 768)),
    portrait=("sca", "4096x4096", "3657338173664010496", [(1600, 1280, 768), (1408, 2304, 640)]),
    cathedral=("sca", "4096x4096", "m8737279395568207187"),
    train=("sca", "2048x4096", "6978695891801939003"),
    still=("sca", "4096x4096", "m947425340308551929"),
    compare=("4096x4096", "m3651203496105491158", (1280, 1280, 768)),
)

MAIN = ["m1057810871814672649", "8142852677968042257", "2100482621711410240",
        "m2729538226706404616", "m6394968997939197542", "1052899493017962912"]
MAIN_BOX = {}  # sid -> (x, y, size) overrides of the automatic crop window

ABL = ["1139740195494332405", "1516768667427572324", "4525259556400020626", "7009273464006889449"]
ABL_METHODS = ["abl_v0", "abl_local", "abl_uniform", "abl_global", "abl_mean", "sca"]

FAIL_DUP = ["7354534197758042049", "3614061571823823193"]  # development prompts
FAIL_SCA = [  # (sid, crop box, short description used in the caption order)
    ("7396058529551312081", (2304, 1408, 1024)),  # thin diagonal sword: staircase blocks
    ("1316997283360365550", (2560, 1024, 768)),   # blocky patch beside the coat
    ("m8648158440845565467", (128, 1280, 1024)),  # dancer: a foot where the rear arm should be
    ("m702025498990232048", (0, 2560, 1024)),     # dense illustration: border dissolves into speckle
]


def used_ids():
    return {HERO[k][2] for k in ("pano", "portrait", "cathedral", "train", "still")} | {HERO["compare"][1]} \
        | set(MAIN) | {s for s, _ in FAIL_SCA}


def fig_hero(a):
    out = os.path.join(imgsel.FIGDIR, "hero")
    l, r, sid, pbox = HERO["pano"]
    ql, qr, qsid, (b1, b2) = HERO["portrait"]
    cres, csid, cbox = HERO["compare"]
    jobs = [
        dict(out=out, name="pano", label=l, res=r, sid=sid, side=1600, boxes=[(pbox, COLORS[0])],
             crops=[(pbox, COLORS[0], "crop")]),
        dict(out=out, name="portrait", label=ql, res=qr, sid=qsid, side=1000,
             boxes=[(b1, COLORS[1]), (b2, COLORS[2])], crops=[(b1, COLORS[1], "crop1"), (b2, COLORS[2], "crop2")]),
        dict(out=out, name="cathedral", label="sca", res=HERO["cathedral"][1], sid=HERO["cathedral"][2], side=1000),
        dict(out=out, name="train", label="sca", res=HERO["train"][1], sid=HERO["train"][2], side=1600),
        dict(out=out, name="still", label="sca", res=HERO["still"][1], sid=HERO["still"][2], side=1000),
        dict(out=out, name="cmp_sca", label="sca", res=cres, sid=csid, side=1000, boxes=[(cbox, COLORS[3])],
             crops=[(cbox, COLORS[3], "crop")]),
        dict(out=out, name="cmp_area", label="area", res=cres, sid=csid, crops=[(cbox, COLORS[3], "crop")]),
        dict(out=out, name="cmp_full", label="full", res=cres, sid=csid, crops=[(cbox, COLORS[3], "crop")]),
    ]
    t = run(jobs, a.workers)
    rA, _ = row([(3, t["pano"]["full"], wxh(r)), (1, t["pano"]["crop"], "1:1 crop")])
    rB, _ = row([(1, t["portrait"]["full"], wxh(qr)),
                 ("stack", [(t["portrait"]["crop1"], None), (t["portrait"]["crop2"], None)]),
                 (1, t["cathedral"]["full"], wxh(HERO["cathedral"][1])),
                 (2, t["train"]["full"], wxh(HERO["train"][1]))])
    rC, _ = row([(1, t["cmp_sca"]["full"], wxh(cres)), (1, t["cmp_full"]["crop"], "Full attention"),
                 (1, t["cmp_area"]["crop"], "AReA"), (1, t["cmp_sca"]["crop"], r"\textbf{SCA}"),
                 (1, t["still"]["full"], wxh(HERO["still"][1]))])
    cap = (r"\textbf{SCA generates coherent, detailed images up to 6144$\times$2048 from FLUX.1-dev without "
           r"training.} All images are SCA outputs at seed 0 (labels give width$\times$height); framed tiles are "
           r"1:1 crops at native resolution from the outlined windows. Bottom row: the same prompt and seed at "
           r"4096$\times$4096 with full attention, which collapses into a blurred field, with AReA, whose sky breaks "
           r"into tiled facets, and with SCA.")
    snippet("hero", [rA, rB, rC], cap, "fig:hero",
            comments=prompt_comments([(l, r, sid), (ql, qr, qsid), ("sca", HERO["cathedral"][1], HERO["cathedral"][2]),
                                      ("sca", HERO["train"][1], HERO["train"][2]),
                                      ("sca", cres, csid), ("sca", HERO["still"][1], HERO["still"][2])]))


def fig_main(a):
    out = os.path.join(imgsel.FIGDIR, "qual_main")
    methods = a.methods or ["full", "area", "sca"]
    res, size = a.res, 512
    auto = boxes_for([dict(key=s, labels=methods, res=res, sid=s, size=size) for s in MAIN if s not in MAIN_BOX],
                     a.workers)
    box = {s: MAIN_BOX.get(s, auto.get(s)) for s in MAIN}
    jobs = [dict(out=out, name=f"{s}_{m}", label=m, res=res, sid=s, side=700, boxes=[(box[s], COLORS[0])],
                 crops=[(box[s], COLORS[0], "crop")]) for s in MAIN for m in methods]
    t = run(jobs, a.workers)
    lines = []
    for blk in (MAIN[:3], MAIN[3:]):
        for m in methods:
            items = []
            for s in blk:
                items += [(1, t[f"{s}_{m}"]["full"]), (1, t[f"{s}_{m}"]["crop"])]
            lines.append(row(items, label=NAMES.get(m, m))[0])
    cap = (r"\textbf{At 4096$\times$4096, full attention collapses, AReA loses detail and adds texture "
           r"artifacts, and SCA keeps both layout and fine detail.} FLUX.1-dev, same prompt and seed (0) per "
           r"column pair; each full image is followed by a 1:1 crop of its outlined 512$\times$512 window. "
           r"Crop windows are chosen automatically as the region with the most edge energy averaged over the "
           r"three methods. Prompts were picked for variety from the 191 evaluation prompts; uncurated random "
           r"samples are in Figure~\ref{fig:gallery_4096}.")
    snippet("qual_main", lines, cap, "fig:qual_main", comments=prompt_comments([("sca", res, s) for s in MAIN]))


def gallery_ids(res, n, seed, labels, exclude=()):
    common = None
    for l in labels:
        ids = {imgsel.short_id(k) for k in imgsel.index(l, res)}
        common = ids if common is None else common & ids
    pool = sorted(common - imgsel.dev4_ids() - used_ids() - set(exclude))
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))))


def fig_gallery_square(a):
    """AReA vs SCA (or --methods) at a square resolution, random prompts, 16 per page."""
    methods = a.methods or ["area", "sca"]
    res = a.res
    tag = res.split("x")[0] if len(set(res.split("x"))) == 1 else res
    name = f"gallery_{tag}"
    out = os.path.join(imgsel.FIGDIR, name)
    ids = gallery_ids(res, a.n, a.seed, methods)
    size = 512 * int(res.split("x")[0]) // 4096 if int(res.split("x")[0]) > 4096 else 512
    auto = boxes_for([dict(key=s, labels=methods, res=res, sid=s, size=size) for s in ids], a.workers)
    jobs = [dict(out=out, name=f"{s}_{m}", label=m, res=res, sid=s, side=560, boxes=[(auto[s], COLORS[0])],
                 crops=[(auto[s], COLORS[0], "crop")]) for s in ids for m in methods]
    t = run(jobs, a.workers)
    per_row = 2 if len(methods) <= 2 else 1
    per_page = 16 if per_row == 2 else 6
    pages = [ids[i:i + per_page] for i in range(0, len(ids), per_page)]
    for pi, page in enumerate(pages):
        lines = []
        for i in range(0, len(page), per_row):
            items = []
            for s in page[i:i + per_row]:
                for m in methods:
                    items += [(1, t[f"{s}_{m}"]["full"]), (1, t[f"{s}_{m}"]["crop"])]
            line, h = row(items)
            if i == 0:
                lines.append(header([(2 + GAP / h, NAMES.get(m, m)) for m in methods] * per_row, h))
            lines.append(line)
        cap = (rf"\textbf{{Random prompts at {wxh(res)}, {' vs '.join(NAMES.get(m, m) for m in methods)}"
               rf"{' (continued)' if pi else ''}.}} ")
        if pi == 0:
            cap += (rf"{len(ids)} prompts sampled uniformly at random (seed {a.seed}) from the evaluation prompts, "
                    r"excluding the development prompts and those shown in the main text; not curated. "
                    r"Each full image is followed by a 1:1 crop of its outlined window, chosen automatically "
                    r"as the region with the most edge energy averaged over the methods. FLUX.1-dev, seed 0.")
        else:
            cap += r"Same protocol as Figure~\ref{fig:" + name + "}."
        lab = f"fig:{name}" + ("" if pi == 0 else f"_p{pi + 1}")
        snippet(name + ("" if pi == 0 else f"_p{pi + 1}"), lines, cap, lab,
                comments=prompt_comments([("sca" if "sca" in methods else methods[0], res, s) for s in page]))


def fig_gallery_wide(a):
    methods = a.methods or ["area", "sca"]
    taken = set(gallery_ids("4096x4096", a.n, a.seed, ["area", "sca"]))
    for res in (a.wide or ["2048x4096", "2048x6144"]):
        h, w = (int(v) for v in res.split("x"))
        name = f"gallery_{w}x{h}"
        out = os.path.join(imgsel.FIGDIR, name)
        # disjoint from the square gallery and from earlier wide galleries, so every prompt appears once
        ids = gallery_ids(res, a.n_wide, a.seed + 1 + len(taken), methods, exclude=taken)
        taken |= set(ids)
        size = 512
        auto = boxes_for([dict(key=s, labels=methods, res=res, sid=s, size=size) for s in ids], a.workers)
        jobs = [dict(out=out, name=f"{s}_{m}", label=m, res=res, sid=s, side=1200 if w / h > 2.5 else 1000,
                     boxes=[(auto[s], COLORS[0])], crops=[(auto[s], COLORS[0], "crop")]) for s in ids for m in methods]
        t = run(jobs, a.workers)
        asp = w / h
        lines = []
        for i, s in enumerate(ids):
            items = []
            for m in methods:
                items += [(asp, t[f"{s}_{m}"]["full"]), (1, t[f"{s}_{m}"]["crop"])]
            line, hh = row(items)
            if i == 0:
                lines.append(header([(asp + 1 + GAP / hh, NAMES.get(m, m)) for m in methods], hh))
            lines.append(line)
        cap = (rf"\textbf{{Random prompts at {w}$\times${h} (width$\times$height), "
               rf"{' vs '.join(NAMES.get(m, m) for m in methods)}.}} "
               rf"{len(ids)} prompts sampled at random from the evaluation prompts, disjoint from the other galleries and excluding the "
               r"development prompts and those in the main text; not curated. Each image is followed by a 1:1 crop "
               r"of its outlined 512$\times$512 window, chosen automatically as the region with the most edge energy "
               r"averaged over the methods. FLUX.1-dev, seed 0.")
        snippet(name, lines, cap, f"fig:{name}",
                comments=prompt_comments([("sca", res, s) for s in ids]))


def fig_ablation(a):
    out = os.path.join(imgsel.FIGDIR, "ablation")
    res, size = "4096x4096", 512
    methods = a.methods or ABL_METHODS
    auto = boxes_for([dict(key=s, labels=methods, res=res, sid=s, size=size) for s in ABL], a.workers)
    jobs = [dict(out=out, name=f"{s}_{m}", label=m, res=res, sid=s, side=640, boxes=[(auto[s], COLORS[0])],
                 crops=[(auto[s], COLORS[0], "crop")]) for s in ABL for m in methods]
    t = run(jobs, a.workers)
    lines = []
    for i, s in enumerate(ABL):
        rf_, h = row([(1, t[f"{s}_{m}"]["full"]) for m in methods])
        rc, _ = row([(1, t[f"{s}_{m}"]["crop"]) for m in methods])
        if i == 0:
            lines.append(header([(1, NAMES.get(m, m)) for m in methods], h))
        lines += [rf_, rc]
    cap = (r"\textbf{Removing the gate duplicates subjects; dropping either branch or the per-head weights "
           r"costs detail or coherence.} Four ablation prompts at 4096$\times$4096, seed 0; each prompt shows the "
           r"full images and, below them, 1:1 crops of the outlined 512$\times$512 window (chosen automatically as "
           r"the region with the most edge energy averaged over the six variants). No gate: the calibrated mix "
           r"from the first step. $\pi=1$ and $\pi=0$: after the gate, the local or the global branch alone. "
           r"Uniform $\pi(\sigma)$: one weight shared by all heads. Mean pooling: average-pooled instead of "
           r"subsampled global keys. Without the gate the subject repeats across the canvas. $\pi=0$ crops show "
           r"the coarse grid of the pooled branch, and mean pooling loses the layout. $\pi=1$ and uniform "
           r"$\pi(\sigma)$ look close to SCA in these crops; their deficits show in the metrics.")
    snippet("ablation", lines, cap, "fig:ablation_grid", comments=prompt_comments([("sca", res, s) for s in ABL]))


def fig_failures(a):
    out = os.path.join(imgsel.FIGDIR, "failures")
    res = "4096x4096"
    dup_methods = ["abl_v0", "dev_w16", "sca"]
    jobs = [dict(out=out, name=f"dup_{s}_{m}", label=m, res=res, sid=s, side=640) for s in FAIL_DUP for m in dup_methods]
    for s, b in FAIL_SCA:
        jobs.append(dict(out=out, name=f"err_{s}_sca", label="sca", res=res, sid=s, side=640,
                         boxes=[(b, COLORS[0])], crops=[(b, COLORS[0], "crop")]))
        jobs.append(dict(out=out, name=f"err_{s}_area", label="area", res=res, sid=s, crops=[(b, COLORS[0], "crop")]))
    t = run(jobs, a.workers)
    lines = []
    r1, h = row([(1, t[f"dup_{s}_{m}"]["full"]) for s in FAIL_DUP for m in dup_methods])
    dup_names = ["No gate, window 64", "No gate, window 16", NAMES["sca"]]
    lines.append(header([(1, t_) for t_ in dup_names] * 2, h))
    lines.append(r1)
    hdr_done = False
    for pair in (FAIL_SCA[:2], FAIL_SCA[2:]):
        items = []
        for s, _ in pair:
            items += [(1, t[f"err_{s}_sca"]["full"]), (1, t[f"err_{s}_sca"]["crop"]), (1, t[f"err_{s}_area"]["crop"])]
        rr, h = row(items)
        if not hdr_done:
            lines.append(header([(1, "SCA"), (1, "SCA crop"), (1, "AReA crop")] * 2, h))
            hdr_done = True
        lines.append(rr)
    cap = (r"\textbf{Failure modes.} Top: without the gate, local windows decide the layout at high noise and "
           r"duplicate the subject (no gate, 64-token window; no gate, 16-token window, which gives smaller and more "
           r"numerous copies); SCA keeps one subject. Two development prompts, 4096$\times$4096. Middle and bottom: "
           r"errors SCA still makes on evaluation prompts, with 1:1 crops of the outlined window and the same "
           r"window from AReA. A thin diagonal sword is drawn as a staircase of short blocks; a patch beside the "
           r"coat turns into a coarse grid; a dancer's rear arm ends in a foot; and the border of a dense "
           r"illustration dissolves into speckle. AReA's images show none of these four errors, though its "
           r"compositions differ, so its crops are not of the same objects. All seed 0.")
    snippet("failures", lines, cap, "fig:failures",
            comments=prompt_comments([("sca", res, s) for s in FAIL_DUP] + [("sca", res, s) for s, _ in FAIL_SCA]))


FIGS = {"hero": fig_hero, "qual_main": fig_main, "gallery": fig_gallery_square, "gallery_wide": fig_gallery_wide,
        "ablation": fig_ablation, "failures": fig_failures}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fig", choices=list(FIGS) + ["all"])
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--res", default="4096x4096")
    ap.add_argument("--wide", nargs="*", help="non-square resolutions HxW for gallery_wide")
    ap.add_argument("--n", type=int, default=32, help="square gallery prompts")
    ap.add_argument("--n-wide", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    for name in (FIGS if a.fig == "all" else [a.fig]):
        FIGS[name](a)


if __name__ == "__main__":
    main()
