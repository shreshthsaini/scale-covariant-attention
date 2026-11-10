"""Uncurated appendix gallery (fig:gallery_prog): AReA, HRDiT, single-pass SCA and progressive SCA at 4096^2.

N = 6 evaluation prompts, in draw order. Rows are the methods, top to bottom AReA, HRDiT, single-pass SCA and
progressive SCA (the headline configuration, last), labelled by rotated two-line names in an 18 pt column; columns are
the prompts, three per block, two blocks. Each cell pair is the full image with its crop window outlined, then the
native 1:1 crop of that window. The macros are those of the other uncurated galleries (snippets/_scagrid.tex;
fig_qual.Geo, g_row, t_full and t_crop): 1 pt between an image and its crop, 2 pt between prompts and rows, 6 pt
between the blocks, full images Lanczos-downscaled to GRID_DPI of their printed size, crops native, every JPEG at
JPEG_Q. Single-pass SCA's label is in the accent colour, progressive SCA's in the accent colour and bold. This layout
(tiles of about 61.8 pt filling one page) replaced the first one (one prompt per row, 48.4 pt tiles, the page about
41 percent blank) after review on 2026-11-07; the prompts, draw and windows are unchanged (LAYOUT_HISTORY).

Selection (frozen in paper/figures/gallery_prog/provenance.json on the first run, 2026-11-07, before any image of
the draw was viewed): random.Random(SEED).sample(sorted(eligible_ids), N), rows in draw order. eligible_ids are the
195 Aesthetic-4K evaluation prompts (the prompt ids of progressive SCA's per-image CSV, eval/prog195_sca_v2) minus
the four development prompts (DEV4) minus every prompt displayed in a curated figure, i.e. a figure whose prompts
were not drawn at random (CURATED: the teaser of fig_hero.py, zoom4k, the showcase, the duplication examples,
heads_view, method, ablation and failures), read from each figure's provenance at freeze time; 167 prompts. The
uncurated galleries and the random fig:progressive do not shrink the pool. No image content or score entered the
choice. Later runs recompute the draw from the frozen pool and stop unless it reproduces the frozen prompts, check
every source hash, and recompute every crop window by the frozen rule. The page-1 hero (fig_hero_prog.py) was
curated after the draw and excludes the drawn prompts; this script stops if a drawn prompt appears in its record.

Crop windows are automatic, never placed by hand: fig_qual.auto_box, the SIZE x SIZE (512 x 512) window with the
highest mean edge energy (PIL FIND_EDGES of the 8x box-downsampled greyscale image) averaged over the four methods,
at least 256 px from every border, as in the other 4096^2 galleries.

Sources: each PNG is the one its 195-prompt per-image CSV scored (AReA eval/a4k195_main, out/main/area; HRDiT
eval/a4k195_panel, out/baseline_expansion_20261017_{v1,rest145}/hrdit_dev_author; single-pass SCA
eval/a4k195_main_sca_v2, out/dupfix/sca_v2, checked to be imgsel.final_twin of the published image; progressive SCA
eval/prog195_sca_v2, out/dupfix_prog/sca_v2_prog), with a complete sidecar whose png_sha256 matches the file. Every
row shares the prompt and derived seed (same prompt, same initial noise), and each method's sampling configuration
is asserted from its sidecar (CONFIG), so the caption states only checked settings.

Writes paper/figures/gallery_prog/ (JPEG tiles and provenance.json) and paper/figures/snippets/gallery_prog.tex (one
complete figure float). It never writes snippets/_scagrid.tex or any other gallery's files.

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python fig_gallery_prog.py
"""

import csv
import json
import os
import random
import re
from multiprocessing import Pool

import imgsel
from fig_qual import G_GROUP, G_PAIR, G_ROW, GRID_DPI, JPEG_Q, SNIP, Geo, auto_box, g_row, rel, t_crop, t_full

NAME = "gallery_prog"
LABEL = "fig:gallery_prog"
SCRIPT = "paper/figures/src/fig_gallery_prog.py"
OUT = os.path.join(imgsel.FIGDIR, NAME)
MANIFEST = os.path.join(OUT, "provenance.json")
SEED, N, RES, SIZE = 0, 6, "4096x4096", 512
FROZEN_ON = "2026-11-07"
EVAL = "/scratch/09032/saini_2/sca/eval"
POOL_CSV = f"{EVAL}/prog195_sca_v2/base/per_image.csv"  # its sca_v2_prog rows name the 195 evaluation prompts
METHODS = [  # key, row label (TeX), per-image CSV that scored the image, method name in that CSV
    ("area", "AReA", f"{EVAL}/a4k195_main/base/per_image.csv", "area"),
    ("hrdit", "HRDiT", f"{EVAL}/a4k195_panel/base/per_image.csv", "hrdit_dev_author"),
    ("sca", r"\textcolor{scaAccent}{\shortstack{Single-pass\\SCA}}", f"{EVAL}/a4k195_main_sca_v2/base/per_image.csv",
     "sca_v2"),
    ("prog", r"\scaOurs{\shortstack{Progressive\\SCA}}", f"{EVAL}/prog195_sca_v2/base/per_image.csv", "sca_v2_prog")]
PER_BLOCK = 3      # prompts per block of columns; the draw fills two blocks
G_BLOCKS = 6       # pt between the two blocks
# Row labels: an 18 pt column of two 8 pt lines, centred (as fig:failures, fig_qual.FAIL_LABEL_PRE)
LABEL_PRE = [r"\setlength{\scaLW}{18pt}%",
             r"\def\scaRowLabel#1#2{\makebox[\scaLW][c]{\rotatebox{90}{\makebox[#1][c]{\color{scaInk}#2}}}}%"]
LAYOUT_HISTORY = [
    dict(date="2026-11-07", layout="one prompt per row; per method the full image, then its crop; method headers",
         tile_pt=48.44, why="first version"),
    dict(date="2026-11-07", layout=("methods as rows (rotated two-line labels), prompts as columns in two blocks of "
                                    "three; each image followed by its crop"),
         tile_pt="about 61.8",
         why=("reviewer: tiles of about 50 pt and the page 41 percent blank; the same six drawn prompts, windows and "
              "draw, larger tiles that fill the page"))]
CONFIG = {  # asserted from every sidecar; the caption states exactly these
    "area": dict(method="area", steps=28, guidance=4.5, model="black-forest-labs/FLUX.1-dev"),
    "sca": dict(method="sca_v2", steps=28, guidance=4.5, model="black-forest-labs/FLUX.1-dev"),
    "prog": dict(method="sca_v2_prog", steps=57, guidance=[3.5, 4.5, 6.0], model="black-forest-labs/FLUX.1-dev",
                 attention="stock@native+sca@highres",
                 stages=["1024x1024", "2048x2048", "4096x4096"], base_steps=30, highres_steps=[17, 10]),
    "hrdit": dict(method="hrdit_dev_author", profile_steps=[30, 17, 10], profile_guidance=[3.5, 4.5, 6.0],
                  profile_model="dev")}
# Curated figures: every figure whose prompts were not drawn at random. Displayed prompt ids are read from the listed
# fields of each provenance file (a4k_eval_4096_<id> strings anywhere inside those fields).
CURATED = {
    "teaser (fig_hero.py)": ("teaser/provenance.json", ["records"]),
    "zoom4k (fig:qual_main)": ("zoom4k_provenance.json", ["examples"]),
    "showcase (fig:hero)": ("showcase/provenance.json", ["records"]),
    "duplication examples (fig:dup)": ("dup_provenance.json", ["panel_a", "panel_b", "panel_c", "sources"]),
    "heads_view": ("heads_view_provenance.json", ["selection", "sources", "panel_a", "panel_b", "panel_c", "panel_d"]),
    "method": ("method_provenance.json", ["output_image", "calibration_image"]),
    "ablation": ("ablation/provenance.json", ["prompts"]),
    "failures": ("failures/provenance.json", ["prompts"]),
}
HERO_PROG = os.path.join(imgsel.FIGDIR, "hero_prog", "provenance.json")  # curated after the draw; must not overlap
PID = re.compile(r"a4k_eval_4096_(?:minus)?[0-9][0-9A-Za-z.]*")


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


def displayed(path, fields):
    """Short ids of the evaluation prompts named in the given fields of a figure's provenance."""
    with open(os.path.join(imgsel.FIGDIR, path)) as fh:
        man = json.load(fh)
    ids = set()
    for f in fields:
        for s in strings(man.get(f)):
            ids.update(imgsel.short_id(m.group(0)) for m in PID.finditer(s))
    if "prompts" in fields:  # ablation and failures also record short ids directly
        ids.update(e["short_id"] for e in man["prompts"])
    if "examples" in fields:  # zoom4k records short ids only
        ids.update(e["prompt_id"] for e in man["examples"])
    return ids


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def all195():
    ids = sorted({imgsel.short_id(r["prompt_id"]) for r in read_rows(POOL_CSV)
                  if r["method"] == "sca_v2_prog" and r["seed"] == "0" and r["res"] == RES})
    if len(ids) != 195:
        raise ValueError(f"{POOL_CSV}: {len(ids)} prompts, expected 195")
    return ids


def freeze_pool():
    """The eligible pool and the draw, computed from figure records only (no image is opened)."""
    base = all195()
    dev = imgsel.dev4_ids()
    if not dev <= set(base) or len(dev) != 4:
        raise ValueError(f"development prompts {sorted(dev)} are not four of the 195")
    curated = {name: sorted(displayed(path, fields)) for name, (path, fields) in CURATED.items()}
    excluded = set().union(*map(set, curated.values()))
    pool = sorted(set(base) - dev - excluded)
    draw = random.Random(SEED).sample(pool, N)
    return dict(n_evaluation_prompts=len(base), development_prompts=sorted(dev), curated_figures=curated,
                curated_in_195=sorted(excluded & set(base)), eligible_count=len(pool), eligible_ids=pool,
                draw=draw)


def sources(sid):
    """Per method: the PNG its per-image CSV scored for this prompt, checked against the sidecar."""
    out, cfgs = {}, {}
    for key, _, path, name in METHODS:
        hits = [r for r in rows[path] if r["method"] == name and r["seed"] == "0" and r["res"] == RES
                and imgsel.short_id(r["prompt_id"]) == sid]
        if len(hits) != 1:
            raise LookupError(f"{path} {name} {sid}: {len(hits)} rows")
        png = hits[0]["path"]
        with open(png[:-4] + ".json") as fh:
            meta = json.load(fh)
        digest = imgsel.sha256(png)
        if meta.get("complete") is not True or meta.get("png_sha256") != digest:
            raise ValueError(f"incomplete or hash-mismatched source {png}")
        if not png.endswith("_seed0.png"):
            raise ValueError(f"not seed 0: {png}")
        out[key] = dict(png=png, sha256=digest, csv=path, csv_method=name, csv_prompt_id=hits[0]["prompt_id"])
        cfgs[key] = meta["config"]
    twin, twin_sha = imgsel.final_png(RES, sid)
    if os.path.abspath(twin) != os.path.abspath(out["sca"]["png"]) or twin_sha != out["sca"]["sha256"]:
        raise ValueError(f"{sid}: single-pass SCA is not the final operator's twin of the published image")
    for key, want in CONFIG.items():
        c = cfgs[key]
        got = {"method": c["method"]}
        if key == "hrdit":
            prof = c["profile"]
            got.update(profile_steps=prof["steps"], profile_guidance=prof["guidance"], profile_model=prof["model"])
        else:
            got.update(steps=c["steps"], guidance=c["guidance"], model=c["model"])
            if key == "prog":
                p = c["progressive"]
                got.update(attention=c["attention"], stages=p["stages"], base_steps=p["base_steps"],
                           highres_steps=p["highres_steps"])
        if got != want:
            raise ValueError(f"{sid} {key}: configuration {got} differs from {want}")
        if c["res"] != RES or int(c["seed"]) != 0:
            raise ValueError(f"{sid} {key}: res {c['res']} seed {c['seed']}")
    prompts = {c["prompt"] for c in cfgs.values()}
    seeds = {c["derived_seed"] for c in cfgs.values()}
    if len(prompts) != 1 or len(seeds) != 1:
        raise ValueError(f"{sid}: prompts or derived seeds differ across methods: {seeds}")
    return out, prompts.pop(), seeds.pop()


def window(job):
    sid, pngs = job
    return sid, list(auto_box([imgsel.load(p) for p in pngs], SIZE))


def tiles(job):
    """Full image at px on its long side (Lanczos) and the native crop, both at JPEG_Q (imgsel.save_jpeg)."""
    sid, key, png, box, px = job
    img = imgsel.load(png)
    full = imgsel.save_jpeg(img, os.path.join(OUT, f"{sid}_{key}.jpg"), JPEG_Q, px)
    crop = imgsel.save_jpeg(imgsel.crop(img, box), os.path.join(OUT, f"{sid}_{key}_crop.jpg"), JPEG_Q, box[2])
    return (sid, key), (full, crop)


CAPTION = (
    r"\textbf{Samples at $4096^2$.} Six prompts, by selection seed~%d) from the %d of the "
    r"%d evaluation prompts that are neither development prompts nor shown in another figure "
    r"when the draw was made. "
    r"FLUX.1-dev, generation seed~0. AReA and single-pass \sca{}: 28 steps at $4096^2$, guidance 4.5. HRDiT and "
    r"progressive \sca{}: HRDiT's schedule of 30 steps at $1024^2$, 17 at $2048^2$ and 10 at $4096^2$ (guidance 3.5, "
    r"4.5 and 6.0); progressive \sca{} uses \sca{} attention in the last two stages. Rows are methods; columns are "
    r"the prompts in draw order, three per block. Each image is followed by a 1:1 crop of its outlined "
    r"$%d\times%d$ window of highest edge energy averaged over the four methods.")


def main():
    global rows
    os.makedirs(OUT, exist_ok=True)
    rows = {path: read_rows(path) for _, _, path, _ in METHODS}
    if os.path.exists(MANIFEST):  # frozen: recompute the draw from the frozen pool and check it
        with open(MANIFEST) as fh:
            man = json.load(fh)
        pool = man["eligible_ids"]
        if pool != sorted(pool) or len(pool) != man["eligible_count"]:
            raise ValueError("eligible_ids are not the sorted frozen pool")
        if random.Random(man["selection_seed"]).sample(pool, man["n"]) != man["draw"]:
            raise ValueError("the recomputed draw does not reproduce the frozen selection")
        frozen = {e["short_id"]: e for e in man["prompts"]}
    else:
        man = dict(schema=1, figure=NAME, label=LABEL, script=SCRIPT, resolution=RES, frozen_on=FROZEN_ON,
                   selection_seed=SEED, n=N, **freeze_pool())
        frozen = {}
    draw = man["draw"]
    if os.path.exists(HERO_PROG):
        with open(HERO_PROG) as fh:
            hero = {imgsel.short_id(m.group(0)) for s in strings(json.load(fh).get("shown", [])) for m in PID.finditer(s)}
        if hero & set(draw):
            raise ValueError(f"drawn prompts shown in the curated hero: {sorted(hero & set(draw))}")

    srcs = {sid: sources(sid) for sid in draw}
    for sid in draw:  # a frozen record must still name the same images
        if sid in frozen and {k: v["sha256"] for k, v in srcs[sid][0].items()} != \
                {k: v["sha256"] for k, v in frozen[sid]["sources"].items()}:
            raise ValueError(f"{sid}: a source image changed since the selection was frozen")
    with Pool(min(len(draw), 6)) as pool:
        boxes = dict(pool.map(window, [(sid, [srcs[sid][0][k]["png"] for k, *_ in METHODS]) for sid in draw]))
    for sid in draw:
        if sid in frozen and frozen[sid]["crop_xy_size"] != boxes[sid]:
            raise ValueError(f"{sid}: the crop rule does not reproduce the frozen window")

    geo = Geo(2 * PER_BLOCK, PER_BLOCK * G_PAIR + (PER_BLOCK - 1) * G_GROUP, label=True)
    px = geo.px()  # sized with fig_qual.LAB_PT (11 pt), so a little larger than the printed tile
    for old in os.listdir(OUT):  # this folder holds only this script's output
        if old.endswith(".jpg"):
            os.remove(os.path.join(OUT, old))
    jobs = [(sid, k, srcs[sid][0][k]["png"], tuple(boxes[sid]), px) for sid in draw for k, *_ in METHODS]
    with Pool(12) as pool:
        t = dict(pool.map(tiles, jobs))

    body = []
    blocks = [draw[i:i + PER_BLOCK] for i in range(0, len(draw), PER_BLOCK)]
    for b, sids in enumerate(blocks):
        for k, (key, label, _, _) in enumerate(METHODS):
            cells = []
            for j, sid in enumerate(sids):
                full, crop = t[sid, key]
                cells += ([G_GROUP] if j else []) + [t_full(full, tuple(boxes[sid]), RES), G_PAIR, t_crop(crop)]
            last = k == len(METHODS) - 1
            body.append((g_row(cells, label=label), G_BLOCKS if last and b < len(blocks) - 1 else G_ROW))
    caption = CAPTION % (man["selection_seed"], man["eligible_count"], man["n_evaluation_prompts"], SIZE, SIZE)
    lines = [f"% Generated by {SCRIPT}; do not edit by hand.",
             f"% Draw: random.Random({man['selection_seed']}).sample(sorted(eligible_ids), {man['n']}) over the "
             f"{man['eligible_count']} prompts frozen in figures/{NAME}/provenance.json, rows in draw order.",
             "% Rows: AReA, HRDiT, single-pass SCA, progressive SCA. Columns: prompts 1-3, then 4-6 (second block); "
             "each full image followed by its native 1:1 crop."]
    lines += [f"% {sid} ({RES}): {srcs[sid][1][:150].replace(chr(0x2014), ' - ').replace(chr(10), ' ')}"
              for sid in draw]
    lines += [r"\input{figures/snippets/_scagrid}%", r"\begin{figure}[!htbp]", r"\begingroup\scaGridStyle"]
    lines += LABEL_PRE + [geo.tex()]
    lines += [tex + (r"\par\vskip%gpt" % gap if i < len(body) - 1 else r"\par") for i, (tex, gap) in enumerate(body)]
    lines += [r"\endgroup", rf"\caption{{{caption}}}", rf"\label{{{LABEL}}}", r"\end{figure}", ""]
    snippet = os.path.join(SNIP, f"{NAME}.tex")
    with open(snippet, "w") as fh:
        fh.write("\n".join(lines))

    man.update(
        methods=[k for k, *_ in METHODS],
        selection=(f"random.Random(selection_seed).sample(sorted(eligible_ids), n); rows in draw order. "
                   f"eligible_ids: the {man['n_evaluation_prompts']} evaluation prompts (sca_v2_prog rows of "
                   f"{POOL_CSV}) minus the development prompts (imgsel.DEV4) minus every prompt displayed in a curated "
                   f"figure (curated_figures, read from each figure's provenance on {man['frozen_on']}); frozen before "
                   f"any image of the draw was viewed; no image content or score entered the choice. Uncurated "
                   f"galleries and the random fig:progressive do not shrink the pool. The page-1 hero (hero_prog) was "
                   f"curated afterwards and excludes the drawn prompts (checked on every run)."),
        crop_rule=(f"fig_qual.auto_box: the {SIZE}x{SIZE} window with the highest mean edge energy (PIL FIND_EDGES of "
                   f"the 8x box-downsampled greyscale image) averaged over the four methods, at least 256 px from "
                   f"every border; recomputed and checked on every run"),
        freeze_note=("The pool and the seed-0 draw were first computed on 2026-11-07 in a scratch selection record, "
                     "before any image was viewed. The script's first run read zoom4k's prompt ids with the full-id "
                     "pattern only, missed its three short ids, and froze a 170-prompt pool with a different draw; "
                     "that record and its tiles were deleted unviewed and the pool re-frozen after the fix. The "
                     "frozen 167-prompt pool and draw are identical to the scratch record."),
        configs=dict(CONFIG, note="asserted from every sidecar of every row"),
        prompts=[dict(short_id=sid, id=srcs[sid][0]["prog"]["csv_prompt_id"], prompt=srcs[sid][1],
                      derived_seed=srcs[sid][2], crop_xy_size=boxes[sid], sources=srcs[sid][0]) for sid in draw],
        layout=dict(rows=("methods as rows with rotated two-line labels in an 18 pt column; the prompts as "
                          "columns in draw order, three per block, two blocks; each full image followed by its "
                          "native crop"),
                    tile_pt="(\\linewidth - 7 pt - 18 pt - 1.5 pt) / 6, about 61.8 pt at 397.5 pt",
                    full_image_px=px, full_image_dpi=GRID_DPI, jpeg_quality=JPEG_Q, crop_px=SIZE,
                    crop_ppi_about=round(SIZE / (61.8 / 72.27), 1),
                    labels=("single-pass SCA in the accent colour, progressive SCA in the accent colour and bold; "
                            "labels repeated on both blocks"),
                    history=LAYOUT_HISTORY),
        caption=caption,
        outputs={rel(p): imgsel.sha256(p) for v in t.values() for p in v})
    with open(MANIFEST, "w") as fh:
        fh.write(json.dumps(man, indent=2) + "\n")
    size = sum(os.path.getsize(p) for v in t.values() for p in v)
    print(f"wrote {snippet}; {NAME}: {len(draw)} prompts {draw}; tile {geo.tile_in() * 72.27:.2f} pt, full {px} px, "
          f"crops {SIZE} px ({SIZE / geo.tile_in():.0f} ppi); JPEGs {size / 1e6:.2f} MB")


rows = {}

if __name__ == "__main__":
    main()
