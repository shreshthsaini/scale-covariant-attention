"""Appendix galleries at a readable size: the first k prompts of each frozen random selection.

fig_qual.py (gallery_4096, gallery_wide) and fig_phase2.py (flux2) froze each gallery's random
prompts, crop windows and source hashes in paper/figures/<gallery>/provenance.json. Their layouts
printed tiles 0.65 to 0.9 in tall (eight tiles per row at 4096^2 and for FLUX.2). This script keeps
every frozen record and shows a deterministic prefix of each selection in the layout of
fig:gallery_6144: one prompt per row, each full image followed by its native-pixel crop, method
headers above. Wide images (2048x4096, 2048x6144) stack the two methods as labelled rows, so each
row spans the text width.

Selection: the seeded draw order is recomputed from the frozen eligible_ids and seed. It must
reproduce the frozen selection exactly; the first k prompts are shown, in draw order. No image
content or score enters the choice. Source PNG hashes and the frozen crop JPEG hashes are checked
before anything is written. Only full images are re-encoded (Lanczos to GRID_DPI of their new
printed size, JPEG quality 85); crops stay the frozen native 1:1 JPEGs. Each provenance.json gains a
"display" record and updated "outputs" hashes for re-encoded tiles; nothing else in it changes.
Tiles of prompts that are no longer shown stay on disk unchanged.

Final operator (2026-10-30): in the galleries named in FINAL the SCA column shows the final operator
(sca_v2, out/dupfix/sca_v2) instead of the published operator (sca_nb). to_final() keeps the frozen prompts
and draw order, points every prompt's SCA source at the final operator's image of the same prompt and seed
(imgsel.final_twin), recomputes each crop window by the frozen crop rule over the methods shown (checking
first that the rule reproduces the frozen window from the published images), rewrites that prompt's crops
and SCA full image, and keeps the published source and window under "published_sca". Reruns reproduce the
same record.

Run on a compute node with the sca env, from paper/figures/src:
    <sca python> fig_gallery_large.py                  # all four galleries
    <sca python> fig_gallery_large.py gallery_flux2    # one gallery
Afterwards do not run fig_qual.py gallery / gallery_wide or fig_phase2.py flux2: they rewrite these
snippets in the superseded small-tile layouts. This script never writes snippets/_scagrid.tex.
"""

import argparse
import json
import os
import random
import re

from PIL import Image

import imgsel
from fig_qual import (G_GROUP, G_PAIR, G_ROW, G_SECTION, GRID_DPI, JPEG_Q, NAMES, OURS_PUB, ROW_OURS, SNIP, Geo,
                      auto_box, g_row, method_head, rel, t_crop, t_full)

SCRIPT = "paper/figures/src/fig_gallery_large.py"
HEAD_NAMES = dict(NAMES, flux2_full="Full attention")
ROW_LABEL = {"area": "AReA", "sca": ROW_OURS}
FINAL = {"gallery_4096", "gallery_4096x2048", "gallery_6144x2048"}  # SCA column: the final operator (sca_v2)

# Captions: a bold claim, then at most 35 words in all; the shared protocol (model, seed, steps, guidance,
# crop placement, exclusions) is stated once in the appendix text, not repeated here.
SPECS = {
    "gallery_4096": dict(
        k=10, parts=[("gallery_4096", 5), ("gallery_4096_p2", 5)], retired=["gallery_4096_p3"],
        captions=[
            r"\textbf{At $4096^2$ SCA keeps texture that AReA blurs or grids, but adds a faint second dancer.} "
            r"Prompts one to five of 32 drawn at random; SCA with the schedule; 1:1 crops follow.",
            r"\textbf{At $4096^2$, continued: SCA turns the steeple into clustered spires and adds two extra "
            r"hikers and a second astronaut.} Prompts six to ten of the random draw in "
            r"Figure~\ref{fig:gallery_4096}; SCA with the schedule.",
        ]),
    "gallery_4096x2048": dict(k=2, captions=[
        r"\textbf{At $2048\times4096$ both methods keep the layout; SCA avoids AReA's faint weave and keeps "
        r"sharper rock texture.} First two of six prompts drawn at random; AReA above, SCA (with the schedule) "
        r"below; 1:1 crops follow."]),
    "gallery_6144x2048": dict(k=2, captions=[
        r"\textbf{At $2048\times6144$ SCA resolves finer texture than AReA but adds two extra climbers.} First "
        r"two of six prompts drawn at random; AReA above, SCA (with the schedule) below; 1:1 crops follow."]),
    "gallery_flux2": dict(  # no sca_v2 run on FLUX.2-dev: the published operator, headed "SCA w/o schedule"
        k=4, captions=[
            r"\textbf{FLUX.2-dev, $4096^2$: full attention blurs and repeats; SCA without the schedule is sharp "
            r"but duplicates subjects in three rows and distorts the portrait.} Four of 50 prompts drawn at "
            r"random; 1:1 crops follow each image."]),
}


def manifest_path(name):
    return os.path.join(imgsel.FIGDIR, name, "provenance.json")


def draw_order(man):
    """Seed, sample size and the frozen selection in draw order, recomputed from eligible_ids."""
    m = re.match(r"random\.Random\((\d+)\)\.sample\(sorted\(eligible_ids\), (\w+)\)", man["selection"])
    if not m:
        raise ValueError(f"{man['figure']}: unknown selection rule {man['selection']!r}")
    seed = int(m.group(1))
    ids = {e.get("id", e["short_id"]): e["short_id"] for e in man["prompts"]}
    n = len(ids) if m.group(2) == "n" else int(m.group(2))
    pool = sorted(man["eligible_ids"])
    if pool != man["eligible_ids"] or len(pool) != man["eligible_count"]:
        raise ValueError(f"{man['figure']}: eligible_ids are not the sorted frozen pool")
    drawn = random.Random(seed).sample(pool, min(n, len(pool)))
    if sorted(drawn) != sorted(ids):
        raise ValueError(f"{man['figure']}: the recomputed draw does not reproduce the frozen selection")
    return seed, n, [ids[p] for p in drawn]


def tiles(name, man, sids, px):
    """(full, crop) JPEG paths per (prompt, method); re-encodes a full image only if its size is stale."""
    out, by, changed, t = os.path.join(imgsel.FIGDIR, name), {e["short_id"]: e for e in man["prompts"]}, {}, {}
    for sid in sids:
        for m in man["methods"]:
            src = by[sid]["sources"][m]
            if imgsel.sha256(src["png"]) != src["sha256"]:
                raise ValueError(f"Changed source image: {src['png']}")
            full, crop = os.path.join(out, f"{sid}_{m}.jpg"), os.path.join(out, f"{sid}_{m}_crop.jpg")
            if imgsel.sha256(crop) != man["outputs"][rel(crop)]:
                raise ValueError(f"Crop differs from the frozen output: {crop}")
            with Image.open(full) as im:
                stale = max(im.size) != px
            if stale or imgsel.sha256(full) != man["outputs"].get(rel(full)):
                imgsel.save_jpeg(imgsel.load(src["png"]), full, JPEG_Q, px)
                changed[rel(full)] = imgsel.sha256(full)
            t[sid, m] = full, crop
    return t, changed


def to_final(name, man, px):
    """Switch the record's SCA column to the final operator; returns the output hashes it rewrote.

    For every prompt of the frozen draw (shown or not): the SCA source becomes imgsel.final_twin of the published
    image; the window is recomputed by the frozen rule (auto_box over every method shown, same size), after checking
    that the rule reproduces the frozen window from the published images; both methods' crops and the SCA full
    image are rewritten (full images of shown prompts at px, the others at their current size)."""
    if man["methods"] != ["area", "sca"]:
        raise ValueError(f"{name}: to_final expects methods area and sca, not {man['methods']}")
    shown = set(man.get("display", {}).get("shown", []))
    out, changed = os.path.join(imgsel.FIGDIR, name), {}
    for e in man["prompts"]:
        pub = e.setdefault("published_sca", dict(png=e["sources"]["sca"]["png"], sha256=e["sources"]["sca"]["sha256"],
                                                 crop_xy_size=list(e["crop_xy_size"])))
        if imgsel.sha256(pub["png"]) != pub["sha256"] or \
                imgsel.sha256(e["sources"]["area"]["png"]) != e["sources"]["area"]["sha256"]:
            raise ValueError(f"Changed source image: {e['short_id']}")
        png, digest = imgsel.final_twin(pub["png"])
        e["sources"]["sca"] = {"png": png, "sha256": digest}
        area = imgsel.load(e["sources"]["area"]["png"])
        size = pub["crop_xy_size"][2]
        if list(auto_box([area, imgsel.load(pub["png"])], size)) != pub["crop_xy_size"]:
            raise ValueError(f"{name} {e['short_id']}: the crop rule does not reproduce the frozen window")
        final = imgsel.load(png)
        box = auto_box([area, final], size)
        e["crop_xy_size"] = list(box)
        for m, img in (("area", area), ("sca", final)):
            crop = os.path.join(out, f"{e['short_id']}_{m}_crop.jpg")
            imgsel.save_jpeg(imgsel.crop(img, box), crop, JPEG_Q, box[2])
            changed[rel(crop)] = imgsel.sha256(crop)
        full = os.path.join(out, f"{e['short_id']}_sca.jpg")
        with Image.open(full) as im:
            side = px if e["short_id"] in shown else max(im.size)
        imgsel.save_jpeg(final, full, JPEG_Q, side)
        changed[rel(full)] = imgsel.sha256(full)
    man["final_operator"] = dict(
        method="sca_v2", images="out/dupfix/sca_v2 (imgsel.final_twin of each published image: same prompt, prompt "
                                "id, derived seed and file name)",
        switched="2026-10-30, by fig_gallery_large.to_final: the frozen prompts and draw order are unchanged; every "
                 "SCA source is the final operator's image; every crop window is recomputed by crop_rule over the "
                 "methods shown; the published source and window are kept per prompt under published_sca")
    return changed


def square_rows(t, sids, methods, by, res, ours):
    head = []
    for k, m in enumerate(methods):
        head += ([G_GROUP] if k else []) + [method_head(m, 2, G_PAIR, HEAD_NAMES, ours=ours)]
    rows = [(g_row(head), G_ROW)]
    for sid in sids:
        box, cells = tuple(by[sid]["crop_xy_size"]), []
        for k, m in enumerate(methods):
            full, crop = t[sid, m]
            cells += ([G_GROUP] if k else []) + [t_full(full, box, res), G_PAIR, t_crop(crop)]
        rows.append((g_row(cells), G_ROW))
    return rows


def stacked_rows(t, sids, methods, by, res, asp):
    """Wide images: one row per method, prompts G_SECTION apart; no header row, and the method names
    appear once, as rotated labels on the first prompt's rows (the caption states the order)."""
    rows = []
    for i, sid in enumerate(sids):
        box = tuple(by[sid]["crop_xy_size"])
        for k, m in enumerate(methods):
            full, crop = t[sid, m]
            rows.append((g_row([t_full(full, box, res), G_PAIR, t_crop(crop)],
                               label=ROW_LABEL[m] if i == 0 else None, indent=i > 0),
                         G_PAIR if k < len(methods) - 1 else G_SECTION))
    return rows


def write_snippet(part, geo, rows, caption, comments):
    lines = [f"% Generated by {SCRIPT}; do not edit by hand."]
    lines += [f"% {c}" for c in comments]
    lines += [r"\input{figures/snippets/_scagrid}%", r"\begin{figure*}[t]", r"\begingroup\scaGridStyle", geo.tex()]
    lines += [tex + (r"\par\vskip%gpt" % gap if i < len(rows) - 1 else r"\par") for i, (tex, gap) in enumerate(rows)]
    lines += [r"\endgroup", rf"\caption{{{caption}}}", rf"\label{{fig:{part}}}", r"\end{figure*}", ""]
    path = os.path.join(SNIP, f"{part}.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", path)


def build(name):
    spec = SPECS[name]
    with open(manifest_path(name)) as fh:
        man = json.load(fh)
    seed, n, draw = draw_order(man)
    k, res, methods = spec["k"], man["resolution"], man["methods"]
    w, h = imgsel.res_size(res)
    asp = w / h
    stacked = asp > 1
    geo = Geo(asp + 1, G_PAIR, label=True) if stacked else Geo(2 * len(methods), len(methods) * G_PAIR + G_GROUP)
    px = geo.px(asp)
    shown = draw[:k]
    if name in FINAL:  # before tiles(), whose checks then hold for the rewritten crops and full images
        man["outputs"].update(to_final(name, man, px))
    t, changed = tiles(name, man, shown, px)
    by = {e["short_id"]: e for e in man["prompts"]}
    parts, start = spec.get("parts", [(name, k)]), 0
    figures = {}
    for (part, count), caption in zip(parts, spec["captions"]):
        sids = shown[start:start + count]
        start += count
        figures[f"fig:{part}"] = sids
        rows = (stacked_rows(t, sids, methods, by, res, asp) if stacked
                else square_rows(t, sids, methods, by, res, "SCA (ours)" if name in FINAL else OURS_PUB))
        span = f"prompts {start - count + 1} to {start} of " if len(parts) > 1 else ""
        comments = [f"Shows {span}the first {k} of the {n} prompts in the frozen draw order of "
                    f"{name}/provenance.json (see its display record), in draw order.",
                    "fig_qual.py and fig_phase2.py skip this gallery while its provenance.json has a display record."]
        comments += [f"{s} ({res}): {by[s]['prompt'][:150].replace(chr(0x2014), ' - ').replace(chr(10), ' ')}"
                     for s in sids]
        write_snippet(part, geo, rows, caption, comments)
    if start != k:
        raise ValueError(f"{name}: the parts show {start} prompts, expected {k}")
    for part in spec.get("retired", []):
        path = os.path.join(SNIP, f"{part}.tex")
        if os.path.exists(path):
            os.remove(path)
            print("removed", path)
    man["outputs"].update(changed)
    man["display"] = {
        "script": SCRIPT,
        "rule": (f"first {k} of the frozen draw random.Random({seed}).sample(sorted(eligible_ids), {n}), "
                 "recomputed from eligible_ids and checked against the frozen selection; shown in draw order"),
        "k": k, "n": n, "draw_order": draw, "shown": shown, "figures": figures,
        "retired_figures": [f"fig:{p}" for p in spec.get("retired", [])],
        "layout": ("one row per method (full image, then its native crop), labelled on the first prompt only; "
                   "prompts stacked" if stacked
                   else "one prompt per row; per method the full image, then its native crop"),
        "tile_height_in": round(geo.tile_in(), 4),
        "full_image_long_side_px": px, "full_image_dpi": GRID_DPI, "jpeg_quality": JPEG_Q,
        "crops": "frozen native 1:1 crop JPEGs, unchanged (hashes under outputs)",
    }
    if k < n:
        man["display"]["unshown_tiles"] = "tiles of the other prompts stay on disk from the superseded layout"
    with open(manifest_path(name), "w") as fh:
        fh.write(json.dumps(man, indent=2) + "\n")
    print(f"{name}: showing {k} of {n} ({', '.join(shown)}); re-encoded {len(changed)} full images at {px} px")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("galleries", nargs="*", help=f"any of {', '.join(SPECS)} (default: all)")
    names = parser.parse_args().galleries or list(SPECS)
    unknown = sorted(set(names) - set(SPECS))
    if unknown:
        parser.error(f"unknown gallery: {', '.join(unknown)}")
    for g in names:
        build(g)
