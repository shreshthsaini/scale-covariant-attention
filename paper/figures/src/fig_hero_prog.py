"""Page-1 hero (fig:prog_hero): a mosaic of curated progressive-SCA outputs at 4096^2, as a TikZ PDF.

Layout, in the style of the MFlow and LILIE heroes (pictures only): one large square on the left and a 2 x 3 grid of
small squares on its right, full text width (397.5 pt) and BIG_SIDE tall (158.1 pt), 1.5 pt white gutters, no image
borders. Every image is a full 4096^2 progressive-SCA output (sca_v2_prog, out/dupfix_prog/sca_v2_prog: HRDiT's
released three-stage schedule, 1024^2 for 30 steps with stock attention, then the last 17 steps at 2048^2 and the last
10 at 4096^2 with SCA attention; seed 0; asserted from each sidecar), Lanczos-downscaled to DPI pixels per printed
inch. Two images carry one native-pixel (1:1) inset each: a square window cut from the PNG without resampling,
shown enlarged in a free corner of the same image, with a 1 pt frame in the zoom colour (scaZoom) on the source window
and around the inset (each over a faint dark halo so that the frame reads on snow and sky). The only text is one
"4096 x 4096" tag on the large image (6.5 pt sans). The caption is written in main.tex.

Selection (curated; recorded in figures/hero_prog/provenance.json):
* Pool: the 195 Aesthetic-4K evaluation prompts minus the four development prompts, minus every prompt displayed in
  another figure (the curated teaser, zoom4k, showcase, duplication examples, heads_view, method, ablation and
  failures; the uncurated galleries as displayed and fig:progressive), minus the six prompts of the uncurated
  gallery fig:gallery_prog, whose draw was frozen before any image was viewed. 137 prompts (pool(), asserted).
* Shortlist: the pool ranked by progressive SCA's ImageReward percentile plus MUSIQ_c percentile (each within its
  own 195 seed-0 images, eval/prog195_sca_v2); the top 60 were viewed as contact sheets of 384 px thumbnails, then
  the promising ones (and the best of each wanted category further down) whole at about 780 px (SHORTLIST, VIEWED).
* Native audit: fourteen candidates were cut into sixteen 1024 px tiles each and every tile was viewed at 1:1 (AUDIT),
  rejecting any image with artifacts, stripes, speckle, garbled lettering, anatomical errors, or extra or repeated
  people or animals (REJECTED gives every rejection and its reason). The seven images shown cover a landscape,
  architecture, two portraits, an animal, a still life, a fantasy scene and a night scene (MOSAIC roles). Each inset window
  was viewed at 1:1 and 2x before it was fixed.
Ranking on progressive SCA's own metrics and curating by eye favour the method; the caption says the images are
curated, and Table tab:main carries the means.

Everything is generated: deterministic JPEG assets (PIL, fixed quality and subsampling), the TikZ source
(hero_prog/hero_prog.tex), its PDF (tectonic, SOURCE_DATE_EPOCH=0), and the provenance with source paths and SHA256,
metric values, crop boxes and inset geometry. No metric or prompt text is printed in the figure.

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python fig_hero_prog.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

import imgsel
import scafig

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
FIG = HERE.parent
REPO = FIG.parent.parent
ASSETS = FIG / "hero_prog"
OUT_PDF = FIG / "hero_prog.pdf"
TECTONIC = os.path.expanduser("~/.local/bin/tectonic")
EVAL = Path("/scratch/09032/saini_2/sca/eval")
PROG_CSV = EVAL / "prog195_sca_v2" / "base" / "per_image.csv"         # MUSIQ_c (musiq_crop1024)
PROG_EXTRA = EVAL / "prog195_sca_v2" / "extra" / "extra_per_image.csv"  # ImageReward (ir), HPSv2.1
METHOD, RES = "sca_v2_prog", "4096x4096"
PROG = dict(method=METHOD, steps=57, attention="stock@native+sca@highres", guidance=[3.5, 4.5, 6.0],
            stages=["1024x1024", "2048x2048", "4096x4096"], base_steps=30, highres_steps=[17, 10],
            model="black-forest-labs/FLUX.1-dev")  # asserted from every sidecar

W = 397.5                    # text width (TeX pt); the layout unit is the TeX point
G = 1.5                      # white gutter (pt)
BIG_SIDE = (W - 1.5 * G) / 2.5   # W = big + G + 3 small + 2 G, small = (big - G) / 2
SMALL_SIDE = (BIG_SIDE - G) / 2
DPI = 600                    # full images: Lanczos to this many px per printed inch
FRAME_LW = 1.0               # zoom-colour frame (pt) on the source window and around the inset
HALO_LW = 1.7                # dark halo under each frame (pt; black at 35 percent), a 0.35 pt rim on each side
M = 2.5                      # margin (pt) of the insets and the size tag from the image edges
TAG_PT = 6.5                 # the one size tag, sans

# slot "big" or (row, col) of the 2 x 3 grid; zoom: native window (x, y, side) in px, inset side in pt, corner
MOSAIC = [
    dict(slot="big", pid="m3112502764021036396", role="landscape and architecture",
         zoom=dict(box=(1600, 580, 384), inset=54.0, corner="bl",
                   content="the bridge's parapet openings, ashlar masonry and the stone ring of the arch"),
         why=("waterfalls in a gorge under an arched stone bridge: the richest scene that passed the audit; the bridge "
              "masonry is the inset; the bottom-left boulders leave room for it")),
    dict(slot=(0, 0), pid="3657338173664010496", role="portrait",
         why="the strongest portrait in the pool; flaws only at 1:1"),
    dict(slot=(0, 1), pid="m5332089088153794307", role="still life",
         why="pitcher of white flowers with three lemons and one petal, as asked"),
    dict(slot=(0, 2), pid="1594791064055740275", role="fantasy",
         why="armoured woman before a red moon and gothic spires"),
    dict(slot=(1, 0), pid="4783241696049918576", role="night",
         why="one figure under green and purple aurora"),
    dict(slot=(1, 1), pid="7265494513082458083", role="portrait",
         zoom=dict(box=(2534, 1894, 256), inset=24.0, corner="tr",
                   content="the embroidered floral band of her bodice"),
         why="young woman with a red fan at a stone window; nearly clean at 1:1"),
    dict(slot=(1, 2), pid="m3118670343091498341", role="animal",
         why=("a woman kneeling with three wolves: the only animal image in the pool that passed the audit (exactly "
              "three wolves, each with one head and four legs)")),
]
SHORTLIST = dict(  # 2026-11-07; ranks in the pool by progressive SCA's ImageReward + MUSIQ_c percentiles
    contact_sheets=[  # ranks 0 to 59, viewed as five sheets of twelve 384 px thumbnails
        "m7320315357894569493.Jpg", "m6362293977233387960", "m3118670343091498341", "7265494513082458083",
        "5826245701465116957", "58350587024203491", "4673559490456019105", "7543519915112968451",
        "3943607367684487957", "1594791064055740275", "1139740195494332405", "m5996982714209002611",
        "6384782444147762726", "m3108444262580777290", "4820545723762191815", "m5332089088153794307",
        "8253776796901638375", "8562792468822776695", "5941097972571555184", "m9039328566106297566",
        "m3880872769316591135", "3305730548209556640", "4783241696049918576", "m2712631331773394876",
        "1195301905298426486", "m2742856837225126754", "m3651203496105491158", "5374762814090317810",
        "m4460699761507361961", "m2729538226706404616", "175243114551627459", "m1222532847814836069",
        "6422600338162192082", "m5795767365495583614", "m5805253890620089726", "m1057810871814672649",
        "m3112502764021036396", "2155209427123413237.1", "6648013380537069782", "6599424322813629311",
        "m3268147817845827695", "m9219623328452982854", "2232973817337527860", "3657338173664010496",
        "m2670470656524755471", "5968998903196670637", "m6858649860006634381", "3826098298648979216",
        "1638867997226496674", "m210790534622161044", "11799097571596124", "m795146951737249951",
        "m947425340308551929", "9082507599514684465", "1118979843958125111", "8689685467512870777",
        "m7878071503822468846", "3411143122470771726", "9112592444131713102", "1041106339713796826"],
    further_down=["6066373865342273390", "m7946139341219184985"],  # the pool's other architecture prompts, viewed
    categories=("landscape, architecture or city, portrait, animal, interior or still life, fantasy or night; "
                "animal prompts in the pool: the owl, two polar bears and the three wolves (keyword search of all "
                "137 prompts)"))
VIEWED = {  # whole at about 780 px (2026-11-07); not rejected, judged on content and variety
    "m3112502764021036396": "waterfalls in a gorge under an arched stone bridge; clean at overview",
    "3657338173664010496": "elderly man with braids, striped headband, one pipe and bead necklaces; clean at overview",
    "7265494513082458083": "young woman with curls at a stone window, one red fan; clean at overview",
    "m5332089088153794307": "grey pitcher of white flowers, three lemons, one petal; clean at overview",
    "1594791064055740275": "woman in dark armour with a staff before a red moon and gothic spires; clean at overview",
    "4783241696049918576": "one figure with a flashlight under green and purple aurora, steam plume; clean at overview",
    "m1222532847814836069": "gnarled tree under the Milky Way; clean at overview",
    "m947425340308551929": "two blue bottles and four pears in a bowl; clean, not chosen (the pitcher is brighter)",
    "m1057810871814672649": "ivy-covered stone arch into a misty forest with toadstools; clean, not chosen",
    "6066373865342273390": "white church with a golden dome and one walker; flat, rendered look, little fine detail",
    "m3880872769316591135": "red-cloaked figure, tree and Milky Way; faint diagonal streaks across the upper sky at overview; not chosen",
}
AUDIT = {  # every one of the sixteen 1024 px tiles viewed at 1:1 (2026-11-07), suspect spots at 2x
    "m3112502764021036396": ("MINOR: a pseudo-emblem (white panel, black star) in a parapet drain slot at the bridge "
                             "crown (x 2075-2108, y 572-628, about 4 px in print); the keystone hangs about 15 px "
                             "below the arch (a thin dark line at print, reads as a keystone); faint contour-like "
                             "striations on the big right boulder and a fine stipple on shaded rock, 1:1 only; one "
                             "bridge, two tiers of falls and a thin distant fall, no people, no lattice or seam"),
    "3657338173664010496": ("MINOR, all 1:1 only: red and white specks in the inner eye corners; a 25 x 45 px hole in "
                            "the pipe bowl wall; a smooth fold with a dark slit on the throat; a 0.3-level lattice in "
                            "the defocused wall, visible only after a 12x high-pass; columns x 4093-4095 about 8 "
                            "levels brighter; one man, one pipe, both braids attached, no hands in frame"),
    "7265494513082458083": ("MINOR, nearly clean: the fan hand is a back-of-hand grip with the lower three fingers "
                            "as a smooth fist (no wrong count); the fan's gold print is a glitter speckle; one "
                            "woman, one fan, no text, no lattice"),
    "m5332089088153794307": ("MINOR, all 1:1 only: concentric ripple rings in the lemons' highlights (worst on the "
                             "back lemon, x 1300-1530, y 2850-3030); a +-1 level vertical banding on the plain wall; "
                             "sub-level seams at y 768, 1536 and 2304 found only by measurement; a dark "
                             "head-and-shoulders reflection in the glaze (x 2436-2696, y 2980-3280), a faint smudge at "
                             "print, accepted as a plausible reflection of the photographer, not a subject in the scene; "
                             "3 lemons, 1 petal, 1 pitcher, no text"),
    "1594791064055740275": ("MINOR, all 1:1 only: rune-like carvings on the staff shaft (x 2765-2870, y 3420-3950); "
                            "the left hand hangs on both sides of the staff without wrapping it (reads as holding it at "
                            "print); a +1.5 level seam in the flat sky at y 1536 (x 768-1024); 2 to 4 px bright edge "
                            "halos on the staff and gauntlet; one woman, no second face"),
    "4783241696049918576": ("MINOR, all 1:1 only: 2 to 4 px overshoot halos on the ridge under the aurora and on the "
                            "figure's legs and gloves (reads as rim light at print); sub-level seams; wind-ripple "
                            "striation on lit snow; one figure, one flashlight beam, a distant village with lights"),
    "m3118670343091498341": ("MINOR, all 1:1 only: exactly three wolves (3 heads, legs and paws accounted for, no "
                             "shared or floating limb) and one woman (4 fingers and a hidden thumb on the visible "
                             "hand); a dark speckled blob between two fingertips (x 2029-2055, y 2602-2637); a thin "
                             "gold arc on the leggings (x 608-684, y 2932-2991); a 5 to 10 px dark sharpening halo "
                             "around the nose, lips and ear rims; very regular whisker-pad dots; no text, no seam"),
    "m1222532847814836069": ("MINOR, all 1:1 only (passed, not chosen: a second night scene beside the aurora): a "
                             "squirrel-like silhouette on a broken branch stub (x 2240-2320, "
                             "y 1375-1465); a 2 to 4 px light halo on dark edges against the sky; a fine grain in "
                             "the bright Milky Way; stylised claw-like root tips; one tree, one Milky Way"),
}
REJECTED = {  # with the stage that found the flaw
    "7543519915112968451": "contact sheet: garbled lettering on the hanging restaurant sign",
    "6384782444147762726": "contact sheet: sheep before a church, the subject of the teaser strip (fig:teaser)",
    "5941097972571555184": "contact sheet: sheep before a church, the subject of the teaser strip (fig:teaser)",
    "175243114551627459": "contact sheet: sheep before a church, the subject of the teaser strip (fig:teaser)",
    "m9039328566106297566": "contact sheet: neon signs of invented characters",
    "m2712631331773394876": "contact sheet: her back in a mirror reads as a second woman",
    "1195301905298426486": "contact sheet: four men for the five the prompt asks for",
    "m7320315357894569493.Jpg": "whole view: garbled lettering on the book's cover and spine",
    "m3268147817845827695": "whole view: a second, distant figure for the lone figure the prompt asks for",
    "m2670470656524755471": "whole view: the white umbrella floats above the walker with no hand on it",
    "m7946139341219184985": "whole view: a garbled name on the red boat's hull",
    "5826245701465116957": ("native audit: a soft, textureless dark smudge caps the main summit (x 1400-1610, y "
                            "700-800, about 35 levels darker than the sky), visible in the large slot"),
    "m2742856837225126754": ("native audit: legible pseudo-text on the main sign of the yellow house (x 933-1163, y "
                             "2337-2380) and six smaller pseudo-text signs; a fused crowd behind the hedge at 1:1"),
    "1139740195494332405": ("native audit: the owl has a third wing (a spotted folded wing across its chest, x "
                            "1360-1870, y 1870-2570), visible at print; a glyph-like squiggle on the outstretched wing"),
    "4820545723762191815": ("native audit: a second, smaller moon (35 px disc at 2437, 1777) and a second small "
                            "person on the right rock shelf (x 3840-3867, y 3002-3054); a pale disc and a dome in the "
                            "water"),
    "m5795767365495583614": ("native audit: a white calligraphic pseudo-signature at the bottom-left waterline (x "
                             "27-206, y 3997-4068), visible in the 1024 px overview; a tiny bird pasted on the bear's "
                             "back; cut-out halos around the birds and the figure"),
    "3943607367684487957": ("native audit: a dark 8 to 12 px cut-out halo around the whole boy, visible at the large "
                            "slot's print size; pseudo-text 'J3' on the shirt badge; echo outlines on blurred straws"),
}

CURATED = {  # figures whose prompts were not drawn at random: provenance file, fields that name the shown prompts
    "teaser (fig_hero.py)": ("teaser/provenance.json", ["records"]),
    "zoom4k (fig:qual_main)": ("zoom4k_provenance.json", ["examples"]),
    "showcase (fig:hero)": ("showcase/provenance.json", ["records"]),
    "duplication examples (fig:dup)": ("dup_provenance.json", ["panel_a", "panel_b", "panel_c", "sources"]),
    "heads_view": ("heads_view_provenance.json", ["selection", "sources", "panel_a", "panel_b", "panel_c", "panel_d"]),
    "method": ("method_provenance.json", ["output_image", "calibration_image"]),
    "ablation": ("ablation/provenance.json", ["prompts"]),
    "failures": ("failures/provenance.json", ["prompts"]),
}
PID = re.compile(r"a4k_eval_4096_(?:minus)?[0-9][0-9A-Za-z.]*")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


def named(man: dict, fields: list[str]) -> set[str]:
    ids = set()
    for f in fields:
        for s in strings(man.get(f)):
            ids.update(imgsel.short_id(m.group(0)) for m in PID.finditer(s))
    if "prompts" in fields:
        ids.update(e["short_id"] for e in man["prompts"])
    if "examples" in fields:
        ids.update(e["prompt_id"] for e in man["examples"])
    return ids


def rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["method"] == METHOD and r["seed"] == "0" and r["res"] == RES]


def metrics() -> dict[str, dict]:
    """Progressive SCA's per-image ImageReward, MUSIQ_c and HPSv2.1, with percentiles within its own 195 images."""
    out = {imgsel.short_id(r["prompt_id"]): dict(csv_prompt_id=r["prompt_id"], path=r["path"],
                                                 musiq_c=float(r["musiq_crop1024"])) for r in rows(PROG_CSV)}
    for r in rows(PROG_EXTRA):
        e = out[imgsel.short_id(r["prompt_id"])]
        assert e["path"] == r["path"], r["prompt_id"]
        e.update(ir=float(r["ir"]), hpsv21=float(r["hpsv21"]))
    assert len(out) == 195, len(out)
    for col in ("ir", "musiq_c", "hpsv21"):
        vals = [e[col] for e in out.values()]
        for e in out.values():
            e[col + "_pct"] = 100 * sum(v <= e[col] for v in vals) // len(vals)
    return out


def pool(all195: list[str]) -> dict:
    """The eligible prompts: 195 minus development, minus every displayed prompt, minus fig:gallery_prog's draw."""
    dev = imgsel.dev4_ids()
    curated = {}
    for name, (path, fields) in CURATED.items():
        with open(FIG / path) as fh:
            curated[name] = sorted(named(json.load(fh), fields))
    shown = {}
    for p in sorted(FIG.glob("gallery_*/provenance.json")):
        man = json.loads(p.read_text())
        if p.parent.name == "gallery_prog":
            shown[p.parent.name] = sorted(man["draw"])
        elif "display" in man:
            shown[p.parent.name] = sorted(man["display"]["shown"])
        else:
            shown[p.parent.name] = sorted(e.get("short_id") or imgsel.short_id(e["id"]) for e in man["prompts"])
    prog = json.loads((FIG / "progressive" / "provenance.json").read_text())["selection"]["drawn"]
    shown["progressive (fig:progressive)"] = sorted(imgsel.short_id(s) for s in prog)
    excluded = set(dev).union(*map(set, curated.values()), *map(set, shown.values()))
    eligible = sorted(set(all195) - excluded)
    return dict(development=sorted(dev), curated=curated, displayed_uncurated=shown, eligible_count=len(eligible),
                eligible_ids=eligible)


def source(sid: str, m: dict) -> tuple[Path, dict]:
    png = Path(m[sid]["path"])
    meta = json.loads(png.with_suffix(".json").read_text())
    if meta.get("complete") is not True or meta.get("png_sha256") != sha(png):
        raise ValueError(f"incomplete or hash-mismatched source {png}")
    c = meta["config"]
    got = dict(method=c["method"], steps=c["steps"], attention=c["attention"], guidance=c["guidance"],
               stages=c["progressive"]["stages"], base_steps=c["progressive"]["base_steps"],
               highres_steps=c["progressive"]["highres_steps"], model=c["model"])
    if got != PROG or c["res"] != RES or int(c["seed"]) != 0 or c["prompt_id"] != m[sid]["csv_prompt_id"]:
        raise ValueError(f"{png}: configuration {got}")
    if [s["res"] for s in c["sca"]["stages"]] != PROG["stages"][1:]:
        raise ValueError(f"{png}: SCA stages {[s['res'] for s in c['sca']['stages']]}")
    return png, meta


def slot_xy(slot) -> tuple[float, float, float]:
    """Top-left corner (x, y) and side of a slot, pt; y grows downward."""
    if slot == "big":
        return 0.0, 0.0, BIG_SIDE
    r, c = slot
    return BIG_SIDE + G + c * (SMALL_SIDE + G), r * (SMALL_SIDE + G), SMALL_SIDE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="trial run: write the PDF, assets and provenance here instead of the paper tree")
    a = ap.parse_args()
    assets = a.out_dir / "hero_prog" if a.out_dir else ASSETS
    out_pdf = a.out_dir / "hero_prog.pdf" if a.out_dir else OUT_PDF
    scafig.write_tikz_colors(str(HERE / "sca_colors.tex"))
    assets.mkdir(parents=True, exist_ok=True)
    for old in assets.glob("*.jpg"):  # this folder holds only this script's output
        old.unlink()
    m = metrics()
    elig = pool(sorted(m))
    assert elig["eligible_count"] == 137, elig["eligible_count"]
    slots = [e["slot"] for e in MOSAIC]
    assert sorted(map(str, slots)) == sorted(map(str, ["big"] + [(r, c) for r in range(2) for c in range(3)])), slots
    assert len({e["pid"] for e in MOSAIC}) == len(MOSAIC)
    assert not {e["pid"] for e in MOSAIC} & set(REJECTED)
    assert all(e["pid"] in elig["eligible_ids"] for e in MOSAIC), [e["pid"] for e in MOSAIC]
    assert sum("zoom" in e for e in MOSAIC) in (1, 2)
    tex, records, written = [], [], []

    def node(x, y, side, name):
        tex.append(r"\node[anchor=north west,inner sep=0pt] at (%.4f,%.4f) "
                   r"{\includegraphics[width=%.4fpt,height=%.4fpt]{%s}};" % (x, -y, side, side, name))

    def frame(x0, y0, x1, y1):
        """The zoom-colour frame just inside the rectangle (pt), over a faint dark halo."""
        d = FRAME_LW / 2
        r = (x0 + d, -(y0 + d), x1 - d, -(y1 - d))
        tex.append(r"\draw[halo] (%.4f,%.4f) rectangle (%.4f,%.4f);" % r)
        tex.append(r"\draw[frame] (%.4f,%.4f) rectangle (%.4f,%.4f);" % r)

    for e in MOSAIC:
        sid = e["pid"]
        png, meta = source(sid, m)
        img = Image.open(png).convert("RGB")
        assert img.size == (4096, 4096), img.size
        x, y, side = slot_xy(e["slot"])
        px = round(DPI * side / 72.27)
        full = assets / f"{sid}_full.jpg"
        img.resize((px, px), Image.Resampling.LANCZOS).save(full, "JPEG", quality=90, optimize=True, subsampling=0)
        written.append(full)
        node(x, y, side, full.name)
        rec = dict(slot=list(e["slot"]) if e["slot"] != "big" else "big", role=e["role"], prompt_id=sid,
                   csv_prompt_id=m[sid]["csv_prompt_id"], prompt=meta["config"]["prompt"],
                   derived_seed=meta["config"]["derived_seed"], source=str(png), png_sha256=meta["png_sha256"],
                   metrics={k: m[sid][k] for k in ("ir", "ir_pct", "musiq_c", "musiq_c_pct", "hpsv21", "hpsv21_pct")},
                   printed_side_pt=round(side, 3), full_image=full.name, full_image_px=px)
        z = e.get("zoom")
        if z:
            bx, by, bs = z["box"]
            q, corner = z["inset"], z["corner"]
            crop = assets / f"{sid}_inset.jpg"
            img.crop((bx, by, bx + bs, by + bs)).save(crop, "JPEG", quality=95, optimize=True, subsampling=0)
            written.append(crop)
            s = side / 4096
            w0, w1 = (x + bx * s, y + by * s), (x + (bx + bs) * s, y + (by + bs) * s)
            ix = x + M if corner[1] == "l" else x + side - M - q
            iy = y + M if corner[0] == "t" else y + side - M - q
            assert not (ix < w1[0] and w0[0] < ix + q and iy < w1[1] and w0[1] < iy + q), (sid, "inset covers window")
            node(ix, iy, q, crop.name)
            frame(*w0, *w1)
            frame(ix, iy, ix + q, iy + q)
            rec["inset"] = dict(window_xys_px=[bx, by, bs], inset_side_pt=q, corner=corner, file=crop.name,
                                window_on_page_pt=[round(v, 3) for v in (*w0, *w1)],
                                inset_on_page_pt=[round(v, 3) for v in (ix, iy, ix + q, iy + q)],
                                inset_ppi=round(bs / (q / 72.27), 1),
                                magnification=round((q / bs) / s, 2), content=z["content"])
        if e["slot"] == "big":
            tex.append(r"\node[tag,anchor=north west] at (%.4f,%.4f) {4096\,\texttimes\,4096};" % (x + M, -(y + M)))
        records.append(rec)

    doc = r"""%% Generated by paper/figures/src/fig_hero_prog.py; edit the script, not this file.
\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{textcomp}
\usepackage{graphicx}
\usepackage{tikz}
\input{sca_colors.tex}
\begin{document}
\begin{tikzpicture}[x=1pt,y=1pt,
  halo/.style={draw=black,draw opacity=0.35,line width=%.2fpt},
  frame/.style={draw=scaZoom,line width=%.2fpt},
  tag/.style={fill=black,fill opacity=0.55,text opacity=1,text=white,rounded corners=1pt,
              inner xsep=1.7pt,inner ysep=1.1pt,font=\sffamily\fontsize{%.1f}{%.1f}\selectfont}]
\useasboundingbox (0,0) rectangle (%.4f,%.4f);
%s
\end{tikzpicture}
\end{document}
""" % (HALO_LW, FRAME_LW, TAG_PT, TAG_PT + 1, W, -BIG_SIDE, "\n".join(tex))
    (assets / "hero_prog.tex").write_text(doc)
    shutil.copy(HERE / "sca_colors.tex", assets / "sca_colors.tex")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([TECTONIC, "-X", "compile", "hero_prog.tex", "--outdir", tmp], cwd=assets, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       env=dict(os.environ, SOURCE_DATE_EPOCH="0"))  # byte-identical PDF per run
        shutil.copy(Path(tmp) / "hero_prog.pdf", out_pdf)

    size = sum(p.stat().st_size for p in written)
    prov = dict(
        figure="hero_prog (fig:prog_hero)", script="paper/figures/src/fig_hero_prog.py",
        pdf=str(out_pdf.relative_to(REPO)) if not a.out_dir else str(out_pdf), pdf_sha256=sha(out_pdf),
        pdf_bytes=out_pdf.stat().st_size,
        shown=[r["csv_prompt_id"] for r in records],
        method=dict(name=METHOD, images="out/dupfix_prog/sca_v2_prog/4096x4096 (paths as listed in eval/prog195_sca_v2)",
                    configuration=PROG, note="asserted from every sidecar"),
        layout_pt=dict(width=W, height=round(BIG_SIDE, 3), big_side=round(BIG_SIDE, 3),
                       small_side=round(SMALL_SIDE, 3), gutter=G, frame_lw=FRAME_LW, halo_lw=HALO_LW,
                       margin=M, tag_pt=TAG_PT, full_image_dpi=DPI,
                       arrangement="one large square left, a 2 x 3 grid of small squares right"),
        jpeg=dict(full=f"Lanczos to {DPI} px per printed inch, quality 90, 4:4:4",
                  inset="native pixels (1:1), quality 95, 4:4:4", total_bytes=size),
        selection=dict(
            pool_rule=("195 evaluation prompts minus the development prompts, minus every prompt displayed in a "
                       "curated figure or an uncurated gallery (as displayed) or fig:progressive, minus "
                       "fig:gallery_prog's frozen draw"),
            pool=elig,
            ranking=("progressive SCA's ImageReward percentile + MUSIQ_c percentile, each within its own 195 seed-0 "
                     "images (eval/prog195_sca_v2); the top 60 viewed as contact sheets"),
            shortlist=SHORTLIST, viewed=VIEWED, native_audit=AUDIT, rejected=REJECTED,
            chosen={e["pid"]: dict(role=e["role"], why=e["why"]) for e in MOSAIC},
            caveat=("curated: ranked on progressive SCA's own metrics and chosen by eye, which favours the method; "
                    "Table tab:main carries the means")),
        metric_files={str(p): sha(p) for p in (PROG_CSV, PROG_EXTRA)},
        records=records,
        asset_sha256={p.name: sha(p) for p in sorted(written)})
    (assets / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"{out_pdf} {out_pdf.stat().st_size / 1e6:.2f} MB, {W} x {BIG_SIDE:.1f} pt, big {BIG_SIDE:.1f} pt, small "
          f"{SMALL_SIDE:.1f} pt, JPEGs {size / 1e6:.2f} MB")
    for r in records:
        print(r["slot"], r["prompt_id"], r["role"], r.get("inset", {}).get("window_xys_px", ""),
              r.get("inset", {}).get("inset_ppi", ""))


if __name__ == "__main__":
    main()
