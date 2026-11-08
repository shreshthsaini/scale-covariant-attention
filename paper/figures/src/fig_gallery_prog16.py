"""Appendix gallery (fig:gallery_prog16, fig:gallery_prog16b): AReA, HRDiT and progressive SCA.

Sixteen evaluation prompts at 4096^2, eight per float, one prompt per row. Columns, left to right: AReA, HRDiT
and progressive SCA (ours, last, bold accent header); per method the full image with its crop window outlined, then
the native 1:1 crop of that window, shared by the row. Layout and macros of the other appendix galleries
(snippets/_scagrid.tex; fig_qual.Geo, g_row, t_full, t_crop, head_cell): 1 pt between an image and its crop, 2 pt
between methods and rows, full images Lanczos to GRID_DPI of their printed size, every JPEG at JPEG_Q. Each float is
a full page ([p]); the snippet holds both floats.

Selection (recorded in figures/gallery_prog16/provenance.json; the captions state the prompt set, seed and protocol):
* Pool and audit as fig_qual_prog.py: the 195 Aesthetic-4K prompts minus the development prompts, every prompt of a
  curated figure (teaser, zoom4k, showcase, duplication examples, heads_view, method, ablation, failures, hero_prog)
  and fig:gallery_prog's draw; every baseline's ImageReward at or above its own 10th percentile; ranked by
  progressive SCA's per-image advantage (MUSIQ_c and ImageReward ranks) over AReA, UltraImage and HRDiT; the top 35
  eligible prompts audited (progressive image whole and as sixteen 1:1 tiles; baselines whole). The prompts of
  fig:qual_prog are not reused (asserted).
* Windows: 320 x 320 px, placed by hand (curated) where the 1:1 crops show progressive SCA's advantage most clearly
  and free of any flaw of the progressive image.
* Images: those scored by the 195-prompt per-image CSVs (AReA eval/a4k195_main, HRDiT eval/a4k195_panel, progressive
  SCA eval/prog195_sca_v2), with complete sidecars whose png_sha256 matches; every row shares prompt and derived seed.

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python fig_gallery_prog16.py
"""

import csv
import hashlib
import json
import os
from multiprocessing import Pool

from PIL import Image

import imgsel
from fig_qual import G_GROUP, G_PAIR, G_ROW, GRID_DPI, JPEG_Q, SNIP, Geo, g_row, head_cell, rel, t_crop, t_full
import fig_qual_prog as qp

Image.MAX_IMAGE_PIXELS = None
NAME = "gallery_prog16"
LABELS = ["fig:gallery_prog16", "fig:gallery_prog16b"]
SCRIPT = "paper/figures/src/fig_gallery_prog16.py"
OUT = os.path.join(imgsel.FIGDIR, NAME)
RES, WIN, PER_FLOAT = "4096x4096", 320, 8
METHODS = [m for m in qp.METHODS if m[0] in ("area", "hrdit", "prog")]
HEADS = {"area": ("AReA", False), "hrdit": ("HRDiT", False), "prog": ("Progressive SCA", True)}

GALLERY = [{'pid': '5374762814090317810',
  'window': (2432, 3264),
  'why': 'mountains, lake and winding road: clarity 4 of 5',
  'crops': {'area': 'flat grey road with uniform grass and a fine lattice',
            'hrdit': 'a flat cartoon road with dashed lines and vector leaves',
            'prog': 'crisp double yellow line, white edge lines, textured gravel shoulder and trees'},
  'audit': 'MINOR: 1:1 only: honeycomb-like fine grain in meadow and lake (3x); fine striation on hazy rock; faint '
           'x~2113 sky line'},
 {'pid': 'm7878071503822468846',
  'window': (1600, 2560),
  'why': 'diver over a coral reef: clarity 3 of 5; HRDiT as crisp as ours in most other coral windows',
  'crops': {'area': 'hazy reef with a crosshatch lattice and white speckle blobs',
            'hrdit': 'mostly soft hazy distant coral heads',
            'prog': 'crisp urchin corals with resolved polyps and textured rock'},
  'audit': 'MINOR: 1:1 only: tiny pseudo-text on the tank label, harness tag and spear blade; small odd objects on '
           'the reef (a yellow gear-like disc, a cyan cube); branching bubble plume; reversed spear'},
 {'pid': '6599424322813629311',
  'window': (288, 3456),
  'why': 'blonde woman: clarity 3 of 5',
  'crops': {'area': 'flat grey wall with a fine diagonal lattice and two specks',
            'hrdit': 'soft wall and a ringed finger',
            'prog': 'crisp ornate gold bracelet with specular beads and wrist tendons'},
  'audit': 'MINOR: 1:1 only: thin dark contour with a green fringe along the out-of-focus sleeve edge'},
 {'pid': 'm3130703310573987496',
  'window': (2176, 2432),
  'why': 'green hills with a tree and a stone wall: clarity 4 of 5',
  'crops': {'area': 'flat green field with a faint lattice',
            'hrdit': 'soft dark hillside with blurred tufts',
            'prog': 'crisp, shaded three-dimensional dry-stone wall'},
  'audit': 'MINOR: 1:1 only: faint light rim on the wall top and fine grain on stone faces (4x); distant small '
           'trees and shrubs read as scenery (as in UltraImage and HRDiT)'},
 {'pid': 'm6362293977233387960',
  'window': (1080, 1928),
  'why': 'fantasy battle: clarity 3 of 5',
  'crops': {'area': 'murky dark skin of its own orc, grain, no structure',
            'hrdit': 'a different region (a forearm with a leather bracer), moderately sharp',
            'prog': 'sharp face with stubble, wrinkles, glowing eye and individual teeth'},
  'audit': 'MINOR: 1:1 only: 4 px dark line on the right border; bright halo rims on dark shapes against fog; '
           'illegible rune-like belt engraving; stylised feet and boots'},
 {'pid': '8689685467512870777',
  'window': (3440, 1240),
  'why': "Milky Way over a lake: clarity 4 of 5; HRDiT's hatch may be stronger than its usual; alternative window "
         '(520, 1400)',
  'crops': {'area': 'flat sky with an obvious diagonal stripe lattice',
            'hrdit': 'starry sky covered by a diagonal hatch',
            'prog': 'crisp pine top with individual needle tufts on a clean sky'},
  'audit': 'MINOR: 1:1 only: fine diagonal weave over the bright Milky Way core (clear at 4x); stars drawn as '
           'small five-point glyphs; tiny specks on the water'},
 {'pid': 'm654082322195551039',
  'window': (840, 3520),
  'why': 'woman with glasses and jewellery: clarity 3 of 5; advantage clear against UltraImage only; HRDiT has a '
         'watermark-like K logo in its corner',
  'crops': {'area': 'smeared out-of-focus skin with faint vertical streaks',
            'hrdit': 'plain shoulder skin with pores, sharp',
            'prog': 'crisp faceted centre stone and clean pave setting'},
  'audit': 'MINOR: 1:1 only: 2-4 px light halos at the glasses frame and finger outlines; fingerprint-like ridges '
           'on the back of the hand; possible echo arcs on the far-eye lashes'},
 {'pid': '60170519444801447',
  'window': (3168, 1056),
  'why': 'frozen waterfall: clarity 3 of 5',
  'crops': {'area': 'blank sky over a flat snow plain with dark specks',
            'hrdit': 'crisp icicles over a brick-like rock grid',
            'prog': 'distinct crisp tiered icicles with snow caps'},
  'audit': 'MINOR: 1:1 only: faint x~2117 vertical line (about 1 level) in sky and snow; 1.9-level step at y 768 '
           'in sky; cut-out rims at snow-water edges'},
 {'pid': '4537479891260587671',
  'window': (720, 2560),
  'why': 'fjord with light beams: clarity 4 of 5',
  'crops': {'area': 'flat blue haze with faint vertical stripes, no structure',
            'hrdit': 'stylised wavy rock texture with white speckles',
            'prog': 'crisp moss-streaked cliff faces with white surf at the waterline'},
  'audit': 'MINOR: 1:1 or overview-subtle: 2-4 px light halos on dark silhouettes (read as rim light); two '
           'straight white bars on the water (tiny dashes at print, read as glints); dotted shore foam; faint '
           'canvas micro-texture in sky and water'},
 {'pid': 'm5685677173773420545',
  'window': (1864, 520),
  'why': 'bald man holding a flower (drawing): clarity 3 of 5',
  'crops': {'area': 'striped background behind a smeared sprig',
            'hrdit': 'soft lotus petals',
            'prog': 'a fine, crisp floret spiral'},
  'audit': 'MINOR: 1:1 only: glossy glove-like hands (no countable finger error at print); spiky nipples; wavy '
           'lower chest contours; a target-like ring at the flower centre; sparkle flecks in the beard'},
 {'pid': '2155209427123413237.1',
  'window': (2400, 2480),
  'why': 'rock towers under the Milky Way: clarity 4 of 5',
  'crops': {'area': 'flat sky with a vertical seam stripe above a talus slope with a mesh lattice',
            'hrdit': 'a crisp Milky Way star field (different content)',
            'prog': 'crisp rock ledges, boulders, a grass tuft and sand ripples'},
  'audit': 'MINOR: 1:1 only: grid-locked 8/16 px mesh (0.8-1.6 levels) in the starry sky with smudgy faint stars; '
           'soft dark blobs in the Milky Way; a short streak reads as a meteor'},
 {'pid': 'm9219623328452982854',
  'window': (1760, 3424),
  'why': "climber on ice: clarity 3 of 5; AReA's rock wall is covered in multicoloured polka dots (speckle at its "
         'extreme)',
  'crops': {'area': 'a crisp boot with white snowflake dots (different content)',
            'hrdit': 'blurred teal ice and a rope',
            'prog': 'crisp granular three-dimensional snow clumps'},
  'audit': 'MINOR: 1:1 only: small pseudo-text on strap labels, patches, boot and axe (unreadable at print); white '
           'halos along the rope, beanie and profile; a blurred word-like smudge in the background'},
 {'pid': '9082507599514684465',
  'window': (2104, 2480),
  'why': 'illustration, three men: clarity 4 of 5',
  'crops': {'area': 'soft green robe fabric with a fine lattice',
            'hrdit': 'staff and robe hem before distant mountains, moderately soft',
            'prog': 'hazy but structured fields, hedgerows, tree rows and small white houses'},
  'audit': 'MINOR: 1:1 only: waxy highlight smears on robes; rubbery bark; 2-3 px bright line on the left border; '
           'a modern wristwatch on one man (also in UltraImage)'},
 {'pid': 'm4844341235185759356',
  'window': (1384, 2264),
  'why': 'lone hiker at sunset: clarity 4 of 5; AReA and HRDiT each have a tiny extra figure outside the window',
  'crops': {'area': 'lake reflection with blocky horizontal streaks and haze',
            'hrdit': 'slope and shoreline over flat dark water, soft',
            'prog': 'scree texture, small pine clusters, a crisp shoreline and turquoise shallows'},
  'audit': 'MINOR: 1:1 only: diagonal 16 px checker (1-2.6 levels) on dark foreground rocks; 2-3 px bright halo '
           'along ridgelines; tiny pseudo-text on the backpack badge; a faint pink disc reads as a cloud puff'},
 {'pid': '6422600338162192082',
  'window': (2304, 1440),
  'why': 'bald bearded man: clarity 3 of 5; advantage clear over UltraImage only',
  'crops': {'area': 'its own brow and eye with odd whitish lashes',
            'hrdit': 'crisp beard curls, an ear and a knit scarf',
            'prog': "dense crisp crow's-feet wrinkles and individual lashes"},
  'audit': 'CLEAN: anatomy correct, no pseudo-text; sub-visible steps of at most 1 level on 256 px grid lines'},
 {'pid': 'm1222532847814836069',
  'window': (1440, 3120),
  'why': 'gnarled tree under the Milky Way: clarity 3 of 5; dark crops; HRDiT equally crisp',
  'crops': {'area': 'thin branches against a sky with a fine crosshatch lattice',
            'hrdit': 'crisp rock with gravel and plants',
            'prog': 'forked roots with fine bark grain'},
  'audit': 'MINOR: from round 1: squirrel-like silhouette on a branch stub; faint light halos on dark edges; grain '
           'in the bright Milky Way; stylised roots (all 1:1)'}]

CAPTIONS = [
    (r"\textbf{Progressive \sca{} beside AReA and HRDiT at $4096^2$.} Sixteen Aesthetic-4K evaluation prompts. "
     r"FLUX.1-dev, seed~0. AReA: 28 steps at $4096^2$, guidance 4.5. HRDiT and progressive \sca{}: "
     r"HRDiT's schedule of 30 steps at $1024^2$, 17 at $2048^2$ and 10 at $4096^2$ (guidance 3.5, 4.5, and 6.0); "
     r"progressive \sca{} uses \sca{} attention in the last two stages. Each image is followed by a 1:1 crop of its outlined $%d\times%d$ window."),
    (r"\textbf{Progressive \sca{} beside AReA and HRDiT at $4096^2$, continued.} Prompts nine to sixteen; same "
     r"protocol as Figure~\ref{fig:gallery_prog16}."),
]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def tiles(job):
    pid, key, png, box, px = job
    img = imgsel.load(png)
    full = imgsel.save_jpeg(img, os.path.join(OUT, f"{pid}_{key}.jpg"), JPEG_Q, px)
    crop = imgsel.save_jpeg(imgsel.crop(img, box), os.path.join(OUT, f"{pid}_{key}_crop.jpg"), JPEG_Q, box[2])
    return (pid, key), (full, crop)


def main():
    os.makedirs(OUT, exist_ok=True)
    assert len(GALLERY) == 2 * PER_FLOAT, len(GALLERY)
    pids = [g["pid"] for g in GALLERY]
    assert len(set(pids)) == len(pids)
    assert not set(pids) & {s["pid"] for s in qp.SELECTED}, "a prompt of fig:qual_prog is reused"
    assert not set(pids) & set(qp.REJECTED), "a rejected prompt is shown"
    data = {k: qp.per_image(k, b, x, n) for k, _, b, x, n in METHODS}
    srcs, records = {}, []
    for g in GALLERY:
        pid, (bx, by) = g["pid"], g["window"]
        assert bx % 8 == 0 and by % 8 == 0 and 256 <= bx <= 4096 - 256 - WIN and 256 <= by <= 4096 - 256 - WIN, pid
        cfgs, rec = {}, {}
        for key, _, _, _, name in METHODS:
            e = data[key][pid]
            if key != "prog":
                assert e["ir_pct"] >= qp.FAIR_PCT, (pid, key, e["ir_pct"])
            meta = json.load(open(e["path"][:-4] + ".json"))
            if meta.get("complete") is not True or meta.get("png_sha256") != sha(e["path"]):
                raise ValueError(f"incomplete or hash-mismatched source {e['path']}")
            cfgs[key] = meta["config"]
            srcs[pid, key] = e["path"]
            rec[key] = dict(csv_method=name, source=e["path"], png_sha256=meta["png_sha256"],
                            metrics={k: e[k] for k in ("ir", "ir_pct", "musiq_c", "hpsv21")})
        assert len({c["prompt"] for c in cfgs.values()}) == 1 and len({str(c["derived_seed"]) for c in cfgs.values()}) == 1
        pc = cfgs["prog"]
        assert dict(method=pc["method"], steps=pc["steps"], attention=pc["attention"],
                    stages=pc["progressive"]["stages"], base_steps=pc["progressive"]["base_steps"],
                    highres_steps=pc["progressive"]["highres_steps"]) == qp.PROG, pid
        records.append(dict(prompt_id=pid, csv_prompt_id=data["prog"][pid]["csv_prompt_id"], prompt=pc["prompt"],
                            derived_seed=pc["derived_seed"], window_xys=[bx, by, WIN], why=g["why"],
                            crops=g.get("crops"), progressive_audit=g.get("audit"), methods=rec))
    geo = Geo(2 * len(METHODS), len(METHODS) * G_PAIR + (len(METHODS) - 1) * G_GROUP)
    px = geo.px()
    for old in os.listdir(OUT):
        if old.endswith(".jpg"):
            os.remove(os.path.join(OUT, old))
    jobs = [(g["pid"], k, srcs[g["pid"], k], (g["window"][0], g["window"][1], WIN), px) for g in GALLERY
            for k, *_ in METHODS]
    with Pool(12) as pool:
        t = dict(pool.map(tiles, jobs))
    head = []
    for k, (key, *_rest) in enumerate(METHODS):
        text, bold = HEADS[key]
        head += ([G_GROUP] if k else []) + [head_cell(text, 2, G_PAIR, accent=bold)]
    lines = [f"% Generated by {SCRIPT}; do not edit by hand.",
             f"% Prompts, windows and sources: figures/{NAME}/provenance.json."]
    lines += [f"% {r['prompt_id']}: {r['prompt'][:140].replace(chr(10), ' ')}" for r in records]
    lines += [r"\input{figures/snippets/_scagrid}%"]
    for f in range(2):
        part = GALLERY[f * PER_FLOAT:(f + 1) * PER_FLOAT]
        body = [(g_row(head), G_ROW)]
        for g in part:
            box, cells = (g["window"][0], g["window"][1], WIN), []
            for k, (key, *_rest) in enumerate(METHODS):
                full, crop = t[g["pid"], key]
                cells += ([G_GROUP] if k else []) + [t_full(full, box, RES), G_PAIR, t_crop(crop)]
            body.append((g_row(cells), G_ROW))
        cap = CAPTIONS[f] % ((WIN, WIN) if f == 0 else ())
        lines += [r"\begin{figure}[p]", r"\begingroup\scaGridStyle", geo.tex()]
        lines += [tx + (r"\par\vskip%gpt" % gap if i < len(body) - 1 else r"\par") for i, (tx, gap) in enumerate(body)]
        lines += [r"\endgroup", rf"\caption{{{cap}}}", rf"\label{{{LABELS[f]}}}", r"\end{figure}"]
    lines.append("")
    snippet = os.path.join(SNIP, f"{NAME}.tex")
    with open(snippet, "w") as fh:
        fh.write("\n".join(lines))
    outputs = {rel(p): sha(p) for v in t.values() for p in v}
    man = dict(figure=NAME, labels=LABELS, script=SCRIPT, curated=True, resolution=RES,
               methods=[k for k, *_ in METHODS], window_px=WIN,
               selection=dict(pool="as fig_qual_prog.py: 154 prompts", fairness=f"baselines ImageReward >= own p{qp.FAIR_PCT}",
                              window_rule="curated: by hand where the 1:1 crops show progressive SCA's advantage most "
                                          "clearly, free of any flaw of the progressive image",
                              not_reused="the prompts of fig:qual_prog (asserted)", rejected=qp.REJECTED),
               layout=dict(rows_per_float=PER_FLOAT, tile_in=round(geo.tile_in(), 4), full_image_px=px,
                           full_image_dpi=GRID_DPI, jpeg_quality=JPEG_Q, crop_ppi=round(WIN / geo.tile_in(), 1)),
               captions=CAPTIONS, prompts=records, outputs=outputs)
    with open(os.path.join(OUT, "provenance.json"), "w") as fh:
        fh.write(json.dumps(man, indent=2) + "\n")
    size = sum(os.path.getsize(p) for v in t.values() for p in v)
    print(f"wrote {snippet}; tile {geo.tile_in() * 72.27:.2f} pt; JPEGs {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
