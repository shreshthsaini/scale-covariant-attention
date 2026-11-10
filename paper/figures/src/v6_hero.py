"""Figure 1 of v6 (paper/figures/v6/hero.pdf): 4096^2 progressive SCA samples with native-pixel crops, and one panel
that carries the headline (ImageReward against clean seconds per 4096^2 image).

Images. Full 4096^2 progressive SCA outputs (sca_v2_prog: HRDiT's three-stage schedule, 1024^2 for 30 steps with
stock attention, the last 17 steps at 2048^2 and the last 10 at 4096^2 with SCA attention; seed 0; asserted from every
sidecar) of the 40 showcase prompts (web/build/showcase_prompts.jsonl; out/showcase/flux1_sca_v2_prog/4096x4096).

Selection rule (one clause in the caption): one prompt from each of the six categories of the showcase prompts,
drawn with seed 0 (random.Random("0:<category>").choice over the category's sorted ids). The large tile is the
portrait draw; the small tiles follow in the order animal, landscape, architecture, dense, product (layout A) or the
first three of that order (layout B). History: the first rule, one evaluation prompt per keyword category of
v6_common.CATEGORIES with seed 0, drew an image with an obvious defect at overview (prompt 4820545723762191815,
animal), so the rule moved to the showcase prompts before any of their images was viewed
(.agent/notes/paper_v6/visuals.md).

Crops. Native 1:1 windows of fixed size at the highest mean edge energy of the image (v6_common.best_windows; the
second window is the best one that does not overlap the first). Each window carries a 1 pt red box on the full image
and its crop a 1 pt red frame; a thin leader joins them.

Panel. Every value comes from fig_speed.load() (results/ CSVs, checked against the printed paper values and the
headline macros): ImageReward of the 195 evaluation prompts at seed 0 against clean batch-1 seconds per image on one
GH200, for the four methods with a clean time and an ImageReward inside one axis (progressive SCA, single-pass SCA,
HRDiT, AReA). Two green arrows give progressive SCA's speedup over AReA and HRDiT (clean over clean, the headline
macros \\hlSpeedProgArea and \\hlSpeedProgHrdit).

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I v6_hero.py [--layout A|B] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib  # noqa: E402

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from PIL import Image  # noqa: E402

import fig_speed  # noqa: E402
import scafig  # noqa: E402
import v6_common as vc  # noqa: E402

SHOWCASE = vc.REPO / "web" / "build" / "showcase_prompts.jsonl"
SC_DIR = Path("/scratch/09032/saini_2/sca/out/showcase/flux1_sca_v2_prog/4096x4096")
SEED = 0
ORDER = ["portrait", "animal", "landscape", "architecture", "dense", "product"]
PROG = dict(method="sca_v2_prog", steps=57, attention="stock@native+sca@highres", guidance=[3.5, 4.5, 6.0],
            stages=["1024x1024", "2048x2048", "4096x4096"], base_steps=30, highres_steps=[17, 10],
            model="black-forest-labs/FLUX.1-dev", seed=0, res="4096x4096")
W = vc.TEXT_PT
G = vc.G


# ------------------------------------------------------------------------------------------------ selection
def showcase() -> dict:
    rows = [json.loads(line) for line in SHOWCASE.read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def draw() -> dict:
    """{category: showcase id}, one per category, seed SEED."""
    sc = showcase()
    out = {}
    for c in ORDER:
        ids = sorted(i for i, r in sc.items() if r["category"] == c)
        out[c] = random.Random(f"{SEED}:{c}").choice(ids)
    return out


def source(sid: str) -> tuple[Path, dict]:
    pngs = sorted(SC_DIR.glob(f"{sid}_*_seed0.png"))
    assert len(pngs) == 1, (sid, pngs)
    meta = vc.sidecar(pngs[0])
    c = meta["config"]
    got = dict(method=c["method"], steps=c["steps"], attention=c["attention"], guidance=c["guidance"],
               stages=c["progressive"]["stages"], base_steps=c["progressive"]["base_steps"],
               highres_steps=c["progressive"]["highres_steps"], model=c["model"], seed=c["seed"], res=c["res"])
    if got != PROG or c["prompt_id"] != sid or c["prompt"] != showcase()[sid]["prompt"]:
        raise ValueError(f"{pngs[0]}: {got}")
    return pngs[0], meta


# ------------------------------------------------------------------------------------------------ panel
def panel(pdf: Path, w_pt: float, h_pt: float, fs: float = 7.5) -> dict:
    """ImageReward against clean seconds per 4096^2 image; returns the plotted values."""
    d = fig_speed.load()
    scafig.use_style()
    scafig.script_floor(fs - 1.0)
    pts = {m: (d["clean"][m], d["ir"][m]) for m in ("sca_v2_prog", "sca_v2", "hrdit_dev_author", "area")}
    fig = plt.figure(figsize=(w_pt / 72.27, h_pt / 72.27))
    left, right, bottom, top = 27.0, 6.0, 25.0, 4.0  # pt margins inside the cell
    ax = fig.add_axes([left / w_pt, bottom / h_pt, 1 - (left + right) / w_pt, 1 - (bottom + top) / h_pt])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=fs - 0.5, length=2.5, width=0.6, pad=1.5)
    scafig.value_grid(ax, "y")
    ax.set_xlim(0, 225)
    ax.set_ylim(0.84, 1.45)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_yticks([0.9, 1.1, 1.3])
    ax.set_xlabel("Seconds per 4096\u00b2 image $\\downarrow$", fontsize=fs, labelpad=2)
    ax.set_ylabel(r"ImageReward $\uparrow$", fontsize=fs, labelpad=2)
    col = {"sca_v2_prog": scafig.SCA, "sca_v2": scafig.SCA, "hrdit_dev_author": "#56626F", "area": scafig.AREA}
    mk = {"sca_v2_prog": ("*", 9.5), "sca_v2": ("o", 4.6), "hrdit_dev_author": ("D", 4.2), "area": ("o", 4.6)}
    for m, (x, y) in pts.items():
        ax.plot([x], [y], marker=mk[m][0], ms=mk[m][1], color=col[m], mec="white", mew=0.5, ls="none", zorder=5)
    xp, yp = pts["sca_v2_prog"]

    def arrow(m, rad):
        x, y = pts[m]
        a = FancyArrowPatch((x, y), (xp, yp), transform=ax.transData, arrowstyle="-|>,head_length=3.2,head_width=1.9",
                            connectionstyle=f"arc3,rad={rad}", color=scafig.SCA, lw=1.1, shrinkA=4.5, shrinkB=6.5,
                            zorder=4)
        ax.add_patch(a)

    arrow("area", -0.32)
    arrow("hrdit_dev_author", 0.0)
    sp = d["speed"]
    lab = dict(fontsize=fs, va="center", textcoords="offset points")

    def note(text, m, dx, dy, ha="left", color=scafig.SCA, bold=False, at=None):
        ax.annotate(text, at or pts[m], xytext=(dx, dy), ha=ha, color=color, fontweight="bold" if bold else None, **lab)

    # Offsets in points from the marker (the upper-left cluster holds three markers within 35 pt).
    note("Progressive SCA", "sca_v2_prog", -9, 14.5, bold=True)
    note("Single-pass SCA", "sca_v2", 5.5, -0.5)
    note("HRDiT", "hrdit_dev_author", 5.5, -3.0, color="#56626F")
    note("AReA", "area", 0, -8, ha="center", color=scafig.AREA)
    (x0, y0), (x1, y1) = pts["hrdit_dev_author"], pts["sca_v2_prog"]
    note(f"{sp['hrdit_dev_author'][0]:.1f}\u00d7", None, 0, -7.5, ha="center", bold=True, at=((x0 + x1) / 2, (y0 + y1) / 2))
    (x0, y0) = pts["area"]
    note(f"{sp['area'][0]:.1f}\u00d7 faster", None, -6, -12, bold=True, at=((x0 + x1) / 2, (y0 + y1) / 2))
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return dict(points={m: dict(seconds_clean=v[0], ir=v[1]) for m, v in pts.items()},
                speedup={m: dict(ratio=sp[m][0], kind=sp[m][1]) for m in ("area", "hrdit_dev_author")})


# ------------------------------------------------------------------------------------------------ layout
def layout(name: str) -> dict:
    """Cells in pt: {role: (x, y, w, h)} and the height. Roles: A (large), a1, a2 (crops of A), plot, B..F small."""
    if name == "A":  # five columns, three rows
        s = (W - 4 * G) / 5
        L = 2 * s + G
        col = [i * (s + G) for i in range(5)]
        row = [i * (s + G) for i in range(3)]
        cells = {"A": (0, 0, L, L), "a1": (col[2], 0, s, s), "a2": (col[2], row[1], s, s),
                 "plot": (col[3], 0, L, L)}
        for k, r in zip("BCDEF", range(5)):
            cells[k] = (col[r], row[2], s, s)
        return dict(cells=cells, height=3 * s + 2 * G, crop_px=256, small=list("BCDEF"))
    if name == "B":  # six columns, three rows: two large images, each with two crops beneath
        s = (W - 5 * G) / 6
        L = 2 * s + G
        col = [i * (s + G) for i in range(6)]
        cells = {"A": (0, 0, L, L), "B": (col[2], 0, L, L), "plot": (col[4], 0, L, L)}
        r2 = L + G
        for k, c in zip(["a1", "a2", "b1", "b2", "C", "D"], range(6)):
            cells[k] = (col[c], r2, s, s)
        return dict(cells=cells, height=r2 + s, crop_px=224, small=list("CD"))
    raise ValueError(name)


def _load(p):
    return Image.open(p).convert("RGB")


def build(layout_name: str, out_dir: Path | None) -> dict:
    lay = layout(layout_name)
    cells, H = lay["cells"], lay["height"]
    if out_dir:
        assets, pdf = out_dir / f"hero_{layout_name}", out_dir / f"hero_{layout_name}.pdf"
    else:
        assets, pdf = vc.ASSETS / "hero", vc.V6 / "hero.pdf"
    picks = draw()
    roles = {"A": picks["portrait"]}
    if layout_name == "A":
        roles.update(zip(lay["small"], [picks[c] for c in ORDER[1:]]))
    else:
        roles["B"] = picks["animal"]
        roles.update(zip(lay["small"], [picks[c] for c in ORDER[2:4]]))
    srcs = {k: source(v) for k, v in roles.items()}
    with Pool(len(srcs)) as pool:
        imgs = dict(zip(srcs, pool.map(_load, [str(p) for p, _ in srcs.values()])))
    for im in imgs.values():
        assert im.size == (4096, 4096)
    cv = vc.Canvas(W, H, assets)
    rec = dict(images={}, crops={})
    # full images
    for k, im in imgs.items():
        x, y, w, h = cells[k]
        name = f"{roles[k]}_full.jpg"
        vc.save_jpeg(vc.thumb(im, w, h), assets / name, vc.THUMB_Q)
        cv.image(x, y, w, h, name)
        rec["images"][k] = dict(id=roles[k], category=showcase()[roles[k]]["category"], source=str(srcs[k][0]),
                                png_sha256=srcs[k][1]["png_sha256"], cell_pt=[round(v, 3) for v in cells[k]],
                                prompt=srcs[k][1]["config"]["prompt"], derived_seed=srcs[k][1]["config"]["derived_seed"])
    # crops: windows by edge energy on the image itself
    cp = lay["crop_px"]
    crop_owner = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    owners = sorted({crop_owner[c] for c in crop_owner if c in cells})
    for o in owners:
        mine = [c for c in ("a1", "a2", "b1", "b2") if c in cells and crop_owner[c] == o]
        cw, ch = cells[mine[0]][2], cells[mine[0]][3]
        wpx, hpx = cp, round(cp * ch / cw)
        wins = vc.best_windows([vc.detail_map(imgs[o])], wpx, hpx, k=len(mine))
        # order the crops left to right / top to bottom by window position (stable, rule-based)
        wins = sorted(wins, key=lambda b: (b[1], b[0])) if layout_name == "A" else sorted(wins, key=lambda b: b[0])
        ox, oy, ow, oh = cells[o]
        sc = ow / 4096
        for c, (bx, by, bw, bh) in zip(mine, wins):
            name = f"{roles[o]}_crop_{bx}_{by}.jpg"
            vc.save_jpeg(imgs[o].crop((bx, by, bx + bw, by + bh)), assets / name, vc.CROP_Q)
            x, y, w, h = cells[c]
            cv.image(x, y, w, h, name)
            cv.box(x, y, x + w, y + h, inside=True)
            cv.text(x + 3, y + 3, "1:1", size=vc.FS_TAG, chip=True)
            wx0, wy0, wx1, wy1 = ox + bx * sc, oy + by * sc, ox + (bx + bw) * sc, oy + (by + bh) * sc
            cv.box(wx0, wy0, wx1, wy1, inside=False)
            rec["crops"][c] = dict(of=o, window_px=[bx, by, bw, bh], tile_pt=[round(v, 3) for v in cells[c]],
                                   tile_ppi=round(bw / (w / 72.27), 1), magnification=round((w / bw) / sc, 2))
            # leader: from the window's facing edge to the crop tile's facing edge
            if layout_name == "A":
                cv.line([(wx1 + 0.5, (wy0 + wy1) / 2), (x - 0.5, y + h / 2)], lw=0.5, opacity=0.8)
            else:
                cv.line([((wx0 + wx1) / 2, wy1 + 0.5), (x + w / 2, y - 0.5)], lw=0.5, opacity=0.8)
    # resolution tag on the large image(s)
    for k in ("A", "B"):
        if k in cells and k in imgs:
            x, y, w, h = cells[k]
            cv.text(x + 3, y + 3, r"4096\,\texttimes\,4096", size=vc.FS_TAG, chip=True)
            break
    # panel
    px, py, pw, ph = cells["plot"]
    pinfo = panel(assets / "panel.pdf", pw, ph)
    cv.image(px, py, pw, ph, "panel.pdf")
    cv.compile(pdf, "hero")
    rec.update(figure="Figure 1 (v6 hero)", layout=layout_name, pdf=str(pdf), pdf_bytes=pdf.stat().st_size,
               size_pt=[W, round(H, 3)], size_in=[round(W / 72.27, 3), round(H / 72.27, 3)],
               selection=dict(rule=("one showcase prompt per category, random.Random('0:<category>').choice over the "
                                    "category's sorted ids"), seed=SEED, draw=picks,
                              prompts_file=str(SHOWCASE.relative_to(vc.REPO)), prompts_sha256=vc.sha256(SHOWCASE),
                              history=("rule 1 (one evaluation prompt per keyword category, seed 0) drew "
                                       "4820545723762191815 (animal), an image with an obvious defect at overview; "
                                       "replaced by this rule before any showcase image was viewed")),
               crop_rule="native windows of fixed size at the highest mean edge energy of the image, non-overlapping",
               panel=pinfo, jpeg=dict(full=f"Lanczos to {vc.THUMB_DPI} ppi, q {vc.THUMB_Q}, 4:4:4",
                                      crop=f"native pixels, q {vc.CROP_Q}, 4:4:4"))
    prov = (out_dir / f"hero_{layout_name}_provenance.json") if out_dir else vc.V6 / "hero_provenance.json"
    rec["assets"] = str(assets)
    prov.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"{pdf}: {pdf.stat().st_size / 1e6:.2f} MB, {W / 72.27:.2f} x {H / 72.27:.2f} in")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="A", choices=["A", "B"])
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    build(a.layout, a.out_dir)


if __name__ == "__main__":
    main()
