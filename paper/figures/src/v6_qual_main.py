"""Figure 5 of v6 (paper/figures/v6/qual_main.pdf): like-for-like 1:1 crops of AReA, UltraImage, HRDiT and progressive
SCA at 4096^2, one prompt per row.

Each row: progressive SCA's full image with a red 1 pt box on a shared 512 x 512 window, then that window cut at
native pixels from AReA, UltraImage, HRDiT and progressive SCA, methods as column heads. Under each crop sits the MUSIQ
of that crop (pyiqa musiq, the KonIQ checkpoint of the evaluation; best of the row bold).

Prompt rule (one clause in the caption): the evaluation prompt whose four images agree most in layout, in each of the
portrait, landscape and architecture categories (keyword rule v6_common.CATEGORIES). Layout agreement is the smallest
pairwise SSIM among the four methods' 256 x 256 greyscale thumbnails (BOX downsampling, seed 0); development prompts
are excluded. The four-way rule is used, not the three-way fallback without AReA: the binding pair is usually HRDiT
against progressive SCA, and the four-way and three-way rankings share their top prompts (visuals.md).

Window rule: the 512 x 512 window that maximises the smallest mean edge energy among the four methods (maximin,
v6_common.best_windows_maximin), at least 1/16 of the side from every border, so every crop has content.

History: the first v6 version (methods as rows, each method's own full image beside a shared window chosen by the mean
edge energy) was rejected in review because the methods' layouts differ and the shared window showed unrelated content.

Images: the PNGs scored by eval/prog195_vs_baselines (seed 0, 4096^2); complete sidecars with matching hashes; same
prompt and derived seed per row (asserted).

Run on compute from paper/figures/src:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I v6_qual_main.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import imgsel  # noqa: E402
import v6_common as vc  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SEED = 0  # images at seed 0; the prompt rule itself draws nothing
CATS = ["portrait", "landscape", "architecture"]
METHODS = ["area", "ultraimage", "hrdit_dev_author", "sca_v2_prog"]
HEAD = {"area": "AReA", "ultraimage": "UltraImage", "hrdit_dev_author": "HRDiT", "sca_v2_prog": "Progressive SCA"}
WIN = 512
TORCH_HOME = "/scratch/09032/saini_2/tmp/torch"  # holds pyiqa's musiq_koniq checkpoint (the evaluation's model)


STRIDE, MARGIN, DOWN = 256, 256, 4     # window grid (px); border margin (1/16 of the side); SSIM crops 128 x 128
DETAIL_Q = 0.90                         # keep windows whose maximin edge energy is in the top decile of all candidates


def _candidates(job):
    """All grid windows of one prompt: (x, y, maximin edge energy, smallest pairwise SSIM of the 128 px crops)."""
    from skimage.metrics import structural_similarity as ssim
    sid, paths4 = job
    maps, grays = [], []
    for p in paths4:
        im = Image.open(p).convert("RGB")
        maps.append(vc.detail_map(im))
        g = im.convert("L").resize((im.width // DOWN, im.height // DOWN), Image.Resampling.BOX)
        grays.append(np.asarray(g, dtype=np.float64) / 255.0)
    pos = range(MARGIN, 4096 - WIN - MARGIN + 1, STRIDE)
    out = []
    for y in pos:
        for x in pos:
            e = min(float(a[y // 8:(y + WIN) // 8, x // 8:(x + WIN) // 8].mean()) for a in maps)
            c = [g[y // DOWN:(y + WIN) // DOWN, x // DOWN:(x + WIN) // DOWN] for g in grays]
            agree = min(ssim(a, b, data_range=1.0) for a, b in itertools.combinations(c, 2))
            out.append((x, y, e, agree))
    return sid, out


def candidates(pool) -> list:
    """[(sid, x, y, maximin energy, local agreement)] for every prompt and grid window (195 x 169)."""
    ids = sorted(pool["sca_v2_prog"])
    with Pool(64) as mp:
        res = mp.map(_candidates, [(s, [pool[m][s]["path"] for m in METHODS]) for s in ids])
    return [(s, x, y, e, a) for s, lst in res for (x, y, e, a) in lst]


def picks_from(pool, texts, cands=None):
    """(info, {category: id}, {category: window}): per category, the (prompt, 512 x 512 window) pair on which the four
    methods agree most in content (smallest pairwise SSIM of 128 x 128 greyscale crops), among windows whose maximin
    edge energy is in the top decile of all candidates (top quintile if a category keeps fewer than five)."""
    cands = cands or candidates(pool)
    dev = imgsel.dev4_ids()
    energies = np.array([c[3] for c in cands])
    for q, name in ((DETAIL_Q, "top decile"), (0.80, "top quintile")):
        cut = float(np.quantile(energies, q))
        hi = [c for c in cands if c[3] >= cut and c[0] not in dev]
        per = {k: [c for c in hi if vc.category(texts[c[0]]) == k] for k in CATS}
        if min(len(v) for v in per.values()) >= 5:
            break
    picks, wins, top5 = {}, {}, {}
    for k in CATS:
        ranked = sorted(per[k], key=lambda c: (-c[4], -c[3], c[0], c[1], c[2]))
        best = ranked[0]
        picks[k], wins[k] = best[0], (best[1], best[2], WIN, WIN)
        seen, top = set(), []
        for c in ranked:                      # top five distinct prompts of the category, for the review sheet
            if c[0] not in seen:
                seen.add(c[0]); top.append(dict(id=c[0], window=[c[1], c[2]], agreement=round(c[4], 4),
                                                energy=round(c[3], 3)))
            if len(top) == 5:
                break
        top5[k] = top
    info = dict(detail_filter=name, energy_cut=round(cut, 4), n_candidates=len(cands), n_high_detail=len(hi),
                per_category=({k: len(v) for k, v in per.items()}), top5=top5,
                picked={k: dict(id=picks[k], window=list(wins[k][:2]),
                                agreement=round(next(c[4] for c in per[k] if c[0] == picks[k] and
                                                     (c[1], c[2]) == wins[k][:2]), 4)) for k in CATS})
    return info, picks, wins


def selection():
    pool = vc.pool195()
    texts = vc.prompts195(pool)
    info, picks, wins = picks_from(pool, texts)
    return pool, texts, info, picks, wins


def _edge(p):
    im = Image.open(p).convert("RGB")
    return vc.detail_map(im)


def musiq_scores(crops: list) -> list:
    """MUSIQ (pyiqa 'musiq', KonIQ checkpoint) of each crop, on CPU."""
    os.environ.setdefault("TORCH_HOME", TORCH_HOME)
    import pyiqa
    import torch
    torch.set_num_threads(32)
    model = pyiqa.create_metric("musiq", device="cpu").eval()
    out = []
    with torch.no_grad():
        for im in crops:
            x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
            out.append(float(model(x).item()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    pool, texts, info, picks, wins_by_cat = selection()
    rows = [picks[c] for c in CATS]
    paths = {s: {m: pool[m][s]["path"] for m in METHODS} for s in rows}
    meta = {s: vc.same_prompt_and_noise(paths[s]) for s in rows}            # verifies every sidecar hash
    sha = {s: {m: json.loads(Path(p).with_suffix(".json").read_text())["png_sha256"] for m, p in paths[s].items()}
           for s in rows}
    wins = {picks[c]: wins_by_cat[c] for c in CATS}
    crops = {}
    for s in rows:
        x, y, w, h = wins[s]
        for m in METHODS:
            crops[(s, m)] = Image.open(paths[s][m]).convert("RGB").crop((x, y, x + w, y + h))
    scores = dict(zip(crops, musiq_scores(list(crops.values()))))

    out = a.out_dir or vc.V6
    assets = (a.out_dir or vc.ASSETS) / "qual_main"
    pdf = out / "qual_main.pdf"
    W, G = vc.TEXT_PT, vc.G
    SEP = 6.0                                   # gap between the full image and the crops
    tile = (W - SEP - 3 * G) / 5                # five square tiles across the text width
    head_h, num_h, row_gap = 19.0, 9.0, 4.0     # column heads with a seconds line; MUSIQ line under each row
    H = head_h + len(rows) * (tile + num_h) + (len(rows) - 1) * row_gap
    cv = vc.Canvas(W, H, assets)
    xs = [0.0] + [tile + SEP + k * (tile + G) for k in range(4)]
    # Seconds per 4096^2 image from the hero panel's data (fig_speed.load): clean batch-1 medians on one GH200;
    # UltraImage has only a campaign mean, marked with Table 1's double dagger.
    import fig_speed
    d = fig_speed.load()
    r1 = fig_speed.r1  # half-up to 0.1 s, as the paper prints (49.15 s -> 49.2 s)
    secs = {"area": "%.1f\\,s" % r1(d["clean"]["area"]), "hrdit_dev_author": "%.1f\\,s" % r1(d["clean"]["hrdit_dev_author"]),
            "sca_v2_prog": "%.1f\\,s" % r1(d["clean"]["sca_v2_prog"]),
            "ultraimage": "%.1f\\,s\\textdaggerdbl" % r1(d["camp"]["ultraimage"])}
    cv.text(xs[0] + tile / 2, head_h - 11.0, "Image", anchor="base")
    for k, m in enumerate(METHODS):
        cv.text(xs[k + 1] + tile / 2, head_h - 11.0, HEAD[m], anchor="base",
                color=vc.SCA if m == "sca_v2_prog" else vc.INK, bold=(m == "sca_v2_prog"))
        cv.text(xs[k + 1] + tile / 2, head_h - 3.0, secs[m], size=6.5, anchor="base", color=vc.MID)
    rec = dict(rows=[])
    for i, s in enumerate(rows):
        y = head_h + i * (tile + num_h + row_gap)
        full = Image.open(paths[s]["sca_v2_prog"]).convert("RGB")
        tn = f"{s}_sca_v2_prog_full.jpg"
        vc.save_jpeg(vc.thumb(full, tile, tile), assets / tn, vc.THUMB_Q)
        cv.image(xs[0], y, tile, tile, tn)
        bx, by, bw, bh = wins[s]
        sc = tile / 4096
        cv.box(xs[0] + bx * sc, y + by * sc, xs[0] + (bx + bw) * sc, y + (by + bh) * sc, inside=False)
        best = max(METHODS, key=lambda m: scores[(s, m)])
        for k, m in enumerate(METHODS):
            cn = f"{s}_{m}_crop_{bx}_{by}.jpg"
            vc.save_jpeg(crops[(s, m)], assets / cn, vc.CROP_Q)
            cv.image(xs[k + 1], y, tile, tile, cn)
            cv.box(xs[k + 1], y, xs[k + 1] + tile, y + tile, inside=True)
            cv.text(xs[k + 1] + tile / 2, y + tile + num_h - 2.2, "%.1f" % scores[(s, m)], size=7.0, anchor="base",
                    color=vc.INK, bold=(m == best))
        rec["rows"].append(dict(
            key=s, category=vc.category(texts[s]),
            local_agreement_min_ssim=info["picked"][vc.category(texts[s])]["agreement"],
            window_px=list(wins[s]), crop_ppi=round(WIN / (tile / 72.27), 1), prompt=meta[s]["prompt"],
            prompt_id=meta[s]["prompt_id"], derived_seed=meta[s]["derived_seed"],
            musiq_crop={m: round(scores[(s, m)], 2) for m in METHODS}, best=best,
            musiq_c_image={m: pool[m][s]["musiq_c"] for m in METHODS},
            sources={m: paths[s][m] for m in METHODS}, sources_sha256=sha[s]))
    cv.compile(pdf, "qual_main")
    rec.update(figure="Figure 5 (v6 qual_main)", pdf=str(pdf), pdf_bytes=pdf.stat().st_size,
               size_in=[round(W / 72.27, 3), round(H / 72.27, 3)], tile_pt=round(tile, 2), assets=str(assets),
               selection=dict(rule=("per category (portrait, landscape, architecture; keyword rule), the evaluation "
                                    "prompt and 512 x 512 window (grid stride 256, 256 px border) on which the four "
                                    "methods agree most in content (smallest pairwise SSIM of 128 x 128 greyscale "
                                    "crops) among high-detail windows (maximin edge energy in the top decile of all "
                                    "candidates); development prompts excluded"), **info),
               seconds=dict(clean={m: d["clean"][m] for m in ("area", "hrdit_dev_author", "sca_v2_prog")},
                            campaign_ultraimage=d["camp"]["ultraimage"]),
               numbers="MUSIQ of each 512 x 512 crop, pyiqa musiq (KonIQ checkpoint), CPU; best of the row bold")
    prov = (a.out_dir / "qual_main_provenance.json") if a.out_dir else vc.V6 / "qual_main_provenance.json"
    prov.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"{pdf}: {pdf.stat().st_size / 1e6:.2f} MB, {W / 72.27:.2f} x {H / 72.27:.2f} in; rows {rows}")
    for r in rec["rows"]:
        print(r["key"], r["category"], r["local_agreement_min_ssim"], r["window_px"], r["musiq_crop"], "best", r["best"])
    print("detail filter:", info["detail_filter"], "cut", info["energy_cut"], "high-detail", info["n_high_detail"],
          "per category", info["per_category"])
    if a.out_dir:  # review sheet: the top five prompts per category, their best window cropped from the four methods
        sheet = []
        for c in CATS:
            for t in info["top5"][c]:
                x, y = t["window"]
                sheet.append([Image.open(pool[m][t["id"]]["path"]).convert("RGB").crop((x, y, x + WIN, y + WIN))
                              .resize((192, 192), Image.Resampling.LANCZOS) for m in METHODS])
        im = Image.new("RGB", (4 * 196, len(sheet) * 196), "white")
        for r_, row in enumerate(sheet):
            for c_, t in enumerate(row):
                im.paste(t, (c_ * 196, r_ * 196))
        im.save(a.out_dir / "top5_pairs.jpg", quality=88)
        (a.out_dir / "top5_pairs.json").write_text(json.dumps(info["top5"], indent=1) + "\n")


if __name__ == "__main__":
    main()
