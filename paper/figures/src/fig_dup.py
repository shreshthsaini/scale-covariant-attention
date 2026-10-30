"""Late duplication at 4096^2: the gate sweep, the rates of every method, and uncurated before/after pairs.

(a) Gate sweep on the 46-prompt ablation set (abl50 minus dev4; 26 prompts name a person or animal with a
    count): share of those 26 prompts whose image shows more copies of the subject than the prompt implies,
    for published SCA (hard switch from the global-only opening to native windows) with the gate at
    sigma_g = 0.93, 0.89 (published), 0.85 and global only at every step. sca_v2 (final operator: windows of
    128, 112, 96, 80 tokens on the four steps after the gate, weights read at the matching ratio, 1,024 bridge
    tokens; 46-prompt run label dup_i_shrink) is the green dot on the published gate's row, AReA the purple
    line. The one annotation is the quality cost of gate 0.85 against gate 0.89: the number of the ten
    per-image metrics whose paired 95% bootstrap interval lies below zero (results/e7_sca_g85_deltas.csv,
    same 46 prompts).
(b) The same rate on the 95 people/animal prompts of the 195-prompt Aesthetic-4K set at 4096^2 for every
    method, with Wilson 95% intervals (results/dup_audit_20261027/rates_a4k195_people.csv). sca_v2's row is
    read from results/a4k195_main_sca_v2/dup_rates_people.csv, or from the eval directory once its DONE marker
    exists; until then it is drawn as an empty outline labelled "pending" and carries no number.
(c) Uncurated pairs, same prompt and seed (identical derived seed checked in the sidecars), published SCA
    (main/sca_nb) and sca_v2 (dupfix/sca_v2, else the identical-flag 46-prompt run dup46/dup_i_shrink).
    Prompt rule, fixed before any sca_v2 image was opened: start at the ice-climber prompt
    a4k_eval_4096_997293625843621481; walk the 95 people/animal prompt ids of per_image_a4k195_people.csv in
    Python string order (wrapping at the end) and take the next prompts the detector flags for published SCA,
    N_PAIRS in all; sca_v2 images are shown whatever they show. Box rule: the detector's kept boxes for the
    subject with the largest excess count (dup_rate.py count_boxes, threshold 0.35), minus the `implied`
    largest (the intended subjects), are the extra copies; the box is the CROP_PX square that holds the most
    extra copies (ties: higher summed score), centred on their union and clamped to the image. The same box is
    drawn on both images and cropped from both.

Every number comes from the CSVs named above; nothing is computed except counts, shares already in them, and
the opening length in steps (sigmas > sigma_g in the image sidecars), which goes to the provenance only.

Regenerate: <sca env python> paper/figures/src/fig_dup.py
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-identical PDF on rerun
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scafig as sf  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

ROOT = HERE.parents[2]
AUDIT = ROOT / "results/dup_audit_20261027"
RATES46 = AUDIT / "rates_46_all_people.csv"      # sca_nb, area, dup_i_shrink (= sca_v2) on the 46-prompt set
RATESABL = AUDIT / "rates_abl46_people.csv"      # gate sweep and ablations on the same 46 prompts
RATES195 = AUDIT / "rates_a4k195_people.csv"     # every method, 195 prompts (95 people/animal)
PERIMG = AUDIT / "per_image_a4k195_people.csv"
SUBJECTS = AUDIT / "subjects_a4k195.json"
E7_G85 = ROOT / "results/e7_sca_g85_deltas.csv"  # gate 0.85 minus gate 0.89, 46 prompts, paired bootstrap
DUP_RATE = ROOT / "scripts/dup_rate.py"          # the detector's own counting rule
V2_RESULT = ROOT / "results/a4k195_main_sca_v2/dup_rates_people.csv"
V2_EVAL = Path("/scratch/09032/saini_2/sca/eval/a4k195_main_sca_v2")
# Progressive SCA with the final operator (sca_v2_prog); the published sca_prog bar is used until this file exists.
PROG_V2_RESULT = ROOT / "results/prog195_sca_v2/dup_rates_people.csv"
DETS = Path("/scratch/09032/saini_2/sca/eval/dup_detect_20261027")
OUTROOT = Path("/scratch/09032/saini_2/sca/out")
PUB_DIR = OUTROOT / "main/sca_nb/4096x4096"
V2_DIRS = [OUTROOT / "dupfix/sca_v2/4096x4096", OUTROOT / "dup46/dup_i_shrink/4096x4096"]
# Detector runs that may hold sca_v2 boxes: the z980 run (195 prompts) and the 46-prompt run (identical images).
V2_DETS = [("sca_v2", V2_EVAL / "det"), ("dup_i_shrink", Path("/scratch/09032/saini_2/sca/eval/dup46_dup_i_shrink/det"))]
GATE_DIRS = {"sca_g93": OUTROOT / "gate/sca_g93/4096x4096", "sca_nb": PUB_DIR,
             "sca_g85": OUTROOT / "gate/sca_g85/4096x4096", "abl_globalonly": OUTROOT / "abl/abl_globalonly/4096x4096"}
OUT = ROOT / "paper/figures/dup"                 # dup.pdf, dup_provenance.json; tiles in dup/
TILES = ROOT / "paper/figures/dup"
RULE_TEX = ROOT / "paper/figures/snippets/dup_examples_rule.tex"  # the selection rule as a caption/text sentence

START = "a4k_eval_4096_997293625843621481"       # the ice climber
N_PAIRS = 3
# Visual check of the selected sca_v2 images (2026-10-29, after the selection rule was fixed): the ice climber
# and the bear are clean (the bear's identical 46-prompt image scores bear 1/1); the astronaut keeps one small
# copy sitting on the rock right of the main one, where published SCA also drew one (box [2746, 3127, 2867, 3434]).
# The caption says so; the detector verdict on the 195-prompt images lands with z980 and goes to the provenance.
OBSERVED = {"a4k_eval_4096_minus1071233499893339092": "one astronaut copy remains"}
CROP_PX = 512                                    # one eighth of the 4096 px side
SIDE = 4096
THR, SIDES, NMS, CONTAIN, CONTAIN_AREA, MAX_FRAC = 0.35, {800, 1333}, 0.5, 0.9, 0.3, 0.9  # dup_rate.py defaults
V2_SCA = {"gate_sigma": 0.89, "win_sched": [[0.88, 128], [0.86, 112], [0.84, 96], [0.82, 80]], "bridge_cells": 8,
          "bridge_k": 16, "win_pi_match": True, "win_pi_interp": True, "window": 64, "pi_variant": "head"}

# Panel (a): rows top to bottom, later gate = longer global-only opening.
GATES = [("sca_g93", "0.93", 0.93), ("sca_nb", "0.89", 0.89), ("sca_g85", "0.85", 0.85),
         ("abl_globalonly", "global\nonly", None)]
# Panel (b): published SCA first, the baselines by rate, ours last.
METHODS = {"sca_nb": "SCA w/o schedule", "sigma": "SigMa", "dype": "DyPE", "sega": "SEGA", "hrdit": "HRDiT",
           "sca_prog": "Progressive SCA", "full": "Full attention", "ultraimage": "UltraImage", "area": "AReA",
           "sca_v2": "SCA (ours)"}

W, H = sf.TEXT, 1.98
RED = "#E41A1C"        # duplicated-region boxes and crop frames (one frame colour)
GREY = "#A3ABB3"       # every baseline bar
GREY_DOT = "#6F7C89"   # the gate-sweep series
BOX_LW = 0.6
DPI_PRINT = 300


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    path = Path(path)
    return str(path.relative_to(ROOT)) if ROOT in path.parents else str(path)


def read_rates(path: Path) -> dict:
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["method"]] = dict(method=r["method"], n=int(r["n"]), dup=int(r["dup"]), rate=float(r["rate"]),
                                lo95=float(r["lo95"]), hi95=float(r["hi95"]))
    return out


# ----------------------------------------------------------------------------- data
def gate_data(read: set) -> dict:
    abl, all46 = read_rates(RATESABL), read_rates(RATES46)
    read.update({RATESABL, RATES46})
    assert abl["sca_nb"] == all46["sca_nb"], "the two 46-prompt tables disagree on published SCA"
    rows = []
    for method, tick, gate in GATES:
        r = dict(abl[method])
        assert r["n"] == 26, f"{method}: expected the 26 people/animal prompts of the 46-prompt set"
        sig = set()
        for js in sorted(GATE_DIRS[method].glob("*.json")):
            meta = json.loads(js.read_text())
            sig.add(tuple(meta["sigmas"]))
            g = meta["config"]["sca"]["gate_sigma"]
            if gate is not None:
                assert abs(g - gate) < 1e-9, f"{js}: gate {g} != {gate}"
            else:
                assert meta["config"]["sca"]["pi_variant"] == "const" and meta["config"]["sca"]["pi_const"] == 0.0
        assert len(sig) == 1, f"{method}: more than one sigma schedule"
        sigmas = list(sig.pop())
        r.update(tick=tick.replace("\n", " "), gate_sigma=gate, steps=len(sigmas),
                 opening_steps=len(sigmas) if gate is None else sum(s > gate for s in sigmas),
                 image_dir=str(GATE_DIRS[method]))
        rows.append(r)
    v2, area = dict(all46["dup_i_shrink"]), dict(all46["area"])
    assert v2["n"] == 26 and area["n"] == 26
    # Quality cost of the later gate: per-image metrics significantly worse than gate 0.89.
    e7 = list(csv.DictReader(open(E7_G85)))
    read.add(E7_G85)
    assert {r["method"] for r in e7} == {"sca_g85"} and {r["reference"] for r in e7} == {"sca_nb"}
    assert {int(r["n_prompts"]) for r in e7} == {46}
    worse = sorted(r["metric"] for r in e7 if float(r["ci_high"]) < 0)
    better = sorted(r["metric"] for r in e7 if float(r["ci_low"]) > 0)
    assert not better, f"gate 0.85 is better on {better}: the annotation would mislead"
    cost = dict(source=rel(E7_G85), n_metrics=len(e7), worse=worse, n_worse=len(worse),
                rule="paired 95% bootstrap interval (gate 0.85 minus gate 0.89) entirely below zero",
                deltas={r["metric"]: [float(r["delta"]), float(r["ci_low"]), float(r["ci_high"])] for r in e7})
    return dict(rows=rows, v2=v2, area=area, cost=cost)


def v2_rate(read: set):
    """sca_v2's 195-prompt rate, or None while a4k195_main_sca_v2 has not landed."""
    path = V2_RESULT if V2_RESULT.exists() else None
    if path is None and (V2_EVAL / "DONE").exists() and (V2_EVAL / "dup_rates_people.csv").exists():
        path = V2_EVAL / "dup_rates_people.csv"
    if path is None:
        return None, None
    rows = read_rates(path)
    r = rows.get("sca_v2")
    if r is None or r["n"] != 95:
        raise ValueError(f"{path}: no complete sca_v2 row over the 95 people/animal prompts ({r})")
    read.add(path)
    return r, path


def method_data(read: set) -> dict:
    rates = read_rates(RATES195)
    read.add(RATES195)
    rows = []
    for m, name in METHODS.items():
        if m == "sca_v2":
            continue
        r = dict(rates[m])
        if m == "sca_prog" and PROG_V2_RESULT.exists():
            r = dict(read_rates(PROG_V2_RESULT)["sca_v2_prog"])
            read.add(PROG_V2_RESULT)
        assert r["n"] == 95, f"{m}: expected the 95 people/animal prompts"
        rows.append(dict(r, name=name))
    first, rest = rows[0], sorted(rows[1:], key=lambda r: (-r["rate"], r["name"]))
    v2, v2_path = v2_rate(read)
    ours = dict(v2, name=METHODS["sca_v2"], pending=False) if v2 else dict(method="sca_v2", name=METHODS["sca_v2"],
                                                                         pending=True)
    return dict(rows=[first] + rest + [ours], v2_source=str(v2_path) if v2_path else None)


# ----------------------------------------------------------------------------- selection and boxes
def dup_rate_module():
    spec = importlib.util.spec_from_file_location("dup_rate", DUP_RATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_prompts(read: set) -> list[str]:
    rows = [r for r in csv.DictReader(open(PERIMG)) if r["method"] == "sca_nb"]
    read.add(PERIMG)
    assert len(rows) == 95
    ids = sorted(r["prompt_id"] for r in rows)
    flag = {r["prompt_id"]: r["dup"] == "1" for r in rows}
    assert flag[START], "the ice climber is not flagged for published SCA"
    i0 = ids.index(START)
    walk = ids[i0:] + ids[:i0]
    return [p for p in walk if flag[p]][:N_PAIRS]


def extra_boxes(dr, det: dict, subjects: dict) -> dict:
    """Kept boxes of the subject with the largest excess; the `implied` largest are the intended subjects."""
    best = None
    for s in subjects["subjects"]:
        if s.get("implied") is None or s.get("kind") not in ("person", "animal"):
            continue
        kept = dr.count_boxes(det["dets"], s["query"], THR, SIDES, NMS, CONTAIN, CONTAIN_AREA, MAX_FRAC,
                              det["size"][0] * det["size"][1])
        excess = len(kept) - s["implied"]
        if excess > 0 and (best is None or excess > best["excess"]):
            ranked = sorted(kept, key=lambda d: -dr.area(d["b"]))
            best = dict(query=s["query"], implied=s["implied"], count=len(kept), excess=excess,
                        intended=[d["b"] for d in ranked[:s["implied"]]],
                        extras=[dict(box=d["b"], score=d["s"]) for d in ranked[s["implied"]:]])
    assert best is not None, f"{det['prompt_id']}: no subject over its implied count"
    return best


def crop_window(extras: list[dict]) -> tuple[list[int], list[dict]]:
    """CROP_PX square holding the most extra copies (ties: higher summed score), centred on their union."""
    best, key = None, None
    for k in range(len(extras), 0, -1):
        for sub in combinations(extras, k):
            x0 = min(e["box"][0] for e in sub)
            y0 = min(e["box"][1] for e in sub)
            x1 = max(e["box"][2] for e in sub)
            y1 = max(e["box"][3] for e in sub)
            if x1 - x0 > CROP_PX or y1 - y0 > CROP_PX:
                continue
            kk = (k, round(sum(e["score"] for e in sub), 6))
            if key is None or kk > key:
                best, key = (sub, (x0, y0, x1, y1)), kk
        if best is not None:
            break
    sub, (x0, y0, x1, y1) = best
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx = int(round(min(max(cx - CROP_PX / 2, 0), SIDE - CROP_PX)))
    by = int(round(min(max(cy - CROP_PX / 2, 0), SIDE - CROP_PX)))
    return [bx, by, bx + CROP_PX, by + CROP_PX], list(sub)


def image(dirs: list[Path], pid: str) -> tuple[Path, dict]:
    for d in dirs:
        hits = sorted(d.glob(f"{pid}_*_seed0.png"))
        if hits:
            assert len(hits) == 1, hits
            png = hits[0]
            meta = json.loads(png.with_suffix(".json").read_text())
            if meta.get("complete") is not True or meta.get("png_sha256") != sha(png):
                raise ValueError(f"incomplete or hash-mismatched image {png}")
            return png, meta
    raise FileNotFoundError(f"{pid}: not yet in {[str(d) for d in dirs]}")


def verdict(dr, det: dict, subj: dict) -> tuple[int, str]:
    """dup_rate.flags for one image, person and animal subjects: (flag, 'query:count/implied;...')."""
    detail, flag = [], False
    for s in subj["subjects"]:
        if s.get("implied") is None or s.get("kind") not in ("person", "animal"):
            continue
        n = len(dr.count_boxes(det["dets"], s["query"], THR, SIDES, NMS, CONTAIN, CONTAIN_AREA, MAX_FRAC,
                               det["size"][0] * det["size"][1]))
        detail.append(f"{s['query']}:{n}/{s['implied']}")
        flag |= n > s["implied"]
    return int(flag), ";".join(detail)


def v2_verdict(dr, pid: str, png_sha: str, subj: dict) -> dict:
    """The detector on this exact sca_v2 image, if a run has scored it (matched by PNG hash)."""
    for method, d in V2_DETS:
        if not d.exists():
            continue
        det = dr.load([str(d)]).get((method, pid))
        if det is not None and sha(Path(det["png"])) == png_sha:
            flag, detail = verdict(dr, det, subj)
            return dict(dup=flag, counts=detail, source=str(d), scored_png=det["png"])
    return dict(dup=None, counts=None, source="not yet scored (z980 detector step pending)")


def pairs(read: set) -> list[dict]:
    dr = dup_rate_module()
    dets = dr.load([str(DETS)])
    subjects = {s["id"]: s for s in json.loads(SUBJECTS.read_text())}
    read.update({SUBJECTS, DUP_RATE})
    out = []
    for pid in select_prompts(read):
        det = dets[("sca_nb", pid)]
        ex = extra_boxes(dr, det, subjects[pid])
        box, inside = crop_window(ex["extras"])
        pub, pmeta = image([PUB_DIR], pid)
        assert Path(det["png"]) == pub, f"the detector scored {det['png']}, not {pub}"
        v2, vmeta = image(V2_DIRS, pid)
        pc, vc = pmeta["config"], vmeta["config"]
        assert pc["method"] == "sca_nb" and pc["sca"]["gate_sigma"] == 0.89 and "win_sched" not in pc["sca"]
        assert vc["method"] in ("sca_v2", "dup_i_shrink")
        assert all(vc["sca"].get(k) == v for k, v in V2_SCA.items()), f"{v2}: not the sca_v2 operator"
        for k in ("prompt_id", "prompt", "seed", "derived_seed", "res", "steps", "guidance", "model"):
            assert pc[k] == vc[k], f"{pid}: {k} differs between the pair"
        twin = None  # the same prompt in the other sca_v2 run, if any: same flags and seed must give the same image
        for d in V2_DIRS:
            hits = sorted(d.glob(f"{pid}_*_seed0.png"))
            if hits and hits[0] != v2:
                twin = dict(path=str(hits[0]), identical_png=sha(hits[0]) == vmeta["png_sha256"])
        out.append(dict(prompt_id=pid, prompt=pc["prompt"], seed=pc["seed"], derived_seed=pc["derived_seed"],
                        detector=dict(subject=ex["query"], implied=ex["implied"], count_published=ex["count"],
                                      published=dict(zip(("dup", "counts"), verdict(dr, det, subjects[pid]))),
                                      ours=v2_verdict(dr, pid, vmeta["png_sha256"], subjects[pid]),
                                      intended_boxes=ex["intended"], extra_boxes=ex["extras"]),
                        box=box, extras_in_box=inside,
                        published=dict(png=str(pub), png_sha256=pmeta["png_sha256"], method=pc["method"]),
                        ours=dict(png=str(v2), png_sha256=vmeta["png_sha256"], method=vc["method"],
                                  sca=vc["sca"], seconds=vmeta.get("seconds"), other_run=twin)))
    return out


def make_tiles(pr: list[dict], tile_in: float) -> None:
    px = int(round(tile_in * DPI_PRINT))
    TILES.mkdir(parents=True, exist_ok=True)
    for p in pr:
        short = p["prompt_id"].replace("a4k_eval_4096_", "")
        x0, y0, x1, y1 = p["box"]
        for role in ("published", "ours"):
            img = Image.open(p[role]["png"]).convert("RGB")
            assert img.size == (SIDE, SIDE)
            files = {}
            for kind, src in (("full", img), ("crop", img.crop((x0, y0, x1, y1)))):
                path = TILES / f"{short}_{'sca_nb' if role == 'published' else 'sca_v2'}_{kind}.jpg"
                src.resize((px, px), Image.LANCZOS).save(path, "JPEG", quality=92, optimize=True, subsampling=0)
                files[kind] = dict(path=rel(path), sha256=sha(path), px=px)
            p[role]["tiles"] = files


# ----------------------------------------------------------------------------- figure
def ax_at(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def letter(fig, x_in, y_in, ch, title=None):
    fig.text(x_in / W, y_in / H, ch, fontsize=sf.FS_PANEL, fontweight="bold", va="baseline", color=sf.INK)
    if title:
        fig.text((x_in + 0.12) / W, y_in / H, title, fontsize=sf.FS_TICK, va="baseline", color=sf.INK)


def style_rate_axis(ax):
    ax.set_xlim(0, 85)
    ax.set_xticks([0, 25, 50, 75])
    ax.set_xlabel("Duplicated (%)", labelpad=2)
    sf.value_grid(ax, "x")
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)


def plot(g: dict, m: dict, pr: list[dict]) -> tuple:
    sf.use_style()
    sf.script_floor(sf.FS_TICK)  # the g of sigma_g in the 7 pt title prints at 6 pt or more
    plt.rcParams["svg.hashsalt"] = "sca-dup"
    fig = plt.figure(figsize=(W, H))
    y_letter = H - 0.115
    bottom, top = 0.33, H - 0.21

    # (a) gate sweep, 46 prompts
    xa, wa = 0.37, 0.93
    ax = ax_at(fig, xa, bottom, wa, top - bottom)
    rows = g["rows"]
    ys = np.arange(len(rows))[::-1].astype(float)
    ax.axvline(100 * g["area"]["rate"], color=sf.AREA, lw=0.9, ls=(0, (3.0, 1.6)), zorder=1)
    ax.plot([100 * r["rate"] for r in rows], ys, "o", ms=4.0, mfc=GREY_DOT, mec="white", mew=0.5, zorder=3)
    y89 = ys[[r["method"] for r in rows].index("sca_nb")]
    x_nb, x_v2 = 100 * rows[1]["rate"], 100 * g["v2"]["rate"]
    ax.add_patch(FancyArrowPatch((x_nb - 3.2, y89), (x_v2 + 3.4, y89), arrowstyle="-|>,head_length=2.6,head_width=1.5",
                                 color=sf.MID, lw=0.7, shrinkA=0, shrinkB=0, zorder=2))
    ax.plot([x_v2], [y89], "o", ms=4.6, mfc=sf.SCA, mec="white", mew=0.5, zorder=4)
    ax.set_yticks(ys, [t for _, t, _ in GATES], fontsize=sf.FS_TICK)
    ax.set_ylim(-0.75, len(rows) - 0.45)
    style_rate_axis(ax)
    lab = dict(fontsize=sf.FS_TICK, va="center")
    ax.text(100 * rows[0]["rate"] - 4.5, ys[0], "SCA w/o\nschedule", ha="right", color=GREY_DOT, linespacing=1.0,
            **lab)
    ax.text(100 * g["area"]["rate"] + 2.2, y89 - 0.36, "SCA (ours)", ha="left", color=sf.SCA, fontweight="bold",
            **lab)
    ax.text(100 * g["area"]["rate"] + 2.2, -0.52, "AReA", ha="left", color=sf.AREA, **lab)
    y85 = ys[[r["method"] for r in rows].index("sca_g85")]
    c = g["cost"]
    ax.text(100 * rows[2]["rate"] + 5.0, y85, f"{c['n_worse']}/{c['n_metrics']} metrics\nworse",
            ha="left", va="center", fontsize=sf.FS_SMALL, color=sf.MID, linespacing=1.0)
    letter(fig, 0.02, y_letter, "a", r"Gate $\sigma_g$, 46 prompts")

    # (b) every method, 195 prompts
    xb, wb = 2.20, 0.86
    bx = ax_at(fig, xb, bottom, wb, top - bottom)
    mrows = m["rows"]
    yb = np.arange(len(mrows))[::-1].astype(float)
    for r, y in zip(mrows, yb):
        ours = r["method"] == "sca_v2"
        col = sf.SCA if ours else sf.AREA if r["method"] == "area" else GREY
        if r.get("pending"):
            bx.add_patch(Rectangle((0, y - 0.31), 4.0, 0.62, fill=False, ec=sf.SCA, lw=0.6, ls=(0, (2.0, 1.2)),
                                   zorder=3))
            bx.text(6.0, y, "pending", ha="left", va="center", fontsize=sf.FS_VALUE, color=sf.SCA)
            continue
        v, lo, hi = 100 * r["rate"], 100 * r["lo95"], 100 * r["hi95"]
        bx.barh(y, v, height=0.62, color=col, lw=0, zorder=2)
        bx.plot([lo, hi], [y, y], color=sf.INK, lw=0.55, solid_capstyle="butt", zorder=3)
        bx.text(hi + 2.0, y, f"{v:.0f}%", ha="left", va="center", fontsize=sf.FS_VALUE,
                color=sf.SCA if ours else sf.INK, fontweight="bold" if ours else "normal")
    bx.set_yticks(yb, [r["name"] for r in mrows], fontsize=sf.FS_TICK)
    for t, r in zip(bx.get_yticklabels(), mrows):
        if r["method"] == "sca_v2":
            t.set_fontweight("bold")
            t.set_color(sf.SCA)
    bx.set_ylim(-0.6, len(mrows) - 0.4)
    style_rate_axis(bx)
    letter(fig, 1.40, y_letter, "b", "195 prompts")

    # (c) uncurated pairs: published thumbnail, crop | crop, ours thumbnail
    gut, mid_gap = 0.022, 0.06
    xc0 = 3.30
    tile = (W - 0.01 - xc0 - 2 * gut - mid_gap) / 4
    xs = [xc0, xc0 + tile + gut, xc0 + 2 * tile + gut + mid_gap, xc0 + 3 * tile + 2 * gut + mid_gap]
    ytop = H - 0.165
    geometry = dict(tile_in=round(tile, 4), columns_in=[round(x, 4) for x in xs], rows_top_in=round(ytop, 4))
    for i, p in enumerate(pr):
        y = ytop - (i + 1) * tile - i * gut
        x0, y0, x1, y1 = p["box"]
        cells = [("published", "full"), ("published", "crop"), ("ours", "crop"), ("ours", "full")]
        for x, (role, kind) in zip(xs, cells):
            a = ax_at(fig, x, y, tile, tile)
            arr = np.asarray(Image.open(ROOT / p[role]["tiles"][kind]["path"]))
            if kind == "full":
                a.imshow(arr, extent=(0, SIDE, SIDE, 0), interpolation="none")
                a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=RED, lw=BOX_LW, zorder=3))
                a.set_xlim(0, SIDE)
                a.set_ylim(SIDE, 0)
            else:
                a.imshow(arr, extent=(0, 1, 1, 0), interpolation="none")
                a.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=RED, lw=BOX_LW * 2, zorder=3, clip_on=True))
                a.set_xlim(0, 1)
                a.set_ylim(1, 0)
            a.axis("off")
    letter(fig, xc0 - 0.13, y_letter, "c")
    hy = y_letter / H
    fig.text((xs[0] + xs[1] + tile) / 2 / W, hy, "SCA w/o schedule", ha="center", va="baseline",
             fontsize=sf.FS_TICK, color=sf.INK)
    fig.text((xs[2] + xs[3] + tile) / 2 / W, hy, "SCA (ours)", ha="center", va="baseline",
             fontsize=sf.FS_TICK, color=sf.SCA, fontweight="bold")
    return fig, geometry


def tile_size() -> float:
    gut, mid_gap, xc0 = 0.022, 0.06, 3.30
    return (W - 0.01 - xc0 - 2 * gut - mid_gap) / 4


def caption(g: dict, m: dict, pr: list[dict]) -> str:
    v2 = m["rows"][-1]
    b = "" if v2.get("pending") else rf" (ours {100 * v2['rate']:.0f}\%)"
    words = {2: "two", 3: "three"}[len(pr) - 1]
    seen = "".join(f"; {OBSERVED[p['prompt_id']]}" for p in pr if p["prompt_id"] in OBSERVED)
    return (r"\textbf{A gradual handover cuts late duplication.} "
            r"(a)~A later gate cuts it but costs quality; ours keeps $\sigma_g=0.89$; "
            r"(b)~all methods, 95 people and animal prompts at $4096^2$" + b + ". "
            rf"(c)~Uncurated, same seed: the ice climber, then the next {words} detector-flagged prompts by id"
            + seen + ".")


def rule_sentence(pr: list[dict]) -> str:
    """One or two sentences (leading space) for the caption or text: the rule and the prompt ids of panel (c)."""
    ids = [p["prompt_id"].replace("a4k_eval_4096_", "") for p in pr]
    words = {2: "two", 3: "three"}[len(ids) - 1]
    tt = [rf"\texttt{{{i}}}" for i in ids]
    listed = ", ".join(tt[1:-1]) + (" and " if len(tt) > 2 else "") + tt[-1]
    return (rf" The examples in panel (c) are not curated: starting from the ice-climber prompt (Aesthetic-4K id "
            rf"{tt[0]}), we took the next {words} prompts in id order that the detector flags for \sca{{}} without "
            rf"the schedule (ids {listed}). Each box is the ${CROP_PX}\times{CROP_PX}$ pixel square holding the most extra "
            r"detections, and the rule was fixed before any image of the final operator was viewed." + "\n")


def main():
    read: set = set()
    g = gate_data(read)
    m = method_data(read)
    pr = pairs(read)
    make_tiles(pr, tile_size())
    fig, geometry = plot(g, m, pr)
    sf.save(fig, str(OUT), formats=("pdf",))
    cap = caption(g, m, pr)
    v2 = m["rows"][-1]
    provenance = dict(
        figure="paper/figures/dup.pdf", script="paper/figures/src/fig_dup.py",
        printed_size_in=[W, H], printed_size_texpt=[round(W * 72.27, 1), round(H * 72.27, 1)],
        caption_draft=cap,
        caption_words=len(cap.replace("~", " ").split()),
        visual_check={k: v for k, v in OBSERVED.items()},
        detector="Grounding DINO boxes (scripts/dup_detect.py), counted by scripts/dup_rate.py at threshold 0.35; "
                  "a prompt is duplicated when a person or animal subject with an implied count is detected more "
                  "often than implied; validated 20/20 on labelled published-SCA images "
                  "(results/dup_audit_20261027/validation_people_dup_reviewed.txt)",
        panel_a=dict(set="46 prompts (abl50 minus dev4), 26 name a person or animal with a count; 4096x4096, seed 0",
                     gate_sweep=g["rows"], sca_v2=dict(g["v2"], label="dup_i_shrink (46-prompt run of sca_v2)"),
                     area=g["area"], annotation=dict(g["cost"], text=f"{g['cost']['n_worse']}/{g['cost']['n_metrics']} "
                                                                        "metrics worse", at="gate 0.85 row"),
                     drawn="grey dots: published operator at each gate; green dot: sca_v2 at the published gate "
                           "(arrow from published SCA); purple dashed line: AReA; no intervals drawn (Wilson 95% "
                           "intervals listed here)"),
        panel_b=dict(set="195 Aesthetic-4K prompts at 4096x4096, seed 0; 95 people/animal prompts",
                     rows=m["rows"], sca_v2_source=m["v2_source"],
                     sca_v2_status="pending: drawn as an empty outline, no number" if v2.get("pending") else "loaded",
                     interval="Wilson 95% (scripts/dup_rate.py wilson), whiskers; value at the whisker end"),
        panel_c=dict(rule=("Fixed before any sca_v2 image was opened: start at " + START + " (ice climber); walk the "
                           "95 people/animal prompt ids of results/dup_audit_20261027/per_image_a4k195_people.csv in "
                           "Python string order, wrapping at the end, and take the next prompts with dup = 1 for "
                           f"sca_nb until there are {N_PAIRS} (count set by the panel layout). sca_v2 images are "
                           "shown whatever they show; failures are kept."),
                     box_rule=(f"Detector boxes kept by dup_rate.count_boxes (thr {THR}, sides 800 and 1333) for the "
                               "subject with the largest excess count; the `implied` largest boxes are the intended "
                               f"subjects, the rest are extra copies; the box is the {CROP_PX} px square holding the "
                               "most extra copies (ties: higher summed score), centred on their union and clamped to "
                               "the image. The same box is drawn on both images; both crops come from it."),
                     prompts=[p["prompt_id"] for p in pr], pairs=pr,
                     tiles=f"Lanczos to {DPI_PRINT} dpi at the printed tile size, JPEG quality 92, 4:4:4; red boxes "
                           "and crop frames 0.6 pt, drawn as vectors", **geometry),
        sources={rel(p): sha(p) for p in sorted(read, key=str)},
        regenerate=("PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/work/09032/saini_2/vista/library/uv-cache uv run "
                    "--no-project --python /work/09032/saini_2/vista/library/uv-envs/sca/bin/python python "
                    "paper/figures/src/fig_dup.py"))
    OUT.with_name(OUT.name + "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    RULE_TEX.write_text(rule_sentence(pr))
    print(cap)
    for r in g["rows"]:
        print(f"gate {r['tick']:12s} {r['dup']:2d}/{r['n']} = {100 * r['rate']:.1f}%  opening {r['opening_steps']} steps")
    print(f"sca_v2 46: {g['v2']['dup']}/26  AReA 46: {g['area']['dup']}/26  g85 worse on {g['cost']['worse']}")
    for r in m["rows"]:
        print(f"{r['name']:18s} " + ("pending" if r.get("pending") else f"{r['dup']:2d}/{r['n']} = {100 * r['rate']:.1f}%"))
    for p in pr:
        print(p["prompt_id"], p["box"], p["detector"]["subject"], p["detector"]["count_published"], "/",
              p["detector"]["implied"], p["ours"]["png"].split("/")[-3], p["ours"]["other_run"],
              "ours:", p["detector"]["ours"]["counts"])


if __name__ == "__main__":
    main()
