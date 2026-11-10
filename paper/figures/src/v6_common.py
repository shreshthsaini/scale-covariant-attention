"""Shared code of the v6 qualitative family: Figure 1 (hero), Figure 5 (qual_main) and the appendix sample pages.

One visual language for every image figure of the paper:
* Grid. Full text width (TEXT_PT = 397.48 pt, 5.5 in), equal 2 pt gutters inside a group (G), 5 pt between groups
  (GG). Full images are squares or keep their aspect; 1:1 crops may be any rectangle and are never resampled in the
  JPEG (LaTeX scales them to the tile).
* Box. A 1 pt BOX-coloured rectangle drawn just outside the cropped window on the full image, and the same 1 pt
  frame drawn just inside the crop tile, both over a faint dark halo so the frame reads on snow and on night skies.
* Labels. Sans (Helvetica through the times package, the same face as the matplotlib figures), 8 pt, ink colour;
  ours in the SCA green, progressive SCA bold. Resolution tags 6.5 pt white on a translucent dark chip. Nothing
  smaller than 6 pt.
* Pixels. Full images Lanczos-downscaled to THUMB_DPI of their printed size (JPEG q 90, 4:4:4); crops native pixels
  (JPEG q 95, 4:4:4). Deterministic bytes, deterministic PDFs (tectonic, SOURCE_DATE_EPOCH=0).
* Selection. Every prompt and every window comes from a rule implemented here or in the figure script: seeded draws
  (random.Random(seed) over sorted ids) within keyword categories of the prompt text, and crop windows of fixed size
  at the highest mean edge energy (PIL FIND_EDGES of the 8x box-downsampled greyscale image), averaged over the
  methods shown, at least 1/16 of the side from every border. No image is placed by eye.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageFilter

import imgsel
import scafig

Image.MAX_IMAGE_PIXELS = None
# Pillow sizes the JPEG encoder buffer at width x height bytes when optimize=True and quality < 95; a high-entropy crop
# (sparks, foliage) compresses to more than that and the save fails with "broken data stream". A 16 MB floor fixes it.
ImageFile.MAXBLOCK = max(ImageFile.MAXBLOCK, 1 << 24)
HERE = Path(__file__).resolve().parent
FIG = HERE.parent
V6 = FIG / "v6"
# Build assets (JPEG tiles, standalone TikZ) live on scratch; the repo keeps the PDF, the snippet and the provenance.
ASSETS = Path(os.environ.get("V6_ASSETS", "/scratch/09032/saini_2/sca/paper_builds/v6_visuals_assets"))
REPO = FIG.parent.parent
TECTONIC = shutil.which("tectonic") or os.path.expanduser("~/.local/bin/tectonic")
EVAL = Path("/scratch/09032/saini_2/sca/eval")
ROOT195 = EVAL / "prog195_vs_baselines"  # all eight 4096^2 methods on the 195 prompts, seed 0
RES = "4096x4096"

TEXT_PT = 397.48   # \linewidth of the ICLR text block
G = 2.0            # gutter inside a group (pt)
GG = 5.0           # gutter between groups (pt)
THUMB_DPI = 300    # full images: pixels per printed inch
CROP_Q, THUMB_Q = 95, 90
# Crop windows and crop frames: the design system's red box. Read from the shared style once scafig defines CROP
# (requested from the style owner in .agent/notes/paper_v6/visuals.md); until then this one value stands in.
BOX = getattr(scafig, "CROP", "#E4572E")
BOX_LW = 1.0       # pt
HALO_LW = 1.8      # pt, black at 35 percent under every box
FS_LABEL = 8.0     # method labels
FS_TAG = 6.5       # resolution tags
INK, MID, SCA, AREA = scafig.INK, scafig.MID, scafig.SCA, scafig.AREA

# Method keys of the 195-prompt evaluation root, their names, and their colours (house style: SCA green, AReA purple,
# other baselines neutral).
NAME = {"sca_v2_prog": "Progressive SCA", "sca_v2": "Single-pass SCA", "area": "AReA", "hrdit_dev_author": "HRDiT",
        "ultraimage": "UltraImage", "dype": "DyPE", "sega_official": "SEGA", "full": "Full attention"}
OURS = ("sca_v2_prog", "sca_v2")

# Subject categories of the prompt text: the first matching rule wins (lower-cased prompt, word boundaries).
CATEGORIES = [
    ("animal", r"\b(bears?|wol(f|ves)|owls?|sheep|birds?|horses?|dogs?|cats?|fox(es)?|deer|leopards?|tigers?|lions?|"
               r"whales?|fish)\b"),
    ("portrait", r"\b(woman|man|girl|boy|model|ballerina|dancer|chef|warrior|men|women|children)\b"),
    ("still life", r"\b(vase|pitcher|bottles?|bowl|plates?|bouquet|still life|chickens?|bread|dining|interior|"
                   r"workspace)\b"),
    ("architecture", r"\b(church|castle|village|town|street|alley|bridge|buildings?|city|harbor|pagoda|great wall|"
                     r"cabin|archway|houses)\b"),
    ("night sky", r"\b(milky way|starry|aurora|night sky)\b"),
    ("landscape", r"."),
]


# Prompts that name a person (lower-cased text, word boundaries). Sample pages that show single-pass SCA, SCA without
# the schedule or the ablation variants draw from the prompts that name no person (person_free); the caption says so.
PERSON = (r"\b(woman|women|man|men|girls?|boys?|persons?|people|figures?|hikers?|climbers?|astronauts?|divers?|surfers?|"
          r"dancers?|ballerinas?|warriors?|characters?|individuals?|couples?|children|child|kids?|chefs?|models?|"
          r"silhouettes?|travell?ers?|walkers?|explorers?|skiers?|lady|ladies|monks?|soldiers?|knights?|crowds?|"
          r"family|wanderers?|humanoids?|robots?|players?|musicians?|brides?|grooms?|queens?|kings?|princes?|"
          r"princess|cowboys?|samurai|fishermen|fisherman|farmers?|sailors?|pilots?|athletes?|runners?|riders?|"
          r"elderly|teenagers?|babies|baby|humans?|faces?)\b")


def names_person(prompt: str) -> bool:
    return re.search(PERSON, prompt.lower()) is not None


def person_free(texts: dict) -> set:
    """Ids whose prompt names no person (keyword rule PERSON)."""
    return {k for k, t in texts.items() if not names_person(t)}


# ------------------------------------------------------------------------------------------------ data
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def pool195(root: Path = ROOT195) -> dict:
    """{method: {short_id: record}} for the eight 4096^2 methods at seed 0 (path, prompt_id and the per-image metrics
    ImageReward ir, HPSv2.1, MUSIQ, MUSIQ_c, PickScore, CLIP, seconds), from the evaluation root's per-image CSVs."""
    out: dict = {}
    for r in _csv(root / "base" / "per_image.csv"):
        if r["seed"] != "0" or r["res"] != RES:
            continue
        out.setdefault(r["method"], {})[imgsel.short_id(r["prompt_id"])] = dict(
            prompt_id=r["prompt_id"], path=r["path"], musiq=float(r["musiq"]), musiq_c=float(r["musiq_crop1024"]),
            pickscore=float(r["pickscore"]), clip=float(r["clip"]), seconds=float(r["seconds"]))
    for r in _csv(root / "extra" / "extra_per_image.csv"):
        if r["seed"] != "0" or r["res"] != RES:
            continue
        e = out[r["method"]][imgsel.short_id(r["prompt_id"])]
        assert e["path"] == r["path"], (r["method"], r["prompt_id"])
        e.update(ir=float(r["ir"]), hpsv21=float(r["hpsv21"]), aesthetic=float(r["aesthetic"]))
    for m, d in out.items():
        assert len(d) == 195, (m, len(d))
    return out


def sidecar(png: str | Path) -> dict:
    """The image's sidecar, checked complete and hash-matched."""
    png = Path(png)
    meta = json.loads(png.with_suffix(".json").read_text())
    if meta.get("complete") is not True or meta.get("png_sha256") != sha256(png):
        raise ValueError(f"incomplete or hash-mismatched image {png}")
    return meta


def same_prompt_and_noise(pngs: dict) -> dict:
    """Assert that the images of one example share prompt id, prompt, seed and derived seed; return those."""
    keys = ("prompt_id", "prompt", "seed", "derived_seed")
    vals = {}
    for m, p in pngs.items():
        c = sidecar(p)["config"]
        vals[m] = tuple(c.get(k) for k in keys)
    if len(set(vals.values())) != 1:
        raise ValueError(f"examples differ in prompt or noise: {vals}")
    return dict(zip(keys, next(iter(vals.values()))))


def category(prompt: str) -> str:
    p = prompt.lower()
    return next(name for name, rx in CATEGORIES if re.search(rx, p))


def prompts195(pool: dict) -> dict:
    """{short_id: prompt text} from progressive SCA's sidecars."""
    return {sid: json.loads(Path(e["path"]).with_suffix(".json").read_text())["config"]["prompt"]
            for sid, e in pool["sca_v2_prog"].items()}


def seeded_draw(ids, k: int, seed: int) -> list:
    """random.Random(seed).sample over the sorted ids: the documented draw of every v6 figure."""
    return random.Random(seed).sample(sorted(ids), k)


def draw_by_category(texts: dict, order: list, seed: int, exclude=()) -> dict:
    """One id per category in `order`, each drawn with its own Random(seed) from that category's sorted ids."""
    out = {}
    for c in order:
        ids = sorted(s for s, t in texts.items() if category(t) == c and s not in exclude)
        out[c] = random.Random(f"{seed}:{c}").choice(ids)
    return out


# ------------------------------------------------------------------------------------------------ windows
def detail_map(img: Image.Image, f: int = 8) -> np.ndarray:
    """Edge energy of the f-times box-downsampled greyscale image (fig_qual.detail_map)."""
    g = img.convert("L").resize((img.width // f, img.height // f), Image.BOX)
    a = np.asarray(g.filter(ImageFilter.FIND_EDGES), dtype=np.float64)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 0
    return a


def best_windows(maps: list, w: int, h: int, k: int = 1, f: int = 8, margin_frac: float = 1 / 16) -> list:
    """The k non-overlapping w x h windows (native px) of highest mean edge energy averaged over `maps`.

    Windows keep at least margin_frac of the side from every border; the second window is the best one that does
    not overlap the first, and so on. Returns [(x, y, w, h)] in native pixels."""
    a = np.mean(maps, axis=0)
    kw, kh = w // f, h // f
    ii = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = ii[kh:, kw:] - ii[:-kh, kw:] - ii[kh:, :-kw] + ii[:-kh, :-kw]
    my, mx = max(1, round(a.shape[0] * margin_frac)), max(1, round(a.shape[1] * margin_frac))
    s[:my, :] = s[-my:, :] = -np.inf
    s[:, :mx] = s[:, -mx:] = -np.inf
    out = []
    for _ in range(k):
        y, x = np.unravel_index(np.argmax(s), s.shape)
        out.append((int(x * f), int(y * f), w, h))
        s[max(0, y - kh + 1): y + kh, max(0, x - kw + 1): x + kw] = -np.inf  # forbid overlap with this window
    return out


def best_windows_maximin(maps: list, w: int, h: int, k: int = 1, f: int = 8, margin_frac: float = 1 / 16) -> list:
    """Like-for-like crop windows: the k non-overlapping w x h windows (native px) that maximise the SMALLEST mean edge
    energy over `maps` (one map per method), so every method's crop of a shared window has content. Same border margin
    and non-overlap rule as best_windows."""
    kw, kh = w // f, h // f
    sums = []
    for a in maps:
        ii = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        sums.append(ii[kh:, kw:] - ii[:-kh, kw:] - ii[kh:, :-kw] + ii[:-kh, :-kw])
    s = np.min(sums, axis=0)
    a0 = maps[0]
    my, mx = max(1, round(a0.shape[0] * margin_frac)), max(1, round(a0.shape[1] * margin_frac))
    s[:my, :] = s[-my:, :] = -np.inf
    s[:, :mx] = s[:, -mx:] = -np.inf
    out = []
    for _ in range(k):
        y, x = np.unravel_index(np.argmax(s), s.shape)
        out.append((int(x * f), int(y * f), w, h))
        s[max(0, y - kh + 1): y + kh, max(0, x - kw + 1): x + kw] = -np.inf
    return out


# ------------------------------------------------------------------------------------------------ tiles
def thumb(img: Image.Image, w_pt: float, h_pt: float | None = None, dpi: int = THUMB_DPI) -> Image.Image:
    """Lanczos-downscale to dpi pixels per printed inch at w_pt x h_pt (never upscale)."""
    h_pt = w_pt * img.height / img.width if h_pt is None else h_pt
    pw, ph = round(dpi * w_pt / 72.27), round(dpi * h_pt / 72.27)
    if pw >= img.width:
        return img.copy()
    return img.resize((pw, ph), Image.Resampling.LANCZOS)


def save_jpeg(img: Image.Image, path: Path, q: int, subsampling: int = 0) -> Path:
    """Deterministic JPEG; subsampling 0 is 4:4:4, 2 is 4:2:0 (used for small appendix insets to save bytes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, "JPEG", quality=q, optimize=True, subsampling=subsampling)
    return path


# ------------------------------------------------------------------------------------------------ TikZ canvas
def _hex(c: str) -> str:
    return c.lstrip("#").upper()


class Canvas:
    """A standalone TikZ picture in TeX points, y growing downward from the top-left corner.

    Images are JPEG/PDF files in `assets`; the picture is compiled with tectonic to `pdf`."""

    def __init__(self, width: float, height: float, assets: Path):
        self.w, self.h, self.assets = width, height, assets
        self.body: list[str] = []
        self.colors: dict[str, str] = {}
        assets.mkdir(parents=True, exist_ok=True)
        for old in assets.glob("*.jpg"):  # a figure's asset folder holds only that figure's tiles
            old.unlink()

    def c(self, color: str) -> str:
        """The xcolor name of a hex colour, defined in the preamble."""
        h = _hex(color)
        return self.colors.setdefault(h, f"vc{len(self.colors)}")

    def image(self, x, y, w, h, name: str):
        self.body.append(r"\node[anchor=north west,inner sep=0pt] at (%.3f,%.3f) "
                         r"{\includegraphics[width=%.3fpt,height=%.3fpt]{%s}};" % (x, -y, w, h, name))

    def box(self, x0, y0, x1, y1, color=BOX, lw=BOX_LW, inside=True, halo=True):
        """A frame on the rectangle (pt): inside=True draws it just inside, False just outside."""
        d = lw / 2 if inside else -lw / 2
        r = (x0 + d, -(y0 + d), x1 - d, -(y1 - d))
        if halo:
            self.body.append(r"\draw[draw=black,draw opacity=0.35,line width=%.2fpt] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                             % ((HALO_LW,) + r))
        self.body.append(r"\draw[draw=%s,line width=%.2fpt] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                         % ((self.c(color), lw) + r))

    def line(self, pts, color=BOX, lw=0.6, opacity=1.0, dash=False):
        path = " -- ".join("(%.3f,%.3f)" % (x, -y) for x, y in pts)
        style = "draw=%s,line width=%.2fpt,draw opacity=%.2f%s" % (self.c(color), lw, opacity,
                                                                           ",densely dashed" if dash else "")
        self.body.append(r"\draw[%s] %s;" % (style, path))

    def text(self, x, y, s, size=FS_LABEL, anchor="north west", color=INK, bold=False, rotate=0, chip=False):
        font = r"\sffamily%s\fontsize{%.1f}{%.1f}\selectfont" % (r"\bfseries" if bold else "", size, size * 1.2)
        if chip:
            style = (r"fill=%s,text=white,rounded corners=1pt," % self.c(INK) +
                     r"inner xsep=1.6pt,inner ysep=1.0pt")
        else:
            style = r"inner sep=0pt,text=%s" % self.c(color)
        if r"\\" in s:  # multi-line label: centred lines at the label size
            style += r",align=center"
        self.body.append(r"\node[anchor=%s,%s,rotate=%g,font=%s] at (%.3f,%.3f) {%s};"
                         % (anchor, style, rotate, font, x, -y, s))

    def tex(self) -> str:
        return (r"""%% Generated by a paper/figures/src/v6_*.py script; edit the script, not this file.
\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{textcomp}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
%s
\begin{document}
\begin{tikzpicture}[x=1pt,y=1pt]
\useasboundingbox (0,0) rectangle (%.3f,%.3f);
%s
\end{tikzpicture}
\end{document}
""" % ("\n".join(r"\definecolor{%s}{HTML}{%s}" % (n, h) for h, n in self.colors.items()),
       self.w, -self.h, "\n".join(self.body)))

    def compile(self, pdf: Path, stem: str) -> Path:
        src = self.assets / f"{stem}.tex"
        src.write_text(self.tex())
        with tempfile.TemporaryDirectory(dir=os.environ.get("SCA_TMP", "/scratch/09032/saini_2/tmp")) as tmp:
            r = subprocess.run([TECTONIC, "-X", "compile", src.name, "--outdir", tmp], cwd=self.assets,
                               capture_output=True, text=True, env=dict(os.environ, SOURCE_DATE_EPOCH="0"))
            if r.returncode:
                raise RuntimeError(r.stderr[-3000:])
            pdf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(tmp) / f"{stem}.pdf", pdf)
        return pdf


def tex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
            .replace("#", r"\#").replace("_", r"\_").replace("{", r"\{").replace("}", r"\}"))
