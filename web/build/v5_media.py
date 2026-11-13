"""Shared helpers of the v5 page videos (web/build/v5_*.py): paths, the vendored ffmpeg, H.264 encodes that fit a
byte budget, WebP posters, contact sheets, ffprobe checks, the manifest web/assets/video/v5/manifest.json, the page's
fonts and colours, numbers from web/build/numbers.json, and easing.

Every v5 video is H.264 High, yuv420p, BT.709 tags, +faststart, even dimensions, no audio track, with a WebP poster.
Frames are rendered as PNG under the scratch build folder (BUILD, never the repo, $HOME or /tmp) and encoded from
there; only the finished MP4, poster and manifest entry go into web/assets/video/v5/.

Run any v5 script with the project interpreter from the repository root:
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I web/build/v5_<name>.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WEB = REPO / "web"
OUT = WEB / "assets" / "video" / "v5"
MANIFEST = OUT / "manifest.json"
NUMBERS = WEB / "build" / "numbers.json"
HEADLINE_TEX = REPO / "paper" / "figures" / "snippets" / "main_tables" / "headline.tex"
FIGSRC = REPO / "paper" / "figures" / "src"
V6 = REPO / "paper" / "figures" / "v6"
BUILD = Path(os.environ.get("V5_BUILD", "/scratch/09032/saini_2/sca/web_builds/v5_webmedia_1"))
FONTS = WEB / "assets" / "fonts"
FFDIR = Path("/work/09032/saini_2/vista/library/ffmpeg-7.1-shared")
FFMPEG, FFPROBE = FFDIR / "bin" / "ffmpeg", FFDIR / "bin" / "ffprobe"
THREADS = int(os.environ.get("V5_FF_THREADS", "8"))
BANNED = re.compile("|".join(["calibrat", "curated", r"\brandom", chr(0x2014), chr(0xA7)]), re.I)


# ------------------------------------------------------------------------------------------------ design tokens
# The page's design variables (.agent/notes/web_v5/design.md, "Videos (webmedia)"): media sits on the darkroom
# "screen" ground #0B0D0C in both themes, so every video uses the dark-theme values. raised/sunk/rule are the dark
# theme's --bg-raised, --bg-sunk and --rule.
PAL = dict(
    screen="#0B0D0C", raised="#141816", sunk="#070908", rule="#252B28", ink="#ECEEE9", ink2="#A5AEA8",
    ink3="#727B76", sca="#3DBB8C", sca_ink="#5CD2A4", sca_soft="#0F2A1F", near="#F0A04B", far="#6FA3E0",
    area="#B592E3", hrdit="#C9D0CB", other="#7F8984", crop="#E4572E",
)


def rgb(h: str, a: float | None = None):
    h = h.lstrip("#")
    c = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return c if a is None else c + (int(round(255 * a)),)


def mix(c1, c2, t: float):
    a, b = (rgb(c1) if isinstance(c1, str) else c1), (rgb(c2) if isinstance(c2, str) else c2)
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


# The designer's TTF builds of the page's faces (full character sets, incl. Greek); the page's own latin WOFF2 files
# (web/assets/fonts) stand in when the scratch copy is gone.
TTF = Path("/scratch/09032/saini_2/sca/web_builds/v5_webdesign_1/fonts")
_FONT_FILES = {
    ("sans", 400): ("Inter-Regular.ttf", "inter-latin-400-normal.woff2"),
    ("sans", 500): ("Inter-Medium.ttf", "inter-latin-500-normal.woff2"),
    ("sans", 600): ("Inter-SemiBold.ttf", "inter-latin-600-normal.woff2"),
    ("serif", 400): ("SourceSerif4-Regular.ttf", "source-serif-4-latin-400-normal.woff2"),
    ("serif", 600): ("SourceSerif4-SemiBold.ttf", "source-serif-4-latin-600-normal.woff2"),
    ("serif", "i"): ("SourceSerif4-Italic.ttf", "source-serif-4-latin-400-italic.woff2"),
}
_FONT_CACHE: dict = {}


def font(size: float, family: str = "sans", weight=400) -> ImageFont.FreeTypeFont:
    """The page's own faces (Inter, Source Serif 4) at a pixel size."""
    key = (family, weight, round(size * 4) / 4)
    if key not in _FONT_CACHE:
        ttf, woff = _FONT_FILES[(family, weight)]
        path = TTF / ttf if (TTF / ttf).exists() else FONTS / woff
        _FONT_CACHE[key] = ImageFont.truetype(str(path), size=max(1, round(size)),
                                              layout_engine=ImageFont.Layout.RAQM)
    return _FONT_CACHE[key]


def mono(size: float) -> ImageFont.FreeTypeFont:
    key = ("mono", round(size))
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype("/usr/share/fonts/adobe-source-code-pro/SourceCodePro-Medium.otf",
                                              size=max(1, round(size)))
    return _FONT_CACHE[key]


def text_w(s: str, f) -> float:
    return f.getlength(s)


def text_alpha(img: Image.Image, xy, s: str, f, fill, alpha: float = 1.0, anchor: str = "la", features=None):
    """Text at an opacity (PIL ignores the alpha of a text colour): glyph coverage x alpha as a paste mask."""
    if alpha <= 0.002 or not s:
        return
    fill = rgb(fill) if isinstance(fill, str) else tuple(fill[:3])
    d = ImageDraw.Draw(img)
    if alpha >= 0.998:
        d.text(xy, s, font=f, fill=fill, anchor=anchor, features=features)
        return
    b = d.textbbox(xy, s, font=f, anchor=anchor, features=features)
    x0, y0 = int(math.floor(b[0])) - 2, int(math.floor(b[1])) - 2
    x1, y1 = int(math.ceil(b[2])) + 2, int(math.ceil(b[3])) + 2
    m = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(m).text((xy[0] - x0, xy[1] - y0), s, font=f, fill=int(round(255 * alpha)), anchor=anchor,
                           features=features)
    img.paste(Image.new("RGB", m.size, fill), (x0, y0), m)


def check_words(*strings: str, where: str = "") -> None:
    for s in strings:
        if s and BANNED.search(s):
            raise ValueError(f"wording rule broken in {where}: {s!r}")


# ------------------------------------------------------------------------------------------------ numbers
def numbers() -> dict:
    return json.loads(NUMBERS.read_text())


def num(key: str, field: str = "value"):
    d = numbers()
    if key not in d:
        raise KeyError(f"{key} missing from {NUMBERS}")
    return d[key][field]


def headline_macros() -> dict:
    """{macro name: text} of the paper's generated headline macros."""
    out = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", HEADLINE_TEX.read_text()):
        out[m.group(1)] = m.group(2)
    return out


# ------------------------------------------------------------------------------------------------ easing
def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def smootherstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def ease_in_out_sine(t):
    t = np.clip(t, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def ease_out_cubic(t):
    t = np.clip(t, 0.0, 1.0)
    return 1 - (1 - t) ** 3


# ------------------------------------------------------------------------------------------------ drawing
def rounded(d: ImageDraw.ImageDraw, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle([round(v) for v in box], radius=r, fill=fill, outline=outline, width=width)


def chip(img: Image.Image, xy, label: str, size: float, *, anchor="lt", pad=(0.62, 0.36), bg=(10, 12, 15, 168),
         fg=(236, 239, 242, 255), weight=500, family="sans", radius=None, dot=None) -> tuple:
    """A translucent dark chip with a label (the paper's resolution tag on video). Returns its box."""
    f = font(size, family, weight)
    tw = f.getlength(label)
    asc, desc = f.getmetrics()
    px, py = pad[0] * size, pad[1] * size
    dot_w = (size * 0.55 + size * 0.45) if dot else 0
    w, h = tw + 2 * px + dot_w, asc * 0.78 + 2 * py + size * 0.18
    x, y = xy
    if anchor[0] == "r":
        x -= w
    elif anchor[0] == "m":
        x -= w / 2
    if anchor[1] == "b":
        y -= h
    elif anchor[1] == "m":
        y -= h / 2
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle([x, y, x + w, y + h], radius=radius if radius is not None else h / 2, fill=bg)
    if dot:
        r = size * 0.24
        cx, cy = x + px + r, y + h / 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=dot)
    d.text((x + px + dot_w, y + h / 2 + size * 0.02), label, font=f, fill=fg, anchor="lm")
    img.alpha_composite(layer) if img.mode == "RGBA" else img.paste(layer, (0, 0), layer)
    return (x, y, x + w, y + h)


# ------------------------------------------------------------------------------------------------ ffmpeg
def ffenv() -> dict:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{FFDIR / 'lib'}:" + env.get("LD_LIBRARY_PATH", "")
    return env


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    r = subprocess.run([str(c) for c in cmd], env=ffenv(), capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({r.returncode}):\n{r.stderr[-3000:]}")
    return r


X264_COMMON = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high", "-color_primaries", "bt709",
               "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv", "-movflags", "+faststart", "-an",
               "-map_metadata", "-1", "-metadata", "title=", "-fflags", "+bitexact"]


def encode(frames: str | Path, out: Path, fps: float, *, crf: float | None = None, kbps: float | None = None,
           preset: str = "slower", tune: str | None = None, gop: int | None = None, x264: str = "",
           threads: int = THREADS, passlog: Path | None = None, vf: str | None = None,
           n_frames: int | None = None) -> Path:
    """Encode an image sequence (printf pattern) to H.264. kbps -> two-pass ABR; crf -> one pass constant quality.
    n_frames caps the frames read (a frame folder is never cleaned, so a shorter re-render must not pick up the tail
    of an older one)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    src = ["-framerate", str(fps), "-i", str(frames)] + (["-frames:v", str(n_frames)] if n_frames else [])
    vfilt = ["-vf", vf] if vf else []
    base = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-threads", str(threads), *src, *vfilt,
            "-threads", str(threads), "-preset", preset, *(["-tune", tune] if tune else []),
            *(["-g", str(gop), "-keyint_min", str(min(gop, 30))] if gop else [])]
    params = "colorprim=bt709:transfer=bt709:colormatrix=bt709" + (":" + x264 if x264 else "")
    if kbps is not None:
        log = passlog or out.with_suffix(".x264pass")
        run(base + ["-b:v", f"{kbps:.0f}k", "-pass", "1", "-passlogfile", str(log), *X264_COMMON,
                    "-x264-params", params, "-f", "mp4", "/dev/null"])
        run(base + ["-b:v", f"{kbps:.0f}k", "-pass", "2", "-passlogfile", str(log), *X264_COMMON,
                    "-x264-params", params, str(out)])
    else:
        run(base + ["-crf", f"{crf if crf is not None else 20}", *X264_COMMON, "-x264-params", params, str(out)])
    return out


def encode_to_size(frames, out: Path, fps: float, n_frames: int, max_bytes: int, *, margin: float = 0.965,
                   crf_floor: float = 16.0, **kw) -> dict:
    """Best quality that fits max_bytes: try CRF crf_floor first; if it is too big, two-pass ABR at the bitrate that
    fills max_bytes * margin (and step down until it fits). Every attempt is written on scratch; only the file that
    fits is copied into place (the repository is committed by a loop, so an oversized attempt must never sit there)."""
    import shutil
    kw["n_frames"] = n_frames
    final = Path(out)
    work = BUILD / "enc_tmp" / final.name
    work.parent.mkdir(parents=True, exist_ok=True)

    def place(rec):
        part = final.with_name("." + final.name + ".part")
        shutil.copyfile(work, part)
        os.replace(part, final)
        return rec

    encode(frames, work, fps, crf=crf_floor, **kw)
    if work.stat().st_size <= max_bytes:
        return place(dict(mode="crf", crf=crf_floor, bytes=work.stat().st_size))
    seconds = n_frames / fps
    kbps = max_bytes * margin * 8 / seconds / 1000
    for _ in range(6):
        encode(frames, work, fps, kbps=kbps, **kw)
        if work.stat().st_size <= max_bytes:
            return place(dict(mode="2pass", kbps=round(kbps), bytes=work.stat().st_size))
        kbps *= 0.94 * max_bytes / work.stat().st_size
    raise RuntimeError(f"{final} does not fit {max_bytes} bytes")


def probe(path: Path) -> dict:
    r = run([FFPROBE, "-v", "error", "-show_entries",
             "stream=codec_type,codec_name,profile,width,height,pix_fmt,r_frame_rate,nb_frames,bit_rate:"
             "format=duration,size,bit_rate", "-of", "json", path])
    d = json.loads(r.stdout)
    v = [s for s in d["streams"] if s["codec_type"] == "video"]
    a = [s for s in d["streams"] if s["codec_type"] == "audio"]
    s = v[0]
    num_, den = (int(x) for x in s["r_frame_rate"].split("/"))
    info = dict(codec=s["codec_name"], profile=s.get("profile"), width=s["width"], height=s["height"],
                pix_fmt=s["pix_fmt"], fps=num_ / den, frames=int(s.get("nb_frames", 0)),
                seconds=round(float(d["format"]["duration"]), 3), bytes=int(d["format"]["size"]),
                kbps=round(int(d["format"]["bit_rate"]) / 1000), audio_streams=len(a))
    head = Path(path).read_bytes()[:4096]
    info["faststart"] = head.find(b"moov") != -1 and (head.find(b"mdat") == -1 or head.find(b"moov") < head.find(b"mdat"))
    return info


def check_video(path: Path, *, max_bytes: int, w: int | None = None, h: int | None = None) -> dict:
    p = probe(path)
    errs = []
    if p["codec"] != "h264" or p["pix_fmt"] != "yuv420p":
        errs.append(f"codec {p['codec']} {p['pix_fmt']}")
    if p["audio_streams"]:
        errs.append("audio stream present")
    if p["width"] % 2 or p["height"] % 2:
        errs.append("odd dimensions")
    if w and (p["width"], p["height"]) != (w, h):
        errs.append(f"size {p['width']}x{p['height']}")
    if p["bytes"] > max_bytes:
        errs.append(f"{p['bytes']} bytes > {max_bytes}")
    if not p["faststart"]:
        errs.append("moov atom not at the front")
    if errs:
        raise RuntimeError(f"{path}: " + "; ".join(errs))
    return p


def decode_frames(path: Path, times: list, width: int | None = None) -> list:
    """Frames of a video at the given times (s), as PIL images (exact seek, RGB)."""
    out = []
    for t in times:
        vf = ["-vf", f"scale={width}:-2:flags=lanczos"] if width else []
        r = subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-threads", "2", "-ss", f"{t:.4f}",
                            "-i", str(path), "-frames:v", "1", *vf, "-f", "image2pipe", "-vcodec", "png", "-"],
                           env=ffenv(), capture_output=True)
        if r.returncode != 0 or not r.stdout:
            raise RuntimeError(r.stderr.decode()[-2000:])
        import io
        out.append(Image.open(io.BytesIO(r.stdout)).convert("RGB"))
    return out


def contact_sheet(video: Path, out_png: Path, n: int = 12, cols: int = 4, tile_w: int = 480, label: str = "",
                  times: list | None = None) -> Path:
    """n frames evenly spaced over the video (first and last included), tiled with their times."""
    p = probe(video)
    dur = p["seconds"]
    if times is None:
        last = max(0.0, dur - 1.0 / p["fps"] - 1e-3)
        times = [last * i / (n - 1) for i in range(n)]
    frames = decode_frames(video, times, tile_w)
    th = frames[0].height
    rows = math.ceil(len(frames) / cols)
    pad, head = 8, 30
    sheet = Image.new("RGB", (cols * tile_w + (cols + 1) * pad, rows * (th + 22) + (rows + 1) * pad + head),
                      (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), f"{label or video.name}  {p['width']}x{p['height']}  {dur:.2f} s  {p['bytes'] / 1e6:.2f} MB  "
                     f"{p['kbps']} kb/s", font=font(15, "sans", 600), fill=(20, 20, 20))
    for i, (t, fr) in enumerate(zip(times, frames)):
        r, c = divmod(i, cols)
        x, y = pad + c * (tile_w + pad), head + pad + r * (th + 22 + pad)
        sheet.paste(fr, (x, y))
        d.text((x, y + th + 3), f"t = {t:.2f} s", font=font(13), fill=(60, 60, 60))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


def frames_sheet(paths: list, out_png: Path, cols: int = 4, tile_w: int = 400, label: str = "") -> Path:
    """Tile rendered PNG frames (before encoding) with their file names."""
    ims = [Image.open(p).convert("RGB") for p in paths]
    th = round(ims[0].height * tile_w / ims[0].width)
    rows = math.ceil(len(ims) / cols)
    pad, head = 8, 30
    sheet = Image.new("RGB", (cols * tile_w + (cols + 1) * pad, rows * (th + 22) + (rows + 1) * pad + head),
                      (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), label, font=font(15, "sans", 600), fill=(20, 20, 20))
    for i, (p, im) in enumerate(zip(paths, ims)):
        r, c = divmod(i, cols)
        x, y = pad + c * (tile_w + pad), head + pad + r * (th + 22 + pad)
        sheet.paste(im.resize((tile_w, th), Image.LANCZOS), (x, y))
        d.text((x, y + th + 3), Path(p).name, font=font(13), fill=(60, 60, 60))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


def seam_sheet(video: Path, out_png: Path, tile_w: int = 640) -> Path:
    """Last frame beside first frame (loop seam check)."""
    p = probe(video)
    fr = decode_frames(video, [max(0.0, p["seconds"] - 1.0 / p["fps"] - 1e-3), 0.0], tile_w)
    sheet = Image.new("RGB", (2 * tile_w + 24, fr[0].height + 40), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), f"{video.name}: last frame | first frame", font=font(15, "sans", 600), fill=(20, 20, 20))
    sheet.paste(fr[0], (8, 32))
    sheet.paste(fr[1], (16 + tile_w, 32))
    sheet.save(out_png)
    return out_png


def poster_webp(img: Image.Image, out: Path, width: int | None = None, quality: int = 82) -> Path:
    if width and img.width != width:
        img = img.resize((width, round(img.height * width / img.width / 2) * 2), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out, "WEBP", quality=quality, method=6)
    return out


# ------------------------------------------------------------------------------------------------ manifest
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_put(entry: dict) -> None:
    """Insert or replace one item (by id) of web/assets/video/v5/manifest.json; keeps the other items."""
    check_words(entry.get("alt", ""), entry.get("caption", ""), where=f"manifest {entry['id']}")
    data = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {
        "about": ("Videos and loops of the v5 page (webmedia). Each item: file and poster (relative to this folder), "
                  "size, seconds, loop, bytes, alt text, a one-line caption, the images or data it was made from, and "
                  "the script that rebuilds it. H.264 High yuv420p, +faststart, no audio track."),
        "items": []}
    items = [e for e in data["items"] if e["id"] != entry["id"]]
    items.append(entry)
    order = {k: i for i, k in enumerate(data.get("order", []))}
    items.sort(key=lambda e: (order.get(e["id"], 999), e.get("order", 999), e["id"]))
    data["items"] = items
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


def entry_for(id_: str, mp4: Path, poster: Path, *, loop: bool, alt: str, caption: str, sources, script: str,
              order: int, extra: dict | None = None) -> dict:
    p = probe(mp4)
    e = dict(id=id_, order=order, file=mp4.relative_to(OUT).as_posix(), poster=poster.relative_to(OUT).as_posix(),
             width=p["width"], height=p["height"], seconds=p["seconds"], fps=p["fps"], loop=loop, bytes=p["bytes"],
             poster_bytes=poster.stat().st_size, alt=alt, caption=caption, sources=sources,
             script=f"web/build/{script}", sha256=sha256(mp4))
    if extra:
        e.update(extra)
    return e


def rel_repo(p) -> str:
    p = Path(p)
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


if __name__ == "__main__":
    # contact sheet of any video: python v5_media.py sheet <video> <out.png> [n]
    if len(sys.argv) >= 4 and sys.argv[1] == "sheet":
        contact_sheet(Path(sys.argv[2]), Path(sys.argv[3]), n=int(sys.argv[4]) if len(sys.argv) > 4 else 12)
        print(sys.argv[3])
