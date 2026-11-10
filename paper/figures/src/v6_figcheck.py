"""Check v6 figure PDFs at final size: page size, smallest text span, words inside the figure, PNG render.

Words follow the gold-standard count: every run of letters of length two or more in the figure's sans text spans (math
fonts, single letters, numbers and units are not words). Spans under 6 pt (PDF points) fail.

Run with the PyMuPDF env (CPU, seconds):
  /work/09032/saini_2/vista/library/uv-envs/csd-figs/bin/python -I v6_figcheck.py OUT_DIR fig1.pdf [fig2.pdf ...]
Writes OUT_DIR/<stem>.png (300 dpi) and prints one report line per figure; exit status 1 if any span < 6 pt.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pymupdf

MIN_PT = 6.0
WORD = re.compile(r"[A-Za-z][A-Za-z\-]+")


def check(pdf: Path, out_dir: Path, dpi: int = 300) -> tuple[bool, str]:
    doc = pymupdf.open(pdf)
    page = doc[0]
    w_in, h_in = page.rect.width / 72, page.rect.height / 72
    spans = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                txt = span["text"].strip()
                if txt:
                    spans.append((round(span["size"], 2), txt, span["font"]))
    small = sorted({s for s, _, _ in spans})
    # words: text set in the sans face only (math fonts, Computer Modern and matplotlib's cm*, hold symbols)
    text = " ".join(t for _, t, f in spans if not re.match(r"(?i)(cm[a-z]*\d|dejavu|stix)", f))
    words = WORD.findall(text)
    under = [(s, t) for s, t, _ in spans if s < MIN_PT - 1e-3]
    out_dir.mkdir(parents=True, exist_ok=True)
    page.get_pixmap(dpi=dpi).save(out_dir / f"{pdf.stem}.png")
    fonts = sorted({f for _, _, f in spans})
    rep = (f"{pdf.name}: {w_in:.3f} x {h_in:.3f} in | sizes {small[:6]} | {len(words)} words: "
           f"{' '.join(words)} | fonts {fonts}")
    if under:
        rep += f" | UNDER {MIN_PT} pt: {under}"
    return not under, rep


def main() -> None:
    out_dir = Path(sys.argv[1])
    ok = True
    for name in sys.argv[2:]:
        good, rep = check(Path(name), out_dir)
        ok &= good
        print(rep)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
