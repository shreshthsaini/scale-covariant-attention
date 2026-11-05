#!/usr/bin/env python3
"""Round 6, Step 2 summaries: the attention probe (scripts/r6_attn_probe.py) and the x0 traces (scripts/sample_trace.py).

probe: for each prompt and step, mean over blocks and heads of the text, subject-word and main-subject mass of full
attention, the window branch and the global branch at the copy, main and background patches; the same split by the
heads' mix weight (pi < 0.5: heads that lean on the global branch); and ratios window / full at the copy patch.
sheets: one contact sheet per traced prompt, the native 512 px crop at the copy for steps 6 to 21, sca_v2 above AReA.
Usage: r6_mechanism.py probe <probe dir> [--out csv] | r6_mechanism.py sheets <trace dir> <out dir>
"""

import csv
import json
from pathlib import Path
import sys

import numpy as np

REGIONS = ("copy", "main", "bg")
OPS = ("full", "loc", "glob")
QTY = ("text", "subj", "main")


def probe(d, out=None):
    rows = []
    for f in sorted(Path(d).glob("*/probe.npz")):
        z = np.load(f)
        st, pi, steps, sig = z["stats"], z["pi"], z["steps"], z["sigma"]     # st [S, layers, heads, R, O, Q]
        for si, step in enumerate(steps):
            for ri, reg in enumerate(REGIONS):
                rec = {"prompt": f.parent.name, "step": int(step), "sigma": round(float(sig[si]), 3), "region": reg}
                for oi, op in enumerate(OPS):
                    for qi, q in enumerate(QTY):
                        v = st[si, :, :, ri, oi, qi]
                        rec[f"{op}_{q}"] = float(v.mean())
                        rec[f"{op}_{q}_glob_heads"] = float(v[pi[si] < 0.5].mean()) if (pi[si] < 0.5).any() else float("nan")
                rows.append(rec)
    if out:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    # compact print: copy patch, mean over prompts, per step
    steps = sorted({r["step"] for r in rows})
    print("copy patch, mean over prompts (all heads | heads with pi < 0.5):")
    print("step sigma | subj full loc glob | text full loc glob | main full loc glob")
    for s in steps:
        rs = [r for r in rows if r["step"] == s and r["region"] == "copy"]
        if not rs:
            continue
        m = lambda k: np.nanmean([r[k] for r in rs])
        print(f"{s:4d} {m('sigma'):.3f} | " + " ".join(f"{m(f'{o}_subj'):.4f}" for o in OPS) + " | "
              + " ".join(f"{m(f'{o}_text'):.3f}" for o in OPS) + " | " + " ".join(f"{m(f'{o}_main'):.4f}" for o in OPS)
              + " || pi<.5: subj " + " ".join(f"{m(f'{o}_subj_glob_heads'):.4f}" for o in OPS)
              + " text " + " ".join(f"{m(f'{o}_text_glob_heads'):.3f}" for o in OPS))
    for reg in ("main", "bg"):
        rs = [r for r in rows if r["region"] == reg and 8 <= r["step"] <= 14]
        m = lambda k: np.nanmean([r[k] for r in rs])
        print(f"{reg} patch, steps 8-14: subj " + " ".join(f"{o} {m(f'{o}_subj'):.4f}" for o in OPS) + " | text "
              + " ".join(f"{o} {m(f'{o}_text'):.3f}" for o in OPS) + " | main " + " ".join(f"{o} {m(f'{o}_main'):.4f}" for o in OPS))
    return rows


def sheets(trace, out):
    from PIL import Image, ImageDraw
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    trace = Path(trace)
    for sid_dir in sorted((trace / "sca_v2").glob("*")):
        sid = sid_dir.name
        cells = []
        for method in ("sca_v2", "area"):
            row = []
            for k in range(6, 22):
                f = trace / method / sid / f"steps/crop_{k:02d}.png"
                im = Image.open(f).convert("RGB").resize((192, 192)) if f.exists() else Image.new("RGB", (192, 192))
                ImageDraw.Draw(im).text((3, 3), f"{method[:4]} {k}", fill=(255, 255, 0))
                row.append(im)
            cells.append(row)
        sheet = Image.new("RGB", (8 * 192, 4 * 192))
        for r, row in enumerate(cells):
            for j, im in enumerate(row):
                sheet.paste(im, ((j % 8) * 192, (2 * r + j // 8) * 192))
        sheet.save(out / f"{sid}.jpg", quality=88)
    print("sheets in", out)


def overread(npz, out):
    """Text over-read of a window: full attention's text mass at 1024^2 (calibration masses) against the text mass of a
    window holding only the near-field share of the image mass, alpha_win = a / (a + share (1 - a)), per layer, head
    and populated sigma bin, for windows of 64 to 128 tokens on a 256-token canvas (emulated radius 64 w / 512)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sca.textgate import near_share
    z = np.load(npz)
    mass, radii, c, pop, pi4 = z["mass_mean"], z["radii"], z["sigma_centers"], z["populated"], z["s4_pi_head"]
    text = mass[0]
    rows = []
    for w in (64, 80, 96, 112, 128):
        share = near_share(mass, radii, 0.5 * 64 * w / 256)
        win = text / (text + share * (1 - text))
        f = win / np.clip(text, 1e-9, None)
        for b in np.flatnonzero(pop):
            fb = f[..., b].ravel()
            g = pi4[..., b].ravel() < 0.5
            rows.append({"window": w, "sigma": round(float(c[b]), 3), "text_full": round(float(text[..., b].mean()), 4),
                         "text_window": round(float(win[..., b].mean()), 4), "overread_median": round(float(np.median(fb)), 3),
                         "overread_p90": round(float(np.percentile(fb, 90)), 3), "heads_over_2x": round(float(np.mean(fb > 2)), 3),
                         "overread_median_pi_below_half": round(float(np.median(fb[g])), 3) if g.any() else ""})
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    if sys.argv[1] == "overread":
        overread(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "probe":
        probe(sys.argv[2], sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--out" else None)
    else:
        sheets(sys.argv[2], sys.argv[3])
