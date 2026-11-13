"""Data of the heads explorer (SCAComp.heads, web/assets/js/c_heads.js): reach belongs to the head.

Everything comes from the paper's own loader, paper/figures/src/v6_observation.py load() (Figure 2), which reads
calib/k1_1024.npz and checks the variance split against results/heads_view/variance_split.csv and the paper's
headline macros (\\hlVarHead, \\hlVarStep, \\hlVarAvg). Nothing is typed here.
  curves   the near share (share of a head's image attention within Chebyshev distance 16 tokens of the query) of
           all 1,368 heads (57 blocks x 24 heads) of FLUX.1-dev at 1024^2, at the 11 populated noise bins after the
           gate (sigma < 0.89), sigma falling; 32 prompts, 256 seeded queries per block and step. Stored x 10^4.
  avg      the head-averaged curve.
  split    the variance split (head, noise level within a head, the head-averaged curve), FLUX.1-dev, plus the
           head shares of FLUX.1-Krea-dev and FLUX.2-dev (Table B.1), as the paper rounds them (half up).
  maps     block 12 (one-based), all 24 heads: the full-attention row at one query (token row 43, column 11, the
           face of the leftmost sheep) at sigma 0.90, 0.61, 0.33 (paper/figures/src/data/v6_attn_block12.npz):
           density relative to uniform, log10 scaled to [0, 1] at 100x and stored as one byte per token (base64),
           exactly the quantity the paper's tiles colour; share16 is the near share of each map.
  featured heads 4 and 7 of block 12, the paper's near-reaching and far-reaching heads (its rule, asserted there).
Images: img/base.webp, the 1024^2 image of that query (paper/figures/attn_focus/image_1024.png) in grey, darkened
for the media ground; fallback.webp, a static drawing of the curves on the media ground.

Run (repository root, CPU, seconds):
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_build_heads.py
"""
import base64
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
REPO = os.path.dirname(WEB)
sys.path.insert(0, os.path.join(REPO, "paper", "figures", "src"))

import k1data  # noqa: E402
import v6_observation as vo  # noqa: E402
from PIL import Image  # noqa: E402

OUT = os.path.join(WEB, "assets", "data", "heads")
DMAX = vo.DMAX
BASE_LO, BASE_HI = 0.07, 0.52          # grey range of the base image on the dark media ground


def quant(full):
    """[..., 64, 64] attention -> uint8 log density (as v6_observation.heat: log10(max(dens, 1)) / log10(DMAX))."""
    n2 = full.shape[-1] * full.shape[-2]
    dens = full / full.sum(axis=(-1, -2), keepdims=True) * n2
    v = np.clip(np.log10(np.maximum(dens, 1.0)) / np.log10(DMAX), 0, 1)
    return np.round(v * 255).astype(np.uint8)


def base_image(path, out, px=640):
    L = np.asarray(Image.open(path).convert("L").resize((px, px), Image.LANCZOS), dtype=np.float64) / 255
    g = np.round(255 * (BASE_LO + (BASE_HI - BASE_LO) * L)).astype(np.uint8)
    Image.fromarray(g, "L").convert("RGB").save(out, "WEBP", quality=82, method=6)


def fallback(d, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("farnear", [(0, "#6FA3E0"), (0.5, "#7F8984"), (1, "#F0A04B")])
    sig, flat = d["sig_cols"], d["flat"]
    means = flat.mean(1)
    fig = plt.figure(figsize=(9.6, 4.8), dpi=120, facecolor="#0B0D0C")
    ax = fig.add_axes([0.08, 0.13, 0.88, 0.82], facecolor="#0B0D0C")
    for i in np.argsort(np.abs(means - 0.5))[::-1]:
        ax.plot(sig, flat[i], color=cm(float(np.clip(means[i], 0, 1))), lw=0.6, alpha=0.3)
    ax.plot(sig, d["curve"], color="#0B0D0C", lw=4.5)
    ax.plot(sig, d["curve"], color="#ECEEE9", lw=2.2)
    ax.set_xlim(sig[0] + 0.02, sig[-1] - 0.02)
    ax.set_ylim(0, 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#727B76")
    ax.tick_params(colors="#A5AEA8", labelsize=11)
    ax.set_xlabel("noise level  → denoising", color="#A5AEA8", fontsize=12)
    ax.set_ylabel("near share", color="#A5AEA8", fontsize=12)
    fig.savefig(out + ".png", facecolor="#0B0D0C")
    plt.close(fig)
    Image.open(out + ".png").convert("RGB").save(out, "WEBP", quality=82, method=6)
    os.remove(out + ".png")


def main():
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
    d = vo.load()
    flat = d["flat"]
    nl, nh = 57, 24
    assert flat.shape[0] == nl * nh
    maps = d["maps"]                                   # [3, 24, 64, 64]
    q = quant(maps.astype(np.float64))
    a = np.load(vo.ATTN)
    rec = dict(
        n_blocks=nl, n_heads=nh, n_double=k1data.N_DOUBLE, radius=vo.RADIUS, gate=vo.GATE,
        sig=[round(float(s), 4) for s in d["sig_cols"]],
        curves=np.round(flat * 1e4).astype(int).ravel().tolist(),
        avg=[round(float(v), 4) for v in d["curve"]],
        split={ck: {k: v for k, v in p.items() if k in ("head", "step", "avg")} for ck, p in d["pct"].items()},
        split_f={ck: {k: round(v, 6) for k, v in p.items() if k.endswith("_f")} for ck, p in d["pct"].items()},
        featured=dict(block=vo.BLOCK + 1, near=vo.HEADS[0] + 1, far=vo.HEADS[1] + 1),
        maps=dict(block=vo.BLOCK + 1, query=list(d["query"]), n=int(maps.shape[-1]), dmax=DMAX,
                  sigma=[round(float(s), 4) for s in d["map_sigma"]], steps=[int(s) for s in a["steps"]],
                  share16=np.round(d["map_share"], 4).tolist(), text=np.round(a["text_mass"], 4).tolist(),
                  q=base64.b64encode(q.tobytes()).decode("ascii"), shape=list(q.shape), image="img/base.webp"),
        source=dict(script="web/build/v5_build_heads.py", loader="paper/figures/src/v6_observation.py load()",
                    reach="calib/k1_1024.npz", split="results/heads_view/variance_split.csv",
                    maps="paper/figures/src/data/v6_attn_block12.npz", image="paper/figures/attn_focus/image_1024.png"),
    )
    base_image(vo.IMG, os.path.join(OUT, "img", "base.webp"))
    with open(os.path.join(OUT, "data.js"), "w") as f:
        f.write("/* Generated by web/build/v5_build_heads.py; do not edit. */\n")
        f.write("window.SCA_DATA = window.SCA_DATA || {};\n")
        f.write("window.SCA_DATA.heads = " + json.dumps(rec, separators=(",", ":")) + ";\n")
    fallback(d, os.path.join(OUT, "fallback.webp"))
    tot = sum(os.path.getsize(os.path.join(p, f)) for p, _, fs in os.walk(OUT) for f in fs)
    print("split", rec["split"], "featured", rec["featured"], f"data {tot / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
