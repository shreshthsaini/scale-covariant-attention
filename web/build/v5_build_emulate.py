"""Data of the reach-emulation toy (SCAComp.emulate, web/assets/js/c_emulate.js).

One query of one 1024^2 image (FLUX.1-dev, the native stage of progressive SCA for prompt
a4k_eval_4096_minus8343371271909676658, seed 0; query token row 43, column 11, the face of the leftmost sheep),
block 12, step 20 (sigma 0.612), every one of the block's 24 heads, exactly the arrays of the paper's method figure
(paper/figures/src/v6_method_assets.py, from paper/figures/src/data/v6_attn_block12.npz):
  loc    the emulated near view at r = 4: the head's attention over the 16 x 16 window of the query's tile
         (image keys; the rest of its softmax went to text, loc_text)
  glob   the emulated far view at r = 4: its attention over the 16 x 16 block-centre keys (image part; glob_text)
  full   its full attention over all 64 x 64 image keys at the same step, stored as the paper's tiles colour it
         (density relative to uniform, log10 scaled to [0, 1] at 100x, one byte per token); m4 = its share inside
         the window; text = its text mass.
  J      the squared output error of the mix, J(pi) = E - 2 pi B + pi^2 A, of the same head at the noise bin centred
         at sigma 0.618, from the archived emulation statistics (results/evidence/calib/k1_1024.npz: 32 prompts, the
         query samples counted in its stats), for r = 2 and r = 4, divided by the full-attention output energy F of
         that head and bin: a = A / F, b = B / F, e = E / F. pi* = clip(B / A, 0, 1) is asserted equal to the paper's
         tables s2_pi_head and s4_pi_head.
The maps exist at r = 4 only (the 4096^2 case); r = 2 enters only through J.
Featured heads: 4 and 7 (the paper's near- and far-reaching heads) and 10 (the method figure's head: the one whose
full attention at this query puts the share closest to one half inside the r = 4 window; v6_method_assets.py rule).
img/base.webp: the 1024^2 image in grey, darkened for the media ground. fallback.webp: the three featured heads'
J curves on the media ground (a static picture for when the script or data cannot load).

Run (repository root, CPU, seconds):
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_build_emulate.py
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
sys.path.insert(0, HERE)

import v6_method_assets as ma  # noqa: E402
from PIL import Image  # noqa: E402
from v5_build_heads import base_image, quant  # noqa: E402

OUT = os.path.join(WEB, "assets", "data", "emulate")


def sig(x, n=5):
    return float(f"{x:.{n}g}")


def fallback(heads, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9.6, 4.0), dpi=120, facecolor="#0B0D0C")
    cols = {"near": "#F0A04B", "mid": "#ECEEE9", "far": "#6FA3E0"}
    for i, (tag, h) in enumerate(heads):
        ax = fig.add_axes([0.06 + i * 0.32, 0.18, 0.27, 0.7], facecolor="#0B0D0C")
        j = h["J"]["4"]
        p = np.linspace(0, 1, 101)
        J = j["e"] - 2 * p * j["b"] + p ** 2 * j["a"]
        ax.plot(p, (J - J.min()) / (J.max() - J.min()), color=cols[tag], lw=2.4)
        ax.axvline(j["pi"], color="#3DBB8C", lw=1.4, ls=(0, (3, 2)))
        ax.set_title(f"head {h['h']}: π* = {j['pi']:.2f}", color="#ECEEE9", fontsize=12)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#727B76")
        ax.tick_params(colors="#A5AEA8", labelsize=10)
        ax.set_xlabel("π", color="#A5AEA8", fontsize=12)
        ax.set_yticks([])
    fig.savefig(out + ".png", facecolor="#0B0D0C")
    plt.close(fig)
    Image.open(out + ".png").convert("RGB").save(out, "WEBP", quality=82, method=6)
    os.remove(out + ".png")


def main():
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
    z = np.load(ma.ATTN)
    q = tuple(int(v) for v in z["query"])
    assert int(z["block"]) == ma.BLOCK and q == (43, 11)
    si = ma.STEP_INDEX
    m4 = z["m4"][si]
    assert int(np.argmin(np.abs(m4 - 0.5))) == ma.HEAD, "method figure's head rule"
    r0, c0, w = ma.window(*q, ma.N1024, ma.RATIO)
    assert [r0, c0, w] == [int(v) for v in z["window_r4"][:3]]
    a = np.load(ma.CALIB)
    J = {}
    for r in (2, 4):
        i = list(a["scales"]).index(r)
        S = a["stats"][i][:, ma.BLOCK, :, ma.BIN]          # A, B, E_fg, E_fl, F2, count
        A, B, E, F = S[0], S[1], S[2], S[4]
        pi_tab = a[f"s{r}_pi_head"][ma.BLOCK, :, ma.BIN]
        star = np.clip(np.where(A > 0, B / np.where(A > 0, A, 1), 1.0), 0, 1)
        assert np.allclose(star, pi_tab, atol=1e-9), r
        J[r] = dict(a=A / F, b=B / F, e=E / F, pi=star, count=S[5])
    fq = quant(z["full"][si].astype(np.float64))         # [24, 64, 64]
    heads = []
    for h in range(z["full"].shape[1]):
        heads.append(dict(
            h=h + 1,
            loc=[sig(v) for v in z["local_r4"][h].ravel()], loc_text=sig(z["local_r4_text"][h]),
            glob=[sig(v) for v in z["global_r4"][h].ravel()], glob_text=sig(z["global_r4_text"][h]),
            text=sig(z["text_mass"][si, h]), m4=sig(m4[h]),
            J={str(r): dict(a=sig(J[r]["a"][h], 7), b=sig(J[r]["b"][h], 7), e=sig(J[r]["e"][h], 7),
                            pi=round(float(J[r]["pi"][h]), 4)) for r in (2, 4)}))
    counts = {str(r): int(J[r]["count"][0]) for r in (2, 4)}
    rec = dict(
        block=ma.BLOCK + 1, query=list(q), n=ma.N1024, r=ma.RATIO, window=[r0, c0, w], grid=ma.N1024 // ma.RATIO,
        sigma=round(float(z["sigma"][si]), 3), step=int(z["steps"][si]), bin_sigma=round(float(a["sigma_centers"][ma.BIN]), 3),
        n_prompts=int(a["n_prompts"]), samples=counts, dmax=ma.DMAX,
        featured=dict(near=ma.NEAR_HEAD + 1, mid=ma.HEAD + 1, far=ma.FAR_HEAD + 1),
        full_q=base64.b64encode(fq.tobytes()).decode("ascii"), heads=heads, image="img/base.webp",
        source=dict(script="web/build/v5_build_emulate.py", maps="paper/figures/src/data/v6_attn_block12.npz",
                    stats="results/evidence/calib/k1_1024.npz", rules="paper/figures/src/v6_method_assets.py"))
    base_image(ma.IMG1024, os.path.join(OUT, "img", "base.webp"))
    with open(os.path.join(OUT, "data.js"), "w") as f:
        f.write("/* Generated by web/build/v5_build_emulate.py; do not edit. */\n")
        f.write("window.SCA_DATA = window.SCA_DATA || {};\n")
        f.write("window.SCA_DATA.emulate = " + json.dumps(rec, separators=(",", ":")) + ";\n")
    fallback([("near", heads[ma.NEAR_HEAD]), ("mid", heads[ma.HEAD]), ("far", heads[ma.FAR_HEAD])], os.path.join(OUT, "fallback.webp"))
    tot = sum(os.path.getsize(os.path.join(p, f)) for p, _, fs in os.walk(OUT) for f in fs)
    for k, hh in rec["featured"].items():
        j = heads[hh - 1]["J"]
        print(k, hh, "pi* r4", j["4"]["pi"], "r2", j["2"]["pi"], "m4", heads[hh - 1]["m4"])
    print("samples", counts, f"data {tot / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
