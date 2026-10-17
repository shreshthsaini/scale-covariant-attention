"""SCA method: native output calibration and transferred attention geometry.

Vector PDF + text-editable SVG at the final 5.5-inch paper width. Geometry is
schematic; the coefficient inset is the exact middle-layer (zero-based28) slice
of the archived scale4 table, with original head/bin order. No synthetic data.

Run on compute:
uv run --no-project --python /work/09032/saini_2/vista/library/uv-envs/csd-figs/bin/python python paper/figures/src/fig_method.py
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from orx_figstyle import use_style

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
DATA = HERE / "data/k1_1024_pi.npz"
W, H = 396, 360  # 5.5 by 5 inches, no later shrinking
INK, GRAY, LINE = "#202830", "#687681", "#C9D1D6"
BLUE, GLOBAL, LOCAL = "#0072B2", "#516779", "#288575"
PALE_BLUE, PALE_GLOBAL, PALE_LOCAL = "#E7F2F8", "#E8EEF2", "#E7F2EF"


def main():
    use_style()
    plt.rcParams.update({"svg.hashsalt": "sca-method-20261017", "font.size": 8,
                         "mathtext.fontset": "stixsans", "svg.fonttype": "none"})
    z = np.load(DATA)
    table = z["s4_pi_head"]
    assert table.shape == (57, 24, 14)
    assert np.isfinite(table).all() and np.all((table >= 0) & (table <= 1))
    a, b = z["s4_stats"][:2]
    fitted = np.clip(np.divide(b, a, out=np.ones_like(a), where=a > 0), 0, 1)
    populated = np.flatnonzero(z["populated"])
    nearest = populated[np.abs(np.arange(14)[:, None] - populated[None, :]).argmin(1)]
    np.testing.assert_allclose(table, fitted[..., nearest], rtol=1e-12, atol=1e-12)
    pi = table[28]
    fig = plt.figure(figsize=(W / 72, H / 72), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, W), ylim=(H, 0))
    ax.axis("off")

    def text(x, y, label, size=8, color=INK, ha="left", weight="normal", **kw):
        return ax.text(x, y, label, fontsize=size, color=color, ha=ha, va="center",
                       fontweight=weight, **kw)

    def rect(x, y, w, h, fill="white", edge=LINE, lw=.5, **kw):
        p = Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=lw, **kw)
        ax.add_patch(p)
        return p

    def line(points, color=LINE, lw=.65, **kw):
        ax.plot(*zip(*points), color=color, linewidth=lw, **kw)

    def arrow(points, color=GRAY, lw=.85, dashed=False):
        if len(points) > 2:
            line(points[:-1], color, lw, linestyle=(0, (3, 2)) if dashed else "-")
        p = FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                            mutation_scale=6, linewidth=lw, color=color,
                            linestyle=(0, (3, 2)) if dashed else "-", shrinkA=0, shrinkB=0)
        ax.add_patch(p)

    def grid(x, y, side, n=16, color=LINE, fill="white", lw=.25):
        rect(x, y, side, side, fill, color, .65)
        for k in range(1, n):
            line([(x+k*side/n, y), (x+k*side/n, y+side)], color, lw)
            line([(x, y+k*side/n), (x+side, y+k*side/n)], color, lw)

    def tensor(x, y, color, name):
        # One per-head output, not a measured feature map.
        ax.add_patch(Polygon([(x+3,y-3),(x+17,y-3),(x+17,y+24),(x+14,y+27),
                              (x,y+27),(x,y)], closed=True,
                             facecolor="white", edgecolor=color, linewidth=.65))
        rect(x, y, 14, 27, "white", color, .65)
        for j in range(1, 6):
            line([(x,y+j*4.5),(x+14,y+j*4.5)], color, .35)
        for j in range(1,3):
            line([(x+j*14/3,y),(x+j*14/3,y+27)], color, .35)
        text(x+7, y+39, name, 9, color, "center")

    def attention(x, y, color, pale, rows):
        # The matrix is a shape schematic, not an attention heatmap.
        rect(x, y, 61, 34, pale, color, .65)
        rect(x, y, 9, 34, "#DDE2E5", color, .5)
        for r in range(1, 8):
            line([(x,y+r*34/8),(x+61,y+r*34/8)], "white", .45)
        for c in range(1, 11):
            line([(x+9+c*52/11,y),(x+9+c*52/11,y+34)], "white", .45)
        text(x+4.5,y-6,"text",6.8,GRAY,"center")
        text(x+35,y-6,r"image $K,V$",7.3,color,"center")
        text(x+30.5,y+44,rows,8,INK,"center")

    # A: measurement and fitting. All observations use native-resolution Q,K,V.
    text(5, 10, "a", 10, BLUE, weight="bold")
    text(19, 10, "CALIBRATE AT NATIVE RESOLUTION", 8.4, weight="bold")
    text(389, 10, "Model weights stay fixed", 7.8, GRAY, "right")
    text(26, 28, r"$1024^2$ image", 8, ha="center")
    grid(5, 39, 42, n=8, fill="#F5F7F8")
    # Highlight the native image grid, without inventing feature intensities.
    rect(5+3*5.25, 39+3*5.25, 2*5.25, 2*5.25, PALE_LOCAL, LOCAL, .8)
    text(26, 91, r"$64\!\times\!64$ tokens", 7.7, ha="center")
    arrow([(51, 60), (63, 60)])
    text(69, 32, r"Emulate scale $s=4$", 8, weight="bold")
    text(69, 49, r"$O^{\rm full}$  full-attention target", 8)
    text(69, 65, r"$O^{\rm glob}$  $16\!\times\!16$ sampled keys", 8, GLOBAL)
    text(69, 81, r"$O^{\rm loc}$  $16\!\times\!16$ local window", 8, LOCAL)
    text(69, 96, "Same native trajectory", 7.5, GRAY)
    line([(185,35),(185,88)], LINE, .65)
    text(196, 35, "FIT EACH LAYER, HEAD, NOISE BIN", 7.3, BLUE, weight="bold")
    text(196, 54, r"$\pi^\star=\mathrm{clip}_{[0,1]}(B/A)$", 10.6, BLUE)
    text(196, 72, r"$A=\sum\|\Delta\|^2,\quad B=\sum\langle R,\Delta\rangle$", 8)
    text(196, 87, r"$\Delta=O^{\rm loc}-O^{\rm glob}$", 8)
    text(196, 100, r"$R=O^{\rm full}-O^{\rm glob}$", 8)
    text(196, 112, r"$A=0:\ \pi^\star=1$", 7.1, GRAY)
    # A genuine slice, with no sorting, smoothing or example intensities.
    hx, hy, hw, hh = 336, 42, 48, 38
    cmap = LinearSegmentedColormap.from_list("sca_pi", ["#F3F6F8", BLUE])
    for h in range(24):
        for b in range(14):
            rect(hx+b*hw/14, hy+h*hh/24, hw/14, hh/24,
                 cmap(float(pi[h,b])), "none", 0)
    rect(hx,hy,hw,hh,"none",BLUE,.65)
    text(hx+hw/2,30,"Fitted table",8,BLUE,"center",weight="bold")
    text(hx-7,hy+hh/2,"head",7,GRAY,"center",rotation=90)
    text(hx+hw/2,89,r"noise bin $\sigma$",7.4,GRAY,"center")
    text(hx+hw/2,102,r"$57\!\times\!24\!\times\!14$",8,BLUE,"center")
    # Compact calibrated-weight key, separate from geometry colors.
    for k in range(30):
        rect(344+k*.9,111,.9,3,cmap(k/29),"none",0)
    text(339,112,"0",6.8,GRAY,"center")
    text(376,112,"1",6.8,GRAY,"center")
    line([(5,121),(391,121)],LINE,.65)

    # B: only the key geometry changes at deployment. All queries are retained.
    text(5, 133, "b", 10, BLUE, weight="bold")
    text(19, 133, "DEPLOY ON THE TARGET GRID", 8.4, weight="bold")
    text(389, 133, r"$4096^2$ example: $s=4$, $N=256^2$", 8, GRAY, "right")
    gx, gy, side = 8, 154, 42
    text(8,146,"GLOBAL",7.5,GLOBAL,weight="bold")
    grid(gx,gy,side,n=16)
    for k in range(0,17,4):
        line([(gx+k*side/16,gy),(gx+k*side/16,gy+side)],GLOBAL,.55)
        line([(gx,gy+k*side/16),(gx+side,gy+k*side/16)],GLOBAL,.55)
    for a in (2,6,10,14):
        for b in (2,6,10,14):
            ax.plot(gx+a*side/16,gy+b*side/16,"o",ms=1.9,color=GLOBAL)
    rect(gx+9*side/16,gy+9*side/16,side/32,side/32,INK,INK,.4)
    arrow([(54,175),(72,175)],GLOBAL)
    grid(78,159,32,n=4,color=GLOBAL,fill=PALE_GLOBAL,lw=.35)
    for a in range(4):
        for b in range(4):
            ax.plot(82+a*8,163+b*8,"o",ms=2,color=GLOBAL)
    text(59,206,r"$64^2$ block-center keys",7.7,GLOBAL,"center")
    text(59,218,r"RoPE coordinates $\lfloor p/4\rfloor$",7.7,GRAY,"center")
    arrow([(115,175),(126,175)],GLOBAL)
    text(134,175,r"all $Q$",7.2,GLOBAL,"center",rotation=90)
    attention(142,158,GLOBAL,PALE_GLOBAL,r"$(N+L)\times(64^2+L)$")
    arrow([(207,175),(222,175)],GLOBAL)
    tensor(228,161,GLOBAL,r"$O_h^{\rm glob}$")

    ly=235
    text(8,227,"LOCAL",7.5,LOCAL,weight="bold")
    grid(gx,ly,side,n=16)
    # Exact relative extents: 64/256 window and 8/256 query tile.
    win=side/4
    rect(gx+side/2,ly+side/2,win,win,PALE_LOCAL,LOCAL,.9)
    rect(gx+side/2+win/2-side/64,ly+side/2+win/2-side/64,
         side/32,side/32,INK,INK,.4)
    # A border tile sees a clamped window of the same size.
    rect(gx,ly,win,win,"none",LOCAL,.7,linestyle=(0,(2,1)))
    rect(gx+.8,ly+.8,side/32,side/32,INK,INK,.4)
    arrow([(54,256),(72,256)],LOCAL)
    grid(78,240,32,n=8,color=LOCAL,fill=PALE_LOCAL,lw=.25)
    rect(90,252,4,4,INK,INK,.4)
    text(59,287,r"$64^2$ keys; $8\!\times\!8$ query tile",7.4,LOCAL,"center")
    text(59,299,r"RoPE $p$; border-clamped windows",7.4,GRAY,"center")
    arrow([(115,256),(126,256)],LOCAL)
    text(134,256,r"tile $Q$",7.2,LOCAL,"center",rotation=90)
    attention(142,239,LOCAL,PALE_LOCAL,r"$8^2\times(64^2+L)$ per tile")
    arrow([(207,256),(222,256)],LOCAL)
    tensor(228,242,LOCAL,r"$O_h^{\rm loc}$")

    # Mixture at one layer/head. Both branches retain all text keys/values.
    arrow([(246,175),(258,175),(258,213),(270,213)],GLOBAL)
    arrow([(246,256),(258,256),(258,232),(270,232)],LOCAL)
    rect(273,195,115,54,PALE_BLUE,BLUE,.85)
    text(330.5,205,"IMAGE OUTPUT",7.5,BLUE,"center",weight="bold")
    text(330.5,222,r"$\pi_h(\sigma)\,O_h^{\rm loc}$",11,INK,"center")
    text(330.5,238,r"$+\;(1-\pi_h(\sigma))\,O_h^{\rm glob}$",10,INK,"center")
    # The transfer edge is dashed to distinguish stored weights from activations.
    arrow([(384,103),(391,103),(391,182),(331,182),(331,192)],BLUE,dashed=True)
    text(329,169,r"Transfer $\pi_h(\sigma)$",8,BLUE,"center")
    text(330,265,r"Text queries: $O^{\rm glob}$",8.3,INK,"center")
    text(330,280,r"Both branches include all $L$ text keys",7.4,GRAY,"center")
    rect(272,291,3,3,INK,INK,.4)
    text(280,293,"query tile; grids are schematic",7.2,GRAY)

    # The step bar is discrete. Its horizontal spacing is not a sigma axis.
    line([(5,311),(391,311)],LINE,.65)
    text(5,324,"c",10,BLUE,weight="bold")
    text(19,324,"GLOBAL OPENING",8.3,weight="bold")
    text(142,324,r"steps 1–8: global only",7.8,GLOBAL)
    text(272,324,r"steps 9–28: fitted mixture",7.8,BLUE)
    barx,bary,barw=19,337,368
    cell=barw/28
    for k in range(28):
        if k<8:
            rect(barx+k*cell,bary,cell-.8,7,GLOBAL,"none",0)
        else:
            rect(barx+k*cell,bary,cell-.8,3.5,GLOBAL,"none",0)
            rect(barx+k*cell,bary+3.5,cell-.8,3.5,LOCAL,"none",0)
    text(barx,352,r"$\sigma=1$",7.4,GRAY)
    text(barx+8*cell,352,r"$\sigma_g=0.89$",7.4,GRAY,"center")
    text(387,352,r"$\sigma\to0$",7.4,GRAY,"right")
    fig.savefig(OUT/"method.pdf",metadata={"Title":"SCA native calibration and deployment",
                                          "Creator":"fig_method.py","CreationDate":None})
    fig.savefig(OUT/"method.svg")
    plt.close(fig)
    receipt = {
        "source": "src/fig_method.py", "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "width_inches": W/72, "height_inches":H/72,
        "calibration_input":"src/data/k1_1024_pi.npz",
        "calibration_sha256":hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "table_key":"s4_pi_head", "table_shape":list(table.shape), "displayed_layer_zero_based":28,
        "numeric_check":"all fitted coefficients rederived from archived A/B statistics, clipped and nearest-bin filled; atol=rtol=1e-12",
        "selection":"middle layer; original head and ascending noise-bin order; no sorting or smoothing",
        "geometry":"schematic s=4, n0=64, N=256^2, local query tile8, window64; exact relative extents",
        "colors":"global slate, local teal, calibrated coefficients blue, query black, text keys gray",
        "schedule":"28-step main FLUX.1 protocol; global-only sigma>0.89 (first8), mixture thereafter",
        "numerical_scope":"geometry labels describe coordinate definitions, not exact bf16 RoPE equivalence",
        "outputs":{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
                   for name in ("method.pdf","method.svg")},
    }
    (OUT/"method_provenance.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
