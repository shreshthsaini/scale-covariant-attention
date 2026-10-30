"""Attention cost of sca_v2 from the formulas (no timing): keys per query, query-key pairs, FLOPs.

Counting follows Proposition 2 and Table D.1 (tab:flops): per head and layer, full attention scores (N + T)^2
query-key pairs, the global operator (N + T)(T + N0), and the local operator N K_loc with K_loc keys per image
query. sca_v2 (sca/attn.py, flags --bridge-cells 8 --bridge-k 16 --win-sched 0.88:128,0.86:112,0.84:96,0.82:80
--win-pi-match --win-pi-interp, gate 0.89) has K_loc = T + R + w(sigma)^2: text keys, the R = 1,024 bridge tokens
(8 x 8 cells x 16, sca_dedup_forward prepends them to the window keys and does not deduplicate them, so a bridge
token inside the window is scored twice) and the w x w window. FLOPs = 4 d per pair (QK^T and PV), d = 128,
24 heads, 57 blocks, 28 steps, as in Table D.1. Sigma schedule: FLUX.1 shift with mu = 1.15 at every size.

Run: python -I cost_v2.py   (numpy only; writes cost_v2.txt next to this file)
"""
import math
from pathlib import Path

T, D, HEADS, BLOCKS, STEPS, N0_SIDE, MU = 512, 128, 24, 57, 28, 64, 1.15
GATE = 0.89
WIN_SCHED = ((0.88, 128), (0.86, 112), (0.84, 96), (0.82, 80))     # (sigma_k, w_k): w_k while sigma > sigma_k
BRIDGE_CELLS, BRIDGE_K = 8, 16


def schedule(steps=STEPS, mu=MU):
    t = [1.0 - k * (1.0 - 1.0 / steps) / (steps - 1) for k in range(steps)]   # linspace(1, 1/steps, steps)
    return [math.exp(mu) / (math.exp(mu) + (1.0 / s - 1.0)) for s in t]


def nearest_divisor(n, target):
    return min((d for d in range(1, n + 1) if n % d == 0), key=lambda d: (abs(d - target), d))


def window(sigma):          # SCAContext.window_now
    for s, w in sorted(WIN_SCHED, reverse=True):
        if sigma > s:
            return w
    return N0_SIDE


def geometry(H, W):
    fh, fw = nearest_divisor(H, H / N0_SIDE), nearest_divisor(W, W / N0_SIDE)          # pool_factors
    ch, cw = nearest_divisor(H, BRIDGE_CELLS), nearest_divisor(W, BRIDGE_CELLS)          # bridge_now
    R = ch * cw * min(BRIDGE_K, (H // ch) * (W // cw))
    return (H // fh) * (W // fw), R


def pairs(H, W, sigma, v2):
    """Query-key pairs per head and layer at one step."""
    N = H * W
    N0, R = geometry(H, W)
    g = (N + T) * (T + N0)                     # global: every query, text + block-centre keys
    if sigma > GATE:
        return g                               # opening phase: global only
    w = window(sigma) if v2 else N0_SIDE
    k_loc = T + (R if v2 else 0) + min(w, H) * min(w, W)
    return g + N * k_loc                       # local: image queries only


def main():
    sig = schedule()
    lines = []
    out = lines.append
    phase = ["opening" if s > GATE else ("shrink" if window(s) != N0_SIDE else "native") for s in sig]
    out("sca_v2 cost from the formulas (cost_v2.py); no timing.")
    out("Schedule (FLUX.1 shift, mu = 1.15, 28 steps), 1-based step: sigma, phase, local window w")
    for k, (s, p) in enumerate(zip(sig, phase), 1):
        if k <= 14 or k == 28:
            out("  step %2d (scheduler index %2d): sigma %.4f  %-7s  w = %s" % (k, k - 1, s, p, "-" if p == "opening" else window(s)))
    n_open, n_shr, n_nat = (phase.count(p) for p in ("opening", "shrink", "native"))
    out("  phases: %d opening, %d shrink, %d native" % (n_open, n_shr, n_nat))
    H = W = 256
    N = H * W
    N0, R = geometry(H, W)
    out("")
    out("4096^2: grid %dx%d, N = %d, T = %d, N0 = %d block centres, bridge R = %d" % (H, W, N, T, N0, R))
    out("Keys per image query, local branch of sca_v2 (T + R + w^2), and the emulated ratio r = 256 / w:")
    out("  %5s %8s %8s %7s %6s" % ("w", "window", "K_loc", "r", "t"))
    for w in (128, 112, 96, 80, 64):
        r = max(H, W) / w
        out("  %5d %8d %8d %7.3f %6.3f" % (w, w * w, T + R + w * w, r, math.log2(r / 2)))
    out("  published local branch: T + n0^2 = %d; global branch (both operators): T + N0 = %d; full: N + T = %d"
        % (T + N0_SIDE ** 2, T + N0, N + T))
    full = sum((N + T) ** 2 for _ in sig)
    pub = sum(pairs(H, W, s, False) for s in sig)
    v2 = sum(pairs(H, W, s, True) for s in sig)
    pub_ng = STEPS * ((N + T) * (T + N0) + N * (T + N0_SIDE ** 2))
    out("")
    out("Query-key pairs per head and layer, one step:")
    out("  full                          %.4e" % ((N + T) ** 2))
    out("  global only (opening)         %.4e" % ((N + T) * (T + N0)))
    out("  published, after the gate     %.4e  (ratio to full %.2f)" % (pairs(H, W, 0.5, False), (N + T) ** 2 / pairs(H, W, 0.5, False)))
    for w, s in zip((128, 112, 96, 80), sig[n_open:n_open + n_shr]):
        out("  sca_v2, w = %3d (sigma %.4f) %.4e  (ratio %.2f)" % (w, s, pairs(H, W, s, True), (N + T) ** 2 / pairs(H, W, s, True)))
    out("  sca_v2, native w = 64         %.4e  (ratio %.2f)" % (pairs(H, W, 0.5, True), (N + T) ** 2 / pairs(H, W, 0.5, True)))
    out("")
    out("Whole 28-step trajectory, pairs per head and layer (sum over steps):")
    out("  %-36s %12s %8s" % ("operator", "pairs", "full/x"))
    out("  %-36s %12.4e %8.2f" % ("full attention", full, 1.0))
    out("  %-36s %12.4e %8.2f" % ("published SCA, no gate", pub_ng, full / pub_ng))
    out("  %-36s %12.4e %8.2f" % ("published SCA (8 opening steps)", pub, full / pub))
    out("  %-36s %12.4e %8.2f" % ("sca_v2 (8 opening, 4 shrink, 16 native)", v2, full / v2))
    out("  sca_v2 / published = %.3f  (+%.1f%% pairs)" % (v2 / pub, 100 * (v2 / pub - 1)))
    shr = sum(pairs(H, W, s, True) - pairs(H, W, s, False) for s, p in zip(sig, phase) if p == "shrink")
    brg = sum(N * R for p in phase if p == "native")
    out("  extra over published: shrink steps (wider windows + bridge) %.4e, native-step bridge keys %.4e" % (shr, brg))
    out("  bridge selection (not a query-key pair): one squared distance per key element, N h d = %.3e elements"
        " per block and step, %.4f%% of that step's sca_v2 attention FLOPs at w = 64" % (N * HEADS * D, 100 * N * HEADS * D / (pairs(H, W, 0.5, True) * HEADS * 4 * D)))
    out("")
    out("Table D.1 (tab:flops) rows, analytic attention FLOPs per image (4 d per pair, 24 heads, 57 blocks, 28 steps):")
    out("  %-11s %-9s %10s %12s %12s %12s %6s %6s" % ("size", "grid", "full", "SCA no gate", "SCA gated", "sca_v2", "f/gat", "f/v2"))
    paper = {(128, 256): (2.17e16, 5.97e15, 5.12e15), (256, 256): (8.56e16, 1.19e16, 1.02e16), (128, 384): (4.84e16, 8.93e15, 7.66e15)}
    flop = 4 * D * HEADS * BLOCKS
    for (h, w), size in (((128, 256), "2048x4096"), ((256, 256), "4096x4096"), ((128, 384), "2048x6144")):
        n = h * w
        n0, _ = geometry(h, w)
        f = flop * STEPS * (n + T) ** 2
        ng = flop * STEPS * ((n + T) * (T + n0) + n * (T + N0_SIDE ** 2))
        ga = flop * sum(pairs(h, w, s, False) for s in sig)
        v = flop * sum(pairs(h, w, s, True) for s in sig)
        for mine, theirs in zip((f, ng, ga), paper[(h, w)]):
            assert abs(float("%.2e" % mine) - theirs) < 1e-6 * theirs, (size, mine, theirs)   # reproduces Table D.1
        out("  %-11s %-9s %10.2e %12.2e %12.2e %12.2e %6.1f %6.1f" % (size, "%dx%d" % (h, w), f, ng, ga, v, f / ga, f / v))
    out("  (full, SCA no gate and SCA gated reproduce the published Table D.1 to three digits)")
    text = "\n".join(lines) + "\n"
    Path(__file__).with_name("cost_v2.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
