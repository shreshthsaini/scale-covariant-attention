#!/usr/bin/env python3
"""Measured hand-off schedule for the reach-annealed sampler (HANDOVER Section 40, experiment A). CPU only.

The calibration npz (calib/k1_1024.npz) stores mass_mean[1 + j, layer, head, bin]: full attention's softmax mass on
the image keys within Chebyshev radius radii[j] of the query (index 0 is the text mass), measured at 1024^2 per sigma
bin. The near share of full attention inside the window of ratio r (image side / window side) is the image mass within
radius 0.5 n0 / r (16 tokens at r = 2, 8 at r = 4) divided by the image mass, exactly the m_h of the round-7 pi blend
(SCAContext.near_share_now via sca.textgate.ctw_beta: per head, log-linear in radius, interpolated over the populated
sigma bins, clamped at the ends). Its head average m_r(sigma) is the mean over every layer and head.

Rule (lead's brief): sigma_r = the first noise level, going down from sigma = 1, where m_r exceeds tau. The sampler
uses the global view (r = 1) while sigma > sigma_2, the r = 2 view while sigma_2 >= sigma > sigma_4, and the native
window (r = 4) below sigma_4. "from step 0" means m_r > tau already at the first step (no stage before it); "never"
means m_r stays below tau down to the last step.

--norm image (default) is the image share above; all divides by all keys (text included, raw mass_mean); near_text
adds the text mass (the local branch's own key set is text plus window).

Usage: handoff_from_mass.py [--npz calib/k1_1024.npz] [--taus 0.4,0.5,0.6] [--steps 28] [--mu 1.15]
"""

import argparse
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def flux_sigmas(steps=28, mu=1.15):
    """FLUX.1-dev sampler sigmas (diffusers FluxPipeline + FlowMatchEulerDiscreteScheduler, exponential time shift).
    mu = max_shift = 1.15 is scripts/sample.py's mu_mode max_shift at every resolution."""
    s = np.linspace(1.0, 1.0 / steps, steps)
    e = math.exp(mu)
    return e / (e + (1.0 / s - 1.0))


def head_mean_share(npz, ratio, sigma, norm="image", n0_calib=None):
    """m_r(sigma): head-averaged share of full attention's mass inside the ratio-r window at noise sigma."""
    from sca.textgate import near_share
    z = np.load(npz)
    res = str(z["res"])
    side = int(res.split("x")[0]) // 16 if n0_calib is None else n0_calib       # calibration grid side (64 at 1024^2)
    rho = 0.5 * side / ratio
    mass = np.asarray(z["mass_mean"], dtype=np.float64)                          # [1 + radii, layers, heads, bins]
    share = near_share(mass, z["radii"], rho)                                    # image share [layers, heads, bins]
    if norm == "all":
        share = share * np.clip(1.0 - mass[0], 1e-6, None)
    elif norm == "near_text":
        share = share * np.clip(1.0 - mass[0], 1e-6, None) + mass[0]
    elif norm != "image":
        raise ValueError(norm)
    logm = np.log(np.clip(share, 1e-12, None))
    c = np.asarray(z["sigma_centers"], dtype=np.float64)
    pop = np.asarray(z["populated"], dtype=bool)
    c, logm = c[pop], logm[..., pop]
    out = []
    for s in np.atleast_1d(sigma):
        j = np.searchsorted(c, s)
        if j <= 0:
            v = logm[..., 0]
        elif j >= len(c):
            v = logm[..., -1]
        else:
            a = (s - c[j - 1]) / (c[j] - c[j - 1])
            v = (1 - a) * logm[..., j - 1] + a * logm[..., j]
        out.append(float(np.exp(v).mean()))
    return np.array(out)


def crossing(npz, ratio, tau, norm="image", fine=20001):
    """Largest sigma in [0, 1] below which m_r exceeds tau, scanning down from sigma = 1: +inf if m_r > tau at
    sigma = 1, -inf if it never exceeds tau. Linear between the two grid points that bracket the crossing."""
    grid = np.linspace(1.0, 0.0, fine)
    m = head_mean_share(npz, ratio, grid, norm)
    above = np.flatnonzero(m > tau)
    if above.size == 0:
        return -math.inf
    i = int(above[0])
    if i == 0:
        return math.inf
    s0, s1, m0, m1 = grid[i - 1], grid[i], m[i - 1], m[i]
    return float(s0 + (tau - m0) * (s1 - s0) / (m1 - m0))


def schedule(sig2, sig4, sigmas):
    """Per step the reach r (1, 2 or 4): 1 while sigma > sig2, 2 while sig2 >= sigma > sig4, else 4."""
    out = []
    for s in sigmas:
        out.append(1 if s > sig2 else (2 if s > sig4 else 4))
    return out


def ranges(r_steps):
    """{r: 'a-b'} step ranges of each reach (contiguous by construction)."""
    out = {}
    for r in (1, 2, 4):
        idx = [i for i, x in enumerate(r_steps) if x == r]
        out[r] = f"{idx[0]}-{idx[-1]}" if idx else "none"
    return out


def flags(sig2, sig4, eps=1e-4):
    """sample.py operator flags for a measured schedule (gate 0 = no global opening; scale-fixed S:2 = r = 2 while
    sigma > S; the native window below)."""
    if sig2 == -math.inf:
        return "(global at every step: no sampler flags)"
    gate = 0.0 if sig2 == math.inf else sig2
    s4 = 0.0 if sig4 == -math.inf else (1.0 + eps if sig4 == math.inf else sig4)
    return f"--gate-sigma {gate:.4f} --scale-fixed {s4:.4f}:2 --lattice-pool mean --bridge-cells 8 --bridge-k 16"


def fmt(s):
    return "from step 0" if s == math.inf else ("never" if s == -math.inf else f"{s:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", type=Path, default=ROOT / "calib/k1_1024.npz")
    p.add_argument("--taus", default="0.4,0.5,0.6")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--mu", type=float, default=1.15)
    p.add_argument("--norm", choices=("image", "all", "near_text"), default="image")
    p.add_argument("--all-norms", action="store_true", help="also print the schedule under the other two normalizations")
    a = p.parse_args()
    sig = flux_sigmas(a.steps, a.mu)
    taus = [float(x) for x in a.taus.split(",")]
    print(f"npz {a.npz}; sampler: {a.steps} steps, mu {a.mu}; step sigmas {[round(float(s), 4) for s in sig]}")
    for norm in ([a.norm] + [n for n in ("image", "all", "near_text") if n != a.norm] if a.all_norms else [a.norm]):
        m2, m4 = head_mean_share(a.npz, 2, sig, norm), head_mean_share(a.npz, 4, sig, norm)
        print(f"\n[norm {norm}] head-averaged near share per step (step: sigma m_2 m_4)")
        print("  " + "  ".join(f"{i}:{s:.3f} {x:.3f} {y:.3f}" for i, (s, x, y) in enumerate(zip(sig, m2, m4))))
        print(f"  range over the trajectory: m_2 {m2.min():.3f}..{m2.max():.3f}, m_4 {m4.min():.3f}..{m4.max():.3f}"
              + ("  (disjoint: no single tau crosses both inside the trajectory)" if m2.min() >= m4.max() else ""))
        print(f"  {'tau':>5} {'sigma_2':>12} {'sigma_4':>12}  steps r=1 / r=2 / r=4   flags")
        for t in taus:
            s2, s4 = crossing(a.npz, 2, t, norm), crossing(a.npz, 4, t, norm)
            r = ranges(schedule(s2, s4, sig))
            print(f"  {t:>5.2f} {fmt(s2):>12} {fmt(s4):>12}  {r[1]:>6} / {r[2]:>6} / {r[4]:>6}   {flags(s2, s4)}")
        print("  two-threshold pairs (tau_2 for r=1->2, tau_4 for r=2->4):")
        for t2 in taus:
            for t4 in taus:
                s2, s4 = crossing(a.npz, 2, t2, norm), crossing(a.npz, 4, t4, norm)
                r = ranges(schedule(s2, s4, sig))
                inner = all(x != "none" for x in r.values())
                print(f"    tau_2 {t2:.2f} tau_4 {t4:.2f}: sigma_2 {fmt(s2)}, sigma_4 {fmt(s4)}; steps r=1 {r[1]}, r=2 {r[2]}, "
                      f"r=4 {r[4]}{'  (walks 1 -> 2 -> 4)' if inner else ''}")


if __name__ == "__main__":
    main()
