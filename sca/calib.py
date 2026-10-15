"""Calibration statistics for SCA and the merge / report that turns them into pi tables.

Per (emulated scale, layer, head, sigma bin) we accumulate over image queries
  A    = sum ||O_loc - O_glob||^2
  B    = sum <O_full - O_glob, O_loc - O_glob>
  E_fg = sum ||O_full - O_glob||^2
  E_fl = sum ||O_full - O_loc||^2
  F2   = sum ||O_full||^2
  count
For a mixture m = O_glob + pi (O_loc - O_glob) the squared error is E_fg - 2 pi B + pi^2 A,
so every pooled or switched pi variant is scored from these sums alone.
"""

from __future__ import annotations

import math

import numpy as np
import torch

QUANT = ("A", "B", "E_fg", "E_fl", "F2", "count")
RADII = (2, 4, 8, 16)


class CalibStats:
    def __init__(self, n_layers, heads, scales, bins=14, radii=RADII, mass_queries=256, device="cpu", seed=0):
        self.scales, self.bins, self.radii = tuple(scales), bins, tuple(radii)
        self.mass_queries, self.seed = mass_queries, seed
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        self.stats = z(len(self.scales), len(QUANT), n_layers, heads, bins)
        self.mass = z(1 + len(self.radii), n_layers, heads, bins)  # text, then Chebyshev radii
        self.mass_n = z(n_layers, bins)
        self.sigma = z(2, bins)  # sum of sigma, number of transformer calls

    def bin(self, sigma):
        return min(max(int(sigma * self.bins), 0), self.bins - 1)

    def add_sigma(self, sigma):
        b = self.bin(sigma)
        self.sigma[0, b] += sigma
        self.sigma[1, b] += 1

    @torch.no_grad()
    def add(self, layer, i, sigma, full, loc, glob):
        f, l, g = full.float(), loc.float(), glob.float()
        dl, df = l - g, f - g
        red = lambda x: x.sum((0, 1, 3)).double()
        n = torch.full_like(red(f[..., :1]), f.shape[0] * f.shape[1])
        vals = torch.stack([red(dl * dl), red(df * dl), red(df * df), red((f - l) ** 2), red(f * f), n])
        self.stats[i, :, layer, :, self.bin(sigma)] += vals

    @torch.no_grad()
    def add_mass(self, layer, sigma, step, qr, kr, L, grid):
        if self.mass_queries <= 0:
            return
        H, W = grid
        N, D, dev = H * W, qr.shape[-1], qr.device
        gen = torch.Generator().manual_seed(self.seed * 1000003 + step * 1009 + layer)
        sel = torch.randperm(N, generator=gen)[: self.mass_queries].to(dev)
        p = torch.einsum("bmhd,bshd->bhms", qr[:, L + sel].float(), kr.float()).div_(math.sqrt(D)).softmax(-1)
        keys = torch.arange(N, device=dev)
        cheb = torch.maximum((sel[:, None] // W - keys[None] // W).abs(), (sel[:, None] % W - keys[None] % W).abs())
        pi = p[..., L:]
        masses = [p[..., :L].sum(-1)] + [(pi * (cheb <= r)).sum(-1) for r in self.radii]
        b = self.bin(sigma)
        self.mass[:, layer, :, b] += torch.stack([m.sum((0, 2)).double() for m in masses])
        self.mass_n[layer, b] += p.shape[0] * sel.numel()

    def state(self):
        return {"stats": self.stats.cpu().numpy(), "mass": self.mass.cpu().numpy(),
                "mass_n": self.mass_n.cpu().numpy(), "sigma": self.sigma.cpu().numpy(),
                "scales": np.array(self.scales), "radii": np.array(self.radii)}


# ----------------------------------------------------------------------------- fitting

def _ratio(b, a):
    return np.clip(np.where(a > 0, b / np.where(a > 0, a, 1.0), 1.0), 0.0, 1.0)


def centers_of(sigma):
    nb = sigma.shape[1]
    mid = (np.arange(nb) + 0.5) / nb
    pop = sigma[1] > 0
    return np.where(pop, sigma[0] / np.maximum(sigma[1], 1), mid), pop


def fill(pi, pop):
    """Copy populated sigma bins into empty ones (nearest populated bin)."""
    idx = np.flatnonzero(pop)
    near = idx[np.abs(np.arange(pop.size)[:, None] - idx[None]).argmin(1)]
    return pi[..., near]


def fit(S, pop):
    """S: [6, layers, heads, bins] for one scale. Returns pi tables (empty bins filled)."""
    A, B = S[0], S[1]
    return {"pi_head": fill(_ratio(B, A), pop),
            "pi_layer": fill(_ratio(B.sum(1), A.sum(1)), pop),
            "pi_uniform": fill(_ratio(B.sum((0, 1)), A.sum((0, 1))), pop)}


def sq_err(S, pi):
    """Total squared error of the mixture with pi broadcast to [layers, heads, bins]."""
    return S[2] - 2 * pi * S[1] + pi ** 2 * S[0]


def rel_err(S, pi, axis=None):
    return sq_err(S, pi).sum(axis) / np.maximum(S[4].sum(axis), 1e-300)


def best_switch(S, centers, pop):
    """pi = 0 (global) for sigma > sigma_star, else 1 (local). Search thresholds between bins."""
    cs = np.sort(centers[pop])
    cands = np.concatenate([[0.0], (cs[1:] + cs[:-1]) / 2, [1.0]])
    best = None
    for s in cands:
        pi = (centers <= s).astype(float)[None, None]
        e = rel_err(S, pi)
        if best is None or e < best[1]:
            best = (float(s), float(e))
    return best


def as_pi(pi):
    """Broadcast layer [Lyr, bins] or uniform [bins] tables to [Lyr|1, heads|1, bins]."""
    return pi[:, None] if pi.ndim == 2 else (pi[None, None] if pi.ndim == 1 else pi)


def evaluate(S, pop, centers, tables=None):
    t = tables or fit(S, pop)
    sstar, e_sw = best_switch(S, centers, pop)
    cell = S[0] > 0
    mid = (t["pi_head"] >= 0.2) & (t["pi_head"] <= 0.8)
    return {"rel_err": {"per_head": float(rel_err(S, t["pi_head"])),
                        "per_layer": float(rel_err(S, as_pi(t["pi_layer"]))),
                        "uniform": float(rel_err(S, as_pi(t["pi_uniform"]))),
                        "switch": e_sw, "glob": float(rel_err(S, 0.0)), "loc": float(rel_err(S, 1.0))},
            "rel_err_per_bin": {"per_head": rel_err(S, t["pi_head"], (0, 1)).tolist(),
                                "glob": rel_err(S, 0.0, (0, 1)).tolist(), "loc": rel_err(S, 1.0, (0, 1)).tolist()},
            "sigma_star": sstar,
            "frac_pi_mid": float((mid & cell & pop).sum() / max((cell & pop).sum(), 1)),
            "pi_head_mean_per_bin": t["pi_head"].mean((0, 1)).tolist(),
            "pi_head_std_over_heads_per_bin": t["pi_head"].std((0, 1)).tolist(),
            "pi_uniform": t["pi_uniform"].tolist()}


def corr(a, b, mask):
    x, y = a[mask], b[mask]
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ----------------------------------------------------------------------------- merge + report

def load_parts(paths):
    return [dict(np.load(p)) for p in paths]


def merge(parts, res, out_npz):
    """Sum per-prompt parts, fit tables, write the pi table npz, return the JSON report."""
    tot = {k: sum(p[k] for p in parts) for k in ("stats", "mass", "mass_n", "sigma")}
    scales, radii = parts[0]["scales"], parts[0]["radii"]
    centers, pop = centers_of(tot["sigma"])
    halves = [[p for j, p in enumerate(parts) if j % 2 == r] for r in (0, 1)]
    out = {"scales": scales, "radii": radii, "res": np.array(res), "sigma_centers": centers, "populated": pop,
           "n_prompts": np.array(len(parts)), **tot}
    report = {"res": res, "n_prompts": len(parts), "sigma_centers": centers.tolist(),
              "populated_bins": pop.astype(int).tolist(), "scales": {}}
    for i, s in enumerate(scales.tolist()):
        S = tot["stats"][i]
        t = fit(S, pop)
        sstar, _ = best_switch(S, centers, pop)
        out.update({f"s{s}_{k}": v for k, v in t.items()})
        out[f"s{s}_sigma_star"] = np.array(sstar)
        r = evaluate(S, pop, centers, t)
        if min(len(h) for h in halves) > 0:
            fh = [fit(sum(p["stats"][i] for p in h), pop) for h in halves]
            mask = (S[0] > 0) & pop[None, None]
            r["split_half_corr"] = {"per_head": corr(fh[0]["pi_head"], fh[1]["pi_head"], mask),
                                    "per_layer": corr(fh[0]["pi_layer"], fh[1]["pi_layer"], mask.any(1))}
            # cross-fit: tables from one half scored on the other half
            r["cross_half_rel_err_per_head"] = float(np.mean(
                [rel_err(sum(p["stats"][i] for p in halves[1 - h]), fh[h]["pi_head"]) for h in (0, 1)]))
        report["scales"][str(s)] = r
    mn = np.maximum(tot["mass_n"], 1)[None, :, None, :]
    mass = tot["mass"] / mn  # [1 + radii, layers, heads, bins]
    out["mass_mean"] = mass
    report["attention_mass"] = {"text_per_bin": mass[0].mean((0, 1)).tolist(),
                                **{f"r{r}_per_bin": mass[j + 1].mean((0, 1)).tolist() for j, r in enumerate(radii.tolist())}}
    np.savez(str(out_npz), **out)
    return report


def transfer(npz_paths):
    """Relative error when a per-head table from one (res, scale) is applied to another."""
    entries = []
    for p in npz_paths:
        z = np.load(p)
        for i, s in enumerate(z["scales"].tolist()):
            entries.append((f"{z['res']}/s{s}", z["stats"][i], z[f"s{s}_pi_head"], z["populated"]))
    m = {}
    for la, _, pi, _ in entries:
        for lb, S, own, pop in entries:
            if pi.shape == own.shape:
                m[f"{la} -> {lb}"] = {"rel_err": float(rel_err(S, pi)), "own": float(rel_err(S, own))}
    return m
