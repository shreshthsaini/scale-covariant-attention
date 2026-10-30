"""Command-line options for the late-duplication fixes (.agent/notes/duplication_fix_20261027.md).

Shared by scripts/sample.py and scripts/sample_progressive.py. Every option defaults to off; info() returns only
the options that are set, so the sidecar block (and the config fingerprint) of the published method is unchanged.
The attention options go to SCAContext (context_kwargs); the latent anchor (--anchor-*, sca/anchor.py) works at the
sampler level and is built by anchor() in scripts/sample.py only.
"""


def win_sched(text):
    """'0.8:128,0.7:96' -> ((0.8, 128), (0.7, 96)): local window 128 while sigma > 0.8, then 96 while > 0.7."""
    out = []
    for item in text.split(","):
        a, b = item.split(":")
        out.append((float(a), int(b)))
    return tuple(sorted(out, reverse=True))


def add_args(p):
    p.add_argument("--pi-ramp", type=float, default=0.0, help="Soft opening: hand pi over from 0 at --gate-sigma "
                   "to the table at this sigma (0 = off, published method)")
    p.add_argument("--pi-ramp-mode", choices=("linear", "head"), default="linear",
                   help="linear: pi * r^gamma; head: head h opens at ramp + (gate - ramp) * pi_h")
    p.add_argument("--pi-ramp-gamma", type=float, default=1.0)
    p.add_argument("--glob-pos", choices=("floor", "frac"), default="floor",
                   help="Global-branch query/key ids: floor(p / f) (published) or p / f")
    p.add_argument("--win-sched", type=win_sched, default=(), metavar="SIGMA:WIN,...",
                   help="Wider local windows at high noise, e.g. 0.8:128,0.7:96 (default: off)")
    p.add_argument("--text-local", type=float, default=0.0, help="Logit bias on text keys in the local branch (0 = off)")
    p.add_argument("--text-local-lo", type=float, default=0.0, help="--text-local applies while sigma > this")
    p.add_argument("--bridge-cells", type=int, default=0, help="AReA bridge: cells per axis (0 = off)")
    p.add_argument("--bridge-k", type=int, default=16, help="Bridge tokens per cell")
    p.add_argument("--bridge-lo", type=float, default=0.0, help="Bridge active while sigma > this")
    p.add_argument("--bridge-k-hi", type=int, default=0, metavar="K_HI", help="Bridge tokens per cell while sigma > "
                   "--bridge-k-hi-lo, --bridge-k below (0 = off: --bridge-k at every sigma)")
    p.add_argument("--bridge-k-hi-lo", type=float, default=0.0, metavar="SIGMA", help="--bridge-k-hi applies while "
                   "sigma > SIGMA")
    p.add_argument("--lse-tau", type=float, default=0.0, help="Per-query LSE mixing strength (0 = per-head pi)")
    p.add_argument("--scale-sched", action="store_true", help="Per-head apparent scale: each head attends n0 x n0 keys "
                   "on the stride-(s / a) lattice around its tile, a growing from 1 (global branch) to s (native window)")
    p.add_argument("--scale-lo", type=float, default=0.5, metavar="SIGMA", help="--scale-sched: head h reaches its "
                   "native window at SIGMA + (gate - SIGMA) * pi_h")
    p.add_argument("--scale-fixed", type=win_sched, default=(), metavar="SIGMA:A,...", help="Every head at apparent "
                   "scale A while sigma > SIGMA (stride s / A lattice, coarse ids, weight from the scale-A table; bridge "
                   "keys at the same coarse ids), the published local branch below (default: off)")
    p.add_argument("--win-pi-match", action="store_true", help="On --win-sched steps read pi from the table at the "
                   "emulated scale max(grid) / window (log-nearest calibrated scale)")
    p.add_argument("--win-heads-below", type=float, default=None, metavar="TAU", help="On --win-sched steps only heads "
                   "whose published pi < TAU take the scheduled window, with the weight at its emulated scale; the "
                   "others keep the base window and the published weight (default: off; 0 = no head widens)")
    p.add_argument("--anchor-lo", type=float, default=0.0, metavar="SIGMA", help="Latent low-band anchor "
                   "(sca/anchor.py): while sigma > SIGMA the low band of x0_hat follows the x0_hat of the last "
                   "global-only step (0 = off)")
    p.add_argument("--anchor-shape", choices=("linear", "cosine"), default="linear",
                   help="Anchor weight from 1 at --gate-sigma to 0 at --anchor-lo")
    p.add_argument("--anchor-pool", type=int, default=0, metavar="P",
                   help="Anchor low band: P x P token average, bilinear up (0 = the global block size)")
    p.add_argument("--anchor-refresh", type=int, default=0, metavar="K", help="Recompute the anchor every K steps "
                   "after the gate from one extra global-only forward (0 = fixed anchor)")
    return p


def context_kwargs(args):
    return dict(ramp_sigma=args.pi_ramp, ramp_mode=args.pi_ramp_mode, ramp_gamma=args.pi_ramp_gamma,
                glob_pos=args.glob_pos, win_sched=args.win_sched, text_local=args.text_local,
                text_local_lo=args.text_local_lo, bridge_cells=args.bridge_cells, bridge_k=args.bridge_k,
                bridge_lo=args.bridge_lo, bridge_k_hi=args.bridge_k_hi, bridge_k_hi_lo=args.bridge_k_hi_lo,
                lse_tau=args.lse_tau, scale_sched=args.scale_sched, scale_lo=args.scale_lo,
                scale_fixed=args.scale_fixed, win_pi_match=args.win_pi_match, win_heads_below=args.win_heads_below)


def anchor(args, ctx):
    """sca.anchor.LatentAnchor for --anchor-lo > 0 (validated against ctx, so before the model loads), else None."""
    if not args.anchor_lo:
        return None
    from sca.anchor import LatentAnchor
    return LatentAnchor(ctx, args.anchor_lo, args.anchor_shape, args.anchor_pool, args.anchor_refresh)


def info(args):
    out = {}
    if args.pi_ramp:
        out.update(pi_ramp=args.pi_ramp, pi_ramp_mode=args.pi_ramp_mode, pi_ramp_gamma=args.pi_ramp_gamma)
    if args.glob_pos != "floor":
        out["glob_pos"] = args.glob_pos
    if args.win_sched:
        out["win_sched"] = [list(x) for x in args.win_sched]
    if args.text_local:
        out.update(text_local=args.text_local, text_local_lo=args.text_local_lo)
    if args.bridge_cells:
        out.update(bridge_cells=args.bridge_cells, bridge_k=args.bridge_k, bridge_lo=args.bridge_lo)
    if args.bridge_k_hi:
        out.update(bridge_k_hi=args.bridge_k_hi, bridge_k_hi_lo=args.bridge_k_hi_lo)
    if args.lse_tau:
        out["lse_tau"] = args.lse_tau
    if args.scale_sched:
        out.update(scale_sched=True, scale_lo=args.scale_lo)
    if args.scale_fixed:
        out["scale_fixed"] = [list(x) for x in args.scale_fixed]
    if args.win_pi_match:
        out["win_pi_match"] = True
    if args.win_heads_below is not None:
        out["win_heads_below"] = args.win_heads_below
    if args.anchor_lo:
        out.update(anchor_lo=args.anchor_lo, anchor_shape=args.anchor_shape, anchor_pool=args.anchor_pool,
                   anchor_refresh=args.anchor_refresh)
    return out
