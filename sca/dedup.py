"""Command-line options for the late-duplication fixes (.agent/notes/duplication_fix_20261027.md).

Shared by scripts/sample.py and scripts/sample_progressive.py. Every option defaults to off; info() returns only
the options that are set, so the sidecar block (and the config fingerprint) of the published method is unchanged.
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
    p.add_argument("--lse-tau", type=float, default=0.0, help="Per-query LSE mixing strength (0 = per-head pi)")
    return p


def context_kwargs(args):
    return dict(ramp_sigma=args.pi_ramp, ramp_mode=args.pi_ramp_mode, ramp_gamma=args.pi_ramp_gamma,
                glob_pos=args.glob_pos, win_sched=args.win_sched, text_local=args.text_local,
                text_local_lo=args.text_local_lo, bridge_cells=args.bridge_cells, bridge_k=args.bridge_k,
                bridge_lo=args.bridge_lo, lse_tau=args.lse_tau)


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
    if args.lse_tau:
        out["lse_tau"] = args.lse_tau
    return out
