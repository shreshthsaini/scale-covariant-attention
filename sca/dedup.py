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
    p.add_argument("--bridge-ratio", type=float, default=0.0, metavar="RHO", help="Bridge size tied to the local window "
                   "as in AReA: round(RHO * w)^2 tokens for window side w, split evenly over the cells (replaces "
                   "--bridge-k; 0 = off). RHO 0.5 gives 16 per cell of 8 x 8 at w = 64")
    p.add_argument("--lse-tau", type=float, default=0.0, help="Per-query LSE mixing strength (0 = per-head pi)")
    p.add_argument("--scale-sched", action="store_true", help="Per-head apparent scale: each head attends n0 x n0 keys "
                   "on the stride-(s / a) lattice around its tile, a growing from 1 (global branch) to s (native window)")
    p.add_argument("--scale-lo", type=float, default=0.5, metavar="SIGMA", help="--scale-sched: head h reaches its "
                   "native window at SIGMA + (gate - SIGMA) * pi_h")
    p.add_argument("--scale-fixed", type=win_sched, default=(), metavar="SIGMA:A,...", help="Every head at apparent "
                   "scale A while sigma > SIGMA (stride s / A lattice, coarse ids, weight from the scale-A table; bridge "
                   "keys at the same coarse ids), the published local branch below (default: off)")
    p.add_argument("--win-pi-interp", action="store_true", help="With --win-pi-match: interpolate pi log-linearly "
                   "between the bracketing calibrated scales (e.g. 112-token windows at 4096^2 sit between scale 2 and 4)")
    p.add_argument("--win-pi-match", action="store_true", help="On --win-sched steps read pi from the table at the "
                   "emulated scale max(grid) / window (log-nearest calibrated scale)")
    p.add_argument("--win-heads-below", type=float, default=None, metavar="TAU", help="On --win-sched steps only heads "
                   "whose published pi < TAU take the scheduled window, with the weight at its emulated scale; the "
                   "others keep the base window and the published weight (default: off; 0 = no head widens)")
    p.add_argument("--text-mode", default="", metavar="MODE", help="Round-6 window text option (sca/textgate.py): "
                   "ctw (calibrated text weight), gtg (global text gate), gtx (global text read-out), ctw+gtg")
    p.add_argument("--text-mode-lo", type=float, default=0.0, metavar="SIGMA", help="--text-mode applies while sigma > this")
    p.add_argument("--text-gamma", type=float, default=1.0, help="gtg: strength of the global text gate")
    p.add_argument("--text-clip", type=float, default=6.0, help="gtg: floor of the gate's log ratio")
    p.add_argument("--text-bw-scale", type=float, default=1.0, help="bw: scale of the bridge offset log(N / R)")
    p.add_argument("--subject-words", default="", metavar="JSON", help="sb: JSON {prompt_id: [words]} (subject words)")
    p.add_argument("--ffl-near", choices=("window", "bridge"), default="window", help="Round 7, text mode ffl: near "
                   "part = the window alone (round 6) or sca_v2's window plus bridge")
    p.add_argument("--ffl-head-frac", type=float, default=0.0, metavar="F", help="Round 7, ffl: only the fraction F of "
                   "heads with the smallest calibrated pi at the hand-off take the far field (0 = all heads)")
    p.add_argument("--ffl-fade-lo", type=float, default=0.0, metavar="SIGMA", help="Round 7, ffl: far share fades "
                   "linearly from its calibrated value at --gate-sigma to 0 at SIGMA (0 = no fade)")
    p.add_argument("--ffl-outside", action="store_true", help="Round 7, ffl: leave lattice keys inside the query's own "
                   "window out of the far field")
    p.add_argument("--ffl-coarse", action="store_true", help="Round 7, ffl: far field at the global branch's coarse ids "
                   "(query-key offsets within the native grid) instead of true positions")
    p.add_argument("--text-bw-lo", type=float, default=None, metavar="SIGMA", help="Round 7: text mode bw applies only "
                   "while sigma > SIGMA (default: while --text-mode applies)")
    p.add_argument("--open-pool", type=int, default=0, metavar="P", help="Round 7: global-only steps with sigma < "
                   "--open-pool-hi pool by P (finer coarse layout before the hand-off; 0 = off)")
    p.add_argument("--open-pool-hi", type=float, default=0.0, metavar="SIGMA")
    p.add_argument("--morph-local", default="", choices=("", "area_tiles"), help="Morph ladder (sca/morph.py): "
                   "area_tiles = AReA's jittered tiles, Halo, DWT splits and merge as SCA's local branch")
    p.add_argument("--morph-text", default="", choices=("", "window_mean", "all_coarse"), help="Morph ladder: "
                   "window_mean = text queries read the mean over windows (AReA) instead of the global branch; "
                   "all_coarse = audit Fix 4, text queries attend every image key at its coarse id")
    p.add_argument("--text-value", type=float, default=1.0, metavar="A", help="Image queries read the text values "
                   "scaled by A in both branches (attention weights unchanged, so the prompt still acts as a sink)")
    p.add_argument("--text-value-lo", type=float, default=0.0, metavar="SIGMA", help="--text-value applies while "
                   "sigma > SIGMA")
    p.add_argument("--subject-anchor", type=float, default=1.0, metavar="A", help="Subject-anchored prompt gating "
                   "(sca/subjgate.py): on window steps, subject-word content outside the opening's subject map is "
                   "scaled by A (needs --subject-words; 1 = off)")
    p.add_argument("--subject-anchor-lo", type=float, default=0.55, metavar="SIGMA", help="--subject-anchor applies "
                   "while sigma > SIGMA")
    p.add_argument("--subject-anchor-dilate", type=int, default=2, metavar="T", help="Subject map dilation, tokens")
    p.add_argument("--subject-anchor-all", action="store_true", help="The subject map gates every text token's content")
    p.add_argument("--subject-anchor-q", type=float, nargs=2, default=(0.5, 0.95), metavar=("QLO", "QHI"),
                   help="Subject map normalization quantiles")
    p.add_argument("--subject-anchor-dump", default="", metavar="DIR", help="Save each image's subject map (npz)")
    p.add_argument("--subject-anchor-smooth", type=int, default=0, metavar="K", help="Average the raw subject map over "
                   "K x K tokens before the ramp (0 = off)")
    p.add_argument("--subject-anchor-rel", type=float, nargs=2, default=None, metavar=("A", "B"), help="Ramp from "
                   "med + A (max - med) to med + B (max - med) instead of the quantiles")
    p.add_argument("--subject-anchor-edge", type=int, default=0, metavar="T", help="Keep full prompt content on a "
                   "border band of T tokens")
    p.add_argument("--guide-map", type=float, default=0.0, metavar="G", help="Spatially varying guidance (sca/guidemap.py): on window steps a second pass at guidance G, blended with the main pass by the opening's subject map (--subject-anchor-* map flags; needs --subject-words); 0 = off")
    p.add_argument("--guide-map-lo", type=float, default=0.58, metavar="SIGMA", help="--guide-map applies while sigma > SIGMA (0.58: steps 8 to 19 of 28)")
    p.add_argument("--text-branch", default="both", choices=("both", "loc", "glob"), help="Where --text-value and "
                   "--subject-anchor act: both branches, the window branch only, or the global branch only")
    p.add_argument("--text-value-subject", action="store_true", help="--text-value scales only the subject-word tokens "
                   "(needs --subject-words)")
    p.add_argument("--qmix", default="", choices=("", "f1", "f13"), help="Per-query mix (sca/qmix.py): f1 = weight "
                   "from count-corrected image partitions; f13 = plus one three-part softmax over text, near, far")
    p.add_argument("--qmix-lo", type=float, default=0.0, metavar="SIGMA", help="--qmix applies while sigma > SIGMA")
    p.add_argument("--qmix-tau", type=float, default=1.0, help="Slope of the per-query log-odds swing")
    p.add_argument("--qmix-gamma", type=float, default=1.0, help="f13: scale of the image part (near and far) against "
                   "the text part; 1 = one plain softmax, < 1 raises the text share")
    p.add_argument("--qmix-far", default="out", choices=("out", "all"), help="f13 far part: block centres outside the "
                   "window (out) or every block centre (all)")
    p.add_argument("--qmix-dose", type=float, default=1.0, help="f13: dose * f13 + (1 - dose) * sca_v2's image output")
    p.add_argument("--qmix-anchor", default="text", choices=("text", "near"), help="f13 head offset on the near part "
                   "(text) or on the far part (near: text and near keep the window softmax's balance)")
    p.add_argument("--pi-blend", type=float, default=0.0, metavar="LAMBDA", help="Audit Fix 2: pi' = (1 - LAMBDA) pi "
                   "+ LAMBDA m_h while sigma > --pi-blend-lo, m_h = full attention's image-key share inside the step's "
                   "window (calibration mass_mean at the emulated radius; 0 = off)")
    p.add_argument("--pi-blend-lo", type=float, default=0.8, metavar="SIGMA", help="--pi-blend applies while sigma > SIGMA")
    p.add_argument("--native-open", type=int, default=0, metavar="K", help="Round 6 candidate d: first K steps at 1024^2 "
                   "with full attention, then upsample, renoise at the next sigma and continue at the target (0 = off)")
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
                bridge_ratio=args.bridge_ratio,
                lse_tau=args.lse_tau, scale_sched=args.scale_sched, scale_lo=args.scale_lo,
                scale_fixed=args.scale_fixed, win_pi_match=args.win_pi_match, win_pi_interp=args.win_pi_interp, win_heads_below=args.win_heads_below,
                text_mode=args.text_mode, text_mode_lo=args.text_mode_lo, text_gamma=args.text_gamma,
                text_clip=args.text_clip, text_bw_scale=args.text_bw_scale, ffl_near=args.ffl_near,
                ffl_head_frac=args.ffl_head_frac, ffl_fade_lo=args.ffl_fade_lo, ffl_outside=args.ffl_outside,
                ffl_coarse=args.ffl_coarse,
                text_bw_lo=args.text_bw_lo, open_pool=args.open_pool, open_pool_hi=args.open_pool_hi,
                morph_local=args.morph_local, morph_text=args.morph_text, qmix=args.qmix, qmix_lo=args.qmix_lo,
                qmix_tau=args.qmix_tau, qmix_gamma=getattr(args, "qmix_gamma", 1.0), qmix_far=getattr(args, "qmix_far", "out"), qmix_dose=getattr(args, "qmix_dose", 1.0), qmix_anchor=getattr(args, "qmix_anchor", "text"), text_value=getattr(args, "text_value", 1.0),
                text_value_lo=getattr(args, "text_value_lo", 0.0),
                text_value_subject=getattr(args, "text_value_subject", False),
                text_branch=getattr(args, "text_branch", "both"),
                sa_value=getattr(args, "subject_anchor", 1.0), sa_lo=getattr(args, "subject_anchor_lo", 0.55),
                sa_dilate=getattr(args, "subject_anchor_dilate", 2), sa_all=getattr(args, "subject_anchor_all", False),
                sa_qlo=float(getattr(args, "subject_anchor_q", (0.5, 0.95))[0]),
                sa_qhi=float(getattr(args, "subject_anchor_q", (0.5, 0.95))[1]),
                sa_dump=getattr(args, "subject_anchor_dump", ""), sa_smooth=getattr(args, "subject_anchor_smooth", 0),
                sa_rel=tuple(getattr(args, "subject_anchor_rel", None) or ()),
                sa_edge=getattr(args, "subject_anchor_edge", 0), pi_blend=getattr(args, "pi_blend", 0.0), pi_blend_lo=getattr(args, "pi_blend_lo", 0.8),
                gm_low=getattr(args, "guide_map", 0.0), gm_lo=getattr(args, "guide_map_lo", 0.58))


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
    if args.bridge_ratio:
        out["bridge_ratio"] = args.bridge_ratio
    if args.lse_tau:
        out["lse_tau"] = args.lse_tau
    if args.scale_sched:
        out.update(scale_sched=True, scale_lo=args.scale_lo)
    if args.scale_fixed:
        out["scale_fixed"] = [list(x) for x in args.scale_fixed]
    if args.win_pi_match:
        out["win_pi_match"] = True
    if args.win_pi_interp:
        out["win_pi_interp"] = True
    if args.win_heads_below is not None:
        out["win_heads_below"] = args.win_heads_below
    if args.text_mode:
        out.update(text_mode=args.text_mode, text_mode_lo=args.text_mode_lo, text_gamma=args.text_gamma,
                   text_clip=args.text_clip)
        if args.text_bw_scale != 1.0:
            out["text_bw_scale"] = args.text_bw_scale
        if args.subject_words:
            out["subject_words"] = args.subject_words
        for key in ("ffl_head_frac", "ffl_fade_lo", "ffl_outside", "ffl_coarse"):
            if getattr(args, key):
                out[key] = getattr(args, key)
        if args.ffl_near != "window":
            out["ffl_near"] = args.ffl_near
        if args.text_bw_lo is not None:
            out["text_bw_lo"] = args.text_bw_lo
    if args.open_pool:
        out.update(open_pool=args.open_pool, open_pool_hi=args.open_pool_hi)
    if getattr(args, "morph_local", ""):
        out["morph_local"] = args.morph_local
    if getattr(args, "morph_text", ""):
        out["morph_text"] = args.morph_text
    if getattr(args, "qmix", ""):
        out.update(qmix=args.qmix, qmix_lo=args.qmix_lo, qmix_tau=args.qmix_tau)
        if getattr(args, "qmix_gamma", 1.0) != 1.0:
            out["qmix_gamma"] = args.qmix_gamma
        if getattr(args, "qmix_far", "out") != "out":
            out["qmix_far"] = args.qmix_far
        if getattr(args, "qmix_dose", 1.0) != 1.0:
            out["qmix_dose"] = args.qmix_dose
        if getattr(args, "qmix_anchor", "text") != "text":
            out["qmix_anchor"] = args.qmix_anchor
    if getattr(args, "text_branch", "both") != "both":
        out["text_branch"] = args.text_branch
    if getattr(args, "guide_map", 0.0):
        out.update(guide_map=args.guide_map, guide_map_lo=args.guide_map_lo)
    if getattr(args, "subject_anchor", 1.0) != 1.0 or getattr(args, "guide_map", 0.0):
        out.update(subject_anchor=args.subject_anchor, subject_anchor_lo=args.subject_anchor_lo,
                   subject_anchor_dilate=args.subject_anchor_dilate)
        if getattr(args, "subject_anchor_all", False):
            out["subject_anchor_all"] = True
        if tuple(getattr(args, "subject_anchor_q", (0.5, 0.95))) != (0.5, 0.95):
            out["subject_anchor_q"] = list(args.subject_anchor_q)
        if getattr(args, "subject_anchor_smooth", 0):
            out["subject_anchor_smooth"] = args.subject_anchor_smooth
        if getattr(args, "subject_anchor_rel", None):
            out["subject_anchor_rel"] = list(args.subject_anchor_rel)
        if getattr(args, "subject_anchor_edge", 0):
            out["subject_anchor_edge"] = args.subject_anchor_edge
    if getattr(args, "text_value", 1.0) != 1.0:
        out.update(text_value=args.text_value, text_value_lo=args.text_value_lo)
        if getattr(args, "text_value_subject", False):
            out["text_value_subject"] = True
    if getattr(args, "pi_blend", 0.0):
        out.update(pi_blend=args.pi_blend, pi_blend_lo=args.pi_blend_lo)
    if getattr(args, "native_open", 0):
        out["native_open"] = args.native_open
    if args.anchor_lo:
        out.update(anchor_lo=args.anchor_lo, anchor_shape=args.anchor_shape, anchor_pool=args.anchor_pool,
                   anchor_refresh=args.anchor_refresh)
    return out
