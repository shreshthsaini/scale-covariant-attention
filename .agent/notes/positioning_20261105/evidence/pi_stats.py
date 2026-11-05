import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=False)
pop = z['populated']; sc = z['sigma_centers']
bins = np.where(pop)[0]
print('populated bins', bins.tolist(), 'sigma', np.round(sc[bins], 3).tolist())
for s in (2, 4):
    P = z[f's{s}_pi_head'][:, :, bins]          # (57, 24, nb)
    v = P.reshape(-1)
    print(f'\n== scale {s}: cells {v.size}')
    edges = [0, 1e-9, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1 - 1e-9, 1.0001]
    h, _ = np.histogram(v, bins=edges)
    labs = ['=0', '(0,.1)', '[.1,.2)', '[.2,.3)', '[.3,.4)', '[.4,.5)', '[.5,.6)', '[.6,.7)', '[.7,.8)', '[.8,.9)', '[.9,1)', '=1']
    print(' hist:', ', '.join(f'{l}:{c/v.size:.3f}' for l, c in zip(labs, h)))
    hm = P.mean(axis=2).reshape(-1)               # per-head mean over sigma
    hs = P.std(axis=2).reshape(-1)                # within-head std over sigma
    print(f' per-head mean over sigma: std across heads {hm.std():.3f}; within-head std over sigma: median {np.median(hs):.3f}, p90 {np.percentile(hs,90):.3f}')
    hh, _ = np.histogram(hm, bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001])
    print(' per-head-mean hist (deciles 0..1):', (hh / hm.size).round(3).tolist())
    # variance decomposition: layer vs head-within-layer vs sigma-within-head
    tot = v.var()
    layer_mean = P.mean(axis=(1, 2))              # (57,)
    head_mean = P.mean(axis=2)                    # (57,24)
    between_layer = ((layer_mean[:, None, None] - v.mean()) ** 2 * np.ones_like(P)).mean()
    within_layer_between_head = ((head_mean[:, :, None] - layer_mean[:, None, None]) ** 2 * np.ones_like(P)).mean()
    within_head = ((P - head_mean[:, :, None]) ** 2).mean()
    print(f' variance share: layer {between_layer/tot:.3f}, head-within-layer {within_layer_between_head/tot:.3f}, sigma-within-head {within_head/tot:.3f}')
    # heads near the extremes for most sigma bins
    frac_hi = (P > 0.9).mean(axis=2).reshape(-1); frac_lo = (P < 0.3).mean(axis=2).reshape(-1)
    print(f' heads with pi>0.9 in >=75% of bins: {(frac_hi>=.75).mean():.3f}; heads with pi<0.3 in >=75% of bins: {(frac_lo>=.75).mean():.3f}')
# rank stability across ratio
P2 = z['s2_pi_head'][:, :, bins]; P4 = z['s4_pi_head'][:, :, bins]
r = np.corrcoef(P2.reshape(-1), P4.reshape(-1))[0, 1]
rh = np.corrcoef(P2.mean(2).reshape(-1), P4.mean(2).reshape(-1))[0, 1]
print(f'\ncorr pi(r=2) vs pi(r=4): cells {r:.3f}, per-head means {rh:.3f}; mean diff (s4-s2) {float((P4-P2).mean()):.3f}; frac cells s4>s2 {float((P4>P2).mean()):.3f}')
# relation to measured near-field mass
mm = z['mass_mean']; radii = z['radii'].tolist()
print('radii', radii, 'mass_mean index 0 = text (assumed)')
for ri, rad in enumerate(radii):
    M = mm[ri + 1][:, :, bins]
    for s in (2, 4):
        P = z[f's{s}_pi_head'][:, :, bins]
        ok = np.isfinite(M) & (M > 0)
        print(f' radius {rad}: corr(pi s{s}, near mass) over cells {np.corrcoef(P[ok], M[ok])[0,1]:.3f}')
T = mm[0][:, :, bins]
print(f' text mass per head-cell: mean {T.mean():.3f}, median {np.median(T):.3f}, p90 {np.percentile(T,90):.3f}, p99 {np.percentile(T,99):.3f}')
for s in (2, 4):
    P = z[f's{s}_pi_head'][:, :, bins]
    print(f' corr(pi s{s}, text mass) {np.corrcoef(P.reshape(-1), T.reshape(-1))[0,1]:.3f}')
