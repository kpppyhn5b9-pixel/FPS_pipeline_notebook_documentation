"""Figure du banc run_considerance_structures.py : un kymographe Rloc(t, n) par structure
(même seed), le projecteur tracé par-dessus, et en bas la surprise des strates liées.
usage: python plot_considerance_structures.py <sortie.png> <dossier1> [<dossier2> ...]
"""
import sys, os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
out = sys.argv[1]; dirs = sys.argv[2:]
fig, axes = plt.subplots(2, len(dirs), figsize=(4.6 * len(dirs), 7.2), sharex=True, gridspec_kw={'height_ratios': [3, 1.3]})
axes = np.atleast_2d(axes.T).T if len(dirs) > 1 else axes.reshape(2, 1)
for j, d in enumerate(dirs):
    K = np.load(os.path.join(d, 'kymo.npz')); S = json.load(open(os.path.join(d, 'summary.json')))
    t, R, s, U, th, ctr = K['t'], K['R'], K['s'], K['U'], K['t_hist'], K['ctr']
    ax = axes[0, j]
    ax.imshow(R.T, aspect='auto', origin='lower', extent=[t[0], t[-1], 0, R.shape[1]], vmin=0, vmax=1, cmap='magma')
    ok = np.isfinite(ctr); ax.plot(t[ok], ctr[ok], '.', color='cyan', ms=1.5)
    eff = np.mean([p['efficience'] for p in S['posts']]); eng = np.mean([p['engagement'] for p in S['posts']]); cont = np.mean([p['contagion'] for p in S['posts']])
    ax.set_title(f"{S['structure']} (seed {S['seed']})\nengagement +{eng:.2f}  contagion {cont:+.3f}  efficience {eff:.0f}", fontsize=10)
    if j == 0: ax.set_ylabel('strate n  —  Rloc(t, n)')
    ax2 = axes[1, j]
    zin = (s > 0.5)
    # surprise : strates sous le projecteur (à cet instant) vs les autres, lissée
    u_in = np.array([np.nanmean(U[i][zin[i]]) if zin[i].any() else np.nan for i in range(len(th))])
    u_out = np.array([np.nanmean(U[i][~zin[i]]) for i in range(len(th))])
    ax2.plot(th, u_out, color='0.6', lw=1, label='autres strates')
    ax2.plot(th, u_in, color='crimson', lw=1.2, label='strates liées')
    for p in (60, 80, 100, 120): ax2.axvline(p, color='0.8', lw=0.6)
    ax2.set_ylim(0, max(0.3, np.nanmax(u_out[th > 30]) * 1.3 if np.isfinite(u_out[th > 30]).any() else 0.3)); ax2.set_xlabel('t (u.t.)')
    if j == 0: ax2.set_ylabel('surprise (mémoire de soi)'); ax2.legend(fontsize=8, loc='upper left')
fig.suptitle("Le contexte se pose (t=60), migre (80, 100), disparaît (120) — même geste, même seed, quatre cascades", fontsize=11)
fig.tight_layout(); fig.savefig(out, dpi=110); print("saved", out)
