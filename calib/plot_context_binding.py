"""Figure : le projecteur de contexte et la cohérence qui le suit (run G : champ moyen d'îlot, K₀=2, κ=1)."""
import sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
d = np.load(sys.argv[1]); out = sys.argv[2]
t, R, s, sig, disp, t_h = d['t'], d['R'], d['s'], d['sigma'], d['disp'], d['t_hist']
N = R.shape[1]; t_on = 60.0
# palette de référence (dataviz) : bleu séquentiel 100→700, catégoriel bleu / orange
seq = LinearSegmentedColormap.from_list('blue_seq', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
BLUE, ORANGE, INK, MUTED = '#2a78d6', '#eb6834', '#1f2933', '#6b7280'
plt.rcParams.update({'font.family': ['DejaVu Sans', 'sans-serif'], 'font.size': 10, 'axes.edgecolor': '#c9ced6', 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1.4, 1.2], 'hspace': 0.12})
fig.patch.set_facecolor('white')
# 1. kymographe Rloc(t, n) + projecteur
ax = axes[0]
im = ax.imshow(R.T, aspect='auto', origin='lower', cmap=seq, vmin=0, vmax=1, extent=[t[0], t[-1], -0.5, N - 0.5], interpolation='nearest')
ctr = np.array([np.argmax(row) if row.max() > 0 else np.nan for row in s], dtype=float)
ax.plot(t, ctr, color=ORANGE, linewidth=2.5, label='projecteur de contexte (strate éclairée)')
ax.axvline(t_on, color=MUTED, linewidth=1, linestyle='--'); ax.text(t_on + 0.6, N - 4, 'allumage', color=MUTED, fontsize=9, va='top')
ax.set_ylabel('strate n'); ax.set_title('La cohérence locale suit le projecteur (liage champ moyen d\'îlot, K₀ = 2, κ = 1)', loc='left', color=INK, fontweight='bold')
ax.legend(loc='upper right', frameon=False)
cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03); cb.set_label('Rloc (0 = voisines opposées, 1 = alignées)'); cb.outline.set_visible(False)
# 2. cohérence dedans / dehors (lissée 2 u.t.)
ax = axes[1]
inside = np.array([R[i][s[i] > 0.5].mean() if (s[i] > 0.5).any() else np.nan for i in range(len(t))])
outside = np.array([R[i][s[i] < 0.05].mean() if (s[i] < 0.05).any() else R[i].mean() for i in range(len(t))])
k = 20; sm = lambda x: np.convolve(np.nan_to_num(x, nan=np.nanmean(x)), np.ones(k) / k, mode='same')
ax.plot(t, sm(outside), color=BLUE, linewidth=2, label='hors du projecteur')
ax.plot(t[t >= t_on], sm(inside)[t >= t_on], color=ORANGE, linewidth=2, label='sous le projecteur')
ax.axhline(2 / np.pi, color=MUTED, linewidth=1, linestyle=':'); ax.text(t[0] + 1, 2 / np.pi + 0.02, 'repos : 2/π = 0.64 (battement)', color=MUTED, fontsize=9)
ax.axvline(t_on, color=MUTED, linewidth=1, linestyle='--')
ax.set_ylim(0.3, 1.05); ax.set_ylabel('Rloc moyen'); ax.legend(loc='lower right', frameon=False, ncol=2); ax.grid(True, axis='y', color='#eef1f4')
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
# 3. gardiens, en × de leur valeur au repos (un seul axe)
ax = axes[2]
ref_sig = np.median(sig[t < t_on]); ax.plot(t, sm(sig / ref_sig), color=BLUE, linewidth=2, label='contraste chimère σ_Rloc')
dd = disp.copy(); ok = np.isfinite(dd); ref_d = np.nanmedian(dd[(t_h < t_on) & (t_h > 40)])
ax.plot(t_h[ok], dd[ok] / ref_d, color=ORANGE, linewidth=2, label='dispersion (std ΣO / √N, lissée)')
ax.axhspan(1 / 1.35, 1.35, color='#eef1f4', zorder=0); ax.text(t[0] + 1, 1.36, 'bande saine de la cloche de dispersion (×1.35)', color=MUTED, fontsize=9, va='bottom')
ax.axhline(1, color=MUTED, linewidth=1, linestyle=':'); ax.axvline(t_on, color=MUTED, linewidth=1, linestyle='--')
ax.set_ylim(0.6, 1.7); ax.set_ylabel('× valeur au repos'); ax.set_xlabel('temps (u.t.)'); ax.legend(loc='upper left', frameon=False, ncol=2); ax.grid(True, axis='y', color='#eef1f4')
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
fig.savefig(out, dpi=110, bbox_inches='tight'); print('saved', out)
