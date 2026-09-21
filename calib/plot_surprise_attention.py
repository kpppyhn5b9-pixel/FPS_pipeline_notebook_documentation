"""Figure : surprise et attention, témoin vs attention-consolidation (traces trace.npz de run_surprise_attention)."""
import sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
off, cons, out = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
DUR, BREF = list(range(40, 49)), list(range(70, 79)); t_dur, t_bref = 90.0, 140.0
BLUE, ORANGE, INK, MUTED = '#2a78d6', '#eb6834', '#1f2933', '#6b7280'
plt.rcParams.update({'font.family': ['DejaVu Sans', 'sans-serif'], 'font.size': 10, 'axes.edgecolor': '#c9ced6'})
k = 30; sm = lambda x: np.convolve(x, np.ones(k) / k, mode='same')
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={'hspace': 0.15})
fig.patch.set_facecolor('white')
for ax, grp, title, t_ev in ((axes[0], DUR, 'Changement DURABLE (9 strates, fréquence propre × 1.15 à t = 90)', t_dur),
                             (axes[1], BREF, 'Événement BREF (9 strates, × 1.15 pendant 2 u.t. à t = 140)', t_bref)):
    for d, col, lab in ((off, BLUE, 'témoin'), (cons, ORANGE, 'attention-consolidation')):
        t = d['t']; u = d['U'][:, grp].mean(1); m = t >= 60
        ax.plot(t[m], sm(u)[m], color=col, linewidth=2, label=f'surprise, {lab}')
    t = cons['t']; s = cons['S'][:, DUR if grp is DUR else BREF].mean(1); m = t >= 60
    ax.fill_between(t[m], 0, sm(s)[m] * 0.1, color=ORANGE, alpha=0.18, label='attention posée (échelle ×0.1)')
    ax.axvline(t_ev, color=MUTED, linewidth=1, linestyle='--'); ax.set_ylim(0, 0.3); ax.set_ylabel('surprise')
    ax.set_title(title, loc='left', fontweight='bold', color=INK); ax.legend(loc='upper right', frameon=False); ax.grid(True, axis='y', color='#eef1f4')
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax = axes[2]
for d, col, lab in ((off, BLUE, 'témoin'), (cons, ORANGE, 'attention-consolidation')):
    t = d['t']; m = t >= 60; a = (d['Fl'][:, DUR] / d['Fn'][:, DUR]).mean(1)
    ax.plot(t[m], sm(a)[m], color=col, linewidth=2, label=lab)
ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=':'); ax.axvline(t_dur, color=MUTED, linewidth=1, linestyle='--')
ax.set_ylim(0.9, 1.02); ax.set_ylabel('mémoire longue / rythme courant'); ax.set_xlabel('temps (u.t.)')
ax.set_title('Assimilation du changement durable par la mémoire longue (1 = assimilé)', loc='left', fontweight='bold', color=INK)
ax.legend(loc='lower right', frameon=False); ax.grid(True, axis='y', color='#eef1f4')
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
fig.savefig(out, dpi=110, bbox_inches='tight'); print('saved', out)
