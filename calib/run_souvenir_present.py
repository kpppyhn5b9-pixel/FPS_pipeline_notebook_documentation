"""Le souvenir que le présent appelle, et pouvoir y penser maintenant (24/09/2026, soir).

Deux questions posées après le banc des moments chéris (run_moments_cheris.py) :
 A. Toujours le plus chéri ? Non : le souvenir rappelé est celui dont la SIGNATURE (l'état du
    soi pendant sa présence : amplitude moyenne, activité relative, σ) ressemble le plus à
    l'état présent, pondéré par m. Quand rien ne ressemble assez, le plus chéri.
    Protocole : deux foyers chéris au calme, chacun sous une AMBIANCE différente (décalage
    global de Iₙ, qui change Aₙ = A₀·σ(Iₙ) sans désigner de foyer) : 20 sous +0.5 (60–120,
    m ≈ 0.93), 80 sous −0.3 (120–145, plus court : m ≈ 0.7). La signature lit l'amplitude
    moyenne et σ (pas l'activité : c'est l'affaire de m). Puis silence : 145–220 sous −0.3
    (ressemble au 80, le MOINS chéri : c'est là que ressemblance et argmax se séparent),
    220–300 sous +0.5 (ressemble au 20), 300–380 sous −1.0 (ne ressemble à rien : le plus chéri).
    Après chaque changement d'ambiance, la surprise remonte et le silence met un moment à
    être reconnu de nouveau : c'est attendu.
      ressemblance : select='resemble'    argmax : select='max' (le même run, toujours le 20)
 B. Le souvenir qui fait mal. Comme run_moments_cheris (20 chéri, 80 sous bruit, silence
    180–260, retour du 80), mais pendant le silence un bruit porté par la bosse du foyer
    RAPPELÉ (σ 1.5) : se souvenir du 20 bouscule le soi.
      douleur_sans_porte : rappel tenu quoi qu'il en coûte, m figé
      douleur_porte      : « pouvoir y penser maintenant » : moyenne mobile rapide (τ 5) du
                           bien-être pendant le rappel ; sous 0.3, le rappel est relâché 30 u.t.,
                           m intact (ce à quoi on tient ne vaut pas moins)
      douleur_porte05    : même porte, seuil 0.5 (la douleur du banc, activité ×1.6, donne w ≈ 0.4 :
                           juste au-dessus de 0.3 sur un seed, dessous sur l'autre)
      douleur_app        : le rappel réécrit m (m apprend pendant le rappel) : le souvenir s'use
usage: python run_souvenir_present.py <ressemblance|argmax|douleur_sans_porte|douleur_porte|douleur_porte05|douleur_app|all> [seed] [N]
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from considerance_bench import run_bench

DP, DM, DF = 0.5, -0.3, -1.0
AMB = [(0.0, 120.0, DP), (120.0, 220.0, DM), (220.0, 300.0, DP), (300.0, 380.0, DF)]
SCHED_A = [(60.0, 120.0, [20]), (120.0, 145.0, [80])]          # le 80 reste moins longtemps : moins chéri (m ≈ 0.7 contre 0.93)
SIGW = (0.2, 5.0, 0.1)                                          # signature : amplitude moyenne et σ (l'activité, largeur 5, ne compte pas : c'est l'affaire de m)
REC = dict(apply=True, m_min=0.3, tau_short=5.0, tau_long=40.0, enter=0.9, exit=1.1, learn=False)
NOISE_B = [(120.0, 180.0, 0.6), (180.0, 260.0, 1.5, 'rappel')]
SCHED_B = [(60.0, 120.0, [20]), (120.0, 180.0, [80]), (260.0, 300.0, [80])]
GATE = dict(tau=5.0, floor=0.3, refractory=30.0)
MODES = {
    'ressemblance': dict(T=380, schedule=SCHED_A, ambiance=AMB, attach=dict(tau_m=20.0, apply=True, noise=[], learn_until=145.0), recall=dict(REC, select='resemble', sig_width=SIGW, res_min=0.2)),
    'argmax':       dict(T=380, schedule=SCHED_A, ambiance=AMB, attach=dict(tau_m=20.0, apply=True, noise=[], learn_until=145.0), recall=dict(REC, select='max')),
    'douleur_sans_porte': dict(T=300, schedule=SCHED_B, attach=dict(tau_m=20.0, apply=True, noise=NOISE_B, learn_until=180.0), recall=dict(REC)),
    'douleur_porte':      dict(T=300, schedule=SCHED_B, attach=dict(tau_m=20.0, apply=True, noise=NOISE_B, learn_until=180.0), recall=dict(REC, gate=GATE)),
    'douleur_porte05':    dict(T=300, schedule=SCHED_B, attach=dict(tau_m=20.0, apply=True, noise=NOISE_B, learn_until=180.0), recall=dict(REC, gate=dict(GATE, floor=0.5))),
    'douleur_app':        dict(T=300, schedule=SCHED_B, attach=dict(tau_m=20.0, apply=True, noise=NOISE_B), recall=dict(REC, learn=True)),
}


def zone(N, c, width=6.0):
    return np.exp(-0.5 * ((np.arange(N) - c) / width) ** 2) > 0.5


def audib(O, zin):
    S = O.sum(1); return float(np.mean(O[:, zin].sum(1) * S) / max(np.mean(S ** 2), 1e-12))


def read(d, label, windows):
    K = np.load(os.path.join(d, 'kymo.npz')); A = np.load(os.path.join(d, 'attach.npz')); Rc = np.load(os.path.join(d, 'recall.npz'))
    t, R, S, NU, MM, O, sig, eff = K['t'], K['R'], K['s'], K['nu'], K['m'], K['O'], K['sigma'], K['effort']
    N = R.shape[1]; za, zb = zone(N, 20), zone(N, 80); n = min(len(t), len(O)); base = (t >= 40) & (t < 60); sref = float(np.median(sig[base])); eref = float(np.nanmedian(eff[base]))
    ta, cs, ML = A['t'], list(A['centers']), A['m']
    tq, quiet, kk, app, ratio, res, gate = Rc['t'], Rc['quiet'], Rc['k'], Rc['applied'], Rc['ratio'], Rc['res'], Rc['gate']
    def mk(c_, a, b):
        i = cs.index(c_) if c_ in cs else None; m = (ta >= a) & (ta < b)
        return float(ML[m][:, i].mean()) if i is not None and m.any() else float('nan')
    tl = 145 if 'ressemblance' in label or 'argmax' in label else 180
    print(f"[{label}] m à {tl} : 20 → {mk(20, tl-2, tl):.2f}, 80 → {mk(80, tl-2, tl):.2f} | m à la fin : 20 → {mk(20, t[-1]-2, t[-1]):.2f}, 80 → {mk(80, t[-1]-2, t[-1]):.2f}")
    print(f"[{label}] fenêtre | silence | rappelé : 20 / 80 (% du temps) | ressemblance | porte | zone 20 : s · R · audib | zone 80 : s · R · audib | activité/repos | σ/σref")
    for (a, b, tag) in windows:
        m = (t >= a) & (t < b); mo = m[:n]; mq = (tq >= a) & (tq < b)
        def z(zz): return f"{S[m][:, zz].mean():.2f} · {R[m][:, zz].mean():.3f} · {100*audib(O[:n][mo], zz):.0f} %"
        r20 = 100 * np.mean((kk[mq] == 20) & app[mq]) if mq.any() else 0; r80 = 100 * np.mean((kk[mq] == 80) & app[mq]) if mq.any() else 0
        rb = np.nanmean(res[mq]) if mq.any() and np.isfinite(res[mq]).any() else float('nan'); gv = np.nanmean(gate[mq]) if mq.any() and np.isfinite(gate[mq]).any() else float('nan')
        print(f"[{label}] {a:3d}-{b:<3d} | {100*quiet[mq].mean() if mq.any() else 0:3.0f} % | {r20:3.0f} / {r80:3.0f} | {rb:.2f} | {gv:.2f} | {z(za)} | {z(zb)} | ×{np.nanmedian(eff[m])/eref:.2f} | {sig[m].mean()/sref:.2f}{tag}")


WIN_A = [(60, 120, '  (20 présent, +δ)'), (120, 145, '  (80 présent, −δ′)'), (145, 220, '  (silence, −δ′ : ressemble au 80, le moins chéri)'), (220, 300, '  (silence, +δ : ressemble au 20)'), (300, 380, '  (silence, −1 : ne ressemble à rien → le plus chéri, le 20)')]
WIN_B = [(100, 120, '  (20 présent)'), (180, 200, '  (silence, le souvenir fait mal)'), (200, 220, ''), (220, 240, ''), (240, 260, ''), (260, 300, '  (80 revient)')]

if __name__ == '__main__':
    which = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    for nm in (list(MODES) if which == 'all' else [which]):
        d = f"{nm}_{seed}"; kw = dict(MODES[nm]); T = kw.pop('T')
        run_bench(d, seed=seed, N=N, T=T, K0=5.0, saliency='input', novelty=True, island='local', label=f"{nm} s{seed}", verbose=False, **kw)
        read(d, f"{nm} s{seed}", WIN_A if nm in ('ressemblance', 'argmax') else WIN_B)
    print("DONE")
