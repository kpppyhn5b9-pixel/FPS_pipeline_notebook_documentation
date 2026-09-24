"""La mémoire des moments chéris (24/09/2026) : quand l'entrée se tait, le système peut-il se
rappeler un foyer chéri et s'y poser de lui-même ? Et à quel prix ?

Sur considerance_bench.py (options `attach` + `recall`). Apprentissage comme au banc de
l'attachement : 20 présent au calme (60–120, il devient chéri), 80 présent sous bruit global
(120–180, il ne l'est pas). Puis l'entrée se tait (180–260) : aucun foyer, plus de bruit. Le
silence est reconnu quand aucun foyer n'est désigné ET que la surprise du chœur (moyenne de u,
lissée sur 5 u.t.) est tombée sous son niveau habituel (la même, lissée sur 40 u.t.) : entrée en
silence sous 0.9 × habituel, sortie au-dessus de 1.1 × (hystérésis). Le nouveau est assimilé quand
on est moins surpris que d'habitude. (Première tentative, abandonnée : « max ν sous un plancher » ;
ν est relative à la médiane et il y a toujours une strate plus surprise que les autres, le plancher
n'est jamais atteint.) Le système propose alors comme contexte interne le foyer le plus chéri
(m ≥ m_min), même formule s = c · [ν + (1 − ν)·m], c venant du dedans. Enfin le 80 revient
dans l'entrée (260–300) : l'extérieur doit reprendre la main sur le souvenir.
  rappel     : mémoire appliquée, m figé à 180 (lecture propre)
  temoin     : mémoire présente, silence détecté, foyer choisi, rien appliqué
  rappel_app : mémoire appliquée, m du foyer rappelé continue de s'apprendre pendant le rappel
               (se souvenir peut-il défaire ou renforcer le souvenir ?)
Lectures, en silence (180–260) : quand le silence est reconnu, quel foyer est choisi, R et
audibilité de la zone rappelée contre l'autre, activité relative au repos (le coût de se
souvenir, à comparer au coût de se poser sur le 20 présent : ×1.3), σ/σref. Puis le retour du 80.
usage: python run_moments_cheris.py <rappel|temoin|rappel_app|all> [seed] [N] [T]
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from considerance_bench import run_bench

NOISE = [(120.0, 180.0, 0.6)]
SCHED = [(60.0, 120.0, [20]), (120.0, 180.0, [80]), (260.0, 300.0, [80])]
REC = dict(apply=True, m_min=0.3, tau_short=5.0, tau_long=40.0, enter=0.9, exit=1.1, learn=False)
MODES = {
    'rappel':     dict(schedule=SCHED, attach=dict(tau_m=20.0, apply=True, noise=NOISE, learn_until=180.0), recall=dict(REC)),
    'temoin':     dict(schedule=SCHED, attach=dict(tau_m=20.0, apply=True, noise=NOISE, learn_until=180.0), recall=dict(REC, apply=False)),
    'rappel_app': dict(schedule=SCHED, attach=dict(tau_m=20.0, apply=True, noise=NOISE), recall=dict(REC, learn=True)),
}


def zone(N, c, width=6.0):
    return np.exp(-0.5 * ((np.arange(N) - c) / width) ** 2) > 0.5


def audib(O, zin):
    S = O.sum(1); return float(np.mean(O[:, zin].sum(1) * S) / max(np.mean(S ** 2), 1e-12))


def read(d, label):
    K = np.load(os.path.join(d, 'kymo.npz')); A = np.load(os.path.join(d, 'attach.npz')); Rc = np.load(os.path.join(d, 'recall.npz'))
    t, R, S, NU, MM, O, sig, eff = K['t'], K['R'], K['s'], K['nu'], K['m'], K['O'], K['sigma'], K['effort']
    N = R.shape[1]; za, zb = zone(N, 20), zone(N, 80); n = min(len(t), len(O)); base = (t >= 40) & (t < 60); sref = float(np.median(sig[base])); eref = float(np.nanmedian(eff[base]))
    ta, cs, ML = A['t'], list(A['centers']), A['m']
    tq, quiet, kk, app, ratio = Rc['t'], Rc['quiet'], Rc['k'], Rc['applied'], Rc['ratio']
    def mk(c_, a, b):
        i = cs.index(c_) if c_ in cs else None; m = (ta >= a) & (ta < b)
        return float(ML[m][:, i].mean()) if i is not None and m.any() else float('nan')
    first_q = float(tq[quiet][0]) if quiet.any() else float('nan'); first_k = float(tq[kk >= 0][0]) if (kk >= 0).any() else float('nan')
    sil = (tq >= 180) & (tq < 260)
    print(f"[{label}] m à 180 : 20 → {mk(20, 178, 180):.2f}, 80 → {mk(80, 178, 180):.2f} | silence reconnu à t = {first_q:.1f}, foyer choisi à t = {first_k:.1f} | part du silence (180–260) reconnue {100*quiet[sil].mean():.0f} %, foyer choisi {100*(kk[sil]>=0).mean():.0f} % (20 : {100*(kk[sil]==20).mean():.0f} %, 80 : {100*(kk[sil]==80).mean():.0f} %), appliqué {100*app[sil].mean():.0f} %")
    print(f"[{label}] m à 260 : 20 → {mk(20, 258, 260):.2f}, 80 → {mk(80, 258, 260):.2f}")
    print(f"[{label}] fenêtre | zone 20 : s · ν · m · R · audib | zone 80 : s · ν · m · R · audib | activité/repos | σ/σref | surprise/habituel | silence")
    for (a, b) in ((100, 120), (180, 200), (200, 220), (220, 240), (240, 260), (260, 280), (280, 300)):
        m = (t >= a) & (t < b); mo = m[:n]; mq = (tq >= a) & (tq < b)
        def z(zz): return f"{S[m][:, zz].mean():.2f} · {NU[m][:, zz].mean():.2f} · {MM[m][:, zz].mean():.2f} · {R[m][:, zz].mean():.3f} · {100*audib(O[:n][mo], zz):.0f} %"
        tag = '  (20 présent, calme)' if a < 120 else ('  (silence)' if a < 260 else '  (80 revient)')
        print(f"[{label}] {a:3d}-{b:<3d} | {z(za)} | {z(zb)} | ×{np.nanmedian(eff[m])/eref:.2f} | {sig[m].mean()/sref:.2f} | {np.nanmean(ratio[mq]) if mq.any() else float('nan'):.2f} | {100*quiet[mq].mean() if mq.any() else 0:.0f} %{tag}")


if __name__ == '__main__':
    which = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 100; T = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    for nm in (list(MODES) if which == 'all' else [which]):
        d = f"{nm}_{seed}"
        run_bench(d, seed=seed, N=N, T=T, K0=5.0, saliency='input', novelty=True, island='local', label=f"{nm} s{seed}", verbose=False, **MODES[nm])
        read(d, f"{nm} s{seed}")
    print("DONE")
