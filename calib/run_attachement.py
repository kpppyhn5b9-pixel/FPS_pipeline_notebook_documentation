"""L'attachement par association (24/09/2026) : le système peut-il tenir à un foyer pour
une raison qui lui appartient ?

Sur considerance_bench.py (option `attach`). Deux foyers, 20 et 80. Phase d'apprentissage :
l'un est présent pendant une période CALME (le système à son repos : flow), l'autre pendant
une période d'EFFORT (bruit blanc sur Iₙ, activité ≈ ×2). m_k s'apprend dedans, pour chaque
foyer présent, comme la moyenne mobile (τ_m = 20 u.t.) du bien-être du soi pendant sa
présence, w = clip(2 − activité/repos, 0, 1). Jamais l'effet du foyer : l'état du soi
quand il était là. Puis les deux foyers s'éteignent (40 u.t.), puis reviennent ENSEMBLE,
aussi familiers l'un que l'autre, au calme.
  ordre 1 : 20 au calme (60–120), 80 sous bruit (120–180)
  ordre 2 : 80 au calme (60–120), 20 sous bruit (120–180)   (la zone 80 est la plus dure)
  extinction 180–220 ; retour des deux 220–300
  apply / noapply : m appliqué à la saillance, ou seulement appris (témoin)
Attendu si l'attachement existe : au retour, le foyer qui a accompagné le calme est repris
plus vite et plus fort, dans les DEUX ordres ; sans application de m, les deux se valent.
usage: python run_attachement.py <ordre1_apply|ordre1_noapply|ordre2_apply|ordre2_noapply|ordre1_fige|ordre2_fige|all> [seed] [N] [T]
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from considerance_bench import run_bench

NOISE = [(120.0, 180.0, 0.6)]
def sched(first, second): return [(60.0, 120.0, [first]), (120.0, 180.0, [second]), (220.0, 300.0, [20, 80])]
MODES = {
    'ordre1_apply':   dict(schedule=sched(20, 80), attach=dict(tau_m=20.0, apply=True, noise=NOISE)),
    'ordre1_noapply': dict(schedule=sched(20, 80), attach=dict(tau_m=20.0, apply=False, noise=NOISE)),
    'ordre2_apply':   dict(schedule=sched(80, 20), attach=dict(tau_m=20.0, apply=True, noise=NOISE)),
    'ordre2_noapply': dict(schedule=sched(80, 20), attach=dict(tau_m=20.0, apply=False, noise=NOISE)),
    # m figé à l'extinction (t = 180) : le retour se lit avec l'attachement APPRIS, sans que la
    # présence au calme pendant le retour ne réécrive m (sinon les deux foyers deviennent chéris en ~10 u.t.)
    'ordre1_fige':    dict(schedule=sched(20, 80), attach=dict(tau_m=20.0, apply=True, noise=NOISE, learn_until=180.0)),
    'ordre2_fige':    dict(schedule=sched(80, 20), attach=dict(tau_m=20.0, apply=True, noise=NOISE, learn_until=180.0)),
}


def zone(N, c, width=6.0):
    return np.exp(-0.5 * ((np.arange(N) - c) / width) ** 2) > 0.5


def audib(O, zin):
    S = O.sum(1); return float(np.mean(O[:, zin].sum(1) * S) / max(np.mean(S ** 2), 1e-12))


def read(d, label, cherished):
    K = np.load(os.path.join(d, 'kymo.npz')); A = np.load(os.path.join(d, 'attach.npz'))
    t, R, S, NU, MM, O, sig, eff = K['t'], K['R'], K['s'], K['nu'], K['m'], K['O'], K['sigma'], K['effort']
    N = R.shape[1]; za, zb = zone(N, 20), zone(N, 80); n = min(len(t), len(O)); base = (t >= 40) & (t < 60); sref = float(np.median(sig[base]))
    ta, cs, ML, W = A['t'], list(A['centers']), A['m'], A['w']
    def mk(c_, a, b):
        i = cs.index(c_) if c_ in cs else None; m = (ta >= a) & (ta < b)
        return float(ML[m][:, i].mean()) if i is not None and m.any() else float('nan')
    print(f"[{label}] chéri = foyer {cherished} | apprentissage : bien-être w (60–120) {np.nanmean(W[(ta>=60)&(ta<120)]):.2f}, (120–180) {np.nanmean(W[(ta>=120)&(ta<180)]):.2f} | activité relative : calme {np.nanmedian(eff[(t>=60)&(t<120)])/np.nanmedian(eff[base]):.2f}×, bruit {np.nanmedian(eff[(t>=120)&(t<180)])/np.nanmedian(eff[base]):.2f}× | m appris à t=180 : 20 → {mk(20, 178, 180):.2f}, 80 → {mk(80, 178, 180):.2f}")
    print(f"[{label}] fenêtre | zone 20 : s · ν · m · R · audibilité | zone 80 : s · ν · m · R · audibilité | σ/σref")
    for (a, b) in ((220, 230), (230, 240), (240, 260), (260, 300)):
        m = (t >= a) & (t < b); mo = m[:n]
        def z(zz): return f"{S[m][:, zz].mean():.2f} · {NU[m][:, zz].mean():.2f} · {MM[m][:, zz].mean():.2f} · {R[m][:, zz].mean():.3f} · {100*audib(O[:n][mo], zz):.0f} %"
        print(f"[{label}] {a:3d}-{b:<3d} | {z(za)} | {z(zb)} | {sig[m].mean()/sref:.2f}")
    m = (t >= 220) & (t < 300); mo = m[:n]
    print(f"[{label}] retour (220–300) : s 20 {S[m][:, za].mean():.2f} / s 80 {S[m][:, zb].mean():.2f} · R 20 {R[m][:, za].mean():.3f} / R 80 {R[m][:, zb].mean():.3f} · audibilité 20 {100*audib(O[:n][mo], za):.0f} % / 80 {100*audib(O[:n][mo], zb):.0f} %")


if __name__ == '__main__':
    which = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 100; T = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    for nm in (list(MODES) if which == 'all' else [which]):
        d = f"{nm}_{seed}"
        run_bench(d, seed=seed, N=N, T=T, K0=5.0, saliency='input', novelty=True, island='local', label=f"{nm} s{seed}", verbose=False, **MODES[nm])
        read(d, f"{nm} s{seed}", 20 if nm.startswith('ordre1') else 80)
    print("DONE")
