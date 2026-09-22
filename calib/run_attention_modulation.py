"""Le contexte modulé (22/09/2026) : deux questions posées au banc considerance_bench.py.

1. HABITUATION / RÉORIENTATION — un contexte tenu 60 u.t. sur la strate 50 (t 60–120), puis
   déplacé sur la 20 (t 120–150), puis éteint. Saillance = entrée Iₙ (miroir) contre
   entrée × nouveauté pour soi (ν = (uₙ/médiane − 1)/2 borné à [0, 1]).
   Attendu si la modulation est juste : l'îlot 50 s'allume puis se relâche de lui-même quand
   la mémoire longue a assimilé ; l'îlot 20 se rallume quand le contexte change de place.
2. DEUX CONTEXTES À LA FOIS — bosses sur 20 et 80 (t 60–120). Champ moyen global (un seul Z)
   contre champ moyen par îlot (un Z par composante connexe). Attendu : le global tire les
   deux îlots vers une même phase (cohérence non locale, |Z_union| ≈ moyenne des |Z_k|) ;
   le local les laisse indépendants (|Z_union| / moyenne ≈ 2/π ≈ 0.64), σ intact.
usage: python run_attention_modulation.py <mode|all> [seed] [N] [T] [K0]
  modes : hold_input, hold_novelty, two_global, two_local, two_K0
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from considerance_bench import run_bench

HOLD = [(60.0, 120.0, [50]), (120.0, 150.0, [20])]
TWO = [(60.0, 120.0, [20, 80])]
MODES = {
    'hold_input':   dict(saliency='input', schedule=HOLD),
    'hold_novelty': dict(saliency='input', schedule=HOLD, novelty=True),
    'two_global':   dict(saliency='input', schedule=TWO, island='global'),
    'two_local':    dict(saliency='input', schedule=TWO, island='local'),
    'two_K0':       dict(saliency='input', schedule=TWO, K0=0.0, kappa=0.0),
}


def zone(N, c, width=6.0):
    return np.exp(-0.5 * ((np.arange(N) - c) / width) ** 2) > 0.5


def audib(O, zin):
    S = O.sum(1); return float(np.mean(O[:, zin].sum(1) * S) / max(np.mean(S ** 2), 1e-12))


def read_hold(d, label):
    K = np.load(os.path.join(d, 'kymo.npz')); t, R, S, U, O, NU, COST, sig = K['t'], K['R'], K['s'], K['U'], K['O'], K['nu'], K['cost'], K['sigma']
    N = R.shape[1]; z50 = zone(N, 50); z20 = zone(N, 20); n = min(len(t), len(O)); base = (t >= 40) & (t < 60)
    sref = float(np.median(sig[base])); R50_0 = R[base][:, z50].mean(); R20_0 = R[base][:, z20].mean()
    print(f"[{label}] avant : R zone 50 {R50_0:.3f}, zone 20 {R20_0:.3f} | σ_ref {sref:.3f}")
    print(f"[{label}] fenêtre | zone 50 : s · ν · R · surprise · coût · audibilité | zone 20 : s · R · audibilité | R ailleurs | σ/σref")
    for a in range(60, 170, 10):
        m = (t >= a) & (t < a + 10); mo = m[:n]
        z50s = S[m][:, z50].mean(); z20s = S[m][:, z20].mean(); others = ~(z50 | z20)
        print(f"[{label}] {a:3d}-{a+10:<3d} | {z50s:.2f} · {NU[m][:, z50].mean():.2f} · {R[m][:, z50].mean():.3f} · {np.nanmean(U[m][:, z50]):.3f} · {COST[m][:, z50].mean():.4f} · {100*audib(O[:n][mo], z50):.0f} % | {z20s:.2f} · {R[m][:, z20].mean():.3f} · {100*audib(O[:n][mo], z20):.0f} % | {R[m][:, others].mean():.3f} | {sig[m].mean()/sref:.2f}")


def read_two(d, label):
    K = np.load(os.path.join(d, 'kymo.npz')); t, R, S, O, TH, sig, COST = K['t'], K['R'], K['s'], K['O'], K['theta'], K['sigma'], K['cost']
    N = R.shape[1]; za = zone(N, 20); zb = zone(N, 80); n = min(len(t), len(O)); base = (t >= 40) & (t < 60); sref = float(np.median(sig[base]))
    print(f"[{label}] fenêtre | R îlot 20 · R îlot 80 · R ailleurs · R global | |Z20| · |Z80| · |Z_union| / moy(|Z_k|)  (≈1 : verrouillés ensemble ; ≈0.64 : indépendants) | σ/σref | audibilité 20 · 80 · union | coût")
    for (a, b) in ((40, 60), (65, 90), (90, 120), (125, 150), (150, 170)):
        m = (t >= a) & (t < b); mo = m[:n]
        z = np.exp(1j * TH[m]); Za = np.abs(z[:, za].mean(1)); Zb = np.abs(z[:, zb].mean(1)); Zu = np.abs(z[:, za | zb].mean(1))
        ratio = float(np.mean(Zu) / max(np.mean((Za + Zb) / 2), 1e-9))
        print(f"[{label}] {a:3d}-{b:<3d} | {R[m][:, za].mean():.3f} · {R[m][:, zb].mean():.3f} · {R[m][:, ~(za | zb)].mean():.3f} · {R[m].mean():.3f} | {Za.mean():.2f} · {Zb.mean():.2f} · {ratio:.2f} | {sig[m].mean()/sref:.2f} | {100*audib(O[:n][mo], za):.0f} % · {100*audib(O[:n][mo], zb):.0f} % · {100*audib(O[:n][mo], za | zb):.0f} % | {COST[m][:, za | zb].mean():.4f}")


if __name__ == '__main__':
    which = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    T = int(sys.argv[4]) if len(sys.argv) > 4 else 170
    K0 = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0
    for nm in (list(MODES) if which == 'all' else [which]):
        kw = dict(MODES[nm]); kw.setdefault('K0', K0)
        d = f"{nm}_{seed}"; run_bench(d, seed=seed, N=N, T=T, label=f"{nm} s{seed}", verbose=False, **kw)
        (read_hold if nm.startswith('hold') else read_two)(d, f"{nm} s{seed}")
    print("DONE")
