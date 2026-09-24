"""Le banc de Gepetto (24/09/2026) : nouveauté ou importance ? Une priorité de maintien m,
fournie de l'extérieur, peut-elle ramener au premier plan un foyer devenu familier, sans
nouveauté sensorielle, et le relâcher quand on la retire ?

Sur considerance_bench.py : s = c · [ν + (1 − ν)·m]. Deux foyers (20 et 80) posés de
t = 60 à 220 et jamais déplacés : tous deux deviennent familiers (ν → 0 vers t ≈ 100).
  t 100–140 : m = 1 sur le foyer 20, 0 sur le 80
  t 140–180 : inversion (0 sur le 20, 1 sur le 80), rien ne change dans les entrées
  t 180–220 : m = 0 partout (retrait)
Runs : maintien (avec m) et sans_m (m absent, notre mécanisme actuel). Lectures par
fenêtre de 20 u.t. et par zone : s, ν, m, R, audibilité, σ/σref.
Attendu si le mécanisme tient : sous m le foyer familier se rallume sans nouveauté,
l'autre non ; à l'inversion l'attention change de foyer ; au retrait elle se relâche.
usage: python run_maintien.py <maintien|sans_m|all> [seed] [N] [T]
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from considerance_bench import run_bench

SCHED = [(60.0, 220.0, [20, 80])]
MAINT = [(100.0, 140.0, {20: 1.0, 80: 0.0}), (140.0, 180.0, {20: 0.0, 80: 1.0})]
MODES = {
    'maintien': dict(saliency='input', novelty=True, island='local', schedule=SCHED, maintain=MAINT),
    'sans_m':   dict(saliency='input', novelty=True, island='local', schedule=SCHED, maintain=None),
}


def zone(N, c, width=6.0):
    return np.exp(-0.5 * ((np.arange(N) - c) / width) ** 2) > 0.5


def audib(O, zin):
    S = O.sum(1); return float(np.mean(O[:, zin].sum(1) * S) / max(np.mean(S ** 2), 1e-12))


def read(d, label):
    K = np.load(os.path.join(d, 'kymo.npz')); t, R, S, NU, MM, O, sig, COST = K['t'], K['R'], K['s'], K['nu'], K['m'], K['O'], K['sigma'], K['cost']
    N = R.shape[1]; za = zone(N, 20); zb = zone(N, 80); n = min(len(t), len(O)); base = (t >= 40) & (t < 60); sref = float(np.median(sig[base]))
    print(f"[{label}] fenêtre | zone 20 : s · ν · m · R · audibilité | zone 80 : s · ν · m · R · audibilité | σ/σref | coût")
    for a in range(60, 220, 20):
        m = (t >= a) & (t < a + 20); mo = m[:n]
        def z(zz): return f"{S[m][:, zz].mean():.2f} · {NU[m][:, zz].mean():.2f} · {MM[m][:, zz].mean():.2f} · {R[m][:, zz].mean():.3f} · {100*audib(O[:n][mo], zz):.0f} %"
        tag = '  (m : 20)' if 100 <= a < 140 else ('  (m : 80)' if 140 <= a < 180 else ('  (retrait)' if a >= 180 else ''))
        print(f"[{label}] {a:3d}-{a+20:<3d} | {z(za)} | {z(zb)} | {sig[m].mean()/sref:.2f} | {COST[m][:, za | zb].mean():.4f}{tag}")


if __name__ == '__main__':
    which = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 100; T = int(sys.argv[4]) if len(sys.argv) > 4 else 230
    for nm in (list(MODES) if which == 'all' else [which]):
        d = f"{nm}_{seed}"; run_bench(d, seed=seed, N=N, T=T, K0=5.0, label=f"{nm} s{seed}", verbose=False, **MODES[nm])
        read(d, f"{nm} s{seed}")
    print("DONE")
