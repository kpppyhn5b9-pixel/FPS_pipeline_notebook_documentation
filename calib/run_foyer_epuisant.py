"""Le foyer épuisant (24/09/2026) : un foyer qui est LUI-MÊME la source de l'effort fait-il
baisser m par le même mécanisme (association, pas récompense) ?

Sur considerance_bench.py (option `attach`, bruit porté par la bosse du foyer : 4e élément
'foyer'). Le bruit n'est plus sur toute l'entrée mais sur les strates du foyer présent, à
l'échelle de sa bosse : c'est lui qui bouscule. La saillance lit la bosse propre (sans bruit).
  epuisant  : 20 au calme (60–120) ; 80 présent 120–180, bruyant SUR LUI-MÊME ; extinction ;
              retour des deux au calme 220–300, m figé à 180. Même lecture que run_attachement.
  lassitude : 20 au calme (60–120, il devient chéri) ; le MÊME foyer 20 reste présent 120–180
              mais devient bruyant sur lui-même ; extinction ; retour 20 + 80 (jamais vu) 220–300,
              m figé à 180. Attendu si le mécanisme est symétrique : m(20) monte puis redescend.
  temoin    : 20 au calme (60–120) ; 80 présent 120–180 SANS bruit ; sert à lire l'effort propre
              du bruit local (activité) contre une présence calme.
usage: python run_foyer_epuisant.py <epuisant|lassitude|temoin|all> [seed] [N] [T] [sd]
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from considerance_bench import run_bench
from run_attachement import read


def modes(sd):
    loc = [(120.0, 180.0, sd, 'foyer')]
    return {
        'epuisant':  dict(schedule=[(60.0, 120.0, [20]), (120.0, 180.0, [80]), (220.0, 300.0, [20, 80])],
                          attach=dict(tau_m=20.0, apply=True, noise=loc, learn_until=180.0)),
        'lassitude': dict(schedule=[(60.0, 120.0, [20]), (120.0, 180.0, [20]), (220.0, 300.0, [20, 80])],
                          attach=dict(tau_m=20.0, apply=True, noise=loc, learn_until=180.0)),
        'temoin':    dict(schedule=[(60.0, 120.0, [20]), (120.0, 180.0, [80]), (220.0, 300.0, [20, 80])],
                          attach=dict(tau_m=20.0, apply=True, noise=[], learn_until=180.0)),
    }


def trace_m(d, label):
    A = np.load(os.path.join(d, 'attach.npz')); ta, cs, ML, W = A['t'], list(A['centers']), A['m'], A['w']
    print(f"[{label}] m au fil du temps (t : w moyen sur 10 u.t. | m 20 | m 80)")
    for a in range(60, 180, 10):
        m = (ta >= a) & (ta < a + 10)
        cols = ' | '.join(f"m {c_} {ML[m][:, i].mean():.2f}" for i, c_ in enumerate(cs))
        print(f"[{label}]   {a:3d}-{a+10:<3d} : w {np.nanmean(W[m]):.2f} | {cols}")


if __name__ == '__main__':
    which = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 100; T = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    sd = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
    M = modes(sd)
    for nm in (list(M) if which == 'all' else [which]):
        d = f"{nm}_{seed}_sd{sd:g}"
        run_bench(d, seed=seed, N=N, T=T, K0=5.0, saliency='input', novelty=True, island='local', label=f"{nm} s{seed}", verbose=False, **M[nm])
        trace_m(d, f"{nm} s{seed} sd{sd:g}")
        read(d, f"{nm} s{seed} sd{sd:g}", 20)
    print("DONE")
