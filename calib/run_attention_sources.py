"""Qui désigne l'îlot ? (22/09/2026) Les sources de saillance du geste de considération,
sur le banc considerance_bench.py, sans toucher au pipeline.

  projector   la saillance est le projecteur (référence d'hier, K₀ = 5)
  context     le contexte RÉEL : une bosse d'entrée Iₙ suit les postes, la saillance est lue
              dans l'amplitude (présent lissé de la mémoire de soi) — se relâche quand Iₙ part
  context_K0  témoin : même bosse d'entrée, aucun geste (l'attention de gain seule)
  surprise    la surprise de la mémoire de soi, avec un ÉVÉNEMENT (f₀ × 1.3 durable sur les
              strates 40–44 à t = 90), sans contexte — s'allume-t-elle là, et se relâche-t-elle ?
  event_K0    témoin : l'événement seul, aucun geste
  both        contexte (postes) + surprise (événement) : cohabitation
  input       la saillance lue sur Iₙ lui-même (bosse active) : gain + cohérence
  context_x3  contexte d'amplitude avec une bosse trois fois plus forte (+1.5)
  surprise_x2 surprise avec un événement franc (f₀ × 2)
usage: python run_attention_sources.py <mode|all> [seed] [N] [T] [K0]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from considerance_bench import run_bench

EVENT = dict(t=90.0, strates=range(40, 45), factor=1.3)
MODES = {
    'projector':  dict(saliency='projector'),
    'context':    dict(saliency='context'),
    'context_K0': dict(saliency='context', K0=0.0, kappa=0.0),
    'surprise':   dict(saliency='surprise', posts=False, event=EVENT),
    'event_K0':   dict(saliency='surprise', posts=False, event=EVENT, K0=0.0, kappa=0.0),
    'both':       dict(saliency='both', event=EVENT),
    'input':      dict(saliency='input'),
    'context_x3': dict(saliency='context', ctx_gain=1.5),
    'surprise_x2': dict(saliency='surprise', posts=False, event=dict(t=90.0, strates=range(40, 45), factor=2.0)),
}

if __name__ == '__main__':
    which = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    T = int(sys.argv[4]) if len(sys.argv) > 4 else 170
    K0 = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0
    for nm in (list(MODES) if which == 'all' else [which]):
        kw = dict(MODES[nm]); kw.setdefault('K0', K0)
        run_bench(f"{nm}_{seed}", seed=seed, N=N, T=T, label=f"{nm} s{seed}", **kw)
    print("DONE")
