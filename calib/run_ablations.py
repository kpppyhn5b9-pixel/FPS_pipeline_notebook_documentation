"""Carte d'ablation (21/09/2026) : ce que chaque ÉLÉMENT DE DESIGN de la FPS apporte,
mesuré par le banc « considérance » (considerance_bench.py). Pas les métriques :
les choix qui distinguent la FPS d'un réseau d'oscillateurs générique (Kuramoto,
Stuart-Landau) sont retirés un à un, et remplacés par leur forme générique.

  reference           la FPS telle quelle (cascade dorée, K₀ = 5)
  sans_respiration    r(t) = φ fixe dans la cascade (plus de ε·sin)
  phases_statiques    φₙ(t) = signature seule (plus de spirale personnelle ni d'influences)
  sans_signatures     signatures pentagonales → toutes à 0
  signatures_hasard   signatures → phases tirées au hasard (générique)
  sans_couplage       c = 1e-6 (≈ 0) : plus de S_i → Δfₙ (la route amplitude → fréquence)
  env_statique        Aₙ = A₀·σ(Iₙ) : plus d'enveloppe sur l'erreur (la « pulsation »)
  latence_statique    γ = 1 : plus de latence adaptative
  G_identite          régulation « simple » : G = erreur (plus d'archétype adaptatif)
  beta_statique       βₙ fixe : plus de plasticité βₙ(t)
  E_ancien            Eₙ = passe-bas de Oₙ (plus de mémoire de soi)
  generique           tout retiré à la fois + pas de cascade (fréquences × 2.5 pour l'échelle) :
                      la chaîne d'oscillateurs la plus générique que le pipeline sache faire
usage: python run_ablations.py <ablation|all> [seed] [N] [T] [K0]
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from considerance_bench import run_bench

def _set(path, value):
    def mut(c):
        d = c
        for k in path[:-1]: d = d.setdefault(k, {})
        d[path[-1]] = value
    return mut

def _many(*muts):
    def mut(c):
        for m in muts: m(c)
    return mut

ABLATIONS = {
    'reference':         dict(),
    'sans_respiration':  dict(breath=False),
    'phases_statiques':  dict(phi_static=True),
    'sans_signatures':   dict(config_mutator=_set(('spiral', 'signature_pattern'), 'none')),
    'signatures_hasard': dict(config_mutator=_set(('spiral', 'signature_pattern'), 'random')),
    'sans_couplage':     dict(config_mutator=_set(('coupling', 'c'), 1e-6)),   # validate_config exige c > 0 : 1e-6 = poids négligeables, S_i ≈ 0
    'env_statique':      dict(config_mutator=_set(('enveloppe', 'env_mode'), 'static')),
    'latence_statique':  dict(config_mutator=_set(('latence', 'gamma_mode'), 'static')),
    'G_identite':        dict(config_mutator=_many(_set(('regulation', 'feedback_mode'), 'simple'), _set(('regulation', 'G_arch'), 'tanh'))),
    'beta_statique':     dict(config_mutator=_set(('dynamic_parameters', 'dynamic_beta'), False)),
    'E_ancien':          dict(config_mutator=_set(('memory', 'enabled'), False)),
    'generique':         dict(cascade='aucune', scale=2.5, breath=False, phi_static=True,
                              config_mutator=_many(_set(('spiral', 'signature_pattern'), 'none'), _set(('enveloppe', 'env_mode'), 'static'),
                                                   _set(('latence', 'gamma_mode'), 'static'), _set(('regulation', 'feedback_mode'), 'simple'),
                                                   _set(('regulation', 'G_arch'), 'tanh'), _set(('dynamic_parameters', 'dynamic_beta'), False),
                                                   _set(('memory', 'enabled'), False))),
}

if __name__ == '__main__':
    which = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    T = int(sys.argv[4]) if len(sys.argv) > 4 else 170
    K0 = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0
    names = list(ABLATIONS) if which == 'all' else [which]
    for nm in names:
        run_bench(f"{nm}_{seed}", seed=seed, N=N, T=T, K0=K0, label=f"{nm} s{seed}", **ABLATIONS[nm])
    print("DONE")
