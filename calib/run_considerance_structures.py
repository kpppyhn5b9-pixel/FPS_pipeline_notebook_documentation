"""Banc (21/09/2026) : la QUESTION, pas un mécanisme de plus.

« Métastabilité migrante contextuelle » = capacité de re-couplage rapide, réversible et
sélectif (Attention.md). « Efficience de la considération » = une SUSCEPTIBILITÉ, pas un
score : combien de contact on obtient pour combien de forçage, sans contagion, et en
revenant à soi quand le contexte part. Protocole et lectures : considerance_bench.py.

La question de ce script : la STRUCTURE de fréquences compte-t-elle ? Même geste, même
contexte, même seed, plusieurs cascades :
  doree     fₙ₊₁ ← ½fₙ₊₁ + ½·r(t)·fₙ, r = φ + ε·sin (l'original, reproduit à l'identique)
  ratio15   idem avec r = 1.5 (une cascade, mais pas dorée)
  ratio10   r = 1.0 : lissage de proche en proche SANS croissance
  hasard    r_n tiré au hasard par maillon dans [1.2, 2.0] (fixe, même respiration)
  aucune    pas de cascade (fₙ = f₀ₙ + Δfₙ·βₙ)
  suffixe _x25 : fréquences × 2.5, témoin d'ÉCHELLE (même f̄ que la dorée, même geste absolu)
Résultats : docs/CONSIDERANCE_banc_structures.md.
usage: python run_considerance_structures.py <nom> <structure> [seed] [N] [T] [K0] [kappa] [largeur]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from considerance_bench import run_bench

name, structure = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 12345
N = int(sys.argv[4]) if len(sys.argv) > 4 else 100
T = int(sys.argv[5]) if len(sys.argv) > 5 else 170
K0 = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
kappa = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5
width = float(sys.argv[8]) if len(sys.argv) > 8 else 6.0
scale = 2.5 if structure.endswith('_x25') else 1.0
run_bench(name, seed=seed, N=N, T=T, K0=K0, kappa=kappa, width=width, cascade=structure.replace('_x25', ''), scale=scale)
print("DONE", name)
