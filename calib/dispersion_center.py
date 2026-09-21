"""Recette du centre de la cloche de dispersion (metrics.DISPERSION_NORM_CENTER).

Le centre est ⟨Aₙ⟩/√2 : pour N sinusoïdes d'amplitude A et de phases
indépendantes, std(ΣOₙ) = A·√(N/2), donc std(ΣO)/√N = A/√2. Ce script mesure les
deux sur un ou plusieurs CSV de run calme (config standard), affiche le rapport
(≈ 1 si le chœur est incohérent au premier ordre) et la distribution du repli
max(v/c, c/v) par fenêtre W_f contre le centre en vigueur. À rejouer quand le
régime d'amplitude change (A₀, échelle d'entrée, enveloppe, couplage).
usage: python dispersion_center.py <run_*.csv> [...] [--N 100] [--Wf 50] [--tmin 60]"""
import sys, os, argparse, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # racine du dépôt
sys.path.insert(0, ROOT)
import metrics

p = argparse.ArgumentParser(); p.add_argument('csv', nargs='+'); p.add_argument('--N', type=int, default=100)
p.add_argument('--Wf', type=int, default=50); p.add_argument('--tmin', type=float, default=60.0)
a = p.parse_args()
C = metrics.DISPERSION_NORM_CENTER; edge = metrics.DISPERSION_FOLD_FACTORS[0]
print(f"centre en vigueur {C:.4f} ; bord du 5 : repli < {edge}")
print("run | std(ΣO)/√N | ⟨Aₙ⟩/√2 | rapport | repli/fenêtre médiane / max | part > bord")
for f in a.csv:
    d = pd.read_csv(f); t = d['t'].values
    O = a.N * d['On_mean(t)'].values.astype(float); A = d['An_mean(t)'].values.astype(float)
    reg = t >= a.tmin
    disp = float(np.std(O[reg])) / np.sqrt(a.N); pred = float(np.mean(A[reg])) / np.sqrt(2)
    idx = np.where(reg)[0]; folds = []
    for i in range(idx[0] + a.Wf, len(O), 10):
        v = float(np.std(O[i - a.Wf:i])) / np.sqrt(a.N); folds.append(max(v / C, C / v))
    folds = np.array(folds)
    print(f"{os.path.basename(f)[:40]:40s} | {disp:.4f} | {pred:.4f} | {disp/pred:.2f} | {np.median(folds):.2f} / {folds.max():.2f} | {np.mean(folds > edge):.2f}")
print("→ pour recalibrer : DISPERSION_NORM_CENTER = ⟨Aₙ⟩/√2 mesuré ici sur la config standard.")
