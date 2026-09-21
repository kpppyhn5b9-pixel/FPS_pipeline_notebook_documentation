"""Tableau de la carte d'ablation : une ligne par variante (moyenne des seeds), substrat + considération.
usage: python table_ablations.py <dossier des runs> [markdown]
"""
import sys, os, json, glob, numpy as np
d = sys.argv[1]; md = len(sys.argv) > 2 and sys.argv[2] == 'markdown'
ORDER = ['reference', 'sans_respiration', 'phases_statiques', 'sans_signatures', 'signatures_hasard', 'sans_couplage',
         'env_statique', 'latence_statique', 'G_identite', 'beta_statique', 'E_ancien', 'generique']
rows = {}
for p in glob.glob(os.path.join(d, '*', 'summary.json')):
    S = json.load(open(p)); nm = os.path.basename(os.path.dirname(p)).rsplit('_', 1)[0]
    rows.setdefault(nm, []).append(S)
def m(ss, f): return float(np.mean([f(s) for s in ss]))
def sd(ss, f): return float(np.std([f(s) for s in ss]))
hdr = ['variante', 'n', 'R', 'σ_Rloc', 'fluidité', 'disp', 'activité', 'C_JS', 'écart vois.', 'engagement', 'contagion', 'coût', 'efficience', 'montée', 'σ/σref', 'rev +5', 'ext +45', 'scores']
sep = ' | ' if md else '  '
print(sep.join(hdr) if not md else '| ' + ' | '.join(hdr) + ' |')
if md: print('|' + '---|' * len(hdr))
for nm in ORDER + sorted(set(rows) - set(ORDER)):
    if nm not in rows: continue
    ss = rows[nm]; sub = lambda k: (lambda s: s['substrat'][k]); res = lambda k: (lambda s: s['resume'][k])
    sc = ss[-1]['scores']; scs = ' '.join(f"{k[:3]}{v}" for k, v in sc.items() if v is not None)
    vals = [nm, str(len(ss)), f"{m(ss, sub('R_mean')):.3f}", f"{m(ss, sub('sigma_ref')):.3f}", f"{m(ss, sub('fluidity')):.3f}", f"{m(ss, sub('disp')):.4f}",
            f"{m(ss, sub('effort')):.0f}", f"{m(ss, sub('cjs')):.3f}", f"{m(ss, sub('f_neigh_gap')):.3f}",
            f"+{m(ss, res('engagement')):.3f}±{sd(ss, res('engagement')):.2f}", f"{m(ss, res('contagion')):+.3f}", f"{m(ss, res('cost')):.4f}",
            f"{m(ss, res('efficience')):.1f}", f"{m(ss, res('rise')):.1f}", f"{m(ss, res('sigma_ratio')):.2f}", f"{m(ss, res('rev5')):+.3f}", f"{m(ss, res('ext45')):+.3f}", scs]
    print(('| ' + ' | '.join(vals) + ' |') if md else sep.join(v.ljust(10) for v in vals))
