"""Prototype (hors pipeline, 21/09/2026) : liage contextuel local dans le canal fréquence.
Résultat : le liage « champ moyen d'îlot » (coupling=mf) fait suivre la cohérence au
contexte (R îlot 0.9–1.0 contre 0.64 dehors), σ_Rloc intact, relâchement immédiat quand
le projecteur part ; la cloche de dispersion joue le gardien (score 4 à K₀=6, 5 à K₀≤3).
Réglage retenu pour une cohérence LÂCHE (jamais totale) : K₀ = 2–3, κ = 0.5–1, gate=none, mf.

Contexte spatial = projecteur gaussien qui se déplace sur l'axe des strates.
Saillance sₙ ∈ [0,1] = le projecteur. Après compute_fn (donc APRÈS la contrainte
spirale), on ajoute un terme de couplage de phase de type Kuramoto restreint au
voisinage de couplage (le même que μ_Rloc) et pondéré par la saillance :
    Δfₙ = Kₙ/(2π) · Σ_j pw_j · sin(θ_j − θ_n),   Kₙ = K₀ · sₙ · garde(σ) · (1 − Rloc_n)
garde(σ) = 1 si le contraste σ_Rloc reste ≥ 0.8 × sa valeur de référence (mesurée
avant l'allumage), sinon décroît linéairement jusqu'à 0 à 0.5×.
On mesure : Rloc dedans/dehors du projecteur, corr(sₙ, Rloc_n), contraste σ_Rloc,
cloche de dispersion, scores, activité, et la réversibilité (Rloc de l'ancienne
zone après départ du projecteur).
usage: python run_context_binding.py <nom> <K0> <T> <t_on> <largeur> [seed] [kappa]
  kappa ∈ [0,1] : COMPRESSION des fréquences dans la zone saillante, fₙ ← fₙ + κ·sₙ·(f̄_s − fₙ)
  (f̄_s = moyenne des fₙ pondérée par sₙ) — « rapprocher les fréquences » (le levier des ratios),
  appliquée APRÈS la contrainte spirale, donc non diluée par son lissage.
"""
import sys, os, json, io, contextlib, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # racine du dépôt
sys.path.insert(0, ROOT)
import simulate, dynamics, metrics
name, K0, T, t_on, width = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 12345
kappa = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
gate = sys.argv[8] if len(sys.argv) > 8 else 'full'
coupling = sys.argv[9] if len(sys.argv) > 9 else 'nn'   # 'nn' = voisins de couplage ; 'mf' = champ moyen de l'îlot
# Diagnostic 21/09 (bis) : sur une chaîne, le Kuramoto premier-voisin symétrique a pour point fixe
# TOUTE torsion uniforme (0.5·sin(δ) + 0.5·sin(−δ) = 0) ; il uniformise la torsion mais ne la fixe pas,
# et Rloc_n = |cos(δ)| dérive avec elle. Pour fermer l'écart il faut tirer chaque strate vers la
# PHASE MOYENNE de l'îlot (paramètre d'ordre Z_s) : pas de mode neutre, Rloc → 1 dans le cœur.   # 'full' = (1−Rloc_n) ; 'floor' = max(0.3, 1−Rloc_n) ; 'none' = 1
# Diagnostic 21/09 : avec (1−Rloc_n) plein, la force de maintien s'annule exactement quand l'îlot est
# aligné ; le résidu de désaccord (bords de la bosse, s<1) re-désynchronise, Rloc retombe, la force
# revient : cycle 0.99 → 0.3 → 0.99. Les rendements décroissants doivent porter sur l'ALLOCATION de
# saillance, pas sur la force qui tient l'îlot.
os.makedirs(name, exist_ok=True); os.chdir(name)
dt = 0.1
c = json.load(open(os.path.join(ROOT, 'config.json')))
c['system']['T'] = T; c['system']['seed'] = seed
c['system']['input']['perturbations'] = [{'type': 'none', 'amplitude': 0.0, 't0': 0.0, 'weight': 1.0}]
c['analysis'] = {**c.get('analysis', {}), 'compare_kuramoto': False}
json.dump(c, open('cfg.json', 'w'))
N = c['system']['N']

def center_at(t):
    # projecteur : immobile 20 u.t. sur chaque poste, postes 20 → 80 → 50 → 20 (cycle)
    if t < t_on: return None
    posts = [20, 80, 50, 20]
    k = int((t - t_on) // 20) % len(posts)
    return posts[k]

st = {'phase': None, 'neigh': None, 'sigma_ref': [], 'log': []}
_orig = dynamics.compute_fn
def patched(t, state, An_t, F, cfg, _o=_orig):
    fn = _o(t, state, An_t, F, cfg)
    n = len(state)
    if st['neigh'] is None:
        st['neigh'] = []
        for i in range(n):
            w = np.asarray(state[i]['w'], dtype=float); idx = np.nonzero(np.abs(w) > 1e-12)[0]
            pw = np.abs(w[idx]); pw = pw / pw.sum() if idx.size else pw
            st['neigh'].append((idx, pw))
        st['phase'] = np.zeros(n)
    hist = cfg.get('history', [])
    phi_prev = np.asarray(hist[-1]['phi_n_t'], dtype=float) if hist and 'phi_n_t' in hist[-1] else np.zeros(n)
    theta = st['phase'] + phi_prev
    z = np.exp(1j * theta)
    rloc = np.array([abs(np.sum(pw * z[idx])) if idx.size else 1.0 for idx, pw in st['neigh']])
    sigma = float(np.std(rloc))
    ctr = center_at(t)
    if ctr is None:
        st['sigma_ref'].append(sigma); s = np.zeros(n); garde = 1.0
    else:
        s = np.exp(-0.5 * ((np.arange(n) - ctr) / width) ** 2)
        ref = float(np.median(st['sigma_ref'][-200:])) if st['sigma_ref'] else sigma
        garde = float(np.clip((sigma / ref - 0.5) / 0.3, 0.0, 1.0))  # 1 au-dessus de 0.8·ref, 0 sous 0.5·ref
    if ctr is not None and (kappa > 0 or K0 > 0):
        if kappa > 0 and s.sum() > 0:
            f_bar = float(np.sum(s * fn) / np.sum(s))          # fréquence moyenne de la zone saillante
            fn = fn + kappa * garde * s * (f_bar - fn)         # rapprocher les fréquences (ratios)
        if K0 > 0:
            if coupling == 'mf':
                Z = np.sum(s * np.exp(1j * theta)) / max(np.sum(s), 1e-12)      # paramètre d'ordre de l'îlot
                dtheta = np.abs(Z) * np.sin(np.angle(Z) - theta)                 # tirer vers la phase moyenne
            else:
                dtheta = np.array([np.sum(pw * np.sin(theta[idx] - theta[i])) if idx.size else 0.0 for i, (idx, pw) in enumerate(st['neigh'])])
            g_r = (1.0 - rloc) if gate == 'full' else (np.maximum(0.3, 1.0 - rloc) if gate == 'floor' else np.ones_like(rloc))
            K = K0 * s * garde * g_r
            fn = fn + K * dtheta / (2 * np.pi)               # accrocher les phases
    st['phase'] = st['phase'] + 2 * np.pi * fn * dt   # même intégration que simulate (phase_acc)
    st['log'].append((t, ctr, sigma, garde, rloc.copy(), s.copy()))
    return fn
dynamics.compute_fn = patched
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf): res = simulate.run_simulation('cfg.json', 'FPS')
except BaseException as e:
    sys.stderr.write(f"ECHEC {type(e).__name__}: {e}\n" + buf.getvalue()[-2500:] + "\n"); raise
finally:
    dynamics.compute_fn = _orig
h = res['history']
def col(k): return np.array([np.nan if (x.get(k) is None or isinstance(x.get(k), str)) else x[k] for x in h], dtype=float)
t = col('t'); eff = col('effort(t)'); dn = col('dispersion_norm'); mu = col('mu_Rloc(t)')
log = st['log']; tl = np.array([l[0] for l in log]); sig = np.array([l[2] for l in log]); gar = np.array([l[3] for l in log])
R = np.array([l[4] for l in log]); Sal = np.array([l[5] for l in log]); ctrs = [l[1] for l in log]
out = []
print(f"[{name}] K0={K0} kappa={kappa} gate={gate} coupling={coupling} largeur={width} | σ_ref (avant allumage) = {np.median(sig[tl < t_on]):.3f}")
for a in range(int(t_on) - 20, T, 10):
    m = (tl >= a) & (tl < a + 10)
    if not m.any(): continue
    Rm = R[m].mean(0); Sm = Sal[m].mean(0); c0 = ctrs[np.where(m)[0][-1]]
    inside = Rm[Sm > 0.5].mean() if (Sm > 0.5).any() else float('nan')
    outside = Rm[Sm < 0.05].mean() if (Sm < 0.05).any() else Rm.mean()
    corr = float(np.corrcoef(Sm, Rm)[0, 1]) if Sm.std() > 0 else float('nan')
    # ancienne zone (poste précédent) pour la réversibilité
    prev = None
    if c0 is not None:
        posts = [20, 80, 50, 20]; k = posts.index(c0)
        prev = posts[k - 1] if k > 0 else None
    old = Rm[max(0, (prev or 0) - 3):(prev or 0) + 4].mean() if prev is not None and prev != c0 else float('nan')
    mh = (t >= a) & (t < a + 10)
    row = dict(t=a, centre=c0, R_dedans=inside, R_dehors=outside, corr=corr, R_ancienne_zone=old, sigma=float(sig[m].mean()), garde=float(gar[m].mean()),
               mu_Rloc=float(np.nanmean(mu[mh])), disp_norm=float(np.nanmedian(dn[mh])) if np.isfinite(dn[mh]).any() else float('nan'), effort=float(np.nanmedian(eff[mh])))
    out.append(row)
    print(f"[{name}] t∈[{a},{a+10}) centre={str(c0):>4s} | R dedans {inside:.3f} dehors {outside:.3f} corr(s,R) {corr:+.2f} | R ancienne zone {old:.3f} | σ_Rloc {row['sigma']:.3f} garde {row['garde']:.2f} | μ_Rloc {row['mu_Rloc']:.3f} disp {row['disp_norm']:.4f} | effort {row['effort']:.0f}")
sc = metrics.compute_reference_scores(h[-metrics.reference_window(c, dt):], dt, signal='O', N=N)
print(f"[{name}] scores finaux :", metrics.labelled_scores(sc), "| résumé activite_rel", res['metrics'].get('activite_rel'), "dispersion_norm", res['metrics'].get('dispersion_norm'))
json.dump({'rows': out, 'scores': sc}, open('summary.json', 'w'), indent=1)
print("DONE", name)
