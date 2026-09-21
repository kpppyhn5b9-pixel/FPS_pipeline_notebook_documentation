"""Banc (21/09/2026) : la QUESTION, pas un mécanisme de plus.

« Métastabilité migrante contextuelle » = capacité de re-couplage rapide, réversible et
sélectif (Attention.md). « Efficience de la considération » = une SUSCEPTIBILITÉ, pas un
score : combien de contact on obtient pour combien de forçage, sans contagion, et en
revenant à soi quand le contexte part.

Ce banc pose la question au substrat : un contexte (projecteur gaussien sur l'axe des
strates) se pose à un poste, tient 20 u.t., se déplace (deux fois), puis disparaît.
Le geste est celui déjà validé (run_context_binding.py, mode mf, gate none) :
    Δfₙ = K₀ · sₙ · garde(σ) · |Z_s| · sin(arg Z_s − θₙ) / 2π   (+ κ·sₙ·(f̄_s − fₙ))
Aucun paramètre nouveau dans le pipeline. La mémoire de soi (Eₙ) est laissée telle
quelle comme SENS (attention désactivée : gains à 0) : la surprise des strates liées
dit si le contact laisse une trace ou s'efface.

Quatre lectures, sans barème :
  ENGAGEMENT    R dedans − R dedans avant (et corr(s, R), temps de montée)
  SÉLECTIVITÉ   R dehors − R dehors avant ; σ_Rloc / σ_ref ; cloche de dispersion
  RÉVERSIBILITÉ R de l'ancienne zone 5 et 15 u.t. après le départ ; tout après l'extinction ;
                surprise des strates liées vs les autres, pendant et après
  COÛT          forçage relatif moyen |Δfₙ|/fₙ sur les strates liées
  → efficience = engagement / coût

Et la vraie question : la STRUCTURE compte-t-elle ? Quatre cascades, même geste, même
contexte, même seed :
  doree     fₙ₊₁ ← ½fₙ₊₁ + ½·r(t)·fₙ, r = φ + ε·sin (l'original, reproduit à l'identique)
  ratio15   idem avec r = 1.5 (une cascade, mais pas dorée)
  hasard    r_n tiré au hasard par maillon dans [1.2, 2.0] (fixe, même respiration)
  aucune    pas de cascade (fₙ = f₀ₙ + Δfₙ·βₙ)
  ratio10   r = 1.0 : lissage de proche en proche SANS croissance (sépare « lisser » de « cascader »)
  suffixe _x25 : fréquences × 2.5, témoin d'ÉCHELLE (même f̄ que la dorée, même geste absolu)
usage: python run_considerance_structures.py <nom> <structure> [seed] [N] [T] [K0] [kappa] [largeur]
"""
import sys, os, json, io, contextlib, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import simulate, dynamics, metrics

name = sys.argv[1]; structure = sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 12345
N = int(sys.argv[4]) if len(sys.argv) > 4 else 100
T = int(sys.argv[5]) if len(sys.argv) > 5 else 170
K0 = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
kappa = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5
width = float(sys.argv[8]) if len(sys.argv) > 8 else 6.0
SCALE = 2.5 if structure.endswith('_x25') else 1.0; base_structure = structure.replace('_x25', '')   # _x25 : témoin d'ÉCHELLE (f × 2.5 ≈ f̄ de la dorée)
assert base_structure in ('doree', 'ratio15', 'ratio10', 'hasard', 'aucune'), structure

T_ON = 60.0; POSTS = [20, 80, 50]; DWELL = 20.0; T_OFF = T_ON + DWELL * len(POSTS)   # 120 : extinction
os.makedirs(name, exist_ok=True); os.chdir(name)
dt = 0.1
c = json.load(open(os.path.join(ROOT, 'config.json')))
c['system']['T'] = T; c['system']['seed'] = seed; c['system']['N'] = N
c['system']['input']['perturbations'] = [{'type': 'none', 'amplitude': 0.0, 't0': 0.0, 'weight': 1.0}]
c['analysis'] = {**c.get('analysis', {}), 'compare_kuramoto': False}
c.setdefault('memory', {})['enabled'] = True; c['memory']['birth_t'] = 20.0          # la mémoire comme sens
c.setdefault('attention', {})['consolidation_gain'] = 0.0; c['attention']['anchor_gain'] = 0.0   # pas de main
json.dump(c, open('cfg.json', 'w'))
sp = c.get('spiral', {}); PHI, EPS, OMEGA, THETA = sp.get('phi', 1.618), sp.get('epsilon', 0.1), sp.get('omega', 0.05), sp.get('theta', 0.0)
rng_struct = np.random.default_rng(777)
R_HASARD = rng_struct.uniform(1.2, 2.0, size=N - 1)

def ratios(t):
    breath = EPS * np.sin(2 * np.pi * OMEGA * t + THETA)
    if base_structure == 'doree':   return np.full(N - 1, PHI) + breath
    if base_structure == 'ratio15': return np.full(N - 1, 1.5) + breath
    if base_structure == 'ratio10': return np.full(N - 1, 1.0) + breath   # lissage de proche en proche sans croissance
    if base_structure == 'hasard':  return R_HASARD + breath
    return None

def center_at(t):
    if t < T_ON or t >= T_OFF: return None
    return POSTS[int((t - T_ON) // DWELL)]

st = {'phase': None, 'neigh': None, 'sigma_ref': [], 'log': []}
_orig = dynamics.compute_fn

def patched(t, state, An_t, F, cfg, _o=_orig):
    # 1) fréquences de base SANS cascade ni feedback (dynamic_phi coupé pour cet appel seulement)
    cfg_nc = dict(cfg); cfg_nc['dynamic_parameters'] = {**cfg.get('dynamic_parameters', {}), 'dynamic_phi': False}
    fn = _o(t, state, An_t, None, cfg_nc)
    # 2) la cascade choisie (même relaxation que l'original : ½ / ½, de proche en proche)
    r = ratios(t)
    if r is not None:
        for n in range(N - 1):
            if fn[n] > 0:
                fn[n + 1] = 0.5 * fn[n + 1] + 0.5 * r[n] * fn[n]
    if SCALE != 1.0:
        fn = fn * SCALE
    # 3) le feedback F, comme l'original (clip ±0.5, plancher)
    if F is not None:
        fn = np.maximum(fn * (1 + np.clip(np.asarray(F, dtype=float)[:N], -0.5, 0.5)), 1e-6)
    f_before = fn.copy()
    # 4) le geste de considération (identique à run_context_binding, mf, gate none)
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
        if t < T_ON: st['sigma_ref'].append(sigma)
        s = np.zeros(n); garde = 1.0
    else:
        s = np.exp(-0.5 * ((np.arange(n) - ctr) / width) ** 2)
        ref = float(np.median(st['sigma_ref'][-200:])) if st['sigma_ref'] else sigma
        garde = float(np.clip((sigma / ref - 0.5) / 0.3, 0.0, 1.0))
        if kappa > 0 and s.sum() > 0:
            f_bar = float(np.sum(s * fn) / np.sum(s)); fn = fn + kappa * garde * s * (f_bar - fn)
        if K0 > 0:
            Z = np.sum(s * np.exp(1j * theta)) / max(np.sum(s), 1e-12)
            dtheta = np.abs(Z) * np.sin(np.angle(Z) - theta)
            fn = fn + K0 * s * garde * dtheta / (2 * np.pi)
    st['phase'] = st['phase'] + 2 * np.pi * fn * dt
    st['log'].append((t, ctr, sigma, garde, rloc.copy(), s.copy(), np.abs(fn - f_before) / np.maximum(f_before, 1e-9), fn.copy()))
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
t = col('t'); eff = col('effort(t)'); dn = col('dispersion_norm'); fl = col('fluidity')
U = np.array([np.asarray(x['surprise'], dtype=float) if x.get('surprise') is not None else np.full(N, np.nan) for x in h])
log = st['log']; tl = np.array([l[0] for l in log]); sig = np.array([l[2] for l in log]); gar = np.array([l[3] for l in log])
R = np.array([l[4] for l in log]); S = np.array([l[5] for l in log]); COST = np.array([l[6] for l in log]); FN = np.array([l[7] for l in log])
ctrs = np.array([np.nan if l[1] is None else l[1] for l in log])

def win(a, b): return (tl >= a) & (tl < b)
def zone(ctr): return np.exp(-0.5 * ((np.arange(N) - ctr) / width) ** 2) > 0.5
base = win(40, 60)
R_base = R[base].mean(0); sig_ref = float(np.median(sig[base])); f_base = FN[base].mean(0)
out = {'structure': structure, 'seed': seed, 'N': N, 'K0': K0, 'kappa': kappa, 'width': width,
       'sigma_ref': sig_ref, 'R_base_mean': float(R_base.mean()), 'f_mean_base': float(f_base.mean()),
       'f_spread_base': float(np.std(np.log(np.maximum(f_base, 1e-9)))), 'posts': []}
print(f"[{name}] structure={structure} seed={seed} N={N} | avant : R moy {R_base.mean():.3f}  σ_Rloc {sig_ref:.3f}  f̄ {f_base.mean():.2f} (étalement log {out['f_spread_base']:.2f})  disp {np.nanmedian(dn[(t>=40)&(t<60)]):.4f}")
prev = None
for k, ctr in enumerate(POSTS):
    a = T_ON + k * DWELL; b = a + DWELL
    zin = zone(ctr); zout = S[win(a, b)].mean(0) < 0.05
    Rw = R[win(a, b)]; Sw = S[win(a, b)]
    R_in_early = Rw[:int(5 / dt)][:, zin].mean(); R_in_late = Rw[int(10 / dt):][:, zin].mean()
    R_in0 = R_base[zin].mean(); R_out = Rw[:, zout].mean(); R_out0 = R_base[zout].mean()
    corr = float(np.mean([np.corrcoef(Sw[i], Rw[i])[0, 1] for i in range(len(Sw))]))
    cost = COST[win(a, b)][:, zin].mean()
    # temps de montée : premier instant où R dedans dépasse R0 + 0.5·(plateau − R0)
    rin_t = Rw[:, zin].mean(1); target = R_in0 + 0.5 * (R_in_late - R_in0)
    idx = np.nonzero(rin_t >= target)[0]; rise = float(idx[0] * dt) if idx.size and R_in_late > R_in0 else float('nan')
    surprise_in = np.nanmean(U[(t >= a) & (t < b)][:, zin]); surprise_out = np.nanmean(U[(t >= a) & (t < b)][:, ~zin])
    row = dict(poste=ctr, R_in0=float(R_in0), R_in_early=float(R_in_early), R_in=float(R_in_late), engagement=float(R_in_late - R_in0),
               rise_t=rise, corr=corr, R_out=float(R_out), R_out0=float(R_out0), contagion=float(R_out - R_out0),
               sigma_ratio=float(sig[win(a, b)].mean() / sig_ref), garde=float(gar[win(a, b)].mean()),
               cost=float(cost), efficience=float((R_in_late - R_in0) / max(cost, 1e-9)),
               surprise_in=float(surprise_in), surprise_out=float(surprise_out),
               disp=float(np.nanmedian(dn[(t >= a) & (t < b)])), fluid=float(np.nanmedian(fl[(t >= a) & (t < b)])), effort=float(np.nanmedian(eff[(t >= a) & (t < b)])))
    if prev is not None:   # réversibilité de l'ancienne zone
        zp = zone(prev)
        row['old_R0'] = float(R_base[zp].mean()); row['old_R_5'] = float(R[win(a + 3, a + 7)][:, zp].mean()); row['old_R_15'] = float(R[win(a + 13, a + 17)][:, zp].mean())
        row['old_surprise_5'] = float(np.nanmean(U[(t >= a + 3) & (t < a + 7)][:, zp])); row['old_surprise_15'] = float(np.nanmean(U[(t >= a + 13) & (t < a + 17)][:, zp]))
    out['posts'].append(row); prev = ctr
    print(f"[{name}] poste {ctr:2d} t∈[{a:.0f},{b:.0f}) | R dedans {R_in0:.3f} → {R_in_early:.3f} (5 u.t.) → {R_in_late:.3f} (fin)  montée {rise:.1f} u.t.  corr {corr:+.2f} | dehors {R_out0:.3f} → {R_out:.3f} | σ/σref {row['sigma_ratio']:.2f} garde {row['garde']:.2f} | coût {cost:.4f}  efficience {row['efficience']:.1f} | surprise dedans {surprise_in:.3f} dehors {surprise_out:.3f} | disp {row['disp']:.4f} fluid {row['fluid']:.3f} effort {row['effort']:.0f}"
          + (f" | ancienne zone {prev if False else POSTS[k-1]} : R {row['old_R0']:.3f} → {row['old_R_5']:.3f} (+5) {row['old_R_15']:.3f} (+15), surprise {row['old_surprise_5']:.3f} / {row['old_surprise_15']:.3f}" if k > 0 else ""))
# extinction
zl = zone(POSTS[-1])
after = {}
for lag in (5, 15, 30, 45):
    m = win(T_OFF + lag - 2, T_OFF + lag + 2); mh = (t >= T_OFF + lag - 2) & (t < T_OFF + lag + 2)
    if not m.any(): continue
    after[lag] = dict(R_last=float(R[m][:, zl].mean()), R_all=float(R[m].mean()), sigma_ratio=float(sig[m].mean() / sig_ref),
                      surprise_last=float(np.nanmean(U[mh][:, zl])), surprise_others=float(np.nanmean(U[mh][:, ~zl])))
    print(f"[{name}] extinction +{lag:2d} u.t. | dernière zone R {after[lag]['R_last']:.3f} (avant {R_base[zl].mean():.3f}) | R tout {after[lag]['R_all']:.3f} (avant {R_base.mean():.3f}) | σ/σref {after[lag]['sigma_ratio']:.2f} | surprise dernière zone {after[lag]['surprise_last']:.3f} autres {after[lag]['surprise_others']:.3f}")
out['after'] = after; out['R_base_last_zone'] = float(R_base[zl].mean())
sc = metrics.compute_reference_scores(h[-metrics.reference_window(c, dt):], dt, signal='O', N=N)
out['scores'] = sc
print(f"[{name}] scores finaux :", metrics.labelled_scores(sc))
json.dump(out, open('summary.json', 'w'), indent=1)
np.savez_compressed('kymo.npz', t=tl, R=R, s=S, sigma=sig, garde=gar, cost=COST, fn=FN, t_hist=t, U=U, disp=dn, effort=eff, ctr=ctrs)
print("DONE", name)
