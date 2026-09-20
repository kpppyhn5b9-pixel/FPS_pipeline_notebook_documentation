"""Campagne rampe interne (latence) : la latence adaptative γ est le mécanisme à
mémoire de la couche lente (fₙ = f₀·(1 + β·γ) ; γ converge vers sa cible par
lissage exponentiel de taux λ, dynamics.py:1002). On enveloppe compute_gamma_
adaptive_aware : γ_out = (1−λ)·γ_prev + λ·γ_cible + σ·ε, ε ~ N(0,1) par pas
(perturbation FIXE sur l'état, modèle de Scheffer), avec λ qui dérive
linéairement de l0 à l1 entre t0 et t1. Taux de rappel qui faiblit = temps de
retour qui s'allonge = ralentissement critique, variance qui monte (σ²/2λ).
Attendu au lag L : ac ≈ (1−λ)^L.
usage: python run_ramp_gamma.py <nom> <l0> <l1> <t0> <t1> <sigma> <T> <seeds>"""
import sys, os, json, io, contextlib, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # racine du dépôt
sys.path.insert(0, ROOT)
name = sys.argv[1]; l0, l1, t0, t1, sigma = map(float, sys.argv[2:7]); T = int(float(sys.argv[7]))
seeds = tuple(int(v) for v in sys.argv[8].split(','))
os.makedirs(name, exist_ok=True); os.chdir(name)
import simulate, dynamics
_orig = dynamics.compute_gamma_adaptive_aware
base = json.load(open(os.path.join(ROOT, 'config.json')))
out = {}
def lam_at(t):
    if t <= t0: return l0
    if t >= t1: return l1
    return l0 + (l1 - l0) * (t - t0) / (t1 - t0)
for seed in seeds:
    rng = np.random.RandomState(seed + 1)
    st = {'g': None}
    def patched(t, state, history, config, journal=None, _o=_orig):
        g_target, regime, journal = _o(t, state, history, config, journal)
        lam = lam_at(t)
        g = g_target if st['g'] is None else (1 - lam) * st['g'] + lam * g_target
        g = float(np.clip(g + sigma * rng.randn(), 0.05, 0.95))
        st['g'] = g
        return g, regime, journal
    dynamics.compute_gamma_adaptive_aware = patched
    c = json.loads(json.dumps(base)); c['system']['T'] = T; c['system']['seed'] = seed
    c['system']['input']['perturbations'] = [{'type': 'none', 'amplitude': 0.0, 't0': 0.0, 'weight': 1.0}]
    c['analysis'] = {**c.get('analysis', {}), 'compare_kuramoto': False}
    cfg_path = f'config_{seed}.json'; json.dump(c, open(cfg_path, 'w'))
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            res = simulate.run_simulation(cfg_path, 'FPS')
    except BaseException as e:
        sys.stderr.write(f"ECHEC seed {seed}: {type(e).__name__}: {e}\n" + buf.getvalue()[-3000:] + "\n"); raise
    finally:
        dynamics.compute_gamma_adaptive_aware = _orig
    h = res['history']
    def col(k): return np.array([np.nan if x.get(k) is None else float(x[k]) for x in h], dtype=float)
    t = col('t'); lag = res['metrics'].get('resilience_lag')
    ac, sm, var, al, q, sc, fn, gam = (col(k) for k in ('resilience_ac', 'resilience_ac_smooth', 'resilience_var', 'resilience_alert', 'resilience_quiet', 'resilience_score', 'fn_mean(t)', 'gamma'))
    rows = []
    for a in range(0, T, 25):
        m = (t >= a) & (t < a + 25); lam = lam_at(a + 12.5)
        r = {'t': a, 'lambda': round(lam, 4), 'expected_ac': (round((1 - lam) ** lag, 3) if lag else None),
             'gamma_std': float(np.nanstd(gam[m])), 'fn_std': float(np.nanstd(fn[m]))}
        if np.any(np.isfinite(ac[m])):
            r.update({'ac': float(np.nanmedian(ac[m])), 'ac_smooth': float(np.nanmedian(sm[m])), 'var': float(np.nanmedian(var[m])),
                      'quiet': float(np.nanmean(q[m])), 'alerts': int(np.nansum(al[m])), 'score': float(np.nanmedian(sc[m]))})
        rows.append(r)
    out[str(seed)] = {'lag': lag, 'rows': rows}
    print(seed, "lag", lag)
    for r in rows:
        print("  " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    sys.stdout.flush()
json.dump(out, open('summary.json', 'w'), indent=1)
print("DONE", name)
