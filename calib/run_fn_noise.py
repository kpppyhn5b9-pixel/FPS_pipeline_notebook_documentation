"""Fluctuation RÉELLE injectée sur la couche lente : bruit AR(1) en mode commun sur fₙ,
de temps de relaxation τ connu (en pas) et d'amplitude relative rel_amp.
Vérité-terrain in-situ : au lag L calibré, l'autocorr attendue vaut exp(−L/τ).
usage: python run_fn_noise.py <nom> <tau_pas> <rel_amp> <T> [t_on] [seeds]  (t_on : la fluctuation ne démarre qu'à t ≥ t_on)"""
import sys, os, json, copy, io, contextlib, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # racine du dépôt
sys.path.insert(0, ROOT)
name, tau, rel_amp, T = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(float(sys.argv[4]))
t_on = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
seeds = tuple(int(v) for v in sys.argv[6].split(',')) if len(sys.argv) > 6 else (12345, 7, 99, 2024)
os.makedirs(name, exist_ok=True); os.chdir(name)
import simulate, dynamics
_orig = dynamics.compute_fn
base = json.load(open(os.path.join(ROOT, 'config.json')))
out = {}
for seed in seeds:
    rng = np.random.RandomState(seed + 1)
    phi = float(np.exp(-1.0 / tau)); st = {'x': 0.0}
    def patched(t, state, An_t, F, cfg, _o=_orig):
        fn = _o(t, state, An_t, F, cfg)
        st['x'] = phi * st['x'] + np.sqrt(1 - phi ** 2) * rng.randn()
        return fn * (1.0 + rel_amp * st['x']) if t >= t_on else fn
    dynamics.compute_fn = patched
    c = copy.deepcopy(base); c['system']['T'] = T; c['system']['seed'] = seed
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
        dynamics.compute_fn = _orig
    h = res['history']
    def col(k): return np.array([np.nan if x.get(k) is None else float(x[k]) for x in h], dtype=float)
    t = col('t'); reg = t > max(200, t_on)
    ac, sm, var, al, q, sc = (col(k) for k in ('resilience_ac', 'resilience_ac_smooth', 'resilience_var', 'resilience_alert', 'resilience_quiet', 'resilience_score'))
    def stats(a):
        a = a[np.isfinite(a)]
        return None if len(a) == 0 else {'n': int(len(a)), 'median': float(np.median(a)), 'q25': float(np.percentile(a, 25)), 'q75': float(np.percentile(a, 75)), 'min': float(a.min()), 'max': float(a.max())}
    lag = res['metrics'].get('resilience_lag')
    out[str(seed)] = {'lag': lag, 'expected_ac': float(np.exp(-lag / tau)) if lag else None,
                      'ac_regime': stats(ac[reg]), 'ac_smooth_regime': stats(sm[reg]), 'var_regime': stats(var[reg]),
                      'quiet_share': float(np.nanmean(q[reg])), 'alerts': int(np.nansum(al)),
                      'alerts_by_50': {int(a): int(np.nansum(al[(t >= a) & (t < a + 50)])) for a in range(200, int(T), 50)},
                      'scores_regime': {int(k): int(v) for k, v in zip(*np.unique(sc[reg & np.isfinite(sc)], return_counts=True))}}
    print(seed, json.dumps(out[str(seed)]), flush=True)
json.dump(out, open('summary.json', 'w'), indent=1)
print("DONE", name)
