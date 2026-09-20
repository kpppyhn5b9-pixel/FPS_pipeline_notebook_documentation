"""Calibration de la résilience sous bruit : 4 seeds × une condition.
usage: python run_condition.py <nom> <amplitude_bruit|none> <T>
Écrit <nom>/summary.json avec, par seed : lag, ac brute/lissée en régime (t>60), alertes, scores."""
import sys, os, json, copy, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # racine du dépôt
sys.path.insert(0, ROOT)
name, amp, T = sys.argv[1], sys.argv[2], int(float(sys.argv[3]))
os.makedirs(name, exist_ok=True); os.chdir(name)
import simulate
base = json.load(open(os.path.join(ROOT, 'config.json')))
out = {}
for seed in (12345, 7, 99, 2024):
    c = copy.deepcopy(base)
    c['system']['T'] = T; c['system']['seed'] = seed
    if amp != 'none':
        c['system']['input']['perturbations'] = [{'type': 'bruit', 'amplitude': float(amp), 't0': 0.0, 'weight': 1.0, 'seed': seed}]
    else:
        c['system']['input']['perturbations'] = [{'type': 'none', 'amplitude': 0.0, 't0': 0.0, 'weight': 1.0}]
    c['analysis'] = {**c.get('analysis', {}), 'compare_kuramoto': False}
    cfg_path = f'config_{seed}.json'; json.dump(c, open(cfg_path, 'w'))
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            res = simulate.run_simulation(cfg_path, 'FPS')
    except BaseException as e:
        sys.stderr.write(f"ECHEC seed {seed}: {type(e).__name__}: {e}\n--- fin de la sortie capturée ---\n" + buf.getvalue()[-3000:] + "\n")
        raise
    h = res['history']
    def col(k): return np.array([np.nan if x.get(k) is None else float(x[k]) for x in h], dtype=float)
    t = col('t'); reg = t > 200
    ac, sm, var, al, sc = col('resilience_ac'), col('resilience_ac_smooth'), col('resilience_var'), col('resilience_alert'), col('resilience_score')
    def stats(a):
        a = a[np.isfinite(a)]
        return None if len(a) == 0 else {'n': int(len(a)), 'median': float(np.median(a)), 'q25': float(np.percentile(a, 25)), 'q75': float(np.percentile(a, 75)), 'min': float(a.min()), 'max': float(a.max())}
    out[str(seed)] = {
        'lag': res['metrics'].get('resilience_lag'),
        'ac_regime': stats(ac[reg]), 'ac_smooth_regime': stats(sm[reg]), 'var_regime': stats(var[reg]),
        'alerts': int(np.nansum(al)),
        'scores_regime': {int(k): int(v) for k, v in zip(*np.unique(sc[reg & np.isfinite(sc)], return_counts=True))},
        'innovation_cjs': res['metrics'].get('innovation_cjs'),
        'fluidity_final': res['metrics'].get('final_fluidity'),
    }
    print(seed, json.dumps(out[str(seed)]), flush=True)
json.dump(out, open('summary.json', 'w'), indent=1)
print("DONE", name)
