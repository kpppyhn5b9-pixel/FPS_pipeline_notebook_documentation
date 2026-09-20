"""Exploration : la FPS a-t-elle une transition quand on monte le gain de couplage α ?
usage: python explore_gain.py <nom> <gain> <T>  — α_n multiplié par gain (constant), pas de bruit."""
import sys, os, json, copy, io, contextlib, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # racine du dépôt
sys.path.insert(0, ROOT)
name, gain, T = sys.argv[1], float(sys.argv[2]), int(float(sys.argv[3]))
os.makedirs(name, exist_ok=True); os.chdir(name)
import simulate, dynamics
_orig = dynamics.compute_fn
done = {'scaled': False}
def patched(t, state, An_t, F, cfg, _o=_orig):
    if not done['scaled']:
        for s in state: s['alpha'] = s['alpha'] * gain
        done['scaled'] = True
    return _o(t, state, An_t, F, cfg)
dynamics.compute_fn = patched
c = json.load(open(os.path.join(ROOT, 'config.json')))
c['system']['T'] = T; c['system']['seed'] = 12345
c['system']['input']['perturbations'] = [{'type': 'none', 'amplitude': 0.0, 't0': 0.0, 'weight': 1.0}]
c['analysis'] = {**c.get('analysis', {}), 'compare_kuramoto': False}
c['resilience']['W_res_t'] = 20.0  # court : on regarde la dynamique, pas la résilience
json.dump(c, open('cfg.json', 'w'))
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        res = simulate.run_simulation('cfg.json', 'FPS')
except BaseException as e:
    sys.stderr.write(f"ECHEC: {type(e).__name__}: {e}\n" + buf.getvalue()[-2000:]); raise
h = res['history']
def col(k): return np.array([np.nan if x.get(k) is None else float(x[k]) for x in h])
t = col('t'); fn = col('fn_mean(t)'); S = col('S(t)'); An = col('An_mean(t)'); eff = col('effort(t)'); err = col('mean_abs_error')
for a, b in ((10, 30), (30, 60)):
    m = (t >= a) & (t < b)
    print(f"gain {gain} t∈[{a},{b}): fn_mean {np.nanmean(fn[m]):.3f}±{np.nanstd(fn[m]):.3f} min {np.nanmin(fn[m]):.3f} max {np.nanmax(fn[m]):.3f} | S std {np.nanstd(S[m]):.3f} | An {np.nanmean(An[m]):.3f} | effort {np.nanmean(eff[m]):.3f} | err {np.nanmean(err[m]):.4f}")
print("DONE", name)
