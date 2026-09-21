"""Prototype (hors pipeline, 21/09/2026) : mémoire de soi à deux vitesses, surprise, attention d'assimilation.

Par strate : mémoire COURTE (τ_s u.t.) et LONGUE (τ_l u.t., suit la courte) de l'amplitude Aₙ et de
la fréquence fₙ. Surprise uₙ = |Â_s − Â_l|/Â_l + |f̂_s − f̂_l|/f̂_l (sans dimension). Une surprise brève
vit dans la courte et s'efface ; une surprise durable passe dans la longue (assimilation).
Attention : saillance sₙ = surprise au-dessus d'un plancher, lissée dans le temps (fluidité) ;
geste = ramener le rythme de la strate surprise vers celui de ses voisines de couplage pondérées par
leur calme (1 − s_j) : Δfₙ = K·sₙ·garde·(f̄_voisines calmes − fₙ), variation limitée par pas.
Événements : DURABLE (f₀ × facteur sur un groupe, à t_dur) et BREF (fₙ × facteur pendant 2 u.t. sur un
autre groupe, à t_bref). On compare attention éteinte (K = 0) et allumée.
usage: python run_surprise_attention.py <nom> <K> <T> [seed] [tau_s] [tau_l] [facteur]"""
import sys, os, json, io, contextlib, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # racine du dépôt
sys.path.insert(0, ROOT)
import simulate, dynamics, metrics
name, K, T = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 12345
tau_s = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
tau_l = float(sys.argv[6]) if len(sys.argv) > 6 else 40.0
fac = float(sys.argv[7]) if len(sys.argv) > 7 else 1.15
os.makedirs(name, exist_ok=True); os.chdir(name)
dt = 0.1; t_mem = 40.0; t_dur = 70.0; t_bref = 120.0
DUR = list(range(40, 49)); BREF = list(range(70, 79))
U0, U1, SLEW, TAU_SAL = 0.03, 0.10, 0.02, 3.0   # plancher / plein de saillance (surprise), limite Δf, lissage saillance
c = json.load(open(os.path.join(ROOT, 'config.json')))
c['system']['T'] = T; c['system']['seed'] = seed
c['system']['input']['perturbations'] = [{'type': 'none', 'amplitude': 0.0, 't0': 0.0, 'weight': 1.0}]
c['analysis'] = {**c.get('analysis', {}), 'compare_kuramoto': False}
json.dump(c, open('cfg.json', 'w'))
N = c['system']['N']
st = {'neigh': None, 'As': None, 'fs': None, 'Al': None, 'fl': None, 'sal': None, 'dprev': None, 'dur_done': False, 'log': [], 'sigma_ref': [], 'phase': None}
a_s, a_l, a_sal = dt / tau_s, dt / tau_l, dt / TAU_SAL
_orig = dynamics.compute_fn
def patched(t, state, An_t, F, cfg, _o=_orig):
    n = len(state)
    if not st['dur_done'] and t >= t_dur:
        for i in DUR: state[i]['f0'] = state[i]['f0'] * fac      # reconfiguration DURABLE
        st['dur_done'] = True
    fn = _o(t, state, An_t, F, cfg)
    if t_bref <= t < t_bref + 2.0:
        fn = fn.copy(); fn[BREF] = fn[BREF] * fac                # événement BREF
    if st['neigh'] is None:
        st['neigh'] = []
        for i in range(n):
            w = np.asarray(state[i]['w'], dtype=float); idx = np.nonzero(np.abs(w) > 1e-12)[0]
            pw = np.abs(w[idx]); pw = pw / pw.sum() if idx.size else pw
            st['neigh'].append((idx, pw))
        st['phase'] = np.zeros(n); st['sal'] = np.zeros(n); st['dprev'] = np.zeros(n)
    A = np.asarray(An_t, dtype=float)
    # ---- mémoire de soi à deux vitesses (après l'échauffement) ----
    u = np.zeros(n); s = st['sal']
    if t >= t_mem:
        if st['As'] is None:
            st['As'], st['fs'] = A.copy(), fn.copy(); st['Al'], st['fl'] = A.copy(), fn.copy()
        st['As'] = (1 - a_s) * st['As'] + a_s * A;  st['fs'] = (1 - a_s) * st['fs'] + a_s * fn
        st['Al'] = (1 - a_l) * st['Al'] + a_l * st['As']; st['fl'] = (1 - a_l) * st['fl'] + a_l * st['fs']
        u = np.abs(st['As'] - st['Al']) / np.maximum(st['Al'], 1e-9) + np.abs(st['fs'] - st['fl']) / np.maximum(st['fl'], 1e-9)
        s_raw = np.clip((u - U0) / (U1 - U0), 0.0, 1.0)
        st['sal'] = (1 - a_sal) * st['sal'] + a_sal * s_raw; s = st['sal']
    # ---- gardien : contraste σ_Rloc (θ intégrée comme simulate) ----
    hist = cfg.get('history', [])
    phi_prev = np.asarray(hist[-1]['phi_n_t'], dtype=float) if hist and 'phi_n_t' in hist[-1] else np.zeros(n)
    theta = st['phase'] + phi_prev; z = np.exp(1j * theta)
    rloc = np.array([abs(np.sum(pw * z[idx])) if idx.size else 1.0 for idx, pw in st['neigh']]); sigma = float(np.std(rloc))
    if t < t_mem: st['sigma_ref'].append(sigma)
    ref = float(np.median(st['sigma_ref'])) if st['sigma_ref'] else sigma
    garde = float(np.clip((sigma / ref - 0.5) / 0.3, 0.0, 1.0))
    # ---- attention : appuyer la strate surprise sur ses voisines calmes ----
    d = np.zeros(n)
    if K > 0 and t >= t_mem:
        f_nb = np.zeros(n)
        for i, (idx, pw) in enumerate(st['neigh']):
            wcalm = pw * (1.0 - s[idx]) if idx.size else pw
            f_nb[i] = np.sum(wcalm * fn[idx]) / wcalm.sum() if idx.size and wcalm.sum() > 1e-9 else fn[i]
        d = K * s * garde * (f_nb - fn)
        d = st['dprev'] + np.clip(d - st['dprev'], -SLEW, SLEW); st['dprev'] = d
        fn = fn + d
    st['phase'] = st['phase'] + 2 * np.pi * fn * dt
    st['log'].append((t, u.copy(), s.copy(), fn.copy(), (st['fl'].copy() if st['fl'] is not None else fn.copy()), sigma, garde, rloc.copy()))
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
th = col('t'); fl_col = col('fluidity'); dn = col('dispersion_norm'); eff = col('effort(t)'); mu = col('mu_Rloc(t)')
L = st['log']; tl = np.array([l[0] for l in L]); U = np.array([l[1] for l in L]); Sal = np.array([l[2] for l in L])
Fn = np.array([l[3] for l in L]); Fl = np.array([l[4] for l in L]); sig = np.array([l[5] for l in L]); gar = np.array([l[6] for l in L])
f0_new = np.array([s_['f0'] for s_ in res['state']]) if 'state' in res else None
print(f"[{name}] K={K} τ_s={tau_s} τ_l={tau_l} facteur={fac} | durable strates {DUR[0]}–{DUR[-1]} à t={t_dur} ; bref strates {BREF[0]}–{BREF[-1]} à t={t_bref} (2 u.t.)")
print(f"[{name}]   t   | surprise durable | attention durable | assimilation f̂_l/f_courant | surprise bref | attention bref | surprise ailleurs | fluidité (score) | dispersion | σ_Rloc | garde | activité")
rows = []
others = [i for i in range(N) if i not in DUR and i not in BREF]
for a in range(int(t_mem), T, 10):
    m = (tl >= a) & (tl < a + 10); mh = (th >= a) & (th < a + 10)
    ud, sd = U[m][:, DUR].mean(), Sal[m][:, DUR].mean(); ub, sb = U[m][:, BREF].mean(), Sal[m][:, BREF].mean(); uo = U[m][:, others].mean()
    assim = float(np.mean(Fl[m][:, DUR] / Fn[m][:, DUR]))   # 1 = la mémoire longue a rejoint le rythme courant
    flm = float(np.nanmedian(fl_col[mh])); row = dict(t=a, u_dur=float(ud), s_dur=float(sd), assim=assim, u_bref=float(ub), s_bref=float(sb), u_autres=float(uo),
        fluid=flm, fluid_score=int(metrics.score_from_brackets(flm, 'fluidity')) if np.isfinite(flm) else None,
        disp=float(np.nanmedian(dn[mh])) if np.isfinite(dn[mh]).any() else None, sigma=float(sig[m].mean()), garde=float(gar[m].mean()), effort=float(np.nanmedian(eff[mh])))
    rows.append(row)
    print(f"[{name}] {a:3d}-{a+10:<3d}| {ud:.3f} | {sd:.2f} | {assim:.3f} | {ub:.3f} | {sb:.2f} | {uo:.3f} | {flm:.3f} ({row['fluid_score']}) | {row['disp'] if row['disp'] is None else round(row['disp'],4)} | {row['sigma']:.3f} | {row['garde']:.2f} | {row['effort']:.0f}")
sc = metrics.compute_reference_scores(h[-metrics.reference_window(c, dt):], dt, signal='O', N=N)
print(f"[{name}] scores finaux : {metrics.labelled_scores(sc)}")
json.dump({'rows': rows, 'scores': sc}, open('summary.json', 'w'), indent=1)
np.savez_compressed('trace.npz', t=tl, U=U, S=Sal, Fn=Fn, Fl=Fl, sigma=sig, garde=gar)
print("DONE", name)
