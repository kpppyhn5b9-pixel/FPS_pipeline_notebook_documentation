"""Prototype (hors pipeline, 21/09/2026) : mémoire de soi à deux vitesses, surprise, attention d'assimilation.

Par strate : mémoire COURTE (τ_s u.t.) et LONGUE (τ_l u.t., suit la courte) de l'amplitude Aₙ et de
la fréquence fₙ. Surprise uₙ = |Â_s − Â_l|/Â_l + |f̂_s − f̂_l|/f̂_l (sans dimension). Une surprise brève
vit dans la courte et s'efface ; une surprise durable passe dans la longue (assimilation).
Attention : saillance sₙ = surprise au-dessus d'un plancher, lissée dans le temps (fluidité) ;
geste = ramener le rythme de la strate surprise vers celui de ses voisines de couplage pondérées par
leur calme (1 − s_j) : Δfₙ = K·sₙ·garde·(f̄_voisines calmes − fₙ), variation limitée par pas.
Événements : DURABLE (f₀ × facteur sur un groupe, à t_dur) et BREF (fₙ × facteur pendant 2 u.t. sur un
autre groupe, à t_bref). On compare attention éteinte (K = 0) et allumée.
usage: python run_surprise_attention.py <nom> <K> <T> [seed] [tau_s] [tau_l] [facteur] [mode]
  mode 'gesture'     : l'attention agit sur le RYTHME (Δfₙ vers les voisines calmes) — inerte sur une cascade
                       lisse et sur un bloc qui change ensemble (f̄_voisines ≈ fₙ), mesuré 21/09 ;
  mode 'consolidate' : l'attention agit sur la MÉMOIRE : la mémoire longue des strates saillantes apprend
                       plus vite (τ_l / (1 + K·sₙ)) — attention = porte de consolidation ; ne touche pas la dynamique.
  mode 'lier'        : l'attention LIE les strates en déficit de RYTHME (tremblement de fₙ : saillance = |Δfₙ| relatif
                       au-dessus de la norme du chœur, lissée) vers leurs voisines calmes, Δf limité par pas ;
  mode 'both'        : consolidation (sur la surprise) + lier (sur le tremblement) — cohabitation.
Troisième événement : TREMBLEMENT (bruit blanc relatif sur fₙ des strates 20–28 à partir de t_jit) — un déficit de
fluidité et d'activité qui vit dans le rythme."""
import sys, os, json, io, contextlib, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # racine du dépôt
sys.path.insert(0, ROOT)
import simulate, dynamics, metrics
name, K, T = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 12345
tau_s = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
tau_l = float(sys.argv[6]) if len(sys.argv) > 6 else 40.0
fac = float(sys.argv[7]) if len(sys.argv) > 7 else 1.15
mode = sys.argv[8] if len(sys.argv) > 8 else 'gesture'
os.makedirs(name, exist_ok=True); os.chdir(name)
dt = 0.1; t_mem = 60.0; t_dur = 90.0; t_bref = 140.0; t_jit = 110.0
JIT = list(range(20, 29)); JIT_AMP = 0.05   # tremblement : ±5 % de fₙ par pas, bruit blanc
DUR = list(range(40, 49)); BREF = list(range(70, 79))
U0, U1, SLEW, TAU_SAL = 0.03, 0.10, 0.02, 3.0   # plancher / plein de saillance (surprise), limite Δf, lissage saillance
c = json.load(open(os.path.join(ROOT, 'config.json')))
c['system']['T'] = T; c['system']['seed'] = seed
c['system']['input']['perturbations'] = [{'type': 'none', 'amplitude': 0.0, 't0': 0.0, 'weight': 1.0}]
c['analysis'] = {**c.get('analysis', {}), 'compare_kuramoto': False}
json.dump(c, open('cfg.json', 'w'))
N = c['system']['N']
st = {'neigh': None, 'As': None, 'fs': None, 'Al': None, 'fl': None, 'sal': None, 'dprev': None, 'dur_done': False, 'log': [], 'sigma_ref': [], 'phase': None,
      'f_prev': None, 'jit_ema': None, 'sal_lier': None, 'rng': np.random.RandomState(seed + 7)}
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
    if t >= t_jit:
        fn = fn.copy(); fn[JIT] = fn[JIT] * (1.0 + JIT_AMP * st['rng'].randn(len(JIT)))   # TREMBLEMENT (déficit de rythme)
    if st['neigh'] is None:
        st['neigh'] = []
        for i in range(n):
            w = np.asarray(state[i]['w'], dtype=float); idx = np.nonzero(np.abs(w) > 1e-12)[0]
            pw = np.abs(w[idx]); pw = pw / pw.sum() if idx.size else pw
            st['neigh'].append((idx, pw))
        st['phase'] = np.zeros(n); st['sal'] = np.zeros(n); st['dprev'] = np.zeros(n)
    # Mémoire de soi sur des quantités RELATIVES au chœur (fₙ / f̄, Aₙ / Ā) : la latence γ fait respirer
    # toutes les fréquences ensemble (±20 %, période 20 u.t.) ; en absolu, la mémoire courte suivait la
    # respiration et la longue la moyennait → surprise permanente partout (premier banc, 21/09).
    # « Toucher au relatif, pas à l'absolu » (Attention.md) vaut aussi pour se souvenir de soi.
    A = np.asarray(An_t, dtype=float)
    A_rel = A / max(float(np.mean(A)), 1e-9); f_rel = fn / max(float(np.mean(fn)), 1e-9)
    # ---- mémoire de soi à deux vitesses (après l'échauffement) ----
    u = np.zeros(n); s = st['sal']
    if t >= t_mem:
        if st['As'] is None:
            st['As'], st['fs'] = A_rel.copy(), f_rel.copy(); st['Al'], st['fl'] = A_rel.copy(), f_rel.copy()
        st['As'] = (1 - a_s) * st['As'] + a_s * A_rel;  st['fs'] = (1 - a_s) * st['fs'] + a_s * f_rel
        # consolidation : là où l'attention se pose, la mémoire longue apprend plus vite (porte de plasticité)
        a_l_n = a_l * (1.0 + K * s) if (mode in ('consolidate', 'both') and K > 0) else a_l
        st['Al'] = (1 - a_l_n) * st['Al'] + a_l_n * st['As']; st['fl'] = (1 - a_l_n) * st['fl'] + a_l_n * st['fs']
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
    garde = float(np.clip((sigma / ref - 0.5) / 0.3, 0.0, 1.0)) if ref > 1e-9 else 1.0   # t=0 : phases nulles, σ = 0
    # ---- saillance de RYTHME : tremblement = |Δfₙ/fₙ| par pas, lissé, au-dessus de la norme du chœur ----
    if st['f_prev'] is None: st['f_prev'] = fn.copy(); st['jit_ema'] = np.zeros(n); st['sal_lier'] = np.zeros(n)
    jit = np.abs(fn - st['f_prev']) / np.maximum(np.abs(fn), 1e-9); st['f_prev'] = fn.copy()
    st['jit_ema'] = 0.9 * st['jit_ema'] + 0.1 * jit                       # ~1 u.t.
    med = max(float(np.median(st['jit_ema'])), 1e-9)
    s_lier_raw = np.clip((st['jit_ema'] / med - 2.0) / 3.0, 0.0, 1.0)     # 0 à 2× la norme, 1 à 5×
    st['sal_lier'] = (1 - a_sal) * st['sal_lier'] + a_sal * s_lier_raw
    s_lier = st['sal_lier']
    # ---- attention : appuyer la strate saillante sur ses voisines calmes ----
    d = np.zeros(n)
    if K > 0 and t >= t_mem and mode in ('gesture', 'lier', 'both'):
        s_g = s if mode == 'gesture' else s_lier                          # 'gesture' : sur la surprise ; 'lier'/'both' : sur le rythme
        f_nb = np.zeros(n)
        for i, (idx, pw) in enumerate(st['neigh']):
            wcalm = pw * (1.0 - s_g[idx]) if idx.size else pw
            f_nb[i] = np.sum(wcalm * fn[idx]) / wcalm.sum() if idx.size and wcalm.sum() > 1e-9 else fn[i]
        d = K * s_g * garde * (f_nb - fn)
        d = st['dprev'] + np.clip(d - st['dprev'], -SLEW, SLEW); st['dprev'] = d
        fn = fn + d
    st['phase'] = st['phase'] + 2 * np.pi * fn * dt
    st['log'].append((t, u.copy(), s.copy(), f_rel.copy(), (st['fl'].copy() if st['fl'] is not None else f_rel.copy()), sigma, garde, rloc.copy(), fn.copy(), s_lier.copy()))
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
print(f"[{name}] K={K} mode={mode} τ_s={tau_s} τ_l={tau_l} facteur={fac} | durable strates {DUR[0]}–{DUR[-1]} à t={t_dur} ; bref strates {BREF[0]}–{BREF[-1]} à t={t_bref} (2 u.t.)")
Fabs = np.array([l[8] for l in L]); Slier = np.array([l[9] for l in L])
def jerk_rel(F):   # jerk relatif par strate : 3ᵉ différence de fₙ / fₙ, moyenne absolue sur la fenêtre
    d3 = np.diff(F, n=3, axis=0) / np.maximum(F[3:], 1e-9)
    return np.abs(d3).mean(0)
print(f"[{name}]   t   | surprise durable | attention durable | assimilation | surprise bref | surprise ailleurs | TREMBLEMENT : jerk groupe / jerk autres | attention-lier groupe | surprise groupe | fluidité (score) | dispersion | σ_Rloc | activité")
rows = []
others = [i for i in range(N) if i not in DUR and i not in BREF and i not in JIT]
for a in range(int(t_mem), T, 10):
    m = (tl >= a) & (tl < a + 10); mh = (th >= a) & (th < a + 10)
    ud, sd = U[m][:, DUR].mean(), Sal[m][:, DUR].mean(); ub, sb = U[m][:, BREF].mean(), Sal[m][:, BREF].mean(); uo = U[m][:, others].mean()
    assim = float(np.mean(Fl[m][:, DUR] / Fn[m][:, DUR]))   # 1 = la mémoire longue a rejoint le rythme courant
    jk = jerk_rel(Fabs[m]); jg, jo = float(jk[JIT].mean()), float(jk[others].mean()); slg = float(Slier[m][:, JIT].mean()); ug = float(U[m][:, JIT].mean())
    flm = float(np.nanmedian(fl_col[mh])); row = dict(t=a, u_dur=float(ud), s_dur=float(sd), assim=assim, u_bref=float(ub), s_bref=float(sb), u_autres=float(uo),
        jerk_jit=jg, jerk_autres=jo, s_lier_jit=slg, u_jit=ug,
        fluid=flm, fluid_score=int(metrics.score_from_brackets(flm, 'fluidity')) if np.isfinite(flm) else None,
        disp=float(np.nanmedian(dn[mh])) if np.isfinite(dn[mh]).any() else None, sigma=float(sig[m].mean()), garde=float(gar[m].mean()), effort=float(np.nanmedian(eff[mh])))
    rows.append(row)
    print(f"[{name}] {a:3d}-{a+10:<3d}| {ud:.3f} | {sd:.2f} | {assim:.3f} | {ub:.3f} | {uo:.3f} | {jg:.4f} / {jo:.4f} | {slg:.2f} | {ug:.3f} | {flm:.3f} ({row['fluid_score']}) | {row['disp'] if row['disp'] is None else round(row['disp'],4)} | {row['sigma']:.3f} | {row['effort']:.0f}")
sc = metrics.compute_reference_scores(h[-metrics.reference_window(c, dt):], dt, signal='O', N=N)
print(f"[{name}] scores finaux : {metrics.labelled_scores(sc)}")
json.dump({'rows': rows, 'scores': sc}, open('summary.json', 'w'), indent=1)
np.savez_compressed('trace.npz', t=tl, U=U, S=Sal, Fn=Fn, Fl=Fl, sigma=sig, garde=gar)
print("DONE", name)
