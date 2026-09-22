"""Cœur du banc « considérance » (21/09/2026), partagé par
run_considerance_structures.py (les cascades) et run_ablations.py (les éléments de design).

Protocole : un contexte (projecteur gaussien sur l'axe des strates, largeur `width`)
se pose sur la strate 20 à t = 60, tient 20 u.t., migre en 80, puis en 50, puis
disparaît à t = 120. Le geste est celui validé (run_context_binding.py, champ moyen
d'îlot, gate none) :
    Δfₙ = K₀ · sₙ · garde(σ) · |Z_s| · sin(arg Z_s − θₙ) / 2π   (+ κ·sₙ·(f̄_s − fₙ))
Rien n'est ajouté au pipeline ; la mémoire de soi reste un SENS (attention à 0).

Lectures, sans barème :
  substrat (t 40–60)  R moyen, σ_Rloc, fluidité (jerk de f̄), dispersion, activité, C_JS, f̄
  ENGAGEMENT          R dedans − R dedans avant ; corr(s, R) ; temps de montée
  SÉLECTIVITÉ         R dehors − R dehors avant ; σ_Rloc / σ_ref ; dispersion pendant
  RÉVERSIBILITÉ       R de l'ancienne zone +5 / +15 u.t. ; tout après extinction ; surprise
  COÛT                forçage relatif moyen |Δfₙ|/fₙ sur les strates liées
  → efficience = engagement / coût

La lentille Rloc est FIXE : voisines n−1, n+1 à poids égaux (la lentille de couplage
de la spirale ouverte standard), pour rester lisible même quand le couplage est ablaté.
"""
import os, sys, json, io, contextlib, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import simulate, dynamics, metrics, init as fps_init

T_ON = 60.0; POSTS = [20, 80, 50]; DWELL = 20.0; T_OFF = T_ON + DWELL * len(POSTS)
CASCADES = ('doree', 'ratio15', 'ratio10', 'hasard', 'aucune')


def run_bench(name, seed=12345, N=100, T=170, K0=2.0, kappa=0.5, width=6.0,
              cascade='doree', scale=1.0, breath=True, phi_static=False,
              config_mutator=None, label=None, verbose=True):
    """Lance un run et écrit `summary.json` + `kymo.npz` dans le dossier `name`. Renvoie le résumé."""
    assert cascade in CASCADES, cascade
    label = label or name
    cwd0 = os.getcwd(); os.makedirs(name, exist_ok=True); os.chdir(name)
    dt = 0.1
    c = json.load(open(os.path.join(ROOT, 'config.json')))
    c['system']['T'] = T; c['system']['seed'] = seed; c['system']['N'] = N
    c['system']['input']['perturbations'] = [{'type': 'none', 'amplitude': 0.0, 't0': 0.0, 'weight': 1.0}]
    c['analysis'] = {**c.get('analysis', {}), 'compare_kuramoto': False}
    c.setdefault('memory', {})['enabled'] = True; c['memory']['birth_t'] = 20.0
    c.setdefault('attention', {})['consolidation_gain'] = 0.0; c['attention']['anchor_gain'] = 0.0
    if config_mutator is not None:
        config_mutator(c)
    json.dump(c, open('cfg.json', 'w'))
    sp = c.get('spiral', {}); PHI, EPS, OMEGA, THETA = sp.get('phi', 1.618), sp.get('epsilon', 0.1), sp.get('omega', 0.05), sp.get('theta', 0.0)
    R_HASARD = np.random.default_rng(777).uniform(1.2, 2.0, size=N - 1)

    def ratios(t):
        b = EPS * np.sin(2 * np.pi * OMEGA * t + THETA) if breath else 0.0
        if cascade == 'doree':   return np.full(N - 1, PHI) + b
        if cascade == 'ratio15': return np.full(N - 1, 1.5) + b
        if cascade == 'ratio10': return np.full(N - 1, 1.0) + b
        if cascade == 'hasard':  return R_HASARD + b
        return None

    def center_at(t):
        if t < T_ON or t >= T_OFF: return None
        return POSTS[int((t - T_ON) // DWELL)]

    st = {'phase': None, 'sigma_ref': [], 'log': []}
    neigh = [(np.array([j for j in (i - 1, i + 1) if 0 <= j < N]),) for i in range(N)]
    neigh = [(idx, np.full(len(idx), 1.0 / len(idx))) for (idx,) in neigh]
    _fn0 = dynamics.compute_fn; _phi0 = dynamics.compute_phi_n; _sig0 = fps_init.apply_signature_pattern

    def patched_fn(t, state, An_t, F, cfg, _o=_fn0):
        cfg_nc = dict(cfg); cfg_nc['dynamic_parameters'] = {**cfg.get('dynamic_parameters', {}), 'dynamic_phi': False}
        fn = _o(t, state, An_t, None, cfg_nc)
        r = ratios(t)
        if r is not None:
            for n in range(N - 1):
                if fn[n] > 0:
                    fn[n + 1] = 0.5 * fn[n + 1] + 0.5 * r[n] * fn[n]
        if scale != 1.0:
            fn = fn * scale
        if F is not None:
            fn = np.maximum(fn * (1 + np.clip(np.asarray(F, dtype=float)[:N], -0.5, 0.5)), 1e-6)
        f_before = fn.copy()
        if st['phase'] is None:
            st['phase'] = np.zeros(N)
        hist = cfg.get('history', [])
        phi_prev = np.asarray(hist[-1]['phi_n_t'], dtype=float) if hist and 'phi_n_t' in hist[-1] else np.zeros(N)
        theta = st['phase'] + phi_prev
        z = np.exp(1j * theta)
        rloc = np.array([abs(np.sum(pw * z[idx])) for idx, pw in neigh])
        sigma = float(np.std(rloc))
        ctr = center_at(t)
        if ctr is None:
            if t < T_ON: st['sigma_ref'].append(sigma)
            s = np.zeros(N); garde = 1.0
        else:
            s = np.exp(-0.5 * ((np.arange(N) - ctr) / width) ** 2)
            ref = float(np.median(st['sigma_ref'][-200:])) if st['sigma_ref'] else sigma
            garde = float(np.clip((sigma / ref - 0.5) / 0.3, 0.0, 1.0))
            if kappa > 0 and s.sum() > 0:
                f_bar = float(np.sum(s * fn) / np.sum(s)); fn = fn + kappa * garde * s * (f_bar - fn)
            if K0 > 0:
                Z = np.sum(s * np.exp(1j * theta)) / max(np.sum(s), 1e-12)
                fn = fn + K0 * s * garde * np.abs(Z) * np.sin(np.angle(Z) - theta) / (2 * np.pi)
        st['phase'] = st['phase'] + 2 * np.pi * fn * dt
        st['log'].append((t, ctr, sigma, garde, rloc.copy(), s.copy(), np.abs(fn - f_before) / np.maximum(f_before, 1e-9), fn.copy()))
        return fn

    def phi_static_fn(t, state, cfg):
        return np.array([s.get('phi', 0.0) for s in state], dtype=float)

    def sig_random(config, _o=_sig0):
        if config.get('spiral', {}).get('signature_pattern') == 'random':
            rng = np.random.default_rng(config['system'].get('seed', 0) + 4242)
            for s in config.get('strates', []):
                s['phi'] = float(rng.uniform(0.0, 2 * np.pi))
            return config
        return _o(config)

    dynamics.compute_fn = patched_fn
    if phi_static: dynamics.compute_phi_n = phi_static_fn
    fps_init.apply_signature_pattern = sig_random
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf): res = simulate.run_simulation('cfg.json', 'FPS')
    except BaseException as e:
        sys.stderr.write(f"ECHEC {type(e).__name__}: {e}\n" + buf.getvalue()[-2500:] + "\n"); raise
    finally:
        dynamics.compute_fn = _fn0; dynamics.compute_phi_n = _phi0; fps_init.apply_signature_pattern = _sig0
        os.chdir(cwd0)
    os.chdir(name)
    try:
        out = _analyse(res, c, st['log'], N, dt, width, label, verbose)
    finally:
        os.chdir(cwd0)
    return out


def _analyse(res, c, log, N, dt, width, label, verbose):
    h = res['history']
    def col(k): return np.array([np.nan if (x.get(k) is None or isinstance(x.get(k), str)) else x[k] for x in h], dtype=float)
    t = col('t'); eff = col('effort(t)'); dn = col('dispersion_norm'); cjs = col('innovation_cjs')
    U = np.array([np.asarray(x['surprise'], dtype=float) if x.get('surprise') is not None else np.full(N, np.nan) for x in h])
    tl = np.array([l[0] for l in log]); sig = np.array([l[2] for l in log]); gar = np.array([l[3] for l in log])
    R = np.array([l[4] for l in log]); S = np.array([l[5] for l in log]); COST = np.array([l[6] for l in log]); FN = np.array([l[7] for l in log])
    ctrs = np.array([np.nan if l[1] is None else l[1] for l in log])
    def win(a, b): return (tl >= a) & (tl < b)
    def hwin(a, b): return (t >= a) & (t < b)
    def zone(ctr): return np.exp(-0.5 * ((np.arange(N) - ctr) / width) ** 2) > 0.5
    base = win(40, 60); R_base = R[base].mean(0); sig_ref = float(np.median(sig[base])); f_base = FN[base].mean(0)
    fm = FN.mean(1); fb = fm[base]; W = 50
    fluid = float(np.median([metrics.compute_fluidity(fb[i:i + W]) for i in range(0, len(fb) - W, 10)])) if len(fb) > W + 10 else float('nan')
    fm_on = fm[win(60, 120)]
    fluid_on = float(np.median([metrics.compute_fluidity(fm_on[i:i + W]) for i in range(0, len(fm_on) - W, 10)])) if len(fm_on) > W + 10 else float('nan')
    cjs_valid = cjs[np.isfinite(cjs)]
    sub = dict(R_mean=float(R_base.mean()), sigma_ref=sig_ref, fluidity=fluid, fluidity_during=fluid_on,
               disp=float(np.nanmedian(dn[hwin(40, 60)])), effort=float(np.nanmedian(eff[hwin(40, 60)])),
               cjs=float(cjs_valid[-1]) if cjs_valid.size else float('nan'),
               f_mean=float(f_base.mean()), f_spread_log=float(np.std(np.log(np.maximum(f_base, 1e-9)))),
               f_neigh_gap=float(np.median(np.abs(np.diff(np.log(np.maximum(f_base, 1e-9)))))))
    out = {'label': label, 'N': N, 'substrat': sub, 'posts': []}
    if verbose:
        print(f"[{label}] substrat : R {sub['R_mean']:.3f} σ_Rloc {sub['sigma_ref']:.3f} fluidité {sub['fluidity']:.3f} disp {sub['disp']:.4f} activité {sub['effort']:.0f} C_JS {sub['cjs']:.3f} | f̄ {sub['f_mean']:.2f} écart voisines (log) {sub['f_neigh_gap']:.3f}")
    prev = None
    for k, ctr in enumerate(POSTS):
        a = T_ON + k * DWELL; b = a + DWELL
        zin = zone(ctr); zout = S[win(a, b)].mean(0) < 0.05
        Rw = R[win(a, b)]; Sw = S[win(a, b)]
        R_in0 = float(R_base[zin].mean()); R_in_late = float(Rw[int(10 / dt):][:, zin].mean()); R_in_early = float(Rw[:int(5 / dt)][:, zin].mean())
        R_out = float(Rw[:, zout].mean()); R_out0 = float(R_base[zout].mean())
        corr = float(np.mean([np.corrcoef(Sw[i], Rw[i])[0, 1] for i in range(len(Sw))]))
        cost = float(COST[win(a, b)][:, zin].mean())
        rin_t = Rw[:, zin].mean(1); target = R_in0 + 0.5 * (R_in_late - R_in0)
        idx = np.nonzero(rin_t >= target)[0]; rise = float(idx[0] * dt) if idx.size and R_in_late > R_in0 else float('nan')
        row = dict(poste=ctr, R_in0=R_in0, R_in_early=R_in_early, R_in=R_in_late, engagement=R_in_late - R_in0, rise_t=rise, corr=corr,
                   R_out=R_out, R_out0=R_out0, contagion=R_out - R_out0, sigma_ratio=float(sig[win(a, b)].mean() / max(sig_ref, 1e-9)),
                   garde=float(gar[win(a, b)].mean()), cost=cost, efficience=(R_in_late - R_in0) / max(cost, 1e-9),
                   surprise_in=float(np.nanmean(U[hwin(a, b)][:, zin])), surprise_out=float(np.nanmean(U[hwin(a, b)][:, ~zin])),
                   disp=float(np.nanmedian(dn[hwin(a, b)])), effort=float(np.nanmedian(eff[hwin(a, b)])))
        if prev is not None:
            zp = zone(prev)
            row.update(old_R0=float(R_base[zp].mean()), old_R_5=float(R[win(a + 3, a + 7)][:, zp].mean()), old_R_15=float(R[win(a + 13, a + 17)][:, zp].mean()),
                       old_surprise_5=float(np.nanmean(U[hwin(a + 3, a + 7)][:, zp])), old_surprise_15=float(np.nanmean(U[hwin(a + 13, a + 17)][:, zp])))
        out['posts'].append(row); prev = ctr
        if verbose:
            print(f"[{label}] poste {ctr:2d} | R dedans {R_in0:.3f} → {R_in_early:.3f} (5) → {R_in_late:.3f} (fin) montée {rise:.1f} corr {corr:+.2f} | dehors {R_out0:.3f} → {R_out:.3f} | σ/σref {row['sigma_ratio']:.2f} garde {row['garde']:.2f} | coût {cost:.4f} eff {row['efficience']:.1f} | surprise {row['surprise_in']:.3f}/{row['surprise_out']:.3f}"
                  + (f" | ancienne {POSTS[k-1]} : {row['old_R0']:.3f} → {row['old_R_5']:.3f} (+5) {row['old_R_15']:.3f} (+15)" if k > 0 else ""))
    zl = zone(POSTS[-1]); after = {}
    for lag in (5, 15, 30, 45):
        m = win(T_OFF + lag - 2, T_OFF + lag + 2); mh = hwin(T_OFF + lag - 2, T_OFF + lag + 2)
        if not m.any(): continue
        after[str(lag)] = dict(R_last=float(R[m][:, zl].mean()), R_all=float(R[m].mean()), sigma_ratio=float(sig[m].mean() / max(sig_ref, 1e-9)),
                               surprise_last=float(np.nanmean(U[mh][:, zl])), surprise_others=float(np.nanmean(U[mh][:, ~zl])))
    out['after'] = after; out['R_base_mean'] = float(R_base.mean()); out['R_base_last_zone'] = float(R_base[zl].mean())
    sc = metrics.compute_reference_scores(h[-metrics.reference_window(c, dt):], dt, signal='O', N=N); out['scores'] = sc
    P = out['posts']
    out['resume'] = dict(engagement=float(np.mean([p['engagement'] for p in P])), contagion=float(np.mean([p['contagion'] for p in P])),
                         cost=float(np.mean([p['cost'] for p in P])), efficience=float(np.mean([p['engagement'] for p in P]) / max(np.mean([p['cost'] for p in P]), 1e-9)),
                         rise=float(np.nanmedian([p['rise_t'] for p in P])), sigma_ratio=float(np.mean([p['sigma_ratio'] for p in P])),
                         rev5=float(np.mean([p['old_R_5'] - p['old_R0'] for p in P if 'old_R0' in p])), rev15=float(np.mean([p['old_R_15'] - p['old_R0'] for p in P if 'old_R0' in p])),
                         ext15=float(after['15']['R_all'] - out['R_base_mean']) if '15' in after else float('nan'),
                         ext45=float(after['45']['R_all'] - out['R_base_mean']) if '45' in after else float('nan'),
                         disp_during=float(np.mean([p['disp'] for p in P])), fluidity_during=sub['fluidity_during'])
    if verbose:
        r = out['resume']
        print(f"[{label}] résumé : engagement +{r['engagement']:.3f} contagion {r['contagion']:+.3f} coût {r['cost']:.4f} efficience {r['efficience']:.1f} montée {r['rise']:.1f} | σ/σref {r['sigma_ratio']:.2f} | réversibilité {r['rev5']:+.3f}/{r['rev15']:+.3f} extinction {r['ext15']:+.3f}/{r['ext45']:+.3f} | scores {metrics.labelled_scores(sc)}")
    json.dump(out, open('summary.json', 'w'), indent=1)
    O = np.array([np.asarray(x['O'], dtype=float) for x in h])   # sorties par strate : pour l'audibilité d'un îlot dans S(t)
    np.savez_compressed('kymo.npz', t=tl, R=R, s=S, sigma=sig, garde=gar, cost=COST, fn=FN, t_hist=t, U=U, disp=dn, effort=eff, ctr=ctrs, O=O)
    return out
