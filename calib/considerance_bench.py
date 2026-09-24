"""Banc « considérance » (21–22/09/2026) : un contexte se pose sur le chœur, tient, migre,
disparaît ; on lit ce que le substrat en fait. Rien n'est ajouté au pipeline.

Le geste (validé sur run_context_binding.py, champ moyen d'îlot) :
    Δfₙ = K₀ · sₙ · garde(σ) · |Z_s| · sin(arg Z_s − θₙ) / 2π   (+ κ·sₙ·(f̄_s − fₙ))
Le gardien garde(σ) lit le contraste σ_Rloc (l'identité chimère) : 1 tant qu'il reste
≥ 0.8 × sa référence, 0 à 0.5 ×.

Qui désigne l'îlot (la SAILLANCE sₙ), au choix :
  projector  le projecteur gaussien lui-même (la saillance est donnée)
  context    le CONTEXTE RÉEL : Iₙ reçoit une bosse gaussienne (+ctx_gain) qui suit les
             postes ; la saillance est lue dans l'AMPLITUDE (excédent relatif du présent
             lissé de la mémoire de soi, a_p / médiane − 1, échelle ctx_scale)
  input      même bosse d'entrée, mais la saillance est lue sur Iₙ lui-même (l'entrée,
             normalisée par la bosse) : le contexte tel qu'il arrive, pas tel qu'il s'exprime
  surprise   la SURPRISE de la mémoire de soi (uₙ / médiane, seuil 3, plein 6, lissée 3 u.t.) :
             là où le soi se reconfigure
  both       max(context, surprise)
Un ÉVÉNEMENT optionnel (f₀ × facteur, durable, sur quelques strates à t_ev) sert à
tester la source « surprise » : s'allume-t-elle sur les strates qui changent, et se
relâche-t-elle une fois le changement assimilé (~τ_l) ?

Lectures, sans barème :
  substrat (t 40–60)  R moyen, σ_Rloc, fluidité (jerk de f̄), dispersion, activité, C_JS, f̄
  ENGAGEMENT          R dedans − R dedans avant ; corr(s, R) ; temps de montée
  SÉLECTIVITÉ         R dehors − R dehors avant ; σ_Rloc / σ_ref ; part de strates saillantes hors zone
  RÉVERSIBILITÉ       R de l'ancienne zone +5 / +15 u.t. ; tout après extinction ; surprise
  COÛT                forçage relatif moyen |Δfₙ|/fₙ sur les strates liées → efficience = engagement / coût
  AUDIBILITÉ          part de la variance de S(t) portée par l'îlot (et somme / voix indépendantes)

La lentille Rloc est FIXE : voisines n−1, n+1 à poids égaux.
"""
import os, sys, json, io, contextlib, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import simulate, dynamics, metrics, perturbations, init as fps_init

T_ON = 60.0; POSTS = [20, 80, 50]; DWELL = 20.0; T_OFF = T_ON + DWELL * len(POSTS)
CASCADES = ('doree', 'ratio15', 'ratio10', 'hasard', 'aucune')
SALIENCIES = ('projector', 'context', 'input', 'surprise', 'both')   # 'input' : la saillance est lue sur Iₙ lui-même (l'entrée), la bosse d'entrée étant active


def run_bench(name, seed=12345, N=100, T=170, K0=2.0, kappa=0.5, width=6.0,
              cascade='doree', scale=1.0, breath=True, phi_static=False,
              saliency='projector', posts=True, ctx_gain=0.5, ctx_scale=0.3,
              surp_floor=3.0, surp_full=6.0, sal_tau=3.0,
              schedule=None, novelty=False, novelty_scale=2.0, island='global',
              maintain=None, attach=None, event=None, config_mutator=None, label=None, verbose=True):
    """Lance un run, écrit `summary.json` + `kymo.npz` dans le dossier `name`, renvoie le résumé.
    event : None ou dict(t=90.0, strates=range(40, 45), factor=1.3) — f₀ × facteur, durable.
    schedule : None (postes 20 → 80 → 50, 20 u.t. chacun) ou liste de (t0, t1, [centres]) : plusieurs
               foyers à la fois, tenue libre. Avec un horaire libre, l'analyse par postes est sautée
               (posts=False), le runner lit kymo.npz (R, s, U, O, theta).
    novelty : saillance × nouveauté pour soi, ν = clip((uₙ / médiane(u) − 1) / novelty_scale, 0, 1) :
              ce qui est déjà assimilé par la mémoire longue cesse d'être saillant (habituation).
    island : 'global' (un seul champ moyen sur toutes les strates saillantes) ou 'local' (un champ
             moyen par îlot = composante connexe de s > 0.05 : jamais de cohérence non locale).
    maintain : None ou liste de (t0, t1, {centre: m}) — la PRIORITÉ DE MAINTIEN de Gepetto (24/09),
               fournie de l'extérieur, par foyer : s = c · [ν + (1 − ν)·m]. m = 0 partout redonne
               s = c·ν ; ν = 0 laisse s = c·m : le familier peut rester considéré sans surprendre.
    attach : None ou dict(tau_m=20.0, apply=True, noise=[(t0, t1, std)]) — l'ATTACHEMENT PAR ASSOCIATION
             (24/09) : m_k s'apprend DEDANS pour chaque foyer k présent, comme une moyenne mobile
             (τ_m) du bien-être du soi pendant sa présence, w = clip(2 − activité/repos, 0, 1) :
             1 quand le système est à son repos ou en dessous, 0 quand il fait le double. Jamais
             l'effet du foyer, seulement l'état du soi quand il était là. `noise` ajoute un bruit
             blanc à Iₙ (par strate) pendant des fenêtres : une période d'effort ; un 4e élément
             'foyer' porte ce bruit par la bosse du foyer présent (le foyer lui-même épuisant). `apply=False`
             apprend m sans l'appliquer (témoin). `learn_until` fige l'apprentissage après cet instant
             (pour lire un retour sans que la présence au calme ne réécrive m)."""
    assert cascade in CASCADES, cascade; assert saliency in SALIENCIES, saliency
    label = label or name
    cwd0 = os.getcwd(); os.makedirs(name, exist_ok=True); os.chdir(name)
    dt = 0.1
    c = json.load(open(os.path.join(ROOT, 'config.json')))
    c['system']['T'] = T; c['system']['seed'] = seed; c['system']['N'] = N
    c['system']['input']['perturbations'] = [{'type': 'none', 'amplitude': 0.0, 't0': 0.0, 'weight': 1.0}]
    c['analysis'] = {**c.get('analysis', {}), 'compare_kuramoto': False}
    c.setdefault('memory', {})['enabled'] = True; c['memory']['birth_t'] = 20.0
    if config_mutator is not None:
        config_mutator(c)
    json.dump(c, open('cfg.json', 'w'))
    birth_t = float(c['memory']['birth_t']); tau_s = float(c['memory'].get('tau_s_t', 2.0)); tau_l = float(c['memory'].get('tau_l_t', 40.0))
    sp = c.get('spiral', {}); PHI, EPS, OMEGA, THETA = sp.get('phi', 1.618), sp.get('epsilon', 0.1), sp.get('omega', 0.05), sp.get('theta', 0.0)
    R_HASARD = np.random.default_rng(777).uniform(1.2, 2.0, size=N - 1)
    ev = None
    if event is not None:
        ev = dict(t=float(event.get('t', 90.0)), strates=np.array(list(event.get('strates', range(40, 45))), dtype=int),
                  factor=float(event.get('factor', 1.3)), done=False)

    def ratios(t):
        b = EPS * np.sin(2 * np.pi * OMEGA * t + THETA) if breath else 0.0
        if cascade == 'doree':   return np.full(N - 1, PHI) + b
        if cascade == 'ratio15': return np.full(N - 1, 1.5) + b
        if cascade == 'ratio10': return np.full(N - 1, 1.0) + b
        if cascade == 'hasard':  return R_HASARD + b
        return None

    if schedule is not None:
        posts = False

    def center_at(t):
        if schedule is not None:
            for (t0, t1, ctrs_) in schedule:
                if t0 <= t < t1: return list(ctrs_)
            return None
        if not posts or t < T_ON or t >= T_OFF: return None
        return POSTS[int((t - T_ON) // DWELL)]

    def bump(ctr):
        cs = ctr if isinstance(ctr, (list, tuple)) else [ctr]
        return np.max([np.exp(-0.5 * ((np.arange(N) - c_) / width) ** 2) for c_ in cs], axis=0)

    st = {'phase': None, 'sigma_ref': [], 'log': [], 'u_prev': np.zeros(N), 's_surp': np.zeros(N), 'in_last': np.zeros(N), 'm_last': np.zeros(N),
          'm_learn': {}, 'm_learn_log': [], 'w_log': []}
    rng_noise = np.random.default_rng(seed + 99)
    bm = dynamics.init_self_memory(N)          # mémoire de soi côté banc (mêmes entrées que le pipeline → même état)
    neigh = [(np.array([j for j in (i - 1, i + 1) if 0 <= j < N]),) for i in range(N)]
    neigh = [(idx, np.full(len(idx), 1.0 / len(idx))) for (idx,) in neigh]
    _fn0 = dynamics.compute_fn; _phi0 = dynamics.compute_phi_n; _sig0 = fps_init.apply_signature_pattern; _in0 = perturbations.compute_In
    k_sal = min(1.0, dt / max(sal_tau, dt))

    def patched_in(t, cfg, state=None, history=None, dt_=0.05, _o=_in0):
        base = _o(t, cfg, state, history, dt_)
        ctr = center_at(t)
        noise = np.zeros(N)
        if attach is not None:
            for spec in attach.get('noise', []):
                t0, t1, sd = spec[:3]; where = spec[3] if len(spec) > 3 else 'global'
                if t0 <= t < t1:
                    noise = rng_noise.normal(0.0, float(sd), N)
                    if where == 'foyer':                                 # le foyer LUI-MÊME est épuisant : bruit porté par sa bosse
                        noise = noise * bump(ctr) if ctr is not None else np.zeros(N)
        if saliency in ('context', 'input', 'both') and ctr is not None:
            v = np.full(N, float(base)) + ctx_gain * bump(ctr)      # le contexte RÉEL : une bosse d'entrée
            st['in_last'] = v - float(base); return v + noise
        st['in_last'] = np.zeros(N); return (np.full(N, float(base)) + noise) if np.any(noise) else base

    def patched_fn(t, state, An_t, F, cfg, _o=_fn0):
        if ev is not None and not ev['done'] and t >= ev['t']:
            for n in ev['strates']: state[n]['f0'] = state[n]['f0'] * ev['factor']
            ev['done'] = True
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
        if t < T_ON: st['sigma_ref'].append(sigma)
        # ---- la saillance : qui désigne l'îlot ----
        s_ctx = np.zeros(N)
        if saliency == 'projector' and ctr is not None:
            s_ctx = bump(ctr)
        elif saliency == 'input':
            s_ctx = np.clip(st.get('in_last', np.zeros(N)) / max(ctx_gain, 1e-9), 0.0, 1.0)   # l'entrée elle-même, normalisée par la bosse
        elif saliency in ('context', 'both') and bm.get('born'):
            ap = bm['a_p']; s_ctx = np.clip((ap / max(float(np.median(ap)), 1e-9) - 1.0) / ctx_scale, 0.0, 1.0)
        if saliency in ('surprise', 'both') and bm.get('born'):
            u = st['u_prev']; raw = np.clip((u / max(float(np.median(u)), 1e-9) - surp_floor) / max(surp_full - surp_floor, 1e-9), 0.0, 1.0)
            st['s_surp'] = st['s_surp'] + k_sal * (raw - st['s_surp'])
        s = np.maximum(s_ctx, st['s_surp']) if saliency == 'both' else (st['s_surp'] if saliency == 'surprise' else s_ctx)
        nu = np.ones(N)
        if novelty and bm.get('born'):
            u = st['u_prev']; nu = np.clip((u / max(float(np.median(u)), 1e-9) - 1.0) / max(novelty_scale, 1e-9), 0.0, 1.0)
            m_vec = np.zeros(N)
            if attach is not None:
                rel = hist[-1].get('activite_rel') if hist else None
                w = float(np.clip(2.0 - float(rel), 0.0, 1.0)) if rel is not None else None
                k_m = min(1.0, dt / max(float(attach.get('tau_m', 20.0)), dt))
                if ctr is not None and w is not None and t < float(attach.get('learn_until', 1e12)):
                    for c_ in ctr:
                        st['m_learn'][c_] = st['m_learn'].get(c_, 0.0) + k_m * (w - st['m_learn'].get(c_, 0.0))
                st['w_log'].append((t, w if w is not None else np.nan)); st['m_learn_log'].append((t, dict(st['m_learn'])))
                if attach.get('apply', True) and ctr is not None:
                    num = np.zeros(N); den = np.zeros(N)
                    for c_ in ctr:
                        b_ = np.exp(-0.5 * ((np.arange(N) - float(c_)) / width) ** 2); num += b_ * st['m_learn'].get(c_, 0.0); den += b_
                    m_vec = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)
            elif maintain is not None and ctr is not None:
                num = np.zeros(N); den = np.zeros(N)
                for (t0, t1, m_by_ctr) in maintain:
                    if t0 <= t < t1:
                        for c_, m_ in m_by_ctr.items():
                            b_ = np.exp(-0.5 * ((np.arange(N) - float(c_)) / width) ** 2); num += b_ * float(m_); den += b_
                m_vec = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)
            s = s * (nu + (1.0 - nu) * m_vec)
        st['m_last'] = m_vec if (novelty and bm.get('born')) else np.zeros(N)
        ref = float(np.median(st['sigma_ref'][-200:])) if st['sigma_ref'] else sigma
        garde = float(np.clip((sigma / ref - 0.5) / 0.3, 0.0, 1.0)) if (ref > 1e-6 and len(st['sigma_ref']) >= 100) else 1.0   # gardien : nul tant que le contraste de référence n'est pas établi
        # ---- le geste ----
        if s.sum() > 1e-9:
            if kappa > 0:
                f_bar = float(np.sum(s * fn) / np.sum(s)); fn = fn + kappa * garde * s * (f_bar - fn)
            if K0 > 0:
                if island == 'local':
                    on = s > 0.05; comps = []; i = 0
                    while i < N:
                        if on[i]:
                            j = i
                            while j + 1 < N and on[j + 1]: j += 1
                            comps.append(np.arange(i, j + 1)); i = j + 1
                        else: i += 1
                    for comp in comps:
                        sc_ = s[comp]; Zk = np.sum(sc_ * np.exp(1j * theta[comp])) / max(np.sum(sc_), 1e-12)
                        fn[comp] = fn[comp] + K0 * sc_ * garde * np.abs(Zk) * np.sin(np.angle(Zk) - theta[comp]) / (2 * np.pi)
                else:
                    Z = np.sum(s * np.exp(1j * theta)) / max(np.sum(s), 1e-12)
                    fn = fn + K0 * s * garde * np.abs(Z) * np.sin(np.angle(Z) - theta) / (2 * np.pi)
        st['phase'] = st['phase'] + 2 * np.pi * fn * dt
        # ---- la mémoire côté banc (mêmes entrées que le pipeline : An_t, fn rendu) ----
        u_now = np.full(N, np.nan)
        if t >= birth_t:
            u_now = dynamics.update_self_memory(bm, An_t, fn, dt, tau_s, tau_l)['u']; st['u_prev'] = u_now
        st['log'].append((t, (ctr[0] if isinstance(ctr, list) else ctr), sigma, garde, rloc.copy(), s.copy(), np.abs(fn - f_before) / np.maximum(f_before, 1e-9), fn.copy(), u_now.copy(), theta.copy(), nu.copy(), st['m_last'].copy()))
        return fn

    def phi_static_fn(t, state, cfg):
        return np.array([s.get('phi', 0.0) for s in state], dtype=float)

    def sig_random(config, _o=_sig0):
        if config.get('spiral', {}).get('signature_pattern') == 'random':
            rng = np.random.default_rng(config['system'].get('seed', 0) + 4242)
            for s_ in config.get('strates', []):
                s_['phi'] = float(rng.uniform(0.0, 2 * np.pi))
            return config
        return _o(config)

    dynamics.compute_fn = patched_fn; perturbations.compute_In = patched_in
    if phi_static: dynamics.compute_phi_n = phi_static_fn
    fps_init.apply_signature_pattern = sig_random
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf): res = simulate.run_simulation('cfg.json', 'FPS')
    except BaseException as e:
        sys.stderr.write(f"ECHEC {type(e).__name__}: {e}\n" + buf.getvalue()[-2500:] + "\n"); raise
    finally:
        dynamics.compute_fn = _fn0; dynamics.compute_phi_n = _phi0; fps_init.apply_signature_pattern = _sig0; perturbations.compute_In = _in0
        os.chdir(cwd0)
    os.chdir(name)
    try:
        out = _analyse(res, c, st['log'], N, dt, width, label, verbose, posts, ev, saliency, st)
    finally:
        os.chdir(cwd0)
    return out


def _audib(O, zin):
    """Audibilité d'une zone dans S = ΣO : (RMS de la somme de la zone, somme / voix indépendantes, part de var(S))."""
    somme = float(np.sqrt(np.mean(O[:, zin].sum(1) ** 2)))
    indep = float(np.sqrt(np.sum(np.mean(O[:, zin] ** 2, 0))))
    S = O.sum(1); part = float(np.mean(O[:, zin].sum(1) * S) / max(np.mean(S ** 2), 1e-12))
    return somme, somme / max(indep, 1e-12), part


def _analyse(res, c, log, N, dt, width, label, verbose, posts, ev, saliency, st_extra=None):
    h = res['history']
    def col(k): return np.array([np.nan if (x.get(k) is None or isinstance(x.get(k), str)) else x[k] for x in h], dtype=float)
    t = col('t'); eff = col('effort(t)'); dn = col('dispersion_norm'); cjs = col('innovation_cjs')
    O = np.array([np.asarray(x['O'], dtype=float) for x in h])
    tl = np.array([l[0] for l in log]); sig = np.array([l[2] for l in log]); gar = np.array([l[3] for l in log])
    R = np.array([l[4] for l in log]); S = np.array([l[5] for l in log]); COST = np.array([l[6] for l in log]); FN = np.array([l[7] for l in log]); U = np.array([l[8] for l in log])
    TH = np.array([l[9] for l in log]); NU = np.array([l[10] for l in log]); MM = np.array([l[11] for l in log])
    ctrs = np.array([np.nan if l[1] is None else l[1] for l in log])
    n_h = min(len(t), len(tl)); t = t[:n_h]; O = O[:n_h]; eff = eff[:n_h]; dn = dn[:n_h]; cjs = cjs[:n_h]
    def win(a, b): return (tl >= a) & (tl < b)
    def hwin(a, b): return (t >= a) & (t < b)
    def zone(ctr): return np.exp(-0.5 * ((np.arange(N) - ctr) / width) ** 2) > 0.5
    base = win(40, 60); R_base = R[base].mean(0); sig_ref = float(np.median(sig[base])); f_base = FN[base].mean(0)
    fm = FN.mean(1); fb = fm[base]; W = 50
    fluid = float(np.median([metrics.compute_fluidity(fb[i:i + W]) for i in range(0, len(fb) - W, 10)])) if len(fb) > W + 10 else float('nan')
    fm_on = fm[win(60, 120)]
    fluid_on = float(np.median([metrics.compute_fluidity(fm_on[i:i + W]) for i in range(0, len(fm_on) - W, 10)])) if len(fm_on) > W + 10 else float('nan')
    cjs_valid = cjs[np.isfinite(cjs)]
    calm = win(30, 60)
    sub = dict(R_mean=float(R_base.mean()), sigma_ref=sig_ref, fluidity=fluid, fluidity_during=fluid_on,
               disp=float(np.nanmedian(dn[hwin(40, 60)])), effort=float(np.nanmedian(eff[hwin(40, 60)])),
               cjs=float(cjs_valid[-1]) if cjs_valid.size else float('nan'),
               f_mean=float(f_base.mean()), f_spread_log=float(np.std(np.log(np.maximum(f_base, 1e-9)))),
               f_neigh_gap=float(np.median(np.abs(np.diff(np.log(np.maximum(f_base, 1e-9)))))),
               calm_salient_share=float(np.mean(S[calm] > 0.5)), calm_cost=float(COST[calm].mean()))
    out = {'label': label, 'N': N, 'saliency': saliency, 'substrat': sub, 'posts': [], 'event': None}
    if verbose:
        print(f"[{label}] substrat : R {sub['R_mean']:.3f} σ_Rloc {sub['sigma_ref']:.3f} fluidité {sub['fluidity']:.3f} disp {sub['disp']:.4f} activité {sub['effort']:.0f} C_JS {sub['cjs']:.3f} | f̄ {sub['f_mean']:.2f} écart voisines {sub['f_neigh_gap']:.3f} | au calme (30–60) : strates saillantes {100*sub['calm_salient_share']:.1f} %, coût {sub['calm_cost']:.4f}")
    prev = None
    if posts:
        for k, ctr in enumerate(POSTS):
            a = T_ON + k * DWELL; b = a + DWELL
            zin = zone(ctr); zout = np.exp(-0.5 * ((np.arange(N) - ctr) / width) ** 2) < 0.05
            Rw = R[win(a, b)]; Sw = S[win(a, b)]
            R_in0 = float(R_base[zin].mean()); R_in_late = float(Rw[int(10 / dt):][:, zin].mean()); R_in_early = float(Rw[:int(5 / dt)][:, zin].mean())
            R_out = float(Rw[:, zout].mean()); R_out0 = float(R_base[zout].mean())
            corr = float(np.nanmean([np.corrcoef(Sw[i], Rw[i])[0, 1] if Sw[i].std() > 0 else np.nan for i in range(len(Sw))]))
            cost = float(COST[win(a, b)][:, zin].mean())
            rin_t = Rw[:, zin].mean(1); target = R_in0 + 0.5 * (R_in_late - R_in0)
            idx = np.nonzero(rin_t >= target)[0]; rise = float(idx[0] * dt) if idx.size and R_in_late > R_in0 else float('nan')
            somme, ratio, part = _audib(O[hwin(a + 5, b)], zin); s0, r0, p0 = _audib(O[hwin(40, 60)], zin)
            row = dict(poste=ctr, R_in0=R_in0, R_in_early=R_in_early, R_in=R_in_late, engagement=R_in_late - R_in0, rise_t=rise, corr=corr,
                       R_out=R_out, R_out0=R_out0, contagion=R_out - R_out0, sigma_ratio=float(sig[win(a, b)].mean() / max(sig_ref, 1e-9)),
                       garde=float(gar[win(a, b)].mean()), cost=cost, efficience=(R_in_late - R_in0) / max(cost, 1e-9),
                       s_in=float(Sw[:, zin].mean()), s_out_share=float(np.mean(Sw[:, zout] > 0.5)),
                       surprise_in=float(np.nanmean(U[win(a, b)][:, zin])), surprise_out=float(np.nanmean(U[win(a, b)][:, ~zin])),
                       audib_part=part, audib_ratio=ratio, audib_part0=p0, audib_ratio0=r0,
                       disp=float(np.nanmedian(dn[hwin(a, b)])), effort=float(np.nanmedian(eff[hwin(a, b)])))
            if prev is not None:
                zp = zone(prev)
                row.update(old_R0=float(R_base[zp].mean()), old_R_5=float(R[win(a + 3, a + 7)][:, zp].mean()), old_R_15=float(R[win(a + 13, a + 17)][:, zp].mean()))
            out['posts'].append(row); prev = ctr
            if verbose:
                print(f"[{label}] poste {ctr:2d} | R dedans {R_in0:.3f} → {R_in_late:.3f} montée {rise:.1f} | dehors {R_out0:.3f} → {R_out:.3f} | s dedans {row['s_in']:.2f} saillantes dehors {100*row['s_out_share']:.1f} % | σ/σref {row['sigma_ratio']:.2f} garde {row['garde']:.2f} | coût {cost:.4f} eff {row['efficience']:.1f} | audibilité {100*p0:.0f} % → {100*part:.0f} % (somme/indép {r0:.2f} → {ratio:.2f})"
                      + (f" | ancienne {POSTS[k-1]} : {row['old_R0']:.3f} → {row['old_R_5']:.3f} (+5) {row['old_R_15']:.3f} (+15)" if k > 0 else ""))
        zl = zone(POSTS[-1]); after = {}
        for lag in (5, 15, 30, 45):
            m = win(T_OFF + lag - 2, T_OFF + lag + 2); mh = hwin(T_OFF + lag - 2, T_OFF + lag + 2)
            if not m.any(): continue
            after[str(lag)] = dict(R_last=float(R[m][:, zl].mean()), R_all=float(R[m].mean()), sigma_ratio=float(sig[m].mean() / max(sig_ref, 1e-9)),
                                   audib_part=_audib(O[mh], zl)[2], salient_share=float(np.mean(S[m] > 0.5)))
        out['after'] = after; out['R_base_mean'] = float(R_base.mean())
    if ev is not None:
        zin = np.zeros(N, dtype=bool); zin[ev['strates']] = True; t0 = ev['t']; rows = []
        for (a, b) in ((t0 - 20, t0), (t0, t0 + 10), (t0 + 10, t0 + 30), (t0 + 30, t0 + 60), (t0 + 60, tl[-1] + dt)):
            m = win(a, b); mh = hwin(a, b)
            if not m.any(): continue
            somme, ratio, part = _audib(O[mh], zin)
            rows.append(dict(t0=float(a), t1=float(b), R_in=float(R[m][:, zin].mean()), R_out=float(R[m][:, ~zin].mean()),
                             s_in=float(S[m][:, zin].mean()), s_out_share=float(np.mean(S[m][:, ~zin] > 0.5)),
                             surprise_in=float(np.nanmean(U[m][:, zin])), surprise_out=float(np.nanmean(U[m][:, ~zin])),
                             cost_in=float(COST[m][:, zin].mean()), sigma_ratio=float(sig[m].mean() / max(sig_ref, 1e-9)), audib_part=part, audib_ratio=ratio))
        out['event'] = dict(t=t0, strates=[int(x) for x in ev['strates']], factor=ev['factor'], rows=rows)
        if verbose:
            for r in rows:
                print(f"[{label}] événement (f₀×{ev['factor']} sur {len(ev['strates'])} strates à t={t0:.0f}) t∈[{r['t0']:.0f},{r['t1']:.0f}) | R zone {r['R_in']:.3f} autres {r['R_out']:.3f} | s zone {r['s_in']:.2f} saillantes ailleurs {100*r['s_out_share']:.1f} % | surprise zone {r['surprise_in']:.3f} autres {r['surprise_out']:.3f} | coût zone {r['cost_in']:.4f} | σ/σref {r['sigma_ratio']:.2f} | audibilité {100*r['audib_part']:.0f} % (somme/indép {r['audib_ratio']:.2f})")
    sc = metrics.compute_reference_scores(h[-metrics.reference_window(c, dt):], dt, signal='O', N=N); out['scores'] = sc
    P = out['posts']
    if P:
        after = out['after']
        out['resume'] = dict(engagement=float(np.mean([p['engagement'] for p in P])), contagion=float(np.mean([p['contagion'] for p in P])),
                             cost=float(np.mean([p['cost'] for p in P])), efficience=float(np.mean([p['engagement'] for p in P]) / max(np.mean([p['cost'] for p in P]), 1e-9)),
                             rise=float(np.nanmedian([p['rise_t'] for p in P])), sigma_ratio=float(np.mean([p['sigma_ratio'] for p in P])),
                             rev5=float(np.mean([p['old_R_5'] - p['old_R0'] for p in P if 'old_R0' in p])), rev15=float(np.mean([p['old_R_15'] - p['old_R0'] for p in P if 'old_R0' in p])),
                             ext15=float(after['15']['R_all'] - out['R_base_mean']) if '15' in after else float('nan'),
                             ext45=float(after['45']['R_all'] - out['R_base_mean']) if '45' in after else float('nan'),
                             audib_part=float(np.mean([p['audib_part'] for p in P])), audib_part0=float(np.mean([p['audib_part0'] for p in P])),
                             s_out_share=float(np.mean([p['s_out_share'] for p in P])),
                             disp_during=float(np.mean([p['disp'] for p in P])), fluidity_during=sub['fluidity_during'])
        if verbose:
            r = out['resume']
            print(f"[{label}] résumé : engagement +{r['engagement']:.3f} contagion {r['contagion']:+.3f} coût {r['cost']:.4f} efficience {r['efficience']:.1f} montée {r['rise']:.1f} | σ/σref {r['sigma_ratio']:.2f} | réversibilité {r['rev5']:+.3f}/{r['rev15']:+.3f} extinction {r['ext15']:+.3f}/{r['ext45']:+.3f} | audibilité {100*r['audib_part0']:.0f} % → {100*r['audib_part']:.0f} % | scores {metrics.labelled_scores(sc)}")
    elif verbose:
        print(f"[{label}] scores {metrics.labelled_scores(sc)}")
    json.dump(out, open('summary.json', 'w'), indent=1)
    np.savez_compressed('kymo.npz', t=tl, R=R, s=S, sigma=sig, garde=gar, cost=COST, fn=FN, U=U, t_hist=t, disp=dn, effort=eff, ctr=ctrs, O=O, theta=TH, nu=NU, m=MM)
    if st_extra is not None and st_extra.get('m_learn_log'):
        cs = sorted({c_ for (_, d_) in st_extra['m_learn_log'] for c_ in d_})
        np.savez_compressed('attach.npz', t=np.array([x[0] for x in st_extra['m_learn_log']]), centers=np.array(cs),
                            m=np.array([[d_.get(c_, 0.0) for c_ in cs] for (_, d_) in st_extra['m_learn_log']]),
                            w=np.array([x[1] for x in st_extra['w_log']]))
    return out
