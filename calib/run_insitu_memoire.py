"""Les bancs de la mémoire des moments chéris, IN SITU (25/09/2026) : le pipeline natif
(simulate.run_simulation, config.json + context.foyers), ses colonnes, ses lecteurs.
Seul le stimulus est sculpté de l'extérieur : foyers et ambiances par `context.foyers`
(une ambiance = un foyer de largeur 1e6 : un décalage global de Iₙ), bruit par un patch de
perturbations.compute_In (fenêtres, global ou porté par une zone). Rien n'est patché dans
dynamics ni simulate.

  chéri    : 20 présent au calme (60–120), 80 sous bruit global (120–180, σ 0.6), silence
             (180–260), retour des deux (260–300). Attendu : m(20) chéri, m(80) non ; en
             silence le 20 est rappelé ; au retour le 20 est tenu, le 80 s'use.
  temoin   : le même run, attachement désactivé (m absent, donc ni rappel ni porte).
  present  : 20 sous ambiance +0.5 (60–120), 80 sous −0.3 (120–145, moins chéri) ; silence
             sous −0.3 (145–220 : ressemble au 80), +0.5 (220–300 : au 20), −1.0 (300–380 :
             à rien → le plus chéri). Attendu : le présent choisit, puis le plus chéri.
  douleur  : comme chéri, mais en silence (180–260) un bruit porté par la zone 20 (σ 1.5) :
             se souvenir du 20 fait mal. porte / sans_porte (rappel.porte.enabled).
usage: python run_insitu_memoire.py <cheri|temoin|present|douleur_porte|douleur_sans_porte|bref|forme|meme_lieu|contigus|all> [seed] [N]
"""
import sys, os, io, json, contextlib, shutil, tempfile, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import simulate, perturbations

def foyer(c, t0, t1, gain=0.5, width=6.0): return {'center': c, 'width': width, 'gain': gain, 't0': t0, 't1': t1}
def ambiance(t0, t1, off): return {'center': 50, 'width': 1e6, 'gain': off, 't0': t0, 't1': t1}
NOISE_CHERI = [(120.0, 180.0, 0.6, None)]
MODES = {
    'cheri':   dict(T=300, foyers=[foyer(20, 60, 120), foyer(80, 120, 180), foyer(20, 260, 300), foyer(80, 260, 300)], noise=NOISE_CHERI),
    'temoin':  dict(T=300, foyers=[foyer(20, 60, 120), foyer(80, 120, 180), foyer(20, 260, 300), foyer(80, 260, 300)], noise=NOISE_CHERI, attach=False),
    'present': dict(T=380, foyers=[foyer(20, 60, 120), foyer(80, 120, 145), ambiance(0, 120, 0.5), ambiance(120, 220, -0.3), ambiance(220, 300, 0.5), ambiance(300, 380, -1.0)], noise=[]),
    'douleur_porte':     dict(T=300, foyers=[foyer(20, 60, 120), foyer(80, 120, 180), foyer(80, 260, 300)], noise=NOISE_CHERI + [(180.0, 260.0, 1.5, 20)]),
    'douleur_sans_porte': dict(T=300, foyers=[foyer(20, 60, 120), foyer(80, 120, 180), foyer(80, 260, 300)], noise=NOISE_CHERI + [(180.0, 260.0, 1.5, 20)], porte=False),
    # ---- les enquêtes de Gepetto (25/09, après-midi) ----
    # bref : pendant le rappel du 20 (silence 180–260), un foyer BREF sur 80 (240–242, 2 u.t.) : le geste le voit-il à l'instant ?
    'bref':     dict(T=300, foyers=[foyer(20, 60, 120), foyer(80, 120, 180), foyer(80, 240, 242), foyer(20, 270, 300), foyer(80, 270, 300)], noise=NOISE_CHERI),
    # forme : le 20 présent en PLATEAU (|n−20| ≤ 7, gain 0.5) au lieu d'une bosse : même profil que le souvenir rappelé
    'forme':    dict(T=300, foyers=[foyer(80, 120, 180), foyer(20, 260, 300), foyer(80, 260, 300)], noise=NOISE_CHERI, plateaus=[(60.0, 120.0, 20, 7, 0.5)]),
    # meme_lieu : deux histoires au même endroit : 20 au calme (60–120), puis 20 sous bruit global (150–210) ; silence 210–290
    'meme_lieu': dict(T=300, foyers=[foyer(20, 60, 120), foyer(20, 150, 210)], noise=[(150.0, 210.0, 0.6, None)]),
    # contigus : deux lieux chéris voisins, 20 (60–120) puis 32 (120–180), au calme ; silence 180–260 : rappelés ensemble ?
    'contigus': dict(T=300, foyers=[foyer(20, 60, 120), foyer(32, 120, 180)], noise=[]),
}
ETATS = {0: 'désactivé', 1: 'extérieur', 2: 'pas de silence', 3: 'rien de chéri', 4: 'réfractaire', 5: 'porte', 6: 'rappel'}
WIN = {
    'bref':      [(230, 240, 'rappel du 20'), (240, 242, 'foyer BREF sur 80'), (242, 250, 'après'), (250, 260, ''), (270, 300, 'les deux reviennent')],
    'forme':     [(100, 120, '20 présent en PLATEAU'), (200, 230, 'silence'), (230, 260, 'silence'), (260, 300, 'les deux reviennent')],
    'meme_lieu': [(100, 120, '20 présent, calme'), (150, 170, '20 présent, bruit'), (190, 210, ''), (210, 240, 'silence'), (240, 290, 'silence')],
    'contigus':  [(100, 120, '20 présent'), (160, 180, '32 présent'), (180, 220, 'silence'), (220, 260, 'silence')],
    'cheri':   [(100, 120, '20 présent, calme'), (160, 180, '80 présent, bruit'), (180, 200, 'silence'), (200, 230, 'silence'), (230, 260, 'silence'), (260, 280, 'les deux reviennent'), (280, 300, '')],
    'present': [(60, 120, '20 présent, +0.5'), (120, 145, '80 présent, −0.3'), (145, 220, 'silence −0.3 : comme le 80'), (220, 300, 'silence +0.5 : comme le 20'), (300, 380, 'silence −1.0 : comme rien')],
    'douleur': [(100, 120, '20 présent'), (180, 200, 'silence, la zone 20 fait mal'), (200, 220, ''), (220, 240, ''), (240, 260, ''), (260, 300, '80 revient')],
}


def run(name, seed, N, T, foyers, noise, attach=True, porte=True, plateaus=()):
    cfg = json.load(open(os.path.join(ROOT, 'config.json')))
    cfg['system']['N'] = N; cfg['system']['T'] = T; cfg['system']['seed'] = seed
    cfg['system']['input']['perturbations'] = [{'type': 'none', 'amplitude': 0.0, 't0': 0.0, 'weight': 1.0}]
    cfg['analysis'] = {**cfg.get('analysis', {}), 'compare_kuramoto': False}
    cfg['context'] = {'foyers': foyers}
    cfg['attention']['attachement']['enabled'] = attach; cfg['attention']['rappel']['porte']['enabled'] = porte
    rng = np.random.default_rng(seed + 99); _in0 = perturbations.compute_In
    def patched(t, c, state=None, history=None, dt=0.05, _o=_in0):
        base = _o(t, c, state, history, dt); v = np.full(N, float(base))
        for (t0, t1, sd, zc) in noise:
            if t0 <= t < t1:
                nz = rng.normal(0.0, sd, N)
                if zc is not None: nz = nz * np.exp(-0.5 * ((np.arange(N) - zc) / 6.0) ** 2)
                v = v + nz
        for (t0, t1, zc, hw, gain) in plateaus:
            if t0 <= t < t1: v = v + gain * (np.abs(np.arange(N) - zc) <= hw)
        return v
    tmp = tempfile.mkdtemp(); cwd = os.getcwd()
    try:
        os.chdir(tmp); json.dump(cfg, open('cfg.json', 'w')); perturbations.compute_In = patched
        with contextlib.redirect_stdout(io.StringIO()): res = simulate.run_simulation('cfg.json', 'FPS')
    finally:
        perturbations.compute_In = _in0; os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)
    h = res['history']; dt = cfg['system']['dt']
    t = np.array([x['t'] for x in h]); O = np.array([x['O'] for x in h]); fn = np.array([x['fn'] for x in h]); phi = np.array([x['phi_n_t'] for x in h])
    theta = 2 * np.pi * np.cumsum(fn, axis=0) * dt + phi
    z = np.exp(1j * theta); R = np.abs(0.5 * np.roll(z, 1, 1) + 0.5 * np.roll(z, -1, 1)); R[:, 0] = np.abs(z[:, 1]); R[:, -1] = np.abs(z[:, -2])
    out = dict(t=t, O=O, R=R, s=np.array([x['attention_s'] for x in h]), m=np.array([x['attention_m'] for x in h]),
               m_max=np.array([x['attention_m_max'] for x in h], dtype=float), silence=np.array([x['attention_silence'] for x in h], dtype=float),
               rappel=np.array([x['attention_rappel'] for x in h], dtype=float), porte=np.array([x['attention_porte'] for x in h], dtype=float),
               etat=np.array([x.get('attention_rappel_etat', np.nan) for x in h], dtype=float),
               audib=np.array([x['attention_audibility'] for x in h], dtype=float), act=np.array([x['activite_rel'] if x.get('activite_rel') is not None else np.nan for x in h], dtype=float),
               surprise=np.array([x['surprise_mean'] if x.get('surprise_mean') is not None else np.nan for x in h], dtype=float))
    np.savez_compressed(os.path.join(ROOT, 'calib', 'attention_sources', f'insitu_{name}_{seed}.npz'), **out)
    json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in res['metrics'].items() if isinstance(v, (int, float, np.floating, str)) and not isinstance(v, bool)},
              open(os.path.join(ROOT, 'calib', 'attention_sources', f'insitu_{name}_{seed}.json'), 'w'), indent=1)
    return out


def zone(N, c, width=6.0): return np.exp(-0.5 * ((np.arange(N) - c) / width) ** 2) > 0.5
def audib(O, zin):
    S = O.sum(1); return float(np.mean(O[:, zin].sum(1) * S) / max(np.mean(S ** 2), 1e-12))


def read(d, label, wins, zones=(20, 80)):
    t = d['t']; N = d['O'].shape[1]; za, zb = zone(N, zones[0]), zone(N, zones[1])
    print(f"[{label}] fenêtre | m 20 / m 80 | silence | rappel : 20 / 80 (% temps) | porte | zone 20 : s · R · audib | zone 80 : s · R · audib | activité/repos")
    for (a, b, tag) in wins:
        w = (t >= a) & (t < b)
        r20 = 100 * np.mean(np.isfinite(d['rappel'][w]) & (np.abs(d['rappel'][w] - zones[0]) <= 8)); r80 = 100 * np.mean(np.isfinite(d['rappel'][w]) & (np.abs(d['rappel'][w] - zones[1]) <= 8))
        g = d['porte'][w]; gv = np.nanmean(g) if np.isfinite(g).any() else float('nan')
        def zz(z_): return f"{d['s'][w][:, z_].mean():.2f} · {d['R'][w][:, z_].mean():.3f} · {100 * audib(d['O'][w], z_):.0f} %"
        et = d['etat'][w]; et_s = ' '.join(f"{ETATS[int(k)]} {100*np.mean(et == k):.0f}%" for k in sorted(set(et[np.isfinite(et)].astype(int))))
        print(f"[{label}] {a:3d}-{b:<3d} | {d['m'][w][:, za].mean():.2f} / {d['m'][w][:, zb].mean():.2f} | {100 * np.nanmean(d['silence'][w]):3.0f} % | {r20:3.0f} / {r80:3.0f} | {gv:.2f} | {zz(za)} | {zz(zb)} | ×{np.nanmedian(d['act'][w]):.2f}  {tag}  [{et_s}]")


if __name__ == '__main__':
    which = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345; N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    for nm in (list(MODES) if which == 'all' else [which]):
        kw = dict(MODES[nm]); d = run(nm, seed, N, **kw)
        key = nm if nm in WIN else ('douleur' if nm.startswith('douleur') else 'cheri')
        read(d, f"{nm} s{seed}", WIN[key], zones=((20, 32) if nm == 'contigus' else (20, 80)))
        if nm == 'bref':                                                   # le délai de reconnaissance de l'extérieur, au pas près
            t = d['t']; N = d['O'].shape[1]; zb = zone(N, 80); s80 = d['s'][:, zb].mean(1); e = d['etat']
            i0 = int(np.searchsorted(t, 240.0)); seen = np.flatnonzero(s80[i0:] > 0.3); ext = np.flatnonzero(e[i0:] == 1)
            print(f"[{nm} s{seed}] foyer bref à t = 240 : saillance zone 80 > 0.3 après {t[i0 + seen[0]] - 240:.1f} u.t. (s80 {s80[i0 + seen[0]]:.2f}) ; « l'extérieur parle » reconnu après {t[i0 + ext[0]] - 240 if len(ext) else float('nan'):.1f} u.t. ; rappel du 20 interrompu {100 * np.mean(np.isfinite(d['rappel'][(t >= 240) & (t < 242)])):.0f} % du temps pendant le foyer bref")
    print("DONE")
