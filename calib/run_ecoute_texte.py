#!/usr/bin/env python3
"""
ÉCOUTER UN TEXTE (20/11) : le pont vers les transformers, première moitié seulement.
Le transformer parle, la FPS écoute, rien ne revient vers le transformer.

Le pont est petit : à chaque token, un vecteur (l'état interne d'un modèle, ou un témoin sans apprentissage) est
projeté sur N directions fixes, une par strate, et devient Iₙ(t) pendant `k` pas ; après le texte, le silence.
Le reste, la FPS le fait déjà : elle lit l'entrée relative à sa médiane, garde ce qui reste (τ_c), chérit ce qui
était là quand elle allait bien, se tait, se souvient. On lit ce qu'on lit déjà : la cloche de dispersion, la garde,
la part saillante, m, le silence, le rappel et son origine. Et une chose de plus : à quels PASSAGES du texte
correspondent les lieux qu'elle a chéris et ceux qu'elle rappelle en silence.

Sources :
  temoin : un plongement de mots sans apprentissage (trigrammes de caractères hachés, 256 dimensions). Il ne porte
           aucune structure apprise : ce que la FPS en fait est le point zéro, à comparer au modèle.
  vierge : un transformer de la forme de GPT-2 small, poids aléatoires : l'architecture sans l'apprentissage. Son attention
           mélange déjà le passé ; si quelque chose reste ici, ce n'est pas le sens, c'est la forme.
  modele : l'état caché d'un petit modèle ouvert (transformers, `--model`, `--layer`), token par token.
  --shuffle : le même texte, mots mélangés AVANT toute lecture (le modèle lit le texte mélangé) : la persistance (ce qui
           reste) dépend-elle de l'ordre ?

Usage :
  python calib/run_ecoute_texte.py docs/ATTENTION_souvenir_present.md --source temoin [--k 4] [--N 100] [--seed 12345]
  python calib/run_ecoute_texte.py <texte> --source modele --model Qwen/Qwen2.5-0.5B --layer 12
Sorties : calib/attention_sources/ecoute_<source>_<nom>_<seed>.npz (+ .json), et la lecture imprimée.
"""
import os, sys, io, json, re, shutil, tempfile, contextlib, argparse, hashlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import simulate, perturbations, metrics

ETATS = {0: 'désactivé', 1: 'extérieur', 2: 'pas de silence', 3: 'rien de chéri', 4: 'réfractaire', 5: 'porte', 6: 'lieu du monde', 8: 'substrat', 9: 'lieu à soi'}
APPEL = {1.0: 'appelé par le présent', 0.0: 'repli « le plus chéri »', 2.0: 'visite du tour', 3.0: 'substrat'}


# ---------------------------------------------------------------- les deux sources : des vecteurs, un par token
def tokens_mots(text):
    return re.findall(r"[^\s]+", text)

def plongement_temoin(mots, d=256):
    """Trigrammes de caractères hachés, signés, normalisés : aucun apprentissage, une forme pour chaque mot."""
    E = np.zeros((len(mots), d))
    for i, w in enumerate(mots):
        s = f"^{w.lower()}$"
        for j in range(len(s) - 2):
            h = hashlib.blake2b(s[j:j + 3].encode('utf-8'), digest_size=8).digest()
            idx = int.from_bytes(h[:4], 'little') % d; sgn = 1.0 if h[4] % 2 == 0 else -1.0
            E[i, idx] += sgn
        n = np.linalg.norm(E[i]); E[i] = E[i] / n if n > 0 else E[i]
    return E

def plongement_modele(text, model_name, layer=None, max_tokens=None):
    """L'état caché d'un modèle ouvert, token par token (couche `layer`, par défaut la moitié). Renvoie (tokens, E)."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name); mdl = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32); mdl.eval()
    ids = tok(text, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
    if max_tokens: ids = ids[:max_tokens]
    win = 512; outs = []
    with torch.no_grad():
        for a in range(0, len(ids), win):                                   # par fenêtres, sans chevauchement (CPU, 4 cœurs)
            o = mdl(input_ids=ids[a:a + win][None], output_hidden_states=True)
            hs = o.hidden_states; L = len(hs) // 2 if layer is None else int(layer)
            outs.append(hs[L][0].numpy())
    E = np.concatenate(outs, 0); mots = [tok.decode([int(i)]) for i in ids]
    return mots, E

def plongement_modele_vierge(mots, seed=0, n_layer=12, n_embd=768, layer=None):
    """Un transformer de la forme de GPT-2 small mais VIERGE (poids aléatoires, graine `seed`) : il n'a rien appris, mais son
    attention mélange déjà le passé de chaque token. Si une persistance apparaît ici, elle vient de l'architecture, pas de
    l'apprentissage. Les mots sont envoyés sur des identifiants par hachage (pas de vocabulaire appris). Aucun téléchargement."""
    import torch
    from transformers import GPT2Config, GPT2Model
    torch.manual_seed(seed)
    cfg = GPT2Config(n_layer=n_layer, n_embd=n_embd, n_head=12, vocab_size=50257, n_positions=1024)
    mdl = GPT2Model(cfg); mdl.eval()
    ids = [int.from_bytes(hashlib.blake2b(w.lower().encode('utf-8'), digest_size=4).digest(), 'little') % 50257 for w in mots]
    ids = torch.tensor(ids); win = 512; outs = []
    with torch.no_grad():
        for a in range(0, len(ids), win):
            o = mdl(input_ids=ids[a:a + win][None], output_hidden_states=True)
            hs = o.hidden_states; L = len(hs) // 2 if layer is None else int(layer)
            outs.append(hs[L][0].numpy())
    return np.concatenate(outs, 0)

def projeter(E, N, dirs_seed, echelle=0.25, dims_relatives=True):
    """N directions fixes (graine `dirs_seed`), projection, puis chaque strate centrée-réduite sur le texte et × échelle :
    une excursion de 2 σ vaut 0.5, le gain des foyers de tous les bancs. Une seule constante, nommée.
    `dims_relatives` (20/11, observé sur Qwen2.5-0.5B) : chaque dimension du modèle est d'abord lue relative à son habitude
    sur le texte (centrée-réduite). Sans ça, une seule dimension (la 62, σ 76 contre 0.3 pour les autres : une « activation
    massive », connue des transformers) et un token-puits (« est », norme 1736 contre 16) font 83 % de ce que les strates
    reçoivent : toutes voient le même nombre, au signe près, et rien ne se détache du chœur. C'est le même principe que la
    FPS applique à son entrée, relative à sa médiane, appliqué une fois en amont ; `--brut` pour voir sans."""
    rng = np.random.default_rng(dirs_seed); d = E.shape[1]
    if dims_relatives: E = (E - E.mean(0)) / (E.std(0) + 1e-9)
    D = rng.normal(0.0, 1.0 / np.sqrt(d), (d, N))
    P = E @ D
    mu = P.mean(0); sd = P.std(0) + 1e-9
    return (P - mu) / sd * echelle


# ---------------------------------------------------------------- écouter : le texte devient Iₙ(t), puis le silence
def ecouter(nom, mots, I_tok, k, seed, silence=200.0, N=None, reve=None):
    cfg = json.load(open(os.path.join(ROOT, 'config.json'))); dt = float(cfg['system']['dt'])
    N = I_tok.shape[1]; T_text = len(mots) * k * dt; T = int(np.ceil(20.0 + T_text + silence))
    cfg['system']['N'] = N; cfg['system']['T'] = T; cfg['system']['seed'] = seed
    cfg['system']['input']['perturbations'] = [{'type': 'none', 'amplitude': 0.0, 't0': 0.0, 'weight': 1.0}]
    cfg['analysis'] = {**cfg.get('analysis', {}), 'compare_kuramoto': False}
    cfg['context'] = {'foyers': []}
    if reve is not None: cfg['attention']['reve']['enabled'] = bool(reve)
    t_debut = 20.0                                                           # le texte commence après le transitoire, comme les foyers des bancs
    _in0 = perturbations.compute_In
    def patched(t, c, state=None, history=None, dt_=dt, _o=_in0):
        base = _o(t, c, state, history, dt_); v = np.full(N, float(base))
        i = int((t - t_debut) / dt_) // k
        if 0 <= i < len(mots): v = v + I_tok[i]
        return v
    tmp = tempfile.mkdtemp(); cwd = os.getcwd()
    try:
        os.chdir(tmp); json.dump(cfg, open('cfg.json', 'w')); perturbations.compute_In = patched
        with contextlib.redirect_stdout(io.StringIO()): res = simulate.run_simulation('cfg.json', 'FPS')
    finally:
        perturbations.compute_In = _in0; os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)
    h = res['history']
    t = np.array([x['t'] for x in h]); tok_idx = np.floor((t - t_debut) / dt).astype(int) // k
    tok_idx = np.where((tok_idx >= 0) & (tok_idx < len(mots)), tok_idx, -1)
    g = lambda key, fill=np.nan: np.array([x[key] if x.get(key) is not None else fill for x in h], dtype=float)
    out = dict(t=t, tok=tok_idx, O=np.array([x['O'] for x in h]),
               s=np.array([x['attention_s'] for x in h]), m=np.array([x['attention_m'] for x in h]),
               m_long=np.array([x['attention_m_long'] if x.get('attention_m_long') is not None else np.zeros(N) for x in h]),
               soi=np.array([x.get('attention_soi', np.zeros(N)) for x in h]),
               m_max=g('attention_m_max'), silence=g('attention_silence'), rappel=g('attention_rappel'), porte=g('attention_porte'),
               etat=g('attention_rappel_etat'), appel=g('attention_rappel_appel'), garde=g('attention_garde'),
               part=g('attention_salient_share'), audib=g('attention_audibility'), disp=g('dispersion_norm'), act=g('activite_rel'),
               surprise=g('surprise_mean'), I_tok=I_tok, k=k, dt=dt, t_debut=t_debut, T_text=T_text)
    base = os.path.join(ROOT, 'calib', 'attention_sources', f'ecoute_{nom}_{seed}')
    np.savez_compressed(base + '.npz', **out)
    json.dump({'mots': mots, 'k': k, 'dt': dt, 'N': N, 'seed': seed, 'T_text': T_text, 'silence': silence}, open(base + '.json', 'w'), ensure_ascii=False)
    return out


# ---------------------------------------------------------------- lire : ce qu'on lit déjà, plus les passages
def passages(I_tok, strates, mots, n=3, largeur=12):
    """Les `n` passages (fenêtres de `largeur` mots) où l'entrée sur ces strates a été la plus haute."""
    x = I_tok[:, strates].mean(1)
    if len(x) < largeur: return [(0, ' '.join(mots))]
    w = np.convolve(x, np.ones(largeur) / largeur, mode='valid')
    order = np.argsort(-w); pris = []
    for i in order:
        if all(abs(i - j) >= largeur for j, _ in pris):
            pris.append((int(i), ' '.join(mots[i:i + largeur])))
        if len(pris) == n: break
    return pris

def ilots(mask):
    idx = np.flatnonzero(mask); comps = []
    if len(idx) == 0: return comps
    start = prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1: comps.append(np.arange(start, prev + 1)); start = i
        prev = i
    comps.append(np.arange(start, prev + 1)); return comps

def lire(d, mots, label, fenetre=100.0):
    t = d['t']; dt = d['dt']; T_text = d['T_text']; t0 = d['t_debut']
    c = metrics.DISPERSION_NORM_CENTER
    print(f"[{label}] {len(mots)} tokens, {d['k']} pas par token, {d['k'] * dt:.2f} u.t. par token, texte {t0:.0f}–{t0 + T_text:.0f}, silence ensuite jusqu'à {t[-1]:.0f}")
    print(f"[{label}] fenêtre | tokens | cloche (médiane, min) | garde min | part saillante | audib | m max | silence | activité/repos")
    a = 20.0
    while a < t[-1]:
        b = min(a + fenetre, t[-1] + 1e-9); w = (t >= a) & (t < b)
        if w.sum() == 0: a = b; continue
        tk = d['tok'][w]; tk = tk[tk >= 0]
        sc = [metrics.score_from_brackets(x, 'dispersion') for x in d['disp'][w] if np.isfinite(x)]
        tag = f"{tk.min()}–{tk.max()}" if len(tk) else 'silence'
        print(f"[{label}] {a:4.0f}-{b:<4.0f} | {tag:>9s} | {np.median(sc) if sc else float('nan'):.0f}, {min(sc) if sc else float('nan'):.0f} | {np.nanmin(d['garde'][w]):.2f} | {100 * np.nanmean(d['part'][w]):4.1f} % | {np.nanmean(d['audib'][w]):.2f} | {np.nanmax(d['m_max'][w]):.2f} | {100 * np.nanmean(d['silence'][w]):3.0f} % | ×{np.nanmedian(d['act'][w]):.2f}")
        a = b
    # ce qu'elle a chéri à la fin du texte, et à quels passages ça correspond
    i_fin = int(np.argmin(np.abs(t - (t0 + T_text)))); mr = np.maximum(d['m'][i_fin], d['m_long'][i_fin])
    comps = ilots(mr >= 0.3)
    print(f"[{label}] à la fin du texte : {len(comps)} lieu(x) chéri(s) (m lu ≥ 0.3), {100 * np.mean(mr >= 0.3):.0f} % des strates ; m max {mr.max():.2f}")
    for comp in comps:
        print(f"[{label}]   strates {comp[0]}–{comp[-1]} (m lu {mr[comp].mean():.2f}, soi {d['soi'][i_fin][comp].mean():.2f}) : passages qui l'ont le plus appelé")
        for i, p in passages(d['I_tok'], comp, mots):
            print(f"[{label}]     mots {i}–{i + 12} : « {p} »")
    # le silence : ce qu'elle rappelle, d'où ça vient, et le passage derrière
    w = t >= t0 + T_text
    if w.sum():
        e = d['etat'][w]; e = e[np.isfinite(e)].astype(int)
        parts = ' · '.join(f"{ETATS[k_]} {100 * np.mean(e == k_):.0f} %" for k_ in sorted(set(e.tolist())))
        print(f"[{label}] silence après le texte : {parts}")
        r = d['rappel'][w]; f = np.isfinite(r)
        if f.any():
            ap = d['appel'][w][f]; orig = ' · '.join(f"{APPEL.get(k_, k_)} {100 * np.mean(ap == k_):.0f} %" for k_ in sorted(set(ap[np.isfinite(ap)].tolist())))
            print(f"[{label}]   rappel {100 * f.mean():.0f} % du silence ; origine : {orig} ; porte moyenne {np.nanmean(d['porte'][w]):.2f}")
            cs, cnt = np.unique(r[f].astype(int), return_counts=True)
            for c_, n_ in sorted(zip(cs, cnt), key=lambda z: -z[1])[:3]:
                comp = [cc for cc in comps if c_ in cc]; comp = comp[0] if comp else np.array([c_])
                print(f"[{label}]   centre {c_} ({100 * n_ / f.sum():.0f} % des rappels), strates {comp[0]}–{comp[-1]} : « {passages(d['I_tok'], comp, mots, n=1)[0][1]} »")
    print(f"[{label}] cloche sur tout le texte : {np.median([metrics.score_from_brackets(x, 'dispersion') for x in d['disp'][(t >= t0) & (t < t0 + T_text)] if np.isfinite(x)]):.0f} ; garde min {np.nanmin(d['garde']):.2f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('texte'); ap.add_argument('--source', default='temoin', choices=['temoin', 'vierge', 'modele'])
    ap.add_argument('--model', default='Qwen/Qwen2.5-0.5B'); ap.add_argument('--layer', type=int, default=None); ap.add_argument('--max-tokens', type=int, default=None)
    ap.add_argument('--k', type=int, default=4); ap.add_argument('--N', type=int, default=100); ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--dirs-seed', type=int, default=0); ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--silence', type=float, default=200.0)
    ap.add_argument('--reve', type=int, default=None); ap.add_argument('--echelle', type=float, default=0.25)
    ap.add_argument('--brut', action='store_true', help='sans lire chaque dimension du modèle relative à son habitude')
    a = ap.parse_args()
    text = open(a.texte, encoding='utf-8').read()
    # (lecture de Codex, 20/11) le mélange se fait sur les MOTS, AVANT toute lecture : pour les trois sources, c'est le texte
    # mélangé qui est lu. Mélanger les vecteurs après coup laisserait au modèle le contexte de l'ordre d'origine.
    mots = tokens_mots(text)
    if a.shuffle: np.random.default_rng(a.seed).shuffle(mots)
    if a.source in ('temoin', 'vierge'):
        if a.max_tokens: mots = mots[:a.max_tokens]
        E = plongement_temoin(mots) if a.source == 'temoin' else plongement_modele_vierge(mots, seed=a.dirs_seed, layer=a.layer)
    else:
        mots, E = plongement_modele(' '.join(mots), a.model, a.layer, a.max_tokens)
    I_tok = projeter(E, a.N, a.dirs_seed, a.echelle, dims_relatives=not a.brut)
    nom = f"{a.source}{'_melange' if a.shuffle else ''}{'_brut' if a.brut else ''}{f'_L{a.layer}' if a.layer is not None else ''}_{os.path.splitext(os.path.basename(a.texte))[0]}_k{a.k}_d{a.dirs_seed}"
    d = ecouter(nom, mots, I_tok, a.k, a.seed, a.silence, reve=a.reve)
    lire(d, mots, nom)
