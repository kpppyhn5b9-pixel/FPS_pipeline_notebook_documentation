"""usage: python calib/observe_signature.py reve_12345 reve_7 pont_12345 (après un run du banc, qui sauve sig, g, present depuis le 30/09)
Un lieu à soi est-il signé par le silence ? (30/09, observation pure sur les sorties insitu_* avec sig/g/present.)
Pour chaque pas en silence : la ressemblance (gaussienne relative, largeurs 0.2/0.1, comme le rappel) de chaque
lieu chéri (m ≥ 0.3, composante connexe) à l'état présent g ; lieu « à soi » si soi moyen > 0.5, « du monde » sinon.
Puis : qui ressemble le plus en silence, qui est appelé en premier à chaque nouveau silence, et la présence des lieux visités."""
import numpy as np, sys
sys.path.insert(0, '.'); from dynamics import attention_islands
W = np.array([0.2, 0.1])
def res_of(g, sigc):
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(sigc))): return np.nan
    z = (g - sigc) / (W * np.abs(sigc) + 1e-9); return float(np.exp(-0.5 * np.sum(z ** 2)))
for nm in sys.argv[1:]:
    d = np.load(f'calib/attention_sources/insitu_{nm}.npz', allow_pickle=True)
    t, m, soi, sig, g, e, sil, c, pres = d['t'], d['m'], d['soi'], d['sig'], d['g'], d['etat'], d['silence'], d['rappel'], d['present']
    rows = []   # (t, type, res, m_mean, is_recalled)
    first = []  # type du premier lieu visité à chaque nouveau silence
    prev_vis = False
    for i in range(len(t)):
        ok = m[i] >= 0.3
        comps = attention_islands(ok.astype(float), 0.5) if ok.any() else []
        vis = e[i] in (6, 9)
        if vis and not prev_vis:
            first.append(('soi' if e[i] == 9 else 'monde', t[i]))
        prev_vis = vis if e[i] not in (2, 1) else False
        if sil[i] != 1 or t[i] < 120: continue
        for comp in comps:
            typ = 'soi' if soi[i][comp].mean() > 0.5 else 'monde'
            sc = sig[i][comp].mean(0); r = res_of(g[i], sc)
            rec = np.isfinite(c[i]) and (comp[0] <= c[i] <= comp[-1])
            rows.append((t[i], typ, r, m[i][comp].mean(), rec, comp[0], comp[-1]))
    rows = np.array(rows, dtype=object)
    print(f"== {nm}")
    for typ in ('monde', 'soi'):
        sel = [r for r in rows if r[1] == typ]
        if not sel: print(f"   {typ:6s}: aucun lieu chéri en silence"); continue
        rr = np.array([r[2] for r in sel], dtype=float); mm = np.array([r[3] for r in sel], dtype=float)
        print(f"   {typ:6s}: {len(sel)} (pas × lieu) en silence ; ressemblance médiane {np.nanmedian(rr):.2f} (quartiles {np.nanpercentile(rr,25):.2f}–{np.nanpercentile(rr,75):.2f}) ; ≥ 0.2 : {100*np.nanmean(rr>=0.2):.0f} % ; m moyen {mm.mean():.2f} ; rappelé {100*np.mean([r[4] for r in sel]):.0f} % de ces pas")
    # aux pas où les deux types existent : qui ressemble le plus ?
    by_t = {}
    for r in rows: by_t.setdefault(r[0], []).append(r)
    both = [v for v in by_t.values() if {x[1] for x in v} == {'soi', 'monde'}]
    if both:
        soi_best = np.mean([max(v, key=lambda x: (x[2] if np.isfinite(x[2]) else -1))[1] == 'soi' for v in both])
        soi_score = np.mean([max(v, key=lambda x: ((x[2] if np.isfinite(x[2]) else 0) * x[3]))[1] == 'soi' for v in both])
        print(f"   pas de silence où monde et soi coexistent : {len(both)} ; le soi ressemble le plus {100*soi_best:.0f} % ; le soi a le meilleur score (res × m) {100*soi_score:.0f} %")
    # ressemblance dans le temps pour les lieux à soi (dérive vers le silence ?)
    for typ in ('soi', 'monde'):
        sel = [r for r in rows if r[1] == typ and np.isfinite(r[2])]
        if len(sel) > 20:
            tt = np.array([r[0] for r in sel]); rr = np.array([r[2] for r in sel], dtype=float)
            qs = np.percentile(tt, [0, 33, 66, 100]); parts = [rr[(tt >= qs[k]) & (tt < qs[k+1] + 1e-9)] for k in range(3)]
            print(f"   {typ:6s}: ressemblance médiane par tiers du temps : " + ' → '.join(f"{np.median(p):.2f}" for p in parts if len(p)))
    if first: print("   premier lieu visité à chaque nouveau silence : " + ', '.join(f"{a}@{b:.0f}" for a, b in first[:14]) + (' …' if len(first) > 14 else ''))
    # présence des lieux visités
    for k, lab in ((9, 'lieu à soi'), (6, 'lieu du monde'), (8, 'substrat')):
        w = (e == k) & np.isfinite(pres)
        if w.any(): print(f"   {lab:14s}: présent (au sens du critère) {100*np.nanmean(pres[w]):.0f} % des pas de visite ({int(w.sum())} pas)")
