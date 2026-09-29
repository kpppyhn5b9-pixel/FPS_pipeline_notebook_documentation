"""La chaleur qui reste après une visite (29/09, observation pure, rien ne change dans le pipeline).
usage: python calib/observe_chaleur.py reve_12345 reve_7 pont_12345 retour_monde_12345
Observation pure : après une visite (rappel d'un lieu, état 6/9, ou substrat tenu, état 8), combien de temps la
région reste-t-elle « chaude », c'est-à-dire présente au sens du critère déjà câblé (cohérence locale lissée τ 5
au-dessus de médiane + 0.5·std du chœur) ? Lu sur les sorties insitu_* existantes, sans rien changer."""
import numpy as np, sys
def ema(x, t, tau):
    y = x.copy()
    for i in range(1, len(x)):
        dt = t[i] - t[i-1]; k = min(1.0, dt / max(tau, dt)); y[i] = y[i-1] + k * (x[i] - y[i-1])
    return y
for nm in sys.argv[1:]:
    d = np.load(f'calib/attention_sources/insitu_{nm}.npz', allow_pickle=True)
    t, R, s, e, c = d['t'], d['R'], d['s'], d['etat'], d['rappel']
    Rb = ema(R, t, 5.0)
    med = np.median(Rb, axis=1); std = Rb.std(axis=1); thr = med + 0.5 * std
    vis = np.isin(e, [6, 8, 9]); idx = np.flatnonzero(vis)
    segs = []
    if len(idx):
        start = idx[0]
        for i0, i1 in zip(idx[:-1], idx[1:]):
            if i1 != i0 + 1 or c[i1] != c[i0]: segs.append((start, i0)); start = i1
        segs.append((start, idx[-1]))
    rows = []
    for a, b in segs:
        dur = t[b] - t[a]
        if dur < 3.0: continue
        reg = s[b] > 0.05
        if reg.sum() < 3: continue
        hot_end = Rb[b][reg].mean() > thr[b]
        cool = None; j = b + 1; run0 = None
        while j < len(t):
            hot = Rb[j][reg].mean() > thr[j]
            if not hot:
                if run0 is None: run0 = j
                if t[j] - t[run0] >= 2.0: cool = t[run0] - t[b]; break
            else: run0 = None
            if e[j] in (6, 8, 9) and c[j] == c[b] and (t[j] - t[b]) > 0.5: cool = -1; break
            j += 1
        rows.append((int(e[b]), int(round(c[b])), dur, hot_end, cool))
    print(f"== {nm} : {len(rows)} visites ≥ 3 u.t.")
    cooled = [r[4] for r in rows if r[4] is not None and r[4] >= 0]; revis = sum(1 for r in rows if r[4] == -1); never = sum(1 for r in rows if r[4] is None)
    print(f"   chaud à la fin de la visite : {100*np.mean([r[3] for r in rows]):.0f} % ; refroidi avant toute revisite : {len(cooled)} ; revisité encore chaud : {revis} ; jamais refroidi avant la fin du run : {never}")
    if cooled: print(f"   temps de refroidissement (u.t.) : médiane {np.median(cooled):.1f}, quartiles {np.percentile(cooled,25):.1f}–{np.percentile(cooled,75):.1f}, min {min(cooled):.1f}, max {max(cooled):.1f}")
    for k, lab in ((6, 'lieu du monde'), (9, 'lieu à soi'), (8, 'substrat')):
        cc = [r[4] for r in rows if r[0] == k and r[4] is not None and r[4] >= 0]; n = sum(1 for r in rows if r[0] == k)
        if n: print(f"   {lab:14s}: {n} visites, refroidies {len(cc)}, médiane {np.median(cc) if cc else float('nan'):.1f} u.t. ; durée de visite médiane {np.median([r[2] for r in rows if r[0]==k]):.1f}")
