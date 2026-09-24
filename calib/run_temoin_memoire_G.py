"""Témoin pour le résultat de Gepetto (24/09/2026) : « masquer la mémoire d'efficacité de G
change la saillance strate par strate (écart ≈ 0.08) sans changer sa moyenne ».

La FPS est sensible à ses conditions : deux trajectoires qui divergent à partir d'un
détail quelconque donnent un écart de saillance par simple bifurcation. Il faut donc
comparer l'écart produit par le masquage à celui produit par une perturbation aussi
petite qui n'est PAS la mémoire (un f₀ multiplié par 1 + 10⁻⁶ sur une strate).

Trois runs par seed, pipeline natif (main : mémoire + considération câblées), même
passé jusqu'à t = 60, foyer constant sur la strate 50 de t = 60 à 160 :
  ref      rien de plus
  maskG    à partir de t = 60, la mémoire d'efficacité par contexte est vidée avant
           chaque décision de régulation (decide_G_adaptive_aware) ; la mémoire de soi reste
  papillon à t = 60, f₀ de la strate 0 × (1 + 10⁻⁶), rien d'autre
Lectures : écart moyen |Δs| strate par strate et pas par pas dans la zone du foyer
(t 60–160), saillance moyenne dans la zone, audibilité, et divergence des fréquences
|Δf|/f (pour voir que les trajectoires bifurquent de toute façon).
usage: python run_temoin_memoire_G.py <ref|maskG|papillon|all> [seed] [N] [T]
"""
import sys, os, json, io, contextlib, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import simulate, dynamics

T_ON, T_OFF, CENTER, WIDTH = 60.0, 160.0, 50, 6.0


def run(mode, seed=12345, N=100, T=170):
    c = json.load(open(os.path.join(ROOT, 'config.json')))
    c['system']['T'] = T; c['system']['seed'] = seed; c['system']['N'] = N
    c['system']['input']['perturbations'] = [{'type': 'none', 'amplitude': 0.0, 't0': 0.0, 'weight': 1.0}]
    c['analysis'] = {**c.get('analysis', {}), 'compare_kuramoto': False}
    c['context'] = {'foyers': [{'center': CENTER, 'width': WIDTH, 'gain': 0.5, 't0': T_ON, 't1': T_OFF}]}
    d = f"{mode}_{seed}"; os.makedirs(d, exist_ok=True); cwd = os.getcwd(); os.chdir(d)
    json.dump(c, open('cfg.json', 'w'))
    _dec = dynamics.decide_G_adaptive_aware; _fn = dynamics.compute_fn; done = {'p': False}

    def dec_mask(t, gamma_current, regulation_state, history, config, error_summary, _o=_dec):
        if t >= T_ON:
            rm = regulation_state.get('regulation_state', {}).get('regulation_memory')
            if rm is not None:
                rm['effectiveness_by_context'] = {}
        return _o(t, gamma_current, regulation_state, history, config, error_summary)

    def fn_papillon(t, state, An_t, F, cfg, _o=_fn):
        if t >= T_ON and not done['p']:
            state[0]['f0'] = state[0]['f0'] * (1.0 + 1e-6); done['p'] = True
        return _o(t, state, An_t, F, cfg)

    if mode == 'maskG': dynamics.decide_G_adaptive_aware = dec_mask
    if mode == 'papillon': dynamics.compute_fn = fn_papillon
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf): res = simulate.run_simulation('cfg.json', 'FPS')
    except BaseException as e:
        sys.stderr.write(f"ECHEC {type(e).__name__}: {e}\n" + buf.getvalue()[-2000:] + "\n"); raise
    finally:
        dynamics.decide_G_adaptive_aware = _dec; dynamics.compute_fn = _fn; os.chdir(cwd)
    h = res['history']
    t = np.array([x['t'] for x in h]); S = np.array([np.asarray(x['attention_s'], dtype=float) for x in h])
    FN = np.array([np.asarray(x['fn'], dtype=float) for x in h]); O = np.array([np.asarray(x['O'], dtype=float) for x in h])
    np.savez_compressed(os.path.join(d, 'trace.npz'), t=t, s=S, fn=FN, O=O)
    print("DONE", d)


def compare(seed):
    R = {m: np.load(f"{m}_{seed}/trace.npz") for m in ('ref', 'maskG', 'papillon')}
    t = R['ref']['t']; N = R['ref']['s'].shape[1]
    zin = np.exp(-0.5 * ((np.arange(N) - CENTER) / WIDTH) ** 2) > 0.5; m = (t >= T_ON) & (t < T_OFF)
    def audib(O):
        S = O.sum(1); return float(np.mean(O[:, zin].sum(1) * S) / max(np.mean(S ** 2), 1e-12))
    print(f"seed {seed} | run | s moyen dans le foyer | audibilité | écart |Δs| au ref (foyer, strate × pas) | écart |Δs| hors foyer | divergence |Δf|/f (toutes strates)")
    for mode in ('ref', 'maskG', 'papillon'):
        s = R[mode]['s'][m]; s0 = R['ref']['s'][m]; f = R[mode]['fn'][m]; f0 = R['ref']['fn'][m]
        print(f"  {mode:8s} | {s[:, zin].mean():.4f} | {100*audib(R[mode]['O'][m]):.0f} % | {np.abs(s - s0)[:, zin].mean():.4f} | {np.abs(s - s0)[:, ~zin].mean():.4f} | {(np.abs(f - f0) / np.maximum(f0, 1e-9)).mean():.4f}")
    # et l'écart entre les deux témoins eux-mêmes
    s1, s2 = R['maskG']['s'][m], R['papillon']['s'][m]
    print(f"  maskG vs papillon : écart |Δs| foyer {np.abs(s1 - s2)[:, zin].mean():.4f}")


if __name__ == '__main__':
    which = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 100; T = int(sys.argv[4]) if len(sys.argv) > 4 else 170
    if which == 'compare':
        compare(seed)
    else:
        for mode in (('ref', 'maskG', 'papillon') if which == 'all' else [which]):
            run(mode, seed, N, T)
