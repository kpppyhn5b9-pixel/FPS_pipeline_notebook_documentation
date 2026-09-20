"""Table de calibration : biais et étalement de l'autocorr CSD mesurée in-situ
(forçage période 200 + 2 harmoniques, fluctuation AR(1) multiplicative rel 0.05)
selon la fenêtre W et le lag, pour τ = k·lag, k ∈ facteurs de ralentissement."""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # racine du dépôt
sys.path.insert(0, ROOT)
import metrics
rng = np.random.RandomState(11)
def ar1(tau, n, burn=800):
    phi = np.exp(-1/tau); e = rng.randn(n+burn); a = np.zeros(n+burn)
    for i in range(1, n+burn): a[i] = phi*a[i-1] + e[i]
    return a[burn:]*np.sqrt(1-phi**2)
KS = (1.0, 1.4, 2.0, 3.0, 5.0, 8.0)
out = {}
for W in (1000, 2000):
    for lag in (10, 20, 40):
        for k in KS:
            tau = k*lag; vals = []; scores = []
            for _ in range(30):
                t = np.arange(W); env = 1+0.3*np.sin(2*np.pi*t/200)+0.05*np.sin(4*np.pi*t/200)+0.01*np.sin(6*np.pi*t/200)
                y = env*(1+0.05*ar1(tau, W))
                r = metrics.compute_resilience_csd(y, 0.1, lag=lag)
                if r: vals.append(r['ac']); scores.append(metrics.score_from_brackets(r['ac'], 'resilience'))
            vals = np.array(vals); nominal = metrics.score_from_brackets(float(np.exp(-1/k)), 'resilience')
            hit = float(np.mean(np.array(scores) == nominal)); within1 = float(np.mean(np.abs(np.array(scores)-nominal) <= 1))
            row = dict(W=W, lag=lag, k=k, tau=tau, expected=float(np.exp(-1/k)), mean=float(vals.mean()), std=float(vals.std()),
                       nominal_score=nominal, hit=hit, within1=within1, score_hist={int(s): int(c) for s, c in zip(*np.unique(scores, return_counts=True))})
            out[f"W{W}_lag{lag}_k{k}"] = row
            print(f"W={W} lag={lag} k={k} tau={tau:.0f}: attendu {row['expected']:.2f} mesuré {row['mean']:.2f}±{row['std']:.2f} score nominal {nominal} exact {hit:.2f} ±1 {within1:.2f} {row['score_hist']}", flush=True)
json.dump(out, open('mc_bias.json', 'w'), indent=1)
print("DONE")
