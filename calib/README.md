# calib/ — campagnes de calibration de la résilience (ralentissement critique)

Scripts autonomes (ils lisent `config.json` à la racine, écrivent dans un
sous-dossier du nom donné et y déposent `summary.json`). Rien ici n'est appelé
par le pipeline : ce sont les bancs qui ont servi à régler la métrique
(`docs/DOUBLONS_metriques.md`, suites quater à sexies du 20/09/2026), à rejouer
quand un réglage change (`resilience` dans `config.json`, détrend, barème).

| script | ce qu'il fait | usage |
|---|---|---|
| `mc_bias.py` | table de biais / étalement de l'autocorr à lag L sur une fenêtre W (forçage période 200 + harmoniques, AR(1) 5 %), score à ±1 cran du nominal | `python mc_bias.py` (≈ 2 min) |
| `run_condition.py` | 4 seeds sous bruit d'ENTRÉE ; a montré que le bruit d'entrée n'atteint pas fₙ (→ plancher quiet) | `python run_condition.py <nom> <amplitude|none> <T>` |
| `run_fn_noise.py` | vérité-terrain in-situ : fluctuation AR(1) de τ connu injectée sur fₙ ; attendu au lag L : exp(−L/τ) ; option de démarrage à t_on | `python run_fn_noise.py <nom> <tau_pas> <rel_amp> <T> [t_on] [seeds]` |
| `explore_gain.py` | le couplage αₙ a-t-il une prise sur fₙ ? (non : ×30 ne change rien) | `python explore_gain.py <nom> <gain> <T>` |
| `run_ramp_gamma.py` | rampe interne : le taux de rappel λ de la latence γ dérive de l0 à l1 (perturbation fixe σ sur l'état) ; le radar doit sonner pendant la montée, pas sur un contrôle l0 = l1 | `python run_ramp_gamma.py <nom> <l0> <l1> <t0> <t1> <sigma> <T> <seeds>` |

Repères (W_res_t = 200, lag 10) :

- `run_fn_noise.py fn 80 0.05 350` → ac ≈ 0.88 (attendu exp(−10/80)), score 1.
- `run_fn_noise.py on 80 0.05 500 300 12345,7` → démarrage brutal : le SCORE
  l'attrape en un verdict, le radar (montée graduelle) ne le voit pas.
- `run_ramp_gamma.py rampe 0.1 0.005 250 450 0.005 500 12345,7` et
  `run_ramp_gamma.py ctrl 0.1 0.1 250 450 0.005 500 12345` → alertes sur la
  rampe à t ≥ 425, aucune sur le contrôle après t = 400.

Convention de lecture : pas de lecture du radar avant une longueur de fenêtre
après le premier verdict (la fenêtre qui se remplit est une montée pour lui).
Un run T = 500 dure ~6 min ; les scripts multi-seeds sont séquentiels, lancer
les conditions en parallèle.
