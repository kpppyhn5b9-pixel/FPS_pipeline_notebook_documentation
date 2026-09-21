# FPS — Doublons de métriques à retirer, une fois le switch pris comme référence

*But : partir d'une décision (les six métriques et le mode de scoring du **switch de
perception** sont LA référence) et lister tout ce qui, ailleurs, calcule la même chose
autrement : fonctions, blocs inline, barèmes, colonnes loggées. Complète
`INVENTAIRE_metrics_py.md` (qui recense sans juger) et `CARTE_metriques.md` (qui
explique le sens).*

**Vérifié le 2026-09-19**, branche `claude/metrics-py-inventory-glxowx`. Les lignes
renvoient à l'état du code AVANT le chantier décrit ci-dessous.

---

## ✅ Chantier de mise à niveau — fait le 19/09/2026 (même branche)

Décisions d'Andréa appliquées : les six métriques du switch sont les seules du
pipeline ; un seul barème (`SCORE_BRACKETS`) ; gamma note **S(t) courant**, tout le
reste note **O(t)** ; plus d'inline remplaçable par un appel ; notebook intouché.

**Ce qui existe maintenant (`metrics.py`)**

| Fonction | Rôle |
|---|---|
| `reference_window(config, dt)` | LA fenêtre de scoring : `max(10, perception.W_f_t/dt)` |
| `compute_dispersion`, `compute_fluidity` (jerk de fₙ), `compute_entropy_S`, `compute_mean_abs_error`, `compute_effort`, `compute_adaptive_resilience` | les six valeurs brutes |
| `compute_reference_metrics(history, dt, signal='O'|'S', N)` | les six valeurs sur une fenêtre |
| `score_reference_metrics`, `compute_reference_scores` | LE scoreur (barème `SCORE_BRACKETS`, neutre 3 sans verdict) |
| `FILTER_TO_SCORE_KEY`, `REFERENCE_SCORE_KEYS`, `SCORE_KEY_LABELS`, `labelled_scores` | clés switch ↔ barème ↔ libellés des figures |

**Qui l'appelle** : le switch (`simulate.py`, cible O), `gamma_adaptive_aware`
(`dynamics.py`, cible S), `visualize.calculate_empirical_scores_notebook`,
`plot_scores_evolution`, `plot_signal_scores_S_vs_O` (cible O),
`analyze.analyze_criteria_statistics` (cible O, déclenchement = score <
`perception.seuil_declenchement`), `compare_modes` (barème), `kuramoto.py`
(fonctions de référence).

**Supprimé** : `compute_cpu_step` (temps mur) et la colonne `wall_time_step(t)`,
`compute_variance_d2S` + colonne, l'ancienne `compute_fluidity` sigmoïde et ses
copies inline (`main.py`, `compare_modes.py`), `compute_fluidity_spectral`,
`compute_max_median_ratio` + colonne, `check_thresholds`, `log_metrics`,
`summarize_metrics`, `detect_chaos_events`, `compute_adaptive_window`,
`compute_scores`, `weighted_average`, `calculate_all_scores`, l'auto-test de
`metrics.py`, `visualize.compute_signal_quality_scores`, `build_O_based_history`,
`main.calculate_empirical_scores` et `_safe_score`, `analyze.refine_cpu`, le score
`cpu_cost` (le coût CPU reste loggé via `compute_cpu_step_deterministic`, seule
fonction CPU), les barèmes inline de résilience (`_rb` dans `simulate.py`,
`resilience_v2.score_brackets`, ceux de `compute_adaptive_resilience`), les
réimplémentations de `kuramoto.py`, les seuils de score de `to_calibrate`
(`variance_d2S`, `fluidity_threshold`, `stability_ratio`, `resilience`, `entropy_S`,
`t_retour`, `cpu_step_ctrl`, `max_chaos_events`) et `adaptive_windows.scoring`.
Gardé : `compute_decorrelation_time`, les dérivés de l'effort (`effort_status`,
`mean_high_effort`, `d_effort_dt`), les deux producteurs de `adaptive_resilience`.

**Aligné sans supprimer** : `dynamics.compute_perception_deficit` utilise
`metrics.compute_fluidity` (jerk de fₙ par strate) et `metrics.compute_entropy_S` ;
la colonne `entropy_S` et la colonne `fluidity` du moteur sont calculées sur la
fenêtre `W_f` ; `adaptive_resilience_score` passe par `score_from_brackets`.

**Reste ouvert (décisions, pas des doublons)**
1. La normalisation `1/(1+t_retour)` du chemin ponctuel ne peut pas atteindre les
   scores 4-5 avec le barème `resilience` (plancher ~2 du settling-time) : à
   calibrer côté barème ou côté normalisation, pas par un second barème.
2. `adaptive_resilience` est produite soit par les enveloppes, soit par le chemin
   typé selon `resilience_v2.mode` ; le switch consomme la colonne quelle que soit
   la source (cf. `CARTE_metriques.md` §4.4).
3. Le notebook garde ses copies (hors périmètre).

Validation : 47 tests verts (`test_fps.py`, dont `TestReferenceScores`), pipeline
complet exécuté sur T=40 (3 simulations, 23 figures, batch, comparaison, rapport).

### Suite 20/09/2026 — innovation : `entropy_S` → `innovation_cjs`

`main` a introduit `metrics.compute_innovation_cjs` (complexité statistique C_JS de
l'enveloppe fₙ, moniteur lent, fenêtre ≥ 200 pas, τ auto-calibré) et la colonne
`innovation_cjs`. Le remplacement est maintenant complet :

- **supprimé** : `compute_entropy_S`, la colonne `entropy_S` (moteur, neutral,
  Kuramoto), `final_entropy_S` / `entropy_S` du résumé de run (→ `innovation_cjs`,
  `innovation_cjs_mean`), l'import `scipy.signal` de `metrics.py` ;
- **rebranché sur `innovation_cjs`** : `compare_modes`, `kuramoto.compare_with_fps`,
  le tableau de bord et les listes de colonnes de `visualize.py`, `main.py`
  (scatter, matrice critères-termes), `analyze.py` (corrélations croisées),
  `explore.py` (anomalies), `config.json` (`log_metrics`, `exploration.metrics`),
  `validate_config.py` ;
- **filtre de perception `innovation`** : `dynamics.compute_perception_deficit`
  lit C_JS par strate sur une fenêtre longue de fₙ (`fn_long_win`, fournie par le
  switch) via `metrics.innovation_deficit` = `1 − C_JS / seuil_haut_du_barème`
  (source unique : `SCORE_BRACKETS['innovation']`) ; fenêtre trop courte → déficit
  nul (aucun verdict, aucun poids) ;
- **robustesse** : cellules vides (warmup) lues comme NaN par `explore.load_csv_data`
  et ignorées par `detect_anomalies` ; le mode alerte ignore les `None`.

Validation : 49 tests verts, pipeline complet sur T=60 (`innovation_cjs` rempli sur
401/600 pas, FPS ≈ 0.26–0.30, Kuramoto 0.0, comparaison et rapport OK).

Points d'attention sur C_JS (banc du 20/09) : avec le τ auto-calibré, un sinus pur
score ≈ 0.30 comme le chaos logistique (≈ 0.29) ; C seule ne sépare donc que
bruit ↔ structure, pas ordre ↔ nouveauté. D'où les deux compléments ci-dessous.
Les ex æquo dans `argsort` ont été vérifiés sur run réel : 0 % de pas consécutifs
égaux dans fₙ, aucun traitement spécial nécessaire.

### Suite 20/09/2026 (bis) — plan (H, C) loggé, innovation « sens, pas main »

- `metrics.compute_innovation_plane(fn, dt)` renvoie `{'H', 'C'}` (entropie de
  permutation normalisée et C_JS) d'un seul geste ; `compute_innovation_cjs` en est
  la coordonnée C (une seule implémentation). Colonne `innovation_H` loggée à côté
  de `innovation_cjs` (config `log_metrics`, `METRIQUES_VALIDES`, résumé de run).
  Lecture : H bas = trop ordonné, H haut = trop bruité, H moyen avec C haut =
  nouveauté structurée (`Attention.md`).
- `metrics.OBSERVE_ONLY_FILTERS = ('innovation',)` : l'innovation reste l'un des six
  scores (switch, gamma, figures, analyse) mais le switch ne l'engage **jamais**
  comme remède : aucun poids de perception n'en découle. Métrique d'identité, à
  observer. `compute_perception_deficit('innovation')` reste disponible comme sens
  (saillance par strate) pour le futur geste « relâcher ».

Validation : 51 tests verts ; run FPS T=60 : H ∈ [0.35, 0.65] (médiane 0.51,
côté « structuré »), C ∈ [0.26, 0.30] ; filtres engagés par le switch : neutre,
fluidité, effort, résilience, jamais innovation.

### Suite 20/09/2026 (ter) — résilience : UNE quantité, le ralentissement critique

Spec : `New_Attention.md` (Scheffer 2009, banc AR(1), sonde de calibration,
règle de décision anti-faux-positif). Les cinq couches de résilience de la
CARTE sont dissoutes dans une seule :

- **`metrics.compute_resilience_csd(fn_env, dt, lag)`** : autocorrélation, à lag
  calibré, des résidus **détrendés** de l'enveloppe fₙ (couche lente), plus la
  variance des résidus. Détrend adaptatif par raies spectrales (`detrend_spectral_
  peaks`, proéminence locale : un spectre rouge lisse n'est pas touché, un forçage
  périodique l'est raie par raie ; remplacé le 20/09 par `detrend_periodic`, cf.
  suite quater). Lag = `relaxation_steps` (1/e, source unique
  partagée avec le τ de l'innovation), **calibré une fois puis figé** dans le run :
  re-dériver le lag à chaque fenêtre depuis le même signal donnerait ~1/e par
  construction et rendrait toute montée invisible. `None` si fenêtre < 200 pas,
  signal plat ou résidus nuls (verdict suspendu → score neutre).
- **`metrics.resilience_alert`** : radar CSD codifiant la règle du cahier — bande
  de référence (médiane du calme + max(0.06, 2σ)), tendance sur plusieurs fenêtres
  distinctes (échantillonnées au lag), convergence (autocorr ET variance montent),
  jamais d'alerte sans référence calme complète.
- **Barème** `SCORE_BRACKETS['resilience']` : direction `lower`, seuils
  provisoires `[0.45, 0.60, 0.75, 0.90]` (calme FPS ≈ 0.33, montée CSD observée
  sur jouet 0.5–0.6 ; recalés en facteurs de ralentissement, cf. suite quater).
  Autocorr basse = retour rapide = 5.
- **Colonnes** : `resilience_ac`, `resilience_var`, `resilience_lag`,
  `resilience_alert`, `resilience_score`. Le scoreur de référence lit la dernière
  `resilience_ac` de la fenêtre (moniteur lent, comme l'innovation).
- **Supprimé** : `compute_t_retour`, `compute_continuous_resilience`,
  `compute_adaptive_resilience`, `init_/update_resilience_envelope`, `_robust_iqr`,
  le bloc config `resilience_v2` (→ bloc `resilience`), `t_choc`, les colonnes
  `t_retour`, `continuous_resilience`, `resilience_env(t)`, `D_*`,
  `resilience_metric_used`, `adaptive_resilience(_score)`, la figure
  `plot_adaptive_resilience` (→ `plot_resilience_csd`), l'entrée
  `continuous_resilience` de `compare_modes`. `mu_Rloc(t)` reste loggée (cohérence
  locale, futur gardien σ). Le déficit par strate `'resilience'` (excursion IQR)
  reste : c'est la résilience-excursion, métrique d'attention.

Banc (tests `TestResilienceCSD`) : AR(1) φ = 0.6 / 0.8 / 0.9 / 0.95 → lag-1 pur
0.602 / 0.786 / 0.900 / 0.949 ; avec forçage sinusoïdal ×5 : brut 0.96–0.99
(aveugle), détrendé 0.600 / 0.784 / 0.896 / 0.936 (table du cahier reproduite).
Enveloppe FPS réelle : lag auto 39–40, autocorr ≈ 0.33 (cahier : 38, 0.339).

Validation : 54 tests verts ; pipeline complet T=80 : `resilience_ac` sur 401/800
pas, lag 40, 0 alerte, score 5 sur 400 pas (3 en warmup).

Points d'attention : (1) l'estimateur par fenêtre glissante de 400 pas au lag 40
n'a que ~10 échantillons indépendants, la valeur pas à pas oscille (−0.46 … 0.47
sur le run) ; le cahier recommande de lire une tendance lissée, pas un point —
le radar le fait, le score lit encore le point ; (2) les seuils du barème sont
provisoires, à recaler sur plusieurs seeds ; (3) `W_res_t` (40 u.t.) et
`alert_calm_n` sont des réglages, pas des vérités ; (4) en régime établi et sans
bruit d'entrée, l'enveloppe fₙ de la FPS est quasi périodique : une fois le
transitoire sorti de la fenêtre, la variance des résidus tombe près de zéro et
l'autocorr (≈ 0.15–0.25) lit un reste de détrend plus qu'une turbulence spontanée.
La métrique prend tout son sens quand quelque chose fluctue (bruit, perturbation
continue, rampe interne) — exactement les régimes que le cahier a testés.

### Suite 20/09/2026 (quater) — résilience : score lissé, barème en facteurs de ralentissement, comportement réel sous bruit

Deux chemins suivis (accord d'Andréa) : un score plus pertinent et moins bruité,
et des campagnes pour mesurer ce que la métrique lit VRAIMENT sous bruit.

**Ce que le score lit.** `resilience_ac_smooth` = médiane des derniers verdicts
sur `smooth_lags` × lag (`smooth_resilience_ac`), quiet compris (= 0.0) : un
calme retrouvé ramène le score à 5. La valeur brute `resilience_ac` reste
loggée (moniteur), le radar ne lit que les fluctuations réelles.

**Barème.** Seuils = exp(−1/k), k ∈ {1.4, 2, 3, 5} → [0.49, 0.607, 0.717, 0.819]
(`RESILIENCE_SLOWING_FACTORS`). Sens : au lag L, un temps de retour τ = k·L
donne une autocorr exp(−1/k). Score 5 = retour plus rapide que 1.4 lag, score 1 =
plus lent que 5 lags. Le lag est donc l'échelle de référence à laquelle le
ralentissement est comparé.

**Campagne 1 — bruit d'entrée (`bruit_leger`, `bruit_fort`, 4 seeds).** Le bruit
sur l'entrée ne touche pas l'enveloppe fₙ (il est orthogonal à la phase) : les
résidus restent à ~1e-10 de la variance de l'enveloppe. Une autocorr lue
là-dessus est un reste numérique, pas une dynamique (un seed flottait 4/5 sur du
vide). → **Plancher quiet** (`quiet_floor_rel` = 1e-6) : ac = 0.0, quiet = 1,
score 5, le radar ignore l'échantillon.

**Campagne 2 — fluctuation injectée sur fₙ (`run_fn_noise.py`).** Vérité-terrain
in-situ : bruit AR(1) multiplicatif en mode commun sur fₙ, τ = 20 / 80 / 160 pas,
5 % d'amplitude ; au lag L, l'autocorr attendue vaut exp(−L/τ). Première lecture
(fenêtre 400 pas, lag 40) : ≈ 0.1 quel que soit τ. Deux causes, corrigées :

1. *Le détrend effaçait la fluctuation.* L'évidement de bins (raie ± 2 voisins)
   dans une fenêtre de 400 pas sur une enveloppe de période 200 couvrait tout ce
   qui est plus lent que ~50 pas. Remplacé par **`detrend_periodic`** : ajustement
   itératif de raies (bin dominant → fréquence affinée par nombre d'or → réajuste-
   ment conjoint Gauss-Newton de toutes les raies). Critère de raie **à deux
   côtés** : P[k] > peak_ratio × médiane à gauche ET à droite (lobe k±1 exclu,
   ≥ 2 bins de chaque côté). Un spectre rouge (AR(1)) est monotone : jamais
   touché (faux positifs ≤ 10 %, une raie sur 2000 bins, sans effet). L'enveloppe
   FPS calme (période 200 + harmoniques jusqu'au bin 14) se réduit à 2e-10 de sa
   variance en 8 raies ; une fluctuation τ=160 injectée ressort corrélée > 0.9 à
   la réalisation de bruit.
2. *La fenêtre était trop courte pour le lag.* Table de calibration
   (`mc_bias.py`, forçage période 200 + 2 harmoniques, AR(1) 5 %, 30 tirages) :
   l'estimateur à lag L sur W pas est biaisé vers le bas d'≈ 2τ/W, étalement en
   √(τ/W). Score à ±1 cran du nominal (τ = k·L, k = 1…8) :

   | W / lag | 1000 / 10 | 1000 / 20 | 1000 / 40 | 2000 / 10 | 2000 / 20 | 2000 / 40 |
   |---|---|---|---|---|---|---|
   | ±1 cran (min sur k) | 87 % | 57 % | 17 % | **100 %** | 93 % | 50 % |
   | biais max (k=8) | −0.06 | −0.16 | −0.34 | **−0.03** | −0.08 | −0.23 |

   → règle : **W ≥ 200 × lag**. `W_res_t` passe à 200 u.t. (2000 pas) et le lag
   auto (relaxation 1/e de l'enveloppe ≈ 40) est plafonné à W/`min_lags_per_
   window` = 10 pas. Une fenêtre trop courte pour séparer forçage et fluctuation
   (W < 20 × relax : la raie n'a pas 4 bins) rend None, verdict suspendu, au lieu
   de lire l'enveloppe elle-même (W=400 : ac ≈ 0.998, faux score 1).

**Résultat in-situ après correction** (W=2000, lag 10, 4 seeds, régime t > 200,
médiane de `resilience_ac` par seed) :

| τ injecté (pas) | attendu exp(−10/τ) | mesuré (4 seeds) | score nominal | scores observés |
|---|---|---|---|---|
| 20 | 0.607 | 0.590 – 0.625 | 3/4 (τ = 2·lag, sur le seuil) | 3 et 4 |
| 80 | 0.882 | 0.863 – 0.892 | 1 | 1 (2 en bordure) |
| 160 | 0.939 | 0.909 – 0.942 | 1 | 1 |
| calme (rien d'injecté) | — | quiet (ac = 0) | 5 | 5 |

La métrique lit maintenant le temps de retour injecté à ~0.02 près, dans le
pipeline complet, à travers l'enveloppe périodique. Radar : 0 alerte sur 3 seeds
sous fluctuation STATIONNAIRE (correct : pas de montée) ; le seed 12345 alerte
(220–410 pas) parce que sa référence calme (100 premiers échantillons) tombe sur
la fin du transitoire, plus basse que le régime — la règle du cahier suppose une
référence représentative ; `alert_calm_n` est à régler avec le début du régime.

**Calme, après correction du lissage** (seed 12345, T=350, W=2000, lag 10) : dès
que le transitoire est sorti de la fenêtre (t ≥ 230), quiet = 100 %, ac = 0,
0 alerte, score 5 sur tout le régime (quiet_share du run 0.94). Entre t = 200 et
230 la fenêtre contient encore la queue du transitoire : 72 % quiet, quelques
scores 1–2 (la fluctuation du transitoire est lente, et c'est vrai). Avant
correction, la médiane lissée restait figée sur ces valeurs du transitoire
(0.41, score 4) pour tout le run : le score lit maintenant TOUS les derniers
verdicts, quiet compris.

**Coût.** Le détrend vaut ~50 ms par appel à W=2000 : calcul tous les `stride`
= 5 pas (le dernier verdict est reporté entre deux), ~50 s par run T=500.

**Ce qui reste ouvert.** (1) Le lag de référence est une convention (10 pas =
1 u.t.) : l'enveloppe calme de la FPS ne fluctue pas, il n'y a pas de τ₀ naturel
à calibrer ; le barème dit « lent par rapport à 1 u.t. ». (2) Sous fluctuation
stationnaire lente, le score est bas (1) sans que rien ne « monte » : le score
lit un NIVEAU de lenteur, le radar lit une TENDANCE ; les deux lectures sont
complémentaires, pas redondantes. (3) Les premiers verdicts arrivent à t = 200
(fenêtre pleine) : sur un run T=500, 60 % du run est lu.

### Suite 20/09/2026 (quinquies) — radar : référence glissante

Suite du point ouvert de la campagne quater : sur le seed 12345, le radar
sonnait 220–410 pas sous fluctuation STATIONNAIRE parce que sa référence calme
(les 100 premiers verdicts du run) tombait sur la queue du transitoire, plus
basse que le régime. Trois options étaient sur la table (décaler le début de la
référence ; référence glissante ; montée relative au dernier calme). Retenue :
la **référence glissante**, la plus juste par rapport au cahier — un signal
précoce lit une MONTÉE, pas un niveau.

**Règle** (`metrics.resilience_alert`, param `gap`) : la référence est faite des
`calm_n` verdicts qui se terminent `gap` verdicts avant les échantillons de
tendance, et non des `calm_n` premiers du run. Bande, tendance sur n_windows
fenêtres distinctes et convergence ac + variance : inchangées. Conséquences :

- une fluctuation stationnaire, même lente (score 1), ne sonne pas : le présent
  ressemble au passé récent ;
- une montée sonne PENDANT qu'elle a lieu (le gap la sépare de sa propre
  référence), puis se tait une fois le plateau atteint. Le score, lui, reste bas
  sur le plateau : niveau (score) et tendance (radar) sont deux lectures ;
- jamais d'alerte sans calm_n + gap + tendance verdicts derrière soi.

**Config** : `alert_calm_n` (en entrées) → `alert_calm_lags` = 20 et
`alert_gap_lags` = 20, en LAGS, convertis en entrées d'historique par simulate
(un lag = lag/stride entrées). Avec lag 10 et stride 5 : référence de 200 pas,
séparée du présent par 200 pas ; premier verdict possible ~430 pas après la
première fluctuation lue.

**Tests** (`test_radar_reference_slides_with_the_run`) : transitoire bas puis
plateau stationnaire → 0 alerte en régime ; montée à mi-parcours → 0 avant,
alerte pendant la montée, 0 sur le plateau.

**Campagne in-situ** (`run_fn_noise.py`, τ = 80 pas, 5 %, W=2000, lag 10) :

| run | seed | alertes (avant) | alertes (glissante) | où |
|---|---|---|---|---|
| stationnaire dès t=0, T=350 | 12345 | 410, tout le run | 250 | t ∈ [200, 300) puis 0 |
| stationnaire dès t=0, T=350 | 2024 | 0 | 55 | t ∈ [250, 300) puis 0 |
| démarrage à t=300, T=500 | 12345 | — | 80 | t ∈ [330, 360) puis 0 |
| démarrage à t=300, T=500 | 7 | — | 0 | — |

Lecture. (1) Le faux positif persistant a disparu : sous fluctuation
stationnaire, plus aucune alerte une fois la lecture installée (t ≥ 300). (2) Les
alertes restantes entre t = 200 et 300 sont la fenêtre qui se remplit : au
premier verdict (t=200) la fenêtre contient encore le transitoire, et
l'autocorr monte vers son plateau à mesure qu'il en sort ; pour le radar, c'est
une montée, et il a raison — c'est la mesure qui s'installe, pas le système.
Convention à retenir : ne pas lire le radar avant une longueur de fenêtre après
le premier verdict (t ≥ W_res_t + première fenêtre). (3) Le démarrage brutal
d'une fluctuation lente n'est PAS une montée graduelle : l'autocorr saute de 0
(quiet) à ~0.9 en un verdict (seed 7 : 0.88 à t=300, 0.96 à t=310) puis
REDESCEND vers son plateau (0.86) à mesure que la fenêtre se remplit. Le radar
(3 fenêtres croissantes, ac ET variance) ne voit pas un saut suivi d'une
descente : 0 alerte sur le seed 7, 80 sur le seed 12345 (qui a mis deux verdicts
à sauter). C'est le SCORE qui attrape le démarrage, immédiatement (5 → 1 en un
verdict) ; le radar est fait pour l'approche graduelle d'une transition
(Scheffer), pas pour un interrupteur. Les deux lectures se complètent.

Point ouvert : le seul signal graduel qu'on ait produit in-situ est celui de la
fenêtre qui se remplit. Un vrai test du radar demande une rampe interne de la
FPS (un paramètre qui dérive lentement vers une transition), pas une injection
qui s'allume ; c'est une campagne à part, côté dynamique.

### Suite 20/09/2026 (sexies) — rampe interne : le radar sur une vraie approche de transition

Question restée ouverte : le radar (montée graduelle de l'autocorr + de la
variance) n'avait été vu que sur la fenêtre qui se remplit. Il fallait une
dérive LENTE d'un mécanisme interne de la FPS vers un temps de retour plus long.

**Ce qui n'a pas de prise.** Multiplier le couplage αₙ par 3, 10 ou 30 ne
change rien à fₙ (stats identiques à la 3ᵉ décimale) : avec Aₙ ≈ 0.03, le terme
αₙ·Sᵢ = αₙ·Σ wⱼ Oⱼ est négligeable devant f₀. Dans ce régime, le couplage inter-
strates n'est pas un levier sur la couche lente.

**Le mécanisme à mémoire de la couche lente, c'est la latence.** fₙ = f₀·(1 +
βₙ·γ) et γ (adaptive_aware) converge vers sa cible par lissage exponentiel
(`dynamics.py:1002`, taux 0.1). Un taux de rappel qui faiblit = un temps de
retour qui s'allonge = le ralentissement critique de Scheffer, littéralement.
Campagne (`run_ramp_gamma.py`, scratch de session) : on enveloppe
`compute_gamma_adaptive_aware` : γ ← (1−λ)·γ + λ·γ_cible + σ·ε, ε ~ N(0,1) par
pas (perturbation FIXE sur l'état, σ = 0.005), λ dérivant linéairement de 0.1 à
0.005 entre t = 250 et 450 (T = 500). Contrôle : même bruit, λ = 0.1 constant.

**Résultats** (W=2000, lag 10, médiane de `resilience_ac` et alertes par tranche
de 25 u.t.) :

| t | 275 | 300 | 325 | 350 | 375 | 400 | 425 | 450 | 475 |
|---|---|---|---|---|---|---|---|---|---|
| λ (rampe) | .082 | .070 | .058 | .047 | .035 | .023 | .011 | .005 | .005 |
| contrôle : ac | .78 | .75 | .76 | .76 | .77 | .78 | .77 | .68 | .63 |
| contrôle : alertes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rampe 12345 : ac | .75 | .75 | .77 | .78 | .79 | .81 | .83 | **.94** | **.97** |
| rampe 12345 : alertes | 9 | 0 | 0 | 0 | 0 | 0 | **31** | **250** | **129** |
| rampe 7 : ac | .75 | .76 | .67 | .66 | .67 | .66 | .72 | **.79** | **.89** |
| rampe 7 : alertes | 0 | 0 | 0 | 0 | 0 | 0 | **80** | **171** | **129** |

Variance des résidus : ×5 sur la rampe (0.001 → 0.005), stable au contrôle.
Score : 2 sur le plateau (les deux runs), 1 en fin de rampe.

Lecture. (1) **Le radar fait son travail** : il sonne sur les deux seeds pendant
la montée (dernier quart de la rampe, puis le plateau lent), et jamais sur le
contrôle une fois la lecture installée. (2) Il sonne TARD : l'autocorr de base
de fₙ sous ce bruit est déjà ≈ 0.75 (deux lissages en cascade, celui de la FPS
et le nôtre, plus la contrainte spiralée), et la dérive ne domine qu'une fois
1/λ au-delà de la mémoire interne (λ ≲ 0.02). La colonne « attendu (1−λ)^lag »
du script ne vaut que pour notre lissage seul : c'est le contrôle qui donne la
base. (3) La bouffée d'alertes du contrôle à t ∈ [250, 275) (145) est, encore,
la fenêtre qui se remplit après le premier verdict (t=200) ; la convention
« pas de lecture du radar avant une longueur de fenêtre après le premier
verdict » tient, et ici c'est t ≥ 400 : toutes les alertes de la rampe sont
après. (4) Le seed 7 montre une baisse d'autocorr (0.76 → 0.66) entre t = 325 et
400 sans alerte : le radar ne lit qu'une MONTÉE, une descente est du calme.

Ce que ça ne dit pas : la rampe est imposée sur le taux de rappel de γ, pas
produite par la FPS elle-même. Le jour où un régime interne (régulation, effort
chronique, φ adaptatif) fera dériver ce taux, le radar est prêt à le lire.

### Suite 20/09/2026 (septies) — noms affichés : une source unique

Nettoyage ultime (accord d'Andréa) : chaque métrique porte le nom de ce
qu'elle MESURE, partout où un humain la lit. Rien ne change dans les appels,
les signatures, les clés de scores, les noms de filtres du switch, les colonnes
du CSV de run, les clés des dicts de résultats ni les clés de config : ce sont
des identifiants consommés par le code (readers, validateur, tests).

| clé de score | filtre | nom affiché (avant) | nom affiché (après) |
|---|---|---|---|
| `dispersion` | `stabilite` | Stabilité | **Dispersion** |
| `regulation` | `erreur` | Régulation | Régulation |
| `fluidity` | `fluidite` | Fluidité | **Fluidité jerk** |
| `resilience` | `resilience` | Résilience | **Résilience CSD** (critical slowing down, ralentissement critique) |
| `innovation` | `innovation` | Innovation | **Innovation C_JS** |
| `activite` | `effort` | Effort interne | **Activité** |

**Source unique** (`metrics.py`) : `SCORE_KEY_LABELS` (clé de score → libellé),
`FILTER_LABELS` (nom de filtre → libellé, + `neutre`), `COLUMN_LABELS` (colonne
ou clé de résultat → libellé, ex. `resilience_ac` → « Résilience CSD
(autocorr) », `mean_high_effort` → « Activité chronique », `mean_abs_error` →
« Régulation (|E−O|) »), et `metric_label(nom)` qui accepte les trois formes et
rend un nom inconnu tel quel (`S(t)`, `C(t)`… restent eux-mêmes).
`labelled_scores` et `labelled_summary` traduisent un dict entier.

**Passés par la source unique** (inventaire du 20/09) : toutes les figures de
`visualize.py` (tableau de bord, S vs O, grille empirique, évolution des scores,
évolution des métriques brutes, détail par métrique, corrélations, nuages de
points, matrice critères ↔ termes, figure résilience, rapport HTML) ; les prints
de `main.py` (figures, métriques finales) et de `simulate.py` (métriques
finales, journal d'alertes : libellé + clé entre parenthèses) ; `analyze.py`
(critères déclenchés, changelog) ; `compare_modes.py` (prints et rapport TXT :
libellé + clé, les clés JSON restent) ; `explore.py` (rapport Markdown, top
corrélations) ; `utils.py` (résumé de `runs_completed.txt`, latent).

Convention : quand un texte sert aussi de trace technique (journal d'alertes,
changelog, rapport TXT de comparaison), le libellé est suivi de la clé entre
parenthèses, pour que le lecteur puisse retrouver la colonne.

Hors périmètre : `notebooks/*.py` (copies figées) et le notebook.

### Suite 20/09/2026 (octies) — activité : score relatif au repos du run (« la grenouille »)

Constat (toutes campagnes, 4 seeds chacune, activité en régime) : calme →
médiane 271, p90 334 ; fluctuation fₙ injectée ou rampe de latence → idem ;
bruit d'entrée léger → 282 ; bruit d'entrée fort → 370 (p90 610). Avec le
barème fixe de juillet [30, 45, 75, 150] (repos d'alors : 59), la FPS au repos
lisait 1/5 par construction, et les seuils fixes de statut (chronique > 75,
transitoire > 150), sous le niveau de repos, rendaient « stable » inatteignable
(calme = 92 % transitoire, 8 % chronique).

**Décision (Andréa, 20/09) : référence relative FIGÉE**, pas de nouveaux seuils
fixes (qui périmeraient à la prochaine dérive d'échelle) et pas de référence
glissante (qui absorberait une montée lente : la grenouille dans l'eau qui
chauffe). L'activité n'a pas d'unité propre (somme de variations de paramètres,
dépend de N, dt, de la dynamique) : la question honnête est « court-elle plus
qu'à son repos ? ».

- **Repos de référence** : `metrics.activity_reference` = médiane de l'activité
  sur [`calib_start_t`, `calib_start_t` + `calib_window_t`] (config, bloc
  `activite`, 20 + 20 u.t.), donc APRÈS l'exploration initiale de γ et G (gros
  pic d'activité qui n'est pas du repos). Calibrée une fois par simulate, figée,
  loggée à chaque pas (`activite_ref`, None avant) ; résumé `activite_ref`.
- **Score `activite`** : `SCORE_BRACKETS['activite']` note le RAPPORT
  activité / repos (`metrics.activity_ratio`), seuils = `ACTIVITY_FACTORS`
  (1.1, 1.25, 1.5, 2) : 5 sous ×1.1, 4 sous ×1.25, 3 sous ×1.5, 2 sous ×2, 1
  au-delà. Sans référence → None → neutre (verdict suspendu, comme la
  résilience en échauffement). Le switch, γ, les figures et l'analyse de batch
  lisent `activite_ref` dans les lignes d'historique / le CSV : aucun appel ne
  change.
- **Statuts** (`compute_effort_status(…, ref=)`) : transitoire = un pas au-delà
  de ×2 du repos (pic) ; chronique = moyenne des 50 derniers pas tenue au-delà
  de ×1.25 ; stable sinon. Sans référence (calibration en cours) : seul un pic à
  +2σ est signalé, jamais « chronique ». Les clés `effort_chronique_threshold` /
  `effort_transitoire_threshold` de `to_calibrate` ne sont plus lues (retirées
  de config, avertissement « obsolète » si présentes).

Ce que ça ne dit pas : un run agité dès sa naissance a une référence agitée et
lit 5 — « pas plus qu'à son repos », ce qui est vrai, mais pas « calme ». Pour
ce cas, la valeur brute `effort(t)` reste loggée et se compare entre runs.

**Vérification in-situ** (seed 12345, T = 140, repos calibré sur [40, 60] ;
`activite_rel` = niveau sur une respiration / repos, par tranche de 20 u.t.) :

| run | repos (`activite_ref`) | t ∈ [60, 80) | [80, 100) | [100, 120) | [120, 140) |
|---|---|---|---|---|---|
| calme | 265 | 1.01 · score 5 · stable 100 % | 1.02 · 5 · stable | 1.01 · 5 · stable | 1.01 · 5 · stable |
| bruit d'entrée fort dès t = 70 | 265 | 1.01 · 5 · stable (transitoire 34 % : premiers pics) | 2.22 · 1 (69 %) · chronique 95 % | 2.70 · 1 · chronique 100 % | 2.38 · 1 · chronique |
| bruit d'entrée fort dès t = 0 | 398 | 0.93 · 5 · stable | 0.92 · 5 · stable | 0.97 · 5 · stable | 1.00 · 5 · stable |

Lecture : au calme, 5 et « stable » sans exception (avant : 1 et 92 %
« transitoire ») ; l'agitation qui s'installe après la calibration est lue en
une respiration (×2.2 → score 1, statut chronique) ; l'agitation présente dès
la naissance donne un repos agité (398) et un score 5 — c'est la limite
annoncée, vraie au sens de la métrique (« pas plus qu'à son repos »), à lire
avec `effort(t)` brut entre runs. Pourquoi le repos se calibre à 40 u.t. : la
médiane par tranche de 20 u.t. au calme vaut 204, 234, puis 270 ± 1 % à partir
de t = 40 (4 seeds) ; et pourquoi le niveau se lit sur 200 pas : sur 50 pas le
rapport oscille entre 0.86 et 1.28 (la respiration de l'enveloppe fₙ), sur 200
entre 0.98 et 1.02.

### Suite 21/09/2026 (nonies) — dispersion : ce qu'est le centre, lecture lissée, sur O seulement

La cloche de dispersion (commit 87f2943, session jumelle) note std(ΣOₙ)/√N
autour de 0.0157, des deux côtés (gel, emballement), en observation seule. Trois
ajustements, à trois mains (Andréa, la session jumelle, celle-ci).

**1. Le centre n'est pas une constante empirique : c'est ⟨Aₙ⟩/√2.** Pour N
sinusoïdes d'amplitude A et de phases indépendantes, std(ΣO) = A·√(N/2), donc
std(ΣO)/√N = A/√2. Vérifié sur toutes les campagnes (rapport std(ΣO)/√N sur
⟨Aₙ⟩/√2) :

| condition | std(ΣO)/√N | ⟨Aₙ⟩/√2 | rapport |
|---|---|---|---|
| calme (4 seeds) | 0.0154–0.0160 | 0.0153 | 1.00–1.05 |
| fluctuation fₙ injectée, rampe de latence | 0.0156–0.0158 | 0.0153 | 1.01–1.03 |
| bruit d'entrée fort | 0.0181–0.0187 | 0.0176 | 1.02–1.07 |

Conséquences : (a) l'invariance en N est expliquée ; (b) le centre bouge avec
le régime d'amplitude (A₀, échelle d'entrée, enveloppe), jamais avec N — c'est
exactement ce qu'il faut surveiller ; (c) la cloche lit amplitude × cohérence
collective : « emballement » = aussi la synchronisation (somme cohérente ∝ N),
« gel » = chute d'amplitude ou annulation en antiphase ; (d) la chimère saine
est incohérente au premier ordre (rapport 1), l'accord spiralé C(t) vit au
second ordre. Recette rangée : `calib/dispersion_center.py` (mesure les deux
sur un CSV et la distribution du repli par fenêtre). **Garde-fou** :
`TestDispersionGuard` fait tourner un run calme réel (N=30, T=40) et exige que
le centre soit ⟨Aₙ⟩/√2 à 15 % près et que la cloche lise 5 sur tout le régime :
un régime d'amplitude qui change casse la suite au lieu de laisser un voyant
d'identité lire 4 en silence (le piège de l'activité en juillet).

**2. Lecture lissée.** Par fenêtre W_f, le repli sain atteint 1.40–1.46 au
calme (1–2 % de fenêtres à 4) et jusqu'à 16 % de fenêtres à 4 sous bruit fort
(régime sain, amplitude plus haute). simulate logge `dispersion_norm` =
std(ΣO)/√N sur W_f, médiane sur `DISPERSION_SMOOTH_WINDOWS` = 3 fenêtres ; le
scoreur lit cette colonne (dernière valeur), repli sur la fenêtre pour un CSV
ancien. Le bord du 5 reste ×1.35 : la cloche garde sa finesse, le point bruité
ne fait plus verdict.

**3. Sur O(t) brut, jamais sur S(t).** La dispersion était la seule métrique
qui dépendait du signal noté, et γ note S(t). Quand un filtre de perception
pondère les strates, std(S) quitte le centre pour une raison qui n'a rien à
voir avec la chimère : γ aurait vu un 4 fabriqué par l'attention elle-même. Un
filtre change ce que le système regarde, pas qui il est : la dispersion rejoint
la fluidité et l'innovation parmi les métriques agnostiques au signal.
Conséquence à nommer : aujourd'hui aucune des six ne dépend du signal, donc les
scores sur S(t) (γ) et sur O(t) (switch) sont identiques. La plomberie
`signal='S'` reste, inerte, pour le jour où une métrique lira légitimement le
perçu (une erreur perçue, par exemple). Le panneau S vs O de la figure le dit
désormais tel quel.

### Suite 21/09/2026 (decies) — métastabilité migrante contextuelle : germe, mécanisme, prototype

Question (Andréa, Attention.md « La métastabilité migrante contextuelle ») : la
migration de la cohérence est-elle en germe dans la FPS, et comment la câbler
pour qu'elle serve sans rien casser ? Résultats du jumeau (kymographes, φ gelé,
c = 0/0.1/0.5, projecteur de contexte) : chimère imposée, non émergente ;
contexte → amplitude parfaitement, cohérence pas du tout. Ici : le mécanisme
exact derrière ces observations, puis un prototype de câblage hors pipeline.

**1. Anatomie de la lentille couplage (mesuré sur la config standard).**

- Le voisinage de couplage est de DEUX strates, n−1 et n+1, poids ∓0.1 (somme
  nulle) ; μ_Rloc normalise en |w| → 0.5/0.5. Donc
  **Rloc_n = |cos((θ_{n+1} − θ_{n−1})/2)|** : la cohérence locale d'une strate est
  l'alignement de ses deux voisines. Sa propre phase n'y entre pas, le contexte
  non plus.
- La contrainte spirale de `compute_fn` (fₙ₊₁ ← ½ fₙ₊₁ + ½ r·fₙ, r ≈ 1.618) est
  une **moyenne mobile exponentielle le long de la chaîne** (mémoire ~5 strates,
  gain 2.6) : corrélation 0.9999 entre fₙ mesuré et cette prédiction sur f₀.
  C'est elle qui fabrique la cascade lisse fₙ ∈ [0.4, 6.3] à partir de f₀ ∈
  [0.4, 2.4], puis × (1 + β·γ) : toutes les fréquences respirent ensemble avec
  γ (± 20 %, période 20 u.t.).
- Le « plaid » des kymographes est donc un **battement** à la fréquence
  f_{n+1} − f_{n−1} (médiane 0.12 cycle/u.t., période 8 u.t.). La moyenne
  temporelle de |cos| vaut 2/π = 0.64 : c'est μ_Rloc au repos (0.63 mesuré).
  La « strate épinglée » vers 30 n'est pas un bord : entre 28 et 31 la cascade
  lissée est plate (Δf ≈ 0.01, battement de ~100 u.t.), un renversement du
  bruit de f₀ ; la strate 96 l'est davantage (Δf = 0.0004).
- Le contraste σ_Rloc (0.31) est invariant parce que chaque Rloc_n balaie [0, 1]
  à son propre rythme : la distribution instantanée sur n ne dépend d'aucun
  paramètre lent. D'où l'invariance à γ, à l'entrée, au couplage.
- **La route contexte → fréquence existe et vaut 2·10⁻⁴** : |α·Sᵢ·β| / f₀,
  médiane 2·10⁻⁴, max 2·10⁻³ (Sᵢ = Σ w·O ≈ 2·10⁻³, poids 0.1 sur deux voisins,
  O ≈ 0.02). Morte par petitesse, pas par absence ; ×30 sur α ne change rien
  (20/09). Le germe de migration-de-cohérence n'est pas là : confirmé et expliqué.

**2. Prototype de câblage (`calib/run_context_binding.py`, hors pipeline).**
Projecteur de contexte gaussien (largeur 4 strates) posé 20 u.t. sur les
strates 20, 80, 50, 20 ; saillance sₙ = le projecteur ; après `compute_fn`
(donc après le lissage spiral), deux gestes pondérés par sₙ et par garde(σ) :

- **rapprocher les fréquences** : fₙ ← fₙ + κ·sₙ·(f̄_s − fₙ) (le levier des
  ratios, celui que le jumeau avait triangulé) ;
- **accrocher les phases** : Δfₙ = Kₙ/2π · Σ_j pw_j sin(θ_j − θ_n) sur le
  voisinage de couplage, Kₙ = K₀·sₙ·garde(σ)·gₙ.

Trois enseignements de banc, dans l'ordre où ils sont venus :

1. *L'accrochage de phase seul ne lie pas* (K₀ = 2, 6, 30 : cohérence dedans =
   dehors). Tirer la strate du milieu vers deux voisines qui s'écartent l'une de
   l'autre à 0.2–0.5 cycle/u.t., c'est tirer dans les deux sens ; verrouiller
   une chaîne de neuf strates contre ce gradient demanderait K ≫ gradient × L²/8.
2. *Rapprocher les fréquences seul ne lie pas non plus* (κ = 0.5, 1.0) : les
   phases cessent de dériver mais restent figées à leur écart du moment ; la
   moyenne de |cos| sur l'îlot reste 0.64. Il faut les DEUX gestes : les
   fréquences pour arrêter la dérive, la phase pour fermer l'écart. C'est la
   double-ancre du jumeau, retrouvée par la dynamique.
3. *Le facteur (1 − Rloc_n) empêche de tenir* : la force s'annule exactement
   quand l'îlot est aligné, le résidu de désaccord aux bords (s < 1) le
   re-désynchronise, Rloc retombe, la force revient : cycle 0.99 → 0.3 → 0.99
   (diagnostic au centre du projecteur). Les rendements décroissants doivent
   porter sur l'allocation de saillance, pas sur la force qui tient.
4. *Le Kuramoto premier-voisin ne ferme pas l'écart, même sans ce facteur* :
   sur une chaîne, le couplage symétrique aux deux voisines a pour point fixe
   TOUTE torsion uniforme (½ sin δ + ½ sin(−δ) = 0). Il uniformise la torsion
   mais ne la fixe pas ; or Rloc_n = |cos δ| dérive avec elle. Le cœur de
   l'îlot se verrouille puis se déverrouille par cycles (0.83 → 0.34 sur deux
   demi-postes). Pour fermer l'écart il faut tirer chaque strate vers la phase
   MOYENNE de l'îlot (paramètre d'ordre Z_s, pas de mode neutre) : c'est le
   liage « champ moyen d'îlot », en cours de mesure.

**Résultats du liage champ moyen d'îlot** (projecteur 20 → 80 → 50 → 20, 20 u.t.
par poste ; R îlot = Rloc moyen des strates où s > 0.5 ; dehors = s < 0.05 ;
« ancienne zone » = le poste précédent) :

| variante | R îlot | R dehors | corr(s, R) | R ancienne zone | σ_Rloc | dispersion (score) | activité |
|---|---|---|---|---|---|---|---|
| contrôle (K₀ = 0) | 0.53–0.70 | 0.61–0.66 | −0.2…+0.1 | 0.65–0.73 | 0.30–0.31 | 0.015 (5) | 1.01 |
| premier-voisin, κ = 1, K₀ = 6, sans (1−R) | 0.34–0.83 (cycles) | 0.62–0.66 | −0.55…+0.40 | 0.57–0.67 | 0.30–0.31 | 0.016 (5) | 1.02 |
| **champ moyen, κ = 1, K₀ = 2** | **0.78–0.99** | 0.62–0.66 | +0.29…+0.52 | 0.57–0.74 | 0.31–0.32 | 0.018–0.022 (5) | 1.00 |
| champ moyen, κ = 0.5, K₀ = 3 | 0.71–0.98 | 0.62–0.66 | +0.20…+0.55 | 0.56–0.74 | 0.31 | 0.017–0.021 (5) | 1.01 |
| champ moyen, κ = 0, K₀ = 6 (phase seule) | 0.85–0.99 | 0.63–0.66 | +0.39…+0.57 | 0.56–0.74 | 0.30–0.31 | 0.021–0.023 (5, limite) | 1.02 |
| champ moyen, κ = 1, K₀ = 6 | 0.95–1.00 | 0.62–0.66 | +0.51…+0.66 | 0.56–0.75 | 0.31–0.32 | 0.021–0.024 (**4**) | 1.01 |

Lecture.

- **La cohérence suit le contexte**, franchement : l'îlot passe de 0.64 (le
  2/π du battement) à 0.8–1.0, le reste du chœur ne bouge pas (0.62–0.66), et
  la corrélation spatiale saillance ↔ cohérence passe de ~0 à +0.5. C'est la
  migration-de-cohérence contextuelle, pilotée, pas en germe mais câblée.
- **Réversible sans mémoire** : l'ancienne zone revient au battement (0.56–0.75)
  dès le poste suivant, parce que le geste s'applique à fₙ pas à pas, jamais à
  f₀ ; retirer le contexte, c'est retirer le geste.
- **La chimère tient** : σ_Rloc reste à 0.31 ± 0.01 dans toutes les variantes ;
  un îlot de 9 strates ne change pas le contraste global. Activité inchangée
  (≈ 1.0 du repos), régulation, fluidité, innovation à 5.
- **Le gardien, c'est la cloche de dispersion** : neuf strates parfaitement
  cohérentes ajoutent une somme cohérente à ΣO (∝ 9 au lieu de √9), std(ΣO)/√N
  monte de 0.015 à 0.022–0.024, repli 1.45 → score 4 à K₀ = 6. À K₀ = 2–3 la
  cohérence reste LÂCHE (0.8–0.95) et la cloche reste à 5. C'est la cohérence
  partielle de Fries, avec sa mesure de garde intégrée : lier sans fusionner.
- Le champ moyen d'îlot seul (κ = 0, K₀ = 6) suffit à verrouiller contre le
  gradient de fréquences ; la compression κ permet de tenir avec un K₀ plus
  faible, donc plus doux. Les deux ancres du jumeau, à nouveau.

**3. Proposition de câblage (à décider, rien n'est branché).** Dans
`simulate.py`, juste après `fn_t = dynamics.compute_fn(...)` (donc après le
lissage spiral, avant `phase_acc += 2π fn dt`) :

    Z_s  = Σ sₙ e^{iθₙ} / Σ sₙ            (θ = phase_inst du pas précédent)
    fₙ  += κ·garde·sₙ·(f̄_s − fₙ)          (rapprocher, optionnel)
    fₙ  += K₀·garde·sₙ · |Z_s| sin(arg Z_s − θₙ) / 2π   (accrocher)

avec sₙ la saillance du filtre courant (déjà : `compute_perception_deficit` +
`_echelle_attention`, ramenée dans [0, 1]), ou un contexte spatial Iₙ quand il
existera ; garde = 1 tant que la cloche de dispersion et σ_Rloc sont sains,
décroissante sinon (les deux voyants existent) ; K₀ ∈ [2, 3], κ ∈ [0.5, 1] ;
pas de facteur (1 − Rloc_n) sur la force (le mettre, si on y tient, sur
l'allocation de sₙ). À logger : |Z_s| (l'engagement : la cohérence locale
suit-elle la saillance) et le nombre de pas où garde < 1 (la rareté des
interventions, l'indice de flow du cahier). Deux gestes seulement : lier (K₀ >
0) ; relâcher = K₀ < 0 sur le champ moyen (diverger), à tester avant d'y croire.

**Figure** : `docs/figures/kymo_context_binding.png` (kymographe Rloc(t, n) avec
le projecteur, cohérence sous / hors projecteur, gardiens en × du repos ;
script `calib/plot_context_binding.py` sur le `kymo.npz` d'un run).

**4. Et sur un déficit ? Premier test, négatif.** Saillance = déficit
(`saliency=deficit` : les 8 strates à plus forte erreur |Eₙ − Oₙ| lissée), liage
champ moyen K₀ = 2, κ = 1, contre un témoin sans liage (même seed, même
sélection) :

| t | erreur moyenne (témoin / lié) | erreur des strates liées (témoin / lié) | autres |
|---|---|---|---|
| [60, 80) | 0.0141 / 0.0142 | 0.0160 / 0.0166 | 0.0141 / 0.0141 |
| [100, 120) | 0.0141 / 0.0140 | 0.0162 / 0.0162 | 0.0139 / 0.0139 |
| [120, 140) | 0.0141 / 0.0140 | 0.0162 / 0.0162 | 0.0140 / 0.0138 |

Lier les strates en erreur ne réduit pas leur erreur (à 0.0002 près, dans les
deux sens), et l'activité monte à ×1.11 (score 4) parce que l'ensemble lié
change de composition d'un pas à l'autre (la saillance-déficit scintille) et
que chaque changement est un saut de fréquence. Lecture : l'erreur |E − O| est
une quantité d'enveloppe (amplitude, cible Eₙ), la cohérence est une quantité
de phase ; aligner les phases ne rapproche pas O de E. Le geste « lier » est
donc non-invasif et pilotable, mais le bénéfice sur un déficit n'est pas
démontré, et pour l'erreur il n'est probablement pas là : la table de valence
du cahier (erreur → lier) est à revoir. Deux pistes honnêtes : (i) mesurer le
bénéfice là où la cohérence compte par construction (une communication entre
strates via leur phase, qui n'existe pas encore dans la dynamique : c'est
peut-être le chaînon manquant, la « communication-through-coherence » de Fries
suppose que quelque chose passe par la cohérence) ; (ii) lisser la saillance
(≥ 5 u.t.) avant de la donner au liage, sinon l'activité paie le scintillement.

**5. Le geste du jumeau sur l'erreur, rejoué, et ce que l'erreur est vraiment.**
Geste : Δfₙ = K₀·sₙ·(f̄_voisines de couplage − fₙ) (`coupling=nnfreq`),
K₀ = 0.5, saillance = 8 pires erreurs lissées, contre témoin (même seed) :

| run | erreur des strates saillantes | erreur toutes | fluidité (score) | activité |
|---|---|---|---|---|
| témoin | 0.0160–0.0163 (les 8 pires, sans geste) | 0.0141 | 0.975 (5) | 265–271 |
| geste, saillance dure (scintille) | 0.0161–0.0164 | 0.0141 | **0.88 (4)** | 264–271 |
| geste, saillance lissée dans le temps (EMA 5 u.t.) + variation de Δf limitée | 0.0152–0.0162 | 0.0141 | **0.975 (5)** | 263–271 |

Le coût de fluidité du cahier (5 → 4) est reproduit avec la saillance dure, et
**disparaît** dès que la saillance est continue dans le temps (une strate entre
et sort de l'attention en 5 u.t., la fréquence ne saute pas) : c'est la
recette pour tout geste futur, pas seulement celui-ci. Le bénéfice sur l'erreur,
lui, n'est pas reproduit (à 0.0002 près). Pourquoi aucun geste de fréquence ne peut agir
sur cette erreur : Eₙ = (1−λ)Eₙ + λ·φ·Oₙ est un passe-bas de la propre sortie
de la strate, τ = 1/λ_E = 10 pas = 1 u.t., coupure ≈ 0.16 cycle/u.t., alors
que fₙ ∈ [0.41, 6.3]. La cible ne suit AUCUNE strate : |E| ≈ 0.07·|O|, et
l'erreur vaut |O| (rapport médian 1.04, corrélation 0.988 ; corrélation avec
fₙ : −0.07). **L'erreur de régulation, telle que définie, est l'amplitude de
la sortie** : les strates « en déficit d'erreur » sont les plus fortes, pas
celles qui décrochent. Ralentir ou lier une strate ne change rien tant qu'on
ne la ralentit pas sous 0.16 cycle/u.t., c'est-à-dire tant qu'on ne l'éteint
pas. Conséquences : (i) la valence « erreur → lier » du cahier n'a pas de
prise sur cette erreur-là ; (ii) c'est la cible Eₙ qui mérite un regard : une
cible qui ne peut pas suivre ne peut pas être visée, et la régulation qui la
lit (G, γ) régule en fait l'amplitude ; (iii) le ×0.6 mesuré par le jumeau
n'est pas reproduit ici avec le geste tel que décrit — à comparer sur le même
script avant d'en tirer quoi que ce soit.

Ce qui reste ouvert : (a) le geste agit sur le voisinage de l'îlot au sens de
la saillance, pas du couplage w (deux voisins ∓0.1) : si la spirale doit
compter, c'est w qu'il faut enrichir ; (b) « relâcher » n'a pas été testé ; (c)
la saillance réelle (déficit par strate) est bruitée et migre vite, il faudra
un lissage court avant de la donner au liage, sinon l'îlot n'a pas le temps de
s'accrocher (~5 u.t. d'après les runs) ; (d) l'effet sur O(t) est celui d'une
synchronisation partielle locale, visible dans la dispersion : c'est voulu, et
c'est ce que le gardien borne.

### Suite 21/09/2026 (undecies) — mémoire de soi à deux vitesses, surprise, attention d'assimilation

Décision de design (Andréa, 21/09), à l'épreuve du banc (`calib/run_surprise_
attention.py`, hors pipeline) : l'erreur devient une **mémoire de soi à deux
vitesses** ; la **surprise** (écart entre les deux) dit quelles strates se
reconfigurent et lesquelles sont en flow ; l'**attention** se pose sur les
strates en surprise pour les aider à assimiler, puis se retire ; la
métastabilité migrante n'est pas le flow, c'est ce qui le rend récupérable.

**Banc.** Par strate, mémoire courte (τ_s = 2 u.t.) et longue (τ_l = 40 u.t.,
suit la courte) de l'amplitude et de la fréquence ; surprise uₙ = écart relatif
entre les deux. Événements : DURABLE (f₀ × 1.15 sur les strates 40–48 à t = 90)
et BREF (fₙ × 1.15 pendant 2 u.t. sur 70–78 à t = 140). Trois runs : témoin ;
attention-geste (rythme vers les voisines calmes) ; attention-consolidation (la
mémoire longue des strates saillantes apprend plus vite : τ_l / (1 + 3·sₙ)).

**Trois leçons, dans l'ordre.**

1. *Se souvenir de soi par rapport au chœur, pas en absolu.* Sur Aₙ et fₙ
   bruts, la surprise était permanente (0.15–0.35) partout : la latence γ fait
   respirer toutes les fréquences ensemble (± 20 %, période 20 u.t.), la
   mémoire courte suivait la respiration et la longue la moyennait. Sur fₙ/f̄
   et Aₙ/Ā la respiration disparaît. « Toucher au relatif, pas à l'absolu »
   vaut pour la mémoire aussi. Et naître à la mémoire une fois posé (t ≥ 60,
   même repère que l'activité) : avant, tout le chœur est « surpris » de naître.
2. *La mémoire à deux vitesses fait ce qu'on voulait.* Le durable apparaît dans
   la surprise du groupe (0.21 contre 0.12 ailleurs), la mémoire longue le
   rattrape (rapport mémoire/rythme 0.926 → 0.98 en ~40 u.t.), la surprise
   redescend : assimilé. Le bref fait une bosse de 10 u.t. et s'efface sans
   toucher la mémoire longue.
3. *L'attention doit agir sur la mémoire, pas sur le rythme.* Le geste
   « rythme vers les voisines » est inerte (témoin et geste identiques à la
   troisième décimale) : sur une cascade lisse la moyenne des voisines vaut déjà
   la strate, et dans un bloc qui change ensemble l'intérieur n'a que des
   voisines changées. Ce levier n'existe que pour UNE strate isolée qui
   décroche. En revanche l'**attention-consolidation** (porte de plasticité :
   là où elle se pose, la mémoire longue apprend plus vite) :

| | témoin | attention-consolidation |
|---|---|---|
| surprise du groupe durable revenue au niveau des autres | t ≈ 150 | t ≈ 110 |
| assimilation (mémoire longue / rythme) ≥ 0.98 | t ≈ 120–130 | t ≈ 100–110 |
| attention sur le groupe durable | — | 0.57 → 0.48 → 0.31 → 0.20 (retrait spontané) |
| événement bref | bosse 10 u.t., effacé | bosse 10 u.t., effacé, attention 0.2 → 0.1 (pas consolidé) |
| fluidité, dispersion, σ_Rloc, activité | 5 / 0.015 / 0.31 / 240–320 | identiques (la dynamique n'est pas touchée) |

   La surprise de naissance (t = 60–90) est elle aussi absorbée deux fois plus
   vite : le système assimile sa propre naissance. Figure :
   `docs/figures/surprise_attention.png` (`calib/plot_surprise_attention.py`).

**Ce que ça dessine pour le cœur du système** (à décider, rien n'est branché) :

- Eₙ → deux mémoires par strate (courte, longue) de (Aₙ/Ā, fₙ/f̄), nées à
  t ≈ 60 ; l'« erreur » de régulation devient la surprise uₙ (sans dimension) ;
  son niveau global est ce que lisent la latence et la régulation : bas = flow,
  ils se taisent ; haut et durable = reconfiguration, ils s'adaptent.
- L'attention lit la surprise (saillance lissée ≥ 3 u.t.) et, là où elle se
  pose, ouvre la consolidation (τ_l plus court). Elle ne touche pas la
  dynamique : aucun coût de fluidité, de dispersion ni de chimère par
  construction, et elle se retire d'elle-même quand la surprise tombe.
- Le geste sur le rythme (lier / relâcher, liage champ moyen d'îlot, suite
  decies) reste disponible pour les déficits qui vivent dans le rythme, sous
  la boucle d'observation « l'effet a-t-il eu lieu, l'identité a-t-elle tenu ».
  Innovation et dispersion : jamais visées.
- Indice de santé (efficience de la considération) : l'attention voyage peu et
  se retire vite. Beaucoup d'îlots qui durent = un système qui n'assimile plus ;
  aucun jamais = un système que rien ne touche.

**Cohabitation, et le rythme (troisième événement : TREMBLEMENT).** Neuf
strates (20–28) reçoivent un bruit blanc de ± 5 % sur fₙ à chaque pas à partir
de t = 110 : un déficit qui vit dans le rythme (fluidité, activité). Saillance
de rythme = |Δfₙ/fₙ| lissé, au-dessus de 2× la norme du chœur. Quatre runs :

| | témoin | lier (voisines calmes, K = 0.5, pente limitée) | consolidation | les deux |
|---|---|---|---|---|
| secousse relative du groupe / autres | 0.18 / 0.0001 | 0.165 / 0.0001 | 0.18 | 0.175 |
| saillance de rythme sur le groupe | 0.66 → 1.0 | idem | idem | idem |
| surprise du groupe qui tremble | 0.04 (pas élevée) | idem | idem | idem |
| fluidité globale | 0.53 (score 1) | 0.53 (1) | 0.53 (1) | 0.53 (1) |
| assimilation du durable ≥ 0.98 | t ≈ 120–130 | idem | t ≈ 100–110 | t ≈ 100–110 |
| dispersion, σ_Rloc, activité | 5 / 0.31 / 240–320 | idem | idem | idem |

Lecture. (a) Les deux saillances séparent deux natures de trouble : le
tremblement n'est PAS une surprise (les mémoires relatives moyennent un bruit
blanc), et le changement durable n'est pas un tremblement ; la surprise
désigne la reconfiguration, la secousse désigne le déficit de rythme. (b) Les
deux actions cohabitent sans interférence : « les deux » = consolidation +
lier, additifs à la troisième décimale, gardiens intacts. (c) Mais « lier vers
les voisines calmes » ne répare presque pas un tremblement (−7 %) : la pente
limitée qui protège la fluidité empêche de suivre un bruit blanc, et à
l'intérieur du bloc les voisines tremblent aussi. Un tremblement se soigne par
de l'INERTIE, pas par du voisinage → geste « ancrer » (la strate qui tremble
est ramenée vers sa propre mémoire courte, lisse par construction et qui suit
un changement durable, donc ne combat pas l'assimilation) : ANCRER_PLACEHOLDER

Points ouverts : le facteur de consolidation (×3 ici) et les deux τ sont des
réglages, à caler comme le reste ; la saillance par strate demande une
mémoire par strate dans simulate (N × 4 nombres, rien de lourd) ; la
résilience par strate n'a pas encore d'événement propre sur ce banc ; et le
« relâcher » n'a toujours pas été testé.

---

*Ci-dessous : l'état des lieux qui a servi de base au chantier (lignes d'avant).*

---

## 0. La référence : les six métriques du switch et leur scoring

Bloc `simulate.py:597-628`, évalué toutes les `T_switch` unités de temps sur une fenêtre
`W_f` (config `perception.W_f_t`, défaut 5 u.t.), sur le signal **brut O(t) = ΣOₙ** :

| Clé switch | Valeur brute (fenêtre `W_f`) | Ligne | Barème `SCORE_BRACKETS` |
|---|---|---|---|
| `stabilite` | `std(ΣOₙ)` | 604 | `dispersion` |
| `fluidite` | jerk de la moyenne de fₙ : `1/(1+std(d²f)/std(d¹f))` | 607-613 | `fluidity` |
| `innovation` | `compute_entropy_S(ΣOₙ, 1/dt)` | 614-615 | `innovation` |
| `erreur` | `mean(|E−O|)` par pas, moyenné | 616-617 | `regulation` |
| `effort` | `mean(effort_history)` (issu de `compute_effort`) | 618-619 | `activite` |
| `resilience` | `mean(adaptive_resilience)` (colonne loggée, None ignorés) | 623-627 | `resilience` |

**Mode de scoring de référence** : `metrics.score_from_brackets(valeur, clé)` sur la table
unique `metrics.SCORE_BRACKETS` (l.1093), **une seule fenêtre `W_f`**, pas de
multi-fenêtres, pas de pondération, pas de facteur de maturité, pas de 7ᵉ score CPU.

Ce qui reste en amont et n'est **pas** un doublon : `compute_effort` (alimente
`effort_history`), `compute_entropy_S` (appelée par le switch), la production de la
colonne `adaptive_resilience` (voir §2.6 pour le choix qui reste ouvert), et les déficits
par strate `dynamics.compute_perception_deficit` (le *remède* du filtre, même formule par
strate, sauf la fluidité : voir §2.2).

---

## 1. Les scoreurs concurrents (mode de calcul ≠ switch)

Tous produisent des scores 1-5 pour les mêmes critères, mais avec un autre signal, une
autre fenêtre ou d'autres seuils. **À remplacer par le mode du switch.**

| # | Où | Ce qu'il fait de différent | Qui le consomme (à rebrancher) |
|---|---|---|---|
| 1 | `metrics.compute_scores` (1124) + `calculate_all_scores` (1260) + `weighted_average` (1213) + `compute_adaptive_window` (1052) | Scores sur **S(t)** (pas O), 4 fenêtres pondérées (`adaptive_windows.scoring`), facteur de maturité `T/dt`, 7 clés dont `cpu_cost` | `dynamics.gamma_adaptive_aware` (773, 804 ; moyenne des scores hors `cpu_cost`, donc déjà les 6), `main.calculate_empirical_scores` (891), `main.py:537,567` → `visualize.plot_scores_evolution` (1016, fenêtre glissante 50, 7 sous-plots) |
| 2 | `visualize.calculate_empirical_scores_notebook` (554-613) | Bons barèmes (`_sfb`) mais sur **S(t)**, fenêtre « derniers 20 % », 7 clés dont CPU. Commentaire l.586-589 l'assume comme DUPLICAT | grille empirique des figures |
| 3 | `notebooks/visualize.py:552` (même nom) | **Seuils en dur, anciens** (0.5/0.7/1.0/1.3 pour la stabilité, 0.9/0.7/0.5/0.3 pour la fluidité sigmoïde…), n'importe pas `metrics` | notebook |
| 4 | `visualize.compute_signal_quality_scores` (672-730) | Seuils en dur anciens, fluidité **sigmoïde**, entropie recalculée sur fenêtre 20 % ; sert à comparer S vs O | `plot_signal_scores_S_vs_O` (774). Comparaison S vs O devient sans objet : le switch note déjà O |
| 5 | `visualize.build_O_based_history` (732-772) | Recalcule `variance_d2S` / `compute_fluidity` / `entropy_S` sur O pour rejouer les scoreurs S | `main.py:567` (shadow_O → `plot_scores_evolution`) |
| 6 | `main.calculate_empirical_scores` fallback (910-1040) | Seuils en dur anciens pour les 7 critères, fluidité sigmoïde réécrite inline (939-942), résilience par t_retour/continue avec ses propres seuils (956-1002) | rapport final quand l'historique < 20 pas ou en cas d'exception |
| 7 | `simulate.py:252` `_rb` + `897-905` | Score de résilience inline, seuils depuis `config.resilience_v2.score_brackets` : **deuxième source de barème** pour la même clé que `SCORE_BRACKETS['resilience']` | colonne `adaptive_resilience_score` (1268) → `main.py:956` |
| 8 | `metrics.compute_adaptive_resilience` (739-749, 773-783) | Renvoie son propre `score` 1-5 avec seuils internes (continue 0.90/0.75/0.60/0.40 ; t_retour 2.5/4/7/11) | `simulate.py:925` (`adaptive_resilience_score` sur le chemin typé) |
| 9 | `compare_modes.py:62-78, 81-105` | `stability = 1/std_S`, `resilience = 1/(t_retour+1)` : ses propres échelles | comparaison FPS / Kuramoto / neutre |
| 10 | nb `cell31:543-758` (`compute_scores`, `weighted_average`, `calculate_all_scores`, `compute_adaptive_window`) | Seuils en dur, **anciennes clés** `stability` / `effort` | nb `cell31:840`, `cell86:1` |

---

## 2. Doublons métrique par métrique (valeurs brutes)

### 2.1 Stabilité (référence : `std(ΣOₙ)` sur `W_f`)

| De trop | Pourquoi |
|---|---|
| `std(S)` dans `compute_scores` (1160) et `calculate_empirical_scores_notebook` (577) | même formule, **autre signal** (S perçu au lieu de O brut) |
| `metrics.compute_max_median_ratio` (467) + `simulate.py:875-879` + colonne `max_median_ratio` | autre quantité (pics/médiane) sous le nom « stability » dans `analyze.py:202`, `main.py:1049`, `visualize.py:1853` |
| `std_S` du résumé de fin de run (`simulate.py:1250`) et `compare_modes.py:62-78` | S-based, échelle 1/std |
| `visualize.compute_signal_quality_scores` : `std(fenêtre 20 %)` (695) | fenêtre et seuils différents |
| nb `cell50` (`compute_max_median_ratio`) | copie |

### 2.2 Fluidité (référence : jerk de fₙ, `simulate.py:607-613`)

| De trop | Pourquoi |
|---|---|
| **Le même jerk réécrit dans le moteur** `simulate.py:779-787` (fenêtre `_fl_win` = 5 u.t. en dur, le switch lit `W_f_t`) | même formule, deux copies du code et deux réglages de fenêtre. Garder **une** fonction, appelée aux deux endroits (ou le switch lit la colonne `fluidity`) |
| `metrics.compute_variance_d2S` (296) + `simulate.py:773` + colonne `variance_d2S` + `final_variance_d2S` | legacy dt-liée, « ne pilote plus la fluidité » (commentaire l.774-775) ; consommée seulement par les scoreurs à retirer (§1.4, 1.5, 1.6), `analyze.py:729,768`, `visualize.py:267-270, 1597, 1845`, `validate_config.py:9,28` |
| `metrics.compute_fluidity` (333) — sigmoïde de `variance_d2S/175` | ancienne définition ; appelée par `visualize.py:703,764`, `test_fps.py:361-383`, nb `cell62:280` |
| `main.py:939-942` | la même sigmoïde **réécrite inline** |
| `metrics.compute_fluidity_spectral` (356) | morte, formule invalidée par l'audit |
| `dynamics.compute_perception_deficit('fluidite')` (1968-1980) | **formule spectrale** (`1 − part sous ¼ Nyquist`) par strate : c'est la formule invalidée, pas le jerk. Pas supprimable (c'est le remède du filtre), mais **à aligner** sur le jerk de fₙ par strate |
| `kuramoto.py:247-251` | variance de d²S inline (variance brute, sans IQR) |
| `analyze.refine_fluidity` (372) + `to_calibrate.fluidity_threshold` (config 0.035, défaut 0.3 dans `validate_config.py:29`) | seuil sur la colonne `fluidity` avec l'échelle de l'ancienne sigmoïde ; second système de seuils (voir §5) |
| nb `cell44` (`compute_variance_d2S`, `compute_fluidity`) et `cell62:279-280` | copies ; le notebook n'a pas le jerk |

### 2.3 Innovation (référence : `compute_entropy_S(ΣOₙ)` sur `W_f`)

| De trop | Pourquoi |
|---|---|
| `simulate.py:789-795` → colonne `entropy_S` (+ `final_entropy_S`, `entropy_S` moyen l.1245-1253) | même fonction, **sur S** et fenêtre 50 : autre signal, autre fenêtre. Consommée par les scoreurs §1, `analyze.py:214,556,729,770`, `main.py:636,1006,1053`, `visualize.py:257,602,1590,1844`, mode alerte `simulate.py:1182` |
| `kuramoto.py:254-262` | entropie spectrale réécrite inline |
| `visualize.py:704, 767` | recalculs pour les scoreurs S vs O |
| nb `cell46` | copie avec buffer persistant de 5 valeurs (diverge du module) |

### 2.4 Régulation / erreur (référence : `mean(|E−O|)` sur `W_f`)

| De trop | Pourquoi |
|---|---|
| Le switch **recalcule** l'erreur inline (616-617) alors que `metrics.compute_mean_abs_error` (495) produit déjà la colonne `mean_abs_error` (`simulate.py:836`) | même formule, deux codes. Un seul à garder (le switch peut lire la colonne) |
| `analyze.py:219-224` critère `regulation` | compare `mean_abs_error` à `2 × to_calibrate.mean_high_effort` : seuil d'un autre critère |
| `notebooks/visualize.py`, `main.py:926-934`, `compute_signal_quality_scores` | seuils en dur anciens (0.1/0.5/1.0/1.5 ou 0.1/0.5/1.0) ≠ `SCORE_BRACKETS['regulation']` (0.1/0.3/0.5/1.0) |
| nb `cell48` | copie identique |

### 2.5 Effort / activité (référence : `mean(effort_history)` sur `W_f`)

| De trop ou à décider | Pourquoi |
|---|---|
| `metrics.compute_effort_status` (152) + colonne `effort_status` + seuils `effort_chronique_threshold` / `effort_transitoire_threshold` | **second système de notation** de l'effort (stable / transitoire / chronique) avec ses propres seuils, hors barème. Consommé par `visualize.py:300-303` et `analyze.refine_chronic_effort` / `refine_transient_effort` (565, 591) |
| `metrics.compute_mean_high_effort` (218), `compute_d_effort_dt` (261) + colonnes + seuils `to_calibrate.mean_high_effort`, `d_effort_dt` | dérivés de l'effort avec leurs propres seuils (`analyze.py:303-338`) ; pas des scores sur 5 |
| `main.py:1028-1037` | seuils en dur anciens (0.5/1.0/2.0) ≠ `activite` (30/45/75/150) |
| nb `cell42`, `cell54` | copies (`effort_status` diverge : seuils 1.5/2.0 contre 75/150) |

### 2.6 Résilience (référence : `mean(adaptive_resilience)` sur `W_f`, barème `resilience`)

| De trop | Pourquoi |
|---|---|
| `simulate.py:252, 897-905` (`_rb`) | score inline, barème doublon en config (§1.7) |
| `metrics.compute_adaptive_resilience` → champ `score` (739-749, 773-783) | troisième calcul du score (§1.8) |
| `main.py:956-1002` | quatrième, seuils en dur |
| `kuramoto.py:265-276` (t_retour inline) et `279-290` (fallback inline de `compute_continuous_resilience`) | réimplémentations pour la comparaison Kuramoto |
| `compare_modes.py:81-105` | `1/(t_retour+1)` |
| nb `cell52` (`compute_t_retour`, `compute_continuous_resilience`, `compute_adaptive_resilience`) | copies **anciennes** (t_retour sur |S| instantané, `1.0` au lieu de `None`, brackets 1/2/5/10) |
| `to_calibrate.resilience`, `to_calibrate.t_retour` + `analyze.refine_resilience` (456) | seuils parallèles sur `t_retour` |

**Reste ouvert (pas un doublon, une décision)** : la colonne `adaptive_resilience` est
produite par deux systèmes selon `resilience_v2.mode` : enveloppes
(`init_/update_resilience_envelope`, `simulate.py:888-905`) ou chemin typé
(`compute_adaptive_resilience` → `compute_t_retour` / `compute_continuous_resilience`,
`simulate.py:907-925`). Le switch consomme la colonne quelle que soit sa source. Choisir
lequel des deux est LA résilience (cf. `CARTE_metriques.md` §4.4) décide du sort de
`compute_t_retour`, `compute_continuous_resilience` et `compute_adaptive_resilience`.

### 2.7 CPU : hors des six

Pas un doublon, mais un **7ᵉ critère** absent du switch : score `cpu_cost` dans
`compute_scores` (1199), `calculate_empirical_scores_notebook` (607-609), `main.py`
(903, 1017-1025), `plot_scores_evolution` (7ᵉ sous-plot), `analyze.py:225-230`
(`cpu_step_ctrl`), `refine_cpu` (546), `compute_correlation_effort_cpu` (942 + wrapper
`analyze.py:698`). `gamma_adaptive_aware` l'exclut déjà de sa moyenne. Amont :
`compute_cpu_step_deterministic` (alimente `cpu_step(t)`), `compute_cpu_step` (temps mur,
profilage). À décider : garder comme diagnostic loggé, sans score.

---

## 3. Code mort (supprimable sans rien rebrancher)

`compute_fluidity_spectral` (356), `detect_chaos_events` (900), `compute_decorrelation_time`
(1559), `summarize_metrics` (866, seul appel commenté `simulate.py:1277-1280`),
`log_metrics` (830, jamais appelée), `check_thresholds` (789, auto-test seulement), et
l'auto-test `__main__` de `metrics.py` (975-1048) qui exerce ces fonctions.

---

## 4. Le notebook (`NOTEBOOK_FPS.ipynb`)

Les 22 copies (cellules 31, 42-57) sont des doublons par construction : le notebook
n'importe pas `metrics.py`. Neuf divergent déjà (détail `INVENTAIRE_metrics_py.md` §3),
dont tout le bloc scores (anciennes clés) et les trois fonctions de résilience. Sa boucle
`run_fps_simulation` (cellule 62) n'a **ni le jerk, ni le switch de perception**.
Décision à prendre : importer le module, ou accepter que le notebook documente un état
antérieur.

---

## 5. Les systèmes de seuils parallèles à `SCORE_BRACKETS`

Trois autres tables de seuils notent les mêmes grandeurs :

| Table | Où | Clés concernées |
|---|---|---|
| `config.to_calibrate` | `config.json`, défauts `validate_config.py:25-40`, consommée par `analyze.analyze_criteria_statistics` (193-230) et `refine_*` (372-591), `update_config_threshold` (558) | `variance_d2S`, `fluidity_threshold`, `stability_ratio`, `resilience`, `t_retour`, `entropy_S`, `cpu_step_ctrl`, `mean_high_effort`, `d_effort_dt`, `effort_chronique_threshold`, `effort_transitoire_threshold`, `max_chaos_events` |
| `config.resilience_v2.score_brackets` | `simulate.py:252` | résilience (mêmes valeurs par défaut que `SCORE_BRACKETS['resilience']`, mais modifiables séparément) |
| seuils en dur | `main.py` fallback, `notebooks/visualize.py`, `visualize.compute_signal_quality_scores`, `compute_adaptive_resilience`, nb `cell31` | les 7 critères |

Si `SCORE_BRACKETS` est la source unique, `to_calibrate` ne devrait plus contenir que ce
qui n'est pas un score (`env_n`, `gamma_n`, `sigma_n`).

---

## 6. À adapter le jour du retrait

- **Tests** : `test_fps.py` `test_compute_variance_d2S` (347), `test_compute_fluidity`
  (361), `test_compute_adaptive_resilience` (385), `test_compute_entropy_S` (437, sur S),
  `test_compute_effort_status` (451), `test_update_config_threshold` (491), et les tests
  d'`analyze.refine_*` (591).
- **Listes de colonnes** citant les colonnes retirées (`variance_d2S`, `max_median_ratio`,
  `mean_high_effort`, `d_effort_dt`, `std_S`, `effort_status`,
  `adaptive_resilience_score`) : `validate_config.py:9-20`, `visualize.py:1589-1597,
  1844-1853`, `explore.py:857`, `main.py:635-636, 751, 1049-1055`, `analyze.py:556`,
  `config.system.logging.log_metrics`.
- **Figures** : `visualize.py:257-270` (entropy/fluidity/variance_d2S), `300-303`
  (effort_status), `446-465` (corrélations fluidity), `plot_scores_evolution` (7 → 6
  sous-plots), `plot_signal_scores_S_vs_O`, `plot_adaptive_resilience` (2932).

---

## 7. En trois lignes

1. **Dix scoreurs** notent les mêmes critères autrement que le switch ; tous se
   rebranchent sur `score_from_brackets` + fenêtre `W_f` sur O.
2. **Par métrique**, les doublons sont surtout : le jerk copié deux fois, toute la lignée
   `variance_d2S` → `compute_fluidity` (module, `main.py`, `visualize.py`, notebook),
   l'`entropy_S` sur S, `max_median_ratio`, et les quatre scorings de résilience.
3. **Trois tables de seuils** concurrencent `SCORE_BRACKETS` ; le notebook et
   `kuramoto.py` réimplémentent tout en local.
