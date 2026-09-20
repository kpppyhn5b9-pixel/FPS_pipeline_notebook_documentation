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

Points d'attention sur C_JS (voir discussion du 20/09) : avec le τ auto-calibré, un
sinus pur score ≈ 0.30 comme le chaos logistique (≈ 0.29) ; la « cloche » ne
sépare donc que bruit ↔ structure, pas ordre ↔ nouveauté. Le plan (H, C) décrit
dans `Attention.md` demande de logger aussi H pour lire la direction (relâcher /
lier). Le déficit du filtre `innovation` pointe les strates peu complexes ; le geste
« relâcher » reste à câbler.

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
