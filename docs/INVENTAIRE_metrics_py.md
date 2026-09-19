# FPS — Inventaire de `metrics.py` : fonctions, appels et fonctions voisines

*But de ce document : savoir, pour chaque fonction de `metrics.py`, **qui l'appelle**
(fichier:ligne) et **quelles fonctions « ressemblantes »** existent ailleurs (doublons du
notebook, réimplémentations inline, variantes par strate). Il complète
`CARTE_metriques.md`, qui explique le *sens* des métriques ; ici on ne fait que
recenser, sans juger.*

**Vérifié le 2026-09-19** sur la branche `claude/metrics-py-inventory-glxowx`
(alignée sur `main` après la PR #9). Périmètre scanné : tous les `.py` du dépôt
(racine + `notebooks/`) et les cellules de code de `notebooks/NOTEBOOK_FPS.ipynb`
et `EMI_etape1_temoignages.ipynb`. Les numéros de ligne renvoient à cet état du code.

> **Note 2026-09-19** : cet inventaire décrit `metrics.py` AVANT le chantier de mise à
> niveau (voir `DOUBLONS_metriques.md`). Depuis, 13 fonctions ont été supprimées et le
> bloc « six métriques de référence » (`compute_reference_scores` et co.) a été ajouté.

---

## 0. Lecture rapide

- `metrics.py` définit **34 fonctions** (dont 1 privée, `_robust_iqr`, et 1 fermeture
  interne `_steps`).
- **Qui importe `metrics`** : `simulate.py`, `dynamics.py`, `visualize.py`, `test_fps.py`,
  `analyze.py`, `main.py` (import local l.890), `kuramoto.py` (import local l.282).
- **Le notebook n'importe pas `metrics.py`** : `NOTEBOOK_FPS.ipynb` redéfinit ses propres
  copies de 22 fonctions (cellules 31, 42 à 57) et c'est *celles-là* que sa boucle
  `run_fps_simulation` (cellule 62) appelle. 13 sont identiques au module, 9 divergent
  (détail en §3).
- **Fonctions sans aucun appel réel** : `compute_fluidity_spectral`, `detect_chaos_events`,
  `compute_decorrelation_time`, `summarize_metrics` (appel commenté), `log_metrics`
  (jamais appelée ; les 37 occurrences du nom sont la clé de config
  `system.logging.log_metrics`), `check_thresholds` (seulement dans l'auto-test
  `__main__` de `metrics.py`).
- **Fonctions appelées uniquement par les tests** : `compute_cpu_step` hors `simulate`
  (voir tableau), `compute_fluidity` (tests + `visualize.py`), `compute_adaptive_window`
  et `weighted_average` (internes à `calculate_all_scores`).

---

## 1. Les fonctions de `metrics.py` et leurs appels

Convention : « nb » = `notebooks/NOTEBOOK_FPS.ipynb`, `cellN:L` = cellule N, ligne L de la
cellule. Dans le notebook, l'appel vise **la copie locale**, pas `metrics.py` (sauf
mention). « auto-test » = bloc `if __name__ == "__main__"` de `metrics.py` (l.975-1048).

### 1.1 Coût CPU

| Fonction (ligne) | Rôle (docstring) | Appelée par |
|---|---|---|
| `compute_cpu_step` (37) | Temps mur mesuré par pas et par strate, **profilage uniquement** | `simulate.py:582` (→ `wall_time_step`), `test_fps.py:331`, auto-test l.986, nb `cell62:255` (copie locale) |
| `compute_cpu_step_deterministic` (67) | Coût CPU par strate, déterministe et reproductible | `simulate.py:585`, `:1393`, `:1486` (→ `cpu_step`) |

### 1.2 Effort

| Fonction (ligne) | Rôle | Appelée par |
|---|---|---|
| `compute_effort` (95) | Effort d'adaptation interne (churn de paramètres, score `activite`) | `simulate.py:766`, `test_fps.py:341`, auto-test l.994, nb `cell62:268` |
| `compute_effort_status` (152) | Statut stable / transitoire / chronique | `simulate.py:770`, `test_fps.py:455,459,464`, nb `cell62:273` |
| `compute_mean_high_effort` (218) | Moyenne haute de l'effort (percentile 80) | `simulate.py:845`, nb `cell62:299` |
| `compute_d_effort_dt` (261) | Dérivée temporelle de l'effort | `simulate.py:851`, nb `cell62:300` |

### 1.3 Fluidité

| Fonction (ligne) | Rôle | Appelée par |
|---|---|---|
| `compute_variance_d2S` (296) | Variance robuste (IQR) de d²S/dt² | `simulate.py:773` (loggée en diagnostic, **ne pilote plus la fluidité**), `visualize.py:702` (`compute_signal_quality_scores`), `visualize.py:762` (`build_O_based_history`), `test_fps.py:352,356`, auto-test l.1004-1005, nb `cell62:279` |
| `compute_fluidity` (333) | Sigmoïde inversée de `variance_d2S/175` | `visualize.py:703`, `visualize.py:764`, `test_fps.py:364-381`, nb `cell62:280`. **Plus appelée par `simulate.py`** (commentaire l.772 seulement) |
| `compute_fluidity_spectral` (356) | Part de puissance spectrale sous ¼ Nyquist | **aucun appel** (morte) |

### 1.4 Innovation / stabilité / régulation

| Fonction (ligne) | Rôle | Appelée par |
|---|---|---|
| `compute_entropy_S` (392) | Entropie spectrale normalisée de S(t) | `simulate.py:793` (moteur, fenêtre 50), `simulate.py:615` (switch de perception, sur O agrégé), `visualize.py:704`, `visualize.py:767`, `test_fps.py:442,446`, auto-test l.1015-1016, nb `cell62:284` |
| `compute_max_median_ratio` (467) | Ratio max/médiane de S (pics) | `simulate.py:877`, nb `cell62:303` |
| `compute_mean_abs_error` (495) | Erreur absolue moyenne En vs On | `simulate.py:836`, nb `cell62:288` |

### 1.5 Résilience

| Fonction (ligne) | Rôle | Appelée par |
|---|---|---|
| `compute_t_retour` (518) | Temps de retour après choc (settling-time) | `simulate.py:857`, `metrics.py:760` (depuis `compute_adaptive_resilience`), auto-test l.1027, nb `cell52:177` (depuis la copie locale d'`adaptive_resilience`) |
| `compute_continuous_resilience` (565) | Maintien sous perturbation continue | `simulate.py:869`, `kuramoto.py:283` (import local, fallback inline en `except`), `metrics.py:732` (depuis `compute_adaptive_resilience`), nb `cell52:150`, nb `cell62:304` |
| `compute_adaptive_resilience` (656) | Choisit t_retour ou continue selon la perturbation | `simulate.py:919` (branche `elif`, seulement si les enveloppes ne sont pas utilisées), `test_fps.py:411,422,432`, nb `cell62:307` |
| `init_resilience_envelope_state` (1607) | État initial de la résilience par enveloppes (v2) | `simulate.py:231` |
| `_robust_iqr` (1663) | IQR robuste (privée) | `metrics.py:1697,1740,1772` (depuis `update_resilience_envelope`) |
| `update_resilience_envelope` (1668) | Un pas de résilience par enveloppes | `simulate.py:889` |

### 1.6 Scores 1-5 et fenêtres adaptatives

| Fonction (ligne) | Rôle | Appelée par |
|---|---|---|
| `compute_adaptive_window` (1052) | Taille de fenêtre adaptative | `metrics.py:1303` (depuis `calculate_all_scores`), nb `cell31:727` (copie) |
| `score_from_brackets` (1113) | Score 1-5 depuis la table unique `SCORE_BRACKETS` (l.1093) | `metrics.py:1160-1207` (depuis `compute_scores`, 7 appels), `simulate.py:604-627` (switch de perception, 6 appels), `visualize.py:556` (importée sous l'alias `_sfb` dans `calculate_empirical_scores_notebook`) |
| `compute_scores` (1124) | Scores pour une tranche d'historique | `metrics.py:1314,1337` (depuis `calculate_all_scores`), nb `cell31:738,752` (copie) |
| `weighted_average` (1213) | Moyenne pondérée des scores par fenêtre | `metrics.py:1340`, nb `cell31:755` (copie) |
| `calculate_all_scores` (1260) | Scores normalisés multi-fenêtres | `main.py:891` (`calculate_empirical_scores`), `main.py:537,567` (passée en callable à `plot_scores_evolution`), `dynamics.py:773,804`, `visualize.py:1016` (via le paramètre callable), nb `cell31:840`, nb `cell86:1` (copie, passée à `visualize.plot_scores_evolution`) |

### 1.7 Temps caractéristiques / décorrélation

| Fonction (ligne) | Rôle | Appelée par |
|---|---|---|
| `compute_tau_parameter` (1345) | Temps caractéristique d'un paramètre loggé | `simulate.py:815,816` (`An_mean(t)`, `fn_mean(t)`), nb `cell62:346,347` |
| `extract_decorrelation_metrics` (1396) | Dérive (decorrelation_time, autocorr_tau) | `simulate.py:827`, nb `cell62:351` |
| `compute_temporal_coherence` (1413) | Prévisibilité temporelle de S (autocorrélation) | `simulate.py:825`, nb `cell62:322` |
| `compute_autocorr_tau` (1501) | Temps de décorrélation d'un signal | `metrics.py:1561` (depuis `compute_decorrelation_time`), `metrics.py:1589` (depuis `compute_multiple_tau`), nb `cell56:122` |
| `compute_decorrelation_time` (1559) | Alias de `compute_autocorr_tau` | **aucun appel** |
| `compute_multiple_tau` (1565) | Tau pour plusieurs signaux | `simulate.py:807`, docstring l.1581, nb `cell62:326` |

### 1.8 Utilitaires / analyse

| Fonction (ligne) | Rôle | Appelée par |
|---|---|---|
| `check_thresholds` (789) | Franchissement de seuils | auto-test l.1043 **uniquement** |
| `log_metrics` (830) | Export CSV/HDF5 des métriques | **jamais appelée**. `init.py`, `simulate.py`, `kuramoto.py` écrivent le CSV eux-mêmes en lisant la clé de config `log_metrics` |
| `summarize_metrics` (866) | Résumé statistique d'un run | **appel commenté** `simulate.py:1277-1280` |
| `detect_chaos_events` (900) | Événements chaotiques (> 3σ) | **aucun appel** |
| `compute_correlation_effort_cpu` (942) | Corrélation effort / CPU | `analyze.py:706` (depuis le wrapper homonyme), `analyze.py:797` (auto-test d'`analyze`) |

---

## 2. Fonctions « ressemblantes » hors `metrics.py`

Par concept. Une ligne = une implémentation distincte trouvée dans le code.

### 2.1 Fluidité (4 formules vivantes + 1 morte)

| Où | Formule | Signal d'entrée | Utilisée pour |
|---|---|---|---|
| `simulate.py:779-787` (moteur) | **jerk** : `1/(1+std(d²f)/std(d¹f))` sur fenêtre `max(10, 5/dt)` | moyenne de fₙ par pas (`fn_history`) | la colonne `fluidity` loggée → score `fluidity` |
| `simulate.py:607-613` (switch de perception) | même jerk, fenêtre `W_f` | idem | score `fluidite` du switch |
| `metrics.compute_fluidity` (333) | sigmoïde de `variance_d2S/175` | S(t) ou O(t) | `visualize.py` (comparaison S vs O), notebook, tests |
| `main.py:939-942` | **réimplémentation inline** de la sigmoïde (`1/(1+exp(5(x-1)))`) en fallback si `final_fluidity` absent | `final_variance_d2S` | `calculate_empirical_scores` (rapport) |
| `dynamics.compute_perception_deficit('fluidite')` (1935, 1978-1980) | `1 − part spectrale sous ¼ Nyquist`, **par strate** | Oₙ fenêtré | déficit par strate pour le gabarit d'attention. C'est la formule de `compute_fluidity_spectral`, réécrite inline |
| `metrics.compute_fluidity_spectral` (356) | part spectrale sous ¼ Nyquist | — | **morte** |

### 2.2 Entropie spectrale / innovation

| Où | Formule | Note |
|---|---|---|
| `metrics.compute_entropy_S` (392) | périodogramme → Shannon normalisé | référence |
| nb `cell46` | même chose **+ buffer persistant** de 5 valeurs (`compute_entropy_S._buffer`) pour les appels scalaires | diverge du module (§3) |
| `kuramoto.py:255-262` | périodogramme → Shannon normalisé, **inline** | comparaison Kuramoto, sans passer par `metrics` |
| `dynamics.compute_perception_deficit('innovation')` (1981-1983) | `1 − entropie` par strate, via FFT `rfft` | par strate |

### 2.3 Variance de d²S/dt²

| Où | Formule |
|---|---|
| `metrics.compute_variance_d2S` (296) | `np.gradient` ×2 puis variance **robuste IQR** |
| `kuramoto.py:247-251` | `np.gradient` ×2 puis `np.var` **brut** (inline) |

### 2.4 Résilience / t_retour

| Où | Ce que ça fait |
|---|---|
| `metrics.compute_t_retour` (518) | settling-time de l'enveloppe (corrigé) |
| nb `cell52:1` | **ancienne** version (retour de `|S|` instantané, tolérance `(1−seuil)·état`) |
| `kuramoto.py:265-276` | t_retour inline sur `order_params` (retour à ±5 % de la moyenne pré-choc sur 50 pas) |
| `kuramoto.py:286-290` | fallback inline de `compute_continuous_resilience` : `min(1, C_mean/(1+C_std))` |
| `simulate.py:897-905` | brackets de score de résilience inline (`_rb`), pour la branche enveloppes |
| `dynamics.compute_perception_deficit('resilience')` (1958-1967) | écart à la médiane / IQR local, par strate |
| `visualize.plot_adaptive_resilience` (2932), `notebooks/visualize.py:2530` | tracé, pas de calcul |
| `analyze.refine_resilience` (456) | raffinement de config, pas de calcul |

### 2.5 Effort

| Où | Ce que ça fait |
|---|---|
| `metrics.compute_effort` (95) et dérivés (`_status`, `mean_high_`, `d_effort_dt`) | référence |
| `dynamics.compute_perception_deficit('effort')` (1954-1957) | churn relatif `|ΔA|/|A| + |Δf|/|f|` par strate |
| `analyze.analyze_effort_criteria` (295), `refine_chronic_effort` (565), `refine_transient_effort` (591) | analyse et raffinement de config, pas de calcul de la métrique |
| `dynamics.py:1361` `compute_phi_adaptive`, nb `cell33:17` | consomme `effort_history`, pas une métrique |

### 2.6 Stabilité / dispersion

| Où | Quantité |
|---|---|
| `metrics.compute_scores` (1160) | `std(S)` → score `dispersion` |
| `simulate.py:604` | `std(O agrégé)` → score `stabilite` du switch |
| `visualize.compute_signal_quality_scores` (695-716) | `std(fenêtre)` avec **brackets inline** (0.5/0.7/1.0/1.3), pas `SCORE_BRACKETS` |
| `dynamics.compute_perception_deficit('stabilite')` (1952-1953) | `std(Oₙ)` par strate |
| `dynamics.py:789,826` | `stability = 1/(1+std(perfs))` (autre notion, gamma) |
| `metrics.compute_max_median_ratio` (467) | ratio pics/médiane, utilisé comme critère `stability` dans `analyze.py` |

### 2.7 Scores 1-5 (les scoreurs parallèles)

| Où | Source des barèmes | Note |
|---|---|---|
| `metrics.compute_scores` / `calculate_all_scores` | `SCORE_BRACKETS` | référence |
| `visualize.calculate_empirical_scores_notebook` (554) | `SCORE_BRACKETS` via `_sfb` | duplicat assumé (commentaire l.586-589 « DETTE CONNUE ») |
| `notebooks/visualize.py:552` (même nom) | **brackets inline**, n'importe pas `metrics` | version notebook, désynchronisée |
| `visualize.compute_signal_quality_scores` (672) | **brackets inline** | S vs O |
| `main.calculate_empirical_scores` (872) | délègue à `calculate_all_scores` si historique ≥ 20, sinon **brackets inline** (l.905 et suivantes) | fallback |
| `simulate.py:604-628` (switch de perception) | `score_from_brackets` | 6 scores recalculés sur O |
| nb `cell31:553-758` | brackets inline (**anciens noms** `stability`/`effort`) | voir §3 |

### 2.8 Temps caractéristiques

| Où | Note |
|---|---|
| nb `cell56`, `cell57` | copies de `compute_temporal_coherence`, `compute_autocorr_tau`, `compute_multiple_tau`, `compute_tau_parameter`, `extract_decorrelation_metrics` |
| `perturbations.py:379` | `np.correlate` S vs perturbation (corrélation croisée, autre usage) |
| `explore.py:398` | `np.gradient` de phase (autre usage) |

### 2.9 CPU et corrélation

| Où | Note |
|---|---|
| nb `cell42:1` | copie identique de `compute_cpu_step` |
| `analyze.compute_correlation_effort_cpu` (698) | wrapper : appelle `metrics.` puis recalcule inline en `except` (doublon déjà noté dans `CARTE_metriques.md` §5) |

---

## 3. Doublons du notebook : identiques ou divergents ?

Comparaison AST de chaque copie de `NOTEBOOK_FPS.ipynb` avec `metrics.py`, en ignorant
annotations de type, docstrings et commentaires.

| Cellule | Fonction | Verdict | Nature de l'écart |
|---|---|---|---|
| 42 | `compute_cpu_step` | identique | — |
| 42 | `compute_effort` | identique | — |
| 42 | `compute_effort_status` | **diverge** | seuils par défaut : nb `1.5 / 2.0`, module `75.0 / 150.0` |
| 44 | `compute_variance_d2S` | identique | — |
| 44 | `compute_fluidity` | identique | — |
| 46 | `compute_entropy_S` | **diverge** | nb garde un buffer persistant de 5 valeurs pour les appels scalaires ; seuil magnitude `0.001` vs `0.1` |
| 48 | `compute_mean_abs_error` | identique | — |
| 50 | `compute_max_median_ratio` | identique | — |
| 52 | `compute_t_retour` | **diverge** | nb = ancienne version (`|S|` instantané) ; module = settling-time d'enveloppe |
| 52 | `compute_continuous_resilience` | **diverge** | nb renvoie `1.0` sans perturbation ou < 20 points ; module renvoie `None` (verdict suspendu) |
| 52 | `compute_adaptive_resilience` | **diverge** | `dt` par défaut (`0.05` vs lu dans la config) ; brackets t_retour `1/2/5/10` vs `2.5/4/7/11` |
| 54 | `compute_mean_high_effort` | identique | — |
| 54 | `compute_d_effort_dt` | identique | — |
| 56 | `compute_temporal_coherence` | identique | — |
| 56 | `compute_autocorr_tau` | identique | — |
| 56 | `compute_multiple_tau` | identique | — |
| 57 | `compute_tau_parameter` | identique (à un `as e` près) | — |
| 57 | `extract_decorrelation_metrics` | identique | — |
| 31 | `compute_adaptive_window` | **diverge** | même logique, `min_absolute` sans défaut côté nb |
| 31 | `compute_scores` | **diverge** | nb : brackets inline, clés `stability` / `effort` ; module : `score_from_brackets`, clés `dispersion` / `activite`, fluidité neutre `0.15` au lieu de `0.5` |
| 31 | `weighted_average` | **diverge** | mêmes anciennes clés ; le module itère sur une liste fixe de 7 métriques |
| 31 | `calculate_all_scores` | **diverge** | anciennes clés ; maturité calculée sur `T/dt` dans le module, sur `total_steps` dans le nb |

Le notebook n'a pas de copie de : `compute_cpu_step_deterministic`, `compute_fluidity_spectral`,
`check_thresholds`, `log_metrics`, `summarize_metrics`, `detect_chaos_events`,
`compute_correlation_effort_cpu`, `score_from_brackets`, `compute_decorrelation_time`,
`init_/update_resilience_envelope`, `_robust_iqr`. Il n'a pas non plus la fluidité jerk :
sa colonne `fluidity` vient toujours de `compute_fluidity(variance_d2S)` (`cell62:280`).

---

## 4. Ce que ça donne, en trois lignes

1. **Le moteur (`simulate.py`) appelle 20 fonctions de `metrics.py`** ; la fluidité est la
   seule métrique majeure calculée *hors* module (jerk inline, deux fois).
2. **Six fonctions du module ne servent à rien aujourd'hui** (`compute_fluidity_spectral`,
   `detect_chaos_events`, `compute_decorrelation_time`, `summarize_metrics`, `log_metrics`,
   `check_thresholds` hors auto-test).
3. **Le notebook vit sa vie** : 22 copies, 9 divergentes, dont les trois de résilience et
   tout le bloc scores (anciens noms `stability` / `effort`).
