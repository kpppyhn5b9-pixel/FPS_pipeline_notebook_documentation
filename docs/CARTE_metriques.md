# FPS — Carte des métriques : état des lieux

*But de ce document : ne plus jamais reperdre le fil. Il répertorie ce qui n'allait
pas, ce qui a été réparé, ce qui n'est pas encore final, et les endroits où plusieurs
versions d'une « même » métrique coexistent sans version unifiée.*

*Principe directeur : **regarder plutôt que supposer.** Toutes les lignes ci-dessous
ont été vérifiées dans le code, pas recopiées de mémoire.*

**Dernière vérification : 2026-09-17** (branche `claude/eager-dirac-8emh3h`, alignée sur
`main` après merge de la PR #6).

> **Mise à jour 2026-09-19 (branche `claude/metrics-py-inventory-glxowx`)** : les
> chevauchements du §4 et les doublons du §5 ont été **résorbés**. Six métriques de
> référence, un barème (`metrics.SCORE_BRACKETS`), un scoreur
> (`metrics.compute_reference_scores`), une fenêtre (`metrics.reference_window`).
> γ note S(t) courant, tout le reste note O(t). Détail du chantier et de ce qui
> reste ouvert : `docs/DOUBLONS_metriques.md` (section « Chantier de mise à niveau »).
> Les tableaux ci-dessous décrivent l'état d'avant et restent utiles comme
> historique.

---

## 0. En une phrase

Le système avait un **regard biaisé sur lui-même** : plusieurs métriques ne mesuraient
pas ce que leur nom prétendait. Quatre ont été corrigées et sont sur `main`. Le reste
n'est **pas cassé** — c'est de la *confusion de noms* (plusieurs définitions sous un
même mot, pas de version unifiée partagée entre les mécanismes) + quelques questions de
recherche ouvertes, honnêtement non résolues.

---

## 1. Ce qui n'allait pas (l'audit)

| Problème | Nature | Statut |
|---|---|---|
| **Fluidité spectrale** classait le bruit / les à-coups comme « plus fluides » qu'une sinusoïde propre | métrique invalide (ne mesure pas la fluidité) | ✅ réparé |
| **t_retour** basé sur `|S|` instantané → retour mal estimé après un choc | métrique imprécise | ✅ réparé |
| Score **`stability`** = simple écart-type de S (= amplitude), pas de la stabilité structurelle | nom trompeur | ✅ renommé `dispersion` |
| Score **`effort`** = churn de paramètres, pas du « stress » | nom trompeur | ✅ renommé `activite` |
| **Bug silencieux** : le switch de perception utilisait encore les vieilles clés (`stability`/`effort`) → `KeyError` avalé par un `try/except`, cassé sans que les tests le voient | bug masqué | ✅ corrigé |

> **Leçon gravée : tests verts ≠ marche**, surtout avec un `except` silencieux. Ce bug
> passait 42 tests au vert tout en étant cassé.

---

## 2. Ce qui a été réparé (sur `main`, PR #6)

| Métrique | Avant | Maintenant | Où |
|---|---|---|---|
| **Fluidité** | puissance spectrale grave (cassée) | **jerk de l'enveloppe fₙ** : `1/(1+std(d²)/std(d¹))` sur une fenêtre lente de fₙ. Mesure la douceur du *tempo*. Validée sur banc de signaux de référence. | `simulate.py:782-787` (moteur) et `:607-613` (switch) — **même** formule aux deux endroits |
| **Résilience (t_retour)** | retour de `|S|` instantané | **settling-time** : retour de l'*enveloppe* à sa base pré-choc (fenêtre glissante), brackets recalibrés | `metrics.py:518` (`compute_t_retour`) |
| **`stability` → `dispersion`** | nom = « stabilité » | clé `dispersion`, écart-type de S = **amplitude** (assumé) | `metrics.py:1094,1160` |
| **`effort` → `activite`** | nom = « effort/stress » | clé `activite`, **churn** de paramètres | `metrics.py:1109`, `compute_effort` `:95` |

Consommateurs mis à jour partout (metrics, simulate, main, visualize). §12 (ré-émergence
Chimera-like) revérifiée indépendamment après ces changements : corrélation de profil
reset-vs-uniforme = **1.000**.

---

## 3. Ce qui n'est PAS encore final (questions ouvertes, honnêtes)

Rien ici n'est cassé au sens « ça plante » — ce sont des limites connues et assumées.

| Sujet | État réel | Direction |
|---|---|---|
| **Axe « structure »** | `compute_temporal_coherence` part dans le mauvais sens *in situ* ; la différenciation est plate sauf en catastrophe | vraie question de recherche, non tranchée |
| **Résilience auto-observée** | le prototype par autocorrélation lag-1 a échoué (contaminé par l'autocorrélation du forçage périodique lui-même, 4/7) | piste = pôle AR(1) |
| **Score `dispersion`** | dormant / peu discriminant en pratique | inoffensif ; à réactiver ou retirer un jour |
| **`compute_fluidity_spectral`** (metrics.py:356) | **code mort** : défini, appelé nulle part | supprimable sans risque |
| **`compute_fluidity`** legacy (metrics.py:333, « nombre magique 175 ») | vivant **uniquement** dans `visualize.py` (reconstruction du signal O) + couvert par des tests | à aligner sur le jerk le jour du nettoyage |

---

## 4. Les versions qui se chevauchent (pas de version unifiée)

**C'est fini, ce n'est pas infini : 5 noms.** Le même patron répété : pour chaque
concept il existe souvent (a) une version **SCORE**, (b) une version **PERCEPTION par
strate**, (c) parfois une version **analyse/comparaison**, (d) parfois du **legacy**.
Une fois le patron vu, la carte entière tient dans la tête.

### 4.1 FLUIDITÉ — 4 versions
| où | définition | statut |
|---|---|---|
| **SCORE + switch** (simulate.py, `SCORE_BRACKETS['fluidity']`) | jerk de l'enveloppe fₙ | ✅ le bon, unifié entre moteur et switch |
| `compute_fluidity_spectral` (metrics.py:356) | fraction de puissance spectrale grave | ❌ cassé — **mort**, supprimable |
| `compute_fluidity` (metrics.py:333) | `1/(1+exp(5·(var_d2S/175−1)))` | ⚪ legacy, vivant seulement dans `visualize.py` |
| `compute_perception_deficit('fluidite')` (dynamics.py:1935) | version par strate | à aligner sur le jerk |

### 4.2 STABILITÉ — 3 quantités distinctes sous un nom
| où | définition |
|---|---|
| **SCORE** = `dispersion` | écart-type de S = amplitude |
| `analyze.py` critère `stability` | `max_median_ratio` (pics/médiane) — AUTRE quantité |
| `compare_modes.py` `stability` | sa propre valeur de comparaison |
| `compute_perception_deficit('stabilite')` | std par strate |

→ *3 choses. À renommer distinctement (dispersion / ratio-pics / …) plus tard.*

### 4.3 COHÉRENCE — 3 concepts LÉGITIMEMENT différents, mal distingués par le nom
| nom | ce que c'est |
|---|---|
| `compute_temporal_coherence` (metrics.py:1413) | prévisibilité temporelle de S (autocorrélation) |
| `compute_C` (dynamics.py:1503) = **C(t)** | cohésion de chaîne : `moy cos(φₙ₊₁−φₙ)` — déjà « cohésion » dans le catalogue |
| `kuramoto_global/local` (visualize) / **Rloc** | synchronisation de Kuramoto |

→ *Trois vraies choses qui MÉRITENT trois noms distincts (temporelle / cohésion / synchro).*

### 4.4 RÉSILIENCE — plusieurs couches empilées
| où | rôle |
|---|---|
| `compute_t_retour` (metrics.py:518) | temps de retour (choc) — corrigé (settling-time) |
| `compute_continuous_resilience` (metrics.py:565) | maintien sous perturbation continue |
| `compute_adaptive_resilience` (metrics.py:656) | le **switch** qui choisit entre les deux (config-dépendant) |
| `init_/update_resilience_envelope` | 3ᵉ système : enveloppe de santé (épisodes, gel) |
| SCORE `resilience` | lit `adaptive_resilience` |
| `compute_perception_deficit('resilience')` | version par strate (IQR local) |

→ *Le plus « en couches ». Un jour calme : décider laquelle est LA résilience, ranger les autres.*

### 4.5 EFFORT → ACTIVITÉ
| où | définition |
|---|---|
| **SCORE** = `activite` | churn des paramètres (`compute_effort`, metrics.py:95) |
| champ de DONNÉES `effort(t)` loggé | valeur brute par pas (≠ le score) |
| `compute_perception_deficit('effort')` | version par strate |
| `compute_effort_status`, `mean_high_effort`, `d_effort_dt` | dérivés (statut, chronique, transitoire) |

---

## 5. Doublons de fonctions (à consolider un jour calme)

~5 fonctions dupliquées repérées : `compute_In`, `compute_Gn`,
`compute_correlation_effort_cpu`, `run_kuramoto_simulation`, `load_run_data`.
Non urgent, aucun impact fonctionnel connu.

---

## Le motif (pour respirer)

**Rien ne presse. Presque rien n'est cassé** (le seul résidu invalide,
`compute_fluidity_spectral`, est mort et inoffensif). C'est le **même patron répété 5
fois**. Le jour du nettoyage : pour chaque nom, une seule définition canonique, les
autres renommées ou rangées. Un après-midi calme, pas une montagne.
