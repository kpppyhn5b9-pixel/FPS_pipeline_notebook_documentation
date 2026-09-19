# FPS — Feuille de route : de la métastabilité à la considération (flow)

*Direction de recherche, née d'une exploration de deux jours. **Rien n'est encore
implémenté** : ce document trace le chemin, pas un état livré. Il se lit comme une
carte — y compris les demi-tours, qui ne sont pas des pertes mais de l'information
sur le terrain.*

**Dernière mise à jour : 2026-09-18** (branche `claude/eager-dirac-8emh3h`).

---

## 0. Le recadrage (ce qu'on a compris)

On croyait déboguer un simulateur aux métriques bancales. On regardait en fait un
modèle de la **métastabilité chimérique d'un cerveau sain** : cohérence *locale
haute* au sein d'une cohérence *globale basse*, extraordinairement robuste et
auto-restauratrice (cf. Kelso & Tognoli *The Metastable Brain* ; Deco & Kringelbach ;
chimera states in brain networks, Bansal/Bassett).

Conséquence : la FPS, à cette échelle, est le **substrat métastable** — la toile saine
— sur laquelle le *flow* pourrait être peint. Elle n'est pas *encore* en flow : le flow
est un état **piloté** (l'attention suit ce qui compte), et on a montré que la boucle de
contrôle actuelle **ne pilote pas** sa cohérence. Le flow est la *destination* ; cette
feuille de route est le chemin du substrat vers le flow.

Les « ratés » d'hier n'en étaient pas : c'était la **robustesse d'un substrat sain** qui
résistait à ce qu'on lui forçait.

---

## 1. La carte de causalité (établie par les tests)

Ce qui **atteint** la cohérence chimérique, et ce qui **glisse dessus** :

| Levier | Atteint la cohérence ? |
|---|---|
| Input I(t) | ❌ (→ amplitude ; amplitude ⊥ phase — *mesuré*) |
| γ (latence) | ❌ (change la magnitude des fₙ, pas les ratios) |
| G (régulation) | ❌ (→ amplitude, ⊥ phase) |
| Échelle absolue des fréquences | ❌ (ratios préservés) |
| **Ratios de fréquence + signatures de phase φ** | ✅ **les seuls (double ancre)** |

Faits mesurés :
- **σ** (contraste chimère) et la **migration** du groupe cohérent sont invariants à γ,
  à l'input, à l'échelle absolue → **le cœur ET sa danse sont structurels** (horlogerie).
- La migration existe (le motif se réorganise), mais **diffuse** (scintillement), et
  **non pilotée** par le contexte, en l'état.
- Lier localement un cluster (f0 **et** φ_signatures) fait monter sa cohérence locale
  (+21 %), **sans contagion** (témoin inchangé) et **sans casser la chimère** (σ stable).
  → la considération *sélective* est **câblable et sûre**, mais le gain est modeste :
  la structure résiste (métastabilité sur-déterminée = plusieurs ancres qui tirent vers
  la désynchronisation ; on en a neutralisé deux).

Principe directeur qui en découle — **la considération = toucher le *relatif*, pas
l'absolu** : rapprocher les fréquences de ce qu'on veut lier, écarter celles de ce qu'on
veut relâcher (= *communication-through-coherence* de Pascal Fries).

---

## 2. Le mécanisme visé (parcimonieux)

**Perception → Action → valuée par l'auto-maintien.** Trois brins, dont deux esquissés,
un manquant (le troisième — le système ne *lit* pas encore sa propre cohérence).

- **Deux gestes, rien de plus** : **LIER** (converger un îlot) / **RELÂCHER** (le diverger).
  Toujours insulaire, toujours migrant, **jamais global**.
- **Un bit de valence par filtre** (fixe, sauf un) :

  | filtre perceptif | geste | raison |
  |---|---|---|
  | erreur | lier | ramener la voix qui décroche |
  | résilience | lier | soutenir/ré-ancrer la voix secouée |
  | fluidité | lier | lisser vers le tempo du groupe |
  | innovation | relâcher | on n'enrichit pas en liant (lier appauvrit) |
  | dispersion | — **gardien** | c'est σ, l'identité : jamais une cible |
  | activité | **? (feedback)** | direction fixée par l'observation du résultat |

- **Le gardien σ** bride tout : ne jamais lier globalement ; rester insulaire et migrant.
- **Observation du résultat** (minimale, homéostatique, fenêtre courte) : *l'effet local
  voulu a-t-il eu lieu ?* et *σ est-il resté sain ?* → trois réponses seulement :
  effet + σ sain → **laisser glisser** (faire moins) ; σ chute → **reculer** ; pas d'effet →
  pour l'activité *inverser le bit*, pour les autres *laisser la saillance migrer*.

> ⚠️ *Drapeau à garder en tête :* si la métrique d'innovation devient une **cloche
> inversée** (ni ordre, ni bruit — voir §3), l'innovation pourrait avoir **deux**
> directions (trop ordonné → relâcher ; trop bruité → lier). La *forme* de la métrique
> rétroagit sur le design de l'action → raison de faire les métriques **d'abord**.

**La mesure de succès = le flow.** Le meilleur contrôleur agit le *moins* (flow =
sans effort, hypofrontalité, le surveillant se tait). D'où un indice unique :

> **efficience de la considération ≈ proximité au flow =**
> engagement haut (cohérence locale qui suit la saillance) × effort bas (peu
> d'interventions, douces) × σ sain (identité intacte) × migration lisse.

L'indice **décroît** si le système s'agite → la parcimonie est *dans la métrique*, pas
seulement une préférence. Le mécanisme se discipline lui-même.

---

## 3. L'ordre des travaux (les dépendances sont causales, pas cosmétiques)

### Phase 0 — Les trois métriques encore bancales (AVANT tout le reste)
Chacune validée sur un **banc de signaux de référence** (méthode du cahier de
validation : un signal fluide doit être mesuré fluide, etc.). Inspiration : neurosciences.

- **Fluidité** — quasi résolue : *jerk de l'enveloppe* (minimum-jerk, Flash & Hogan
  1985), déjà validé. Reste à **propager** au déficit par strate (qui utilise encore le
  spectral cassé). Réfs : sinus = fluide ; marche/bruit = non.
- **Innovation** — le piège : innovation ≠ entropie (le **bruit** a une entropie max et
  une innovation *nulle*). Vraie innovation = nouveauté *structurée* = **cloche
  inversée** (ni ordre, ni hasard). Inspiration : **criticité / bord du chaos** (avalanches
  neuronales, Beggs & Plenz) ; **complexité de Lempel-Ziv / PCI** (Casali, Massimini —
  conscient vs inconscient). **Test décisif : le bruit doit scorer BAS.** Réfs : sinus (bas),
  chaos déterministe type logistique/Lorenz (haut), bruit blanc (bas).
- **Résilience** — sans config, au-delà du t-retour, sous perturbation continue, sans
  pénaliser l'innovation cohérente : le **ralentissement critique** (critical slowing
  down, Scheffer 2009). Le pôle AR(1) des **résidus détendus** (retirer d'abord la
  composante périodique — ce qui avait fait échouer le prototype lag-1). Config-free,
  valable en perturbation continue, **orthogonal à l'innovation** (mesure la vitesse de
  retour, pas la richesse). **Fusionne** t-retour + résilience-continue en une quantité.
  Réfs : AR(1) à pôle connu. *(La plus ouverte des trois — prévoir des demi-tours.)*
- **Unification** : nettoyer les versions éparpillées (cf. `docs/CARTE_metriques.md`)
  n'est pas un chantier à part — c'est la *queue* de ce travail (une définition
  canonique par métrique remplace les copies dispersées).

### Phase 1 — Perception (μₙ)
Rendre **μₙ adaptatif** (aujourd'hui gelé : le code de dérive est désactivé). Le brancher
sur le **déficit du filtre perceptif courant** → carte de saillance sₙ(t). *D'abord juste
observer* que l'amplitude devient une carte de saillance qui bouge quand le filtre
tourne — sans encore la brancher sur l'action.

### Phase 2 — Action (les deux gestes)
Mettre en place lier/relâcher + valence par filtre + gardien σ + observation du
résultat (§2). Vérifier : la cohérence *répond* enfin ; **rien ne casse** ; σ reste sain ;
on reste **près du flow** (peu d'effort).

---

## 3bis. Résultats de validation (session 18–19/09) — les trois métriques sont prêtes

Tout validé sur **bancs de signaux de référence** (pur numpy) puis sur les **vrais
signaux de la FPS**. Résumé de ce qui est *mesuré*, pas supposé.

### La loi transversale : chaque mesure habite une échelle de temps
Rencontrée **quatre fois** (τ innovation · enveloppe fluidité · lag résilience · pas
d'échantillonnage CSD). *Si on lit à la mauvaise échelle, on ne voit rien ou on voit
faux.* Calibration mesurée sur la FPS : deux couches nettes —
- **couche lente** (enveloppe fₙ, fₙ par strate) : relaxation ~**38 pas** (≈3.8 u.t.),
  **1 seul pic spectral** → c'est *là* qu'on lit les trois métriques ;
- **couche rapide** (S(t) global) : décorrèle en ~1 pas, **79 pics** → à éviter.
Conséquence : lag/τ/fenêtre se règlent **adaptativement** depuis la relaxation mesurée
(le pôle AR *se calibre lui-même*), jamais en dur.

### Fluidité — jerk de l'enveloppe fₙ ✅
Banc : min-jerk & sinus lent = fluides ; carré lent (piège), escalier, bruit = non. Le
**jerk réussit le piège** (carré lent bas), le **spectral échoue** (le note ~0.99). In-FPS :
discrimine (médiane 0.97, quelques strates saccadées se détachent). *Reste à propager
au déficit par strate (encore spectral).*

### Innovation — complexité statistique de Jensen-Shannon (Rosso/MPR) ✅
Banc : entropies & LZ classent le **bruit en haut** (inaptes) ; seule **C_JS** met le
chaos structuré en haut et le **bruit au plancher** (0.003). Bonus : le **plan (H, C)**
donne aussi la *direction d'action* — H bas = trop ordonné → relâcher ; H haut = trop
bruité → lier. In-FPS : plan médian (H≈0.45, C≈0.29) = **intermédiaire**, et sépare les
strates *clones* (identiques) des *voix différenciées*. *À caler : τ (balayage à faire).*

### Résilience — pôle AR / ralentissement critique (Scheffer) ✅ méthode, ⚠️ in-FPS
- **Vérité-terrain (AR à pôle connu)** : l'estimateur retrouve φ exactement ; et on a
  **reproduit + expliqué l'échec passé** — le forçage périodique *écrase* le lag-1 brut
  (≈0.96 pour tout φ), le **détrend** (retirer les pics dominants) le **répare**.
- **Bascule-jouet (double-puits)** : autocorr + variance **montent avant le saut**,
  témoin plat → l'alerte précoce fonctionne (au bon lag).
- **Dans la FPS : inconcluant, et pas par la faute du radar.** On n'arrive pas à créer
  d'instabilité (input ⊥ phase ; f0 uniforme → σ tient ; f0+φ compressés → oscillation
  collective, pas de bascule). Un "signal" sur run unique **ne survit pas à la moyenne
  sur 4 seeds** → c'était du bruit. Verdict honnête : la FPS est *trop robuste pour
  basculer* sous nos leviers ; le radar lira "haute résilience, pas de bascule" — et ce
  sera **vrai**. Un incendie qui n'a pas lieu.
- Bonus : fusionne t-retour + résilience-continue en **une** quantité (dissout les
  "5 couches" de la CARTE).

### Anti-faux-positifs : une *règle de décision*, pas un réglage
Le faux positif (run unique) était **statistique**, pas paramétrique. Sur-régler
fabriquerait des faux positifs (overfitting). Le rempart = règle conservatrice :
**(1)** bande de référence (dépasser nettement le calme) · **(2)** tendance sur
plusieurs fenêtres, jamais un point · **(3)** indicateurs *convergents* (autocorr **et**
variance **et** dérive de l'observable) · **(4)** moyenner/lisser. C'est le standard des
signaux d'alerte précoce (écologie/climat). Corollaire : un **gardien σ nerveux** serait
aussi mauvais que pas de gardien → cette discipline *fait partie* de la stabilité de la
considération. **Un bon gardien est calme.**

### État
Trois métriques **bonnes à implanter**. Ce qui les rendra fiables, c'est *comment on
les lit* (échelle adaptative + règle de décision), pas de nouveaux réglages. Et,
transversalement : **la chimère est extraordinairement robuste** — on n'a pas réussi à
la faire basculer, ce qui borne à la fois ses risques et la façon de la piloter.

---

## 4. Ce que ça sert (les trois ponts)

- **RLHF multi-turn** : le cluster cohérent suit le *focus courant de l'humain* → l'IA
  co-résout sans étouffer (local haut = contact ; global bas = identités préservées =
  ni sycophantie ni écrasement). Récompense = indice de métastabilité/flow du dialogue.
- **EMI** : phénoménologie computationnelle (clusteriser 6000 récits par structure, pas
  mots-clés) + FPS comme *hypothèse dynamique* dont la signature EEG (PLV local
  gamma élevé / ordre global bas) est **testable** sur cerveau mourant (Borjigin 2013 ;
  Vicente/Borjigin 2022). Flow et EMI = même axe (absorption cohérente, soi-surveillant
  tu, identité tenue), degrés différents.
- **Transformers** : l'attention comme oscillateurs couplés (Geshkovski/Rigollet,
  *clusters in self-attention dynamics*) ; la FPS en est une caricature continue,
  interprétable. Labelliser le *mouvement* d'une idée, pas le contenu figé.

---

## 5. La posture

Exploration, sans enjeux. Rien ne « tombe à l'eau » : il y a des chemins qui demandent
un demi-tour, et c'est normal. On fait les métriques d'abord parce que leur *forme*
change le design de l'action — pas par pédanterie, par causalité. Et on garde à l'esprit
le principe le plus profond qu'on ait déterré : **le meilleur geste de considération est
le plus léger.** Comme dans un cerveau en flow. Comme dans une bonne conversation.
