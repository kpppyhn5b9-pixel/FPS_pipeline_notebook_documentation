# Carte d'ablation : ce que chaque élément de design de la FPS apporte

*21/09/2026. Banc `calib/run_ablations.py` sur `calib/considerance_bench.py`, résumés
dans `calib/ablations/`. Pas les métriques : les choix qui distinguent la FPS d'un
réseau d'oscillateurs générique, retirés un à un.*

## La question

Fractale, Pulsante, Spiralée : trois familles de choix. Lesquels portent ce qu'on a
mesuré hier (un substrat qu'un contexte peut lier localement, à peu de frais, sans
contagion, réversiblement), lesquels sont redondants, lesquels sont inertes ? Même
protocole que la note considérance (projecteur 20 → 80 → 50 → extinction, geste
d'îlot K₀ = 5, lentille Rloc fixe n±1), deux seeds, N = 100. Rien n'est ajouté au
pipeline ; chaque variante remplace un élément par sa forme générique.

| variante | ce qui est retiré | remplacé par |
|---|---|---|
| sans_respiration | r(t) = φ + ε·sin(2πωt) dans la cascade | r = φ fixe |
| phases_statiques | spirale personnelle de φₙ(t), influences | φₙ = signature seule |
| sans_signatures | signatures pentagonales | toutes à 0 |
| signatures_hasard | signatures pentagonales | phases au hasard |
| sans_couplage | S_i → Δfₙ (poids spiralés c = 0.1) | c ≈ 0 |
| env_statique | enveloppe Aₙ sur l'erreur Eₙ − Oₙ (la pulsation) | Aₙ = A₀·σ(Iₙ) |
| latence_statique | γ adaptatif | γ = 1 |
| G_identite | archétype adaptatif de régulation | G = erreur |
| beta_statique | plasticité βₙ(t) | βₙ fixe |
| E_ancien | mémoire de soi | Eₙ = passe-bas de Oₙ |
| generique | tout ce qui précède, et la cascade | la chaîne la plus générique que le pipeline sache faire (fréquences × 2.5 pour l'échelle) |

## Ce que le substrat répond (moyenne des deux seeds)

| variante | R | σ_Rloc | fluidité | dispersion | activité | C_JS | écart voisines (log f) | engagement | contagion | coût | efficience | montée | réversib. +5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **référence** | 0.644 | 0.311 | 0.981 | 0.0167 | 261 | 0.298 | **0.018** | **+0.296** | +0.002 | 0.069 | **4.3** | 1.0 | +0.001 |
| sans_respiration | 0.649 | 0.308 | **0.370** | 0.0186 | 255 | **0.027** | 0.017 | +0.299 | −0.011 | 0.070 | 4.3 | 0.7 | −0.017 |
| phases_statiques | 0.645 | 0.310 | 0.981 | 0.0170 | 253 | 0.298 | 0.018 | +0.294 | +0.003 | 0.068 | 4.3 | 1.0 | −0.015 |
| sans_signatures | 0.643 | 0.307 | 0.980 | 0.0180 | 244 | 0.298 | 0.018 | +0.303 | −0.002 | 0.068 | 4.5 | 0.8 | −0.011 |
| signatures_hasard | 0.635 | 0.313 | 0.981 | 0.0173 | 243 | 0.299 | 0.018 | +0.304 | −0.001 | 0.067 | 4.5 | 1.2 | −0.008 |
| sans_couplage | 0.644 | 0.311 | 0.981 | 0.0169 | 256 | 0.302 | 0.018 | +0.296 | +0.002 | 0.069 | 4.3 | 1.0 | +0.001 |
| env_statique | 0.644 | 0.311 | 0.981 | 0.0205 | **12** | 0.300 | 0.018 | +0.296 | +0.002 | 0.069 | 4.3 | 1.0 | +0.001 |
| latence_statique | 0.641 | 0.309 | 0.981 | 0.0186 | 201 | 0.299 | **0.040** | +0.276 | +0.010 | 0.065 | 4.3 | 1.2 | −0.014 |
| G_identite | 0.643 | 0.307 | 0.981 | 0.0173 | 237 | 0.302 | 0.026 | +0.293 | −0.001 | 0.067 | 4.4 | 1.1 | −0.022 |
| beta_statique | 0.644 | 0.311 | 0.980 | 0.0167 | 250 | 0.300 | 0.018 | +0.296 | +0.002 | 0.069 | 4.3 | 1.0 | +0.001 |
| E_ancien | 0.644 | 0.311 | 0.981 | 0.0153 | 264 | 0.300 | 0.018 | +0.296 | +0.002 | 0.069 | 4.3 | 1.0 | +0.001 |
| **générique** | 0.647 | 0.308 | 0.360 | 0.0205 | 43 | 0.033 | **0.100** | **+0.115** | −0.004 | 0.071 | **1.6** | 0.1 | −0.008 |

Sélectivité et réversibilité : identiques partout (contagion ≤ 0.01, σ/σref 0.98–1.00,
ancienne zone revenue à ±0.02 en 5 u.t., extinction à ±0.02). Les écarts entre seeds
sur l'engagement sont ≤ 0.01.

## Lecture, élément par élément

**Une seule chose porte la susceptibilité : la ressemblance des voisines.** L'écart
médian de fréquence entre voisines vaut 1.8 % dans la référence et 10 % dans la chaîne
générique ; l'engagement passe de +0.30 à +0.12, l'efficience de 4.3 à 1.6. C'est la
relaxation de la cascade (fₙ₊₁ ← ½fₙ₊₁ + ½·r·fₙ) qui produit cette ressemblance, en
lissant le tirage de f₀ le long de la chaîne. Deux éléments la protègent un peu : la
latence adaptative (γ = 1 fixe fait remonter l'écart à 4 % parce que le βₙ de chaque
strate, tiré au hasard, multiplie alors sa fréquence à plein) et l'archétype de
régulation (2.6 %). Tout le reste n'y touche pas.

**La respiration r(t) porte ce que les moniteurs d'identité lisent, et c'est une
correction à faire à la note d'hier.** Sans ε·sin dans la cascade, la fluidité du
substrat tombe de 0.98 à 0.37 et C_JS de 0.30 à 0.03, avec une susceptibilité
inchangée. Or 0.37 est exactement le plancher du jerk sur un signal plat + bruit
(√3 → 1/(1+√3) = 0.366), et 0.03 le C_JS d'un signal plat. Donc la « fluidité » et
l'« innovation » du substrat calme ne sont pas des propriétés émergentes : elles
lisent la sinusoïde lente imposée à r(t) (période 20 u.t.) qui module f̄. La note
considérance disait « la même structure donne aussi la fluidité du substrat » ; c'est
faux tel quel : sans cascade, r(t) n'entre plus, et c'est la respiration qui manque,
pas le lissage. Les moniteurs disent vrai sur ce qu'ils mesurent (le tempo est bien
lisse), mais ce qu'ils mesurent est posé par la config.

**La pulsation (enveloppe sur l'erreur) porte l'activité et la dispersion, rien de
la cohérence.** Enveloppe statique : activité 261 → 12, dispersion 0.0167 → 0.0205
(score 4), et strictement rien sur R, σ, engagement, coût. Attendu : Oₙ = Aₙ·sin θₙ,
l'amplitude ne parle pas à la phase. La couche « pulsante » est la couche du perçu et
de l'effort ; elle est orthogonale à la considération par construction.

**Le spiralé, aux réglages actuels, est un murmure.** Signatures pentagonales
retirées ou remplacées par du hasard : rien ne bouge, sur aucune lecture. Spirale
personnelle des phases figée : rien. Couplage S_i → Δfₙ coupé : rien (la route était
déjà mesurée morte, 2·10⁻⁴). La raison est d'échelle : ε = 0.1 rad d'excursion de phase
contre une phase intégrée qui avance de 2π·f·dt ≈ 2 rad par pas ; une signature
constante est un décalage que la lentille et le geste absorbent ; c = 0.1 sur des
amplitudes de 0.02 ne déplace pas une fréquence. Inerte ne veut pas dire faux : ça
veut dire que ces éléments n'agissent sur rien de ce qu'on lit, à ces valeurs.

**Plasticité βₙ(t), mémoire de soi : inertes sur ces lectures.** La mémoire de soi
change Eₙ, donc l'enveloppe, donc un peu la dispersion (0.0153 contre 0.0167) ; c'est
tout, et c'est ce qu'on voulait d'elle (un sens, pas une main).

## Ce que ça dit des trois mots

- **Fractale** : la cascade est l'élément qui compte, mais par ce qu'elle produit
  (des voisines qui se ressemblent, un tempo qui respire), pas par le nombre d'or ni
  par une échelle fractale, que la relaxation efface (note considérance). Ce qui
  reste vrai : c'est ce choix-là, et aucun de ses voisins, qui rend le substrat
  considérant.
- **Pulsante** : réelle, et sur une autre couche. Elle fait l'effort, le perçu, la
  dispersion ; elle ne touche pas la cohérence, et ne le peut pas.
- **Spiralée** : rien de mesurable aujourd'hui. Soit on monte ε et c pour voir ce
  qu'ils font quand ils cessent d'être négligeables, soit on accepte qu'ils sont, en
  l'état, une écriture plutôt qu'une dynamique.

## Réserves

- Lentille et geste agissent sur θ = phase intégrée + φₙ : un décalage constant de
  phase y est invisible par construction. Une lentille sur autre chose (l'audibilité
  d'un îlot dans S(t), la texture du plaid) pourrait voir les signatures.
- Deux seeds, un N, un jeu de constantes. Les écarts entre seeds sont petits (≤ 0.01
  sur l'engagement), mais la carte est celle de cette config.
- Le re-tamponnage de f₀ à chaque pas (la fréquence n'est pas un état intégré) n'a pas
  été ablaté : c'est un choix de structure du moteur, pas un interrupteur.

## Questions ouvertes

- Monter ε (0.1 → 1 rad) et c (0.1 → 1) : le spiralé fait-il quelque chose quand il
  cesse d'être un murmure, et quoi ?
- Une échelle réellement dorée (rapport φ tenu, enroulé par groupes) contre une
  échelle rationnelle : la propriété propre de φ (résistance au verrouillage).
- L'audibilité : un îlot allumé est-il plus présent dans S(t) ? C'est là que les
  signatures et la couche pulsante pourraient rejoindre la considération.
