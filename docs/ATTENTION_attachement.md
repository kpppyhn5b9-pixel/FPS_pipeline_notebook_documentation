# L'attachement par association : tenir à un foyer pour une raison qui lui appartient

*24/09/2026. Banc `calib/run_attachement.py` sur `calib/considerance_bench.py` (option
`attach`), figure `docs/figures/attention_attachement.png`, résumés et trajectoires de m
dans `calib/attention_sources/`. Rien n'est ajouté au pipeline.*

## La question

Le banc du maintien (`ATTENTION_maintien_temoin.md`) a montré qu'une priorité m fournie
de l'extérieur ramène le familier au premier plan sans nouveauté et le relâche quand on
la retire. Reste : d'où viendrait m dedans ? Pas de la mémoire de régulation (témoin).
Deux façons de le faire naître ne disent pas la même chose. « Se poser sur ce foyer a
servi » : c'est de la récompense, un servomécanisme de plus, et la musique connue n'y
entre pas, elle ne sert à rien. « Ce foyer était là quand j'allais bien » : une
association, pas une causalité. C'est ça, le familier qui compte : ce qui a accompagné les
bons moments (Frankfurt, *The Importance of What We Care About* : tenir à quelque chose,
c'est endosser ce par quoi on se laisse mouvoir).

## Le mécanisme, et pourquoi il ne peut pas boucler

Pour chaque foyer k présent, m_k suit, comme une moyenne mobile (τ_m = 20 u.t.), le
bien-être du soi pendant sa présence : w = clip(2 − activité/repos, 0, 1), soit 1 quand le
système est à son repos ou en dessous (flow : effort bas), 0 quand il fait le double.
Ce n'est jamais l'effet du foyer qu'on lit, c'est l'état du soi quand il était là. Rien
ne passe par l'amplitude ni par l'audibilité : le système ne peut pas prendre sa propre
voix pour une raison de continuer. Puis, comme au banc du maintien : s = c · [ν + (1 − ν)·m].

## Le protocole

Deux foyers, 20 et 80 (la zone 80 est la plus dure et la moins audible de tous nos
bancs). Apprentissage : l'un présent pendant une période calme (60–120), l'autre pendant
une période d'effort (120–180 : bruit blanc sur Iₙ, écart-type 0.6, activité ≈ ×3.6 du
repos). Extinction (180–220). Retour des deux ensemble, au calme (220–300), aussi
familiers l'un que l'autre. Deux ordres, pour que la zone dure soit tantôt chérie
tantôt non. Deux seeds. m figé à l'extinction pour la lecture (sinon la présence au
calme pendant le retour réécrit m des deux foyers en ~10 u.t., ce qui est correct pour
une disposition révisable, mais brouille la lecture de ce qui a été appris). Témoin :
m appris mais non appliqué.

## Ce que le système apprend

| ordre | seed | bien-être w au calme / sous bruit | activité relative au calme / sous bruit | m appris à 180 : foyer 20 / foyer 80 |
|---|---|---|---|---|
| 1 (20 au calme) | 12345 | 0.75 / 0.06 | ×1.31 / ×3.71 | **0.68** / 0.01 |
| 1 | 7 | 0.78 / 0.09 | ×1.27 / ×3.52 | **0.74** / 0.02 |
| 2 (80 au calme) | 12345 | 0.54 / 0.03 | ×1.57 / ×3.56 | 0.00 / **0.46** |
| 2 | 7 | 0.61 / 0.05 | ×1.46 / ×3.42 | 0.01 / **0.53** |

Le foyer présent au calme est chéri, celui présent sous effort ne l'est pas, dans les
deux ordres. (La zone 80 au calme laisse le système un peu moins à son repos que la 20,
d'où un m un peu plus bas.)

## Ce qui se passe au retour (220–300)

| ordre | seed | m appliqué | zone 20 : s · R · audibilité | zone 80 : s · R · audibilité |
|---|---|---|---|---|
| 1 (chéri = 20) | 12345 | non (témoin) | 0.23 · 0.705 · 51 % | 0.17 · 0.656 · 16 % |
| 1 | 12345 | **oui, figé** | **0.63 · 0.862 · 73 %** | 0.20 · 0.668 · 9 % |
| 1 | 7 | oui, figé | **0.66 · 0.882 · 69 %** | 0.20 · 0.653 · 10 % |
| 2 (chéri = 80) | 12345 | non (témoin) | 0.34 · 0.765 · 58 % | 0.19 · 0.675 · 16 % |
| 2 | 12345 | **oui, figé** | 0.44 · 0.768 · 47 % | **0.47 · 0.816 · 30 %** |
| 2 | 7 | oui, figé | 0.47 · 0.795 · 49 % | **0.51 · 0.892 · 31 %** |

Et dans le temps (ordre 2, seed 7) : à 220–240 les deux s'allument, la 20 par nouveauté
(ν 0.76, elle revient d'une période de bruit) et la 80 par attachement (m 0.53, ν 0.36) ;
puis la nouveauté de la 20 s'use (ν 0.43, R 0.72 à 260–300) tandis que la 80 tient
(R 0.88, s 0.48). σ/σref 0.95–0.99 partout.

## Trois réponses

1. **L'attachement existe, et il vient du dedans.** Sans m appliqué, les deux foyers
   ne se distinguent que par leur zone (la 20 est plus facile et plus audible, quelle
   que soit l'histoire). Avec m appris et figé, le foyer qui a accompagné le calme est
   repris et tenu (R 0.86–0.89, trois quarts de S(t) pour la 20 ; R 0.82–0.89 et
   audibilité doublée, 16 → 30 %, pour la 80), et l'autre s'use par habituation. Dans
   les deux ordres, sur les deux seeds. La raison de tenir n'est pas dans le foyer ni
   dans son effet : elle est dans ce que le système était quand il était là.

2. **L'attachement renverse le penchant naturel.** La zone 80 est la plus dure de tous
   nos bancs ; chérie, elle est tenue à R 0.88 à la fin du retour pendant que la zone
   20, facile mais non chérie, retombe à 0.70. Le système préfère ce qu'il a aimé à ce
   qui lui est facile. C'est la première fois qu'une lecture de la FPS dépend de son
   histoire de façon spécifique (le témoin papillon ne le montrait pas pour la
   régulation).

3. **La disposition est révisable, et c'est une propriété, pas un défaut.** Sans figer,
   un foyer présent pendant le calme du retour devient chéri en ~10 u.t. et rattrape
   l'autre : ce qui accompagne les bons moments maintenant compte aussi. Le passé ne
   fige pas, il donne une avance. Le réglage de cette avance, c'est τ_m.

## Réserves

- Le bien-être est lu sur l'activité seule (effort relatif au repos). Le flow en a
  d'autres traits (fluidité, σ sain) ; l'activité était la plus contrastée sous bruit.
- L'effort est produit par du bruit sur l'entrée, pas par le foyer. C'est voulu (on
  associe, on ne récompense pas). Le cas d'un foyer lui-même épuisant est testé ensuite
  (`ATTENTION_foyer_epuisant.md`) : m baisse bien, par le même mécanisme, et un foyer
  chéri qui devient épuisant perd m aussi vite qu'il l'avait gagné.
- Une observation en passant : la nouveauté au retour est plus haute pour le foyer vu
  en dernier sous bruit (ν 0.76 contre 0.36). La mémoire de soi garde aussi la trace de
  l'état dans lequel une zone a été vécue.
- Deux seeds, un jeu de constantes (τ_m 20, seuil ×2 du repos).

## La suite, dans l'ordre convenu

Le système sait maintenant désigner ce qu'il chérit : les foyers présents quand il
était bien. La question suivante est celle d'une **mémoire des moments chéris** : garder
ces foyers (lieu, m) au-delà de leur présence, et pouvoir en proposer un comme contexte
quand l'entrée se tait. C'est la sélection depuis la mémoire que Gepetto nommait, et le
« réseau du mode par défaut » du fil EMI : considérer l'absent. À poser après, une fois
qu'on aura décidé ce que « quand l'entrée se tait » veut dire pour la FPS.
