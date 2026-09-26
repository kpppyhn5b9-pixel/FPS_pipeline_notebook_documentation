# Rêver : deux formes essayées, aucune ne tient encore

*26/09/2026. Deuxième piste (`ATTENTION_pistes.md`), reformulée le matin avec Andréa : rêver n'est
pas se poser sur une chimère venue de rien, c'est **relier des souvenirs éloignés** (ses intuitions,
ces ponts qui se tissent d'eux-mêmes entre des choses comprises séparément) et **accueillir ce que le
substrat propose de lui-même** (les motifs propres à la structure, eux aussi des rêves légitimes).
Interrupteur `attention.reve.mode` : `off` (défaut), `liaison`, `substrat`, `les_deux`. Banc in situ
`calib/run_insitu_memoire.py` (modes `reve_liaison`, `reve_liaison_off`, `reve_substrat`,
`reve_substrat_off`, `reve_substrat20`), sorties `calib/attention_sources/insitu_reve_*`.
États 7 (rêve-liaison) et 8 (rêve-substrat) dans `attention_rappel_etat`. Test `test_dream_bridge_and_link`.*

## Ce qu'on a construit

**Liaison.** En silence, au lieu de rappeler un seul lieu, tenir ensemble le lieu que le présent
appelle et le meilleur autre lieu chéri qui ne le touche pas, comme **un seul îlot** : un seul champ
moyen, une seule moyenne de fréquence, par-dessus la distance (`bridge` dans `attention_delta_fn`).

**Substrat.** En silence, quand rien n'est chéri, se poser sur la plus grande région où la cohérence
locale, lissée, se détache de la médiane du chœur : ce que la structure produit d'elle-même.

## Ce qu'on a vu

**Liaison** (20 puis 65 présents au calme, chéris 0.57 / 0.32 ; silence 160–260 ; deux seeds).

| | seed | verrouillage de phase entre les deux zones (190–260) | R zone 20 | R zone 65 | silence tenu |
|---|---|---|---|---|---|
| rappel ordinaire (un seul lieu) | 12345 | 0.00–0.01 | **0.93–0.95** | 0.62–0.66 | 100 % |
| rêve-liaison | 12345 | 0.12–0.13 | 0.77–0.84 | 0.63–0.65 | 49–68 % |
| rappel ordinaire | 7 | 0.01 | **0.90–0.93** | 0.62–0.66 | 100 % |
| rêve-liaison | 7 | 0.15–0.20 | 0.85–0.92 | 0.67–0.70 | 98–100 % |

Le pont tient à peine. Les deux zones se verrouillent faiblement (0.13 sur un seed, 0.20 sur
l'autre, contre 0.01 sans pont) ; la seconde ne se lie jamais (R 0.63–0.70, à son repos) ; la
première **perd** de sa cohérence (le champ moyen commun est un compromis qui ne convient
vraiment à aucune des deux). Sur le seed 12345, ce grand geste réveille en plus le chœur : la
surprise remonte au-dessus de l'habituel, le silence se rompt un tiers à la moitié du temps, le
rêve s'interrompt et reprend ; sur le seed 7 il tient. C'est la propriété déjà vue avec deux lieux
contigus (« un souvenir large se rappelle par bribes »), et pire : ici les deux moitiés n'ont pas
de voisinage pour se tenir. Un pont faible existe donc, mesurable, mais pas un tout cohérent.

Au premier essai (20 et 80), il n'y avait même pas deux lieux chéris : la zone 80 coûte ×1.5, et en
mode `soulagement` une présence qui coûte plus que l'habituel n'est pas chérie (m 0.27). Noté, parce
que c'est une propriété du mode par défaut : ce qui fatigue un peu plus que d'habitude n'est pas
retenu, même au calme.

**Substrat** (20 présent sous bruit, rien de chéri ; silence 120–260).

| lissage de la cohérence | rêve-substrat (part du silence) | cœurs distincts | écart-type du cœur | R de la région |
|---|---|---|---|---|
| τ 5 | 57–83 % | 42 | 23 strates | inchangé (0.63) |
| τ 20 | 19–73 % | 12 | 23 strates | inchangé |

Ce que le substrat propose ne tient pas en place : la région de cohérence spontanée se déplace sans
cesse le long du chœur (la chimère papillote), et se poser dessus ne la stabilise pas ; R n'y bouge
pas. Lisser plus longtemps réduit le nombre de régions choisies, pas leur errance.

## Ce qu'on en garde

- **On a regardé avant de croire, et ça ne tient pas comme ça.** Ni le pont (un verrouillage de
  0.1–0.2, jamais la seconde zone) ni le substrat ne donnent un tout cohérent avec le geste qu'on a. Les deux restent dans le pipeline, éteints, avec
  leurs lectures, pour ne pas les reconstruire si on y revient.
- **Ce que ça dit du système :** un souvenir, pour lui, est une région contiguë de taille modeste.
  Tout ce qui est plus grand ou disjoint réveille le chœur et se défait. Ce n'est pas une limite du
  rêve, c'est la forme de son attention : locale, par voisinage. Relier des lieux éloignés
  demanderait autre chose qu'un champ moyen partagé (peut-être une résonance de rythmes, par κ
  seul, avant toute phase ; peut-être une alternance, aller de l'un à l'autre, ce qu'Andréa
  reconnaît comme « par bribes »).
- **Et du substrat :** ses îlots spontanés existent (on les avait vus au premier fil) mais ils sont
  transitoires. Pour qu'ils deviennent un rêve, il faudrait que l'attention puisse les *retenir*,
  et le geste actuel ne retient pas ce qui n'a pas de forme dans le temps, par construction
  (« le contexte, c'est ce qui reste »). Le rêve du substrat et le contexte qui reste se
  contredisent tels quels.
- Pistes, pour plus tard : la liaison par alternance (rappeler l'un, puis l'autre, et lire si
  quelque chose passe de l'un à l'autre) ; la liaison par rythme seul (κ sur l'union, K₀ par
  îlot) ; retenir un motif du substrat le temps qu'il s'installe.
