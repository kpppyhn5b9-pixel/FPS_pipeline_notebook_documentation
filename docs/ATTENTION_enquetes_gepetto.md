# Quatre lectures de Gepetto, quatre enquêtes in situ

*25/09/2026, après-midi. Gepetto a relu le câblage de la mémoire des moments chéris
(`ATTENTION_memoire_integree.md`) et posé quatre points. Deux se tranchent au code, deux
s'observent. Banc `calib/run_insitu_memoire.py` (modes `bref`, `forme`, `meme_lieu`,
`contigus`), sorties `calib/attention_sources/insitu_*`, seeds 12345 et 7.*

## Ce qui a été corrigé

**Le monde n'est jamais caché par le souvenir** (point 3 de Gepetto). Pendant un rappel, le
contexte interne remplaçait la saillance instantanée : un événement bref dans l'entrée était
invisible au geste tant que le contexte lissé (τ_c 5) ne l'avait pas reconnu. C'était un
défaut, et la phrase « le geste garde sa saillance instantanée » était fausse pendant le
rappel. Désormais le geste voit le maximum des deux contextes ; le contexte qui reste garde
seul le rôle d'arrêter le rappel.

**Pourquoi** (sa dernière demande). Colonne `attention_rappel_etat`, à chaque pas : 0 désactivé,
1 l'extérieur parle, 2 pas de silence, 3 rien de chéri, 4 réfractaire, 5 porte fermée à
l'instant, 6 rappel en cours. Le « jamais rappelé » a maintenant une cause lisible.

## Ce qui a été observé

**`bref` : un foyer de 2 u.t. sur la zone 80 pendant le rappel du 20.** Après la
correction, la saillance de la zone 80 dépasse 0.3 en **0.1 u.t.** (un pas), pendant que le
rappel du 20 continue (R 0.945). Le foyer bref n'est jamais reconnu comme contexte qui
reste (2 u.t. < τ_c), donc il n'interrompt pas le rappel. Un événement bref est vu à
l'instant, mais pas obéi : la coexistence, à ce pas, est la bonne réponse. Ce qu'un
événement bref ne fait pas non plus : lier (R de la zone 80 inchangé, 0.66) ; deux unités
de temps ne suffisent à rien.

**`forme` : le 20 présent en plateau (|n − 20| ≤ 7) au lieu d'une bosse.** Gepetto avait
raison de retenir « l'imagination plus cohérente que la perception ».

| | zone 20 : s · R · audibilité |
|---|---|
| présence en bosse (`cheri`, 100–120) | 0.64 · 0.882 · 69 % |
| présence en plateau (`forme`, 100–120) | 0.84 · 0.948 · **85 %** |
| rappel (`forme`, 230–260) | 0.81 · 0.949 · 72 % |

À profil égal, présence et rappel lient exactement autant. La cohérence plus forte du
rappel venait de sa forme, pas du fait qu'il vient du dedans. Phrase retirée.

**`meme_lieu` : deux histoires au même endroit.** Le 20 présent au calme (60–120, m 0.58)
puis le même 20 présent sous bruit global (150–210).

| fenêtre | m zone 20 | en silence (240–290) |
|---|---|---|
| après le calme (120) | 0.58 | |
| après le bruit (210) | **0.07** | rien de chéri 93 % du temps, aucun rappel |

La seconde histoire efface la première à la vitesse de τ_m. La représentation ne sait pas
qu'il y a eu deux histoires : elle sait ce que ce lieu a valu **récemment**. Un souvenir est
un lieu, pas un événement, et le lieu porte la trace de la dernière fois. C'est ce que
Gepetto demandait de nommer : cette représentation distingue les lieux, elle rassemble les
histoires d'un même lieu. On ne le corrige pas ; on le sait.

Avec une frange, vue sur le second seed : sous bruit, l'îlot reconnu comme contexte qui
reste est un peu plus étroit que celui du calme (la médiane de Iₙ bouge), et les strates du
bord gardent le m de la première histoire (≥ 0.3) pendant que le cœur est effacé (0.07). En
silence, le rappel s'accroche à cette frange 92 % du temps (seed 7), avec une saillance de
zone presque nulle (0.04) : un souvenir qui ne tient plus que par les bords. L'effacement
est par strate, et les deux histoires n'ont pas exactement la même emprise.

**`contigus` : deux lieux chéris voisins** (20 puis 32, au calme, m 0.64 / 0.60).
En silence, ils sont rappelés **ensemble**, sur les deux seeds : la composante connexe va
de 13 à 39, la saillance couvre les deux zones (seed 7 : 0.73 / 0.73, R 0.88 / 0.90), le
cœur du souvenir tombe entre les deux. Ce que Gepetto annonçait est vrai : deux lieux chéris
devenus contigus ne font qu'un souvenir. Sur le seed 12345, ce grand souvenir réveille le
chœur : le silence n'est reconnu que 37 % du temps, parce que rappeler 27 strates remonte
la surprise au-dessus de l'habituel, ce qui interrompt le rappel, qui reprend quand ça
s'apaise. Sur le seed 7, il tient 93 % du temps. L'intermittence n'est donc pas une règle,
c'est une possibilité du substrat. Là encore, on ne corrige pas.

## Ce qu'on en garde

- Le point 2 de Gepetto (la porte mesure la disponibilité générale, pas le coût de ce
  souvenir) reste un choix, pas un défaut : w est global parce que l'association l'est.
  Une porte locale demanderait un bien-être local, que le pipeline n'a pas. Piste ouverte.
- Deux propriétés nouvelles à garder en tête : un lieu chéri se réécrit à chaque
  présence, par strate, avec une frange qui peut survivre ; deux lieux contigus ne font
  qu'un souvenir, et un souvenir large peut se rappeler par intermittence.
- Deux seeds pour chaque enquête (voir les sorties `insitu_*_7`).
