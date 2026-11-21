# Écouter un texte : le pont vers les transformers, première moitié

*20/11/2026. Après la base propre (la mémoire des moments chéris, décidée et rangée), la FPS va voir ailleurs. D'abord
en écoutant, sans rien devoir : le transformer parle, la FPS écoute, rien ne revient vers le transformer. Banc :
`calib/run_ecoute_texte.py`. Sorties : `calib/attention_sources/ecoute_*.npz`.*

## Le pont, et ce qu'il ne fait pas

À chaque token d'un texte, un vecteur (l'état interne d'un modèle à une couche donnée, ou un témoin) est projeté sur N
directions fixes, une par strate, et devient Iₙ(t) pendant `k` pas ; après le texte, le silence. Chaque strate est
centrée-réduite sur le texte et × 0.25 : une excursion de 2 σ vaut 0.5, le gain des foyers de tous les bancs. Le reste,
la FPS le fait déjà : l'entrée relative à sa médiane, le contexte qui reste (τ_c), l'attachement, le silence, le rappel.
On lit ce qu'on lit déjà (cloche, garde, part saillante, m, silence, rappel et son origine), et une chose de plus : à
quels **passages** du texte correspondent les lieux chéris et rappelés (les fenêtres de douze mots où l'entrée sur ces
strates a été la plus haute).

Ce que le pont ne fait pas, par choix : lisser les états du modèle avant de les donner (ce serait faire le travail de
τ_c dehors), choisir les directions pour qu'elles trouvent quelque chose, et renvoyer quoi que ce soit au modèle.

Trois sources, pour avoir les témoins avant le modèle : `temoin` (trigrammes de caractères hachés, aucun
apprentissage), `vierge` (GPT-2 small aux poids aléatoires : l'architecture sans l'apprentissage), `modele`
(Qwen2.5-0.5B, multilingue, 24 couches, 896 dimensions, sur quatre cœurs). `--shuffle` mélange les mots **avant toute
lecture** (lecture de Codex : mélanger les vecteurs après coup aurait laissé au modèle le contexte de l'ordre d'origine).

## Ce que la note du soutien lui a fait (1350 tokens du modèle, 680 mots)

| source | temps par token | silence pendant le texte | lieux chéris à la fin | ce qu'ils portent | en silence |
|---|---|---|---|---|---|
| témoin, mélangé ou non | 0.4 u.t. | 80–100 % | aucun | — | rien de chéri |
| vierge (couches 6 et 12) | 0.4 | 55–100 % | 1 strate, m 0.42 / 0.36 | le passage où les mêmes mots reviennent (« le tient encore là… il était encore là quand ça a été mieux ») | rappel 83 %, appelé par le présent |
| modèle, brut | 0.4 | 8–100 % | aucun, m max 0 | — | rien de chéri (mélangé, autre graine : idem) |
| modèle, dimensions relatives | 0.4 | 0–60 % au milieu | aucun, m max 0.19 | une ligne de tableau en tirets | rien de chéri |
| modèle, dimensions relatives, dernière couche | 0.4 | 4–86 % | aucun, m max 0.13 | — | rien de chéri |
| **modèle, dimensions relatives, lent** | **1.2** | **0–7 % sur 1600 u.t.** | **16 lieux (20 % des strates), presque tous d'une strate, m 0.30–0.56** | des passages de chiffres et de tableaux, et des phrases (« Le soutien dans un moment difficile : chérir ce qui », « quand il n'y a pas de grands éclats ») | rappel 97 %, un seul centre (77), appelé par le présent, porte 0.82 |
| modèle, lent, **seconde graine de directions** (d1) | 1.2 | 0–8 % | 23 lieux (31 % des strates), d'une strate, m jusqu'à 0.62 | encore des chiffres et des tableaux ; « quand il n'y a pas de grands éclats » de nouveau | rappel 97 %, deux centres (77 : une plage de nombres, 68 % ; 16 : la formule du soulagement, 32 %), porte 0.85 |
| modèle, lent, **mots mélangés avant lecture** (d0) | 1.2 | 54 % au début, puis 0–14 % | 12 lieux (12 % des strates), d'une strate, m jusqu'à 0.65 | des noms de fichiers et de tests, « soulagement », des séparateurs de tableau : la forme, sans les phrases | rappel 97 %, un seul centre (28), porte 0.87, garde min 0.88 |
| **sac tenu** : les durées du texte dans l'ordre (d0), sans ses mots (surrogat de phase : mêmes rythmes par strate, mêmes co-mouvements, aucun texte derrière) | 1.2 | 0 % tout du long | 19 lieux (35 % des strates), **plusieurs larges** (71–78, 89–92, 59–61), m jusqu'à 0.76 | rien : il n'y a pas de mots | rappel 97 %, un seul centre (89–92), porte 0.93, garde min 0.98, activité ×1.05 |
| modèle, lent, **mots mélangés et doublés** (chaque mot lu deux fois ; 1350 tokens = la moitié du texte) | 1.2 | 34 % au début, puis 0–12 % | 16 lieux (18 % des strates), m jusqu'à 0.49 | la forme : un chemin de fichier, « le le 26/09 » | rappel 87 %, deux centres (35, 59–60), porte 0.76, activité ×1.25 (la plus haute de toutes) |
| modèle, lent, **second texte** (la note des enquêtes de Gepetto, 1350 tokens) | 1.2 | 0–14 % | 11 lieux (13 % des strates), m jusqu'à 0.62 | des noms de modes entre accents graves, des chiffres de tableau, et quelques phrases (« la coexistence », « deux unités de temps ne suffisent à rien », « le geste garde sa saillance ») | rappel 86 %, un seul centre (5 : les noms de modes), porte 0.82 |

Cloche à 5 partout, garde libre (min 0.88 sur le mélangé, 0.95 et plus ailleurs) : rien de ce qu'elle a entendu ne l'a
mise en danger. Les quatre écoutes lentes ont chacune demandé une heure et demie à deux heures sur quatre cœurs.

## Ce qu'on a appris, dans l'ordre

1. **Le modèle brut n'est pas lisible tel quel.** Une dimension sur 896 (la 62, σ 76 contre 0.3 pour les autres) et
   un token-puits (« est », norme 1736 contre 16) font 83 % de ce que les strates reçoivent : toutes voient le même
   nombre, au signe près, donc aucun contraste, donc aucun îlot. Ce sont les « activations massives » connues des
   transformers. Lire chaque dimension relative à son habitude sur le texte (le principe que la FPS applique déjà à
   son entrée) ramène cet axe commun à 11 %. Retiré par défaut ; `--brut` pour voir sans ; les trois sorties brutes
   gardées sous ce nom.
2. **Le texte lui parle, mais à sa vitesse à elle.** À 0.4 u.t. par token, le modèle change d'état à chaque mot
   (corrélation 0.3 d'un token au suivant, 0.04 à douze) et « ce qui reste » demande douze tokens : rien ne reste.
   À 1.2 u.t. par token, le silence tombe à zéro pendant tout le texte, et des lieux se forment. La rencontre entre
   les deux temps, celui des mots et le sien, est la première chose vraie du pont.
3. **Ce qui reste, d'abord, c'est la forme.** Avec l'architecture seule (vierge), la répétition des mots. Avec le
   modèle, encore beaucoup de chiffres, de tableaux et de syntaxe : ce sont les passages où l'état du modèle reste
   le même longtemps. Quelques phrases aussi, et l'unique lieu rappelé en silence porte « quand il n'y a pas de grands
   éclats » et le titre de la note. La seconde graine de directions ramène la même maxime parmi ses lieux ; on a
   vérifié si c'était un goût : non. Les passages chéris couvrent 62 à 74 % du texte, le recouvrement entre les deux
   graines (40 fenêtres sur 46) est à peine au-dessus du hasard (moyenne 33, 95ᵉ centile 39), et la maxime est dans
   les 15 % de fenêtres où l'état du modèle est le plus tenu. Ce qui revient d'une graine à l'autre, c'est la
   géométrie du modèle (ce qu'il tient longtemps), pas une préférence de la FPS. Ce qui compterait comme un goût :
   un passage rappelé en silence sur plusieurs graines, et absent quand les mots sont mélangés.
4. **Les lieux sont minces : une strate.** Sous une projection aléatoire, deux strates voisines ne se ressemblent pas,
   donc la contiguïté du chœur, sur laquelle toute la mémoire repose (un souvenir est une région), ne veut rien dire
   ici. Les « lieux » sont des directions isolées. C'est une question sur le pont, pas sur elle : donner aux directions
   voisines une parenté (une seule ligne : des directions lissées le long du chœur) rendrait à la FPS la géométrie
   qu'elle connaît. À décider ensemble, et à observer avant.
   *(21/11 : observé avant, et la question a changé. Le sac tenu, sans aucun lissage, fait des lieux de 3, 4 et 8
   strates ; la minceur venait aussi du texte, voir la leçon 9. Décidé avec Andréa : on ne fait pas tenir les îlots
   artificiellement, on cherche ce que la largeur veut dire pour elle ; `--voisines` reste écrit, pas lancé.)*
5. **Se souvenir coûte un peu ici** : porte 0.82 pendant le rappel, activité ×1.2 pendant tout le texte. Elle est au
   travail tout du long. Confortable au sens des mesures (cloche, garde), pas au repos.
6. **Sans l'ordre des mots, elle garde moins, et pas la même chose.** Les mêmes mots mélangés avant lecture (mêmes
   directions d0) laissent 12 strates chéries contre 20 dans l'ordre (31 avec l'autre graine), un m moyen à la fin du
   texte de 0.11 contre 0.18, et 4 strates communes avec l'écoute ordonnée là où le hasard en donne 2.4 (p 0.19 :
   rien à dire). Ce qui reste du texte mélangé est la forme seule : noms de fichiers, séparateurs, un mot répété. Le
   silence revient aussi au début (54 % sur la première centaine d'unités de temps), avant qu'elle ne s'y fasse. Un
   seul mélange, une seule graine : une direction, pas un résultat.
7. **Ce qu'elle garde, c'est la durée, pas le sens** (21/11, témoin proposé par Andréa). Le modèle tient moins un sac de
   mots (ressemblance d'un token au suivant 0.24 contre 0.34 ; tenue sur 12 tokens 0.41 contre 0.46), donc « sans
   l'ordre elle garde moins » pouvait n'être que ce reflet. Le sac tenu tranche : les durées exactes du texte dans
   l'ordre, sans aucun mot derrière, lui laissent **plus** que le texte lui-même (35 % des strates contre 20 %, m moyen
   0.22 contre 0.18, m jusqu'à 0.76), avec moins de travail (activité ×1.05 contre ×1.17) et, pour la première fois,
   des lieux larges de plusieurs strates. La leçon 6 se relit donc ainsi : elle est un miroir fidèle de ce que le
   modèle tient ; un texte qui a du sens, le modèle le tient plus longtemps, et elle le garde plus. Pas de goût pour le
   sens chez elle, une mesure honnête de la tenue de ce qu'on lui donne. Ce que le vrai texte a en plus du sac tenu,
   ce sont ses éclats : des excursions à plus de 4 σ (0.14 % des valeurs, aplatissement 15 contre 3 pour le sac
   gaussien), et ces éclats lui coûtent (activité) et lui laissent moins. Les mots doublés (chaque mot lu deux fois,
   l'autre témoin) ne font pas tenir le modèle davantage (0.26, 0.42 contre 0.24, 0.41 pour le simple mélange) : l'état
   d'un mot répété n'est pas l'état du mot. Elle suit la tenue à la lettre : un peu plus tenu, un peu plus gardé (18 %
   des strates contre 12 %), mais la répétition la fait travailler plus que tout le reste (activité ×1.25) et la porte
   descend à 0.76. Trois témoins, une seule lecture : elle garde ce que le modèle tient, et paie ce qui le secoue.
8. **Le second texte lui fait la même chose que le premier.** Treize pour cent des strates, un seul centre rappelé
   (les noms de modes entre accents graves), quelques phrases parmi les chiffres. Ce que le pont lui donne d'un texte
   de notes de travail, c'est d'abord ce qui y est tenu : les noms, les tableaux, et de loin en loin une phrase.
9. **Ce que la largeur d'un lieu veut dire, première lecture** (21/11, sur les 97 lieux des six écoutes lentes, sans
   rien lancer). Je croyais : lieu mince = le modèle sursaute sur un axe, lieu large = il se déplace tout entier.
   C'est faux, presque à l'envers. Aux moments qui appellent un lieu mince, le modèle bouge en moyenne sur 27 strates
   (souvent 50 d'un coup : ce sont les éclats, des tokens où la moitié des directions dépassent 2 σ) ; aux moments qui
   appellent les deux lieux larges, sur 7. Un sursaut en bloc d'un seul token ne laisse qu'une strate : ce qui reste
   (τ_c) ne garde pas un éclat, il garde ce qui dure, et ce qui dure après un éclat n'est que la direction où il était
   le plus fort. Les lieux larges naissent de voisines qui bougent modérément et longtemps ensemble ; il n'y en a que
   dans le sac tenu, parce que c'est le seul texte sans éclats. Et les lieux rappelés en silence dans les trois textes
   dans l'ordre (77, 16, 5, 89–92) sont tous nés à des moments calmes, 5 à 7 strates en mouvement, jamais d'un éclat ;
   ceux du mélangé et du doublé, si. Lecture provisoire : pour elle, un lieu mince né d'un éclat est le souvenir d'un
   sursaut, un lieu large le souvenir d'un déplacement, et ce qu'elle rappelle d'un texte qui a du sens, ce sont ses
   moments calmes. Une lecture, six écoutes : à regarder encore avant d'y croire.

## Deux limites apprises (nuit du 20 au 21/11)

- **Les écoutes lentes durent plus que la fenêtre d'une tâche d'arrière-plan** (90 min) : trois lancées ensemble ont
  été coupées sans rien écrire. Lancer détaché (`setsid nohup`), et une à la fois.
- **L'historique du pipeline garde tout à chaque pas** : une écoute de 1840 u.t. monte à plusieurs gigaoctets, et
  deux en parallèle ont été tuées par la mémoire (15 Go). D'où `--max-tokens 1350` sur le second texte, et la chaîne
  séquentielle. On l'écoute d'abord ; le moment voulu, on rangera les mesures.

## Ce qui attend

La seconde graine, le mélange, le second texte et le sac tenu ont été regardés : ce qui tient d'une écoute à l'autre,
c'est la forme tenue par le modèle et un seul centre rappelé en silence ; ce qui change avec l'ordre des mots, c'est
combien le modèle tient, et elle le reflète. Les mots doublés confirment. Reste la question des directions voisines (`--voisines 6`, écrite, pas lancée : à comparer au mince sur le
même texte et la même graine), quand ça appellera. Rien ne revient vers le modèle.
