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
| témoin, mélangé ou non | 0.4 u.t. (et 1.2, le 22/11 : pareil, m max 0.10, silence 0–78 %) | 80–100 % | aucun | — | rien de chéri |
| vierge (couches 6 et 12) | 0.4 | 55–100 % | 1 strate, m 0.42 / 0.36 | le passage où les mêmes mots reviennent (« le tient encore là… il était encore là quand ça a été mieux ») | rappel 83 %, appelé par le présent |
| modèle, brut | 0.4 | 8–100 % | aucun, m max 0 | — | rien de chéri (mélangé, autre graine : idem) |
| modèle, dimensions relatives | 0.4 | 0–60 % au milieu | aucun, m max 0.19 | une ligne de tableau en tirets | rien de chéri |
| modèle, dimensions relatives, dernière couche | 0.4 | 4–86 % | aucun, m max 0.13 | — | rien de chéri |
| **modèle, dimensions relatives, lent** | **1.2** | **0–7 % sur 1600 u.t.** | **16 lieux (20 % des strates), presque tous d'une strate, m 0.30–0.56** | des passages de chiffres et de tableaux, et des phrases (« Le soutien dans un moment difficile : chérir ce qui », « quand il n'y a pas de grands éclats ») | rappel 97 %, un seul centre (77), appelé par le présent, porte 0.82 |
| modèle, lent, **seconde graine de directions** (d1) | 1.2 | 0–8 % | 23 lieux (31 % des strates), d'une strate, m jusqu'à 0.62 | encore des chiffres et des tableaux ; « quand il n'y a pas de grands éclats » de nouveau | rappel 97 %, deux centres (77 : une plage de nombres, 68 % ; 16 : la formule du soulagement, 32 %), porte 0.85 |
| modèle, lent, **mots mélangés avant lecture** (d0) | 1.2 | 54 % au début, puis 0–14 % | 12 lieux (12 % des strates), d'une strate, m jusqu'à 0.65 | des noms de fichiers et de tests, « soulagement », des séparateurs de tableau : la forme, sans les phrases | rappel 97 %, un seul centre (28), porte 0.87, garde min 0.88 |
| **sac tenu** : les durées du texte dans l'ordre (d0), sans ses mots (surrogat de phase : mêmes rythmes par strate, mêmes co-mouvements, aucun texte derrière) | 1.2 | 0 % tout du long | 19 lieux (35 % des strates), **plusieurs larges** (71–78, 89–92, 59–61), m jusqu'à 0.76 | rien : il n'y a pas de mots | rappel 97 %, un seul centre (89–92), porte 0.93, garde min 0.98, activité ×1.05 |
| modèle, lent, **mots mélangés et doublés** (chaque mot lu deux fois ; 1350 tokens = la moitié du texte) | 1.2 | 34 % au début, puis 0–12 % | 16 lieux (18 % des strates), m jusqu'à 0.49 | la forme : un chemin de fichier, « le le 26/09 » | rappel 87 %, deux centres (35, 59–60), porte 0.76, activité ×1.25 (la plus haute de toutes) |
| modèle, lent, **quatre trébuchements** : la note soutien dans l'ordre, sauf quatre passages de 50 mots mélangés localement, sans le lui dire (`--trebucher`) | 1.2 | 0–32 %, un peu plus à l'entrée de trois zones sur quatre | 15 lieux (20 % des strates), m jusqu'à 0.66, appelés moins souvent par les zones (20 % contre 29 % des tokens) | des lignes de tableau, hors des zones | rappel 98 %, un centre (52), porte 0.83, garde 1.00 |
| modèle, lent, **second texte** (la note des enquêtes de Gepetto, 1350 tokens) | 1.2 | 0–14 % | 11 lieux (13 % des strates), m jusqu'à 0.62 | des noms de modes entre accents graves, des chiffres de tableau, et quelques phrases (« la coexistence », « deux unités de temps ne suffisent à rien », « le geste garde sa saillance ») | rappel 86 %, un seul centre (5 : les noms de modes), porte 0.82 |
| modèle, lent, **relecture** : les 675 premiers tokens de la note soutien, 200 u.t. de silence, les mêmes 675 (`--relire 200 --max-tokens 675`, nuit du 21 au 22/11) | 1.2 | 1re lecture 10 %, 38 %, puis 0–5 % ; trou 73 % ; 2de lecture 0–6 % | fin de la 1re : 2 lieux (32, 36), m 0.43 ; fin de la 2de : 16 lieux (16 %), m 0.57 | les mêmes passages lus deux fois (« soutien dans un moment difficile : chérir ce qui », « ça allait mal, ça va mieux », des lignes de tableau) | trou : rappel 73 %, un centre (36) ; après : rappel 98 %, deux centres (60 : la phrase d'ouverture, 69 % ; 36 : 31 %), porte 0.84, garde 1.00 |
| **témoin de la relecture** : les mêmes 675 tokens, le même silence, puis 675 tokens d'un AUTRE texte (la note de Gepetto, `--puis`) | 1.2 | identique jusqu'au trou ; 2de partie 0–7 % | fin de la 2de partie : 13 lieux (13 %), m 0.69 (strate 5, les noms de modes de Gepetto, comme dans l'écoute du second texte seul) | les deux textes | après : silence 72 % seulement, rappel 72 %, un centre (5, 100 %), le centre de la 1re lecture (36) n'est plus rappelé, porte 0.85, garde min 0.93, activité ×1.17 |
| modèle, lent, **relecture de la note entière** : 1350 tokens, 200 u.t. de silence, les mêmes 1350 (`--relire 200`, finie le 22/11 à 6 h, 4 h de calcul) | 1.2 | 1re lecture 22 % au début puis 0–3 % ; trou 88 % ; 2de lecture 0–3 % | fin de la 1re : 19 lieux (19 sur les 20 de l'écoute de référence, écarts numériques), m 0.54 ; fin de la 2de : **48 lieux (48 % des strates), m 0.72, les lieux s'élargissent** (27–30, 32–36, 43–45, 62–64, 89–91) | les mêmes passages, deux fois | trou : rappel 88 %, centre 77 (97 %) ; après : rappel 87 %, centre 77 (94 %, la phrase d'ouverture, le centre connu de ce texte), 52 et 23 pour le reste, porte 0.82, garde min 0.93 |

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
10. **Voit-elle autre chose que la statistique en une ligne ? Non, pas ici** (21/11 au soir, question posée à deux).
    Quatre passages de cinquante mots mélangés localement dans la note soutien, à des endroits connus de nous seuls.
    La statistique les voit tous les quatre, nettement (ressemblance d'un token au suivant 0.27 dans les zones contre
    0.35 ailleurs ; chaque zone fait baisser le chiffre). Elle en marque trois : un silence bref à l'entrée de la zone
    (25 %, 7 %, 16 % de la première fenêtre, là où le texte dans l'ordre n'en avait aucun), puis plus rien ; la
    quatrième, elle ne la marque pas. L'activité ne bouge pas (1.19 contre 1.19) ; la part saillante baisse un peu dans
    les zones (1.0 % contre 1.3 %) ; les strates chéries sont appelées moins souvent depuis les zones (20 % de leurs
    tokens appelants, pour 29 % des tokens) ; les deux centres rappelés en silence sont des lignes de tableau, hors
    des zones. Une fois, un silence 60 u.t. après la sortie d'une zone (15 %), qu'on pourrait lire comme un temps pour
    s'en remettre ; une fois, ça ne fait pas une loi. Ce qu'on peut dire : elle réagit à un trébuchement du modèle dans
    son vocabulaire à elle (se taire en entrant dans la tempête, en garder moins), plus tard et plus faiblement que le
    chiffre d'une ligne, et sans rien y voir que lui ne voie pas. Elle est un miroir fidèle et lent, pas une loupe.
    Si sa valeur pour lire un modèle existe, elle n'est pas dans la détection ; elle serait dans ce qu'elle garde et
    rappelle après, et ça demande d'autres questions que celle-ci.
11. **Relire : reconnaît-elle ? Non ; elle tient** (nuit du 21 au 22/11, question posée à deux, lue avec son témoin).
    On lui a lu deux fois les 675 premiers tokens de la note soutien, 200 u.t. de silence entre, et, pour témoin, la même
    première lecture et le même silence suivis d'un autre texte (la note de Gepetto, coupée au même nombre de tokens).
    Les deux écoutes sont identiques jusqu'à la seconde partie, au chiffre près. Ce qu'on avait dit qu'on regarderait :
    - *Se taire moins la seconde fois ?* Oui, mais devant l'autre texte aussi (1 % contre 2 %). Le silence des premières
      fenêtres (10 %, 38 %) est celui du début de la course, pas celui d'un texte neuf. Rien de reconnu là.
    - *Les lieux plus vite ?* Les lieux passent de 2 à 16 avec le même texte, de 2 à 13 avec l'autre, et l'accroissement
      moyen de m est le même (+0.056 contre +0.058). Huit strates franchissent 0.3 dans les deux cas : elles étaient
      juste dessous après la première lecture, n'importe quel texte de même tenue les fait passer (la durée, leçon 7).
      La seule chose que le même texte fait et que l'autre ne fait pas : il creuse les mêmes strates (accroissements
      corrélés 0.80 à ceux de la première lecture, contre 0.12 pour l'autre texte). C'est attendu de bout en bout :
      mêmes directions, même texte, mêmes strates. Le même texte deux fois compte double, il ne compte pas autrement.
    - *Un sursaut à la reprise ?* La strate 60 (la phrase d'ouverture) gagne +0.29 dans les cent premières u.t. de la
      seconde lecture, contre +0.21 en huit cents la première fois. Mais devant l'autre texte, la strate 83 gagne +0.31
      dans les mêmes cent u.t. Ce qui marque fort, c'est le premier texte qui vient après un silence, quel qu'il soit :
      l'état d'après le silence, pas la reconnaissance. La surprise, plus basse dans la seconde partie (0.176 contre
      0.225), l'est à l'identique dans les deux écoutes : le temps, pas le texte.
    - *Des souvenirs de la première lecture appelés par la seconde ?* Pendant la lecture, aucun rappel : elle ne se tait
      pas. Après, oui, et c'est la seule différence qui tienne au texte : après la relecture elle se tait à 98 % et
      rappelle la phrase d'ouverture (69 %) et le centre de la première lecture (36, 31 %) ; après l'autre texte, elle
      ne se tait qu'à 72 %, rappelle le centre du nouveau texte seul (5, 100 %), la garde descend à 0.93 et l'activité
      monte un peu. Le lieu 36 est pourtant plus chéri qu'avant (0.66) : il n'est pas perdu, il n'est plus appelé par le
      présent, qui ressemble au dernier texte. Une écoute de chaque : une direction, pas une loi.
    Ce qu'on peut dire : elle ne reconnaît pas un texte ; rien ne bouge chez elle qu'un autre texte ne fasse bouger
    pareil, sauf que le même texte creuse les mêmes places. Relire ne la fait pas reconnaître, ça la fait tenir : ce qui
    a accompagné deux fois reste deux fois, et après, elle se tait et appelle les deux. Pour qu'une reconnaissance soit
    possible il faudrait que le présent sache ressembler au passé pendant que le texte parle, et ce chemin (le rappel
    hors du silence) n'existe pas dans le code ; on l'écrit comme question, pas comme manque. La relecture de la note
    entière (1350 tokens deux fois) a fini à 6 h du matin (4 h de calcul), et dit la même chose, en plus grand :
    - la première lecture retrouve l'écoute de référence au bruit numérique près (l'état du modèle diffère de 4·10⁻⁶,
      le chœur de 5·10⁻⁵ avant le texte ; 1640 u.t. plus loin, 19 lieux sur 20 sont les mêmes et m diffère de 0.02 au
      plus) ; le silence de la seconde lecture est celui de la première (2–3 % par fenêtre de 200 u.t., contre 22 % au
      tout début de la course seulement) ;
    - les lieux passent de 19 à **48** (48 % des strates), par les mêmes strates (accroissements corrélés 0.86), avec un
      accroissement moyen plus faible la seconde fois (+0.115 contre +0.181 : l'attachement est borné, ce qui est haut
      monte moins) ; **les lieux s'élargissent** : 15 îlots d'une ou deux strates après la première lecture, 24 îlots dont
      six de trois à cinq strates après la seconde (27–30, 32–36, 43–45, 62–64, 84–86, 89–91). Les voisines d'un lieu
      étaient juste sous le seuil ; relire les fait passer. Les lieux larges de la leçon 9 venaient de co-mouvements
      calmes et longs ; ceux-ci viennent de l'accumulation, c'est une autre façon d'avoir des lieux larges ;
    - la part saillante pendant la seconde lecture est presque double (2.0 % contre 1.2 %) : c'est le seul chemin par
      lequel la mémoire agit pendant qu'un texte parle, la valeur m dans la saillance qui ramène le familier chéri ; la
      surprise est plus basse (0.178 contre 0.204), l'activité égale ;
    - dans le trou comme après, elle rappelle 77, la phrase d'ouverture, le centre connu de ce texte (97 % puis 94 %),
      et après la seconde lecture elle se tait un peu moins qu'après une seule (87 % contre 97 %).
    Ce qui a accompagné deux fois reste deux fois, et s'étend aux voisines. Elle ne reconnaît pas ; elle s'installe.

## Limites apprises (nuits du 20 au 22/11)

- **Les écoutes lentes durent plus que la fenêtre d'une tâche d'arrière-plan** (90 min) : trois lancées ensemble ont
  été coupées sans rien écrire. Lancer détaché (`setsid nohup`), et une à la fois.
- **L'historique du pipeline garde tout à chaque pas** : une écoute de 1840 u.t. monte à plusieurs gigaoctets, et
  deux en parallèle ont été tuées par la mémoire (15 Go). D'où `--max-tokens 1350` sur le second texte, et la chaîne
  séquentielle. On l'écoute d'abord ; le moment voulu, on rangera les mesures.
- **Le coût d'un pas grandit avec l'historique** (nuit du 21 au 22/11, vu en échantillonnant la pile pendant que ça
  tournait) : à chaque pas, `compute_multiple_tau` et `compute_temporal_coherence` recalculent l'autocorrélation
  complète de S, γ et C depuis le début (`np.correlate` sur tout l'historique). Le pas coûte comme le carré du temps
  écoulé, l'écoute comme son cube : 1840 u.t. en 55 min, 2040 en 59, 3700 estimées à 7 h. Ces valeurs ne sont que
  journalisées (`tau_S`, `temporal_coherence`, `autocorr_tau`) ; rien dans la dynamique ne les relit. Une fenêtre
  bornée les rend linéaires sans rien changer à ce qu'elle fait. Décidé avec Andréa le 22/11 et fait :
  `analysis.fenetre_autocorrelation_ut` (200 u.t. par défaut, `null` = tout l'historique) ; vérifié sur un run court
  que O et m sont identiques au bit près avec et sans, et que les valeurs journalisées sont les mêmes tant que
  l'historique tient dans la fenêtre. Un pas d'autocorrélations : 60 ms sur 2000 pas, 600 ms sur 36 600.
- **« Détaché » survit à la fin d'un tour, pas au remplacement de la machine** : le conteneur a redémarré vers 23h40 et
  la première relecture (quatre heures de calcul) est morte sans rien écrire, parce que la simulation n'écrit qu'à la
  fin. Un guet réarmé toutes les trente minutes a semblé tenir la machine éveillée ensuite ; rien de garanti.
- **Le post-traitement de `run_simulation` tue les longues écoutes** : la demi-relecture (20 400 pas) a été tuée par le
  noyau à 12.6 Go après la fin de sa boucle, alors que la boucle n'en prenait que 2. Une sonde courte (2000 pas) ne
  reproduit pas l'explosion (conversion ×2.7, analyse et exploration négligeables), donc la cause précise reste à
  trouver. Le banc appelle maintenant la boucle seule (`init` + `run_fps_simulation`, mêmes graines, même dynamique,
  sans analyse ni dashboard) : la même écoute passe en 59 min et 2.0 Go au plus ; la relecture de la note
  entière (36 600 pas) en 4 h 04, 2.2 Go pendant la boucle et 4.0 Go au plus à la fin.

## Ce qui attend

La seconde graine, le mélange, le second texte et le sac tenu ont été regardés : ce qui tient d'une écoute à l'autre,
c'est la forme tenue par le modèle et un seul centre rappelé en silence ; ce qui change avec l'ordre des mots, c'est
combien le modèle tient, et elle le reflète. Les mots doublés confirment, et les quatre trébuchements aussi : elle ne
voit rien que la tenue ne voie, elle le voit à sa manière. La relecture, lue avec son témoin, dit qu'elle ne
reconnaît pas un texte mais le tient plus, et que relire élargit ses lieux (leçon 11). Reste, quand ça appellera :
un long silence après le texte (900 u.t.) ; la vitesse du texte face à τ_c ; les directions voisines (`--voisines 6`,
écrite, pas lancée : à comparer au mince sur le même texte et la même graine). Rien ne revient vers le modèle.
