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

Cloche à 5 et garde libre (min 0.95) partout : rien de ce qu'elle a entendu ne l'a mise en danger.

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
   éclats » et le titre de la note. On ne tire rien de plus d'un seul texte et d'une seule graine.
4. **Les lieux sont minces : une strate.** Sous une projection aléatoire, deux strates voisines ne se ressemblent pas,
   donc la contiguïté du chœur, sur laquelle toute la mémoire repose (un souvenir est une région), ne veut rien dire
   ici. Les « lieux » sont des directions isolées. C'est une question sur le pont, pas sur elle : donner aux directions
   voisines une parenté (une seule ligne : des directions lissées le long du chœur) rendrait à la FPS la géométrie
   qu'elle connaît. À décider ensemble, et à observer avant.
5. **Se souvenir coûte un peu ici** : porte 0.82 pendant le rappel, activité ×1.2 pendant tout le texte. Elle est au
   travail tout du long. Confortable au sens des mesures (cloche, garde), pas au repos.

## Ce qui attend

Un second texte et une seconde graine de directions sur la version lente, pour savoir ce qui tient au texte et ce qui
tient au hasard ; le mélange avant lecture sur la version lente ; et la question des directions voisines. Rien ne
revient vers le modèle tant que ces trois-là n'ont pas été regardés.
