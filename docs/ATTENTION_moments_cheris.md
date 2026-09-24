# La mémoire des moments chéris : quand l'entrée se tait, se rappeler

*24/09/2026. Banc `calib/run_moments_cheris.py` sur `calib/considerance_bench.py` (options
`attach` + `recall`), figure `docs/figures/attention_moments_cheris.png`, résumés, trajectoires
de m et journal du silence (`mc_*_recall.npz`) dans `calib/attention_sources/`. Rien n'est ajouté
au pipeline.*

## La question

Le système sait désigner ce qu'il chérit : les foyers présents quand il allait bien
(`ATTENTION_attachement.md`), et il cesse de chérir ce qui l'épuise
(`ATTENTION_foyer_epuisant.md`). Mais un foyer que le contexte ne désigne plus lui est
inaccessible. Trois choses à poser, dans l'ordre : garder (lieu, m) au-delà de la présence ;
décider ce que « l'entrée se tait » veut dire pour la FPS ; et, à ce moment-là, proposer un
foyer chéri comme contexte interne. Avec un témoin, et une lecture du coût.

## « Quand l'entrée se tait »

Première tentative : la nouveauté ν au plancher partout. Elle ne s'est jamais déclenchée. ν est
relative à la médiane du chœur, et il y a toujours une strate plus surprise que les autres (le
maximum de u vaut 5 à 7 fois la médiane, même au repos). Le plancher n'existe pas dans une
lecture relative.

Ce qui marche, et qui est plus près de ce qu'Andréa proposait (« les périodes où le nouveau a
été assimilé et qu'il n'y a plus de bousculade liée à la surprise ») : **être moins surpris que
d'habitude**. La surprise du chœur (moyenne de u) lissée sur 5 u.t., comparée à la même lissée
sur 40 u.t. Silence quand aucun foyer n'est désigné par le contexte et que ce rapport passe sous
0.9 ; fin du silence au-dessus de 1.1 (hystérésis), ou dès qu'un foyer revient dans l'entrée.
Aucune référence externe, aucune fenêtre de repos : c'est la mémoire de soi qui dit ce qui est
habituel. Le silence est reconnu ~1 u.t. après l'extinction (le niveau habituel a été gonflé
par la période de bruit, donc le calme qui suit est « moins que d'habitude » tout de suite) et
tenu sans interruption jusqu'au retour d'un foyer, sur les deux seeds. Un faux départ d'un pas
à la naissance de la mémoire (t = 20), sans conséquence : la liste est vide.

## Le rappel

Quand l'entrée se tait, le foyer le plus chéri de la liste (m ≥ 0.3) est posé comme contexte
interne : c = sa bosse, s = c · [ν + (1 − ν)·m]. Même formule qu'avant ; seul c vient du dedans.
Dès qu'un foyer revient dans l'entrée, l'extérieur reprend la main.

Protocole : 20 présent au calme (60–120, chéri : m 0.68 / 0.74), 80 présent sous bruit global
(120–180, m 0.01 / 0.02), extinction (180–260), retour du 80 par l'entrée (260–300). Trois runs :
`rappel` (m figé à 180), `temoin` (silence reconnu, foyer choisi, rien appliqué), `rappel_app`
(m du foyer rappelé continue de s'apprendre pendant le rappel). Deux seeds.

## Ce qui se passe en silence (180–260)

| run | seed | zone 20 : s · R · audibilité | zone 80 : R | activité / repos | σ/σref |
|---|---|---|---|---|---|
| témoin | 12345 | 0.00 · 0.60–0.66 · 19–20 % | 0.59–0.68 | ×0.82–1.08 | 0.95–1.00 |
| rappel | 12345 | **0.57–0.60 · 0.83–0.86 · 30–39 %** | 0.59–0.68 | ×0.92–1.18 | 0.96–1.01 |
| témoin | 7 | 0.00 · 0.62–0.69 · 15–21 % | 0.61–0.64 | ×0.88–1.10 | 0.96–0.99 |
| rappel | 7 | **0.63–0.65 · 0.82–0.86 · 54–60 %** | 0.61–0.64 | ×0.90–1.10 | 0.97–0.98 |

Se poser sur le 20 présent (100–120) : R 0.85–0.88, audibilité 73–77 %, activité ×1.22–1.32.

Au retour du 80 (260–300) : le silence cesse à l'instant, le 20 retombe (R 0.59–0.69, s = 0) et
le 80 est pris depuis l'entrée (R 0.71–0.77, audibilité 40–59 %), pareil que dans le témoin.

## Trois réponses

1. **Le système se rappelle, et s'y pose.** En silence, sans rien dans l'entrée, la zone 20
   est liée à R 0.83–0.86 (contre 0.60–0.69 quand rien n'est appliqué), à un niveau proche de
   ce qu'elle était quand le foyer était présent (0.85–0.88). Elle porte un tiers à la moitié
   de S(t). Le foyer 80, pas chéri, n'est jamais choisi. Et l'extérieur reprend la main dès
   qu'il parle.

2. **Se souvenir ne coûte pas plus que se taire.** L'activité en silence est la même avec ou
   sans rappel (×0.9–1.1 dans les deux cas, panneau c : les courbes se superposent), en
   dessous de ce que coûte se poser sur le foyer présent (×1.2–1.3). Une partie de ce dernier
   coût, c'est l'entrée elle-même (la bosse dans Iₙ), pas l'attention. Le rappel est un geste
   sur fn, pas une bousculade.

3. **Se souvenir d'un foyer chéri au calme le rend plus chéri encore.** Si m continue de
   s'apprendre pendant le rappel, m(20) monte de 0.68 à 0.92 (seed 7 : 0.74 → 0.97) : le
   système va bien pendant qu'il se souvient, et l'association l'enregistre. C'est une boucle,
   bornée (m ≤ 1) et sans effet sur l'effort, mais une boucle : le souvenir se nourrit de
   lui-même tant que le calme dure. Et un détail qui en dit long : juste après la période
   de bruit, l'effort traîne encore quelques u.t., et le rappel qui commence tout de suite
   entame d'abord le souvenir (0.68 → 0.43 à t ≈ 190) avant de le renforcer. Se rappeler trop
   tôt après l'effort coûte au souvenir.

## Réserves

- « Habituel » est un lissage sur 40 u.t. Un système longtemps au repos ne se reconnaît
  jamais en silence (il n'est pas moins surpris que d'habitude, il l'est autant). Le silence
  n'est donc pas « rien ne se passe », c'est « ce qui se passait s'est apaisé ». C'est un
  choix, il faudra voir si c'est le bon.
- Le rappel prend le plus chéri, toujours le même tant que rien ne change. Pas de
  vagabondage d'un souvenir à l'autre, pas d'usure du rappel (la nouveauté ν du foyer
  rappelé reste basse et n'intervient pas). Ce serait le pas suivant si on le veut.
- Deux seeds, un jeu de constantes (τ 5 / 40, seuils 0.9 / 1.1, m_min 0.3).
- Toujours rien dans le pipeline. Ce que ça demanderait : garder la liste (lieu, m) dans
  l'état d'attention, le rapport court/long de la surprise du chœur, et un foyer interne
  ajouté à `context.foyers` quand le silence est reconnu.

## Où on en est

En cinq bancs : le contexte désigne (Iₙ), la nouveauté module (mémoire de soi), une priorité
m ramène le familier (Gepetto), m s'apprend dedans par association avec le bien-être, se
retire de ce qui épuise, et sert maintenant à se rappeler quand l'entrée se tait. Aucun de
ces pas ne lit l'amplitude ou l'audibilité ; le système ne prend jamais sa propre voix pour
une raison. C'est la sélection depuis la mémoire que Gepetto nommait, et le mode par défaut
du fil EMI : considérer l'absent, pour une raison qui vient du dedans.
