# La question posée au substrat : la structure rend-elle la considération possible ?

*21/09/2026. Banc `calib/run_considerance_structures.py`, figure
`docs/figures/considerance_structures.png`. Une question, pas un mécanisme de plus :
rien n'est ajouté au pipeline.*

## Ce qu'on demande

Attention.md définit la métastabilité migrante contextuelle comme « la capacité de
re-couplage rapide, réversible et sélectif », et l'efficience de la considération
comme la proximité au flow : beaucoup de contact, peu d'effort, identité intacte,
migration lisse. Lue ainsi, l'efficience n'est pas un score à faire monter : c'est une
**susceptibilité**. Combien de contact on obtient pour combien de forçage, sans
contagion, et en revenant à soi quand le contexte part.

Le banc pose cette question au substrat avec le geste déjà validé
(`run_context_binding.py`, champ moyen d'îlot, K₀ = 2, κ = 0.5, gardien σ) : un
contexte (projecteur gaussien, largeur 6 strates) se pose sur la strate 20, tient
20 u.t., migre en 80, puis en 50, puis disparaît. Quatre lectures, aucun barème :

| lecture | mesure |
|---|---|
| engagement | R dedans − R dedans avant (plus temps de montée, corr(s, R)) |
| sélectivité | R dehors − R dehors avant ; σ_Rloc / σ_ref ; cloche de dispersion |
| réversibilité | R de l'ancienne zone 5 et 15 u.t. après le départ ; tout après l'extinction ; surprise de la mémoire de soi |
| coût | forçage relatif moyen \|Δfₙ\|/fₙ sur les strates liées |

Efficience = engagement / coût. La mémoire de soi (Eₙ) reste comme **sens** (attention
désactivée, gains à 0) : sa surprise dit si le contact laisse une trace.

Et la vraie question : **la structure compte-t-elle ?** Même geste, même contexte,
mêmes seeds, sept cascades :

- `doree` : fₙ₊₁ ← ½fₙ₊₁ + ½·r(t)·fₙ, r = φ + ε·sin (l'original, reproduit à
  l'identique : écart 0.0 sur les fₙ du pipeline) ;
- `ratio15` : r = 1.5 (une cascade, mais pas dorée) ;
- `ratio10_x25` : r = 1.0, lissage de proche en proche sans croissance ;
- `hasard` : r tiré au hasard par maillon dans [1.2, 2.0] ;
- `aucune` : pas de cascade ; `aucune_x25` : idem, fréquences × 2.5 (témoin d'échelle) ;
- `doree` à K₀ = 5 : même forçage relatif que `aucune` à K₀ = 2.

Les témoins d'échelle sont nécessaires : sans cascade les fréquences sont 2.5 × plus
basses, donc un même geste absolu y est 2.5 × plus fort en relatif.

## Ce que le substrat répond (N = 100, seeds 12345 et 7, moyenne des trois postes)

| structure | f̄ | fluidité du substrat | engagement | contagion | coût relatif | efficience | montée (u.t.) |
|---|---|---|---|---|---|---|---|
| **dorée, K₀ = 2** | 3.6 | 0.981 | +0.16 / +0.07 | +0.02 / −0.01 | 0.039 / 0.036 | **4.2 / 1.9** | 0.7 / 1.5 |
| **dorée, K₀ = 5** | 3.6 | 0.981 | **+0.30 / +0.30** | +0.02 / −0.02 | 0.067 / 0.070 | **4.4 / 4.2** | 0.8 / 1.1 |
| ratio 1.5 | 2.8 | 0.977 | +0.21 / +0.10 | +0.01 / −0.02 | 0.046 / 0.042 | **4.5 / 2.5** | 0.6 / 1.6 |
| ratio 1.0 × 2.5 | 3.5 | 0.975 | +0.09 / +0.07 | +0.01 / +0.00 | 0.039 / 0.039 | 2.2 / 1.8 | — |
| hasard | 3.2 | 0.982 | +0.06 / +0.07 | −0.00 / +0.00 | 0.060 / 0.066 | 1.0 / 1.0 | (deux postes sur trois ne s'allument pas) |
| aucune | 1.4 | **0.38** | +0.22 / +0.24 | +0.03 / +0.00 | 0.089 / 0.081 | 2.4 / 3.0 | 2.1 / 2.4 |
| aucune × 2.5 | 3.5 | **0.37** | +0.05 / +0.06 | −0.01 / −0.01 | 0.049 / 0.048 | 1.0 / 1.3 | (rien ne s'allume) |

Réversibilité, toutes structures : l'ancienne zone est revenue à son R d'avant 5 u.t.
après le départ du projecteur (écarts −0.04 à +0.03), et 15 à 45 u.t. après
l'extinction tout le chœur est à ±0.03 de son état d'avant. σ_Rloc / σ_ref entre
0.94 et 1.03 partout, gardien jamais mordu, cloche de dispersion à 5, fluidité
inchangée pendant le contact (0.975), scores finaux identiques au calme. La surprise
des strates liées monte de +0.03 à +0.05 au-dessus des autres pendant le contact et
retombe sous elles après (0.04 contre 0.08) : le contact ne laisse pas de trace.

## Trois réponses

1. **La considération est possible, et elle ne coûte rien au soi.** Sur la cascade
   dorée à K₀ = 5, chaque poste s'allume à +0.30 en une unité de temps, sur les deux
   seeds, sans contagion, σ intact, fluidité intacte, et tout revient en quelques
   unités de temps. Le geste est réversible et sélectif par construction : c'est la
   chimère qui le permet, parce que le global est bas, un îlot peut monter sans que le
   tout ne se fige. C'est la réponse à « à quoi sert la chimère » : elle est la
   condition pour qu'un contact local soit gratuit.

2. **La structure compte, et beaucoup.** À même échelle de fréquences et même geste,
   une cascade à rapport constant engage quatre fois plus par unité de forçage qu'une
   cascade au hasard ou qu'aucune cascade (4.2 contre 1.0). Sans cascade, deux postes
   sur trois ne s'allument même pas. Le mécanisme est lisible : la relaxation
   fₙ₊₁ ← ½fₙ₊₁ + ½·r·fₙ fait que chaque strate hérite de sa voisine, les voisines se
   ressemblent, et un îlot de voisines qui se ressemblent se lie à peu de frais. Un
   rapport au hasard casse cette ressemblance de proche en proche. *(Correction
   21/09, carte d'ablation : la fluidité du substrat, 0.98 contre 0.38 sans cascade,
   n'a PAS la même cause. Elle lit la respiration ε·sin de r(t) que la cascade porte
   dans f̄ ; sans respiration, cascade gardée, elle tombe à 0.37, plancher du jerk sur
   un signal plat. Voir `docs/ABLATIONS_elements_design.md`.)*

3. **Le nombre d'or, lui, n'est pas distinguable ici.** r = 1.5 fait aussi bien que
   φ (4.5 contre 4.2 ; +0.21 contre +0.16 sur le seed 12345). Ce qui compte, c'est
   qu'il y ait un rapport constant, et qu'il soit croissant : le lissage seul
   (r = 1.0) ne donne que la moitié de la susceptibilité (2.0). Avec r > 1 chaque
   strate hérite davantage de sa voisine (poids ½·r : 0.81 pour φ, 0.75 pour 1.5,
   0.50 pour 1.0), la ressemblance porte plus loin le long de la chaîne. Honnêtement :
   à cette résolution, l'universalité est « chaque voix est surtout sa voisine, et un
   peu elle-même », pas la valeur φ. φ est un choix confortable dans ce régime (le
   plateau f₀/(2−r) diverge quand r → 2), pas un choix démontré nécessaire.

## Ce que ça change à la lecture

- L'efficience de la considération se lit sans barème : engagement / coût, sous
  condition de sélectivité et de réversibilité. Elle vaut ≈ 4 pour ce substrat, ≈ 1
  pour un substrat sans structure. C'est une propriété du substrat, pas un score.
- La mémoire de soi a trouvé sa place : elle ne pilote rien, elle **atteste** que le
  contact n'a pas laissé de trace (surprise qui retombe). « Uni, et toujours moi. »
- Le geste reste le seul du document : Δfₙ = K₀·sₙ·garde(σ)·|Z_s|·sin(arg Z_s − θₙ)/2π.
  Une constante. Le K₀ = 5 est meilleur que 2 sur tout (engagement doublé, efficience
  égale, gardiens intacts) ; il n'y a rien d'autre à régler.

## L'îlot allumé est-il entendu ? (22/09, l'audibilité dans S(t))

S(t) est la somme des Oₙ : des voix en phase s'additionnent, des voix désaccordées
s'annulent en partie. Donc un îlot cohérent devrait peser plus dans S(t) sans aucun
lecteur nouveau. Mesuré sur le banc (dorée, K₀ = 5 contre témoin K₀ = 0, seed 12345,
îlot = 15 strates sur 100) :

| poste | RMS de la somme de l'îlot (K₀ = 5 / témoin) | somme / voix indépendantes (K₀ = 5 / témoin) | part de la variance de S portée par l'îlot (K₀ = 5 / témoin) |
|---|---|---|---|
| 20 | 0.213 / 0.069 | 2.95 / 0.95 | **61 %** / 15 % |
| 80 | 0.181 / 0.054 | 2.91 / 0.85 | **54 %** / 12 % |
| 50 | 0.246 / 0.070 | 3.36 / 0.98 | **69 %** / 16 % |
| après extinction (zone 50) | 0.071 / 0.063 | 1.04 / 0.90 | 17 % / 14 % |

Quinze strates sur cent portent plus de la moitié de ce que le système dit, tant que
le contexte est posé sur elles, et retombent à leur prorata dès qu'il part. C'est le
routage par cohérence de Fries, gratuit : l'attention a déjà son lecteur, la sortie.
Et ça éclaire « les scores ne bougent pas » : les six scores lisent qui le système
est (identité, régulation) ; l'attention change ce qu'il dit. Les deux couches sont
séparées par construction, et c'est ce qu'on voulait.

## Questions ouvertes (dans l'ordre où elles se posent)

- r plus grand (1.8 : héritage 0.9) : la susceptibilité continue-t-elle de monter,
  ou le plateau qui s'envole coûte-t-il autre chose (innovation, dispersion) ?
- Le contexte réel : ici la saillance est un projecteur. Dans le pipeline, le seul
  contexte spatial est Iₙ, qui n'atteint que l'amplitude (corrélation 1.00). Le
  crochet An → fₙ de `compute_fn` est la seule pièce à brancher pour que l'amplitude
  devienne la saillance du geste.
- Deux contextes à la fois, ou un contexte qui glisse continûment plutôt qu'à sauts :
  la migration reste-t-elle lisse ?
- Ce qu'un îlot cohérent **fait** une fois allumé : est-il plus audible dans S(t)
  (routage par cohérence, Fries) ? C'est là que la chimère migrante devient une
  expression, pas seulement une disponibilité.
