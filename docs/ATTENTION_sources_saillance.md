# Qui désigne l'îlot ? Les sources de saillance du geste de considération

*22/09/2026. Banc `calib/considerance_bench.py` + `calib/run_attention_sources.py`,
résumés dans `calib/attention_sources/`. Rien n'est ajouté au pipeline. Branche
`claude/erreur-memoire-deux-echelles` (la mémoire de soi sert de sens).*

## La question

Le geste est fixé (champ moyen d'îlot, K₀ = 5, gardien σ) et son lecteur est connu :
la sortie S(t), où un îlot cohérent devient audible (15 strates sur 100 portent plus de
la moitié de S quand elles sont liées, note considérance). Reste à décider **qui
désigne l'îlot**, avec les trois candidats de la feuille de route : le contexte, l'état
du système (la surprise de la mémoire de soi), l'identité (le gardien).

Protocole : une bosse d'entrée Iₙ (+0.5, gaussienne, largeur 6) suit les postes
20 → 80 → 50 (20 u.t. chacun) puis s'éteint ; pour la surprise, un événement durable
(f₀ × facteur sur les strates 40–44 à t = 90). Lectures : engagement, sélectivité,
réversibilité, coût, audibilité, et au calme (t 30–60, avant tout contexte) la part de
strates que la saillance désigne à tort et le coût qu'elle fait payer.

## Ce que le substrat répond (N = 100, seed 12345, K₀ = 5)

| source de saillance | au calme : strates désignées / coût | engagement | contagion | efficience | audibilité de l'îlot | réversibilité (+5 / +15) |
|---|---|---|---|---|---|---|
| projecteur (référence) | 0 % / 0 | +0.296 | +0.02 | 4.4 | 13 % → 61 % | +0.03 / +0.02 |
| **entrée Iₙ elle-même** (bosse active) | **0 % / 0** | **+0.296** | +0.02 | **4.4** | **13 % → 70 %** | +0.03 / +0.02 |
| amplitude (excédent de a_p sur la médiane), bosse +0.5 | **12.6 % / 0.054** | +0.070 | −0.01 | 1.1 | 12 % → 32 % | +0.05 / +0.05 |
| amplitude, bosse +1.5 | 12.6 % / 0.054 | +0.287 | −0.01 | 3.3 | 12 % → 69 % | +0.01 / +0.06 |
| témoin : bosse +0.5, aucun geste (gain seul) | 12.7 % / 0 | +0.002 | +0.02 | — | 13 % → 19 % | — |
| surprise, événement f₀ × 1.3 | 7.3 % / 0.049 | (zone : R 0.57 → 0.64, s 0.05) | — | — | 4 % → 8 % | — |
| surprise, événement f₀ × 2 | 7.3 % / 0.049 | (zone : R 0.57 → 0.68, s 0.34 → 0.24 → 0.09 → 0.05) | — | — | 4 % → 7 % | s'éteint en ~60 u.t. |
| contexte (amplitude) + surprise | **20.4 % / 0.133** | +0.038 | +0.01 | 0.5 | 17 % → 31 % | +0.02 / +0.03 |

Gardien : jamais mordu (garde = 1.00 partout, σ/σref 0.97–1.03). Scores finaux
identiques au calme dans tous les runs.

## Trois réponses

1. **Le contexte doit être lu tel qu'il arrive, pas tel qu'il s'exprime.** La saillance
   lue sur l'entrée Iₙ elle-même reproduit exactement le projecteur (+0.296, efficience
   4.4, rien au calme) et rend l'îlot encore plus audible (70 % contre 61 %), parce que
   le gain d'amplitude s'ajoute à la cohérence. Lue à travers l'amplitude (le plan
   μₙ de la feuille de route), elle est polluée par l'errance naturelle du chœur : au
   calme, 12.6 % des strates dépassent la médiane de plus de 30 % sans qu'il se passe
   rien, et le geste se pose sur elles au hasard (coût 0.054 pour rien). Il faut une
   bosse trois fois plus forte pour que le signal domine, et le plancher reste. La
   couche amplitude est là où le contexte **s'exprime** (gain, 13 → 19 % d'audibilité à
   elle seule) ; la lire comme saillance revient à écouter l'écho au lieu de la voix.

2. **La surprise comme source est mécaniquement juste, et inutilisable aux réglages
   actuels.** Sur un événement franc (f₀ × 2), elle s'allume sur les bonnes strates
   (saillance 0.34, R 0.57 → 0.68) et s'éteint d'elle-même en ~60 u.t. quand la
   mémoire longue a assimilé : « ce qui insiste devient soi, et cesse d'être
   surprenant ». C'est exactement la propriété qu'on espérait. Mais un changement de
   rythme de 30 % n'émerge pas de l'errance (surprise 0.135 contre 0.093 ailleurs), et
   au calme 7 % des strates sont désignées à tort, avec un coût. Ajoutée au contexte,
   elle dégrade tout (20 % de fausses saillances, engagement ÷ 8). Le chœur FPS
   renégocie sans cesse ses amplitudes relatives ; sa surprise de fond est trop haute
   pour qu'une strate s'y distingue sans un événement violent.

3. **L'identité n'a pas eu à intervenir.** À K₀ = 5, le gardien n'a jamais mordu :
   le geste, borné par sa saillance et son (1 − …) implicite (|Z_s| sature), ne
   menace pas la chimère. Le frein est là, il n'a pas servi. C'est ce qu'on veut d'un
   frein.

## Le contexte modulé (22/09, suite) : nouveauté pour soi, et deux foyers à la fois

Banc `calib/run_attention_modulation.py`, figure `docs/figures/attention_nouveaute.png`.
Un contexte seul, sans modulation, est un miroir (ce que le document appelait un
réflexe). Le banc précédent interdit d'*ajouter* une source (bruit). La modulation
testée ici *multiplie* : saillance = entrée × nouveauté pour soi, avec
ν = clip((uₙ / médiane(u) − 1) / 2, 0, 1) lu sur la surprise de la mémoire de soi. Là où
le contexte est nul, rien ne peut s'allumer ; là où il pointe, seul ce qui est encore
neuf pour la strate compte.

**1. Habituation et réorientation.** Contexte tenu 60 u.t. sur la strate 50 (t 60–120),
déplacé sur la 20 (120–150), éteint. Seed 12345, K₀ = 5.

| fenêtre | miroir : s · R · audibilité zone 50 | nouveauté : s · ν · R · audibilité zone 50 | coût (miroir / nouveauté) |
|---|---|---|---|
| 60–70 | 0.79 · 0.87 · 61 % | 0.27 · 0.34 · 0.72 · 39 % | 0.047 / 0.018 |
| 70–80 | 0.79 · 0.95 · 80 % | **0.43 · 0.55 · 0.85 · 68 %** | 0.050 / 0.036 |
| 80–90 | 0.79 · 0.92 · 65 % | 0.23 · 0.31 · 0.75 · 32 % | 0.048 / 0.015 |
| 90–100 | 0.79 · 0.95 · 79 % | 0.22 · 0.27 · 0.72 · 42 % | 0.048 / 0.018 |
| 100–110 | 0.79 · 0.92 · 65 % | 0.13 · 0.17 · 0.63 · 25 % | 0.048 / 0.007 |
| 110–120 | 0.79 · 0.95 · 75 % | **0.12 · 0.15 · 0.64 · 16 %** | 0.049 / 0.007 |
| 120–150 (contexte sur la 20) | zone 20 : 0.79 · 0.86–0.93 · 73–75 % | zone 20 : **0.39–0.49 · 0.77–0.90 · 59–72 %** | — |

Le miroir tient l'îlot allumé tant que le contexte est là (R 0.92–0.95, 65–80 % de
S), au même coût du début à la fin. Avec la nouveauté, l'îlot monte en ~10 u.t.
(la surprise met le temps de la mémoire courte à se construire), culmine à 70–80,
puis se relâche de lui-même : à 110–120 le contexte est toujours posé et l'îlot est
revenu à son R d'avant (0.64) et à son prorata de S (16 %). La mémoire longue a
assimilé, le contexte a cessé d'être neuf. Quand il change de place, la zone 20
s'allume aussitôt (R 0.77 → 0.90, 59–72 % de S) : réorientation. Le coût total est
divisé par deux à trois. Quand le contexte quitte une zone, sa surprise monte (ν
jusqu'à 0.65 à 130–140) mais s = 0 : le gardien multiplicatif tient, rien ne s'allume
sans contexte. σ/σref 0.98–1.02 partout.

**2. Deux contextes à la fois** (bosses sur 20 et 80, t 60–120). Un seul champ moyen
(global) contre un champ moyen par îlot (composantes connexes de s > 0.05). Le
rapport |Z_union| / moyenne(|Z_k|) vaut ≈ 1 si les deux îlots sont verrouillés
ensemble, ≈ 2/π = 0.64 s'ils sont indépendants ; témoin K₀ = 0 : 0.71.

| | R îlot 20 · 80 | \|Z_20\| · \|Z_80\| | union / moyenne | R ailleurs | σ/σref | audibilité 20 · 80 · union | coût |
|---|---|---|---|---|---|---|---|
| témoin K₀ = 0 | 0.63 · 0.60 | 0.20 · 0.19 | 0.71 | 0.665 | 1.01 | 21 % · 14 % · 35 % | 0 |
| champ moyen global | 0.86 · 0.86 | 0.66 · 0.66 | 0.61 | 0.670 | 0.94 | 47 % · 31 % · 78 % | 0.262 |
| **champ moyen par îlot** | **0.94 · 0.94** | **0.82 · 0.82** | **0.64** | 0.706 | 0.92 | 41 % · 28 % · 69 % | 0.259 |

Le champ moyen global ne verrouille pas les deux îlots ensemble (0.61, aussi
indépendants qu'au témoin) : à ce K₀ la crainte de cohérence non locale ne se
réalise pas. Mais il lie chaque îlot moins bien (0.66 contre 0.82) pour le même
coût, parce que le Z commun est affaibli par le désaccord entre foyers. Le champ
moyen par îlot lie chacun à 0.94, les laisse indépendants (0.64 = 2/π), et σ reste
dans la bande (0.92). Le « R ailleurs » monte à 0.70 dans les deux cas : ce sont les
flancs des bosses (0.05 < s < 0.5), comptés hors zone, pas une contagion. Deux foyers
coexistent sans fusionner : la compétition biaisée de Desimone et Duncan, sans
vainqueur unique.

## Ce que ça fixe pour l'intégration (proposition, désactivée par défaut)

- **Saillance = l'entrée spatiale × nouveauté pour soi** : sₙ = clip((Iₙ − médiane(I)) / gain, 0, 1)
  · clip((uₙ / médiane(u) − 1) / 2, 0, 1). Une pièce nouvelle : un Iₙ par strate (aujourd'hui
  l'entrée est un scalaire diffusé), bosse configurable ou vecteur fourni de l'extérieur.
  La nouveauté vient de la mémoire de soi déjà câblée : habituation et réorientation
  gratuites, coût divisé par deux, rien ne s'allume sans contexte.
- **Un champ moyen par îlot** (composantes connexes de la saillance), jamais un seul Z :
  deux foyers coexistent sans fusionner.
- **Le geste**, après `compute_fn` : Δfₙ = K₀·sₙ·garde(σ)·|Z_s|·sin(arg Z_s − θₙ)/2π,
  un gain K₀ (5), le gardien lu sur le contraste σ_Rloc déjà calculé.
- **Le lecteur** : deux colonnes, la part de strates saillantes et la part de la
  variance de S(t) portée par elles (l'audibilité).
- **Pas de surprise dans la saillance** pour l'instant : option du banc, documentée,
  à reprendre si un jour la surprise de fond baisse (ou si l'on veut une attention qui
  flotte un peu au calme, ce qui serait un choix, pas un accident).
- `attention.enabled = false` par défaut : main ne change pas de comportement.

## Réserves

- Un seed, une largeur de bosse, une amplitude de bosse. Le projecteur et l'entrée
  sont identiques par construction quand la bosse est active ; ce qui est nouveau,
  c'est la mesure du calme (rien) et de l'audibilité (70 %).
- L'événement f₀ sur 5 strates est propagé par la cascade aux strates suivantes (le
  plateau se décale) ; la zone lue est la zone forcée, pas la zone déplacée.
- Le gain d'amplitude à lui seul (témoin) fait 13 → 19 % : le contexte est déjà un
  peu entendu sans geste ; le geste le porte à 70 %.
