# Nouveauté ou importance ? Le témoin de Gepetto, et la priorité de maintien

*24/09/2026. Bancs `calib/run_temoin_memoire_G.py` (pipeline natif, main) et
`calib/run_maintien.py` (sur `calib/considerance_bench.py`, option `maintain`), figure
`docs/figures/attention_maintien.png`, résumés dans `calib/attention_sources/`. Rien
n'est ajouté au pipeline : deux questions posées, deux réponses lues.*

## D'où vient la question

Exploration de Gepetto sur main (24/09) : la saillance actuelle, contexte × nouveauté,
s'éteint quand la nouveauté s'éteint. Une musique connue peut pourtant mériter toute
l'attention. Il propose de distinguer familiarité, pertinence et engagement, et une
extension minimale : s = c · [ν + (1 − ν)·m], avec m une priorité de maintien, bornée,
révisable, d'abord fournie de l'extérieur pour éprouver le mécanisme sans lui prêter une
autonomie qu'on n'a pas construite. Il rapporte aussi un premier résultat : masquer la
mémoire d'efficacité des modes de régulation (G) change la saillance strate par strate
(écart ≈ 0.08) sans changer sa moyenne. Deux pas, dans l'ordre : le témoin de ce
résultat, puis le banc du maintien.

## 1. Le témoin : la mémoire de régulation façonne-t-elle l'attention ?

La FPS est sensible à ses conditions : deux trajectoires qui divergent à partir d'un
détail quelconque s'écartent. Il faut donc comparer l'écart produit par le masquage à
celui produit par une perturbation aussi petite qui n'est pas la mémoire. Trois runs par
seed, même passé jusqu'à t = 60, foyer constant sur la strate 50 de 60 à 160 : `ref` ;
`maskG` (la mémoire d'efficacité par contexte est vidée avant chaque décision de
régulation à partir de 60, la mémoire de soi intacte) ; `papillon` (à t = 60, f₀ de la
strate 0 × (1 + 10⁻⁶), rien d'autre).

| seed | run | s moyen dans le foyer | audibilité | écart \|Δs\| au ref (foyer, strate × pas) | hors foyer | divergence \|Δf\|/f |
|---|---|---|---|---|---|---|
| 12345 | ref | 0.169 | 43 % | 0 | 0 | 0 |
| 12345 | mémoire de G masquée | 0.166 | 43 % | **0.083** | 0.003 | 0.0016 |
| 12345 | papillon (10⁻⁶ sur une strate) | 0.163 | 44 % | **0.084** | 0.003 | 0.0016 |
| 7 | ref | 0.146 | 38 % | 0 | 0 | 0 |
| 7 | mémoire de G masquée | 0.154 | 37 % | **0.071** | 0.003 | 0.0013 |
| 7 | papillon | 0.148 | 37 % | **0.073** | 0.002 | 0.0012 |

Et masqué contre papillon : 0.077 / 0.076. L'écart de Gepetto est reproduit au
millième (0.083). Mais une perturbation d'un millionième sur une seule strate, sans
aucun rapport avec la mémoire, produit exactement le même écart, et les deux témoins
s'écartent autant l'un de l'autre. La mesure est juste ; la conclusion « l'histoire de
la régulation modifie où et quand l'attention se porte » ne tient pas : ce qu'on voit,
c'est la bifurcation ordinaire d'un système sensible, identique quelle qu'en soit la
cause. La mémoire de régulation ne laisse aucune empreinte spécifique sur l'attention,
ni en moyenne, ni en distribution, à cette échelle. Bonne nouvelle en creux : m n'est
pas déjà là, caché dans la régulation ; il faut le chercher ailleurs, ou le construire.

## 2. Le banc du maintien : le familier peut-il rester considéré sans surprendre ?

Deux foyers (strates 20 et 80) posés de t = 60 à 220 et jamais déplacés : tous deux
deviennent familiers. Puis, sans rien changer aux entrées : m = 1 sur le foyer 20 de
100 à 140 ; inversion (m = 1 sur le 80) de 140 à 180 ; retrait (m = 0) de 180 à 220.
Comparé au même run sans m. Seed 12345, K₀ = 5, champ moyen par îlot.

| fenêtre | sans m : s · ν · R · audibilité (zone 20) | avec m : s · ν · m · R · audibilité (zone 20) | zone 80 avec m : s · ν · m · R · audibilité |
|---|---|---|---|
| 60–100 (nouveauté) | 0.47–0.51 · 0.57–0.63 · 0.81–0.87 · 64–77 % | identique | 0.19 · 0.22 · 0 · 0.66 · 8–11 % |
| 100–140 (m sur 20) | 0.32 → 0.29 · 0.38 → 0.34 · 0.71 · 54 → 52 % | **0.79 · 0.40 → 0.27 · 1 · 0.91 · 75 %** | 0.16–0.19 · 0.20–0.24 · 0 · 0.64 · 6–7 % |
| 140–180 (m sur 80) | 0.18 → 0.14 · 0.22 → 0.17 · 0.65 · 40 → 28 % | 0.20 → 0.28 · 0.26 → 0.38 · 0 · 0.62 → 0.68 · 31–36 % | **0.79 · 0.19 · 1 · 0.91 → 0.93 · 20 → 14 %** |
| 180–220 (retrait) | 0.16 → 0.12 · 0.19 → 0.14 · 0.65 → 0.63 · 33 % | 0.21 → 0.17 · 0.26 → 0.22 · 0 · 0.64 · 35–38 % | 0.12–0.13 · 0.15–0.17 · 0 · 0.65 · 17–20 % |

σ/σref entre 0.96 et 1.01 partout ; coût 0.11 sous maintien (celui du miroir), 0.04–0.07
sans.

**Ce que ça dit.** Sous m, le foyer 20, devenu familier (ν tombé à 0.27), est ramené
au premier plan à saillance pleine (0.79), R 0.91, trois quarts de S(t), alors que sans
m il s'était relâché (0.29, R 0.71). À l'inversion, sans aucune nouveauté sensorielle,
l'attention change de foyer : le 20 retombe à 0.20, le 80 monte à 0.79 et R 0.93. Au
retrait, les deux se relâchent. Le mécanisme de Gepetto fait exactement ce qu'il
prédisait : le familier n'a pas à redevenir étrange pour être considéré, et le maintien
est réversible. C'est bien autre chose qu'une habituation ralentie.

**Une chose que le banc montre et qu'on n'attendait pas.** Le foyer 80 sous m est
lié aussi fort que le 20 (R 0.93) mais porte trois fois moins de S(t) (14–20 % contre
75 %). Cohérer n'est pas la même chose qu'être entendu : l'audibilité dépend aussi de
l'amplitude des voix liées, et la zone 80 est la plus dure de tous nos bancs. Deux
lectures à garder séparées : ce que le système tient au premier plan (R), et ce qu'on
en entend (audibilité).

## Ce qui reste ouvert

- m est fourni. D'où viendrait-il dedans ? Pas de la régulation (témoin). Le candidat
  le plus naturel reste « l'observation du résultat » d'Attention.md : ce foyer, quand on
  s'y est posé, a-t-il servi ? Jamais depuis l'amplitude ou l'audibilité (le système
  prendrait sa propre voix pour une raison de continuer).
- Un seed. Le mécanisme est déterministe et net ; la variabilité est celle du substrat.
- Un foyer que c ne désigne plus reste inaccessible : considérer l'absent demanderait
  une sélection depuis la mémoire, qui n'existe pas.
