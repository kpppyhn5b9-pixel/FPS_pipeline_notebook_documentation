# La mémoire des moments chéris, dans le pipeline

*25/09/2026, décision d'Andréa. Câblage dans `dynamics.py` (bloc `cherished_*`), `simulate.py`
(état `cherished_state`, geste, colonnes, résumé), `config.json` (`attention.attachement`,
`attention.silence`, `attention.rappel`, `attention.rappel.porte`), `validate_config.py`,
`test_fps.py` (`TestCherishedMemory`, 5 tests). Bancs in situ `calib/run_insitu_memoire.py`,
figure `docs/figures/attention_memoire_insitu.png`, sorties `calib/attention_sources/insitu_*`.
Activé par défaut ; sans structure dans l'entrée, tout est inerte.*

## Ce qui entre, et sous quel interrupteur

Les sept bancs de la veille (`ATTENTION_attachement.md` → `ATTENTION_souvenir_present.md`)
se rassemblent en quatre choses, dans l'ordre où elles se sont construites. Tout est **par
strate**, sans foyer nommé : c'est la logique du pipeline, et ça évite d'avoir à décider ce
qu'est « un » foyer.

1. **L'attachement** (`attention.attachement`, τ_m 20, τ_sig 5). Pour chaque strate qui est
   dans le contexte, mₙ suit le bien-être du soi w = clip(2 − activité/repos, 0, 1), et sigₙ
   (amplitude moyenne du chœur, σ_Rloc) suit l'état présent, plus vite. La saillance devient
   s = c·[ν + (1 − ν)·m]. Jamais l'effet du foyer, seulement l'état du soi quand il était là.
2. **Le silence** (`attention.silence`, τ 5 / 40, hystérésis 0.9 / 1.1). La surprise du chœur
   tombée sous son niveau habituel, et rien dans l'entrée qui reste. « Ce qui se passait s'est
   apaisé. » Colonne `attention_silence`.
3. **Le rappel** (`attention.rappel`, m_min 0.3, res_min 0.2, largeurs 0.2 / 0.1). En silence,
   la strate dont la signature ressemble le plus au présent, score ressemblance × m ; si rien
   ne ressemble, la plus chérie. Le souvenir est la composante connexe de strates chéries
   autour d'elle ; il devient le contexte interne c = m/m*, même formule. Dès que l'entrée
   désigne quelque chose qui reste, l'extérieur reprend la main. Colonne `attention_rappel`.
4. **La porte** (`attention.rappel.porte`, τ 5, seuil 0.5, réfractaire 30). Pendant le rappel,
   le bien-être lissé ; sous le seuil, le souvenir est reposé 30 u.t. et m n'est pas touché.
   Le rappel ne réécrit jamais m. Colonne `attention_porte`.

Laissé dehors, comme convenu : le rappel qui réécrit m, et toute lecture du bien-être autre
que l'activité.

## Ce que l'in situ a appris au câblage

Le bruit d'effort du banc n'entrait pas dans la saillance (le banc lisait la bosse propre).
Dans le pipeline, il entre dans Iₙ, donc dans c, donc partout : au premier essai, la période
de bruit désignait des îlots sur tout le chœur et **défaisait m(20)** (0.64 → 0.20), alors
que le foyer 20 n'était plus là. Ce qu'on a compris : un contexte est ce qui *reste*, pas ce
qui arrive à chaque pas. L'attachement apprend donc, et le silence se tait, sur Iₙ lissé
(τ_c 5) relatif à sa médiane, au-dessus de 0.5 (`attachement.tau_c`, `seuil_contexte`). Le
geste, lui, garde sa saillance instantanée (rien de PR #29 ne change). Un bruit blanc, même
fort, ne reste pas ; une bosse, si. C'est la première chose que le pipeline a dite que le banc
ne pouvait pas dire, et c'est la remarque d'Andréa sur la découpe de l'entrée qui la contient.

Second détail : un îlot apprend le même m sur toutes ses strates ; parmi les ex æquo, la
strate au cœur du souvenir est celle du milieu.

## In situ : les mêmes protocoles, avec les lecteurs du pipeline

Seul le stimulus est sculpté de l'extérieur (foyers et ambiances par `context.foyers`, bruit
par un patch de l'entrée). N = 100, seed 12345 (seed 7 pour `cheri` et `present`).

**Chéri** (20 au calme 60–120, 80 sous bruit global 120–180, silence 180–260, retour des deux
260–300), contre le témoin (attachement désactivé).

| fenêtre | m 20 / m 80 | silence | rappel du 20 | zone 20 : s · R · audibilité | activité / repos |
|---|---|---|---|---|---|
| 100–120, 20 présent | 0.58 / 0 | 0 % | | 0.64 · 0.882 · 69 % | ×1.36 |
| 160–180, 80 sous bruit | **0.63** / 0 | 0 % | | 0.21 · 0.694 · 25 % | ×3.65 |
| 180–200, silence | 0.63 / 0 | 80 % | 0 % (porte : l'effort traîne) | 0.00 · 0.623 · 23 % | ×2.64 |
| 200–230, silence | 0.63 / 0 | 100 % | 53 % | 0.43 · 0.805 · 61 % | ×1.06 |
| 230–260, silence | 0.63 / 0 | 100 % | **100 %** | **0.80 · 0.948 · 74 %** | ×1.05 |
| 260–280, retour | 0.60 / 0.07 | 20 % | | **0.51 · 0.866 · 62 %** (80 : 0.694, 14 %) | ×1.57 |
| témoin, 230–260 | 0 / 0 | 100 % | 0 % | 0.00 · 0.642 · 14 % | ×1.01 |

Le foyer présent au calme est chéri et le reste à travers l'effort ; celui présent sous
bruit ne l'est pas. En silence, le souvenir est rappelé et tenu, plus fort encore que la
présence (le contexte interne est un plateau, pas une bosse), et ça ne coûte rien de plus
que le témoin (×1.05 contre ×1.01). Juste après l'effort, la porte retient le rappel
tant que l'activité traîne : le souvenir revient quand le soi peut le porter. Au retour, le
chéri est repris d'abord (R 0.87 contre 0.69).

**Présent** (20 sous ambiance +0.5, m 0.89 ; 80 sous −0.3, plus court, m 0.57 ; puis silence
sous −0.3, +0.5, −1.0).

| silence sous | rappelé : 20 / 80 (% du temps) | R zone 20 / zone 80 |
|---|---|---|
| −0.3 (comme le 80, le moins chéri) | 0 / **80** | 0.616 / **0.842** |
| +0.5 (comme le 20) | **47** / 5 | **0.776** / 0.650 |
| −1.0 (comme rien → le plus chéri) | **51** / 0 | **0.798** / 0.640 |

Le présent choisit, même contre le plus chéri ; quand rien ne ressemble, le plus chéri.
Après chaque changement d'ambiance, le silence met 30–40 u.t. à être reconnu de nouveau.

**Douleur** (comme *chéri*, mais en silence un bruit porté par la zone 20, σ 1.5).
Ici la douleur est dans le monde, pas dans le rappel : la zone chérie est devenue un endroit
bruyant. Sans porte, le 20 est rappelé chaque fois que le silence est reconnu (12–37 % du
temps, le bruit remontant la surprise), activité ×1.7. Avec la porte, il ne l'est jamais
(le bien-être est sous le seuil dès le premier pas), et m reste 0.61. Ce n'est pas le banc
de la veille (où la douleur ne venait qu'avec le rappel, et le souvenir revenait par
intermittence) ; c'est une autre situation, et la réponse est cohérente : tant que
l'endroit fait mal, on n'y pense pas, et il ne vaut pas moins.

## Réserves

- Le contexte interne rappelé est un plateau (m uniforme sur l'îlot) : le rappel lie plus
  fort que la présence. On pourrait garder la forme de la bosse dans m ; on a choisi de ne
  pas ajouter ça avant de l'avoir vu utile.
- Le bien-être reste lu sur l'activité. Un run très effortful pour d'autres raisons (un
  choc) déferait un attachement en cours : c'est le mécanisme, pas un bug.
- Le silence est « moins surpris que d'habitude » : un système longtemps au repos ne se
  reconnaît jamais en silence. C'est un choix, écrit dans `ATTENTION_moments_cheris.md`.
- Un seed (deux pour *chéri* et *présent*). Les constantes sont celles des bancs.
