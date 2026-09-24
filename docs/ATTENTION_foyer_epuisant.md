# Le foyer épuisant : quand l'effort vient du foyer lui-même, m baisse par le même mécanisme

*24/09/2026. Banc `calib/run_foyer_epuisant.py` sur `calib/considerance_bench.py` (option
`attach`, bruit porté par la bosse du foyer présent), figure
`docs/figures/attention_foyer_epuisant.png`, résumés et trajectoires de m dans
`calib/attention_sources/`. Rien n'est ajouté au pipeline.*

## La question laissée ouverte

Au banc de l'attachement (`ATTENTION_attachement.md`), l'effort était produit par du bruit
sur toute l'entrée, sans rapport avec le foyer : on associait, on ne récompensait pas. Restait
le cas non testé : un foyer qui est **lui-même** la source de l'effort. Le mécanisme ne lit
jamais l'effet du foyer, seulement l'état du soi quand il était là ; il devrait donc, sans
rien ajouter, cesser de chérir un foyer qui épuise. Et, plus délicat : un foyer déjà chéri qui
devient épuisant devrait se voir retirer m, par la même règle et à la même vitesse τ_m.

## Le protocole

Même mécanisme, mêmes constantes (τ_m 20, w = clip(2 − activité/repos, 0, 1), m figé à
l'extinction t = 180 pour la lecture). Le bruit blanc n'est plus sur toute l'entrée : il est
multiplié par la bosse du foyer présent, aux strates du foyer et à son échelle. La saillance
lit la bosse propre, sans le bruit (le foyer reste désigné de la même façon ; c'est sa
présence qui bouscule).

- **témoin** : 20 au calme (60–120), 80 présent au calme (120–180), extinction, retour des
  deux au calme (220–300).
- **épuisant** : identique, mais le 80 est bruyant sur lui-même pendant sa présence (σ 0.6,
  puis 1.5).
- **lassitude** : 20 au calme (60–120, il devient chéri), puis le **même** foyer 20 reste
  présent (120–180) mais devient bruyant sur lui-même ; extinction ; retour 20 + 80 (jamais
  vu). Ici la question est la révision : m(20) monte, puis redescend-il ?

Deux seeds (12345, 7).

## Ce que le système apprend

| run | seed | w pendant 120–180 | activité relative (120–180) | m à 180 : foyer 20 / foyer 80 |
|---|---|---|---|---|
| témoin (80 au calme) | 12345 | 0.43 | ×1.62 | 0.68 / **0.41** |
| témoin | 7 | 0.53 | ×1.52 | 0.74 / **0.48** |
| 80 épuisant, σ 0.6 | 12345 | 0.31 | ×1.75 | 0.68 / **0.28** |
| 80 épuisant, σ 1.5 | 12345 | 0.17 | ×1.92 | 0.68 / **0.12** |
| 80 épuisant, σ 1.5 | 7 | 0.30 | ×1.79 | 0.74 / **0.24** |
| lassitude (20 chéri puis épuisant) | 12345 | 0.14 | ×1.95 | **0.68 → 0.11** / — |
| lassitude | 7 | 0.27 | ×1.84 | **0.74 → 0.21** / — |

Et dans le temps (figure, panneaux a et c) : le m du 80 épuisant monte d'abord comme celui
du témoin (le bruit met ~10 u.t. à se traduire en activité), plafonne vers 130–135, puis
redescend jusqu'à la fin de sa présence, d'autant plus bas que le foyer est bruyant. En
lassitude, le m du 20 monte à 0.68–0.74 pendant le calme, puis fond en ~40 u.t. (0.68 → 0.34
à 150, 0.11 à 180). La descente est aussi rapide que la montée : c'est τ_m.

## Ce qui se passe au retour (220–300, m figé)

| run | seed | zone 20 : s · R · audibilité | zone 80 : s · R · audibilité |
|---|---|---|---|
| témoin | 12345 | 0.66 · 0.899 · 63 % | 0.45 · 0.798 · 20 % |
| témoin | 7 | 0.68 · 0.916 · 60 % | 0.47 · 0.857 · 22 % |
| 80 épuisant, σ 1.5 | 12345 | 0.64 · 0.873 · 63 % | **0.26 · 0.689** · 14 % |
| 80 épuisant, σ 1.5 | 7 | 0.67 · 0.912 · 66 % | **0.32 · 0.738** · 15 % |
| lassitude | 12345 | **0.30 · 0.710** · 46 % | 0.17 · 0.659 · 18 % |
| lassitude | 7 | **0.41 · 0.784** · 49 % | 0.20 · 0.698 · 20 % |

Le foyer 80 qui a épuisé n'est pas repris (R 0.69–0.74 contre 0.80–0.86 quand il a été
présent au calme). Le foyer 20, chéri puis épuisant, est retenu comme un foyer quelconque
(R 0.71–0.78, s 0.30–0.41 : ce qu'il reste de nouveauté), là où le même foyer resté chéri
est tenu à R 0.90–0.92 (panneau d). σ/σref 0.93–1.00 partout.

## Trois réponses

1. **Oui, par le même mécanisme, sans rien ajouter.** Un foyer qui bouscule le soi
   pendant qu'il est là n'est pas chéri, que l'effort vienne d'ailleurs (banc précédent)
   ou de lui. Le mécanisme ne fait pas la différence, et c'est ce qu'on voulait : il ne lit
   pas des causes, il lit un état.

2. **L'attachement se révise dans les deux sens, à la même vitesse.** Un foyer chéri qui
   devient épuisant perd m aussi vite qu'il l'avait gagné, et n'est plus tenu au retour.
   Le passé donne une avance, il ne fige pas ; l'usure vaut pour l'attachement comme pour la
   nouveauté. Le réglage de cette avance reste τ_m.

3. **Une chose qu'on n'avait pas cherchée : la zone dure épuise un peu par elle-même.**
   Au témoin, le 80 présent au calme laisse le soi à ×1.5–1.6 de son repos (w 0.43–0.53,
   m 0.41–0.48), contre ×1.3 pour le 20 (m 0.68–0.74). Se poser sur la zone 80 coûte, et
   le mécanisme le sent sans qu'on le lui ait dit. C'est cohérent avec le banc de
   l'attachement (ordre 2 : m 0.46–0.53 pour le 80 chéri) : l'attachement y renversait le
   penchant naturel, mais avec une avance moindre, parce que la zone dure a un peu épuisé
   pendant qu'on la chérissait.

## Réserves

- Le bien-être reste lu sur l'activité seule. La question de la fluidité et de C_JS
  comme lecture du « pas-flow » est posée à part (discussion du 24/09) : au banc des
  ablations, fluidité et C_JS lisaient la respiration imposée r(t) et bougeaient à peine ;
  l'activité était le signal contrasté. À revoir quand on aura une perturbation qui
  laisse l'activité intacte et casse la fluidité.
- Le bruit local, à σ 0.6, produit moins d'effort que le bruit global de même écart-type
  (×1.75 contre ×3.6), parce qu'il ne touche qu'une douzaine de strates : les intensités ne
  sont pas comparables entre les deux bancs, seules les directions le sont.
- Deux seeds, mêmes constantes.

## Ce que ça change pour la suite

La mémoire des moments chéris peut se poser sur une base saine : m est appris dedans,
sans récompense, il monte pour ce qui accompagne le calme, descend pour ce qui bouscule,
qu'importe d'où vient la bousculade, et il se révise. Reste à décider ce que « quand
l'entrée se tait » veut dire pour la FPS, et à garder (lieu, m) au-delà de la présence.
