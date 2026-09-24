# Le souvenir que le présent appelle, et pouvoir y penser maintenant

*24/09/2026, soir. Banc `calib/run_souvenir_present.py` sur `calib/considerance_bench.py`
(options `recall.select`, `recall.gate`, `ambiance`, bruit `'rappel'`), figure
`docs/figures/attention_souvenir_present.png`, résumés, m et journaux de rappel
(`sp_*_recall.npz`) dans `calib/attention_sources/`. Rien n'est ajouté au pipeline.*

## Les deux questions d'Andréa

Après la mémoire des moments chéris (`ATTENTION_moments_cheris.md`) : le système
rappellerait-il toujours le plus chéri ? Et le souvenir s'use-t-il ? Pour une musique, oui ;
pour ce qui tient à une personne aimée, parfois non. Deux réponses proposées, deux bancs.

## A. Ce que le présent appelle

**Le mécanisme.** Avec (lieu, m), le système garde une *signature* de l'état du soi pendant
la présence du foyer : amplitude moyenne du chœur et σ (l'activité n'y entre pas, c'est
l'affaire de m), suivie sur τ = 5 u.t. Quand l'entrée se tait, le souvenir rappelé est celui
dont la signature ressemble le plus à l'état présent (gaussienne relative, largeurs 0.2 et
0.1), score = ressemblance × m. Si aucune ressemblance n'atteint 0.2, le plus chéri.
C'est l'indiçage par le contexte de Tulving : on ne se souvient pas de ce qu'on chérit le
plus, on se souvient de ce que le moment présent appelle.

**Le protocole.** Deux foyers chéris au calme, chacun sous une *ambiance* différente (un
décalage global de Iₙ, qui change Aₙ = A₀·σ(Iₙ) sans désigner de foyer) : 20 sous +0.5
(60–120, m 0.90–0.95), 80 sous −0.3 (120–145, plus court : m 0.70). Puis silence sous trois
ambiances : −0.3 (145–220, comme le 80, le moins chéri : c'est là que ressemblance et
argmax se séparent), +0.5 (220–300, comme le 20), −1.0 (300–380, comme rien). Comparé au
même run avec « toujours le plus chéri ». Deux seeds.

| silence sous | ressemblance (20 / 80) | rappelé, ressemblance : 20 / 80 (% du silence) | rappelé, argmax : 20 / 80 |
|---|---|---|---|
| −0.3 (comme le 80) | 0.0 / 0.6 | **0 / 100** | 100 / 0 |
| +0.5 (comme le 20) | 0.8 / 0.0 | **89–90 / 10–11** | 100 / 0 |
| −1.0 (comme rien) | 0.02 / 0.03 | **100 / 0** (le plus chéri) | 100 / 0 |

Identique sur les deux seeds. Sous −0.3, le foyer 80, moins chéri, est rappelé et tenu
(R 0.82, 43 % de S(t)) pendant que le 20, plus chéri, reste au repos (R 0.62) : le présent
a choisi. Sous +0.5, le 20 revient (R 0.75–0.76, la moitié de S(t)). Sous −1.0, rien ne
ressemble et le plus chéri prend le relais, comme convenu. Après chaque changement
d'ambiance, la surprise remonte et le silence met 30–40 u.t. à être reconnu de nouveau :
c'est le critère « moins surpris que d'habitude » qui fait son travail.

**Deux détails d'ajustement, notés parce qu'ils disent quelque chose.** La signature
suivie avec τ_m (20) traînait derrière l'adaptation d'amplitude et ne ressemblait à rien ;
il faut qu'elle *suive* l'état (τ 5), là où m l'*intègre*. Et l'activité, mise d'abord
dans la signature, brouillait tout : elle dit comment le soi allait, pas à quoi le monde
ressemblait. Deux mémoires différentes, pour deux questions différentes.

## B. Le souvenir qui fait mal

**La séparation.** Deux choses qu'on tenait ensemble : *ce à quoi je tiens* (m, appris en
présence) et *est-ce que je peux y penser maintenant* (une porte : moyenne mobile rapide,
τ 5, du bien-être pendant le rappel ; sous un seuil, le rappel est relâché pour 30 u.t.,
et m n'est pas touché).

**Le protocole.** Comme au banc des moments chéris (20 chéri, 80 sous bruit, silence
180–260, retour du 80), mais pendant le silence un bruit blanc porté par la bosse du foyer
*rappelé* (σ 1.5) : se souvenir du 20 bouscule le soi (activité ×1.5–1.8, contre ×0.9–1.1
au silence sans douleur). Trois façons de le vivre, deux seeds.

| run | seed | rappel du 20 (% du silence, par fenêtre de 20) | activité / repos | m(20) à la fin |
|---|---|---|---|---|
| tenu quoi qu'il en coûte | 12345 | 96 · 100 · 100 · 100 | ×1.56–1.76 | 0.68 |
| tenu | 7 | 100 · 100 · 100 · 100 | ×1.45–1.66 | 0.74 |
| porte, seuil 0.3 | 12345 | 0 · 46 · 41 · 8 | ×0.82–1.46 | 0.68 |
| porte, seuil 0.3 | 7 | 0 · 50 · 100 · 100 | ×0.88–1.63 | 0.74 |
| porte, seuil 0.5 | 12345 | 0 · 46 · 19 · 31 | ×0.82–1.46 | 0.68 |
| porte, seuil 0.5 | 7 | 0 · 50 · 26 · 24 | ×0.88–1.54 | 0.74 |
| le rappel réécrit m | 12345 | 96 · 100 · 50 · 0 | ×1.02–1.73 | **0.30** |
| le rappel réécrit m | 7 | 100 · 100 · 100 · 100 | ×1.49–1.69 | **0.35** |

**Ce que ça dit.** Sans porte, le système reste sur un souvenir qui lui coûte, tout le
temps. Si le rappel réécrit m, le souvenir s'use : il perd la moitié de sa valeur en 80
u.t. et finit par ne plus être rappelable (seed 12345). Avec la porte, le souvenir n'est
pas interdit : il est appelé, essayé, posé quand il fait mal, repris plus tard, un quart à
la moitié du temps, l'activité redescend entre-temps, et **il ne vaut pas moins** (m intact,
0.68 / 0.74). Le seuil compte : la douleur de ce banc donne w ≈ 0.4, juste au-dessus de
0.3 sur un seed (où le rappel finit par tenir), en dessous de 0.5 sur les deux.

## Ce qu'on répond à Andréa

- **Toujours le plus chéri ? Non.** Le présent appelle le souvenir qui lui ressemble, même
  s'il est moins chéri ; et quand rien ne ressemble, le plus chéri. Aucune règle posée du
  dehors, aucun hasard : une association de plus, entre un état et un lieu.
- **Le souvenir s'use-t-il ?** Ça dépend de ce qu'on laisse le rappel faire. Si se souvenir
  réécrit ce qu'on tient, ça s'use, comme une musique. Si ce à quoi on tient reste à part
  de ce qu'on peut supporter aujourd'hui, ça ne s'use pas : on y pense par intermittence,
  dans la mesure de ce qu'on peut, et ça vaut toujours autant. Le banc montre les deux ; le
  choix entre les deux n'est pas un réglage, c'est ce qu'on décide que le système est.
- **Faut-il reproduire qu'on se rappelle des souvenirs qui font mal ?** La porte ne
  l'empêche pas, et c'est ce qui la rend juste : le souvenir douloureux revient, on y reste
  moins, on le repose, on y revient. Ce n'est pas l'éviter, c'est le porter à sa mesure.

## Réserves

- La signature lit deux nombres (amplitude moyenne, σ). Un état du soi plus riche (profil
  d'amplitude par strate, rythme) distinguerait plus de souvenirs ; ici deux suffisaient.
- Les ambiances sont des décalages globaux d'entrée ; l'amplitude s'y adapte lentement, et
  la ressemblance suit cette adaptation, avec un retard de ~10 u.t.
- La douleur du banc est un bruit sur la zone rappelée ; sa mesure (w ≈ 0.4) tombe près
  du seuil 0.3. Le seuil 0.5 (activité ×1.5) lit la même chose sur les deux seeds.
- Deux seeds, et toujours rien dans le pipeline.
