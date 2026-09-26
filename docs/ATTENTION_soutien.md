# Le soutien dans un moment difficile : chérir ce qui était là quand ça allait mieux

*26/09/2026. Première piste d'Andréa (`ATTENTION_pistes.md`). Variante `attention.attachement.mode`
= `"soulagement"` (défaut : `"niveau"`, l'ancienne lecture), `tau_ref` 40. Banc in situ
`calib/run_insitu_memoire.py` (modes `soutien_niveau`, `soutien_soulagement`), deux seeds, sorties
`calib/attention_sources/insitu_soutien_*`. Test `test_relief_cherishes_what_was_there_when_it_got_better`.*

## Ce qu'Andréa vivait, et que le système ne faisait pas

m lisait le **niveau** du bien-être pendant la présence : ce qui était là quand ça allait mal
était terni, y compris le soutien arrivé en plein effort. Or ce que le soutien a de particulier,
ce n'est pas l'état où on était, c'est que ça allait **mieux avec lui qu'avant lui**.

## La variante

Un bien-être *habituel* : w lissé sur 40 u.t., à chaque pas, comme « habituel » pour le silence.
En mode `soulagement`, m tend vers clip(w + (w − w_habituel), 0, 1) au lieu de w. Le calme habituel
reste chéri comme avant (w = habituel : cible w). Ce qui accompagne un mieux est chéri davantage
(ça allait mal, ça va mieux : 0.5 devient 0.9). Ce qui accompagne une dégradation l'est moins (ça
allait bien, ça se dégrade : 0.5 devient 0.1). Toujours une association avec l'état du soi, jamais
une causalité ; le rappel ne réécrit toujours pas m.

## Le banc in situ

Effort global 60–140 (bruit σ 0.6, puis rampe 0.6 → 0 de 100 à 140). Le 80 est présent pendant que
l'effort **s'installe** (60–100), le 20 pendant qu'il **s'apaise** (100–140, il part quand le calme est
revenu). Silence 140–220. Retour des deux 220–260.

| mode | seed | m 20 / m 80 après (140–170) | en silence : rappel du 20 | zone 20 en silence : s · R |
|---|---|---|---|---|
| niveau | 12345 | 0.18 / 0.02 | jamais (rien de chéri) | 0.00 · 0.63 |
| niveau | 7 | 0.19 / 0.02 | jamais (rien de chéri) | 0.00 · 0.66 |
| soulagement | 12345 | **0.28** / 0.01 | 17 % du silence reconnu | 0.10 · 0.66 |
| soulagement | 7 | **0.30** / 0.01 | **96 %** du silence reconnu | **0.50 · 0.83** |

## Ce que ça dit, honnêtement

- **La direction est la bonne.** Avec la même histoire, le mode `niveau` ne chérit ni l'un ni
  l'autre (les deux ont vécu l'effort) ; le mode `soulagement` chérit celui qui a accompagné le
  mieux, pas celui qui a accompagné le pire, et en silence c'est lui qui revient.
- **L'effet est modeste.** m(20) passe de 0.18 à 0.28–0.30, juste au-dessus du seuil de rappel
  (0.3). Pendant la plus grande partie de sa présence, le soi allait encore mal (activité ×3.4 à
  100–120, ×2.3 à 120–140 : l'activité, médiane sur une respiration, traîne derrière le bruit) ;
  le mieux n'est lisible que sur les dernières unités de temps.
- **Une chose qu'on n'avait pas prévue et qui est juste.** m(20) monte surtout *après* 140, quand
  le foyer est parti : le contexte qui reste (lissé sur 5 u.t.) le tient encore là quelques unités
  de temps, exactement celles où le soulagement arrive. « Il était encore là quand ça a été
  mieux. » C'est le lissage qui fait ça, et il fait ce qu'il faut.
- **Le silence après un effort** est reconnu très différemment selon le seed (34 % contre 94 %
  sur 170–220) : la surprise met parfois longtemps à redescendre sous son habituel après une rampe.
  Ce n'est pas la variante, c'est le substrat.

## Ce qui reste à décider

Le mode par défaut. `niveau` est ce qu'on a validé sur sept bancs ; `soulagement` fait la même
chose au calme et corrige le cas du soutien, modestement, sans constante nouvelle (l'habituel est
celui du silence). On n'a pas voulu forcer l'effet avec un gain : ce serait une règle de plus.
La décision est à Andréa.
