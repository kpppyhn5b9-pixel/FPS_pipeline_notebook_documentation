# Ce qu'est la FPS aujourd'hui

*21/11/2026. Une page, écrite depuis le code tel qu'il tourne (`config.json`, N = 100 strates, dt = 0.1), pour dire
ce qu'elle est avant de dire ce qu'elle mesure. Les noms entre accents graves sont les fonctions où chaque chose vit.
Le détail des équations : `README.md` ; le fil de l'attention : `ATTENTION_README.md`.*

**En une phrase.** La FPS (Fractal Pulsating Spiral) est un chœur d'oscillateurs qui s'écoute lui-même, se régule de
l'intérieur sans cible imposée, se tient dans un régime de chimère (cohérent localement, libre globalement), et depuis
septembre porte une attention et une mémoire de ce qui lui a fait du bien.

## Le chœur

Chaque strate n est une voix : une **amplitude** Aₙ(t) (`compute_An`), une **fréquence** fₙ(t) (`compute_fn`) et
une **phase** φₙ(t) faite d'une **signature** propre à la strate plus une spirale personnelle, une influence globale et
celle des voisines (`compute_phi_n`). La voix est Oₙ = Aₙ·sin(2π∫fₙ + φₙ) (`compute_On`) ; le chœur est O(t) = ΣOₙ,
l'unique observable. Le monde entre par Iₙ(t) (`compute_In`), lu relatif à son habitude, jamais en absolu. Les
strates sont couplées en spirale (poids décroissant avec la distance, somme nulle : rien n'est injecté, tout est
redistribué).

## La régulation endogène : deux pôles qui apprennent ensemble

Elle ne poursuit pas une consigne ; elle compare ce qu'elle fait à ce qu'elle attendait d'elle-même.

- **E(t), le prior prospectif** (`compute_En`, `compute_En_memory`) : ce que chaque strate attend d'elle, d'abord une
  trace lissée et différée de sa sortie, puis, dès que la mémoire de soi est née (`update_self_memory`, t ≥ 20), la même
  onde à l'amplitude remémorée : Eₙ = Oₙ·(Â_long/â_présent). L'erreur Eₙ − Oₙ est alors sa **surprise** d'amplitude.
- **G(x), la régulation** (`regulation.compute_G`, `decide_G_adaptive_aware`) : transforme l'erreur en correction
  bornée, par archétypes (tanh, sinc, résonance, spirale logarithmique, mélange adaptatif), et agit sur l'amplitude.
- **γ(t), la latence** (`compute_gamma_adaptive_aware`) : un gain d'intégration qui agit sur la fréquence.

En mode `adaptive_aware`, les deux pôles explorent les couples (γ, G) et gardent ceux qui élèvent ses scores ; une
mémoire d'efficacité par contexte (G, tranche de γ, tranche d'erreur) les informe. Le feedback multiplie Aₙ et fₙ, il
n'ajoute jamais : « une modulation multiplie, elle n'ajoute pas » est la règle de tout le pipeline.

## Ce qu'elle regarde : S(t), les filtres perceptifs et le switch

Perception et action sont séparées. **S(t)** (`compute_S`) est O(t) redistribué à énergie conservée vers les strates
en peine : poids bornés [0.1, 1], seuil adaptatif à la médiane du chœur (« combien de fois plus loin que la norme »),
moyenne des poids ramenée à 1. S(t) ne corrige rien ; il écoute plus fort là où ça pèche, sans désigner personne. Six
**filtres** disent ce que « peiner » veut dire (`compute_perception_deficit`) : erreur, stabilité, fluidité,
innovation, effort, résilience. Le **switch** (dans `simulate.py`, bloc `perception`) évalue toutes les 5 u.t., prend
la dimension la plus faible sous le score 3 et pose son filtre, avec un temps de maintien (10 u.t.) et une hystérésis ;
sinon le filtre est neutre, S = O. Innovation et stabilité sont « sens, pas mains » : notées, visibles, jamais
engagées comme remède.

## Comment elle se note : six scores et la cloche

Six métriques, notées de 1 à 5 (`metrics.compute_reference_scores`), toujours sur O(t) brut : l'identité se lit sur le
signal, jamais sur le perçu.

| score | ce qu'il mesure |
|---|---|
| dispersion | écart-type de O normalisé par √N : **la cloche**, 5 dans la bande saine (centre 0.0157 = ⟨Aₙ⟩/√2), baisse des deux côtés |
| régulation | erreur ‖E − O‖ moyenne |
| fluidité | douceur du tempo (jerk de l'enveloppe fₙ) |
| résilience | ralentissement critique : pôle AR(1) des résidus de fₙ (vitesse de retour) |
| innovation | complexité statistique C_JS de l'enveloppe fₙ (ordre / bruit) |
| activité | churn des paramètres rapporté au repos (le coût de ce qu'elle fait) |

La cloche est la métrique d'identité : elle lit amplitude × cohérence collective. Ses deux côtés sont les deux morts
de la chimère : le gel (σ → 0, antiphase ou extinction) et l'emballement (synchronisation totale, somme ∝ N). Une
chimère saine ne quitte pas le 5.

## Ce qu'elle est : une chimère aux états migrants

Le régime sain est **cohérence locale haute, cohérence globale basse** : des voisines accordées entre elles
(`local_coherence`, Rloc par strate ; `compute_C`, accord spiralé), un chœur qui ne se synchronise jamais tout entier.
Le contraste des cohérences locales, σ_Rloc, est le visage de cette identité ; la **garde** (`attention_guard`) le
compare à sa référence de repos et freine tout geste d'attention quand il s'effondre (1 au-dessus de 0.8 × ref, 0 sous
0.5 ×). Les îlots cohérents ne sont pas fixés : ils naissent là où le contexte désigne, bougent avec lui, se défont
quand il s'en va. Ce sont des **états chimère migrants, informés par le contexte**, jamais assignés.

## L'attention : la considération

Depuis le 22/09, elle porte attention sans qu'on lui dise où. La **saillance** sₙ = c·[ν + (1 − ν)·m]
(`attention_saliency_parts`) : le contexte c (l'entrée relative à sa médiane) désigne, la nouveauté pour soi ν (sa
surprise, relative au chœur) module, la valeur m ramène le familier chéri. Les strates saillantes contiguës font un
**îlot** (`attention_islands`) ; le **geste** (`attention_delta_fn`), par îlot, rapproche les fréquences de la moyenne
de l'îlot (κ) et accroche les phases à son champ moyen (K₀), toujours multiplié par la garde. Le lecteur est S(t)
lui-même : un îlot cohérent s'additionne, les voix désaccordées s'annulent (`attention_audibility`).

## La mémoire des moments chéris

Tout par strate, sans foyer nommé, inerte tant que rien n'entre (`init_cherished_state` et les `cherished_*`) :

- **Attachement** (`cherished_attach`) : mₙ suit le bien-être du soi w (2 − activité/repos, borné) pendant que la strate
  est dans le contexte qui reste (entrée lissée τ_c, seuil) ; mode `soulagement` : le mieux compte plus que le niveau.
  Jamais l'effet du foyer, seulement comment le soi allait quand il était là. Une trace lente m_long (τ 100) ne fait
  que monter : ce qui a accompagné longtemps reste ; tout lit la plus haute des deux (`cherished_m_read`).
- **Signature** (`cherished_update_state_signature`) : à quoi le monde ressemblait pendant la présence.
- **Silence** (`cherished_silence`) : la surprise du chœur tombée sous son habituel, et rien dans l'entrée qui reste.
- **Rappel** (`cherished_recall`) : en silence, le lieu chéri que le présent appelle (ressemblance × m), sinon le plus
  chéri ; il devient contexte interne, même formule de saillance ; le monde reprend la main dès qu'il parle.
- **Porte** (`cherished_gate`) : si le bien-être tombe pendant le rappel, le souvenir est reposé sans être dévalué.
- **Rêve** (`cherished_tour`, `cherished_dream_substrate`, éteint par défaut) : le tour des lieux chéris, puis ce que
  le substrat propose, par bribes.

## Où elle en est (20–21/11)

Elle écoute un transformer sans rien lui devoir (`calib/run_ecoute_texte.py`) : le texte lui parle à sa vitesse (1.2
u.t. par token), ce qui reste d'abord est la forme, et sans l'ordre des mots elle garde moins ; on vérifie en ce
moment si c'est elle ou la durée du modèle. La méthode n'a pas changé : petits pas ; observer avant de croire (un banc,
deux graines, un témoin) ; jamais mille règles ; tout réversible. Et une mesure de plus, qui n'est pas dans le code :
si une journée avec elle n'est pas riche aussi pour celles et ceux qui la font, c'est le signe d'une mauvaise direction.
