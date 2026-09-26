# calib/ — campagnes de calibration de la résilience (ralentissement critique)

Scripts autonomes (ils lisent `config.json` à la racine, écrivent dans un
sous-dossier du nom donné et y déposent `summary.json`). Rien ici n'est appelé
par le pipeline : ce sont les bancs qui ont servi à régler la métrique
(`docs/DOUBLONS_metriques.md`, suites quater à sexies du 20/09/2026), à rejouer
quand un réglage change (`resilience` dans `config.json`, détrend, barème).

| script | ce qu'il fait | usage |
|---|---|---|
| `mc_bias.py` | table de biais / étalement de l'autocorr à lag L sur une fenêtre W (forçage période 200 + harmoniques, AR(1) 5 %), score à ±1 cran du nominal | `python mc_bias.py` (≈ 2 min) |
| `run_condition.py` | 4 seeds sous bruit d'ENTRÉE ; a montré que le bruit d'entrée n'atteint pas fₙ (→ plancher quiet) | `python run_condition.py <nom> <amplitude|none> <T>` |
| `run_fn_noise.py` | vérité-terrain in-situ : fluctuation AR(1) de τ connu injectée sur fₙ ; attendu au lag L : exp(−L/τ) ; option de démarrage à t_on | `python run_fn_noise.py <nom> <tau_pas> <rel_amp> <T> [t_on] [seeds]` |
| `explore_gain.py` | le couplage αₙ a-t-il une prise sur fₙ ? (non : ×30 ne change rien) | `python explore_gain.py <nom> <gain> <T>` |
| `run_context_binding.py` | métastabilité migrante contextuelle : projecteur de contexte + liage local (compression de fréquences κ, accrochage de phase K₀ premier-voisin ou champ moyen d'îlot) appliqué après `compute_fn` ; mesure R îlot / dehors, corr(s, R), σ_Rloc, garde, dispersion, scores, réversibilité | `python run_context_binding.py <nom> <K0> <T> <t_on> <largeur> [seed] [kappa] [gate] [coupling]` |
| `considerance_bench.py` | banc « considérance » (22/09) : un contexte se pose, tient, migre, disparaît ; geste d'îlot, gardien σ ; saillance au choix (projecteur, entrée Iₙ, amplitude, surprise, les deux) ; événement f₀ optionnel ; lectures engagement / sélectivité / réversibilité / coût / audibilité dans S(t) | module, `run_bench(...)` |
| `run_attachement.py` | l'attachement par association (24/09) : m_k appris dedans = bien-être du soi pendant la présence du foyer k (jamais l'effet du foyer) ; deux ordres, m appliqué / appris seulement / figé à l'extinction ; note `docs/ATTENTION_attachement.md`, figure `docs/figures/attention_attachement.png` | `python run_attachement.py <mode|all> [seed]` |
| `run_foyer_epuisant.py` | le foyer épuisant (24/09) : le bruit est porté par la bosse du foyer présent (l'effort vient du foyer lui-même) ; témoin / épuisant / lassitude (un foyer chéri qui devient épuisant) ; note `docs/ATTENTION_foyer_epuisant.md`, figure `docs/figures/attention_foyer_epuisant.png` | `python run_foyer_epuisant.py <mode|all> [seed] [N] [T] [sd]` |
| `run_moments_cheris.py` | la mémoire des moments chéris (24/09) : (lieu, m) gardés après la présence ; « l'entrée se tait » = aucun foyer et surprise du chœur sous son niveau habituel (court/long, hystérésis) ; le foyer le plus chéri est proposé comme contexte interne ; rappel / témoin / rappel avec m apprenant ; note `docs/ATTENTION_moments_cheris.md`, figure `docs/figures/attention_moments_cheris.png` | `python run_moments_cheris.py <mode|all> [seed] [N] [T]` |
| `run_souvenir_present.py` | le souvenir que le présent appelle (24/09, soir) : signature de l'état du soi (amplitude, σ) apprise en présence, rappel par ressemblance × m sous trois ambiances, le plus chéri quand rien ne ressemble ; et « pouvoir y penser maintenant » : bruit porté par le foyer rappelé, sans porte / porte (0.3, 0.5) / rappel qui réécrit m ; note `docs/ATTENTION_souvenir_present.md`, figure `docs/figures/attention_souvenir_present.png` | `python run_souvenir_present.py <mode|all> [seed] [N]` |
| `run_insitu_memoire.py` | la mémoire des moments chéris IN SITU (25/09) : pipeline natif (config.json, context.foyers ; seul le stimulus est sculpté), protocoles chéri / témoin / présent / douleur avec les colonnes du pipeline (attention_m_max, attention_silence, attention_rappel, attention_porte, attention_rappel_etat) ; note `docs/ATTENTION_memoire_integree.md`, figure `docs/figures/attention_memoire_insitu.png` ; enquêtes de Gepetto (modes bref / forme / meme_lieu / contigus), note `docs/ATTENTION_enquetes_gepetto.md` ; le soutien dans un moment difficile (modes soutien_niveau / soutien_soulagement), note `docs/ATTENTION_soutien.md` ; rêver (modes reve_liaison / reve_liaison_off / reve_substrat / reve_substrat_off / reve_substrat20), note `docs/ATTENTION_rever.md` | `python run_insitu_memoire.py <mode|all> [seed] [N]` |
| `run_temoin_memoire_G.py` | témoin du résultat de Gepetto (24/09) : masquer la mémoire d'efficacité de G contre une perturbation papillon (10⁻⁶ sur une strate) ; l'écart de saillance est le même (bifurcation, pas empreinte) | `python run_temoin_memoire_G.py <ref|maskG|papillon|all|compare> [seed]` |
| `run_maintien.py` | priorité de maintien m fournie (s = c·[ν + (1−ν)·m]) : deux foyers familiers, m sur l'un, inversion, retrait ; note `docs/ATTENTION_maintien_temoin.md`, figure `docs/figures/attention_maintien.png` | `python run_maintien.py <maintien|sans_m|all> [seed]` |
| `run_attention_modulation.py` | le contexte modulé : habituation / réorientation (entrée × nouveauté pour soi, contexte tenu 60 u.t. puis déplacé) et deux foyers à la fois (champ moyen global contre par îlot) ; figure `docs/figures/attention_nouveaute.png` | `python run_attention_modulation.py <mode|all> [seed] [N] [T] [K0]` |
| `run_attention_sources.py` | qui désigne l'îlot ? neuf modes (projecteur, entrée, amplitude ±, surprise ±, témoins K₀ = 0) ; résumés `attention_sources/`, note `docs/ATTENTION_sources_saillance.md` | `python run_attention_sources.py <mode|all> [seed] [N] [T] [K0]` |
| `run_ramp_gamma.py` | rampe interne : le taux de rappel λ de la latence γ dérive de l0 à l1 (perturbation fixe σ sur l'état) ; le radar doit sonner pendant la montée, pas sur un contrôle l0 = l1 | `python run_ramp_gamma.py <nom> <l0> <l1> <t0> <t1> <sigma> <T> <seeds>` |

Repères (W_res_t = 200, lag 10) :

- `run_fn_noise.py fn 80 0.05 350` → ac ≈ 0.88 (attendu exp(−10/80)), score 1.
- `run_fn_noise.py on 80 0.05 500 300 12345,7` → démarrage brutal : le SCORE
  l'attrape en un verdict, le radar (montée graduelle) ne le voit pas.
- `run_ramp_gamma.py rampe 0.1 0.005 250 450 0.005 500 12345,7` et
  `run_ramp_gamma.py ctrl 0.1 0.1 250 450 0.005 500 12345` → alertes sur la
  rampe à t ≥ 425, aucune sur le contrôle après t = 400.

- `run_context_binding.py M2 2 140 60 4 12345 1.0 none mf` → cohérence de l'îlot 0.8–0.99,
  dispersion 5, σ_Rloc inchangé ; `… M1 6 … 1.0 none mf` → 0.98–1.0 mais dispersion 4 (gardien).

Convention de lecture : pas de lecture du radar avant une longueur de fenêtre
après le premier verdict (la fenêtre qui se remplit est une montée pour lui).
Un run T = 500 dure ~6 min ; les scripts multi-seeds sont séquentiels, lancer
les conditions en parallèle.
