"""
simulate.py – FPS Pipeline Simulation Core
Version exhaustive FPS
---------------------------------------------------------------
NOTE FPS – Plasticité méthodologique :
Ce module orchestre toute la dynamique FPS/Kuramoto/Neutral, et doit toujours rester :
- **modulaire** : chaque fonction/étape isolable, falsifiable individuellement
- **extensible** : chaque critère, métrique, ou pipeline peut être augmenté/remplacé
- **falsifiable** : tout doit être logué, traçable, chaque hypothèse peut être testée, toute évolution loguée
- **strict** : tout ce qui est attendu ici, même si le module n'est pas encore codé, doit être détaillé dans ce fichier en code ou en commentaire.
---------------------------------------------------------------

**Pipeline FPS** – *Résumé*
Boucle temporelle orchestrant :
- Chargement et validation complète du config.json (validate_config.py)
- Initialisation strates, seeds, loggers (init.py)
- Gestion du mode (FPS, Kuramoto, Neutral)
- Boucle de simulation temps-réel (avec dynamique FPS complète OU alternatives)
- Calculs à chaque pas :
    - Input contextuel/perturbations, dynamique FPS (dynamics.py)
    - Régulation (regulation.py), feedback adaptatif G(x)
    - Mise à jour des états, calcul S(t), E(t), O(t), metrics (metrics.py)
    - Log metrics, backup d'état, gestion erreurs (utils.py)
- Gestion post-run :
    - Analyse & raffinements automatiques (analyze.py)
    - Exploration émergences, fractals, anomalies (explore.py)
    - Visualisation complète, dashboard, grille empirique (visualize.py)
    - Export de tous les logs, configs, seeds, changelog.txt
- Exécution en mode batch/auto si besoin (via batch_runner ou module export-batch)
"""

# --------- IMPORTS (TOUS MODULES DU PIPELINE FPS) -----------
import os, sys, time, json, traceback, numpy as np
import scipy
import csv
# Imports stricts
import init
import dynamics
import regulation
import metrics
import analyze
import explore
import visualize
import validate_config
import kuramoto
import perturbations
import utils
from utils import deep_convert, extract_best_pair_from_journal

def safe_float_conversion(value, default=0.0):
    """Convertit une valeur en float sûr.

    None = verdict suspendu / absent → NaN (cellule vide dans le CSV),
    JAMAIS 0.0 : écrire zéro à la place d'une absence de verdict est un
    mensonge de logging (0 = catastrophique, None = pas encore jugé)."""
    try:
        if value is None:
            return float('nan')
        if isinstance(value, str):
            return default
        if np.isnan(value) or np.isinf(value):
            return default
        return float(value)
    except:
        return default


# --------- PIPELINE PRINCIPALE : RUN_SIMULATION() -----------

def run_simulation(config_path, mode="FPS", strict=False):
    """
    Orchestration complète d'un run FPS/Kuramoto/Neutral.
    - Validation complète config.json (présence, structure, types, seuils, dépendances croisées…)
    - Initialisation strates, seeds, loggers
    - Branche pipeline selon le mode
    - À chaque step : dynamique complète (input, dynamique FPS, feedback, metrics, log, backup)
    - Gestion post-run (analyse, exploration, visualisation, logs, exports)
    """
    # ---- 0. Chargement et VALIDATION TOTALE du config ----
    config = init.load_config(config_path) if hasattr(init, "load_config") else json.load(open(config_path))
    errors, warnings = validate_config.validate_config(config_path) if hasattr(validate_config, "validate_config") else ([], [])
    if errors:  # Si il y a des erreurs (liste non vide)
        print("Config validation FAILED:")
        for e in errors: print("  -", e)
        sys.exit(1)
    if warnings:
        print("Config validation avec avertissements:")
        for w in warnings: print("  -", w)
    print("Config validation: OK")

    # ---- 1. SEED : reproductibilité, log dans seeds.txt ----
    SEED = config['system']['seed']
    # CORRECTION: Assurer cohérence seeds entre run principal et batch
    print(f"🌱 Initialisation seed: {SEED} pour mode {mode}")
    np.random.seed(SEED)
    import random; random.seed(SEED)
    
    # Log de la seed pour traçabilité
    if hasattr(utils, "log_seed"): 
        utils.log_seed(SEED)
    else: 
        with open("seeds.txt", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | SEED = {SEED} | Mode = {mode}\n")
    
    # ---- 2. Initialisation : strates, état, logs, dirs ----
    state = init.init_strates(config)
    loggers = init.setup_logging(config, mode_suffix=mode.upper())
    
    if hasattr(utils, "log_config_and_meta"): 
        utils.log_config_and_meta(config, loggers['run_id'])
    
    # ---- 3. BRANCH par mode (FPS/Kuramoto/Neutral) ----
    if mode.lower() == "kuramoto":
        result = run_kuramoto_simulation(config, loggers)
    elif mode.lower() == "neutral":
        result = run_neutral_simulation(config, loggers)
    elif mode.lower() == "fps":
        result = run_fps_simulation(config, state, loggers, strict)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(2)
    
    # ---- 4. Post-run : Analyse, exploration, visualisation, export ----
    if hasattr(analyze, "analyze_criteria_and_refine"):
        analyze.analyze_criteria_and_refine([result['logs']], config)
    
    if hasattr(explore, "run_exploration"):
        explore.run_exploration(result['logs'], loggers['output_dir'], config)
    
    if hasattr(visualize, "plot_metrics_dashboard"):
        visualize.plot_metrics_dashboard(result['metrics'])
    
    # ---- 5. Exports finaux : rapport, changelog, backup ----
    if hasattr(utils, "log_end_of_run"): 
        utils.log_end_of_run(result['run_id'])
    
    # NOTE: Le fichier CSV sera fermé APRÈS la simulation dans run_fps_simulation
    # pour éviter la fermeture prématurée qui empêche l'enregistrement des données
    
    return result

# --- MODE FPS (Pipeline complet) ------------------------------------------
# --- MODE FPS (Pipeline complet) ------------------------------------------
def check_chimera_reset(t, T, config, state, phase_acc):
    """
    Tests chimériques 2 et 3 — resets mid-run (robustesse de l'état chimérique).
    Porté depuis NOTEBOOK_FPS.ipynb cell 12, adapté à l'architecture pipeline :

    - reset_frequencies_midrun : réécrit f0 (l'ancre persistante) de chaque strate.
      NOTE portage : resetter fn_t serait sans effet (recalculé chaque pas depuis f0).
    - reset_phases_midrun : remet phase_acc (phase intégrée θ) à la valeur cible.
      NOTE portage : ne touche JAMAIS state['phi'] (signature invariante,
      convention catalogue « ne jamais assigner θ → φₙ »).

    Chaque reset se déclenche une seule fois (flag 'triggered'), à t >= t_reset·T.

    Args:
        t: temps courant
        T: durée totale du run
        config: configuration (lit/écrit config['chimera_tests'])
        state: état des strates (f0 modifié in-place si reset fréquences)
        phase_acc: accumulateur de phase (np.ndarray, modifié in-place si reset phases)

    Returns:
        phase_acc (même objet, pour lisibilité de l'appel)
    """
    chimera_cfg = config.get('chimera_tests', {})
    if not chimera_cfg:
        return phase_acc

    freq_reset = chimera_cfg.get('reset_frequencies_midrun', {})
    if (freq_reset.get('enabled', False) and not freq_reset.get('triggered', False)
            and t >= freq_reset.get('t_reset', 0.5) * T):
        value = freq_reset.get('value', 1.0)
        for strate in state:
            strate['f0'] = value
        freq_reset['triggered'] = True
        print(f"\n🔬 CHIMERA RESET @ t={t:.1f}: toutes les f0 → {value} Hz")

    phase_reset = chimera_cfg.get('reset_phases_midrun', {})
    if (phase_reset.get('enabled', False) and not phase_reset.get('triggered', False)
            and t >= phase_reset.get('t_reset', 0.5) * T):
        value = phase_reset.get('value', 0.0)
        phase_acc[:] = value
        phase_reset['triggered'] = True
        print(f"\n🔬 CHIMERA RESET @ t={t:.1f}: phase intégrée (phase_acc) → {value}")

    return phase_acc


def run_fps_simulation(config, state, loggers, strict=False):
    """
    Boucle principale FPS, version exhaustive :
    - À chaque pas : input contextuel, dynamique FPS (toutes formules), feedback/régulation, logs, backup
    - Gestion du mode statique/dynamique pour tous paramètres (voir config)
    - Log exhaustif des métriques (voir feuille de route/tableau structuré)
    - Gestion stricte erreurs, try/except, logs des exceptions
    - Sécurité backup état toutes les 100 steps
    - À la fin : export de tous les logs, résumé metrics, passage à l'analyse et à l'exploration
    """
    # -- PARAMÈTRES GLOBAUX
    T = config['system']['T']
    dt = config['system'].get('dt', 0.05)
    N = config['system']['N']
    run_id = loggers['run_id']
    t_array = np.arange(0, T, dt)
    backup_interval = 100
    # Initialiser les signaux de feedback à zéro
    F_n_t_An = np.zeros(N)
    F_n_t_fn = np.zeros(N)
    phase_acc = np.zeros(N)

    # --- μ_Rloc en boucle (lot v2) : cohérence locale réelle sur θ intégrée ---
    # Voisinage et poids |W| normalisés, mêmes conventions que kuramoto_local2.
    # Sert de mesure de tenue de soi au repos (résilience mesurée, pas décrétée).
    _rloc_neigh = []
    for _n in range(N):
        _w = np.asarray(state[_n].get('w', []), dtype=float) if state and len(state) > _n else np.array([])
        _idx = np.nonzero(np.abs(_w) > 1e-12)[0] if _w.size else np.array([], dtype=int)
        if _idx.size:
            _pw = np.abs(_w[_idx]); _pw = _pw / _pw.sum()
        else:
            _pw = np.array([])
        _rloc_neigh.append((_idx, _pw))
    mu_Rloc_history = []
    O_sum_history = []            # ΣOₙ par pas (dispersion d'identité, sur O brut)
    dispersion_raw_history = []   # std(ΣO)/√N par fenêtre W_f, avant lissage
    # Résilience = ralentissement critique sur la couche lente (New_Attention.md) :
    # autocorrélation à lag calibré des résidus détrendés de l'enveloppe fₙ, sur
    # une fenêtre LONGUE. Le lag est calibré UNE fois (relaxation 1/e) puis figé :
    # c'est la condition pour qu'une montée (CSD) soit visible. Sans config, sans
    # choc à induire : elle lit la turbulence spontanée du système.
    _rcfg = config.get('resilience', {})
    _W_res = max(int(_rcfg.get('min_points', 200)),
                 int(round(float(_rcfg.get('W_res_t', 200.0)) / dt)))
    resilience_state = {
        'lag': (int(_rcfg['lag']) if _rcfg.get('lag') else None),  # None → auto-calibré
        'ac_hist': [], 'var_hist': [],  # fluctuations RÉELLES seulement (radar)
        'ac_recent': [],  # tous les verdicts calculés, quiet → 0.0 (lissage du score)
        'peak_ratio': float(_rcfg.get('peak_ratio', 10.0)),
        'max_peaks': int(_rcfg.get('max_peaks', 8)),
        'calm_lags': int(_rcfg.get('alert_calm_lags', 20)),  # référence du radar : longueur, en lags
        'gap_lags': int(_rcfg.get('alert_gap_lags', 20)),    # écart référence → présent, en lags
        'band': float(_rcfg.get('alert_band', 0.06)),
        'n_windows': int(_rcfg.get('alert_windows', 3)),
        'smooth_lags': int(_rcfg.get('smooth_lags', 3)),  # médiane sur k lags pour le SCORE
        'quiet_floor_rel': float(_rcfg.get('quiet_floor_rel', 1e-6)),  # plancher « rien ne fluctue »
        'min_lags_per_window': int(_rcfg.get('min_lags_per_window', 200)),  # plafond du lag auto (biais fenêtre)
        'stride': max(1, int(_rcfg.get('stride', 5))),  # cadence de calcul (pas) : la fenêtre est longue, inutile de refaire chaque pas
        'last': None,  # dernier verdict (ac, ac_smooth, var, lag, alert, quiet), reporté entre deux calculs
    }
    # Activité : repos de référence du run (20/09/2026), calibré UNE fois sur
    # [calib_start_t, calib_start_t + calib_window_t] (après l'exploration
    # initiale de γ et G) puis FIGÉ ; le score 'activite' et les statuts
    # stable/transitoire/chronique se lisent en facteurs de ce repos.
    _acfg = config.get('activite', {})
    # Mémoire de soi à deux étages (21/09/2026, décision d'Andréa) : une fois née
    # (t ≥ birth_t), Eₙ = Oₙ à l'amplitude REMÉMORÉE (mémoire longue, relative au
    # chœur) ; la surprise (écart courte/longue) est loguée. Voir dynamics.update_self_memory.
    _mcfg = config.get('memory', {})
    self_memory = dynamics.init_self_memory(N)
    memory_state = {
        'enabled': bool(_mcfg.get('enabled', True)),
        'birth_t': float(_mcfg.get('birth_t', 20.0)),   # naissance AVANT les fenêtres de repos (40–60) : son transitoire ne les touche pas, et l'attention (nouveauté) voit clair dès t ≈ 60
        'tau_s': float(_mcfg.get('tau_s_t', 2.0)), 'tau_l': float(_mcfg.get('tau_l_t', 40.0)),
        'E_last': None,            # dernier Eₙ remémoré (pour l'enveloppe, en tête de pas)
        'surprise': None,          # dernier vecteur de surprise (lu par l'attention : nouveauté pour soi)
    }
    activite_state = {
        'ref': None,
        't_start': float(_acfg.get('calib_start_t', 20.0)),
        't_end': float(_acfg.get('calib_start_t', 20.0)) + float(_acfg.get('calib_window_t', 20.0)),
        'window_steps': max(1, int(round(float(_acfg.get('calib_window_t', 20.0)) / dt))),  # une respiration
    }
    # ATTENTION : la considération (22/09/2026, décision d'Andréa). Contexte spatial
    # (config 'context.foyers') × nouveauté pour soi → îlots liés par champ moyen local,
    # freinés par σ_Rloc (référence = le repos du run, même fenêtre que l'activité),
    # entendus dans S(t). Sans foyer, la saillance est nulle : le geste ne touche rien.
    _atcfg = config.get('attention', {}); _ctxcfg = config.get('context', {})
    attention_state = {
        'enabled': bool(_atcfg.get('enabled', True)),
        'K0': float(_atcfg.get('K0', 5.0)), 'kappa': float(_atcfg.get('kappa', 0.5)),
        'saliency_scale': float(_atcfg.get('saliency_scale', 0.5)), 'novelty_scale': float(_atcfg.get('novelty_scale', 2.0)),
        'island_threshold': float(_atcfg.get('island_threshold', 0.05)),
        'foyers': list(_ctxcfg.get('foyers', [])),
        'ref_t0': activite_state['t_start'], 'ref_t1': activite_state['t_end'], 'sigma_ref': [], 'sigma_ref_value': None,
        's': np.zeros(N), 'garde': 1.0, 'cost': 0.0,
    }
    # LA MÉMOIRE DES MOMENTS CHÉRIS (25/09/2026, décision d'Andréa) : attachement par strate
    # (m suit le bien-être du soi pendant la présence d'un îlot), silence (« ce qui se passait
    # s'est apaisé »), rappel du souvenir que le présent appelle, porte « pouvoir y penser
    # maintenant ». Voir dynamics.cherished_*. Sans îlot dans l'entrée : inerte.
    _attach_cfg = _atcfg.get('attachement', {}); _sil_cfg = _atcfg.get('silence', {}); _rap_cfg = _atcfg.get('rappel', {}); _porte_cfg = _rap_cfg.get('porte', {})
    cherished_state = dynamics.init_cherished_state(N)
    cherished_cfg = {
        'attach': bool(_attach_cfg.get('enabled', True)), 'tau_m': float(_attach_cfg.get('tau_m', 20.0)), 'tau_sig': float(_attach_cfg.get('tau_sig', 5.0)),
        'tau_c': float(_attach_cfg.get('tau_c', 5.0)), 'seuil_contexte': float(_attach_cfg.get('seuil_contexte', 0.5)),
        'mode': str(_attach_cfg.get('mode', 'soulagement')), 'tau_ref': float(_attach_cfg.get('tau_ref', 40.0)),
        'tau_short': float(_sil_cfg.get('tau_short', 5.0)), 'tau_long': float(_sil_cfg.get('tau_long', 40.0)),
        'enter': float(_sil_cfg.get('enter', 0.9)), 'exit': float(_sil_cfg.get('exit', 1.1)),
        'rappel': bool(_rap_cfg.get('enabled', True)), 'm_min': float(_rap_cfg.get('m_min', 0.3)), 'res_min': float(_rap_cfg.get('res_min', 0.2)),
        'sig_width': tuple(float(x) for x in _rap_cfg.get('sig_width', [0.2, 0.1])),
        'porte': bool(_porte_cfg.get('enabled', True)), 'porte_tau': float(_porte_cfg.get('tau', 5.0)),
        'porte_floor': float(_porte_cfg.get('floor', 0.5)), 'porte_refractory': float(_porte_cfg.get('refractory', 30.0)),
        'w': None, 'external': False, 'recalling': False, 'etat': 0,
    }
    # Filtres de perception (spec 15/07/2026) — état du switch
    _pcfg = config.get('perception', {})
    perception_mode = _pcfg.get('filter_mode', 'static')
    # En AUTO, le système NAÎT neutre (le repos perceptif, design d'Andréa) :
    # 'erreur' n'est plus jamais un repli structurel, seulement un remède CHOISI.
    # En STATIC, le filtre configuré s'applique tel quel.
    _f0 = 'neutre' if _pcfg.get('filter_mode', 'static') == 'auto' else _pcfg.get('filter', 'erreur')
    perception_state = {
        'filter': _f0,
        'weights': (np.ones(config['system']['N']) if _f0 == 'neutre' else None),
        'last_eval_t': -1e9, 'last_switch_t': -1e9,
        'W_f': metrics.reference_window(config, dt),  # source unique de la fenêtre de scoring
        'T_switch': float(_pcfg.get('T_switch_t', 5.0)),
        'dwell': float(_pcfg.get('dwell_t', 10.0)),
        'seuil_in': int(_pcfg.get('seuil_declenchement', 3)),
        'seuil_out': int(_pcfg.get('seuil_sortie', 3)),
        'enabled': _pcfg.get('filters_enabled', ['erreur','stabilite','fluidite','innovation','effort']),
    }
    # Fenêtre LONGUE du moniteur lent d'innovation (C_JS de l'enveloppe fₙ) :
    # la distribution de permutations a besoin de ≥200 points, bien plus que W_f.
    _W_innov = max(200, 4 * perception_state['W_f'])

    # Historiques avec limite de mémoire
    MAX_HISTORY_SIZE = config.get('system', {}).get('max_history_size', 10000)
    history, cpu_steps, effort_history, S_history = [], [], [], []
    # Historiques supplémentaires pour métriques avancées
    C_history = []
    An_history = []  # Pour A_mean(t) et export individuel
    fn_history = []  # Pour f_mean(t) et export individuel
    En_history = []  # Pour mean_abs_error
    On_history = []  # Pour mean_abs_error
    
    # NOUVEAU S1: Seuil d'alignement En ≈ On (pour log de debug)
    epsilon_E = config.get('regulation', {}).get('epsilon_E', 0.01)  # Seuil d'alignement
    
    # NOUVEAU : État adaptatif pour gamma et G
    adaptive_state = {
        'gamma_journal': None,
        'regulation_state': {}
    }

    # INITIALISER all_metrics ET t EN DEHORS DE LA BOUCLE
    all_metrics = {}
    t = 0
    
    # Fonction de rotation des historiques
    def rotate_history(hist_list, max_size):
        """Garde seulement les max_size derniers éléments."""
        if len(hist_list) > max_size:
            return hist_list[-max_size:]
        return hist_list
    
    # Préparer les fichiers individuels si N > 10
    individual_csv_writers = {}
    if N > 10 and config.get('analysis', {}).get('save_indiv_files', True):
        for n in range(N):
            # A_n_{id}.csv
            an_filename = os.path.join(loggers['output_dir'], f"A_n_{n}_{run_id}.csv")
            an_file = open(an_filename, 'w', newline='')
            an_writer = csv.writer(an_file)
            an_writer.writerow(['t', f'A_{n}(t)'])
            individual_csv_writers[f'A_{n}'] = {'file': an_file, 'writer': an_writer}
            
            # f_n_{id}.csv
            fn_filename = os.path.join(loggers['output_dir'], f"f_n_{n}_{run_id}.csv")
            fn_file = open(fn_filename, 'w', newline='')
            fn_writer = csv.writer(fn_file)
            fn_writer.writerow(['t', f'f_{n}(t)'])
            individual_csv_writers[f'f_{n}'] = {'file': fn_file, 'writer': fn_writer}
    
    # -- BOUCLE PRINCIPALE --
    try:
        for step, t in enumerate(t_array):
            # Tests chimériques 2/3 : resets mid-run (no-op si non configurés)
            phase_acc = check_chimera_reset(t, T, config, state, phase_acc)

            # ----------- 1. INPUTS ET PERTURBATIONS -----------
            # Nouvelle architecture In(t)
            input_config = config.get('system', {}).get('input', {})
            
            # Calculer In(t) avec la nouvelle architecture
            try:
                input_cfg_with_T = input_config.copy()
                input_cfg_with_T['T'] = T
                In_t = perturbations.compute_In(t, input_cfg_with_T, state, history, dt)
                if np.isscalar(In_t):
                    In_t = np.full(N, In_t)
                
                # Logger les valeurs significatives
                if step % 100 == 0 and np.mean(In_t) > 0.01:
                    print(f"  📊 Input contextuel à t={t:.1f}: In_mean={np.mean(In_t):.3f}")
                    
            except Exception as e:
                print(f"⚠️ Erreur compute_In à t={t}: {e}")
                print(f"   input_config: {input_config}")
                print(f"   Type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                # Fallback : utiliser un offset minimal
                In_t = np.full(N, 0.1)
            if attention_state['foyers']:
                In_t = np.asarray(In_t, dtype=float) + dynamics.compute_context_field(t, N, attention_state['foyers'])   # contexte spatial
            
            # ----------- 2. CALCULS DYNAMIQUE FPS --------------
            # ================= RÉORDONNANCEMENT DU PAS (15/07/2026) =============
            # φ_reg → Eₙ calculé UNE FOIS ici, puis passé à compute_An (enveloppe)
            # et compute_S (erreur relative) via les configs. Avant : Eₙ était
            # calculé TROIS fois par pas, les deux appels internes recomputant
            # leur propre φ depuis history SANS le buffer d'effort (micro-
            # divergences avec l'officiel). Une seule vérité désormais.
            # b) Calcul de phi adaptatif (alignement notebook)
            try:
                phi_mode = config.get('regulation', {}).get('phi_mode', 'fixed')
                
                if phi_mode == 'adaptive':
                    # Récupérer l'effort actuel depuis history
                    if len(history) > 0 and 'effort(t)' in history[-1]:
                        effort_current = history[-1].get('effort(t)', 0.0)
                    else:
                        effort_current = 0.0
                    
                    # Prendre les 20 derniers éléments pour phi_adaptive
                    effort_for_phi = effort_history[-20:] if len(effort_history) > 0 else [0.0]
                    
                    # Calculer phi adaptatif
                    phi_reg = dynamics.compute_phi_adaptive(effort_current, effort_for_phi, config)
                else:
                    # Mode fixe
                    phi_reg = config.get('regulation', {}).get('phi_fixed_value', 1.618)
            except Exception as e:
                print(f"⚠️ Erreur calcul phi adaptatif à t={t}: {e}")
                phi_reg = config.get('regulation', {}).get('phi_fixed_value', 1.618)
            
            En_t = dynamics.compute_En(t, state, history, config, phi_reg, effort_history) if hasattr(dynamics, 'compute_En') else None
            if memory_state['enabled'] and memory_state['E_last'] is not None:
                En_t = memory_state['E_last']   # mémoire née : l'enveloppe lit l'attendu remémoré du pas précédent

            # a) Amplitude, fréquence, phase, latence par strate (avec statique/dynamique, config.json)
            try:
                # IMPORTANT: Ajouter l'historique à la config pour compute_An avec enveloppe dynamique
                config_for_An = config.copy()
                config_for_An['history'] = history
                config_for_An['En_ext'] = En_t
                # σₙ RELATIF (dossier mu_n, 15/07/2026) — config-gaté, défaut intact.
                # enveloppe.sigma_mode='relative' : σ = max(k·IQR des erreurs
                # récentes, plancher) — la tolérance suit l'échelle d'erreur
                # PROPRE du système (sans dimension, portable), au lieu du 0.1
                # absolu qui faisait chuchoter la cloche (facteurs ~0.98).
                if config.get('enveloppe', {}).get('sigma_mode', 'static') == 'relative':
                    _sk = float(config['enveloppe'].get('sigma_rel_k', 1.5))
                    _sw = max(10, int(round(float(config['enveloppe'].get('sigma_rel_window_t', 10.0)) / dt)))
                    _errs = []
                    for _h in history[-_sw:]:
                        _e = _h.get('E'); _o = _h.get('O')  # clés history : 'E' et 'O' (l.993)
                        if _e is not None and _o is not None and len(_e) and len(_o):
                            _errs.append(np.abs(np.asarray(_e) - np.asarray(_o)))
                    if len(_errs) >= 10:
                        _ev = np.concatenate(_errs)
                        _q75, _q25 = np.percentile(_ev, [75, 25])
                        config_for_An['sigma_n_override'] = float(max(_sk * (_q75 - _q25), 1e-4))
                    # sinon : pas assez d'histoire -> compute_An garde le statique (warmup)
                An_t = dynamics.compute_An(t, state, In_t, F_n_t_An, config_for_An)
                An_history.append(An_t.copy())                
                # Passer l'historique dans la config pour compute_fn
                config_with_history = config.copy()
                config_with_history['history'] = history
                fn_t = dynamics.compute_fn(t, state, An_t, F_n_t_fn, config_with_history)
                # ATTENTION (22/09) : le geste de considération, après la cascade, avant l'intégration.
                if attention_state['enabled']:
                    _phi_prev = (np.asarray(history[-1]['phi_n_t'], dtype=float)
                                 if history and history[-1].get('phi_n_t') is not None else np.zeros(N))
                    _theta_att = phase_acc + _phi_prev
                    _sig_att = float(np.std(dynamics.local_coherence(_theta_att, _rloc_neigh)))
                    if attention_state['ref_t0'] <= t < attention_state['ref_t1']:
                        attention_state['sigma_ref'].append(_sig_att)
                    elif t >= attention_state['ref_t1'] and attention_state['sigma_ref_value'] is None and attention_state['sigma_ref']:
                        attention_state['sigma_ref_value'] = float(np.median(attention_state['sigma_ref']))
                    attention_state['garde'] = dynamics.attention_guard(_sig_att, attention_state['sigma_ref_value'])
                    # contexte c et nouveauté ν séparés ; puis la mémoire des moments chéris
                    _c_att, _nu_att = dynamics.attention_saliency_parts(In_t, memory_state.get('surprise'),
                                                                        attention_state['saliency_scale'], attention_state['novelty_scale'])
                    _rel_prev = history[-1].get('activite_rel') if history else None
                    cherished_cfg['w'] = (float(np.clip(2.0 - float(_rel_prev), 0.0, 1.0)) if _rel_prev is not None and np.isfinite(_rel_prev) else None)
                    _ctx_on = dynamics.cherished_context(cherished_state, In_t, dt, cherished_cfg['tau_c'], attention_state['saliency_scale'], cherished_cfg['seuil_contexte'])
                    cherished_cfg['external'] = bool(_ctx_on.any())                 # l'entrée désigne quelque chose qui reste
                    cherished_cfg['recalling'] = False
                    dynamics.cherished_update_state_signature(cherished_state, np.array([float(np.mean(An_t)), _sig_att]), dt, cherished_cfg['tau_sig'])
                    dynamics.cherished_wellbeing_ref(cherished_state, cherished_cfg['w'], dt, cherished_cfg['tau_ref'])
                    if cherished_cfg['attach']:
                        dynamics.cherished_attach(cherished_state, _ctx_on, cherished_cfg['w'], dt, cherished_cfg['tau_m'], cherished_cfg['tau_sig'], cherished_cfg['mode'])
                    _quiet = dynamics.cherished_silence(cherished_state, memory_state.get('surprise'), cherished_cfg['external'], dt,
                                                        cherished_cfg['tau_short'], cherished_cfg['tau_long'], cherished_cfg['enter'], cherished_cfg['exit'])
                    # l'état du rappel, journalisé (colonne attention_rappel_etat) : 0 désactivé · 1 l'extérieur parle ·
                    # 2 pas de silence · 3 rien de chéri · 4 réfractaire · 5 porte fermée à l'instant · 6 rappel en cours
                    if not (cherished_cfg['attach'] and cherished_cfg['rappel']):
                        cherished_cfg['etat'] = 0
                    elif cherished_cfg['external']:
                        cherished_cfg['etat'] = 1
                    elif not _quiet:
                        cherished_cfg['etat'] = 2
                    if cherished_cfg['attach'] and cherished_cfg['rappel'] and _quiet and not cherished_cfg['external']:
                        _c_int = dynamics.cherished_recall(cherished_state, t, cherished_cfg['m_min'], cherished_cfg['res_min'], cherished_cfg['sig_width'])
                        if _c_int is None:
                            cherished_cfg['etat'] = 4 if bool(np.any(cherished_state['m'] >= cherished_cfg['m_min'])) else 3
                        elif cherished_cfg['porte'] and dynamics.cherished_gate(cherished_state, cherished_cfg['w'], t, dt, cherished_cfg['porte_tau'], cherished_cfg['porte_floor'], cherished_cfg['porte_refractory']):
                            _c_int = None; cherished_cfg['etat'] = 5                # on n'y pense pas maintenant ; m intact
                        if _c_int is not None:
                            # le contexte vient du dedans, mais le monde n'est jamais caché par le souvenir (Gepetto, 25/09) :
                            # le geste voit le maximum des deux ; un événement bref est vu à l'instant, et le contexte
                            # qui reste (lissé) décide seul de la fin du rappel
                            _c_att = np.maximum(_c_att, _c_int); cherished_cfg['recalling'] = True; cherished_cfg['etat'] = 6
                    else:
                        cherished_state['recall_center'] = None; cherished_state['recall_res'] = float('nan'); cherished_state['recall_comp'] = None
                        cherished_state['gate'] = None; cherished_state['gate_center'] = None
                    attention_state['s'] = _c_att * (_nu_att + (1.0 - _nu_att) * cherished_state['m']) if cherished_cfg['attach'] else _c_att * _nu_att
                    _dfn = dynamics.attention_delta_fn(fn_t, _theta_att, attention_state['s'], attention_state['K0'], attention_state['kappa'],
                                                       attention_state['garde'], attention_state['island_threshold'])
                    _sal = attention_state['s'] > 0.5
                    attention_state['cost'] = float(np.mean(np.abs(_dfn[_sal]) / np.maximum(fn_t[_sal], 1e-9))) if _sal.any() else 0.0
                    fn_t = np.maximum(fn_t + _dfn, 1e-6)
                fn_history.append(fn_t.copy())
                # Calculer et stocker delta_fn pour l'historique
                delta_fn_t = np.zeros(N)
                for n in range(N):
                    S_i = dynamics.compute_S_i(t, n, history, state)
                    delta_fn_t[n] = dynamics.compute_delta_fn(t, state[n]['alpha'], S_i)
                phi_n_t = dynamics.compute_phi_n(t, state, config)
                
                # Calculer gamma global pour le logging
                latence_config = config.get('latence', {})
                gamma_mode = latence_config.get('gamma_mode', 'static')
                gamma_dynamic = latence_config.get('gamma_dynamic', {})
                k = gamma_dynamic.get('k', None)
                t0 = gamma_dynamic.get('t0', None)
                
                # CORRECTION : Calculer gamma adaptatif AVANT gamma_n_t
                if gamma_mode == 'adaptive_aware':
                    # Calculer gamma avec conscience de G
                    gamma_t, gamma_regime, gamma_journal = dynamics.compute_gamma_adaptive_aware(
                        t, state, history, config, adaptive_state['gamma_journal']
                    )
                    adaptive_state['gamma_journal'] = gamma_journal
                else:
                    # Modes classiques
                    gamma_t = dynamics.compute_gamma(t, gamma_mode, config.get('system', {}).get('T', 100), k, t0)
                    gamma_regime = 'classic'
                
                # Latence expressive gamma par strate (utilise gamma_t calculé)
                # NOUVEAU : Passer l'historique pour modulation locale basée sur t-1
                gamma_n_t = dynamics.compute_gamma_n(t, state, config, gamma_t,
                                                   An_array=An_t,  # An_t est déjà calculé
                                                   fn_array=fn_t,  # fn_t est déjà calculé  
                                                   history=history)
                    
            except Exception as e:
                print(f"❌ ERREUR CRITIQUE calculs dynamiques à t={t}: {e}")
                import traceback
                traceback.print_exc()
                # DÉBOGAGE : Détailler l'exception pour identifier la source
                print(f"🔍 DÉTAILS EXCEPTION:")
                print(f"  - Type: {type(e).__name__}")
                print(f"  - State: {len(state) if state else 'None'}")
                print(f"  - In_t: {In_t.shape if hasattr(In_t, 'shape') else type(In_t)}")
                print(f"  - Config keys: {list(config.keys())}")
                print(f"  - Enveloppe mode: {config.get('enveloppe', {}).get('env_mode', 'None')}")
                
                # En cas d'erreur, utiliser des valeurs de fallback mais alerter clairement
                print(f"🚨 UTILISATION VALEURS FALLBACK - ceci explique les problèmes de reproductibilité !")
                An_t = np.ones(N)
                fn_t = np.ones(N)
                phi_n_t = np.zeros(N)
                gamma_n_t = np.ones(N)
            
            # (φ_reg est désormais calculé en TÊTE de pas — réordonnancement 15/07/2026)
            
            phase_acc += 2 * np.pi * fn_t * dt
            phase_inst = phase_acc + phi_n_t 

            # μ_Rloc(t) : cohérence locale moyenne sur la phase intégrée θ.
            # (La strate sans voisins — bord de spirale ouverte — vaut 1.0 par
            # convention, comme dans kuramoto_local2.)
            _z = np.exp(1j * phase_inst)
            _rl = np.array([np.abs(np.sum(pw * _z[idx])) if idx.size else 1.0
                            for idx, pw in _rloc_neigh])
            mu_Rloc_t = float(np.mean(_rl))
            mu_Rloc_history.append(mu_Rloc_t)

            # c) Sorties observée/attendue
            try:
                On_t = dynamics.compute_On(t, state, An_t, fn_t, phi_n_t, phase_inst, config) if hasattr(dynamics, 'compute_On') else An_t
                if En_t is None:
                    En_t = An_t  # fallback : compute_En absent du module
            except Exception as e:
                print(f"⚠️ Erreur compute On à t={t}: {e}")
                On_t = An_t
                if En_t is None:
                    En_t = An_t

            # MÉMOIRE DE SOI (21/09) : un pas de mémoire une fois née (t ≥ birth_t),
            # Eₙ = Oₙ à l'amplitude remémorée, surprise par strate.
            surprise_mean = None; surprise_max = None
            if memory_state['enabled'] and t >= memory_state['birth_t']:
                _u = dynamics.update_self_memory(self_memory, An_t, fn_t, dt, memory_state['tau_s'], memory_state['tau_l'])
                surprise_mean = float(np.mean(_u['u'])); surprise_max = float(np.max(_u['u'])); memory_state['surprise'] = _u['u']
                _E_mem = dynamics.compute_En_memory(On_t, self_memory)
                if _E_mem is not None:
                    En_t = _E_mem; memory_state['E_last'] = _E_mem

            # d) Régulation/adaptation feedback
            try:                
                # NOUVEAU : Logger détaillé pour diagnostic
                debug_log_data = {
                    't': t,
                    'In_t': In_t.copy() if isinstance(In_t, np.ndarray) else In_t,
                    'An_t': An_t.copy() if isinstance(An_t, np.ndarray) else An_t,
                    'En_t': En_t.copy() if isinstance(En_t, np.ndarray) else En_t,
                    'On_t': On_t.copy() if isinstance(On_t, np.ndarray) else On_t,
                    'phi_n_t': phi_n_t.copy() if isinstance(phi_n_t, np.ndarray) else phi_n_t,  # NOUVEAU: phases
                    'fn_t': fn_t.copy() if isinstance(fn_t, np.ndarray) else fn_t,  # NOUVEAU: fréquences
                    'gamma_n_t': gamma_n_t.copy() if isinstance(gamma_n_t, np.ndarray) else gamma_n_t,  # NOUVEAU: gamma par strate
                    'erreur_n': (En_t - On_t).copy() if isinstance(En_t, np.ndarray) else (En_t - On_t),
                    'G_values': [],
                    'G_archs': []
                }
                
                # NOUVEAU : Calculer et logger r(t) si mode spirale activé
                dynamic_params = config.get('dynamic_parameters', {})
                if dynamic_params.get('dynamic_phi', False):
                    spiral_config = config.get('spiral', {})
                    phi = spiral_config.get('phi', 1.618)
                    epsilon = spiral_config.get('epsilon', 0.05)
                    omega = spiral_config.get('omega', 0.1)
                    theta = spiral_config.get('theta', 0.0)
                    r_t = dynamics.compute_r(t, phi, epsilon, omega, theta)
                    debug_log_data['r_t'] = r_t
                else:
                    debug_log_data['r_t'] = 1.618  # Valeur par défaut
                
                # NOUVEAU : Vérifier si on utilise G adaptatif
                regulation_config = config.get('regulation', {})
                G_arch_mode = regulation_config.get('G_arch', 'tanh')
                G_archs_used = []
                
                # Initialiser G_params par défaut pour éviter l'erreur dans le bloc except
                G_params = {
                    'lambda': regulation_config.get('lambda', 1.0),
                    'alpha': regulation_config.get('alpha', 1.0),
                    'beta': regulation_config.get('beta', 2.0)
                }
                
                # « Un style par instant » : la décision de régulation se prend
                # UNE fois par pas, sur la médiane des erreurs (robuste aux
                # strates aberrantes), AVANT d'écouter qui que ce soit.
                G_plan = None
                if G_arch_mode == 'adaptive_aware':
                    error_summary = float(np.median(np.abs(En_t - On_t)))
                    G_plan = dynamics.decide_G_adaptive_aware(
                        t, gamma_t, adaptive_state['regulation_state'], history, config, error_summary
                    )
                
                for n in range(N):
                    beta_n = state[n]['beta']
                    error_n = En_t[n] - On_t[n]                    
                    # NOUVEAU : Calculer G selon le mode configuré
                    if G_arch_mode == 'adaptive_aware':
                        # « Une écoute par voix » : G évalué sur l'erreur de CETTE
                        # strate, params modulés par CETTE erreur (loi unique).
                        G_value = dynamics.evaluate_G_adaptive_aware(error_n, gamma_t, G_plan)
                        G_arch_used = G_plan['G_arch_used']
                        G_archs_used.append(G_arch_used)
                    else:
                        # G statique selon config
                        if config.get('regulation', {}).get('feedback_mode', 'simple') == 'simple':
                            G_value = error_n  # Pas de régulation
                            G_arch_used = 'identity'
                        else:
                            # Calculer G selon l'archétype configuré
                            G_arch = config.get('regulation', {}).get('G_arch', 'tanh')
                            G_params = {
                                'lambda': config.get('regulation', {}).get('lambda', 1.0),
                                'alpha': config.get('regulation', {}).get('alpha', 1.0),
                                'beta': config.get('regulation', {}).get('beta', 2.0)
                            }
                            G_value = regulation.compute_G(error_n, G_arch, G_params)
                            G_arch_used = G_arch
                        G_archs_used.append(G_arch_used)
                    
                    # Calculer le feedback avec G_value déjà calculé
                    F_n_t_fn[n] = state[n]['beta'] * gamma_t
                    F_n_t_An[n] = state[n]['beta'] * G_value
                    # Log G value per strate for analysis overlay
                    debug_log_data['G_values'].append(G_value)
                    debug_log_data['G_archs'].append(G_arch_used)
                
                # Archétype dominant pour le logging
                # sorted() : ordre d'itération déterministe — l'ordre d'un set de
                # strings dépend de PYTHONHASHSEED, donc en cas d'égalité de votes
                # le tie-break variait d'un process à l'autre (non-reproductible).
                # Égalité résolue par ordre alphabétique, stable et documenté.
                if G_archs_used:
                    G_arch_dominant = max(sorted(set(G_archs_used)), key=G_archs_used.count)
                else:
                    G_arch_dominant = G_arch_mode

            except Exception as e:
                print(f"⚠️ Erreur régulation à t={t}: {e}")
            
            # d) Update état complet du système
            try:
                state = dynamics.update_state(state, An_t, fn_t, phi_n_t, gamma_n_t, F_n_t_fn, F_n_t_An) if hasattr(dynamics, 'update_state') else state
                # Coût CPU : UNE fonction, déterministe et reproductible (plus de
                # temps mur : non reproductible, il n'est plus mesuré).
                cpu_step = metrics.compute_cpu_step_deterministic(N)
            except Exception as e:
                print(f"⚠️ Erreur update state à t={t}: {e}")
            
            # ----------- 3. SIGNAUX GLOBAUX ET MÉTRIQUES -------------
            try:
                # Préparer config enrichie pour mode étendu de S(t)
                # ALIGNEMENT NOTEBOOK: config complète + gamma_n_t pré-calculé
                # ===== SWITCH DE PERCEPTION (spec 15/07/2026) =====
                # Évaluation toutes les T_switch u.t. ; poids GELÉS entre deux
                # évaluations. Repos = neutre (perception nue, doublon de O),
                # pire dimension < seuil -> son filtre-remède (dwell + hystérésis).
                _need_w = perception_mode == 'auto' or perception_state['filter'] not in ('erreur',)
                if _need_w and (t - perception_state['last_eval_t']) >= perception_state['T_switch']:
                    perception_state['last_eval_t'] = t
                    _Wf = perception_state['W_f']
                    if len(history) >= _Wf:
                        # LES SIX MÉTRIQUES DE RÉFÉRENCE sur O(t) brut, fenêtre W_f,
                        # barème SCORE_BRACKETS — LE scoreur du pipeline (metrics.
                        # compute_reference_*), le même que gamma (cible S), les
                        # figures, les rapports et l'analyse (cible O). Rien d'inline.
                        _raw_ref = metrics.compute_reference_metrics(history[-_Wf:], dt, signal='O', N=N)
                        _ref_scores = metrics.score_reference_metrics(_raw_ref)
                        _o_scores = {f: _ref_scores[k] for f, k in metrics.FILTER_TO_SCORE_KEY.items()}
                        if _raw_ref.get('resilience') is None:
                            # Aucun verdict de résilience dans la fenêtre : le filtre
                            # ne peut ni s'engager ni sortir (pas de note inventée).
                            _o_scores.pop('resilience', None)
                        _o_scores = {k: v for k, v in _o_scores.items() if k in perception_state['enabled']}
                        if perception_mode == 'auto':
                            # ANTI-CAMPEMENT (règle d'Andréa) : un remède tenu 3 x dwell
                            # sans amélioration est mis en retrait jusqu'à ce que son
                            # score bouge — on n'insiste pas sur ce qui ne soigne pas.
                            _cur = perception_state['filter']
                            _bl = perception_state.setdefault('blacklist', {})
                            for _k in list(_bl):
                                if _o_scores.get(_k, 5) != _bl[_k]:
                                    del _bl[_k]
                            if _cur != 'neutre' and _cur in _o_scores:
                                _held = t - perception_state['last_switch_t']
                                if _held >= 3 * perception_state['dwell'] and \
                                   _o_scores[_cur] <= perception_state.get('engaged_score', 5):
                                    _bl[_cur] = _o_scores[_cur]
                                    perception_state['filter'] = 'neutre'
                                    perception_state['last_switch_t'] = t
                                    perception_state['weights'] = np.ones(N)
                            # SENS, PAS MAINS : un filtre « observation seule » (innovation,
                            # métrique d'identité) est noté et visible, jamais engagé.
                            _cands = {k: v for k, v in _o_scores.items()
                                      if k not in _bl and k not in metrics.OBSERVE_ONLY_FILTERS}
                            _cur = perception_state['filter']
                            _can_switch = (t - perception_state['last_switch_t']) >= perception_state['dwell']
                            _target = _cur
                            if _cur != 'neutre' and _cur in _o_scores and _o_scores[_cur] >= perception_state['seuil_out'] \
                               and (not _cands or min(_cands.values()) >= perception_state['seuil_in']):
                                _target = 'neutre'
                            if _cands and min(_cands.values()) < perception_state['seuil_in']:
                                _target = min(_cands, key=_cands.get)
                            if _target != _cur and _can_switch:
                                perception_state['filter'] = _target
                                perception_state['last_switch_t'] = t
                                perception_state['engaged_score'] = _o_scores.get(_target, 3)
                        # (re)calcul des poids gelés du filtre actif
                        _f = perception_state['filter']
                        if _f == 'erreur':
                            perception_state['weights'] = None      # chemin natif par pas
                        elif _f == 'neutre':
                            perception_state['weights'] = np.ones(N)
                        else:
                            _Ow = np.array([h['O'] for h in history[-_Wf:]])
                            _Aw = np.array([h['An'] for h in history[-_Wf:]])
                            _Fw = np.array([h['fn'] for h in history[-_Wf:]])
                            # 'innovation' lit C_JS par strate : fenêtre LONGUE de fₙ
                            _Fw_long = (np.array([h['fn'] for h in history[-_W_innov:]])
                                        if _f == 'innovation' else None)
                            _def = dynamics.compute_perception_deficit(_f, _Ow, _Aw, _Fw, dt,
                                                                       fn_long_win=_Fw_long)
                            perception_state['weights'] = dynamics._echelle_attention(_def)
                config_for_S = config.copy()
                config_for_S['state'] = state
                config_for_S['history'] = history
                config_for_S['En_ext'] = En_t
                # UNIFICATION On (15/07 soir, même patron que le En) : le S
                # consomme LE On du pas (section c, celui de l'history et des
                # erreurs) au lieu de le recalculer sur un état déjà muté par
                # la régulation. Rend le filtre neutre EXACTEMENT ΣOₙ.
                config_for_S['On_ext'] = On_t
                # Sentinelle EXPLICITE (15/07, question d'Andréa) : jamais de None
                # ambigu — 'erreur' est demandée en toutes lettres, sinon des poids.
                config_for_S['perception_weights'] = (perception_state['weights']
                    if perception_state['weights'] is not None else 'erreur')
                config_for_S['gamma'] = gamma_t
                S_t = dynamics.compute_S(t, An_t, fn_t, phi_n_t, phase_inst, config_for_S, gamma_n_t=gamma_n_t) if hasattr(dynamics, 'compute_S') else 0.0
                
                # NOUVEAU : Ajouter S(t) au debug log
                if 'debug_log_data' in locals():
                    debug_log_data['S_t'] = S_t
                    
                    # Écrire dans un fichier de debug séparé toutes les 10 itérations
                    if step % 10 == 0 and config.get('debug', {}).get('log_detailed', False):
                        debug_file = os.path.join(loggers['output_dir'], f"debug_detailed_{run_id}.csv")
                        
                        # Créer l'en-tête si c'est la première fois
                        if step == 0:
                            with open(debug_file, 'w') as f:
                                # En-tête avec toutes les strates + r(t) et fréquences
                                header = ['t', 'S_t', 'r_t']
                                for n in range(N):
                                    header.extend([f'In_{n}', f'An_{n}', f'En_{n}', f'On_{n}', f'phi_{n}', f'fn_{n}', f'gamma_{n}', f'erreur_{n}', f'G_{n}', f'G_arch_{n}'])
                                f.write(','.join(header) + '\n')
                        
                        # Écrire les données
                        with open(debug_file, 'a') as f:
                            row = [f"{debug_log_data['t']:.3f}", f"{debug_log_data['S_t']:.12g}", f"{debug_log_data['r_t']:.12g}"]
                            for n in range(N):
                                row.extend([
                                    f"{debug_log_data['In_t'][n] if hasattr(debug_log_data['In_t'], '__getitem__') else debug_log_data['In_t']:.12g}",
                                    f"{debug_log_data['An_t'][n]:.12g}",
                                    f"{debug_log_data['En_t'][n]:.12g}",
                                    f"{debug_log_data['On_t'][n]:.12g}",
                                    f"{debug_log_data['phi_n_t'][n]:.12g}",
                                    f"{debug_log_data['fn_t'][n]:.12g}",
                                    f"{debug_log_data['gamma_n_t'][n]:.12g}",
                                    f"{debug_log_data['erreur_n'][n]:.12g}",
                                    f"{debug_log_data['G_values'][n]:.12g}",
                                    f"{debug_log_data['G_archs'][n]}"
                                ])
                            f.write(','.join(row) + '\n')
                
                C_t = dynamics.compute_C(t, phi_n_t) if hasattr(dynamics, 'compute_C') else 0.0
                delta_fn_array = np.zeros(N)
                for n in range(N):
                    S_i = dynamics.compute_S_i(t, n, history, state)
                    delta_fn_array[n] = dynamics.compute_delta_fn(t, state[n]['alpha'], S_i)
                
                A_t = dynamics.compute_A(t, delta_fn_array) if hasattr(dynamics, 'compute_A') else 0.0
                A_spiral_t = dynamics.compute_A_spiral(t, C_t, A_t) if hasattr(dynamics, 'compute_A_spiral') else 0.0
                E_t = dynamics.compute_E(t, An_t) if hasattr(dynamics, 'compute_E') else 0.0
                # L(t) selon FPS_Paper: argmaxₙ |dAₙ(t)/dt| - nécessite historique An
                if hasattr(dynamics, 'compute_L') and len(An_history) >= 2:
                    L_t = dynamics.compute_L(t, An_history, dt)
                else:
                    L_t = 0
            except Exception as e:
                print(f"⚠️ Erreur signaux globaux à t={t}: {e}")
                S_t = 0.0
                C_t = 0.0
                A_t = 0.0
                A_spiral_t = 0.0
                E_t = 0.0
                L_t = 0.0
            
            # ----------- CALCUL DES MÉTRIQUES AVANCÉES -------------
            # Calcul des moyennes pour A_mean(t) et f_mean(t)
            A_mean_t = np.mean(An_t) if isinstance(An_t, np.ndarray) else An_t
            f_mean_t = np.mean(fn_t) if isinstance(fn_t, np.ndarray) else fn_t
            S_history.append(S_t)
            C_history.append(C_t)

            # Calcul de effort(t) avec deltas depuis history (aligné notebook)
            if len(history) > 0:
                delta_An = An_t - history[-1].get('An', An_t)
                delta_fn = fn_t - history[-1].get('fn', fn_t)
                delta_gamma_n = gamma_n_t - history[-1].get('gamma_n', gamma_n_t)
            else:
                delta_An = np.zeros(N)
                delta_fn = np.zeros(N)
                delta_gamma_n = np.zeros(N)
            
            # EFFORT EN TAUX (lot v3, 14/07/2026) : changement relatif PAR UNITÉ
            # DE TEMPS (avant : par pas — dépendant du dt, audit mesuré 0.67x à
            # dt=0.05). Seuils et barèmes x10 partout : équivalence stricte à dt=0.1.
            effort_t = metrics.compute_effort(delta_An, delta_fn, delta_gamma_n,
                np.mean(An_t), np.mean(fn_t), gamma_t) / dt

            effort_history.append(effort_t)
            if activite_state['ref'] is None and t >= activite_state['t_end']:
                activite_state['ref'] = metrics.activity_reference(
                    [h.get('t') for h in history] + [t], effort_history,
                    activite_state['t_start'], activite_state['t_end'])  # calibré une fois, figé
            activite_ref = activite_state['ref']
            _level = metrics.activity_level(effort_history, activite_state['window_steps']) if activite_ref else None
            activite_rel = (float(_level / activite_ref) if (_level is not None and activite_ref) else None)
            effort_status = metrics.compute_effort_status(effort_t, effort_history, config,
                                                          ref=activite_ref, window=activite_state['window_steps'])
            
            # FLUIDITÉ loggée par pas : la fonction de RÉFÉRENCE (metrics.
            # compute_fluidity = jerk de l'enveloppe fₙ) sur la fenêtre W_f.
            _W_ref = perception_state['W_f']
            fluidity = metrics.compute_fluidity([float(np.mean(f)) for f in fn_history[-_W_ref:]])

            # DISPERSION = métrique d'IDENTITÉ loguée par pas, sur O(t) BRUT :
            # std(ΣOₙ)/√N sur la fenêtre W_f (cloche autour de l'amplitude saine,
            # metrics.DISPERSION_NORM_CENTER = ⟨Aₙ⟩/√2), puis médiane sur
            # DISPERSION_SMOOTH_WINDOWS fenêtres : un point bruité ne fait pas verdict.
            O_sum_history.append(float(np.sum(On_t)))
            dispersion_norm = None
            if len(O_sum_history) >= _W_ref:
                _dn = float(np.std(O_sum_history[-_W_ref:])) / np.sqrt(N)
                dispersion_raw_history.append(_dn)
                dispersion_norm = float(np.median(dispersion_raw_history[-(metrics.DISPERSION_SMOOTH_WINDOWS * _W_ref):]))

            # INNOVATION = moniteur LENT (métrique d'identité, pas d'attention) :
            # complexité statistique C_JS de l'enveloppe fₙ sur une fenêtre LONGUE
            # (_W_innov ≥ 200, bien plus large que W_f~50) ; tant qu'on n'a pas
            # assez de points → None (verdict suspendu, score neutre côté scoreur).
            # C'est LA colonne d'innovation : plus d'entropie spectrale.
            _W_long = max(_W_innov, _W_res)
            _fn_env_long = [float(np.mean(f)) for f in fn_history[-_W_long:]]
            if len(fn_history) >= 200:
                _fn_env = _fn_env_long[-_W_innov:]
                _plane = metrics.compute_innovation_plane(_fn_env, dt)
                innovation_cjs = None if _plane is None else _plane['C']
                innovation_H = None if _plane is None else _plane['H']  # côté de la cloche
            else:
                innovation_cjs = None
                innovation_H = None

            # === TAU MULTI-ÉCHELLES D'ABORD (tau_S requis par decorrelation après) ===
            if len(S_history) >= 20 and len(history) >= 20:
                try:
                    # Signaux alignés notebook : S, gamma, C
                    signals_for_tau = {
                        'S': S_history,
                        'gamma': [h.get('gamma', gamma_t) for h in history][-len(S_history):],
                        'C': C_history
                    }
                    if hasattr(metrics, 'compute_multiple_tau') and len(signals_for_tau) > 1:
                        multi_tau = metrics.compute_multiple_tau(signals_for_tau, dt)
                        tau_S = multi_tau.get('tau_S', dt)
                        tau_gamma = multi_tau.get('tau_gamma', dt)
                        tau_C = multi_tau.get('tau_C', dt)
                    else:
                        tau_S = tau_gamma = tau_C = dt
                    
                    # tau_A_mean et tau_f_mean via compute_tau_parameter (aligné notebook)
                    tau_A_mean = metrics.compute_tau_parameter(history, 'An_mean(t)') if hasattr(metrics, 'compute_tau_parameter') else dt
                    tau_f_mean = metrics.compute_tau_parameter(history, 'fn_mean(t)') if hasattr(metrics, 'compute_tau_parameter') else dt
                    
                except Exception:
                    tau_S = tau_gamma = tau_C = tau_A_mean = tau_f_mean = dt
            else:
                tau_S = tau_gamma = tau_C = tau_A_mean = tau_f_mean = dt
            
            # === COHÉRENCE TEMPORELLE + DÉCORRÉLATION (utilise tau_S calculé au-dessus) ===
            if len(S_history) >= 20:
                temporal_coherence = metrics.compute_temporal_coherence(S_history, dt) if hasattr(metrics, 'compute_temporal_coherence') else 0.5
                if hasattr(metrics, 'extract_decorrelation_metrics'):
                    decorrelation_time, autocorr_tau = metrics.extract_decorrelation_metrics(temporal_coherence, tau_S)
                else:
                    decorrelation_time = autocorr_tau = dt
            else:
                temporal_coherence = 0.5
                autocorr_tau = dt
                decorrelation_time = dt
            
            # Calcul mean_abs_error (régulation)
            mean_abs_error = metrics.compute_mean_abs_error(En_t, On_t) if hasattr(metrics, 'compute_mean_abs_error') else np.mean(np.abs(En_t - On_t))
            
            # NOUVEAU S1: Détecter alignement En ≈ On (log de debug uniquement)
            if mean_abs_error < epsilon_E:
                if config.get('debug', {}).get('log_alignments', False):
                    print(f"[S1] Alignement à t={t:.2f}: |E-O|={mean_abs_error:.4f} < {epsilon_E}")
            
            # Calcul mean_high_effort (effort chronique) - nécessite historique
            if len(effort_history) >= 10:
                mean_high_effort = metrics.compute_mean_high_effort(effort_history, 80) if hasattr(metrics, 'compute_mean_high_effort') else np.percentile(effort_history[-100:], 80)
            else:
                mean_high_effort = 0
            
            # Calcul d_effort_dt (effort transitoire)
            if len(effort_history) >= 2:
                d_effort_dt = metrics.compute_d_effort_dt(effort_history, dt) if hasattr(metrics, 'compute_d_effort_dt') else (effort_history[-1] - effort_history[-2]) / dt
            else:
                d_effort_dt = 0.0
            
            # ----------- RÉSILIENCE : ralentissement critique (couche lente) -----------
            # UNE quantité (New_Attention.md) : autocorrélation à lag calibré des
            # résidus détrendés de l'enveloppe fₙ + variance des résidus + radar.
            resilience_ac = None
            resilience_ac_smooth = None
            resilience_var = None
            resilience_lag = resilience_state['lag']
            resilience_alert_flag = 0
            resilience_quiet = 0
            if len(fn_history) >= _W_res:
                _rs = resilience_state
                if (len(fn_history) - _W_res) % _rs['stride'] == 0:
                    _csd = metrics.compute_resilience_csd(
                        _fn_env_long[-_W_res:], dt, lag=_rs['lag'],
                        min_points=_W_res, peak_ratio=_rs['peak_ratio'],
                        max_peaks=_rs['max_peaks'],
                        quiet_floor_rel=_rs['quiet_floor_rel'],
                        min_lags_per_window=_rs['min_lags_per_window'])
                    if _csd is not None:
                        if _rs['lag'] is None:
                            _rs['lag'] = _csd['lag']  # calibré une fois, figé ensuite
                        _lag = _csd['lag']
                        if not _csd['quiet']:
                            # Le radar ne lit que des fluctuations RÉELLES : les pas
                            # « quiet » ne nourrissent ni la référence calme ni la tendance.
                            _rs['ac_hist'].append(_csd['ac'])
                            _rs['var_hist'].append(_csd['var'])
                        # L'historique est échantillonné tous les `stride` pas : un lag
                        # vaut lag/stride entrées (lissage du score, cadence du radar).
                        # Le score lit la médiane des DERNIERS verdicts, quiet compris
                        # (= 0.0) : un calme retrouvé ramène le score à 5, il ne reste
                        # pas figé sur les fluctuations du transitoire.
                        _lag_entries = max(1, int(round(_lag / _rs['stride'])))
                        _rs['ac_recent'].append(_csd['ac'])
                        _ac_smooth = metrics.smooth_resilience_ac(
                            _rs['ac_recent'], _lag_entries, _rs['smooth_lags'])
                        _alert = metrics.resilience_alert(
                            _rs['ac_hist'], _rs['var_hist'],
                            calm_n=_rs['calm_lags'] * _lag_entries, band=_rs['band'],
                            n_windows=_rs['n_windows'], stride=_lag_entries,
                            gap=_rs['gap_lags'] * _lag_entries)
                        _rs['last'] = (_csd['ac'], _ac_smooth, _csd['var'], _lag, _alert, int(_csd['quiet']))
                if _rs['last'] is not None:
                    (resilience_ac, resilience_ac_smooth, resilience_var,
                     resilience_lag, resilience_alert_flag, resilience_quiet) = _rs['last']
            # UN SEUL barème (SCORE_BRACKETS['resilience']) sur la valeur LISSÉE ;
            # None = verdict suspendu → neutre
            resilience_score = (metrics.NEUTRAL_SCORE if resilience_ac_smooth is None
                                else metrics.score_from_brackets(float(resilience_ac_smooth), 'resilience'))
            
            # NOUVEAU: Calcul des moyennes En, On et gamma pour le logging
            En_mean_t = np.mean(En_t) if isinstance(En_t, np.ndarray) else En_t
            On_mean_t = np.mean(On_t) if isinstance(On_t, np.ndarray) else On_t
            gamma_mean_t = np.mean(gamma_n_t) if isinstance(gamma_n_t, np.ndarray) else gamma_n_t[0] if len(gamma_n_t) > 0 else 1.0
            In_mean_t = np.mean(In_t) if isinstance(In_t, np.ndarray) else In_t
            
            # CRÉER all_metrics ICI
            # ATTENTION : lecteurs (part de strates saillantes ; audibilité = part de la variance
            # de S(t) portée par elles sur la fenêtre de référence ; gardien).
            attention_salient_share = None; attention_audibility = None; attention_garde = None
            if attention_state['enabled']:
                _sal = attention_state['s'] > 0.5
                attention_salient_share = float(np.mean(_sal)); attention_garde = float(attention_state['garde']); attention_audibility = 0.0
                if _sal.any() and len(history) >= 10:
                    _Ow = np.array([_h['O'] for _h in history[-perception_state['W_f']:]], dtype=float)
                    _Sw = _Ow.sum(1)
                    attention_audibility = float(np.mean(_Ow[:, _sal].sum(1) * _Sw) / max(np.mean(_Sw ** 2), 1e-12))
            # LA MÉMOIRE DES MOMENTS CHÉRIS : lecteurs (m max du chœur ; silence 0/1 ; strate rappelée ; porte)
            attention_m_max = None; attention_silence = None; attention_rappel = None; attention_porte = None; attention_rappel_etat = None
            if attention_state['enabled']:
                attention_rappel_etat = float(cherished_cfg['etat'])
                attention_m_max = float(np.max(cherished_state['m'])); attention_silence = float(bool(cherished_state['quiet']))
                attention_rappel = (float(cherished_state['recall_center']) if cherished_cfg['recalling'] and cherished_state['recall_center'] is not None else float('nan'))
                attention_porte = (float(cherished_state['gate']) if cherished_state['gate'] is not None else float('nan'))
            all_metrics = {
                't': t,
                'S(t)': S_t,
                'C(t)': C_t,
                'A_spiral(t)': A_spiral_t,
                'E(t)': E_t,
                'L(t)': L_t,
                'cpu_step(t)': cpu_step,
                'effort(t)': effort_t,
                'activite_ref': activite_ref,  # repos de référence du run (figé), None en calibration
                'activite_rel': activite_rel,  # niveau (médiane sur une respiration) / repos : ce que le score lit
                'A_mean(t)': A_mean_t,
                'f_mean(t)': f_mean_t,
                'fluidity': fluidity,  # jerk de l'enveloppe fₙ (métrique de référence)
                'dispersion_norm': dispersion_norm,  # std(ΣO)/√N lissé, sur O brut (identité) ; None en warmup
                'surprise_mean': surprise_mean,      # mémoire de soi : surprise moyenne du chœur (None avant la naissance)
                'surprise_max': surprise_max,
                'attention_salient_share': attention_salient_share,   # part de strates désignées par la considération
                'attention_audibility': attention_audibility,         # part de S(t) portée par elles (prorata = leur part)
                'attention_garde': attention_garde,                   # frein σ_Rloc (1 = libre, 0 = bloqué)
                'attention_m_max': attention_m_max,                   # le plus chéri du chœur (0 = rien n'est chéri)
                'attention_silence': attention_silence,               # l'entrée se tait (1) : moins surpris que d'habitude, aucun îlot
                'attention_rappel': attention_rappel,                 # strate au cœur du souvenir rappelé (NaN = pas de rappel)
                'attention_porte': attention_porte,                   # bien-être pendant le rappel (NaN hors rappel)
                'attention_rappel_etat': attention_rappel_etat,       # pourquoi : 0 désactivé · 1 l'extérieur parle · 2 pas de silence · 3 rien de chéri · 4 réfractaire · 5 porte · 6 rappel
                'innovation_cjs': innovation_cjs,  # C_JS enveloppe fₙ (moniteur lent d'identité)
                'innovation_H': innovation_H,  # entropie de permutation : H bas = ordre, H haut = bruit
                'temporal_coherence': temporal_coherence,  # Cohérence temporelle
                'autocorr_tau': autocorr_tau,  # Temps de décorrélation
                'decorrelation_time': decorrelation_time,  # Alias pour cohérence
                'tau_S': tau_S,  # Tau du signal global (surface fluide)
                'tau_gamma': tau_gamma,  # Tau de gamma (structure profonde)
                'tau_A_mean': tau_A_mean,  # Tau des amplitudes moyennes
                'tau_f_mean': tau_f_mean,  # Tau des fréquences moyennes
                'mean_abs_error': mean_abs_error,
                'mean_high_effort': mean_high_effort,
                'd_effort_dt': d_effort_dt,
                'mu_Rloc(t)': mu_Rloc_t,
                'resilience_ac': resilience_ac,        # autocorr CSD des résidus de fₙ (moniteur lent, brute)
                'resilience_ac_smooth': resilience_ac_smooth,  # médiane sur quelques lags : ce que le score lit
                'resilience_var': resilience_var,      # variance des résidus (radar : monte avec l'ac)
                'resilience_lag': resilience_lag,      # lag calibré (figé), en pas
                'resilience_alert': resilience_alert_flag,  # radar CSD : 1 = alerte précoce
                'resilience_quiet': resilience_quiet,       # 1 = aucune fluctuation mesurable (ac=0 par convention)
                'resilience_score': resilience_score,
                'perception_filter': perception_state['filter'],
                'En_mean(t)': En_mean_t,
                'On_mean(t)': On_mean_t,
                'gamma': gamma_t,  # Ajouter gamma global
                'gamma_mean(t)': gamma_mean_t,
                'phi': phi_reg,  # Phi adaptatif de régulation (alignement notebook)
                'In_mean(t)': In_mean_t,
                'An_mean(t)': A_mean_t,
                'fn_mean(t)': f_mean_t,
                'G_arch_used': G_arch_dominant if 'G_arch_dominant' in locals() else 'tanh',
            }

            all_metrics['tau_C'] = tau_C
            # Extraire la meilleure paire (γ, G) depuis le gamma_journal
            # ALIGNEMENT NOTEBOOK: utilise gamma_journal au lieu de regulation_memory
            best_gamma, best_G, best_score = extract_best_pair_from_journal(
                adaptive_state.get('gamma_journal')
            )
            
            all_metrics['best_pair_gamma'] = float(best_gamma) if best_gamma is not None else float('nan')
            all_metrics['best_pair_G'] = str(best_G) if best_G is not None else ''
            all_metrics['best_pair_score'] = float(best_score) if best_score is not None else float('nan')
            
            # Résilience : None = verdict suspendu (humilité, sous perturbation
            # mais pas assez vécu). On le garde comme "donnée absente" (cellule
            # vide via NaN), jamais un 0 trompeur qui ressemblerait à un effondrement.
            for _rk in ('resilience_ac', 'resilience_ac_smooth', 'resilience_var', 'resilience_lag', 'innovation_cjs', 'innovation_H', 'activite_ref', 'activite_rel', 'dispersion_norm', 'surprise_mean', 'surprise_max', 'attention_salient_share', 'attention_audibility', 'attention_garde', 'attention_m_max', 'attention_silence', 'attention_rappel', 'attention_porte', 'attention_rappel_etat'):
                if all_metrics.get(_rk) is None:
                    all_metrics[_rk] = float('nan')

            # Appliquer safe_float_conversion à toutes les métriques
            # SAUF les champs textuels et ceux où NaN est intentionnel
            skip_safe_convert = {'effort_status', 'G_arch_used', 'best_pair_G',
                                 'best_pair_gamma', 'best_pair_score', 'tau_A_mean', 'tau_f_mean',
                                 'resilience_ac', 'resilience_ac_smooth', 'resilience_var', 'resilience_lag',
                                 'innovation_cjs', 'innovation_H', 'perception_filter', 'activite_ref', 'activite_rel', 'dispersion_norm', 'surprise_mean', 'surprise_max', 'attention_salient_share', 'attention_audibility', 'attention_garde', 'attention_m_max', 'attention_silence', 'attention_rappel', 'attention_porte', 'attention_rappel_etat'}
            for key in all_metrics:
                if key in skip_safe_convert:
                    continue
                all_metrics[key] = safe_float_conversion(all_metrics[key])
            
            # ----------- 4. VÉRIFICATION NaN/Inf SYSTÉMATIQUE -------------
            # Champs où NaN est intentionnel (= pas de données disponibles)
            nan_ok_fields = {'best_pair_gamma', 'best_pair_score', 'tau_A_mean', 'tau_f_mean',
                             'resilience_ac', 'resilience_ac_smooth', 'resilience_var', 'resilience_lag',
                             'innovation_cjs', 'innovation_H', 'perception_filter', 'activite_ref', 'activite_rel', 'dispersion_norm', 'surprise_mean', 'surprise_max', 'attention_salient_share', 'attention_audibility', 'attention_garde', 'attention_m_max', 'attention_silence', 'attention_rappel', 'attention_porte', 'attention_rappel_etat'}
            nan_inf_detected = False
            for metric_name, metric_value in all_metrics.items():
                if metric_name == 't' or metric_name in nan_ok_fields:
                    continue
                if isinstance(metric_value, (int, float)) and (np.isnan(metric_value) or np.isinf(metric_value)):
                    nan_inf_detected = True
                    alert_msg = f"ALERTE : NaN/Inf détecté à t={t} pour {metrics.metric_label(metric_name)} ({metric_name})={metric_value}"
                    print(alert_msg)
                    os.makedirs(loggers['output_dir'], exist_ok=True)
                    with open(os.path.join(loggers['output_dir'], f"alerts_{run_id}.log"), "a") as alert_file:
                        alert_file.write(f"{alert_msg}\n")
                    if np.isnan(metric_value):
                        all_metrics[metric_name] = 0.0
                    elif np.isinf(metric_value):
                        all_metrics[metric_name] = 1e6 if metric_value > 0 else -1e6
            
            # Si trop de NaN/Inf, option d'arrêter le run
            if nan_inf_detected and config.get('system', {}).get('stop_on_nan_inf', False):
                raise ValueError(f"NaN/Inf détecté à t={t}, arrêt du run")
            
            # ----------- 5. LOGGING DE TOUTES LES MÉTRIQUES ------------
            metrics_dict = all_metrics.copy()
            metrics_dict['effort_status'] = effort_status
            
            # Créer la ligne de log dans l'ordre des colonnes de config
            log_metrics_order = config['system']['logging'].get('log_metrics', sorted(metrics_dict.keys()))
            row_data = []
            for metric in log_metrics_order:
                if metric in metrics_dict:
                    value = metrics_dict[metric]
                    # Formater correctement les valeurs
                    if isinstance(value, str):
                        row_data.append(value)
                    elif isinstance(value, (int, float)):
                        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                            row_data.append('')  # NaN/Inf intentionnel → cellule vide dans le CSV
                        else:
                            row_data.append(f"{value:.12g}" if value != int(value) else str(int(value)))
                    else:
                        row_data.append(str(value))
                else:
                    row_data.append("0.0")
            
            if hasattr(loggers['csv_writer'], 'writerow'):
                loggers['csv_writer'].writerow(row_data)
                # Flush toutes les 100 itérations pour éviter la perte de données
                if step % 100 == 0 and 'csv_file' in loggers:
                    loggers['csv_file'].flush()
            
            # ----------- 6. EXPORT FICHIERS INDIVIDUELS SI N > 10 ------------
            if N > 10 and individual_csv_writers:
                for n in range(N):
                    # Export A_n(t)
                    if f'A_{n}' in individual_csv_writers:
                        individual_csv_writers[f'A_{n}']['writer'].writerow([t, An_t[n]])
                    # Export f_n(t)
                    if f'f_{n}' in individual_csv_writers:
                        individual_csv_writers[f'f_{n}']['writer'].writerow([t, fn_t[n]])
            
            # ----------- 7. BACKUP AUTOMATIQUE ------------------------
            if step % backup_interval == 0 and step > 0:
                if hasattr(utils, "save_simulation_state"):
                    # Créer le dossier checkpoints s'il n'existe pas
                    checkpoint_dir = os.path.join(loggers.get('output_dir', '.'), 'checkpoints')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f"{run_id}_backup_{step}.pkl")
                    utils.save_simulation_state(state, checkpoint_path)
            
            # NOUVEAU : Sauvegarder périodiquement les découvertes couplées
            # Pour T=50 avec dt=0.1, on a 500 steps, donc sauvegarder tous les 100 steps
            save_interval = min(1000, max(100, int(len(t_array) / 5)))  # Au moins 5 sauvegardes
            if gamma_mode == 'adaptive_aware' and step % save_interval == 0 and step > 0:
                if adaptive_state['gamma_journal'] is not None:
                    discoveries_path = os.path.join(
                        loggers.get('output_dir', '.'), 
                        f"discoveries_coupled_{run_id}_step{step}.json"
                    )
                    utils.save_coupled_discoveries(
                        adaptive_state['gamma_journal'], 
                        adaptive_state['regulation_state'], 
                        discoveries_path
                    )
                    if config.get('debug', {}).get('save_discoveries', False):
                        print(f"  💡 Découvertes sauvegardées: {discoveries_path}")
            
            # ----------- 8. HISTORIQUE POUR ANALYSE -------------------
            # Calculer G_values_array et S_contrib_t pour la visualisation par strate
            G_values_list = debug_log_data.get('G_values', [])
            G_values_array = np.array(G_values_list, dtype=float) if len(G_values_list) == N else np.zeros(N)
            # S_contrib_t = On_t * G_values_array (contribution de chaque strate au signal)
            S_contrib_t = (On_t * G_values_array) if isinstance(On_t, np.ndarray) else np.zeros(N)

            history.append({
                't': t, 'S': S_t, 'O': On_t.copy(), 'E': En_t.copy(), 'F': F_n_t_An,
                'gamma_n': gamma_n_t.copy(), 'An': An_t.copy(), 'fn': fn_t.copy(),
                'phi_n_t': phi_n_t,  # AJOUT PR: phases par strate (requis par visualize_stratum_patterns)
                'S_contrib': S_contrib_t,  # AJOUT PR: contribution de chaque strate à S(t)
                'G_values_array': G_values_array,  # AJOUT PR: G par strate (array numpy)
                'C': C_t, 'A_spiral': A_spiral_t,
                'innovation_cjs': innovation_cjs,  # C_JS enveloppe fₙ (moniteur lent d'identité)
                'innovation_H': innovation_H,
                'delta_fn': delta_fn_t, 'S(t)': S_t, 'C(t)': C_t,
                'effort(t)': effort_t, 'activite_ref': activite_ref, 'activite_rel': activite_rel, 'cpu_step(t)': cpu_step,
                'A_mean(t)': A_mean_t, 'f_mean(t)': f_mean_t,
                'fluidity': fluidity,
                'dispersion_norm': dispersion_norm,
                'surprise_mean': surprise_mean, 'surprise_max': surprise_max,
                'attention_salient_share': attention_salient_share, 'attention_audibility': attention_audibility, 'attention_garde': attention_garde,
                'attention_s': (attention_state['s'].copy() if attention_state['enabled'] else None),
                'attention_m_max': attention_m_max, 'attention_silence': attention_silence, 'attention_rappel': attention_rappel, 'attention_porte': attention_porte, 'attention_rappel_etat': attention_rappel_etat,
                'attention_m': (cherished_state['m'].copy() if attention_state['enabled'] else None),
                'mean_abs_error': mean_abs_error,
                'effort_status': effort_status,
                'En_mean(t)': En_mean_t,
                'On_mean(t)': On_mean_t,
                'gamma_mean(t)': gamma_mean_t,
                'In_mean(t)': In_mean_t,
                'In': In_t,
                'An_mean(t)': A_mean_t,
                'fn_mean(t)': f_mean_t,
                'mu_Rloc(t)': mu_Rloc_t,
                'resilience_ac': resilience_ac,
                'resilience_ac_smooth': resilience_ac_smooth,
                'resilience_var': resilience_var,
                'resilience_lag': resilience_lag,
                'resilience_alert': resilience_alert_flag,
                'resilience_quiet': resilience_quiet,
                'resilience_score': resilience_score,
                'perception_filter': perception_state['filter'],
                'gamma': gamma_t,
                'phi': phi_reg,  # Phi adaptatif de régulation
                'G_arch_used': G_arch_dominant if 'G_arch_dominant' in locals() else 'tanh',
                'gamma_regime': gamma_regime if 'gamma_regime' in locals() else 'unknown',
                'best_pair': {'gamma': best_gamma, 'G_arch': best_G, 'score': best_score} if best_gamma is not None else {},
                'best_pair_gamma': float(best_gamma) if best_gamma is not None else None,
                'best_pair_G': str(best_G) if best_G is not None else None,
                'best_pair_score': float(best_score) if best_score is not None else float('nan'),
                'G_values': G_values_list,
                'G_archs': debug_log_data.get('G_archs', []),
                'tau_C': tau_C
            })
            cpu_steps.append(cpu_step)
            En_history.append(En_t.copy() if isinstance(En_t, np.ndarray) else En_t)
            On_history.append(On_t.copy() if isinstance(On_t, np.ndarray) else On_t)
            
            # Rotation des historiques pour éviter explosion mémoire
            if len(history) > MAX_HISTORY_SIZE:
                history = rotate_history(history, MAX_HISTORY_SIZE)
                cpu_steps = rotate_history(cpu_steps, MAX_HISTORY_SIZE)
                effort_history = rotate_history(effort_history, MAX_HISTORY_SIZE)
                S_history = rotate_history(S_history, MAX_HISTORY_SIZE)
                An_history = rotate_history(An_history, MAX_HISTORY_SIZE)
                fn_history = rotate_history(fn_history, MAX_HISTORY_SIZE)
                En_history = rotate_history(En_history, MAX_HISTORY_SIZE)
                On_history = rotate_history(On_history, MAX_HISTORY_SIZE)
                
                # Logger la rotation
                if step % 1000 == 0:
                    print(f"  📊 Rotation des historiques à t={t:.1f} (max_size={MAX_HISTORY_SIZE})")
            
            # ----------- 9. DÉTECTION MODE ALERTE (>3σ) ---------------
            # Vérifier si un critère non-déclencheur présente un écart > 3σ
            alert_sigma = config.get('validation', {}).get('alert_sigma', 3)
            if len(S_history) >= 100:  # Besoin d'historique pour calculer σ
                # Métriques non-déclencheuses surveillées (innovation_cjs : None =
                # warmup, ignoré). Valeur courante ET historique doivent exister.
                for metric_name in ['innovation_cjs', 'fluidity', 'mean_high_effort', 'd_effort_dt']:
                    _cur = all_metrics.get(metric_name)
                    if _cur is None or not np.isfinite(_cur):
                        continue
                    if metric_name in all_metrics:
                        metric_history = [h[metric_name] for h in history[-100:]
                                          if h.get(metric_name) is not None and np.isfinite(h[metric_name])]
                        if len(metric_history) > 10:
                            m_mean = np.mean(metric_history)
                            m_std = np.std(metric_history)
                            if m_std > 0 and abs(all_metrics[metric_name] - m_mean) > alert_sigma * m_std:
                                alert_msg = f"MODE ALERTE : {metrics.metric_label(metric_name)} ({metric_name})={all_metrics[metric_name]:.4f} dévie de >{alert_sigma}σ à t={t}"
                                with open(os.path.join(loggers['output_dir'], f"alerts_{run_id}.log"), "a") as alert_file:
                                    alert_file.write(f"{alert_msg}\n")

            # (print [DEBUG] par pas retiré — item catalogue « ne garder que
            # le log normal ». Le garde config.get('debug', False) était de
            # toute façon truthy par accident : 'debug' est un dict en config.)

        # -- FERMETURE DES FICHIERS INDIVIDUELS --
        if N > 10 and individual_csv_writers:
            for key, writer_info in individual_csv_writers.items():
                writer_info['file'].close()
            print(f"Fichiers individuels A_n_*.csv et f_n_*.csv exportés pour N={N}")
        
        # -- SAUVEGARDE FINALE DES DÉCOUVERTES ADAPTATIVES --
        if gamma_mode == 'adaptive_aware' and adaptive_state['gamma_journal'] is not None:
            final_discoveries_path = os.path.join(
                loggers.get('output_dir', '.'), 
                f"discoveries_coupled_{run_id}_final.json"
            )
            utils.save_coupled_discoveries(
                adaptive_state['gamma_journal'], 
                adaptive_state['regulation_state'], 
                final_discoveries_path
            )
            print(f"  💡 Découvertes finales sauvegardées: {final_discoveries_path}")

        # -- EXPORTS ET SYNTHÈSE FINALE --
        logs = loggers['log_file']
        
        # Fermer le fichier CSV maintenant que la simulation est terminée
        if 'csv_file' in loggers and loggers['csv_file']:
            try:
                loggers['csv_file'].close()
                print(f"✓ Fichier CSV fermé : {logs}")
            except:
                pass
        
        # Calcul des statistiques finales pour le résumé
        # Résilience (moniteur lent CSD) : dernière valeur, moyenne, lag, nb d'alertes
        res_ac_values = [h['resilience_ac'] for h in history
                         if h.get('resilience_ac') is not None and np.isfinite(h['resilience_ac'])]
        res_alerts = int(sum(1 for h in history if h.get('resilience_alert')))
        
        # Innovation (moniteur lent) : dernière valeur et moyenne des verdicts rendus
        innov_values = [h['innovation_cjs'] for h in history
                        if h.get('innovation_cjs') is not None and np.isfinite(h['innovation_cjs'])]
        innov_H_values = [h['innovation_H'] for h in history
                          if h.get('innovation_H') is not None and np.isfinite(h['innovation_H'])]
        
        metrics_summary = {
            'mean_S': np.mean(S_history) if S_history else 0.0,
            'std_S': np.std(S_history) if S_history else 0.0,
            'dispersion_norm': (float(dispersion_norm) if 'dispersion_norm' in locals() and dispersion_norm is not None else None),
            'surprise_mean': (float(np.nanmean([h['surprise_mean'] for h in history if h.get('surprise_mean') is not None]))
                              if any(h.get('surprise_mean') is not None for h in history) else None),
            'surprise_final': (float(surprise_mean) if 'surprise_mean' in locals() and surprise_mean is not None else None),
            'attention_salient_steps': int(sum(1 for h in history if (h.get('attention_salient_share') or 0.0) > 0)),
            'attention_audibility_mean': (float(np.mean([h['attention_audibility'] for h in history if (h.get('attention_salient_share') or 0.0) > 0]))
                                          if any((h.get('attention_salient_share') or 0.0) > 0 for h in history) else None),
            'attention_m_max': (float(np.max(cherished_state['m'])) if attention_state['enabled'] else None),
            'attention_silence_share': (float(np.mean([h['attention_silence'] for h in history if h.get('attention_silence') is not None]))
                                        if any(h.get('attention_silence') is not None for h in history) else None),
            'attention_rappel_steps': int(sum(1 for h in history if h.get('attention_rappel') is not None and np.isfinite(h['attention_rappel']))),
            'activite_ref': (float(activite_state['ref']) if activite_state['ref'] is not None else None),
            'activite_rel': (float(activite_rel) if 'activite_rel' in locals() and activite_rel is not None else None),
            'mean_effort': np.mean(effort_history) if effort_history else 0.0,
            'max_effort': np.max(effort_history) if effort_history else 0.0,
            'mean_cpu_step': np.mean(cpu_steps) if cpu_steps else 0.0,
            'innovation_cjs': float(innov_values[-1]) if innov_values else float('nan'),
            'innovation_cjs_mean': float(np.mean(innov_values)) if innov_values else float('nan'),
            'innovation_H': float(innov_H_values[-1]) if innov_H_values else float('nan'),
            'final_fluidity': float(fluidity) if 'fluidity' in locals() and fluidity is not None else 0.0,
            'final_mean_abs_error': float(mean_abs_error) if mean_abs_error is not None else 0.0,
            'mean_C': float(np.mean(C_history)) if C_history else float('nan'),
            # Résilience CSD : None = verdict suspendu (run trop court) → NaN, jamais un 0/1 trompeur.
            'resilience_ac': float(res_ac_values[-1]) if res_ac_values else float('nan'),
            'resilience_ac_mean': float(np.mean(res_ac_values)) if res_ac_values else float('nan'),
            'resilience_ac_smooth': (float(resilience_ac_smooth)
                                     if 'resilience_ac_smooth' in locals() and resilience_ac_smooth is not None
                                     else float('nan')),
            'resilience_lag': (int(resilience_state['lag']) if resilience_state['lag'] else None),
            'resilience_alerts': res_alerts,
            'resilience_quiet_share': (float(np.mean([1.0 if h.get('resilience_quiet') else 0.0
                                                      for h in history if h.get('resilience_ac') is not None]))
                                       if any(h.get('resilience_ac') is not None for h in history) else float('nan')),
            'resilience_score': int(resilience_score) if 'resilience_score' in locals() else metrics.NEUTRAL_SCORE,
            'total_steps': len(t_array),
            'recorded_steps': len(S_history),
            'dt': float(dt),
            'N': int(N),
            'mode': 'FPS'
        }
        
        # Calcul du checksum des logs pour intégrité
        if hasattr(utils, "compute_checksum") and os.path.exists(logs):
            checksum = utils.compute_checksum(logs)
            with open(os.path.join(loggers['output_dir'], f"checksum_{run_id}.txt"), "w") as f:
                f.write(f"Checksum for {logs}: {checksum}\n")
        
        # 👉 Vérifier que la simulation a couvert toutes les étapes prévues
        expected_steps = len(t_array)
        actual_steps = len(S_history)
        if actual_steps < expected_steps:
            msg = f"La simulation s'est terminée prématurément : {actual_steps}/{expected_steps} steps enregistrés"
            print(f"⚠️  {msg}")
            if strict:
                raise RuntimeError(msg)
        
        return deep_convert({
            'history': history,
            'logs': logs,
            'metrics': metrics_summary,
            'cpu_steps': cpu_steps,
            'effort_history': effort_history,
            'S_history': S_history,
            'run_id': run_id,
            'An_history': An_history,
            'fn_history': fn_history,
            'En_history': En_history,
            'On_history': On_history,
            'gamma_journal': adaptive_state.get('gamma_journal'),
        })

    except Exception as e:
        # -- LOG D'ERREUR CRITIQUE POUR POST-MORTEM --
        err_path = os.path.join(loggers['output_dir'], f"error_{run_id}.log")
        with open(err_path, "a") as f: 
            f.write(f"Error at t={t if 't' in locals() else 'unknown'}\n")
            f.write(traceback.format_exc())
        
        # Fermer les fichiers individuels en cas d'erreur
        if N > 10 and 'individual_csv_writers' in locals():
            for key, writer_info in individual_csv_writers.items():
                try:
                    writer_info['file'].close()
                except:
                    pass
        
        if hasattr(utils, "handle_crash_recovery"): 
            crash_state = {
                'strates': state,
                't': t if 't' in locals() else 0,
                'mode': config['system'].get('mode', 'FPS'),
                'error_info': str(e),
                'all_metrics': all_metrics  # Maintenant toujours défini
            }
            utils.handle_crash_recovery(crash_state, loggers, e)
        raise e

# --- MODE KURAMOTO (GROUPE CONTRÔLE) ---------------------------
def run_kuramoto_simulation(config, loggers):
    """
    Implémentation stricte de la dynamique Kuramoto (phase-only).
    - Utilise K=0.5, ωᵢ~U[0,1], phases initialisées random, N strates, dt, T comme FPS
    - Log les mêmes métriques que FPS pour comparaison
    """
    if hasattr(kuramoto, "run_kuramoto_simulation"):
        return kuramoto.run_kuramoto_simulation(config, loggers)
    else:
        # Implémentation minimale Kuramoto
        N = config['system']['N']
        T = config['system']['T']
        dt = config['system'].get('dt', 0.05)
        K = 0.5
        
        # Initialisation
        phases = np.random.uniform(0, 2*np.pi, N)
        frequencies = np.random.uniform(0, 1, N)
        t_array = np.arange(0, T, dt)
        
        history = []
        S_history = []
        C_history = []
        cpu_steps = []
        
        for t in t_array:
            # Équation Kuramoto
            dphases_dt = frequencies.copy()
            for i in range(N):
                coupling_sum = 0.0
                for j in range(N):
                    coupling_sum += np.sin(phases[j] - phases[i])
                dphases_dt[i] += (K / N) * coupling_sum
            
            # Mise à jour
            phases += dphases_dt * dt
            phases = phases % (2 * np.pi)
            
            # Ordre global (paramètre d'ordre de Kuramoto)
            order_param = np.abs(np.mean(np.exp(1j * phases)))
            
            # Cohérence des phases adjacentes (équivalent à C(t))
            if N > 1:
                C_t = np.mean([np.cos(phases[(i+1)%N] - phases[i]) for i in range(N)])
            else:
                C_t = 1.0
            
            # Signal global (somme des oscillateurs)
            S_t = np.sum(np.sin(phases))

            # Coût CPU : une seule fonction, déterministe
            cpu_step = metrics.compute_cpu_step_deterministic(N)

            # Log
            metrics_dict = {
                't': t,
                'S(t)': S_t,
                'C(t)': C_t,
                'E(t)': order_param,
                'L(t)': 0,
                'cpu_step(t)': cpu_step,
                'effort(t)': 0.0,
                'A_mean(t)': 1.0,
                'f_mean(t)': np.mean(frequencies),
                'effort_status': 'stable'
            }
            
            # Écrire dans le CSV
            log_metrics_order = config['system']['logging'].get('log_metrics', sorted(metrics_dict.keys()))
            row_data = []
            for metric in log_metrics_order:
                if metric in metrics_dict:
                    value = metrics_dict[metric]
                    if isinstance(value, str):
                        row_data.append(value)
                    else:
                        row_data.append(f"{value:.12g}")
                else:
                    row_data.append("0.0")
            
            if hasattr(loggers['csv_writer'], 'writerow'):
                loggers['csv_writer'].writerow(row_data)
            
            history.append({'t': t, 'S': S_t, 'C': C_t, 'order': order_param})
            S_history.append(S_t)
            C_history.append(C_t)
            cpu_steps.append(cpu_step)
        
        # Fermer le fichier CSV
        if 'csv_file' in loggers and loggers['csv_file']:
            try:
                loggers['csv_file'].close()
            except:
                pass
        
        return deep_convert({
            "logs": loggers.get('log_file', 'kuramoto_log.csv'),
            "metrics": {
                'mean_S': float(np.mean(S_history)) if S_history else 0.0,
                'std_S': float(np.std(S_history)) if S_history else 0.0,
                'mean_C': float(np.mean(C_history)) if C_history else 0.0,
                'mean_cpu_step': float(np.mean(cpu_steps)) if cpu_steps else 0.0,
                'final_order': float(order_param) if order_param is not None else 0.0,
                'mode': 'Kuramoto'
            },
            "history": history,
            "run_id": loggers['run_id'],
            "S_history": S_history
        })

# --- MODE NEUTRAL (OSCs FIXES, PAS DE FEEDBACK) ----------------
def run_neutral_simulation(config, loggers):
    """
    Simulation neutre : phases/amplitudes fixes, sans rétroaction ni spiralisation.
    Sert de contrôle pour valider l'émergence spécifique de la FPS.
    """
    N = config['system']['N']
    T = config['system']['T']
    dt = config['system'].get('dt', 0.05)
    t_array = np.arange(0, T, dt)
    
    # Paramètres fixes
    amplitudes = np.ones(N)
    frequencies = np.linspace(0.8, 1.2, N)  # Fréquences légèrement différentes
    phases = np.zeros(N)
    
    history = []
    S_history = []
    cpu_steps = []
    
    for t in t_array:
        # Signal sans feedback
        S_t = np.sum(amplitudes * np.sin(2 * np.pi * frequencies * t + phases))
        
        # Pas de modulation, pas de feedback
        C_t = 1.0  # Cohérence fixe
        E_t = np.max(amplitudes)
        L_t = 0
        
        # Coût CPU : une seule fonction, déterministe
        cpu_step = metrics.compute_cpu_step_deterministic(N)

        # Log
        metrics_dict = {
            't': t,
            'S(t)': S_t,
            'C(t)': C_t,
            'E(t)': E_t,
            'L(t)': L_t,
            'cpu_step(t)': cpu_step,
            'effort(t)': 0.0,
            'A_mean(t)': 1.0,
            'f_mean(t)': np.mean(frequencies),
            'effort_status': 'stable',
            'innovation_cjs': 0.0,  # fréquences fixes : ordre pur → C_JS nul
            'mean_abs_error': 0.0
        }
        
        # Écrire dans le CSV
        log_metrics_order = config['system']['logging'].get('log_metrics', sorted(metrics_dict.keys()))
        row_data = []
        for metric in log_metrics_order:
            if metric in metrics_dict:
                value = metrics_dict[metric]
                if isinstance(value, str):
                    row_data.append(value)
                else:
                    row_data.append(f"{value:.12g}")
            else:
                row_data.append("0.0")
        
        if hasattr(loggers['csv_writer'], 'writerow'):
            loggers['csv_writer'].writerow(row_data)
        
        history.append({'t': t, 'S': S_t})
        S_history.append(S_t)
        cpu_steps.append(cpu_step)
    
    # Fermer le fichier CSV
    if 'csv_file' in loggers and loggers['csv_file']:
        try:
            loggers['csv_file'].close()
        except:
            pass
    
    return deep_convert({
        "logs": loggers.get('log_file', 'neutral_log.csv'),
        "metrics": {
            'mean_S': float(np.mean(S_history)) if S_history else 0.0,
            'std_S': float(np.std(S_history)) if S_history else 0.0,
            'mean_cpu_step': float(np.mean(cpu_steps)) if cpu_steps else 0.0,
            'mode': 'Neutral'
            },
        "history": history,
        "run_id": loggers['run_id'],
        "S_history": S_history
    })

# --------- AFFICHAGE TODO ET INCOMPLETS DU PIPELINE -------------
def list_todos():
    print("\n--- TODO FPS PIPELINE ---")
    print("Compléter/raffiner tous les modules suivants pour une version 100% rigoureuse et falsifiable :")
    print("  * Animation 3D de la spirale")
    print("  * Dashboard temps réel")
    print("- Documentation : README, matrice critère-terme, journal seuils\n")

# --------- ARGPARSE POUR EXECUTION CLI -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run FPS Simulation")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--mode", type=str, default="FPS", choices=["FPS", "Kuramoto", "neutral"], help="Simulation mode")
    parser.add_argument("--list-todos", action="store_true", help="List all TODO items")
    parser.add_argument("--strict", action="store_true", help="Interrompre si le run ne couvre pas tous les pas prévus (falsifiabilité stricte)")
    args = parser.parse_args()
    
    if args.list_todos:
        list_todos()
    else:
        result = run_simulation(args.config, args.mode, strict=args.strict)
        print(f"\nSimulation terminée : {result['run_id']}")
        print(f"Logs : {result['logs']}")
        print(f"Métriques finales : {metrics.labelled_summary(result['metrics'])}")

"""
-----------------------
FIN simulate.py FPS V1.3 (exhaustif, strict, explicite)
Ce fichier doit servir de **référence** pour toutes les évolutions du pipeline FPS. 
Chaque fonction à compléter/raffiner l'est dans les modules correspondants.
"""