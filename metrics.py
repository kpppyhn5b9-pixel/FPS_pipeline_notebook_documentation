"""
metrics.py - Calcul des métriques FPS
Version exhaustive conforme à la feuille de route FPS V1.3
---------------------------------------------------------------
NOTE FPS – Plasticité méthodologique :
Les métriques (effort, innovation, etc.) sont ajustables : toute
modification ou alternative testée doit être documentée et laissée
ouverte dans la config.
---------------------------------------------------------------

Ce module quantifie tous les aspects du système FPS :
- Performance computationnelle (CPU, mémoire)
- Effort d'adaptation (interne, transitoire, chronique)
- Qualité dynamique (fluidité, stabilité, innovation)
- Résilience et régulation
- Détection d'états (stable, transitoire, chronique, chaotique)

Les métriques sont le miroir empirique du système, permettant
la falsification et le raffinement continu.

(c) 2025 Gepetto & Andréa Gadal & Claude 🌀
"""

import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
import warnings


# ============== MÉTRIQUES DE PERFORMANCE ==============


# Coût nominal d'une opération de couplage (secondes "modèle", indépendant de
# la machine). Calibré pour que les tailles N usuelles tombent dans une bande
# réaliste (N≈100 → ~1e-4 s/strate), comme le temps mur observé jadis.
CPU_NOMINAL_OP_COST = 1e-6


def compute_cpu_step_deterministic(N: int, op_cost: float = CPU_NOMINAL_OP_COST) -> float:
    """
    Coût CPU par strate, DÉTERMINISTE et reproductible.

    Remplace le temps mur afin que deux runs identiques produisent exactement
    les mêmes logs, sur n'importe quelle machine. Le coût CPU est LOGGÉ
    (diagnostic), il n'est pas noté : il ne fait pas partie des six métriques
    de référence et ne pilote rien.

    Modèle : le travail par strate est dominé par la somme de couplage sur les
    N strates → O(N) opérations par strate. On renvoie op_cost * N : une
    estimation RELATIVE, machine-indépendante, identique d'un run à l'autre.
    C'est LA mesure de coût CPU du pipeline (colonne cpu_step(t)) : le temps
    mur n'est plus mesuré, il n'est pas reproductible.

    Args:
        N: nombre de strates
        op_cost: coût nominal d'une opération de couplage (s "modèle")

    Returns:
        float: coût CPU par strate déterministe
    """
    if N <= 0:
        return 0.0
    return op_cost * N


# ============== MÉTRIQUES D'EFFORT ==============

def compute_effort(delta_An_array: np.ndarray, delta_fn_array: np.ndarray, 
                   delta_gamma_n_array: np.ndarray, An_ref: float, fn_ref: float, 
                   gamma_ref: float) -> float:
    """
    Calcule l'effort d'adaptation interne du système.
    
    Version corrigée qui évite l'explosion quand les valeurs de référence → 0 :
    effort = Σₙ [|ΔAₙ|/(|Aₙ|+ε) + |Δfₙ|/(|fₙ|+ε) + |Δγₙ|/(|γₙ|+ε)]
    
    Args:
        delta_An_array: variations d'amplitude
        delta_fn_array: variations de fréquence
        delta_gamma_n_array: variations de latence
        An_ref: valeur de référence pour An (mean ou max)
        fn_ref: valeur de référence pour fn
        gamma_ref: valeur de référence pour gamma
    
    Returns:
        float: effort total normalisé
    
    Note:
        L'effort mesure maintenant le changement RELATIF par rapport
        aux valeurs actuelles, pas par rapport au maximum global.
    """
    effort = 0.0
    
    # Protection contre division par zéro avec epsilon adaptatif
    # Pour éviter overflow quand les valeurs de référence sont très petites
    # Epsilon s'adapte à l'échelle des valeurs de référence
    ref_scale = max(abs(An_ref), abs(fn_ref), abs(gamma_ref))
    epsilon = max(0.001, 0.01 * ref_scale)
    
    # Calcul de l'effort comme changement relatif
    # Si la référence est trop petite, on utilise epsilon
    An_denom = max(abs(An_ref), epsilon)
    fn_denom = max(abs(fn_ref), epsilon) 
    gamma_denom = max(abs(gamma_ref), epsilon)
    
    # Somme des efforts relatifs pour chaque composante
    effort_An = np.sum(np.abs(delta_An_array)) / An_denom
    effort_fn = np.sum(np.abs(delta_fn_array)) / fn_denom
    effort_gamma = np.sum(np.abs(delta_gamma_n_array)) / gamma_denom
    
    effort = effort_An + effort_fn + effort_gamma
    
    # Vérification finale contre NaN/Inf et saturation
    if np.isnan(effort) or np.isinf(effort):
        # Si malgré tout on a un problème, retourner une valeur par défaut
        effort = 0.0
    
    # NOUVEAU : Saturation pour éviter les valeurs extrêmes
    MAX_EFFORT = 100.0  # Limite raisonnable pour l'effort
    effort = min(effort, MAX_EFFORT)
    
    return effort


def compute_effort_status(effort_t: float, effort_history: List[float], 
                          config: Dict) -> str:
    """
    Détermine le statut de l'effort : stable, transitoire ou chronique.
    
    Args:
        effort_t: effort actuel
        effort_history: historique des efforts
        config: configuration (seuils)
    
    Returns:
        str: "stable", "transitoire" ou "chronique"
    
    Logique:
        - stable: effort dans la norme
        - transitoire: pic temporaire (adaptation ponctuelle)
        - chronique: effort élevé persistant (système en lutte)
    """
    if len(effort_history) < 10:
        # Pas assez d'historique - on considère stable par défaut
        return "stable"
    
    # Calcul des statistiques sur les 10 derniers pas
    recent_efforts = effort_history[-10:]
    mean_recent = np.mean(recent_efforts)
    
    # Seuils adaptatifs basés sur l'historique complet
    if len(effort_history) >= 50:
        # Moyenne et écart-type sur une fenêtre plus large
        long_term = effort_history[-50:]
        mean_long = np.mean(long_term)
        std_long = np.std(long_term)
        
        # Détection transitoire : pic > 2σ
        if effort_t > mean_long + 2 * std_long:
            return "transitoire"
        
        # Détection chronique : moyenne récente élevée
        # Si std_long est très petit (effort constant), utiliser un seuil absolu
        if std_long < 0.01:
            # Effort constant - utiliser les seuils de config
            thresholds = config.get('to_calibrate', {})
            if mean_recent > thresholds.get('effort_chronique_threshold', 75.0):
                return "chronique"
        else:
            # Effort variable - utiliser la logique adaptative
            if mean_recent > mean_long + std_long:
                # Vérifier la persistance
                high_count = sum(1 for e in recent_efforts if e > mean_long + std_long)
                if high_count >= 7:  # 70% du temps récent
                    return "chronique"
    
    # Sinon, utiliser des seuils fixes depuis config
    thresholds = config.get('to_calibrate', {})
    
    # Seuil transitoire
    if effort_t > thresholds.get('effort_transitoire_threshold', 150.0):
        return "transitoire"
    
    # Seuil chronique sur la moyenne
    if mean_recent > thresholds.get('effort_chronique_threshold', 75.0):
        return "chronique"
    
    return "stable"


def compute_mean_high_effort(effort_history: List[float], percentile: int = 80) -> float:
    """
    Calcule la moyenne haute de l'effort (percentile élevé).
    
    Mesure l'effort chronique en regardant les valeurs hautes.
    
    Args:
        effort_history: historique complet des efforts
        percentile: percentile à considérer (80 par défaut)
    
    Returns:
        float: moyenne des efforts au-dessus du percentile
    """
    if len(effort_history) == 0:
        return 0.0
    
    if len(effort_history) < 10:
        # Pas assez de données - retourner la moyenne simple
        return np.mean(effort_history)
    
    # Filtrer les valeurs aberrantes (NaN, Inf, et valeurs énormes)
    MAX_REASONABLE_EFFORT = 1e6
    filtered_efforts = []
    
    for effort in effort_history:
        if np.isfinite(effort) and abs(effort) < MAX_REASONABLE_EFFORT:
            filtered_efforts.append(effort)
    
    if len(filtered_efforts) == 0:
        return 0.0
    
    # Calculer le seuil du percentile sur les valeurs filtrées
    threshold = np.percentile(filtered_efforts, percentile)
    
    # Moyenner les valeurs au-dessus
    high_efforts = [e for e in filtered_efforts if e >= threshold]
    
    if len(high_efforts) == 0:
        return threshold
    
    return np.mean(high_efforts)


def compute_d_effort_dt(effort_history: List[float], dt: float) -> float:
    """
    Calcule la dérivée temporelle de l'effort.
    
    Mesure les variations brusques d'effort (transitoires).
    
    Args:
        effort_history: historique des efforts
        dt: pas de temps
    
    Returns:
        float: dérivée d_effort/dt
    """
    if len(effort_history) < 2:
        return 0.0
    
    # Dérivée simple entre les deux derniers points
    d_effort = effort_history[-1] - effort_history[-2]
    d_effort_dt = d_effort / dt
    
    # Protection contre les valeurs aberrantes
    # Si la dérivée est énorme, c'est probablement dû à un overflow
    MAX_DERIVATIVE = 1e6  # Limite raisonnable pour une dérivée
    
    if np.isnan(d_effort_dt) or np.isinf(d_effort_dt):
        return 0.0
    elif abs(d_effort_dt) > MAX_DERIVATIVE:
        # Clipper à la valeur max avec le bon signe
        return MAX_DERIVATIVE if d_effort_dt > 0 else -MAX_DERIVATIVE
    
    return d_effort_dt


# ============== MÉTRIQUES DE QUALITÉ DYNAMIQUE ==============


# ============== MÉTRIQUES DE RÉGULATION ==============

def compute_mean_abs_error(En_array: np.ndarray, On_array: np.ndarray) -> float:
    """
    Calcule l'erreur absolue moyenne entre attendu et observé.
    
    mean(|Eₙ(t) - Oₙ(t)|)
    
    Mesure la qualité de la régulation : faible = bonne convergence.
    
    Args:
        En_array: sorties attendues
        On_array: sorties observées
    
    Returns:
        float: erreur absolue moyenne
    """
    if len(En_array) == 0 or len(On_array) == 0:
        return 0.0
    
    return np.mean(np.abs(En_array - On_array))


# ============== MÉTRIQUES DE RÉSILIENCE ==============

def compute_t_retour(S_history: List[float], t_choc: int, dt: float, 
                     threshold: float = 0.95) -> float:
    """
    Calcule le temps de retour à l'équilibre après perturbation.
    
    Temps pour revenir à 95% de l'état pré-choc.
    
    Args:
        S_history: historique du signal
        t_choc: indice temporel du choc
        dt: pas de temps
        threshold: seuil de retour (0.95 = 95%)
    
    Returns:
        float: temps de retour en unités de temps
    
    Note (v2, settling-time) : on mesure le retour de l'ENVELOPPE (|S| lissé, le
    NIVEAU) vers la bande pré-choc, et non le |S| INSTANTANÉ vers sa moyenne — un
    signal oscillant ne se pose jamais sur sa moyenne, l'ancienne version ratait
    donc le retour (validé sur signaux fabriqués : un retour rapide était noté au
    pire). Le lissage ajoute ~1 u.t. de latence, intégrée dans les barèmes.
    """
    if t_choc >= len(S_history) or t_choc < 10:
        return 0.0

    S_abs = np.abs(np.asarray(S_history, dtype=float))
    # Enveloppe causale = |S| lissé sur ~une période (le niveau, pas l'oscillation).
    w = max(5, int(round(1.2 / dt)))
    env = np.array([float(np.mean(S_abs[max(0, i-w):i+1])) for i in range(len(S_abs))])

    # Niveau de référence pré-choc (sur la même fenêtre de lissage).
    base = float(np.mean(env[max(0, t_choc-w):t_choc]))
    if base < 1e-6:
        base = float(np.max(env[max(0, t_choc-w):t_choc]))
    if base < 1e-6:
        # Signal quasi nul avant le choc : pas de retour mesurable → pénalité max.
        return (len(S_history) - t_choc) * dt

    tolerance = (1 - threshold) * base
    for i in range(t_choc + 1, len(env)):
        if abs(env[i] - base) <= tolerance:
            return (i - t_choc) * dt

    # Pas encore revenu à l'équilibre
    return (len(S_history) - t_choc) * dt


def compute_continuous_resilience(C_history: List[float], S_history: List[float],
                                 perturbation_active: bool = True) -> Optional[float]:
    """
    Calcule la résilience sous perturbation continue.

    Pour une perturbation continue (ex: sinusoïdale), mesure la capacité
    du système à maintenir sa cohérence et stabilité.

    Args:
        C_history: historique de la cohérence C(t)
        S_history: historique du signal S(t)
        perturbation_active: si une perturbation est active

    Returns:
        float: score de résilience continue [0, 1]
        - 1.0 = excellente résilience (maintien de cohérence)
        - 0.0 = mauvaise résilience (perte de cohérence)
        None = verdict suspendu (sous perturbation mais pas encore assez vécu)

    Note:
        Métrique complémentaire à t_retour pour perturbations non-ponctuelles.
        Sans perturbation, cette métrique ne s'applique pas (None) : la tenue
        de soi au repos est mesurée par μ_Rloc en boucle (lot v2), jamais
        décrétée. Sous perturbation mais avec moins de 20 points, on ne sait
        PAS encore juger → verdict suspendu (None, humilité de démarrage).
    """
    if not perturbation_active:
        # Pas de perturbation : la tenue de soi est MESURÉE ailleurs (μ_Rloc
        # en boucle, lot v2), jamais décrétée. Verdict non applicable ici.
        return None
    if len(C_history) < 20:
        # Sous perturbation mais pas assez vécu pour juger : verdict suspendu.
        return None
    
    # Utiliser une fenêtre récente
    window_size = min(100, len(C_history))
    C_recent = C_history[-window_size:]
    S_recent = S_history[-window_size:]
    
    # 1. Stabilité de la cohérence sous perturbation
    # Une bonne résilience = C(t) reste élevé malgré la perturbation
    C_mean = np.mean(C_recent)
    C_std = np.std(C_recent)
    
    # Score de stabilité de C(t)
    C_stability_score = C_mean / (1 + C_std) if C_mean > 0 else 0.0
    
    # 2. Stabilité ET expressivité de S(t)
    S_mean = np.mean(S_recent)
    S_std = np.std(S_recent)
    S_power = np.mean(np.abs(S_recent))
    
    if S_power > 0:
        # Coefficient de variation - mesure de stabilité relative
        S_cv = S_std / S_power
        
        # Score de stabilité de S(t) : CV faible = signal stable
        if S_cv < 0.5:  # Très stable
            S_stability_score = 1.0
        elif S_cv < 1.0:  # Stable
            S_stability_score = 0.9 - 0.2 * (S_cv - 0.5)
        elif S_cv < 2.0:  # Moyennement stable
            S_stability_score = 0.7 - 0.3 * (S_cv - 1.0)
        else:  # Instable
            S_stability_score = 0.4
        
        # Score d'expressivité : le signal garde son amplitude
        expressivity_score = np.tanh(2 * S_power)  # Saturation douce
        
        # Combiner stabilité et expressivité de S(t)
        S_combined_score = 0.6 * S_stability_score + 0.4 * expressivity_score
    else:
        # Signal effondré
        S_combined_score = 0.0
    
    # 3. Résilience combinée
    # Pondération : C(t) reste le plus important, mais S(t) compte significativement
    continuous_resilience = 0.6 * C_stability_score + 0.4 * S_combined_score
    
    return float(np.clip(continuous_resilience, 0.0, 1.0))


# ── PANNEAU (lisibilité, 15/07/2026) ────────────────────────────────────────
# Tu cherches le calcul de résilience par ENVELOPPES (μ_Rloc / effort / erreur,
# médiane ± k·IQR, gel pendant épisodes, recalibration prudente) ? Il vit dans
# init_resilience_envelope_state (l.~1592) et update_resilience_envelope
# (l.~1653). La fonction ci-dessous est L'AUTRE BRANCHE de l'aiguillage
# (simulate l.~868) : resilience_v2.mode='auto' sert les enveloppes AU REPOS
# et bascule ici PENDANT les perturbations déclarées (machinerie t_retour /
# récupération d'épisode). Deux juges, chacun son régime — pas un doublon.
# ─────────────────────────────────────────────────────────────────────────────
def compute_adaptive_resilience(config: Dict, metrics: Dict,
                               C_history: List[float] = None,
                               S_history: List[float] = None,
                               t_choc: int = None, dt: float = None) -> Dict[str, Any]:
    """
    Calcule la résilience adaptative selon le type de perturbation.
    
    Cette fonction unifie t_retour et continuous_resilience en sélectionnant
    automatiquement la métrique appropriée selon le type de perturbation configuré.
    
    Args:
        config: configuration complète
        metrics: métriques déjà calculées (pour t_retour existant)
        C_history: historique de cohérence (pour continuous_resilience)
        S_history: historique du signal
        t_choc: indice temporel du choc (pour t_retour)
        dt: pas de temps
    
    Returns:
        Dict contenant:
        - 'type': 'punctual' ou 'continuous'
        - 'value': valeur de résilience [0, 1] ou temps
        - 'score': score normalisé [1-5]
        - 'metric_used': 't_retour' ou 'continuous_resilience'
    """
    # dt : utiliser l'horloge réelle de la config, jamais un défaut fantôme.
    # (L'ancien défaut 0.05 ≠ dt=0.1 de la config faussait t_retour si un
    # appelant oubliait de passer dt.)
    if dt is None:
        dt = config.get('system', {}).get('dt', 0.1) if config else 0.1

    # Déterminer le type de perturbation
    has_continuous_perturbation = False
    perturbation_type = 'none'
    
    # Nouvelle structure avec input.perturbations
    input_cfg = config.get('system', {}).get('input', {})
    perturbations = input_cfg.get('perturbations', [])
    
    for pert in perturbations:
        pert_type = pert.get('type', 'none')
        if pert_type in ['sinus', 'bruit', 'rampe']:
            has_continuous_perturbation = True
            perturbation_type = 'continuous'
            break
        elif pert_type == 'choc':
            perturbation_type = 'punctual'
    
    # Si pas trouvé dans la nouvelle structure, vérifier l'ancienne (compatibilité)
    if perturbation_type == 'none' and config:
        old_pert = config.get('system', {}).get('perturbation', {})
        old_type = old_pert.get('type', 'none')
        if old_type in ['sinus', 'bruit', 'rampe']:
            has_continuous_perturbation = True
            perturbation_type = 'continuous'
        elif old_type == 'choc':
            perturbation_type = 'punctual'
    
    result = {
        'type': perturbation_type,
        'metric_used': None,
        'value': None,
        'score': 3,  # Score par défaut
        'raw_value': None
    }
    
    if has_continuous_perturbation:
        # Utiliser continuous_resilience
        result['metric_used'] = 'continuous_resilience'
        
        # Préférer la valeur moyenne si disponible
        cont_resilience = metrics.get('continuous_resilience_mean', 
                                     metrics.get('continuous_resilience', None))
        
        # Si pas disponible, calculer
        if cont_resilience is None and C_history is not None and S_history is not None:
            cont_resilience = compute_continuous_resilience(C_history, S_history, True)
        
        if cont_resilience is not None:
            result['value'] = cont_resilience
            result['raw_value'] = cont_resilience
            # UN SEUL barème : celui du switch (SCORE_BRACKETS['resilience']).
            result['score'] = score_from_brackets(float(cont_resilience), 'resilience')
    
    else:
        # Utiliser t_retour pour perturbation ponctuelle ou absence
        result['metric_used'] = 't_retour'
        
        # Récupérer t_retour existant
        t_retour = metrics.get('resilience_t_retour', 
                               metrics.get('t_retour', None))
        
        # Si pas disponible et qu'on a les données nécessaires, calculer
        if t_retour is None and S_history is not None and t_choc is not None:
            t_retour = compute_t_retour(S_history, t_choc, dt)
        
        if t_retour is not None:
            result['raw_value'] = t_retour
            # Normaliser t_retour en score [0, 1] pour cohérence
            # Plus t_retour est petit, meilleure est la résilience
            if t_retour == 0:
                result['value'] = 1.0
            else:
                result['value'] = 1.0 / (1.0 + t_retour)
            # UN SEUL barème : celui du switch, appliqué à la valeur normalisée
            # (c'est ce que le switch et les scoreurs consomment). NB : avec
            # 1/(1+t_retour) et le plancher ~2 du settling-time, l'échelle
            # 4-5 n'est pas atteignable — calibration de la normalisation à
            # faire côté barème, pas par un second barème ici.
            result['score'] = score_from_brackets(result['value'], 'resilience')
    
    return result


# ============== FONCTIONS SPÉCIALISÉES ==============


def compute_correlation_effort_cpu(effort_history: List[float], 
                                   cpu_history: List[float]) -> float:
    """
    Calcule la corrélation entre l'effort et le coût CPU.
    
    Permet de détecter si l'effort interne se traduit en charge computationnelle.
    
    Args:
        effort_history: historique des efforts
        cpu_history: historique des temps CPU
    
    Returns:
        float: coefficient de corrélation [-1, 1]
    """
    if len(effort_history) < 10 or len(cpu_history) < 10:
        return 0.0
    
    # Aligner les longueurs
    min_len = min(len(effort_history), len(cpu_history))
    effort_aligned = effort_history[-min_len:]
    cpu_aligned = cpu_history[-min_len:]
    
    # Corrélation de Pearson
    correlation_matrix = np.corrcoef(effort_aligned, cpu_aligned)
    
    if correlation_matrix.shape == (2, 2):
        return correlation_matrix[0, 1]
    else:
        return 0.0


# ============== LES SIX MÉTRIQUES DE RÉFÉRENCE ET LEUR SCORING ==============
#
# Décision 19/09/2026 : le switch de perception (simulate.py) est LA référence.
# Six métriques, une fenêtre W_f, un barème (SCORE_BRACKETS), un scoreur
# (compute_reference_scores). Tout ce qui note le système passe par ici :
#   - le switch, les visualisations, les rapports, l'analyse de batch :
#     signal='O' (O(t) = ΣOₙ, le signal brut non pondéré) ;
#   - gamma_adaptive_aware : signal='S' (S(t) perçu courant) — mêmes
#     métriques, même barème, seule la CIBLE change.
# Plus de scoreur parallèle, plus de barème en dur, plus d'inline.
#
# ============================================================================
# SOURCE UNIQUE DES BARÈMES DE SCORES (unification 14/07/2026, ré-appliquée
# le 15/07). Consommée par compute_reference_scores, donc par le switch de
# perception, gamma, visualize, main, analyze. Toute calibration se fait ICI.
# ============================================================================
# NOMS HONNÊTES (audit de validité, cf. cahier) — clés RENOMMÉES :
#   'dispersion' (ex-'stability') : écart-type de S = AMPLITUDE des variations,
#                  PAS la structure. Aveugle à l'ordre. Score DORMANT (constant →
#                  inoffensif). Le vrai axe "structure" reste ouvert (cahier).
#   'activite'   (ex-'effort') : churn des paramètres (taux de changement), PAS
#                  du stress. Compteur d'activité, pas de souffrance.
SCORE_BRACKETS = {
    'dispersion':  {'direction': 'lower',  'thresholds': [0.5, 1.0, 2.0, 3.0]},  # amplitude, pas structure
    'regulation': {'direction': 'lower',  'thresholds': [0.1, 0.3, 0.5, 1.0]},
    # fluidity : mesurée par le JERK de l'enveloppe fₙ (cf. cahier de validation).
    # Seuils calibrés in-situ par balayage d'intensité (repos 0,94 → bruit fort
    # 0,78, monotone) pour que toute l'échelle 1-5 soit atteignable :
    #   repos→5 · bruit1→4 · bruit2→3 · bruit4→2 · bruit≥8→1.
    'fluidity':   {'direction': 'higher', 'thresholds': [0.91, 0.87, 0.83, 0.80], 'ge': False},
    'resilience': {'direction': 'higher', 'thresholds': [0.90, 0.75, 0.60, 0.40], 'ge': True},
    # innovation : complexité statistique C_JS de l'enveloppe fₙ (cloche inversée
    # native — bruit ET ordre → bas ; nouveauté structurée → haut). Métrique
    # d'IDENTITÉ : voyant de santé, ~invariant en régime sain (FPS ~0.30 → 5).
    # Monotone en C_JS suffit : C_JS encode déjà la cloche ordre/bruit.
    'innovation': {'direction': 'higher', 'thresholds': [0.28, 0.22, 0.15, 0.08], 'ge': False},
    # effort : PROVISOIRE (15/07, dossier 4 seeds — médiane repos 59, p75 73,
    # p90 229, corrélation effort<->erreur +0.54, choc invisible). Le score 1
    # permanent des anciens seuils était connu faux ; ceux-ci notent la
    # croisière 3 EN ATTENDANT la certification sur-effort des campagnes
    # (question OUVERTE : l'effort est un compteur d'activité, pas de stress).
    'activite':   {'direction': 'lower',  'thresholds': [30.0, 45.0, 75.0, 150.0]},  # ex-'effort' : churn, pas stress
}


def score_from_brackets(value: float, key: str) -> int:
    """Score 1-5 depuis la table unique SCORE_BRACKETS."""
    b = SCORE_BRACKETS[key]
    t = b['thresholds']
    if b['direction'] == 'lower':
        return 5 if value < t[0] else 4 if value < t[1] else 3 if value < t[2] else 2 if value < t[3] else 1
    if b.get('ge', False):
        return 5 if value >= t[0] else 4 if value >= t[1] else 3 if value >= t[2] else 2 if value >= t[3] else 1
    return 5 if value > t[0] else 4 if value > t[1] else 3 if value > t[2] else 2 if value > t[3] else 1


# Clés du switch (filtres de perception) ↔ clés de barème (SCORE_BRACKETS).
FILTER_TO_SCORE_KEY = {
    'stabilite':  'dispersion',
    'fluidite':   'fluidity',
    'innovation': 'innovation',
    'erreur':     'regulation',
    'effort':     'activite',
    'resilience': 'resilience',
}
SCORE_KEY_TO_FILTER = {v: k for k, v in FILTER_TO_SCORE_KEY.items()}
REFERENCE_SCORE_KEYS = ('dispersion', 'regulation', 'fluidity', 'resilience', 'innovation', 'activite')
# Libellés des figures et rapports, dans l'ordre d'affichage.
SCORE_KEY_LABELS = {
    'dispersion': 'Stabilité',
    'regulation': 'Régulation',
    'fluidity':   'Fluidité',
    'resilience': 'Résilience',
    'innovation': 'Innovation',
    'activite':   'Effort interne',
}
NEUTRAL_SCORE = 3


def reference_window(config: Optional[Dict], dt: float) -> int:
    """
    Fenêtre W_f du switch de perception, en pas : max(10, W_f_t / dt).

    Source unique de la fenêtre de scoring (config perception.W_f_t, défaut
    5 unités de temps) : le switch, gamma, les figures et l'analyse de batch
    regardent tous la même durée.
    """
    w_t = float((config or {}).get('perception', {}).get('W_f_t', 5.0))
    return max(10, int(round(w_t / max(float(dt), 1e-12))))


def compute_dispersion(signal_window) -> float:
    """
    Dispersion (score 'dispersion', filtre 'stabilite') : écart-type du signal
    sur la fenêtre. C'est l'AMPLITUDE des variations, pas la structure.
    """
    x = np.asarray(signal_window, dtype=float).ravel()
    x = x[np.isfinite(x)]
    return float(np.std(x)) if len(x) else 0.0


def compute_fluidity(fn_mean_window) -> float:
    """
    Fluidité (score 'fluidity', filtre 'fluidite') : douceur du TEMPO du
    système = jerk de l'enveloppe de fréquence fₙ.

        fluidity = 1 / (1 + std(d²f) / std(d¹f))   sur la moyenne de fₙ par pas

    Validée sur banc de signaux de référence et calibrée in-situ (cahier de
    validation). Sans dimension, invariante à l'échelle. < 4 points → 1.0
    (démarrage : rien de saccadé encore).

    Args:
        fn_mean_window: moyenne de fₙ à chaque pas de la fenêtre
    """
    f = np.asarray(fn_mean_window, dtype=float).ravel()
    if len(f) < 4:
        return 1.0
    d1, d2 = np.diff(f), np.diff(f, 2)
    return 1.0 / (1.0 + float(np.std(d2) / (np.std(d1) + 1e-12)))


def compute_innovation_cjs(fn_mean_window, dt: float, d: int = 4,
                           min_points: int = 200) -> Optional[float]:
    """
    Innovation (score 'innovation') = complexité statistique de Jensen-Shannon
    (Rosso/MPR) de l'enveloppe fₙ.

    C'est une CLOCHE inversée native : ~0 pour l'ordre pur (sinus) ET pour le
    bruit pur, maximale pour la nouveauté *structurée* (chaos déterministe).
    Le bruit score donc BAS — contrairement à l'entropie, qui le note "innovant"
    (défaut validé au banc de référence).

    Métrique d'IDENTITÉ, pas d'attention : un MONITEUR lent. τ est auto-calibré
    depuis la relaxation (1/e de l'autocorrélation) de l'enveloppe, de sorte que
    la fenêtre ordinale (d-1)·τ couvre ~un temps de relaxation. Fenêtre longue
    requise (permutation stable) : renvoie None si < min_points (→ score neutre).

    Args:
        fn_mean_window: moyenne de fₙ par pas (l'enveloppe), fenêtre LONGUE
        dt: pas de temps (non utilisé directement ; garde la signature homogène)
        d: ordre des motifs de permutation (défaut 4 → 24 motifs)
        min_points: longueur minimale pour un C_JS stable (défaut 200)

    Returns:
        float dans [0, ~0.5] (C_JS), ou None si la fenêtre est trop courte.
    """
    from itertools import permutations
    x = np.asarray(fn_mean_window, dtype=float).ravel()
    if len(x) < min_points:
        return None
    xc = x - x.mean()
    if float(np.dot(xc, xc)) < 1e-12:
        return 0.0  # signal plat : ordre pur → innovation nulle
    ac = np.correlate(xc, xc, mode='full')[len(x) - 1:]
    ac = ac / ac[0]
    below = np.where(ac < 1.0 / np.e)[0]
    relax = int(below[0]) if len(below) else 10
    tau = max(1, int(round(relax / (d - 1))))
    # garde : la fenêtre ordinale doit tenir largement dans le signal
    tau = min(tau, max(1, (len(x) // 2) // (d - 1)))
    idx = {p: i for i, p in enumerate(permutations(range(d)))}
    counts = np.zeros(len(idx))
    n = len(x) - (d - 1) * tau
    if n <= 0:
        return None
    for i in range(n):
        counts[idx[tuple(np.argsort(x[i:i + d * tau:tau]))]] += 1
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    Nc = len(p)
    pe = np.ones(Nc) / Nc

    def _sh(q):
        qq = q[q > 0]
        return float(-(qq * np.log(qq)).sum())

    H = _sh(p) / np.log(Nc)
    js = _sh((p + pe) / 2) - 0.5 * _sh(p) - 0.5 * _sh(pe)
    delta = np.zeros(Nc); delta[0] = 1.0
    js_max = _sh((delta + pe) / 2) - 0.5 * _sh(delta) - 0.5 * _sh(pe)
    return float((js / js_max) * H) if js_max > 0 else 0.0


def innovation_deficit(c_js: Optional[float]) -> float:
    """
    Déficit d'innovation (filtre de perception 'innovation'), sur la même
    métrique que le score : 1 − C_JS / seuil_haut, borné à 0. Le seuil haut est
    celui du barème (SCORE_BRACKETS['innovation'], score 5) : source unique.
    None (fenêtre trop courte) → 0 (aucun verdict, aucun poids).
    """
    if c_js is None or not np.isfinite(c_js):
        return 0.0
    c_top = SCORE_BRACKETS['innovation']['thresholds'][0]
    return float(max(0.0, 1.0 - float(c_js) / c_top))


def _num(v) -> Optional[float]:
    """float fini ou None (cellule vide d'un CSV, None, NaN, texte)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def signal_series(history_window: List[Dict], signal: str, N: Optional[int] = None) -> List[float]:
    """Série du signal cible : 'O' = ΣOₙ (brut), 'S' = S(t) (perçu)."""
    out = []
    if signal == 'S':
        for h in history_window:
            v = _num(h.get('S(t)', h.get('S')))
            if v is not None:
                out.append(v)
        return out
    if N is None:
        for h in history_window:
            O = h.get('O')
            if O is not None and np.ndim(O) > 0:
                N = len(O)
                break
    for h in history_window:
        O = h.get('O')
        if O is not None and np.ndim(O) > 0:
            out.append(float(np.sum(O)))
        elif _num(h.get('On_mean(t)')) is not None:
            # ΣOₙ = N · moyenne (CSV rechargé : seule la moyenne est loggée)
            out.append(_num(h['On_mean(t)']) * (N if N else 1))
    return out


def _fn_mean_series(history_window: List[Dict]) -> List[float]:
    out = []
    for h in history_window:
        fn = h.get('fn')
        if fn is not None and np.ndim(fn) > 0:
            out.append(float(np.mean(fn)))
        elif _num(h.get('fn_mean(t)')) is not None:
            out.append(_num(h['fn_mean(t)']))
    return out


def compute_reference_metrics(history_window: List[Dict], dt: float,
                              signal: str = 'O', N: Optional[int] = None) -> Dict[str, Optional[float]]:
    """
    Valeurs brutes des six métriques de référence sur une fenêtre d'historique.

    Mêmes calculs que le switch de perception, à l'identique :
      dispersion  = std(signal)                       (compute_dispersion)
      fluidity    = jerk de la moyenne de fₙ           (compute_fluidity)
      innovation  = complexité statistique C_JS de fₙ  (compute_innovation_cjs,
                    moniteur lent lu dans la colonne innovation_cjs)
      regulation  = moyenne de mean_abs_error par pas  (compute_mean_abs_error)
      activite    = moyenne de effort(t)               (compute_effort)
      resilience  = moyenne de adaptive_resilience     (compute_adaptive_resilience
                    ou enveloppes) ; None si aucun verdict dans la fenêtre

    Args:
        history_window: liste de dicts par pas (history[-W:])
        dt: pas de temps
        signal: 'O' (brut, ΣOₙ — switch, figures, rapports, analyse) ou
                'S' (perçu — gamma_adaptive_aware)
        N: nombre de strates (utile quand seul On_mean(t) est disponible)
    """
    series = [v for v in signal_series(history_window, signal, N) if np.isfinite(v)]
    fn_means = _fn_mean_series(history_window)

    errors = []
    for h in history_window:
        e = _num(h.get('mean_abs_error'))
        if e is None and h.get('E') is not None and h.get('O') is not None:
            e = _num(compute_mean_abs_error(np.asarray(h['E'], dtype=float), np.asarray(h['O'], dtype=float)))
        if e is not None:
            errors.append(e)
    efforts = [v for v in (_num(h.get('effort(t)')) for h in history_window) if v is not None]
    resil = [v for v in (_num(h.get('adaptive_resilience')) for h in history_window) if v is not None]
    # innovation = MONITEUR LENT lu depuis les logs (C_JS de l'enveloppe fₙ,
    # calculé sur fenêtre longue dans simulate → 'innovation_cjs'). On lit la
    # dernière valeur disponible dans la fenêtre ; None (warmup / CSV sans la
    # colonne) → score neutre. Métrique d'identité, signal-agnostique (gamma S
    # et switch O lisent la même valeur).
    innov = [v for v in (_num(h.get('innovation_cjs')) for h in history_window) if v is not None]

    return {
        'dispersion': compute_dispersion(series) if series else 0.0,
        'fluidity':   compute_fluidity(fn_means),
        'innovation': (innov[-1] if innov else None),
        'regulation': float(np.mean(errors)) if errors else 0.0,
        'activite':   float(np.mean(efforts)) if efforts else 0.0,
        'resilience': float(np.mean(resil)) if resil else None,
    }


def score_reference_metrics(raw: Dict[str, Optional[float]]) -> Dict[str, int]:
    """Scores 1-5 des six métriques via SCORE_BRACKETS. Valeur absente → neutre 3."""
    scores = {}
    for key in REFERENCE_SCORE_KEYS:
        v = raw.get(key)
        if v is None or not np.isfinite(v):
            scores[key] = NEUTRAL_SCORE  # verdict suspendu : neutre, jamais inventé
        else:
            scores[key] = int(score_from_brackets(float(v), key))
    return scores


def neutral_reference_scores() -> Dict[str, int]:
    return {key: NEUTRAL_SCORE for key in REFERENCE_SCORE_KEYS}


def compute_reference_scores(history_window: List[Dict], dt: float,
                             signal: str = 'O', N: Optional[int] = None,
                             min_points: int = 3) -> Dict[str, int]:
    """
    LE scoreur du pipeline : scores 1-5 des six métriques de référence sur
    une fenêtre d'historique, barème SCORE_BRACKETS.

    Utilisé par le switch de perception (signal='O'), gamma_adaptive_aware
    (signal='S'), les figures, les rapports et l'analyse de batch (signal='O').
    Moins de min_points pas → scores neutres.
    """
    if history_window is None or len(history_window) < min_points:
        return neutral_reference_scores()
    return score_reference_metrics(compute_reference_metrics(history_window, dt, signal, N))


def labelled_scores(scores: Dict[str, int]) -> Dict[str, int]:
    """{'Stabilité': 4, 'Régulation': 3, ...} pour les figures et rapports."""
    return {SCORE_KEY_LABELS[k]: int(scores.get(k, NEUTRAL_SCORE)) for k in REFERENCE_SCORE_KEYS}


def compute_tau_parameter(history, param_name, min_samples=20):
    """
    Calcule le temps caractéristique d'un paramètre depuis l'historique.
    Utilise l'autocorrélation pour estimer le temps de décorrélation.
    """
    if len(history) < min_samples:
        return None
    
    values = []
    for h in history[-min_samples:]:
        val = h.get(param_name)
        if val is None and param_name == 'An_mean(t)':
            val = h.get('An_mean')
        if val is None and param_name == 'fn_mean(t)':
            val = h.get('fn_mean')
        if val is not None:
            values.append(val)
    
    if not values or len(values) < 15:
        return None
    
    try:
        values_array = np.array(values)
        values_centered = values_array - np.mean(values_array)
        
        if np.std(values_centered) < 1e-10:
            return None
        
        autocorr = np.correlate(values_centered, values_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        if autocorr[0] > 1e-10:
            autocorr = autocorr / autocorr[0]
        else:
            return None
        
        positive_autocorr = np.maximum(autocorr, 0)
        
        if len(history) >= 2:
            dt = history[-1]['t'] - history[-2]['t']
        else:
            dt = 0.1
        
        tau = np.sum(positive_autocorr) * dt
        tau = min(tau, 10.0)
        
        return tau
    except Exception:
        return None


def extract_decorrelation_metrics(temporal_coherence_val, tau_S_val):
    """
    Extrait les métriques de décorrélation depuis les calculs existants.
    """
    decorrelation_time = tau_S_val if tau_S_val is not None else None
    
    if temporal_coherence_val is not None and tau_S_val is not None:
        autocorr_tau = tau_S_val * (1 + temporal_coherence_val)
    else:
        autocorr_tau = None
    
    return (decorrelation_time, autocorr_tau)


# ============== MÉTRIQUES DE COHÉRENCE TEMPORELLE ==============

def compute_temporal_coherence(S_history: List[float], dt: float, 
                              tau_target: float = 1.0) -> float:
    """
    Mesure la cohérence temporelle du système - sa capacité à maintenir
    une "mémoire" douce sans être ni trop rigide ni chaotique.
    
    Calcule le temps de décorrélation et compare à une cible optimale.
    Un système idéal garde une mémoire modérée (ni oubli immédiat ni rigidité).
    
    Args:
        S_history: historique du signal global S(t)
        dt: pas de temps
        tau_target: temps de décorrélation cible (défaut: 1.0 seconde)
    
    Returns:
        float: score de cohérence temporelle entre 0 et 1
        
    Note:
        - Score proche de 1 = mémoire optimale
        - Score proche de 0 = trop chaotique ou trop rigide
    """
    if len(S_history) < 20:
        return 0.5  # Pas assez de données, score neutre
    
    try:
        # Conversion et centrage pour robustesse
        S_array = np.array(S_history) - np.mean(S_history)
        
        # Protection contre variance nulle (signal constant)
        if np.std(S_array) < 1e-10:
            return 0.8  # Signal stable = bonne cohérence temporelle
        
        # Autocorrélation normalisée (méthode robuste)
        autocorr = np.correlate(S_array, S_array, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Garder seulement lag positifs
        
        # Normalisation avec protection division par zéro
        if autocorr[0] > 1e-10:
            autocorr = autocorr / autocorr[0]
        else:
            return 0.5  # Problème de normalisation, score neutre
        
        # Temps de décorrélation - méthode intégrée plus robuste
        # Calcule la surface sous la courbe d'autocorrélation (plus stable)
        # Cette méthode évite les problèmes de premier croisement dans les signaux complexes
        
        # Méthode 1: Intégrale de l'autocorrélation positive
        positive_autocorr = np.maximum(autocorr, 0)  # Garder seulement les valeurs positives
        tau_integrated = np.sum(positive_autocorr) * dt
        
        # Méthode 2: Ajustement exponentiel robuste sur les premiers points
        try:
            # Prendre les 20 premiers points pour un fit exponentiel
            n_fit = min(20, len(autocorr))
            x_fit = np.arange(n_fit) * dt
            y_fit = autocorr[:n_fit]
            
            # Fit robuste : log(max(y, 0.01)) pour éviter log(0)
            y_log = np.log(np.maximum(y_fit, 0.01))
            
            # Régression linéaire simple
            if len(x_fit) > 2:
                slope = (y_log[-1] - y_log[0]) / (x_fit[-1] - x_fit[0])
                tau_exponential = -1.0 / slope if slope < -0.01 else tau_target
            else:
                tau_exponential = tau_target
        except:
            tau_exponential = tau_target
        
        # Choisir la méthode la plus conservative (temps le plus long)
        tau_decorr = max(tau_integrated, tau_exponential)
        
        # Limitation raisonnable
        tau_decorr = min(tau_decorr, 10.0 * tau_target)
        
        # Score basé sur la proximité à tau_target
        # Utilise une gaussienne centrée pour pénaliser les écarts
        deviation = abs(tau_decorr - tau_target) / max(tau_target, 0.1)
        score = np.exp(-0.5 * deviation**2)  # Gaussienne plus douce que l'exponentielle
        
        # Clamp final pour sécurité
        return float(np.clip(score, 0.0, 1.0))
        
    except Exception:
        # Fallback robuste en cas de problème imprévu
        return 0.5


def compute_autocorr_tau(S_history: List[float], dt: float, 
                        method: str = 'one_over_e') -> float:
    """
    Calcule le temps de décorrélation (tau) d'un signal.
    
    Alias plus explicite pour la cohérence temporelle, retournant
    directement la valeur tau en secondes.
    
    Args:
        S_history: historique du signal
        dt: pas de temps
        method: méthode de calcul ('one_over_e', 'first_zero', 'integrated')
    
    Returns:
        float: temps de décorrélation en secondes
    """
    if len(S_history) < 10:
        return dt  # Minimum possible
    
    try:
        S_array = np.array(S_history) - np.mean(S_history)
        
        if np.std(S_array) < 1e-10:
            return 10.0 * dt  # Signal constant = longue corrélation
        
        # Autocorrélation
        autocorr = np.correlate(S_array, S_array, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        if autocorr[0] > 1e-10:
            autocorr = autocorr / autocorr[0]
        else:
            return dt
        
        # Différentes méthodes selon le choix
        if method == 'one_over_e':
            threshold = 1.0 / np.e
        elif method == 'first_zero':
            threshold = 0.0
        elif method == 'integrated':
            # Temps intégré = intégrale de l'autocorrélation
            return float(np.sum(autocorr) * dt)
        else:
            threshold = 1.0 / np.e  # Défaut
        
        # Chercher le premier croisement
        for i in range(1, min(len(autocorr), 100)):
            if autocorr[i] < threshold:
                return float(i * dt)
        
        # Si pas trouvé, corrélation très longue
        return float(min(len(autocorr), 100) * dt)
        
    except Exception:
        return dt


# Alias pour compatibilité
def compute_decorrelation_time(S_history: List[float], dt: float) -> float:
    """Alias pour compute_autocorr_tau avec méthode par défaut."""
    return compute_autocorr_tau(S_history, dt, method='one_over_e')


# Fonction utilitaire pour calculer plusieurs tau en une fois
def compute_multiple_tau(signals_dict: Dict[str, List[float]], dt: float) -> Dict[str, float]:
    """
    Calcule les temps de décorrélation pour plusieurs signaux simultanément.
    
    Utile pour observer les différentes échelles temporelles du système
    (surface vs structure profonde).
    
    Args:
        signals_dict: dictionnaire {nom_signal: historique_valeurs}
        dt: pas de temps
    
    Returns:
        dict: {nom_signal: tau_en_secondes}
        
    Exemple:
        signals = {'S': S_history, 'gamma': gamma_history}
        taus = compute_multiple_tau(signals, 0.1)
        # → {'S': 0.1, 'gamma': 2.4}
    """
    result = {}
    
    for signal_name, signal_history in signals_dict.items():
        if len(signal_history) >= 20:  # Minimum pour autocorrélation
            try:
                tau = compute_autocorr_tau(signal_history, dt)
                result[f'tau_{signal_name}'] = tau
            except Exception:
                result[f'tau_{signal_name}'] = dt  # Fallback
        else:
            result[f'tau_{signal_name}'] = dt  # Pas assez de données
    
    return result

# ============================================================================
# RÉSILIENCE SIGNAL-DRIVEN PAR ENVELOPPES DE SANTÉ (étage 2, spec 13/07/2026)
# État EXPLICITE passé en argument (jamais d'attribut de fonction caché).
# Entièrement déterministe : aucun RNG, aucun temps mur.
# ============================================================================

from collections import deque as _deque


def init_resilience_envelope_state(config: Dict) -> Dict:
    """Initialise l'état de la résilience par enveloppes (bloc resilience_v2)."""
    cfg = (config or {}).get('resilience_v2', {})
    # Fenêtres en UNITÉS DE TEMPS (hygiène sans-dimension, 14/07/2026) :
    # les clés *_t priment et sont converties en pas via dt ; les anciennes
    # clés en pas restent acceptées (rétro-compatibilité). Les défauts en
    # temps reconvertissent EXACTEMENT aux anciens pas à dt=0.1 (neutre).
    _dt = (config or {}).get('system', {}).get('dt', 0.1)
    def _steps(key_t, default_t, key_steps, default_steps):
        if key_t in cfg:
            return max(1, int(round(cfg[key_t] / _dt)))
        if key_steps in cfg:
            return int(cfg[key_steps])
        return max(1, int(round(default_t / _dt)))
    W_env = _steps('W_env_t', 10.0, 'W_env', 100)
    signals = ['mu_Rloc', 'effort', 'mean_abs_error']
    directions = {'mu_Rloc': ['low'], 'effort': ['high'], 'mean_abs_error': ['high']}
    if cfg.get('rigidity_watch', False):
        directions['mu_Rloc'] = ['low', 'high']
    return {
        'cfg': {
            'W_env': W_env,
            'k_iqr': float(cfg.get('k_iqr', 1.5)),
            'iqr_floor_rel': float(cfg.get('iqr_floor_rel', 0.05)),
            'iqr_floor_abs': float(cfg.get('iqr_floor_abs', 1e-6)),
            'warmup': _steps('warmup_t', 2.0, 'warmup', 20),
            'debounce': _steps('debounce_t', 0.3, 'debounce', 3),
            'T_recal': _steps('T_recal_t', 20.0, 'T_recal', 200),
            'W_mem': _steps('W_mem_t', 50.0, 'W_mem', 500),
            'tau_ref': float(cfg.get('tau_ref', 10.0)),
            'w_depth': float(cfg.get('w_depth', 0.5)),
            'w_time': float(cfg.get('w_time', 0.5)),
            'aggregation': str(cfg.get('aggregation', 'rms')),  # mean|max|rms (rms : campagne 13/07)
            'depth_memory': str(cfg.get('depth_memory', 'worst')),  # worst|mean (worst : campagne 13/07,
            # « aussi résilient que sa pire récupération récente » — la moyenne diluait les catastrophes)
            'depth_compression': str(cfg.get('depth_compression', 'log')),  # log|linear (log : discrimine
            # mieux les sévérités, la sigmoïde d'entrée saturant déjà les intensités extrêmes)
            'recal_stability_factor': float(cfg.get('recal_stability_factor', 1.5)),
            'spike_threshold': float(cfg.get('spike_threshold', 1.0)),  # D >= 1 largeur d'enveloppe
            # → épisode immédiat (l'anti-rebond filtre le scintillement de bord, pas la violence)
        },
        'signals': signals,
        'directions': directions,
        'healthy': {s: _deque(maxlen=W_env) for s in signals},   # échantillons sains
        'raw': {s: _deque(maxlen=W_env) for s in signals},       # bruts (pour recalibration)
        'pre_iqr': {s: None for s in signals},                   # IQR sain avant épisode
        'episode': None,          # épisode courant : dict ou None
        'out_streak': 0,          # pas consécutifs D>0 (anti-rebond début)
        'pending_Dmax': 0.0,      # pic accumulé PENDANT l'anti-rebond (jamais perdu)
        'pending_peaks': None,    # pics par signal pendant l'anti-rebond
        'in_streak': 0,           # pas consécutifs D=0 (anti-rebond fin)
        'episodes': [],           # épisodes clos : (t_end, D_max, t_ret, recalibrated)
        'step': 0,
    }


def _robust_iqr(values, med, cfg):
    q75, q25 = np.percentile(values, [75, 25])
    return max(q75 - q25, cfg['iqr_floor_rel'] * abs(med), cfg['iqr_floor_abs'])


def update_resilience_envelope(env: Dict, signals_t: Dict[str, float],
                               t: float, dt: float) -> Dict:
    """
    Un pas de la résilience par enveloppes.

    Args:
        env: état (init_resilience_envelope_state), muté in-place
        signals_t: {'mu_Rloc': ..., 'effort': ..., 'mean_abs_error': ...}
        t: temps courant ; dt: pas de temps

    Returns dict:
        D, D_mean, D_max, D_rms : profondeurs d'excursion (les 3 agrégations loggées)
        value : résilience [0,1] ou None (warmup)
        metric_used : 'episodes' | 'tenue_de_soi' | None
        in_episode : bool ; recalibrated : bool (ce pas)
    """
    cfg = env['cfg']
    env['step'] += 1
    for s in env['signals']:
        env['raw'][s].append(float(signals_t[s]))

    # --- Enveloppes et profondeurs par signal ---
    depths = {}
    envelopes = {}
    ready = all(len(env['healthy'][s]) >= cfg['warmup'] for s in env['signals'])
    if ready:
        for s in env['signals']:
            vals = np.asarray(env['healthy'][s], dtype=float)
            med = float(np.median(vals))
            iqr = _robust_iqr(vals, med, cfg)
            half = cfg['k_iqr'] * iqr
            lo, hi = med - half, med + half
            envelopes[s] = (lo, hi, half)
            x = float(signals_t[s]); d = 0.0
            if 'low' in env['directions'][s] and x < lo:
                d = max(d, (lo - x) / half)
            if 'high' in env['directions'][s] and x > hi:
                d = max(d, (x - hi) / half)
            depths[s] = d
        dv = np.array([depths[s] for s in env['signals']], dtype=float)
        D_mean = float(np.mean(dv)); D_max = float(np.max(dv))
        D_rms = float(np.sqrt(np.mean(dv ** 2)))
        D = {'mean': D_mean, 'max': D_max, 'rms': D_rms}[cfg['aggregation']]
    else:
        depths = {s: 0.0 for s in env['signals']}
        D_mean = D_max = D_rms = D = 0.0

    recalibrated_now = False
    ep = env['episode']

    if ep is None:
        # Hors épisode : les pas sains nourrissent les buffers
        if D <= 0.0:
            for s in env['signals']:
                env['healthy'][s].append(float(signals_t[s]))
            env['out_streak'] = 0
            env['pending_Dmax'] = 0.0
            env['pending_peaks'] = None
        elif ready:
            env['out_streak'] += 1
            env['pending_Dmax'] = max(env['pending_Dmax'], D)
            if env['pending_peaks'] is None:
                env['pending_peaks'] = dict(depths)
            else:
                for s in env['signals']:
                    env['pending_peaks'][s] = max(env['pending_peaks'][s], depths[s])
            if env['out_streak'] >= cfg['debounce'] or D >= cfg['spike_threshold']:
                # Début d'épisode : gel, snapshot des IQR sains.
                # D_max initialisé au PIC accumulé (le sommet d'un choc bref
                # tombé pendant l'anti-rebond n'est jamais perdu).
                for s in env['signals']:
                    vals = np.asarray(env['healthy'][s], dtype=float)
                    env['pre_iqr'][s] = _robust_iqr(vals, float(np.median(vals)), cfg)
                env['episode'] = {'t_start': t - env['out_streak'] * dt,
                                  'D_max': env['pending_Dmax'],
                                  'peak_by_signal': dict(env['pending_peaks']),
                                  'steps': env['out_streak']}
                env['out_streak'] = 0
                env['pending_Dmax'] = 0.0
                env['pending_peaks'] = None
    else:
        ep['steps'] += 1
        ep['D_max'] = max(ep['D_max'], D)
        for s in env['signals']:
            ep['peak_by_signal'][s] = max(ep['peak_by_signal'][s], depths[s])
        if D <= 0.0:
            env['in_streak'] += 1
            if env['in_streak'] >= cfg['debounce']:
                # Fin d'épisode : retour dans l'enveloppe
                t_ret = (t - cfg['debounce'] * dt) - ep['t_start']
                env['episodes'].append({'t_end': t, 'D_max': ep['D_max'],
                                        't_ret': max(t_ret, dt), 'recalibrated': False})
                env['episode'] = None
                env['in_streak'] = 0
        else:
            env['in_streak'] = 0
            # Recalibration (régime légitime) : temps ET santé ET stabilité
            # (condition d'Andréa : on ne normalise jamais une lutte).
            if ep['steps'] >= cfg['T_recal']:
                excursing = [s for s in env['signals'] if depths[s] > 0.0]
                others_ok = all(depths[s] <= 0.0 for s in env['signals'] if s not in excursing)
                settled = True
                for s in excursing:
                    recent = np.asarray(list(env['raw'][s])[-max(cfg['W_env'] // 2, cfg['warmup']):], dtype=float)
                    rec_iqr = _robust_iqr(recent, float(np.median(recent)), cfg)
                    if env['pre_iqr'][s] and rec_iqr > cfg['recal_stability_factor'] * env['pre_iqr'][s]:
                        settled = False
                        break
                if others_ok and settled:
                    for s in env['signals']:
                        env['healthy'][s] = _deque(env['raw'][s], maxlen=cfg['W_env'])
                    t_ret = min(ep['steps'] * dt, cfg['T_recal'] * dt)
                    env['episodes'].append({'t_end': t, 'D_max': ep['D_max'],
                                            't_ret': t_ret, 'recalibrated': True})
                    env['episode'] = None
                    recalibrated_now = True

    # Borne mémoire des épisodes (par temps)
    env['episodes'] = [e for e in env['episodes'] if (t - e['t_end']) <= cfg['W_mem'] * dt]

    # --- Valeur de résilience ---
    if not ready:
        value, metric_used = None, None
    elif env['episodes']:
        _g = (lambda x: np.log1p(x)) if cfg['depth_compression'] == 'log' else (lambda x: x)
        if cfg['depth_memory'] == 'worst':
            r_depth = float(1.0 / (1.0 + _g(max(e['D_max'] for e in env['episodes']))))
        else:
            r_depth = float(np.mean([1.0 / (1.0 + _g(e['D_max'])) for e in env['episodes']]))
        r_time = float(np.mean([1.0 / (1.0 + e['t_ret'] / cfg['tau_ref']) for e in env['episodes']]))
        value = cfg['w_depth'] * r_depth + cfg['w_time'] * r_time
        metric_used = 'episodes'
    else:
        # Aucun épisode en mémoire : résilience non testée → tenue de soi (étage 1)
        mu_hist = list(env['raw']['mu_Rloc'])
        value = float(np.mean(mu_hist[-min(100, len(mu_hist)):]))
        metric_used = 'tenue_de_soi'

    return {'D': D, 'D_mean': D_mean, 'D_max': D_max, 'D_rms': D_rms,
            'value': value, 'metric_used': metric_used,
            'in_episode': env['episode'] is not None,
            'recalibrated': recalibrated_now}
