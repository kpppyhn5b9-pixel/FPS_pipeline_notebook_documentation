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


# ============== MÉTRIQUES DE RÉSILIENCE : RALENTISSEMENT CRITIQUE ==============
#
# Décision 20/09/2026 (New_Attention.md) : UNE seule quantité remplace t_retour,
# la résilience continue, l'aiguillage adaptatif et les enveloppes de santé :
# le ralentissement critique (Scheffer 2009) mesuré comme l'autocorrélation, à
# un lag calibré sur la relaxation, des RÉSIDUS DÉTRENDÉS de la couche lente
# (l'enveloppe fₙ). Près d'un point de bascule, un système récupère plus
# lentement de ses propres fluctuations : autocorrélation ET variance montent.
#   - sans config : lit la turbulence spontanée, aucun choc à induire ;
#   - valable sous perturbation continue ;
#   - orthogonal à l'innovation (vitesse de retour, pas richesse).
# Validé au banc (cahier) : AR(1) à pôle connu retrouvé ; AR(1) + forçage
# périodique → le lag-1 brut est aveugle (~0.96 partout), le détrend le répare.
# Lecture : autocorr BASSE = retour rapide = résilient (score 5) ; qui MONTE =
# alarme latente (barème monotone unilatéral).

def relaxation_steps(x, default: int = 10) -> int:
    """
    Temps de relaxation en pas : premier passage de l'autocorrélation sous 1/e.
    Source unique de l'échelle de temps (innovation τ, lag de résilience).
    """
    x = np.asarray(x, dtype=float).ravel()
    xc = x - x.mean()
    if len(xc) < 3 or float(np.dot(xc, xc)) < 1e-12:
        return default
    ac = np.correlate(xc, xc, mode='full')[len(xc) - 1:]
    ac = ac / ac[0]
    below = np.where(ac < 1.0 / np.e)[0]
    return int(below[0]) if len(below) else default


def detrend_spectral_peaks(x, peak_ratio: float = 10.0, max_peaks: int = 5):
    """
    Retire adaptativement les pics spectraux DOMINANTS (la composante
    déterministe / périodique : le forçage) et renvoie (résidus, nb_pics).

    Un pic est une RAIE : sa puissance dépasse peak_ratio × la médiane de son
    voisinage spectral (±neigh bins, les 3 bins centraux exclus). Un spectre
    rouge lisse (AR(1)) n'a pas de raie et n'est donc pas touché ; un forçage
    périodique en a une par harmonique. Au plus max_peaks raies ; chaque raie
    est retirée avec ses deux voisins de bin (fuite spectrale). Sur l'enveloppe
    fₙ de la FPS, une seule raie domine (cahier) ; la règle reste adaptative.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    xc = x - x.mean()
    if n < 8 or float(np.dot(xc, xc)) < 1e-12:
        return xc, 0
    X = np.fft.rfft(xc)
    P = np.abs(X) ** 2
    P[0] = 0.0
    neigh = max(5, len(P) // 20)
    n_peaks = 0
    for _ in range(max_peaks):
        k = int(np.argmax(P))
        if k == 0 or P[k] <= 0:
            break
        lo, hi = max(1, k - neigh), min(len(P), k + neigh + 1)
        around = np.concatenate([P[lo:max(lo, k - 1)], P[min(hi, k + 2):hi]])
        around = around[around > 0]
        if len(around) < 3 or P[k] <= peak_ratio * float(np.median(around)):
            break
        for j in (k - 1, k, k + 1):
            if 1 <= j < len(P):
                X[j] = 0.0
                P[j] = 0.0
        n_peaks += 1
    return np.fft.irfft(X, n=n), n_peaks


def lag_autocorrelation(x, lag: int) -> Optional[float]:
    """Autocorrélation de x au décalage lag (None si dégénérée)."""
    x = np.asarray(x, dtype=float).ravel()
    lag = int(lag)
    if lag < 1 or len(x) <= lag + 2:
        return None
    a, b = x[:-lag], x[lag:]
    a = a - a.mean(); b = b - b.mean()
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if den < 1e-15:
        return None
    return float(np.dot(a, b) / den)


def compute_resilience_csd(fn_mean_window, dt: float, lag: Optional[int] = None,
                           min_points: int = 200, peak_ratio: float = 10.0,
                           max_peaks: int = 5) -> Optional[Dict[str, float]]:
    """
    Résilience (score 'resilience') = ralentissement critique sur la couche
    lente : autocorrélation à lag des résidus détrendés de l'enveloppe fₙ.

    Étapes : (1) détrend adaptatif des pics spectraux dominants ; (2) lag =
    celui fourni (figé après calibration, cf. simulate) ou la relaxation 1/e de
    l'enveloppe brute ; (3) autocorrélation des résidus à ce lag + leur variance.

    Pourquoi figer le lag une fois calibré : si on le re-dérivait à chaque
    fenêtre depuis le même signal, l'autocorrélation à « son propre 1/e »
    vaudrait ~1/e par construction et ne pourrait jamais monter. Le radar CSD
    n'existe qu'à lag FIXE (c'est ainsi que le cahier l'a validé).

    Args:
        fn_mean_window: moyenne de fₙ par pas, fenêtre LONGUE (≥ min_points)
        dt: pas de temps (signature homogène ; l'échelle vient du signal)
        lag: décalage en pas, ou None → auto (relaxation)

    Returns:
        {'ac': autocorr [-1, 1], 'var': variance des résidus, 'lag': int,
         'relax': int, 'n_peaks': int} ou None (fenêtre trop courte, signal plat).
    """
    x = np.asarray(fn_mean_window, dtype=float).ravel()
    if len(x) < min_points:
        return None
    relax = relaxation_steps(x)
    resid, n_peaks = detrend_spectral_peaks(x, peak_ratio, max_peaks)
    if float(np.var(resid)) < 1e-15:
        return None  # rien ne fluctue : pas de retour à mesurer, verdict suspendu
    L = int(lag) if lag is not None else relax
    L = int(min(max(1, L), max(1, len(x) // 4)))
    ac = lag_autocorrelation(resid, L)
    if ac is None:
        return None
    return {'ac': float(ac), 'var': float(np.var(resid)), 'lag': L,
            'relax': int(relax), 'n_peaks': int(n_peaks)}


def resilience_alert(ac_series, var_series, calm_n: int = 100, band: float = 0.06,
                     n_windows: int = 3, stride: int = 1) -> int:
    """
    Radar CSD, règle de décision conservatrice (cahier, anti-faux-positif) :
      1. bande de référence : ac doit dépasser nettement la variabilité du calme
         (médiane des calm_n premières valeurs + max(band, 2·σ_calme)) ;
      2. tendance, pas un point : n_windows échantillons consécutifs croissants
         (échantillonnés tous les `stride` pas, pour lire des fenêtres distinctes) ;
      3. convergence : la variance monte aussi, sur les mêmes échantillons.
    Renvoie 1 (alerte) ou 0. Jamais d'alerte sans référence calme complète.
    """
    ac = np.asarray([v for v in ac_series if v is not None and np.isfinite(v)], dtype=float)
    var = np.asarray([v for v in var_series if v is not None and np.isfinite(v)], dtype=float)
    stride = max(1, int(stride))
    need = calm_n + n_windows * stride
    if len(ac) < need or len(var) < need:
        return 0
    calm_ac, calm_var = ac[:calm_n], var[:calm_n]
    thr_ac = float(np.median(calm_ac)) + max(band, 2.0 * float(np.std(calm_ac)))
    thr_var = float(np.median(calm_var))
    s_ac = ac[-1 - (n_windows - 1) * stride::stride] if n_windows > 1 else ac[-1:]
    s_var = var[-1 - (n_windows - 1) * stride::stride] if n_windows > 1 else var[-1:]
    rising_ac = all(np.diff(s_ac) > 0) if len(s_ac) > 1 else True
    rising_var = all(np.diff(s_var) > 0) if len(s_var) > 1 else True
    return int(rising_ac and rising_var and s_ac[-1] > thr_ac and s_var[-1] > thr_var)



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
    # resilience : autocorrélation (lag calibré) des résidus détrendés de fₙ.
    # BASSE = retour rapide = résilient. PROVISOIRE (20/09) : calme FPS ≈ 0.33,
    # bande du témoin ±0.06 ; la montée CSD observée sur jouet atteint 0.5–0.6.
    'resilience': {'direction': 'lower',  'thresholds': [0.45, 0.60, 0.75, 0.90]},
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
# Filtres « sens, pas mains » (Attention.md) : notés, visibles du switch et de
# gamma, mais JAMAIS engagés comme remède (aucun poids de perception n'en
# découle). L'innovation est une métrique d'IDENTITÉ : on l'observe, on
# n'intervient pas dessus. Le geste « relâcher » n'existe pas encore.
OBSERVE_ONLY_FILTERS = ('innovation',)
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


def compute_innovation_plane(fn_mean_window, dt: float, d: int = 4,
                             min_points: int = 200) -> Optional[Dict[str, float]]:
    """
    Plan entropie–complexité (Rosso / MPR) de l'enveloppe fₙ :
      H  = entropie de permutation normalisée dans [0, 1]
      C  = complexité statistique de Jensen-Shannon C_JS = Q_J · H

    C est LA métrique d'innovation (score 'innovation', cf. compute_innovation_cjs).
    H dit de quel côté de la cloche on est (Attention.md, « diagnostic
    d'action ») : H bas = trop ordonné, H haut = trop bruité, H moyen avec C
    haut = nouveauté structurée. Les deux sont calculés d'un seul geste ; on
    logge les deux ('innovation_cjs', 'innovation_H').

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
        {'H': float dans [0, 1], 'C': float dans [0, ~0.5]} ou None si trop court.
    """
    from itertools import permutations
    x = np.asarray(fn_mean_window, dtype=float).ravel()
    if len(x) < min_points:
        return None
    xc = x - x.mean()
    if float(np.dot(xc, xc)) < 1e-12:
        return {'H': 0.0, 'C': 0.0}  # signal plat : ordre pur → innovation nulle
    relax = relaxation_steps(x)  # source unique de l'échelle de temps
    tau = max(1, int(round(relax / (d - 1))))
    # garde : la fenêtre ordinale doit tenir largement dans le signal
    tau = min(tau, max(1, (len(x) // 2) // (d - 1)))
    idx = {p: i for i, p in enumerate(permutations(range(d)))}
    counts = np.zeros(len(idx))
    n = len(x) - (d - 1) * tau
    if n <= 0:
        return None
    # NB ex æquo : argsort est stable (l'index le plus ancien passe devant). Sur
    # l'enveloppe fₙ de la FPS, aucun pas consécutif n'est strictement égal
    # (vérifié 20/09 sur run réel) : pas de traitement spécial nécessaire.
    for i in range(n):
        counts[idx[tuple(np.argsort(x[i:i + d * tau:tau]))]] += 1
    s = counts.sum()
    if s <= 0:
        return {'H': 0.0, 'C': 0.0}
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
    C = float((js / js_max) * H) if js_max > 0 else 0.0
    return {'H': float(H), 'C': C}


def compute_innovation_cjs(fn_mean_window, dt: float, d: int = 4,
                           min_points: int = 200) -> Optional[float]:
    """
    Innovation (score 'innovation') = complexité statistique C_JS de l'enveloppe
    fₙ — la coordonnée C de compute_innovation_plane (une seule implémentation).

    Cloche native : ~0 pour le bruit pur, maximale pour la nouveauté structurée.
    NB (banc 20/09, τ auto) : un sinus pur score aussi haut que le chaos (~0.30) ;
    c'est H, pas C, qui distingue « trop ordonné » de « nouveauté ».
    Renvoie None si la fenêtre est trop courte (→ score neutre).
    """
    plane = compute_innovation_plane(fn_mean_window, dt, d, min_points)
    return None if plane is None else plane['C']


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
      resilience  = dernière valeur de resilience_ac    (compute_resilience_csd,
                    moniteur lent lu dans les logs) ; None si aucun verdict

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
    # résilience = MONITEUR LENT lu dans les logs (autocorr CSD de l'enveloppe fₙ,
    # calculée sur fenêtre longue dans simulate → 'resilience_ac'), dernière valeur
    # disponible ; None (warmup / CSV sans la colonne) → score neutre.
    resil = [v for v in (_num(h.get('resilience_ac')) for h in history_window) if v is not None]
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
        'resilience': (resil[-1] if resil else None),
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


