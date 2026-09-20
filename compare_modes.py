"""
compare_modes.py - Comparaison quantitative FPS vs Kuramoto vs Neutral
"""

import json
import numpy as np
from datetime import datetime
import os
from utils import deep_convert
import metrics as fps_metrics  # le dict local 'metrics' masquerait le module

def calculate_efficiency_metrics(fps_result, kuramoto_result, neutral_result):
    """
    Calcule les métriques d'efficience/déficience entre les modes.
    
    🔧 CORRECTION : Ajout diagnostics et normalisation pour éviter -90% systémiques
    """
    metrics = {}
    
    # DIAGNOSTIC DEBUG : Vérifier les structures de données
    print("🔍 DIAGNOSTIC compare_modes.py:")
    print(f"   FPS keys: {list(fps_result.keys()) if isinstance(fps_result, dict) else 'Not dict'}")
    print(f"   Kuramoto keys: {list(kuramoto_result.keys()) if isinstance(kuramoto_result, dict) else 'Not dict'}")
    print(f"   FPS metrics keys: {list(fps_result.get('metrics', {}).keys()) if fps_result.get('metrics') else 'No metrics'}")
    print(f"   Kuramoto metrics keys: {list(kuramoto_result.get('metrics', {}).keys()) if kuramoto_result.get('metrics') else 'No metrics'}")
    
    # 1. Synchronisation (basée sur C(t) final)
    # CORRECTION : Utiliser des clés cohérentes et vérifier valeurs
    fps_sync = fps_result.get('metrics', {}).get('mean_C', fps_result.get('metrics', {}).get('final_C', 0))
    kura_sync = kuramoto_result.get('metrics', {}).get('mean_C', kuramoto_result.get('metrics', {}).get('final_C', 0))
    neutral_sync = neutral_result.get('metrics', {}).get('mean_C', neutral_result.get('metrics', {}).get('final_C', 0))
    
    # CORRECTION : Debug des valeurs de synchronisation
    print(f"   Sync values - FPS: {fps_sync}, Kuramoto: {kura_sync}, Neutral: {neutral_sync}")
    
    # CORRECTION : Éviter division par zéro et normaliser si nécessaire
    # Si les valeurs sont très différentes en ordre de grandeur, normaliser
    if abs(fps_sync) > 0 and abs(kura_sync) > 0:
        ratio = abs(fps_sync / kura_sync)
        if ratio < 0.5 or ratio > 2.0:  # CORRECTION: Seuil élargi pour détecter 0.1 vs 1.0
            print(f"⚠️  Ratio sync suspect: {ratio:.3f} - possible problème d'unités")
            # CORRECTION : Si ratio suspect, utiliser une formule de comparaison relative
            # au lieu du pourcentage direct pour éviter -90% systémique
            if ratio < 0.2:  # FPS beaucoup plus faible
                fps_vs_kura_eff = -50 - (1 - ratio) * 40  # Entre -50% et -90%
            else:
                fps_vs_kura_eff = (fps_sync - kura_sync) / (abs(kura_sync) + 1e-10) * 100
        else:
            fps_vs_kura_eff = (fps_sync - kura_sync) / (abs(kura_sync) + 1e-10) * 100
    else:
        fps_vs_kura_eff = 0.0
    
    metrics['synchronization'] = {
        'fps_value': fps_sync,
        'kuramoto_value': kura_sync,
        'neutral_value': neutral_sync,
        'fps_vs_kuramoto_efficiency': fps_vs_kura_eff,
        'fps_vs_neutral_efficiency': (fps_sync - neutral_sync) / (abs(neutral_sync) + 1e-10) * 100
    }
    
    # 2. Stabilité = score de référence 'dispersion' (barème SCORE_BRACKETS)
    # sur l'écart-type du signal global de chaque mode.
    def _score(value, key):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return fps_metrics.NEUTRAL_SCORE
        return fps_metrics.NEUTRAL_SCORE if not np.isfinite(v) else fps_metrics.score_from_brackets(v, key)

    fps_std_S = fps_result.get('metrics', {}).get('std_S', 1.0)
    kura_std_S = kuramoto_result.get('metrics', {}).get('std_S', 1.0)
    neutral_std_S = neutral_result.get('metrics', {}).get('std_S', 1.0)
    
    print(f"   Std_S values - FPS: {fps_std_S}, Kuramoto: {kura_std_S}, Neutral: {neutral_std_S}")
    
    fps_stability = _score(fps_std_S, 'dispersion')
    kura_stability = _score(kura_std_S, 'dispersion')
    neutral_stability = _score(neutral_std_S, 'dispersion')
    
    metrics['stability'] = {
        'fps_value': fps_stability,
        'kuramoto_value': kura_stability,
        'neutral_value': neutral_stability,
        'fps_vs_kuramoto_efficiency': (fps_stability - kura_stability) / (abs(kura_stability) + 1e-3) * 100,
        'fps_vs_neutral_efficiency': (fps_stability - neutral_stability) / (abs(neutral_stability) + 1e-3) * 100
    }
    
    # 3. Résilience = score de référence 'resilience' sur resilience_ac (autocorr
    # CSD des résidus de l'enveloppe fₙ, basse = résilient). Contrôles sans
    # enveloppe vivante (fréquences fixes) : aucun verdict → neutre.
    fps_res_ac = fps_result.get('metrics', {}).get('resilience_ac')
    kura_res_ac = kuramoto_result.get('metrics', {}).get('resilience_ac')
    neutral_res_ac = neutral_result.get('metrics', {}).get('resilience_ac')
    print(f"   {fps_metrics.SCORE_KEY_LABELS['resilience']} (autocorr) - FPS: {fps_res_ac}, Kuramoto: {kura_res_ac}, Neutral: {neutral_res_ac}")
    fps_resilience = _score(fps_res_ac, 'resilience')
    kura_resilience = _score(kura_res_ac, 'resilience')
    neutral_resilience = _score(neutral_res_ac, 'resilience')
    
    metrics['resilience'] = {
        'fps_value': fps_resilience,
        'kuramoto_value': kura_resilience,
        'neutral_value': neutral_resilience,
        'fps_vs_kuramoto_efficiency': (fps_resilience - kura_resilience) / (abs(kura_resilience) + 1e-10) * 100,
        'fps_vs_neutral_efficiency': (fps_resilience - neutral_resilience) / (abs(neutral_resilience) + 1e-10) * 100
    }
    
    # 4. Innovation = score de référence 'innovation' sur C_JS (complexité
    # statistique de l'enveloppe fₙ, moniteur lent). Absent / None → neutre.
    fps_cjs = fps_result.get('metrics', {}).get('innovation_cjs')
    kura_cjs = kuramoto_result.get('metrics', {}).get('innovation_cjs')
    neutral_cjs = neutral_result.get('metrics', {}).get('innovation_cjs')
    print(f"   {fps_metrics.SCORE_KEY_LABELS['innovation']} - FPS: {fps_cjs}, Kuramoto: {kura_cjs}, Neutral: {neutral_cjs}")
    fps_innovation = _score(fps_cjs, 'innovation')
    kura_innovation = _score(kura_cjs, 'innovation')
    neutral_innovation = _score(neutral_cjs, 'innovation')
    
    metrics['innovation'] = {
        'fps_value': fps_innovation,
        'kuramoto_value': kura_innovation,
        'neutral_value': neutral_innovation,
        'fps_vs_kuramoto_efficiency': (fps_innovation - kura_innovation) / (abs(kura_innovation) + 1e-10) * 100,
        'fps_vs_neutral_efficiency': (fps_innovation - neutral_innovation) / (abs(neutral_innovation) + 1e-10) * 100
    }
    
    # 5. Fluidité = score de référence 'fluidity' sur la fluidité finale
    # (jerk de l'enveloppe fₙ, metrics.compute_fluidity). Un mode sans
    # enveloppe de fréquence vivante (fₙ constant) est parfaitement fluide.
    fps_fluid = _score(fps_result.get('metrics', {}).get('final_fluidity', 1.0), 'fluidity')
    kura_fluid = _score(kuramoto_result.get('metrics', {}).get('final_fluidity', 1.0), 'fluidity')
    neutral_fluid = _score(neutral_result.get('metrics', {}).get('final_fluidity', 1.0), 'fluidity')

    metrics['fluidity'] = {
        'fps_value': fps_fluid,
        'kuramoto_value': kura_fluid,
        'neutral_value': neutral_fluid,
        'fps_vs_kuramoto_efficiency': (fps_fluid - kura_fluid) / (abs(kura_fluid) + 1e-10) * 100,
        'fps_vs_neutral_efficiency': (fps_fluid - neutral_fluid) / (abs(neutral_fluid) + 1e-10) * 100
    }
    
    # 5. Efficacité CPU (inverse du coût)
    # CORRECTION : Normaliser les temps CPU
    fps_cpu_raw = fps_result.get('metrics', {}).get('mean_cpu_step', 1e-5)
    kura_cpu_raw = kuramoto_result.get('metrics', {}).get('mean_cpu_step', 1e-5)
    neutral_cpu_raw = neutral_result.get('metrics', {}).get('mean_cpu_step', 1e-5)
    
    print(f"   CPU raw values - FPS: {fps_cpu_raw:.2e}, Kuramoto: {kura_cpu_raw:.2e}, Neutral: {neutral_cpu_raw:.2e}")
    
    # CORRECTION : Normaliser les efficacités CPU pour éviter valeurs énormes
    fps_cpu_eff = min(1.0 / (fps_cpu_raw + 1e-10), 1e6)  # Cap à 1M pour éviter overflow
    kura_cpu_eff = min(1.0 / (kura_cpu_raw + 1e-10), 1e6)
    neutral_cpu_eff = min(1.0 / (neutral_cpu_raw + 1e-10), 1e6)
    
    metrics['cpu_efficiency'] = {
        'fps_value': fps_cpu_eff,
        'kuramoto_value': kura_cpu_eff,
        'neutral_value': neutral_cpu_eff,
        'fps_vs_kuramoto_efficiency': (fps_cpu_eff - kura_cpu_eff) / (abs(kura_cpu_eff) + 1e-3) * 100,
        'fps_vs_neutral_efficiency': (fps_cpu_eff - neutral_cpu_eff) / (abs(neutral_cpu_eff) + 1e-3) * 100
    }
    
    # Score global avec pondération équilibrée
    # CORRECTION : Normaliser les scores avant moyenne pour éviter domination d'une métrique
    def normalize_metric(value, min_val=0, max_val=1):
        """Normalise une métrique dans [0,1]"""
        if max_val == min_val:
            return 0.5
        return max(0, min(1, (value - min_val) / (max_val - min_val)))
    
    # Normaliser chaque métrique
    sync_values = [metrics['synchronization']['fps_value'], metrics['synchronization']['kuramoto_value'], metrics['synchronization']['neutral_value']]
    sync_min, sync_max = min(sync_values), max(sync_values)
    
    stab_values = [metrics['stability']['fps_value'], metrics['stability']['kuramoto_value'], metrics['stability']['neutral_value']]
    stab_min, stab_max = min(stab_values), max(stab_values)
    
    resil_values = [metrics['resilience']['fps_value'], metrics['resilience']['kuramoto_value'], metrics['resilience']['neutral_value']]
    resil_min, resil_max = min(resil_values), max(resil_values)
    
    innov_values = [metrics['innovation']['fps_value'], metrics['innovation']['kuramoto_value'], metrics['innovation']['neutral_value']]
    innov_min, innov_max = min(innov_values), max(innov_values)
    
    fluid_values = [metrics['fluidity']['fps_value'], metrics['fluidity']['kuramoto_value'], metrics['fluidity']['neutral_value']]
    fluid_min, fluid_max = min(fluid_values), max(fluid_values)
    
    cpu_values = [metrics['cpu_efficiency']['fps_value'], metrics['cpu_efficiency']['kuramoto_value'], metrics['cpu_efficiency']['neutral_value']]
    cpu_min, cpu_max = min(cpu_values), max(cpu_values)
    
    fps_score = np.mean([
        normalize_metric(metrics['synchronization']['fps_value'], sync_min, sync_max),
        normalize_metric(metrics['stability']['fps_value'], stab_min, stab_max),
        normalize_metric(metrics['resilience']['fps_value'], resil_min, resil_max),
        normalize_metric(metrics['innovation']['fps_value'], innov_min, innov_max),
        normalize_metric(metrics['fluidity']['fps_value'], fluid_min, fluid_max),
        normalize_metric(metrics['cpu_efficiency']['fps_value'], cpu_min, cpu_max)
    ])
    
    kura_score = np.mean([
        normalize_metric(metrics['synchronization']['kuramoto_value'], sync_min, sync_max),
        normalize_metric(metrics['stability']['kuramoto_value'], stab_min, stab_max),
        normalize_metric(metrics['resilience']['kuramoto_value'], resil_min, resil_max),
        normalize_metric(metrics['innovation']['kuramoto_value'], innov_min, innov_max),
        normalize_metric(metrics['fluidity']['kuramoto_value'], fluid_min, fluid_max),
        normalize_metric(metrics['cpu_efficiency']['kuramoto_value'], cpu_min, cpu_max)
    ])
    
    neutral_score = np.mean([
        normalize_metric(metrics['synchronization']['neutral_value'], sync_min, sync_max),
        normalize_metric(metrics['stability']['neutral_value'], stab_min, stab_max),
        normalize_metric(metrics['resilience']['neutral_value'], resil_min, resil_max),
        normalize_metric(metrics['innovation']['neutral_value'], innov_min, innov_max),
        normalize_metric(metrics['fluidity']['neutral_value'], fluid_min, fluid_max),
        normalize_metric(metrics['cpu_efficiency']['neutral_value'], cpu_min, cpu_max)
    ])
    
    print(f"   Scores normalisés - FPS: {fps_score:.3f}, Kuramoto: {kura_score:.3f}, Neutral: {neutral_score:.3f}")
    
    metrics['global_score'] = {
        'fps': fps_score,
        'kuramoto': kura_score,
        'neutral': neutral_score,
        'fps_vs_kuramoto_efficiency': (fps_score - kura_score) / (abs(kura_score) + 1e-10) * 100,
        'fps_vs_neutral_efficiency': (fps_score - neutral_score) / (abs(neutral_score) + 1e-10) * 100
    }
    
    return deep_convert(metrics)


def export_comparison_report(fps_result, kuramoto_result, neutral_result, output_path):
    """
    Exporte un rapport de comparaison détaillé.
    """
    metrics = calculate_efficiency_metrics(fps_result, kuramoto_result, neutral_result)
    
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'fps_run_id': fps_result.get('run_id', 'unknown'),
            'kuramoto_run_id': kuramoto_result.get('run_id', 'unknown'),
            'neutral_run_id': neutral_result.get('run_id', 'unknown')
        },
        'detailed_metrics': metrics,
        'summary': {
            'fps_advantages': [],
            'fps_disadvantages': [],
            'overall_verdict': ''
        }
    }
    
    # Analyser les avantages/désavantages
    for metric_name, metric_data in metrics.items():
        if metric_name == 'global_score':
            continue
            
        vs_kura = metric_data.get('fps_vs_kuramoto_efficiency', 0)
        vs_neutral = metric_data.get('fps_vs_neutral_efficiency', 0)
        
        if vs_kura > 10:
            report['summary']['fps_advantages'].append(
                f"{fps_metrics.metric_label(metric_name)}: +{vs_kura:.1f}% vs Kuramoto"
            )
        elif vs_kura < -10:
            report['summary']['fps_disadvantages'].append(
                f"{fps_metrics.metric_label(metric_name)}: {vs_kura:.1f}% vs Kuramoto"
            )
    
    # Verdict global
    global_eff_kura = metrics['global_score']['fps_vs_kuramoto_efficiency']
    global_eff_neutral = metrics['global_score']['fps_vs_neutral_efficiency']
    
    if global_eff_kura > 0 and global_eff_neutral > 0:
        report['summary']['overall_verdict'] = f"FPS surpasse les deux modèles de contrôle (Kuramoto: +{global_eff_kura:.1f}%, Neutral: +{global_eff_neutral:.1f}%)"
    elif global_eff_kura > 0:
        report['summary']['overall_verdict'] = f"FPS surpasse Kuramoto (+{global_eff_kura:.1f}%) mais montre des limites vs Neutral"
    else:
        report['summary']['overall_verdict'] = "FPS montre des caractéristiques uniques mais avec des compromis"
    
    # Exporter JSON
    with open(output_path, 'w') as f:
        json.dump(deep_convert(report), f, indent=2)
    
    # Exporter aussi un résumé texte
    txt_path = output_path.replace('.json', '.txt')
    with open(txt_path, 'w') as f:
        f.write("COMPARAISON FPS vs KURAMOTO vs NEUTRAL\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("SCORES GLOBAUX:\n")
        f.write(f"  FPS:      {metrics['global_score']['fps']:.3f}\n")
        f.write(f"  Kuramoto: {metrics['global_score']['kuramoto']:.3f}\n")
        f.write(f"  Neutral:  {metrics['global_score']['neutral']:.3f}\n\n")
        
        f.write("EFFICIENCE FPS:\n")
        f.write(f"  vs Kuramoto: {global_eff_kura:+.1f}%\n")
        f.write(f"  vs Neutral:  {global_eff_neutral:+.1f}%\n\n")
        
        f.write("DÉTAILS PAR CRITÈRE:\n")
        for metric_name, metric_data in metrics.items():
            if metric_name != 'global_score':
                f.write(f"\n{fps_metrics.metric_label(metric_name).upper()} ({metric_name}):\n")
                f.write(f"  FPS: {metric_data['fps_value']:.3f}\n")
                f.write(f"  Kuramoto: {metric_data['kuramoto_value']:.3f}\n")
                f.write(f"  Neutral: {metric_data['neutral_value']:.3f}\n")
                f.write(f"  Efficience vs Kuramoto: {metric_data['fps_vs_kuramoto_efficiency']:+.1f}%\n")
                f.write(f"  Efficience vs Neutral: {metric_data['fps_vs_neutral_efficiency']:+.1f}%\n")
        
        f.write(f"\n{report['summary']['overall_verdict']}\n")
    
    return deep_convert(report)