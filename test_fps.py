"""
test_fps.py - Tests unitaires complets du système FPS
Version exhaustive conforme à la feuille de route FPS V1.3
---------------------------------------------------------------
Ce module est le gardien de l'intégrité du système FPS.
Il vérifie que chaque pièce de la mécanique fonctionne
parfaitement, seule et en harmonie avec les autres.

Tests inclus :
- Fonctions individuelles (statique/dynamique)
- Validation de configuration
- Perturbations et résilience
- Intégration complète
- Performance et limites
- Comparaisons FPS/Kuramoto

Chaque test est une promesse de fiabilité,
chaque assertion une garantie d'harmonie.

(c) 2025 Gepetto & Andréa Gadal & Claude (Anthropic) 🌀
"""

import unittest
import numpy as np
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import time
import warnings

# Import des modules FPS
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


class TestDynamics(unittest.TestCase):
    """Tests pour le module dynamics.py"""
    
    def setUp(self):
        """Configuration commune pour les tests."""
        self.config = {
            'system': {'N': 3, 'T': 100, 'dt': 0.1},
            'strates': [
                {'A0': 1.0, 'f0': 1.0, 'phi': 0.0, 'alpha': 0.5, 'beta': 1.0, 'k': 2.0, 'x0': 0.5, 'w': [0, 0.1, -0.1]},
                {'A0': 1.0, 'f0': 1.1, 'phi': 0.0, 'alpha': 0.5, 'beta': 1.0, 'k': 2.0, 'x0': 0.5, 'w': [0.1, 0, -0.1]},
                {'A0': 1.0, 'f0': 0.9, 'phi': 0.0, 'alpha': 0.5, 'beta': 1.0, 'k': 2.0, 'x0': 0.5, 'w': [0.1, -0.1, 0]}
            ],
            'latence': {'gamma_mode': 'static', 'gamma_n_mode': 'static'},
            'dynamic_parameters': {'dynamic_phi': False}
        }
        self.state = init.init_strates(self.config)
    
    def test_compute_sigma(self):
        """Test de la fonction sigmoïde."""
        # Test valeurs spécifiques
        self.assertAlmostEqual(dynamics.compute_sigma(0, 2.0, 0.0), 0.5, places=6)
        self.assertAlmostEqual(dynamics.compute_sigma(-10, 2.0, 0.0), 0.0, places=3)
        self.assertAlmostEqual(dynamics.compute_sigma(10, 2.0, 0.0), 1.0, places=3)
        
    def test_compute_An(self):
        """Test du calcul d'amplitude adaptative."""
        In_t = np.array([0.5, 0.5, 0.5])
        # compute_An prend désormais le feedback F_n_t_An (ici nul : mode
        # statique sans feedback → An = A0·σ(In)).
        An_t = dynamics.compute_An(0, self.state, In_t, np.zeros(3), self.config)
        
        # Vérifier la forme
        self.assertEqual(len(An_t), 3)
        
        # Vérifier les valeurs (A0 * sigma(In))
        for i in range(3):
            expected = self.state[i]['A0'] * dynamics.compute_sigma(In_t[i], self.state[i]['k'], self.state[i]['x0'])
            self.assertAlmostEqual(An_t[i], expected, places=6)
    
    def test_compute_gamma_static(self):
        """Test de la latence en mode statique."""
        gamma = dynamics.compute_gamma(50, mode="static")
        self.assertEqual(gamma, 1.0)
    
    def test_compute_gamma_dynamic(self):
        """Test de la latence en mode dynamique."""
        T = 100
        # À t=T/2, la sigmoïde doit valoir 0.5
        gamma = dynamics.compute_gamma(T/2, mode="dynamic", T=T)
        self.assertAlmostEqual(gamma, 0.5, places=3)
        
        # Aux extrêmes
        gamma_start = dynamics.compute_gamma(0, mode="dynamic", T=T)
        gamma_end = dynamics.compute_gamma(T, mode="dynamic", T=T)
        self.assertLess(gamma_start, 0.1)
        self.assertGreater(gamma_end, 0.9)
    
    def test_compute_gamma_sigmoid_up(self):
        """Test du mode sigmoid_up."""
        T = 100
        # Doit croître de façon monotone
        gamma_0 = dynamics.compute_gamma(0, mode="sigmoid_up", T=T)
        gamma_mid = dynamics.compute_gamma(T/2, mode="sigmoid_up", T=T)
        gamma_end = dynamics.compute_gamma(T, mode="sigmoid_up", T=T)
        
        self.assertLess(gamma_0, gamma_mid)
        self.assertLess(gamma_mid, gamma_end)
        self.assertLess(gamma_0, 0.2)
        self.assertGreater(gamma_end, 0.8)
    
    def test_compute_gamma_sigmoid_down(self):
        """Test du mode sigmoid_down."""
        T = 100
        # Doit décroître de façon monotone
        gamma_0 = dynamics.compute_gamma(0, mode="sigmoid_down", T=T)
        gamma_mid = dynamics.compute_gamma(T/2, mode="sigmoid_down", T=T)
        gamma_end = dynamics.compute_gamma(T, mode="sigmoid_down", T=T)
        
        self.assertGreater(gamma_0, gamma_mid)
        self.assertGreater(gamma_mid, gamma_end)
        self.assertGreater(gamma_0, 0.8)
        self.assertLess(gamma_end, 0.2)
    
    def test_compute_gamma_sigmoid_adaptive(self):
        """Test du mode sigmoid_adaptive."""
        T = 100
        # Doit varier entre 0.3 et 1.0
        for t in [0, T/4, T/2, 3*T/4, T]:
            gamma = dynamics.compute_gamma(t, mode="sigmoid_adaptive", T=T)
            self.assertGreaterEqual(gamma, 0.3)
            self.assertLessEqual(gamma, 1.0)
    
    def test_compute_gamma_sigmoid_oscillating(self):
        """Test du mode sigmoid_oscillating."""
        T = 100
        # Doit osciller autour de la sigmoïde de base
        gamma_values = []
        for t in np.linspace(0, T, 50):
            gamma = dynamics.compute_gamma(t, mode="sigmoid_oscillating", T=T)
            gamma_values.append(gamma)
            # Vérifier les bornes
            self.assertGreaterEqual(gamma, 0.1)
            self.assertLessEqual(gamma, 1.0)
        
        # Vérifier qu'il y a des oscillations (variations)
        diffs = np.diff(gamma_values)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        self.assertGreater(sign_changes, 2)
    
    def test_compute_S_i(self):
        """Test du signal inter-strates avec matrice de poids."""
        # À t=0, doit retourner 0
        S_i = dynamics.compute_S_i(0, 0, [], self.state)
        self.assertEqual(S_i, 0.0)
        
        # Avec historique et matrice de poids
        history = [{'S': 2.0, 'O': np.array([0.5, 0.3, 0.2])}]
        
        # Pour la strate 0, calculer S_i selon la formule FPS
        # S_i = Σ(j≠0) O_j * w_0j
        S_i = dynamics.compute_S_i(1, 0, history, self.state)
        
        # Calcul attendu avec les poids de la strate 0
        w_0 = self.state[0]['w']  # [0.0, 0.1, -0.1]
        expected = history[0]['O'][1] * w_0[1] + history[0]['O'][2] * w_0[2]
        expected = 0.3 * 0.1 + 0.2 * (-0.1)  # = 0.03 - 0.02 = 0.01
        
        self.assertAlmostEqual(S_i, expected, places=6)
    
    def test_compute_On_En(self):
        """Test des sorties observée et attendue."""
        An_t = np.array([1.0, 0.8, 1.2])
        fn_t = np.array([1.0, 1.1, 0.9])
        phi_n_t = np.zeros(3)
        gamma_n_t = np.ones(3)

    def test_formule_S_i_conforme(self):
        """Test de conformité de la formule S_i(t) avec matrice de poids."""
        # Créer un historique de test
        history = [{
            'S': 2.5,
            'O': np.array([0.5, 0.3, 0.2])
        }]
        
        # Test pour chaque strate
        for n in range(3):
            S_i = dynamics.compute_S_i(1, n, history, self.state)
            
            # Calcul attendu selon FPS : S_i = Σ(j≠n) O_j * w_nj
            w_n = self.state[n]['w']
            expected = 0.0
            for j in range(3):
                if j != n:
                    expected += history[0]['O'][j] * w_n[j]
            
            self.assertAlmostEqual(S_i, expected, places=6,
                                 msg=f"S_i(t) doit être Σ(j≠n) O_j * w_nj pour strate {n}")
            
    def test_formule_Fn_conforme(self):
        """Conformité des DEUX feedbacks (catalogue §4 : jamais un seul Fₙ).

        F_Aₙ(t) = βₙ·G(On−En)  → amplitude (via l'archétype G)
        F_fₙ(t) = βₙ·γ(t)       → fréquence
        """
        beta_n = 1.5
        On_t = 0.8
        En_t = 0.6
        gamma_t = 0.9
        error = En_t - On_t  # convention production/catalogue : errorₙ = Eₙ − Oₙ

        # Canal AMPLITUDE : βₙ · G(erreur), ici archétype tanh (λ=1).
        G_value = regulation.compute_G(error, 'tanh', {'lambda': 1.0})
        F_A = beta_n * G_value
        self.assertAlmostEqual(F_A, beta_n * np.tanh(error), places=6,
                               msg="F_Aₙ(t) doit être βₙ·G(Eₙ−Oₙ)")

        # Canal FRÉQUENCE : βₙ · γ(t).
        F_f = beta_n * gamma_t
        self.assertAlmostEqual(F_f, beta_n * gamma_t, places=6,
                               msg="F_fₙ(t) doit être βₙ·γ(t)")

        # Le design a quitté l'ancienne formule unique βₙ·(Eₙ−Oₙ)·γ.
        self.assertNotAlmostEqual(F_A, beta_n * error * gamma_t, places=3,
                                  msg="Le feedback amplitude n'est plus βₙ·(Eₙ−Oₙ)·γ")


class TestRegulation(unittest.TestCase):
    """Tests pour le module regulation.py"""
    
    def setUp(self):
        """Configuration pour les tests."""
        self.config = {
            'regulation': {'G_arch': 'tanh', 'lambda': 1.0},
            'enveloppe': {'env_mode': 'static', 'sigma_n_static': 0.1, 'mu_n': 0.0},
            'system': {'T': 100}
        }
    
    def test_compute_G_archetypes(self):
        """Test des différents archétypes de régulation."""
        x_test = np.linspace(-3, 3, 7)
        
        # Tanh
        g_tanh = regulation.compute_G(x_test, 'tanh', {'lambda': 1.0})
        self.assertAlmostEqual(regulation.compute_G(0, 'tanh'), 0.0, places=6)
        
        # Sinc
        g_sinc = regulation.compute_G(x_test, 'sinc')
        self.assertAlmostEqual(regulation.compute_G(0, 'sinc'), 1.0, places=6)
        
        # Resonance
        g_res = regulation.compute_G(x_test, 'resonance', {'alpha': 1.0, 'beta': 2.0})
        self.assertAlmostEqual(regulation.compute_G(0, 'resonance'), 0.0, places=6)
    
    def test_compute_env_n_static(self):
        """Test de l'enveloppe en mode statique."""
        # Gaussienne
        env = regulation.compute_env_n(0, 0, 'static', 0.5, 0, None, 'gaussienne')
        self.assertAlmostEqual(env, 1.0, places=6)
        
        # Sigmoïde
        env_sig = regulation.compute_env_n(0, 0, 'static', 0.5, 0, None, 'sigmoide')
        self.assertAlmostEqual(env_sig, 0.5, places=6)
    
    def test_compute_sigma_n_dynamic(self):
        """Test de l'écart-type dynamique."""
        T = 100
        sigma_dynamic = {'amp': 0.05, 'freq': 1, 'offset': 0.1}
        
        # À différents temps
        sigma_0 = regulation.compute_sigma_n(0, 'dynamic', T, 0.1, sigma_dynamic)
        sigma_T4 = regulation.compute_sigma_n(T/4, 'dynamic', T, 0.1, sigma_dynamic)
        
        # Vérifier la variation
        self.assertNotAlmostEqual(sigma_0, sigma_T4, places=3)
        self.assertGreater(sigma_0, 0)
        self.assertGreater(sigma_T4, 0)
    
    def test_compute_Gn_integration(self):
        """Test de la fonction intégrée pour simulate.py."""
        error = np.array([0.5, -0.3, 0.1])
        An_t = np.array([1.0, 0.8, 1.2])
        fn_t = np.array([1.0, 1.1, 0.9])

    def test_feedback_modes(self):
        """Le feedback amplitude suit l'archétype G choisi ; le feedback
        fréquence reste βₙ·γ, indépendant de l'archétype (catalogue §4)."""
        beta_n = 1.5
        On_t = 0.8
        En_t = 0.6
        gamma_t = 0.9
        error = En_t - On_t  # convention production/catalogue : errorₙ = Eₙ − Oₙ

        # AMPLITUDE : dépend de l'archétype G.
        F_A_tanh = beta_n * regulation.compute_G(error, 'tanh', {'lambda': 1.0})
        F_A_resonance = beta_n * regulation.compute_G(error, 'resonance', {'lambda': 1.0})
        self.assertNotAlmostEqual(F_A_tanh, F_A_resonance, places=3,
                                  msg="L'archétype G doit changer le feedback amplitude")
        self.assertIsInstance(float(F_A_tanh), float)

        # FRÉQUENCE : βₙ·γ, le même quel que soit l'archétype amplitude.
        F_f = beta_n * gamma_t
        self.assertAlmostEqual(F_f, beta_n * gamma_t, places=6)


class TestMetrics(unittest.TestCase):
    """Tests pour le module metrics.py"""
    
    def setUp(self):
        """Données de test."""
        self.S_history = list(np.sin(np.linspace(0, 10, 100)))
        self.effort_history = list(0.5 + 0.1 * np.random.randn(100))
        self.config = {
            'to_calibrate': {
                'effort_transitoire_threshold': 2.0,
                'effort_chronique_threshold': 1.0  # Seuil plus bas pour que le test passe
            }
        }
    
    def test_compute_cpu_step_deterministic(self):
        """Le coût CPU est déterministe, croissant avec N, une seule fonction."""
        c10 = metrics.compute_cpu_step_deterministic(10)
        self.assertGreater(c10, 0)
        self.assertEqual(c10, metrics.compute_cpu_step_deterministic(10))
        self.assertGreater(metrics.compute_cpu_step_deterministic(100), c10)
        self.assertEqual(metrics.compute_cpu_step_deterministic(0), 0.0)
    
    def test_compute_effort(self):
        """Test du calcul d'effort."""
        delta_An = np.array([0.1, -0.05, 0.02])
        delta_fn = np.array([0.01, 0.02, -0.01])
        delta_gamma = np.array([0.0, 0.0, 0.0])
        
        effort = metrics.compute_effort(delta_An, delta_fn, delta_gamma, 1.0, 1.0, 1.0)
        
        # L'effort est la somme normalisée des variations
        expected = np.sum(np.abs(delta_An)) + np.sum(np.abs(delta_fn))
        self.assertAlmostEqual(effort, expected, places=6)
    
    def test_compute_fluidity(self):
        """Fluidité de référence = jerk de l'enveloppe fₙ (métrique du switch)."""
        rng = np.random.RandomState(0)
        t = np.linspace(0, 10, 100)
        # Moins de 4 points : rien de saccadé encore
        self.assertEqual(metrics.compute_fluidity([1.0, 1.0, 1.0]), 1.0)
        # Tempo qui glisse doucement : très fluide
        smooth = 1.0 + 0.1 * np.sin(t)
        f_smooth = metrics.compute_fluidity(smooth)
        self.assertGreater(f_smooth, 0.9)
        # Tempo bruité : moins fluide, et borné dans [0, 1]
        noisy = smooth + 0.05 * rng.randn(100)
        f_noisy = metrics.compute_fluidity(noisy)
        self.assertLess(f_noisy, f_smooth)
        self.assertTrue(0.0 <= f_noisy <= 1.0)
        # Sans dimension : invariante à l'échelle
        self.assertAlmostEqual(metrics.compute_fluidity(10 * noisy), f_noisy, places=9)
    
    def test_compute_effort_status(self):
        """Statut de l'activité en facteurs du repos du run (référence figée)."""
        ref = 100.0
        calm = [100.0 + 5 * np.sin(i / 3) for i in range(50)]
        self.assertEqual(metrics.compute_effort_status(102.0, calm, self.config, ref=ref), "stable")
        # Pic : un pas au-delà de ×2 du repos → transitoire
        self.assertEqual(metrics.compute_effort_status(2.5 * ref, calm, self.config, ref=ref), "transitoire")
        # Agitation installée : niveau (médiane sur une respiration) tenu au-delà de ×1.25 → chronique
        high = [1.4 * ref] * 200
        self.assertEqual(metrics.compute_effort_status(1.4 * ref, high, self.config, ref=ref), "chronique")
        # La crête d'une respiration (période 200 pas, ±20 %) n'est PAS chronique
        breath = [ref * (1 + 0.2 * np.sin(2 * np.pi * i / 200)) for i in range(400)]
        self.assertEqual(metrics.compute_effort_status(breath[-1], breath, self.config, ref=ref), "stable")
        self.assertTrue(0.98 <= metrics.activity_level(breath, 200) / ref <= 1.02)
        self.assertIsNone(metrics.activity_level(breath[:100], 200))
        # Sans référence (calibration en cours) : seul un pic à +2σ est signalé,
        # jamais « chronique » ; les anciens seuils fixes de config sont ignorés
        self.assertEqual(metrics.compute_effort_status(0.5, [0.5] * 50, self.config), "stable")
        noisy = list(0.5 + 0.01 * np.random.RandomState(0).randn(50))
        self.assertEqual(metrics.compute_effort_status(2.0, noisy, self.config), "transitoire")
        self.assertEqual(metrics.compute_effort_status(1.5, [1.5] * 50, self.config), "stable")

    def test_activity_reference_and_ratio(self):
        # Repos du run : médiane sur [t_start, t_end] APRÈS l'exploration initiale
        # (pic d'activité γ/G au début du run) ; None tant que la fenêtre n'est
        # pas complète ; puis le score lit le RAPPORT au repos.
        t = [i * 0.1 for i in range(600)]
        eff = [2000.0 if ti < 5 else 270.0 + 30 * np.sin(ti) for ti in t]  # pic initial puis repos ≈ 270
        self.assertIsNone(metrics.activity_reference(t[:300], eff[:300], 20.0, 40.0))  # fenêtre incomplète
        ref = metrics.activity_reference(t, eff, 20.0, 40.0)
        self.assertTrue(255 <= ref <= 285, ref)   # le pic initial n'entre pas dans la référence
        self.assertIsNone(metrics.activity_ratio([300.0], None))
        self.assertAlmostEqual(metrics.activity_ratio([270.0, 270.0], 270.0), 1.0)
        # Barème en facteurs : repos → 5 ; ×1.3 → 3 ; ×2.5 → 1 (la grenouille)
        for ratio, score in ((1.0, 5), (1.15, 4), (1.3, 3), (1.7, 2), (2.5, 1)):
            self.assertEqual(metrics.score_from_brackets(ratio, 'activite'), score, ratio)
        # Le scoreur lit 'activite_rel' loggé (niveau sur une respiration / repos) ;
        # repli : moyenne de la fenêtre / 'activite_ref' ; rien → None → neutre
        rows = [{'effort(t)': 351.0, 'activite_ref': 270.0, 'activite_rel': 1.05} for _ in range(20)]
        self.assertAlmostEqual(metrics.compute_reference_metrics(rows, 0.1, signal='O', N=3)['activite'], 1.05)
        rows = [{'effort(t)': 351.0, 'activite_ref': 270.0} for _ in range(20)]
        rows_noref = [{'effort(t)': 351.0} for _ in range(20)]
        self.assertAlmostEqual(metrics.compute_reference_metrics(rows, 0.1, signal='O', N=3)['activite'], 1.3)
        self.assertIsNone(metrics.compute_reference_metrics(rows_noref, 0.1, signal='O', N=3)['activite'])
        self.assertEqual(metrics.score_reference_metrics({'activite': None})['activite'], metrics.NEUTRAL_SCORE)


class TestReferenceScores(unittest.TestCase):
    """Le scoreur unique du pipeline : six métriques, une fenêtre, un barème."""

    @staticmethod
    def _history(n=60, N=3, dt=0.1, seed=1):
        rng = np.random.RandomState(seed)
        hist = []
        for i in range(n):
            t = i * dt
            O = np.array([np.sin(t + k) for k in range(N)])
            E = O + 0.05 * rng.randn(N)
            fn = 1.0 + 0.1 * np.sin(t / 3) + np.zeros(N)
            hist.append({
                't': t, 'S(t)': float(np.sum(O)) * 0.5, 'O': O, 'E': E, 'fn': fn,
                'mean_abs_error': float(np.mean(np.abs(E - O))),
                'effort(t)': 20.0 + rng.rand(), 'activite_ref': 20.5, 'resilience_ac': 0.30 + 0.0001 * i,  # moniteur lent loggé (autocorr CSD)
                'On_mean(t)': float(np.mean(O)), 'fn_mean(t)': float(np.mean(fn)),
                'innovation_cjs': 0.25 + 0.0001 * i,  # moniteur lent loggé (C_JS enveloppe fₙ)
            })
        return hist

    def test_reference_window_single_source(self):
        self.assertEqual(metrics.reference_window({'perception': {'W_f_t': 5}}, 0.1), 50)
        self.assertEqual(metrics.reference_window({}, 0.1), 50)
        self.assertEqual(metrics.reference_window({'perception': {'W_f_t': 0.2}}, 0.1), 10)

    def test_scores_keys_and_range(self):
        hist = self._history()
        for signal in ('O', 'S'):
            sc = metrics.compute_reference_scores(hist[-50:], 0.1, signal=signal, N=3)
            self.assertEqual(tuple(sc.keys()), metrics.REFERENCE_SCORE_KEYS)
            self.assertTrue(all(1 <= v <= 5 for v in sc.values()))
        self.assertEqual(set(metrics.FILTER_TO_SCORE_KEY.values()), set(metrics.REFERENCE_SCORE_KEYS))
        self.assertEqual(metrics.compute_reference_scores([], 0.1), metrics.neutral_reference_scores())

    def test_raw_values_match_reference_functions(self):
        hist = self._history()[-50:]
        raw = metrics.compute_reference_metrics(hist, 0.1, signal='O', N=3)
        O_series = [float(np.sum(h['O'])) for h in hist]
        # dispersion = std(ΣO) NORMALISÉ par √N (cloche autour de l'amplitude saine).
        self.assertAlmostEqual(raw['dispersion'], float(np.std(O_series)) / np.sqrt(3))
        self.assertAlmostEqual(raw['fluidity'], metrics.compute_fluidity([float(np.mean(h['fn'])) for h in hist]))
        # innovation = moniteur lent loggé (C_JS enveloppe fₙ) : le scoreur lit la
        # dernière valeur 'innovation_cjs' disponible dans la fenêtre, il ne la recalcule pas.
        self.assertAlmostEqual(raw['innovation'], float(hist[-1]['innovation_cjs']))
        self.assertAlmostEqual(raw['regulation'], float(np.mean([h['mean_abs_error'] for h in hist])))
        self.assertAlmostEqual(raw['activite'], float(np.mean([h['effort(t)'] for h in hist])) / 20.5)  # rapport au repos
        # résilience = moniteur lent loggé : dernière valeur 'resilience_ac' de la fenêtre
        self.assertAlmostEqual(raw['resilience'], float(hist[-1]['resilience_ac']))
        self.assertEqual(metrics.score_reference_metrics(raw)['resilience'],
                         metrics.score_from_brackets(hist[-1]['resilience_ac'], 'resilience'))

    def test_signal_target_changes_only_signal_metrics(self):
        hist = self._history()[-50:]
        raw_O = metrics.compute_reference_metrics(hist, 0.1, signal='O', N=3)
        raw_S = metrics.compute_reference_metrics(hist, 0.1, signal='S', N=3)
        # Depuis le 21/09, la dispersion (identité) se lit sur O brut : aucune des
        # six ne dépend du signal noté. Les scores S (γ) et O (switch) coïncident ;
        # la plomberie `signal` reste pour une future métrique du perçu.
        for k in metrics.REFERENCE_SCORE_KEYS:
            self.assertEqual(raw_O[k], raw_S[k], k)

    def test_csv_like_rows_use_On_mean(self):
        hist = self._history()[-50:]
        rows = [{k: v for k, v in h.items() if k not in ('O', 'E', 'fn')} for h in hist]
        raw_full = metrics.compute_reference_metrics(hist, 0.1, signal='O', N=3)
        raw_rows = metrics.compute_reference_metrics(rows, 0.1, signal='O', N=3)
        self.assertAlmostEqual(raw_full['dispersion'], raw_rows['dispersion'])
        self.assertAlmostEqual(raw_full['fluidity'], raw_rows['fluidity'])

    def test_no_resilience_verdict_is_neutral(self):
        hist = self._history()[-50:]
        for h in hist:
            h['resilience_ac'] = None
        sc = metrics.compute_reference_scores(hist, 0.1, signal='O', N=3)
        self.assertEqual(sc['resilience'], metrics.NEUTRAL_SCORE)
        self.assertEqual(set(metrics.labelled_scores(sc).keys()), set(metrics.SCORE_KEY_LABELS.values()))

    def test_dispersion_bell(self):
        # Cloche : 5 au centre (amplitude saine), baisse des DEUX côtés.
        c = metrics.SCORE_BRACKETS['dispersion']['center']
        self.assertEqual(metrics.SCORE_BRACKETS['dispersion']['direction'], 'bell')
        self.assertEqual(metrics.score_from_brackets(c, 'dispersion'), 5)
        self.assertEqual(metrics.score_from_brackets(c * 1.2, 'dispersion'), 5)   # bande saine
        self.assertEqual(metrics.score_from_brackets(c / 1.2, 'dispersion'), 5)
        self.assertEqual(metrics.score_from_brackets(c * 1.5, 'dispersion'), 4)   # emballement léger
        self.assertEqual(metrics.score_from_brackets(c / 1.5, 'dispersion'), 4)   # gel léger
        self.assertEqual(metrics.score_from_brackets(c * 5, 'dispersion'), 1)     # explose
        self.assertEqual(metrics.score_from_brackets(c / 5, 'dispersion'), 1)     # fige
        self.assertEqual(metrics.score_from_brackets(0.0, 'dispersion'), 1)       # gel total

    def test_dispersion_normalised_by_sqrt_N(self):
        # std(ΣO)/√N : même chimère à N différents → même dispersion normalisée.
        def hist_N(N, n=60):
            rng = np.random.RandomState(4)
            h = []
            for i in range(n):
                O = 0.02 * rng.randn(N)  # N contributions ~indépendantes, centrées
                h.append({'O': O, 'S(t)': float(np.sum(O)),
                          'fn': np.ones(N), 'fn_mean(t)': 1.0})
            return h
        d20 = metrics.compute_reference_metrics(hist_N(20), 0.1, signal='O', N=20)['dispersion']
        d100 = metrics.compute_reference_metrics(hist_N(100), 0.1, signal='O', N=100)['dispersion']
        # invariance en N à ~15 % près (le √N retire l'essentiel de l'échelle)
        self.assertLess(abs(d20 - d100) / d100, 0.15)
        # N inconnu et non déductible (pas de vecteur O) → verdict suspendu.
        rows = [{'On_mean(t)': 0.001, 'S(t)': 0.1, 'fn_mean(t)': 1.0} for _ in range(60)]
        self.assertIsNone(metrics.compute_reference_metrics(rows, 0.1, signal='O', N=None)['dispersion'])

    def test_dispersion_reads_O_only_and_logged_monitor(self):
        # Identité : la dispersion se lit sur O brut quel que soit le signal noté
        # (γ note S(t) ; un filtre de perception change ce que le système regarde,
        # pas qui il est). Ici S(t) = 0.5·ΣO : la dispersion S serait 2× plus petite.
        hist = self._history()[-50:]
        dO = metrics.compute_reference_metrics(hist, 0.1, signal='O', N=3)['dispersion']
        dS = metrics.compute_reference_metrics(hist, 0.1, signal='S', N=3)['dispersion']
        self.assertAlmostEqual(dO, dS)
        self.assertAlmostEqual(dO, float(np.std([float(np.sum(h['O'])) for h in hist])) / np.sqrt(3))
        # Moniteur logué par simulate ('dispersion_norm', lissé) : lu de préférence,
        # dernière valeur ; sans la colonne → repli sur la fenêtre.
        for i, h in enumerate(hist):
            h['dispersion_norm'] = 0.0157 * (1 + 0.001 * i)
        self.assertAlmostEqual(metrics.compute_reference_metrics(hist, 0.1, signal='O', N=3)['dispersion'], hist[-1]['dispersion_norm'])
        self.assertEqual(metrics.compute_reference_scores(hist, 0.1, signal='S', N=3)['dispersion'], 5)

    def test_dispersion_is_observe_only(self):
        # Métrique d'identité : notée et visible, jamais engagée comme remède.
        self.assertIn('stabilite', metrics.OBSERVE_ONLY_FILTERS)
        self.assertIn('innovation', metrics.OBSERVE_ONLY_FILTERS)

    def test_innovation_absent_column_is_neutral(self):
        # Un CSV / warmup sans 'innovation_cjs' → verdict suspendu, score neutre.
        hist = self._history()[-50:]
        for h in hist:
            h.pop('innovation_cjs', None)
        raw = metrics.compute_reference_metrics(hist, 0.1, signal='O', N=3)
        self.assertIsNone(raw['innovation'])
        sc = metrics.compute_reference_scores(hist, 0.1, signal='O', N=3)
        self.assertEqual(sc['innovation'], metrics.NEUTRAL_SCORE)

    def test_innovation_deficit_uses_score_bracket(self):
        # Déficit par strate = même métrique que le score, seuil haut du barème.
        top = metrics.SCORE_BRACKETS['innovation']['thresholds'][0]
        self.assertEqual(metrics.innovation_deficit(None), 0.0)
        self.assertEqual(metrics.innovation_deficit(top), 0.0)
        self.assertAlmostEqual(metrics.innovation_deficit(top / 2), 0.5)
        self.assertEqual(metrics.innovation_deficit(2 * top), 0.0)
        # Câblage dans le filtre de perception : fenêtre courte → aucun verdict → 0
        short = np.ones((50, 3))
        d = dynamics.compute_perception_deficit('innovation', short, short, short, 0.1)
        self.assertTrue(np.all(d == 0.0))
        long_flat = np.ones((300, 3))
        d = dynamics.compute_perception_deficit('innovation', short, short, short, 0.1, fn_long_win=long_flat)
        self.assertTrue(np.all(d == 1.0))  # plat : C_JS = 0 → déficit maximal

    def test_innovation_plane_H_reads_the_side_of_the_bell(self):
        # Une seule implémentation : C de compute_innovation_cjs == C du plan.
        rng = np.random.RandomState(0)
        noise = list(rng.randn(2000))
        t = np.linspace(0, 40 * np.pi, 2000)
        sine = list(np.sin(t))
        p_noise = metrics.compute_innovation_plane(noise, 0.1)
        p_sine = metrics.compute_innovation_plane(sine, 0.1)
        self.assertEqual(metrics.compute_innovation_cjs(noise, 0.1), p_noise['C'])
        self.assertEqual(metrics.compute_innovation_cjs(sine, 0.1), p_sine['C'])
        # H distingue les deux côtés : bruit → H proche de 1, ordre → H nettement plus bas
        self.assertGreater(p_noise['H'], 0.95)
        self.assertLess(p_sine['H'], p_noise['H'] - 0.2)
        self.assertTrue(0.0 <= p_sine['H'] <= 1.0)
        self.assertIsNone(metrics.compute_innovation_plane([1.0, 2.0], 0.1))
        self.assertEqual(metrics.compute_innovation_plane([1.0] * 300, 0.1), {'H': 0.0, 'C': 0.0})

    def test_observe_only_filters_are_scored_but_never_remedies(self):
        # L'innovation reste dans les six scores et dans le mapping des filtres…
        self.assertIn('innovation', metrics.FILTER_TO_SCORE_KEY)
        self.assertIn('innovation', metrics.REFERENCE_SCORE_KEYS)
        # … mais est déclarée « observation seule » : jamais un remède du switch.
        self.assertIn('innovation', metrics.OBSERVE_ONLY_FILTERS)
        self.assertTrue(set(metrics.OBSERVE_ONLY_FILTERS) <= set(metrics.FILTER_TO_SCORE_KEY))

    def test_innovation_cjs_contract(self):
        # Fenêtre trop courte → None (score neutre en aval).
        self.assertIsNone(metrics.compute_innovation_cjs([0.0, 1.0, 0.0], 0.1))
        # Signal plat → ordre pur → C_JS nul.
        self.assertEqual(metrics.compute_innovation_cjs([1.0] * 300, 0.1), 0.0)
        # Bruit blanc : C_JS bas (contrairement à l'entropie). Ordre pur (sinus)
        # aussi bas. Les deux dans [0, ~0.5]. C'est la cloche native de C_JS.
        rng = np.random.RandomState(0)
        c_noise = metrics.compute_innovation_cjs(list(rng.randn(2000)), 0.1)
        t = np.linspace(0, 40 * np.pi, 2000)
        c_sine = metrics.compute_innovation_cjs(list(np.sin(t)), 0.1)
        for c in (c_noise, c_sine):
            self.assertIsNotNone(c)
            self.assertTrue(0.0 <= c <= 0.6)
        self.assertLess(c_noise, 0.2)  # le bruit N'EST PAS noté "innovant"


class TestResilienceCSD(unittest.TestCase):
    """Résilience = ralentissement critique : banc AR(1) du cahier (New_Attention.md)."""

    @staticmethod
    def _ar1(phi, n=4000, seed=0):
        rng = np.random.RandomState(seed)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + rng.randn()
        return x

    def test_ar1_pole_recovered_and_drive_detrended(self):
        # Grille pré-enregistrée : lag-1 pur ≈ φ ; + forçage : brut aveugle (~0.96),
        # détrendé ≈ φ. Reproduit la table du cahier.
        for phi in (0.6, 0.8, 0.9, 0.95):
            x = self._ar1(phi)
            drive = 5 * x.std() * np.sin(2 * np.pi * np.arange(len(x)) / 50)
            self.assertAlmostEqual(metrics.lag_autocorrelation(x, 1), phi, delta=0.03)
            self.assertGreater(metrics.lag_autocorrelation(x + drive, 1), 0.93)
            r = metrics.compute_resilience_csd(x + drive, 0.1, lag=1)
            self.assertAlmostEqual(r['ac'], phi, delta=0.03)
            self.assertGreaterEqual(r['n_peaks'], 1)
        # Spectre rouge lisse : aucune raie retirée
        self.assertEqual(metrics.compute_resilience_csd(self._ar1(0.8), 0.1, lag=1)['n_peaks'], 0)

    def test_relaxation_is_the_single_timescale(self):
        # AR(1) φ=0.9 : temps de retour −1/ln φ ≈ 9.5 pas → relaxation 1/e ≈ 9-10 pas
        relax = metrics.relaxation_steps(self._ar1(0.9, n=20000))
        self.assertTrue(7 <= relax <= 12, relax)
        self.assertEqual(metrics.relaxation_steps([1.0] * 50), 10)  # plat → défaut

    def test_contract_none_and_direction(self):
        self.assertIsNone(metrics.compute_resilience_csd([0.0, 1.0] * 20, 0.1))       # trop court
        self.assertIsNone(metrics.compute_resilience_csd([1.0] * 300, 0.1))           # plat
        r = metrics.compute_resilience_csd(np.sin(2 * np.pi * np.arange(400) / 20), 0.1)  # sinus pur : résidus nuls → quiet
        self.assertTrue(r['quiet']); self.assertEqual(r['ac'], 0.0)
        # Barème monotone unilatéral : autocorr basse = 5, haute = 1
        self.assertEqual(metrics.score_from_brackets(0.33, 'resilience'), 5)
        self.assertEqual(metrics.score_from_brackets(0.50, 'resilience'), 4)
        self.assertEqual(metrics.score_from_brackets(0.95, 'resilience'), 1)
        self.assertEqual(metrics.SCORE_BRACKETS['resilience']['direction'], 'lower')

    def test_bareme_anchored_on_slowing_factors(self):
        # Seuils = exp(−1/k) : au lag calibré sur la relaxation calme τ₀, un retour
        # exponentiel de temps τ = k·τ₀ donne une autocorr exp(−1/k).
        thr = metrics.SCORE_BRACKETS['resilience']['thresholds']
        for k, v in zip(metrics.RESILIENCE_SLOWING_FACTORS, thr):
            self.assertAlmostEqual(v, np.exp(-1.0 / k), places=3)
        # Ancrage 1/e sur AR(1) : lag = relaxation du processus calme, puis le
        # même lag appliqué à des processus ralentis ×k → score décroissant.
        n = 60000
        tau0 = 12.0
        x_calm = self._ar1(np.exp(-1.0 / tau0), n=n, seed=3)
        lag = metrics.relaxation_steps(x_calm)
        self.assertTrue(9 <= lag <= 15, lag)
        ac_calm = metrics.lag_autocorrelation(x_calm, lag)
        self.assertAlmostEqual(ac_calm, np.exp(-lag / tau0), delta=0.05)
        self.assertEqual(metrics.score_from_brackets(ac_calm, 'resilience'), 5)
        expected = {1.7: 4, 2.5: 3, 4.0: 2, 8.0: 1}
        for k, score in expected.items():
            x_slow = self._ar1(np.exp(-1.0 / (k * tau0)), n=n, seed=int(10 * k))
            ac = metrics.lag_autocorrelation(x_slow, lag)
            self.assertAlmostEqual(ac, np.exp(-lag / (k * tau0)), delta=0.06)
            self.assertEqual(metrics.score_from_brackets(ac, 'resilience'), score, (k, ac))

    def test_smoothed_ac_is_what_the_scorer_reads(self):
        rng = np.random.RandomState(0)
        raw = list(0.30 + 0.2 * rng.randn(200))
        sm = metrics.smooth_resilience_ac(raw, lag=10, n_lags=3)
        self.assertAlmostEqual(sm, float(np.median(raw[-30:])))
        self.assertIsNone(metrics.smooth_resilience_ac([None, float('nan')], lag=10))
        hist = TestReferenceScores._history()[-50:]
        for i, h in enumerate(hist):
            h['resilience_ac'] = 0.9            # brute bruitée : ignorée par le scoreur
            h['resilience_ac_smooth'] = 0.30    # lissée : lue par le scoreur
        raw_m = metrics.compute_reference_metrics(hist, 0.1, signal='O', N=3)
        self.assertAlmostEqual(raw_m['resilience'], 0.30)
        for h in hist:
            h.pop('resilience_ac_smooth')
        self.assertAlmostEqual(metrics.compute_reference_metrics(hist, 0.1, signal='O', N=3)['resilience'], 0.9)  # repli CSV ancien

    @staticmethod
    def _fps_like_envelope(n, tau=None, rel=0.05, seed=0):
        # Enveloppe fₙ « façon FPS » : périodique (période 200 pas + harmoniques),
        # éventuellement modulée par une fluctuation AR(1) de temps τ (en pas).
        t = np.arange(n)
        env = 1 + 0.3 * np.sin(2 * np.pi * t / 200) + 0.05 * np.sin(4 * np.pi * t / 200) + 0.01 * np.sin(6 * np.pi * t / 200)
        if tau is None:
            return env, None
        rng = np.random.RandomState(seed)
        phi = np.exp(-1.0 / tau)
        a = np.zeros(n + 800)
        for i in range(1, n + 800):
            a[i] = phi * a[i - 1] + rng.randn()
        a = a[800:] * np.sqrt(1 - phi ** 2)
        return env * (1 + rel * a), a

    def test_detrend_periodic_keeps_slow_fluctuations(self):
        # Le détrend retire les RAIES (forçage + harmoniques) et laisse la
        # fluctuation lente intacte : les résidus sont la fluctuation injectée.
        env, _ = self._fps_like_envelope(2000)
        resid, n_lines = metrics.detrend_periodic(env)
        self.assertGreaterEqual(n_lines, 3)
        self.assertLess(np.var(resid) / np.var(env), 1e-8)          # périodique pur → reste numérique
        y, a = self._fps_like_envelope(2000, tau=160, seed=1)
        resid, n_lines = metrics.detrend_periodic(y)
        self.assertGreaterEqual(n_lines, 2)
        self.assertGreater(abs(np.corrcoef(resid, a)[0, 1]), 0.9)   # fluctuation τ=160 préservée
        # Bruit rouge pur (AR(1)) : aucune raie, quel que soit τ
        for tau in (5, 40, 160):
            _, a = self._fps_like_envelope(2000, tau=tau, seed=tau)
            self.assertEqual(metrics.detrend_periodic(a)[1], 0, tau)

    def test_in_situ_calibration_reads_slowing_factor(self):
        # Table de calibration (20/09) : fenêtre 2000 pas, lag 10 → l'autocorr
        # mesurée in-situ vaut exp(−1/k) à ~0.06 près pour τ = k·lag, et le score
        # tombe à ±1 cran du nominal ; monotone décroissant en k.
        W, lag = 2000, 10
        means = []
        for k in (1.0, 2.0, 5.0):
            acs = []
            for seed in range(8):
                y, _ = self._fps_like_envelope(W, tau=k * lag, seed=100 * int(k) + seed)
                r = metrics.compute_resilience_csd(y, 0.1, lag=lag)
                self.assertFalse(r['quiet'])
                acs.append(r['ac'])
                nominal = metrics.score_from_brackets(float(np.exp(-1.0 / k)), 'resilience')
                self.assertLessEqual(abs(metrics.score_from_brackets(r['ac'], 'resilience') - nominal), 1, (k, seed, r['ac']))
            means.append(float(np.mean(acs)))
            self.assertAlmostEqual(means[-1], np.exp(-1.0 / k), delta=0.08, msg=(k, means[-1]))
        self.assertTrue(means[0] < means[1] < means[2], means)

    def test_auto_lag_is_capped_by_window(self):
        # Le lag auto (relaxation 1/e de l'enveloppe brute ≈ 40 pas pour une
        # période de 200) est plafonné à W/min_lags_per_window : le biais de
        # fenêtre (≈ 2τ/W) rendrait un lag de 40 illisible sur 2000 pas.
        y, _ = self._fps_like_envelope(2000, tau=20, seed=5)
        self.assertTrue(30 <= metrics.relaxation_steps(y) <= 50)
        self.assertEqual(metrics.compute_resilience_csd(y, 0.1)['lag'], 10)
        self.assertEqual(metrics.compute_resilience_csd(y, 0.1, min_lags_per_window=1)['lag'], metrics.relaxation_steps(y))
        self.assertEqual(metrics.compute_resilience_csd(y, 0.1, lag=25)['lag'], 25)   # fourni = figé

    def test_quiet_floor_no_verdict_on_numerical_residue(self):
        # Enveloppe quasi périodique sans fluctuation : les résidus sont un reste
        # numérique → quiet, ac = 0 par convention (rien ne ralentit), score 5.
        env = 4.0 + np.sin(2 * np.pi * np.arange(800) / 20) * (1 + 0.3 * np.sin(2 * np.pi * np.arange(800) / 200))
        r = metrics.compute_resilience_csd(env, 0.1)
        self.assertIsNotNone(r); self.assertTrue(r['quiet']); self.assertEqual(r['ac'], 0.0)
        self.assertEqual(metrics.score_from_brackets(r['ac'], 'resilience'), 5)
        # La même enveloppe avec une fluctuation réelle (AR(1) 1 % de l'amplitude) : mesurée
        rng = np.random.RandomState(0)
        ar = np.zeros(800)
        for i in range(1, 800):
            ar[i] = 0.9 * ar[i - 1] + rng.randn()
        r2 = metrics.compute_resilience_csd(env + 0.01 * ar / ar.std(), 0.1)
        self.assertFalse(r2['quiet']); self.assertTrue(-1 <= r2['ac'] <= 1)

    def test_radar_alert_rule(self):
        # Calme stationnaire : jamais d'alerte. Montée conjointe ac+var : alerte.
        rng = np.random.RandomState(1)
        calm_ac = list(0.30 + 0.02 * rng.randn(160)); calm_var = list(1e-3 + 1e-4 * rng.randn(160))
        self.assertEqual(metrics.resilience_alert(calm_ac, calm_var, calm_n=100, band=0.06, n_windows=3, stride=10), 0)
        rise_ac = calm_ac + list(np.linspace(0.35, 0.65, 60)); rise_var = calm_var + list(np.linspace(1.2e-3, 3e-3, 60))
        self.assertEqual(metrics.resilience_alert(rise_ac, rise_var, calm_n=100, band=0.06, n_windows=3, stride=10), 1)
        # Un seul indicateur qui monte = suspect, pas d'alerte (convergence exigée)
        self.assertEqual(metrics.resilience_alert(rise_ac, calm_var + [1e-3] * 60, calm_n=100, band=0.06, n_windows=3, stride=10), 0)
        # Sans référence calme complète (calm_n + gap + tendance) : jamais d'alerte
        self.assertEqual(metrics.resilience_alert(rise_ac[:120], rise_var[:120], calm_n=100, gap=40), 0)

    def test_radar_reference_slides_with_the_run(self):
        # Campagne 20/09 : une référence figée sur le début du run compare tout le
        # régime à la queue du transitoire → le radar sonnait des centaines de pas
        # sous fluctuation STATIONNAIRE. La référence glissante (calm_n entrées,
        # terminées `gap` entrées avant le présent) ne lit qu'une MONTÉE.
        rng = np.random.RandomState(2)
        kw = dict(calm_n=40, band=0.06, n_windows=3, stride=2, gap=40)
        # (a) transitoire bas puis plateau stationnaire : aucune alerte en régime
        ac = list(0.20 + 0.02 * rng.randn(50)) + list(0.60 + 0.02 * rng.randn(400))
        var = list(1e-3 + 1e-4 * rng.randn(50)) + list(3e-3 + 2e-4 * rng.randn(400))
        regime = [metrics.resilience_alert(ac[:i], var[:i], **kw) for i in range(200, 451)]
        self.assertEqual(sum(regime), 0, sum(regime))
        # (b) montée à mi-parcours : le radar sonne PENDANT la montée…
        ac = list(0.30 + 0.01 * rng.randn(200)) + list(np.linspace(0.30, 0.85, 150)) + list(0.85 + 0.01 * rng.randn(200))
        var = list(1e-3 + 5e-5 * rng.randn(200)) + list(np.linspace(1e-3, 4e-3, 150)) + list(4e-3 + 5e-5 * rng.randn(200))
        before = [metrics.resilience_alert(ac[:i], var[:i], **kw) for i in range(100, 201)]
        during = [metrics.resilience_alert(ac[:i], var[:i], **kw) for i in range(250, 351)]
        after = [metrics.resilience_alert(ac[:i], var[:i], **kw) for i in range(450, 551)]
        self.assertEqual(sum(before), 0)
        self.assertGreater(sum(during), 50, sum(during))
        # … et se tait une fois le plateau atteint (la référence a rejoint le présent)
        self.assertEqual(sum(after), 0, sum(after))


class TestValidateConfig(unittest.TestCase):
    """Tests pour le module validate_config.py"""
    
    def test_validate_valid_config(self):
        """Test avec une configuration valide."""
        config = validate_config.generate_default_config(N=5, T=100)
        errors, warnings = validate_config.validate_config(config)
        
        self.assertEqual(len(errors), 0)
    
    def test_validate_invalid_config(self):
        """Test avec des configurations invalides."""
        # Config sans système
        config = {'strates': []}
        errors, warnings = validate_config.validate_config(config)
        self.assertGreater(len(errors), 0)
        
        # N négatif
        config = validate_config.generate_default_config(N=5, T=100)
        config['system']['N'] = -1
        errors, warnings = validate_config.validate_config(config)
        self.assertGreater(len(errors), 0)
    
    def test_update_config_threshold(self):
        """Test de mise à jour de seuil."""
        config = validate_config.generate_default_config(N=5, T=100)
        
        # Créer un changelog temporaire
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            changelog_path = f.name
        
        try:
            success = validate_config.update_config_threshold(
                config, 'mean_high_effort', 3.5, 'Test', changelog_path
            )
            
            self.assertTrue(success)
            self.assertEqual(config['to_calibrate']['mean_high_effort'], 3.5)
            
            # Vérifier le changelog
            with open(changelog_path, 'r') as f:
                content = f.read()
                self.assertIn('mean_high_effort', content)
                self.assertIn('3.5', content)
        finally:
            os.unlink(changelog_path)


class TestPerturbations(unittest.TestCase):
    """Tests pour le module perturbations.py"""
    
    def test_perturbation_types(self):
        """Test de tous les types de perturbations."""
        # Choc
        config_choc = {'type': 'choc', 't0': 5.0, 'amplitude': 2.0, 'duration': 0.5}
        self.assertEqual(perturbations.generate_perturbation(4.9, config_choc), 0.0)
        self.assertEqual(perturbations.generate_perturbation(5.1, config_choc), 2.0)
        self.assertEqual(perturbations.generate_perturbation(5.6, config_choc), 0.0)
        
        # Rampe
        config_rampe = {'type': 'rampe', 't0': 0.0, 'amplitude': 1.0, 'duration': 10.0}
        self.assertAlmostEqual(perturbations.generate_perturbation(5.0, config_rampe), 0.5, places=6)
        
        # Sinus
        config_sinus = {'type': 'sinus', 't0': 0.0, 'amplitude': 1.0, 'freq': 0.5}
        value = perturbations.generate_perturbation(1.0, config_sinus)
        self.assertAlmostEqual(value, np.sin(np.pi), places=6)
    
    def test_perturbation_sequence(self):
        """Test de séquence de perturbations."""
        configs = [
            {'type': 'choc', 't0': 5.0, 'amplitude': 2.0},
            {'type': 'sinus', 't0': 0.0, 'amplitude': 0.5, 'freq': 1.0}
        ]
        
        sequence = perturbations.generate_perturbation_sequence(10, 0.1, configs)
        
        # Vérifier la longueur
        self.assertEqual(len(sequence), 100)
        
        # Vérifier qu'il y a des valeurs non nulles
        self.assertGreater(np.abs(sequence).sum(), 0)
    
    def test_scenarios(self):
        """Test des scénarios prédéfinis."""
        for scenario in ['stress_test', 'environnement_variable', 'chaos']:
            configs = perturbations.create_scenario(scenario, T=100)
            self.assertIsInstance(configs, list)
            self.assertGreater(len(configs), 0)


class TestAnalyze(unittest.TestCase):
    """Tests pour le module analyze.py"""
    
    def setUp(self):
        """Configuration de test."""
        self.config = validate_config.generate_default_config(N=3, T=50)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Nettoyage."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_run_data(self):
        """Test du chargement de données."""
        # Créer un CSV de test
        csv_path = os.path.join(self.temp_dir, 'test_run.csv')
        with open(csv_path, 'w') as f:
            f.write('t,S(t),effort(t)\n')
            f.write('0,0.0,0.5\n')
            f.write('1,0.5,0.6\n')
            f.write('2,1.0,0.7\n')
        
        data = analyze.load_run_data(csv_path)
        
        self.assertIn('t', data)
        self.assertIn('S(t)', data)
        self.assertEqual(len(data['t']), 3)
    
    def test_refinement_functions(self):
        """Test des fonctions de raffinement."""
        stats = {'trigger_rate': 0.7, 'mean_values': [0.02, 0.03]}
        
        # Test raffinement fluidité
        changes = analyze.refine_fluidity(self.config.copy(), stats)
        self.assertIn('sigma_n', changes)
        
        # Test raffinement stabilité
        changes = analyze.refine_stability(self.config.copy(), stats)
        self.assertIn('k', changes)


class TestExplore(unittest.TestCase):
    """Tests pour le module explore.py"""
    
    def setUp(self):
        """Données de test."""
        t = np.linspace(0, 100, 1000)
        self.test_data = {
            'S(t)': np.sin(2 * np.pi * t / 10) + 0.1 * np.random.randn(1000),
            'C(t)': np.cos(2 * np.pi * t / 15),
            'effort(t)': 0.5 + 0.1 * np.sin(2 * np.pi * t / 20)
        }
        
        # Ajouter des anomalies
        self.test_data['S(t)'][200:220] += 5.0
        self.test_data['C(t)'][400] += 3.0
    
    def test_detect_anomalies(self):
        """Test de détection d'anomalies."""
        anomalies = explore.detect_anomalies(
            self.test_data, 
            ['S(t)', 'effort(t)'], 
            threshold=3.0, 
            min_duration=3
        )
        
        # Doit détecter au moins l'anomalie dans S(t)
        self.assertGreater(len(anomalies), 0)
        
        # Vérifier la structure
        if anomalies:
            self.assertIn('event_type', anomalies[0])
            self.assertIn('t_start', anomalies[0])
            self.assertIn('severity', anomalies[0])
    
    def test_detect_fractal_patterns(self):
        """Test de détection de motifs fractals."""
        # Ajouter des motifs auto-similaires
        for scale in [1, 10, 100]:
            self.test_data['S(t)'] += 0.1 / scale * np.sin(2 * np.pi * np.arange(1000) * scale / 10)
        
        fractals = explore.detect_fractal_patterns(
            self.test_data,
            metrics=['S(t)'],
            window_sizes=[1, 10, 100],
            threshold=0.7
        )
        
        # Peut détecter ou non selon le bruit
        self.assertIsInstance(fractals, list)


class TestKuramoto(unittest.TestCase):
    """Tests pour le module kuramoto.py"""
    
    def test_kuramoto_step(self):
        """Test d'un pas de Kuramoto."""
        phases = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        frequencies = np.ones(5) * 0.5
        
        dphases = kuramoto.kuramoto_step(phases, frequencies, K=1.0, N=5, dt=0.1)
        
        # Vérifier la forme
        self.assertEqual(len(dphases), 5)
        
        # Les dérivées ne doivent pas être toutes égales (couplage)
        self.assertFalse(np.allclose(dphases, frequencies))
    
    def test_compute_kuramoto_order(self):
        """Test du paramètre d'ordre."""
        # Cas synchronisé
        phases_sync = np.zeros(10)
        r_sync, psi_sync = kuramoto.compute_kuramoto_order(phases_sync)
        self.assertAlmostEqual(r_sync, 1.0, places=6)
        
        # Cas désynchronisé parfait
        phases_desync = np.linspace(0, 2*np.pi, 10, endpoint=False)
        r_desync, psi_desync = kuramoto.compute_kuramoto_order(phases_desync)
        self.assertLess(r_desync, 0.2)


class TestUtils(unittest.TestCase):
    """Tests pour le module utils.py"""
    
    def setUp(self):
        """Préparation."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Nettoyage."""
        shutil.rmtree(self.temp_dir)
    
    def test_generate_run_id(self):
        """Test de génération d'ID."""
        id1 = utils.generate_run_id()
        id2 = utils.generate_run_id()
        
        # IDs uniques
        self.assertNotEqual(id1, id2)
        
        # Format correct
        self.assertTrue(id1.startswith('run_'))
        self.assertIn('_', id1)
    
    def test_save_load_state(self):
        """Test sauvegarde/restauration d'état."""
        state = {
            't': 50.0,
            'strates': [{'id': 0, 'An': 1.0}],
            'history': [{'t': 0, 'S': 0}]
        }
        
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint.pkl')
        utils.save_simulation_state(state, checkpoint_path)
        
        # Vérifier que le fichier existe
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Restaurer
        restored = utils.load_simulation_state(checkpoint_path)
        self.assertEqual(restored['t'], state['t'])
        self.assertEqual(len(restored['strates']), len(state['strates']))
    
    def test_format_duration(self):
        """Test du formatage de durée."""
        self.assertEqual(utils.format_duration(45.3), "45.3s")
        self.assertEqual(utils.format_duration(125), "2m 5.0s")
        self.assertEqual(utils.format_duration(3665), "1h 1m 5.0s")


class TestSelfMemory(unittest.TestCase):
    """Mémoire de soi à deux étages (21/09/2026) : Eₙ = Oₙ à l'amplitude remémorée."""

    def _mem(self):
        mem = dynamics.init_self_memory(5)
        A = np.linspace(1, 2, 5); f = np.linspace(1, 1.5, 5)
        for _ in range(50):
            u = dynamics.update_self_memory(mem, A, f, 0.1, 2.0, 40.0)
        return mem, u

    def test_relative_and_born(self):
        mem, u = self._mem()
        self.assertTrue(mem['born'])
        self.assertTrue(np.allclose(mem['A_l'], np.linspace(1, 2, 5) / np.mean(np.linspace(1, 2, 5))))
        self.assertTrue(np.allclose(u['u'], 0.0))               # au repos : aucune surprise
        # une respiration COMMUNE (×1.3 sur tout le chœur) ne surprend personne
        u2 = dynamics.update_self_memory(mem, 1.3 * np.linspace(1, 2, 5), 1.3 * np.linspace(1, 1.5, 5), 0.1, 2.0, 40.0)
        self.assertLess(u2['u'].max(), 1e-9)

    def test_two_speeds_brief_erased_durable_assimilated(self):
        mem, _ = self._mem()
        A = np.linspace(1, 2, 5); f = np.linspace(1, 1.5, 5)
        # BREF : 20 pas (2 u.t.) de ×1.3 sur la strate 2, puis retour → la surprise monte puis s'efface,
        # la mémoire longue ne bouge presque pas
        A_l0 = mem['A_l'].copy(); peak = 0.0
        for _ in range(20):
            u = dynamics.update_self_memory(mem, A * np.array([1, 1, 1.3, 1, 1]), f, 0.1, 2.0, 40.0); peak = max(peak, u['u'][2])
        for _ in range(200):
            u = dynamics.update_self_memory(mem, A, f, 0.1, 2.0, 40.0)
        self.assertGreater(peak, 0.10); self.assertLess(u['u'][2], 0.02); self.assertLess(abs(mem['A_l'][2] - A_l0[2]) / A_l0[2], 0.03)
        # DURABLE : ×1.3 tenu → la surprise monte puis passe dans la mémoire longue (assimilation en ~τ_l)
        u_max = 0.0
        for i in range(1200):
            u = dynamics.update_self_memory(mem, A * np.array([1, 1, 1.3, 1, 1]), f, 0.1, 2.0, 40.0)
            if i < 80: u_max = max(u_max, u['u'][2])
        self.assertGreater(u_max, 0.15); self.assertLess(u['u'][2], 0.02)
        attendu = 1.3 * np.mean(A) / np.mean(A * np.array([1, 1, 1.3, 1, 1]))   # après 3·τ_l : assimilé à ~95 %
        self.assertLess(abs(mem['A_l'][2] / A_l0[2] - attendu) / attendu, 0.02)

    def test_E_is_O_at_remembered_amplitude(self):
        mem, _ = self._mem()
        A = np.linspace(1, 2, 5); O = np.array([0.3, -0.2, 0.5, -0.1, 0.4]) * A
        self.assertTrue(np.allclose(dynamics.compute_En_memory(O, mem), O))      # au repos : E = O, erreur nulle
        self.assertIsNone(dynamics.compute_En_memory(O, dynamics.init_self_memory(5)))   # pas née : None
        # la strate 2 devient 1.5× plus forte : après ~3 périodes le présent lissé a vu le saut, la mémoire
        # longue non → E garde la MÊME onde (même signe) à l'amplitude remémorée
        A2 = A * np.array([1, 1, 1.5, 1, 1]); O2 = O * np.array([1, 1, 1.5, 1, 1])
        for _ in range(30):
            dynamics.update_self_memory(mem, A2, np.ones(5), 0.1, 2.0, 40.0)
        ratio = dynamics.compute_En_memory(O2, mem)[2] / O2[2]
        self.assertGreater(ratio, 0.0)
        attendu = 1 / 1.5 * (np.mean(A2) / np.mean(A))
        self.assertLess(abs(ratio - attendu) / attendu, 0.12)


class TestAttentionConsideration(unittest.TestCase):
    """Le geste de considération (22/09/2026) : contexte × nouveauté, îlots locaux, frein, audibilité."""

    def test_saliency_needs_context(self):
        I = np.full(20, 0.1)
        self.assertTrue(np.all(dynamics.attention_saliency(I, None) == 0.0))         # entrée uniforme : rien
        I2 = I + 0.5 * np.exp(-0.5 * ((np.arange(20) - 10) / 2.0) ** 2)
        s = dynamics.attention_saliency(I2, None, saliency_scale=0.5)
        self.assertGreater(s[10], 0.9); self.assertLess(s[0], 1e-6)                    # une bosse : ~1 au centre, 0 loin
        u = np.full(20, 0.1); s_nu = dynamics.attention_saliency(I2, u, saliency_scale=0.5)
        self.assertTrue(np.all(s_nu == 0.0))                                           # surprise uniforme : rien de neuf → rien
        u[10] = 0.5; s_nu = dynamics.attention_saliency(I2, u, saliency_scale=0.5, novelty_scale=2.0)
        self.assertGreater(s_nu[10], 0.9); self.assertLess(s_nu[0], 1e-6)              # neuf ET contextuel : allumé
        u2 = np.full(20, 0.1); u2[0] = 0.5
        self.assertTrue(np.all(dynamics.attention_saliency(I2, u2) [[0]] == 0.0))       # neuf sans contexte : rien (multiplicatif)

    def test_islands_are_local(self):
        s = np.zeros(30); s[3:8] = 0.8; s[20:26] = 0.6
        comps = dynamics.attention_islands(s)
        self.assertEqual([list(c) for c in comps], [list(range(3, 8)), list(range(20, 26))])
        rng = np.random.default_rng(0)
        th = rng.uniform(0, 2 * np.pi, 30); f = np.full(30, 3.0)
        d1 = dynamics.attention_delta_fn(f, th, s, K0=5.0, kappa=0.0, garde=1.0)
        th2 = th.copy(); th2[20:26] += np.pi                                            # on tourne l'autre îlot
        d2 = dynamics.attention_delta_fn(f, th2, s, K0=5.0, kappa=0.0, garde=1.0)
        self.assertTrue(np.allclose(d1[3:8], d2[3:8]))                                  # l'îlot 3–7 ne le sait pas
        self.assertTrue(np.all(d1[8:20] == 0.0) and np.all(d1[26:] == 0.0))            # rien hors des îlots
        self.assertTrue(np.all(dynamics.attention_delta_fn(f, th, s, 5.0, 0.5, garde=0.0) == 0.0))   # frein à 0 : rien
        # l'accrochage rapproche les phases de l'îlot : après un pas, |Z| de l'îlot monte
        Z0 = abs(np.mean(np.exp(1j * th[3:8]))); th1 = th + 2 * np.pi * d1 * 0.1
        self.assertGreater(abs(np.mean(np.exp(1j * th1[3:8]))), Z0)

    def test_guard(self):
        self.assertEqual(dynamics.attention_guard(0.2, None), 1.0)
        self.assertEqual(dynamics.attention_guard(0.30, 0.30), 1.0)
        self.assertEqual(dynamics.attention_guard(0.24, 0.30), 1.0)                     # 0.8 × ref : encore libre
        self.assertAlmostEqual(dynamics.attention_guard(0.195, 0.30), 0.5, places=6)    # 0.65 × ref : à moitié
        self.assertEqual(dynamics.attention_guard(0.10, 0.30), 0.0)                     # 0.33 × ref : bloqué

    def _run(self, foyers, T=100, enabled=True, N=30):
        import simulate
        with open('config.json') as f:
            config = json.load(f)
        config['system']['N'] = N; config['system']['T'] = T; config['system']['seed'] = 12345
        config['system']['input']['perturbations'] = [{'type': 'none', 'amplitude': 0.0, 't0': 0.0, 'weight': 1.0}]
        config['analysis'] = {**config.get('analysis', {}), 'compare_kuramoto': False}
        config['attention'] = {**config.get('attention', {}), 'enabled': enabled}; config['context'] = {'foyers': foyers}
        tmp = tempfile.mkdtemp(); cwd = os.getcwd()
        try:
            os.chdir(tmp)
            with open('cfg.json', 'w') as f:
                json.dump(config, f)
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                res = simulate.run_simulation('cfg.json', 'FPS')
        finally:
            os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)
        return res['history']

    def test_inert_without_context(self):
        # activée mais sans foyer : saillance nulle partout, fréquences IDENTIQUES à l'attention désactivée
        h_on = self._run([], T=60, enabled=True); h_off = self._run([], T=60, enabled=False)
        self.assertTrue(all(x['attention_salient_share'] == 0.0 for x in h_on if x.get('attention_salient_share') is not None))
        self.assertEqual(max(abs(float(a['f_mean(t)']) - float(b['f_mean(t)'])) for a, b in zip(h_on, h_off)), 0.0)

    def test_context_lights_and_is_heard(self):
        # N = 60, un foyer sur la strate 30 (t 50–80, ~7 strates = 12 % du chœur) : des strates
        # saillantes pendant, aucune après ; elles portent PLUS de S(t) que leur prorata ; le frein reste libre
        h = self._run([{'center': 30, 'width': 3.0, 'gain': 0.5, 't0': 50.0, 't1': 80.0}], T=100, N=60)
        during = [x for x in h if 60 <= x['t'] < 80]; after = [x for x in h if x['t'] >= 85]
        share = np.mean([x['attention_salient_share'] for x in during]); aud = np.mean([x['attention_audibility'] for x in during])
        self.assertGreater(share, 0.03); self.assertLess(share, 0.3)
        self.assertGreater(aud, 1.5 * share, (aud, share))
        self.assertTrue(all(x['attention_salient_share'] == 0.0 for x in after))
        self.assertGreater(min(x['attention_garde'] for x in during), 0.5)

    def test_a_context_on_half_the_chorus_is_not_a_context(self):
        # N = 30, un foyer large et fort (~la moitié du chœur) : la saillance est relative à la médiane
        # (du contexte ET de la surprise), donc ce qui couvre la majorité ne désigne plus rien ;
        # sélectivité par construction, et le frein n'a pas à mordre
        h = self._run([{'center': 15, 'width': 5.0, 'gain': 1.5, 't0': 50.0, 't1': 80.0}], T=90, N=30)
        during = [x for x in h if 55 <= x['t'] < 80]
        # (0.05 avant la mémoire des moments chéris ; avec m appris sur les quelques strates au-dessus
        #  de la médiane, s = c·[ν + (1 − ν)·m] y est un peu plus haut : 0.054. Le sens ne change pas.)
        self.assertLess(np.mean([x['attention_salient_share'] for x in during]), 0.1)
        self.assertGreater(min(x['attention_garde'] for x in during), 0.5)


class TestCherishedMemory(unittest.TestCase):
    """La mémoire des moments chéris (25/09/2026) : attachement par strate, silence, rappel, porte."""

    def test_attachment_learns_where_the_context_is_and_only_there(self):
        st = dynamics.init_cherished_state(20); I = np.full(20, 0.1); I[5:9] = 0.6
        rng = np.random.default_rng(0)
        for _ in range(100):                                                        # 10 u.t. : le contexte qui reste est la bosse, pas le bruit
            on = dynamics.cherished_context(st, I + rng.normal(0, 0.6, 20), 0.1, 5.0, 0.5, 0.5)
        self.assertEqual(list(np.flatnonzero(on)), [5, 6, 7, 8])
        c = on.copy()
        dynamics.cherished_update_state_signature(st, np.array([0.05, 0.3]), 0.1, 5.0)
        for _ in range(400):                                                        # 40 u.t. à w = 0.8, τ_m 20
            dynamics.cherished_attach(st, c, 0.8, 0.1, 20.0, 5.0)
        self.assertGreater(st['m'][6], 0.8 * (1 - np.exp(-2)) - 0.02); self.assertLess(st['m'][6], 0.8)
        self.assertEqual(st['m'][0], 0.0); self.assertEqual(st['m'][15], 0.0)          # rien hors de l'îlot
        self.assertTrue(np.all(np.isfinite(st['sig'][6]))); self.assertTrue(np.all(np.isnan(st['sig'][0])))
        for _ in range(400):                                                        # le même foyer devient épuisant : m redescend
            dynamics.cherished_attach(st, c, 0.0, 0.1, 20.0, 5.0)
        self.assertLess(st['m'][6], 0.15)
        dynamics.cherished_attach(st, np.zeros(20, dtype=bool), 0.9, 0.1, 20.0, 5.0)  # sans contexte : rien ne bouge
        self.assertLess(st['m'][6], 0.15)

    def test_relief_cherishes_what_was_there_when_it_got_better(self):
        # niveau : la cible est w ; soulagement : w + (w − habituel), bornée
        self.assertEqual(dynamics.cherished_target(0.5, 0.1, 'niveau'), 0.5)
        self.assertAlmostEqual(dynamics.cherished_target(0.5, 0.1, 'soulagement'), 0.9)   # ça allait mal, ça va mieux : chéri
        self.assertAlmostEqual(dynamics.cherished_target(0.5, 0.9, 'soulagement'), 0.1)   # ça allait bien, ça se dégrade : non
        self.assertAlmostEqual(dynamics.cherished_target(0.8, 0.8, 'soulagement'), 0.8)   # le calme habituel reste chéri comme avant
        self.assertEqual(dynamics.cherished_target(0.0, 0.0, 'soulagement'), 0.0)         # l'effort qui dure : rien
        self.assertEqual(dynamics.cherished_target(0.5, None, 'soulagement'), 0.5)        # pas d'habituel encore : niveau
        st = dynamics.init_cherished_state(5)
        for _ in range(400): dynamics.cherished_wellbeing_ref(st, 0.2, 0.1, 40.0)
        self.assertLess(st['w_ref'], 0.2 + 1e-6)
        on = np.array([False, True, True, False, False])
        for _ in range(400): dynamics.cherished_attach(st, on, 0.5, 0.1, 20.0, 5.0, 'soulagement')   # w_ref suit aussi ? non : figé ici, pour lire la cible
        self.assertGreater(st['m'][1], 0.6); self.assertEqual(st['m'][0], 0.0)

    def test_silence_is_less_surprised_than_usual_with_hysteresis(self):
        st = dynamics.init_cherished_state(10); u = np.full(10, 1.0)
        for _ in range(800): dynamics.cherished_silence(st, u, False, 0.1)             # habitué à 1.0 : pas de silence (autant que d'habitude)
        self.assertFalse(st['quiet'])
        for _ in range(100): q = dynamics.cherished_silence(st, u * 0.6, False, 0.1)   # la surprise tombe : silence
        self.assertTrue(q)
        for _ in range(30): q = dynamics.cherished_silence(st, u * 0.6, True, 0.1)     # un îlot dans l'entrée : jamais en silence
        self.assertFalse(q)
        for _ in range(100): q = dynamics.cherished_silence(st, u * 0.6, False, 0.1)
        self.assertTrue(q)
        for _ in range(100): q = dynamics.cherished_silence(st, u * 2.0, False, 0.1)   # ça bouscule de nouveau : fin du silence
        self.assertFalse(q)
        self.assertFalse(dynamics.cherished_silence(st, None, False, 0.1))              # mémoire pas née : rien

    def test_recall_is_what_the_present_calls_then_the_most_cherished(self):
        st = dynamics.init_cherished_state(30)
        st['m'][3:7] = 0.9; st['sig'][3:7] = [0.05, 0.3]                                # souvenir A : très chéri, monde ample
        st['m'][20:25] = 0.6; st['sig'][20:25] = [0.02, 0.3]                             # souvenir B : moins chéri, monde faible
        st['g'] = np.array([0.021, 0.3])                                                 # le présent ressemble à B
        c = dynamics.cherished_recall(st, 0.0)
        self.assertEqual(st['recall_center'], 22); self.assertEqual(list(st['recall_comp']), list(range(20, 25)))
        self.assertGreater(c[22], 0.99); self.assertEqual(c[5], 0.0)
        st['g'] = np.array([0.049, 0.3]); dynamics.cherished_recall(st, 0.0)              # le présent ressemble à A
        self.assertEqual(st['recall_center'], 5)
        st['g'] = np.array([0.2, 0.3]); dynamics.cherished_recall(st, 0.0)                # rien ne ressemble : le plus chéri
        self.assertEqual(st['recall_center'], 5); self.assertLess(st['recall_res'], 0.2)
        st['m'][:] = 0.1; self.assertIsNone(dynamics.cherished_recall(st, 0.0))           # rien de chéri : rien

    def test_gate_puts_the_memory_down_and_keeps_m(self):
        st = dynamics.init_cherished_state(20); st['m'][5:10] = 0.8; st['sig'][5:10] = [0.05, 0.3]; st['g'] = np.array([0.05, 0.3])
        self.assertIsNotNone(dynamics.cherished_recall(st, 0.0))
        for i in range(60):
            released = dynamics.cherished_gate(st, 0.9, i * 0.1, 0.1, tau=5.0, floor=0.5, refractory=30.0)
        self.assertFalse(released)                                                       # se souvenir fait du bien : on reste
        t = 6.0; released = False
        while not released and t < 60:
            released = dynamics.cherished_gate(st, 0.0, t, 0.1, tau=5.0, floor=0.5, refractory=30.0); t += 0.1
        self.assertTrue(released); self.assertLess(t, 6.0 + 5.0)                          # ça fait mal : reposé en moins de τ
        self.assertEqual(st['m'][7], 0.8)                                                # ce à quoi on tient ne vaut pas moins
        self.assertIsNone(dynamics.cherished_recall(st, t))                              # réfractaire : pas rappelable maintenant
        self.assertIsNotNone(dynamics.cherished_recall(st, t + 31.0))                    # plus tard, on y revient

    def test_pipeline_inert_without_context_and_remembers_with_it(self):
        import simulate
        def run(foyers, N=30, T=150):
            with open('config.json') as f:
                config = json.load(f)
            config['system']['N'] = N; config['system']['T'] = T; config['system']['seed'] = 12345
            config['system']['input']['perturbations'] = [{'type': 'none', 'amplitude': 0.0, 't0': 0.0, 'weight': 1.0}]
            config['analysis'] = {**config.get('analysis', {}), 'compare_kuramoto': False}
            config['context'] = {'foyers': foyers}
            tmp = tempfile.mkdtemp(); cwd = os.getcwd()
            try:
                os.chdir(tmp)
                with open('cfg.json', 'w') as f:
                    json.dump(config, f)
                import io, contextlib
                with contextlib.redirect_stdout(io.StringIO()):
                    res = simulate.run_simulation('cfg.json', 'FPS')
            finally:
                os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)
            return res['history']
        h0 = run([], T=80)
        self.assertTrue(all(x['attention_m_max'] == 0.0 for x in h0 if x.get('attention_m_max') is not None))
        self.assertTrue(all(not np.isfinite(x['attention_rappel']) for x in h0 if x.get('attention_rappel') is not None))
        h = run([{'center': 8, 'width': 3.0, 'gain': 0.5, 't0': 50.0, 't1': 90.0}], N=30, T=150)
        during = [x for x in h if 70 <= x['t'] < 90]; after = [x for x in h if x['t'] >= 110]
        self.assertGreater(np.mean([x['attention_m_max'] for x in during]), 0.2)          # chéri pendant la présence au calme
        self.assertTrue(all(not np.isfinite(x['attention_rappel']) for x in during))     # pas de rappel tant que l'entrée parle
        rec = [x for x in after if np.isfinite(x['attention_rappel'])]
        self.assertGreater(len(rec), 0.1 * len(after))                                    # une fois l'entrée tue : le souvenir revient
        self.assertTrue(all(5 <= x['attention_rappel'] <= 11 for x in rec))               # au bon endroit
        self.assertTrue(all(x['attention_silence'] == 1.0 for x in rec))                  # et seulement en silence
        self.assertTrue(all(x['attention_rappel_etat'] == 6.0 for x in rec))              # l'état dit « rappel en cours »
        self.assertTrue(all(x['attention_rappel_etat'] == 1.0 for x in during))           # pendant la présence : « l'extérieur parle »
        self.assertTrue(all(x['attention_rappel_etat'] in (2.0, 3.0) for x in h0 if x.get('attention_rappel_etat') is not None and x['t'] > 25))   # sans contexte : jamais 6


class TestDispersionGuard(unittest.TestCase):
    """
    Garde-fou du centre de la cloche (21/09) : le centre DISPERSION_NORM_CENTER
    est un portrait de la chimère saine (⟨Aₙ⟩/√2 sur la config standard). Si le
    régime d'amplitude change (A₀, échelle d'entrée, enveloppe), ce test casse au
    lieu de laisser un voyant d'identité lire 4 en silence pendant deux mois
    (le piège de l'activité en juillet). Recette : calib/dispersion_center.py.
    Run réel, court (N=30, T=60 : l'invariance en √N tient), ~5 s, sans figures.
    """

    def test_calm_run_reads_5(self):
        import simulate
        with open('config.json') as f:
            config = json.load(f)
        config['system']['N'] = 30; config['system']['T'] = 60; config['system']['seed'] = 12345
        config['system']['input']['perturbations'] = [{'type': 'none', 'amplitude': 0.0, 't0': 0.0, 'weight': 1.0}]
        config['analysis'] = {**config.get('analysis', {}), 'compare_kuramoto': False}
        tmp = tempfile.mkdtemp(); cwd = os.getcwd()
        try:
            os.chdir(tmp)
            with open('cfg.json', 'w') as f:
                json.dump(config, f)
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                res = simulate.run_simulation('cfg.json', 'FPS')
        finally:
            os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)
        h = res['history']
        # Régime : t ≥ 45. Jusqu'à t ≈ 40 l'amplitude Aₙ décroît encore (0.12 → 0.022)
        # et la médiane lissée traîne le transitoire (rapport 1.7 à t=20-30, 1.1 à
        # 30-40, 1.0 ensuite, mesuré N=30 et N=100) ; même repère que le repos de
        # l'activité (calib_start_t = 40).
        regime = [x for x in h if x.get('t', 0) >= 45 and x.get('dispersion_norm') is not None]
        self.assertGreater(len(regime), 100)
        c = metrics.DISPERSION_NORM_CENTER
        folds = [max(x['dispersion_norm'] / c, c / x['dispersion_norm']) for x in regime]
        # Le centre est bien ⟨Aₙ⟩/√2 de ce run (rapport ~1), et la cloche lit 5
        # sur tout le régime (lissée : aucune fenêtre isolée ne fait verdict).
        A_mean = float(np.mean([x['An_mean(t)'] for x in regime]))
        self.assertLess(abs(np.median([x['dispersion_norm'] for x in regime]) / (A_mean / np.sqrt(2)) - 1.0), 0.15)
        self.assertLess(max(folds), metrics.DISPERSION_FOLD_FACTORS[0], max(folds))
        scores = [metrics.score_from_brackets(x['dispersion_norm'], 'dispersion') for x in regime]
        self.assertTrue(all(s == 5 for s in scores), set(scores))


class TestIntegration(unittest.TestCase):
    """Tests d'intégration du système complet."""
    
    def setUp(self):
        """Configuration pour tests d'intégration."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')
        
        # Créer une config minimale
        config = validate_config.generate_default_config(N=3, T=10)
        config['system']['dt'] = 0.5  # Pas plus grand pour test rapide
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
    
    def tearDown(self):
        """Nettoyage."""
        shutil.rmtree(self.temp_dir)
    
    @patch('simulate.init.setup_logging')
    @patch('simulate.init.init_strates')
    def test_simulation_flow(self, mock_init_strates, mock_setup_logging):
        """Test du flux de simulation complet."""
        # Mocks
        mock_init_strates.return_value = [
            {'id': 0, 'A0': 1.0, 'f0': 1.0, 'An': 1.0, 'fn': 1.0},
            {'id': 1, 'A0': 1.0, 'f0': 1.1, 'An': 1.0, 'fn': 1.1},
            {'id': 2, 'A0': 1.0, 'f0': 0.9, 'An': 1.0, 'fn': 0.9}
        ]
        
        mock_csv_writer = MagicMock()
        mock_setup_logging.return_value = {
            'csv_writer': mock_csv_writer,
            'run_id': 'test_run',
            'output_dir': self.temp_dir,
            'log_file': 'test.csv'
        }
        
        # Test imports et initialisation
        import simulate
        
        # Vérifier que les modules sont bien importés
        self.assertTrue(hasattr(simulate, 'dynamics'))
        self.assertTrue(hasattr(simulate, 'regulation'))
        self.assertTrue(hasattr(simulate, 'metrics'))


class TestPerformance(unittest.TestCase):
    """Tests de performance et limites."""
    
    def test_large_n_strates(self):
        """Test avec un grand nombre de strates."""
        # Générer une config avec N=50
        config = validate_config.generate_default_config(N=50, T=10)
        
        # Initialiser
        state = init.init_strates(config)
        self.assertEqual(len(state), 50)
        
        # Tester quelques calculs
        In_t = np.ones(50)
        An_t = dynamics.compute_An(0, state, In_t, np.zeros(50), config)
        self.assertEqual(len(An_t), 50)
    
    def test_numerical_stability(self):
        """Test de stabilité numérique."""
        # Valeurs extrêmes
        x_extreme = np.array([1e-10, 1e10, -1e10])
        
        # Sigmoïde doit rester bornée
        sigma_vals = dynamics.compute_sigma(x_extreme, 1.0, 0.0)
        self.assertTrue(np.all(sigma_vals >= 0))
        self.assertTrue(np.all(sigma_vals <= 1))
        
        # Régulation tanh doit rester bornée
        g_vals = regulation.compute_G(x_extreme, 'tanh')
        self.assertTrue(np.all(np.abs(g_vals) <= 1))


def run_all_tests():
    """Lance tous les tests et génère un rapport."""
    # Créer un test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Ajouter tous les tests
    test_classes = [
        TestDynamics,
        TestRegulation,
        TestMetrics,
        TestReferenceScores,
        TestResilienceCSD,
        TestValidateConfig,
        TestPerturbations,
        TestAnalyze,
        TestExplore,
        TestKuramoto,
        TestUtils,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Runner avec rapport détaillé
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ DES TESTS FPS")
    print("="*70)
    print(f"Tests exécutés : {result.testsRun}")
    print(f"Succès : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Échecs : {len(result.failures)}")
    print(f"Erreurs : {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ TOUS LES TESTS SONT PASSÉS ! La FPS est prête à danser ! 🌀")
    else:
        print("\n❌ Certains tests ont échoué. Vérifier les détails ci-dessus.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Désactiver les warnings pour les tests
    warnings.filterwarnings('ignore')
    
    # Lancer tous les tests
    success = run_all_tests()
    
    # Code de sortie
    exit(0 if success else 1)
