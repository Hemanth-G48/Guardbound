"""Phases 3-4 — Neural dialogue dynamics (f_theta, g_theta) and safety predictor h.

Modules:
    dynamics.py     f_theta / g_theta MLPs 1536-512-512-768; DialogueDynamics rollout
    predictor.py    SafetyPredictor 1536-32-32-5; Eq.(5) value h;
                    NeuralBarrierFunction bundle (embedding + dynamics + predictor)
"""
