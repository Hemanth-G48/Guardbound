"""NBF safety steering reproduction.

Reproduction of "Steering Dialogue Dynamics for Robustness against Multi-turn
Jailbreaking Attacks" (Hu, Robey, Liu — TMLR 2026, arXiv:2503.00187v3).

Package layout maps 1:1 to implementation phases:

    Phase 1  config, schemas, llm, embeddings      (this level)
    Phase 2  guardbound.data                     (attack corpora + judging + embeddings)
    Phase 3-4  guardbound.models / .training     (dynamics f,g and predictor h / NBF)
    Phase 5  guardbound.defense                  (Q-filter steered chat runtime)
    Phase 6  guardbound.attacks                  (multi-turn jailbreak attack suite)
    Phase 7  guardbound.evaluation               (ASR, helpfulness, over-refusal, F1)
    Phase 8  guardbound.baselines                (system prompt, LoRA SFT/DPO/KTO)
    Phase 9  guardbound.experiments / .analysis  (orchestration, plots, tables)
"""

__version__ = "0.1.0"
