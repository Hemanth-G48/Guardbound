# Contribution Statement

**Paper:** Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks (Hu, Robey, Liu — TMLR 02/2026, arXiv:2503.00187v3)

**Paper type:** Defense / safety framework for LLMs (learning-based safe control + lightweight guardrail).

**Single core contribution:** A control-theoretic safety steering framework that (1) models multi-turn dialogue as neural state-space dynamics in sentence-embedding space, and (2) trains a Neural Barrier Function (NBF) safety predictor that is used online as a Q-filter to proactively filter harmful queries before they reach the LLM, achieving invariant safety against multi-turn jailbreaks.

**Supporting contributions:** Invariant-safety losses (safe-set loss L_SS, safety-invariance loss L_SI) derived from a corollary of an invariant-safety certificate theorem; empirical demonstration across many LLMs and three attack methods with better safety/helpfulness/over-refusal trade-off than alignment baselines; adaptive-attack analysis.

**Implementation surface:** data pipeline (attack conversation generation + GPT-4o judging + embedding), two small MLP training stages (dynamics f_θ,g_θ then predictor h), runtime Q-filter defense wrapper around arbitrary chat LLMs, attack harnesses (ActorAttack, Crescendo, Opposite-day, Acronym, RedQueen, adaptive variant), evaluation suite (ASR, MMLU post-filtering, MTBench, over-refusal benchmarks, guardrail F1 comparison), baselines (system prompt, LoRA SFT/DPO/KTO).
