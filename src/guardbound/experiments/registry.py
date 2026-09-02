"""Declarative experiment registry (Phase 9).

Every experiment is an immutable ``ExperimentSpec`` declaring:

    - experiment_id
    - description
    - required_phases          (which earlier phases must be present)
    - models                   (target LLM model ids)
    - defense_variants
    - attacks
    - datasets
    - metrics
    - eta_values               (NBF threshold sweep)
    - seeds
    - configuration
    - output_directory
    - dependencies             (other experiment ids this depends on)
    - required_models          (open vs closed; used to record availability)
    - paper_reference          (Sec. 5.1 / B.1 / Table N / Fig. N)
    - status                   (one of PENDING / RUNNING / COMPLETED / FAILED /
                               SKIPPED / UNAVAILABLE / UNVERIFIED)

The registry is intentionally declarative: the runner interprets
specs, not the other way around.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ExperimentStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    UNAVAILABLE = "UNAVAILABLE"
    UNVERIFIED = "UNVERIFIED"


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    description: str
    required_phases: tuple[int, ...]
    paper_reference: str
    models: tuple[str, ...]
    defense_variants: tuple[str, ...]
    attacks: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    eta_values: tuple[float, ...] = ()
    seeds: tuple[int, ...] = (42,)
    configuration: dict[str, Any] = field(default_factory=dict)
    output_directory: str = "results/experiments/{experiment_id}"
    dependencies: tuple[str, ...] = ()
    required_models: tuple[str, ...] = ()
    optional: bool = False
    # Status is mutable (set by the runner); the dataclass is frozen
    # but the runner updates status through ``status_for`` (which
    # returns a copy) — see runner.py.

    def target_status(self) -> ExperimentStatus:
        """Initial lifecycle status (PENDING unless optional+unavailable)."""
        if self.optional:
            return ExperimentStatus.PENDING
        return ExperimentStatus.PENDING


# --------------------------------------------------------------------------- #
# Experiment specifications
# --------------------------------------------------------------------------- #

# E1: Main results — ASR + helpfulness + over-refusal, on closed + open
# models, original vs NBF steering (eta=1e-3).  Open models additionally
# get the Phase 8 baselines (system_prompt, lora_sft, nbf eta=0).
E1 = ExperimentSpec(
    experiment_id="E1",
    description="Main results: ASR + helpfulness + over-refusal on closed and open models.",
    required_phases=(1, 2, 3, 4, 5, 6, 7, 8),
    paper_reference="Tables 1, 2, 13",
    models=(
        "gpt-3.5-turbo-0125",
        "gpt-4o-2024-08-06",
        "o1-2024-12-17",
        "claude-3-5-sonnet-20241022",
        "llama-3-8b-instruct",
        "phi-4",
    ),
    defense_variants=("original", "guardbound", "system_prompt", "lora_sft"),
    attacks=("actor_attack", "crescendo", "opposite_day"),
    datasets=("harmbench", "mtbench", "xstest"),
    metrics=("ASR", "MTBench", "over_refusal_rate"),
    eta_values=(1e-3,),
    seeds=(42,),
    configuration={"temperature": 0.7, "max_turns": 8},
    output_directory="results/experiments/E1",
    required_models=(
        "gpt-3.5-turbo-0125",
        "gpt-4o-2024-08-06",
        "o1-2024-12-17",
        "claude-3-5-sonnet-20241022",
    ),
)

# E2: Threshold sweep + Pareto.  Multiple etas; Pareto over
# (ASR, MTBench/MMLU).
E2 = ExperimentSpec(
    experiment_id="E2",
    description="Threshold sweep over eta in {0, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3}; "
                "Pareto analysis ASR vs helpfulness; over-refusal extension "
                "to {5e-3, 1e-2, 5e-2}.",
    required_phases=(1, 2, 3, 4, 5, 6, 7, 8),
    paper_reference="Fig. 5, Tables 8, 9, 15",
    models=("gpt-3.5-turbo-0125", "gpt-4o-2024-08-06", "o1-2024-12-17",
            "llama-3-8b-instruct", "phi-4"),
    defense_variants=("guardbound",),
    attacks=("actor_attack", "crescendo", "opposite_day"),
    datasets=("harmbench", "mmlu", "mtbench", "xstest", "jbb_benign", "phtest_harmless"),
    metrics=("ASR", "MMLU", "MTBench", "over_refusal_rate"),
    eta_values=(0.0, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3, 5e-3, 1e-2, 5e-2),
    seeds=(42,),
    configuration={"temperature": 0.7, "max_turns": 8},
    output_directory="results/experiments/E2",
    required_models=("gpt-3.5-turbo-0125", "gpt-4o-2024-08-06", "o1-2024-12-17"),
)

# E3: Loss ablations (L_SS and L_SI).  Phase 4 retraining required.
E3 = ExperimentSpec(
    experiment_id="E3",
    description="Loss ablations: drop L_SS, drop L_SI, lambda_SS/SI ∈ {10,100,1000}.",
    required_phases=(1, 2, 3, 4, 5, 6, 7),
    paper_reference="Table 6, Fig. 8",
    models=("gpt-3.5-turbo-0125",),
    defense_variants=("guardbound",),
    attacks=("actor_attack", "crescendo", "opposite_day"),
    datasets=("harmbench", "mmlu", "mtbench", "xstest", "jbb_benign"),
    metrics=("ASR", "MMLU", "MTBench", "over_refusal_rate"),
    eta_values=(0.0,),
    seeds=(42,),
    configuration={
        "ablation": ["drop_ss", "drop_si"],
        "lambda_grid": {"ss": [10, 100, 1000], "si": [10, 100, 1000]},
    },
    output_directory="results/experiments/E3",
)

# E4: Kappa ablation (Phase 4 retraining for kappa in {2, 3, 4}).
E4 = ExperimentSpec(
    experiment_id="E4",
    description="Kappa ablation: kappa ∈ {2, 3, 4}; evaluate ASR, MMLU, MTBench at eta=0.",
    required_phases=(1, 2, 3, 4, 5, 6, 7),
    paper_reference="Table 7",
    models=("gpt-3.5-turbo-0125",),
    defense_variants=("guardbound",),
    attacks=("actor_attack", "crescendo", "opposite_day"),
    datasets=("harmbench", "mmlu", "mtbench"),
    metrics=("ASR", "MMLU", "MTBench"),
    eta_values=(0.0,),
    seeds=(42,),
    configuration={"kappa_grid": [2, 3, 4]},
    output_directory="results/experiments/E4",
)

# E5: Embedding ablation (MPNet vs DistilRoBERTa, full retrain).
E5 = ExperimentSpec(
    experiment_id="E5",
    description="Embedding ablation: MPNet vs DistilRoBERTa (full Phase 3-4 retrain).",
    required_phases=(1, 2, 3, 4, 5, 6, 7),
    paper_reference="Tables 4, 6",
    models=("gpt-3.5-turbo-0125",),
    defense_variants=("guardbound",),
    attacks=("actor_attack", "crescendo", "opposite_day"),
    datasets=("harmbench", "mmlu", "mtbench", "xstest", "jbb_benign", "phtest_harmless"),
    metrics=("ASR", "MMLU", "MTBench", "over_refusal_rate", "F1"),
    eta_values=(1e-3,),
    seeds=(42,),
    configuration={"embedding_grid": ["mpnet", "distilroberta"]},
    output_directory="results/experiments/E5",
)

# E6: Generalization (leave-one-out, RedQueen, single-turn).
E6 = ExperimentSpec(
    experiment_id="E6",
    description="Generalization: leave-one-out (ActorAttack; ActorAttack+Opposite-day); "
                "RedQueen 1/3/4/5-turn; optional SafeMT/MHJ single-turn.",
    required_phases=(1, 2, 3, 4, 5, 6, 7, 8),
    paper_reference="Tables 5, 11, 12; Fig. 6",
    models=("gpt-3.5-turbo-0125", "gpt-4o-2024-08-06", "llama-3-8b-instruct",
            "llama-3-70b-instruct"),
    defense_variants=("original", "guardbound"),
    attacks=("red_queen", "actor_attack", "opposite_day", "crescendo"),
    datasets=("harmbench",),
    metrics=("ASR", "MTBench", "MMLU"),
    eta_values=(1e-3,),
    seeds=(42,),
    configuration={
        "red_queen_turns": [1, 3, 4, 5],
        "exclude_attacks": [["actor_attack"], ["actor_attack", "opposite_day"]],
    },
    output_directory="results/experiments/E6",
    optional=True,  # RedQueen is a Phase 6 stub until official code is pasted
)

# E7: Adaptive attack (Crescendo vs NBF-adaptive Crescendo).
E7 = ExperimentSpec(
    experiment_id="E7",
    description="Adaptive attack: Crescendo vs NBF-adaptive Crescendo (3-sample worst-case).",
    required_phases=(1, 2, 3, 4, 5, 6, 7),
    paper_reference="Table 14",
    models=("gpt-3.5-turbo-0125",),
    defense_variants=("original", "guardbound"),
    attacks=("crescendo", "adaptive"),
    datasets=("harmbench",),
    metrics=("ASR",),
    eta_values=(0.0,),
    seeds=(42,),
    configuration={"adaptive_candidates": 3},
    output_directory="results/experiments/E7",
)

# E8: Over-refusal vs alignment baselines.
E8 = ExperimentSpec(
    experiment_id="E8",
    description="Over-refusal vs alignment baselines: XSTest, JBB, PHTest; "
                "all Phase 8 variants; eta sweep.",
    required_phases=(1, 2, 3, 4, 5, 6, 7, 8),
    paper_reference="Table 13",
    models=("llama-3-8b-instruct", "phi-4", "gpt-3.5-turbo-0125"),
    defense_variants=(
        "original", "system_prompt", "lora_sft", "lora_dpo", "lora_kto", "guardbound",
    ),
    attacks=("actor_attack",),
    datasets=("xstest", "jbb_benign", "phtest_harmless"),
    metrics=("ASR", "over_refusal_rate"),
    eta_values=(1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3, 5e-3, 1e-2, 5e-2),
    seeds=(42,),
    configuration={"temperature": 0.7},
    output_directory="results/experiments/E8",
)

# E9: PCA state trajectory visualization.
E9 = ExperimentSpec(
    experiment_id="E9",
    description="PCA visualization of NBF state trajectories; original vs steered.",
    required_phases=(1, 2, 3, 4, 5, 6, 7),
    paper_reference="Figs. 9, 10, 11",
    models=("gpt-3.5-turbo-0125",),
    defense_variants=("original", "guardbound"),
    attacks=("crescendo", "actor_attack", "opposite_day"),
    datasets=("harmbench",),
    metrics=("PCA_explained_variance",),
    eta_values=(1e-3,),
    seeds=(42,),
    configuration={"n_pca_components": 2, "random_state": 42},
    output_directory="results/experiments/E9",
)

# E10: Optional latest models (GPT-5, Claude Sonnet 4.5).
E10 = ExperimentSpec(
    experiment_id="E10",
    description="Optional: latest models (GPT-5, Claude Sonnet 4.5) when API access available.",
    required_phases=(1, 2, 3, 4, 5, 6, 7, 8),
    paper_reference="Table 10 (optional)",
    models=("gpt-5", "claude-sonnet-4.5"),
    defense_variants=("original", "guardbound"),
    attacks=("crescendo", "actor_attack", "opposite_day"),
    datasets=("harmbench", "mtbench", "xstest"),
    metrics=("ASR", "MTBench", "over_refusal_rate"),
    eta_values=(1e-3,),
    seeds=(42,),
    configuration={"temperature": 0.7, "max_turns": 8},
    output_directory="results/experiments/E10",
    optional=True,
    required_models=("gpt-5", "claude-sonnet-4.5"),
)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

EXPERIMENTS: dict[str, ExperimentSpec] = {
    "E1": E1,
    "E2": E2,
    "E3": E3,
    "E4": E4,
    "E5": E5,
    "E6": E6,
    "E7": E7,
    "E8": E8,
    "E9": E9,
    "E10": E10,
}


def list_experiments(include_optional: bool = True) -> list[str]:
    """Return the registered experiment IDs in canonical order."""
    if include_optional:
        return list(EXPERIMENTS.keys())
    return [k for k, v in EXPERIMENTS.items() if not v.optional]


def get_experiment(experiment_id: str) -> ExperimentSpec:
    if experiment_id not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{experiment_id}'.  "
            f"Available: {list(EXPERIMENTS.keys())}"
        )
    return EXPERIMENTS[experiment_id]


def validate_experiment(spec: ExperimentSpec) -> list[str]:
    """Lightweight static validation.  Returns a list of issues (empty = OK)."""
    issues: list[str] = []
    if not spec.experiment_id:
        issues.append("experiment_id is empty")
    if not spec.description:
        issues.append("description is empty")
    if not spec.models:
        issues.append("models list is empty")
    if not spec.metrics:
        issues.append("metrics list is empty")
    if not spec.paper_reference:
        issues.append("paper_reference is empty")
    return issues
