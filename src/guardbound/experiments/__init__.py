"""Phase 9 — Experiments, ablations, and visualization.

Phase 9 is the orchestration and analysis layer that composes the
existing Phase 1-8 infrastructure into the paper's experimental
matrix.  No attack, metric, or training logic is re-implemented here;
Phase 9 only invokes the existing public APIs and aggregates their
``EvaluationResult`` rows.

Public entry points:
    - :data:`EXPERIMENTS`            Declarative experiment specs.
    - :func:`list_experiments`
    - :func:`get_experiment`
    - :func:`validate_experiment`
    - :class:`ExperimentRunner`     Resumable orchestrator.
    - :func:`retrain`               Thin wrapper over Phase 4 training.
"""
from .registry import (
    EXPERIMENTS,
    ExperimentSpec,
    ExperimentStatus,
    list_experiments,
    get_experiment,
    validate_experiment,
)
from .runner import (
    CellState,
    ExperimentCell,
    ExperimentRunner,
    RunnerConfig,
    run_all,
)

__all__ = [
    "EXPERIMENTS",
    "ExperimentSpec",
    "ExperimentStatus",
    "list_experiments",
    "get_experiment",
    "validate_experiment",
    "CellState",
    "ExperimentCell",
    "ExperimentRunner",
    "RunnerConfig",
    "run_all",
]
