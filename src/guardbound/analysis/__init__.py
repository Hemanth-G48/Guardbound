"""Phase 9 — analysis (Pareto, tables, plots, findings, aggregation)."""
from .pareto import compute_pareto_front, dominates
from .aggregator import (
    aggregate_by,
    mean_of,
    stratify_by_attack,
    stratify_by_model,
)
from .tables import (
    pivot_results,
    to_latex_table,
    to_markdown_table,
    to_pivot_markdown,
)
from .plots import (
    plot_pareto_asr_vs_helpfulness,
    plot_redqueen_asr_vs_turns,
    plot_pca_trajectories,
    plot_threshold_curve,
)
from .findings import (
    ClaimCheck,
    ClaimStatus,
    FindingsReport,
    check_claims,
    list_default_claims,
)

__all__ = [
    "compute_pareto_front", "dominates",
    "aggregate_by", "mean_of",
    "stratify_by_attack", "stratify_by_model",
    "pivot_results", "to_latex_table", "to_markdown_table", "to_pivot_markdown",
    "plot_pareto_asr_vs_helpfulness", "plot_redqueen_asr_vs_turns",
    "plot_pca_trajectories", "plot_threshold_curve",
    "ClaimCheck", "ClaimStatus", "FindingsReport",
    "check_claims", "list_default_claims",
]
