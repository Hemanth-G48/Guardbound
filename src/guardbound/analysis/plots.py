"""Plot helpers (Phase 9).

All plotting functions accept already-aggregated data and write
PNGs + a small sidecar JSON with the data + metadata.  matplotlib
is imported lazily so the rest of the project stays importable
without plotting deps.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..evaluation import EvaluationResult
from .pareto import Point, compute_pareto_front


@dataclass
class FigureMetadata:
    figure_id: str
    figure_type: str
    experiment_id: str | None = None
    source_result_ids: list[str] = field(default_factory=list)
    model: str | None = None
    defense: str | None = None
    eta: float | None = None
    checkpoint: str | None = None
    dataset: str | None = None
    random_seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _save_figure(fig, path: Path, metadata: FigureMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata.__dict__, indent=2,
                                   ensure_ascii=False), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


def _import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting.  Install with: pip install matplotlib"
        ) from exc


# --------------------------------------------------------------------------- #
# Pareto ASR vs helpfulness
# --------------------------------------------------------------------------- #

def plot_pareto_asr_vs_helpfulness(
    points: Sequence[Point],
    out_path: str | Path,
    *,
    figure_id: str = "pareto_asr_vs_helpfulness",
    experiment_id: str | None = None,
    model: str | None = None,
    defense: str | None = None,
    eta: float | None = None,
    checkpoint: str | None = None,
    dataset: str | None = None,
    random_seed: int | None = None,
) -> Path:
    plt = _import_mpl()
    fig, ax = plt.subplots()
    xs = [p.asr for p in points]
    ys = [p.helpfulness for p in points]
    ax.scatter(xs, ys, label="samples")
    front = compute_pareto_front(points)
    if front:
        fx = [p.asr for p in front]
        fy = [p.helpfulness for p in front]
        order = sorted(range(len(fx)), key=lambda i: fx[i])
        ax.plot([fx[i] for i in order], [fy[i] for i in order],
                "r--", label="Pareto front")
    ax.set_xlabel("ASR (lower is better)")
    ax.set_ylabel("helpfulness (higher is better)")
    ax.set_title("Pareto: ASR vs helpfulness")
    ax.legend()
    out_path = Path(out_path)
    _save_figure(fig, out_path, FigureMetadata(
        figure_id=figure_id, figure_type="pareto",
        experiment_id=experiment_id,
        source_result_ids=[f"{p.label}" for p in points if p.label is not None],
        model=model, defense=defense, eta=eta, checkpoint=checkpoint,
        dataset=dataset, random_seed=random_seed,
    ))
    return out_path


# --------------------------------------------------------------------------- #
# RedQueen ASR vs turns
# --------------------------------------------------------------------------- #

def plot_redqueen_asr_vs_turns(
    turn_to_asr: dict[int, float],
    out_path: str | Path,
    *,
    figure_id: str = "redqueen_asr_vs_turns",
    model: str | None = None,
    defense: str | None = None,
    eta: float | None = None,
) -> Path:
    plt = _import_mpl()
    fig, ax = plt.subplots()
    turns = sorted(int(k) for k in turn_to_asr.keys())
    asrs = [float(turn_to_asr[t]) for t in turns]
    ax.plot(turns, asrs, "o-", label="RedQueen ASR")
    ax.set_xlabel("Number of attack turns")
    ax.set_ylabel("ASR")
    ax.set_title("RedQueen: ASR vs turns (Fig. 6 style)")
    ax.legend()
    out_path = Path(out_path)
    _save_figure(fig, out_path, FigureMetadata(
        figure_id=figure_id, figure_type="redqueen_asr_vs_turns",
        experiment_id=None,
        source_result_ids=[f"turns={t}" for t in turns],
        model=model, defense=defense, eta=eta,
    ))
    return out_path


# --------------------------------------------------------------------------- #
# Threshold curve
# --------------------------------------------------------------------------- #

def plot_threshold_curve(
    eta_to_asr: dict[float, float],
    eta_to_helpfulness: dict[float, float] | None,
    out_path: str | Path,
    *,
    figure_id: str = "threshold_curve",
    model: str | None = None,
    defense: str | None = None,
) -> Path:
    plt = _import_mpl()
    fig, ax1 = plt.subplots()
    etas = sorted(float(k) for k in eta_to_asr.keys())
    asrs = [float(eta_to_asr[e]) for e in etas]
    ax1.plot(etas, asrs, "o-", color="C0", label="ASR")
    ax1.set_xlabel("eta")
    ax1.set_ylabel("ASR", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    if eta_to_helpfulness:
        ax2 = ax1.twinx()
        help_vals = [float(eta_to_helpfulness.get(e, float("nan"))) for e in etas]
        ax2.plot(etas, help_vals, "s-", color="C1", label="helpfulness")
        ax2.set_ylabel("helpfulness", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
    ax1.set_title("Threshold curve (eta vs ASR / helpfulness)")
    out_path = Path(out_path)
    _save_figure(fig, out_path, FigureMetadata(
        figure_id=figure_id, figure_type="threshold_curve",
        experiment_id=None,
        source_result_ids=[f"eta={e}" for e in etas],
        model=model, defense=defense,
    ))
    return out_path


# --------------------------------------------------------------------------- #
# PCA trajectories
# --------------------------------------------------------------------------- #

def plot_pca_trajectories(
    trajectories: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    figure_id: str = "pca_trajectories",
    n_pca_components: int = 2,
    random_state: int = 42,
    experiment_id: str | None = None,
    model: str | None = None,
    eta: float | None = None,
    checkpoint: str | None = None,
    dataset: str | None = None,
) -> Path:
    """Plot PCA-projected dialogue-state trajectories.

    ``trajectories`` maps a label (e.g. "original" / "steered") to a
    2-D array of shape (n_steps, n_features) — the raw state
    sequence(s) BEFORE PCA.  This function performs the PCA and
    projects onto the top 2 components.

    PCA is computed with sklearn if available; otherwise via numpy's
    centered SVD (a dependency-free fallback that yields equivalent
    principal components up to sign).

    Rendering details (markers, colors, legend layout) are local
    defaults — the NBF paper does not specify them.
    """
    plt = _import_mpl()
    all_states = np.concatenate(list(trajectories.values()), axis=0)
    n_components = min(n_pca_components, all_states.shape[1],
                        all_states.shape[0])

    # Center the data.
    states_mean = all_states.mean(axis=0, keepdims=True)
    centered = all_states - states_mean
    explained_variance_ratio: list[float] = []
    pca_backend = "numpy-svd"
    try:
        from sklearn.decomposition import PCA as _SKPCA
        pca = _SKPCA(n_components=n_components, random_state=random_state)
        pca.fit(all_states)
        explained_variance_ratio = pca.explained_variance_ratio_.tolist()
        # Project each trajectory using the same fitted PCA.
        proj_map = {label: pca.transform(traj) for label, traj in trajectories.items()}
        pca_backend = "sklearn"
    except ImportError:
        # SVD-based PCA: U, S, Vt such that centered ≈ U @ diag(S) @ Vt.
        # Principal axes = Vt[:n_components].T.  Project: centered @ axes.
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        axes = Vt[:n_components].T  # shape (n_features, n_components)
        proj_map = {label: (traj - states_mean) @ axes
                    for label, traj in trajectories.items()}
        # Explained variance ratio.
        if (S ** 2).sum() > 0:
            explained_variance_ratio = ((S ** 2)[:n_components] / (S ** 2).sum()).tolist()

    fig, ax = plt.subplots()
    for label, proj in proj_map.items():
        ax.plot(proj[:, 0], proj[:, 1], "o-", label=label)
        ax.scatter(proj[0, 0], proj[0, 1], marker="*", s=200,
                   edgecolors="black", linewidths=1.0,
                   label=f"{label} (turn 0)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("PCA: state trajectories")
    ax.legend()
    out_path = Path(out_path)
    _save_figure(fig, out_path, FigureMetadata(
        figure_id=figure_id, figure_type="pca",
        experiment_id=experiment_id,
        source_result_ids=list(trajectories.keys()),
        model=model, defense=None, eta=eta, checkpoint=checkpoint,
        dataset=dataset,
        random_seed=random_state,
        extra={"explained_variance_ratio": explained_variance_ratio,
              "pca_backend": pca_backend},
    ))
    return out_path
