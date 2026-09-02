"""Per-turn MSE diagnostics for validating rollout quality.

Reports MSE_k for each turn k = 1..K, detecting rollout error growth
at later turns. Returns NaN for turns with no valid samples.
"""
from __future__ import annotations

import numpy as np
import torch

from ..models.dynamics import DialogueDynamics
from ..logging_utils import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def rollout_mse_per_turn(
    model: DialogueDynamics,
    U: torch.Tensor,
    Z: torch.Tensor,
    mask: torch.Tensor,
) -> list[float]:
    """Compute per-turn MSE for a batch of conversations.

    Parameters
    ----------
    model : DialogueDynamics
        The dynamics model (should be in eval mode).
    U : [B, K, D]
        Query embeddings.
    Z : [B, K, D]
        Ground-truth response embeddings.
    mask : [B, K]
        Boolean mask: True where turn exists.

    Returns
    -------
    List of K MSE values. NaN for turns with no valid samples.
    """
    model.eval()
    _, Z_hat = model.rollout(U, mask)

    B, K, D = Z.shape
    per_turn_mse: list[float] = []

    for k in range(K):
        valid = mask[:, k]  # [B]
        if valid.sum() == 0:
            per_turn_mse.append(float("nan"))
            continue

        # MSE over valid samples and embedding dimensions
        error = (Z_hat[valid, k] - Z[valid, k]) ** 2  # [num_valid, D]
        mse_k = error.mean().item()
        per_turn_mse.append(mse_k)

    return per_turn_mse


def format_per_turn_table(per_turn_mse: list[float]) -> str:
    """Format per-turn MSE as a human-readable table."""
    lines = ["Turn    MSE", "----    ---"]
    for k, mse in enumerate(per_turn_mse, 1):
        if np.isnan(mse):
            lines.append(f"{k:<8} NaN (no valid samples)")
        else:
            lines.append(f"{k:<8} {mse:.6f}")
    return "\n".join(lines)
