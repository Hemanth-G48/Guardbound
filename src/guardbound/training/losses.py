"""Dynamics training loss: masked MSE for temporal rollout.

Paper Eq. (4):

    L_dyn = (1/N) * SUM_i SUM_k ||z_k^(i) - g_theta(x_k^(i), u_k^(i))||^2

where x_k = f_theta(x_{k-1}, u_k) and x_0 = 0.

The loss is masked so that padded turns do not contribute.
"""
from __future__ import annotations

import torch


def dynamics_loss(
    Z_hat: torch.Tensor,
    Z: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked mean squared error for dynamics prediction.

    Parameters
    ----------
    Z_hat : [B, K, D]
        Predicted response embeddings from g_theta rollout.
    Z : [B, K, D]
        Ground-truth response embeddings.
    mask : [B, K] or None
        Boolean mask: True where turn exists, False for padding.

    Returns
    -------
    Scalar loss tensor.
    """
    if Z_hat.shape != Z.shape:
        raise ValueError(
            f"Shape mismatch: Z_hat {Z_hat.shape} != Z {Z.shape}"
        )

    # Per-element squared error: [B, K, D]
    squared_error = (Z_hat - Z) ** 2

    if mask is not None:
        # Expand mask to [B, K, 1] for broadcasting over embedding dim
        mask_expanded = mask.float().unsqueeze(-1)  # [B, K, 1]

        # Zero out masked positions
        masked_error = squared_error * mask_expanded

        # Count valid elements
        num_valid = mask.float().sum().clamp(min=1.0)
        num_elements = num_valid * Z.shape[-1]  # total valid float elements

        # Mean over valid elements only
        loss = masked_error.sum() / num_elements
    else:
        # No mask: simple mean over all elements
        loss = squared_error.mean()

    return loss
