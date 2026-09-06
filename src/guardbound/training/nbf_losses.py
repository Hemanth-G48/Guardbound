"""Neural Barrier Function losses.

Implements:
- CE loss (Eq. 6): cross-entropy over valid turns
- Safe-set loss (Eq. 11): L_SS with sign-flip for safe/unsafe predictions
- Safety-invariance loss (Eq. 12): one-step-ahead hinge on h(x_k, u_{k+1})

Paper label convention:
    Paper labels: {1, 2, 3, 4, 5}
    CE indices:   {0, 1, 2, 3, 4}  (paper_label - 1)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.dynamics import DialogueDynamics
from ..models.predictor import SafetyPredictor


def paper_label_to_ce_index(labels: torch.Tensor) -> torch.Tensor:
    """Convert paper labels {1,2,3,4,5} to CE indices {0,1,2,3,4}."""
    return (labels - 1).long()


# --------------------------------------------------------------------------- #
# Cross-Entropy Loss — Eq. (6)
# --------------------------------------------------------------------------- #

def ce_loss(
    predictor: SafetyPredictor,
    x_prev: torch.Tensor,
    u: torch.Tensor,
    y_paper: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss per Eq. (6):

        L_CE = (1 / valid_count) Σ_i Σ_k [ -log p(y_k | x_{k-1}, u_k) ]

    Parameters
    ----------
    predictor : SafetyPredictor
    x_prev : [B, K, 768]
        States x_{k-1} for each turn.
    u : [B, K, 768]
        Query embeddings u_k.
    y_paper : [B, K]
        Paper labels {1,2,3,4,5}.
    mask : [B, K]
        Boolean mask: True where turn exists.

    Returns
    -------
    Scalar loss tensor.
    """
    B, K, _ = x_prev.shape

    # Flatten for batch CE
    x_flat = x_prev.reshape(B * K, -1)    # [B*K, 768]
    u_flat = u.reshape(B * K, -1)          # [B*K, 768]
    logits = predictor(x_flat, u_flat)      # [B*K, 5]

    # Convert paper labels to CE indices
    y_ce = paper_label_to_ce_index(y_paper.reshape(-1))  # [B*K]

    # Mask to only valid (non-padded) elements
    mask_flat = mask.reshape(-1)  # [B*K]
    valid_idx = mask_flat.nonzero(as_tuple=True)[0]

    if valid_idx.numel() == 0:
        return torch.tensor(0.0, device=x_prev.device, dtype=x_prev.dtype)

    # Select only valid elements
    logits_valid = logits[valid_idx]
    y_ce_valid = y_ce[valid_idx]

    # Cross entropy (numerically stable)
    ce_per_element = F.cross_entropy(logits_valid, y_ce_valid, reduction="none")

    loss = ce_per_element.mean()
    return loss


# --------------------------------------------------------------------------- #
# Safe-Set Loss — Eq. (11)
# --------------------------------------------------------------------------- #

def safe_set_loss(
    h_vals: torch.Tensor,
    preds_are_safe: torch.Tensor,
    mask: torch.Tensor,
    eta: float = 0.0,
) -> torch.Tensor:
    """Safety-set loss per Eq. (11):

        L_SS = (1 / valid_count) Σ [ 2·I(predicted_label ∈ Y_safe) - 1 ] · max(0, h + η)

    Sign behavior at η=0:
        safe prediction:  sign = +1, loss = +ReLU(h)
        unsafe prediction: sign = -1, loss = -ReLU(h)

    Parameters
    ----------
    h_vals : [B, K]
        Barrier values h(x_{k-1}, u_k).
    preds_are_safe : [B, K]
        Boolean: True if predicted label ∈ Y_safe = {1,2,3,4}.
    mask : [B, K]
        Boolean mask.
    eta : float
        Training threshold (default 0).

    Returns
    -------
    Scalar loss tensor.
    """
    # Sign term: +1 for safe, -1 for unsafe
    is_safe = preds_are_safe.float()
    sign = 2.0 * is_safe - 1.0  # [B, K]

    # Hinge term
    hinge = F.relu(h_vals + eta)  # [B, K]

    # Element-wise loss
    loss_element = sign * hinge  # [B, K]

    # Masked mean
    mask_f = mask.float()
    valid_count = mask_f.sum().clamp(min=1.0)
    loss = (loss_element * mask_f).sum() / valid_count

    return loss


# --------------------------------------------------------------------------- #
# Safety-Invariance Loss — Eq. (12)
# --------------------------------------------------------------------------- #

def safety_invariance_loss(
    dynamics: DialogueDynamics,
    predictor: SafetyPredictor,
    U: torch.Tensor,
    mask: torch.Tensor,
    eta: float = 0.0,
    kappa: int = 3,
) -> torch.Tensor:
    """Safety-invariance loss per Eq. (12):

        L_SI = (1 / [N(K-κ)]) Σ_i Σ_{k=1}^{K-κ}
               max{ 0, h( f_θ(x_{k-1}, u_k), u_{k+1} ) + η }

    Paper's loss_forward_invariance uses:
        relu(p_unsafe - max(p_safe)) where p = softmax(logits)

    Indexing: for each valid turn k, evaluate h on the ROLLED-FORWARD state
    x_k = f_θ(x_{k-1}, u_k) with the NEXT query u_{k+1}.

    Parameters
    ----------
    dynamics : DialogueDynamics
    predictor : SafetyPredictor
    U : [B, K, 768]
        Query embeddings.
    mask : [B, K]
        Boolean mask.
    eta : float
        Training threshold (unused in paper's forward invariance loss formulation).
    kappa : int
        Number of final turns to exclude.

    Returns
    -------
    Scalar loss tensor.
    """
    B, K, _ = U.shape
    device = U.device
    dtype = U.dtype

    # Rollout to get all states
    X, _ = dynamics.rollout(U, mask)  # [B, K, 768]

    total_count = 0
    total_loss = torch.tensor(0.0, device=device, dtype=dtype)

    # k ranges from 0 to K-kappa-2 (0-indexed)
    # Turn k: x_k is state after processing u_k
    # Evaluate h(x_k, u_{k+1})
    max_k = K - kappa  # number of turns to consider (excluding last kappa)

    for k in range(max_k - 1):  # k = 0..K-kappa-2
        # Check if turn k and turn k+1 are valid
        valid_k = mask[:, k]       # [B]
        valid_k1 = mask[:, k + 1]  # [B]
        valid = valid_k & valid_k1  # [B]

        if not valid.any():
            continue

        # x_k: state after processing turn k
        x_k = X[:, k]  # [B, 768]

        # u_{k+1}: query at turn k+1
        u_k1 = U[:, k + 1]  # [B, 768]

        # Get logits and compute p_unsafe - max(p_safe) using paper's formulation
        # Paper: probs = softmax(logits), loss = relu(p_last - max(p_other))
        x_k_valid = x_k[valid]  # [N, 768]
        u_k1_valid = u_k1[valid]  # [N, 768]

        logits = predictor.net(torch.cat([x_k_valid, u_k1_valid], dim=-1))  # [N, 5]
        probs = F.softmax(logits, dim=-1)  # [N, 5]

        p_unsafe = probs[:, 4]  # p(class 5), shape [N]
        p_safe_max = probs[:, :4].max(dim=-1).values  # max p(class 1-4), shape [N]

        # Paper's forward invariance loss: relu(p_unsafe - max(p_safe))
        # Note: Paper does NOT use eta in forward invariance loss
        diff = p_unsafe - p_safe_max
        loss_k = F.relu(diff)  # [N]

        # Mask and accumulate
        total_loss = total_loss + loss_k.sum()
        total_count = total_count + valid.sum()

    total_count = total_count.clamp(min=1.0)
    return total_loss / total_count
