"""Safety predictor h_theta and Neural Barrier Function (NBF) bundle.

Implements:
- SafetyPredictor: MLP 1536→32→32→5 for 5-class safety scoring
- Eq. (5): h(x, u) = p(unsafe) - max(p(safe))
- NeuralBarrierFunction: bundles dynamics + predictor for Phase 5 handoff
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..logging_utils import get_logger
from .dynamics import DialogueDynamics, load_dynamics

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Safety Predictor
# --------------------------------------------------------------------------- #

class SafetyPredictor(nn.Module):
    """5-class safety predictor: [x_{k-1}; u_k] -> logits(5).

    Architecture (paper Sec. 5.1):
        Linear(1536, 32) -> ReLU -> Linear(32, 32) -> ReLU -> Linear(32, 5)

    Paper label convention:
        Y = {1, 2, 3, 4, 5}  (1-based)
        Y_safe = {1, 2, 3, 4}
        Unsafe = 5

    PyTorch CE expects 0-based indices:
        CE index = paper_label - 1
    """

    # Paper label {1,2,3,4,5} to CE index {0,1,2,3,4}
    PAPER_LABEL_TO_CE = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    CE_TO_PAPER_LABEL = {v: k for k, v in PAPER_LABEL_TO_CE.items()}

    def __init__(
        self,
        state_dim: int = 768,
        embedding_dim: int = 768,
    ):
        super().__init__()
        input_dim = state_dim + embedding_dim  # 1536

        # Paper: 1536 -> 32 -> 32 -> 5
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

        self.state_dim = state_dim
        self.embedding_dim = embedding_dim

    def forward(self, x_prev: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits.

        Parameters
        ----------
        x_prev : [B, state_dim]
            Previous latent state.
        u : [B, embedding_dim]
            Current query embedding.

        Returns
        -------
        logits : [B, 5]
            Raw logits for 5 classes.
        """
        inp = torch.cat([x_prev, u], dim=-1)  # [B, 1536]
        return self.net(inp)

    def class_probs(self, x_prev: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities p(y | x_{k-1}, u_k).

        Returns
        -------
        p : [B, 5]
            Probability distribution over {1,2,3,4,5} (ordered by CE index).
        """
        logits = self.forward(x_prev, u)
        return F.softmax(logits, dim=-1)

    def predictor_value(self, x_prev: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Compute barrier value h per Eq. (5):

            h(x_{k-1}, u_k) = p(ŷ ∉ Y_safe) - max_{y ∈ Y_safe} p(ŷ = y)

        Since class 5 (index 4) is unsafe:
            h = p[:, 4] - max(p[:, 0:4])

        Returns
        -------
        h : [B]
            Scalar barrier value per sample.
        """
        p = self.class_probs(x_prev, u)
        p_unsafe = p[:, 4]                    # p(score=5)
        p_safe = p[:, :4]                     # p(score=1..4)
        h = p_unsafe - p_safe.max(dim=-1).values
        return h

    def predicted_label(self, x_prev: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Return predicted label in paper convention {1,2,3,4,5}.

        Returns
        -------
        labels : [B]
            Predicted class label in {1,2,3,4,5}.
        """
        p = self.class_probs(x_prev, u)
        ce_indices = p.argmax(dim=-1)  # {0,1,2,3,4}
        # Convert to paper labels {1,2,3,4,5}
        paper_labels = ce_indices + 1
        return paper_labels

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# Neural Barrier Function Bundle
# --------------------------------------------------------------------------- #

class NeuralBarrierFunction(nn.Module):
    """Bundles pretrained dynamics + safety predictor for inference.

    Exposes the API needed by Phase 5:
        h(state, query_embedding) -> barrier value
        filter_query(state, query_text, eta) -> (allowed, h_value)

    Keep h() differentiable w.r.t. state and query embedding.
    """

    def __init__(
        self,
        dynamics: DialogueDynamics,
        predictor: SafetyPredictor,
        embedding_model: str = "all-mpnet-base-v2",
    ):
        super().__init__()
        self.dynamics = dynamics
        self.predictor = predictor
        self.embedding_model = embedding_model

    def h(
        self,
        state: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute barrier value h(x, u).

        Parameters
        ----------
        state : [B, 768]
            Current latent state.
        query_embedding : [B, 768]
            Embedded query.

        Returns
        -------
        h : [B]
            Barrier value.
        """
        return self.predictor.predictor_value(state, query_embedding)

    def filter_query(
        self,
        state: torch.Tensor,
        query_embedding: torch.Tensor,
        eta: float,
    ) -> tuple[bool, float]:
        """Determine whether a query should be filtered.

        Parameters
        ----------
        state : [1, 768] or [768]
            Current latent state.
        query_embedding : [1, 768] or [768]
            Embedded query.
        eta : float
            Steering threshold.

        Returns
        -------
        allowed : bool
            True if query passes the filter (h + eta < 0).
        h_value : float
            The barrier value.
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if query_embedding.dim() == 1:
                query_embedding = query_embedding.unsqueeze(0)

            h_val = self.h(state, query_embedding)
            h_value = h_val.item()
            allowed = (h_value + eta) < 0
            return allowed, h_value

    def advance_state(
        self,
        state: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the latent state using f_theta.

        Parameters
        ----------
        state : [B, 768]
        query_embedding : [B, 768]

        Returns
        -------
        new_state : [B, 768]
        """
        f_input = torch.cat([state, query_embedding], dim=-1)
        return self.dynamics.f_theta(f_input)

    @classmethod
    def load(
        cls,
        dynamics_dir: Path,
        predictor_path: Path,
        device: str | torch.device | None = None,
    ) -> NeuralBarrierFunction:
        """Load a complete NBF bundle from checkpoints."""
        dynamics = load_dynamics(dynamics_dir, device)
        ckpt = torch.load(predictor_path, map_location=device or "cpu", weights_only=False)

        arch = ckpt.get("architecture", {})
        predictor = SafetyPredictor(
            state_dim=arch.get("state_dim", 768),
            embedding_dim=arch.get("embedding_dim", 768),
        )
        predictor.load_state_dict(ckpt["predictor_state_dict"])

        nbf = cls(
            dynamics=dynamics,
            predictor=predictor,
            embedding_model=ckpt.get("embedding_model", "all-mpnet-base-v2"),
        )
        nbf.eval()
        return nbf

    def save(
        self,
        predictor_path: Path,
        config: dict | None = None,
        training_mode: str = "frozen",
        eta: float = 0.0,
        kappa: int = 3,
        loss_weights: dict | None = None,
        epoch: int = 0,
        seed: int = 42,
    ) -> None:
        """Save the predictor and metadata."""
        predictor_path = Path(predictor_path)
        predictor_path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "predictor_state_dict": self.predictor.state_dict(),
            "dynamics_state_dict": self.dynamics.state_dict(),
            "architecture": {
                "state_dim": self.predictor.state_dim,
                "embedding_dim": self.predictor.embedding_dim,
            },
            "embedding_model": self.embedding_model,
            "training_mode": training_mode,
            "eta": eta,
            "kappa": kappa,
            "loss_weights": loss_weights or {},
            "epoch": epoch,
            "seed": seed,
            "config": config or {},
        }
        torch.save(ckpt, predictor_path)
        logger.info("Saved NBF checkpoint to %s", predictor_path)
