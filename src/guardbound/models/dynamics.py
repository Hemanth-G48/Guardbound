"""Neural dialogue dynamics: f_theta and g_theta.

Implements the state-space model from Eq. (3):

    x_k = f_theta(x_{k-1}, u_k)
    z_k = g_theta(x_k, u_k)
    x_0 = 0_m

Architecture (paper Sec. 5.1):
    f_theta: R^1536 -> R^768  (Linear-ReLU-Linear-ReLU-Linear)
    g_theta: R^1536 -> R^768  (Linear-ReLU-Linear-ReLU-Linear)

Both MLPs share the same architecture but have **separate parameters**.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..logging_utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Single MLP
# --------------------------------------------------------------------------- #

class MLPDynamics(nn.Module):
    """A single 3-layer MLP: input_dim -> hidden[0] -> hidden[1] -> output_dim.

    Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear (no activation on output).
    Paper: not specified in the paper — default: linear output for regression.
    """

    def __init__(
        self,
        input_dim: int = 1536,    # 768 + 768 (state + embedding)
        hidden_dims: list[int] | None = None,
        output_dim: int = 768,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]  # paper: 512, 512

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))  # linear output

        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# Dialogue Dynamics
# --------------------------------------------------------------------------- #

class DialogueDynamics(nn.Module):
    """Full dialogue dynamics model containing f_theta and g_theta.

    Performs temporal rollout per Eq. (3):

        x_0 = zeros(768)
        for k in 1..K:
            x_k = f_theta([x_{k-1}, u_k])
            z_hat_k = g_theta([x_k, u_k])
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        state_dim: int = 768,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims or [512, 512]

        # f_theta: [x_{k-1}, u_k] -> x_k
        self.f_theta = MLPDynamics(
            input_dim=state_dim + embedding_dim,
            hidden_dims=self.hidden_dims,
            output_dim=state_dim,
        )

        # g_theta: [x_k, u_k] -> z_hat_k
        self.g_theta = MLPDynamics(
            input_dim=state_dim + embedding_dim,
            hidden_dims=self.hidden_dims,
            output_dim=embedding_dim,
        )

    def forward(
        self,
        U: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Complete temporal rollout.

        Parameters
        ----------
        U : [B, K, embedding_dim]
            Query embeddings for each turn.
        mask : [B, K] or None
            Boolean mask: True where turn exists, False for padding.

        Returns
        -------
        X : [B, K, state_dim]
            Latent states x_k for each turn.
        Z_hat : [B, K, embedding_dim]
            Predicted response embeddings for each turn.
        """
        return self.rollout(U, mask)

    def rollout(
        self,
        U: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Temporal rollout through all K turns.

        Preserves causal dependencies: x_k depends only on x_{k-1} and u_k.
        Masked turns do not modify the state.
        """
        B, K, _ = U.shape
        device = U.device
        dtype = U.dtype

        # x_0 = zeros
        x = torch.zeros(B, self.state_dim, device=device, dtype=dtype)

        X = torch.zeros(B, K, self.state_dim, device=device, dtype=dtype)
        Z_hat = torch.zeros(B, K, self.embedding_dim, device=device, dtype=dtype)

        for k in range(K):
            u_k = U[:, k]  # [B, emb_dim]

            if mask is not None:
                valid = mask[:, k]  # [B]
            else:
                valid = torch.ones(B, dtype=torch.bool, device=device)

            # Always compute (needed for autograd graph), but conditionally update
            # Compute f_theta([x, u_k])
            f_input = torch.cat([x, u_k], dim=-1)  # [B, state_dim + emb_dim]
            x_next = self.f_theta(f_input)  # [B, state_dim]

            # Compute g_theta([x_next, u_k])
            g_input = torch.cat([x_next, u_k], dim=-1)  # [B, state_dim + emb_dim]
            z_hat_k = self.g_theta(g_input)  # [B, emb_dim]

            # Conditionally update state: only for valid turns
            # Use where to avoid data-dependent control flow issues
            valid_f = valid.float().unsqueeze(-1)  # [B, 1]
            x_new = valid_f * x_next + (1.0 - valid_f) * x

            X[:, k] = x_new
            Z_hat[:, k] = z_hat_k
            x = x_new

        return X, Z_hat

    def f_theta_param_count(self) -> int:
        return self.f_theta.param_count()

    def g_theta_param_count(self) -> int:
        return self.g_theta.param_count()

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# Checkpoint save / load
# --------------------------------------------------------------------------- #

def save_dynamics(
    model: DialogueDynamics,
    checkpoint_dir: Path,
    optimizer_state: dict | None = None,
    epoch: int = 0,
    config: dict | None = None,
    dataset_metadata: dict | None = None,
    seed: int = 42,
    extra_metadata: dict | None = None,
) -> Path:
    """Save a structured checkpoint with full reproducibility metadata.

    Returns the path to the main checkpoint file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save individual components
    torch.save(model.f_theta.state_dict(), checkpoint_dir / "f_theta.pt")
    torch.save(model.g_theta.state_dict(), checkpoint_dir / "g_theta.pt")

    # Save complete model state
    main_ckpt = {
        "model_state_dict": model.state_dict(),
        "f_theta_state_dict": model.f_theta.state_dict(),
        "g_theta_state_dict": model.g_theta.state_dict(),
        "optimizer_state_dict": optimizer_state,
        "epoch": epoch,
        "seed": seed,
        "architecture": {
            "embedding_dim": model.embedding_dim,
            "state_dim": model.state_dim,
            "hidden_dims": model.hidden_dims,
        },
        "config": config or {},
        "dataset_metadata": dataset_metadata or {},
        "extra_metadata": extra_metadata or {},
    }

    ckpt_path = checkpoint_dir / "dialogue_dynamics.pt"
    torch.save(main_ckpt, ckpt_path)
    logger.info("Saved dynamics checkpoint to %s", ckpt_path)
    return ckpt_path


def load_dynamics(
    checkpoint_dir: Path,
    device: str | torch.device | None = None,
) -> DialogueDynamics:
    """Load a pretrained DialogueDynamics from a checkpoint directory.

    Returns the model in eval mode. Does NOT freeze parameters —
    Phase 4 decides whether to freeze f_theta/g_theta.
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Try main checkpoint first
    ckpt_path = checkpoint_dir / "dialogue_dynamics.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device or "cpu", weights_only=False)
        arch = ckpt.get("architecture", {})
        model = DialogueDynamics(
            embedding_dim=arch.get("embedding_dim", 768),
            state_dim=arch.get("state_dim", 768),
            hidden_dims=arch.get("hidden_dims", [512, 512]),
        )
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # Fallback: load individual components
        f_path = checkpoint_dir / "f_theta.pt"
        g_path = checkpoint_dir / "g_theta.pt"
        if not f_path.exists() or not g_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_dir}. "
                "Expected dialogue_dynamics.pt or f_theta.pt + g_theta.pt."
            )
        model = DialogueDynamics()
        model.f_theta.load_state_dict(torch.load(f_path, map_location=device or "cpu", weights_only=True))
        model.g_theta.load_state_dict(torch.load(g_path, map_location=device or "cpu", weights_only=True))

    model.eval()
    logger.info("Loaded dynamics from %s (eval mode)", checkpoint_dir)
    return model
