"""Stage-1 dynamics trainer: trains f_theta and g_theta on L_dyn.

Paper: Adam lr=1e-4, 200 epochs, loss L_dyn only.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import load_config, ProjectConfig
from ..logging_utils import get_logger
from ..models.dynamics import DialogueDynamics, save_dynamics, load_dynamics
from ..training.losses import dynamics_loss
from ..training.diagnostics import rollout_mse_per_turn, format_per_turn_table

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def load_phase2_dataset(
    dataset_dir: Path,
    embedding_model: str = "all-mpnet-base-v2",
    split: str = "train",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load U, Z, Y, MASK from Phase-2 npz files.

    Returns (U, Z, Y, MASK) as torch tensors.
    """
    npz_path = Path(dataset_dir) / f"{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    data = np.load(npz_path)
    U = torch.from_numpy(data["U"]).float()
    Z = torch.from_numpy(data["Z"]).float()
    Y = torch.from_numpy(data["Y"]).long()
    MASK = torch.from_numpy(data["MASK"]).bool()

    logger.info(
        "Loaded %s split: U=%s Z=%s Y=%s MASK=%s",
        split, U.shape, Z.shape, Y.shape, MASK.shape,
    )
    return U, Z, Y, MASK


def validate_dataset(U: torch.Tensor, Z: torch.Tensor, MASK: torch.Tensor) -> None:
    """Validate dataset shapes and dtypes before training."""
    errors = []
    if U.dtype != torch.float32:
        errors.append(f"U.dtype={U.dtype}, expected float32")
    if Z.dtype != torch.float32:
        errors.append(f"Z.dtype={Z.dtype}, expected float32")
    if MASK.dtype != torch.bool:
        errors.append(f"MASK.dtype={MASK.dtype}, expected bool")
    if U.shape[-1] != 768:
        errors.append(f"U.shape[-1]={U.shape[-1]}, expected 768")
    if Z.shape[-1] != 768:
        errors.append(f"Z.shape[-1]={Z.shape[-1]}, expected 768")
    if U.shape[:2] != Z.shape[:2]:
        errors.append(f"U/Z shape mismatch: {U.shape} vs {Z.shape}")
    if U.shape[:2] != MASK.shape:
        errors.append(f"U/MASK shape mismatch: {U.shape[:2]} vs {MASK.shape}")
    if MASK.sum() == 0:
        errors.append("MASK is all False — no valid turns")
    if errors:
        raise ValueError("Invalid dataset:\n  - " + "\n  - ".join(errors))


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Not specified in the paper — local reproducibility default.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

@dataclass
class TrainMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    per_turn_mse: list[float]
    learning_rate: float


class DynamicsTrainer:
    """Trains f_theta and g_theta on L_dyn with Adam."""

    def __init__(
        self,
        model: DialogueDynamics,
        config: ProjectConfig,
        device: torch.device | None = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)

        # Paper: Adam, lr=1e-4
        stage1 = config.training.stage1_dynamics
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=stage1.lr,
            weight_decay=config.training_extra.weight_decay,
        )

        self.epochs = stage1.epochs
        self.batch_size = config.training_extra.batch_size

        # No scheduler by default (not specified in the paper)
        self.scheduler = None
        if hasattr(config.training_extra, 'scheduler') and config.training_extra.scheduler != "none":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=50, gamma=0.5
            )

        # Gradient clipping (disabled by default, not specified in paper)
        self.grad_clip_norm = None
        if hasattr(config.training_extra, 'gradient_clip_norm') and config.training_extra.gradient_clip_norm is not None:
            self.grad_clip_norm = config.training_extra.gradient_clip_norm

    def train(
        self,
        train_U: torch.Tensor,
        train_Z: torch.Tensor,
        train_MASK: torch.Tensor,
        val_U: torch.Tensor,
        val_Z: torch.Tensor,
        val_MASK: torch.Tensor,
        run_dir: Path,
        checkpoint_dir: Path,
        resume: bool = False,
    ) -> list[TrainMetrics]:
        """Run the full training loop.

        Returns list of per-epoch metrics.
        """
        run_dir = Path(run_dir)
        checkpoint_dir = Path(checkpoint_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Validate datasets
        validate_dataset(train_U, train_Z, train_MASK)
        validate_dataset(val_U, val_Z, val_MASK)

        # Create dataloader
        train_dataset = TensorDataset(train_U, train_Z, train_MASK)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            drop_last=False,
        )

        # Resume from checkpoint
        start_epoch = 0
        if resume:
            ckpt_path = checkpoint_dir / "dialogue_dynamics.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                logger.info("Resumed from epoch %d", start_epoch)

        # Metrics history
        all_metrics: list[TrainMetrics] = []
        best_val_loss = float("inf")

        logger.info("Starting dynamics training: %d epochs, batch_size=%d, lr=%s",
                     self.epochs - start_epoch, self.batch_size,
                     self.config.training.stage1_dynamics.lr)
        logger.info("Device: %s", self.device)
        logger.info("Model parameters: %d", self.model.total_param_count())

        for epoch in range(start_epoch, self.epochs):
            # ---- Train ----
            self.model.train()
            train_losses = []
            for batch_U, batch_Z, batch_MASK in train_loader:
                batch_U = batch_U.to(self.device)
                batch_Z = batch_Z.to(self.device)
                batch_MASK = batch_MASK.to(self.device)

                self.optimizer.zero_grad()
                _, Z_hat = self.model.rollout(batch_U, batch_MASK)
                loss = dynamics_loss(Z_hat, batch_Z, batch_MASK)

                # NaN/Inf detection
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at epoch {epoch}: {loss.item()}. "
                        "Check for NaN/Inf in inputs."
                    )

                loss.backward()

                # Gradient clipping
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                self.optimizer.step()
                train_losses.append(loss.item())

            if self.scheduler:
                self.scheduler.step()

            avg_train_loss = sum(train_losses) / len(train_losses)

            # ---- Validate ----
            val_loss, per_turn_mse = self._validate(val_U, val_Z, val_MASK)

            # ---- Log ----
            current_lr = self.optimizer.param_groups[0]["lr"]
            metrics = TrainMetrics(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                per_turn_mse=per_turn_mse,
                learning_rate=current_lr,
            )
            all_metrics.append(metrics)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f  lr=%.2e",
                    epoch + 1, self.epochs, avg_train_loss, val_loss, current_lr,
                )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(
                    checkpoint_dir, epoch, val_loss,
                    self._get_dataset_metadata(train_U, train_Z, train_MASK),
                )

            # Save metrics
            self._save_metrics(all_metrics, run_dir)

        # Save final checkpoint
        self._save_checkpoint(
            checkpoint_dir, self.epochs - 1, val_loss,
            self._get_dataset_metadata(train_U, train_Z, train_MASK),
        )

        # Save loss plot
        self._save_loss_plot(all_metrics, run_dir)

        # Save per-turn MSE
        self._save_per_turn_mse(all_metrics, run_dir)

        # Print final validation report
        self._print_validation_report(per_turn_mse, val_loss)

        return all_metrics

    @torch.no_grad()
    def _validate(
        self,
        val_U: torch.Tensor,
        val_Z: torch.Tensor,
        val_MASK: torch.Tensor,
    ) -> tuple[float, list[float]]:
        """Compute validation loss and per-turn MSE."""
        self.model.eval()

        val_U = val_U.to(self.device)
        val_Z = val_Z.to(self.device)
        val_MASK = val_MASK.to(self.device)

        _, Z_hat = self.model.rollout(val_U, val_MASK)
        val_loss = dynamics_loss(Z_hat, val_Z, val_MASK).item()

        per_turn_mse = rollout_mse_per_turn(self.model, val_U, val_Z, val_MASK)

        self.model.train()
        return val_loss, per_turn_mse

    def _save_checkpoint(
        self,
        checkpoint_dir: Path,
        epoch: int,
        val_loss: float,
        dataset_metadata: dict,
    ) -> None:
        """Save structured checkpoint."""
        config_dict = {
            "embedding_dim": self.model.embedding_dim,
            "state_dim": self.model.state_dim,
            "hidden_dims": self.model.hidden_dims,
            "lr": self.config.training.stage1_dynamics.lr,
            "epochs": self.config.training.stage1_dynamics.epochs,
            "batch_size": self.batch_size,
            "weight_decay": self.config.training_extra.weight_decay,
        }

        save_dynamics(
            model=self.model,
            checkpoint_dir=checkpoint_dir,
            optimizer_state=self.optimizer.state_dict(),
            epoch=epoch,
            config=config_dict,
            dataset_metadata=dataset_metadata,
            seed=self.config.training_extra.seed,
            extra_metadata={"best_val_loss": val_loss},
        )

    def _save_metrics(self, metrics: list[TrainMetrics], run_dir: Path) -> None:
        """Save training metrics to CSV."""
        csv_path = run_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["epoch", "train_loss", "val_loss", "learning_rate"]
            header += [f"MSE_{k+1}" for k in range(8)]
            writer.writerow(header)
            for m in metrics:
                row = [m.epoch, f"{m.train_loss:.8f}", f"{m.val_loss:.8f}",
                       f"{m.learning_rate:.2e}"]
                row += [f"{v:.8f}" if not np.isnan(v) else "NaN" for v in m.per_turn_mse]
                writer.writerow(row)

    def _save_loss_plot(self, metrics: list[TrainMetrics], run_dir: Path) -> None:
        """Generate training/validation loss curve."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = [m.epoch + 1 for m in metrics]
            train_losses = [m.train_loss for m in metrics]
            val_losses = [m.val_loss for m in metrics]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs, train_losses, label="Training Loss", linewidth=2)
            ax.plot(epochs, val_losses, label="Validation Loss", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("L_dyn (MSE)")
            ax.set_title("Dynamics Training Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(run_dir / "loss.png", dpi=150)
            plt.close(fig)
            logger.info("Saved loss plot to %s", run_dir / "loss.png")
        except ImportError:
            logger.warning("matplotlib not available, skipping loss plot")

    def _save_per_turn_mse(self, metrics: list[TrainMetrics], run_dir: Path) -> None:
        """Save per-turn MSE over epochs."""
        if not metrics:
            return
        K = len(metrics[-1].per_turn_mse)
        csv_path = run_dir / "per_turn_mse.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + [f"MSE_{k+1}" for k in range(K)])
            for m in metrics:
                row = [m.epoch + 1]
                row += [f"{v:.8f}" if not np.isnan(v) else "NaN" for v in m.per_turn_mse]
                writer.writerow(row)

    def _get_dataset_metadata(
        self,
        U: torch.Tensor,
        Z: torch.Tensor,
        MASK: torch.Tensor,
    ) -> dict:
        """Collect dataset metadata for checkpoint."""
        return {
            "U_shape": list(U.shape),
            "Z_shape": list(Z.shape),
            "MASK_shape": list(MASK.shape),
            "num_samples": U.shape[0],
            "num_turns": U.shape[1],
            "embedding_dim": U.shape[2],
            "valid_turns": int(MASK.sum()),
        }

    def _print_validation_report(
        self,
        per_turn_mse: list[float],
        val_loss: float,
    ) -> None:
        """Print final validation report."""
        logger.info("\nValidation rollout MSE")
        logger.info(format_per_turn_table(per_turn_mse))
        logger.info("\nOverall validation L_dyn: %.6f", val_loss)
