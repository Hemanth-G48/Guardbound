"""Stage-2 NBF trainer: trains SafetyPredictor with L_CE, L_SS, L_SI.

Paper: Adam lr=1e-3, 200 epochs.
Default mode: freeze dynamics, train only predictor.
Optional: --joint for joint fine-tuning.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import load_config, ProjectConfig
from ..logging_utils import get_logger
from ..models.dynamics import DialogueDynamics, load_dynamics
from ..models.predictor import SafetyPredictor, NeuralBarrierFunction
from ..training.losses import dynamics_loss
from ..training.nbf_losses import ce_loss, safe_set_loss, safety_invariance_loss
from ..training.diagnostics import rollout_mse_per_turn

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #

def load_phase2_dataset(
    dataset_dir: Path,
    split: str = "train",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load U, Z, Y, MASK from Phase-2 npz files."""
    npz_path = Path(dataset_dir) / f"{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    data = np.load(npz_path)
    U = torch.from_numpy(data["U"]).float()
    Z = torch.from_numpy(data["Z"]).float()
    Y = torch.from_numpy(data["Y"]).long()
    MASK = torch.from_numpy(data["MASK"]).bool()

    logger.info("Loaded %s split: U=%s Z=%s Y=%s MASK=%s",
                split, U.shape, Z.shape, Y.shape, MASK.shape)
    return U, Z, Y, MASK


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
# Metrics
# --------------------------------------------------------------------------- #

@dataclass
class NBFMetrics:
    epoch: int
    train_total: float
    train_dyn: float
    train_ce: float
    train_ss: float
    train_si: float
    val_total: float
    val_dyn: float
    val_ce: float
    val_ss: float
    val_si: float
    val_accuracy: float
    mean_h_safe: float
    mean_h_unsafe: float


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

class NBFTrainer:
    """Trains SafetyPredictor with L_CE, L_SS, L_SI on frozen or joint dynamics."""

    def __init__(
        self,
        dynamics: DialogueDynamics,
        predictor: SafetyPredictor,
        config: ProjectConfig,
        freeze_dynamics: bool = True,
        device: torch.device | None = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.freeze_dynamics = freeze_dynamics

        self.dynamics = dynamics.to(self.device)
        self.predictor = predictor.to(self.device)

        # Freeze dynamics if requested
        if freeze_dynamics:
            for p in self.dynamics.parameters():
                p.requires_grad = False
            logger.info("Dynamics FROZEN — only predictor will be trained")
        else:
            logger.info("Joint mode — dynamics + predictor will be trained")

        # Optimizer: paper Adam lr=1e-3
        stage2 = config.training.stage2_predictor
        params_to_train = list(self.predictor.parameters())
        if not freeze_dynamics:
            params_to_train += list(self.dynamics.parameters())

        self.optimizer = torch.optim.Adam(
            params_to_train,
            lr=stage2.lr,
            weight_decay=config.training_extra.weight_decay,
        )

        self.epochs = stage2.epochs
        self.batch_size = config.training_extra.batch_size

        # Loss weights from config
        self.lambda_dyn = config.training.loss_weights["lambda_dyn"]
        self.lambda_ce = config.training.loss_weights["lambda_ce"]
        self.lambda_ss = config.training.loss_weights["lambda_ss"]
        self.lambda_si = config.training.loss_weights["lambda_si"]

        # Training eta and kappa
        self.eta = config.training.eta_train
        self.kappa = config.training.kappa_noninvariant_turns

    def train(
        self,
        train_U: torch.Tensor,
        train_Z: torch.Tensor,
        train_Y: torch.Tensor,
        train_MASK: torch.Tensor,
        val_U: torch.Tensor,
        val_Z: torch.Tensor,
        val_Y: torch.Tensor,
        val_MASK: torch.Tensor,
        run_dir: Path,
        checkpoint_dir: Path,
        resume: bool = False,
    ) -> list[NBFMetrics]:
        """Run the full NBF training loop."""
        run_dir = Path(run_dir)
        checkpoint_dir = Path(checkpoint_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Resume
        start_epoch = 0
        if resume:
            ckpt_path = checkpoint_dir / "checkpoint.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                self.predictor.load_state_dict(ckpt["predictor_state_dict"])
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                logger.info("Resumed from epoch %d", start_epoch)

        all_metrics: list[NBFMetrics] = []
        best_val_loss = float("inf")

        logger.info("Starting NBF training: %d epochs, lr=%s, eta=%s, kappa=%d",
                     self.epochs - start_epoch,
                     self.config.training.stage2_predictor.lr,
                     self.eta, self.kappa)
        logger.info("Loss weights: dyn=%.1f, ce=%.1f, ss=%.1f, si=%.1f",
                     self.lambda_dyn, self.lambda_ce, self.lambda_ss, self.lambda_si)

        for epoch in range(start_epoch, self.epochs):
            # ---- Train ----
            train_metrics = self._train_epoch(train_U, train_Z, train_Y, train_MASK)

            # ---- Validate ----
            val_metrics = self._validate(val_U, val_Z, val_Y, val_MASK)

            metrics = NBFMetrics(
                epoch=epoch,
                **train_metrics,
                **val_metrics,
            )
            all_metrics.append(metrics)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d  total=%.4f ce=%.4f ss=%.4f si=%.4f  "
                    "val_acc=%.3f mean_h_safe=%.4f mean_h_unsafe=%.4f",
                    epoch + 1, self.epochs,
                    metrics.train_total, metrics.train_ce, metrics.train_ss, metrics.train_si,
                    metrics.val_accuracy, metrics.mean_h_safe, metrics.mean_h_unsafe,
                )

            # Save best
            if metrics.val_total < best_val_loss:
                best_val_loss = metrics.val_total
                self._save_checkpoint(checkpoint_dir, epoch, metrics)

            # Save metrics
            self._save_metrics(all_metrics, run_dir)

        # Final checkpoint
        self._save_checkpoint(checkpoint_dir, self.epochs - 1, metrics)
        self._save_loss_plot(all_metrics, run_dir)
        self._save_sanity_report(all_metrics[-1], run_dir)

        return all_metrics

    def _train_epoch(
        self,
        U: torch.Tensor, Z: torch.Tensor, Y: torch.Tensor, MASK: torch.Tensor,
    ) -> dict[str, float]:
        """Single training epoch."""
        self.dynamics.train()
        self.predictor.train()

        total_losses = {"train_total": 0, "train_dyn": 0, "train_ce": 0, "train_ss": 0, "train_si": 0}
        n_batches = 0

        # Simple batching (not using DataLoader for simplicity with small datasets)
        indices = torch.randperm(len(U))

        for start in range(0, len(U), self.batch_size):
            idx = indices[start:start + self.batch_size]
            batch_U = U[idx].to(self.device)
            batch_Z = Z[idx].to(self.device)
            batch_Y = Y[idx].to(self.device)
            batch_MASK = MASK[idx].to(self.device)

            self.optimizer.zero_grad()

            # Rollout
            X, Z_hat = self.dynamics.rollout(batch_U, batch_MASK)

            # L_dyn (reported but dynamics not updated in frozen mode)
            l_dyn = dynamics_loss(Z_hat, batch_Z, batch_MASK)

            # Predictor inputs: [x_{k-1}; u_k]
            # x_{k-1} for turn 0 is x_0 = 0
            B, K, D = batch_U.shape
            x_prev = torch.zeros(B, K, D, device=self.device)
            x_prev[:, 1:] = X[:, :-1]  # x_0=0, x_1, ..., x_{K-1}

            # L_CE
            l_ce = ce_loss(self.predictor, x_prev, batch_U, batch_Y, batch_MASK)

            # L_SS
            h_vals = torch.zeros(B, K, device=self.device)
            preds_safe = torch.zeros(B, K, dtype=torch.bool, device=self.device)
            for k in range(K):
                valid = batch_MASK[:, k]
                if valid.any():
                    h_k = self.predictor.predictor_value(x_prev[valid, k], batch_U[valid, k])
                    h_vals[valid, k] = h_k
                    preds_k = self.predictor.predicted_label(x_prev[valid, k], batch_U[valid, k])
                    preds_safe[valid, k] = (preds_k <= 4)  # safe = {1,2,3,4}

            l_ss = safe_set_loss(h_vals, preds_safe, batch_MASK, self.eta)

            # L_SI
            l_si = safety_invariance_loss(
                self.dynamics, self.predictor, batch_U, batch_MASK,
                self.eta, self.kappa,
            )

            # Total loss
            l_total = (
                self.lambda_dyn * l_dyn +
                self.lambda_ce * l_ce +
                self.lambda_ss * l_ss +
                self.lambda_si * l_si
            )

            # NaN check
            if not torch.isfinite(l_total):
                raise RuntimeError(f"Non-finite total loss: {l_total.item()}")

            l_total.backward()
            self.optimizer.step()

            total_losses["train_total"] += l_total.item()
            total_losses["train_dyn"] += l_dyn.item()
            total_losses["train_ce"] += l_ce.item()
            total_losses["train_ss"] += l_ss.item()
            total_losses["train_si"] += l_si.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    @torch.no_grad()
    def _validate(
        self,
        U: torch.Tensor, Z: torch.Tensor, Y: torch.Tensor, MASK: torch.Tensor,
    ) -> dict[str, float]:
        """Validation with accuracy and h-value statistics."""
        self.dynamics.eval()
        self.predictor.eval()

        U = U.to(self.device)
        Z = Z.to(self.device)
        Y = Y.to(self.device)
        MASK = MASK.to(self.device)

        # Rollout
        X, Z_hat = self.dynamics.rollout(U, MASK)

        # Predictor inputs
        B, K, D = U.shape
        x_prev = torch.zeros(B, K, D, device=self.device)
        x_prev[:, 1:] = X[:, :-1]

        # Compute losses
        l_dyn = dynamics_loss(Z_hat, Z, MASK)
        l_ce = ce_loss(self.predictor, x_prev, U, Y, MASK)

        h_vals = torch.zeros(B, K, device=self.device)
        preds_safe = torch.zeros(B, K, dtype=torch.bool, device=self.device)
        all_preds = torch.zeros(B, K, dtype=torch.long, device=self.device)

        for k in range(K):
            valid = MASK[:, k]
            if valid.any():
                h_k = self.predictor.predictor_value(x_prev[valid, k], U[valid, k])
                h_vals[valid, k] = h_k
                preds_k = self.predictor.predicted_label(x_prev[valid, k], U[valid, k])
                all_preds[valid, k] = preds_k
                preds_safe[valid, k] = (preds_k <= 4)

        l_ss = safe_set_loss(h_vals, preds_safe, MASK, self.eta)
        l_si = safety_invariance_loss(self.dynamics, self.predictor, U, MASK, self.eta, self.kappa)

        l_total = (
            self.lambda_dyn * l_dyn +
            self.lambda_ce * l_ce +
            self.lambda_ss * l_ss +
            self.lambda_si * l_si
        )

        # Accuracy
        correct = (all_preds == Y) & MASK
        total_valid = MASK.sum().float().clamp(min=1.0)
        accuracy = correct.float().sum() / total_valid

        # h-value statistics
        safe_mask = (Y <= 4) & MASK
        unsafe_mask = (Y == 5) & MASK

        mean_h_safe = h_vals[safe_mask].mean().item() if safe_mask.any() else float("nan")
        mean_h_unsafe = h_vals[unsafe_mask].mean().item() if unsafe_mask.any() else float("nan")

        self.dynamics.train()
        self.predictor.train()

        return {
            "val_total": l_total.item(),
            "val_dyn": l_dyn.item(),
            "val_ce": l_ce.item(),
            "val_ss": l_ss.item(),
            "val_si": l_si.item(),
            "val_accuracy": accuracy.item(),
            "mean_h_safe": mean_h_safe,
            "mean_h_unsafe": mean_h_unsafe,
        }

    def _save_checkpoint(
        self,
        checkpoint_dir: Path,
        epoch: int,
        metrics: NBFMetrics,
    ) -> None:
        """Save NBF checkpoint."""
        ckpt = {
            "predictor_state_dict": self.predictor.state_dict(),
            "dynamics_state_dict": self.dynamics.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "architecture": {
                "state_dim": self.predictor.state_dim,
                "embedding_dim": self.predictor.embedding_dim,
            },
            "training_mode": "frozen" if self.freeze_dynamics else "joint",
            "eta": self.eta,
            "kappa": self.kappa,
            "loss_weights": {
                "lambda_dyn": self.lambda_dyn,
                "lambda_ce": self.lambda_ce,
                "lambda_ss": self.lambda_ss,
                "lambda_si": self.lambda_si,
            },
            "seed": self.config.training_extra.seed,
            "config": {
                "lr": self.config.training.stage2_predictor.lr,
                "epochs": self.config.training.stage2_predictor.epochs,
                "batch_size": self.batch_size,
            },
        }

        ckpt_path = checkpoint_dir / "checkpoint.pt"
        torch.save(ckpt, ckpt_path)

        # Also save standalone predictor
        torch.save(self.predictor.state_dict(), checkpoint_dir / "predictor_h.pt")
        logger.info("Saved NBF checkpoint to %s", ckpt_path)

    def _save_metrics(self, metrics: list[NBFMetrics], run_dir: Path) -> None:
        csv_path = run_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_total", "train_dyn", "train_ce", "train_ss", "train_si",
                "val_total", "val_dyn", "val_ce", "val_ss", "val_si",
                "val_accuracy", "mean_h_safe", "mean_h_unsafe",
            ])
            for m in metrics:
                writer.writerow([
                    m.epoch,
                    f"{m.train_total:.8f}", f"{m.train_dyn:.8f}",
                    f"{m.train_ce:.8f}", f"{m.train_ss:.8f}", f"{m.train_si:.8f}",
                    f"{m.val_total:.8f}", f"{m.val_dyn:.8f}",
                    f"{m.val_ce:.8f}", f"{m.val_ss:.8f}", f"{m.val_si:.8f}",
                    f"{m.val_accuracy:.6f}",
                    f"{m.mean_h_safe:.6f}" if not np.isnan(m.mean_h_safe) else "NaN",
                    f"{m.mean_h_unsafe:.6f}" if not np.isnan(m.mean_h_unsafe) else "NaN",
                ])

    def _save_loss_plot(self, metrics: list[NBFMetrics], run_dir: Path) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = [m.epoch + 1 for m in metrics]
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Total loss
            axes[0, 0].plot(epochs, [m.train_total for m in metrics], label="Train")
            axes[0, 0].plot(epochs, [m.val_total for m in metrics], label="Val")
            axes[0, 0].set_title("Total Loss")
            axes[0, 0].legend()
            axes[0, 0].set_ylabel("L_total")

            # Component losses
            axes[0, 1].plot(epochs, [m.train_ce for m in metrics], label="L_CE")
            axes[0, 1].plot(epochs, [m.train_ss for m in metrics], label="L_SS")
            axes[0, 1].plot(epochs, [m.train_si for m in metrics], label="L_SI")
            axes[0, 1].set_title("Component Losses (Train)")
            axes[0, 1].legend()

            # Accuracy
            axes[1, 0].plot(epochs, [m.val_accuracy for m in metrics])
            axes[1, 0].set_title("Validation Accuracy")
            axes[1, 0].set_ylabel("Accuracy")

            # h-value means
            safe_h = [m.mean_h_safe for m in metrics if not np.isnan(m.mean_h_safe)]
            unsafe_h = [m.mean_h_unsafe for m in metrics if not np.isnan(m.mean_h_unsafe)]
            if safe_h:
                axes[1, 1].plot(range(1, len(safe_h) + 1), safe_h, label="Safe turns")
            if unsafe_h:
                axes[1, 1].plot(range(1, len(unsafe_h) + 1), unsafe_h, label="Unsafe turns")
            axes[1, 1].set_title("Mean h-value by Safety Class")
            axes[1, 1].legend()

            fig.suptitle("NBF Training Losses")
            fig.tight_layout()
            fig.savefig(run_dir / "losses.png", dpi=150)
            plt.close(fig)
            logger.info("Saved loss plot to %s", run_dir / "losses.png")
        except ImportError:
            logger.warning("matplotlib not available, skipping loss plot")

    def _save_sanity_report(self, metrics: NBFMetrics, run_dir: Path) -> None:
        """Save post-training sanity report."""
        report = {
            "val_accuracy": metrics.val_accuracy,
            "mean_h_safe": metrics.mean_h_safe,
            "mean_h_unsafe": metrics.mean_h_unsafe,
            "final_train_loss": metrics.train_total,
            "final_val_loss": metrics.val_total,
            "training_mode": "frozen" if self.freeze_dynamics else "joint",
            "eta": self.eta,
            "kappa": self.kappa,
        }

        report_path = run_dir / "sanity_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        # Text version
        txt_path = run_dir / "sanity_report.txt"
        with open(txt_path, "w") as f:
            f.write("NBF Training Sanity Report\n")
            f.write("=" * 40 + "\n")
            f.write(f"Training mode: {report['training_mode']}\n")
            f.write(f"Val accuracy: {report['val_accuracy']:.4f}\n")
            f.write(f"Mean h (safe turns): {report['mean_h_safe']:.6f}\n")
            f.write(f"Mean h (unsafe turns): {report['mean_h_unsafe']:.6f}\n")
            f.write(f"Final train loss: {report['final_train_loss']:.6f}\n")
            f.write(f"Final val loss: {report['final_val_loss']:.6f}\n")

        logger.info("Saved sanity report to %s", run_dir / "sanity_report.json")
