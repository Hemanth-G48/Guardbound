#!/usr/bin/env python
"""Train neural dialogue dynamics (f_theta, g_theta) — Stage 1.

Paper: Adam lr=1e-4, 200 epochs, L_dyn only.

Usage:
    python scripts/train_dynamics.py --embedding mpnet --epochs 200
    python scripts/train_dynamics.py --embedding mpnet --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.config import load_config
from guardbound.logging_utils import setup_logging, get_logger
from guardbound.models.dynamics import DialogueDynamics

logger = get_logger(__name__)

# Embedding model name mapping
EMBEDDING_MAP = {
    "mpnet": "all-mpnet-base-v2",
    "distilroberta": "all-distilroberta-v1",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--embedding", default="mpnet",
        choices=["mpnet", "distilroberta"],
        help="Embedding model (default: mpnet)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (default: from config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: from config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: from config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: from config)")
    parser.add_argument("--device", default=None, help="Device: cpu/cuda (default: auto)")
    parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint output dir")
    parser.add_argument("--run-dir", default=None, help="Metrics/run output dir")
    parser.add_argument("--dataset-dir", default=None, help="Phase-2 dataset dir")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no training")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(level=__import__("logging").DEBUG if args.verbose else __import__("logging").INFO)

    # Load config
    cfg = load_config(args.config)

    # Resolve embedding model
    emb_name = EMBEDDING_MAP.get(args.embedding, args.embedding)

    # Resolve paths
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = Path(cfg.paths.checkpoints) / f"dynamics_{args.embedding}"

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path(cfg.paths.runs) / f"dynamics_{args.embedding}"

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        # Default: data/processed/datasets/<embedding_model>
        dataset_dir = Path("data/processed/datasets") / emb_name.replace("/", "_")

    # Override config with CLI args
    if args.epochs is not None:
        cfg.training.stage1_dynamics.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training_extra.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.stage1_dynamics.lr = args.lr
    if args.seed is not None:
        cfg.training_extra.seed = args.seed

    import torch
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Print info
    logger.info("Phase 3: Neural Dialogue Dynamics Training")
    logger.info("  Embedding model: %s", emb_name)
    logger.info("  Dataset dir: %s", dataset_dir)
    logger.info("  Checkpoint dir: %s", checkpoint_dir)
    logger.info("  Run dir: %s", run_dir)
    logger.info("  Device: %s", device)
    logger.info("  Epochs: %d", cfg.training.stage1_dynamics.epochs)
    logger.info("  LR: %s", cfg.training.stage1_dynamics.lr)
    logger.info("  Batch size: %d", cfg.training_extra.batch_size)
    logger.info("  Seed: %d", cfg.training_extra.seed)

    if args.dry_run:
        logger.info("[DRY RUN] Would train dynamics model")
        model = DialogueDynamics(
            embedding_dim=cfg.dims.embedding_dim_n,
            state_dim=cfg.dims.state_dim_m,
            hidden_dims=cfg.dynamics.hidden_dims if hasattr(cfg, "dynamics") else [512, 512],
        )
        logger.info("  f_theta params: %d", model.f_theta_param_count())
        logger.info("  g_theta params: %d", model.g_theta_param_count())
        logger.info("  Total params: %d", model.total_param_count())
        return 0

    # Load datasets
    from guardbound.training.train_dynamics import load_phase2_dataset, DynamicsTrainer, set_seeds

    set_seeds(cfg.training_extra.seed)

    train_U, train_Z, train_Y, train_MASK = load_phase2_dataset(dataset_dir, emb_name, "train")
    val_U, val_Z, val_Y, val_MASK = load_phase2_dataset(dataset_dir, emb_name, "val")

    # Create model
    hidden_dims = cfg.dynamics.hidden_dims if hasattr(cfg, "dynamics") else [512, 512]
    model = DialogueDynamics(
        embedding_dim=cfg.dims.embedding_dim_n,
        state_dim=cfg.dims.state_dim_m,
        hidden_dims=hidden_dims,
    )

    logger.info("  f_theta params: %d", model.f_theta_param_count())
    logger.info("  g_theta params: %d", model.g_theta_param_count())
    logger.info("  Total params: %d", model.total_param_count())

    # Train
    trainer = DynamicsTrainer(model=model, config=cfg, device=device)

    metrics = trainer.train(
        train_U=train_U, train_Z=train_Z, train_MASK=train_MASK,
        val_U=val_U, val_Z=val_Z, val_MASK=val_MASK,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        resume=args.resume,
    )

    logger.info("Training complete. Artifacts saved to:")
    logger.info("  Metrics: %s/metrics.csv", run_dir)
    logger.info("  Loss plot: %s/loss.png", run_dir)
    logger.info("  Per-turn MSE: %s/per_turn_mse.csv", run_dir)
    logger.info("  Checkpoint: %s/dialogue_dynamics.pt", checkpoint_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
