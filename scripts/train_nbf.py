#!/usr/bin/env python
"""Train the NBF safety predictor (Stage 2).

Paper: Adam lr=1e-3, 200 epochs, frozen dynamics by default.

Usage:
    python scripts/train_nbf.py --embedding mpnet --freeze-dynamics --epochs 200
    python scripts/train_nbf.py --embedding mpnet --joint --epochs 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.config import load_config
from guardbound.logging_utils import setup_logging, get_logger
from guardbound.models.dynamics import DialogueDynamics, load_dynamics
from guardbound.models.predictor import SafetyPredictor

logger = get_logger(__name__)

EMBEDDING_MAP = {
    "mpnet": "all-mpnet-base-v2",
    "distilroberta": "all-distilroberta-v1",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--embedding", default="mpnet", choices=["mpnet", "distilroberta"])
    parser.add_argument("--freeze-dynamics", action="store_true", default=True)
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--dynamics-dir", default=None, help="Phase-3 checkpoint dir")
    parser.add_argument("--eta", type=float, default=None, help="Training eta (default: from config)")
    parser.add_argument("--kappa", type=int, default=None, help="Kappa (default: from config)")
    parser.add_argument("--lambda-dyn", type=float, default=None)
    parser.add_argument("--lambda-ce", type=float, default=None)
    parser.add_argument("--lambda-ss", type=float, default=None)
    parser.add_argument("--lambda-si", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(level=__import__("logging").DEBUG if args.verbose else __import__("logging").INFO)

    if args.joint and args.freeze_dynamics:
        parser.error("Cannot use both --joint and --freeze-dynamics. Choose one.")

    cfg = load_config(args.config)

    # Resolve
    emb_name = EMBEDDING_MAP.get(args.embedding, args.embedding)
    freeze = not args.joint

    if args.dynamics_dir:
        dynamics_dir = Path(args.dynamics_dir)
    else:
        dynamics_dir = Path(cfg.paths.checkpoints) / f"dynamics_{args.embedding}"

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = Path(cfg.paths.checkpoints) / f"nbf_{args.embedding}"

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path(cfg.paths.runs) / f"nbf_{args.embedding}"

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        dataset_dir = Path("data/processed/datasets") / emb_name.replace("/", "_")

    # Override config
    if args.eta is not None:
        cfg.training.eta_train = args.eta
    if args.kappa is not None:
        cfg.training.kappa_noninvariant_turns = args.kappa
    if args.lambda_dyn is not None:
        cfg.training.loss_weights["lambda_dyn"] = args.lambda_dyn
    if args.lambda_ce is not None:
        cfg.training.loss_weights["lambda_ce"] = args.lambda_ce
    if args.lambda_ss is not None:
        cfg.training.loss_weights["lambda_ss"] = args.lambda_ss
    if args.lambda_si is not None:
        cfg.training.loss_weights["lambda_si"] = args.lambda_si
    if args.lr is not None:
        cfg.training.stage2_predictor.lr = args.lr
    if args.epochs is not None:
        cfg.training.stage2_predictor.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training_extra.batch_size = args.batch_size
    if args.seed is not None:
        cfg.training_extra.seed = args.seed

    logger.info("Phase 4: NBF Safety Predictor Training")
    logger.info("  Mode: %s", "FROZEN dynamics" if freeze else "JOINT training")
    logger.info("  Dynamics dir: %s", dynamics_dir)
    logger.info("  Dataset dir: %s", dataset_dir)
    logger.info("  Checkpoint dir: %s", checkpoint_dir)
    logger.info("  Run dir: %s", run_dir)
    logger.info("  Epochs: %d, LR: %s, Batch: %d",
                cfg.training.stage2_predictor.epochs,
                cfg.training.stage2_predictor.lr,
                cfg.training_extra.batch_size)
    logger.info("  eta=%s, kappa=%d", cfg.training.eta_train, cfg.training.kappa_noninvariant_turns)
    logger.info("  λ_dyn=%.1f, λ_ce=%.1f, λ_ss=%.1f, λ_si=%.1f",
                cfg.training.loss_weights["lambda_dyn"],
                cfg.training.loss_weights["lambda_ce"],
                cfg.training.loss_weights["lambda_ss"],
                cfg.training.loss_weights["lambda_si"])

    if args.dry_run:
        import torch
        predictor = SafetyPredictor()
        logger.info("[DRY RUN] Predictor params: %d", predictor.param_count())
        logger.info("[DRY RUN] Would train for %d epochs", cfg.training.stage2_predictor.epochs)
        return 0

    import torch
    from guardbound.training.train_nbf import NBFTrainer, load_phase2_dataset, set_seeds

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    set_seeds(cfg.training_extra.seed)

    # Load dynamics
    dynamics = load_dynamics(dynamics_dir, device)

    # Create predictor
    predictor = SafetyPredictor(
        state_dim=cfg.dims.state_dim_m,
        embedding_dim=cfg.dims.embedding_dim_n,
    )
    logger.info("Predictor params: %d", predictor.param_count())

    # Load datasets
    train_U, train_Z, train_Y, train_MASK = load_phase2_dataset(dataset_dir, "train")
    val_U, val_Z, val_Y, val_MASK = load_phase2_dataset(dataset_dir, "val")

    # Train
    trainer = NBFTrainer(
        dynamics=dynamics,
        predictor=predictor,
        config=cfg,
        freeze_dynamics=freeze,
        device=device,
    )

    metrics = trainer.train(
        train_U=train_U, train_Z=train_Z, train_Y=train_Y, train_MASK=train_MASK,
        val_U=val_U, val_Z=val_Z, val_Y=val_Y, val_MASK=val_MASK,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        resume=args.resume,
    )

    logger.info("Training complete. Artifacts saved to:")
    logger.info("  Metrics: %s/metrics.csv", run_dir)
    logger.info("  Loss plot: %s/losses.png", run_dir)
    logger.info("  Sanity report: %s/sanity_report.json", run_dir)
    logger.info("  Checkpoint: %s/checkpoint.pt", checkpoint_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
