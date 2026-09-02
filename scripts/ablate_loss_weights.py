#!/usr/bin/env python
"""Ablate loss weights to reproduce Fig. 8 qualitative trends.

Sweeps λ_SS ∈ {10, 100, 1000} and λ_SI ∈ {10, 100, 1000}.
Saves results under runs/ablations/.

Not specified in the paper — ablation implementation details may use local defaults.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.config import load_config
from guardbound.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


def run_ablation(cfg, ablation_dir: Path, weight_name: str, values: list[float],
                 dynamics_dir: Path, dataset_dir: Path, device) -> list[dict]:
    """Run ablation for a single weight."""
    from guardbound.models.dynamics import load_dynamics
    from guardbound.models.predictor import SafetyPredictor
    from guardbound.training.train_nbf import NBFTrainer, load_phase2_dataset, set_seeds

    set_seeds(cfg.training_extra.seed)
    train_U, train_Z, train_Y, train_MASK = load_phase2_dataset(dataset_dir, "train")
    val_U, val_Z, val_Y, val_MASK = load_phase2_dataset(dataset_dir, "val")

    results = []
    for val in values:
        logger.info("Ablating %s = %.1f", weight_name, val)
        cfg_copy = load_config()  # fresh config
        cfg_copy.training.loss_weights[weight_name] = val
        # Copy relevant settings
        cfg_copy.training.stage2_predictor = cfg.training.stage2_predictor
        cfg_copy.training.eta_train = cfg.training.eta_train
        cfg_copy.training.kappa_noninvariant_turns = cfg.training.kappa_noninvariant_turns
        cfg_copy.training_extra = cfg.training_extra
        cfg_copy.dims = cfg.dims

        dynamics = load_dynamics(dynamics_dir, device)
        predictor = SafetyPredictor(
            state_dim=cfg.dims.state_dim_m,
            embedding_dim=cfg.dims.embedding_dim_n,
        )

        out_dir = ablation_dir / f"{weight_name}_{val:.0f}"
        trainer = NBFTrainer(
            dynamics=dynamics, predictor=predictor, config=cfg_copy,
            freeze_dynamics=True, device=device,
        )

        # Train for fewer epochs in ablation (not specified in paper — local default)
        cfg_copy.training.stage2_predictor.epochs = min(cfg.training.stage2_predictor.epochs, 20)

        metrics = trainer.train(
            train_U=train_U, train_Z=train_Z, train_Y=train_Y, train_MASK=train_MASK,
            val_U=val_U, val_Z=val_Z, val_Y=val_Y, val_MASK=val_MASK,
            run_dir=out_dir, checkpoint_dir=out_dir / "ckpt",
        )

        last = metrics[-1]
        results.append({
            "weight": weight_name,
            "value": val,
            "final_train_loss": last.train_total,
            "final_val_loss": last.val_total,
            "val_accuracy": last.val_accuracy,
            "train_ce": last.train_ce,
            "train_ss": last.train_ss,
            "train_si": last.train_si,
        })

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", default="mpnet", choices=["mpnet", "distilroberta"])
    parser.add_argument("--dynamics-dir", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(level=__import__("logging").DEBUG if args.verbose else __import__("logging").INFO)

    cfg = load_config(args.config)
    import torch
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    emb_name = {"mpnet": "all-mpnet-base-v2", "distilroberta": "all-distilroberta-v1"}.get(args.embedding)
    dynamics_dir = Path(args.dynamics_dir) if args.dynamics_dir else Path(cfg.paths.checkpoints) / f"dynamics_{args.embedding}"
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path("data/processed/datasets") / emb_name.replace("/", "_")
    ablation_dir = Path(cfg.paths.runs) / "ablations"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    # Sweep lambda_SS
    logger.info("=== Ablating lambda_SS ===")
    ss_results = run_ablation(cfg, ablation_dir, "lambda_ss", [10, 100, 1000],
                              dynamics_dir, dataset_dir, device)

    # Sweep lambda_SI
    logger.info("=== Ablating lambda_SI ===")
    si_results = run_ablation(cfg, ablation_dir, "lambda_si", [10, 100, 1000],
                              dynamics_dir, dataset_dir, device)

    all_results = ss_results + si_results
    results_path = ablation_dir / "ablation_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Ablation results saved to %s", results_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
