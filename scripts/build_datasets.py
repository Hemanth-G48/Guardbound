#!/usr/bin/env python
"""Build tensor datasets (U/Z/Y/MASK) from embedded conversations.

Usage:
    python scripts/build_datasets.py --input data/processed/conversations_embedded.jsonl
    python scripts/build_datasets.py --input ... --model all-distilroberta-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.config import load_config
from guardbound.data.dataset import package_dataset
from guardbound.logging_utils import setup_logging, get_logger
from guardbound.schemas import load_conversations_jsonl

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input JSONL path (embedded conversations)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: from config)")
    parser.add_argument("--model", default=None, help="Embedding model name (default: all-mpnet-base-v2)")
    parser.add_argument("--train-ratio", type=float, default=None, help="Train ratio (default: 0.9)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be executed")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(level=__import__("logging").DEBUG if args.verbose else __import__("logging").INFO)

    cfg = load_config(args.config)

    model_name = args.model or cfg.embeddings.default_model
    train_ratio = args.train_ratio if args.train_ratio is not None else cfg.data.train_val_split
    seed = args.seed if args.seed is not None else cfg.training_extra.seed

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(cfg.dataset.output_dir if hasattr(cfg, "dataset") else "data/processed/datasets")
        output_dir = output_dir / model_name.replace("/", "_")

    if args.dry_run:
        logger.info("[DRY RUN] Would build dataset from %s", args.input)
        logger.info("[DRY RUN] Output: %s", output_dir)
        return 0

    # Load conversations
    conversations = load_conversations_jsonl(args.input)
    logger.info("Loaded %d conversations", len(conversations))

    # Compute config hash for manifest (deterministic, includes all dataset-generation params)
    config_hash = hashlib.sha256(
        json.dumps({
            "model_name": model_name,
            "train_ratio": train_ratio,
            "seed": seed,
            "max_turns": cfg.dialogue.max_turns_k,
            "embedding_dim": cfg.dims.embedding_dim_n,
            "judge_model": cfg.llm.openai.get("gpt4o", "gpt-4o-2024-08-06"),
            "temperature": cfg.dialogue.temperature,
        }, sort_keys=True).encode()
    ).hexdigest()[:16]

    # Package
    manifest = package_dataset(
        conversations=conversations,
        output_dir=output_dir,
        embedding_model=model_name,
        max_turns=cfg.dialogue.max_turns_k,
        embedding_dim=cfg.dims.embedding_dim_n,
        train_ratio=train_ratio,
        seed=seed,
        judge_model=cfg.llm.openai.get("gpt4o", "gpt-4o-2024-08-06"),
        temperature=cfg.dialogue.temperature,
        config_hash=config_hash,
    )

    logger.info("Dataset built:")
    logger.info("  Conversations: %d", manifest.num_conversations)
    logger.info("  Train: %d, Val: %d", manifest.train_count, manifest.validation_count)
    logger.info("  Judge coverage: %.1f%%", manifest.judge_coverage * 100)
    logger.info("  Attack counts: %s", manifest.attack_counts)
    logger.info("  Success rates: %s", manifest.attack_success_rates)
    logger.info("  Output: %s", output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
