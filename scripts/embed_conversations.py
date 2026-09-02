#!/usr/bin/env python
"""Extract sentence embeddings from judged conversations.

Usage:
    python scripts/embed_conversations.py --input data/processed/conversations_labeled.jsonl
    python scripts/embed_conversations.py --input ... --model all-distilroberta-v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.config import load_config
from guardbound.data.embedder import ConversationEmbedder
from guardbound.logging_utils import setup_logging, get_logger
from guardbound.schemas import load_conversations_jsonl, save_conversations_jsonl

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input JSONL path (labeled conversations)")
    parser.add_argument("--output", default=None, help="Output JSONL path (default: <input>_embedded.jsonl)")
    parser.add_argument("--model", default=None, help="Embedding model (default: all-mpnet-base-v2)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: from config)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be executed")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(level=__import__("logging").DEBUG if args.verbose else __import__("logging").INFO)

    cfg = load_config(args.config)

    model_name = args.model or cfg.embeddings.default_model
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else (cfg.embedding_extraction.batch_size if hasattr(cfg, "embedding_extraction") else 64)
    )
    cache_dir = Path(cfg.cache.dir) / "embeddings" if hasattr(cfg, "cache") else Path("data/processed/.cache/embeddings")

    output_path = Path(args.output) if args.output else Path(args.input).with_name(
        Path(args.input).stem + "_embedded.jsonl"
    )

    if args.dry_run:
        logger.info("[DRY RUN] Would embed conversations from %s", args.input)
        logger.info("[DRY RUN] Model: %s, batch size: %d", model_name, batch_size)
        return 0

    # Load conversations
    conversations = load_conversations_jsonl(args.input)
    logger.info("Loaded %d conversations", len(conversations))

    # Create embedder
    embedder = ConversationEmbedder(
        model_name=model_name,
        batch_size=batch_size,
        cache_dir=cache_dir,
    )

    # Embed
    results = embedder.embed_conversations(conversations, dry_run=args.dry_run)

    # Save
    save_conversations_jsonl(conversations, output_path)

    total_cached = sum(r.num_cached for r in results)
    total_computed = sum(r.num_computed for r in results)

    logger.info("Embedding complete:")
    logger.info("  Model: %s", model_name)
    logger.info("  Conversations: %d", len(conversations))
    logger.info("  Cached: %d", total_cached)
    logger.info("  Computed: %d", total_computed)
    logger.info("  Output: %s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
