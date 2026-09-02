#!/usr/bin/env python
"""Generate multi-turn attack conversations.

Usage:
    python scripts/build_conversations.py --attack acronym --num-goals 1000
    python scripts/build_conversations.py --attack crescendo --dry-run --num-goals 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.config import load_config
from guardbound.data.attack_runner import AttackRunner
from guardbound.data.sources import load_circuit_breakers, load_harmbench
from guardbound.attacks import get_attack, ATTACK_PROVENANCE
from guardbound.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--attack", required=True,
        choices=["crescendo", "opposite_day", "actor_attack", "acronym"],
        help="Attack method to generate",
    )
    parser.add_argument(
        "--target-model", default=None,
        help="Target LLM model (default: from config)",
    )
    parser.add_argument(
        "--num-goals", type=int, default=1000,
        help="Number of goals to generate conversations for (default: 1000)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=None,
        help="Max turns per conversation (default: from config)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Generation temperature (default: from config)",
    )
    parser.add_argument(
        "--input-goals", type=str, default=None,
        help="Path to JSON file with goals (default: auto-load from sources)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (default: data/processed/conversations/<attack>.jsonl)",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Configuration file (default: configs/default.yaml)",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume", action="store_true", default=True,
        help="Resume from existing output (default: True)",
    )
    resume_group.add_argument(
        "--no-resume", action="store_true",
        help="Do not resume; start fresh",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for goal selection (default: from config)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be executed without making API calls",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    # Validate --num-goals
    if args.num_goals <= 0:
        parser.error("--num-goals must be a positive integer")

    # Validate --max-turns
    if args.max_turns is not None and args.max_turns < 1:
        parser.error("--max-turns must be at least 1")

    # Validate --temperature
    if args.temperature is not None and args.temperature < 0:
        parser.error("--temperature must be non-negative")

    setup_logging(level=__import__("logging").DEBUG if args.verbose else __import__("logging").INFO)

    # Load config
    cfg = load_config(args.config)

    # Resolve settings
    target_model = args.target_model or cfg.llm.openai.get("gpt35_turbo", "gpt-3.5-turbo-0125")
    max_turns = (
        args.max_turns
        if args.max_turns is not None
        else cfg.dialogue.max_turns_k
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else cfg.dialogue.temperature
    )
    seed = args.seed if args.seed is not None else cfg.training_extra.seed

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = cfg.paths.data_processed / "conversations" / f"{args.attack}.jsonl"

    # Cache dir
    cache_dir = Path(cfg.cache.dir) if hasattr(cfg, "cache") else cfg.paths.data_processed / ".cache"

    # Load goals
    import json
    if args.input_goals:
        goals = json.loads(Path(args.input_goals).read_text())
    else:
        goals, _ = load_circuit_breakers(
            cfg.paths.data_raw,
            num_goals=args.num_goals,
            seed=seed,
        )

    # Validate goals is a list/sequence
    if not isinstance(goals, (list, tuple)):
        logger.error("Goals must be a list or tuple, got %s", type(goals).__name__)
        return 1

    # Remove empty goals
    original_count = len(goals)
    goals = [g for g in goals if g and isinstance(g, str) and g.strip()]
    if len(goals) < original_count:
        logger.warning("Removed %d empty/invalid goals", original_count - len(goals))

    if not goals:
        logger.error("No usable goals loaded after filtering empty values")
        return 1

    goals = goals[:args.num_goals]

    # Show provenance
    prov = ATTACK_PROVENANCE.get(args.attack, {})
    logger.info("Attack: %s", prov.get("method_name", args.attack))
    logger.info("Implementation status: %s", prov.get("implementation_status", "unknown"))
    logger.info("Original paper: %s", prov.get("original_paper", "unknown"))

    if args.dry_run:
        logger.info("[DRY RUN] Would generate %d conversations", len(goals))
        logger.info("[DRY RUN] Target model: %s", target_model)
        logger.info("[DRY RUN] Output: %s", output_path)
        logger.info("[DRY RUN] Goals: %s", [g[:40] for g in goals[:5]])
        return 0

    # Create attack adapter
    attack = get_attack(args.attack)

    # Create LLM client
    from guardbound.llm import OpenAIChatLLM
    llm = OpenAIChatLLM(target_model)

    # Resolve resume flag: --resume/--no-resume are mutually exclusive
    # --resume defaults to True, so resume is enabled unless --no-resume is set
    resume_flag = args.resume and not args.no_resume

    # Create runner
    runner = AttackRunner(
        attack=attack,
        target_llm=llm,
        output_path=output_path,
        cache_dir=cache_dir / "generation",
        target_model=target_model,
        temperature=temperature,
        max_turns=max_turns,
        retry_max=cfg.attack.retry_max if hasattr(cfg, "attack") else 3,
        retry_backoff_base=cfg.attack.retry_backoff_base if hasattr(cfg, "attack") else 2.0,
        resume=resume_flag,
    )

    # Generate
    results = runner.generate_batch(goals, dry_run=args.dry_run)

    # Report
    successes = sum(1 for r in results if r.error is None)
    failures = sum(1 for r in results if r.error is not None)
    cached = sum(1 for r in results if r.from_cache)

    logger.info("Generation complete:")
    logger.info("  Total: %d", len(results))
    logger.info("  Success: %d", successes)
    logger.info("  Cached: %d", cached)
    logger.info("  Failed: %d", failures)
    logger.info("  Output: %s", output_path)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
