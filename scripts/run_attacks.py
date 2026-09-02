#!/usr/bin/env python3
"""CLI for running multi-turn attacks with optional NBF steering.

Supports both bare-LLM and NBF-steered modes.

Examples
--------
Bare LLM (no defense):

    python scripts/run_attacks.py \\
        --attack crescendo \\
        --goals data/raw/harmbench/raw_data.jsonl \\
        --target gpt-3.5-turbo-0125 \\
        --max-turns 8 \\
        --out data/processed/crescendo_eval.jsonl

With NBF defense:

    python scripts/run_attacks.py \\
        --attack crescendo \\
        --goals data/raw/harmbench/raw_data.jsonl \\
        --target gpt-3.5-turbo-0125 \\
        --barrier-checkpoint checkpoints/nbf_mpnet/checkpoint.pt \\
        --eta 5e-4 \\
        --max-turns 8 \\
        --out data/processed/crescendo_nbf.jsonl

Offline (mock):

    python scripts/run_attacks.py \\
        --attack crescendo \\
        --goals data/raw/harmbench/raw_data.jsonl \\
        --mock \\
        --max-turns 3 \\
        --out data/processed/crescendo_mock.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-turn attacks with optional NBF steering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--attack",
        required=True,
        choices=[
            "crescendo", "actor_attack", "opposite_day",
            "acronym", "red_queen", "adaptive",
        ],
        help="Attack method to use",
    )
    parser.add_argument(
        "--goals",
        required=True,
        help="Path to JSONL file with goals (one 'goal' field per line)",
    )
    parser.add_argument(
        "--target",
        default="gpt-3.5-turbo-0125",
        help="Target LLM model name (default: gpt-3.5-turbo-0125)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--barrier-checkpoint",
        default=None,
        help="Path to NBF checkpoint directory (omit for bare-LLM mode)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="NBF threshold (default: 0.0 = no filtering)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Max dialogue turns per conversation (default: 8)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)",
    )
    parser.add_argument(
        "--embedding",
        default="mpnet",
        choices=["mpnet", "distilroberta"],
        help="Embedding model for NBF (default: mpnet)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockChatLLM for offline testing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of goals to process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip already-completed conversations (default: True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without making API calls",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def load_goals(path: str, limit: int | None = None) -> list[dict]:
    """Load goals from JSONL file."""
    goals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            goals.append(record)
            if limit and len(goals) >= limit:
                break
    return goals


def make_target_llm(args: argparse.Namespace):
    """Create target LLM based on CLI args."""
    if args.mock:
        from guardbound.llm.mock import MockChatLLM
        return MockChatLLM()

    # Try configured LLM backends
    target = args.target.lower()

    if "gpt" in target or "o1" in target:
        from guardbound.llm.openai_client import OpenAIChatLLM
        return OpenAIChatLLM(model=args.target)

    if "claude" in target or "anthropic" in target:
        from guardbound.llm.anthropic_client import AnthropicChatLLM
        return AnthropicChatLLM(model=args.target)

    # Default to local
    from guardbound.llm.local_client import LocalChatLLM
    return LocalChatLLM(model_name=args.target)


def make_barrier(args: argparse.Namespace):
    """Create NBF barrier if checkpoint is provided."""
    if args.barrier_checkpoint is None:
        return None

    from guardbound.models.predictor import NeuralBarrierFunction
    return NeuralBarrierFunction.load(
        dynamics_dir=args.barrier_checkpoint,
        embedding_model=args.embedding,
    )


def make_embed_fn(args: argparse.Namespace):
    """Create embedding function for NBF mode.

    Returns ``None`` when no NBF is in use, or in --mock mode (the
    barrier is a stub there, so we just embed with a fixed-dim random
    function for the mocked path).
    """
    if args.barrier_checkpoint is None:
        return None
    if args.mock:
        # Use a deterministic-dim placeholder so the NBF path is exercised.
        import torch
        return lambda text: torch.zeros(1, 768)
    from guardbound.embeddings import get_embed_fn
    return get_embed_fn(args.embedding)


def main() -> None:
    args = parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Import attacks
    from guardbound.attacks.registry import get_attack, available_attacks
    from guardbound.attacks.runner import run_attack

    print(f"Attack: {args.attack}")
    print(f"Target: {args.target}")
    print(f"Max turns: {args.max_turns}")
    print(f"Temperature: {args.temperature}")
    print(f"Mock: {args.mock}")

    # Load goals
    goals_data = load_goals(args.goals, limit=args.limit)
    print(f"Loaded {len(goals_data)} goals from {args.goals}")

    if not goals_data:
        print("No goals found. Exiting.")
        sys.exit(1)

    # Create attack
    attack = get_attack(args.attack)
    print(f"Attack instance: {attack.__class__.__name__}")

    # Create target LLM
    target_llm = make_target_llm(args)
    print(f"Target LLM: {target_llm.__class__.__name__}")

    # Create barrier
    barrier = make_barrier(args)
    if barrier is not None:
        print(f"NBF barrier loaded from: {args.barrier_checkpoint}")
        print(f"Eta: {args.eta}")
    else:
        print("No NBF barrier (bare-LLM mode)")

    # Create embed function (only used in NBF mode)
    embed_fn = make_embed_fn(args)

    # Prepare output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing for resume
    existing_goals: set[str] = set()
    if args.resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        existing_goals.add(rec.get("goal", ""))
                    except json.JSONDecodeError:
                        pass
        print(f"Resume: {len(existing_goals)} existing conversations found")

    # Run attacks
    from guardbound.schemas import Conversation

    results: list[dict] = []
    start_time = time.time()

    for i, goal_record in enumerate(goals_data):
        goal_text = (
            goal_record.get("goal")
            or goal_record.get("behavior")
            or goal_record.get("text")
            or ""
        )

        # Resume check
        if goal_text in existing_goals:
            print(f"[{i+1}/{len(goals_data)}] SKIP (already completed): {goal_text[:50]}...")
            continue

        print(f"[{i+1}/{len(goals_data)}] Attacking: {goal_text[:50]}...")

        if args.dry_run:
            print(f"  [DRY RUN] Would run {args.attack} with max_turns={args.max_turns}")
            continue

        try:
            conv = run_attack(
                attack=attack,
                goal=goal_text,
                target_llm=target_llm,
                barrier=barrier,
                embed_fn=embed_fn,
                eta=args.eta,
                max_turns=args.max_turns,
                temperature=args.temperature,
                attack_method=args.attack,
                target_llm_name=args.target,
            )

            # Record
            result = conv.to_dict()
            results.append(result)

            # Append to output file
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            # Print summary
            n_turns = len(conv.turns)
            n_filtered = sum(1 for t in conv.turns if t.was_filtered)
            print(f"  -> {n_turns} turns, {n_filtered} filtered")

        except Exception as exc:
            print(f"  -> ERROR: {exc}")

    elapsed = time.time() - start_time
    print(f"\nDone. {len(results)} conversations in {elapsed:.1f}s")
    print(f"Output: {out_path}")

    if not args.dry_run and results:
        # Summary
        total_turns = sum(len(r.get("turns", [])) for r in results)
        total_filtered = sum(
            sum(1 for t in r.get("turns", []) if t.get("was_filtered", False))
            for r in results
        )
        print(f"Total turns: {total_turns}")
        print(f"Total filtered: {total_filtered}")


if __name__ == "__main__":
    main()
