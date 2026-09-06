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

GPT-4 Crescendo (paper-style with backtracking):

    python scripts/run_attacks.py \\
        --attack crescendo_paper \\
        --goals data/raw/harmbench/raw_data.jsonl \\
        --target gpt-3.5-turbo-0125 \\
        --attacker gpt-4o \\
        --use-backtracking \\
        --max-turns 8 \\
        --out data/processed/crescendo_gpt4_eval.jsonl

GPT-4 Crescendo with NBF defense:

    python scripts/run_attacks.py \\
        --attack crescendo_paper \\
        --goals data/raw/harmbench/raw_data.jsonl \\
        --target gpt-3.5-turbo-0125 \\
        --attacker gpt-4o \\
        --use-backtracking \\
        --barrier-checkpoint checkpoints/nbf_mpnet/checkpoint.pt \\
        --eta 5e-4 \\
        --max-turns 8 \\
        --out data/processed/crescendo_gpt4_nbf.jsonl

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
            "crescendo", "crescendo_paper", "actor_attack", "opposite_day",
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
    parser.add_argument(
        "--concurrent",
        type=int,
        default=4,
        help="Max concurrent attacks (default: 4)",
    )
    parser.add_argument(
        "--attacker",
        default=None,
        help="Attacker LLM for crescendo_paper (e.g., gpt-4o, gpt-4-turbo)",
    )
    parser.add_argument(
        "--use-backtracking",
        action="store_true",
        help="Use backtracking runner for paper-style attacks (crescendo_paper)",
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

    from pathlib import Path
    target_path = Path(args.target)

    if target_path.exists() and target_path.is_dir():
        from guardbound.llm.local_client import HFLocalChatLLM
        return HFLocalChatLLM(model_id=args.target, device_map="cuda")

    target = args.target.lower()

    if "gpt" in target or "o1" in target or "chatgpt" in target:
        from guardbound.llm.openai_client import OpenAIChatLLM
        return OpenAIChatLLM(model=args.target)

    if "claude" in target or "anthropic" in target:
        from guardbound.llm.anthropic_client import AnthropicChatLLM
        return AnthropicChatLLM(model=args.target)

    if "ollama/" in target:
        from guardbound.llm.ollama_client import OllamaChatLLM
        return OllamaChatLLM(model=args.target.replace("ollama/", ""))

    # Default to local HF model
    from guardbound.llm.local_client import HFLocalChatLLM
    return HFLocalChatLLM(model_id=args.target, device_map="cuda")


def make_attacker_llm(args: argparse.Namespace):
    """Create attacker LLM for crescendo_paper attacks."""
    if args.attacker is None:
        return None

    if args.mock:
        from guardbound.llm.mock import MockChatLLM
        return MockChatLLM()

    attacker = args.attacker.lower()

    if "gpt" in attacker or "o1" in attacker or "chatgpt" in attacker:
        from guardbound.llm.openai_client import OpenAIChatLLM
        return OpenAIChatLLM(model=args.attacker)

    if "claude" in attacker or "anthropic" in attacker:
        from guardbound.llm.anthropic_client import AnthropicChatLLM
        return AnthropicChatLLM(model=args.attacker)

    if "ollama/" in attacker:
        from guardbound.llm.ollama_client import OllamaChatLLM
        return OllamaChatLLM(model=args.attacker.replace("ollama/", ""))

    # Default to OpenAI if it looks like a model name
    from guardbound.llm.openai_client import OpenAIChatLLM
    return OpenAIChatLLM(model=args.attacker)


def make_barrier(args: argparse.Namespace):
    """Create NBF barrier if checkpoint is provided."""
    if args.barrier_checkpoint is None:
        return None

    from guardbound.models.predictor import NeuralBarrierFunction
    from pathlib import Path

    barrier_path = Path(args.barrier_checkpoint)
    if barrier_path.is_file():
        # Single checkpoint file - use adapted format
        dynamics_dir = barrier_path.parent / 'dynamics'
        predictor_path = barrier_path.parent / 'predictor.pt'
    else:
        # Directory format
        dynamics_dir = barrier_path
        predictor_path = barrier_path / 'predictor.pt'

    return NeuralBarrierFunction.load(
        dynamics_dir=dynamics_dir,
        predictor_path=predictor_path,
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

    from guardbound.attacks.registry import get_attack, available_attacks
    from guardbound.attacks.runner import run_attack_batch_parallel
    import asyncio

    print(f"Attack: {args.attack}")
    print(f"Target: {args.target}")
    print(f"Max turns: {args.max_turns}")
    print(f"Temperature: {args.temperature}")
    print(f"Mock: {args.mock}")
    print(f"Concurrent: {args.concurrent}")

    goals_data = load_goals(args.goals, limit=args.limit)
    print(f"Loaded {len(goals_data)} goals from {args.goals}")

    if not goals_data:
        print("No goals found. Exiting.")
        sys.exit(1)

    attack = get_attack(args.attack)
    print(f"Attack instance: {attack.__class__.__name__}")

    target_llm = make_target_llm(args)
    print(f"Target LLM: {target_llm.__class__.__name__}")

    attacker_llm = make_attacker_llm(args)
    if attacker_llm is not None:
        print(f"Attacker LLM: {attacker_llm.__class__.__name__}")
        if hasattr(attack, 'set_attacker_llm'):
            attack.set_attacker_llm(attacker_llm)
    else:
        print("No attacker LLM (using simple Crescendo templates)")

    barrier = make_barrier(args)
    if barrier is not None:
        print(f"NBF barrier loaded from: {args.barrier_checkpoint}")
        print(f"Eta: {args.eta}")
    else:
        print("No NBF barrier (bare-LLM mode)")

    embed_fn = make_embed_fn(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_paper_runner = args.use_backtracking and args.attack in [
        "crescendo_paper", "opposite_day", "actor_attack", "acronym"
    ]

    async def run_all():
        if use_paper_runner:
            from guardbound.attacks.runner import run_attack_with_backtracking_async
            from guardbound.attacks.crescendo_paper import CrescendoAttackPaper
            from guardbound.attacks.opposite_day import OppositeDayAttack
            from guardbound.attacks.actor_attack import ActorAttack
            from guardbound.attacks.acronym import AcronymAttack
            from guardbound.schemas import load_conversations_jsonl
            supported = (CrescendoAttackPaper, OppositeDayAttack, ActorAttack, AcronymAttack)
            if not isinstance(attack, supported):
                raise ValueError(
                    f"--use-backtracking requires one of: crescendo_paper, opposite_day, actor_attack, acronym"
                )

            goals_list = [g.get("goal") or g.get("behavior") or g.get("text") or "" for g in goals_data]
            existing_goals = set()
            if args.resume and out_path.exists():
                try:
                    for c in load_conversations_jsonl(out_path):
                        existing_goals.add(c.goal)
                except Exception:
                    pass

            conversations = []
            for i, goal in enumerate(goals_list):
                if goal in existing_goals:
                    for c in load_conversations_jsonl(out_path):
                        if c.goal == goal:
                            conversations.append(c)
                            break
                    continue

                conv = await run_attack_with_backtracking_async(
                    attack=attack,
                    goal=goal,
                    target_llm=target_llm,
                    embed_fn=embed_fn,
                    barrier=barrier,
                    eta=args.eta,
                    max_turns=args.max_turns,
                    temperature=args.temperature,
                    target_llm_name=args.target,
                    attack_method=args.attack,
                )
                conversations.append(conv)
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(conv.to_json() + "\n")
                logger.info(f"[{i+1}/{len(goals_list)}] {goal[:50]} -> {len(conv.turns)} turns")

            return conversations

        return await run_attack_batch_parallel(
            attack=attack,
            goals=[g.get("goal") or g.get("behavior") or g.get("text") or "" for g in goals_data],
            target_llm=target_llm,
            output_path=out_path,
            embed_fn=embed_fn,
            barrier=barrier,
            eta=args.eta,
            max_turns=args.max_turns,
            temperature=args.temperature,
            target_llm_name=args.target,
            attack_method=args.attack,
            resume=args.resume,
            dry_run=args.dry_run,
            max_concurrent=args.concurrent,
        )

    start_time = time.time()
    results = asyncio.run(run_all())
    elapsed = time.time() - start_time

    print(f"\nDone. {len(results)} conversations in {elapsed:.1f}s")
    print(f"Output: {out_path}")

    if not args.dry_run and results:
        total_turns = sum(len(r.turns) for r in results)
        total_filtered = sum(
            sum(1 for t in r.turns if t.was_filtered)
            for r in results
        )
        print(f"Total turns: {total_turns}")
        print(f"Total filtered: {total_filtered}")


if __name__ == "__main__":
    main()
