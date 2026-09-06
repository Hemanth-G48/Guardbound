#!/usr/bin/env python3
"""Run paper-reproducible experiments with proper methodology.

Key changes from previous experiments:
1. Uses local Llama-3.1-8B-Instruct (not Ollama llama3)
2. Uses 160 Harmbench behaviors (matching paper)
3. Uses LLM judge for ASR evaluation (not keyword matching)
4. NBF applied as Q-filter during generation

Usage:
    # Baseline experiment
    python scripts/run_paper_experiment.py --mode baseline --out results/llama31_baseline.jsonl

    # NBF experiment
    python scripts/run_paper_experiment.py --mode nbf --out results/llama31_nbf.jsonl

    # Evaluate results with LLM judge
    python scripts/judge_asr.py --corpus results/llama31_baseline.jsonl --out results/baseline_judged.jsonl
    python scripts/judge_asr.py --corpus results/llama31_nbf.jsonl --out results/nbf_judged.jsonl

    # Compare results
    python final_analysis.py --baseline results/baseline_judged.jsonl --nbf results/nbf_judged.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run paper-reproducible NBF experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "nbf"],
        help="Experiment mode: baseline (no defense) or nbf (with NBF steering)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path for conversation results",
    )
    parser.add_argument(
        "--goals",
        default="data/raw/harmbench/harmbench_160.jsonl",
        help="Path to goals JSONL (default: 160-behavior subset)",
    )
    parser.add_argument(
        "--target",
        default="models/Llama-3.1-8B-Instruct",
        help="Target model path or name (default: local Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--barrier-checkpoint",
        default=None,
        help="Path to NBF checkpoint (required for NBF mode)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=5e-4,
        help="NBF threshold (default: 5e-4)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Max dialogue turns (default: 8)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of goals (for testing)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=2,
        help="Max concurrent attacks (default: 2, lower for local model)",
    )
    parser.add_argument(
        "--embedding",
        default="mpnet",
        choices=["mpnet", "distilroberta"],
        help="Embedding model for NBF (default: mpnet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("PAPER-REPRODUCIBLE NBF EXPERIMENT")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Target: {args.target}")
    print(f"Goals: {args.goals}")
    print(f"Output: {args.out}")

    if args.mode == "nbf":
        if args.barrier_checkpoint is None:
            print("ERROR: --barrier-checkpoint required for NBF mode")
            sys.exit(1)
        print(f"Barrier: {args.barrier_checkpoint}")
        print(f"Eta: {args.eta}")

    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from scripts.run_attacks import load_goals, make_target_llm, make_barrier, make_embed_fn
    from guardbound.attacks.registry import get_attack
    from guardbound.attacks.runner import run_attack_batch_parallel
    import asyncio

    goals_data = load_goals(args.goals, limit=args.limit)
    print(f"\nLoaded {len(goals_data)} goals")

    attack = get_attack("crescendo")
    target_llm = make_target_llm(argparse.Namespace(
        mock=False,
        target=args.target,
    ))
    barrier = make_barrier(argparse.Namespace(barrier_checkpoint=args.barrier_checkpoint)) if args.mode == "nbf" else None
    embed_fn = make_embed_fn(argparse.Namespace(barrier_checkpoint=args.barrier_checkpoint, mock=False, embedding=args.embedding)) if args.mode == "nbf" else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async def run_all():
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
            attack_method="crescendo",
            resume=True,
            dry_run=args.dry_run,
            max_concurrent=args.concurrent,
        )

    import time
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

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    if args.mode == "baseline":
        print("1. Evaluate with LLM judge:")
        print(f"   python scripts/judge_asr.py --corpus {args.out} --out results/baseline_judged.jsonl")
    else:
        print("1. Evaluate with LLM judge:")
        print(f"   python scripts/judge_asr.py --corpus {args.out} --out results/nbf_judged.jsonl")
        print("2. Compare results after running both baseline and NBF:")
        print("   python final_analysis.py --baseline results/baseline_judged.jsonl --nbf results/nbf_judged.jsonl")


if __name__ == "__main__":
    main()
