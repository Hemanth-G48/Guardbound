#!/usr/bin/env python3
"""MMLU CLI.

Evaluates a target LLM on an MMLU-shaped JSONL dataset, with optional
NBF steering post-filtering.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt-3.5-turbo-0125",
                   help="Target LLM model id (default: gpt-3.5-turbo-0125).")
    p.add_argument("--dataset", required=True,
                   help="Path to MMLU-shaped JSONL (question, choices, answer).")
    p.add_argument("--barrier-checkpoint", default=None,
                   help="Path to NBF checkpoint (omit for bare-LLM mode).")
    p.add_argument("--dynamics-dir", default=None,
                   help="Path to dynamics checkpoint directory (paired with --barrier-checkpoint).")
    p.add_argument("--embedding", default="mpnet",
                   choices=["mpnet", "distilroberta"],
                   help="Embedding model (default: mpnet).")
    p.add_argument("--eta", type=float, default=5e-4,
                   help="Steering threshold (default: 5e-4, paper recommended).")
    p.add_argument("--out", required=True,
                   help="Output JSONL path for per-question results.")
    p.add_argument("--mock", action="store_true",
                   help="Use MockChatLLM (for tests).")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def make_llm(args):
    if args.mock:
        from guardbound.llm.mock import MockChatLLM
        return MockChatLLM(responses=[
            "A\nBecause the answer is A.",
            "B\nBecause the answer is B.",
        ] * 5000)
    target = args.model.lower()
    if "gpt" in target or target.startswith("o1") or target.startswith("o3"):
        from guardbound.llm.openai_client import OpenAIChatLLM
        return OpenAIChatLLM(model_id=args.model)
    if "claude" in target or "anthropic" in target:
        from guardbound.llm.anthropic_client import AnthropicChatLLM
        return AnthropicChatLLM(model=args.model)
    from guardbound.llm.local_client import HFLocalChatLLM
    return HFLocalChatLLM(model_id=args.model)


def make_barrier_and_embed(args):
    if not args.barrier_checkpoint:
        return None, None
    from guardbound.models.predictor import NeuralBarrierFunction
    from guardbound.embeddings import get_embed_fn
    barrier = NeuralBarrierFunction.load(
        dynamics_dir=args.dynamics_dir or args.barrier_checkpoint,
        predictor_path=args.barrier_checkpoint,
    )
    embed_fn = get_embed_fn(args.embedding)
    return barrier, embed_fn


def main() -> None:
    args = parse_args()
    from guardbound.evaluation import load_mmlu_jsonl, evaluate_mmlu

    questions = load_mmlu_jsonl(args.dataset)
    if args.limit:
        questions = questions[: args.limit]
    print(f"Loaded {len(questions)} MMLU questions from {args.dataset}")

    llm = make_llm(args)
    barrier, embed_fn = make_barrier_and_embed(args)

    result = evaluate_mmlu(
        questions, llm,
        barrier=barrier, embed_fn=embed_fn,
        eta=args.eta,
        out_path=args.out,
    )
    print(f"Total   : {result.total}")
    print(f"Correct : {result.correct}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Defense : {result.defense}")
    if result.eta is not None:
        print(f"eta     : {result.eta}")
    print(f"Per-question JSONL : {args.out}")
    print(f"Results sidecar    : {Path(args.out).with_suffix('.results.jsonl')}")


if __name__ == "__main__":
    main()
