#!/usr/bin/env python3
"""MTBench CLI.

Evaluates a target LLM on an MTBench-shaped JSONL dataset, with
optional NBF steering (B.1 refusal-replacement rule).
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
    p.add_argument("--model", default="gpt-3.5-turbo-0125")
    p.add_argument("--dataset", required=True,
                   help="Path to MTBench-shaped JSONL (question, turn2_question?, ...).")
    p.add_argument("--judge-model", default="gpt-4o-2024-08-06",
                   help="LLM judge model (default: gpt-4o-2024-08-06).")
    p.add_argument("--judge-prompt", default=None,
                   help="Optional path to a custom judge prompt.")
    p.add_argument("--barrier-checkpoint", default=None)
    p.add_argument("--dynamics-dir", default=None)
    p.add_argument("--embedding", default="mpnet",
                   choices=["mpnet", "distilroberta"])
    p.add_argument("--eta", type=float, default=5e-4)
    p.add_argument("--cache-dir",
                   default="data/processed/.cache/judge_mtbench")
    p.add_argument("--out", required=True)
    p.add_argument("--mock", action="store_true",
                   help="Use MockChatLLM for both target and judge.")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def make_llm(args, judge: bool = False):
    if args.mock:
        from guardbound.llm.mock import MockChatLLM
        return MockChatLLM(responses=[
            "Some helpful answer.\nA",
            "5\nA good answer.",
        ] * 5000)
    target = (args.judge_model if judge else args.model).lower()
    if "gpt" in target or target.startswith("o1") or target.startswith("o3"):
        from guardbound.llm.openai_client import OpenAIChatLLM
        return OpenAIChatLLM(model_id=(args.judge_model if judge else args.model))
    if "claude" in target or "anthropic" in target:
        from guardbound.llm.anthropic_client import AnthropicChatLLM
        return AnthropicChatLLM(model=(args.judge_model if judge else args.model))
    from guardbound.llm.local_client import HFLocalChatLLM
    return HFLocalChatLLM(model_id=(args.judge_model if judge else args.model))


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
    from guardbound.evaluation import load_mtbench_jsonl, evaluate_mtbench

    examples = load_mtbench_jsonl(args.dataset)
    if args.limit:
        examples = examples[: args.limit]
    print(f"Loaded {len(examples)} MTBench examples from {args.dataset}")

    target_llm = make_llm(args, judge=False)
    judge_llm = make_llm(args, judge=True)
    barrier, embed_fn = make_barrier_and_embed(args)

    result = evaluate_mtbench(
        examples, target_llm, judge_llm,
        barrier=barrier, embed_fn=embed_fn,
        eta=args.eta,
        prompt_path=args.judge_prompt,
        cache_dir=args.cache_dir,
        out_path=args.out,
    )
    print(f"n          : {result.n}")
    print(f"Mean score : {result.mean_score:.4f}")
    print(f"Turn 1 mean: {result.turn1_mean:.4f}")
    print(f"Turn 2 mean: {result.turn2_mean:.4f}")
    print(f"Defense    : {result.defense}")
    print(f"Per-example JSONL : {args.out}")
    print(f"Results sidecar   : {Path(args.out).with_suffix('.results.jsonl')}")


if __name__ == "__main__":
    main()
