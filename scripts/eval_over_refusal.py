#!/usr/bin/env python3
"""Over-refusal CLI.

Evaluates a target LLM on XSTest / JBB-Benign / PHTest-Harmless,
with optional NBF steering.  Uses the shared refusal detector.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import _cloud_common as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   choices=["xstest", "jbb_benign", "phtest_harmless"],
                   help="Over-refusal dataset to evaluate.")
    p.add_argument("--data-path", default=None,
                   help="Path to the dataset JSONL.  Defaults to data/raw/<dataset>.jsonl.")
    p.add_argument("--model", default="gpt-3.5-turbo-0125")
    p.add_argument("--barrier-checkpoint", default=None)
    p.add_argument("--dynamics-dir", default=None)
    p.add_argument("--embedding", default="mpnet",
                   choices=["mpnet", "distilroberta"])
    p.add_argument("--eta", type=float, default=1e-3,
                   help="Steering margin η. Default 1e-3 = the paper's GPT-4o "
                        "Table 1 row (was 5e-4 before Phase 29).")
    p.add_argument("--out", required=True)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    common.add_paid_authorization_args(p)
    return p.parse_args()


def make_llm(args):
    if args.mock:
        from guardbound.llm.mock import MockChatLLM
        return MockChatLLM(responses=[
            "Sure, here you go.",
            "I'm sorry, but I cannot help with that.",
        ] * 5000)
    target = args.model.lower()
    if "gpt" in target or target.startswith("o1") or target.startswith("o3"):
        return common.build_paid_chat_llm(args.model, args=args)
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
    from guardbound.evaluation import (
        OVER_REFUSAL_LOADERS,
        evaluate_over_refusal,
    )
    from guardbound.evaluation.readiness import (
        STATUS_BLOCKED,
        check_utility_dataset,
    )
    loader = OVER_REFUSAL_LOADERS[args.dataset]
    path = args.data_path or f"data/raw/{args.dataset}.jsonl"
    if args.dataset == "xstest":
        inventory = check_utility_dataset("xstest", path)
        if inventory["status"] == STATUS_BLOCKED:
            print(f"{STATUS_BLOCKED}: XSTest data not usable at {path} "
                  f"(exists={inventory['exists']}, rows={inventory['rows']}, "
                  f"missing={inventory['missing_fields']})")
            print("  Provide a JSONL with a `prompt` (or `text`) field per line.")
            print("  No dataset is downloaded, fabricated or substituted.")
            raise SystemExit(2)
    examples = loader(path)
    if args.limit:
        examples = examples[: args.limit]
    print(f"Loaded {len(examples)} {args.dataset} examples from {path}")

    llm = make_llm(args)
    barrier, embed_fn = make_barrier_and_embed(args)

    defense = "guardbound" if barrier is not None else "original"

    result = evaluate_over_refusal(
        examples, llm,
        dataset_name=args.dataset,
        defense=defense,
        eta=args.eta if defense == "guardbound" else None,
        out_path=args.out,
    )
    print(f"Total   : {result.total}")
    print(f"Refused : {result.refused}")
    print(f"Rate    : {result.refusal_rate:.4f}")
    print(f"Defense : {result.defense}")
    print(f"Per-example JSONL : {args.out}")
    print(f"Results sidecar   : {Path(args.out).with_suffix('.results.jsonl')}")


if __name__ == "__main__":
    main()
