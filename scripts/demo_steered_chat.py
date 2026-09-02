#!/usr/bin/env python
"""Demo: SteeredLLMChat with NBF Q-filter defense.

Usage:
    python scripts/demo_steered_chat.py --mock
    python scripts/demo_steered_chat.py --dynamics-dir checkpoints/dynamics_mpnet --predictor-path checkpoints/nbf_mpnet/predictor_h.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from guardbound.config import load_config
from guardbound.llm.mock import MockChatLLM
from guardbound.logging_utils import setup_logging, get_logger
from guardbound.models.dynamics import DialogueDynamics
from guardbound.models.predictor import SafetyPredictor, NeuralBarrierFunction
from guardbound.defense.steered_chat import SteeredLLMChat, SteeredResult

logger = get_logger(__name__)


def _print_turn(result: SteeredResult, query: str) -> None:
    """Pretty-print a single turn."""
    print("-" * 60)
    print(f"Turn {result.turn_index + 1}")
    print(f"Query: {query}")
    print(f"h(x,u): {result.h_value:.5f}")
    print(f"Decision: {'FILTERED' if result.filtered else 'ALLOWED'}")
    print(f"Response: {result.response[:80] + '...' if result.response and len(result.response) > 80 else result.response}")
    print(f"State norm: {result.state_norm:.4f}")
    if result.refusal:
        print("  [LLM REFUSAL DETECTED]")
    print("-" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Use mock models (offline)")
    parser.add_argument("--dynamics-dir", default=None)
    parser.add_argument("--predictor-path", default=None)
    parser.add_argument("--eta", type=float, default=5e-4)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(level=__import__("logging").DEBUG if args.verbose else __import__("logging").INFO)

    if args.mock:
        print("=" * 60)
        print("NBF Safety Steering — Offline Demo (Mock Mode)")
        print("=" * 60)
        print()

        dynamics = DialogueDynamics(hidden_dims=[32, 32])
        predictor = SafetyPredictor()
        barrier = NeuralBarrierFunction(dynamics, predictor)
        barrier.eval()

        target_llm = MockChatLLM(responses=[
            "Here is information about that topic...",
            "The process involves several steps...",
            "I can explain how this works...",
            "Let me provide details on this...",
            "This is a complex subject with many aspects...",
        ])

        embed_fn = lambda text: torch.randn(1, 768)  # mock embeddings

    else:
        if not args.dynamics_dir or not args.predictor_path:
            parser.error("--dynamics-dir and --predictor-path required for real mode")

        barrier = NeuralBarrierFunction.load(
            Path(args.dynamics_dir), Path(args.predictor_path),
        )

        from guardbound.llm import OpenAIChatLLM
        target_llm = OpenAIChatLLM("gpt-3.5-turbo-0125")

        from guardbound.embeddings import SentenceEmbedder
        emb = SentenceEmbedder("all-mpnet-base-v2")
        embed_fn = lambda text: torch.from_numpy(emb.embed([text])).float()

    # Create steered chat
    chat = SteeredLLMChat(
        target_llm=target_llm,
        barrier=barrier,
        eta=args.eta,
        max_turns=args.max_turns,
    )

    print(f"eta: {args.eta}")
    print(f"max_turns: {args.max_turns}")
    print()

    # Demo queries
    queries = [
        "Can you explain the general concept of machine learning?",
        "What are common techniques used in cybersecurity?",
        "How do encryption algorithms work in practice?",
        "Can you provide a step-by-step guide for network analysis?",
        "What tools are commonly used for system administration?",
    ]

    for query in queries:
        result = chat.chat(query, embed_fn)
        _print_turn(result, query)
        print()

    print("=" * 60)
    print(f"Total turns: {chat.turn_index}")
    print(f"Accepted: {chat.accepted_turn_index}")
    print(f"Filtered: {chat.turn_index - chat.accepted_turn_index}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
