#!/usr/bin/env python
"""Convert paper's conversation data to Guardbound training format.

Usage:
    python scripts/convert_paper_data.py --input "nbf_original_stuff/supplementary files/TMLR_supp_code/data/LoRA_SFT/gpt-3-5-turbo_circuit_breakers_1k_all_attacks.json"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.schemas import Conversation, Turn


def convert_to_guardbound_format(data: list[dict]) -> list[Conversation]:
    """Convert paper format to Guardbound Conversation format.

    Paper format:
        {"instruction": "...", "messages": [{"role": "user/assistant", "content": "..."}]}

    Guardbound format:
        {"goal": "...", "attack_method": "...", "target_llm": "...", "turns": [...]}
    """
    conversations = []

    for item in data:
        instruction = item.get("instruction", "")
        messages = item.get("messages", [])

        if not instruction or not messages:
            continue

        turns = []
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages):
                user_msg = messages[i] if messages[i].get("role") == "user" else None
                assistant_msg = messages[i + 1] if messages[i + 1].get("role") == "assistant" else None

                if user_msg and assistant_msg:
                    turn = Turn(
                        query=user_msg.get("content", ""),
                        response=assistant_msg.get("content", ""),
                        was_filtered=False,
                    )
                    turns.append(turn)

        conv = Conversation(
            goal=instruction,
            attack_method="crescendo",
            target_llm="gpt-3.5-turbo",
            turns=turns,
            max_turns=8,
        )
        conversations.append(conv)

    return conversations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSON file from paper")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of conversations")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_guardbound.jsonl"
    )

    print(f"Loading data from {input_path}...")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} conversations")

    if args.limit:
        data = data[:args.limit]
        print(f"Limited to {len(data)} conversations")

    print("Converting to Guardbound format...")
    conversations = convert_to_guardbound_format(data)

    print(f"Converted {len(conversations)} conversations")

    print(f"Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(conv.to_json() + "\n")

    print(f"Done! Saved {len(conversations)} conversations to {output_path}")

    turns_dist = {}
    for conv in conversations:
        n = conv.num_turns
        turns_dist[n] = turns_dist.get(n, 0) + 1

    print("\nTurns distribution:")
    for n in sorted(turns_dist.keys()):
        print(f"  {n} turns: {turns_dist[n]} conversations")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
