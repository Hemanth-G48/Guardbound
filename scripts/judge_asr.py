#!/usr/bin/env python3
"""ASR CLI.

Resumable, cached GPT-4o judging of multi-turn attack transcripts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True,
                   help="Path to JSONL of conversations (Phase 1 schema).")
    p.add_argument("--judge-model", default="gpt-4o-2024-08-06",
                   help="OpenAI judge model (default: gpt-4o-2024-08-06, paper).")
    p.add_argument("--judge-prompt",
                   default="configs/judge_prompts/asr_judge.txt",
                   help="Path to judge prompt (default: configs/judge_prompts/asr_judge.txt).")
    p.add_argument("--cache-dir",
                   default="data/processed/.cache/judge_asr",
                   help="Cache directory (default: data/processed/.cache/judge_asr).")
    p.add_argument("--out", required=True,
                   help="Output JSONL path (per-conversation verdicts).")
    p.add_argument("--mock", action="store_true",
                   help="Use MockChatLLM instead of an OpenAI client (for tests).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Skip the judge call; report cache coverage only.")
    return p.parse_args()


def make_judge(args: argparse.Namespace):
    if args.mock:
        from guardbound.llm.mock import MockChatLLM
        judge_llm = MockChatLLM(responses=["SAFE"] * 10000)
    else:
        try:
            from guardbound.llm.openai_client import OpenAIChatLLM
        except ImportError as exc:
            raise SystemExit(
                f"OpenAI client unavailable: {exc}.  Use --mock for tests."
            )
        judge_llm = OpenAIChatLLM(model_id=args.judge_model)
    from guardbound.evaluation import ASRJudge
    return ASRJudge(
        judge_llm=judge_llm, prompt_path=args.judge_prompt,
        cache_dir=args.cache_dir,
    )


def load_corpus(path: str) -> list:
    """Load a corpus that may be either Phase 1 Conversation JSONL or
    a flat goal+turns JSONL (with `text` / `goal` / `behavior` keys)."""
    import json
    from guardbound.schemas import (
        Conversation, Turn, load_conversations_jsonl,
    )
    p = Path(path)
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = []
    for r in rows:
        if "turns" in r and "attack_method" in r:
            out.append(Conversation.from_dict(r))
        else:
            # Flat goal-only record; treat as a single-turn placeholder
            # with no response (judge will be called on a 0-turn
            # conversation and produce SAFE by default).
            goal = (
                r.get("goal")
                or r.get("behavior")
                or r.get("text")
                or r.get("prompt")
                or ""
            )
            out.append(Conversation(
                goal=goal,
                attack_method=r.get("attack_method") or r.get("attack") or "unknown",
                target_llm=r.get("target_llm") or r.get("model") or "unknown",
                turns=[Turn(query=t, response=t) for t in r.get("turns", [])],
                max_turns=int(r.get("max_turns", 8)),
            ))
    return out


def main() -> None:
    args = parse_args()
    from guardbound.evaluation import evaluate_asr

    conversations = load_corpus(args.corpus)
    if args.limit:
        conversations = conversations[: args.limit]
    print(f"Loaded {len(conversations)} conversations from {args.corpus}")

    if args.dry_run:
        print("[DRY RUN] Skipping judge calls; cache coverage only.")
        return

    judge = make_judge(args)
    verdicts, aggregate = evaluate_asr(conversations, judge, out_path=args.out)
    print(f"Total behaviors : {aggregate['total_behaviors']}")
    print(f"Successful      : {aggregate['successful_behaviors']}")
    print(f"ASR (overall)   : {aggregate['asr']:.4f}")
    print("ASR by attack   :")
    for atk, sub in aggregate["by_attack"].items():
        print(f"  {atk:20s} total={sub['total']:4d}  ASR={sub['asr']:.4f}")
    print(f"Verdicts        : {args.out}")
    print(f"Results sidecar : {Path(args.out).with_suffix('.results.jsonl')}")


if __name__ == "__main__":
    main()
