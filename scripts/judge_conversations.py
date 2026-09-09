#!/usr/bin/env python
"""Judge conversations using GPT-4o safety scoring.

Usage:
    python scripts/judge_conversations.py --input data/processed/conversations/acronym.jsonl
    python scripts/judge_conversations.py --input ... --output ... --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.config import load_config
from guardbound.data.judge import SafetyJudge
from guardbound.logging_utils import setup_logging, get_logger
from guardbound.schemas import Conversation, Turn, load_conversations_jsonl, save_conversations_jsonl

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", default=None, help="Output JSONL path (default: <input>_labeled.jsonl)")
    parser.add_argument("--judge-model", default=None, help="Judge model (default: from config)")
    parser.add_argument("--prompt", default=None, help="Judge prompt file (default: from config)")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true", default=True, help="Resume from existing output (default: True)")
    resume_group.add_argument("--no-resume", action="store_true", help="Do not resume")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of conversations to judge")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be executed")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--provider", default="openai", choices=["openai", "ollama"],
                        help="Model provider: openai (cloud) or ollama (local)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(level=__import__("logging").DEBUG if args.verbose else __import__("logging").INFO)

    cfg = load_config(args.config)

    judge_model = args.judge_model or cfg.llm.openai.get("gpt4o", "gpt-4o-2024-08-06")
    prompt_path = args.prompt or (cfg.judge.prompt_path if hasattr(cfg, "judge") else "configs/judge_prompt.txt")
    cache_dir = Path(cfg.cache.dir) / "judge" if hasattr(cfg, "cache") else Path("data/processed/.cache/judge")

    output_path = Path(args.output) if args.output else Path(args.input).with_name(
        Path(args.input).stem + "_labeled.jsonl"
    )

    resume_flag = args.resume and not args.no_resume

    # Load conversations
    conversations = load_conversations_jsonl(args.input)
    if args.limit:
        conversations = conversations[:args.limit]
    logger.info("Loaded %d conversations from %s", len(conversations), args.input)

    # Load existing judged conversations for resume
    existing_judged: dict[str, Conversation] = {}
    if resume_flag and output_path.exists():
        try:
            for c in load_conversations_jsonl(output_path):
                existing_judged[c.conversation_id] = c
            logger.info("Loaded %d previously judged conversations for resume", len(existing_judged))
        except Exception as exc:
            logger.warning("Could not load existing judged file: %s", exc)

    if args.dry_run:
        logger.info("[DRY RUN] Would judge %d conversations", len(conversations))
        return 0

    # Create judge
    from guardbound.llm import make_chat_llm
    if args.provider == "ollama":
        judge_llm = make_chat_llm({"provider": "ollama", "model": judge_model})
    else:
        judge_llm = make_chat_llm({"provider": "openai", "model": judge_model})
    logger.info("Judge LLM: %s (provider: %s)", judge_model, args.provider)

    judge = SafetyJudge(
        judge_llm=judge_llm,
        judge_model=judge_model,
        prompt_path=prompt_path,
        cache_dir=cache_dir,
        retry_max=cfg.judge.retry_max if hasattr(cfg, "judge") else 3,
        retry_backoff_base=cfg.judge.retry_backoff_base if hasattr(cfg, "judge") else 2.0,
    )

    # Judge conversations
    judged_conversations: list[Conversation] = []
    total_turns = 0
    judged_turns = 0

    for idx, conv in enumerate(conversations):
        conv_id = conv.conversation_id

        # Check if already judged (resume)
        if conv_id in existing_judged:
            existing = existing_judged[conv_id]
            # Copy existing scores to current conversation
            for i, turn in enumerate(conv.turns):
                if i < len(existing.turns) and existing.turns[i].judge_score is not None:
                    turn.judge_score = existing.turns[i].judge_score
            judged_conversations.append(conv)
            conv_turns = len(conv.turns)
            conv_judged = sum(1 for t in conv.turns if t.judge_score is not None)
            judged_turns += conv_judged
            total_turns += conv_turns
            if conv_judged < conv_turns:
                logger.debug("Resume: %d/%d turns already judged for conversation %d", conv_judged, conv_turns, idx)
            else:
                logger.debug("Resume: all %d turns already judged for conversation %d", conv_turns, idx)
            continue

        # Judge each turn
        for turn in conv.turns:
            total_turns += 1
            if turn.judge_score is not None:
                judged_turns += 1
                continue
            score = judge.judge_turn(
                goal=conv.goal,
                query=turn.query,
                response=turn.response or "",
                dry_run=args.dry_run,
            )
            turn.judge_score = score
            if score is not None:
                judged_turns += 1

        judged_conversations.append(conv)

        if (idx + 1) % 10 == 0:
            coverage = judged_turns / total_turns if total_turns > 0 else 0.0
            logger.info("Judged %d/%d conversations (%d/%d turns, %.1f%% coverage)",
                       idx + 1, len(conversations), judged_turns, total_turns, coverage * 100)
            # Save intermediate results
            save_conversations_jsonl(judged_conversations, output_path)

    # Final save
    save_conversations_jsonl(judged_conversations, output_path)

    coverage = judged_turns / total_turns if total_turns > 0 else 0.0
    logger.info("Judging complete:")
    logger.info("  Conversations: %d", len(judged_conversations))
    logger.info("  Total turns: %d", total_turns)
    logger.info("  Judged turns: %d", judged_turns)
    logger.info("  Coverage: %.1f%%", coverage * 100)
    logger.info("  Output: %s", output_path)

    if coverage < 0.95:
        logger.warning("Judge coverage %.1f%% is below 95%% threshold", coverage * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
