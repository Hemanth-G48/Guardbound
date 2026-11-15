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

import _cloud_common as common  # noqa: E402


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
    common.add_paid_authorization_args(p)
    return p.parse_args()


def make_judge(args: argparse.Namespace):
    if args.mock:
        from guardbound.llm.mock import MockChatLLM
        judge_llm = MockChatLLM(responses=["SAFE"] * 10000)
    else:
        try:
            judge_llm = common.build_paid_chat_llm(
                args.judge_model, args=args)
        except ImportError as exc:
            raise SystemExit(
                f"OpenAI client unavailable: {exc}.  Use --mock for tests."
            )
    from guardbound.evaluation import ASRJudge
    return ASRJudge(
        judge_llm=judge_llm, prompt_path=args.judge_prompt,
        cache_dir=args.cache_dir,
    )


def load_corpus_split(path: str) -> tuple[list, dict]:
    """Load judgeable conversations and report what was excluded, and why.

    The ASR denominator is the number of **valid completed behaviours**. Three
    kinds of row must therefore never become a conversation:

    * ``failed: true`` / ``outcome_category: infrastructure_failure`` — a provider
      error, timeout or interrupted goal is not an unsuccessful attack;
    * ``valid_for_asr: false`` / ``evaluation_status: invalid`` — no usable
      evaluator outcome, so no verdict exists;
    * a flat goal-only record with no turns — it cannot be judged, and converting
      it into a 0-turn conversation silently depressed ASR (Phase 23 defect D-03).
    """
    import json
    from guardbound.schemas import Conversation, Turn
    p = Path(path)
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    stats = {
        "rows": len(rows),
        "valid": 0,
        "excluded_infrastructure_failure": 0,
        "excluded_invalid_evaluator": 0,
        "excluded_no_turns": 0,
    }
    out = []
    for r in rows:
        if r.get("failed") is True or r.get("outcome_category") == "infrastructure_failure":
            stats["excluded_infrastructure_failure"] += 1
            continue
        if r.get("valid_for_asr") is False or r.get("evaluation_status") == "invalid":
            stats["excluded_invalid_evaluator"] += 1
            continue
        turns = r.get("turns")
        if turns is not None and not turns:
            stats["excluded_no_turns"] += 1
            continue
        if "turns" in r and "attack_method" in r and r.get("target_llm"):
            out.append(Conversation.from_dict(r))
            stats["valid"] += 1
            continue
        if not turns:
            stats["excluded_no_turns"] += 1
            continue
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
            turns=[Turn(query=t, response=t) for t in turns],
            max_turns=int(r.get("max_turns", 8)),
        ))
        stats["valid"] += 1
    return out, stats


def load_corpus(path: str) -> list:
    """Judgeable conversations only (see :func:`load_corpus_split`)."""
    return load_corpus_split(path)[0]


def main() -> None:
    args = parse_args()
    from guardbound.evaluation import evaluate_asr
    from guardbound.paper_protocol import PAPER_ASR_SOURCE, SECONDARY_ASR_SOURCE

    # Phase 29: this CLI is a *secondary* judge with a local prompt. The paper
    # metric is the attack-time Qi et al. rubric that scripts/cloud_pilot.py
    # applies to every target turn, so this report can never stand in for R1.
    print(f"ASR source: {SECONDARY_ASR_SOURCE.upper()} (diagnostic) — "
          f"NOT the R1 paper path.")
    print(f"  R1 metric    : {PAPER_ASR_SOURCE} "
          f"(evaluation.asr_source in the paper config)")
    print(f"  This report  : local judge prompt {args.judge_prompt!r}, own model "
          f"{args.judge_model!r} — never report it as the paper ASR.")

    conversations, stats = load_corpus_split(args.corpus)
    if args.limit:
        conversations = conversations[: args.limit]
    print(f"Loaded {len(conversations)} valid conversations from {args.corpus}")
    print(f"  rows                          {stats['rows']}")
    print(f"  excluded infrastructure        {stats['excluded_infrastructure_failure']}")
    print(f"  excluded invalid evaluator     {stats['excluded_invalid_evaluator']}")
    print(f"  excluded (no turns)            {stats['excluded_no_turns']}")

    if not conversations:
        # ASR is undefined without a denominator. Reporting 0% here would present
        # a broken run as a perfectly defended model (Phase 23 defect D-03).
        print(
            "No valid completed goals: ASR is undefined, not 0%. "
            "Refusing to write a report.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if args.dry_run:
        print("[DRY RUN] Skipping judge calls; cache coverage only.")
        return

    judge = make_judge(args)
    verdicts, aggregate = evaluate_asr(conversations, judge, out_path=args.out)
    # Machine-readable provenance: any consumer of these rows learns which judge
    # produced them and that they are not paper-comparable.
    marker = Path(args.out).with_suffix(".asr_source.json")
    marker.write_text(json.dumps({
        "asr_source": SECONDARY_ASR_SOURCE,
        "paper_comparable": False,
        "paper_asr_source": PAPER_ASR_SOURCE,
        "corpus": str(args.corpus),
        "judge_model": args.judge_model,
        "judge_prompt": str(args.judge_prompt),
        "note": (
            "Secondary judge with a local prompt. R1 ASR is the attack-time "
            "Qi et al. rubric recorded by scripts/cloud_pilot.py."
        ),
    }, indent=2), encoding="utf-8")
    print(f"ASR source marker : {marker}")
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
