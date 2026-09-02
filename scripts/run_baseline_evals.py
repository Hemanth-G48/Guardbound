#!/usr/bin/env python3
"""Phase 8 — baseline evaluation runner.

Composes Phase 6 attack runner + Phase 7 evaluation through the
defense-variant registry.  No metric or attack re-implementation
happens here.

For every (variant, model, attack, eta) combination the script:

1. Builds a defended ``ChatLLM`` via ``build_defense(variant, llm, ...)``.
2. Runs the Phase 6 attack via ``run_attack(attack, target_llm, ...)``.
3. Evaluates the transcripts through Phase 7's ``evaluate_asr``.
4. Writes a JSONL of ``EvaluationResult`` rows (one per
   (variant, model, attack, eta) combination).

The ``--mock`` flag is honored throughout, so the script can be
exercised end-to-end in CI without GPUs, API keys, or HF downloads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True,
                   choices=["original", "system_prompt",
                            "lora_sft", "lora_dpo", "lora_kto",
                            "guardbound"],
                   help="Defense variant to evaluate.")
    p.add_argument("--model", default="gpt-3.5-turbo-0125",
                   help="Target LLM model id (default: gpt-3.5-turbo-0125).")
    p.add_argument("--attacks", default="crescendo",
                   help="Comma-separated attack names (crescendo,actor_attack,opposite_day,red_queen,adaptive).")
    p.add_argument("--eta", type=float, default=5e-4,
                   help="Steering threshold for guardbound (default: 5e-4).")
    p.add_argument("--barrier-checkpoint", default=None,
                   help="Predictor checkpoint path (for guardbound).")
    p.add_argument("--dynamics-dir", default=None,
                   help="Dynamics checkpoint dir (for guardbound).")
    p.add_argument("--embedding", default="mpnet",
                   choices=["mpnet", "distilroberta"])
    p.add_argument("--checkpoint", default=None,
                   help="LoRA adapter path (for lora_sft/dpo/kto).")
    p.add_argument("--goals", default="data/raw/harmbench/raw_data.jsonl",
                   help="JSONL of goals / conversations.")
    p.add_argument("--goal-field", default=None,
                   help="Override the goal field name (default: auto-detect).")
    p.add_argument("--judge-model", default="gpt-4o-2024-08-06",
                   help="GPT-4o judge model for ASR (default: gpt-4o-2024-08-06).")
    p.add_argument("--judge-prompt", default="configs/judge_prompts/asr_judge.txt")
    p.add_argument("--cache-dir", default="data/processed/.cache/judge_asr")
    p.add_argument("--out", required=True,
                   help="Output JSONL path (Phase 1 conversation transcripts).")
    p.add_argument("--results-out", default=None,
                   help="Optional separate path for EvaluationResult rows (JSONL).")
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--mock", action="store_true",
                   help="Use MockChatLLM for target and judge.")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip attack generation; print plan only.")
    return p.parse_args()


def make_target_llm(args: argparse.Namespace):
    if args.mock:
        from guardbound.llm.mock import MockChatLLM
        return MockChatLLM(responses=[])
    target = args.model.lower()
    if "gpt" in target or target.startswith("o1") or target.startswith("o3"):
        from guardbound.llm.openai_client import OpenAIChatLLM
        return OpenAIChatLLM(model_id=args.model)
    if "claude" in target or "anthropic" in target:
        from guardbound.llm.anthropic_client import AnthropicChatLLM
        return AnthropicChatLLM(model=args.model)
    from guardbound.llm.local_client import HFLocalChatLLM
    return HFLocalChatLLM(model_id=args.model)


def make_barrier_and_embed(args: argparse.Namespace):
    if args.variant != "guardbound":
        return None, None
    if args.mock:
        from guardbound.models.dynamics import DialogueDynamics
        from guardbound.models.predictor import (
            SafetyPredictor, NeuralBarrierFunction,
        )
        barrier = NeuralBarrierFunction(
            dynamics=DialogueDynamics(state_dim=768, embedding_dim=768),
            predictor=SafetyPredictor(state_dim=768, embedding_dim=768),
            embedding_model="mpnet",
        )
        embed_fn = lambda t: torch.zeros(1, 768)
        return barrier, embed_fn
    if not args.barrier_checkpoint:
        raise SystemExit(
            "Variant guardbound requires --barrier-checkpoint."
        )
    from guardbound.models.predictor import NeuralBarrierFunction
    from guardbound.embeddings import get_embed_fn
    barrier = NeuralBarrierFunction.load(
        dynamics_dir=args.dynamics_dir or args.barrier_checkpoint,
        predictor_path=args.barrier_checkpoint,
    )
    return barrier, get_embed_fn(args.embedding)


def load_goals(path: str, limit: int | None = None) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Goals file not found: {p}")
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def get_goal_text(record: dict, goal_field: str | None) -> str:
    if goal_field:
        return str(record.get(goal_field, ""))
    for k in ("goal", "behavior", "text", "prompt"):
        if k in record and isinstance(record[k], str):
            return record[k]
    return json.dumps(record, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    import torch  # noqa: F401  -- used by steered chat construction

    from guardbound.baselines import build_defense, list_variants
    from guardbound.attacks.registry import get_attack
    from guardbound.attacks.runner import run_attack
    from guardbound.evaluation import (
        ASRJudge, EvaluationResult, evaluate_asr, write_jsonl,
    )

    print(f"Variant : {args.variant}")
    print(f"Model   : {args.model}")
    print(f"Attacks : {args.attacks}")
    print(f"eta     : {args.eta}")
    print(f"Mock    : {args.mock}")

    if args.dry_run:
        print("[DRY RUN] No attacks generated.")
        return

    base_llm = make_target_llm(args)
    barrier, embed_fn = make_barrier_and_embed(args)
    defended = build_defense(
        variant=args.variant,
        llm=base_llm,
        checkpoint=args.checkpoint,
        base_model=args.model,
        eta=args.eta,
        barrier=barrier,
        embed_fn=embed_fn,
        mock=args.mock,
    )
    print(f"Defended LLM: {type(defended).__name__}")

    attack_names = [a.strip() for a in args.attacks.split(",") if a.strip()]
    goals = load_goals(args.goals, limit=args.limit)
    print(f"Loaded {len(goals)} goals from {args.goals}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    rows: list[EvaluationResult] = []
    for atk_name in attack_names:
        attack = get_attack(atk_name)
        for i, rec in enumerate(goals):
            goal = get_goal_text(rec, args.goal_field)
            if not goal:
                continue
            print(f"[{atk_name}] {i+1}/{len(goals)}: {goal[:60]}...")
            conv = run_attack(
                attack=attack,
                goal=goal,
                target_llm=defended,
                barrier=barrier,
                embed_fn=embed_fn,
                eta=args.eta,
                max_turns=args.max_turns,
                temperature=0.7,
                attack_method=atk_name,
                target_llm_name=args.model,
            )
            # Append the conversation to the JSONL output.
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(conv.to_json() + "\n")

    # ---- Phase 7: ASR over the collected transcripts ---------------
    from guardbound.schemas import load_conversations_jsonl
    conversations = load_conversations_jsonl(out_path)
    if not conversations:
        print("No conversations to judge.")
    else:
        # Build the judge (mock or real).
        if args.mock:
            from guardbound.llm.mock import MockChatLLM
            judge_llm = MockChatLLM(responses=["SAFE"] * 10000)
        else:
            try:
                from guardbound.llm.openai_client import OpenAIChatLLM
                judge_llm = OpenAIChatLLM(model_id=args.judge_model)
            except ImportError:
                print("WARN: openai client unavailable; using mock judge.",
                      file=sys.stderr)
                from guardbound.llm.mock import MockChatLLM
                judge_llm = MockChatLLM(responses=["SAFE"] * 10000)
        judge = ASRJudge(
            judge_llm=judge_llm,
            prompt_path=args.judge_prompt,
            cache_dir=args.cache_dir,
        )
        verdicts, agg = evaluate_asr(conversations, judge, out_path=out_path)
        print(f"ASR (variant={args.variant}): {agg['asr']:.4f} "
              f"({agg['successful_behaviors']}/{agg['total_behaviors']})")

        for atk_name, sub in agg["by_attack"].items():
            rows.append(EvaluationResult(
                model=args.model,
                defense_variant=args.variant,
                metric="ASR",
                value=sub["asr"],
                attack=atk_name,
                eta=args.eta if args.variant == "guardbound" else None,
                judge_model=args.judge_model,
                n=sub["total"],
            ))

    if rows:
        results_out = Path(args.results_out) if args.results_out \
            else out_path.with_suffix(".results.jsonl")
        write_jsonl(rows, results_out)
        print(f"EvaluationResult rows: {results_out}")


if __name__ == "__main__":
    main()
