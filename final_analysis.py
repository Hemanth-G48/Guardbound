#!/usr/bin/env python3
"""ASR evaluation using LLM judge (paper methodology).

This script evaluates conversation results using the proper LLM judge
with rubric-based evaluation, matching the paper's methodology.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ASR using LLM judge")
    parser.add_argument("--baseline", required=True, help="Baseline results JSONL")
    parser.add_argument("--nbf", required=True, help="NBF results JSONL")
    parser.add_argument("--judge-model", default="gpt-4o-2024-08-06",
                       help="Judge model (default: gpt-4o-2024-08-06)")
    parser.add_argument("--out-dir", default="results/eval",
                       help="Output directory for evaluation results")
    parser.add_argument("--mock-judge", action="store_true",
                       help="Use mock judge for testing")
    return parser.parse_args()


def load_conversations(path: str):
    """Load conversations from JSONL file."""
    from guardbound.schemas import Conversation
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, dict) and "turns" in data:
                conversations.append(Conversation.from_dict(data))
    return conversations


def make_judge(model_name: str, mock: bool = False):
    """Create LLM judge."""
    if mock:
        from guardbound.llm.mock import MockChatLLM
        judge_llm = MockChatLLM(responses=["SAFE"] * 10000)
    else:
        from guardbound.llm.openai_client import OpenAIChatLLM
        judge_llm = OpenAIChatLLM(model_id=model_name)
    
    from guardbound.evaluation import ASRJudge
    return ASRJudge(
        judge_llm=judge_llm,
        prompt_path="configs/judge_prompts/asr_judge.txt",
        cache_dir="data/processed/.cache/judge_asr",
    )


def evaluate_corpus(conversations, judge, out_path: Path):
    """Evaluate a corpus of conversations."""
    from guardbound.evaluation import evaluate_asr
    verdicts, aggregate = evaluate_asr(conversations, judge, out_path=out_path)
    return verdicts, aggregate


def main():
    args = parse_args()
    
    print("Loading conversations...")
    baseline_convs = load_conversations(args.baseline)
    nbf_convs = load_conversations(args.nbf)
    print(f"Baseline: {len(baseline_convs)} conversations")
    print(f"NBF: {len(nbf_convs)} conversations")
    
    judge = make_judge(args.judge_model, args.mock_judge)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nEvaluating baseline...")
    baseline_out = out_dir / "baseline_verdicts.jsonl"
    _, baseline_agg = evaluate_corpus(baseline_convs, judge, baseline_out)
    
    print("\nEvaluating NBF...")
    nbf_out = out_dir / "nbf_verdicts.jsonl"
    _, nbf_agg = evaluate_corpus(nbf_convs, judge, nbf_out)
    
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print()
    print(f"{'Metric':<30} {'Baseline':<15} {'NBF':<15}")
    print("-" * 60)
    print(f"{'Total behaviors':<30} {baseline_agg['total_behaviors']:<15} {nbf_agg['total_behaviors']:<15}")
    print(f"{'Successful attacks':<30} {baseline_agg['successful_behaviors']:<15} {nbf_agg['successful_behaviors']:<15}")
    print(f"{'ASR (%)':<30} {baseline_agg['asr']*100:<15.1f} {nbf_agg['asr']*100:<15.1f}")
    
    baseline_asr = baseline_agg['asr'] * 100
    nbf_asr = nbf_agg['asr'] * 100
    reduction = baseline_asr - nbf_asr
    
    print()
    print(f"NBF Effect: {'+' if reduction < 0 else ''}{reduction:.1f} percentage points")
    if reduction > 0:
        print("NBF REDUCES attack success rate (matches paper expectation)")
    else:
        print("NBF INCREASES attack success rate (opposite of paper)")
    
    print()
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
