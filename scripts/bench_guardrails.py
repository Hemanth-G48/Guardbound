#!/usr/bin/env python3
"""Guardrail F1 benchmarking CLI.

Runs the NBF guard and optional baseline guards (OpenAI Moderation,
ShieldGemma, LLaMA-Guard) on HarmBench / Aegis / WildGuard and emits
a Table-4-style comparison.
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
    p.add_argument("--nbf-checkpoints", default="mpnet",
                   help="Comma-separated embedding aliases (mpnet,distilroberta).")
    p.add_argument("--datasets", default="harmbench",
                   help="Comma-separated dataset names (harmbench,aegis,wildguard).")
    p.add_argument("--data-path", default=None,
                   help="Optional template: e.g. 'data/raw/{dataset}.jsonl'.  "
                        "Defaults to data/raw/<dataset>.jsonl.")
    p.add_argument("--nbf-dynamics-dir", default=None,
                   help="Dynamics checkpoint dir (defaults to --nbf-checkpoints value).")
    p.add_argument("--nbf-predictor", default=None,
                   help="Predictor checkpoint path (defaults to --nbf-checkpoints value).")
    p.add_argument("--include-openai", action="store_true",
                   help="Include the OpenAI Moderation guard (requires API key).")
    p.add_argument("--include-shieldgemma", action="store_true",
                   help="Include the ShieldGemma-2B guard (requires weights + GPU).")
    p.add_argument("--include-llama-guard", action="store_true",
                   help="Include the LLaMA-Guard-7B guard (requires weights + GPU).")
    p.add_argument("--out", required=True,
                   help="Output JSONL path (EvaluationResult rows).")
    p.add_argument("--mock", action="store_true",
                   help="Use a mock NBF guard (random argmax) for tests.")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def make_nbf_guard(embedding_alias: str, mock: bool, dynamics_dir: str | None, predictor_path: str | None):
    if mock:
        import random
        from guardbound.evaluation.guards.base import PromptGuard
        class _RandomGuard(PromptGuard):
            name = f"mock_nbf_{embedding_alias}"
            def predict(self, text: str) -> str:
                return random.choice(["harmful", "harmless"])
        return _RandomGuard()
    from guardbound.evaluation import make_nbf_guard_from_checkpoint
    from guardbound.embeddings import get_embed_fn
    embed_fn = get_embed_fn(embedding_alias)
    return make_nbf_guard_from_checkpoint(
        dynamics_dir=dynamics_dir or "checkpoints",
        predictor_path=predictor_path or "checkpoints/predictor_h.pt",
        embed_fn=embed_fn,
    )


def make_baseline_guard(name: str, mock: bool):
    if mock:
        from guardbound.evaluation.guards.base import PromptGuard
        class _MockGuard(PromptGuard):
            def __init__(self, label):
                self.name = f"mock_{label}"
            def predict(self, text):
                return "harmful" if "kill" in text.lower() else "harmless"
        return _MockGuard(name)
    if name == "openai":
        from guardbound.evaluation.guards import OpenAIModerationGuard
        return OpenAIModerationGuard()
    if name == "shieldgemma":
        from guardbound.evaluation.guards import ShieldGemmaGuard
        return ShieldGemmaGuard()
    if name == "llama_guard":
        from guardbound.evaluation.guards import LlamaGuardGuard
        return LlamaGuardGuard()
    raise ValueError(f"Unknown guard: {name}")


def main() -> None:
    args = parse_args()
    from guardbound.evaluation import (
        GUARDRAIL_LOADERS,
        evaluate_guard,
        write_jsonl,
        to_markdown_table,
        to_latex_table,
        EvaluationResult,
    )

    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    embedding_aliases = [e.strip() for e in args.nbf_checkpoints.split(",") if e.strip()]

    guards = []
    for alias in embedding_aliases:
        guards.append((f"nbf_{alias}",
                       make_nbf_guard(alias, args.mock,
                                      args.nbf_dynamics_dir, args.nbf_predictor)))
    if args.include_openai:
        guards.append(("openai_moderation", make_baseline_guard("openai", args.mock)))
    if args.include_shieldgemma:
        guards.append(("shieldgemma_2b", make_baseline_guard("shieldgemma", args.mock)))
    if args.include_llama_guard:
        guards.append(("llama_guard_7b", make_baseline_guard("llama_guard", args.mock)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out_path.with_suffix(".results.jsonl")
    all_rows: list[EvaluationResult] = []

    for ds_name in dataset_names:
        loader = GUARDRAIL_LOADERS[ds_name]
        if args.data_path:
            path = args.data_path.format(dataset=ds_name)
        else:
            path = f"data/raw/{ds_name}.jsonl"
        try:
            examples = loader(path)
        except FileNotFoundError as exc:
            print(f"[SKIP] {ds_name}: {exc}")
            continue
        if args.limit:
            examples = examples[: args.limit]
        for guard_name, guard in guards:
            result = evaluate_guard(
                examples, guard, dataset_name=ds_name,
                out_path=out_path.with_name(
                    f"{out_path.stem}__{ds_name}__{guard_name}.jsonl"
                ),
            )
            m = result.metrics
            print(f"  {ds_name:10s} {guard_name:24s}  "
                  f"P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f} "
                  f"acc={m.accuracy:.3f} n={m.n}")
            for metric_name, value in [
                ("F1", m.f1),
                ("precision", m.precision),
                ("recall", m.recall),
                ("accuracy", m.accuracy),
            ]:
                all_rows.append(EvaluationResult(
                    model=guard_name, defense_variant="guardrail",
                    metric=metric_name, value=value,
                    dataset=ds_name, n=m.n,
                ))

    write_jsonl(all_rows, sidecar)
    print(f"Results sidecar : {sidecar}")
    if all_rows:
        md = to_markdown_table(all_rows, title="Guardrail F1 (Table-4-style)")
        print("\n" + md)
        latex = to_latex_table(all_rows, title="Guardrail F1 (Table-4-style)")
        (out_path.parent / f"{out_path.stem}__table.tex").write_text(latex)


if __name__ == "__main__":
    main()
