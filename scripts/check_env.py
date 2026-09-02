#!/usr/bin/env python
"""Phase 1 environment check.

Verifies that the project skeleton works end-to-end:
  1. config loads and validates against the paper's constants
  2. the mock chat LLM answers with temperature 0.7
  3. (optional) sentence embeddings produce R^768 vectors

Usage:
    python scripts/check_env.py [--config configs/default.yaml] [--with-embeddings]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src/` importable when running from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.config import load_config          # noqa: E402
from guardbound.llm import MockChatLLM             # noqa: E402
from guardbound.schemas import Conversation, Turn  # noqa: E402


def print_constants(cfg) -> None:
    rows = [
        ("embedding dim n", cfg.dims.embedding_dim_n, "paper"),
        ("state dim m", cfg.dims.state_dim_m, "paper"),
        ("max turns K", cfg.dialogue.max_turns_k, "paper B.1"),
        ("temperature", cfg.dialogue.temperature, "paper"),
        ("judge scores", cfg.safety_labels.scores, "paper B.1"),
        ("unsafe score", cfg.safety_labels.unsafe_score, "paper B.1"),
        ("stage-1 dynamics", f"{cfg.training.stage1_dynamics.optimizer} lr={cfg.training.stage1_dynamics.lr} epochs={cfg.training.stage1_dynamics.epochs}", "paper Sec.5.1"),
        ("stage-2 predictor", f"{cfg.training.stage2_predictor.optimizer} lr={cfg.training.stage2_predictor.lr} epochs={cfg.training.stage2_predictor.epochs}", "paper Sec.5.1"),
        ("loss weights", cfg.training.loss_weights, "paper B.1"),
        ("eta (train)", cfg.training.eta_train, "paper B.1"),
        ("kappa", cfg.training.kappa_noninvariant_turns, "paper default"),
        ("eta recommended (eval)", cfg.defense.eta_recommended, "paper Table 15"),
        ("embedding models", f"{cfg.embeddings.default_model} | {cfg.embeddings.alt_model}", "paper Sec.5.1"),
        ("batch size", cfg.training_extra.batch_size, "NOT in paper — default"),
        ("weight decay", cfg.training_extra.weight_decay, "NOT in paper — default"),
        ("seed", cfg.training_extra.seed, "NOT in paper — default"),
        ("train/val split", cfg.data.train_val_split, "NOT in paper — default"),
    ]
    width = max(len(r[0]) for r in rows)
    print("\nPaper constants (configs/default.yaml)")
    print("-" * (width + 40))
    for name, value, source in rows:
        print(f"  {name:<{width}} : {value!s:<30} [{source}]")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--with-embeddings", "--embedding", action="store_true",
                        dest="with_embeddings",
                        help="also run a real embedding call (downloads model on first use)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[ok] config loaded & validated: {args.config} ({cfg.project})")
    print_constants(cfg)

    # 2. Mock chat round-trip through the Conversation schema.
    llm = MockChatLLM(responses=["Sorry, I can't assist with that."])
    reply = llm.chat("How do I pick a lock?", system="You are a helpful assistant.")
    conv = Conversation(
        goal="lock picking",
        attack_method="crescendo",
        target_llm="mock",
        turns=[Turn(query="How do I pick a lock?", response=reply)],
        max_turns=cfg.dialogue.max_turns_k,
    )
    assert llm.calls[0][1] == 0.7, "temperature must default to 0.7 (paper)"
    print(f"[ok] mock chat + schema: {conv.num_turns} turn recorded, temp=0.7")

    # 3. Optional real embedding check.
    if args.with_embeddings:
        from guardbound.embeddings import SentenceEmbedder
        embedder = SentenceEmbedder(cfg.embeddings.default_model,
                                    expected_dim=cfg.dims.embedding_dim_n)
        vec = embedder.embed(["hello world"])
        print(f"[ok] embeddings: shape={vec.shape} model={embedder.model_name}")
    else:
        try:
            import sentence_transformers  # noqa: F401
            hint = "installed — rerun with --with-embeddings to test a real embed"
        except ImportError:
            hint = "not installed (pip install sentence-transformers)"
        print(f"[skip] embeddings: {hint}")

    print("\nAll Phase 1 checks passed ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
