"""Tests for context initialization — MMLU-style pre-question state setup."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardbound.defense.context_init import initialize_with_context
from guardbound.models.dynamics import DialogueDynamics
from guardbound.models.predictor import SafetyPredictor, NeuralBarrierFunction

D = 768


def _make_nbf():
    dynamics = DialogueDynamics(hidden_dims=[32, 32])
    predictor = SafetyPredictor()
    nbf = NeuralBarrierFunction(dynamics, predictor)
    nbf.eval()
    return nbf


class TestContextInit:
    def test_manual_rollout_matches(self):
        """verify: initialize_with_context([A,B,C]) == manual f(0,A)->f(x1,B)->f(x2,C)"""
        nbf = _make_nbf()
        # Fixed embeddings for reproducibility
        embeddings = [torch.randn(1, D) for _ in range(3)]
        idx = [0]
        def embed_fn(text):
            val = embeddings[idx[0]]
            idx[0] += 1
            return val

        texts = ["context A", "context B", "context C"]

        # Manual computation
        idx[0] = 0
        state = torch.zeros(1, D)
        for text in texts:
            u = embed_fn(text)
            state = nbf.advance_state(state, u)
        expected = state.clone()

        # Using initialize_with_context
        idx[0] = 0
        initial = torch.zeros(1, D)
        result = initialize_with_context(nbf, initial, texts, embed_fn)

        torch.testing.assert_close(result, expected)

    def test_empty_context(self):
        """Empty context returns initial state unchanged."""
        nbf = _make_nbf()
        embed_fn = lambda text: torch.randn(1, D)

        initial = torch.randn(1, D)
        result = initialize_with_context(nbf, initial, [], embed_fn)

        torch.testing.assert_close(result, initial)

    def test_single_context(self):
        """Single context text advances state once."""
        nbf = _make_nbf()
        embed_val = torch.randn(1, D)
        embed_fn = lambda text: embed_val

        initial = torch.zeros(1, D)
        result = initialize_with_context(nbf, initial, ["hello"], embed_fn)

        # Manually compute
        expected = nbf.advance_state(initial, embed_val)

        torch.testing.assert_close(result, expected)
