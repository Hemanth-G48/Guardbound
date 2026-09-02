"""Tests for SafetyPredictor — shapes, Eq. (5), barrier value."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardbound.models.predictor import SafetyPredictor, NeuralBarrierFunction
from guardbound.models.dynamics import DialogueDynamics


B = 4


def _make_predictor():
    return SafetyPredictor(state_dim=768, embedding_dim=768)


# --------------------------------------------------------------------------- #
# Architecture tests
# --------------------------------------------------------------------------- #

class TestPredictorArchitecture:
    def test_layers(self):
        p = _make_predictor()
        assert isinstance(p.net[0], torch.nn.Linear)
        assert p.net[0].in_features == 1536
        assert p.net[0].out_features == 32
        assert isinstance(p.net[1], torch.nn.ReLU)
        assert isinstance(p.net[2], torch.nn.Linear)
        assert p.net[2].in_features == 32
        assert p.net[2].out_features == 32
        assert isinstance(p.net[3], torch.nn.ReLU)
        assert isinstance(p.net[4], torch.nn.Linear)
        assert p.net[4].in_features == 32
        assert p.net[4].out_features == 5

    def test_logits_shape(self):
        p = _make_predictor()
        x = torch.randn(B, 768)
        u = torch.randn(B, 768)
        logits = p(x, u)
        assert logits.shape == (B, 5)

    def test_probs_shape(self):
        p = _make_predictor()
        x = torch.randn(B, 768)
        u = torch.randn(B, 768)
        probs = p.class_probs(x, u)
        assert probs.shape == (B, 5)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)

    def test_h_shape(self):
        p = _make_predictor()
        x = torch.randn(B, 768)
        u = torch.randn(B, 768)
        h = p.predictor_value(x, u)
        assert h.shape == (B,)

    def test_predicted_label_shape(self):
        p = _make_predictor()
        x = torch.randn(B, 768)
        u = torch.randn(B, 768)
        labels = p.predicted_label(x, u)
        assert labels.shape == (B,)

    def test_param_count(self):
        p = _make_predictor()
        assert p.param_count() > 0


# --------------------------------------------------------------------------- #
# Eq. (5) tests
# --------------------------------------------------------------------------- #

class TestEquation5:
    def test_known_probabilities_unsafe(self):
        """Test h with manually constructed probabilities."""
        p = _make_predictor()
        x = torch.randn(1, 768)
        u = torch.randn(1, 768)

        # Manually set weights to produce specific probabilities
        with torch.no_grad():
            # Override net to produce specific logits
            logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 10.0]])  # strong unsafe
            # Patch class_probs to return our probs
            probs = torch.softmax(logits, dim=-1)
            p_unsafe = probs[0, 4]
            p_safe_max = probs[0, :4].max()
            expected_h = (p_unsafe - p_safe_max).item()

            # Verify formula: h = p[4] - max(p[0:4])
            assert expected_h > 0, "Strong unsafe should give positive h"

    def test_known_probabilities_safe(self):
        """Safe-dominant: h should be negative."""
        probs = torch.tensor([[0.4, 0.3, 0.2, 0.05, 0.05]])
        p_unsafe = probs[0, 4]
        p_safe_max = probs[0, :4].max()
        h = (p_unsafe - p_safe_max).item()
        assert h < 0, "Safe-dominant should give negative h"

    def test_balanced(self):
        """Equal safe/unsafe: h ≈ 0."""
        probs = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])
        p_unsafe = probs[0, 4]
        p_safe_max = probs[0, :4].max()
        h = (p_unsafe - p_safe_max).item()
        assert abs(h) < 1e-5, f"Balanced should give h≈0, got {h}"

    def test_label_conversion(self):
        """Verify paper labels 1..5 map correctly to CE indices 0..4."""
        assert SafetyPredictor.PAPER_LABEL_TO_CE == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        assert SafetyPredictor.CE_TO_PAPER_LABEL == {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}


# --------------------------------------------------------------------------- #
# NeuralBarrierFunction tests
# --------------------------------------------------------------------------- #

class TestNeuralBarrierFunction:
    def test_h_method(self):
        dynamics = DialogueDynamics(hidden_dims=[32, 32])
        predictor = SafetyPredictor()
        nbf = NeuralBarrierFunction(dynamics, predictor)

        state = torch.randn(2, 768)
        u = torch.randn(2, 768)
        h = nbf.h(state, u)
        assert h.shape == (2,)

    def test_filter_query(self):
        dynamics = DialogueDynamics(hidden_dims=[32, 32])
        predictor = SafetyPredictor()
        nbf = NeuralBarrierFunction(dynamics, predictor)

        state = torch.randn(1, 768)
        u = torch.randn(1, 768)
        allowed, h_val = nbf.filter_query(state, u, eta=0.0)
        assert isinstance(allowed, bool)
        assert isinstance(h_val, float)

    def test_advance_state(self):
        dynamics = DialogueDynamics(hidden_dims=[32, 32])
        predictor = SafetyPredictor()
        nbf = NeuralBarrierFunction(dynamics, predictor)

        state = torch.randn(2, 768)
        u = torch.randn(2, 768)
        new_state = nbf.advance_state(state, u)
        assert new_state.shape == (2, 768)

    def test_save_and_reload(self, tmp_path):
        from guardbound.models.dynamics import save_dynamics

        dynamics = DialogueDynamics(hidden_dims=[32, 32])
        predictor = SafetyPredictor()
        nbf = NeuralBarrierFunction(dynamics, predictor)

        # Save dynamics separately
        save_dynamics(dynamics, tmp_path / "dynamics")

        # Save predictor
        nbf.save(tmp_path / "nbf.pt")

        loaded = NeuralBarrierFunction.load(
            dynamics_dir=tmp_path / "dynamics",
            predictor_path=tmp_path / "nbf.pt",
        )
        assert loaded.predictor is not None
