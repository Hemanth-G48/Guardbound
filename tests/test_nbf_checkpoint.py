"""Tests for NBF checkpoint save/load and reproducibility."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardbound.models.dynamics import DialogueDynamics
from guardbound.models.predictor import SafetyPredictor, NeuralBarrierFunction


D = 768


class TestNBFCheckpoint:
    def test_save_and_reload_predictor(self, tmp_path):
        predictor = SafetyPredictor()
        ckpt = {"predictor_state_dict": predictor.state_dict()}
        path = tmp_path / "predictor.pt"
        torch.save(ckpt, path)

        predictor2 = SafetyPredictor()
        ckpt2 = torch.load(path, weights_only=True)
        predictor2.load_state_dict(ckpt2["predictor_state_dict"])

        # Verify identical outputs
        x = torch.randn(2, D)
        u = torch.randn(2, D)
        h1 = predictor.predictor_value(x, u)
        h2 = predictor2.predictor_value(x, u)
        torch.testing.assert_close(h1, h2, atol=1e-6, rtol=1e-6)

    def test_nbf_bundle_save_and_reload(self, tmp_path):
        dynamics = DialogueDynamics(hidden_dims=[32, 32])
        predictor = SafetyPredictor()
        nbf = NeuralBarrierFunction(dynamics, predictor)

        # Save dynamics separately
        from guardbound.models.dynamics import save_dynamics
        save_dynamics(dynamics, tmp_path / "dynamics")

        # Save predictor
        nbf.save(tmp_path / "nbf.pt")

        # Reload
        loaded = NeuralBarrierFunction.load(
            dynamics_dir=tmp_path / "dynamics",
            predictor_path=tmp_path / "nbf.pt",
        )

        # Verify identical predictor outputs
        x = torch.randn(2, D)
        u = torch.randn(2, D)
        h1 = nbf.h(x, u)
        h2 = loaded.h(x, u)
        torch.testing.assert_close(h1, h2, atol=1e-6, rtol=1e-6)

    def test_checkpoint_metadata(self, tmp_path):
        dynamics = DialogueDynamics(hidden_dims=[32, 32])
        predictor = SafetyPredictor()
        nbf = NeuralBarrierFunction(dynamics, predictor)
        nbf.save(tmp_path / "nbf.pt", eta=0.0, kappa=3, training_mode="frozen")

        ckpt = torch.load(tmp_path / "nbf.pt", weights_only=False)
        assert ckpt["eta"] == 0.0
        assert ckpt["kappa"] == 3
        assert ckpt["training_mode"] == "frozen"
        assert "architecture" in ckpt
        assert ckpt["architecture"]["state_dim"] == 768
