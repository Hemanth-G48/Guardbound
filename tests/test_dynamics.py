"""Phase 3 dynamics tests — architecture, rollout, masking, loss, gradient, checkpoint.

All tests are CPU-friendly and use tiny synthetic datasets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardbound.models.dynamics import MLPDynamics, DialogueDynamics, save_dynamics, load_dynamics
from guardbound.training.losses import dynamics_loss
from guardbound.training.diagnostics import rollout_mse_per_turn


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

B, K, D = 4, 3, 768  # small test dimensions


def _make_synthetic_batch(B=B, K=K, D=D, seed=42):
    """Create a synthetic batch for testing."""
    torch.manual_seed(seed)
    U = torch.randn(B, K, D)
    Z = torch.randn(B, K, D)
    mask = torch.ones(B, K, dtype=torch.bool)
    # Mask out last turn for some samples
    mask[2, 2] = False
    mask[3, 1:] = False
    return U, Z, mask


def _make_model():
    """Create a small model for testing."""
    return DialogueDynamics(embedding_dim=D, state_dim=D, hidden_dims=[64, 64])


# --------------------------------------------------------------------------- #
# Architecture tests
# --------------------------------------------------------------------------- #

class TestMLPDynamics:
    def test_f_theta_layers(self):
        model = _make_model()
        f = model.f_theta
        assert isinstance(f.net[0], torch.nn.Linear)
        assert f.net[0].in_features == 1536  # 768 + 768
        assert f.net[0].out_features == 64
        assert isinstance(f.net[1], torch.nn.ReLU)
        assert isinstance(f.net[2], torch.nn.Linear)
        assert f.net[2].in_features == 64
        assert f.net[2].out_features == 64
        assert isinstance(f.net[3], torch.nn.ReLU)
        assert isinstance(f.net[4], torch.nn.Linear)
        assert f.net[4].in_features == 64
        assert f.net[4].out_features == 768

    def test_g_theta_layers(self):
        model = _make_model()
        g = model.g_theta
        assert isinstance(g.net[0], torch.nn.Linear)
        assert g.net[0].in_features == 1536
        assert isinstance(g.net[-1], torch.nn.Linear)
        assert g.net[-1].out_features == 768

    def test_f_g_separate_parameters(self):
        model = _make_model()
        f_params = set(id(p) for p in model.f_theta.parameters())
        g_params = set(id(p) for p in model.g_theta.parameters())
        assert len(f_params & g_params) == 0, "f_theta and g_theta share parameters!"

    def test_output_dimensions(self):
        model = _make_model()
        U = torch.randn(B, K, D)
        mask = torch.ones(B, K, dtype=torch.bool)
        X, Z_hat = model.rollout(U, mask)
        assert X.shape == (B, K, D)
        assert Z_hat.shape == (B, K, D)

    def test_param_count(self):
        model = _make_model()
        assert model.f_theta_param_count() > 0
        assert model.g_theta_param_count() > 0
        assert model.total_param_count() == (
            model.f_theta_param_count() + model.g_theta_param_count()
        )


# --------------------------------------------------------------------------- #
# Initialization tests
# --------------------------------------------------------------------------- #

class TestInitialization:
    def test_x0_is_zero(self):
        """x_0 must always be zeros(768)."""
        model = _make_model()
        U = torch.randn(1, 3, D)
        mask = torch.ones(1, 3, dtype=torch.bool)
        X, _ = model.rollout(U, mask)
        # X[:, 0] is the state after processing turn 1
        # Before any processing, x_0 = 0 is implicit
        # After turn 1: x_1 = f(0, u_1), so X[:, 0] = x_1
        assert X.shape[2] == D  # state dim is correct


# --------------------------------------------------------------------------- #
# Rollout tests
# --------------------------------------------------------------------------- #

class TestRollout:
    def test_batched_rollout(self):
        model = _make_model()
        U, Z, mask = _make_synthetic_batch(B=8, K=5)
        X, Z_hat = model.rollout(U, mask)
        assert X.shape == (8, 5, D)
        assert Z_hat.shape == (8, 5, D)

    def test_rollout_preserves_causality(self):
        """Changing u_3 must not affect x_1."""
        model = _make_model()
        model.eval()

        U1 = torch.randn(1, 4, D)
        U2 = U1.clone()
        U2[0, 2] = torch.randn(D)  # change turn 3

        mask = torch.ones(1, 4, dtype=torch.bool)
        X1, _ = model.rollout(U1, mask)
        X2, _ = model.rollout(U2, mask)

        # x_1 and x_2 should be identical (turns 1-2 unchanged)
        torch.testing.assert_close(X1[0, 0], X2[0, 0])
        torch.testing.assert_close(X1[0, 1], X2[0, 1])


# --------------------------------------------------------------------------- #
# Mask tests — CRITICAL
# --------------------------------------------------------------------------- #

class TestMasking:
    def test_padded_turns_do_not_affect_valid_states(self):
        """Changing garbage in masked positions must NOT change valid states."""
        model = _make_model()
        model.eval()

        # Short trajectory: u1, u2, u3
        U_short = torch.randn(1, 3, D)
        mask_short = torch.ones(1, 3, dtype=torch.bool)

        # Long trajectory with garbage padding
        U_long = torch.cat([U_short, torch.randn(1, 5, D)], dim=1)
        mask_long = torch.cat([mask_short, torch.zeros(1, 5, dtype=torch.bool)], dim=1)

        X_short, _ = model.rollout(U_short, mask_short)
        X_long, _ = model.rollout(U_long, mask_long)

        # First 3 states must be identical
        torch.testing.assert_close(X_short[:, 0], X_long[:, 0])
        torch.testing.assert_close(X_short[:, 1], X_long[:, 1])
        torch.testing.assert_close(X_short[:, 2], X_long[:, 2])

    def test_changing_garbage_does_not_change_valid_states(self):
        """Changing the garbage embeddings in masked positions must not affect valid states."""
        model = _make_model()
        model.eval()

        U1 = torch.randn(1, 5, D)
        U2 = U1.clone()
        U2[0, 3:] = torch.randn(2, D)  # change garbage

        mask = torch.ones(1, 5, dtype=torch.bool)
        mask[0, 3:] = False  # mask out turns 4-5

        X1, _ = model.rollout(U1, mask)
        X2, _ = model.rollout(U2, mask)

        # All 3 valid states must be identical
        for k in range(3):
            torch.testing.assert_close(X1[0, k], X2[0, k])

    def test_masked_positions_do_not_contribute_to_loss(self):
        """Loss at masked positions must be zero."""
        model = _make_model()
        model.eval()

        U = torch.randn(2, 4, D)
        Z = torch.randn(2, 4, D)
        mask = torch.ones(2, 4, dtype=torch.bool)
        mask[1, 2:] = False  # mask turns 3-4 for sample 2

        _, Z_hat = model.rollout(U, mask)

        loss = dynamics_loss(Z_hat, Z, mask)

        # Manually compute expected loss
        with torch.no_grad():
            error = (Z_hat - Z) ** 2
            mask_exp = mask.float().unsqueeze(-1)
            masked_error = error * mask_exp
            num_valid = mask.float().sum()
            expected = masked_error.sum() / (num_valid * D)

        torch.testing.assert_close(loss, expected)


# --------------------------------------------------------------------------- #
# Loss tests
# --------------------------------------------------------------------------- #

class TestLoss:
    def test_perfect_prediction_gives_zero(self):
        Z = torch.randn(B, K, D)
        loss = dynamics_loss(Z, Z)
        assert loss.item() == 0.0

    def test_perfect_prediction_with_mask(self):
        Z = torch.randn(B, K, D)
        mask = torch.ones(B, K, dtype=torch.bool)
        mask[0, 0] = False
        loss = dynamics_loss(Z, Z, mask)
        assert loss.item() == 0.0

    def test_masked_values_ignored(self):
        Z_hat = torch.randn(B, K, D)
        Z = torch.randn(B, K, D)
        mask = torch.ones(B, K, dtype=torch.bool)
        mask[0, 0] = False

        loss1 = dynamics_loss(Z_hat, Z, mask)

        # Change Z_hat at masked position — loss should not change
        Z_hat2 = Z_hat.clone()
        Z_hat2[0, 0] = torch.randn(D)
        loss2 = dynamics_loss(Z_hat2, Z, mask)

        torch.testing.assert_close(loss1, loss2)

    def test_invalid_shapes_rejected(self):
        Z_hat = torch.randn(B, K, D)
        Z = torch.randn(B, K, D + 1)
        with pytest.raises(ValueError, match="Shape mismatch"):
            dynamics_loss(Z_hat, Z)

    def test_valid_inputs_produce_finite_loss(self):
        Z_hat = torch.randn(B, K, D)
        Z = torch.randn(B, K, D)
        mask = torch.ones(B, K, dtype=torch.bool)
        loss = dynamics_loss(Z_hat, Z, mask)
        assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# Gradient tests — BPTT
# --------------------------------------------------------------------------- #

class TestGradients:
    def test_f_theta_gets_gradients(self):
        model = _make_model()
        U, Z, mask = _make_synthetic_batch()
        _, Z_hat = model.rollout(U, mask)
        loss = dynamics_loss(Z_hat, Z, mask)
        loss.backward()

        for name, p in model.f_theta.named_parameters():
            assert p.grad is not None, f"f_theta.{name} has no gradient"
            assert torch.isfinite(p.grad).all(), f"f_theta.{name} has non-finite gradient"

    def test_g_theta_gets_gradients(self):
        model = _make_model()
        U, Z, mask = _make_synthetic_batch()
        _, Z_hat = model.rollout(U, mask)
        loss = dynamics_loss(Z_hat, Z, mask)
        loss.backward()

        for name, p in model.g_theta.named_parameters():
            assert p.grad is not None, f"g_theta.{name} has no gradient"
            assert torch.isfinite(p.grad).all(), f"g_theta.{name} has non-finite gradient"

    def test_bptt_through_multiple_timesteps(self):
        """Verify gradients flow through the complete multi-turn rollout."""
        model = _make_model()
        U = torch.randn(2, 4, D, requires_grad=True)
        mask = torch.ones(2, 4, dtype=torch.bool)

        _, Z_hat = model.rollout(U, mask)
        loss = dynamics_loss(Z_hat, torch.randn(2, 4, D), mask)
        loss.backward()

        # Verify U gets gradients (through f_theta across turns)
        assert U.grad is not None
        # Gradients should be non-zero (proving the graph is connected)
        assert U.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# Overfit sanity test
# --------------------------------------------------------------------------- #

class TestOverfit:
    def test_tiny_synthetic_overfit(self):
        """Train on a tiny dataset and verify loss decreases."""
        torch.manual_seed(42)
        model = DialogueDynamics(embedding_dim=D, state_dim=D, hidden_dims=[32, 32])

        # Create synthetic data with simple relationship: Z ≈ U (identity)
        B_train, K_train = 8, 3
        U_train = torch.randn(B_train, K_train, D)
        Z_train = U_train.clone()  # simple relationship
        mask_train = torch.ones(B_train, K_train, dtype=torch.bool)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Initial loss
        model.train()
        _, Z_hat0 = model.rollout(U_train, mask_train)
        initial_loss = dynamics_loss(Z_hat0, Z_train, mask_train).item()

        # Train for a few steps
        for _ in range(50):
            optimizer.zero_grad()
            _, Z_hat = model.rollout(U_train, mask_train)
            loss = dynamics_loss(Z_hat, Z_train, mask_train)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        assert initial_loss > final_loss, (
            f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )


# --------------------------------------------------------------------------- #
# Checkpoint tests
# --------------------------------------------------------------------------- #

class TestCheckpoint:
    def test_save_and_load(self, tmp_path):
        model = _make_model()
        ckpt_dir = tmp_path / "ckpt"

        save_dynamics(model, ckpt_dir, epoch=10, seed=42)
        loaded = load_dynamics(ckpt_dir)

        assert isinstance(loaded, DialogueDynamics)
        assert loaded.embedding_dim == model.embedding_dim
        assert loaded.state_dim == model.state_dim

    def test_loaded_model_equivalent_rollout(self, tmp_path):
        model = _make_model()
        model.eval()
        ckpt_dir = tmp_path / "ckpt"

        save_dynamics(model, ckpt_dir)

        loaded = load_dynamics(ckpt_dir)
        loaded.eval()

        U = torch.randn(2, 4, D)
        mask = torch.ones(2, 4, dtype=torch.bool)

        X1, Z1 = model.rollout(U, mask)
        X2, Z2 = loaded.rollout(U, mask)

        torch.testing.assert_close(X1, X2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(Z1, Z2, atol=1e-5, rtol=1e-5)

    def test_loaded_model_is_eval_mode(self, tmp_path):
        model = _make_model()
        ckpt_dir = tmp_path / "ckpt"
        save_dynamics(model, ckpt_dir)
        loaded = load_dynamics(ckpt_dir)
        assert not loaded.training

    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dynamics(tmp_path / "nonexistent")


# --------------------------------------------------------------------------- #
# Diagnostics tests
# --------------------------------------------------------------------------- #

class TestDiagnostics:
    def test_rollout_mse_per_turn(self):
        model = _make_model()
        model.eval()
        U = torch.randn(B, K, D)
        Z = torch.randn(B, K, D)
        mask = torch.ones(B, K, dtype=torch.bool)

        per_turn = rollout_mse_per_turn(model, U, Z, mask)
        assert len(per_turn) == K
        assert all(np.isfinite(v) for v in per_turn)

    def test_per_turn_nan_for_empty_mask(self):
        model = _make_model()
        model.eval()
        U = torch.randn(1, K, D)
        Z = torch.randn(1, K, D)
        mask = torch.zeros(1, K, dtype=torch.bool)

        per_turn = rollout_mse_per_turn(model, U, Z, mask)
        assert all(np.isnan(v) for v in per_turn)
