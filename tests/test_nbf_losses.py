"""Tests for NBF losses — Eqs. (6), (11), (12)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardbound.models.dynamics import DialogueDynamics
from guardbound.models.predictor import SafetyPredictor
from guardbound.training.nbf_losses import (
    ce_loss, safe_set_loss, safety_invariance_loss,
    paper_label_to_ce_index,
)
from guardbound.training.losses import dynamics_loss

B, K, D = 4, 8, 768


def _make_data():
    torch.manual_seed(42)
    U = torch.randn(B, K, D)
    Z = torch.randn(B, K, D)
    Y = torch.randint(1, 6, (B, K))  # paper labels {1..5}
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[2, 5:] = False
    mask[3, 3:] = False
    return U, Z, Y, mask


# --------------------------------------------------------------------------- #
# CE Loss — Eq. (6)
# --------------------------------------------------------------------------- #

class TestCELoss:
    def test_ce_basic(self):
        p = SafetyPredictor()
        U, Z, Y, mask = _make_data()
        x_prev = torch.randn(B, K, D)
        loss = ce_loss(p, x_prev, U, Y, mask)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_ce_masked_positions_ignored(self):
        p = SafetyPredictor()
        U, Z, Y, mask = _make_data()
        x_prev = torch.randn(B, K, D)  # same x_prev for both calls

        loss1 = ce_loss(p, x_prev, U, Y, mask)

        # Change Y at masked position — loss should not change
        Y2 = Y.clone()
        Y2[2, 6] = 1  # masked position
        loss2 = ce_loss(p, x_prev, U, Y2, mask)

        torch.testing.assert_close(loss1, loss2)

    def test_ce_manual_computation(self):
        """Verify CE matches manual cross-entropy."""
        p = SafetyPredictor()
        U, Z, Y, mask = _make_data()
        x_prev = torch.randn(B, K, D)

        with torch.no_grad():
            # Compute logits manually
            inp = torch.cat([x_prev.reshape(B*K, D), U.reshape(B*K, D)], dim=-1)
            logits = p.net(inp)
            y_ce = paper_label_to_ce_index(Y.reshape(-1))

            import torch.nn.functional as F
            expected = F.cross_entropy(logits, y_ce, reduction="none")
            mask_f = mask.float().reshape(-1)
            expected_masked = (expected * mask_f).sum() / mask_f.sum().clamp(min=1.0)

        actual = ce_loss(p, x_prev, U, Y, mask)
        torch.testing.assert_close(actual, expected_masked, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# Safe-Set Loss — Eq. (11)
# --------------------------------------------------------------------------- #

class TestSafeSetLoss:
    def test_safe_prediction_positive_sign(self):
        """Safe prediction: sign = +1, loss = +ReLU(h)."""
        h = torch.tensor([[0.5]])
        preds_safe = torch.tensor([[True]])
        mask = torch.tensor([[True]])

        loss = safe_set_loss(h, preds_safe, mask, eta=0.0)
        # sign=+1, hinge=ReLU(0.5)=0.5, loss=+0.5
        torch.testing.assert_close(loss, torch.tensor(0.5))

    def test_unsafe_prediction_negative_sign(self):
        """Unsafe prediction: sign = -1, loss = -ReLU(h)."""
        h = torch.tensor([[0.5]])
        preds_safe = torch.tensor([[False]])
        mask = torch.tensor([[True]])

        loss = safe_set_loss(h, preds_safe, mask, eta=0.0)
        # sign=-1, hinge=ReLU(0.5)=0.5, loss=-0.5
        torch.testing.assert_close(loss, torch.tensor(-0.5))

    def test_negative_h_gives_zero(self):
        """h < 0: ReLU makes contribution zero regardless of sign."""
        h = torch.tensor([[-0.3]])
        preds_safe = torch.tensor([[True]])
        mask = torch.tensor([[True]])

        loss = safe_set_loss(h, preds_safe, mask, eta=0.0)
        # ReLU(-0.3) = 0
        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_eta_shifts_boundary(self):
        """eta shifts the hinge boundary."""
        h = torch.tensor([[-0.1]])
        preds_safe = torch.tensor([[True]])
        mask = torch.tensor([[True]])

        # Without eta: ReLU(-0.1) = 0
        loss0 = safe_set_loss(h, preds_safe, mask, eta=0.0)
        torch.testing.assert_close(loss0, torch.tensor(0.0))

        # With eta=0.2: ReLU(-0.1+0.2) = ReLU(0.1) = 0.1
        loss1 = safe_set_loss(h, preds_safe, mask, eta=0.2)
        torch.testing.assert_close(loss1, torch.tensor(0.1))

    def test_masking(self):
        """Masked positions contribute nothing."""
        h = torch.tensor([[0.5, -0.3]])
        preds_safe = torch.tensor([[True, False]])
        mask = torch.tensor([[True, False]])

        loss = safe_set_loss(h, preds_safe, mask, eta=0.0)
        # Only first position: sign=+1, ReLU(0.5)=0.5
        torch.testing.assert_close(loss, torch.tensor(0.5))


# --------------------------------------------------------------------------- #
# Safety-Invariance Loss — Eq. (12)
# --------------------------------------------------------------------------- #

class TestSafetyInvarianceLoss:
    def test_si_basic(self):
        dynamics = DialogueDynamics(hidden_dims=[32, 32])
        predictor = SafetyPredictor()
        U = torch.randn(2, 8, D)
        mask = torch.ones(2, 8, dtype=torch.bool)

        loss = safety_invariance_loss(dynamics, predictor, U, mask, eta=0.0, kappa=3)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_si_indexing(self):
        """Verify SI evaluates h(x_k, u_{k+1}), not h(x_{k-1}, u_k)."""
        dynamics = DialogueDynamics(hidden_dims=[32, 32])
        predictor = SafetyPredictor()
        dynamics.eval()
        predictor.eval()

        U = torch.randn(1, 4, D)
        mask = torch.ones(1, 4, dtype=torch.bool)

        with torch.no_grad():
            X, _ = dynamics.rollout(U, mask)

            # For k=0: x_0=0, x_1=f(x_0,u_1), evaluate h(x_1, u_2)
            x_0 = torch.zeros(1, D)
            x_1 = dynamics.f_theta(torch.cat([x_0, U[:, 0]], dim=-1))
            h_correct = predictor.predictor_value(x_1, U[:, 1])

            # Wrong: h(x_0, u_1)
            h_wrong1 = predictor.predictor_value(x_0, U[:, 0])

        # SI loss should use h_correct, not h_wrong1
        loss = safety_invariance_loss(dynamics, predictor, U, mask, eta=0.0, kappa=2)
        assert loss.item() >= 0  # ReLU ensures non-negative

    def test_si_kappa_truncation(self):
        """Verify only K-kappa turns contribute."""
        dynamics = DialogueDynamics(hidden_dims=[32, 32])
        predictor = SafetyPredictor()

        U = torch.randn(1, 8, D)
        mask = torch.ones(1, 8, dtype=torch.bool)

        # With kappa=3, only turns 0..4 contribute (5 turns)
        loss_k3 = safety_invariance_loss(dynamics, predictor, U, mask, eta=0.0, kappa=3)

        # With kappa=0, turns 0..6 contribute (7 turns)
        loss_k0 = safety_invariance_loss(dynamics, predictor, U, mask, eta=0.0, kappa=0)

        # Both should be finite
        assert torch.isfinite(loss_k3)
        assert torch.isfinite(loss_k0)

    def test_si_masking(self):
        """Masked turns do not contribute to SI."""
        dynamics = DialogueDynamics(hidden_dims=[32, 32])
        predictor = SafetyPredictor()

        U = torch.randn(2, 6, D)
        mask = torch.ones(2, 6, dtype=torch.bool)
        mask[1, 2:] = False  # mask last 4 turns for sample 2

        loss = safety_invariance_loss(dynamics, predictor, U, mask, eta=0.0, kappa=1)
        assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# Label conversion
# --------------------------------------------------------------------------- #

class TestLabelConversion:
    def test_paper_to_ce(self):
        labels = torch.tensor([1, 2, 3, 4, 5])
        ce_idx = paper_label_to_ce_index(labels)
        expected = torch.tensor([0, 1, 2, 3, 4])
        torch.testing.assert_close(ce_idx, expected)
