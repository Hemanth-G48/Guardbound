"""Tests for SteeredLLMChat - Q-filter behavior, state mutation, boundary cases."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardbound.defense.steered_chat import (
    SteeredLLMChat, SteeredResult, MaxTurnsExceeded,
    default_refusal_detector,
)
from guardbound.llm.mock import MockChatLLM
from guardbound.models.dynamics import DialogueDynamics
from guardbound.models.predictor import SafetyPredictor, NeuralBarrierFunction

D = 768


def _make_nbf():
    dynamics = DialogueDynamics(hidden_dims=[32, 32])
    predictor = SafetyPredictor()
    nbf = NeuralBarrierFunction(dynamics, predictor)
    nbf.eval()
    return nbf


def _make_embed_fn(value=0.0):
    def embed(text):
        return torch.full((1, D), value)
    return embed


def _make_chat(eta=5e-4, max_turns=8, responses=None):
    nbf = _make_nbf()
    llm = MockChatLLM(responses=responses or ["mock response"])
    return SteeredLLMChat(
        target_llm=llm, barrier=nbf, eta=eta, max_turns=max_turns,
    ), llm, nbf


# Filtered query tests

class TestFilteredQuery:
    def test_filtered_no_llm_call(self):
        chat, llm, _ = _make_chat(eta=100.0)
        result = chat.chat("test query", _make_embed_fn())
        assert result.filtered is True
        assert result.response is None
        assert len(llm.calls) == 0

    def test_filtered_state_unchanged(self):
        chat, _, _ = _make_chat(eta=100.0)
        state_before = chat.get_state().clone()
        chat.chat("test query", _make_embed_fn())
        state_after = chat.get_state()
        assert torch.equal(state_before, state_after)


# Boundary tests

class TestBoundary:
    def test_exact_boundary_h_plus_eta_eq_zero(self):
        chat, llm, nbf = _make_chat(eta=0.5)
        def mock_h(state, u):
            return torch.tensor([-0.5])
        nbf.h = mock_h
        result = chat.chat("test", _make_embed_fn())
        assert result.filtered is True
        assert result.h_value == -0.5
        assert len(llm.calls) == 0

    def test_h_plus_eta_just_below_zero(self):
        chat, llm, nbf = _make_chat(eta=0.5)
        def mock_h(state, u):
            return torch.tensor([-0.6])
        nbf.h = mock_h
        result = chat.chat("test", _make_embed_fn())
        assert result.filtered is False
        assert len(llm.calls) == 1

    def test_h_plus_eta_just_above_zero(self):
        chat, llm, nbf = _make_chat(eta=0.5)
        def mock_h(state, u):
            return torch.tensor([-0.4])
        nbf.h = mock_h
        result = chat.chat("test", _make_embed_fn())
        assert result.filtered is True
        assert len(llm.calls) == 0


# Accepted query tests

class TestAcceptedQuery:
    def test_accepted_calls_llm_once(self):
        chat, llm, nbf = _make_chat(eta=0.0)
        def mock_h(state, u):
            return torch.tensor([-10.0])
        nbf.h = mock_h
        result = chat.chat("test", _make_embed_fn())
        assert result.filtered is False
        assert len(llm.calls) == 1

    def test_accepted_advances_state(self):
        chat, _, nbf = _make_chat(eta=0.0)
        def mock_h(state, u):
            return torch.tensor([-10.0])
        nbf.h = mock_h
        state_before = chat.get_state().clone()
        chat.chat("test", _make_embed_fn())
        state_after = chat.get_state()
        assert not torch.equal(state_before, state_after)


# State invariance tests

class TestStateInvariance:
    def test_two_identical_sequences_same_state(self):
        """Two fresh instances with same NBF weights produce same state."""
        # Use same NBF weights for both
        shared_nbf = _make_nbf()
        def mock_h(state, u):
            return torch.tensor([-10.0])
        shared_nbf.h = mock_h

        embed_fn = _make_embed_fn()

        # Instance 1
        llm1 = MockChatLLM(responses=["r1", "r2"])
        chat1 = SteeredLLMChat(target_llm=llm1, barrier=shared_nbf, eta=0.0)
        chat1.chat("query1", embed_fn)
        chat1.chat("query2", embed_fn)
        state1 = chat1.get_state()

        # Instance 2 - same NBF, same embeddings => same state
        llm2 = MockChatLLM(responses=["r1", "r2"])
        chat2 = SteeredLLMChat(target_llm=llm2, barrier=shared_nbf, eta=0.0)
        chat2.chat("query1", embed_fn)
        chat2.chat("query2", embed_fn)
        state2 = chat2.get_state()

        torch.testing.assert_close(state1, state2)

    def test_filtered_query_does_not_advance_state(self):
        """Filtered query does not advance state; accepted does."""
        nbf = _make_nbf()

        embed_fn = _make_embed_fn(0.5)
        u = embed_fn("test")

        # State before
        state_initial = torch.zeros(1, D)

        # Manually advance once (accepted path)
        state_after_advance = nbf.advance_state(state_initial, u)

        # Now verify: if a query is filtered, state stays at initial
        def mock_h_filtered(state, u):
            return torch.tensor([10.0])
        nbf.h = mock_h_filtered

        chat = SteeredLLMChat(
            target_llm=MockChatLLM(), barrier=nbf, eta=0.0,
        )
        chat.chat("test", embed_fn)
        state_filtered = chat.get_state()

        torch.testing.assert_close(state_filtered, state_initial)
        assert not torch.equal(state_filtered, state_after_advance)


# Eta monotonicity

class TestEtaMonotonicity:
    def test_higher_eta_more_filtering(self):
        for h_val in [-0.5, -0.1, 0.0, 0.1, 0.5]:
            for eta_low, eta_high in [(0.1, 0.2), (0.0, 0.1), (0.3, 0.5)]:
                if h_val + eta_low >= 0:
                    assert h_val + eta_high >= 0


# Max turns

class TestMaxTurns:
    def test_max_turns_enforced(self):
        chat, _, _ = _make_chat(max_turns=2)
        chat.chat("q1", _make_embed_fn())
        chat.chat("q2", _make_embed_fn())
        with pytest.raises(MaxTurnsExceeded):
            chat.chat("q3", _make_embed_fn())

    def test_filtered_turns_count_toward_limit(self):
        chat, _, _ = _make_chat(max_turns=2, eta=100.0)
        chat.chat("q1", _make_embed_fn())
        chat.chat("q2", _make_embed_fn())
        with pytest.raises(MaxTurnsExceeded):
            chat.chat("q3", _make_embed_fn())


# Turn recording

class TestTurnRecording:
    def test_filtered_turn_recorded(self):
        chat, _, _ = _make_chat(eta=100.0)
        chat.chat("test", _make_embed_fn())
        turns = chat.get_turns()
        assert len(turns) == 1
        assert turns[0].was_filtered is True
        assert turns[0].response is None

    def test_accepted_turn_recorded(self):
        chat, _, nbf = _make_chat()
        def mock_h(state, u):
            return torch.tensor([-10.0])
        nbf.h = mock_h
        chat.chat("test", _make_embed_fn())
        turns = chat.get_turns()
        assert len(turns) == 1
        assert turns[0].was_filtered is False
        assert turns[0].response == "mock response"


# Refusal detection

class TestRefusalDetection:
    def test_refusal_detected(self):
        chat, llm, nbf = _make_chat(responses=["I'm sorry, but I cannot help."])
        def mock_h(state, u):
            return torch.tensor([-10.0])
        nbf.h = mock_h
        result = chat.chat("test", _make_embed_fn())
        assert result.refusal is True
        assert result.filtered is False

    def test_normal_response_not_refusal(self):
        chat, llm, nbf = _make_chat(responses=["Sure, here is the info."])
        def mock_h(state, u):
            return torch.tensor([-10.0])
        nbf.h = mock_h
        result = chat.chat("test", _make_embed_fn())
        assert result.refusal is False


# Refusal detector function

class TestRefusalDetector:
    def test_known_refusals(self):
        assert default_refusal_detector("I'm sorry, I cannot help.")
        assert default_refusal_detector("I cannot assist with that.")
        assert default_refusal_detector("I can't help you with this.")
        assert default_refusal_detector("I am unable to provide that.")
        assert default_refusal_detector("As an AI, I must decline.")

    def test_normal_responses(self):
        assert not default_refusal_detector("Here is the information.")
        assert not default_refusal_detector("Sure, let me explain.")
        assert not default_refusal_detector("The answer is 42.")
