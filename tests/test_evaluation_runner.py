"""Tests for evaluation runner — Phase 6/7 handoff interface."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardbound.defense.evaluation_runner import run_attack_under_steering
from guardbound.defense.steered_chat import SteeredLLMChat
from guardbound.llm.mock import MockChatLLM
from guardbound.models.dynamics import DialogueDynamics
from guardbound.models.predictor import SafetyPredictor, NeuralBarrierFunction
from guardbound.schemas import Conversation

D = 768


class MockAttack:
    """Minimal mock attack for testing the runner interface."""

    def generate(self, goal, target_llm, max_turns=8, temperature=0.7):
        # Simulate a 2-turn attack
        conv = Conversation(
            goal=goal,
            attack_method="mock_attack",
            target_llm="mock",
            max_turns=max_turns,
        )
        return conv


class TestEvaluationRunner:
    def test_runner_returns_conversation(self):
        nbf = NeuralBarrierFunction(
            DialogueDynamics(hidden_dims=[32, 32]),
            SafetyPredictor(),
        )
        nbf.eval()
        llm = MockChatLLM(responses=["response"])
        embed_fn = lambda text: torch.randn(1, D)
        attack = MockAttack()

        conv = run_attack_under_steering(
            attack=attack,
            target_llm=llm,
            barrier=nbf,
            eta=5e-4,
            embed_fn=embed_fn,
            goal="test goal",
            attack_method="mock_attack",
        )

        assert isinstance(conv, Conversation)
        assert conv.goal == "test goal"

    def test_steered_chat_as_target(self):
        """SteeredLLMChat can be used as a target for attacks."""
        nbf = NeuralBarrierFunction(
            DialogueDynamics(hidden_dims=[32, 32]),
            SafetyPredictor(),
        )
        nbf.eval()
        llm = MockChatLLM(responses=["response"])
        embed_fn = lambda text: torch.randn(1, D)

        chat = SteeredLLMChat(
            target_llm=llm,
            barrier=nbf,
            eta=5e-4,
        )

        # chat has the generate interface needed by attacks
        assert hasattr(chat, 'chat')
        assert hasattr(chat, 'get_state')
        assert hasattr(chat, 'get_turns')
