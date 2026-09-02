"""Tests for Phase 6 multi-turn attack framework.

These tests verify:

- Interface conformance for all attack classes.
- Stub contract: the 5 paper-strict attack stubs raise
  ``NotImplementedError`` from ``next_query`` until the official
  implementations are pasted in.  This is intentional under
  ``paper-strict`` mode.
- Adaptive attack selector (paper-defined) still works end-to-end.
- Runner / registry / resume plumbing are correct using a tiny local
  ``MultiTurnAttack`` subclass — these prove the harness is correct
  independent of the (not-yet-pasted) official attack code.

All tests are offline — no API calls.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from guardbound.attacks.base import MultiTurnAttack
from guardbound.attacks.crescendo import CrescendoAttack
from guardbound.attacks.actor_attack import ActorAttack
from guardbound.attacks.opposite_day import OppositeDayAttack
from guardbound.attacks.acronym import AcronymAttack
from guardbound.attacks.red_queen import RedQueenAttack
from guardbound.attacks.runner import run_attack
from guardbound.attacks.registry import (
    get_attack,
    available_attacks,
    DEFAULT_EVALUATION_ATTACKS,
)
from guardbound.llm.mock import MockChatLLM
from guardbound.schemas import Conversation, Turn
from guardbound.models.dynamics import DialogueDynamics
from guardbound.models.predictor import NeuralBarrierFunction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm(responses: list[str] | None = None) -> MockChatLLM:
    return MockChatLLM(responses=responses)


def _make_barrier() -> NeuralBarrierFunction:
    from guardbound.models.predictor import SafetyPredictor
    dynamics = DialogueDynamics(state_dim=768, embedding_dim=768)
    predictor = SafetyPredictor(state_dim=768, embedding_dim=768)
    return NeuralBarrierFunction(
        dynamics=dynamics, predictor=predictor, embedding_model="mpnet",
    )


class _LocalEchoAttack(MultiTurnAttack):
    """Local-only test attack: returns a deterministic, non-empty query.

    Used to exercise the runner / registry / resume plumbing without
    depending on the (not-yet-pasted) official attack implementations.
    """
    name = "local_echo"

    def __init__(self, max_turns: int = 8):
        self._max_turns = max_turns

    def next_query(self, goal, history):
        return f"echo:{goal}:turn{len(history)}"

    def is_finished(self, history, max_turns=8):
        return len(history) >= max_turns


# ---------------------------------------------------------------------------
# TEST A — Interface conformance
# ---------------------------------------------------------------------------

ATTACK_CLASSES = [
    CrescendoAttack,
    ActorAttack,
    OppositeDayAttack,
    AcronymAttack,
    RedQueenAttack,
]


@pytest.mark.parametrize("cls", ATTACK_CLASSES, ids=lambda c: c.__name__)
def test_attack_is_multiturn(cls):
    """All attacks must implement MultiTurnAttack."""
    attack = cls()
    assert isinstance(attack, MultiTurnAttack)
    assert hasattr(attack, "next_query")
    assert hasattr(attack, "is_finished")
    assert hasattr(attack, "name")
    # name must be a non-empty string
    assert isinstance(attack.name, str) and attack.name


def test_adaptive_is_multiturn():
    """AdaptiveNBFAttack must also implement MultiTurnAttack."""
    from guardbound.attacks.adaptive import AdaptiveNBFAttack
    assert issubclass(AdaptiveNBFAttack, MultiTurnAttack)


# ---------------------------------------------------------------------------
# TEST B — Paper-strict stub contract
# ---------------------------------------------------------------------------

STUB_CLASSES = [
    CrescendoAttack,
    ActorAttack,
    OppositeDayAttack,
    AcronymAttack,
    RedQueenAttack,
]


@pytest.mark.parametrize("cls", STUB_CLASSES, ids=lambda c: c.__name__)
def test_stub_next_query_raises_not_implemented(cls):
    """Under paper-strict mode the 5 non-adaptive attacks ship as stubs.

    next_query() must raise NotImplementedError so the failure mode is
    loud and explicit when these attacks are invoked before their
    official implementations are pasted in.
    """
    attack = cls()
    with pytest.raises(NotImplementedError) as excinfo:
        attack.next_query("test goal", [])
    assert "paper-strict" in str(excinfo.value).lower() or "stub" in str(excinfo.value).lower()


@pytest.mark.parametrize("cls", STUB_CLASSES, ids=lambda c: c.__name__)
def test_stub_is_finished_works(cls):
    """is_finished() must still work on the stubs (no LLM call needed)."""
    attack = cls()
    # No history -> not finished
    assert attack.is_finished([], max_turns=3) is False
    # 3 turns -> finished
    history = [Turn(query=f"q{i}", response=f"r{i}") for i in range(3)]
    assert attack.is_finished(history, max_turns=3) is True


def test_acronym_is_training_only():
    """Acronym must be flagged as training-only and absent from the
    default evaluation list (NBF paper Sec. 5.1)."""
    assert "acronym" not in DEFAULT_EVALUATION_ATTACKS


# ---------------------------------------------------------------------------
# TEST C — Registry
# ---------------------------------------------------------------------------

def test_registry_lists_all_attacks():
    """Registry must expose every attack by its canonical name."""
    expected = {"crescendo", "actor_attack", "opposite_day", "acronym",
                "red_queen", "adaptive"}
    assert expected.issubset(set(available_attacks()))


def test_get_attack_returns_correct_types():
    for name, cls in [
        ("crescendo", CrescendoAttack),
        ("actor_attack", ActorAttack),
        ("opposite_day", OppositeDayAttack),
        ("acronym", AcronymAttack),
        ("red_queen", RedQueenAttack),
    ]:
        assert isinstance(get_attack(name), cls)


def test_get_attack_unknown_raises():
    with pytest.raises(ValueError, match="Unknown attack"):
        get_attack("nonexistent_attack")


def test_get_attack_adaptive_returns_instance():
    """get_attack('adaptive') must succeed and produce a MultiTurnAttack."""
    attack = get_attack("adaptive")
    assert isinstance(attack, MultiTurnAttack)


# ---------------------------------------------------------------------------
# TEST D — Runner plumbing (uses local echo attack)
# ---------------------------------------------------------------------------

def test_runner_bare_llm_mode_produces_turns():
    """The runner must execute the attack loop in bare-LLM mode."""
    attack = _LocalEchoAttack()
    llm = _make_mock_llm()
    conv = run_attack(attack=attack, goal="test goal", target_llm=llm, max_turns=3)
    assert len(conv.turns) == 3
    assert all(t.response is not None and not t.was_filtered for t in conv.turns)
    assert conv.attack_method == "local_echo"


def test_runner_respects_max_turns():
    attack = _LocalEchoAttack()
    llm = _make_mock_llm()
    conv = run_attack(attack=attack, goal="g", target_llm=llm, max_turns=5)
    assert conv.max_turns == 5
    assert len(conv.turns) == 5


def test_runner_nbf_mode_allow_all():
    """With a barrier that returns very negative h, all queries pass."""
    attack = _LocalEchoAttack()
    llm = _make_mock_llm()
    barrier = _make_barrier()
    barrier.h = lambda s, u: torch.full((s.shape[0],), -10.0)

    conv = run_attack(
        attack=attack, goal="g", target_llm=llm, max_turns=3,
        embed_fn=lambda t: torch.zeros(1, 768),
        barrier=barrier, eta=0.0,
    )
    assert len(conv.turns) == 3
    for t in conv.turns:
        assert t.response is not None
        assert not t.was_filtered
    # LLM was actually called
    assert len(llm.calls) == 3


def test_runner_nbf_mode_filter_all():
    """With a barrier that returns very positive h, no LLM calls happen."""
    attack = _LocalEchoAttack()
    llm = _make_mock_llm()
    barrier = _make_barrier()
    barrier.h = lambda s, u: torch.full((s.shape[0],), 10.0)

    conv = run_attack(
        attack=attack, goal="g", target_llm=llm, max_turns=4,
        embed_fn=lambda t: torch.zeros(1, 768),
        barrier=barrier, eta=0.0,
    )
    assert len(conv.turns) == 4
    for t in conv.turns:
        assert t.response is None
        assert t.was_filtered is True
    # No LLM calls
    assert len(llm.calls) == 0


# ---------------------------------------------------------------------------
# TEST E — Adaptive attack selector (paper-defined)
# ---------------------------------------------------------------------------

def test_adaptive_evaluates_exactly_3_candidates():
    """Adaptive attack must evaluate exactly 3 candidates per next_query call."""
    from guardbound.attacks.adaptive import AdaptiveNBFAttack
    base = _LocalEchoAttack()
    barrier = _make_barrier()
    counter = [0]
    barrier.h = lambda s, u: (counter.__setitem__(0, counter[0] + 1) or torch.zeros(s.shape[0]))
    attack = AdaptiveNBFAttack(
        base_attack=base, embed_fn=lambda t: torch.zeros(1, 768),
        barrier=barrier, state_fn=lambda: torch.zeros(1, 768),
        num_candidates=3,
    )
    attack.next_query("g", [])
    assert counter[0] == 3


def test_adaptive_selects_argmax_h_with_distinct_candidates():
    """argmax-h selection: with candidates alpha/beta/gamma and h=-0.2/+0.4/+0.1
    the selector must return 'beta'."""
    from guardbound.attacks.adaptive import AdaptiveNBFAttack

    class StatefulBase(MultiTurnAttack):
        name = "stateful"
        def __init__(self):
            self.i = 0
        def next_query(self, goal, history):
            q = ["alpha", "beta", "gamma"][self.i % 3]
            self.i += 1
            return q
        def is_finished(self, history, max_turns=8):
            return len(history) >= max_turns

    base = StatefulBase()
    barrier = _make_barrier()
    queue = [-0.2, 0.4, 0.1]
    call_count = [0]
    def h_seq(s, u):
        v = queue[call_count[0]] if call_count[0] < len(queue) else 0.0
        call_count[0] += 1
        return torch.tensor([v], dtype=torch.float32, device=s.device)
    barrier.h = h_seq

    attack = AdaptiveNBFAttack(
        base_attack=base, embed_fn=lambda t: torch.zeros(1, 768),
        barrier=barrier, state_fn=lambda: torch.zeros(1, 768),
        num_candidates=3,
    )
    q = attack.next_query("g", [])
    assert call_count[0] == 3
    assert q == "beta", f"argmax-h should select 'beta' (h=+0.4), got {q!r}"


def test_adaptive_num_candidates_override():
    """num_candidates must drive the number of h-evaluations."""
    from guardbound.attacks.adaptive import AdaptiveNBFAttack
    base = _LocalEchoAttack()
    barrier = _make_barrier()
    counter = [0]
    barrier.h = lambda s, u: (counter.__setitem__(0, counter[0] + 1) or torch.zeros(s.shape[0]))
    attack = AdaptiveNBFAttack(
        base_attack=base, embed_fn=lambda t: torch.zeros(1, 768),
        barrier=barrier, state_fn=lambda: torch.zeros(1, 768),
        num_candidates=5,
    )
    attack.next_query("g", [])
    assert counter[0] == 5


def test_adaptive_wires_state_via_runner():
    """The runner must auto-wire the adaptive attack's state_fn to the
    steered chat's current state."""
    from guardbound.attacks.adaptive import AdaptiveNBFAttack
    from guardbound.defense.steered_chat import SteeredLLMChat

    class StatefulBase(MultiTurnAttack):
        name = "stateful"
        def __init__(self):
            self.i = 0
        def next_query(self, goal, history):
            q = ["alpha", "beta", "gamma"][self.i % 3]
            self.i += 1
            return q
        def is_finished(self, history, max_turns=8):
            return len(history) >= max_turns

    base = StatefulBase()
    barrier = _make_barrier()
    queue = [-0.2, 0.4, 0.1]
    counter = [0]
    def h_seq(s, u):
        v = queue[counter[0]] if counter[0] < len(queue) else 0.0
        counter[0] += 1
        return torch.tensor([v], dtype=torch.float32, device=s.device)
    barrier.h = h_seq

    attack = AdaptiveNBFAttack(
        base_attack=base, embed_fn=lambda t: torch.zeros(1, 768),
        barrier=barrier, state_fn=None,  # runner should wire this
        num_candidates=3,
    )
    llm = _make_mock_llm()
    conv = run_attack(
        attack=attack, goal="g", target_llm=llm, max_turns=1,
        embed_fn=attack.embed_fn, barrier=barrier, eta=-10.0,
    )
    # First 3 h-calls are the 3 candidate evaluations; argmax selects beta
    assert conv.turns[0].query == "beta"


# ---------------------------------------------------------------------------
# TEST F — Resume (using local echo attack)
# ---------------------------------------------------------------------------

def test_resume_skips_existing():
    from guardbound.attacks.runner import run_attack_batch
    attack = _LocalEchoAttack()
    llm = _make_mock_llm()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "conversations.jsonl"
        run_attack_batch(
            attack=attack, goals=["Goal A", "Goal B"],
            target_llm=llm, output_path=out_path, max_turns=2,
        )
        # Second run: must skip already-completed goals
        convs2 = run_attack_batch(
            attack=attack, goals=["Goal A", "Goal B", "Goal C"],
            target_llm=llm, output_path=out_path, max_turns=2,
        )
        assert len(convs2) == 3
        with open(out_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 3


def test_resume_no_duplicate_turns():
    from guardbound.attacks.runner import run_attack_batch
    from guardbound.schemas import load_conversations_jsonl
    attack = _LocalEchoAttack()
    llm = _make_mock_llm()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "conversations.jsonl"
        run_attack_batch(
            attack=attack, goals=["Repeat me"],
            target_llm=llm, output_path=out_path, max_turns=3,
        )
        first = load_conversations_jsonl(out_path)
        assert len(first) == 1
        first_turns = len(first[0].turns)
        run_attack_batch(
            attack=attack, goals=["Repeat me"],
            target_llm=llm, output_path=out_path, max_turns=3,
        )
        second = load_conversations_jsonl(out_path)
        assert len(second) == 1
        assert len(second[0].turns) == first_turns


# ---------------------------------------------------------------------------
# TEST G — Determinism (using local echo attack)
# ---------------------------------------------------------------------------

def test_determinism_local_echo():
    """Identical inputs (deterministic attack) must produce identical queries."""
    a1, a2 = _LocalEchoAttack(), _LocalEchoAttack()
    llm1 = _make_mock_llm(responses=["r1", "r2"])
    llm2 = _make_mock_llm(responses=["r1", "r2"])
    conv1 = run_attack(attack=a1, goal="g", target_llm=llm1, max_turns=2)
    conv2 = run_attack(attack=a2, goal="g", target_llm=llm2, max_turns=2)
    assert [t.query for t in conv1.turns] == [t.query for t in conv2.turns]


# ---------------------------------------------------------------------------
# TEST H — JSONL round-trip (uses local echo)
# ---------------------------------------------------------------------------

def test_jsonl_roundtrip():
    from guardbound.schemas import save_conversations_jsonl, load_conversations_jsonl
    attack = _LocalEchoAttack()
    llm = _make_mock_llm()
    conv = run_attack(attack=attack, goal="g", target_llm=llm, max_turns=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "c.jsonl"
        save_conversations_jsonl([conv], out)
        loaded = load_conversations_jsonl(out)
        assert len(loaded) == 1
        assert loaded[0].goal == conv.goal
        assert loaded[0].attack_method == conv.attack_method
        assert len(loaded[0].turns) == len(conv.turns)
        for t1, t2 in zip(loaded[0].turns, conv.turns):
            assert t1.query == t2.query
            assert t1.response == t2.response
            assert t1.was_filtered == t2.was_filtered


# ---------------------------------------------------------------------------
# TEST I — Filtered state unchanged (Phase 5 contract)
# ---------------------------------------------------------------------------

def test_filtered_state_unchanged():
    """Filtered queries must not change the dialogue state (Phase 5 contract)."""
    llm = _make_mock_llm()
    barrier = _make_barrier()
    barrier.h = lambda s, u: torch.full((s.shape[0],), 10.0)
    from guardbound.defense.steered_chat import SteeredLLMChat
    chat = SteeredLLMChat(
        target_llm=llm, barrier=barrier, eta=0.0, max_turns=5,
    )
    initial_state = chat._state.clone()
    result = chat.chat("test", lambda t: torch.zeros(1, 768))
    assert result.filtered is True
    assert torch.equal(chat._state, initial_state), "State changed after filtering!"
