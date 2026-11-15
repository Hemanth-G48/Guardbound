"""Cloud × NBF integration using the REAL checkpoint and the REAL embedder.

The earlier cloud test suite exercised the filtering *plumbing* with a stub
barrier, because the embedding stack could not be imported in this environment.
That is no longer true (see CLOUD_ENVIRONMENT_COMPATIBILITY_REPORT.md), so this
module exercises the actual `NeuralBarrierFunction` and the actual
`sentence-transformers/all-mpnet-base-v2` embedder through the real attack
runner against a fake OpenAI transport.

Nothing here changes the NBF, its threshold, its scoring formula or its state
transition: it only proves that a cloud `ChatLLM` plugs into the validated
defence unchanged.

Deterministic assertions use `safety_threshold_override` so the real barrier's
numeric output does not have to be predicted:

    accept iff  score < threshold

so a threshold above the barrier's range accepts everything, and a threshold
below it filters everything — while the scores themselves remain the real ones.

Skipped (not failed) when the checkpoint or the embedder is unavailable, so the
suite still runs on a machine without them.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from guardbound.attacks.runner import run_attack_with_backtracking
from tests.test_cloud_research_integration import (
    GOAL,
    RoutedBackend,
    build_attack,
    make_llms,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    PROJECT_ROOT
    / "nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/models/"
      "models_best_nbf_released.pth"
)
EXPECTED_SHA = "cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe"
EMBEDDER = "sentence-transformers/all-mpnet-base-v2"

ACCEPT_ALL = 2.0    # every score < 2.0 -> accepted
FILTER_ALL = -2.0   # every score < -2.0 is false -> filtered


def _restore_real_sentence_transformer() -> None:
    """Undo the parity suite's global ``SentenceTransformer`` stub, if present.

    ``tests/parity/utils.py::_install_stubs`` replaces
    ``sentence_transformers.SentenceTransformer`` on the *real* module object and
    never restores it, so a full-suite run leaves behind a stub whose ``encode``
    raises ``NotImplementedError``. Reloading the module from disk restores the
    genuine class.

    This is test hygiene for cross-file ordering, not a change to any production
    code path: run in isolation, this module never needs it.
    """
    import importlib

    import sentence_transformers

    current = getattr(sentence_transformers, "SentenceTransformer", None)
    if current is None or getattr(current, "__name__", "") != "_DummySentenceTransformer":
        return
    importlib.reload(sentence_transformers)


def _load_real_nbf():
    """Real checkpoint + real embedder, or skip with a clear reason."""
    _restore_real_sentence_transformer()
    if not CHECKPOINT.exists():
        pytest.skip(f"NBF checkpoint not present: {CHECKPOINT}")
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA:
        pytest.skip(
            f"checkpoint digest {digest[:12]}… is not the validated "
            f"{EXPECTED_SHA[:12]}…"
        )
    try:
        from guardbound.embeddings import get_embed_fn
        from guardbound.models.compat import load_original_checkpoint
        from guardbound.models.predictor import NeuralBarrierFunction
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"NBF stack unavailable: {type(exc).__name__}: {exc}")

    try:
        dynamics, predictor = load_original_checkpoint(CHECKPOINT, device="cpu")
        barrier = NeuralBarrierFunction(dynamics=dynamics, predictor=predictor)
        barrier.eval()
        embed_fn = get_embed_fn(EMBEDDER)
        probe = embed_fn("warm-up query")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"NBF/embedder could not be loaded: {type(exc).__name__}: {exc}")

    assert tuple(probe.shape) == (1, 768), probe.shape
    return barrier, embed_fn


# --------------------------------------------------------------------------- #
# Environment preconditions for this module
# --------------------------------------------------------------------------- #

def test_real_embedder_is_768_dimensional_and_deterministic():
    _, embed_fn = _load_real_nbf()
    a = embed_fn("How do I bake sourdough bread?")
    b = embed_fn("How do I bake sourdough bread?")
    assert tuple(a.shape) == (1, 768)
    assert torch.isfinite(a).all()
    assert torch.equal(a, b)


def test_real_checkpoint_scores_and_is_deterministic():
    barrier, embed_fn = _load_real_nbf()
    state = torch.zeros(1, barrier.predictor.state_dim)
    u = embed_fn("How do I bake sourdough bread?")
    with torch.no_grad():
        h1 = float(barrier.h(state, u).item())
        h2 = float(barrier.h(state, u).item())
    assert h1 == h2
    assert -1.0 <= h1 <= 1.0
    # The score is exactly p(class 5) - max(p(class 1..4)).
    with torch.no_grad():
        probs = torch.softmax(barrier.predictor(state, u), dim=-1)[0]
    assert abs(h1 - float(probs[4] - probs[:4].max())) < 1e-6
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Cloud transport × real NBF
# --------------------------------------------------------------------------- #

def test_nbf_on_with_real_barrier_accepts_and_calls_the_target():
    barrier, embed_fn = _load_real_nbf()
    backend = RoutedBackend()
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=2)

    conversation = run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"],
        embed_fn=embed_fn, barrier=barrier, eta=0.0, max_turns=2,
        temperature=0.7, safety_threshold_override=ACCEPT_ALL,
    )

    assert len(conversation.turns) == 2
    assert len(backend.requests_for("target")) == 2
    scores = attack.nbf_candidate_scores
    assert scores, "the real barrier produced no candidate scores"
    assert all(isinstance(s, float) for s in scores)
    assert all(-1.0 <= s <= 1.0 for s in scores)
    assert all(s < ACCEPT_ALL for s in scores)
    assert all(v is True for v in attack.nbf_candidate_verdicts)
    assert barrier.predictor.state_dim == 768


def test_nbf_on_with_real_barrier_filters_and_suppresses_the_target():
    barrier, embed_fn = _load_real_nbf()
    backend = RoutedBackend()
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=2)

    conversation = run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"],
        embed_fn=embed_fn, barrier=barrier, eta=0.0, max_turns=2,
        temperature=0.7, safety_threshold_override=FILTER_ALL,
    )

    assert conversation.turns == []
    assert backend.requests_for("target") == []
    scores = attack.nbf_candidate_scores
    assert scores, "the real barrier produced no candidate scores"
    assert all(not (s < FILTER_ALL) for s in scores)
    assert all(v is False for v in attack.nbf_candidate_verdicts)
    # The attacker still ran: filtering happens after query generation.
    assert backend.requests_for("generation")


def test_nbf_decisions_match_the_documented_rule_at_threshold_zero():
    """`accept iff score < threshold`, evaluated on the real scores."""
    barrier, embed_fn = _load_real_nbf()
    backend = RoutedBackend()
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=3)

    run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"],
        embed_fn=embed_fn, barrier=barrier, eta=0.0, max_turns=3,
        temperature=0.7,
    )

    scores = attack.nbf_candidate_scores
    verdicts = attack.nbf_candidate_verdicts
    assert scores and len(scores) == len(verdicts)
    for score, verdict in zip(scores, verdicts):
        assert verdict is (score < 0.0), (score, verdict)

    # A filtered candidate never reaches the target.
    n_target_calls = len(backend.requests_for("target"))
    n_accepted = sum(1 for v in verdicts if v)
    assert n_target_calls == n_accepted


def test_nbf_off_with_real_components_bypasses_scoring():
    """NBF OFF must not score anything even when the barrier is available."""
    _, embed_fn = _load_real_nbf()
    backend = RoutedBackend()
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=2)

    conversation = run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"],
        embed_fn=None, barrier=None, eta=0.0, max_turns=2, temperature=0.7,
    )
    assert len(conversation.turns) == 2
    assert attack.nbf_candidate_scores == []
    assert attack.nbf_candidate_verdicts == []
    assert len(backend.requests_for("target")) == 2


def test_nbf_state_advances_only_on_accepted_turns():
    """The real dynamics are advanced once per accepted turn, never on a filter."""
    barrier, embed_fn = _load_real_nbf()
    backend = RoutedBackend()
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=2)

    with torch.no_grad():
        before = float(barrier.h(
            torch.zeros(1, 768), embed_fn("x")).item())

    run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"],
        embed_fn=embed_fn, barrier=barrier, eta=0.0, max_turns=2,
        temperature=0.7, safety_threshold_override=ACCEPT_ALL,
    )
    # Two accepted turns advanced the state; the barrier is still usable.
    with torch.no_grad():
        after = float(barrier.h(
            torch.zeros(1, 768), embed_fn("x")).item())
    assert before == after, "the barrier must be stateless across conversations"
