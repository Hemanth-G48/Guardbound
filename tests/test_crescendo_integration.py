"""Phase 1 tests for Crescendo attack integration.

Tests verify:
- JSON generation support in ChatLLM backends
- CrescendoAttackPaper query generation
- Multi-turn Crescendo with mock LLMs
- Refusal/backtracking behavior
- NBF parity between original and Guardbound
- Checkpoint conversion compatibility
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


ORIGINAL_TRAIN_PATH = (
    Path(__file__).resolve().parents[1]
    / "nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/train.py"
)


def _load_original_train():
    """Import the original author's train.py via importlib.

    The module executes ``SentenceTransformer('all-mpnet-base-v2')`` and imports
    ``from torch.utils.tensorboard import SummaryWriter`` at import time
    (module-level side effects). We stub both out so tests stay cheap and
    offline; the architecture classes under test need neither the embedder
    nor tensorboard.
    """
    import importlib.util
    import sys
    import types

    import sentence_transformers

    class _DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *args, **kwargs):
            raise NotImplementedError("stubbed out in tests")

    class _DummySummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_scalars(self, *args, **kwargs):
            pass

        def add_histogram(self, *args, **kwargs):
            pass

        def close(self):
            pass

    original_st = sentence_transformers.SentenceTransformer
    fake_tb = types.ModuleType("torch.utils.tensorboard")
    fake_tb.SummaryWriter = _DummySummaryWriter
    had_fake_tb = "torch.utils.tensorboard" in sys.modules
    original_tb = sys.modules.get("torch.utils.tensorboard")

    sentence_transformers.SentenceTransformer = _DummySentenceTransformer
    sys.modules["torch.utils.tensorboard"] = fake_tb
    try:
        spec = importlib.util.spec_from_file_location("original_train", ORIGINAL_TRAIN_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sentence_transformers.SentenceTransformer = original_st
        if had_fake_tb:
            sys.modules["torch.utils.tensorboard"] = original_tb
        else:
            sys.modules.pop("torch.utils.tensorboard", None)
    return module


# ============================================================================
# Test A — JSON generation support
# ============================================================================

class TestJSONGeneration:
    """Verify ChatLLM.generate() json_format parameter works correctly."""

    def test_mock_generate_returns_str_by_default(self):
        """generate() without json_format returns a plain string."""
        from guardbound.llm.mock import MockChatLLM
        llm = MockChatLLM(responses=["hello world"])
        result = llm.generate([{"role": "user", "content": "hi"}])
        assert isinstance(result, str)
        assert result == "hello world"

    def test_mock_generate_json_format_dict(self):
        """generate(json_format=True) with valid JSON returns a dict."""
        from guardbound.llm.mock import MockChatLLM
        llm = MockChatLLM(responses=['{"key": "value", "num": 42}'])
        result = llm.generate([{"role": "user", "content": "hi"}], json_format=True)
        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["num"] == 42

    def test_mock_generate_json_format_non_dict(self):
        """generate(json_format=True) with a JSON list returns that list."""
        from guardbound.llm.mock import MockChatLLM
        llm = MockChatLLM(responses=['["a", "b", "c"]'])
        result = llm.generate([{"role": "user", "content": "hi"}], json_format=True)
        assert isinstance(result, list)
        assert result == ["a", "b", "c"]

    def test_mock_generate_json_format_malformed(self):
        """generate(json_format=True) with malformed JSON returns the raw string."""
        from guardbound.llm.mock import MockChatLLM
        llm = MockChatLLM(responses=["this is not json"])
        result = llm.generate([{"role": "user", "content": "hi"}], json_format=True)
        assert isinstance(result, str)
        assert result == "this is not json"

    def test_mock_generate_json_format_empty_response(self):
        """generate(json_format=True) with empty response returns empty string."""
        from guardbound.llm.mock import MockChatLLM
        llm = MockChatLLM(responses=[""])
        result = llm.generate([{"role": "user", "content": "hi"}], json_format=True)
        assert result == ""

    def test_mock_generate_json_format_uses_responses_exhaustively(self):
        """Each generate() call consumes one response, json_format or not."""
        from guardbound.llm.mock import MockChatLLM
        llm = MockChatLLM(responses=['{"a": 1}', '{"b": 2}', "plain text"])
        r1 = llm.generate([{"role": "user", "content": "q1"}], json_format=True)
        r2 = llm.generate([{"role": "user", "content": "q2"}], json_format=True)
        r3 = llm.generate([{"role": "user", "content": "q3"}])
        assert r1 == {"a": 1}
        assert r2 == {"b": 2}
        assert r3 == "plain text"


# ============================================================================
# Test B — Attacker LLM provisioning
# ============================================================================

class TestAttackerLLMProvisioning:
    """Verify CrescendoAttackPaper can be configured with a separate attacker LLM."""

    def test_set_attacker_llm_via_constructor(self):
        """Attacker LLM can be passed in constructor."""
        from guardbound.llm.mock import MockChatLLM
        from guardbound.attacks.crescendo_paper import CrescendoAttackPaper
        attacker = MockChatLLM(responses=['{"generatedQuestion": "q1", "lastResponseSummary": ""}'])
        attack = CrescendoAttackPaper(attacker_llm=attacker)
        assert attack._attacker_llm is attacker

    def test_set_attacker_llm_via_setter(self):
        """Attacker LLM can be set after construction."""
        from guardbound.llm.mock import MockChatLLM
        from guardbound.attacks.crescendo_paper import CrescendoAttackPaper
        attack = CrescendoAttackPaper()
        attacker = MockChatLLM(responses=['{"generatedQuestion": "q1", "lastResponseSummary": ""}'])
        attack.set_attacker_llm(attacker)
        assert attack._attacker_llm is attacker

    def test_missing_attacker_llm_raises(self):
        """Calling generate_question_for_turn without attacker LLM raises ValueError."""
        from guardbound.attacks.crescendo_paper import CrescendoAttackPaper
        attack = CrescendoAttackPaper()
        with pytest.raises(ValueError, match="Attacker LLM not set"):
            attack.generate_question_for_turn(goal="test", turn_num=1, last_response="")


# ============================================================================
# Test C — First Crescendo query
# ============================================================================

class TestFirstCrescendoQuery:
    """Verify that next_query() produces a non-empty query on the first turn."""

    def test_first_query_non_empty(self):
        """With empty history, next_query returns a generated question."""
        from guardbound.llm.mock import MockChatLLM
        from guardbound.attacks.crescendo_paper import CrescendoAttackPaper
        attacker = MockChatLLM(responses=['{"generatedQuestion": "What is the history?", "lastResponseSummary": ""}'])
        attack = CrescendoAttackPaper(attacker_llm=attacker)
        query = attack.next_query(goal="test goal", history=[])
        assert query == "What is the history?"
        assert query.strip()

    def test_first_query_calls_attacker_llm(self):
        """next_query triggers an attacker LLM call."""
        from guardbound.llm.mock import MockChatLLM
        from guardbound.attacks.crescendo_paper import CrescendoAttackPaper
        attacker = MockChatLLM(responses=['{"generatedQuestion": "q1", "lastResponseSummary": ""}'])
        attack = CrescendoAttackPaper(attacker_llm=attacker)
        _ = attack.next_query(goal="test", history=[])
        assert len(attacker.calls) == 1
        # Verify the call included the system prompt
        messages, temp, json_fmt = attacker.calls[0]
        system_contents = [m["content"] for m in messages if m["role"] == "system"]
        assert len(system_contents) == 1
        assert "conversationObjective" in system_contents[0]
        assert "test" in system_contents[0]

    def test_first_query_json_format_used(self):
        """next_query triggers attacker LLM with json_format=True."""
        from guardbound.llm.mock import MockChatLLM
        from guardbound.attacks.crescendo_paper import CrescendoAttackPaper
        attacker = MockChatLLM(responses=['{"generatedQuestion": "q1", "lastResponseSummary": ""}'])
        attack = CrescendoAttackPaper(attacker_llm=attacker)
        _ = attack.next_query(goal="test", history=[])
        messages, temp, json_fmt = attacker.calls[0]
        assert json_fmt is True

    def test_next_query_regenerates_each_call(self):
        """next_query generates a fresh query on every call (official behavior).

        The official Crescendo run loop invokes the attacker LLM once per round;
        there is no caching of previously generated questions.
        """
        from guardbound.llm.mock import MockChatLLM
        from guardbound.attacks.crescendo_paper import CrescendoAttackPaper
        attacker = MockChatLLM(
            responses=[
                '{"generatedQuestion": "q1", "lastResponseSummary": ""}',
                '{"generatedQuestion": "q2", "lastResponseSummary": ""}',
            ]
        )
        attack = CrescendoAttackPaper(attacker_llm=attacker)
        q1 = attack.next_query(goal="test", history=[])
        assert q1 == "q1"
        # Second call regenerates from the attacker rather than returning cached
        q2 = attack.next_query(goal="test", history=[])
        assert q2 == "q2"
        assert len(attacker.calls) == 2


# ============================================================================
# Test D — Multi-turn Crescendo
# ============================================================================

class TestMultiTurnCrescendo:
    """Verify multi-turn Crescendo with mocked attacker and target LLMs."""

    def test_two_turn_conversation(self):
        """Two turns produce two attacker calls with correct context."""
        from guardbound.llm.mock import MockChatLLM
        from guardbound.attacks.crescendo_paper import CrescendoAttackPaper
        from guardbound.schemas import Turn

        attacker_respuestas = [
            '{"generatedQuestion": "q1", "lastResponseSummary": "summary1"}',
            '{"generatedQuestion": "q2", "lastResponseSummary": "summary2"}',
        ]
        attacker = MockChatLLM(responses=attacker_respuestas)
        attack = CrescendoAttackPaper(attacker_llm=attacker, max_turns=2)

        # Turn 1
        q1 = attack.next_query(goal="my goal", history=[])
        assert q1 == "q1"

        # Record turn 1
        attack.record_turn(q1, "response1", 2)

        # Turn 2 — should call attacker with context from turn 1
        q2 = attack.next_query(goal="my goal", history=[
            Turn(query=q1, response="response1"),
        ])
        assert q2 == "q2"
        assert len(attacker.calls) == 2

        # Verify second call included turn 1 context. Official message
        # structure (crescendomation/run.py): the first-round boilerplate
        # prompt is ALWAYS present, plus one history pair per prior turn
        # (prior attacker query as an assistant message, prior target response
        # inside the "round number N" user prompt).
        messages2, _, _ = attacker.calls[1]
        user_contents = [m["content"] for m in messages2 if m["role"] == "user"]
        assistant_contents = [m["content"] for m in messages2 if m["role"] == "assistant"]
        assert len(user_contents) == 2
        assert any("first round" in c for c in user_contents)
        assert any("response1" in c for c in user_contents)
        assert "q1" in assistant_contents[0]

    def test_round_number_in_prompt(self):
        """Round number increases correctly."""
        from guardbound.llm.mock import MockChatLLM
        from guardbound.attacks.crescendo_paper import CrescendoAttackPaper
        from guardbound.schemas import Turn

        attacker = MockChatLLM(responses=[
            '{"generatedQuestion": "q1", "lastResponseSummary": ""}',
            '{"generatedQuestion": "q2", "lastResponseSummary": ""}',
        ])
        attack = CrescendoAttackPaper(attacker_llm=attacker)

        _ = attack.next_query(goal="g", history=[])
        attack.record_turn("q1", "r1", 3)
        _ = attack.next_query(goal="g", history=[Turn(query="q1", response="r1")])
        messages2, _, _ = attacker.calls[1]
        user_contents = [m["content"] for m in messages2 if m["role"] == "user"]
        # The second turn's user message should mention round 2
        assert any("round number 2" in c for c in user_contents)


# ============================================================================
# Test E — Refusal/backtracking
# ============================================================================

class TestRefusalBacktracking:
    """Verify Crescendo refusal detection and backtracking."""

    def test_check_refusal_regex_mode(self):
        """check_refusal with no attacker LLM uses regex patterns."""
        from guardbound.attacks.crescendo_paper import CrescendoAttackPaper
        attack = CrescendoAttackPaper()
        assert attack.check_refusal("any prompt", "I'm sorry, I cannot do that")
        assert not attack.check_refusal("any prompt", "Here is the information you requested")

    def test_check_refusal_llm_mode(self):
        """check_refusal with attacker LLM uses LLM-based detection."""
        from guardbound.llm.mock import MockChatLLM
        from guardbound.attacks.crescendo_paper import CrescendoAttackPaper
        # Mock returns a JSON indicating refusal with high confidence
        attacker = MockChatLLM(
            responses=['{"value": true, "metadata": 95, "rationale": "refusal"}']
        )
        attack = CrescendoAttackPaper(attacker_llm=attacker)
        assert attack.check_refusal("prompt", "I can't help with that")

    def test_refusal_detection_backtracks(self):
        """On refusal, should_backtrack returns True and increments counter."""
        from guardbound.llm.mock import MockChatLLM
        from guardbound.attacks.crescendo_paper import CrescendoAttackPaper
        attacker = MockChatLLM(responses=[
            '{"generatedQuestion": "q1", "lastResponseSummary": ""}',
        ])
        attack = CrescendoAttackPaper(attacker_llm=attacker, max_refusal_retries=3)
        assert attack.should_backtrack()
        attack.increment_refusal()
        assert attack.get_refusal_count() == 1
        assert attack.should_backtrack()
        # Exhaust retries: 3 refusals == max_refusal_retries
        attack.increment_refusal()
        attack.increment_refusal()
        assert attack.get_refusal_count() == 3
        assert not attack.should_backtrack()

    def test_record_turn_tracks_history(self):
        """record_turn populates internal history."""
        from guardbound.llm.mock import MockChatLLM
        from guardbound.attacks.crescendo_paper import CrescendoAttackPaper
        attacker = MockChatLLM(responses=['{"generatedQuestion": "q1", "lastResponseSummary": ""}'])
        attack = CrescendoAttackPaper(attacker_llm=attacker)
        q = attack.next_query(goal="g", history=[])
        attack.record_turn(q, "resp", 3)
        assert attack._history_attacker == [q]
        assert attack._history_target == ["resp"]
        assert attack._scores == [3]


# ============================================================================
# Test F — NBF parity
# ============================================================================

class TestNBFParity:
    """Compare original NBF scoring with Guardbound NBF scoring."""

    def test_nbf_score_formula_matches(self):
        """The NBF score formula is identical: p(unsafe) - max(p(safe))."""
        from guardbound.models.predictor import SafetyPredictor
        import torch.nn.functional as F

        predictor = SafetyPredictor()
        x = torch.randn(1, 768)
        u = torch.randn(1, 768)

        with torch.no_grad():
            logits = predictor(x, u)
            probs = F.softmax(logits, dim=-1)
            # Original formula: last_class_prob - max_other_class_prob
            orig_score = probs[:, -1] - torch.max(probs[:, :-1], dim=1).values
            # Guardbound predictor_value
            guard_score = predictor.predictor_value(x, u)

        assert torch.allclose(orig_score, guard_score)

    def test_nbf_score_shape(self):
        """NBF score is a scalar per sample."""
        from guardbound.models.predictor import SafetyPredictor
        predictor = SafetyPredictor()
        x = torch.randn(3, 768)
        u = torch.randn(3, 768)
        h = predictor.predictor_value(x, u)
        assert h.shape == (3,)

    def test_original_vs_guardbound_architecture_equivalence(self):
        """Verify that original NeuralBarrierFunction and SafetyPredictor
        have identical architectures by comparing layer shapes."""
        from guardbound.models.predictor import SafetyPredictor

        OriginalNBF = _load_original_train().NeuralBarrierFunction

        orig = OriginalNBF(state_dim=768, input_dim=768, hidden_dim=32, class_num=5)
        guard = SafetyPredictor(state_dim=768, embedding_dim=768)

        # Verify each layer matches (orig.nbf layers: [Linear, ReLU, Linear, ReLU, Linear];
        # guard.net layers: [Linear, ReLU, Linear, ReLU, Linear])
        for orig_layer, guard_layer in zip(orig.nbf, guard.net):
            assert type(orig_layer) == type(guard_layer), f"{type(orig_layer)} vs {type(guard_layer)}"
            if hasattr(orig_layer, "in_features"):
                assert orig_layer.in_features == guard_layer.in_features
            if hasattr(orig_layer, "out_features"):
                assert orig_layer.out_features == guard_layer.out_features


# ============================================================================
# Test G — Checkpoint conversion
# ============================================================================

class TestCheckpointCompat(TestNBFParity):
    """Test the original checkpoint compatibility loader."""

    def test_remap_nbf_keys(self):
        """Original nbf.X keys get remapped to net.X keys."""
        from guardbound.models.compat import _remap_original_nbf_keys

        original = {
            "nbf.0.weight": torch.randn(32, 1536),
            "nbf.0.bias": torch.randn(32),
            "nbf.2.weight": torch.randn(32, 32),
            "nbf.2.bias": torch.randn(32),
            "nbf.4.weight": torch.randn(5, 32),
            "nbf.4.bias": torch.randn(5),
        }
        remapped = _remap_original_nbf_keys(original)
        assert "net.0.weight" in remapped
        assert "nbf.0.weight" not in remapped
        assert remapped["net.0.weight"].shape == (32, 1536)
        assert remapped["net.4.weight"].shape == (5, 32)

    def test_remap_ssm_keys(self):
        """Original state_transition.X and observation_model.X get remapped."""
        from guardbound.models.compat import _remap_original_ssm_keys

        original = {
            "state_transition.0.weight": torch.randn(512, 1536),
            "state_transition.0.bias": torch.randn(512),
            "state_transition.2.weight": torch.randn(512, 512),
            "state_transition.2.bias": torch.randn(512),
            "state_transition.4.weight": torch.randn(768, 512),
            "state_transition.4.bias": torch.randn(768),
            "observation_model.0.weight": torch.randn(512, 1536),
            "observation_model.0.bias": torch.randn(512),
            "observation_model.2.weight": torch.randn(512, 512),
            "observation_model.2.bias": torch.randn(512),
            "observation_model.4.weight": torch.randn(768, 512),
            "observation_model.4.bias": torch.randn(768),
        }
        remapped = _remap_original_ssm_keys(original)
        assert "f_theta.net.0.weight" in remapped
        assert "g_theta.net.0.weight" in remapped
        assert remapped["f_theta.net.0.weight"].shape == (512, 1536)
        assert remapped["g_theta.net.0.weight"].shape == (512, 1536)
        assert remapped["g_theta.net.4.weight"].shape == (768, 512)
        assert remapped["g_theta.net.4.bias"].shape == (768,)

    def test_remap_preserves_other_keys(self):
        """Keys that don't match any prefix are preserved as-is."""
        from guardbound.models.compat import _remap_original_nbf_keys
        original = {"nbf.0.weight": torch.randn(32, 1536), "extra.key": torch.randn(3, 3)}
        remapped = _remap_original_nbf_keys(original)
        assert "extra.key" in remapped
        assert remapped["extra.key"].shape == (3, 3)

    def test_remap_original_state_dict(self):
        """Full checkpoint remapping produces expected structure."""
        from guardbound.models.compat import remap_original_state_dict

        original_ckpt = {
            "ssm": {
                "state_transition.0.weight": torch.randn(512, 1536),
                "state_transition.0.bias": torch.randn(512),
                "state_transition.2.weight": torch.randn(512, 512),
                "state_transition.2.bias": torch.randn(512),
                "state_transition.4.weight": torch.randn(768, 512),
                "state_transition.4.bias": torch.randn(768),
                "observation_model.0.weight": torch.randn(512, 1536),
                "observation_model.0.bias": torch.randn(512),
                "observation_model.2.weight": torch.randn(512, 512),
                "observation_model.2.bias": torch.randn(512),
                "observation_model.4.weight": torch.randn(768, 512),
                "observation_model.4.bias": torch.randn(768),
            },
            "nbf": {
                "nbf.0.weight": torch.randn(32, 1536),
                "nbf.0.bias": torch.randn(32),
                "nbf.2.weight": torch.randn(32, 32),
                "nbf.2.bias": torch.randn(32),
                "nbf.4.weight": torch.randn(5, 32),
                "nbf.4.bias": torch.randn(5),
            },
        }
        result = remap_original_state_dict(original_ckpt)
        assert "dynamics_state_dict" in result
        assert "predictor_state_dict" in result
        assert "f_theta.net.0.weight" in result["dynamics_state_dict"]
        assert "g_theta.net.0.weight" in result["dynamics_state_dict"]
        assert "net.0.weight" in result["predictor_state_dict"]
        # Verify predictor state dict values are preserved
        assert torch.allclose(result["predictor_state_dict"]["net.0.weight"], original_ckpt["nbf"]["nbf.0.weight"])
        # Verify SSM state dict values are preserved
        assert torch.allclose(result["dynamics_state_dict"]["f_theta.net.0.weight"], original_ckpt["ssm"]["state_transition.0.weight"])

    def test_nbf_architecture_matching(self):
        """Original NeuralBarrierFunction and Guardbound SafetyPredictor
        have identical architectures (same layer sizes)."""
        from guardbound.models.predictor import SafetyPredictor

        OriginalNBF = _load_original_train().NeuralBarrierFunction

        orig = OriginalNBF(state_dim=768, input_dim=768, hidden_dim=32, class_num=5)
        guard = SafetyPredictor(state_dim=768, embedding_dim=768)

        # Verify each layer matches (orig.nbf layers: [Linear, ReLU, Linear, ReLU, Linear];
        # guard.net layers: [Linear, ReLU, Linear, ReLU, Linear])
        for orig_layer, guard_layer in zip(orig.nbf, guard.net):
            assert type(orig_layer) == type(guard_layer), f"{type(orig_layer)} vs {type(guard_layer)}"
            if hasattr(orig_layer, "in_features"):
                assert orig_layer.in_features == guard_layer.in_features
            if hasattr(orig_layer, "out_features"):
                assert orig_layer.out_features == guard_layer.out_features

    def test_ssm_architecture_matching(self):
        """Original NeuralStateSpaceModel state_transition matches Guardbound f_theta."""
        from guardbound.models.dynamics import MLPDynamics

        NeuralStateSpaceModel = _load_original_train().NeuralStateSpaceModel

        orig_ssm = NeuralStateSpaceModel(768, 768, 768, 512)
        guard_f = MLPDynamics(input_dim=1536, hidden_dims=[512, 512], output_dim=768)

        for orig_layer, guard_layer in zip(orig_ssm.state_transition, guard_f.net):
            assert type(orig_layer) == type(guard_layer), f"{type(orig_layer)} vs {type(guard_layer)}"
            if hasattr(orig_layer, "in_features"):
                assert orig_layer.in_features == guard_layer.in_features
            if hasattr(orig_layer, "out_features"):
                assert orig_layer.out_features == guard_layer.out_features
