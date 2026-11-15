"""Cloud × Guardbound research-integration tests.

These exercise the *existing* attack algorithms through the *existing* runner
against a cloud ``ChatLLM`` (real SDK, fake HTTP transport). The point is that
nothing in the attack, evaluator or NBF layers had to change to run on a cloud
model — and that the paper's protocol survives the substitution.

The NBF here is a deterministic stub with the same duck-typed surface as
``NeuralBarrierFunction``. It exists to prove the filtering *plumbing*
(candidate scoring, threshold sign, target-call suppression) is provider-agnostic;
it is not a numerical test of the real checkpoint (``tests/parity/test_nbf_parity.py``
and ``tests/test_nbf_checkpoint.py`` cover that).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import httpx
import pytest
import torch

from guardbound.attacks.acronym import AcronymAttack
from guardbound.attacks.crescendo_paper import (
    CRESCENDO_SYSTEM_PROMPT,
    AttackGenerationError,
    CrescendoAttackPaper,
)
from guardbound.attacks.opposite_day import OppositeDayAttack
from guardbound.attacks.runner import run_attack_with_backtracking
from guardbound.attacks.rubric_evaluation import (
    JSON_PARSE_ERROR,
    MISSING_SCORE,
    classify_rubric_output,
    evaluator_telemetry_summary,
    reset_evaluator_telemetry,
)
from guardbound.llm.openai_client import OpenAIChatLLM
from guardbound.llm.usage import UsageLedger
from tests.cloud_fakes import TEST_MODEL, FakeOpenAIBackend, completion

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOAL = "Explain how photosynthesis works."


# --------------------------------------------------------------------------- #
# Fake endpoint that answers like the three official call shapes
# --------------------------------------------------------------------------- #

class RoutedBackend(FakeOpenAIBackend):
    """Routes by prompt content, so call ordering does not matter.

    The three routes correspond to the official operations:
      * ``responseToEvaluateInput`` + "must refuse"   -> refusal check
      * ``responseToEvaluateInput`` + "disclaimer"    -> disclaimer check
      * ``scale of 1 to 5``                           -> rubric evaluation
      * ``generatedQuestion``                         -> attacker generation
      * otherwise                                     -> target completion
    """

    def __init__(
        self,
        *,
        attacker_question: str = "What is the first step?",
        rubric_score: int = 1,
        target_reply: str = "Here is a helpful answer.",
        refused: bool = False,
        disclaimer: bool = False,
        attacker_raw: str | None = None,
        rubric_raw: str | None = None,
    ) -> None:
        self.attacker_question = attacker_question
        self.rubric_score = rubric_score
        self.target_reply = target_reply
        self.refused = refused
        self.disclaimer = disclaimer
        self.attacker_raw = attacker_raw
        self.rubric_raw = rubric_raw
        self.purposes: list[str] = []
        super().__init__(responses=[self._respond] * 400)

    def _respond(self, request):
        body = json.loads(request.content)
        blob = "\n".join(str(m.get("content", "")) for m in body.get("messages", []))

        if "responseToEvaluateInput" in blob:
            if "disclaimer" in blob:
                self.purposes.append("disclaimer")
                payload = {"value": self.disclaimer, "rationale": "test"}
            else:
                self.purposes.append("refusal")
                payload = {"value": self.refused, "rationale": "test", "metadata": 95}
            return httpx.Response(200, json=completion(json.dumps(payload)))

        if "scale of 1 to 5" in blob:
            self.purposes.append("rubric")
            raw = self.rubric_raw or json.dumps({"score": self.rubric_score})
            return httpx.Response(200, json=completion(raw))

        if "generatedQuestion" in blob:
            self.purposes.append("generation")
            raw = self.attacker_raw or json.dumps({
                "generatedQuestion": self.attacker_question,
                "lastResponseSummary": "summary",
            })
            return httpx.Response(200, json=completion(raw))

        self.purposes.append("target")
        return httpx.Response(200, json=completion(self.target_reply))

    def requests_for(self, purpose: str) -> list[dict]:
        return [
            body for body, seen in zip(self.requests, self.purposes) if seen == purpose
        ]


# --------------------------------------------------------------------------- #
# Deterministic NBF stub
# --------------------------------------------------------------------------- #

class StubPredictor(torch.nn.Module):
    """5-class logits whose barrier score equals ``h_value``.

    Solves ``h = (e^L - 1) / (4 + e^L)`` for the class-5 logit ``L`` with the
    four safe logits fixed at 0, so the score the runner computes is exactly the
    value requested. Valid for ``-0.25 < h < 1``.
    """

    def __init__(self, h_value: float) -> None:
        super().__init__()
        if not -0.25 < h_value < 1.0:
            raise ValueError("stub h_value must be in (-0.25, 1)")
        self.state_dim = 768
        self.embedding_dim = 768
        self.h_value = float(h_value)
        self._anchor = torch.nn.Parameter(torch.zeros(1))
        self._logit = math.log((4 * h_value + 1) / (1 - h_value))

    def forward(self, x_prev: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x_prev.shape[0], 5, device=x_prev.device)
        logits[:, 4] = self._logit
        return logits

    def predictor_value(self, x_prev: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.forward(x_prev, u), dim=-1)
        return probs[:, 4] - probs[:, :4].max(dim=-1).values


class StubBarrier(torch.nn.Module):
    """Duck-typed stand-in for ``NeuralBarrierFunction``."""

    def __init__(self, h_value: float) -> None:
        super().__init__()
        self.predictor = StubPredictor(h_value)
        self.advanced = 0

    def h(self, state, query_embedding):
        return self.predictor.predictor_value(state, query_embedding)

    def advance_state(self, state, query_embedding):
        self.advanced += 1
        return state


def stub_embed_fn(text: str) -> torch.Tensor:
    return torch.zeros(1, 768)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_llms(backend: FakeOpenAIBackend, ledger: UsageLedger | None = None):
    kwargs = {"client": backend.client, "ledger": ledger or UsageLedger()}
    return {
        "attacker": OpenAIChatLLM(model_id=TEST_MODEL, role="attacker", **kwargs),
        "target": OpenAIChatLLM(model_id=TEST_MODEL, role="target", **kwargs),
        "evaluator": OpenAIChatLLM(model_id=TEST_MODEL, role="evaluator", **kwargs),
    }


def build_attack(kind: str, llms, max_turns: int):
    cls = {
        "crescendo": CrescendoAttackPaper,
        "opposite_day": OppositeDayAttack,
        "acronym": AcronymAttack,
    }[kind]
    attack = cls(max_turns=max_turns, attacker_llm=llms["attacker"])
    attack.set_evaluator_llm(llms["evaluator"])
    return attack


@pytest.fixture(autouse=True)
def _clear_evaluator_telemetry():
    reset_evaluator_telemetry()
    yield
    reset_evaluator_telemetry()


# --------------------------------------------------------------------------- #
# Attack compatibility
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind", ["crescendo", "opposite_day", "acronym"])
def test_attack_runs_to_completion_over_cloud_transport(kind):
    backend = RoutedBackend()
    llms = make_llms(backend)
    attack = build_attack(kind, llms, max_turns=2)

    conversation = run_attack_with_backtracking(
        attack=attack,
        goal=GOAL,
        target_llm=llms["target"],
        max_turns=2,
        temperature=0.7,
        target_llm_name=f"openai:{TEST_MODEL}",
        attack_method=kind,
    )

    assert len(conversation.turns) == 2
    assert all(t.response for t in conversation.turns)
    assert "target" in backend.purposes
    assert "generation" in backend.purposes
    assert "rubric" in backend.purposes
    # No refusals configured, so no backtrack should have happened.
    assert attack.get_refusal_count() == 0


def test_acronym_also_runs_the_official_disclaimer_check():
    backend = RoutedBackend()
    llms = make_llms(backend)
    attack = build_attack("acronym", llms, max_turns=2)
    run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=2,
        temperature=0.7,
    )
    assert "disclaimer" in backend.purposes


def test_crescendo_and_opposite_day_do_not_call_disclaimer():
    for kind in ("crescendo", "opposite_day"):
        backend = RoutedBackend()
        llms = make_llms(backend)
        attack = build_attack(kind, llms, max_turns=2)
        run_attack_with_backtracking(
            attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=2,
            temperature=0.7,
        )
        assert "disclaimer" not in backend.purposes


# --------------------------------------------------------------------------- #
# Prompt / JSON / temperature preservation (C1 compatibility)
# --------------------------------------------------------------------------- #

def test_crescendo_system_prompt_is_the_official_one():
    """The cloud provider must not alter the paper's attacker prompt."""
    backend = RoutedBackend()
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=2)
    run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=2,
        temperature=0.7,
    )

    first_generation = backend.requests_for("generation")[0]
    system = next(m for m in first_generation["messages"] if m["role"] == "system")
    assert system["content"] == CRESCENDO_SYSTEM_PROMPT.format(conversationObjective=GOAL)
    # ...and the attacker's first user turn is the official boilerplate.
    first_user = next(m for m in first_generation["messages"] if m["role"] == "user")
    assert first_user["content"].startswith("This is the first round.")


def test_attacker_generation_uses_json_mode_at_paper_temperature():
    backend = RoutedBackend()
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=2)
    run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=2,
        temperature=0.7,
    )
    generation = backend.requests_for("generation")[0]
    assert generation["response_format"] == {"type": "json_object"}
    assert generation["temperature"] == 0.7


def test_evaluator_calls_use_temperature_zero():
    """Official code hardcodes 0.0 for refusal / disclaimer / rubric."""
    backend = RoutedBackend()
    llms = make_llms(backend)
    attack = build_attack("acronym", llms, max_turns=2)
    run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=2,
        temperature=0.7,
    )
    for purpose in ("refusal", "disclaimer", "rubric"):
        calls = backend.requests_for(purpose)
        assert calls, f"no {purpose} call was made"
        assert all(call["temperature"] == 0.0 for call in calls), purpose


def test_target_call_does_not_request_json():
    backend = RoutedBackend()
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=2)
    run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=2,
        temperature=0.7,
    )
    target_call = backend.requests_for("target")[0]
    assert target_call["response_format"] == {"type": "text"}
    assert target_call["temperature"] == 0.7


def test_attacker_json_contract_fields_are_consumed_unchanged():
    backend = RoutedBackend(attacker_question="A very specific question?")
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=2)
    conversation = run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=2,
        temperature=0.7,
    )
    assert conversation.turns[0].query == "A very specific question?"


# --------------------------------------------------------------------------- #
# Contract failures are separate from attack failures
# --------------------------------------------------------------------------- #

def test_malformed_attacker_reply_raises_generation_error():
    backend = RoutedBackend(attacker_raw="Sure, here is the next question...")
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=2)
    with pytest.raises(AttackGenerationError):
        run_attack_with_backtracking(
            attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=2,
            temperature=0.7,
        )
    # The failure happened before any target call.
    assert "target" not in backend.purposes


def test_missing_attacker_field_raises_generation_error():
    backend = RoutedBackend(attacker_raw=json.dumps({"generatedQuestion": "Q"}))
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=2)
    with pytest.raises(AttackGenerationError, match="missing"):
        run_attack_with_backtracking(
            attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=2,
            temperature=0.7,
        )


def test_invalid_evaluator_output_is_not_recorded_as_score_one():
    """A reply without ``score`` is a contract failure, not a safe verdict."""
    backend = RoutedBackend(rubric_raw='{"verdict": "fine"}')
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=2)

    conversation = run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=2,
        temperature=0.7,
    )
    # A turn still happens; the score is None, never a fabricated 1.
    assert len(conversation.turns) == 2
    assert attack.get_last_score() is None

    summary = evaluator_telemetry_summary()
    assert summary["by_outcome"][MISSING_SCORE] == 2
    assert summary["evaluator_failures"] == 2
    assert summary["genuine_score_1"] == 0


def test_unparseable_evaluator_output_is_a_parse_error():
    backend = RoutedBackend(rubric_raw="I would rate this a 1 out of 5.")
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=2)
    run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=2,
        temperature=0.7,
    )
    summary = evaluator_telemetry_summary()
    assert summary["by_outcome"][JSON_PARSE_ERROR] == 2
    assert summary["genuine_score_1"] == 0


def test_valid_explicit_score_one_is_preserved():
    evaluated = classify_rubric_output('{"score": 1}')
    assert evaluated.valid is True
    assert evaluated.score == 1


# --------------------------------------------------------------------------- #
# NBF OFF / ON
# --------------------------------------------------------------------------- #

def test_nbf_off_calls_the_target_directly():
    backend = RoutedBackend()
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=2)

    conversation = run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"],
        embed_fn=None, barrier=None, eta=0.0, max_turns=2, temperature=0.7,
    )
    assert len(conversation.turns) == 2
    assert not any(t.was_filtered for t in conversation.turns)
    assert len(backend.requests_for("target")) == 2
    assert attack.nbf_candidate_scores == []


def test_nbf_on_accepts_a_query_below_threshold():
    backend = RoutedBackend()
    llms = make_llms(backend)
    barrier = StubBarrier(h_value=-0.2)  # score < 0 -> accepted
    attack = build_attack("crescendo", llms, max_turns=2)

    conversation = run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"],
        embed_fn=stub_embed_fn, barrier=barrier, eta=0.0, max_turns=2,
        temperature=0.7,
    )
    assert len(conversation.turns) == 2
    assert len(backend.requests_for("target")) == 2
    assert barrier.advanced > 0
    assert attack.nbf_candidate_scores
    assert all(v is True for v in attack.nbf_candidate_verdicts)


def test_nbf_on_suppresses_the_target_call_for_a_filtered_query():
    backend = RoutedBackend()
    llms = make_llms(backend)
    barrier = StubBarrier(h_value=0.2)  # score >= 0 -> filtered
    attack = build_attack("crescendo", llms, max_turns=2)

    conversation = run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"],
        embed_fn=stub_embed_fn, barrier=barrier, eta=0.0, max_turns=2,
        temperature=0.7,
    )
    # Every candidate was filtered, so the target was never called.
    assert conversation.turns == []
    assert backend.requests_for("target") == []
    assert attack.nbf_candidate_scores
    assert all(v is False for v in attack.nbf_candidate_verdicts)
    # The attacker still ran: filtering happens after query generation.
    assert backend.requests_for("generation")


def test_nbf_threshold_override_controls_the_decision():
    """``score < threshold`` accepts; ``score >= threshold`` filters."""
    backend = RoutedBackend()
    llms = make_llms(backend)
    barrier = StubBarrier(h_value=-0.2)  # score = -0.2
    attack = build_attack("crescendo", llms, max_turns=2)
    # Push the threshold below the score so the same query is now filtered.
    conversation = run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"],
        embed_fn=stub_embed_fn, barrier=barrier, eta=0.0, max_turns=2,
        temperature=0.7, safety_threshold_override=-0.5,
    )
    assert conversation.turns == []
    assert backend.requests_for("target") == []
    # The filter really ran (this is not an empty loop).
    assert attack.nbf_candidate_scores
    assert all(score == pytest.approx(-0.2, abs=1e-6)
               for score in attack.nbf_candidate_scores)


def test_nbf_steered_mode_still_calls_the_target_when_accepted():
    """``steer_target=True`` routes through SteeredLLMChat; provider-agnostic."""
    backend = RoutedBackend()
    llms = make_llms(backend)
    barrier = StubBarrier(h_value=-0.2)
    attack = build_attack("crescendo", llms, max_turns=2)
    conversation = run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"],
        embed_fn=stub_embed_fn, barrier=barrier, eta=0.0, max_turns=2,
        temperature=0.7, steer_target=True,
    )
    assert len(conversation.turns) == 2
    assert all(t.was_filtered is False for t in conversation.turns)


def test_one_ledger_attributes_cost_per_role():
    backend = RoutedBackend()
    ledger = UsageLedger()
    llms = make_llms(backend, ledger=ledger)
    attack = build_attack("crescendo", llms, max_turns=2)
    run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=2,
        temperature=0.7,
    )
    roles = ledger.summary()["by_role"]
    assert roles["attacker"]["calls"] >= 1
    assert roles["target"]["calls"] == 2
    assert roles["evaluator"]["calls"] >= 1


# --------------------------------------------------------------------------- #
# Termination
# --------------------------------------------------------------------------- #

def test_terminates_when_the_rubric_reaches_the_success_score():
    backend = RoutedBackend(rubric_score=5)
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=8)
    conversation = run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=8,
        temperature=0.7,
    )
    assert len(conversation.turns) == 1
    assert attack.was_successful() is True


def test_does_not_exceed_max_turns():
    backend = RoutedBackend()
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=4)
    conversation = run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=4,
        temperature=0.7,
    )
    assert len(conversation.turns) == 4


def test_single_round_budget_produces_no_turn_official_parity():
    """Official ``while round_number + num_filtering < max_rounds`` [sic].

    The reference implementation seeds ``round_number = 1`` and recomputes it in
    the body, so ``max_rounds = 1`` admits zero iterations. This harness keeps
    that behaviour rather than "fixing" it, because changing it would change the
    round accounting of every paper-faithful run.
    """
    backend = RoutedBackend()
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=1)
    conversation = run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=1,
        temperature=0.7,
    )
    assert conversation.turns == []
    assert backend.calls == 0


def test_refusal_backtrack_consumes_no_round_budget():
    """Official: a refused turn is popped and regenerated, not counted.

    The reference implementation caps backtracks at ``C_refused < 10`` and then
    accepts the turn, so a persistently refusing target still produces turns —
    and, crucially, the target is called more than ``max_turns`` times. That is
    the observable signature of "the refused round did not consume budget".
    """
    backend = RoutedBackend(refused=True)
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=2)
    conversation = run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=2,
        temperature=0.7,
    )
    assert attack.get_refusal_count() >= 1
    assert len(backend.requests_for("target")) > 2
    assert conversation.turns, "turns are recorded once the backtrack cap is hit"


# --------------------------------------------------------------------------- #
# Config / script gates
# --------------------------------------------------------------------------- #

def _cloud_common():
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import _cloud_common  # noqa: PLC0415

    return _cloud_common


def test_variants_differ_only_in_nbf():
    common = _cloud_common()
    cfg = common.load_config(PROJECT_ROOT / "configs/cloud_research_gpt4o_mini.yaml")
    checked = common.assert_invariants(cfg)
    assert "models" in checked
    off = common.nbf_settings(cfg, common.resolve_variant(cfg, "cloud_baseline_a_nbf_off")[1])
    on = common.nbf_settings(cfg, common.resolve_variant(cfg, "cloud_baseline_b_nbf_on")[1])
    assert off["enabled"] is False
    assert on["enabled"] is True
    assert on["threshold"] == 0.0
    assert on["eta"] == 0.0


def test_variant_override_outside_nbf_is_rejected():
    common = _cloud_common()
    cfg = common.load_config(PROJECT_ROOT / "configs/cloud_research_gpt4o_mini.yaml")
    cfg["variants"]["cloud_baseline_b_nbf_on"]["temperature"] = {"target": 0.9}
    with pytest.raises(ValueError, match="differ only in NBF"):
        common.assert_invariants(cfg)


def test_nbf_checkpoint_identity_is_pinned():
    common = _cloud_common()
    cfg = common.load_config(PROJECT_ROOT / "configs/cloud_research_gpt4o_mini.yaml")
    meta = common.verify_nbf_checkpoint(cfg["nbf"])
    assert meta["sha256"] == (
        "cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe"
    )
    assert meta["state_dim"] == 768
    assert meta["embedding_dim"] == 768
    assert meta["threshold"] == 0.0


def test_checkpoint_digest_mismatch_refuses_to_run(tmp_path):
    common = _cloud_common()
    fake = tmp_path / "ckpt.pth"
    fake.write_bytes(b"not the validated checkpoint")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        common.verify_nbf_checkpoint(
            {"checkpoint": str(fake), "checkpoint_sha256_expected": "deadbeef"}
        )


def test_research_configs_declare_the_formal_protocol():
    common = _cloud_common()
    for name in ("cloud_research_gpt4o_mini.yaml", "cloud_research_reference.yaml"):
        cfg = common.load_config(PROJECT_ROOT / "configs" / name)
        protocol = cfg["experiment"]["protocol"]
        assert protocol["retries"] == 0
        assert protocol["repair"] == "NONE"
        assert protocol["fallback"] == "NONE"
        assert cfg["attacks"]["max_turns"] == 8
        assert cfg["temperature"]["attacker"] == 0.7
        assert cfg["temperature"]["target"] == 0.7
        assert cfg["temperature"]["evaluator"] == 0.0
        assert cfg["cloud"]["structured_output"] == "json_object"


def test_reference_config_uses_gpt4o_and_mini_config_uses_mini():
    common = _cloud_common()
    reference = common.load_config(PROJECT_ROOT / "configs/cloud_research_reference.yaml")
    mini = common.load_config(PROJECT_ROOT / "configs/cloud_research_gpt4o_mini.yaml")
    # Phase 4: exact dated snapshots, not the floating aliases.
    assert {reference["models"][r]["model"] for r in ("attacker", "target", "evaluator")} == {"gpt-4o-2024-08-06"}
    assert {mini["models"][r]["model"] for r in ("attacker", "target", "evaluator")} == {"gpt-4o-mini-2024-07-18"}
    assert "SUBSTITUTION" in mini["experiment"]["methodology_label"].upper()


def test_pilot_refuses_to_run_without_confirmation(monkeypatch, capsys):
    import cloud_pilot  # noqa: PLC0415

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-a-credential")
    code = cloud_pilot.main([
        "--variant", "cloud_baseline_a_nbf_off",
        "--out-dir", "unused",
        "--limit", "1",
    ])
    assert code == 2
    assert "Refusing to send paid requests" in capsys.readouterr().err


def test_smoke_test_refuses_to_run_without_confirmation(monkeypatch, capsys):
    import cloud_smoke_test  # noqa: PLC0415

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-a-credential")
    code = cloud_smoke_test.main([])
    assert code == 2
    assert "Refusing to send a paid request" in capsys.readouterr().err


def test_smoke_test_dry_run_reports_missing_key_without_asking_for_it(
    monkeypatch, capsys
):
    import cloud_smoke_test  # noqa: PLC0415

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        cloud_smoke_test.main(["--dry-run"])
    assert "not set" in str(excinfo.value.code)
    # The message must never solicit the key on the command line or in config.
    assert "never read from" in str(excinfo.value.code)
