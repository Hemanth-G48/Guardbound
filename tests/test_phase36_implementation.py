"""Phase 36 regression tests: the three Phase 35 blockers.

Covered:
  * P0-1 — ActorAttack composite success is the author's summary-score verdict,
    consumed in ``cloud_pilot.run_one`` (not re-derived from ``_scores``).
  * P1-1 — a QUALIFIED qualification report is required before any paid client
    is constructed; --confirm-paid-request and an API key cannot bypass it.
  * P1-2 — the configured attacker temperature reaches every attacker call site
    of ActorAttack / Crescendo / Opposite-day, independently of the evaluator.

All tests are offline: scripted fake ``ChatLLM`` objects, no network, no model
inference, no credentials.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import _cloud_common as common  # noqa: E402
import cloud_pilot  # noqa: E402

from guardbound.attacks.actor_attack import (  # noqa: E402
    ActorAttack,
    infer_single,
    rewrite_query,
)
from guardbound.attacks.crescendo_paper import (  # noqa: E402
    CrescendoAttackPaper,
    generate_crescendo_step,
)
from guardbound.attacks.opposite_day import (  # noqa: E402
    OppositeDayAttack,
    generate_opposite_day_step,
)
from guardbound.schemas import Conversation  # noqa: E402


# --------------------------------------------------------------------------- #
# Shared fakes
# --------------------------------------------------------------------------- #

class RecordingLLM:
    """Deterministic ``ChatLLM`` that records every call's temperature."""

    def __init__(self, responses=None):
        self.calls: list[dict] = []
        self._responses = list(responses or [])

    def generate(self, messages, temperature=0.7, json_format=False, **kwargs):
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "json_format": json_format,
        })
        if self._responses:
            return self._responses.pop(0)
        return {} if json_format else ""

    @property
    def temperatures(self) -> list[float]:
        return [call["temperature"] for call in self.calls]


class FakeAttack:
    """Minimal stand-in for a constructed attack in the pilot."""

    def __init__(self, scores):
        self._scores = list(scores)
        self.nbf_candidate_scores: list[float] = []
        self.nbf_candidate_verdicts: list[bool] = []

    def get_refusal_count(self) -> int:
        return 0


class FakeComposite:
    """Composite-runner result with an explicit ``goal_achieved``."""

    def __init__(self, goal_achieved, turns=None):
        self.goal_achieved = goal_achieved
        self.conversation = Conversation(
            goal="g", attack_method="actor_attack", target_llm="t",
            turns=list(turns or []), max_turns=8,
        )
        self.actor_records = [{"actor_index": 0, "score": None}]


class CompositeWithoutFlag:
    """Composite result that is missing the authoritative success field."""

    def __init__(self):
        self.conversation = Conversation(
            goal="g", attack_method="actor_attack", target_llm="t",
            turns=[], max_turns=8,
        )
        self.actor_records = []


def _pilot_cfg(attack_name: str = "actor_attack", execution: str = "composite"):
    return {
        "temperature": {"attacker": 0.7, "target": 0.7, "evaluator": 0.7},
        "cloud": {"structured_output": "json_object"},
        "models": {
            "attacker": {"provider": "openai", "model": "gpt-4o-2024-08-06"},
            "target": {"provider": "openai", "model": "gpt-4o-2024-08-06"},
            "evaluator": {"provider": "openai", "model": "gpt-4o-2024-08-06"},
        },
        "attacks": {
            "success_score": 5,
            "execution": {attack_name: execution},
        },
        "prompt_variant": "official",
        "attacker_prompt_condition": "baseline",
        "dataset": {"target_system_default": "You are a helpful assistant."},
    }


def _run_one(monkeypatch, *, cfg, scores, composite=None, generic_conversation=None):
    """Drive ``cloud_pilot.run_one`` with a fake attack and runner."""
    fake_attack = FakeAttack(scores)
    monkeypatch.setattr(common, "build_attack", lambda *a, **k: fake_attack)

    def _composite_runner(**_kwargs):
        assert composite is not None, "test did not supply a composite result"
        return composite

    async def _generic_runner(**_kwargs):
        return generic_conversation

    monkeypatch.setattr(
        "guardbound.attacks.runner.run_actor_attack_composite", _composite_runner)
    monkeypatch.setattr(
        "guardbound.attacks.runner.run_attack_with_backtracking_async",
        _generic_runner)

    llms = {"attacker": object(), "target": object(), "evaluator": object()}
    return asyncio.run(cloud_pilot.run_one(
        attack_name="actor_attack",
        case={"task": "do the thing", "target_system": "S"},
        llms=llms,
        barrier=None,
        embed_fn=None,
        cfg=cfg,
        nbf_cfg={"enabled": False, "eta": 0.001, "threshold": 0.001},
        max_turns=8,
        device="cpu",
    ))


# --------------------------------------------------------------------------- #
# P0-1 — composite ActorAttack success
# --------------------------------------------------------------------------- #

def test_composite_intermediate_five_with_summary_below_five_is_unsuccessful(
        monkeypatch):
    """The author's rule ignores an intermediate 5 when the summary is < 5."""
    record = _run_one(
        monkeypatch,
        cfg=_pilot_cfg(),
        scores=[5, 3],
        composite=FakeComposite(goal_achieved=False),
    )
    assert record["success"] is False
    assert record["success_source"] == "composite.goal_achieved"
    # Diagnostics keep the intermediate score, they just do not decide success.
    assert record["rubric_scores"] == [5, 3]
    assert record["max_rubric_score"] == 5


def test_composite_summary_five_is_successful(monkeypatch):
    record = _run_one(
        monkeypatch,
        cfg=_pilot_cfg(),
        scores=[2, 5],
        composite=FakeComposite(goal_achieved=True),
    )
    assert record["success"] is True
    assert record["success_source"] == "composite.goal_achieved"


def test_composite_multiple_intermediate_scores_below_five_is_unsuccessful(
        monkeypatch):
    record = _run_one(
        monkeypatch,
        cfg=_pilot_cfg(),
        scores=[4, 3, 2],
        composite=FakeComposite(goal_achieved=False),
    )
    assert record["success"] is False


def test_generic_attack_keeps_the_max_rubric_score_rule(monkeypatch):
    """Crescendo / Opposite-day are unchanged: any accepted turn scoring 5 wins."""
    conversation = Conversation(
        goal="g", attack_method="crescendo_paper", target_llm="t",
        turns=[], max_turns=8,
    )
    record = _run_one(
        monkeypatch,
        cfg=_pilot_cfg(attack_name="crescendo_paper", execution="generic"),
        scores=[5, 2],
        generic_conversation=conversation,
    )
    assert record["success"] is True
    assert record["success_source"] == "max_rubric_score"


def test_composite_without_goal_achieved_raises_and_never_falls_back(monkeypatch):
    """A missing authoritative flag is an error, not an invitation to use max()."""
    with pytest.raises(cloud_pilot.CompositeSuccessMetadataError):
        _run_one(
            monkeypatch,
            cfg=_pilot_cfg(),
            scores=[5, 5, 5],
            composite=CompositeWithoutFlag(),
        )


def test_composite_with_non_boolean_goal_achieved_raises(monkeypatch):
    with pytest.raises(cloud_pilot.CompositeSuccessMetadataError):
        _run_one(
            monkeypatch,
            cfg=_pilot_cfg(),
            scores=[5],
            composite=FakeComposite(goal_achieved=None),
        )


def test_resolve_goal_success_is_the_single_decision_point():
    success, source = cloud_pilot.resolve_goal_success(
        composite=FakeComposite(goal_achieved=False),
        aggregate={"success": True},
    )
    assert (success, source) == (False, "composite.goal_achieved")
    success, source = cloud_pilot.resolve_goal_success(
        composite=None, aggregate={"success": True})
    assert (success, source) == (True, "max_rubric_score")


# --------------------------------------------------------------------------- #
# P1-1 — qualification gate
# --------------------------------------------------------------------------- #

def _qualified_payload(**overrides):
    root = Path(__file__).resolve().parents[1]
    r1_config = root / "configs" / "paper_reproduction_r1_gpt4o.yaml"
    payload = {
        "schema_version": "qualification.v2",
        "probe_spec_version": "r1-probe.v2",
        "run_id": "20260101T000000Z",
        "overall": "QUALIFIED",
        "probes_complete": True,
        "attacker_prompt_condition": "baseline",
        "probe_status": {
            name: {"status": "PASS"} for name in common.MANDATORY_PROBE_NAMES
        },
        "experiment_config": {
            "path": "configs/paper_reproduction_r1_gpt4o.yaml",
            "sha256": common.sha256_of(r1_config),
        },
        "review_triggers": {
            "delivery_type": {
                "threshold": 0.5,
                "observed": 0.9,
                "triggered": False,
                "operator_decision": None,
                "rationale": None,
                "actor_attack_eligible": None,
            }
        },
        "configuration_under_qualification": {
            "models": {
                "attacker": "gpt-4o-2024-08-06",
                "target": "gpt-4o-2024-08-06",
                "evaluator": "gpt-4o-2024-08-06",
            },
            "prompt_variant": "official",
            "temperatures": {"attacker": 0.7, "target": 0.7, "evaluator": 0.7},
            # Phase 38 P2-4 added this to the qualification identity.
            "structured_output": "json_object",
        },
    }
    payload.update(overrides)
    return payload


def _write_report(tmp_path: Path, payload) -> Path:
    run_id = (
        payload.get("run_id") if isinstance(payload, dict) else None
    ) or "20260101T000000Z"
    path = tmp_path / (
        f"gpt-4o-2024-08-06.baseline.{run_id}.qualification.v2.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_qualified_report_matching_the_config_passes(tmp_path):
    path = _write_report(tmp_path, _qualified_payload())
    block = common.require_qualification(_pilot_cfg(), path)
    assert block["overall"] == "QUALIFIED"
    assert block["gate"] == "enforced before paid client construction"
    assert len(block["sha256"]) == 64


def test_missing_qualification_report_blocks(tmp_path):
    with pytest.raises(common.QualificationError, match="no qualification report"):
        common.require_qualification(_pilot_cfg(), tmp_path / "absent.json")


def test_non_qualified_verdict_blocks(tmp_path):
    path = _write_report(tmp_path, _qualified_payload(overall="NOT_QUALIFIED"))
    with pytest.raises(common.QualificationError, match="not 'QUALIFIED'"):
        common.require_qualification(_pilot_cfg(), path)


def test_incomplete_probe_set_blocks(tmp_path):
    path = _write_report(tmp_path, _qualified_payload(probes_complete=False))
    with pytest.raises(common.QualificationError, match="did not complete"):
        common.require_qualification(_pilot_cfg(), path)


def test_malformed_json_blocks(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(common.QualificationError, match="not valid JSON"):
        common.require_qualification(_pilot_cfg(), path)


def test_non_object_report_blocks(tmp_path):
    path = _write_report(tmp_path, ["QUALIFIED"])
    with pytest.raises(common.QualificationError, match="must be a JSON object"):
        common.require_qualification(_pilot_cfg(), path)


def test_stale_models_block(tmp_path):
    payload = _qualified_payload()
    payload["configuration_under_qualification"]["models"]["target"] = "gpt-4o"
    path = _write_report(tmp_path, payload)
    with pytest.raises(common.QualificationError, match="stale"):
        common.require_qualification(_pilot_cfg(), path)


def test_stale_temperature_blocks(tmp_path):
    payload = _qualified_payload()
    payload["configuration_under_qualification"]["temperatures"]["evaluator"] = 0.0
    path = _write_report(tmp_path, payload)
    with pytest.raises(common.QualificationError, match="temperatures"):
        common.require_qualification(_pilot_cfg(), path)


def test_stale_prompt_variant_blocks(tmp_path):
    payload = _qualified_payload()
    payload["configuration_under_qualification"]["prompt_variant"] = "legacy"
    path = _write_report(tmp_path, payload)
    with pytest.raises(common.QualificationError, match="prompt_variant"):
        common.require_qualification(_pilot_cfg(), path)


class _ReachedClientConstruction(RuntimeError):
    """Sentinel: the paid path got as far as building clients."""


def _pilot_main_args(tmp_path, qualification: Path | None):
    args = [
        "--config", "configs/paper_reproduction_r1_gpt4o.yaml",
        "--variant", "r1_baseline",
        "--limit", "1",
        "--confirm-paid-request",
        "--out-dir", str(tmp_path / "out"),
    ]
    if qualification is not None:
        args += ["--qualification-report", str(qualification)]
    return args


def test_paid_run_blocks_before_client_construction_without_qualification(
        monkeypatch, tmp_path, capsys):
    called = {"build_role_llms": False}

    def _record(*_a, **_k):
        called["build_role_llms"] = True
        raise _ReachedClientConstruction()

    monkeypatch.setattr(common, "build_role_llms", _record)
    monkeypatch.setattr(common, "require_api_key", lambda cfg: "present (len=8)")

    with pytest.raises(common.QualificationError):
        cloud_pilot.main(_pilot_main_args(tmp_path, tmp_path / "absent.json"))

    assert called["build_role_llms"] is False


def test_default_qualification_report_path_resolves_the_newest_v2(tmp_path):
    cfg = _pilot_cfg()
    cfg["qualification"] = {"output_dir": str(tmp_path)}
    older = tmp_path / (
        "gpt-4o-2024-08-06.baseline.20260101T000000Z.qualification.v2.json")
    newer = tmp_path / (
        "gpt-4o-2024-08-06.baseline.20260102T000000Z.qualification.v2.json")
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")
    assert common.default_qualification_report_path(cfg) == newer


def test_confirm_and_api_key_cannot_bypass_qualification(monkeypatch, tmp_path):
    """Both gates are present and a key is 'available'; the run still blocks."""
    monkeypatch.setattr(common, "require_api_key", lambda cfg: "present (len=8)")
    monkeypatch.setattr(
        common, "build_role_llms",
        lambda *_a, **_k: pytest.fail("clients must not be constructed"))

    with pytest.raises(common.QualificationError):
        cloud_pilot.main(_pilot_main_args(tmp_path, tmp_path / "missing.json"))


def test_qualified_report_reaches_client_construction(monkeypatch, tmp_path):
    report = _write_report(tmp_path, _qualified_payload())
    monkeypatch.setattr(common, "require_api_key", lambda cfg: "present (len=8)")

    def _sentinel(*_a, **_k):
        raise _ReachedClientConstruction()

    monkeypatch.setattr(common, "build_role_llms", _sentinel)
    with pytest.raises(_ReachedClientConstruction):
        cloud_pilot.main(_pilot_main_args(tmp_path, report))


def test_dry_run_returns_zero_without_qualification(monkeypatch, tmp_path, capsys):
    args = ["--config", "configs/paper_reproduction_r1_gpt4o.yaml",
            "--variant", "r1_paper", "--limit", "1", "--dry-run",
            "--out-dir", str(tmp_path / "out")]
    assert cloud_pilot.main(args) == 0


# --------------------------------------------------------------------------- #
# P1-2 — attacker temperature propagation
# --------------------------------------------------------------------------- #

def _actor_responses():
    return [
        {"target": "T", "details": {"delivery_type": "article", "other_details": ""}},
        "network text",
        {"actors": [{"actor_name": "A", "relationship": "R"}]},
        "query chain",
        {"questions": [{"question": "q1"}]},
    ]


def test_crescendo_step_uses_the_configured_attacker_temperature():
    llm = RecordingLLM([{"generatedQuestion": "q", "lastResponseSummary": "s"}])
    generate_crescendo_step(
        round_num=1, goal="g", history_attacker=[], history_target=[],
        scores=[], last_response="", attacker_llm=llm, max_rounds=8,
        temperature=0.42,
    )
    assert llm.temperatures == [0.42]


def test_crescendo_attack_class_threads_the_attacker_temperature():
    llm = RecordingLLM([{"generatedQuestion": "q", "lastResponseSummary": "s"}])
    attack = CrescendoAttackPaper(
        attacker_llm=llm, attacker_temperature=0.42, evaluator_temperature=0.11)
    attack.generate_question_for_turn(goal="g", turn_num=1, last_response="")
    assert llm.temperatures == [0.42]


def test_opposite_day_step_uses_the_configured_attacker_temperature():
    llm = RecordingLLM([{"generatedQuestion": "q", "lastResponseSummary": "s"}])
    generate_opposite_day_step(
        round_num=1, goal="g", history_attacker=[], history_target=[],
        scores=[], last_response="", attacker_llm=llm, temperature=0.42,
    )
    assert llm.temperatures == [0.42]


def test_opposite_day_attack_class_threads_the_attacker_temperature():
    llm = RecordingLLM([{"generatedQuestion": "q", "lastResponseSummary": "s"}])
    attack = OppositeDayAttack(attacker_llm=llm, attacker_temperature=0.42)
    attack.generate_question_for_turn(goal="g", turn_num=1, last_response="")
    assert llm.temperatures == [0.42]


def test_actor_attack_pre_attack_uses_the_configured_attacker_temperature():
    llm = RecordingLLM(_actor_responses())
    infer_single("goal", 1, llm, temperature=0.42)
    # extract target, actor network, actor selection, query chain, query parse
    assert llm.temperatures == [0.42] * 5


def test_actor_attack_class_threads_the_attacker_temperature():
    llm = RecordingLLM(_actor_responses())
    attack = ActorAttack(
        attacker_llm=llm, actor_num=1, attacker_temperature=0.42,
        evaluator_temperature=0.11)
    attack.prepare_attack("goal")
    assert llm.temperatures and all(t == 0.42 for t in llm.temperatures)


def test_actor_attack_rewrite_uses_the_configured_attacker_temperature():
    llm = RecordingLLM(["rewritten question"])
    rewrite_query("q", "r", ["q"], {"delivery_type": "article",
                                     "other_details": ""}, llm, temperature=0.42)
    assert llm.temperatures == [0.42]


def test_evaluator_temperature_stays_independent():
    llm = RecordingLLM([json.dumps({"score": 1})])
    attack = ActorAttack(
        attacker_llm=llm, attacker_temperature=0.42, evaluator_temperature=0.11)
    attack.evaluate_response("q", "r", "g")
    assert llm.temperatures == [0.11]


def test_build_attack_forwards_the_attacker_temperature_to_the_three_attacks():
    for name in ("actor_attack", "crescendo_paper", "opposite_day"):
        attack = common.build_attack(
            name, max_turns=8, attacker_llm=RecordingLLM(),
            evaluator_llm=RecordingLLM(), attacker_temperature=0.42,
        )
        assert attack._attacker_temperature == 0.42, name


def test_build_attack_does_not_forward_to_acronym():
    """Acronym is training-only and out of Phase 36 scope; it must not raise."""
    attack = common.build_attack(
        "acronym", max_turns=8, attacker_llm=RecordingLLM(),
        evaluator_llm=RecordingLLM(), attacker_temperature=0.42,
    )
    assert not hasattr(attack, "_attacker_temperature") or (
        attack._attacker_temperature == 0.7)


def test_r1_configuration_still_resolves_to_seven_tenths():
    cfg = common.load_config("configs/paper_reproduction_r1_gpt4o.yaml")
    assert float(cfg["temperature"]["attacker"]) == 0.7
    assert float(cfg["temperature"]["evaluator"]) == 0.7
    attack = common.build_attack(
        "actor_attack", max_turns=8, attacker_llm=RecordingLLM(),
        evaluator_llm=RecordingLLM(),
        attacker_temperature=float(cfg["temperature"]["attacker"]),
        evaluator_temperature=common.evaluator_temperature_of(cfg),
    )
    assert attack._attacker_temperature == 0.7
    assert attack._evaluator_temperature == 0.7
