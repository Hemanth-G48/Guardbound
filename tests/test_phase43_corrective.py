"""Phase 43 — corrective regression suite (offline only).

Covers the Phase 42 findings:
  P0-1  centralized R1 paid authorization across entry points
  P1-1  delivery-type denominators over completed responses
  P1-2  raw vs kept actor counts
  P1-3  review decisions bound to the exact report/run/config/arm
  P2-1  raw `questions` payload validation
  P2-2  malformed payload vs request-error classification
  P2-3  run_id and review_triggers required by the gate
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from guardbound.llm.base import DEFAULT_TEMPERATURE, ChatLLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
R1_CONFIG = PROJECT_ROOT / "configs" / "paper_reproduction_r1_gpt4o.yaml"
R1_MODEL = "gpt-4o-2024-08-06"
UNSET = object()


def _qual():
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import cloud_model_qualification  # noqa: PLC0415

    return cloud_model_qualification


def _common():
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import _cloud_common  # noqa: PLC0415

    return _cloud_common


# --------------------------------------------------------------------------- #
# Scripted actor transport
# --------------------------------------------------------------------------- #

def _actors(distinct=3):
    return [
        {"actor_name": f"a{i}", "relationship": "colleague"}
        for i in range(distinct)
    ]


class ActorFake(ChatLLM):
    name = "phase43-actor"

    def __init__(self, actors=None, details=UNSET, questions=UNSET,
                 fail_query_on_sample=None):
        self.actors = _actors() if actors is None else actors
        self.details = (
            [{"delivery_type": "article"}] if details is UNSET else details)
        self.questions = (
            [{"question": "q"}] if questions is UNSET else questions)
        self.fail_query_on_sample = fail_query_on_sample
        self.sample = -1
        self.actor_seen = False
        self.failed = False

    def generate(self, messages, temperature=DEFAULT_TEMPERATURE,
                 max_turns_context=None, json_format=False,
                 structured_output_mode=None):
        text = "\n".join(str(m.get("content", "")) for m in messages)
        if "Extract the core harmful target" in text:
            self.sample += 1
            self.actor_seen = False
            details = self.details
            if isinstance(details, list):
                details = details[min(self.sample, len(details) - 1)]
            return {"target": "t", "details": copy.deepcopy(details)}
        if "choose most relevant" in text:
            self.actor_seen = True
            return {"actors": copy.deepcopy(self.actors)}
        if "you need extract the content of questions" in text:
            if self.questions is None:
                return {"questions": None}
            if self.questions is ...:  # absent key
                return {}
            return {"questions": copy.deepcopy(self.questions)}
        if (self.actor_seen and not self.failed
                and self.sample == self.fail_query_on_sample):
            self.failed = True
            raise RuntimeError("scripted transport failure")
        return "ok"


def _probe(fake, samples=1, goals=None):
    q = _qual()
    return q.probe_actor_attack_pre_attack(
        fake, goals or [{"task": f"g{i}"} for i in range(samples)],
        samples=samples)


# --------------------------------------------------------------------------- #
# P1-1 — delivery-type denominators
# --------------------------------------------------------------------------- #

def test_delivery_rates_exclude_a_query_phase_request_error():
    result = _probe(ActorFake(
        details=[{"delivery_type": ""}, {"delivery_type": "article"}],
        fail_query_on_sample=1,
    ), samples=2)
    assert result["attempted_count"] == 2
    assert result["successful_response_count"] == 1
    assert result["request_error_count"] == 1
    assert result["delivery_type_valid_rate"] == 0.0
    assert result["delivery_type_present_rate"] == 1.0


def test_delivery_rates_all_valid():
    result = _probe(ActorFake(details={"delivery_type": "article"}), samples=2)
    assert result["delivery_type_valid_rate"] == 1.0
    assert result["delivery_type_present_rate"] == 1.0


def test_delivery_rates_all_invalid():
    result = _probe(ActorFake(details={"delivery_type": ""}), samples=2)
    assert result["delivery_type_valid_rate"] == 0.0
    assert result["delivery_type_present_rate"] == 1.0


def test_delivery_rates_mixed():
    result = _probe(ActorFake(
        details=[{"delivery_type": "article"}, {"delivery_type": ""}]),
        samples=2)
    assert result["delivery_type_valid_rate"] == 0.5
    assert result["delivery_type_present_rate"] == 1.0


def test_delivery_rates_none_when_no_successful_response():
    result = _probe(ActorFake(fail_query_on_sample=0), samples=1)
    assert result["successful_response_count"] == 0
    assert result["delivery_type_valid_rate"] is None
    assert result["delivery_type_present_rate"] is None
    assert 0.0 <= (result["request_error_rate"] or 0.0) <= 1.0


def test_structural_and_request_denominators_are_unchanged():
    result = _probe(ActorFake(
        actors=["not-a-mapping"], fail_query_on_sample=None), samples=1)
    assert result["attempted_count"] == 1
    assert result["successful_response_count"] == 1
    assert result["structural_validity_rate"] == 0.0
    assert result["payload_valid_rate"] == result["structural_validity_rate"]


# --------------------------------------------------------------------------- #
# P1-2 — raw vs kept actors
# --------------------------------------------------------------------------- #

def test_three_distinct_actors_pass():
    result = _probe(ActorFake(actors=_actors(3)))
    sample = result["samples"][0]
    assert sample["raw_actor_count"] == 3
    assert sample["kept_actor_count"] == 3
    assert sample["actor_count"] == 3
    assert sample["structural_valid"] is True


def test_three_raw_with_two_distinct_actors_fail():
    actors = _actors(3)
    actors[1]["actor_name"] = actors[0]["actor_name"]
    sample = _probe(ActorFake(actors=actors))["samples"][0]
    assert sample["raw_actor_count"] == 3
    assert sample["kept_actor_count"] == 2
    assert sample["failure_reason"] == "wrong_actor_count"
    assert sample["structural_valid"] is False


def test_three_raw_with_one_repeated_name_fail():
    actors = [{"actor_name": "a", "relationship": "r"} for _ in range(3)]
    sample = _probe(ActorFake(actors=actors))["samples"][0]
    assert sample["raw_actor_count"] == 3
    assert sample["kept_actor_count"] == 1
    assert sample["failure_reason"] == "wrong_actor_count"


def test_fewer_than_three_raw_actors_fail():
    sample = _probe(ActorFake(actors=_actors(2)))["samples"][0]
    assert sample["raw_actor_count"] == 2
    assert sample["failure_reason"] == "wrong_actor_count"


def test_more_than_three_raw_actors_fail_even_though_filter_keeps_three():
    sample = _probe(ActorFake(actors=_actors(4)))["samples"][0]
    assert sample["raw_actor_count"] == 4
    assert sample["kept_actor_count"] == 3
    assert sample["failure_reason"] == "wrong_actor_count"


def test_invalid_actor_entries_still_report_their_specific_failure():
    sample = _probe(ActorFake(actors=["x"]))["samples"][0]
    assert sample["failure_reason"] == "actor_not_mapping"
    assert sample["rejected_non_mapping"] == 1


# --------------------------------------------------------------------------- #
# P2-1 / P2-2 — raw questions payload
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("questions,reason", [
    ("abcd", "invalid_query_payload"),
    ({"a": "q"}, "invalid_query_payload"),
    (5, "invalid_query_payload"),
    (None, "invalid_query_payload"),
    (["ok", 3], "non_string_query"),
    ([{"other": "x"}], "missing_query_field"),
    ([{"question": ""}], "empty_query"),
])
def test_malformed_questions_are_structural_not_request_errors(
        questions, reason):
    sample = _probe(ActorFake(questions=questions))["samples"][0]
    assert sample["failure_reason"] == reason
    assert sample["request_error"] is False
    assert sample["structural_valid"] is False


def test_absent_questions_is_a_structural_failure():
    sample = _probe(ActorFake(questions=...))["samples"][0]
    assert sample["failure_reason"] == "missing_questions"
    assert sample["request_error"] is False


def test_empty_questions_list_is_empty_chain():
    sample = _probe(ActorFake(questions=[]))["samples"][0]
    assert sample["failure_reason"] == "empty_chain"
    assert sample["structural_valid"] is False


def test_valid_questions_list_passes():
    sample = _probe(ActorFake(
        questions=[{"question": "q1"}, "q2"]))["samples"][0]
    assert sample["structural_valid"] is True


def test_get_init_queries_default_path_is_unchanged():
    from guardbound.attacks.actor_attack import QueryPayloadError, get_init_queries

    fake = ActorFake(questions="abcd")
    queries, _ = get_init_queries(
        "t", {"actor_name": "a", "relationship": "r"}, fake)
    assert queries == ["a", "b", "c", "d"]

    sink: dict = {}
    with pytest.raises(QueryPayloadError):
        get_init_queries(
            "t", {"actor_name": "a", "relationship": "r"}, fake,
            diagnostics=sink)
    assert sink["questions_type"] == "str"
    assert sink["questions_is_list"] is False


# --------------------------------------------------------------------------- #
# Report fixtures for the gate
# --------------------------------------------------------------------------- #

def _cfg():
    return {
        "models": {
            role: {"provider": "openai", "model": R1_MODEL}
            for role in ("attacker", "target", "evaluator")
        },
        "temperature": {"attacker": 0.7, "target": 0.7, "evaluator": 0.7},
        "prompt_variant": "official",
        "attacker_prompt_condition": "baseline",
        "cloud": {"structured_output": "json_object"},
    }


def _report_payload(**overrides):
    common = _common()
    payload = {
        "schema_version": "qualification.v2",
        "probe_spec_version": "r1-probe.v2",
        "run_id": "20260101T000000Z",
        "overall": "QUALIFIED",
        "probes_complete": True,
        "attacker_prompt_condition": "baseline",
        "experiment_config": {
            "path": "configs/paper_reproduction_r1_gpt4o.yaml",
            "sha256": common.sha256_of(R1_CONFIG),
        },
        "configuration_under_qualification": {
            "models": {
                "attacker": R1_MODEL,
                "target": R1_MODEL,
                "evaluator": R1_MODEL,
            },
            "prompt_variant": "official",
            "temperatures": {"attacker": 0.7, "target": 0.7, "evaluator": 0.7},
            "structured_output": "json_object",
        },
        "probe_status": {
            name: {"status": "PASS"} for name in common.MANDATORY_PROBE_NAMES
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
    }
    payload.update(overrides)
    return payload


def _write(tmp_path, payload, name=None):
    run_id = payload.get("run_id") or "20260101T000000Z"
    path = tmp_path / (
        name or f"{R1_MODEL}.baseline.{run_id}.qualification.v2.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _decision(block, **overrides):
    payload = {
        "qualification_report_sha256": block["sha256"],
        "run_id": block["run_id"],
        "configuration_sha256": block["experiment_config"]["sha256"],
        "arm": block["attacker_prompt_condition"],
        "trigger": "delivery_type",
        "decision": "accept_degradation",
        "rationale": "documented degradation",
        "decided_by": "operator",
        "decided_at": "2026-11-12T00:00:00Z",
        "actor_attack_eligible": True,
    }
    payload.update(overrides)
    return payload


def _triggered_block(tmp_path):
    common = _common()
    payload = _report_payload()
    payload["review_triggers"]["delivery_type"].update(
        {"observed": 0.1, "triggered": True})
    path = _write(tmp_path, payload)
    return common, common.require_qualification(_cfg(), path)


# --------------------------------------------------------------------------- #
# P1-3 — review decision binding
# --------------------------------------------------------------------------- #

def test_valid_bound_decision_is_accepted(tmp_path):
    common, block = _triggered_block(tmp_path)
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(_decision(block)), encoding="utf-8")
    assert common.require_review_decision(block, path)["decision"] == (
        "accept_degradation")


@pytest.mark.parametrize("field,value", [
    ("qualification_report_sha256", "0" * 64),
    ("run_id", "20260102T000000Z"),
    ("configuration_sha256", "1" * 64),
    ("arm", "c1_generation_point"),
])
def test_decision_for_another_report_is_rejected(tmp_path, field, value):
    common, block = _triggered_block(tmp_path)
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(_decision(block, **{field: value})),
                    encoding="utf-8")
    with pytest.raises(common.QualificationError, match="does not belong"):
        common.require_review_decision(block, path)


def test_decision_missing_a_binding_field_is_rejected(tmp_path):
    common, block = _triggered_block(tmp_path)
    payload = _decision(block)
    payload.pop("qualification_report_sha256")
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(common.QualificationError, match="missing"):
        common.require_review_decision(block, path)


def test_decision_with_invalid_timestamp_is_rejected(tmp_path):
    common, block = _triggered_block(tmp_path)
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(_decision(block, decided_at="not-a-date")),
                    encoding="utf-8")
    with pytest.raises(common.QualificationError, match="ISO-8601"):
        common.require_review_decision(block, path)


def test_decision_with_ineligible_exclusion_is_rejected(tmp_path):
    common, block = _triggered_block(tmp_path)
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(_decision(
        block, decision="exclude_actor_attack", actor_attack_eligible=True)),
        encoding="utf-8")
    with pytest.raises(common.QualificationError, match="eligible"):
        common.require_review_decision(block, path)


def test_decision_with_inactive_trigger_is_rejected(tmp_path):
    common, block = _triggered_block(tmp_path)
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(_decision(block, trigger="other")),
                    encoding="utf-8")
    with pytest.raises(common.QualificationError, match="trigger"):
        common.require_review_decision(block, path)


def test_unused_decision_is_rejected_not_silently_discarded(tmp_path):
    common = _common()
    block = common.require_qualification(
        _cfg(), _write(tmp_path, _report_payload()))
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(_decision(block)), encoding="utf-8")
    with pytest.raises(common.QualificationError, match="no review trigger"):
        common.require_review_decision(block, path)


# --------------------------------------------------------------------------- #
# P2-3 — run_id and review_triggers gate
# --------------------------------------------------------------------------- #

def test_missing_run_id_is_rejected(tmp_path):
    common = _common()
    payload = _report_payload()
    payload.pop("run_id")
    with pytest.raises(common.QualificationError, match="run_id"):
        common.require_qualification(_cfg(), _write(tmp_path, payload))


def test_invalid_run_id_is_rejected(tmp_path):
    common = _common()
    with pytest.raises(common.QualificationError, match="run_id"):
        common.require_qualification(
            _cfg(), _write(tmp_path, _report_payload(run_id="2026-01-01")))


def test_filename_run_id_mismatch_is_rejected(tmp_path):
    common = _common()
    payload = _report_payload()
    path = _write(
        tmp_path, payload,
        name=f"{R1_MODEL}.baseline.20260102T000000Z.qualification.v2.json")
    with pytest.raises(common.QualificationError, match="canonical filename"):
        common.require_qualification(_cfg(), path)


def test_non_canonical_filename_is_rejected(tmp_path):
    common = _common()
    path = _write(tmp_path, _report_payload(), name="qual.json")
    with pytest.raises(common.QualificationError, match="not canonical"):
        common.require_qualification(_cfg(), path)


def test_missing_review_triggers_is_rejected(tmp_path):
    common = _common()
    payload = _report_payload()
    payload.pop("review_triggers")
    with pytest.raises(common.QualificationError, match="review_triggers"):
        common.require_qualification(_cfg(), _write(tmp_path, payload))


def test_missing_delivery_type_trigger_is_rejected(tmp_path):
    common = _common()
    payload = _report_payload(review_triggers={})
    with pytest.raises(common.QualificationError, match="delivery_type"):
        common.require_qualification(_cfg(), _write(tmp_path, payload))


@pytest.mark.parametrize("field,value,match", [
    ("threshold", 1.5, "threshold"),
    ("observed", 2.0, "observed"),
    ("triggered", "yes", "triggered"),
    ("threshold", "0.5", "threshold"),
])
def test_invalid_trigger_fields_are_rejected(tmp_path, field, value, match):
    common = _common()
    payload = _report_payload()
    payload["review_triggers"]["delivery_type"][field] = value
    with pytest.raises(common.QualificationError, match=match):
        common.require_qualification(_cfg(), _write(tmp_path, payload))


def test_inconsistent_trigger_is_rejected(tmp_path):
    common = _common()
    payload = _report_payload()
    payload["review_triggers"]["delivery_type"].update(
        {"observed": 0.1, "triggered": False})
    with pytest.raises(common.QualificationError, match="inconsistent"):
        common.require_qualification(_cfg(), _write(tmp_path, payload))


def test_valid_no_trigger_report_passes(tmp_path):
    common = _common()
    block = common.require_qualification(
        _cfg(), _write(tmp_path, _report_payload()))
    assert block["overall"] == "QUALIFIED"
    assert block["review_triggers"]["delivery_type"]["triggered"] is False


# --------------------------------------------------------------------------- #
# P0-1 — centralized paid authorization
# --------------------------------------------------------------------------- #

def _missing(tmp_path):
    return tmp_path / "missing.qualification.v2.json"


def test_r1_paid_model_without_report_is_blocked(tmp_path):
    common = _common()
    with pytest.raises(common.QualificationError):
        common.require_paid_authorization(
            R1_MODEL, qualification_report=_missing(tmp_path))


def test_non_r1_model_is_not_gated(tmp_path):
    common = _common()
    block = common.require_paid_authorization(
        "gpt-4o-mini-2024-07-18", qualification_report=_missing(tmp_path))
    assert block["gated"] is False


def test_old_qualification_purpose_string_is_rejected(tmp_path):
    common = _common()
    with pytest.raises(TypeError):
        common.require_paid_authorization(
            R1_MODEL, qualification_report=_missing(tmp_path),
            purpose="qualification")


def test_r1_paid_model_with_valid_report_and_no_trigger_passes(tmp_path):
    common = _common()
    report = _write(tmp_path, _report_payload())
    block = common.require_paid_authorization(
        R1_MODEL, qualification_report=report)
    assert block["gated"] is True
    assert block["qualification"]["overall"] == "QUALIFIED"
    assert block["review_decision"] is None


def test_r1_paid_model_with_trigger_needs_bound_decision(tmp_path):
    common = _common()
    report_payload = _report_payload()
    report_payload["review_triggers"]["delivery_type"].update(
        {"observed": 0.1, "triggered": True})
    report = _write(tmp_path, report_payload)
    with pytest.raises(common.QualificationError, match="review"):
        common.require_paid_authorization(
            R1_MODEL, qualification_report=report)

    verified = common.require_qualification(_cfg(), report)
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(_decision(verified)), encoding="utf-8")
    block = common.require_paid_authorization(
        R1_MODEL, qualification_report=report,
        review_decision=decision_path)
    assert block["review_decision"]["decision"] == "accept_degradation"


def test_shared_factory_is_gated_for_r1(tmp_path):
    common = _common()
    from guardbound.llm.usage import UsageLedger

    cfg = _cfg()
    with pytest.raises(common.QualificationError):
        common.build_role_llms(
            cfg, UsageLedger(), qualification_report=_missing(tmp_path))


def test_shared_factory_is_not_gated_for_non_r1():
    common = _common()
    from guardbound.llm.usage import UsageLedger

    cfg = {
        "models": {
            role: {"provider": "openai", "model": "gpt-4o-mini-2024-07-18",
                   "client": object()}
            for role in ("attacker", "target", "evaluator")
        },
        "attacker_prompt_condition": "baseline",
    }
    # The guard returns before any client construction; no exception.
    assert common.is_r1_paid_model("gpt-4o-mini-2024-07-18", cfg) is False


# --------------------------------------------------------------------------- #
# P0-1 — direct entry-point coverage
# --------------------------------------------------------------------------- #

def _args(**overrides):
    base = {
        "mock": False,
        "qualification_report": None,
        "review_decision": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_eval_entry_points_block_r1_without_authorization(tmp_path):
    import eval_mmlu
    import eval_mtbench
    import eval_over_refusal

    missing = str(_missing(tmp_path))
    with pytest.raises(_common().QualificationError):
        eval_mmlu.make_llm(_args(model=R1_MODEL,
                                 qualification_report=missing))
    with pytest.raises(_common().QualificationError):
        eval_mtbench.make_llm(_args(model="gpt-3.5-turbo-0125",
                                    judge_model=R1_MODEL,
                                    qualification_report=missing), judge=True)
    with pytest.raises(_common().QualificationError):
        eval_over_refusal.make_llm(_args(model=R1_MODEL,
                                         qualification_report=missing))


def test_judge_and_runner_entry_points_block_r1(tmp_path):
    import judge_asr
    import run_attacks
    import run_baseline_evals

    missing = str(_missing(tmp_path))
    with pytest.raises(_common().QualificationError):
        judge_asr.make_judge(_args(
            judge_model=R1_MODEL, qualification_report=missing,
            judge_prompt="configs/judge_prompts/asr_judge.txt",
            cache_dir=str(tmp_path / "cache")))
    with pytest.raises(_common().QualificationError):
        run_attacks.make_target_llm(_args(
            target=R1_MODEL, qualification_report=missing))
    with pytest.raises(_common().QualificationError):
        run_attacks.make_attacker_llm(_args(
            attacker=R1_MODEL, qualification_report=missing))
    with pytest.raises(_common().QualificationError):
        run_baseline_evals.make_target_llm(_args(
            model=R1_MODEL, qualification_report=missing))


def test_full_evaluation_and_reproduction_block_r1(tmp_path):
    import run_full_evaluation
    import run_reproduction

    missing = str(_missing(tmp_path))
    with pytest.raises(_common().QualificationError):
        run_full_evaluation.setup_judge_model(
            _args(qualification_report=missing))
    with pytest.raises(_common().QualificationError):
        run_full_evaluation.setup_target_model(
            "gpt-4o", _args(qualification_report=missing))
    cfg = {"attacker": {"model": R1_MODEL}}
    with pytest.raises(_common().QualificationError):
        run_reproduction.make_llm(
            cfg, "attacker", _args(attacker=None, target=None,
                                   qualification_report=missing),
            mock=False)


def test_smoke_test_blocks_r1_before_key_access(tmp_path):
    import cloud_smoke_test

    missing = str(_missing(tmp_path))
    with pytest.raises(_common().QualificationError):
        cloud_smoke_test.main([
            "--model", R1_MODEL,
            "--confirm-paid-request",
            "--qualification-report", missing,
        ])


def test_mock_paths_never_hit_the_gate(tmp_path):
    import eval_mmlu
    import judge_asr
    import run_attacks

    missing = str(_missing(tmp_path))
    args = _args(mock=True, model=R1_MODEL, qualification_report=missing,
                 judge_model=R1_MODEL, target=R1_MODEL,
                 judge_prompt="configs/judge_prompts/asr_judge.txt",
                 cache_dir=str(tmp_path / "cache"))
    assert eval_mmlu.make_llm(args) is not None
    assert run_attacks.make_target_llm(args) is not None
    judge = judge_asr.make_judge(args)
    assert judge is not None
