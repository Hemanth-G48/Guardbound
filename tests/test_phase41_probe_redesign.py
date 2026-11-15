"""Phase 41 — offline regression suite for the v2 qualification probes.

Every test uses scripted local fakes: no network, no credentials, no model
load. The suite maps one-to-one onto T1–T45 of the Phase 40B specification.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from guardbound.llm.base import DEFAULT_TEMPERATURE, ChatLLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
R1_CONFIG = PROJECT_ROOT / "configs" / "paper_reproduction_r1_gpt4o.yaml"
SYSTEM = "You are a helpful assistant."


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
# Scripted fakes
# --------------------------------------------------------------------------- #

def _valid_actors():
    return [
        {"actor_name": f"a{i}", "relationship": "colleague"}
        for i in range(3)
    ]


def _questions():
    return [["q1", "q2"], ["q3"], ["q4", "q5"]]


_UNSET = object()


class ActorLLM(ChatLLM):
    """Scripted attacker transport for the pre-attack sequence."""

    name = "actor-fake"

    def __init__(
        self,
        actors=None,
        actors_payload=None,
        details=_UNSET,
        details_present=True,
        questions=None,
        fail_on_sample=None,
        empty_actors_on=None,
    ):
        self.actors = _valid_actors() if actors is None else actors
        self.actors_payload = (
            actors_payload if actors_payload is not None
            else {"actors": self.actors}
        )
        self.details = {} if details is _UNSET else details
        self.details_present = details_present
        self.questions = _questions() if questions is None else questions
        self.fail_on_sample = fail_on_sample
        self.empty_actors_on = set(empty_actors_on or ())
        self.sample = -1
        self.query_call = 0
        self.calls: list[dict] = []

    def generate(
        self,
        messages,
        temperature=DEFAULT_TEMPERATURE,
        max_turns_context=None,
        json_format=False,
        structured_output_mode=None,
    ):
        text = "\n".join(str(m.get("content", "")) for m in messages)
        self.calls.append({
            "temperature": temperature,
            "json_format": bool(json_format),
            "messages": copy.deepcopy(messages),
        })
        if "Extract the core harmful target" in text:
            self.sample += 1
            if self.fail_on_sample == self.sample:
                raise RuntimeError("scripted transport failure")
            payload = {"target": "t"}
            if self.details_present:
                payload["details"] = copy.deepcopy(self.details)
            return payload
        if "choose most relevant" in text:
            if self.sample in self.empty_actors_on:
                return {"actors": []}
            return copy.deepcopy(self.actors_payload)
        if "you need extract the content of questions" in text:
            index = min(self.query_call, len(self.questions) - 1)
            self.query_call += 1
            return {"questions": copy.deepcopy(self.questions[index])}
        return "ok"


class TargetLLM(ChatLLM):
    """Scripted target transport; one entry per assistant reply."""

    name = "target-fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.index = 0
        self.calls: list[dict] = []

    def generate(
        self,
        messages,
        temperature=DEFAULT_TEMPERATURE,
        max_turns_context=None,
        json_format=False,
        structured_output_mode=None,
    ):
        self.calls.append({
            "messages": copy.deepcopy(messages),
            "temperature": temperature,
        })
        reply = self.replies[self.index] if self.index < len(self.replies) else "ok"
        self.index += 1
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _run_actor(llm, goals=None, samples=1):
    q = _qual()
    return q.probe_actor_attack_pre_attack(
        llm, goals or [{"task": "goal"}], samples=samples)


# --------------------------------------------------------------------------- #
# T1–T18, T39–T40 — ActorAttack pre-attack contract
# --------------------------------------------------------------------------- #

def test_T01_valid_payload_passes():
    result = _run_actor(ActorLLM())
    assert result["structural_validity_rate"] == 1.0
    assert result["samples"][0]["structural_valid"] is True
    assert result["samples"][0]["actor_count"] == 3
    assert result["samples"][0]["query_count"] == 5


def test_T02_empty_delivery_type_is_diagnostic_only():
    result = _run_actor(ActorLLM(details={"delivery_type": ""}))
    sample = result["samples"][0]
    assert sample["structural_valid"] is True
    assert sample["delivery_type_state"] == "empty"
    assert result["delivery_type_valid_rate"] == 0.0


def test_T03_missing_delivery_type_state():
    result = _run_actor(ActorLLM(details={}))
    assert result["samples"][0]["delivery_type_state"] == "missing"
    assert result["samples"][0]["structural_valid"] is True


def test_T04_null_delivery_type_state_is_distinct():
    result = _run_actor(ActorLLM(details={"delivery_type": None}))
    assert result["samples"][0]["delivery_type_state"] == "null"
    assert result["samples"][0]["structural_valid"] is True


def test_T05_invalid_type_delivery_type_state():
    result = _run_actor(ActorLLM(details={"delivery_type": 5}))
    assert result["samples"][0]["delivery_type_state"] == "invalid_type"
    assert result["samples"][0]["structural_valid"] is True


def test_T06_missing_relationship_fails():
    actors = _valid_actors()
    actors[1].pop("relationship")
    result = _run_actor(ActorLLM(actors=actors))
    assert result["samples"][0]["failure_reason"] == "missing_relationship"
    assert result["structural_validity_rate"] == 0.0


def test_T07_empty_relationship_fails():
    actors = _valid_actors()
    actors[1]["relationship"] = ""
    result = _run_actor(ActorLLM(actors=actors))
    assert result["samples"][0]["failure_reason"] == "invalid_relationship"


def test_T08_non_mapping_actor_is_not_silently_dropped():
    result = _run_actor(ActorLLM(actors_payload={"actors": ["x"]}))
    sample = result["samples"][0]
    assert sample["failure_reason"] == "actor_not_mapping"
    assert sample["rejected_non_mapping"] == 1


def test_T09_missing_actor_name_fails_via_diagnostics():
    actors = _valid_actors()
    actors[2] = {"relationship": "colleague"}
    result = _run_actor(ActorLLM(actors=actors))
    sample = result["samples"][0]
    assert sample["failure_reason"] == "missing_actor_name"
    assert sample["rejected_missing_actor_name"] == 1


def test_T10_empty_query_chain_fails():
    result = _run_actor(ActorLLM(questions=[[], ["q"], ["q"]]))
    assert result["samples"][0]["failure_reason"] == "empty_chain"
    assert result["empty_chain_rate"] == 1.0


def test_T11_non_string_or_empty_query_fails():
    result = _run_actor(ActorLLM(questions=[["ok", "", 3], ["q"], ["q"]]))
    assert result["samples"][0]["failure_reason"] in {
        "empty_query", "non_string_query"
    }


def test_T12_wrong_actor_count_fails():
    result = _run_actor(ActorLLM(actors=_valid_actors()[:2]))
    assert result["samples"][0]["failure_reason"] == "wrong_actor_count"


def test_T13_absent_query_details_is_recorded_as_absent():
    result = _run_actor(ActorLLM(details={}, details_present=False))
    sample = result["samples"][0]
    assert sample["query_details_state"] == "absent"
    assert sample["delivery_type_state"] is None
    assert sample["structural_valid"] is True


def test_T14_null_query_details_passes():
    result = _run_actor(ActorLLM(details=None))
    sample = result["samples"][0]
    assert sample["query_details_state"] == "null"
    assert sample["structural_valid"] is True


def test_T15_non_mapping_query_details_fails():
    result = _run_actor(ActorLLM(details="text"))
    assert result["samples"][0]["failure_reason"] == "invalid_query_details"


def test_T16_request_errors_leave_the_structural_denominator():
    goals = [{"task": f"goal-{i}"} for i in range(30)]
    result = _run_actor(
        ActorLLM(fail_on_sample=0, empty_actors_on={1, 2}),
        goals=goals, samples=30)
    assert result["attempted_count"] == 30
    assert result["request_error_count"] == 1
    assert result["successful_response_count"] == 29
    assert result["structural_valid_count"] == 27
    assert result["structural_validity_rate"] == round(27 / 29, 4)
    assert result["request_error_rate"] == round(1 / 30, 4)


def test_T17_zero_responses_is_insufficient_not_zero():
    goals = [{"task": f"goal-{i}"} for i in range(3)]
    q = _qual()
    llm = ActorLLM()

    def _always_fail(messages, **kwargs):
        raise RuntimeError("transport down")

    llm.generate = _always_fail
    result = q.probe_actor_attack_pre_attack(llm, goals, samples=3)
    assert result["successful_response_count"] == 0
    assert result["structural_validity_rate"] is None
    decision = q.decide(
        "attacker",
        {"actor_attack_payload_validity_rate": result["structural_validity_rate"]},
        {"actor_attack_payload_validity_rate": {"op": ">=", "value": 0.9}},
    )
    assert decision["decision"] == q.INSUFFICIENT_DATA


def test_T18_goal_order_is_file_order_and_goal_text_is_not_stored():
    goals = [{"task": "goal-A"}, {"task": "goal-B"}]
    result = _run_actor(ActorLLM(), goals=goals, samples=2)
    assert [s["goal_index"] for s in result["samples"]] == [0, 1]
    assert "goal-A" not in json.dumps(result)
    assert "goal-B" not in json.dumps(result)


def test_T39_invalid_actor_field_types():
    actors = _valid_actors()
    actors[0]["actor_name"] = 42
    actors[1]["relationship"] = ["x"]
    result = _run_actor(ActorLLM(actors=actors))
    assert result["samples"][0]["failure_reason"] in {
        "invalid_actor_name", "invalid_relationship"
    }


def test_T40_valid_delivery_type_state():
    result = _run_actor(ActorLLM(details={"delivery_type": "Tutorial"}))
    sample = result["samples"][0]
    assert sample["delivery_type_state"] == "valid"
    assert result["delivery_type_valid_rate"] == 1.0


# --------------------------------------------------------------------------- #
# T19–T24, T41–T42 — continuity transcript
# --------------------------------------------------------------------------- #

def _run_continuity(replies, conversations=1):
    q = _qual()
    return q.probe_target(
        TargetLLM(replies), [], 1,
        target_temperature=0.7,
        system_prompt=SYSTEM,
        conversations=conversations,
    )


def test_T19_exactly_nine_messages_in_order():
    result = _run_continuity(["stored", "4", "red", "Miso"])
    conversation = result["multi_turn"]["conversations"][0]
    messages = conversation["messages"]
    assert len(messages) == 9
    assert [m["role"] for m in messages] == [
        "system", "user", "assistant", "user", "assistant",
        "user", "assistant", "user", "assistant",
    ]
    assert all(messages[i]["content"].strip() for i in (2, 4, 6))
    assert messages[7]["chars"] == len("What is the name of my cat?")
    assert messages[8]["content"] == "Miso"


def test_T20_system_prompt_is_first_and_hashed():
    result = _run_continuity(["stored", "4", "red", "Miso"])
    messages = result["multi_turn"]["conversations"][0]["messages"]
    expected = hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest()[:16]
    assert messages[0]["role"] == "system"
    assert messages[0]["content_sha256_16"] == expected
    assert result["multi_turn"]["system_prompt_sent"] is True


def test_T21_all_four_assistant_replies_are_retained():
    result = _run_continuity(["stored", "4", "red", "Miso"])
    messages = result["multi_turn"]["conversations"][0]["messages"]
    assert [messages[i]["content"] for i in (2, 4, 6, 8)] == [
        "stored", "4", "red", "Miso"]


def test_T22_final_reply_is_a_reference_not_a_message():
    result = _run_continuity(["stored", "4", "red", "Miso"])
    conversation = result["multi_turn"]["conversations"][0]
    assert conversation["final_message_index"] == 8
    assert conversation["final_reply"] == "Miso"
    assert len(conversation["messages"]) == 9


def test_T23_refusal_counts_as_failed_continuity():
    result = _run_continuity(
        ["stored", "4", "red", "I'm sorry, but I can't help with that."])
    block = result["multi_turn"]
    assert block["conversations"][0]["outcome"] == "refusal"
    assert block["refusal_count"] == 1
    assert block["pass_count"] == 0
    assert block["continuity_rate"] == 0.0


def test_T24_incorrect_is_not_a_refusal():
    result = _run_continuity(["stored", "4", "red", "I don't know"])
    assert result["multi_turn"]["conversations"][0]["outcome"] == "incorrect"


def test_T41_missing_system_prompt_fails_closed():
    q = _qual()
    with pytest.raises(ValueError, match="system prompt"):
        q.probe_target(TargetLLM([]), [], 1, conversations=1)


def test_T42_continuity_request_errors_are_separate():
    replies = ["stored", "4", "red", "Miso"] * 2
    replies.append(RuntimeError("transport down"))
    replies += ["stored", "4", "red", "Miso"] * 2
    result = _run_continuity(replies, conversations=5)
    block = result["multi_turn"]
    assert block["request_error_count"] == 1
    assert block["request_error_rate"] == 0.2
    assert block["successful_conversation_count"] == 4
    assert block["pass_count"] == 4
    assert block["continuity_rate"] == 1.0


# --------------------------------------------------------------------------- #
# Report identity, persistence and resolution
# --------------------------------------------------------------------------- #

def _cfg(out_dir=None):
    cfg = {
        "models": {
            role: {"provider": "openai", "model": "gpt-4o-2024-08-06"}
            for role in ("attacker", "target", "evaluator")
        },
        "temperature": {"attacker": 0.7, "target": 0.7, "evaluator": 0.7},
        "prompt_variant": "official",
        "attacker_prompt_condition": "baseline",
        "cloud": {"structured_output": "json_object"},
    }
    if out_dir is not None:
        cfg["qualification"] = {"output_dir": str(out_dir)}
    return cfg


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
                "attacker": "gpt-4o-2024-08-06",
                "target": "gpt-4o-2024-08-06",
                "evaluator": "gpt-4o-2024-08-06",
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


def _write(tmp_path, payload, name=None, run_id=None):
    run_id = run_id or payload.get("run_id") or "20260101T000000Z"
    path = tmp_path / (
        name or f"gpt-4o-2024-08-06.baseline.{run_id}.qualification.v2.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_report():
    q = _qual()
    from guardbound.llm.attacker_arms import resolve_arm

    models = {
        role: {"provider": "openai", "model": "gpt-4o-2024-08-06"}
        for role in q.ROLES
    }
    return q._base_report(
        cfg=_cfg(),
        args=SimpleNamespace(offline_mock=True),
        models=models,
        arm=resolve_arm("baseline"),
        prompt_variant="official",
        thresholds={},
        evaluator_temperature=0.7,
        target_temperature=0.7,
        run_id="20261112T144310Z",
        config_path=R1_CONFIG,
    )


def test_T25_report_versions_and_sortable_run_id():
    q = _qual()
    report = _base_report()
    assert report["schema_version"] == q.SCHEMA_VERSION
    assert report["probe_spec_version"] == q.PROBE_SPEC_VERSION
    assert report["run_id"] == "20261112T144310Z"
    first = q.utc_run_id(datetime(2026, 1, 1, tzinfo=timezone.utc))
    second = q.utc_run_id(datetime(2026, 1, 2, tzinfo=timezone.utc))
    assert first == "20260101T000000Z"
    assert first < second


class _Ledger:
    def records(self):
        return []

    def summary(self):
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                "estimated_cost_usd": 0.0}


def _persist(tmp_path, run_id="20260101T000000Z"):
    q = _qual()
    results = {
        "schema_version": q.SCHEMA_VERSION,
        "probe_spec_version": q.PROBE_SPEC_VERSION,
        "overall": "QUALIFIED",
        "probes_complete": True,
        "decisions": {},
    }
    return q._persist_report(
        out_dir=tmp_path,
        tag="gpt-4o-2024-08-06",
        arm_name="baseline",
        run_id=run_id,
        results=results,
        ledger=_Ledger(),
        usage_jsonl=tmp_path / "usage.jsonl",
        offline_mock=True,
    )


def test_T26_canonical_filename(tmp_path):
    path = _persist(tmp_path)
    assert path.name == (
        "gpt-4o-2024-08-06.baseline.20260101T000000Z.qualification.v2.json")


def test_T27_persistence_refuses_overwrite(tmp_path):
    path = _persist(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        _persist(tmp_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_T28_resolver_picks_newest(tmp_path):
    common = _common()
    _write(tmp_path, _report_payload(run_id="20260101T000000Z"))
    newest = _write(tmp_path, _report_payload(run_id="20260102T000000Z"))
    assert common.resolve_qualification_report(_cfg(tmp_path)) == newest


def test_T29_same_timestamp_requires_explicit_path():
    common = _common()
    name = "gpt-4o-2024-08-06.baseline.20260101T000000Z.qualification.v2.json"
    with pytest.raises(common.QualificationError, match="share run_id"):
        common._select_newest_report([Path("a") / name, Path("b") / name])


def test_T30_no_fallback_to_an_older_report(tmp_path):
    common = _common()
    _write(tmp_path, _report_payload(run_id="20260101T000000Z"))
    _write(tmp_path, _report_payload(
        run_id="20260102T000000Z", overall="NOT_QUALIFIED"))
    with pytest.raises(common.QualificationError, match="not 'QUALIFIED'"):
        common.require_qualification(_cfg(tmp_path))


# --------------------------------------------------------------------------- #
# T31–T38, T43–T45 — gate validation
# --------------------------------------------------------------------------- #

def _gate(tmp_path, payload):
    common = _common()
    path = _write(tmp_path, payload)
    return common.require_qualification(_cfg(), path)


def test_T31_legacy_v1_report_is_readable_but_ineligible(tmp_path):
    common = _common()
    payload = _report_payload()
    payload.pop("schema_version")
    payload.pop("probe_spec_version")
    path = _write(tmp_path, payload)
    with pytest.raises(common.QualificationError, match="legacy v1"):
        common.require_qualification(_cfg(), path)
    assert path.exists()


def test_T32_unknown_schema_is_rejected(tmp_path):
    common = _common()
    with pytest.raises(common.QualificationError, match="schema_version"):
        _gate(tmp_path, _report_payload(schema_version="qualification.v9"))


def test_T33_wrong_probe_spec_is_rejected(tmp_path):
    common = _common()
    with pytest.raises(common.QualificationError, match="probe spec"):
        _gate(tmp_path, _report_payload(probe_spec_version="r1-probe.v1"))


def test_T34_missing_mandatory_probe_is_rejected(tmp_path):
    common = _common()
    payload = _report_payload()
    payload["probe_status"].pop("target")
    with pytest.raises(common.QualificationError, match="missing"):
        _gate(tmp_path, payload)


def test_T35_failed_or_unexecuted_probe_is_rejected(tmp_path):
    common = _common()
    for state in ("NOT_EXECUTED", "FAIL", "INSUFFICIENT_DATA"):
        payload = _report_payload()
        payload["probe_status"]["target"] = {"status": state}
        with pytest.raises(common.QualificationError, match="not 'PASS'"):
            _gate(tmp_path, payload)


def test_T36_diagnostics_cannot_affect_completeness():
    q = _qual()
    probes = {
        "attacker_contract": {"contract_validity_rate": 1.0, "attack_coverage": 1.0,
                              "empty_question_rate": 0.0, "request_error_rate": 0.0},
        "attacker_multi_turn": {"multi_turn_usable_rate": 1.0},
        "actor_attack_pre_attack": {"payload_valid_rate": 1.0,
                                    "empty_chain_rate": 0.0},
        "target": {"completion": {"non_empty_rate": 1.0, "truncation_rate": 0.0,
                                  "request_error_rate": 0.0},
                   "multi_turn": {"context_recall_rate": 1.0}},
        "evaluator_rubric": {"rubric_validity_rate": 1.0,
                             "rubric_determinism_rate": 1.0,
                             "rubric_agreement": 1.0,
                             "structural_violation_rate": 0.0},
        "evaluator_refusal": {"refusal_agreement": 1.0, "request_error_rate": 0.0},
        "evaluator_disclaimer": {"disclaimer_agreement": 1.0},
        "nbf_integration": {
            "checkpoint_verified": True, "load_ok": True,
            "embedding_dim_ok": True, "score_valid": True,
            "threshold_applied": True, "deterministic": True,
        },
        "candidate_filter": {"filter_ok": True, "both_branches": True},
        "diagnostics": {"target_code_word": {"status": q.PROBE_FAIL}},
    }
    metrics = q.assemble_metrics(probes)
    thresholds = {role: q.thresholds_for(_cfg(), role) for role in q.DECISION_ROLES}
    statuses = q.probes_status(probes, metrics, thresholds)
    assert "target_code_word" not in statuses
    assert set(statuses) == set(q.MANDATORY_PROBE_NAMES)
    assert q.probes_complete(statuses) is True


def test_T37_review_trigger_is_recorded_without_a_decision():
    q = _qual()
    block = q.build_review_triggers({"delivery_type_valid_rate": 0.1})
    assert block["delivery_type"]["triggered"] is True
    assert block["delivery_type"]["observed"] == 0.1
    assert block["delivery_type"]["operator_decision"] is None
    assert block["delivery_type"]["rationale"] is None
    assert block["delivery_type"]["actor_attack_eligible"] is None


def test_T38_historical_report_is_untouched(tmp_path):
    legacy = tmp_path / "gpt-4o-2024-08-06.baseline.qualification.json"
    legacy.write_text(json.dumps({"overall": "NOT_QUALIFIED"}), encoding="utf-8")
    digest = hashlib.sha256(legacy.read_bytes()).hexdigest()
    _persist(tmp_path, run_id="20260102T000000Z")
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == digest


def test_T43_probes_complete_false_is_rejected(tmp_path):
    common = _common()
    with pytest.raises(common.QualificationError, match="probes_complete"):
        _gate(tmp_path, _report_payload(probes_complete=False))


def test_T44_review_decision_is_bound_to_the_report(tmp_path):
    common = _common()
    report_payload = _report_payload()
    report_payload["review_triggers"]["delivery_type"].update(
        {"observed": 0.1, "triggered": True})
    report_path = _write(tmp_path, report_payload)
    block = common.require_qualification(_cfg(), report_path)

    with pytest.raises(common.QualificationError, match="review"):
        common.require_review_decision(block, None)

    decision = {
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
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    assert common.require_review_decision(block, path)["decision"] == (
        "accept_degradation")

    stale = dict(decision)
    stale["qualification_report_sha256"] = "0" * 64
    path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(common.QualificationError, match="does not belong"):
        common.require_review_decision(block, path)

    contradiction = dict(decision)
    contradiction["decision"] = "exclude_actor_attack"
    path.write_text(json.dumps(contradiction), encoding="utf-8")
    with pytest.raises(common.QualificationError, match="eligible"):
        common.require_review_decision(block, path)

    no_trigger = common.require_qualification(
        _cfg(), _write(tmp_path, _report_payload(run_id="20260102T000000Z")))
    path.write_text(json.dumps(decision), encoding="utf-8")
    with pytest.raises(common.QualificationError, match="no review trigger"):
        common.require_review_decision(no_trigger, path)


def test_T45_missing_probe_spec_is_rejected(tmp_path):
    common = _common()
    payload = _report_payload()
    payload.pop("probe_spec_version")
    with pytest.raises(common.QualificationError, match="probe_spec_version"):
        _gate(tmp_path, payload)


# --------------------------------------------------------------------------- #
# Parity: the optional diagnostics sink must not change the attack path
# --------------------------------------------------------------------------- #

def test_diagnostics_sink_does_not_change_get_actors_or_infer_single():
    from guardbound.attacks.actor_attack import get_actors, infer_single

    llm = ActorLLM()
    plain_actors, plain_messages = get_actors("target", 3, llm)
    sink: dict = {}
    actors, messages = get_actors("target", 3, llm, diagnostics=sink)
    assert actors == plain_actors
    assert messages == plain_messages
    assert sink["kept"] == 3
    assert sink["raw_actor_count"] == 3

    plain = infer_single("goal", 3, ActorLLM())
    sink = {}
    with_sink = infer_single("goal", 3, ActorLLM(), diagnostics=sink)
    assert with_sink == plain
    assert sink["raw_is_mapping"] is True
    assert sink["details_present"] is True


def test_gate_and_harness_share_the_mandatory_probe_names():
    q = _qual()
    common = _common()
    assert tuple(q.MANDATORY_PROBES) == common.MANDATORY_PROBE_NAMES
    assert q.SCHEMA_VERSION == common.QUALIFICATION_SCHEMA_VERSION
    assert q.PROBE_SPEC_VERSION == common.QUALIFICATION_PROBE_SPEC_VERSION


def test_config_declares_the_frozen_phase40b_parameters():
    import yaml

    q = _qual()
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs/cloud_qualification.yaml").read_text(
            encoding="utf-8"))
    assert cfg["qualification"]["actor_attack_probe_samples"] == (
        q.ACTOR_ATTACK_PROBE_SAMPLES)
    assert cfg["qualification"]["continuity_conversations"] == (
        q.CONTINUITY_CONVERSATIONS)
    assert cfg["qualification"]["delivery_type_review_threshold"] == (
        q.DELIVERY_TYPE_REVIEW_THRESHOLD)
    declared = cfg["thresholds"]["target"]["context_recall_rate"]["value"]
    assert declared == q.DEFAULT_THRESHOLDS["target"]["context_recall_rate"]["value"]
