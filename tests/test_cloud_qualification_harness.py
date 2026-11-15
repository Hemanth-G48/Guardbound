"""Qualification-harness tests — thresholds, decisions, gating, prompt fidelity.

The harness must never rubber-stamp a model. These tests pin that:

* the decision function passes, fails and reports insufficient data correctly;
* the thresholds in the shipped config are the ones the harness would apply;
* a run without explicit confirmation spends nothing;
* an offline-mock run exercises every probe and does NOT return QUALIFIED for a
  transport that cannot recall context or classify refusals;
* each attack's probe sends that attack's own official system prompt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from guardbound.llm.openai_client import OpenAIChatLLM
from tests.cloud_fakes import TEST_MODEL, FakeOpenAIBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _qual():
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import cloud_model_qualification  # noqa: PLC0415

    return cloud_model_qualification


# --------------------------------------------------------------------------- #
# Decision logic
# --------------------------------------------------------------------------- #

def test_decide_passes_when_every_metric_meets_its_threshold():
    q = _qual()
    thresholds = {
        "validity": {"op": ">=", "value": 0.9},
        "errors": {"op": "<=", "value": 0.1},
    }
    decision = q.decide("attacker", {"validity": 0.95, "errors": 0.0}, thresholds)
    assert decision["decision"] == q.QUALIFIED
    assert decision["failed_metrics"] == []


def test_decide_fails_on_a_single_missed_threshold():
    q = _qual()
    thresholds = {"validity": {"op": ">=", "value": 0.9}}
    decision = q.decide("attacker", {"validity": 0.5}, thresholds)
    assert decision["decision"] == q.NOT_QUALIFIED
    assert decision["failed_metrics"] == ["validity"]


def test_decide_treats_boundary_values_as_passing():
    q = _qual()
    assert q.decide("r", {"m": 0.9}, {"m": {"op": ">=", "value": 0.9}})["decision"] == q.QUALIFIED
    assert q.decide("r", {"m": 0.1}, {"m": {"op": "<=", "value": 0.1}})["decision"] == q.QUALIFIED


def test_decide_reports_insufficient_data_rather_than_passing():
    q = _qual()
    decision = q.decide("target", {}, {"validity": {"op": ">=", "value": 0.9}})
    assert decision["decision"] == q.INSUFFICIENT_DATA
    assert decision["unmeasured_metrics"] == ["validity"]
    assert decision["checks"] == []


def test_decide_ignores_unmeasured_metrics_when_others_are_measured():
    q = _qual()
    decision = q.decide(
        "target", {"a": 1.0}, {"a": {"op": ">=", "value": 0.5},
                               "b": {"op": ">=", "value": 0.5}})
    assert decision["decision"] == q.QUALIFIED
    assert decision["unmeasured_metrics"] == ["b"]


def test_overall_decision_is_the_worst_role():
    q = _qual()
    ok = {"decision": q.QUALIFIED}
    bad = {"decision": q.NOT_QUALIFIED}
    unknown = {"decision": q.INSUFFICIENT_DATA}
    assert q.overall_decision({"a": ok, "b": ok}) == q.QUALIFIED
    assert q.overall_decision({"a": ok, "b": bad}) == q.NOT_QUALIFIED
    assert q.overall_decision({"a": ok, "b": unknown}) == q.INSUFFICIENT_DATA
    assert q.overall_decision({"a": bad, "b": unknown}) == q.NOT_QUALIFIED


# --------------------------------------------------------------------------- #
# Thresholds declared in the config == thresholds the harness applies
# --------------------------------------------------------------------------- #

def test_shipped_thresholds_match_the_harness_defaults():
    q = _qual()
    cfg = q.load_config(PROJECT_ROOT / "configs/cloud_qualification.yaml")
    for role in q.ROLES:
        declared = (cfg.get("thresholds") or {}).get(role) or {}
        assert declared, f"{role}: no thresholds declared in the config"
        effective = q.thresholds_for(cfg, role)
        for name, spec in declared.items():
            assert effective[name] == spec, (
                f"{role}.{name}: config says {spec}, harness applies "
                f"{effective[name]}"
            )
        # The config must not silently drop a default metric.
        for name in q.DEFAULT_THRESHOLDS[role]:
            assert name in effective


def test_config_has_no_custom_budget_cap_and_a_separate_output_dir():
    """Phase 22 removed the $1.50 cap: the paper specifies no API budget."""
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs/cloud_qualification.yaml").read_text(encoding="utf-8"))
    assert "budget_usd" not in cfg["qualification"]
    assert "budget_usd" not in (cfg.get("cloud") or {})
    assert "qualification" in cfg["qualification"]["output_dir"]


def test_budget_flag_is_gone_from_the_qualification_cli():
    q = _qual()
    with pytest.raises(SystemExit):
        q.parse_args(["--budget-usd", "1.50"])
    assert not hasattr(q.parse_args([]), "budget_usd")


def test_offline_run_reports_usage_without_a_cap(tmp_path, passing_local_gate):
    """Usage is still measured and persisted — it just cannot block anything.

    The local NBF gate is mandatory (Phase 27), so this test supplies a passing
    gate and exercises the paid-probe path.
    """
    q = _qual()
    q.main(["--offline-mock", "--out-dir", str(tmp_path)])
    cfg = q.load_config(PROJECT_ROOT / "configs/cloud_qualification.yaml")
    model = cfg["models"]["attacker"]["model"]
    output = json.loads(
        _qualification_report(tmp_path, model).read_text(encoding="utf-8"))
    usage = output["usage"]
    assert "budget_usd" not in usage
    assert "remaining_usd" not in usage
    assert usage["estimated_cost_usd"] is not None
    assert usage["cost_basis"] == "rate_card"
    assert usage["usage_unavailable_calls"] == 0
    # Spend is persisted, not just printed.
    assert (tmp_path / "qualification.jsonl").exists()



# --------------------------------------------------------------------------- #
# Gating
# --------------------------------------------------------------------------- #

def test_refuses_without_confirmation(monkeypatch, capsys):
    q = _qual()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-a-credential")
    code = q.main(["--model", TEST_MODEL, "--out-dir", "unused"])
    assert code == 2
    assert "Refusing to send paid requests" in capsys.readouterr().err


def test_dry_run_lists_probes_and_thresholds_without_spending(capsys):
    q = _qual()
    code = q.main(["--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "thresholds (declared before the run)" in out
    assert "contract_validity_rate" in out
    assert "DRY RUN" in out


# --------------------------------------------------------------------------- #
# Offline end-to-end: the harness must not rubber-stamp
# --------------------------------------------------------------------------- #

def _qualification_report(tmp_path: Path, model: str, arm: str = "baseline") -> Path:
    """The single canonical v2 report for this model and condition."""
    matches = list(tmp_path.glob(f"{model}.{arm}.*.qualification.v2.json"))
    assert len(matches) == 1, matches
    return matches[0]


def test_offline_mock_run_executes_probes_and_does_not_pass(tmp_path, capsys,
                                                            passing_local_gate):
    """The mock cannot recall context or classify refusals, so it must fail."""
    q = _qual()
    code = q.main(["--offline-mock", "--out-dir", str(tmp_path)])
    model = q.load_config(
        PROJECT_ROOT / "configs/cloud_qualification.yaml")["models"]["attacker"]["model"]
    output = json.loads(_qualification_report(tmp_path, model).read_text(encoding="utf-8"))

    assert code == 1, "a mock that fails the criteria must not exit successfully"
    assert output["result_kind"].startswith("HARNESS VALIDATION ONLY")
    assert output["overall"] == q.NOT_QUALIFIED
    assert output["decisions"]["target"]["decision"] == q.NOT_QUALIFIED
    assert "context_recall_rate" in output["decisions"]["target"]["failed_metrics"]
    # Every documented probe ran.
    for probe in ("attacker_contract", "attacker_multi_turn",
                  "attacker_consistency", "attacker_refusal_judge", "target",
                  "evaluator_rubric", "evaluator_refusal",
                  "evaluator_disclaimer", "evaluator_consistency"):
        assert probe in output["probes"], probe
    # Attacker contract really is valid on the mock, so the attack role passes —
    # showing the decision is discriminating, not uniformly negative.
    assert output["decisions"]["attacker"]["decision"] == q.QUALIFIED
    # Claim separation is explicit.
    assert set(output["claim_separation"]) == {
        "api_connectivity", "protocol_compatibility", "research_role_suitability"}
    assert output["limitations"]


def test_offline_mock_records_model_identity(tmp_path, passing_local_gate):
    q = _qual()
    q.main(["--offline-mock", "--out-dir", str(tmp_path)])
    cfg = q.load_config(PROJECT_ROOT / "configs/cloud_qualification.yaml")
    model = cfg["models"]["attacker"]["model"]
    output = json.loads(
        _qualification_report(tmp_path, model).read_text(encoding="utf-8"))
    identity = output["model_identity"]
    assert identity["roles"]["evaluator"]["is_dated_snapshot"] is True
    assert identity["execution_date_utc"]


# --------------------------------------------------------------------------- #
# Prompt fidelity: each probe uses the attack's own official system prompt
# --------------------------------------------------------------------------- #

def _client(backend: FakeOpenAIBackend) -> OpenAIChatLLM:
    return OpenAIChatLLM(model_id=TEST_MODEL, client=backend.client, role="attacker")


def test_attacker_contract_probe_uses_all_three_official_system_prompts(monkeypatch):
    q = _qual()
    from guardbound.attacks import acronym, crescendo_paper, opposite_day
    from tests.cloud_fakes import completion

    payload = json.dumps(
        {"generatedQuestion": "What is the general background?",
         "lastResponseSummary": "summary"})
    backend = FakeOpenAIBackend([completion(payload)] * 64)
    monkeypatch.setattr(q.common, "OfflineMockClient", None, raising=False)

    result = q.probe_attacker_contract(_client(backend), maxima=8, samples_per_attack=1)

    sent_systems = {
        next(m["content"] for m in body["messages"] if m["role"] == "system")
        for body in backend.requests
    }
    assert crescendo_paper.CRESCENDO_SYSTEM_PROMPT.format(
        conversationObjective=q.BENIGN_GOALS[0]) in sent_systems
    assert opposite_day.OPPOSITE_DAY_SYSTEM_PROMPT.format(
        conversationObjective=q.BENIGN_GOALS[0]) in sent_systems
    assert acronym.ACRONYM_SYSTEM_PROMPT.format(
        conversationObjective=q.BENIGN_GOALS[0]) in sent_systems
    assert result["attack_coverage"] == 1.0
    assert result["contract_validity_rate"] == 1.0


def test_attacker_contract_probe_counts_malformed_replies(monkeypatch):
    q = _qual()
    from tests.cloud_fakes import completion

    backend = FakeOpenAIBackend([completion("Sure, here is a question...")] * 16)
    result = q.probe_attacker_contract(_client(backend), maxima=8, samples_per_attack=1)
    assert result["outcomes"]["non_json"] == 3
    assert result["contract_validity_rate"] == 0.0
    assert result["attack_coverage"] == 0.0


def test_disclaimer_probe_uses_the_official_disclaimer_prompt():
    q = _qual()
    from guardbound.attacks.acronym import CHECK_DISCLAIMER_SYSTEM_PROMPT
    from tests.cloud_fakes import completion

    backend = FakeOpenAIBackend(
        [completion(json.dumps({"value": True, "rationale": "x"}))] * 8)
    q.probe_evaluator_disclaimer(_client(backend))
    systems = {
        next(m["content"] for m in body["messages"] if m["role"] == "system")
        for body in backend.requests
    }
    assert CHECK_DISCLAIMER_SYSTEM_PROMPT in systems


def test_refusal_probe_uses_the_official_refusal_prompt():
    q = _qual()
    from guardbound.evaluation.refusal_checker import REFUSAL_CHECK_SYSTEM_PROMPT
    from tests.cloud_fakes import completion

    backend = FakeOpenAIBackend(
        [completion(json.dumps({"value": True, "metadata": 99}))] * 8)
    q.probe_evaluator_refusal(_client(backend))
    systems = {
        next(m["content"] for m in body["messages"] if m["role"] == "system")
        for body in backend.requests
    }
    assert REFUSAL_CHECK_SYSTEM_PROMPT in systems


def test_evaluator_probes_request_temperature_zero():
    q = _qual()
    from tests.cloud_fakes import completion

    backend = FakeOpenAIBackend([completion(json.dumps({"score": 1}))] * 8)
    q.probe_evaluator_rubric(_client(backend), repeats=3)
    assert all(body["temperature"] == 0.0 for body in backend.requests)


def test_target_probe_uses_the_dataset_system_prompt():
    q = _qual()
    from tests.cloud_fakes import completion

    backend = FakeOpenAIBackend([completion("An answer.")] * 12)
    case = {"task": "Explain photosynthesis.",
            "target_system": "You are a careful scientific assistant."}
    q.probe_target(_client(backend), [case], repeats=1,
                   system_prompt="You are a helpful assistant.")
    systems = [
        m["content"] for body in backend.requests
        for m in body["messages"] if m["role"] == "system"
    ]
    assert "You are a careful scientific assistant." in systems


def test_rubric_probe_reports_agreement_with_the_published_meaning():
    q = _qual()
    from tests.cloud_fakes import completion

    backend = FakeOpenAIBackend([completion(json.dumps({"score": 1}))] * 8)
    result = q.probe_evaluator_rubric(_client(backend), repeats=4)
    assert result["rubric_validity_rate"] == 1.0
    assert result["rubric_determinism_rate"] == 1.0
    assert result["rubric_agreement"] == 1.0
    assert result["structural_violation_rate"] == 0.0


def test_rubric_probe_does_not_accept_an_out_of_range_score():
    q = _qual()
    from tests.cloud_fakes import completion

    backend = FakeOpenAIBackend([completion(json.dumps({"score": 9}))] * 8)
    result = q.probe_evaluator_rubric(_client(backend), repeats=2)
    assert result["rubric_validity_rate"] == 0.0
    assert result["structural_violations"] == 2
    assert result["rubric_agreement"] is None
