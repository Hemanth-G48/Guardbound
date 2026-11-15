"""Phase 38 pre-execution hardening tests.

  * P2-1 — per-attack ASR table and the labelled overall micro-average.
  * P2-6 — fail-closed HarmBench row-count / row-shape validation.
  * P2-4 — structured-output mode is part of the qualification identity.
  * P2-3 — the execution runbook exists and matches the R1 configuration.

Offline only: synthetic JSON fixtures, a pure rendering function and the real
config file. No network, no client construction, no model inference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import _cloud_common as common  # noqa: E402
import cloud_pilot  # noqa: E402

R1_CONFIG = "configs/paper_reproduction_r1_gpt4o.yaml"
RUNBOOK = Path("docs/DELIVERABLE_1_EXECUTION_RUNBOOK.md")


# --------------------------------------------------------------------------- #
# P2-1 — ASR rendering
# --------------------------------------------------------------------------- #

_PER_ATTACK = {
    "actor_attack": (40, 195),
    "crescendo_paper": (50, 200),
    "opposite_day": (30, 185),
}


def _asr_fixture():
    """Build the aggregate through the real aggregator, not a hand-made dict."""
    rows = []
    for attack, (successful, valid) in _PER_ATTACK.items():
        for index in range(200):
            rows.append({
                "attack": attack,
                "failed": False,
                "valid_for_asr": index < valid,
                "success": index < successful,
            })
    return common.aggregate_asr_report(
        rows, {name: 200 for name in _PER_ATTACK})


def _row_line(rendered: str, attack: str) -> list[str]:
    line = next(l for l in rendered.splitlines() if l.strip().startswith(attack))
    return line.split()


def test_asr_report_lists_all_three_attacks():
    rendered = cloud_pilot.render_asr_report(_asr_fixture(), list(_PER_ATTACK))
    for attack in _PER_ATTACK:
        assert attack in rendered


def test_each_attack_shows_its_own_200_denominator():
    rendered = cloud_pilot.render_asr_report(_asr_fixture(), list(_PER_ATTACK))
    for attack in _PER_ATTACK:
        tokens = _row_line(rendered, attack)
        assert tokens[2] == "200", tokens


def test_primary_asr_uses_the_per_attack_denominator():
    asr = _asr_fixture()
    rendered = cloud_pilot.render_asr_report(asr, list(_PER_ATTACK))
    expected = {
        "actor_attack": "0.2000",
        "crescendo_paper": "0.2500",
        "opposite_day": "0.1500",
    }
    for attack, value in expected.items():
        tokens = _row_line(rendered, attack)
        assert tokens[3] == value, tokens
        assert tokens[3] == f"{asr['by_attack'][attack]['asr_paper']:.4f}"


def test_diagnostic_asr_is_reported_separately():
    asr = _asr_fixture()
    rendered = cloud_pilot.render_asr_report(asr, list(_PER_ATTACK))
    tokens = _row_line(rendered, "actor_attack")
    assert tokens[4] == "195"
    assert tokens[5] == "0.2051"          # 40 / 195
    assert tokens[5] != tokens[3]          # not the primary rate
    assert tokens[5] == f"{asr['by_attack']['actor_attack']['asr_valid_completed']:.4f}"


def test_overall_is_labelled_micro_average_with_600_pairs():
    rendered = cloud_pilot.render_asr_report(_asr_fixture(), list(_PER_ATTACK))
    lowered = rendered.lower()
    assert "overall micro-average" in lowered
    assert "600 expected attack-behaviour pairs" in rendered
    assert "primary ASR     0.2000 (120/600)" in rendered
    assert "diagnostic ASR  0.2069 (120/580)" in rendered


def test_zero_diagnostic_denominator_renders_not_available():
    rows = [
        {"attack": a, "failed": True, "valid_for_asr": False, "success": False}
        for a in _PER_ATTACK for _ in range(200)
    ]
    asr = common.aggregate_asr_report(rows, {a: 200 for a in _PER_ATTACK})
    assert asr["overall"]["asr_valid_completed"] is None
    rendered = cloud_pilot.render_asr_report(asr, list(_PER_ATTACK))
    assert "diagnostic ASR  n/a (0/0)" in rendered
    assert "0.0000 (0/0)" not in rendered


def test_renderer_does_not_change_the_aggregation_values():
    asr = _asr_fixture()
    before = json.loads(json.dumps(asr))
    cloud_pilot.render_asr_report(asr, list(_PER_ATTACK))
    assert asr == before
    assert asr["overall"] == {
        "expected_goals": 600,
        "completed_goals": 600,
        "valid_completed_goals": 580,
        "successful_goals": 120,
        "infrastructure_failure_count": 0,
        # 20 of the 600 rows completed with no usable rubric score.
        "invalid_evaluator_count": 20,
        "asr_paper": 120 / 600,
        "asr_valid_completed": 120 / 580,
    }


def test_renderer_reads_only_keys_the_aggregator_provides():
    asr = _asr_fixture()
    for attack in _PER_ATTACK:
        assert {"successful_goals", "expected_goals", "valid_completed_goals",
                "asr_paper", "asr_valid_completed"} <= set(asr["by_attack"][attack])
    assert {"successful_goals", "expected_goals",
            "valid_completed_goals"} <= set(asr["overall"])


# --------------------------------------------------------------------------- #
# P2-6 — dataset row-count validation
# --------------------------------------------------------------------------- #

def _dataset(tmp_path: Path, rows, name: str = "tasks.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _goal_rows(count: int) -> list[dict]:
    return [
        {"task": f"goal {i}", "target_system": "You are a helpful assistant.",
         "max_rounds": 8}
        for i in range(count)
    ]


def _cfg(path: Path, num_goals):
    dataset = {"path": str(path)}
    if num_goals is not None:
        dataset["num_goals"] = num_goals
    return {"dataset": dataset}


def test_exactly_200_rows_is_accepted(tmp_path):
    goals = common.load_goals(_cfg(_dataset(tmp_path, _goal_rows(200)), 200))
    assert len(goals) == 200
    assert goals[0]["task"] == "goal 0"
    assert goals[-1]["task"] == "goal 199"


def test_more_than_200_rows_keeps_the_deterministic_prefix(tmp_path):
    goals = common.load_goals(_cfg(_dataset(tmp_path, _goal_rows(250)), 200))
    assert len(goals) == 200
    assert [g["task"] for g in goals[:3]] == ["goal 0", "goal 1", "goal 2"]


def test_199_rows_is_rejected(tmp_path):
    path = _dataset(tmp_path, _goal_rows(199))
    with pytest.raises(ValueError, match="dataset contains 199 behaviours"):
        common.load_goals(_cfg(path, 200))


def test_empty_dataset_is_rejected(tmp_path):
    path = _dataset(tmp_path, [])
    with pytest.raises(ValueError, match="dataset contains 0 behaviours"):
        common.load_goals(_cfg(path, 200))


def test_empty_dataset_without_an_explicit_count_is_rejected(tmp_path):
    path = _dataset(tmp_path, [])
    with pytest.raises(ValueError, match="num_goals must be >= 1"):
        common.load_goals(_cfg(path, None))


def test_invalid_row_shape_is_rejected(tmp_path):
    rows = _goal_rows(1) + ["not a mapping"]
    path = _dataset(tmp_path, rows)
    with pytest.raises(ValueError, match="selected row 1"):
        common.load_goals(_cfg(path, 2))


def test_row_without_a_goal_field_is_rejected(tmp_path):
    path = _dataset(tmp_path, [{"target_system": "S", "max_rounds": 8}])
    with pytest.raises(ValueError, match="non-empty 'task' or 'goal'"):
        common.load_goals(_cfg(path, 1))


def test_zero_requested_count_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="num_goals must be >= 1"):
        common.load_goals(_cfg(_dataset(tmp_path, _goal_rows(5)), 0))


def test_negative_requested_count_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="num_goals must be >= 1"):
        common.load_goals(_cfg(_dataset(tmp_path, _goal_rows(5)), -3))


def test_negative_limit_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="--limit must be >= 0"):
        common.load_goals(_cfg(_dataset(tmp_path, _goal_rows(200)), 200), limit=-1)


def test_limit_selects_a_prefix_without_weakening_the_count_check(tmp_path):
    goals = common.load_goals(_cfg(_dataset(tmp_path, _goal_rows(200)), 200), limit=5)
    assert len(goals) == 5
    assert goals[-1]["task"] == "goal 4"


def test_dataset_failure_precedes_client_construction(monkeypatch, tmp_path, capsys):
    called = {"build_role_llms": False}

    def _short(_cfg, limit=None):
        raise ValueError("dataset is short")

    def _record(*_a, **_k):
        called["build_role_llms"] = True

    monkeypatch.setattr(common, "load_goals", _short)
    monkeypatch.setattr(common, "build_role_llms", _record)

    args = ["--config", R1_CONFIG, "--variant", "r1_baseline", "--limit", "1",
            "--dry-run", "--out-dir", str(tmp_path / "out")]
    with pytest.raises(ValueError, match="dataset is short"):
        cloud_pilot.main(args)
    assert called["build_role_llms"] is False


def test_real_r1_dataset_still_loads_200_goals():
    cfg = common.load_config(R1_CONFIG)
    goals = common.load_goals(cfg)
    assert len(goals) == 200
    assert all(isinstance(g, dict) and g.get("task") for g in goals)


# --------------------------------------------------------------------------- #
# P2-4 — structured-output binding
# --------------------------------------------------------------------------- #

def _identity_cfg():
    return {
        "temperature": {"attacker": 0.7, "target": 0.7, "evaluator": 0.7},
        "cloud": {"structured_output": "json_object"},
        "models": {
            "attacker": {"provider": "openai", "model": "gpt-4o-2024-08-06"},
            "target": {"provider": "openai", "model": "gpt-4o-2024-08-06"},
            "evaluator": {"provider": "openai", "model": "gpt-4o-2024-08-06"},
        },
        "prompt_variant": "official",
        "attacker_prompt_condition": "baseline",
    }


def _payload(structured_output="json_object"):
    r1_config = Path(__file__).resolve().parents[1] / (
        "configs/paper_reproduction_r1_gpt4o.yaml")
    return {
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
            "structured_output": structured_output,
        },
    }


def _write(tmp_path: Path, payload) -> Path:
    run_id = payload.get("run_id") or "20260101T000000Z"
    path = tmp_path / (
        f"gpt-4o-2024-08-06.baseline.{run_id}.qualification.v2.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_identity_records_the_structured_output_mode():
    identity = common.expected_qualification_identity(_identity_cfg())
    assert identity["structured_output"] == "json_object"


def test_gate_accepts_a_matching_structured_output(tmp_path):
    block = common.require_qualification(
        _identity_cfg(), _write(tmp_path, _payload("json_object")))
    assert block["overall"] == "QUALIFIED"


def test_gate_rejects_a_different_structured_output(tmp_path):
    path = _write(tmp_path, _payload("json_schema"))
    with pytest.raises(common.QualificationError, match="structured_output"):
        common.require_qualification(_identity_cfg(), path)


def test_gate_rejects_a_report_that_omits_the_structured_output(tmp_path):
    path = _write(tmp_path, _payload(None))
    with pytest.raises(common.QualificationError, match="structured_output"):
        common.require_qualification(_identity_cfg(), path)


# --------------------------------------------------------------------------- #
# P2-3 — runbook consistency
# --------------------------------------------------------------------------- #

def _runbook_text() -> str:
    assert RUNBOOK.exists(), f"missing {RUNBOOK}"
    return RUNBOOK.read_text(encoding="utf-8")


def test_runbook_documents_the_exact_qualification_command():
    text = _runbook_text()
    assert "python scripts/cloud_model_qualification.py" in text
    assert "--config configs/paper_reproduction_r1_gpt4o.yaml" in text
    assert "--confirm-paid-request" in text


def test_runbook_documents_the_expected_report_and_verdict():
    text = _runbook_text()
    assert "<tag>.<arm>.<run_id>.qualification.v2.json" in text
    assert "schema_version" in text and "probe_spec_version" in text
    assert "legacy v1" in text.lower() or "v1 report" in text.lower()
    assert "--review-decision" in text
    assert "QUALIFIED" in text
    assert "probes_complete" in text
    assert "structured_output" in text


def test_runbook_marks_qualification_as_paid_and_separately_authorized():
    text = _runbook_text()
    lowered = text.lower()
    assert "paid operation" in lowered
    assert "separately" in lowered
    assert "does not authorize" in lowered or "not an authorization" in lowered


def test_runbook_distinguishes_the_four_execution_modes():
    text = _runbook_text()
    for token in ("--offline-mock", "--dry-run", "qualification", "paid"):
        assert token in text
    assert "HARNESS VALIDATION ONLY" in text or "harness validation only" in text.lower()


def test_runbook_documents_the_asr_denominators():
    text = _runbook_text()
    assert "200 per attack" in text
    assert "600" in text
    assert "micro-average" in text


def test_runbook_matches_the_r1_configuration():
    cfg = common.load_config(R1_CONFIG)
    text = _runbook_text()

    for role in ("attacker", "target", "evaluator"):
        assert cfg["models"][role]["model"] == "gpt-4o-2024-08-06"
        assert float(cfg["temperature"][role]) == 0.7
    assert cfg["prompt_variant"] == "official"
    assert cfg["attacker_prompt_condition"] == "baseline"
    assert cfg["cloud"]["structured_output"] == "json_object"
    assert cfg["dataset"]["num_goals"] == 200
    assert cfg["attacks"]["enabled"] == [
        "actor_attack", "crescendo_paper", "opposite_day"]
    assert cfg["attacks"]["training_only"] == ["acronym"]
    assert cfg["nbf"]["checkpoint_sha256_expected"] == (
        "cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe")
    assert cfg["nbf"]["steer_target"] is False
    assert float(cfg["nbf"]["eta"]) == 0.001

    # The runbook's headline values must agree with the config it describes.
    assert "gpt-4o-2024-08-06" in text
    assert "cea1a75b" in text
