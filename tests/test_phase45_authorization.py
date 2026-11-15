"""Phase 45 — authorization boundary and trigger hardening (offline only).

This module is deliberately excluded from the autouse R1 test-context fixture in
``tests/conftest.py``; it must observe the fail-closed behavior itself.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.cloud_fakes import FakeOpenAIBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(PROJECT_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
R1_CONFIG = PROJECT_ROOT / "configs" / "paper_reproduction_r1_gpt4o.yaml"
R1_MODEL = "gpt-4o-2024-08-06"


def _common():
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import _cloud_common  # noqa: PLC0415

    return _cloud_common


def _args(**overrides):
    base = {"qualification_report": None, "review_decision": None}
    base.update(overrides)
    return SimpleNamespace(**base)


def _missing(tmp_path):
    return tmp_path / "missing.qualification.v2.json"


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
        "_source_path": str(R1_CONFIG),
        "_source_sha256": _common().sha256_of(R1_CONFIG),
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


def _role_cfg(backend=None):
    return {
        "provider": "openai",
        "model": R1_MODEL,
        "client": (backend or FakeOpenAIBackend()).client,
    }


# --------------------------------------------------------------------------- #
# P0-1 — the old caller-controlled exemption is gone
# --------------------------------------------------------------------------- #

def test_old_purpose_string_is_rejected_everywhere(tmp_path):
    common = _common()
    with pytest.raises(TypeError):
        common.require_paid_authorization(
            R1_MODEL, qualification_report=_missing(tmp_path),
            purpose="qualification")
    with pytest.raises(TypeError):
        common.build_single_role_llm(
            "target", _role_cfg(), object(), purpose="qualification")
    with pytest.raises(TypeError):
        common.build_role_llms(
            _cfg(), object(), purpose="qualification")


def test_qualification_builder_requires_an_explicit_capability():
    common = _common()
    with pytest.raises(TypeError):
        common._build_qualification_role_llm(
            "target", _role_cfg(), object())


# --------------------------------------------------------------------------- #
# P0-2 — the client layer fails closed
# --------------------------------------------------------------------------- #

def test_direct_r1_constructor_requires_context():
    from guardbound.llm.openai_client import OpenAIChatLLM
    from guardbound.llm.paid_authorization import PaidAuthorizationRequired

    with pytest.raises(PaidAuthorizationRequired):
        OpenAIChatLLM(R1_MODEL, client=FakeOpenAIBackend().client)


def test_direct_r1_constructor_works_inside_context():
    from guardbound.llm.openai_client import OpenAIChatLLM
    from guardbound.llm.paid_authorization import _authorized

    with _authorized(R1_MODEL):
        llm = OpenAIChatLLM(R1_MODEL, client=FakeOpenAIBackend().client)
    assert llm.model_id == R1_MODEL


def test_public_provider_factory_requires_context():
    from guardbound.llm.paid_authorization import (
        PaidAuthorizationRequired,
        _authorized,
    )
    from guardbound.llm.provider_factory import build_cloud_role_llm

    with pytest.raises(PaidAuthorizationRequired):
        build_cloud_role_llm("target", _role_cfg())
    with _authorized(R1_MODEL):
        llm = build_cloud_role_llm("target", _role_cfg())
    assert llm.model_id == R1_MODEL


def test_public_three_model_stack_requires_context():
    from guardbound.llm.paid_authorization import (
        PaidAuthorizationRequired,
        _authorized,
    )
    from guardbound.llm.provider_factory import build_three_model_stack

    models = {
        role: _role_cfg() for role in ("attacker", "target", "evaluator")
    }
    with pytest.raises(PaidAuthorizationRequired):
        build_three_model_stack(models)
    with _authorized(R1_MODEL):
        manager, llms = build_three_model_stack(models)
    assert set(llms) == {"attacker", "target", "evaluator"}


def test_public_make_chat_llm_requires_context():
    from guardbound.llm import make_chat_llm
    from guardbound.llm.paid_authorization import (
        PaidAuthorizationRequired,
        _authorized,
    )

    with pytest.raises(PaidAuthorizationRequired):
        make_chat_llm("openai", R1_MODEL, client=FakeOpenAIBackend().client)
    with _authorized(R1_MODEL):
        llm = make_chat_llm(
            "openai", R1_MODEL, client=FakeOpenAIBackend().client)
    assert llm.model_id == R1_MODEL


def test_gated_helpers_construct_only_with_a_valid_report(tmp_path):
    common = _common()
    with pytest.raises(common.QualificationError):
        common.build_paid_chat_llm(
            R1_MODEL, args=_args(qualification_report=_missing(tmp_path)),
            model_kwargs={"client": FakeOpenAIBackend().client})
    report = _write(tmp_path, _report_payload())
    llm = common.build_paid_chat_llm(
        R1_MODEL, args=_args(qualification_report=report),
        model_kwargs={"client": FakeOpenAIBackend().client})
    assert llm.model_id == R1_MODEL


def test_build_single_role_and_stack_are_gated(tmp_path):
    common = _common()
    role_cfg = _role_cfg()
    with pytest.raises(common.QualificationError):
        common.build_single_role_llm(
            "target", role_cfg, object(),
            qualification_report=_missing(tmp_path))
    report = _write(tmp_path, _report_payload())
    llm = common.build_single_role_llm(
        "target", role_cfg, object(), qualification_report=report)
    assert llm.model_id == R1_MODEL

    cfg = _cfg()
    for role in cfg["models"]:
        cfg["models"][role]["client"] = FakeOpenAIBackend().client
    with pytest.raises(common.QualificationError):
        common.build_role_llms(
            cfg, object(), qualification_report=_missing(tmp_path))
    manager, llms = common.build_role_llms(
        cfg, object(), qualification_report=report)
    assert set(llms) == {"attacker", "target", "evaluator"}


def test_non_r1_path_is_not_gated(tmp_path):
    common = _common()
    llm = common.build_paid_chat_llm(
        "gpt-4o-mini",
        args=_args(qualification_report=_missing(tmp_path)),
        model_kwargs={"client": FakeOpenAIBackend().client})
    assert llm.model_id == "gpt-4o-mini"
    assert common.is_r1_paid_model("gpt-4o") is False


def test_offline_mock_stack_needs_no_report():
    common = _common()
    cfg = _cfg()
    for role in cfg["models"]:
        cfg["models"][role] = {
            "provider": "openai",
            "model": R1_MODEL,
            "client": common.OfflineMockClient(),
        }
    manager, llms = common.build_role_llms(cfg, object())
    assert set(llms) == {"attacker", "target", "evaluator"}


# --------------------------------------------------------------------------- #
# P0-2 — final_analysis and run_paper_experiment
# --------------------------------------------------------------------------- #

def test_final_analysis_judge_is_gated(tmp_path):
    import final_analysis

    with pytest.raises(_common().QualificationError):
        final_analysis.make_judge(
            R1_MODEL, mock=False,
            args=_args(qualification_report=_missing(tmp_path)))
    assert final_analysis.make_judge(R1_MODEL, mock=True) is not None


def test_run_paper_experiment_forwards_auth_paths_and_mocks_dry_run():
    import run_paper_experiment

    args = SimpleNamespace(
        dry_run=False, target=R1_MODEL,
        qualification_report="report.json", review_decision="decision.json")
    namespace = run_paper_experiment.target_llm_namespace(args)
    assert namespace.mock is False
    assert namespace.target == R1_MODEL
    assert namespace.qualification_report == "report.json"
    assert namespace.review_decision == "decision.json"

    args.dry_run = True
    assert run_paper_experiment.target_llm_namespace(args).mock is True


def test_smoke_test_dry_run_does_not_gate_or_construct(monkeypatch):
    import cloud_smoke_test

    monkeypatch.setattr(cloud_smoke_test, "preflight", lambda args: object())
    assert cloud_smoke_test.main(["--dry-run", "--model", R1_MODEL]) == 0


# --------------------------------------------------------------------------- #
# P1-1 — QUALIFIED reports require completed trigger fields
# --------------------------------------------------------------------------- #

def _gate(tmp_path, **changes):
    common = _common()
    payload = _report_payload()
    delivery = payload["review_triggers"]["delivery_type"]
    delivery.update(changes)
    return common.require_qualification(
        _cfg(), _write(tmp_path, payload))


def test_qualified_report_rejects_null_observed(tmp_path):
    with pytest.raises(_common().QualificationError, match="observed"):
        _gate(tmp_path, observed=None, triggered=None)


def test_qualified_report_rejects_null_triggered(tmp_path):
    with pytest.raises(_common().QualificationError, match="triggered"):
        _gate(tmp_path, observed=0.9, triggered=None)


def test_qualified_report_rejects_boolean_observed(tmp_path):
    with pytest.raises(_common().QualificationError, match="observed"):
        _gate(tmp_path, observed=True, triggered=False)


# --------------------------------------------------------------------------- #
# P1-2 — frozen 0.50 threshold
# --------------------------------------------------------------------------- #

def test_threshold_boundaries_are_frozen(tmp_path):
    block = _gate(tmp_path, threshold=0.5, observed=0.49, triggered=True)
    assert block["overall"] == "QUALIFIED"
    block = _gate(tmp_path, threshold=0.5, observed=0.5, triggered=False)
    assert block["overall"] == "QUALIFIED"


@pytest.mark.parametrize("threshold", [
    0.01, 0.51, 0.0, 1.0, True, "0.5", float("nan"), float("inf"),
])
def test_non_frozen_threshold_is_rejected(tmp_path, threshold):
    with pytest.raises(_common().QualificationError, match="threshold"):
        _gate(tmp_path, threshold=threshold)


def test_threshold_0_01_cannot_suppress_the_review(tmp_path):
    with pytest.raises(_common().QualificationError, match="threshold"):
        _gate(tmp_path, threshold=0.01, observed=0.1, triggered=False)
