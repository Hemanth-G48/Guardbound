"""Phase 51 — effective runtime model identity binding (offline only).

Every scenario uses a synthetic v2 report and, where needed, a synthetic
review decision, plus a fake SDK client. No key, live client, or network call.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.cloud_fakes import FakeOpenAIBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(PROJECT_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import _cloud_common as common  # noqa: E402
import run_reproduction  # noqa: E402
from guardbound.llm.openai_client import OpenAIChatLLM  # noqa: E402

R1 = "gpt-4o-2024-08-06"
NON_R1 = "gpt-4o-mini"
ALIAS = "gpt-4o"
DUMMY_KEY = "phase51-dummy-not-a-credential"
RUN_ID = "20260101T000000Z"
TRIGGERED_RUN_ID = "20260102T000000Z"
ROLES = ("attacker", "target", "evaluator")


# --------------------------------------------------------------------------- #
# Fixtures: config, report, decision, fake builder
# --------------------------------------------------------------------------- #

def _write_cfg(
    tmp_path: Path,
    *,
    target: str = R1,
    attacker: str = R1,
    evaluator: str = R1,
    paper: bool = True,
    use_models: bool = False,
    omit: str | None = None,
    name: str = "phase51",
) -> tuple[dict, Path]:
    data = {
        "temperature": {role: 0.7 for role in ROLES},
        "prompt_variant": "official",
        "attacker_prompt_condition": "baseline",
        "cloud": {"structured_output": "json_object"},
    }
    if paper:
        data["paper_reproduction"] = True
    if use_models:
        data["models"] = {
            "attacker": {"provider": "openai", "model": attacker},
            "target": {"provider": "openai", "model": target},
            "evaluator": {"provider": "openai", "model": evaluator},
        }
    else:
        for role, model in (("attacker", attacker), ("target", target)):
            if role == omit:
                continue
            data[role] = {"model": model}
        if omit != "evaluator":
            data["evaluator"] = {
                "rubric_model": evaluator,
                "refusal_model": evaluator,
                "disclaimer_model": evaluator,
                "actor_step_judge_model": evaluator,
                "temperature": 0.7,
            }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = run_reproduction.load_config(str(path))
    return cfg, path


def _report(
    tmp_path: Path,
    cfg: dict,
    config_path: Path,
    *,
    triggered: bool = False,
    run_id: str = RUN_ID,
    name: str | None = None,
) -> Path:
    payload = {
        "schema_version": "qualification.v2",
        "probe_spec_version": "r1-probe.v2",
        "run_id": run_id,
        "overall": "QUALIFIED",
        "probes_complete": True,
        "attacker_prompt_condition": "baseline",
        "experiment_config": {
            "path": str(config_path),
            "sha256": cfg["_source_sha256"],
        },
        "configuration_under_qualification": {
            "models": {role: R1 for role in ROLES},
            "prompt_variant": "official",
            "temperatures": {role: 0.7 for role in ROLES},
            "structured_output": "json_object",
        },
        "probe_status": {
            probe: {"status": "PASS"} for probe in common.MANDATORY_PROBE_NAMES
        },
        "review_triggers": {
            "delivery_type": {
                "threshold": 0.50,
                "observed": 0.10 if triggered else 0.90,
                "triggered": bool(triggered),
                "operator_decision": None,
                "rationale": None,
                "actor_attack_eligible": None,
            }
        },
    }
    path = tmp_path / (
        name or f"{R1}.baseline.{run_id}.qualification.v2.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _decision(tmp_path: Path, block: dict) -> Path:
    payload = {
        "qualification_report_sha256": block["sha256"],
        "run_id": block["run_id"],
        "configuration_sha256": block["experiment_config"]["sha256"],
        "arm": block["attacker_prompt_condition"],
        "trigger": "delivery_type",
        "decision": "accept_degradation",
        "rationale": "phase51 scenario H: valid R1 decision",
        "decided_by": "phase51-test",
        "decided_at": "2026-11-14T00:00:00Z",
        "actor_attack_eligible": True,
    }
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _args(report=None, decision=None, *, target=None, attacker=None):
    return SimpleNamespace(
        qualification_report=str(report) if report else None,
        review_decision=str(decision) if decision else None,
        target=target,
        attacker=attacker,
    )


class _Builder:
    """Calls the real gate/builder but injects a fake SDK client."""

    def __init__(self):
        self.real = common.build_paid_chat_llm
        self.backend = FakeOpenAIBackend()
        self.calls: list[dict] = []

    def __call__(self, model, *, args=None, cfg=None, role=None, model_kwargs=None):
        self.calls.append({"model": model, "cfg": cfg, "role": role})
        return self.real(
            model, args=args, cfg=cfg, role=role,
            model_kwargs={"client": self.backend.client})


def _spy(monkeypatch, target, name):
    calls = []

    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError(f"{name} must not be reached")

    monkeypatch.setattr(target, name, fail)
    return calls


# --------------------------------------------------------------------------- #
# A — exact frozen R1 single-model configuration
# --------------------------------------------------------------------------- #

def test_scenario_a_exact_r1_passes_and_factory_gets_r1(tmp_path, monkeypatch):
    cfg, path = _write_cfg(tmp_path, target=R1, attacker=R1, evaluator=R1)
    report = _report(tmp_path, cfg, path)
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)

    result = run_reproduction.make_llm(
        cfg, "target", _args(report), mock=False)

    assert isinstance(result, OpenAIChatLLM)
    assert result.model_id == R1
    assert builder.calls == [{"model": R1, "cfg": cfg, "role": "target"}]


def test_scenario_a_non_triggered_report_needs_no_decision(tmp_path, monkeypatch):
    cfg, path = _write_cfg(tmp_path)
    report = _report(tmp_path, cfg, path, triggered=False)
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)

    result = run_reproduction.make_llm(
        cfg, "target", _args(report), mock=False)

    assert result.model_id == R1
    assert builder.calls[0]["role"] == "target"


# --------------------------------------------------------------------------- #
# B/C/D — each effective role must match the R1 report
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("role", "target", "attacker", "evaluator"),
    [
        ("target", NON_R1, R1, R1),
        ("attacker", R1, NON_R1, R1),
        ("evaluator", R1, R1, NON_R1),
    ],
    ids=["B-target", "C-attacker", "D-evaluator"],
)
def test_scenario_b_c_d_paper_non_r1_role_is_rejected(
    tmp_path, monkeypatch, role, target, attacker, evaluator,
):
    cfg, path = _write_cfg(
        tmp_path, target=target, attacker=attacker, evaluator=evaluator)
    report = _report(tmp_path, cfg, path)
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)

    with pytest.raises(common.QualificationError, match="stale"):
        run_reproduction.make_llm(cfg, role, _args(report), mock=False)

    assert builder.calls == []


# --------------------------------------------------------------------------- #
# E/F — CLI overrides enter the effective identity
# --------------------------------------------------------------------------- #

def test_scenario_e_target_override_is_rejected(tmp_path, monkeypatch):
    cfg, path = _write_cfg(tmp_path, target=R1)
    report = _report(tmp_path, cfg, path)
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)

    with pytest.raises(common.QualificationError, match="stale"):
        run_reproduction.make_llm(
            cfg, "target", _args(report, target=NON_R1), mock=False)

    assert builder.calls == []


def test_scenario_f_attacker_override_is_rejected(tmp_path, monkeypatch):
    cfg, path = _write_cfg(tmp_path, attacker=R1)
    report = _report(tmp_path, cfg, path)
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)

    with pytest.raises(common.QualificationError, match="stale"):
        run_reproduction.make_llm(
            cfg, "attacker", _args(report, attacker=NON_R1), mock=False)

    assert builder.calls == []


# --------------------------------------------------------------------------- #
# G — alias must not inherit R1 authorization
# --------------------------------------------------------------------------- #

def test_scenario_g_alias_is_rejected(tmp_path, monkeypatch):
    cfg, path = _write_cfg(tmp_path, target=ALIAS)
    report = _report(tmp_path, cfg, path)
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)

    with pytest.raises(common.QualificationError, match="stale"):
        run_reproduction.make_llm(
            cfg, "target", _args(report), mock=False)

    assert builder.calls == []


# --------------------------------------------------------------------------- #
# H — a valid R1-bound decision cannot authorize a non-R1 target
# --------------------------------------------------------------------------- #

def test_scenario_h_r1_decision_cannot_authorize_non_r1_target(
    tmp_path, monkeypatch,
):
    cfg, path = _write_cfg(tmp_path, target=NON_R1)
    report = _report(
        tmp_path, cfg, path, triggered=True, run_id=TRIGGERED_RUN_ID)

    # The report and decision are valid for the R1 role identity (same source
    # file hash); this is the "correctly bound R1 decision" in the scenario.
    validation_cfg = dict(cfg)
    validation_cfg["attacker"] = {"model": R1}
    validation_cfg["target"] = {"model": R1}
    validation_cfg["evaluator"] = {
        "rubric_model": R1,
        "refusal_model": R1,
        "disclaimer_model": R1,
        "actor_step_judge_model": R1,
        "temperature": 0.7,
    }
    block = common.require_qualification(validation_cfg, report)
    decision = _decision(tmp_path, block)
    assert common.require_review_decision(block, decision)["trigger"] == (
        "delivery_type")

    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)
    with pytest.raises(common.QualificationError, match="stale"):
        run_reproduction.make_llm(
            cfg, "target", _args(report, decision), mock=False)
    assert builder.calls == []


def test_scenario_h_triggered_report_without_decision_is_rejected(
    tmp_path, monkeypatch,
):
    cfg, path = _write_cfg(tmp_path, target=R1)
    report = _report(
        tmp_path, cfg, path, triggered=True, run_id=TRIGGERED_RUN_ID)
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)

    with pytest.raises(common.QualificationError, match="review"):
        run_reproduction.make_llm(
            cfg, "target", _args(report), mock=False)
    assert builder.calls == []


# --------------------------------------------------------------------------- #
# I — three-model path retains stale-identity rejection
# --------------------------------------------------------------------------- #

def test_scenario_i_three_model_non_r1_target_is_rejected(
    tmp_path, monkeypatch,
):
    cfg, path = _write_cfg(
        tmp_path, target=NON_R1, use_models=True)
    report = _report(tmp_path, cfg, path)
    factory_calls = _spy(
        monkeypatch, __import__("guardbound.llm.provider_factory",
                                fromlist=["x"]), "build_role_llm")

    with pytest.raises(common.QualificationError, match="stale"):
        run_reproduction.make_llm(
            cfg, "target", _args(report), mock=False, manager=object())
    assert factory_calls == []


# --------------------------------------------------------------------------- #
# J — missing effective role identity fails closed
# --------------------------------------------------------------------------- #

def test_scenario_j_missing_role_identity_fails_closed(
    tmp_path, monkeypatch,
):
    cfg, path = _write_cfg(tmp_path, paper=True, omit="target")
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)

    with pytest.raises(common.QualificationError, match="missing model IDs"):
        run_reproduction.make_llm(cfg, "attacker", _args(), mock=False)
    assert builder.calls == []


def test_scenario_j_direct_helper_does_not_substitute_r1(tmp_path):
    cfg, _ = _write_cfg(tmp_path, paper=True, omit="target")
    report = tmp_path / "missing.qualification.v2.json"
    with pytest.raises(common.QualificationError, match="missing model IDs"):
        common.build_paid_chat_llm(
            R1, args=_args(report), cfg=cfg, role="target",
            model_kwargs={"client": FakeOpenAIBackend().client})


def test_scenario_j_build_role_llms_fails_closed_without_identity(
    tmp_path, monkeypatch,
):
    cfg, _ = _write_cfg(tmp_path, paper=True, omit="target")
    pf = __import__("guardbound.llm.provider_factory", fromlist=["x"])
    stack_calls = _spy(monkeypatch, pf, "build_three_model_stack")
    with pytest.raises(common.QualificationError):
        common.build_role_llms(cfg, object())
    assert stack_calls == []


# --------------------------------------------------------------------------- #
# K — non-paper compatibility
# --------------------------------------------------------------------------- #

def test_scenario_k_non_paper_non_r1_remains_compatible(tmp_path, monkeypatch):
    cfg, _ = _write_cfg(tmp_path, target=NON_R1, attacker=NON_R1,
                        evaluator=NON_R1, paper=False)
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)

    result = run_reproduction.make_llm(
        cfg, "target", _args(), mock=False)
    assert result.model_id == NON_R1
    assert builder.calls == [{"model": NON_R1, "cfg": cfg, "role": "target"}]


# --------------------------------------------------------------------------- #
# L — public helpers use the same effective identity
# --------------------------------------------------------------------------- #

def test_scenario_l_helpers_reject_r1_report_for_non_r1_target(
    tmp_path, monkeypatch,
):
    cfg, path = _write_cfg(tmp_path, target=NON_R1, use_models=True)
    report = _report(tmp_path, cfg, path)
    args = _args(report)
    role_cfg = {"provider": "openai", "model": NON_R1,
                "client": FakeOpenAIBackend().client}

    oc = __import__("guardbound.llm.openai_client", fromlist=["x"])
    pf = __import__("guardbound.llm.provider_factory", fromlist=["x"])
    chat_calls = _spy(monkeypatch, oc, "OpenAIChatLLM")
    role_calls = _spy(monkeypatch, pf, "build_role_llm")
    single_calls = _spy(monkeypatch, common, "_build_cloud_role")
    stack_calls = _spy(monkeypatch, pf, "build_three_model_stack")

    with pytest.raises(common.QualificationError, match="stale"):
        common.guard_paid_model(
            NON_R1, args=args, cfg=cfg, role="target")
    with pytest.raises(common.QualificationError, match="stale"):
        common.build_paid_chat_llm(
            NON_R1, args=args, cfg=cfg, role="target",
            model_kwargs={"client": FakeOpenAIBackend().client})
    # No explicit role: the model is present in the effective config, so the
    # identity is still derived from the actual target and must be rejected.
    with pytest.raises(common.QualificationError, match="stale"):
        common.guard_paid_model(NON_R1, args=args, cfg=cfg)
    with pytest.raises(common.QualificationError, match="stale"):
        common.build_paid_chat_llm(
            NON_R1, args=args, cfg=cfg,
            model_kwargs={"client": FakeOpenAIBackend().client})
    with pytest.raises(common.QualificationError, match="stale"):
        common.build_paid_role_llm(
            role_cfg, "target", args=args, cfg=cfg)
    with pytest.raises(common.QualificationError, match="stale"):
        common.build_single_role_llm(
            "target", role_cfg, object(), cfg=cfg,
            qualification_report=report)
    with pytest.raises(common.QualificationError, match="stale"):
        common.build_role_llms(
            cfg, object(), qualification_report=report)

    assert chat_calls == []
    assert role_calls == []
    assert single_calls == []
    assert stack_calls == []


def test_scenario_l_helper_rejects_unidentifiable_model_without_role(
    tmp_path, monkeypatch,
):
    cfg, path = _write_cfg(tmp_path, target=NON_R1, use_models=True)
    report = _report(tmp_path, cfg, path)
    args = _args(report)

    oc = __import__("guardbound.llm.openai_client", fromlist=["x"])
    chat_calls = _spy(monkeypatch, oc, "OpenAIChatLLM")

    with pytest.raises(common.QualificationError, match="cannot determine"):
        common.guard_paid_model(ALIAS, args=args, cfg=cfg)
    with pytest.raises(common.QualificationError, match="cannot determine"):
        common.build_paid_chat_llm(
            ALIAS, args=args, cfg=cfg,
            model_kwargs={"client": FakeOpenAIBackend().client})
    assert chat_calls == []


# --------------------------------------------------------------------------- #
# Effective identity design checks
# --------------------------------------------------------------------------- #

def test_effective_identity_includes_cli_override(tmp_path):
    cfg, _ = _write_cfg(tmp_path, target=R1, paper=True)
    effective = common._effective_runtime_config(
        cfg, role="target", model_id=NON_R1)
    assert effective["models"]["target"]["model"] == NON_R1
    assert effective["models"]["attacker"]["model"] == R1
    assert effective["models"]["evaluator"]["model"] == R1
    assert cfg["target"]["model"] == R1  # caller config is not mutated


def test_effective_identity_prefers_models_block_for_three_model(tmp_path):
    cfg, _ = _write_cfg(tmp_path, target=NON_R1, use_models=True)
    cfg["target"] = {"model": R1}  # conflicting top-level role
    effective = common._effective_runtime_config(cfg, model_id=R1)
    assert effective["models"]["target"]["model"] == NON_R1
    assert effective["models"]["attacker"]["model"] == R1
    assert effective["models"]["evaluator"]["model"] == R1
    assert "attacker" not in {
        key for key in effective if key in ROLES
    }


def test_effective_identity_defaults_structured_output_to_json_object(tmp_path):
    cfg, _ = _write_cfg(tmp_path)
    cfg["cloud"] = {}
    identity = common.expected_qualification_identity(cfg)
    assert identity["structured_output"] == "json_object"


def test_effective_identity_does_not_substitute_r1_for_supplied_config(tmp_path):
    cfg, _ = _write_cfg(tmp_path, paper=True, omit="target")
    with pytest.raises(common.QualificationError, match="missing model IDs"):
        common._effective_runtime_config(cfg)
