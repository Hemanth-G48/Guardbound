"""Phase 47 — qualification capability, offline identity, and scope lifetime.

This module is excluded from the legacy R1 test-context fixture in
``tests/conftest.py`` so the fail-closed cases observe the real boundary.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.cloud_fakes import FakeOpenAIBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(PROJECT_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

R1 = "gpt-4o-2024-08-06"
R1_CONFIG = PROJECT_ROOT / "configs" / "paper_reproduction_r1_gpt4o.yaml"


def _common():
    import _cloud_common  # noqa: PLC0415

    return _cloud_common


def _paid():
    from guardbound.llm import paid_authorization  # noqa: PLC0415

    return paid_authorization


def _args(**overrides):
    base = {"qualification_report": None, "review_decision": None}
    base.update(overrides)
    return SimpleNamespace(**base)


def _missing(tmp_path):
    return tmp_path / "missing.qualification.v2.json"


def _role_cfg(client=None):
    return {"provider": "openai", "model": R1, "client": client}


class _FactoryRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, role, role_cfg, ledger=None):
        self.calls.append({
            "role": role,
            "model": role_cfg.get("model"),
            "client": role_cfg.get("client"),
        })
        return object()


def _cfg(paper: bool = False):
    cfg = {
        "models": {
            role: {"provider": "openai", "model": R1}
            for role in ("attacker", "target", "evaluator")
        },
        "temperature": {"attacker": 0.7, "target": 0.7, "evaluator": 0.7},
        "prompt_variant": "official",
        "attacker_prompt_condition": "baseline",
        "cloud": {"structured_output": "json_object"},
        "_source_path": str(R1_CONFIG),
        "_source_sha256": _common().sha256_of(R1_CONFIG),
    }
    if paper:
        cfg["paper_reproduction"] = True
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
            "models": {"attacker": R1, "target": R1, "evaluator": R1},
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


def _write(tmp_path, payload):
    run_id = payload.get("run_id") or "20260101T000000Z"
    path = tmp_path / f"{R1}.baseline.{run_id}.qualification.v2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Qualification builder: explicit capability, no filename magic
# --------------------------------------------------------------------------- #

def test_builder_requires_a_capability():
    common = _common()
    with pytest.raises(TypeError):
        common._build_qualification_role_llm(
            "target", _role_cfg(common.OfflineMockClient()), object())


def test_spoofed_filename_does_not_grant_without_a_capability(tmp_path):
    common = _common()
    source = f"""
import sys
sys.path.insert(0, {SCRIPTS!r})
import _cloud_common as common
common._build_qualification_role_llm(
    "target",
    {{"provider": "openai", "model": "gpt-4o-2024-08-06", "client": None}},
    object(),
)
"""
    path = tmp_path / "cloud_model_qualification.py"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(TypeError):
        exec(compile(source, str(path), "exec"), {"__file__": str(path)})


def test_offline_capability_requires_actual_offline_client(monkeypatch, tmp_path):
    common = _common()
    paid = _paid()
    import guardbound.llm.provider_factory as pf

    recorder = _FactoryRecorder()
    monkeypatch.setattr(pf, "build_cloud_role_llm", recorder)
    cap = common._qualification_capability(
        offline_mock=True, confirmed=False)

    for bad_client in (None, FakeOpenAIBackend().client, object()):
        with pytest.raises(common.QualificationError, match="OfflineMockClient"):
            common._build_qualification_role_llm(
                "target", _role_cfg(bad_client), object(), capability=cap)
    assert recorder.calls == []


def test_offline_capability_accepts_offline_mock(monkeypatch):
    common = _common()
    paid = _paid()
    import guardbound.llm.provider_factory as pf

    recorder = _FactoryRecorder()
    monkeypatch.setattr(pf, "build_cloud_role_llm", recorder)
    cap = common._qualification_capability(
        offline_mock=True, confirmed=False)
    mock = common.OfflineMockClient()
    common._build_qualification_role_llm(
        "target", _role_cfg(mock), object(), capability=cap)
    assert recorder.calls == [
        {"role": "target", "model": R1, "client": mock}]
    assert paid.authorized_models() == frozenset()


def test_capability_requires_the_issuer_seal():
    paid = _paid()
    with pytest.raises(TypeError, match="issued"):
        paid.QualificationCapability(
            offline_mock=True, confirmed=False, _seal=object())


def test_live_capability_requires_confirmation():
    common = _common()
    with pytest.raises(ValueError, match="confirmation"):
        common._qualification_capability(
            offline_mock=False, confirmed=False)


def test_live_capability_reaches_only_the_fake_factory(monkeypatch):
    common = _common()
    from guardbound.llm.paid_authorization import QualificationCapability

    import guardbound.llm.provider_factory as pf

    recorder = _FactoryRecorder()
    monkeypatch.setattr(pf, "build_cloud_role_llm", recorder)
    cap = common._qualification_capability(
        offline_mock=False, confirmed=True)
    assert isinstance(cap, QualificationCapability)
    fake = FakeOpenAIBackend().client
    common._build_qualification_role_llm(
        "target", _role_cfg(fake), object(), capability=cap)
    assert recorder.calls == [
        {"role": "target", "model": R1, "client": fake}]


def test_builder_rejects_a_non_capability_object(monkeypatch):
    common = _common()
    import guardbound.llm.provider_factory as pf

    recorder = _FactoryRecorder()
    monkeypatch.setattr(pf, "build_cloud_role_llm", recorder)
    with pytest.raises(common.QualificationError, match="QualificationCapability"):
        common._build_qualification_role_llm(
            "target", _role_cfg(common.OfflineMockClient()), object(),
            capability="qualification")
    assert recorder.calls == []


def test_failed_builder_does_not_leave_a_grant(monkeypatch):
    common = _common()
    paid = _paid()
    import guardbound.llm.provider_factory as pf

    recorder = _FactoryRecorder()
    monkeypatch.setattr(pf, "build_cloud_role_llm", recorder)
    cap = common._qualification_capability(
        offline_mock=True, confirmed=False)
    with pytest.raises(ValueError, match="qualification only supports cloud"):
        common._build_qualification_role_llm(
            "target",
            {"provider": "local", "model": R1,
             "client": common.OfflineMockClient()},
            object(), capability=cap)
    assert paid.authorized_models() == frozenset()
    with pytest.raises(paid.PaidAuthorizationRequired):
        paid.require_construction_authorization(R1)


# --------------------------------------------------------------------------- #
# Authorization scope lifetime
# --------------------------------------------------------------------------- #

def test_direct_r1_construction_is_blocked():
    paid = _paid()
    from guardbound.llm.openai_client import OpenAIChatLLM

    with pytest.raises(paid.PaidAuthorizationRequired):
        OpenAIChatLLM(R1, client=FakeOpenAIBackend().client)


def test_authorized_construction_works_and_resets():
    paid = _paid()
    from guardbound.llm.openai_client import OpenAIChatLLM

    with paid._authorized(R1):
        llm = OpenAIChatLLM(R1, client=FakeOpenAIBackend().client)
        assert paid.authorized_models() == frozenset({R1})
    assert llm.model_id == R1
    assert paid.authorized_models() == frozenset()
    with pytest.raises(paid.PaidAuthorizationRequired):
        paid.require_construction_authorization(R1)


def test_exception_exit_resets_authorization():
    paid = _paid()
    with pytest.raises(RuntimeError, match="boom"):
        with paid._authorized(R1):
            raise RuntimeError("boom")
    assert paid.authorized_models() == frozenset()


def test_nested_scopes_restore_the_outer_state():
    paid = _paid()
    with paid._authorized(R1):
        with paid._authorized("gpt-4o-mini"):
            assert paid.authorized_models() == frozenset({R1, "gpt-4o-mini"})
        assert paid.authorized_models() == frozenset({R1})
    assert paid.authorized_models() == frozenset()


def test_model_isolation():
    paid = _paid()
    with paid._authorized("gpt-4o-mini"):
        with pytest.raises(paid.PaidAuthorizationRequired):
            paid.require_construction_authorization(R1)


def test_async_child_cannot_construct_after_parent_exits():
    paid = _paid()
    from guardbound.llm.openai_client import OpenAIChatLLM

    fake = FakeOpenAIBackend()

    async def scenario():
        gate = asyncio.Event()
        outcome = {}

        async def child():
            await gate.wait()
            try:
                OpenAIChatLLM(R1, client=fake.client)
                outcome["result"] = "constructed"
            except paid.PaidAuthorizationRequired:
                outcome["result"] = "blocked"

        with paid._authorized(R1):
            task = asyncio.create_task(child())
        gate.set()
        await task
        return outcome["result"]

    assert asyncio.run(scenario()) == "blocked"


def test_copied_context_cannot_extend_authorization():
    paid = _paid()
    copied = None
    with paid._authorized(R1):
        copied = contextvars.copy_context()
    with pytest.raises(paid.PaidAuthorizationRequired):
        copied.run(paid.require_construction_authorization, R1)


# --------------------------------------------------------------------------- #
# Configuration-level enforcement and review-gate preservation
# --------------------------------------------------------------------------- #

def test_exact_r1_is_gated_without_a_config(tmp_path):
    common = _common()
    with pytest.raises(common.QualificationError):
        common.build_paid_chat_llm(
            R1, args=_args(qualification_report=_missing(tmp_path)),
            model_kwargs={"client": FakeOpenAIBackend().client})


def test_paper_config_gates_a_non_r1_model_when_supplied(tmp_path):
    common = _common()
    with pytest.raises(common.QualificationError):
        common.build_paid_chat_llm(
            "gpt-4o-mini", cfg=_cfg(paper=True),
            args=_args(qualification_report=_missing(tmp_path)),
            model_kwargs={"client": FakeOpenAIBackend().client})


def test_other_helpers_apply_a_supplied_paper_config(tmp_path):
    common = _common()
    paper = _cfg(paper=True)
    missing = _missing(tmp_path)
    role_cfg = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "client": FakeOpenAIBackend().client,
    }
    with pytest.raises(common.QualificationError):
        common.build_paid_role_llm(
            role_cfg, "target", cfg=paper,
            args=_args(qualification_report=missing))
    with pytest.raises(common.QualificationError):
        common.build_single_role_llm(
            "target", role_cfg, object(), cfg=paper,
            qualification_report=missing)


def test_cfg_none_is_the_documented_non_paper_path(tmp_path):
    common = _common()
    llm = common.build_paid_chat_llm(
        "gpt-4o-mini", cfg=None,
        args=_args(qualification_report=_missing(tmp_path)),
        model_kwargs={"client": FakeOpenAIBackend().client})
    assert llm.model_id == "gpt-4o-mini"


def test_alias_policy_is_exact():
    paid = _paid()
    assert paid.is_r1_model(R1) is True
    assert paid.is_r1_model("gpt-4o") is False
    assert paid.is_r1_model("gpt-4o-2024-05-13") is False


def test_review_gate_preserved(tmp_path):
    common = _common()
    with pytest.raises(common.QualificationError, match="threshold"):
        payload = _report_payload()
        payload["review_triggers"]["delivery_type"]["threshold"] = 0.01
        common.require_qualification(_cfg(), _write(tmp_path, payload))

    with pytest.raises(common.QualificationError, match="observed"):
        payload = _report_payload()
        payload["review_triggers"]["delivery_type"].update(
            {"observed": None, "triggered": None})
        common.require_qualification(_cfg(), _write(tmp_path, payload))

    payload = _report_payload()
    payload["review_triggers"]["delivery_type"].update(
        {"observed": 0.49, "triggered": True})
    block = common.require_qualification(_cfg(), _write(tmp_path, payload))
    assert block["overall"] == "QUALIFIED"
