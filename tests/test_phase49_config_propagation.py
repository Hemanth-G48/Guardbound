"""Phase 49 — configuration propagation through run_reproduction.make_llm."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.cloud_fakes import FakeOpenAIBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(PROJECT_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import _cloud_common as common  # noqa: E402
import run_reproduction  # noqa: E402

MODEL = "gpt-4o-mini"
R1 = "gpt-4o-2024-08-06"


def _args(**overrides):
    base = {
        "attacker": None,
        "target": None,
        "qualification_report": None,
        "review_decision": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _models(model=MODEL):
    return {
        role: {"provider": "openai", "model": model}
        for role in ("attacker", "target", "evaluator")
    }


def test_three_model_path_forwards_the_exact_config(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_build(role_cfg, role, *, args=None, cfg=None, manager=None, **kwargs):
        captured.update({
            "role_cfg": role_cfg, "role": role, "args": args,
            "cfg": cfg, "manager": manager,
        })
        return sentinel

    monkeypatch.setattr(common, "build_paid_role_llm", fake_build)
    cfg = {"models": _models()}
    manager = object()
    result = run_reproduction.make_llm(
        cfg, "target", _args(), mock=False, manager=manager)
    assert result is sentinel
    assert captured["cfg"] is cfg
    assert captured["role_cfg"] is cfg["models"]["target"]
    assert captured["manager"] is manager


def test_single_model_path_forwards_the_exact_config(monkeypatch):
    guard_calls = []
    build_calls = []
    sentinel = object()

    def fake_guard(model, args, *, cfg=None, role=None):
        guard_calls.append({"model": model, "cfg": cfg, "args": args,
                            "role": role})
        return {"gated": False}

    def fake_build(model, *, args=None, cfg=None, role=None, model_kwargs=None):
        build_calls.append({"model": model, "cfg": cfg, "args": args,
                            "role": role})
        return sentinel

    monkeypatch.setattr(common, "guard_paid_model", fake_guard)
    monkeypatch.setattr(common, "build_paid_chat_llm", fake_build)
    monkeypatch.setenv("OPENAI_API_KEY", "phase49-dummy-not-a-credential")
    cfg = {"attacker": {"model": MODEL}, "target": {"model": MODEL}}
    result = run_reproduction.make_llm(cfg, "target", _args(), mock=False)
    assert result is sentinel
    assert len(guard_calls) == 1
    assert guard_calls[0]["model"] == MODEL
    assert guard_calls[0]["cfg"] is cfg
    assert build_calls[0]["cfg"] is cfg
    assert build_calls[0]["model"] == MODEL


def test_hybrid_paper_three_model_non_r1_is_rejected_without_authorization():
    cfg = {"models": _models(), "paper_reproduction": True}
    with pytest.raises(common.QualificationError):
        run_reproduction.make_llm(
            cfg, "target", _args(), mock=False, manager=object())


def test_hybrid_paper_single_model_non_r1_is_rejected_without_authorization():
    cfg = {
        "attacker": {"model": MODEL},
        "target": {"model": MODEL},
        "paper_reproduction": True,
    }
    with pytest.raises(common.QualificationError):
        run_reproduction.make_llm(cfg, "target", _args(), mock=False)


def test_non_paper_non_r1_remains_compatible(monkeypatch):
    sentinel = object()

    def fake_build(model, *, args=None, cfg=None, role=None, model_kwargs=None):
        assert model == MODEL
        return sentinel

    monkeypatch.setattr(common, "build_paid_chat_llm", fake_build)
    monkeypatch.setenv("OPENAI_API_KEY", "phase49-dummy-not-a-credential")
    cfg = {"attacker": {"model": MODEL}, "target": {"model": MODEL}}
    assert run_reproduction.make_llm(
        cfg, "target", _args(), mock=False) is sentinel


def test_exact_r1_is_blocked_even_with_cfg_none(tmp_path):
    missing = tmp_path / "missing.qualification.v2.json"
    with pytest.raises(common.QualificationError):
        common.build_paid_chat_llm(
            R1, cfg=None,
            args=_args(qualification_report=missing),
            model_kwargs={"client": FakeOpenAIBackend().client})


def test_exact_r1_is_blocked_through_the_reproduction_entry_path():
    cfg = {"attacker": {"model": R1}, "target": {"model": R1}}
    with pytest.raises(common.QualificationError):
        run_reproduction.make_llm(cfg, "target", _args(), mock=False)


def test_reproduction_validation_is_unchanged():
    with pytest.raises(SystemExit):
        run_reproduction.validate_config({})
