"""Phase 53 — actual single-model evaluator identity and config consistency."""
from __future__ import annotations

import copy
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
SOURCE_CONFIG = PROJECT_ROOT / "configs" / "reproduction.yaml"
R1_CONFIG = PROJECT_ROOT / "configs" / "paper_reproduction_r1_gpt4o.yaml"
THREE_CONFIG = PROJECT_ROOT / "configs" / "reproduction_three_model.yaml"
DUMMY = "phase53-dummy-not-a-credential"
ROLES = ("attacker", "target", "evaluator")


def _actual_cfg(
    tmp_path,
    *,
    target=R1,
    attacker=R1,
    rubric=R1,
    refusal=None,
    paper=True,
    omit_rubric=False,
    empty_rubric=False,
    evaluator_temperature=0.7,
):
    data = copy.deepcopy(yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8")))
    data["prompt_variant"] = "official"
    data["attacker_prompt_condition"] = "baseline"
    if paper:
        data["paper_reproduction"] = True
    data["attacker"]["model"] = attacker
    data["target"]["model"] = target
    evaluator = data.setdefault("evaluator", {})
    if omit_rubric:
        evaluator.pop("rubric_model", None)
    elif empty_rubric:
        evaluator["rubric_model"] = ""
    else:
        evaluator["rubric_model"] = rubric
    evaluator["refusal_model"] = refusal or rubric
    evaluator["disclaimer_model"] = refusal or rubric
    evaluator["actor_step_judge_model"] = refusal or rubric
    evaluator["temperature"] = evaluator_temperature
    path = tmp_path / "phase53.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return run_reproduction.load_config(str(path)), path


def _report(tmp_path, cfg, config_path, *, run_id="20260101T000000Z"):
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
            name: {"status": "PASS"} for name in common.MANDATORY_PROBE_NAMES
        },
        "review_triggers": {
            "delivery_type": {
                "threshold": 0.50,
                "observed": 0.90,
                "triggered": False,
                "operator_decision": None,
                "rationale": None,
                "actor_attack_eligible": None,
            }
        },
    }
    path = tmp_path / f"{R1}.baseline.{run_id}.qualification.v2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _args(report=None, *, target=None, attacker=None):
    return SimpleNamespace(
        qualification_report=str(report) if report else None,
        review_decision=None,
        target=target,
        attacker=attacker,
    )


class _Builder:
    def __init__(self):
        self.real = common.build_paid_chat_llm
        self.backend = FakeOpenAIBackend()
        self.calls = []

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
# A–D: actual single-model schema
# --------------------------------------------------------------------------- #

def test_a_actual_schema_resolves_target_and_evaluator(tmp_path, monkeypatch):
    cfg, path = _actual_cfg(tmp_path)
    report = _report(tmp_path, cfg, path)
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY)

    target = run_reproduction.make_llm(
        cfg, "target", _args(report), mock=False)
    evaluator = run_reproduction.make_llm(
        cfg, "evaluator", _args(report), mock=False)

    assert isinstance(target, OpenAIChatLLM)
    assert isinstance(evaluator, OpenAIChatLLM)
    assert target.model_id == R1
    assert evaluator.model_id == R1
    assert builder.calls == [
        {"model": R1, "cfg": cfg, "role": "target"},
        {"model": R1, "cfg": cfg, "role": "evaluator"},
    ]


@pytest.mark.parametrize(
    ("role", "target", "attacker", "rubric"),
    [
        ("target", NON_R1, R1, R1),
        ("attacker", R1, NON_R1, R1),
        ("evaluator", R1, R1, NON_R1),
    ],
    ids=["B-target", "C-attacker", "D-evaluator-rubric"],
)
def test_b_d_non_r1_effective_role_is_rejected(
    tmp_path, monkeypatch, role, target, attacker, rubric,
):
    cfg, path = _actual_cfg(
        tmp_path, target=target, attacker=attacker, rubric=rubric)
    report = _report(tmp_path, cfg, path)
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)

    with pytest.raises(common.QualificationError, match="stale"):
        run_reproduction.make_llm(
            cfg, role, _args(report), mock=False)
    assert builder.calls == []


# --------------------------------------------------------------------------- #
# E: evaluator subroles
# --------------------------------------------------------------------------- #

def test_e_evaluator_subroles_use_the_single_evaluator_model(tmp_path, monkeypatch):
    cfg, path = _actual_cfg(
        tmp_path, rubric=R1, refusal=NON_R1)
    report = _report(tmp_path, cfg, path)
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY)

    evaluator = run_reproduction.make_llm(
        cfg, "evaluator", _args(report), mock=False)
    assert evaluator.model_id == R1

    meta = run_reproduction.build_metadata(cfg, _args(), mock=False)
    subroles = meta["evaluator_subroles"]
    assert subroles["rubric"]["consumer_model"] == R1
    assert subroles["refusal"]["declared_model"] == NON_R1
    assert subroles["refusal"]["consumer_model"] == R1
    assert subroles["disclaimer"]["consumer_model"] == R1
    assert subroles["actor_attack_judge"]["consumer_model"] == R1


# --------------------------------------------------------------------------- #
# F/G: missing/empty evaluator identity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", ["missing", "empty"])
def test_f_g_missing_or_empty_rubric_fails_closed(
    tmp_path, monkeypatch, mode,
):
    cfg, path = _actual_cfg(
        tmp_path, omit_rubric=mode == "missing",
        empty_rubric=mode == "empty")
    report = _report(tmp_path, cfg, path)
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)

    with pytest.raises(common.QualificationError, match="missing model IDs"):
        run_reproduction.make_llm(
            cfg, "target", _args(report), mock=False)
    assert builder.calls == []


# --------------------------------------------------------------------------- #
# H/I: single-model CLI overrides
# --------------------------------------------------------------------------- #

def test_h_target_override_is_rejected_before_factory(tmp_path, monkeypatch):
    cfg, path = _actual_cfg(tmp_path)
    report = _report(tmp_path, cfg, path)
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)

    with pytest.raises(common.QualificationError, match="stale"):
        run_reproduction.make_llm(
            cfg, "target", _args(report, target=NON_R1), mock=False)
    assert builder.calls == []


def test_i_attacker_override_is_rejected_before_factory(tmp_path, monkeypatch):
    cfg, path = _actual_cfg(tmp_path)
    report = _report(tmp_path, cfg, path)
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)

    with pytest.raises(common.QualificationError, match="stale"):
        run_reproduction.make_llm(
            cfg, "attacker", _args(report, attacker=NON_R1), mock=False)
    assert builder.calls == []


# --------------------------------------------------------------------------- #
# J: unsupported three-model CLI override
# --------------------------------------------------------------------------- #

def test_j_three_model_cli_override_fails_explicitly(tmp_path, monkeypatch):
    cfg = common.load_config(R1_CONFIG)
    report = _report(tmp_path, cfg, R1_CONFIG)
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)

    with pytest.raises(ValueError, match="not supported for three-model"):
        run_reproduction.make_llm(
            cfg, "target", _args(report, target=NON_R1), mock=False,
            manager=object())
    assert builder.calls == []


# --------------------------------------------------------------------------- #
# K: manifest
# --------------------------------------------------------------------------- #

def test_k_manifest_records_effective_evaluator_and_subroles(tmp_path):
    cfg, _ = _actual_cfg(tmp_path)
    args = SimpleNamespace(
        config="phase53.yaml", mode="test", limit=1,
        target=None, attacker=None)
    meta = run_reproduction.build_metadata(cfg, args, mock=False)
    assert meta["evaluator_model"] == R1
    assert meta["evaluator_subroles"]["rubric"]["consumer_model"] == R1

    manifest_path = run_reproduction.write_manifest(
        cfg, args, tmp_path / "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["evaluator_model"] == R1
    assert manifest["evaluator"]["model"] == R1
    assert manifest["evaluator_subroles"]["rubric"]["consumer_model"] == R1
    assert "config_hash" in manifest


# --------------------------------------------------------------------------- #
# L/M/N/O: three-model path
# --------------------------------------------------------------------------- #

def test_l_three_model_r1_roles_reach_factory(tmp_path, monkeypatch):
    cfg = common.load_config(R1_CONFIG)
    report = _report(tmp_path, cfg, R1_CONFIG)
    calls = []
    import guardbound.llm.provider_factory as pf
    original = pf.build_role_llm

    def fake(role_cfg, role, **kwargs):
        calls.append({"role": role, "model": role_cfg.get("model")})
        return type("LLM", (), {"model_id": role_cfg.get("model")})()

    pf.build_role_llm = fake
    try:
        for role in ROLES:
            run_reproduction.make_llm(
                cfg, role, _args(report), mock=False, manager=object())
    finally:
        pf.build_role_llm = original
    assert {call["role"] for call in calls} == set(ROLES)
    assert all(call["model"] == R1 for call in calls)


def test_m_three_model_non_r1_role_is_rejected(tmp_path, monkeypatch):
    cfg = run_reproduction.load_config(str(THREE_CONFIG))
    cfg["paper_reproduction"] = True
    cfg["prompt_variant"] = "official"
    cfg["attacker_prompt_condition"] = "baseline"
    # The source file is unchanged; the report is for the R1 role triple.
    cfg["_source_sha256"] = common.sha256_of(THREE_CONFIG)
    report = _report(tmp_path, cfg, THREE_CONFIG)
    import guardbound.llm.provider_factory as pf
    calls = _spy(monkeypatch, pf, "build_role_llm")

    with pytest.raises(common.QualificationError, match="stale"):
        run_reproduction.make_llm(
            cfg, "target", _args(report), mock=False, manager=object())
    assert calls == []


# --------------------------------------------------------------------------- #
# P–V: source/report identity
# --------------------------------------------------------------------------- #

def test_p_loaded_config_and_matching_report_pass(tmp_path):
    cfg, path = _actual_cfg(tmp_path)
    report = _report(tmp_path, cfg, path)
    block = common.require_paid_authorization(
        R1, cfg=cfg, qualification_report=report)
    assert block["gated"] is True
    assert block["effective_models"]["evaluator"]["model"] == R1


def test_q_source_file_change_is_rejected(tmp_path):
    cfg, path = _actual_cfg(tmp_path)
    report = _report(tmp_path, cfg, path)
    path.write_text(path.read_text(encoding="utf-8") + "\n# changed\n",
                    encoding="utf-8")
    with pytest.raises(common.QualificationError, match="hash changed"):
        common.require_qualification(cfg, report)


def test_r_hand_built_paper_config_without_source_fails_closed(tmp_path):
    cfg, path = _actual_cfg(tmp_path)
    report = _report(tmp_path, cfg, path)
    cfg.pop("_source_path", None)
    cfg.pop("_source_sha256", None)
    with pytest.raises(common.QualificationError, match="verified source file"):
        common.require_qualification(cfg, report)


def test_s_fabricated_source_hash_is_rejected(tmp_path):
    cfg, _ = _actual_cfg(tmp_path)
    cfg["_source_sha256"] = "0" * 64
    report = _report(tmp_path, cfg, SOURCE_CONFIG)
    with pytest.raises(common.QualificationError, match="hash changed"):
        common.require_qualification(cfg, report)


def test_t_source_matches_but_effective_model_differs(tmp_path):
    cfg, path = _actual_cfg(tmp_path, target=NON_R1)
    report = _report(tmp_path, cfg, path)
    with pytest.raises(common.QualificationError, match="models"):
        common.require_qualification(cfg, report)


def test_u_effective_model_matches_but_report_hash_differs(tmp_path):
    cfg, path = _actual_cfg(tmp_path)
    report = _report(tmp_path, cfg, path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["experiment_config"]["sha256"] = "0" * 64
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(common.QualificationError, match="experiment_config"):
        common.require_qualification(cfg, report)


def test_v_old_schema_is_not_silently_accepted(tmp_path):
    cfg, path = _actual_cfg(tmp_path)
    report = _report(tmp_path, cfg, path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["schema_version"] = "qualification.v1"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(common.QualificationError, match="schema_version"):
        common.require_qualification(cfg, report)


# --------------------------------------------------------------------------- #
# W/X/Y/AC
# --------------------------------------------------------------------------- #

def test_w_authorization_and_factory_see_same_evaluator(tmp_path, monkeypatch):
    cfg, path = _actual_cfg(tmp_path)
    report = _report(tmp_path, cfg, path)
    builder = _Builder()
    monkeypatch.setattr(common, "build_paid_chat_llm", builder)
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY)
    llm = run_reproduction.make_llm(
        cfg, "evaluator", _args(report), mock=False)
    assert builder.calls[0]["model"] == llm.model_id == R1


def test_y_defensive_snapshot_blocks_mapping_flip(tmp_path):
    class Flip(dict):
        def __init__(self):
            super().__init__(
                model=R1, client=FakeOpenAIBackend().client,
                provider="openai")
            self.calls = 0

        def get(self, key, default=None):
            if key == "model":
                self.calls += 1
                return R1 if self.calls <= 1 else NON_R1
            return super().get(key, default)

    flip = Flip()
    cfg = {
        "models": {
            "attacker": {"provider": "openai", "model": R1},
            "target": flip,
            "evaluator": {"provider": "openai", "model": R1},
        },
        "paper_reproduction": True,
        "temperature": {role: 0.7 for role in ROLES},
        "prompt_variant": "official",
        "attacker_prompt_condition": "baseline",
        "cloud": {"structured_output": "json_object"},
        "_source_path": str(R1_CONFIG),
        "_source_sha256": common.sha256_of(R1_CONFIG),
    }
    report = _report(tmp_path, cfg, R1_CONFIG)
    llm = common.build_paid_role_llm(
        flip, "target",
        args=SimpleNamespace(qualification_report=str(report),
                             review_decision=None),
        cfg=cfg)
    assert llm.model_id == R1


def test_ac_non_paper_local_config_still_works(tmp_path):
    cfg = run_reproduction.load_config(str(SOURCE_CONFIG))
    evaluator = run_reproduction.make_llm(
        cfg, "evaluator", _args(), mock=True)
    assert evaluator.name == "mock-json"
