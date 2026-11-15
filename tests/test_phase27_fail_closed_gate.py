"""Phase 27 — the local qualification gate fails closed (offline, deterministic).

A qualification may construct provider clients only when the mandatory local NBF
decision is ``QUALIFIED``. Every other local outcome — ``NOT_QUALIFIED``,
``INSUFFICIENT_DATA``, a skipped probe, a missing/corrupt checkpoint, an embedder
failure, a bad margin, non-determinism, or a candidate-filter failure — must stop
the run before any client exists, persist a report, and record zero usage calls.

The tests instrument ``common._build_qualification_role_llm``; no provider is contacted.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUAL_CONFIG = PROJECT_ROOT / "configs" / "qualification_r1_gpt4o.yaml"


def _qual():
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import cloud_model_qualification  # noqa: PLC0415

    return cloud_model_qualification


def _instrument_clients(monkeypatch, q) -> dict:
    """Count provider-client constructions (the only path to a request)."""
    state = {"clients": 0}
    real = q.common._build_qualification_role_llm

    def counting(*args, **kwargs):
        state["clients"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(q.common, "_build_qualification_role_llm", counting)
    return state


def _run(q, tmp_path, config=QUAL_CONFIG, extra=()):
    return q.main(["--config", str(config), "--offline-mock",
                   "--out-dir", str(tmp_path), *extra])


def _report(tmp_path) -> dict:
    path = next(Path(tmp_path).glob("*.qualification.v2.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def _usage_rows(payload: dict) -> list[dict]:
    path = Path(payload["usage_jsonl"])
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# --------------------------------------------------------------------------- #
# Local probe fixtures (dicts the probe functions would return)
# --------------------------------------------------------------------------- #

def _passing_nbf() -> dict:
    return {
        "skipped": False,
        "checkpoint": {"path": "checkpoint", "sha256": "x", "expected": "x",
                       "verified": True},
        "checkpoint_verified": True,
        "load_ok": True,
        "device": "cpu",
        "embedding_dim": 768,
        "embedding_dim_ok": True,
        "state_dim": 768,
        "h_value": -0.4,
        "deterministic": True,
        "score_formula_matches": True,
        "score_valid": True,
        "eta": 0.001,
        "threshold": 0.001,
        "raw_threshold": -0.001,
        "threshold_applied": True,
    }


def _passing_cf() -> dict:
    return {
        "skipped": False,
        "eta": 0.001,
        "raw_threshold": -0.001,
        "real_cases": [],
        "boundary_cases": [],
        "real_filtered": 1,
        "real_admitted": 1,
        "boundary_filtered": 1,
        "boundary_admitted": 1,
        "observed_filtered": 2,
        "observed_admitted": 2,
        "both_branches": True,
        "real_consistent": True,
        "boundary_consistent": True,
        "filter_ok": True,
        "target_calls_total": 2,
        "errors": [],
    }


def _patched_probes(monkeypatch, q, *, nbf=None, cf=None):
    nbf_probe = dict(_passing_nbf())
    nbf_probe.update(nbf or {})
    cf_probe = dict(_passing_cf())
    cf_probe.update(cf or {})
    monkeypatch.setattr(q, "probe_nbf_integration", lambda cfg, skip: dict(nbf_probe))
    monkeypatch.setattr(q, "probe_candidate_filter", lambda cfg, skip: dict(cf_probe))


# --------------------------------------------------------------------------- #
# 1 — a QUALIFIED local gate reaches client construction
# --------------------------------------------------------------------------- #

def test_qualified_local_gate_reaches_client_construction(
        tmp_path, monkeypatch, capsys, passing_local_gate):
    q = _qual()
    clients = _instrument_clients(monkeypatch, q)

    code = _run(q, tmp_path)
    capsys.readouterr()
    payload = _report(tmp_path)

    assert code == 1  # the mock still fails target/evaluator
    assert clients["clients"] == 3, "a passing gate must construct the three clients"
    assert payload["stopped_before_paid_probes"] is False
    assert payload["decisions"]["local"]["decision"] == q.QUALIFIED
    assert payload["probe_status"]["nbf_integration"]["status"] == q.PROBE_PASS
    # The full probe sequence ran.
    for probe in q.MANDATORY_PROBE_NAMES:
        assert probe in payload["probe_status"]
    assert "attacker_contract" in payload["probes"]
    assert payload["usage"]["calls"] > 0


def test_passing_gate_with_a_synthetic_qualified_stack_is_qualified(
        tmp_path, monkeypatch, capsys):
    """End-to-end QUALIFIED: passing probes and passing role metrics."""
    q = _qual()
    clients = _instrument_clients(monkeypatch, q)
    _patched_probes(monkeypatch, q)

    def passing_probe(*args, **kwargs):
        return {"contract_validity_rate": 1.0, "attack_coverage": 1.0,
                "empty_question_rate": 0.0, "multi_turn_usable_rate": 1.0,
                "request_error_rate": 0.0, "per_attack": {}, "samples": []}

    monkeypatch.setattr(q, "probe_attacker_contract", passing_probe)
    monkeypatch.setattr(q, "probe_attacker_multi_turn",
                        lambda *a, **k: {"multi_turn_usable_rate": 1.0})
    monkeypatch.setattr(q, "probe_attacker_consistency",
                        lambda *a, **k: {"contract_validity_rate": 1.0})
    monkeypatch.setattr(q, "probe_attacker_refusal_judge",
                        lambda *a, **k: {"request_error_rate": 0.0})
    monkeypatch.setattr(q, "probe_actor_attack_pre_attack",
                        lambda *a, **k: {
                            "payload_valid_rate": 1.0,
                            "structural_validity_rate": 1.0,
                            "structural_valid_count": 30,
                            "successful_response_count": 30,
                            "request_error_count": 0,
                            "empty_chain_rate": 0.0,
                            "delivery_type_valid_rate": 1.0,
                            "samples": []})
    monkeypatch.setattr(q, "probe_target", lambda *a, **k: {
        "completion": {"non_empty_rate": 1.0, "truncation_rate": 0.0,
                       "request_error_rate": 0.0, "non_empty": 1, "of": 1,
                       "truncated": 0},
        "multi_turn": {"context_recall_rate": 1.0, "continuity_rate": 1.0,
                       "pass_count": 5, "successful_conversation_count": 5,
                       "request_error_count": 0}})
    monkeypatch.setattr(q, "probe_evaluator_rubric", lambda *a, **k: {
        "rubric_validity_rate": 1.0, "rubric_determinism_rate": 1.0,
        "rubric_agreement": 1.0, "structural_violation_rate": 0.0,
        "scores": [1]})
    monkeypatch.setattr(q, "probe_evaluator_refusal",
                        lambda *a, **k: {"refusal_agreement": 1.0,
                                         "request_error_rate": 0.0})
    monkeypatch.setattr(q, "probe_evaluator_disclaimer",
                        lambda *a, **k: {"disclaimer_agreement": 1.0})
    monkeypatch.setattr(q, "probe_evaluator_consistency",
                        lambda *a, **k: {"false_refusal_rate": 0.0})

    code = _run(q, tmp_path)
    capsys.readouterr()
    payload = _report(tmp_path)

    assert clients["clients"] == 3
    assert payload["probes_complete"] is True
    assert payload["overall"] == q.QUALIFIED
    assert code == 0


# --------------------------------------------------------------------------- #
# 2-7 — every non-passing local outcome stops before client construction
# --------------------------------------------------------------------------- #

def _embedded_failure(monkeypatch, q):
    def boom(cfg_):
        raise RuntimeError("embedder unavailable")

    monkeypatch.setattr(q, "load_local_nbf", boom)


SCENARIOS = {
    # name: (setup, expected overall)
    "skip_nbf": ("skip", "INSUFFICIENT_DATA"),
    "missing_checkpoint": ("missing_checkpoint", "NOT_QUALIFIED"),
    "embedder_failure": ("embedder", "NOT_QUALIFIED"),
    "non_finite_score": ("nbf", "NOT_QUALIFIED"),
    "threshold_mismatch": ("nbf", "NOT_QUALIFIED"),
    "threshold_sign": ("nbf", "NOT_QUALIFIED"),
    "nondeterministic": ("nbf", "NOT_QUALIFIED"),
    "unmeasurable_determinism": ("nbf", "INSUFFICIENT_DATA"),
    "candidate_filter_failure": ("cf", "NOT_QUALIFIED"),
    "candidate_filter_untested": ("cf", "INSUFFICIENT_DATA"),
}

NBF_OVERRIDES = {
    "non_finite_score": {"score_valid": False},
    "threshold_mismatch": {"threshold_applied": False, "raw_threshold": 0.001},
    "threshold_sign": {"threshold_applied": False, "raw_threshold": 0.001},
    "nondeterministic": {"deterministic": False},
    "unmeasurable_determinism": {"deterministic": None},
}

CF_OVERRIDES = {
    "candidate_filter_failure": {"filter_ok": False, "real_consistent": False},
    "candidate_filter_untested": {"both_branches": None},
}


def _prepare(monkeypatch, q, tmp_path, name):
    setup, expected = SCENARIOS[name]
    extra: tuple = ()
    config = QUAL_CONFIG
    if setup == "skip":
        extra = ("--skip-nbf",)
    elif setup == "missing_checkpoint":
        cfg = copy.deepcopy(q.load_config(QUAL_CONFIG))
        cfg["nbf"]["checkpoint"] = str(Path(tmp_path) / "absent.pth")
        config = Path(tmp_path) / "cfg.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    elif setup == "embedder":
        _embedded_failure(monkeypatch, q)
    elif setup == "nbf":
        _patched_probes(monkeypatch, q, nbf=NBF_OVERRIDES[name])
    elif setup == "cf":
        _patched_probes(monkeypatch, q, cf=CF_OVERRIDES[name])
    return config, extra, expected


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_non_passing_local_gate_never_constructs_a_client(
        name, tmp_path, monkeypatch, capsys):
    q = _qual()
    clients = _instrument_clients(monkeypatch, q)
    config, extra, expected = _prepare(monkeypatch, q, tmp_path, name)

    code = _run(q, tmp_path, config=config, extra=extra)
    capsys.readouterr()

    assert code == 1, name
    assert clients["clients"] == 0, f"{name}: no provider client may be constructed"
    payload = _report(tmp_path)
    assert payload["overall"] == expected, name
    assert payload["overall"] != q.QUALIFIED
    assert payload["stopped_before_paid_probes"] is True, name
    assert payload["probes_complete"] is False


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_non_passing_outcome_persists_report_status_and_zero_usage(
        name, tmp_path, monkeypatch, capsys):
    q = _qual()
    config, extra, expected = _prepare(monkeypatch, q, tmp_path, name)
    _run(q, tmp_path, config=config, extra=extra)
    capsys.readouterr()
    payload = _report(tmp_path)

    statuses = payload["probe_status"]
    local_non_pass = [name_ for name_ in ("nbf_integration", "candidate_filter")
                      if statuses[name_]["status"] != q.PROBE_PASS]
    assert local_non_pass, f"{name}: the stop must be caused by a local probe"
    # Every paid probe is NOT_EXECUTED: the gate stopped before them.
    for paid in ("attacker", "actor_attack_pre_attack", "target", "evaluator"):
        assert statuses[paid]["status"] == q.PROBE_NOT_EXECUTED, name
    assert payload["decisions"]["local"]["decision"] != q.QUALIFIED
    assert payload["stop_reason"], name
    # No request was made, so the ledger is empty…
    assert payload["usage"]["calls"] == 0, name
    assert _usage_rows(payload) == [], name
    # …and the run row is written with the stop marker.
    rows = [json.loads(line) for line in
            (Path(tmp_path) / "qualification.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    assert rows and rows[-1]["stopped_before_paid_probes"] is True
    assert rows[-1]["overall"] == expected


def test_skip_nbf_stops_before_client_construction(tmp_path, monkeypatch, capsys):
    """The Phase 26 §10 demonstration must now stop at the gate."""
    q = _qual()
    clients = _instrument_clients(monkeypatch, q)
    code = _run(q, tmp_path, extra=("--skip-nbf",))
    out = capsys.readouterr().out
    payload = _report(tmp_path)

    assert code == 1
    assert clients["clients"] == 0
    assert payload["probe_status"]["nbf_integration"]["status"] == q.PROBE_INSUFFICIENT
    assert payload["overall"] == q.INSUFFICIENT_DATA
    assert "local NBF readiness gate incomplete" in out
    assert "no paid probe was run" in out


def test_gate_runs_before_the_key_check_and_client_factory(tmp_path, monkeypatch):
    """A failing gate must not even reach `require_api_key` or the factory."""
    q = _qual()
    monkeypatch.setattr(q, "probe_nbf_integration",
                        lambda cfg, skip: {"skipped": True, "reason": "--skip-nbf"})
    monkeypatch.setattr(q, "probe_candidate_filter",
                        lambda cfg, skip: {"skipped": True, "reason": "--skip-nbf"})

    def forbidden(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("no client/key path may run after a failed gate")

    monkeypatch.setattr(q.common, "_build_qualification_role_llm", forbidden)
    code = q.main(["--config", str(QUAL_CONFIG), "--offline-mock",
                   "--out-dir", str(tmp_path)])
    assert code == 1


def test_paid_mode_gate_failure_never_reaches_the_provider(tmp_path, monkeypatch):
    """Without --offline-mock, a failed gate must still stop before clients."""
    q = _qual()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-a-credential")
    calls = _instrument_clients(monkeypatch, q)
    monkeypatch.setattr(q, "probe_nbf_integration",
                        lambda cfg, skip: {"skipped": True, "reason": "--skip-nbf"})
    monkeypatch.setattr(q, "probe_candidate_filter",
                        lambda cfg, skip: {"skipped": True, "reason": "--skip-nbf"})

    code = q.main(["--config", str(QUAL_CONFIG), "--confirm-paid-request",
                   "--out-dir", str(tmp_path)])
    assert code == 1
    assert calls["clients"] == 0
    payload = _report(tmp_path)
    assert payload["usage"]["calls"] == 0
    assert payload["stopped_before_paid_probes"] is True
