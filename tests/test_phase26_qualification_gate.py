"""Phase 26 — qualification gate hardening (offline, deterministic).

Covers the three P1 corrections and the P2/P3 improvements:

* P1-1 the NBF readiness checks are a mandatory ``local`` decision;
* P1-2 ``--model`` / ``--attacker-prompt-condition`` are rejected on paper configs;
* P1-3 per-probe status and a global ``probes_complete`` gate;
* P2-1 the local gate runs before any client is constructed;
* P2-2 per-call usage records are persisted incrementally;
* P2-3 the attack-level candidate filter is exercised with real metrics;
* P3   ``nbf.enabled``, unambiguous NBF keys, ``error_category`` in samples.

No network: model loads are stubbed (the real checkpoint + MPNet path is covered
by ``tests/test_cloud_nbf_integration.py`` and by the Phase 25/26 probes, both run
with ``HF_HUB_OFFLINE=1`` at process start).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUAL_CONFIG = PROJECT_ROOT / "configs" / "qualification_r1_gpt4o.yaml"
LEGACY_CONFIG = PROJECT_ROOT / "configs" / "cloud_qualification.yaml"


def _qual():
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import cloud_model_qualification  # noqa: PLC0415

    return cloud_model_qualification


def _local_thresholds(q, cfg=None):
    cfg = cfg or q.load_config(QUAL_CONFIG)
    return q.thresholds_for(cfg, q.LOCAL_ROLE)


def _all_local_pass() -> dict:
    return {
        "nbf_checkpoint_verified": 1.0,
        "nbf_loaded": 1.0,
        "nbf_embedding_dim_ok": 1.0,
        "nbf_score_valid": 1.0,
        "nbf_threshold_applied": 1.0,
        "nbf_deterministic": 1.0,
        "candidate_filter_ok": 1.0,
        "candidate_filter_both_branches": 1.0,
    }


class _StubEmbed:
    """Deterministic embedder: the encoded value *is* the barrier score h."""

    def __init__(self, values: dict[str, float], default: float):
        self.values = values
        self.default = default

    def __call__(self, text: str):
        return torch.tensor([[float(self.values.get(str(text), self.default))]])


class _StubBarrier:
    """Barrier whose score comes from the stub embedding."""

    class _Predictor:
        state_dim = 1
        embedding_dim = 1

        def __call__(self, x, u):
            h = float(u.reshape(-1)[0].item())
            other = (1.0 - h) / 5.0
            probs = torch.tensor([[other] * 4 + [h + other]], dtype=torch.float64)
            return torch.log(probs)

    def __init__(self):
        self.predictor = self._Predictor()

    def h(self, state, u):
        return torch.tensor([[float(u.reshape(-1)[0].item())]], dtype=torch.float64)

    def advance_state(self, state, u):
        return state

    def parameters(self):
        yield torch.zeros(1)


def _stub_bundle(q, values=None, default=-0.002) -> dict:
    """A local-NBF bundle that answers deterministically without a model load."""
    mapping = dict(values or {})
    mapping.setdefault(q.CANDIDATE_FILTER_QUERIES[0], 0.0)  # filtered at eta>0
    return {
        "barrier": _StubBarrier(),
        "embed_fn": _StubEmbed(mapping, default),
        "device": "cpu",
        "state_dim": 1,
        "embedding_dim": 768,
        "checkpoint_sha256": "stub",
        "checkpoint_path": "stub",
    }


# --------------------------------------------------------------------------- #
# P1-3 — per-probe status and probes_complete
# --------------------------------------------------------------------------- #

def test_probe_status_passes_when_every_metric_is_present_and_in_bounds():
    q = _qual()
    status = q.probe_status("nbf_integration", _all_local_pass(), _local_thresholds(q))
    assert status["status"] == q.PROBE_PASS
    assert status["missing_metrics"] == [] and status["failed_metrics"] == []


def test_probe_status_is_insufficient_when_a_metric_is_missing():
    q = _qual()
    metrics = _all_local_pass()
    metrics["nbf_deterministic"] = None
    status = q.probe_status("nbf_integration", metrics, _local_thresholds(q))
    assert status["status"] == q.PROBE_INSUFFICIENT
    assert status["missing_metrics"] == ["nbf_deterministic"]


def test_probe_status_fails_on_a_measured_violation():
    q = _qual()
    metrics = _all_local_pass()
    metrics["nbf_loaded"] = 0.0
    status = q.probe_status("nbf_integration", metrics, _local_thresholds(q))
    assert status["status"] == q.PROBE_FAIL
    assert status["failed_metrics"] == ["nbf_loaded"]


def test_absent_probes_are_not_executed_never_passing():
    q = _qual()
    statuses = q.probes_status({}, {}, {})
    assert set(statuses) == set(q.MANDATORY_PROBE_NAMES)
    assert all(s["status"] == q.PROBE_NOT_EXECUTED for s in statuses.values())
    assert q.probes_complete(statuses) is False


def test_probes_complete_requires_every_probe_to_pass():
    q = _qual()
    passing = {name: {"status": q.PROBE_PASS} for name in q.MANDATORY_PROBE_NAMES}
    assert q.probes_complete(passing) is True
    for state in (q.PROBE_FAIL, q.PROBE_INSUFFICIENT, q.PROBE_NOT_EXECUTED):
        broken = copy.deepcopy(passing)
        broken["target"] = {"status": state}
        assert q.probes_complete(broken) is False


def test_final_decision_requires_roles_probes_and_config():
    q = _qual()
    ok = {"decision": q.QUALIFIED}
    bad = {"decision": q.NOT_QUALIFIED}
    unknown = {"decision": q.INSUFFICIENT_DATA}
    passing = {name: {"status": q.PROBE_PASS} for name in q.MANDATORY_PROBE_NAMES}

    assert q.final_decision({"local": ok}, passing) == q.QUALIFIED
    # a failing probe blocks even when every role passed
    broken = copy.deepcopy(passing)
    broken["nbf_integration"] = {"status": q.PROBE_FAIL}
    assert q.final_decision({"local": ok}, broken) == q.NOT_QUALIFIED
    # a skipped / missing probe can never pass
    skipped = copy.deepcopy(passing)
    skipped["nbf_integration"] = {"status": q.PROBE_INSUFFICIENT}
    assert q.final_decision({"local": ok}, skipped) == q.INSUFFICIENT_DATA
    missing = copy.deepcopy(passing)
    missing.pop("candidate_filter")
    assert q.final_decision({"local": ok}, missing) != q.QUALIFIED
    # an unapproved configuration blocks
    assert q.final_decision({"local": ok}, passing, config_intact=False) == q.NOT_QUALIFIED
    # a failing role blocks
    assert q.final_decision({"local": bad}, passing) == q.NOT_QUALIFIED
    assert q.final_decision({"local": unknown}, passing) == q.INSUFFICIENT_DATA


def test_local_decision_from_missing_probes_is_insufficient():
    q = _qual()
    metrics = q._local_metrics({}, {})
    decision = q.decide(q.LOCAL_ROLE, metrics, _local_thresholds(q))
    assert decision["decision"] == q.INSUFFICIENT_DATA
    statuses = q.probes_status({}, {q.LOCAL_ROLE: metrics},
                               {q.LOCAL_ROLE: _local_thresholds(q)})
    assert q.final_decision({q.LOCAL_ROLE: decision}, statuses) != q.QUALIFIED


# --------------------------------------------------------------------------- #
# P1-1 — the NBF gate itself
# --------------------------------------------------------------------------- #

def test_nbf_probe_skip_is_insufficient(monkeypatch):
    q = _qual()
    cfg = q.load_config(QUAL_CONFIG)
    probe = q.probe_nbf_integration(cfg, skip=True)
    assert probe["skipped"] is True
    metrics = q._local_metrics(probe, {"skipped": True})
    assert all(value is None for value in metrics.values())
    decision = q.decide(q.LOCAL_ROLE, metrics, _local_thresholds(q, cfg))
    assert decision["decision"] == q.INSUFFICIENT_DATA


def test_nbf_probe_reports_a_missing_checkpoint_as_failure():
    q = _qual()
    cfg = {"nbf": {"checkpoint": "does/not/exist.pth"}}
    probe = q.probe_nbf_integration(cfg, skip=False)
    assert probe["checkpoint_verified"] is False
    assert probe["load_ok"] is False
    metrics = q._local_metrics(probe, {})
    decision = q.decide(q.LOCAL_ROLE, metrics, _local_thresholds(q))
    assert decision["decision"] == q.NOT_QUALIFIED


def test_nbf_probe_hash_mismatch_is_failure():
    q = _qual()
    cfg = {"nbf": {
        "checkpoint": str(PROJECT_ROOT / "nbf_original_stuff/orginal_code_give_by_author"
                          "/NBF-LLM/models/models_best_nbf_released.pth"),
        "checkpoint_sha256_expected": "0" * 64,
    }}
    probe = q.probe_nbf_integration(cfg, skip=False)
    assert probe["checkpoint_verified"] is False
    metrics = q._local_metrics(probe, {})
    assert q.decide(q.LOCAL_ROLE, metrics, _local_thresholds(q))["decision"] == \
        q.NOT_QUALIFIED


def test_nbf_probe_embedder_failure_is_failure(monkeypatch):
    q = _qual()
    cfg = q.load_config(QUAL_CONFIG)

    def boom(cfg_):
        raise RuntimeError("embedder unavailable")

    monkeypatch.setattr(q, "load_local_nbf", boom)
    probe = q.probe_nbf_integration(cfg, skip=False)
    assert probe["checkpoint_verified"] is True
    assert probe["load_ok"] is False
    assert probe["error_category"] == "embedder"
    metrics = q._local_metrics(probe, {})
    assert q.decide(q.LOCAL_ROLE, metrics, _local_thresholds(q, cfg))["decision"] == \
        q.NOT_QUALIFIED


def _patched_bundle(q, monkeypatch, *, h=0.0, h2=None):
    class _B:
        class _P:
            state_dim = 1
            embedding_dim = 1

            def __call__(self, x, u):
                other = (1.0 - h) / 5.0
                return torch.log(torch.tensor(
                    [[other] * 4 + [h + other]], dtype=torch.float64))

        def __init__(self):
            self.predictor = self._P()
            self._values = [h, h if h2 is None else h2]
            self._i = 0

        def h(self, state, u):
            value = self._values[min(self._i, len(self._values) - 1)]
            self._i += 1
            return torch.tensor([[value]], dtype=torch.float64)

        def advance_state(self, state, u):
            return state

        def parameters(self):
            yield torch.zeros(1)

    bundle = {
        "barrier": _B(),
        "embed_fn": lambda text: torch.zeros(1, 1),
        "device": "cpu",
        "state_dim": 1,
        "embedding_dim": 768,
        "checkpoint_sha256": "stub",
        "checkpoint_path": "stub",
    }
    monkeypatch.setattr(q, "load_local_nbf", lambda cfg_: bundle)
    return bundle


def test_nbf_probe_passes_with_a_loaded_barrier(monkeypatch):
    q = _qual()
    cfg = q.load_config(QUAL_CONFIG)
    _patched_bundle(q, monkeypatch, h=-0.4)
    probe = q.probe_nbf_integration(cfg, skip=False)
    assert probe["checkpoint_verified"] and probe["load_ok"]
    assert probe["embedding_dim_ok"] is True
    assert probe["deterministic"] is True
    assert probe["score_valid"] is True
    assert probe["threshold_applied"] is True
    assert probe["raw_threshold"] == -0.001
    metrics = q._local_metrics(probe, _all_local_pass_from(q))
    assert q.decide(q.LOCAL_ROLE, metrics, _local_thresholds(q, cfg))["decision"] == \
        q.QUALIFIED


def _all_local_pass_from(q):
    """Candidate-filter half of the local metrics (the probe itself is tested below)."""
    return {"filter_ok": True, "both_branches": True}


@pytest.mark.parametrize("threshold,eta,expected", [
    (0.001, 0.001, True),
    (-0.001, 0.001, False),   # sign inverted: threshold must equal eta, raw = -eta
    (0.5, 0.001, False),      # wrong margin
    (0.001, 0.01, False),     # eta itself disagrees with the paper value
])
def test_nbf_threshold_application_matrix(monkeypatch, threshold, eta, expected):
    q = _qual()
    cfg = q.load_config(QUAL_CONFIG)
    cfg = copy.deepcopy(cfg)
    cfg["nbf"]["eta"] = eta
    cfg["nbf"]["threshold"] = threshold
    _patched_bundle(q, monkeypatch, h=-0.4)
    probe = q.probe_nbf_integration(cfg, skip=False)
    assert probe["threshold_applied"] is expected


def test_nbf_determinism_gap_is_insufficient_not_a_pass(monkeypatch):
    q = _qual()
    cfg = q.load_config(QUAL_CONFIG)
    _patched_bundle(q, monkeypatch, h=-0.4, h2=-0.3)  # non-deterministic
    probe = q.probe_nbf_integration(cfg, skip=False)
    assert probe["deterministic"] is False
    metrics = q._local_metrics(probe, _all_local_pass_from(q))
    decision = q.decide(q.LOCAL_ROLE, metrics, _local_thresholds(q, cfg))
    assert decision["decision"] == q.NOT_QUALIFIED


def test_missing_determinism_metric_is_insufficient():
    q = _qual()
    metrics = _all_local_pass()
    metrics["nbf_deterministic"] = None
    assert q.decide(q.LOCAL_ROLE, metrics, _local_thresholds(q))["decision"] == \
        q.INSUFFICIENT_DATA


def test_filter_behaviour_not_tested_is_insufficient():
    q = _qual()
    metrics = _all_local_pass()
    metrics["candidate_filter_both_branches"] = None
    assert q.decide(q.LOCAL_ROLE, metrics, _local_thresholds(q))["decision"] == \
        q.INSUFFICIENT_DATA


# --------------------------------------------------------------------------- #
# P2-3 — candidate-filter probe
# --------------------------------------------------------------------------- #

def test_candidate_filter_probe_blocks_and_admits(monkeypatch):
    q = _qual()
    cfg = q.load_config(QUAL_CONFIG)
    monkeypatch.setattr(q, "load_local_nbf", lambda cfg_: _stub_bundle(q))
    probe = q.probe_candidate_filter(cfg, skip=False)

    assert probe["filter_ok"] is True
    assert probe["both_branches"] is True
    assert probe["eta"] == 0.001 and probe["raw_threshold"] == -0.001
    assert probe["real_filtered"] >= 1 and probe["real_admitted"] >= 1
    assert probe["boundary_filtered"] == 1 and probe["boundary_admitted"] == 1

    for case in probe["real_cases"] + probe["boundary_cases"]:
        assert case["consistent"] is True
        if case["filtered"]:
            assert case["target_calls"] == 0, case
        else:
            assert case["target_calls"] >= 1, case

    metrics = q._local_metrics(
        {"checkpoint_verified": True, "load_ok": True, "embedding_dim_ok": True,
         "score_valid": True, "threshold_applied": True, "deterministic": True},
        probe)
    assert q.decide(q.LOCAL_ROLE, metrics, _local_thresholds(q, cfg))["decision"] == \
        q.QUALIFIED


def test_candidate_filter_probe_skip_is_insufficient():
    q = _qual()
    cfg = q.load_config(QUAL_CONFIG)
    probe = q.probe_candidate_filter(cfg, skip=True)
    assert probe.get("skipped") is True
    metrics = q._local_metrics({}, probe)
    assert metrics["candidate_filter_ok"] is None
    assert q.decide(q.LOCAL_ROLE, metrics, _local_thresholds(q, cfg))["decision"] == \
        q.INSUFFICIENT_DATA


# --------------------------------------------------------------------------- #
# P1-2 — CLI override protection
# --------------------------------------------------------------------------- #

def _forbid_clients(monkeypatch, q):
    calls = []

    def _boom(*args, **kwargs):
        calls.append(args)
        raise AssertionError("a client must not be constructed")

    monkeypatch.setattr(q.common, "_build_qualification_role_llm", _boom)
    monkeypatch.setattr(q.common, "require_api_key", _boom)
    return calls


def test_paper_config_rejects_model_override(monkeypatch, capsys):
    q = _qual()
    calls = _forbid_clients(monkeypatch, q)
    code = q.main(["--config", str(QUAL_CONFIG),
                   "--model", "gpt-4o-mini-2024-07-18"])
    assert code == 2
    assert calls == []
    err = capsys.readouterr().err
    assert "--model" in err and "paper reproduction" in err


def test_paper_config_rejects_c1_condition(monkeypatch, capsys):
    q = _qual()
    calls = _forbid_clients(monkeypatch, q)
    code = q.main(["--config", str(QUAL_CONFIG),
                   "--attacker-prompt-condition", "c1_generation_point"])
    assert code == 2
    assert calls == []
    assert "--attacker-prompt-condition" in capsys.readouterr().err


def test_paper_config_rejects_both_overrides(monkeypatch, capsys):
    q = _qual()
    calls = _forbid_clients(monkeypatch, q)
    code = q.main(["--config", str(QUAL_CONFIG),
                   "--model", "gpt-4o-mini-2024-07-18",
                   "--attacker-prompt-condition", "c1_generation_point"])
    assert code == 2
    assert calls == []
    err = capsys.readouterr().err
    assert "--model" in err and "--attacker-prompt-condition" in err


def test_paper_config_without_overrides_is_accepted(monkeypatch, capsys):
    q = _qual()
    calls = _forbid_clients(monkeypatch, q)
    code = q.main(["--config", str(QUAL_CONFIG), "--dry-run"])
    assert code == 0
    assert calls == []
    out = capsys.readouterr().out
    assert "R1 protocol        validated" in out
    assert "Refusing --model" not in out


def test_legacy_config_keeps_its_override_behaviour(monkeypatch, capsys):
    q = _qual()
    calls = _forbid_clients(monkeypatch, q)
    code = q.main(["--config", str(LEGACY_CONFIG), "--model", "gpt-4o-mini-2024-07-18",
                   "--dry-run"])
    assert code == 0
    assert calls == []
    out = capsys.readouterr().out
    assert "gpt-4o-mini-2024-07-18" in out, "the override must still apply"
    assert "Refusing" not in out


# --------------------------------------------------------------------------- #
# P2-1 — local gate runs before any client construction
# --------------------------------------------------------------------------- #

def test_local_gate_failure_stops_before_client_construction(monkeypatch, tmp_path, capsys):
    q = _qual()
    calls = _forbid_clients(monkeypatch, q)

    def boom(cfg_):
        raise RuntimeError("embedder unavailable")

    monkeypatch.setattr(q, "load_local_nbf", boom)
    code = q.main(["--config", str(QUAL_CONFIG), "--offline-mock",
                   "--out-dir", str(tmp_path)])
    assert code == 1
    assert calls == [], "no client may be constructed after a local gate failure"

    report = next(tmp_path.glob("*.qualification.v2.json"))
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["overall"] == q.NOT_QUALIFIED
    assert payload["stopped_before_paid_probes"] is True
    assert payload["probes_complete"] is False
    assert payload["probe_status"]["nbf_integration"]["status"] == q.PROBE_FAIL
    assert payload["usage"]["calls"] == 0
    assert (tmp_path / "qualification.jsonl").exists()
    assert "no paid probe was run" in capsys.readouterr().out


def test_missing_checkpoint_stops_before_client_construction(monkeypatch, tmp_path):
    q = _qual()
    calls = _forbid_clients(monkeypatch, q)
    cfg = q.load_config(QUAL_CONFIG)
    cfg = copy.deepcopy(cfg)
    cfg["nbf"]["checkpoint"] = str(tmp_path / "absent.pth")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    code = q.main(["--config", str(cfg_path), "--offline-mock",
                   "--out-dir", str(tmp_path / "out")])
    assert code == 1
    assert calls == []


# --------------------------------------------------------------------------- #
# P2-2 — per-call usage persistence
# --------------------------------------------------------------------------- #

def _run_mock(q, tmp_path, extra=()):
    # The local NBF gate is mandatory (Phase 27): a test that exercises the paid
    # probes supplies a passing gate via the ``passing_local_gate`` fixture.
    return q.main(["--config", str(QUAL_CONFIG), "--offline-mock",
                   "--out-dir", str(tmp_path), *extra])


def test_usage_jsonl_has_one_row_per_call(tmp_path, capsys, passing_local_gate):
    q = _qual()
    code = _run_mock(q, tmp_path)
    capsys.readouterr()
    report = next(tmp_path.glob("*.qualification.v2.json"))
    payload = json.loads(report.read_text(encoding="utf-8"))
    usage_jsonl = Path(payload["usage_jsonl"])
    rows = [json.loads(line) for line in usage_jsonl.read_text(
        encoding="utf-8").splitlines() if line.strip()]

    assert code == 1  # the mock cannot satisfy the role criteria
    assert len(rows) == payload["usage"]["calls"] > 0
    assert {r["role"] for r in rows} <= {"attacker", "target", "evaluator"}
    for key in ("model", "input_tokens", "cached_tokens", "output_tokens",
                "total_tokens", "duration_s", "request_id", "error_category",
                "usage_available", "estimated_cost_usd", "cost_basis"):
        assert key in rows[0], key


def test_usage_jsonl_survives_an_interruption(monkeypatch, tmp_path, capsys,
                                              passing_local_gate):
    q = _qual()

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("simulated operator interrupt")

    monkeypatch.setattr(q, "probe_target", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _run_mock(q, tmp_path)
    capsys.readouterr()

    usage_files = list(tmp_path.glob("*.usage.jsonl"))
    assert usage_files, "per-call usage must exist after an interruption"
    rows = [json.loads(line) for line in usage_files[0].read_text(
        encoding="utf-8").splitlines() if line.strip()]
    assert rows, "calls made before the interruption must be retained"


# --------------------------------------------------------------------------- #
# P3 — configuration and reporting
# --------------------------------------------------------------------------- #

def test_r1_qualification_config_declares_nbf_enabled():
    cfg = yaml.safe_load(QUAL_CONFIG.read_text(encoding="utf-8"))
    assert cfg["nbf"]["enabled"] is True
    assert cfg["nbf"]["eta"] == 0.001 and cfg["nbf"]["threshold"] == 0.001
    assert cfg["thresholds"]["local"]["nbf_loaded"]["value"] == 1.0


def test_stats_expose_the_error_category():
    q = _qual()

    class _Client:
        last_request_stats = {
            "duration_s": 0.1, "input_tokens": 10, "output_tokens": 2,
            "outcome": "error", "finish_reason": None,
            "error_category": "timeout", "error_message": "read timed out",
        }

    stats = q._stats(_Client())
    assert stats["error_category"] == "timeout"
    assert stats["error_message"] == "read timed out"


def test_report_records_probe_status_and_completeness(tmp_path, capsys,
                                                      passing_local_gate):
    q = _qual()
    _run_mock(q, tmp_path)
    capsys.readouterr()
    payload = json.loads(next(tmp_path.glob("*.qualification.v2.json")).read_text(
        encoding="utf-8"))
    assert set(payload["probe_status"]) == set(q.MANDATORY_PROBE_NAMES)
    assert payload["probes_complete"] in (True, False)
    assert "local" in payload["decisions"]
    # The gate passed (fixture), so the paid probes ran.
    assert payload["stopped_before_paid_probes"] is False
    assert payload["probe_status"]["nbf_integration"]["status"] == q.PROBE_PASS
    # The mock still cannot satisfy the target/evaluator criteria.
    assert payload["overall"] != q.QUALIFIED
