"""Make ``src/`` importable for the test suite without installing the package."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def fake_backend():
    """A scripted OpenAI endpoint behind the real SDK (no network).

    Defined here rather than in ``tests/cloud_fakes.py`` so pytest discovers it
    in every test module; the transport itself lives there.
    """
    from tests.cloud_fakes import FakeOpenAIBackend

    return FakeOpenAIBackend()


@pytest.fixture(autouse=True)
def _authorize_frozen_r1_for_legacy_tests(request):
    """Phase 45: legacy tests construct the frozen R1 client to inspect request
    shaping, retry, and payload behavior. Grant the client-layer context for
    those tests; the Phase 45 module exercises the fail-closed path itself.
    """
    if any(
        name in request.module.__name__
        for name in (
            "test_phase45_authorization",
            "test_phase47_authorization_lifecycle",
            "test_phase51_runtime_identity",
            "test_phase53_identity_correction",
        )
    ):
        yield
        return
    from guardbound.llm.paid_authorization import _authorized

    with _authorized("gpt-4o-2024-08-06"):
        yield


@pytest.fixture()
def passing_local_gate(monkeypatch):
    """Force the qualification's mandatory local gate to pass.

    Phase 27 made the gate fail closed: *any* local verdict other than QUALIFIED
    stops the run before provider clients are constructed. Tests whose subject is
    the paid-probe behaviour (usage persistence, role decisions, report shape)
    therefore supply a passing local result instead of skipping the NBF probe.
    A test whose subject *is* the gate must not use this fixture.
    """
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import cloud_model_qualification as q  # noqa: PLC0415

    nbf = {
        "skipped": False,
        "checkpoint": {"verifiable": True},
        "checkpoint_verified": True,
        "load_ok": True,
        "embedding_dim": 768,
        "embedding_dim_ok": True,
        "state_dim": 768,
        "device": "cpu",
        "h_value": -0.4,
        "deterministic": True,
        "score_formula_matches": True,
        "score_valid": True,
        "eta": 0.001,
        "threshold": 0.001,
        "raw_threshold": -0.001,
        "threshold_applied": True,
    }
    candidate_filter = {
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
    monkeypatch.setattr(q, "probe_nbf_integration",
                        lambda cfg, skip: dict(nbf))
    monkeypatch.setattr(q, "probe_candidate_filter",
                        lambda cfg, skip: dict(candidate_filter))
    return q
