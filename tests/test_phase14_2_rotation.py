"""Phase 14.2 — three-model pinned-slot rotation.

Offline only: no model weights are loaded. The fake backend simulates the one
property that matters for residency — ``_get_pipeline`` populates the shared
``_pipeline_cache`` and ``_pipeline``, and only clearing BOTH releases the
weights.

Covers:
  * real eviction (cache entry + backend handle)
  * pinned roles are never evicted
  * the attacker/target/evaluator rotation state machine
  * VRAM gates: pre-load, post-load, per-run, hard ceiling
  * ``structured_output_mode`` forwarding through both previously-broken paths
  * three-distinct-model enforcement
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from guardbound.llm import local_client  # noqa: E402
from guardbound.llm import model_manager as mm  # noqa: E402
from guardbound.llm.base import ChatLLM  # noqa: E402
from guardbound.llm.local_client import pipeline_cache_key  # noqa: E402
from guardbound.llm.model_manager import (  # noqa: E402
    ManagedLocalChatLLM,
    ModelManager,
    VRAMGuardError,
)
from guardbound.llm.provider_factory import build_role_llm  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_cache():
    local_client._pipeline_cache.clear()
    yield
    local_client._pipeline_cache.clear()


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Never hit the Hub while detecting a model family."""
    monkeypatch.setattr(local_client, "_maybe_thinking_kwargs", lambda model_id: {})


class FreeVram:
    """Mutable driver-level free-VRAM reading for gate tests."""

    def __init__(self, value: float = 100.0):
        self.value = value


@pytest.fixture
def free_vram(monkeypatch):
    holder = FreeVram()
    monkeypatch.setattr(mm, "_free_vram_gb", lambda: holder.value)
    return holder


@pytest.fixture
def reserved_vram(monkeypatch):
    holder = FreeVram(0.0)
    monkeypatch.setattr(mm, "_reserved_vram_gb", lambda: holder.value)
    return holder


class FakeLocalBackend(ChatLLM):
    """Simulates HFLocalChatLLM's load/release contract without any weights."""

    name = "fake-local"

    def __init__(self, model_id: str, device_map: str = "cuda",
                 footprint_gb: float = 0.0, on_load=None):
        self.model_id = model_id
        self.device_map = device_map
        self.structured_output_mode = None
        self._pipeline = None
        self._tokenizer = None
        self.load_calls = 0
        self.generate_kwargs: list[dict] = []
        self._footprint_gb = footprint_gb
        self._on_load = on_load

    def _get_pipeline(self):
        self.load_calls += 1
        self._pipeline = {"model_id": self.model_id}
        local_client._pipeline_cache[
            pipeline_cache_key(self.model_id, self.device_map)
        ] = self._pipeline
        if self._on_load is not None:
            self._on_load()
        return self._pipeline

    def generate(self, messages, temperature=0.7, **kwargs):
        self.generate_kwargs.append(kwargs)
        return "fake-response"


def _register_three(manager: ModelManager, **kwargs) -> dict:
    backends = {}
    for role, model_id in (("attacker", "fake/attacker"),
                           ("target", "fake/target"),
                           ("evaluator", "fake/evaluator")):
        backends[role] = FakeLocalBackend(model_id, **kwargs)
        manager.register(role, backends[role], model_id)
    return backends


# --------------------------------------------------------------------------- #
# 1. Real eviction
# --------------------------------------------------------------------------- #

class TestRealEviction:
    def test_evict_releases_pipeline_cache_entry(self):
        """The bug being fixed: _evict freed 0 bytes because the cache held on."""
        manager = ModelManager()
        backends = _register_three(manager)

        key = pipeline_cache_key("fake/target", "cuda")
        manager.ensure_resident("target")
        assert key in local_client._pipeline_cache
        assert backends["target"]._pipeline is not None

        manager._evict("target")

        assert key not in local_client._pipeline_cache, (
            "evict must remove the pipeline-cache reference"
        )
        assert backends["target"]._pipeline is None, (
            "evict must also drop the backend's own handle"
        )

    def test_evict_records_free_vram_before_and_after(self):
        manager = ModelManager()
        _register_three(manager)
        manager.ensure_resident("target")
        manager._evict("target")

        evict = [e for e in manager.events if e["event"] == "evict"][0]
        assert evict["role"] == "target"
        assert evict["model_id"] == "fake/target"
        assert "free_before_gb" in evict and "free_after_gb" in evict
        assert evict["cache_released"] is True
        assert evict["duration_s"] >= 0.0

    def test_release_pipeline_returns_false_when_absent(self):
        assert local_client.release_pipeline("no/such-model", "cuda") is False

    def test_unload_all_releases_every_role_including_pins(self):
        manager = ModelManager()
        _register_three(manager)
        manager.pin("attacker")
        for role in ("attacker", "target", "evaluator"):
            manager.ensure_resident(role)

        manager.unload_all()

        assert local_client._pipeline_cache == {}
        assert manager.resident_roles == []


# --------------------------------------------------------------------------- #
# 2. Shared-model ownership semantics
# --------------------------------------------------------------------------- #

class TestSharedCacheSafety:
    def test_shared_entry_survives_while_sibling_still_resident(self):
        """A shared physical model must not be released out from under a role
        that is still resident and still needs those exact weights."""
        manager = ModelManager()
        for role in ("attacker", "evaluator"):
            manager.register(role, FakeLocalBackend("shared/model"), "shared/model")
        manager.register("target", FakeLocalBackend("fake/target"), "fake/target")

        manager.ensure_resident("attacker")
        manager.ensure_resident("evaluator")  # same physical model
        manager.pin("evaluator")

        manager.activate("target")

        # The attacker is evicted, but the cache entry must survive for the
        # pinned evaluator, which still references the same physical model.
        assert any(e["event"] == "evict_shared_deferred" for e in manager.events)
        assert pipeline_cache_key("shared/model", "cuda") in local_client._pipeline_cache
        assert "shared/model" in manager.resident_model_ids

    def test_shared_entry_released_once_no_referencing_role_remains(self):
        manager = ModelManager()
        for role in ("attacker", "evaluator"):
            manager.register(role, FakeLocalBackend("shared/model"), "shared/model")
        manager.register("target", FakeLocalBackend("fake/target"), "fake/target")

        manager.ensure_resident("attacker")
        manager.ensure_resident("evaluator")
        # Both shared roles are non-pinned, so activating target evicts both:
        # the first defers, the second releases.
        manager.activate("target")

        assert any(e["event"] == "evict_shared_deferred" for e in manager.events)
        assert pipeline_cache_key("shared/model", "cuda") not in local_client._pipeline_cache
        assert manager.resident_roles == ["target"]


# --------------------------------------------------------------------------- #
# 3. Pinned-slot rotation
# --------------------------------------------------------------------------- #

class TestPinnedRotation:
    def test_pinned_attacker_is_never_evicted(self):
        manager = ModelManager()
        backends = _register_three(manager)
        manager.pin("attacker")

        manager.ensure_resident("attacker")
        manager.ensure_resident("target")
        assert set(manager.resident_roles) == {"attacker", "target"}

        manager.ensure_resident("evaluator")
        assert set(manager.resident_roles) == {"attacker", "evaluator"}
        assert pipeline_cache_key("fake/attacker", "cuda") in local_client._pipeline_cache
        assert backends["attacker"].load_calls == 1, "pinned attacker must not reload"

    def test_slot2_alternates_and_reloads(self):
        manager = ModelManager()
        backends = _register_three(manager)
        manager.pin("attacker")

        for role in ("attacker", "target", "evaluator", "target", "evaluator"):
            manager.ensure_resident(role)
            # Invariant at every step: at most two distinct models resident,
            # and the attacker is always one of them.
            assert len(manager.resident_model_ids) <= 2
            assert "fake/attacker" in manager.resident_model_ids

        assert backends["attacker"].load_calls == 1
        assert backends["target"].load_calls == 2
        assert backends["evaluator"].load_calls == 2

    def test_unknown_model_never_triples_residency(self):
        manager = ModelManager()
        _register_three(manager)
        manager.pin("attacker")
        manager.ensure_resident("attacker")
        manager.ensure_resident("target")
        manager.ensure_resident("evaluator")
        snapshot = manager.residency_snapshot()
        assert snapshot["resident_count"] == 2
        assert snapshot["pinned_roles"] == ["attacker"]


# --------------------------------------------------------------------------- #
# 4. VRAM gates
# --------------------------------------------------------------------------- #

class TestVramGates:
    def test_preload_gate_refuses_over_budget_load(self, free_vram):
        manager = ModelManager(min_free_before_load_gb=3.0)
        backends = _register_three(manager)
        free_vram.value = 8.0  # unknown-model estimate is 10 GB -> refused

        with pytest.raises(VRAMGuardError, match="pre-load gate REFUSED"):
            manager.ensure_resident("target")

        assert backends["target"].load_calls == 0, "must not force the allocation"
        assert any(e["event"] == "load_refused" for e in manager.events)

    def test_preload_gate_admits_a_load_that_fits(self, free_vram):
        manager = ModelManager(min_free_before_load_gb=3.0)
        backends = _register_three(manager)
        free_vram.value = 30.0

        manager.ensure_resident("target")
        assert backends["target"].load_calls == 1
        assert [e["event"] for e in manager.events].count("load") == 1

    def test_postload_verify_evicts_and_raises(self, free_vram):
        manager = ModelManager(min_free_before_load_gb=3.0,
                               min_free_after_load_gb=1.5)
        backends = _register_three(
            manager,
            on_load=lambda: setattr(free_vram, "value", 0.5),
        )
        free_vram.value = 30.0

        with pytest.raises(VRAMGuardError, match="post-load verify FAILED"):
            manager.ensure_resident("target")

        assert "target" not in manager.resident_roles
        assert pipeline_cache_key("fake/target", "cuda") not in local_client._pipeline_cache
        assert backends["target"].load_calls == 1

    def test_per_run_floor(self, free_vram):
        manager = ModelManager(min_free_per_run_gb=6.0)
        _register_three(manager)
        free_vram.value = 5.9
        with pytest.raises(VRAMGuardError, match="per-run guard REFUSED"):
            manager.pre_run_check("OFF_crescendo_000")
        free_vram.value = 6.1
        manager.pre_run_check("OFF_crescendo_000")  # no raise

    def test_reserved_ceiling(self, reserved_vram):
        manager = ModelManager(max_reserved_gb=22.0)
        _register_three(manager)
        reserved_vram.value = 21.9
        manager.ceiling_check("OFF_crescendo_000")
        reserved_vram.value = 22.5
        with pytest.raises(VRAMGuardError, match="hard VRAM ceiling exceeded"):
            manager.ceiling_check("OFF_crescendo_000")


# --------------------------------------------------------------------------- #
# 5. structured_output_mode forwarding (previously silently dropped)
# --------------------------------------------------------------------------- #

class TestStructuredOutputForwarding:
    def test_managed_generate_forwards_mode_to_inner_backend(self):
        manager = ModelManager()
        backend = FakeLocalBackend("fake/attacker")
        manager.register("attacker", backend, "fake/attacker")
        managed = ManagedLocalChatLLM(backend, manager, "attacker")

        managed.generate([{"role": "user", "content": "hi"}],
                         json_format=True,
                         structured_output_mode="constrained_json")

        assert backend.generate_kwargs[0]["structured_output_mode"] == "constrained_json"

    def test_managed_generate_forwards_none_when_not_requested(self):
        manager = ModelManager()
        backend = FakeLocalBackend("fake/attacker")
        manager.register("attacker", backend, "fake/attacker")
        ManagedLocalChatLLM(backend, manager, "attacker").generate(
            [{"role": "user", "content": "hi"}]
        )
        assert backend.generate_kwargs[0]["structured_output_mode"] is None

    def test_mode_is_readable_through_the_wrappers(self):
        manager = ModelManager()
        backend = FakeLocalBackend("fake/attacker")
        backend.structured_output_mode = "constrained_json"
        manager.register("attacker", backend, "fake/attacker")
        managed = ManagedLocalChatLLM(backend, manager, "attacker")
        assert managed.structured_output_mode == "constrained_json"

    def test_provider_factory_forwards_mode(self):
        manager = ModelManager()
        llm = build_role_llm(
            {"provider": "local", "model": "fake/attacker",
             "max_new_tokens": 8},
            "attacker",
            manager=manager,
            structured_output_mode="constrained_json",
        )
        assert llm.structured_output_mode == "constrained_json"

    def test_per_role_config_key_wins(self):
        manager = ModelManager()
        llm = build_role_llm(
            {"provider": "local", "model": "fake/target", "max_new_tokens": 8,
             "structured_output_mode": "constrained_json"},
            "target",
            manager=manager,
            structured_output_mode=None,
        )
        assert llm.structured_output_mode == "constrained_json"


# --------------------------------------------------------------------------- #
# 6. Three-distinct-model enforcement (Phase 14 entry point)
# --------------------------------------------------------------------------- #

def _phase14_cfg(attacker: str, target: str, evaluator: str) -> dict:
    def role(model_id):
        return {"provider": "local", "model": model_id,
                "max_new_tokens": 8, "temperature": 0.7}
    return {
        "models": {"attacker": role(attacker), "target": role(target),
                   "evaluator": role(evaluator)},
        "hardware": {"device": "cuda"},
    }


class TestThreeDistinctModels:
    def test_duplicate_attacker_evaluator_is_rejected(self):
        import phase14_full_reproduction as p14

        cfg = _phase14_cfg("fake/Shared", "fake/target", "fake/Shared")
        with pytest.raises(p14.ModelIdentityError, match="three DISTINCT"):
            p14.make_models(cfg)

    def test_three_distinct_models_are_accepted_and_attacker_pinned(self):
        import phase14_full_reproduction as p14

        cfg = _phase14_cfg("fake/attacker", "fake/target", "fake/evaluator")
        attacker, target, evaluator, manager = p14.make_models(
            cfg, structured_output_mode="constrained_json"
        )
        assert manager is not None
        assert manager.pinned_roles == ["attacker"]
        assert manager.roles == ["attacker", "target", "evaluator"]
        assert len({manager._model_ids[r] for r in manager.roles}) == 3
        for llm in (attacker, target, evaluator):
            assert llm.structured_output_mode == "constrained_json"

    def test_missing_role_section_fails(self):
        import phase14_full_reproduction as p14

        cfg = _phase14_cfg("fake/attacker", "fake/target", "fake/evaluator")
        del cfg["models"]["evaluator"]
        with pytest.raises(RuntimeError, match="models.evaluator"):
            p14.make_models(cfg)


# --------------------------------------------------------------------------- #
# 7. Failure classification
# --------------------------------------------------------------------------- #

class TestFailureClassification:
    def test_vram_guard_maps_to_vram_safety_failure(self):
        import phase14_full_reproduction as p14

        cls, _, _ = p14.classify_exception(VRAMGuardError("nope"))
        assert cls == "VRAM_SAFETY_FAILURE"

    def test_semantic_errors_carry_their_own_class(self):
        import phase14_full_reproduction as p14

        assert p14.classify_exception(p14.ModelIdentityError("x"))[0] == \
            "MODEL_IDENTITY_FAILURE"
        assert p14.classify_exception(p14.NBFSemanticsError("x"))[0] == \
            "NBF_SEMANTICS_FAILURE"
        assert p14.classify_exception(p14.StructuredJSONError("x"))[0] == \
            "STRUCTURED_JSON_FAILURE"
        assert p14.classify_exception(p14.ResidencyInvariantError("x"))[0] == \
            "RESIDENCY_STATE_FAILURE"

    def test_preexisting_classifications_unchanged(self):
        import phase14_full_reproduction as p14

        assert p14.classify_exception(RuntimeError("CUDA out of memory"))[0] == \
            "INFRASTRUCTURE_ERROR"
        assert p14.classify_exception(NotImplementedError("x"))[0] == \
            "LOCAL_MODEL_CAPABILITY_LIMIT"

    def test_load_error_from_backend_failure(self, free_vram):
        manager = ModelManager()
        backend = FakeLocalBackend("fake/target")
        backend._get_pipeline = lambda: (_ for _ in ()).throw(
            RuntimeError("disk read failed")
        )
        manager.register("target", backend, "fake/target")
        free_vram.value = 30.0

        with pytest.raises(mm.ModelLoadError):
            manager.ensure_resident("target")


# --------------------------------------------------------------------------- #
# 8. Per-run guard must release reusable cache before measuring (Phase 15)
# --------------------------------------------------------------------------- #

class TestPreRunGuardReleasesCacheFirst:
    """The per-run guard must release PyTorch's reusable cached blocks BEFORE
    reading free VRAM, exactly as ``_load_with_guards`` does.

    Phase 15 pilot regression: without that step the guard reads the raw
    driver-level free value, which counts cached-but-unused memory as consumed.
    The pilot aborted at 108/180 on a raw 4.04 GB free while 11.789 GB sat in
    reusable cache (effective availability 15.829 GB).
    """

    def test_releases_cache_before_measuring(self, monkeypatch):
        order: list[str] = []
        monkeypatch.setattr(mm.torch.cuda, "empty_cache",
                            lambda *a, **k: order.append("empty_cache"))
        monkeypatch.setattr(mm.torch.cuda, "synchronize",
                            lambda *a, **k: order.append("synchronize"))
        monkeypatch.setattr(
            mm, "_free_vram_gb",
            lambda: (order.append("measure"), 100.0)[1],
        )

        ModelManager(min_free_per_run_gb=6.0).pre_run_check("OFF_crescendo_018")

        assert "measure" in order, "guard never measured free VRAM"
        assert "empty_cache" in order, (
            "guard measured free VRAM WITHOUT releasing reusable cache — the "
            "Phase 15 false-positive condition"
        )
        assert order.index("empty_cache") < order.index("measure")

    def test_still_refuses_genuinely_low_memory(self, free_vram):
        manager = ModelManager(min_free_per_run_gb=6.0)
        free_vram.value = 5.9
        with pytest.raises(VRAMGuardError, match="per-run guard REFUSED"):
            manager.pre_run_check("OFF_crescendo_018")

    def test_still_admits_sufficient_memory(self, free_vram):
        manager = ModelManager(min_free_per_run_gb=6.0)
        free_vram.value = 6.1
        manager.pre_run_check("OFF_crescendo_018")  # no raise

    def test_effective_free_after_release_is_what_is_measured(self, monkeypatch):
        """Reproduces the Phase 15 abort numerically: raw 4.04 GB free with
        11.789 GB reusable -> 15.829 GB effective, which must be ADMITTED."""
        state = {"raw": 4.04, "cached": 11.789}

        def fake_empty_cache(*a, **k):
            state["raw"] += state["cached"]  # blocks returned to the driver
            state["cached"] = 0.0

        monkeypatch.setattr(mm.torch.cuda, "empty_cache", fake_empty_cache)
        monkeypatch.setattr(mm.torch.cuda, "synchronize", lambda *a, **k: None)
        monkeypatch.setattr(mm, "_free_vram_gb", lambda: state["raw"])

        manager = ModelManager(min_free_per_run_gb=6.0)
        manager.pre_run_check("OFF_acronym_018")  # must NOT raise

        assert state["raw"] == pytest.approx(15.829, abs=1e-3)

    def test_safety_thresholds_unchanged(self):
        manager = ModelManager()
        assert manager.min_free_before_load_gb == 3.0
        assert manager.min_free_after_load_gb == 1.5
        assert manager.min_free_per_run_gb == 6.0
        assert manager.max_reserved_gb == 22.0

    def test_post_load_verification_still_enforced(self, free_vram):
        """The fix must not have weakened the post-load check."""
        manager = ModelManager(min_free_before_load_gb=3.0,
                               min_free_after_load_gb=1.5)
        _register_three(manager,
                        on_load=lambda: setattr(free_vram, "value", 0.5))
        free_vram.value = 30.0
        with pytest.raises(VRAMGuardError, match="post-load verify FAILED"):
            manager.ensure_resident("target")
