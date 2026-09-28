"""Model lifecycle management for multi-model local experiments.

The three-model configuration (attacker + target + evaluator) cannot keep all
weights resident on a 24 GB GPU. For the Phase 14 stack the measured footprints
are attacker 8.45 GB + target 7.62 GB + evaluator 8.05 GB = 24.12 GB of weights
alone, against 24.39 GB free at idle — so at most two may be resident at once.

``ModelManager`` implements a **pinned-slot rotation**:

    attacker PINNED (never evicted for another role)
    slot 2: target <-> evaluator

    activate(target)    -> evict evaluator, load target if needed
    activate(evaluator) -> evict target,    load evaluator if needed
    activate(attacker)  -> evict any non-pinned role with a different model id

The manager is a **substitution boundary** for the provider layer only: attacks
see ordinary ``ChatLLM`` objects and never know models are being swapped. The
logical call order the attacks issue is never altered — no batching, no
deferral, no response caching.

RELEASING VRAM REQUIRES TWO RELEASES. Every backend delegates its heavy work to
the shared ``_pipeline_cache`` in ``local_client``. A loaded pipeline is held by
*both* that cache entry *and* the backend's own ``_pipeline`` attribute, so
nulling either one alone frees nothing. ``_evict`` therefore drops both, then
hands the blocks back to the driver with ``empty_cache``. A model released this
way must be reloaded from disk on its next use, which is the measured cost of
the rotation (11.25 s warm / 23.53 s cold for the Phase 14 attacker).

VRAM gates use ``torch.cuda.mem_get_info`` (driver-level free), never
``memory_allocated`` alone: the latter counts only this process's PyTorch
allocations and is blind to the ~1.38 GB the desktop and compositor hold.
"""
from __future__ import annotations

import gc
import threading
import time
from pathlib import Path
from typing import Callable

import torch

from ..logging_utils import get_logger
from .base import ChatLLM, DEFAULT_TEMPERATURE, Message
from .local_client import pipeline_cache_key, release_pipeline

logger = get_logger(__name__)

_HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"

# Used when a model's on-disk size cannot be determined. Deliberately generous:
# under-estimating an incoming footprint would admit an over-budget load, which
# is the failure mode the gate exists to prevent.
_UNKNOWN_FOOTPRINT_GB = 10.0


class ResidencyError(RuntimeError):
    """Base class for residency-layer failures (carries its failure class)."""

    failure_class = "INFRASTRUCTURE_FAILURE"


class VRAMGuardError(ResidencyError):
    """A VRAM gate refused an operation. Never forced past."""

    failure_class = "VRAM_SAFETY_FAILURE"


class ModelLoadError(ResidencyError):
    failure_class = "MODEL_LOAD_FAILURE"


class ModelEvictionError(ResidencyError):
    failure_class = "MODEL_EVICTION_FAILURE"


class CacheReleaseError(ResidencyError):
    failure_class = "CACHE_RELEASE_FAILURE"


class ResidencyStateError(ResidencyError):
    failure_class = "RESIDENCY_STATE_FAILURE"


def estimate_model_footprint_gb(model_id: str) -> float:
    """Conservative upper bound on a model's GPU footprint, in GB.

    Uses the on-disk safetensors size of the local snapshot: bf16 weights are
    ~2 bytes/parameter, so the file size is a safe over-estimate of the resident
    allocation. Gating on an over-estimate is correct here — the pre-load gate
    must refuse rather than admit an over-budget load.
    """
    snapshots: list[Path] = []
    local = Path(model_id)
    if local.exists() and local.is_dir():
        snapshots.append(local)
    else:
        base = _HF_HUB / ("models--" + model_id.replace("/", "--")) / "snapshots"
        if base.is_dir():
            snapshots.extend(sorted(p for p in base.iterdir() if p.is_dir()))
    for snapshot in snapshots:
        weights = list(snapshot.glob("*.safetensors"))
        if not weights:
            continue
        total = 0
        for weight_file in weights:
            try:
                total += weight_file.stat().st_size
            except OSError:
                continue
        if total:
            return total / 1e9
    return _UNKNOWN_FOOTPRINT_GB


def _vram_gb() -> float | None:
    """Total allocated VRAM in GB, or None when CUDA is unavailable."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return None


def _free_vram_gb() -> float | None:
    """Driver-level free VRAM in GB, or None when CUDA is unavailable.

    Unlike ``memory_allocated`` this accounts for every process on the card —
    including the ~1.38 GB the desktop compositor holds on this machine — so it
    is the only correct basis for a safety gate.
    """
    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        return free / 1e9
    return None


def _reserved_vram_gb() -> float | None:
    """PyTorch-reserved VRAM in GB (live allocations plus cached blocks)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1e9
    return None


class ModelManager:
    """Pinned-slot model lifecycle manager for a single GPU.

    Responsibilities:
      * cache one backend per logical role
      * ``activate(role)``: evict every *non-pinned* resident role backed by a
        different physical model, then mark ``role`` resident
      * ``ensure_resident(role)``: ``activate`` plus a gated, timed load
      * ``pin(role)``: exempt a role from eviction (the pinned attacker slot)
      * ``unload_all()``: release everything, pins included
      * record every load/evict/refusal event with driver-level VRAM before and
        after, for the manifest

    VRAM gates (all on driver-level free memory):
      * pre-load:  ``free - incoming_footprint >= min_free_before_load_gb``
      * post-load: ``free >= min_free_after_load_gb``
      * per-run:   ``free >= min_free_per_run_gb``
      * ceiling:   ``reserved <= max_reserved_gb``
    """

    def __init__(
        self,
        device: str = "cuda",
        min_free_before_load_gb: float = 3.0,
        min_free_after_load_gb: float = 1.5,
        min_free_per_run_gb: float = 6.0,
        max_reserved_gb: float = 22.0,
    ):
        self.device = device
        self._backends: dict[str, ChatLLM] = {}
        self._model_ids: dict[str, str] = {}
        self._events: list[dict] = []
        self._active_role: str | None = None
        # Reentrant: ensure_resident() holds the lock across its call to
        # activate(), which acquires it again.
        self._lock = threading.RLock()
        self._pinned: set[str] = set()
        self._resident: set[str] = set()
        self._footprints_gb: dict[str, float] = {}
        self._run_id: str | None = None
        self.min_free_before_load_gb = min_free_before_load_gb
        self.min_free_after_load_gb = min_free_after_load_gb
        self.min_free_per_run_gb = min_free_per_run_gb
        self.max_reserved_gb = max_reserved_gb

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register(self, role: str, backend: ChatLLM, model_id: str) -> None:
        """Register a backend for a logical role."""
        self._backends[role] = backend
        self._model_ids[role] = model_id

    def get(self, role: str) -> ChatLLM:
        if role not in self._backends:
            raise KeyError(
                f"role {role!r} not registered (have: {list(self._backends)})"
            )
        return self._backends[role]

    @property
    def roles(self) -> list[str]:
        return list(self._backends)

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    @property
    def resident_roles(self) -> list[str]:
        """Roles whose weights are currently GPU-resident."""
        return sorted(self._resident)

    @property
    def resident_model_ids(self) -> list[str]:
        """Distinct physical models currently GPU-resident."""
        return sorted(
            {self._model_ids[r] for r in self._resident if r in self._model_ids}
        )

    @property
    def pinned_roles(self) -> list[str]:
        return sorted(self._pinned)

    def pin(self, role: str) -> None:
        """Exempt ``role`` from eviction triggered by another role's activation."""
        if role not in self._backends:
            raise KeyError(f"cannot pin unregistered role {role!r}")
        self._pinned.add(role)

    def unpin(self, role: str) -> None:
        self._pinned.discard(role)

    def set_run_context(self, run_id: str | None) -> None:
        """Attach the current run id to subsequent telemetry events."""
        self._run_id = run_id

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _log(self, event: str, role: str | None = None, **extra) -> None:
        rec = {
            "event": event,
            "role": role,
            "run_id": self._run_id,
            "timestamp": time.time(),
            "vram_gb": _vram_gb(),
            "free_gb": _free_vram_gb(),
            "resident": self.resident_model_ids,
            **extra,
        }
        self._events.append(rec)
        if role:
            logger.info("[model-manager] %s role=%s vram=%.2fGB free=%.2fGB %s",
                        event, role, rec["vram_gb"] or 0.0, rec["free_gb"] or 0.0,
                        extra or "")
        else:
            logger.info("[model-manager] %s vram=%.2fGB free=%.2fGB %s",
                        event, rec["vram_gb"] or 0.0, rec["free_gb"] or 0.0,
                        extra or "")

    def activate(self, role: str) -> None:
        """Mark ``role`` resident, evicting every non-pinned conflicting role.

        Eviction is decided by *residency*, not by recency: every role currently
        holding GPU weights whose physical model differs from ``role``'s is
        released, except pinned roles. That is what makes the pinned-attacker
        rotation work — activating the evaluator releases the target and leaves
        the pinned attacker untouched.

        Roles sharing one physical model id (cloud backends, the Phase 7
        single-model stack, Phase 12's attacker==evaluator) never evict each
        other: the shared pipeline cache entry keeps serving both.
        """
        with self._lock:
            if self._active_role == role and role in self._resident:
                return
            target_id = self._model_ids[role]
            for other in sorted(self._resident):
                if other == role or other in self._pinned:
                    continue
                if self._model_ids.get(other) == target_id:
                    continue
                self._evict(other)
            # Activating a role means its model is (re)loaded: clear any stale
            # "already evicted" marker. Without this, a role evicted in an
            # earlier turn would never be evicted again, so its weights would
            # stay resident and accumulate across turns until the GPU OOMs.
            if getattr(self, "_evicted", None) is not None:
                self._evicted.discard(role)
            self._resident.add(role)
            self._log("activate", role, model_id=target_id)
            self._active_role = role

    def _evict(self, role: str) -> None:
        """Release a role's weights: backend handles AND the pipeline cache.

        Both releases are required. Before this was fixed, only the backend
        attributes were nulled while ``_pipeline_cache`` kept the pipeline, so
        an evict freed 0.000 GB of driver-level VRAM and the model re-activated
        in 0.00 s — it had never left the GPU.
        """
        backend = self._backends.get(role)
        model_id = self._model_ids.get(role)
        if backend is None:
            return
        if getattr(self, "_evicted", None) is None:
            self._evicted: set[str] = set()
        if role in self._evicted:
            return  # already released — no repeated event
        self._evicted.add(role)

        # Shared physical model: another resident role may still need the same
        # pipeline cache entry. Release it only when no other resident role
        # references the same model id.
        siblings_resident = any(
            other_id == model_id and r in self._resident
            for r, other_id in self._model_ids.items()
            if r != role
        )
        self._resident.discard(role)
        if siblings_resident:
            self._log("evict_shared_deferred", role, model_id=model_id)
            return

        device_map = getattr(backend, "device_map", self.device)
        free_before = _free_vram_gb()
        started = time.perf_counter()

        # 1) drop the backend's own handles
        for attr in ("_pipeline", "_tokenizer", "_model", "_client"):
            if hasattr(backend, attr):
                try:
                    setattr(backend, attr, None)
                except Exception:  # noqa: BLE001
                    pass

        # 2) drop the pipeline-cache reference and hand the blocks back to the
        #    driver. Without this the weights stay resident (measured).
        cache_released = release_pipeline(model_id, device_map)
        if not cache_released:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        free_after = _free_vram_gb()
        self._log(
            "evict", role, model_id=model_id, action="evict",
            duration_s=round(time.perf_counter() - started, 3),
            free_before_gb=free_before, free_after_gb=free_after,
            freed_gb=(None if free_before is None or free_after is None
                      else round(free_after - free_before, 3)),
            cache_released=cache_released,
        )

    def ensure_resident(self, role: str) -> None:
        """Make ``role``'s model resident, enforcing the VRAM gates.

        Single entry point used by the serving wrapper. Performs the eviction
        handover, then — only when the weights are not already loaded — runs the
        pre-load gate, times the load, verifies the post-load state and records
        one ``load`` event. A refusal raises ``VRAMGuardError`` and is never
        retried with a forced allocation.
        """
        with self._lock:
            backend = self._backends[role]
            already_loaded = getattr(backend, "_pipeline", None) is not None
            self.activate(role)
            if already_loaded or not hasattr(backend, "_get_pipeline"):
                return
            self._load_with_guards(role, backend)

    def _load_with_guards(self, role: str, backend: ChatLLM) -> None:
        """Gated, timed load of one role's weights. Caller holds ``_lock``."""
        model_id = self._model_ids[role]
        device_map = getattr(backend, "device_map", self.device)
        needed_gb = self._footprint_gb(model_id)

        # Return cached-but-unused blocks to the driver *before* reading free
        # VRAM, so the gate sees what is genuinely available rather than the
        # allocator's high-water reservation. Measured without this: a freshly
        # loaded 8.5 GB model left the driver reporting 7.43 GB free while
        # 8.41 GB sat in reusable cached blocks — the gate would have refused
        # every legitimate rotation.
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        free_before = _free_vram_gb()
        reserved_before = _reserved_vram_gb()
        allocated_before = _vram_gb()

        if (free_before is not None
                and free_before - needed_gb < self.min_free_before_load_gb):
            self._log(
                "load_refused", role, model_id=model_id, action="load_refused",
                free_before_gb=free_before, needed_gb=round(needed_gb, 3),
                reserved_before_gb=reserved_before,
                reason=(f"free {free_before:.2f}GB - incoming {needed_gb:.2f}GB "
                        f"< {self.min_free_before_load_gb:.2f}GB reserve"),
            )
            raise VRAMGuardError(
                f"pre-load gate REFUSED {role} ({model_id}): free "
                f"{free_before:.2f}GB - footprint {needed_gb:.2f}GB < "
                f"{self.min_free_before_load_gb:.2f}GB reserve"
            )

        started = time.perf_counter()
        try:
            backend._get_pipeline()
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"loading {role} ({model_id}) failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_s = time.perf_counter() - started
        free_after = _free_vram_gb()
        allocated_after = _vram_gb()

        # Footprint for future gates comes from the LIVE allocation delta, not
        # the driver-free delta. The latter includes the allocator's transient
        # reservation, which can transiently exceed the model's real size (a
        # 7.67 GB model has been measured at a 15.35 GB driver-free delta under
        # a mis-set PYTORCH_CUDA_ALLOC_CONF). Caching that transient would make
        # every later gate believe the model is twice its size and refuse
        # legitimate loads.
        live_delta = None
        if allocated_before is not None and allocated_after is not None:
            live_delta = allocated_after - allocated_before
            if live_delta > self._footprints_gb.get(model_id, 0.0):
                self._footprints_gb[model_id] = live_delta

        self._log(
            "load", role, model_id=model_id, action="load",
            duration_s=round(load_s, 3),
            free_before_gb=free_before, free_after_gb=free_after,
            reserved_before_gb=reserved_before,
            estimated_gb=round(needed_gb, 3),
            measured_gb=(None if live_delta is None else round(live_delta, 3)),
            driver_free_delta_gb=(
                None if free_before is None or free_after is None
                else round(free_before - free_after, 3)
            ),
            device_map=device_map,
        )

        if free_after is not None and free_after < self.min_free_after_load_gb:
            self._evict(role)
            raise VRAMGuardError(
                f"post-load verify FAILED for {role} ({model_id}): free "
                f"{free_after:.2f}GB < {self.min_free_after_load_gb:.2f}GB"
            )

    def _footprint_gb(self, model_id: str) -> float:
        """Best known footprint: the measured value, else a conservative estimate."""
        measured = self._footprints_gb.get(model_id)
        if measured is not None:
            return measured
        return estimate_model_footprint_gb(model_id)

    def pre_run_check(self, run_id: str) -> None:
        """Per-run guard: refuse to start a run with too little free VRAM.

        Releases PyTorch's reusable cached blocks BEFORE measuring, exactly as
        ``_load_with_guards`` already does. Without that step the raw
        driver-level free value counts cached-but-unused memory as consumed, so
        this guard can trip on a harmless allocator high-water mark rather than
        on genuine pressure: the Phase 15 pilot aborted at 108/180 with a raw
        4.04 GB free while 11.789 GB sat in reusable cache and effective
        availability was 15.829 GB.
        """
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        free = _free_vram_gb()
        if free is None:
            return
        if free < self.min_free_per_run_gb:
            self._log("run_refused", None, run_id=run_id, free_before_gb=free,
                      reason="per-run free-VRAM floor")
            raise VRAMGuardError(
                f"per-run guard REFUSED {run_id}: free {free:.2f}GB < "
                f"{self.min_free_per_run_gb:.2f}GB"
            )

    def ceiling_check(self, run_id: str) -> None:
        """Hard ceiling: abort before the physical limit is approached."""
        reserved = _reserved_vram_gb()
        if reserved is None:
            return
        if reserved > self.max_reserved_gb:
            self._log("ceiling_exceeded", None, run_id=run_id,
                      reserved_gb=round(reserved, 3), reason="hard VRAM ceiling")
            raise VRAMGuardError(
                f"hard VRAM ceiling exceeded during {run_id}: reserved "
                f"{reserved:.2f}GB > {self.max_reserved_gb:.2f}GB"
            )

    def residency_snapshot(self) -> dict:
        """Current residency state, for telemetry and invariant assertions."""
        reserved = _reserved_vram_gb()
        allocated = _vram_gb()
        return {
            "resident_roles": self.resident_roles,
            "resident_model_ids": self.resident_model_ids,
            "resident_count": len(self.resident_model_ids),
            "pinned_roles": self.pinned_roles,
            "free_gb": _free_vram_gb(),
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            # Reusable cached blocks: reserved but not live. Included because
            # driver-free alone understates what the next load can actually use.
            "cached_free_gb": (
                None if reserved is None or allocated is None
                else round(reserved - allocated, 3)
            ),
        }

    def unload_all(self) -> None:
        """Release every model, pins included (end of experiment)."""
        with self._lock:
            pinned = set(self._pinned)
            self._pinned.clear()
            try:
                for role in list(self._backends):
                    self._evict(role)
            finally:
                self._pinned |= pinned
            self._active_role = None
            self._resident.clear()
            if getattr(self, "_evicted", None) is not None:
                self._evicted.clear()
            self._log("unload_all")

    def reset_vram_peak(self) -> None:
        """Start a new per-run peak-VRAM window (Phase 12/13 telemetry).

        ``_events`` is append-only for the whole process, so a per-run window
        needs an index watermark, not a value baseline: any value-based filter
        either (a) keeps prior runs' events (cumulative peak, the Phase 12
        monotonic-growth bug) or (b) drops the current run's steady-state
        samples that do not exceed the pre-run level (the Phase 12
        "0.0 after run 2" defect when the baseline was seeded with the
        historical max). A watermark at the current event count makes
        ``peak_vram_gb()`` the true max over exactly this run's events.
        """
        with self._lock:
            self._peak_watermark = len(self._events)

    def sample_vram(self, role: str = "", note: str = "resident") -> float | None:
        """Record the *current* allocated VRAM for the manifest.

        ``activate`` logs before the weights are faulted in, so on its own it
        under-reports the footprint. Sampling after a generation reflects the
        real resident model size and is what ``peak_vram_gb`` should report.
        """
        with self._lock:
            return self._log(
                "vram_sample", role, model_id=self._model_ids.get(role), note=note
            )

    def peak_vram_gb(self) -> float:
        """Highest VRAM allocation recorded within the current window.

        Per-run peak (Phase 12/13): the maximum over events recorded since
        the last ``reset_vram_peak()`` watermark — this catches transient
        double-residency (e.g. mid-run role swaps that momentarily fault in
        new weights before the old role's eviction completes). If the window
        has no events (e.g. a pre-attack-only run whose shared model never
        re-activates), fall back to the *current* allocation so the reported
        figure still reflects real residency instead of a false 0.0.
        """
        with self._lock:
            wm = getattr(self, "_peak_watermark", 0)
            vals = [
                e["vram_gb"] for e in self._events[wm:]
                if e.get("vram_gb")
            ]
            peak = max(vals) if vals else 0.0
            current = _vram_gb()
            if current is not None:
                peak = max(peak, current)
            return peak

    def summary(self) -> dict:
        """Machine-readable lifecycle summary for the manifest."""
        return {
            "roles": {
                role: {
                    "model_id": self._model_ids[role],
                    "backend": type(self._backends[role]).__name__,
                }
                for role in self._backends
            },
            "events": self._events,
            "peak_vram_gb": round(self.peak_vram_gb(), 3),
        }


class ManagedLocalChatLLM(ChatLLM):
    """A local ``HFLocalChatLLM`` wrapper that ensures its role on each call.

    Wrapping (rather than modifying ``HFLocalChatLLM``) keeps the provider
    abstraction clean: the manager decides *when* models swap; the backend
    decides *how* generation happens. Attacks see a plain ``ChatLLM`` and their
    call order is never altered — the residency layer adapts underneath it.
    """

    def __init__(self, inner: ChatLLM, manager: ModelManager, role: str):
        self._inner = inner
        self._manager = manager
        self._role = role

    @property
    def name(self) -> str:  # noqa: D401
        return f"managed[{self._role}]"

    @property
    def structured_output_mode(self) -> str | None:
        """Decoding mode actually in effect on the wrapped backend.

        Readable through this wrapper so a caller can verify that the mode it
        requested survived the provider layer instead of being dropped.
        """
        return getattr(self._inner, "structured_output_mode", None)

    def generate(
        self,
        messages: list[Message],
        temperature: float = DEFAULT_TEMPERATURE,
        max_turns_context: int | None = None,
        json_format: bool = False,
        structured_output_mode: str | None = None,
    ) -> str | dict:
        self._manager.ensure_resident(self._role)
        out = self._inner.generate(
            messages,
            temperature=temperature,
            max_turns_context=max_turns_context,
            json_format=json_format,
            # Forward the decoding mode. Dropping it here silently disabled
            # Phase 14.1 constrained JSON for every managed role, because the
            # inner backend falls back to its constructor default (None).
            structured_output_mode=structured_output_mode,
        )
        # Weights are now resident: sample the true footprint for the manifest.
        self._manager.sample_vram(self._role)
        return out


def build_managed_local(
    manager: ModelManager,
    role: str,
    model_id: str,
    device_map: str = "cuda",
    max_new_tokens: int = 256,
    dtype: str = "bfloat16",
) -> ManagedLocalChatLLM:
    """Register a role backed by a local HF model with lifecycle management."""
    from .local_client import HFLocalChatLLM

    backend = HFLocalChatLLM(
        model_id=model_id,
        device_map=device_map,
        max_new_tokens=max_new_tokens,
    )
    manager.register(role, backend, model_id)
    return ManagedLocalChatLLM(backend, manager, role)
