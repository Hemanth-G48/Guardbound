"""Phase 8 — model lifecycle management for multi-model local experiments.

The three-model configuration (attacker + target + evaluator) cannot keep all
weights resident on a 24 GB GPU (Gemma-12B alone occupies ~24.4 GB bf16).
``ModelManager`` implements the required execution model:

    Load Qwen3.5-4B      -> attacker calls     -> unload
    Run NBF (always resident, ~12 MB)
    Load Llama-3.1-8B    -> target calls       -> unload
    Load Gemma-12B       -> evaluator calls    -> unload

The manager is a **substitution boundary** for the provider layer only: attacks
see ordinary ``ChatLLM`` objects and never know models are being swapped.
Every backend below delegates its heavy work to the shared ``_pipeline_cache``
in ``local_client``, so re-instantiating the same model id reuses the same
weights; unloading evicts it from cache AND from GPU/``CPU`` memory.

VRAM events are recorded for the manifest (load/unload with peak usage), which
lets the report show the swap cadence actually executed.
"""
from __future__ import annotations

import gc
import threading
from typing import Callable

import torch

from ..logging_utils import get_logger
from .base import ChatLLM, DEFAULT_TEMPERATURE, Message

logger = get_logger(__name__)


def _vram_gb() -> float | None:
    """Total allocated VRAM in GB, or None when CUDA is unavailable."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return None


class ModelManager:
    """Sequential model lifecycle manager for a single GPU.

    Responsibilities:
      * cache one loaded backend per (role, model_id)
      * ``activate(role)``: ensure that role's model is on GPU and others are
        not (unless they are the same physical model)
      * ``unload_all()``: move everything off GPU
      * record every load/unload/peak-VRAM event for the manifest
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._backends: dict[str, ChatLLM] = {}
        self._model_ids: dict[str, str] = {}
        self._events: list[dict] = []
        self._active_role: str | None = None
        self._lock = threading.Lock()

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

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _log(self, event: str, role: str | None = None, **extra) -> None:
        rec = {"event": event, "role": role, "vram_gb": _vram_gb(), **extra}
        self._events.append(rec)
        if role:
            logger.info("[model-manager] %s role=%s vram=%.2fGB %s",
                        event, role, rec["vram_gb"] or 0.0, extra or "")
        else:
            logger.info("[model-manager] %s vram=%.2fGB %s",
                        event, rec["vram_gb"] or 0.0, extra or "")

    def activate(self, role: str) -> None:
        """Ensure ``role``'s model is GPU-resident; evict *other* models.

        When two roles share the same physical model id (e.g. cloud backends,
        the Phase 7 single-model setup, or Phase 12's attacker==evaluator
        stack) both stay valid without any swap — the shared pipeline cache
        entry keeps serving the still-resident weights. Eviction is
        idempotent per (role) and a shared model is only truly released when
        no sibling role still references it.
        """
        with self._lock:
            if self._active_role == role:
                return
            target_id = self._model_ids[role]
            # Evict only the previously active role (the only other role
            # with possibly-resident weights) — and only when it is a
            # DIFFERENT physical model. Sibling roles sharing the target id
            # keep the shared cache entry alive on purpose.
            other_role = self._active_role
            if (
                other_role is not None
                and other_role != role
                and self._model_ids.get(other_role) != target_id
            ):
                self._evict(other_role)
            # Activating a role means its model is (re)loaded: clear any stale
            # "already evicted" marker. Without this, a role evicted in an
            # earlier turn would never be evicted again, so its weights would
            # stay resident and accumulate across turns until the GPU OOMs.
            if getattr(self, "_evicted", None) is not None:
                self._evicted.discard(role)
            self._log("activate", role, model_id=target_id)
            self._active_role = role

    def _evict(self, role: str) -> None:
        backend = self._backends.get(role)
        model_id = self._model_ids.get(role)
        if backend is None:
            return
        if getattr(self, "_evicted", None) is None:
            self._evicted: set[str] = set()
        if role in self._evicted:
            return  # already unloaded — no repeated event
        self._evicted.add(role)

        # Shared physical model: another role may still hold the weights via
        # the same pipeline cache key. Release the cache entry only when no
        # non-evicted sibling references the same model id, and clear this
        # backend's cached handles only when it actually references them.
        siblings_active = any(
            other_id == model_id and r not in self._evicted
            for r, other_id in self._model_ids.items()
            if r != role
        )
        if siblings_active:
            self._log("evict_shared_deferred", role, model_id=model_id)
            return

        # 1) drop the ChatLLM's own cached handles
        for attr in ("_pipeline", "_tokenizer", "_model", "_client"):
            if hasattr(backend, attr):
                try:
                    setattr(backend, attr, None)
                except Exception:  # noqa: BLE001
                    pass

        # 2) drop the shared pipeline cache for this model id
        from . import local_client

        for key in [k for k in local_client._pipeline_cache if model_id in str(k)]:
            local_client._pipeline_cache.pop(key, None)

        # 3) free GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._log("evict", role, model_id=model_id)

    def unload_all(self) -> None:
        """Move every model off the GPU (end of experiment or between runs)."""
        with self._lock:
            for role in list(self._backends):
                self._evict(role)
            self._active_role = None
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
    """A local ``HFLocalChatLLM`` wrapper that activates its role on each call.

    Wrapping (rather than modifying ``HFLocalChatLLM``) keeps the provider
    abstraction clean: the manager decides *when* models swap; the backend
    decides *how* generation happens. Attacks see a plain ``ChatLLM``.
    """

    def __init__(self, inner: ChatLLM, manager: ModelManager, role: str):
        self._inner = inner
        self._manager = manager
        self._role = role

    @property
    def name(self) -> str:  # noqa: D401
        return f"managed[{self._role}]"

    def generate(
        self,
        messages: list[Message],
        temperature: float = DEFAULT_TEMPERATURE,
        max_turns_context: int | None = None,
        json_format: bool = False,
    ) -> str | dict:
        self._manager.activate(self._role)
        out = self._inner.generate(
            messages,
            temperature=temperature,
            max_turns_context=max_turns_context,
            json_format=json_format,
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
    return ManagedLocalChatLLM(manager, role, model_id)
