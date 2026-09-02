"""Content-addressed cache for LLM judge / classifier calls.

This module provides a small, dependency-free JSON-file cache so that
expensive, paid, or non-deterministic LLM judge calls can be reused
across runs without re-billing.

Design:
    - Keys are deterministic SHA-256 digests over a JSON-serialized
      payload that includes the prompt text, prompt version, model
      identifier, the input content (e.g. conversation), and any
      parameters that affect the result.
    - Values are JSON files containing the original request payload
      plus the recorded response.
    - Writes are atomic (write to ``.tmp`` then rename) so a crash
      never leaves a corrupt file in place.
    - All access is by string key; concurrent readers are safe
      because the cache only ever fully replaces files.

Not specified in the NBF paper — this is a local reproducibility aid.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable


def make_cache_key(parts: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 hex digest over the JSON-serialized parts.

    ``parts`` should be a JSON-serializable dict.  Keys are sorted to
    guarantee a stable order.
    """
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JudgeCache:
    """A simple on-disk content-addressed cache for judge calls."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Shard by first 2 hex chars to keep directories small.
        return self.cache_dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, record: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write
        fd, tmp = tempfile.mkstemp(prefix=".tmp_cache_", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:
            # Best-effort cleanup
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def call(
        self,
        parts: dict[str, Any],
        judge_fn: Callable[[], Any],
        *,
        transform: Callable[[Any], dict[str, Any]] | None = None,
    ) -> tuple[Any, bool]:
        """Call ``judge_fn`` (or load from cache) and return ``(result, hit)``.

        ``parts`` defines the cache key.  ``judge_fn`` must return a
        JSON-serializable value.  ``transform`` (optional) is applied
        to the result before storing; the transformed record is what
        is returned by ``get()`` (after reversing the transform via
        ``untransform`` if provided) — by default the result and the
        stored record are identical.
        """
        key = make_cache_key(parts)
        cached = self.get(key)
        if cached is not None:
            return cached, True

        result = judge_fn()
        if transform is not None:
            record = transform(result)
        else:
            # Store a JSON-safe view of the result
            record = {"result": _to_jsonable(result), "cached": False}
        record["cache_key"] = key
        self.set(key, record)
        return result, False


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of a Python value to a JSON-safe form."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return str(obj)
