# Copyright (c) Guardbound contributors.
"""Paid-client construction authorization (Phases 45-47).

Threat model (explicit)
-----------------------
The repository uses a **trusted in-process code** model:

* repository-owned application code is trusted not to deliberately import and
  misuse internal authorization capabilities;
* public builders and supported scripts must enforce the report/review gate;
* accidental direct construction fails closed;
* offline mock execution must not create a live provider client;
* internal capabilities are implementation details, not security primitives;
* arbitrary hostile Python code in the same process is **outside** the
  protection guarantee.

A Python ``ContextVar``, an underscore-prefixed function, a private module, or
an inspect-stack check cannot prevent arbitrary code in the same process from
deliberately bypassing application policy. If protection against untrusted
in-process code is required, credential access and paid transport must move
behind a separate process/OS-level broker. This module does not claim to be
such a boundary.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

R1_MODEL_IDS = frozenset({"gpt-4o-2024-08-06"})

_SYNC_OWNER = object()
_CAPABILITY_SEAL = object()


class PaidAuthorizationRequired(RuntimeError):
    """An R1 paid client was constructed without an active authorization scope."""


class _Scope:
    """One authorization scope, owned by one execution context."""

    __slots__ = ("owner", "released")

    def __init__(self, owner: object) -> None:
        self.owner = owner
        self.released = False


# model id -> active scope. ContextVar copies (child tasks / copied contexts)
# keep the same scope object, so the ``released`` flag is shared and the grant
# cannot outlive the builder that opened it.
_ACTIVE: ContextVar[dict[str, _Scope] | None] = ContextVar(
    "guardbound_paid_authorization", default=None)


@dataclass(frozen=True, slots=True)
class QualificationCapability:
    """Explicit harness capability for the qualification-only builder.

    Created only by ``_issue_qualification_capability`` in this module. The
    seal makes accidental construction fail; it is not a cryptographic token
    and does not protect against arbitrary hostile in-process code.
    """

    offline_mock: bool
    confirmed: bool
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _CAPABILITY_SEAL:
            raise TypeError(
                "QualificationCapability must be issued by "
                "_issue_qualification_capability"
            )
        if not isinstance(self.offline_mock, bool):
            raise TypeError("offline_mock must be a boolean")
        if not isinstance(self.confirmed, bool):
            raise TypeError("confirmed must be a boolean")
        if not self.offline_mock and not self.confirmed:
            raise ValueError(
                "a live qualification capability requires explicit confirmation"
            )


def _issue_qualification_capability(
    *, offline_mock: bool, confirmed: bool
) -> QualificationCapability:
    """Issue the internal harness capability. Private by convention."""
    return QualificationCapability(
        offline_mock=offline_mock,
        confirmed=confirmed,
        _seal=_CAPABILITY_SEAL,
    )


def is_r1_model(model_id: Any, cfg: dict[str, Any] | None = None) -> bool:
    """True for the frozen R1 snapshot or a paper-reproduction configuration.

    Detection is intentionally exact: only ``gpt-4o-2024-08-06`` (the frozen
    paper snapshot) or ``paper_reproduction: true``. Aliases such as ``gpt-4o``
    are not silently treated as R1; broadening this rule requires an explicit
    decision and tests.
    """
    if str(model_id or "").strip() in R1_MODEL_IDS:
        return True
    return bool(cfg and cfg.get("paper_reproduction"))


def _current_owner() -> object:
    """The current asyncio task, or the synchronous-owner sentinel."""
    try:
        import asyncio

        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return task if task is not None else _SYNC_OWNER


def require_construction_authorization(
    model_id: Any, cfg: dict[str, Any] | None = None
) -> None:
    """Fail closed if an R1 paid client is constructed outside an active,
    unreleased scope owned by the current execution context."""
    model = str(model_id or "").strip()
    if not is_r1_model(model, cfg):
        return
    scope = (_ACTIVE.get() or {}).get(model)
    if (scope is None or scope.released
            or scope.owner is not _current_owner()):
        raise PaidAuthorizationRequired(
            f"paid construction of R1 model {model!r} requires a validated "
            f"qualification report and review decision; construct it through "
            f"scripts._cloud_common.build_paid_chat_llm / build_paid_role_llm "
            f"or the qualification harness."
        )


@contextmanager
def _authorized(*model_ids: Any) -> Iterator[None]:
    """Internal capability scope.

    The grant is tied to the owning execution context (the current asyncio task
    or the synchronous sentinel) and is released in ``finally``. ContextVar
    copies made by child tasks or ``copy_context().run`` still reference the
    same scope object, so they see ``released=True`` after this builder exits
    and cannot extend the authorization lifetime.
    """
    owner = _current_owner()
    scope = _Scope(owner)
    current = dict(_ACTIVE.get() or {})
    for model_id in model_ids:
        current[str(model_id or "").strip()] = scope
    token = _ACTIVE.set(current)
    try:
        yield
    finally:
        scope.released = True
        _ACTIVE.reset(token)


def authorized_models() -> frozenset[str]:
    """Models authorized for the current execution context (diagnostics/tests)."""
    owner = _current_owner()
    return frozenset(
        model_id
        for model_id, scope in (_ACTIVE.get() or {}).items()
        if not scope.released and scope.owner is owner
    )
