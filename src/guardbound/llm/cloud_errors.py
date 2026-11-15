"""Classified failures for cloud LLM providers, plus credential redaction.

Formal research mode runs with ``retries = 0``, no repair, and no fallback
(Phase 13 protocol).  Every provider failure therefore has to surface as an
explicit, *classified* error instead of being absorbed by a retry loop or
replaced by a substituted response — otherwise a run silently changes its own
methodology.

The taxonomy below is the failure vocabulary recorded in every run manifest and
usage record.  Classification works with the OpenAI SDK installed *and* with a
test double that only mimics its class names, so offline tests exercise the same
classifier as production.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Failure categories (stable strings: they appear in manifests and reports)
# --------------------------------------------------------------------------- #

CATEGORY_AUTH = "authentication"
CATEGORY_PERMISSION = "permission"
CATEGORY_RATE_LIMIT = "rate_limit"
CATEGORY_TIMEOUT = "timeout"
CATEGORY_CONNECTION = "connection"
CATEGORY_SERVER = "server"
CATEGORY_BAD_REQUEST = "invalid_request"
CATEGORY_CONTEXT_LENGTH = "context_length"
CATEGORY_CONTENT_FILTER = "content_filter"
CATEGORY_PROVIDER = "provider_error"

ALL_CATEGORIES = (
    CATEGORY_AUTH,
    CATEGORY_PERMISSION,
    CATEGORY_RATE_LIMIT,
    CATEGORY_TIMEOUT,
    CATEGORY_CONNECTION,
    CATEGORY_SERVER,
    CATEGORY_BAD_REQUEST,
    CATEGORY_CONTEXT_LENGTH,
    CATEGORY_CONTENT_FILTER,
    CATEGORY_PROVIDER,
)


# --------------------------------------------------------------------------- #
# Credential redaction
# --------------------------------------------------------------------------- #

# Provider key shapes we can recognise without knowing the actual secret.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),          # OpenAI-style
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{8,}"),     # project-scoped OpenAI
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),           # AWS access key id
    re.compile(r"\bBearer\s+[A-Za-z0-9_\-\.=]{8,}", re.IGNORECASE),
)

REDACTED = "[REDACTED]"


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Strip credentials from ``text``.

    Applied to every log line, exception message and manifest field this module
    emits.  ``secrets`` carries the *actual* configured key (when known) so a key
    shape we do not recognise is still removed; the shape patterns catch keys we
    were never told about, such as one echoed back inside a server error body.
    """
    if not text:
        return text
    out = str(text)
    for secret in secrets:
        if secret and len(secret) >= 8:
            out = out.replace(secret, REDACTED)
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub(REDACTED, out)
    return out


def describe_api_key(api_key: str | None) -> str:
    """A safe one-line description of a key: presence and length only.

    Never returns any part of the key — not even a suffix — because a partial
    key is still credential material in a log file.
    """
    if not api_key:
        return "absent"
    return f"present (len={len(api_key)})"


# --------------------------------------------------------------------------- #
# Error hierarchy
# --------------------------------------------------------------------------- #

class CloudProviderError(RuntimeError):
    """Base class for cloud provider failures.

    ``category`` is the taxonomy bucket; ``status_code`` and ``request_id`` are
    whatever the provider exposed (``None`` when it did not).
    """

    category: str = CATEGORY_PROVIDER

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
        secrets: Iterable[str] = (),
        original: BaseException | None = None,
    ) -> None:
        self.model = model
        self.status_code = status_code
        self.request_id = request_id
        self.original = original
        self.message = redact(message, secrets)
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "category": self.category,
            "message": self.message,
            "model": self.model,
            "status_code": self.status_code,
            "request_id": self.request_id,
        }

    def __str__(self) -> str:
        bits = [self.message]
        if self.model:
            bits.append(f"model={self.model}")
        if self.status_code is not None:
            bits.append(f"status={self.status_code}")
        if self.request_id:
            bits.append(f"request_id={self.request_id}")
        return " | ".join(bits)


class CloudAuthError(CloudProviderError):
    """Missing, malformed or revoked credentials (HTTP 401)."""

    category = CATEGORY_AUTH


class CloudPermissionError(CloudProviderError):
    """Authenticated but not allowed to use the model (HTTP 403)."""

    category = CATEGORY_PERMISSION


class CloudRateLimitError(CloudProviderError):
    """Quota or rate limit hit (HTTP 429). Not retried in research mode."""

    category = CATEGORY_RATE_LIMIT


class CloudTimeoutError(CloudProviderError):
    """The request exceeded the configured timeout."""

    category = CATEGORY_TIMEOUT


class CloudConnectionError(CloudProviderError):
    """Transport-level failure: DNS, TLS, reset connection."""

    category = CATEGORY_CONNECTION


class CloudServerError(CloudProviderError):
    """Provider-side failure (HTTP 5xx)."""

    category = CATEGORY_SERVER


class CloudBadRequestError(CloudProviderError):
    """The request itself was rejected (HTTP 400/404/422)."""

    category = CATEGORY_BAD_REQUEST


class CloudContextLengthError(CloudBadRequestError):
    """The conversation exceeded the model's context window."""

    category = CATEGORY_CONTEXT_LENGTH


class CloudContentFilterError(CloudProviderError):
    """The provider's own safety layer blocked the exchange."""

    category = CATEGORY_CONTENT_FILTER


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

# Checked in order: a context-length 400 must not be reported as a generic 400.
_NAME_RULES: tuple[tuple[tuple[str, ...], type[CloudProviderError]], ...] = (
    (("APITimeoutError", "Timeout", "TimeoutError", "ReadTimeout", "ConnectTimeout"),
     CloudTimeoutError),
    (("RateLimitError",), CloudRateLimitError),
    (("AuthenticationError",), CloudAuthError),
    (("PermissionDeniedError", "PermissionError"), CloudPermissionError),
    (("APIConnectionError", "ConnectError", "ConnectionError", "RemoteProtocolError"),
     CloudConnectionError),
    (("InternalServerError", "APIStatusError"), CloudServerError),
    (("BadRequestError", "UnprocessableEntityError", "NotFoundError"),
     CloudBadRequestError),
)

_CONTEXT_TOKENS = (
    "context length",
    "context_length",
    "maximum context",
    "reduce the length",
    "too many tokens",
    "context window",
)
_FILTER_TOKENS = ("content_filter", "content management policy", "safety system")


def _status_code_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _request_id_of(exc: BaseException) -> str | None:
    request_id = getattr(exc, "request_id", None)
    if isinstance(request_id, str):
        return request_id
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            value = headers.get("x-request-id")
        except Exception:  # noqa: BLE001 - headers object may be exotic
            value = None
        if isinstance(value, str):
            return value
    return None


def classify_exception(
    exc: BaseException,
    *,
    model: str | None = None,
    secrets: Iterable[str] = (),
) -> CloudProviderError:
    """Map a provider exception onto the taxonomy, preserving the original.

    Already-classified errors are returned unchanged so a re-raise never
    re-wraps (and never loses the original category).
    """
    if isinstance(exc, CloudProviderError):
        return exc

    text = f"{type(exc).__name__}: {exc}".lower()
    status = _status_code_of(exc)
    request_id = _request_id_of(exc)

    def _wrap(cls: type[CloudProviderError]) -> CloudProviderError:
        return cls(
            f"{type(exc).__name__}: {exc}",
            model=model,
            status_code=status,
            request_id=request_id,
            secrets=secrets,
            original=exc,
        )

    if any(token in text for token in _FILTER_TOKENS):
        return _wrap(CloudContentFilterError)
    if any(token in text for token in _CONTEXT_TOKENS):
        return _wrap(CloudContextLengthError)

    name = type(exc).__name__
    for names, cls in _NAME_RULES:
        if name in names:
            return _wrap(cls)

    if status == 401:
        return _wrap(CloudAuthError)
    if status == 403:
        return _wrap(CloudPermissionError)
    if status == 404:
        return _wrap(CloudBadRequestError)
    if status == 408:
        return _wrap(CloudTimeoutError)
    if status == 429:
        return _wrap(CloudRateLimitError)
    if status is not None and 400 <= status < 500:
        return _wrap(CloudBadRequestError)
    if status is not None and status >= 500:
        return _wrap(CloudServerError)

    return _wrap(CloudProviderError)


def failure_categories() -> tuple[str, ...]:
    """The stable category vocabulary, for manifests and reports."""
    return ALL_CATEGORIES


# --------------------------------------------------------------------------- #
# Retry policy (Phase 30)
# --------------------------------------------------------------------------- #

# Categories a retry cannot fix. The author's release retried *any* exception
# (``generate.py:129-151``); retrying a bad key or an unparseable request only
# burns the schedule, so Guardbound fails fast on these and still retries every
# transient category (rate limit, timeout, connection, server, provider).
NON_RETRYABLE_CATEGORIES: tuple[str, ...] = (
    CATEGORY_AUTH,
    CATEGORY_PERMISSION,
    CATEGORY_BAD_REQUEST,
    CATEGORY_CONTEXT_LENGTH,
    CATEGORY_CONTENT_FILTER,
)


class RetriesExhausted(CloudProviderError):
    """Every bounded attempt failed.

    A typed infrastructure failure — never an empty result that a caller could
    mistake for a valid reply. ``category`` is the last attempt's category so the
    taxonomy stays stable; the schedule facts live in the attributes and in the
    client's ``retry_summary()``.
    """

    def __init__(
        self,
        *,
        attempts: int,
        retries: int,
        backoff_seconds: Iterable[float],
        total_scheduled_backoff_seconds: float,
        error_categories: Iterable[str],
        model: str | None = None,
        last_error: BaseException | None = None,
    ) -> None:
        categories = tuple(error_categories)
        last = categories[-1] if categories else CATEGORY_PROVIDER
        super().__init__(
            f"{attempts} attempts exhausted after "
            f"{total_scheduled_backoff_seconds:.0f}s of scheduled backoff "
            f"(categories: {', '.join(categories) or 'none'})",
            model=model,
            original=last_error,
        )
        self.category = last
        self.attempts = attempts
        self.retries = retries
        self.backoff_seconds = tuple(backoff_seconds)
        self.total_scheduled_backoff_seconds = total_scheduled_backoff_seconds
        self.error_categories = categories

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "attempts": self.attempts,
            "retries": self.retries,
            "backoff_seconds": list(self.backoff_seconds),
            "total_scheduled_backoff_seconds": self.total_scheduled_backoff_seconds,
            "error_categories": list(self.error_categories),
            "exhausted": True,
        }
