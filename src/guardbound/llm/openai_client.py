"""OpenAI cloud backend (Chat Completions).

Integration choices, verified against the current OpenAI documentation and the
official reference implementation
(``nbf_original_stuff/.../NBF-LLM/attacks/utils/generate.py``):

* **Chat Completions, not the Responses API.**  The official code calls
  ``client.chat.completions.create`` and the project-wide ``ChatLLM`` contract is
  a ``messages`` list of ``{role, content}`` dicts.  Chat Completions accepts that
  representation directly, so the existing interface is preserved instead of
  being translated to ``input``/``text.format``.
* **JSON mode by default.**  The official code sends
  ``response_format={"type": "json_object"}`` when ``json_format=True`` and
  ``{"type": "text"}`` otherwise.  That is preserved exactly.  Structured
  Outputs (``json_schema``) is available as an explicit opt-in
  (``structured_output="json_schema"``) because it is a *different* generation
  contract than the paper used; it is never enabled implicitly.
* **No output cap unless configured.**  The official code sends no
  ``max_tokens``.  ``max_output_tokens`` is supported for operators who want a
  hard cap, and is absent by default so paper-faithful runs are unchanged.
* **Reasoning models** (``o1``/``o3`` ids) reject a custom ``temperature`` and
  ``response_format``; the official code omits both, and so does this client.

Parameter mapping (local HF parameter -> OpenAI request field)
-------------------------------------------------------------
    temperature        -> temperature                 (omitted for o1/o3)
    top_p              -> top_p                       (only when configured)
    max_new_tokens     -> max_tokens                  (only when configured;
                                                       official sends nothing)
    json_format=True   -> response_format json_object  (or json_schema, opt-in)
    json_format=False  -> response_format text         (official parity)
    messages           -> messages                     (unchanged order)

Local-only parameters have **no** cloud equivalent (``top_k``, ``do_sample``,
``quantization``, ``chat_template_kwargs``, ``revision``, stopping criteria).
They are rejected by :func:`guardbound.llm.provider_factory.build_role_llm`
rather than silently dropped.

Reliability contract
--------------------
Source-verified retry semantics of the author's release
(``attacks/utils/generate.py`` lines 127-152, inspected in Phase 31):

* ``for ind_ in range(10)`` → ten attempts, indexed ``ind_ = 0 … 9``;
* ``time.sleep(2 * ind_ + 1)`` is the **last statement of the ``except``
  handler** (line 151), i.e. it runs after *every* failed attempt — including the
  tenth — and never after a successful one;
* ``return {}`` (line 152) is a sibling of the loop, so it is reached only after
  the tenth failure, and only after waiting a further 19 s.

Consequently the reference schedule is ``1, 3, …, 19`` and the two paths differ:

    success on attempt 10 → 10 attempts,  9 waits,  81 s scheduled
    failure after attempt 10 → 10 attempts, 10 waits, 100 s scheduled, returns {}

A second, easily-missed property of that handler: it prints
``response.choices[0].message.content`` *before* sleeping.  On the first attempt
``response`` is still unbound, so an exception raised by
``chat.completions.create()`` on attempt 1 escapes the handler as
``UnboundLocalError`` and the loop never retries.  The reference therefore retries
post-response failures (including ``json.loads`` errors) but **not** a first-attempt
transport failure.

This client reproduces the attempt count and the success-path backoff exactly
(nine waits, 1 s … 17 s, 81 s) and keeps the SDK's own hidden retry loop disabled
(``max_retries=0``).  Four deliberate, documented differences remain (see
:func:`retry_policy_record`, published in every run manifest):

* non-retryable categories (authentication, permission, invalid request, context
  length, content filter) fail immediately instead of burning the schedule;
* exhaustion raises a typed
  :class:`~guardbound.llm.cloud_errors.RetriesExhausted` instead of returning
  ``{}``, so a failure can never be mistaken for a reply and the surrounding run
  can record it as an infrastructure failure instead of crashing;
* no wait is scheduled *after* the final failed attempt (the reference waits 19 s
  and then discards the result, so the wait changes nothing except wall-clock);
* a malformed JSON reply is not regenerated: it is surfaced as a typed
  ``json_parse_error`` outcome, so one logical call consumes exactly one sampling
  draw set instead of silently re-sampling at the configured temperature.

The applied policy is published as
``official_retry_schedule + Guardbound_typed_failure_handling``.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Iterable, Sequence

from ..logging_utils import get_logger
from .base import ChatLLM, DEFAULT_TEMPERATURE, Message
from .paid_authorization import require_construction_authorization
from .cloud_errors import (
    NON_RETRYABLE_CATEGORIES,
    CloudAuthError,
    CloudProviderError,
    RetriesExhausted,
    classify_exception,
    describe_api_key,
    redact,
)
from .usage import (
    COST_BASIS_UNAVAILABLE,
    OUTCOME_CONTENT_FILTER,
    OUTCOME_EMPTY,
    OUTCOME_ERROR,
    OUTCOME_JSON_PARSE_ERROR,
    OUTCOME_OK,
    OUTCOME_REFUSAL,
    OUTCOME_TRUNCATED,
    UsageLedger,
    UsageRecord,
    estimate_cost,
)

logger = get_logger(__name__)

CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"

# Structured-output modes. ``json_object`` is the official reference behaviour.
STRUCTURED_OUTPUT_JSON_OBJECT = "json_object"
STRUCTURED_OUTPUT_JSON_SCHEMA = "json_schema"
STRUCTURED_OUTPUT_NONE = "none"
STRUCTURED_OUTPUT_MODES = (
    STRUCTURED_OUTPUT_JSON_OBJECT,
    STRUCTURED_OUTPUT_JSON_SCHEMA,
    STRUCTURED_OUTPUT_NONE,
)

# --------------------------------------------------------------------------- #
# Retry schedule — the author's release, as inspected (Phase 31)
# --------------------------------------------------------------------------- #
# attacks/utils/generate.py:
#   129  for ind_ in range(10):
#   130      try: ...
#   148      except Exception as e:
#   149          print("Error in generate: ", e)
#   150          print("generate: ", response.choices[0].message.content)
#   151          time.sleep(2*ind_+1)      <- inside the handler
#   152  return {}                        <- after the loop
#
# So the expression `2*ind_+1` is evaluated for ind_ = 0..9: ten possible waits.
# A successful call can only ever perform nine of them (the tenth attempt returns
# instead of raising); exhaustion performs all ten, the last one *after* the tenth
# failure, and only then returns {}.
RETRY_POLICY_NAME = (
    "official_retry_schedule + Guardbound_typed_failure_handling"
)
RETRY_POLICY_VERSION = "phase31"

OFFICIAL_RETRY_ATTEMPTS = 10
#: ``2 * ind_ + 1`` for ``ind_ = 0 … 9`` — one entry per *failed* attempt.
OFFICIAL_RETRY_BACKOFF_SECONDS: tuple[float, ...] = tuple(
    2.0 * index + 1.0 for index in range(OFFICIAL_RETRY_ATTEMPTS)
)
#: Waits a *successful* call can perform: the tenth attempt returns, never waits.
OFFICIAL_RETRY_SUCCESS_BACKOFF_SECONDS: tuple[float, ...] = (
    OFFICIAL_RETRY_BACKOFF_SECONDS[:-1]
)
#: Waits the reference performs when every attempt fails (includes the 19 s wait
#: that follows the tenth failure).
OFFICIAL_RETRY_EXHAUSTION_BACKOFF_SECONDS: tuple[float, ...] = (
    OFFICIAL_RETRY_BACKOFF_SECONDS
)
#: Preserved name/meaning: the most a *successful* call can be made to wait.
OFFICIAL_RETRY_TOTAL_BACKOFF_SECONDS = sum(OFFICIAL_RETRY_SUCCESS_BACKOFF_SECONDS)
OFFICIAL_RETRY_EXHAUSTION_TOTAL_BACKOFF_SECONDS = sum(
    OFFICIAL_RETRY_EXHAUSTION_BACKOFF_SECONDS
)

# The applied policy: the success-path schedule; no wait after the final failure.
APPLIED_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (
    OFFICIAL_RETRY_SUCCESS_BACKOFF_SECONDS
)
APPLIED_RETRY_SLEEPS_AFTER_FINAL_FAILURE = False

RETRY_SOURCE_FILE = (
    "nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/utils/"
    "generate.py"
)


def retry_policy_record() -> dict[str, Any]:
    """Official (source-verified) and applied retry semantics, for manifests.

    The official half is transcribed from the author's ``generate()`` body, not
    from a summary; :func:`tests.test_phase31_retry_semantics` re-executes those
    source bytes to prove the numbers.  The applied half is what this client
    actually does.  The two are deliberately reported side by side so a reader can
    never mistake the applied policy for exact equivalence.
    """
    return {
        "policy_name": RETRY_POLICY_NAME,
        "policy_version": RETRY_POLICY_VERSION,
        "paper": "the paper specifies no retry policy",
        "official_implementation": {
            "file": RETRY_SOURCE_FILE,
            "function": "generate",
            "lines": "127-152",
            "loop": "for ind_ in range(10)",
            "sleep_statement": "time.sleep(2 * ind_ + 1)  # line 151, inside `except`",
            "exhaustion_return": "return {}  # line 152, after the loop",
            "attempts": OFFICIAL_RETRY_ATTEMPTS,
            "backoff_seconds": list(OFFICIAL_RETRY_BACKOFF_SECONDS),
            "success_path_backoff_seconds": list(
                OFFICIAL_RETRY_SUCCESS_BACKOFF_SECONDS
            ),
            "success_path_total_backoff_seconds": OFFICIAL_RETRY_TOTAL_BACKOFF_SECONDS,
            "exhaustion_path_backoff_seconds": list(
                OFFICIAL_RETRY_EXHAUSTION_BACKOFF_SECONDS
            ),
            "exhaustion_path_total_backoff_seconds": (
                OFFICIAL_RETRY_EXHAUSTION_TOTAL_BACKOFF_SECONDS
            ),
            "sleeps_after_final_failed_attempt": True,
            "retried_errors": (
                "any exception raised inside the try block — including "
                "json.loads failures and response-processing errors"
            ),
            "first_attempt_exception": (
                "not retried: the handler prints `response.choices[...]` before "
                "sleeping, and `response` is unbound on attempt 1, so the handler "
                "itself raises UnboundLocalError and leaves generate()"
            ),
            "json_parse_failures_retried": True,
            "after_final_attempt": (
                "returns {} -> the caller indexes it (data['generatedQuestion'] or "
                "data['score']) and raises KeyError, aborting the whole run"
            ),
        },
        "applied_guardbound": {
            "attempts": OFFICIAL_RETRY_ATTEMPTS,
            "backoff_seconds": list(APPLIED_RETRY_BACKOFF_SECONDS),
            "total_scheduled_backoff_seconds": OFFICIAL_RETRY_TOTAL_BACKOFF_SECONDS,
            "sleeps_after_final_failed_attempt": (
                APPLIED_RETRY_SLEEPS_AFTER_FINAL_FAILURE
            ),
            "transport": "OpenAIChatLLM(..., max_retries=0), schedule in generate()",
            "non_retryable_categories": list(NON_RETRYABLE_CATEGORIES),
            "json_parse_failures_retried": False,
            "first_attempt_transport_error": (
                "retried (no unbound-variable dereference in this implementation)"
            ),
            "after_final_attempt": (
                "typed RetriesExhausted -> the goal is recorded as "
                "infrastructure_failure and the run continues"
            ),
        },
        "differences": [
            "non-retryable categories fail after one attempt instead of retrying",
            "exhaustion raises a typed error instead of returning {} (which would "
            "crash the caller with a KeyError)",
            "no wait is scheduled after the final failed attempt (19 s saved; the "
            "reference discards the result immediately afterwards)",
            "replies that fail JSON parsing are surfaced as a typed outcome and not "
            "regenerated, so one call = one set of sampling draws",
            "a first-attempt transport failure is retried (the reference aborts with "
            "an UnboundLocalError from inside its own handler)",
        ],
        "status": "ALIGNED WITH DOCUMENTED DEVIATIONS",
    }


def is_reasoning_model(model_id: str) -> bool:
    """True for the o-series ids that reject ``temperature``/``response_format``.

    Uses the official implementation's substring test (``'o1' in model or 'o3'
    in model``) so the two code paths agree.
    """
    return "o1" in model_id or "o3" in model_id


class OpenAIChatLLM(ChatLLM):
    """``ChatLLM`` backed by the official OpenAI Python SDK.

    Parameters
    ----------
    model_id:
        Provider model identifier, e.g. ``gpt-4o-mini`` or
        ``gpt-4o-2024-08-06``.
    api_key_env:
        Environment variable holding the key. The key is read at client-build
        time and is never logged, stored, or echoed.
    role:
        ``attacker`` / ``target`` / ``evaluator``, used for cost attribution.
    ledger:
        Shared :class:`UsageLedger`. When omitted, a local ledger is created so
        ``last_request_stats`` still works for one-off calls.
    structured_output:
        ``json_object`` (default, official), ``json_schema`` (opt-in), or
        ``none`` (send no ``response_format``).
    json_schema:
        JSON Schema used when ``structured_output="json_schema"``.
    client:
        Pre-built SDK client, or a test double. Injecting one bypasses the
        API-key check entirely, which is how the offline tests run without a key.
    """

    name = "openai"

    def __init__(
        self,
        model_id: str,
        api_key_env: str = "OPENAI_API_KEY",
        *,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        timeout_s: float | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        structured_output: str | None = None,
        json_schema: dict[str, Any] | None = None,
        json_schema_name: str = "guardbound_output",
        role: str = "",
        ledger: UsageLedger | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        supports_temperature: bool | None = None,
        retry_attempts: int = OFFICIAL_RETRY_ATTEMPTS,
        retry_backoff: Sequence[float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        if not model_id:
            raise ValueError("model_id is required for OpenAIChatLLM")
        # Phase 45: the lowest common enforcement point. R1 paid construction
        # requires an active authorization context opened by the audited
        # builders; this runs before any key or client access.
        require_construction_authorization(model_id)
        mode = (structured_output or STRUCTURED_OUTPUT_JSON_OBJECT).strip().lower()
        if mode not in STRUCTURED_OUTPUT_MODES:
            raise ValueError(
                f"unsupported structured_output {structured_output!r}; "
                f"supported: {list(STRUCTURED_OUTPUT_MODES)}"
            )
        if mode == STRUCTURED_OUTPUT_JSON_SCHEMA and not json_schema:
            raise ValueError(
                "structured_output='json_schema' requires json_schema"
            )

        self.model_id = model_id
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.organization = organization
        self.project = project
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.structured_output = mode
        self.json_schema = json_schema
        self.json_schema_name = json_schema_name
        self.role = role
        self.ledger = ledger or UsageLedger()
        self.supports_temperature = (
            (not is_reasoning_model(model_id))
            if supports_temperature is None
            else bool(supports_temperature)
        )

        self._client = client
        self._explicit_api_key = api_key
        self.last_request_stats: dict[str, Any] = {}

        if int(retry_attempts) < 1:
            raise ValueError("retry_attempts must be >= 1")
        self.retry_attempts = int(retry_attempts)
        backoff = tuple(
            float(value) for value in (
                retry_backoff
                if retry_backoff is not None
                else APPLIED_RETRY_BACKOFF_SECONDS
            )
        )
        if len(backoff) < self.retry_attempts - 1:
            raise ValueError(
                f"retry_backoff needs {self.retry_attempts - 1} waits for "
                f"{self.retry_attempts} attempts, got {len(backoff)}"
            )
        self.retry_backoff = backoff
        self._sleep = sleep_fn or time.sleep
        self._retry_stats: dict[str, Any] = {
            "calls": 0,
            "attempts": 0,
            "retries": 0,
            "scheduled_backoff_seconds": 0.0,
            "elapsed_backoff_seconds": 0.0,
            "successful_calls": 0,
            "exhausted_calls": 0,
            "non_retryable_calls": 0,
            "by_category": {},
        }

    def retry_summary(self) -> dict[str, Any]:
        """Cumulative retry telemetry for manifests and per-goal attribution.

        ``attempts`` / ``actual_attempts`` are *cumulative across calls*; the
        per-call attempt count of the most recent call lives in
        ``last_request_stats["retry"]["actual_attempts"]``.  The official and
        applied exhaustion totals are both reported: a successful call can never
        schedule more than 81 s, and Guardbound never schedules the reference's
        post-final-attempt 19 s.
        """
        stats = {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in self._retry_stats.items()
        }
        schedule = list(self.retry_backoff[: self.retry_attempts - 1])
        return {
            "policy": RETRY_POLICY_NAME,
            "policy_version": RETRY_POLICY_VERSION,
            "model": self.model_id,
            "role": self.role,
            "max_attempts": self.retry_attempts,
            "backoff_schedule": schedule,
            "max_scheduled_backoff_seconds": float(sum(schedule)),
            "official_success_total_backoff_seconds": (
                OFFICIAL_RETRY_TOTAL_BACKOFF_SECONDS
            ),
            "official_exhaustion_total_backoff_seconds": (
                OFFICIAL_RETRY_EXHAUSTION_TOTAL_BACKOFF_SECONDS
            ),
            "applied_exhaustion_total_backoff_seconds": float(sum(schedule)),
            "sleeps_after_final_failed_attempt": (
                APPLIED_RETRY_SLEEPS_AFTER_FINAL_FAILURE
            ),
            **stats,
            "actual_attempts": stats["attempts"],
            "error_categories": sorted(stats["by_category"]),
        }

    def _call_retry_record(
        self,
        *,
        attempt: int,
        scheduled_backoff: float,
        elapsed_backoff: float,
        categories: Sequence[str],
        succeeded: bool,
        exhausted: bool,
        terminal_category: str | None,
    ) -> dict[str, Any]:
        """Per-call retry facts, distinguishing success from exhaustion.

        ``scheduled_backoff_seconds`` is what the policy *asked* for;
        ``elapsed_backoff_seconds`` is how much wall-clock the sleeps actually
        consumed (they differ when a sleep is interrupted or the injected sleep
        returns early).  They are never treated as interchangeable.
        """
        return {
            # Phase 30 keys, preserved for existing manifest consumers.
            "attempts": attempt,
            "retries": attempt - 1,
            "scheduled_backoff_seconds": scheduled_backoff,
            "error_categories": list(categories),
            # Phase 31 additions.
            "policy": RETRY_POLICY_NAME,
            "policy_version": RETRY_POLICY_VERSION,
            "max_attempts": self.retry_attempts,
            "actual_attempts": attempt,
            "backoff_schedule": list(self.retry_backoff[: max(attempt - 1, 0)]),
            "elapsed_backoff_seconds": elapsed_backoff,
            "succeeded": succeeded,
            "exhausted": exhausted,
            "terminal_category": terminal_category,
            "official_exhaustion_total_backoff_seconds": (
                OFFICIAL_RETRY_EXHAUSTION_TOTAL_BACKOFF_SECONDS
            ),
            "sleeps_after_final_failed_attempt": (
                APPLIED_RETRY_SLEEPS_AFTER_FINAL_FAILURE
            ),
        }

    # ------------------------------------------------------------------ #
    # Client construction
    # ------------------------------------------------------------------ #

    def _resolve_api_key(self) -> str | None:
        if self._explicit_api_key:
            return self._explicit_api_key
        return os.environ.get(self.api_key_env)

    def _get_client(self):
        """Build (once) and return the SDK client.

        Raises a classified ``CloudAuthError`` when the key is missing, instead
        of letting the SDK raise a message that can vary by version.
        """
        if self._client is not None:
            return self._client

        try:
            import openai
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "The 'openai' package is required for OpenAIChatLLM. "
                "Install it with: pip install openai"
            ) from exc

        api_key = self._resolve_api_key()
        if not api_key:
            raise CloudAuthError(
                f"{self.api_key_env} is not set, so the OpenAI provider cannot "
                f"authenticate (key status: {describe_api_key(None)}). Set the "
                f"environment variable or switch this role to the local "
                f"provider.",
                model=self.model_id,
            )

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            # Research protocol: no hidden SDK retries.
            "max_retries": 0,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.organization:
            kwargs["organization"] = self.organization
        if self.project:
            kwargs["project"] = self.project
        if self.timeout_s is not None:
            kwargs["timeout"] = self.timeout_s

        try:
            self._client = openai.OpenAI(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalise to our taxonomy
            raise classify_exception(
                exc, model=self.model_id, secrets=(api_key,)
            ) from exc
        return self._client

    # ------------------------------------------------------------------ #
    # Request shaping
    # ------------------------------------------------------------------ #

    @staticmethod
    def _trim(messages: list[Message], max_turns_context: int | None) -> list[Message]:
        if max_turns_context is None or max_turns_context <= 0:
            return messages
        system_msgs = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        return system_msgs + non_system[-(max_turns_context * 2):]

    def _response_format(self, json_format: bool) -> dict[str, Any] | None:
        if self.structured_output == STRUCTURED_OUTPUT_NONE:
            return None
        if json_format:
            if self.structured_output == STRUCTURED_OUTPUT_JSON_SCHEMA:
                return {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.json_schema_name,
                        "strict": True,
                        "schema": self.json_schema,
                    },
                }
            return {"type": "json_object"}
        # Official parity: the reference implementation explicitly requests text.
        return {"type": "text"}

    def build_request(
        self,
        messages: list[Message],
        temperature: float = DEFAULT_TEMPERATURE,
        json_format: bool = False,
    ) -> dict[str, Any]:
        """The exact kwargs sent to ``chat.completions.create``.

        Exposed so tests and the qualification script can assert the mapping
        without a network call, and so a run manifest can record it.
        """
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
        }
        if self.supports_temperature:
            kwargs["temperature"] = temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens
        if self.supports_temperature:
            response_format = self._response_format(json_format)
            if response_format is not None:
                kwargs["response_format"] = response_format
        return kwargs

    def parameter_map(self, json_format: bool = False) -> dict[str, Any]:
        """Which local generation parameters were mapped, dropped, or unused."""
        response_format = (
            self._response_format(json_format) if self.supports_temperature else None
        )
        return {
            "temperature": "temperature" if self.supports_temperature else "omitted (reasoning model)",
            "top_p": "top_p" if self.top_p is not None else "unset",
            "max_new_tokens": (
                "max_tokens" if self.max_output_tokens is not None
                else "unset (official reference sends no output cap)"
            ),
            "json_format": (
                response_format["type"] if response_format else "omitted (reasoning model)"
            ),
            "messages": "messages",
        }

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate(
        self,
        messages: list[Message],
        temperature: float = DEFAULT_TEMPERATURE,
        max_turns_context: int | None = None,
        json_format: bool = False,
        structured_output_mode: str | None = None,
    ) -> str | dict:
        """Return the assistant reply.

        Contract (unchanged from ``base.ChatLLM``): a ``dict`` when
        ``json_format=True`` and the reply parsed cleanly, otherwise the raw
        text.  A malformed reply is returned as text — never repaired, never
        regenerated — so the attack layer can classify it as a contract failure
        distinct from an attack outcome.

        ``structured_output_mode`` is the local-backend decoding switch and has
        no cloud meaning; it is accepted for interface compatibility and
        ignored (logged at debug level, since it changes nothing here).
        """
        if structured_output_mode is not None:
            logger.debug(
                "OpenAIChatLLM ignores structured_output_mode=%r "
                "(local decoding switch, no cloud equivalent)",
                structured_output_mode,
            )

        messages = self._trim(messages, max_turns_context)
        ledger = self.ledger

        client = self._get_client()
        kwargs = self.build_request(messages, temperature, json_format)

        started = time.perf_counter()
        self._retry_stats["calls"] += 1
        attempt = 0
        categories: list[str] = []
        scheduled_backoff = 0.0
        elapsed_backoff = 0.0
        while True:
            attempt += 1
            self._retry_stats["attempts"] += 1
            try:
                response = client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - re-raised as classified error
                error = classify_exception(
                    exc,
                    model=self.model_id,
                    secrets=(self._resolve_api_key(),),
                )
                categories.append(error.category)
                self._retry_stats["by_category"][error.category] = (
                    self._retry_stats["by_category"].get(error.category, 0) + 1
                )
                if (error.category not in NON_RETRYABLE_CATEGORIES
                        and attempt < self.retry_attempts):
                    wait = float(self.retry_backoff[attempt - 1])
                    scheduled_backoff += wait
                    self._retry_stats["retries"] += 1
                    # Accumulated immediately so an interrupt during the wait is
                    # still represented in the telemetry (Phase 30).
                    self._retry_stats["scheduled_backoff_seconds"] += wait
                    logger.warning(
                        "OpenAI %s attempt %d/%d failed (%s); retrying in %.0fs",
                        self.model_id, attempt, self.retry_attempts,
                        error.category, wait,
                    )
                    wait_started = time.perf_counter()
                    self._sleep(wait)
                    waited = time.perf_counter() - wait_started
                    elapsed_backoff += waited
                    self._retry_stats["elapsed_backoff_seconds"] += waited
                    continue

                # Terminal: exactly one error usage row for the logical call.
                duration = time.perf_counter() - started
                if (attempt > 1
                        and attempt >= self.retry_attempts
                        and error.category not in NON_RETRYABLE_CATEGORIES):
                    # Typed exhaustion only when retries were actually attempted;
                    # with a single-attempt policy the classified error is the
                    # truthful signal (no retry was exhausted).
                    self._retry_stats["exhausted_calls"] += 1
                    terminal: CloudProviderError = RetriesExhausted(
                        attempts=attempt,
                        retries=attempt - 1,
                        backoff_seconds=self.retry_backoff[: attempt - 1],
                        total_scheduled_backoff_seconds=scheduled_backoff,
                        error_categories=categories,
                        model=self.model_id,
                        last_error=exc,
                    )
                else:
                    self._retry_stats["non_retryable_calls"] += 1
                    terminal = error

                record = UsageRecord(
                    provider="openai",
                    model=self.model_id,
                    role=self.role,
                    endpoint=CHAT_COMPLETIONS_ENDPOINT,
                    duration_s=round(duration, 6),
                    outcome=OUTCOME_ERROR,
                    error_category=terminal.category,
                    error_message=str(terminal),
                    cost_basis=COST_BASIS_UNAVAILABLE,
                    temperature=temperature,
                    json_format=json_format,
                )
                ledger.record(record)
                self.last_request_stats = {
                    **record.to_dict(),
                    "error": terminal.to_dict(),
                    "raw_reply": None,
                    "parameter_map": self.parameter_map(json_format),
                    "retry": self._call_retry_record(
                        attempt=attempt,
                        scheduled_backoff=scheduled_backoff,
                        elapsed_backoff=elapsed_backoff,
                        categories=categories,
                        succeeded=False,
                        exhausted=isinstance(terminal, RetriesExhausted),
                        terminal_category=terminal.category,
                    ),
                }
                logger.warning(
                    "OpenAI %s call failed (%s) after %d attempt(s) and %.3fs",
                    self.model_id, terminal.category, attempt, duration,
                )
                raise terminal from exc

        self._retry_stats["successful_calls"] += 1
        duration = time.perf_counter() - started
        content, usage, telemetry = self._extract(
            response, json_format=json_format
        )

        rate = ledger.rate_card.rate_for(self.model_id)
        # No usage object means the token counts are unknown, not zero: an
        # estimate of $0.00 would silently understate spend, so report "unknown".
        cost = (
            estimate_cost(
                rate,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cached_tokens=usage["cached_tokens"],
            )
            if usage["usage_available"]
            else None
        )
        record = UsageRecord(
            provider="openai",
            model=self.model_id,
            role=self.role,
            endpoint=CHAT_COMPLETIONS_ENDPOINT,
            input_tokens=usage["input_tokens"],
            cached_tokens=usage["cached_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            duration_s=round(duration, 6),
            request_id=telemetry["request_id"],
            finish_reason=telemetry["finish_reason"],
            refusal=telemetry["refusal"],
            outcome=telemetry["outcome"],
            estimated_cost_usd=cost,
            cost_basis=ledger.rate_card.cost_basis if cost is not None else COST_BASIS_UNAVAILABLE,
            temperature=temperature,
            json_format=json_format,
            usage_available=usage["usage_available"],
        )
        ledger.record(record)

        self.last_request_stats = {
            **record.to_dict(),
            "raw_reply": telemetry["raw_reply"],
            "json_parse_ok": telemetry["json_parse_ok"],
            "reasoning_tokens": usage["reasoning_tokens"],
            "parameter_map": self.parameter_map(json_format),
            "response_format": (
                kwargs.get("response_format", {}).get("type")
                if isinstance(kwargs.get("response_format"), dict) else None
            ),
            "retry": self._call_retry_record(
                attempt=attempt,
                scheduled_backoff=scheduled_backoff,
                elapsed_backoff=elapsed_backoff,
                categories=categories,
                succeeded=True,
                exhausted=False,
                terminal_category=None,
            ),
        }

        logger.debug(
            "OpenAI %s replied (%d chars, %d in / %d out, %.3fs)",
            self.model_id, len(content or ""),
            usage["input_tokens"], usage["output_tokens"], duration,
        )
        return content

    # ------------------------------------------------------------------ #
    # Response extraction
    # ------------------------------------------------------------------ #

    def _extract(
        self,
        response: Any,
        *,
        json_format: bool,
    ) -> tuple[str | dict, dict[str, Any], dict[str, Any]]:
        """Pull text/usage out of an SDK response without repairing anything."""
        usage_obj = getattr(response, "usage", None)
        details = getattr(usage_obj, "prompt_tokens_details", None)
        completion_details = getattr(usage_obj, "completion_tokens_details", None)
        input_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage_obj, "total_tokens", 0) or 0)
        if not total_tokens:
            total_tokens = input_tokens + output_tokens
        cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
        reasoning_tokens = int(getattr(completion_details, "reasoning_tokens", 0) or 0)

        choices = getattr(response, "choices", None) or []
        if not choices:
            # A 200 with no choices is a contract failure, not an empty answer.
            return "", {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached_tokens,
                "total_tokens": total_tokens,
                "reasoning_tokens": reasoning_tokens,
                "usage_available": usage_obj is not None,
            }, {
                "request_id": getattr(response, "id", None),
                "finish_reason": None,
                "refusal": False,
                "raw_reply": "",
                "json_parse_ok": False,
                "outcome": OUTCOME_EMPTY,
            }

        choice = choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        refusal_text = getattr(message, "refusal", None)
        finish_reason = getattr(choice, "finish_reason", None)
        request_id = getattr(response, "id", None)

        raw_reply = content if content is not None else ""
        refusal = bool(refusal_text) or finish_reason == "content_filter"

        if refusal and not raw_reply:
            # A refusal is a legitimate model output. It is surfaced as text so
            # the caller sees exactly what happened; nothing is fabricated.
            raw_reply = refusal_text or ""

        if finish_reason == "content_filter":
            outcome = OUTCOME_CONTENT_FILTER
        elif refusal:
            outcome = OUTCOME_REFUSAL
        elif not str(raw_reply).strip():
            outcome = OUTCOME_EMPTY
        elif finish_reason == "length":
            outcome = OUTCOME_TRUNCATED
        else:
            outcome = OUTCOME_OK

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
            # A 200 with no usage object cannot be priced: it is recorded as
            # unknown (cost stays None) so the reported total is a lower bound.
            "usage_available": usage_obj is not None,
        }

        json_parse_ok = False
        result: str | dict = raw_reply
        if json_format and raw_reply:
            if isinstance(raw_reply, str):
                try:
                    parsed = json.loads(raw_reply)
                    result = parsed
                    json_parse_ok = True
                except json.JSONDecodeError:
                    json_parse_ok = False
                    if outcome == OUTCOME_OK:
                        outcome = OUTCOME_JSON_PARSE_ERROR
                    logger.warning(
                        "OpenAI %s returned non-JSON for json_format=True: %s",
                        self.model_id, redact(str(raw_reply)[:200]),
                    )
            else:
                result = raw_reply
                json_parse_ok = True

        return result, usage, {
            "request_id": request_id,
            "finish_reason": finish_reason,
            "refusal": refusal,
            "raw_reply": raw_reply,
            "json_parse_ok": json_parse_ok,
            "outcome": outcome,
        }

    def __repr__(self) -> str:
        return (
            f"OpenAIChatLLM(model_id={self.model_id!r}, "
            f"role={self.role!r}, structured_output={self.structured_output!r})"
        )
