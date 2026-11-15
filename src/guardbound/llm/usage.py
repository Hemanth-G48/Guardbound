"""Usage and cost accounting for cloud providers (observational only).

One ``UsageRecord`` is written for *every* provider call — successful or not —
so cost, latency and failure categories can be attributed per role
(attacker / target / evaluator) and per model, which is what the paper's
experiment protocol needs to report.

Design notes
------------
* Prices are **never** hardcoded here. They come from a ``RateCard`` loaded from
  a YAML file (``configs/cloud_pricing.yaml``) so the operator can update them
  without touching code and so the version/dating is visible in every report.
* A model with no rate yields ``estimated_cost_usd = None`` and
  ``cost_basis = "unavailable"``. It is never guessed as zero.
* This module **does not limit spending**. Phase 22 removed the custom budget
  cap: the paper specifies no API budget, the author's release has none, and a
  research run must not be blocked by a number Guardbound invented. Usage
  numbers are reported, never enforced.
* Failures are recorded with their category and cost nothing on the provider
  side only when the request never reached it; a failure that carries usage
  (e.g. a truncated 200 response) is still billed by the provider and is
  recorded as such.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..logging_utils import get_logger
from .cloud_errors import redact

logger = get_logger(__name__)

COST_BASIS_RATE_CARD = "rate_card"
COST_BASIS_UNVERIFIED = "unverified_rate_card"
COST_BASIS_UNAVAILABLE = "unavailable"

# Outcome vocabulary for a single provider call. Contract failures are kept
# distinct from provider failures and from attack outcomes.
OUTCOME_OK = "ok"
OUTCOME_EMPTY = "empty_response"
OUTCOME_REFUSAL = "refusal"
OUTCOME_JSON_PARSE_ERROR = "json_parse_error"
OUTCOME_TRUNCATED = "truncated"
OUTCOME_CONTENT_FILTER = "content_filter"
OUTCOME_ERROR = "error"

ALL_OUTCOMES = (
    OUTCOME_OK,
    OUTCOME_EMPTY,
    OUTCOME_REFUSAL,
    OUTCOME_JSON_PARSE_ERROR,
    OUTCOME_TRUNCATED,
    OUTCOME_CONTENT_FILTER,
    OUTCOME_ERROR,
)


# --------------------------------------------------------------------------- #
# Rate card
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelRate:
    """Prices in USD per 1,000,000 tokens for one model id."""

    model: str
    input_per_1m: float | None = None
    cached_input_per_1m: float | None = None
    output_per_1m: float | None = None
    # Published retirement date (YYYY-MM-DD) when the provider has announced
    # one. Carried so a run can warn that its reproducibility window is finite.
    shutdown_date: str | None = None

    @property
    def usable(self) -> bool:
        return self.input_per_1m is not None and self.output_per_1m is not None


@dataclass(frozen=True)
class RateCard:
    """A versioned, dated price table.

    ``verified`` records whether the operator confirmed the numbers against the
    provider's official pricing page. Unverified rates still produce an estimate,
    but the estimate is labelled ``unverified_rate_card`` everywhere it appears.
    """

    models: dict[str, ModelRate] = field(default_factory=dict)
    as_of: str = ""
    source: str = ""
    verified: bool = False
    version: str = ""

    def rate_for(self, model: str) -> ModelRate | None:
        if model in self.models:
            return self.models[model]
        # Fall back to a dated-snapshot prefix match ("gpt-4o-mini" for
        # "gpt-4o-mini-2024-07-18"). Longest prefix wins so a more specific
        # entry is never shadowed by a shorter one.
        candidates = [
            key for key in self.models if model.startswith(key)
        ]
        if not candidates:
            return None
        return self.models[max(candidates, key=len)]

    @property
    def cost_basis(self) -> str:
        return COST_BASIS_RATE_CARD if self.verified else COST_BASIS_UNVERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "as_of": self.as_of,
            "source": self.source,
            "verified": self.verified,
            "models": {k: asdict(v) for k, v in self.models.items()},
        }


EMPTY_RATE_CARD = RateCard()


def load_rate_card(path: str | Path | None) -> RateCard:
    """Load a rate card from YAML. A missing/None path yields an empty card."""
    if path is None:
        return EMPTY_RATE_CARD
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Rate card not found: {path}. Create it from "
            f"configs/cloud_pricing.yaml, or omit --rate-card to run without "
            f"cost estimation."
        )
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    models: dict[str, ModelRate] = {}
    for model_id, entry in (raw.get("models") or {}).items():
        entry = entry or {}
        models[str(model_id)] = ModelRate(
            model=str(model_id),
            input_per_1m=_maybe_float(entry.get("input_per_1m")),
            cached_input_per_1m=_maybe_float(entry.get("cached_input_per_1m")),
            output_per_1m=_maybe_float(entry.get("output_per_1m")),
            shutdown_date=(
                str(entry["shutdown_date"]) if entry.get("shutdown_date") else None
            ),
        )
    return RateCard(
        models=models,
        as_of=str(raw.get("as_of", "")),
        source=str(raw.get("source", "")),
        verified=bool(raw.get("verified", False)),
        version=str(raw.get("version", "")),
    )


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def estimate_cost(
    rate: ModelRate | None,
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float | None:
    """Cost in USD, or ``None`` when the model has no usable rate.

    Cached input tokens are billed at the cached rate when one is configured and
    deducted from the full-price input count; when no cached rate is configured
    they are billed at the normal input rate (the conservative, non-understating
    choice).
    """
    if rate is None or not rate.usable:
        return None

    total_input = max(0, int(input_tokens or 0))
    # Clamp: never bill more cached tokens than the provider reported as input,
    # which keeps inconsistent usage payloads from overstating cost.
    cached = min(max(0, int(cached_tokens or 0)), total_input)
    uncached = total_input - cached

    cost = uncached * float(rate.input_per_1m) / 1_000_000
    if rate.cached_input_per_1m is not None:
        cost += cached * float(rate.cached_input_per_1m) / 1_000_000
    else:
        cost += cached * float(rate.input_per_1m) / 1_000_000
    cost += max(0, int(output_tokens or 0)) * float(rate.output_per_1m) / 1_000_000
    return round(cost, 8)


# --------------------------------------------------------------------------- #
# Usage record
# --------------------------------------------------------------------------- #

@dataclass
class UsageRecord:
    """One provider call, successful or not."""

    provider: str
    model: str
    role: str = ""
    endpoint: str = ""
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_s: float = 0.0
    request_id: str | None = None
    finish_reason: str | None = None
    refusal: bool = False
    outcome: str = OUTCOME_OK
    error_category: str | None = None
    error_message: str | None = None
    estimated_cost_usd: float | None = None
    cost_basis: str = COST_BASIS_UNAVAILABLE
    temperature: float | None = None
    json_format: bool = False
    # False when the provider answered but returned no usage object. Such a call
    # cannot be priced, so its cost stays ``None`` (never 0) and the run's total
    # is reported as a lower bound.
    usage_available: bool = True
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def ok(self) -> bool:
        return self.outcome != OUTCOME_ERROR

    @property
    def billable_tokens(self) -> int:
        return int(self.total_tokens or (self.input_tokens + self.output_tokens))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #

class UsageLedger:
    """Thread-safe accumulator for ``UsageRecord`` rows.

    One ledger is shared by every role in a run, which is what makes the
    per-role / per-model / whole-experiment split possible.
    """

    def __init__(
        self,
        rate_card: RateCard | None = None,
        secrets: Iterable[str] = (),
    ) -> None:
        self.rate_card = rate_card or EMPTY_RATE_CARD
        self._secrets = tuple(secrets)
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()

    # -- observed spend (telemetry, never a limit) ------------------------- #

    @property
    def spent_usd(self) -> float:
        """Sum of the priced calls. Reported for the manifest; nothing is capped."""
        with self._lock:
            return round(sum(
                r.estimated_cost_usd or 0.0
                for r in self._records
                if r.estimated_cost_usd is not None
            ), 8)

    @property
    def unavailable_usage_calls(self) -> int:
        """Calls whose response carried no usage object (cost unknown, not zero)."""
        with self._lock:
            return sum(1 for r in self._records if not r.usage_available)

    # -- recording --------------------------------------------------------- #

    def record(self, record: UsageRecord) -> UsageRecord:
        if record.error_message:
            record.error_message = redact(record.error_message, self._secrets)
        with self._lock:
            self._records.append(record)
        return record

    def records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    # -- reporting --------------------------------------------------------- #

    def summary(self) -> dict[str, Any]:
        """Totals split by role, by model and for the whole experiment."""
        rows = self.records()
        by_role: dict[str, dict[str, Any]] = {}
        by_model: dict[str, dict[str, Any]] = {}
        by_outcome: dict[str, int] = {}
        by_error: dict[str, int] = {}

        for row in rows:
            by_outcome[row.outcome] = by_outcome.get(row.outcome, 0) + 1
            if row.error_category:
                by_error[row.error_category] = by_error.get(row.error_category, 0) + 1
            for bucket, key in ((by_role, row.role or "unassigned"),
                                (by_model, row.model)):
                agg = bucket.setdefault(key, {
                    "calls": 0,
                    "input_tokens": 0,
                    "cached_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "duration_s": 0.0,
                    "estimated_cost_usd": 0.0,
                    "cost_unavailable_calls": 0,
                })
                agg["calls"] += 1
                agg["input_tokens"] += row.input_tokens
                agg["cached_tokens"] += row.cached_tokens
                agg["output_tokens"] += row.output_tokens
                agg["total_tokens"] += row.billable_tokens
                agg["duration_s"] = round(agg["duration_s"] + row.duration_s, 6)
                if row.estimated_cost_usd is None:
                    agg["cost_unavailable_calls"] += 1
                else:
                    agg["estimated_cost_usd"] = round(
                        agg["estimated_cost_usd"] + row.estimated_cost_usd, 8)

        total_cost = round(sum(
            r.estimated_cost_usd or 0.0 for r in rows
            if r.estimated_cost_usd is not None), 8)
        priced = sum(1 for r in rows if r.estimated_cost_usd is not None)

        return {
            "calls": len(rows),
            "failed_calls": sum(1 for r in rows if not r.ok),
            "usage_unavailable_calls": sum(1 for r in rows if not r.usage_available),
            "input_tokens": sum(r.input_tokens for r in rows),
            "cached_tokens": sum(r.cached_tokens for r in rows),
            "output_tokens": sum(r.output_tokens for r in rows),
            "total_tokens": sum(r.billable_tokens for r in rows),
            "duration_s": round(sum(r.duration_s for r in rows), 6),
            "estimated_cost_usd": total_cost,
            "priced_calls": priced,
            "unpriced_calls": len(rows) - priced,
            "cost_basis": self.rate_card.cost_basis if priced else COST_BASIS_UNAVAILABLE,
            "rate_card": self.rate_card.to_dict(),
            "by_role": by_role,
            "by_model": by_model,
            "by_outcome": by_outcome,
            "by_error_category": by_error,
        }

    def write_jsonl(self, path: str | Path) -> Path:
        """Append every record to ``path`` (one JSON object per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            for row in self.records():
                handle.write(json.dumps(row.to_dict(), default=str) + "\n")
        return path

    def write_summary(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2, default=str)
        return path


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def build_ledger(
    rate_card_path: str | Path | None = None,
    secrets: Iterable[str] = (),
) -> UsageLedger:
    return UsageLedger(
        rate_card=load_rate_card(rate_card_path),
        secrets=secrets,
    )
