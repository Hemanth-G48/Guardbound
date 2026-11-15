"""Cloud usage and cost accounting — offline only, observational (no budget cap)."""
from __future__ import annotations

import json
import threading

import pytest

from guardbound.llm.cloud_errors import (
    CATEGORY_AUTH,
    CATEGORY_CONNECTION,
    CATEGORY_CONTEXT_LENGTH,
    CATEGORY_RATE_LIMIT,
    CATEGORY_SERVER,
    CATEGORY_TIMEOUT,
    CloudAuthError,
    CloudContextLengthError,
    CloudProviderError,
    classify_exception,
    describe_api_key,
    redact,
)
from guardbound.llm.usage import (
    COST_BASIS_RATE_CARD,
    COST_BASIS_UNAVAILABLE,
    COST_BASIS_UNVERIFIED,
    ALL_OUTCOMES,
    ModelRate,
    RateCard,
    UsageLedger,
    UsageRecord,
    estimate_cost,
    load_rate_card,
)


def write_card(tmp_path, *, verified: bool = True):
    path = tmp_path / "card.yaml"
    path.write_text(
        f"""
version: "test-1"
as_of: "2026-01-01"
source: "unit-test"
verified: {str(verified).lower()}
models:
  gpt-4o-mini:
    input_per_1m: 0.15
    cached_input_per_1m: 0.075
    output_per_1m: 0.60
  gpt-4o-mini-2024-07-18:
    input_per_1m: 0.20
    cached_input_per_1m: 0.10
    output_per_1m: 0.80
  unpriced-model:
    input_per_1m: null
    cached_input_per_1m: null
    output_per_1m: null
""",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- #
# Rate card
# --------------------------------------------------------------------------- #

def test_load_rate_card_reads_models_and_provenance(tmp_path):
    card = load_rate_card(write_card(tmp_path))
    assert card.version == "test-1"
    assert card.as_of == "2026-01-01"
    assert card.source == "unit-test"
    assert card.verified is True
    assert card.cost_basis == COST_BASIS_RATE_CARD
    assert card.rate_for("gpt-4o-mini").input_per_1m == 0.15


def test_unverified_rate_card_is_labelled(tmp_path):
    card = load_rate_card(write_card(tmp_path, verified=False))
    assert card.cost_basis == COST_BASIS_UNVERIFIED


def test_missing_rate_card_raises_with_guidance(tmp_path):
    with pytest.raises(FileNotFoundError, match="Rate card not found"):
        load_rate_card(tmp_path / "nope.yaml")


def test_rate_lookup_supports_dated_snapshot_prefixes(tmp_path):
    card = load_rate_card(write_card(tmp_path))
    # Exact id wins over the shorter family prefix.
    assert card.rate_for("gpt-4o-mini-2024-07-18").input_per_1m == 0.20
    # Unknown snapshot of a known family falls back to the family rate.
    assert card.rate_for("gpt-4o-mini-2099-01-01").input_per_1m == 0.15
    assert card.rate_for("entirely-unknown") is None


def test_unpriced_model_is_not_usable(tmp_path):
    card = load_rate_card(write_card(tmp_path))
    assert card.rate_for("unpriced-model").usable is False


# --------------------------------------------------------------------------- #
# Cost estimation
# --------------------------------------------------------------------------- #

def test_cost_uses_separate_cached_input_rate():
    rate = ModelRate("m", input_per_1m=10.0, cached_input_per_1m=1.0, output_per_1m=30.0)
    # 1000 total input of which 400 cached, 500 output:
    # 600*10/1e6 + 400*1/1e6 + 500*30/1e6
    expected = 600 * 10.0 / 1e6 + 400 * 1.0 / 1e6 + 500 * 30.0 / 1e6
    assert estimate_cost(rate, input_tokens=1000, output_tokens=500,
                         cached_tokens=400) == pytest.approx(expected)


def test_cost_bills_cached_tokens_at_input_rate_when_no_cached_rate():
    rate = ModelRate("m", input_per_1m=10.0, cached_input_per_1m=None, output_per_1m=0.0)
    assert estimate_cost(rate, input_tokens=1000, output_tokens=0,
                         cached_tokens=400) == pytest.approx(1000 * 10.0 / 1e6)


def test_cost_is_none_not_zero_when_price_is_unknown():
    assert estimate_cost(None, input_tokens=100, output_tokens=100) is None
    assert estimate_cost(ModelRate("m"), input_tokens=100, output_tokens=100) is None


def test_cached_tokens_cannot_exceed_input_in_the_calculation():
    rate = ModelRate("m", input_per_1m=10.0, cached_input_per_1m=1.0, output_per_1m=0.0)
    # cached > input: the uncached portion floors at zero, never negative.
    assert estimate_cost(rate, input_tokens=100, output_tokens=0,
                         cached_tokens=500) == pytest.approx(100 * 1.0 / 1e6)


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #

def _record(role="attacker", model="gpt-4o-mini", cost=0.001, **kwargs):
    return UsageRecord(provider="openai", model=model, role=role,
                       estimated_cost_usd=cost, **kwargs)


def test_summary_splits_cost_by_role_and_model(tmp_path):
    ledger = UsageLedger(rate_card=load_rate_card(write_card(tmp_path)))
    ledger.record(_record("attacker", cost=0.001, input_tokens=100, output_tokens=20,
                          total_tokens=120))
    ledger.record(_record("target", cost=0.002, input_tokens=200, output_tokens=40,
                          total_tokens=240))
    ledger.record(_record("evaluator", cost=0.003, input_tokens=300, output_tokens=60,
                          total_tokens=360))

    summary = ledger.summary()
    assert summary["calls"] == 3
    assert summary["estimated_cost_usd"] == pytest.approx(0.006)
    assert summary["total_tokens"] == 720
    assert set(summary["by_role"]) == {"attacker", "target", "evaluator"}
    assert summary["by_role"]["attacker"]["estimated_cost_usd"] == pytest.approx(0.001)
    assert summary["by_model"]["gpt-4o-mini"]["calls"] == 3


def test_summary_counts_unpriced_calls_separately(tmp_path):
    ledger = UsageLedger(rate_card=load_rate_card(write_card(tmp_path)))
    ledger.record(_record(cost=None))
    ledger.record(_record(cost=0.001))
    summary = ledger.summary()
    assert summary["priced_calls"] == 1
    assert summary["unpriced_calls"] == 1
    assert summary["by_role"]["attacker"]["cost_unavailable_calls"] == 1


def test_summary_records_failure_categories():
    ledger = UsageLedger()
    ledger.record(UsageRecord(provider="openai", model="m", outcome="error",
                              error_category="rate_limit"))
    ledger.record(UsageRecord(provider="openai", model="m", outcome="json_parse_error"))
    summary = ledger.summary()
    assert summary["failed_calls"] == 1
    assert summary["by_error_category"] == {"rate_limit": 1}
    assert summary["by_outcome"]["json_parse_error"] == 1
    assert set(ALL_OUTCOMES) >= set(summary["by_outcome"])


def test_contract_failure_is_distinct_from_provider_failure():
    """A JSON parse failure is not counted as a failed request."""
    ledger = UsageLedger()
    ledger.record(UsageRecord(provider="openai", model="m", outcome="json_parse_error"))
    summary = ledger.summary()
    assert summary["failed_calls"] == 0
    assert summary["by_outcome"]["json_parse_error"] == 1


# --------------------------------------------------------------------------- #
# Observational spend (Phase 22 removed the custom budget cap)
# --------------------------------------------------------------------------- #

def test_ledger_has_no_spending_limit_surface():
    """The cap is gone by design: no constructor arg, no check, no remaining."""
    ledger = UsageLedger()
    assert not hasattr(ledger, "budget_usd")
    assert not hasattr(ledger, "check_budget")
    assert not hasattr(ledger, "remaining_usd")
    assert "budget_usd" not in ledger.summary()
    assert "remaining_usd" not in ledger.summary()


def test_spent_usd_is_reported_for_the_manifest():
    ledger = UsageLedger()
    ledger.record(_record(model="gpt-4o-mini", cost=0.25))
    assert ledger.spent_usd == pytest.approx(0.25)


def test_summary_counts_unpriceable_calls():
    ledger = UsageLedger()
    ledger.record(UsageRecord(provider="openai", model="m", usage_available=False))
    ledger.record(UsageRecord(provider="openai", model="m", usage_available=True))
    summary = ledger.summary()
    assert summary["usage_unavailable_calls"] == 1
    assert ledger.unavailable_usage_calls == 1


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def test_records_and_summary_round_trip_to_disk(tmp_path):
    ledger = UsageLedger(rate_card=load_rate_card(write_card(tmp_path)))
    ledger.record(_record(input_tokens=10, output_tokens=5, total_tokens=15))

    jsonl = ledger.write_jsonl(tmp_path / "usage.jsonl")
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["model"] == "gpt-4o-mini"
    assert rows[0]["input_tokens"] == 10

    summary_path = ledger.write_summary(tmp_path / "usage.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["calls"] == 1
    assert summary["rate_card"]["version"] == "test-1"


def test_ledger_is_thread_safe():
    ledger = UsageLedger()

    def worker():
        for _ in range(200):
            ledger.record(_record(cost=0.0))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(ledger.records()) == 1600


# --------------------------------------------------------------------------- #
# Redaction and classification
# --------------------------------------------------------------------------- #

def test_redact_removes_known_and_shaped_secrets():
    assert "sk-" not in redact("key sk-abc1234567890 leaked")
    assert "secretvalue" not in redact(
        "token secretvalue1234", secrets=("secretvalue1234",)
    )
    assert "Bearer" not in redact("Authorization: Bearer abcdefghijkl")
    assert redact("") == ""


def test_describe_api_key_returns_no_key_material():
    assert describe_api_key(None) == "absent"
    described = describe_api_key("sk-abcdef1234567890")
    assert "abcdef" not in described
    assert "sk-" not in described
    assert "len=" in described


def test_error_message_is_redacted_at_construction():
    error = CloudProviderError(
        "request failed for key sk-abcdef1234567890",
        secrets=("sk-abcdef1234567890",),
    )
    assert "sk-abcdef" not in str(error)
    assert "[REDACTED]" in str(error)


@pytest.mark.parametrize(
    ("exc", "category"),
    [
        (TimeoutError("read timed out"), CATEGORY_TIMEOUT),
        (ConnectionError("connection reset"), CATEGORY_CONNECTION),
        (RuntimeError("internal server error 503"), "provider_error"),
        (ValueError("This model's maximum context length is 8192 tokens"),
         CATEGORY_CONTEXT_LENGTH),
    ],
)
def test_classify_exception_by_name_and_text(exc, category):
    assert classify_exception(exc).category == category


def test_classify_exception_prefers_context_length_over_status():
    class BadRequestError(Exception):
        status_code = 400

    classified = classify_exception(BadRequestError("maximum context length exceeded"))
    assert isinstance(classified, CloudContextLengthError)


def test_classify_exception_is_idempotent():
    original = CloudAuthError("nope")
    assert classify_exception(original) is original


def test_classify_preserves_original_exception_and_status():
    class RateLimitError(Exception):
        status_code = 429
        request_id = "req-1"

    classified = classify_exception(RateLimitError("slow down"))
    assert classified.category == CATEGORY_RATE_LIMIT
    assert classified.status_code == 429
    assert classified.request_id == "req-1"
    assert isinstance(classified.original, RateLimitError)


def test_classify_server_by_status():
    class APIStatusError(Exception):
        status_code = 502

    assert classify_exception(APIStatusError("bad gateway")).category == CATEGORY_SERVER


def test_default_empty_rate_card_marks_every_cost_unavailable():
    assert RateCard().rate_for("gpt-4o-mini") is None
    assert RateCard().cost_basis == COST_BASIS_UNVERIFIED
    ledger = UsageLedger()
    ledger.record(_record(cost=None))
    assert ledger.summary()["cost_basis"] == COST_BASIS_UNAVAILABLE
