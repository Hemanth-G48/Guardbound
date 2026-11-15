"""Cloud provider tests — provider factory, messages, generation, errors.

Everything runs offline through the real OpenAI SDK over a fake HTTP transport
(see ``tests/cloud_fakes.py``). No API key is required and no request leaves the
process.
"""
from __future__ import annotations

import httpx
import pytest

from guardbound.llm.cloud_errors import (
    CloudAuthError,
    CloudBadRequestError,
    CloudConnectionError,
    CloudContextLengthError,
    CloudPermissionError,
    CloudProviderError,
    CloudRateLimitError,
    CloudServerError,
    CloudTimeoutError,
    RetriesExhausted,
)
from guardbound.llm.openai_client import (
    OpenAIChatLLM,
    STRUCTURED_OUTPUT_JSON_OBJECT,
    STRUCTURED_OUTPUT_JSON_SCHEMA,
    STRUCTURED_OUTPUT_NONE,
)
from guardbound.llm.provider_factory import (
    build_cloud_role_llm,
    build_role_llm,
    normalize_provider,
    validate_cloud_role_config,
)
from guardbound.llm.usage import UsageLedger, UsageRecord, load_rate_card
from tests.cloud_fakes import (
    TEST_API_KEY,
    TEST_MODEL,
    FakeOpenAIBackend,
    completion,
    error_response,
)


def make_llm(backend: FakeOpenAIBackend, **kwargs) -> OpenAIChatLLM:
    """Provider-level tests exercise one attempt.

    The retry schedule has dedicated coverage in
    ``tests/test_phase30_retry_provenance.py``; here a single attempt keeps the
    taxonomy, extraction and telemetry assertions focused and fast.
    """
    kwargs.setdefault("retry_attempts", 1)
    return OpenAIChatLLM(model_id=TEST_MODEL, client=backend.client, **kwargs)


# --------------------------------------------------------------------------- #
# Provider factory
# --------------------------------------------------------------------------- #

def test_normalize_provider_accepts_cloud_aliases():
    for alias in ("openai", "cloud", "api", "OPENAI", " cloud "):
        assert normalize_provider(alias) == "cloud"
    for alias in ("local", "hf_local", "local"):
        assert normalize_provider(alias) == "local"
    with pytest.raises(ValueError, match="Unknown provider"):
        normalize_provider("bedrock")


def test_factory_builds_cloud_role_with_injected_client(fake_backend):
    llm = build_cloud_role_llm(
        "attacker",
        {"provider": "openai", "model": TEST_MODEL, "client": fake_backend.client},
    )
    assert isinstance(llm, OpenAIChatLLM)
    assert llm.model_id == TEST_MODEL
    assert llm.role == "attacker"


def test_factory_requires_api_key_when_no_client_injected(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        build_cloud_role_llm("target", {"provider": "openai", "model": TEST_MODEL})


def test_factory_rejects_unknown_cloud_backend():
    with pytest.raises(ValueError, match="unsupported cloud backend"):
        build_cloud_role_llm(
            "target",
            {"provider": "openai", "model": TEST_MODEL, "backend": "azure", "client": object()},
        )


def test_factory_rejects_local_only_keys_for_cloud_role():
    """Reporting beats silently dropping: top_k has no cloud equivalent."""
    with pytest.raises(ValueError, match="local-only keys"):
        validate_cloud_role_config(
            "attacker", {"provider": "openai", "model": TEST_MODEL, "top_k": 50}
        )
    for key in ("chat_template_kwargs", "quantization", "revision", "do_sample"):
        with pytest.raises(ValueError, match="local-only keys"):
            validate_cloud_role_config("x", {"model": TEST_MODEL, key: 1})


def test_factory_requires_schema_when_json_schema_selected():
    with pytest.raises(ValueError, match="requires a 'json_schema' block"):
        validate_cloud_role_config(
            "evaluator",
            {"model": TEST_MODEL, "structured_output": "json_schema"},
        )


def test_factory_maps_max_new_tokens_to_cloud_output_cap(fake_backend):
    llm = build_cloud_role_llm(
        "target",
        {
            "provider": "openai",
            "model": TEST_MODEL,
            "client": fake_backend.client,
            "max_new_tokens": 64,
        },
    )
    assert llm.max_output_tokens == 64
    llm.generate([{"role": "user", "content": "hi"}])
    assert fake_backend.requests[0]["max_tokens"] == 64


def test_local_provider_still_builds(fake_backend):
    """The local path is untouched by the cloud work."""
    from guardbound.llm.model_manager import ManagedLocalChatLLM, ModelManager

    manager = ModelManager(device="cpu")
    llm = build_role_llm(
        {"provider": "local", "model": "meta-llama/Meta-Llama-3-8B-Instruct"},
        "target",
        manager=manager,
    )
    assert isinstance(llm, ManagedLocalChatLLM)


def test_factory_model_selection_is_per_role(fake_backend):
    attacker = build_cloud_role_llm(
        "attacker",
        {"provider": "openai", "model": "gpt-4o", "client": fake_backend.client},
    )
    evaluator = build_cloud_role_llm(
        "evaluator",
        {"provider": "openai", "model": TEST_MODEL, "client": fake_backend.client},
    )
    assert attacker.model_id == "gpt-4o"
    assert evaluator.model_id == TEST_MODEL


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #

def test_system_user_assistant_messages_are_passed_in_order(fake_backend):
    llm = make_llm(fake_backend)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    llm.generate(messages)
    assert fake_backend.requests[0]["messages"] == messages


def test_multi_turn_history_is_not_reordered(fake_backend):
    llm = make_llm(fake_backend)
    history = [{"role": "system", "content": "s"}]
    for i in range(6):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})
    llm.generate(history)
    assert fake_backend.requests[0]["messages"] == history


def test_long_context_is_preserved_in_full(fake_backend):
    llm = make_llm(fake_backend)
    long_text = "x" * 40_000
    llm.generate([{"role": "system", "content": "s"},
                  {"role": "user", "content": long_text}])
    assert fake_backend.requests[0]["messages"][-1]["content"] == long_text


def test_max_turns_context_keeps_system_and_trims_pairs(fake_backend):
    llm = make_llm(fake_backend)
    messages = [{"role": "system", "content": "s"}]
    for i in range(5):
        messages.append({"role": "user", "content": f"q{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
    llm.generate(messages, max_turns_context=2)
    sent = fake_backend.requests[0]["messages"]
    assert sent[0] == {"role": "system", "content": "s"}
    assert [m["content"] for m in sent[1:]] == ["q3", "a3", "q4", "a4"]


# --------------------------------------------------------------------------- #
# Generation parameters
# --------------------------------------------------------------------------- #

def test_temperature_is_forwarded(fake_backend):
    make_llm(fake_backend).generate([{"role": "user", "content": "hi"}], temperature=0.0)
    assert fake_backend.requests[0]["temperature"] == 0.0


def test_default_temperature_matches_paper_setting(fake_backend):
    make_llm(fake_backend).generate([{"role": "user", "content": "hi"}])
    assert fake_backend.requests[0]["temperature"] == 0.7


def test_top_p_forwarded_only_when_configured(fake_backend):
    make_llm(fake_backend).generate([{"role": "user", "content": "hi"}])
    assert "top_p" not in fake_backend.requests[0]

    backend = FakeOpenAIBackend()
    make_llm(backend, top_p=0.9).generate([{"role": "user", "content": "hi"}])
    assert backend.requests[0]["top_p"] == 0.9


def test_no_output_cap_by_default_matching_official_reference(fake_backend):
    """The paper's reference implementation sends no max_tokens."""
    make_llm(fake_backend).generate([{"role": "user", "content": "hi"}])
    body = fake_backend.requests[0]
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body


def test_output_cap_is_mapped_when_configured(fake_backend):
    make_llm(fake_backend, max_output_tokens=32).generate(
        [{"role": "user", "content": "hi"}]
    )
    assert fake_backend.requests[0]["max_tokens"] == 32


def test_reasoning_model_omits_temperature_and_response_format(fake_backend):
    llm = OpenAIChatLLM(model_id="o1-2024-12-17", client=fake_backend.client)
    llm.generate([{"role": "user", "content": "hi"}], temperature=0.7, json_format=True)
    body = fake_backend.requests[0]
    assert "temperature" not in body
    assert "response_format" not in body
    assert llm.parameter_map(json_format=True)["temperature"].startswith("omitted")


def test_parameter_map_reports_every_mapping():
    llm = OpenAIChatLLM(model_id=TEST_MODEL)
    mapping = llm.parameter_map(json_format=True)
    assert mapping["temperature"] == "temperature"
    assert mapping["messages"] == "messages"
    assert mapping["json_format"] == "json_object"
    assert "official reference" in mapping["max_new_tokens"]


# --------------------------------------------------------------------------- #
# Structured output
# --------------------------------------------------------------------------- #

def test_json_format_uses_json_object_mode_and_parses(fake_backend):
    backend = FakeOpenAIBackend([
        completion('{"generatedQuestion": "q", "lastResponseSummary": "s"}')
    ])
    result = make_llm(backend).generate(
        [{"role": "user", "content": "hi"}], json_format=True
    )
    assert result == {"generatedQuestion": "q", "lastResponseSummary": "s"}
    assert backend.requests[0]["response_format"] == {"type": "json_object"}


def test_plain_generation_requests_text_format_like_official(fake_backend):
    make_llm(fake_backend).generate([{"role": "user", "content": "hi"}])
    assert fake_backend.requests[0]["response_format"] == {"type": "text"}


def test_json_schema_mode_sends_strict_schema(fake_backend):
    schema = {
        "type": "object",
        "properties": {"score": {"type": "integer"}},
        "required": ["score"],
        "additionalProperties": False,
    }
    llm = make_llm(
        fake_backend,
        structured_output=STRUCTURED_OUTPUT_JSON_SCHEMA,
        json_schema=schema,
        json_schema_name="rubric",
    )
    llm.generate([{"role": "user", "content": "hi"}], json_format=True)
    fmt = fake_backend.requests[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "rubric"
    assert fmt["json_schema"]["schema"] == schema


def test_structured_output_none_sends_no_response_format(fake_backend):
    llm = make_llm(fake_backend, structured_output=STRUCTURED_OUTPUT_NONE)
    llm.generate([{"role": "user", "content": "hi"}], json_format=True)
    assert "response_format" not in fake_backend.requests[0]


def test_default_structured_output_mode_is_json_object():
    assert OpenAIChatLLM(model_id=TEST_MODEL).structured_output == STRUCTURED_OUTPUT_JSON_OBJECT


def test_invalid_structured_output_mode_rejected():
    with pytest.raises(ValueError, match="unsupported structured_output"):
        OpenAIChatLLM(model_id=TEST_MODEL, structured_output="yaml")


# --------------------------------------------------------------------------- #
# Response extraction: the nine research-critical cases
# --------------------------------------------------------------------------- #

def test_case1_valid_attacker_json(fake_backend):
    backend = FakeOpenAIBackend([
        completion('{"generatedQuestion": "Q", "lastResponseSummary": "S"}')
    ])
    llm = make_llm(backend)
    out = llm.generate([{"role": "user", "content": "x"}], json_format=True)
    assert out == {"generatedQuestion": "Q", "lastResponseSummary": "S"}
    assert llm.last_request_stats["json_parse_ok"] is True
    assert llm.last_request_stats["outcome"] == "ok"


def test_case2_missing_required_fields_are_not_fabricated(fake_backend):
    backend = FakeOpenAIBackend([completion('{"generatedQuestion": "Q"}')])
    out = make_llm(backend).generate([{"role": "user", "content": "x"}], json_format=True)
    assert out == {"generatedQuestion": "Q"}
    assert "lastResponseSummary" not in out


def test_case3_empty_generated_question_passes_through_unchanged(fake_backend):
    backend = FakeOpenAIBackend([
        completion('{"generatedQuestion": "", "lastResponseSummary": "S"}')
    ])
    llm = make_llm(backend)
    out = llm.generate([{"role": "user", "content": "x"}], json_format=True)
    assert out == {"generatedQuestion": "", "lastResponseSummary": "S"}
    assert llm.last_request_stats["outcome"] == "ok"


def test_case4_prose_instead_of_json_returns_text_not_a_repair(fake_backend):
    backend = FakeOpenAIBackend([completion("Sure! Here is the next question: ...")])
    llm = make_llm(backend)
    out = llm.generate([{"role": "user", "content": "x"}], json_format=True)
    assert isinstance(out, str)
    assert out == "Sure! Here is the next question: ..."
    assert llm.last_request_stats["outcome"] == "json_parse_error"
    assert llm.last_request_stats["json_parse_ok"] is False


def test_case5_invalid_json_returns_text_and_is_classified(fake_backend):
    backend = FakeOpenAIBackend([completion('{"generatedQuestion": "unclosed"')])
    llm = make_llm(backend)
    out = llm.generate([{"role": "user", "content": "x"}], json_format=True)
    assert isinstance(out, str)
    assert llm.last_request_stats["outcome"] == "json_parse_error"


def test_case6_refusal_is_surfaced_as_text(fake_backend):
    backend = FakeOpenAIBackend([completion(None, refusal="I can't help with that.")])
    llm = make_llm(backend)
    out = llm.generate([{"role": "user", "content": "x"}], json_format=True)
    assert out == "I can't help with that."
    assert llm.last_request_stats["refusal"] is True
    assert llm.last_request_stats["outcome"] == "refusal"


def test_case7_truncated_response_is_flagged(fake_backend):
    backend = FakeOpenAIBackend([completion('{"generatedQuestion": "trunc', finish_reason="length")])
    llm = make_llm(backend)
    llm.generate([{"role": "user", "content": "x"}], json_format=True)
    assert llm.last_request_stats["outcome"] == "truncated"
    assert llm.last_request_stats["finish_reason"] == "length"


def test_empty_response_is_recorded_as_empty(fake_backend):
    backend = FakeOpenAIBackend([completion("")])
    llm = make_llm(backend)
    assert llm.generate([{"role": "user", "content": "x"}]) == ""
    assert llm.last_request_stats["outcome"] == "empty_response"


def test_content_filter_finish_reason(fake_backend):
    backend = FakeOpenAIBackend([completion("", finish_reason="content_filter")])
    llm = make_llm(backend)
    llm.generate([{"role": "user", "content": "x"}])
    assert llm.last_request_stats["outcome"] == "content_filter"
    assert llm.last_request_stats["refusal"] is True


def test_usage_and_request_metadata_are_captured(fake_backend):
    backend = FakeOpenAIBackend([
        completion("hi", prompt_tokens=120, completion_tokens=30, cached_tokens=40,
                   request_id="chatcmpl-abc")
    ])
    llm = make_llm(backend)
    llm.generate([{"role": "user", "content": "x"}], temperature=0.7)
    stats = llm.last_request_stats
    assert stats["input_tokens"] == 120
    assert stats["output_tokens"] == 30
    assert stats["cached_tokens"] == 40
    assert stats["total_tokens"] == 150
    assert stats["request_id"] == "chatcmpl-abc"
    assert stats["duration_s"] >= 0.0
    assert stats["model"] == TEST_MODEL
    assert stats["provider"] == "openai"
    assert stats["endpoint"] == "/v1/chat/completions"


def test_structured_output_mode_argument_is_accepted_and_ignored(fake_backend):
    """Local decoding switches must not break the shared ChatLLM contract."""
    llm = make_llm(fake_backend)
    out = llm.generate(
        [{"role": "user", "content": "x"}], structured_output_mode="constrained_json"
    )
    assert out == "ok"


# --------------------------------------------------------------------------- #
# Errors — one test per required category
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (error_response(401), CloudAuthError),
        (error_response(403), CloudPermissionError),
        (error_response(429), CloudRateLimitError),
        (error_response(500), CloudServerError),
        (error_response(400), CloudBadRequestError),
        (error_response(404), CloudBadRequestError),
        (httpx.ReadTimeout("timed out"), CloudTimeoutError),
        (httpx.ConnectError("no route"), CloudConnectionError),
    ],
)
def test_error_categories(spec, expected):
    backend = FakeOpenAIBackend([spec])
    llm = make_llm(backend)
    with pytest.raises(expected) as excinfo:
        llm.generate([{"role": "user", "content": "x"}])
    assert isinstance(excinfo.value, CloudProviderError)
    assert excinfo.value.category
    assert llm.last_request_stats["outcome"] == "error"
    assert llm.last_request_stats["error_category"] == excinfo.value.category


def test_context_length_error_is_not_reported_as_generic_bad_request():
    backend = FakeOpenAIBackend([
        error_response(400, "This model's maximum context length is 128000 tokens.")
    ])
    with pytest.raises(CloudContextLengthError, match="[Cc]ontext"):
        make_llm(backend).generate([{"role": "user", "content": "x"}])


def test_failed_request_is_recorded_in_the_ledger():
    ledger = UsageLedger()
    backend = FakeOpenAIBackend([error_response(429)])
    llm = make_llm(backend, ledger=ledger, role="attacker")
    with pytest.raises(CloudRateLimitError):
        llm.generate([{"role": "user", "content": "x"}])
    records = ledger.records()
    assert len(records) == 1
    assert records[0].ok is False
    assert records[0].error_category == "rate_limit"
    assert records[0].role == "attacker"


def test_no_hidden_retries_on_a_failing_request(monkeypatch):
    """The SDK never retries; the client's attempts are visible and bounded.

    Phase 30 replaced the one-attempt protocol with the author's 10-attempt
    schedule. The property this test guards is unchanged: no *hidden* retry
    layer exists — the ten calls are the client's own, and there is no eleventh.
    """
    import guardbound.llm.openai_client as oc

    monkeypatch.setattr(oc.time, "sleep", lambda seconds: None)
    backend = FakeOpenAIBackend(
        [error_response(500)] * 10 + [completion("would-be-11th")]
    )
    with pytest.raises(RetriesExhausted):
        make_llm(backend, retry_attempts=10).generate(
            [{"role": "user", "content": "x"}])
    assert backend.calls == 10, "ten visible attempts, no eleventh, no SDK retry"


def test_no_retry_on_success_either():
    backend = FakeOpenAIBackend([completion("one")])
    make_llm(backend).generate([{"role": "user", "content": "x"}])
    assert backend.calls == 1


def test_missing_key_raises_actionable_auth_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = OpenAIChatLLM(model_id=TEST_MODEL)
    with pytest.raises(CloudAuthError, match="OPENAI_API_KEY is not set"):
        llm.generate([{"role": "user", "content": "x"}])


def test_key_is_never_echoed_in_the_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-supersecretvalue1234567890")
    backend = FakeOpenAIBackend([error_response(401, "bad key sk-supersecretvalue1234567890")])
    llm = OpenAIChatLLM(model_id=TEST_MODEL, client=backend.client)
    with pytest.raises(CloudAuthError) as excinfo:
        llm.generate([{"role": "user", "content": "x"}])
    assert "sk-supersecret" not in str(excinfo.value)
    assert "[REDACTED]" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Usage telemetry only — Phase 22 removed the custom budget cap
# --------------------------------------------------------------------------- #

def test_usage_telemetry_never_blocks_a_request(tmp_path):
    """Cost is reported; it is not a spending limit and cannot stop a call."""
    card_path = tmp_path / "card.yaml"
    card_path.write_text(
        "version: test\nas_of: today\nsource: unit-test\nverified: true\n"
        "models:\n  gpt-4o-mini:\n    input_per_1m: 1.0\n"
        "    cached_input_per_1m: 0.5\n    output_per_1m: 2.0\n",
        encoding="utf-8",
    )
    ledger = UsageLedger(rate_card=load_rate_card(card_path))
    backend = FakeOpenAIBackend(
        [completion("ok", prompt_tokens=1000, completion_tokens=1000)] * 3
    )
    llm = make_llm(backend, ledger=ledger)

    for _ in range(3):
        llm.generate([{"role": "user", "content": "x"}])

    assert backend.calls == 3
    assert ledger.spent_usd > 0


def test_missing_usage_is_recorded_honestly_and_does_not_stop_the_run():
    payload = completion("ok")
    payload.pop("usage")
    backend = FakeOpenAIBackend([payload, completion("second")])
    ledger = UsageLedger()
    llm = make_llm(backend, ledger=ledger)

    llm.generate([{"role": "user", "content": "x"}])
    record = ledger.records()[0]
    assert record.usage_available is False
    assert record.total_tokens == 0
    assert record.estimated_cost_usd is None, "unknown cost must never be reported as 0"
    assert ledger.summary()["usage_unavailable_calls"] == 1

    llm.generate([{"role": "user", "content": "y"}])
    assert backend.calls == 2, "an unpriced call must not block the next request"


def test_usage_accounting_does_not_alter_the_provider_payload(fake_backend):
    """The ledger observes; it must not rewrite messages or request fields."""
    backend = FakeOpenAIBackend([completion("ok"), completion("ok")])
    plain = make_llm(backend, ledger=None)
    observed = make_llm(backend, ledger=UsageLedger())

    messages = [{"role": "user", "content": "same"}]
    plain.generate(messages, temperature=0.7, json_format=True)
    observed.generate(messages, temperature=0.7, json_format=True)

    assert backend.calls == 2
    first, second = backend.requests[0], backend.requests[1]
    assert first == second

