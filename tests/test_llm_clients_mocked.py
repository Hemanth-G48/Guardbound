"""Tests for chat LLM clients (Phase 1) — all SDKs faked, no network.

We inject lightweight fake modules into ``sys.modules`` to verify call
plumbing (model id, temperature default of 0.7, message handling) without the
real packages installed.
"""
from __future__ import annotations

import sys
import types

import pytest

from guardbound.llm import (
    AnthropicChatLLM,
    ChatLLM,
    HFLocalChatLLM,
    MockChatLLM,
    OpenAIChatLLM,
)


# --------------------------------------------------------------------------- #
# Mock backend


def test_mock_llm_scripts_then_echoes():
    llm = MockChatLLM(responses=["first", "second"])
    assert llm.chat("q1") == "first"
    assert llm.chat("q2") == "second"
    assert llm.chat("q3") == "[mock reply to]: q3"


def test_mock_llm_defaults_to_paper_temperature():
    llm = MockChatLLM()
    llm.chat("hello")
    _, temperature, json_format = llm.calls[0]
    assert temperature == 0.7  # paper-wide generation setting
    assert json_format is False


def test_chatllm_is_abstract():
    with pytest.raises(TypeError):
        ChatLLM()  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
# OpenAI client plumbing


class _FakeCompletions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs

        class _Msg:
            content = "fake-openai-reply"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _FakeOpenAIClient:
    def __init__(self, **kwargs):
        global _FAKE_COMPLETIONS
        _FAKE_COMPLETIONS = _FakeCompletions()
        self.chat = types.SimpleNamespace(completions=_FAKE_COMPLETIONS)


def test_openai_client_sends_model_and_temperature(monkeypatch):
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    llm = OpenAIChatLLM("gpt-3.5-turbo-0125")  # paper version pin
    reply = llm.generate([{"role": "user", "content": "hi"}])
    assert reply == "fake-openai-reply"
    kwargs = _FAKE_COMPLETIONS.last_kwargs
    assert kwargs["model"] == "gpt-3.5-turbo-0125"
    assert kwargs["temperature"] == 0.7
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_o1_client_skips_custom_temperature(monkeypatch):
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    OpenAIChatLLM("o1-2024-12-17").generate(
        [{"role": "user", "content": "hi"}], temperature=0.7)
    assert "temperature" not in _FAKE_COMPLETIONS.last_kwargs


def test_openai_missing_sdk_error_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)  # forces ImportError
    llm = OpenAIChatLLM("gpt-4o-2024-08-06")
    with pytest.raises(ImportError, match="pip install openai"):
        llm.generate([{"role": "user", "content": "hi"}])


# --------------------------------------------------------------------------- #
# Anthropic client plumbing


class _FakeAnthropicMessages:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs

        class _Block:
            text = "fake-claude-reply"

        class _Resp:
            content = [_Block()]

        return _Resp()


class _FakeAnthropicClient:
    def __init__(self, **kwargs):
        global _FAKE_MESSAGES
        _FAKE_MESSAGES = _FakeAnthropicMessages()
        self.messages = _FAKE_MESSAGES


def test_anthropic_client_splits_system_and_sets_temperature(monkeypatch):
    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = _FakeAnthropicClient
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    llm = AnthropicChatLLM("claude-3-5-sonnet-20241022")  # paper version pin
    reply = llm.chat("hello there", system="You are helpful.")
    assert reply == "fake-claude-reply"
    kwargs = _FAKE_MESSAGES.last_kwargs
    assert kwargs["model"] == "claude-3-5-sonnet-20241022"
    assert kwargs["system"] == "You are helpful."
    assert kwargs["temperature"] == 0.7
    assert kwargs["messages"] == [{"role": "user", "content": "hello there"}]


def test_anthropic_missing_sdk_error_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    llm = AnthropicChatLLM("claude-3-5-sonnet-20241022")
    with pytest.raises(ImportError, match="pip install anthropic"):
        llm.chat("hi")


# --------------------------------------------------------------------------- #
# Local client (lazy; only checks error path without transformers)


def test_local_client_lazy_without_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    llm = HFLocalChatLLM("meta-llama/Meta-Llama-3-8B-Instruct")
    with pytest.raises(ImportError, match="transformers"):
        llm.chat("hi")


# --------------------------------------------------------------------------- #
# Factory


def test_factory_dispatch():
    from guardbound.llm import make_chat_llm
    assert isinstance(make_chat_llm("mock", ""), MockChatLLM)
    assert isinstance(make_chat_llm("openai", "gpt-4o-2024-08-06"), OpenAIChatLLM)
    assert isinstance(make_chat_llm("local", "microsoft/phi-4"), HFLocalChatLLM)
    with pytest.raises(ValueError, match="Unknown LLM kind"):
        make_chat_llm("nope", "")
