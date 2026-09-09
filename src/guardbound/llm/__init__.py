"""Chat backends: base interface, API clients, local HF backend, and mock.

All concrete backends (OpenAI, Anthropic, HF-Local) are imported lazily so
that ``import guardbound.llm`` never requires API keys or GPU packages.
"""

from .base import ChatLLM, Message, DEFAULT_TEMPERATURE
from .mock import MockChatLLM

__all__ = [
    "ChatLLM",
    "Message",
    "DEFAULT_TEMPERATURE",
    "MockChatLLM",
    "OpenAIChatLLM",
    "AnthropicChatLLM",
    "HFLocalChatLLM",
    "OllamaChatLLM",
    "make_chat_llm",
]


def __getattr__(name: str):
    """Lazy-import concrete backends so SDKs are loaded only on demand."""
    if name == "OpenAIChatLLM":
        from .openai_client import OpenAIChatLLM
        return OpenAIChatLLM
    if name == "AnthropicChatLLM":
        from .anthropic_client import AnthropicChatLLM
        return AnthropicChatLLM
    if name == "HFLocalChatLLM":
        from .local_client import HFLocalChatLLM
        return HFLocalChatLLM
    if name == "OllamaChatLLM":
        from .ollama_client import OllamaChatLLM
        return OllamaChatLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_chat_llm(
    kind: str | dict,
    model_id: str | None = None,
    **kwargs,
) -> ChatLLM:
    """Factory used by later-phase scripts.

    Supports two calling conventions:

    1. String kind (backward compat):
       make_chat_llm("openai", "gpt-4o")
       make_chat_llm("ollama", "qwen3:14b")

    2. Config dict (unified interface):
       make_chat_llm({
           "provider": "openai",       # required
           "model": "gpt-4o",         # required
           "api_key_env": "OPENAI_API_KEY",   # optional, for openai
           "base_url": "...",          # optional, overrides default
       })

    Supported providers: "mock", "openai", "anthropic", "local", "ollama"
    """
    if isinstance(kind, dict):
        config = kind
        provider = config.get("provider")
        model = config.get("model", model_id)
        if model is None:
            raise ValueError("model is required in config or as model_id argument")
        extra_kwargs: dict = {k: v for k, v in config.items() if k not in ("provider", "model")}
        extra_kwargs.update(kwargs)
        kwargs = extra_kwargs
        kind = provider
    else:
        model = model_id

    if kind == "mock":
        return MockChatLLM()
    if kind == "openai":
        from .openai_client import OpenAIChatLLM
        return OpenAIChatLLM(model, **kwargs)
    if kind == "anthropic":
        from .anthropic_client import AnthropicChatLLM
        return AnthropicChatLLM(model, **kwargs)
    if kind == "local":
        from .local_client import HFLocalChatLLM
        return HFLocalChatLLM(model, **kwargs)
    if kind == "ollama":
        from .ollama_client import OllamaChatLLM
        return OllamaChatLLM(model=model, **kwargs)
    raise ValueError(f"Unknown LLM kind: {kind!r}")
