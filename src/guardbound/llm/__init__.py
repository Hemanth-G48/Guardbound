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


def make_chat_llm(kind: str, model_id: str, **kwargs) -> ChatLLM:
    """Factory used by later-phase scripts.

    kind ∈ {"mock", "openai", "anthropic", "local", "ollama"}
    """
    if kind == "mock":
        return MockChatLLM()
    if kind == "openai":
        from .openai_client import OpenAIChatLLM
        return OpenAIChatLLM(model_id, **kwargs)
    if kind == "anthropic":
        from .anthropic_client import AnthropicChatLLM
        return AnthropicChatLLM(model_id, **kwargs)
    if kind == "local":
        from .local_client import HFLocalChatLLM
        return HFLocalChatLLM(model_id, **kwargs)
    if kind == "ollama":
        from .ollama_client import OllamaChatLLM
        return OllamaChatLLM(model=model_id, **kwargs)
    raise ValueError(f"Unknown LLM kind: {kind!r}")
