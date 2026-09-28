"""Phase 8 — provider-agnostic model factory.

Builds the attacker/target/evaluator ``ChatLLM`` objects from the experiment
config. Provider selection lives here and only here:

    attacker.provider: local  -> HFLocalChatLLM (HF transformers pipeline)
    attacker.provider: cloud  -> OpenAIChatLLM  (OpenAI-compatible API)

Attacks never see the provider — they receive ordinary ``ChatLLM`` objects.
Switching local <-> cloud is a config change, never an attack-code change.

Local multi-model runs route every backend through the ``ModelManager`` so the
three models swap on a 24 GB GPU instead of co-residing. Cloud backends need
no lifecycle management (the API host owns the weights) and bypass the manager.

Required cloud credentials are validated lazily per-provider:
  * local            -> requires the model path/repo id to exist (validated here)
  * cloud (openai)   -> requires OPENAI_API_KEY at construction time
"""
from __future__ import annotations

from typing import Any

from ..logging_utils import get_logger
from .base import ChatLLM
from .model_manager import ManagedLocalChatLLM, ModelManager

logger = get_logger(__name__)

SUPPORTED_PROVIDERS = ("local", "cloud")
CLOUD_BACKENDS = ("openai",)


def _norm_provider(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p in ("hf_local", "huggingface_local", "local"):
        return "local"
    if p in ("openai", "api", "cloud"):
        return "cloud"
    raise ValueError(
        f"Unknown provider {provider!r}. Supported: {SUPPORTED_PROVIDERS}"
    )


def build_role_llm(
    role_cfg: dict[str, Any],
    role: str,
    manager: ModelManager | None = None,
    default_max_new_tokens: int = 256,
    default_temperature: float = 0.7,
    structured_output_mode: str | None = None,
) -> ChatLLM:
    """Build the ChatLLM for one role from its config block.

    Parameters
    ----------
    role_cfg
        e.g. ``{"provider": "local", "model": "Qwen/Qwen3.5-4B",
        "parameters": "4B"}``
    role
        "attacker" | "target" | "evaluator"
    manager
        Required for local providers: models register here for GPU swapping.
    structured_output_mode
        Decoding mode for the local backend (e.g. ``"constrained_json"``).
        Applied to every local role built here; a per-role
        ``structured_output_mode`` key in ``role_cfg`` takes precedence.
    """
    provider = _norm_provider(role_cfg.get("provider", "local"))
    model_id = role_cfg.get("model")
    if not model_id:
        raise ValueError(f"{role}.model is required")

    temperature = role_cfg.get("temperature", default_temperature)
    max_new_tokens = role_cfg.get("max_new_tokens", default_max_new_tokens)

    if provider == "cloud":
        backend_kind = role_cfg.get("backend", "openai")
        if backend_kind not in CLOUD_BACKENDS:
            raise ValueError(
                f"{role}: unsupported cloud backend {backend_kind!r}; "
                f"supported: {CLOUD_BACKENDS}"
            )
        import os

        api_key_env = role_cfg.get("api_key_env", "OPENAI_API_KEY")
        if not os.environ.get(api_key_env):
            raise ValueError(
                f"{role}: cloud provider selected but environment variable "
                f"{api_key_env} is not set. Set it or switch "
                f"{role}.provider to 'local'."
            )
        from .openai_client import OpenAIChatLLM

        logger.info("[factory] %s -> cloud/%s model=%s",
                    role, backend_kind, model_id)
        return OpenAIChatLLM(model_id=model_id, api_key_env=api_key_env)

    # ---- local provider ------------------------------------------------- #
    if manager is None:
        raise ValueError(
            "local provider requires a ModelManager for GPU lifecycle handling"
        )
    from .local_client import HFLocalChatLLM

    device_map = role_cfg.get("device_map", "cuda")
    # Optional chat-template kwargs (e.g. Qwen3.5 enable_thinking=False).
    # Only pass them when declared — other templates reject unknown kwargs.
    ctk = role_cfg.get("chat_template_kwargs")
    extra = {"chat_template_kwargs": ctk} if ctk else {}
    # A per-role config key wins over the caller-supplied default, so a mixed
    # stack (e.g. constrained JSON on attacker+evaluator only) stays expressible.
    mode = role_cfg.get("structured_output_mode", structured_output_mode)
    backend = HFLocalChatLLM(
        model_id=model_id,
        device_map=device_map,
        max_new_tokens=max_new_tokens,
        structured_output_mode=mode,
        **extra,
    )
    manager.register(role, backend, model_id)
    logger.info("[factory] %s -> local model=%s dtype=bfloat16 max_new_tokens=%s "
                "structured_output_mode=%s",
                role, model_id, max_new_tokens, mode)
    return ManagedLocalChatLLM(backend, manager, role)


def build_three_model_stack(
    models_cfg: dict[str, dict[str, Any]],
    device: str = "cuda",
    default_max_new_tokens: int = 256,
    structured_output_mode: str | None = None,
) -> tuple[ModelManager, dict[str, ChatLLM]]:
    """Build attacker/target/evaluator from the ``models:`` config section.

    Returns (manager, {"attacker": llm, "target": llm, "evaluator": llm}).

    No role is pinned here: eviction policy is a property of the caller's
    architecture, not of the stack. Callers that need a pinned slot call
    ``manager.pin(role)`` explicitly.
    """
    manager = ModelManager(device=device)
    llms: dict[str, ChatLLM] = {}
    for role in ("attacker", "target", "evaluator"):
        role_cfg = models_cfg.get(role)
        if not role_cfg:
            raise ValueError(f"models.{role} section missing from config")
        llms[role] = build_role_llm(
            role_cfg,
            role,
            manager=manager,
            default_max_new_tokens=default_max_new_tokens,
            structured_output_mode=structured_output_mode,
        )
    return manager, llms
