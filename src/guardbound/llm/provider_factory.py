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

import os
from typing import Any

from ..logging_utils import get_logger
from .attacker_profiles import (
    BACKEND_GLM4V,
    BACKEND_HF_LOCAL,
    BACKEND_ORNITH15,
    BACKEND_QWEN38_NATIVE,
)
from .base import ChatLLM
from .model_manager import ManagedLocalChatLLM, ModelManager

logger = get_logger(__name__)

SUPPORTED_PROVIDERS = ("local", "cloud")
CLOUD_BACKENDS = ("openai",)

# Keys that only make sense for the local HF backend. They have no cloud
# equivalent, so a cloud role declaring one is rejected instead of silently
# dropping it (a dropped sampling parameter would change the generation
# contract without appearing anywhere in the run manifest).
LOCAL_ONLY_ROLE_KEYS = (
    "top_k",
    "do_sample",
    "chat_template_kwargs",
    "quantization",
    "device_map",
    "revision",
    "dtype",
    "residency",
    "parameters",
    "actual_parameters",
    "reasoning_mode",
)

# Local backends. "hf_local" is the transformers text-generation pipeline used by
# every existing role; "glm4v" is the native processor/model path for GLM-4.6V.
_LOCAL_BACKEND_ALIASES = {
    "hf_local": "hf_local",
    "huggingface_local": "hf_local",
    "hf": "hf_local",
    "glm4v": "glm4v",
    "glm": "glm4v",
    "glm4v_local": "glm4v",
    "ornith15": "ornith15",
    "ornith": "ornith15",
    "ornith15_local": "ornith15",
    # Phase 17 Stage 3: the declared native architecture for Qwen3.8-27B
    # (`Qwen3_5ForConditionalGeneration`) rather than the text-generation
    # pipeline's `Qwen3_5ForCausalLM` substitution.
    "qwen38_native": "qwen38_native",
    "qwen3_8_native": "qwen38_native",
    "qwen38": "qwen38_native",
}


def _norm_local_backend(backend: str | None) -> str:
    """Canonicalise a local backend name; unknown names fail loudly."""
    b = (backend or "hf_local").strip().lower()
    if b not in _LOCAL_BACKEND_ALIASES:
        raise ValueError(
            f"unknown local backend {backend!r}; supported: "
            f"{sorted(set(_LOCAL_BACKEND_ALIASES))}"
        )
    return _LOCAL_BACKEND_ALIASES[b]


def _norm_provider(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p in ("hf_local", "huggingface_local", "local"):
        return "local"
    if p in ("openai", "api", "cloud"):
        return "cloud"
    raise ValueError(
        f"Unknown provider {provider!r}. Supported: {SUPPORTED_PROVIDERS}"
    )


def normalize_provider(provider: str) -> str:
    """Public alias for the provider-name normaliser.

    Scripts need the same local/cloud decision the factory makes, so
    ``provider: openai`` is not mistaken for something exotic just because it
    is not spelled ``cloud``.
    """
    return _norm_provider(provider)


def validate_cloud_role_config(role: str, role_cfg: dict[str, Any]) -> None:
    """Reject a cloud role that declares parameters the cloud API cannot honour.

    Reporting beats ignoring: a silently dropped ``top_k`` or
    ``chat_template_kwargs`` would change the generation contract while leaving
    no trace in the manifest.
    """
    offending = sorted(k for k in LOCAL_ONLY_ROLE_KEYS if k in role_cfg)
    if offending:
        raise ValueError(
            f"{role}: provider=cloud cannot honour local-only keys "
            f"{offending}. Remove them, or switch {role}.provider to 'local'."
        )
    if role_cfg.get("structured_output") == "json_schema" and not role_cfg.get("json_schema"):
        raise ValueError(
            f"{role}: structured_output='json_schema' requires a 'json_schema' "
            f"block. Omit structured_output to use the official reference "
            f"behaviour (JSON mode)."
        )


def build_cloud_role_llm(
    role: str,
    role_cfg: dict[str, Any],
    *,
    ledger=None,
) -> ChatLLM:
    """Build the OpenAI-backed ``ChatLLM`` for one role.

    ``role_cfg`` keys (all optional except ``model``):
        provider, model, backend, api_key_env, base_url, organization, project,
        timeout_s, max_output_tokens, top_p, structured_output, json_schema,
        json_schema_name, client (test injection), supports_temperature.
    """
    backend_kind = role_cfg.get("backend", "openai")
    if backend_kind not in CLOUD_BACKENDS:
        raise ValueError(
            f"{role}: unsupported cloud backend {backend_kind!r}; "
            f"supported: {CLOUD_BACKENDS}"
        )
    validate_cloud_role_config(role, role_cfg)

    model_id = role_cfg.get("model")
    if not model_id:
        raise ValueError(f"{role}.model is required")

    # Phase 45: check before the API-key read below and before OpenAIChatLLM.
    from .paid_authorization import require_construction_authorization
    require_construction_authorization(model_id)

    api_key_env = role_cfg.get("api_key_env", "OPENAI_API_KEY")
    injected_client = role_cfg.get("client")
    # Fail fast on a missing key for a real run. An injected client (offline
    # tests, smoke harnesses) already owns its transport and needs no key.
    if injected_client is None and not os.environ.get(api_key_env):
        raise ValueError(
            f"{role}: cloud provider selected but environment variable "
            f"{api_key_env} is not set. Set it or switch "
            f"{role}.provider to 'local'."
        )

    from .openai_client import OpenAIChatLLM

    kwargs: dict[str, Any] = {
        "api_key_env": api_key_env,
        "role": role,
        "client": injected_client,
    }
    for key in (
        "base_url",
        "organization",
        "project",
        "timeout_s",
        "max_output_tokens",
        "top_p",
        "structured_output",
        "json_schema",
        "json_schema_name",
        "supports_temperature",
    ):
        if key in role_cfg and role_cfg[key] is not None:
            kwargs[key] = role_cfg[key]

    # Explicit local->cloud mapping for the output cap. The official reference
    # sends no cap at all, so a declared cap is honoured but announced: it is a
    # deviation from the paper, not a silent default.
    if "max_output_tokens" not in kwargs and role_cfg.get("max_new_tokens") is not None:
        kwargs["max_output_tokens"] = int(role_cfg["max_new_tokens"])
        logger.warning(
            "[factory] %s: mapping max_new_tokens=%s to OpenAI max_tokens. "
            "The paper's reference implementation sends no output cap.",
            role, kwargs["max_output_tokens"],
        )
    if "temperature" in role_cfg:
        # Recorded for the manifest only: the paper drives sampling per call
        # (attacker/generator 0.7, evaluator 0.0), not per role.
        logger.info("[factory] %s: declared role temperature %s (per-call "
                    "temperatures still govern generation)", role,
                    role_cfg["temperature"])

    if ledger is not None:
        kwargs["ledger"] = ledger

    logger.info("[factory] %s -> cloud/%s model=%s structured_output=%s",
                role, backend_kind, model_id,
                role_cfg.get("structured_output", "json_object"))
    return OpenAIChatLLM(model_id=model_id, **kwargs)


def build_role_llm(
    role_cfg: dict[str, Any],
    role: str,
    manager: ModelManager | None = None,
    default_max_new_tokens: int = 256,
    default_temperature: float = 0.7,
    structured_output_mode: str | None = None,
    ledger=None,
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
    ledger
        Optional shared :class:`UsageLedger` for cloud roles (cost accounting).
        Ignored by the local provider, which has no per-token cost.
    """
    provider = _norm_provider(role_cfg.get("provider", "local"))
    model_id = role_cfg.get("model")
    if not model_id:
        raise ValueError(f"{role}.model is required")

    temperature = role_cfg.get("temperature", default_temperature)
    max_new_tokens = role_cfg.get("max_new_tokens", default_max_new_tokens)

    if provider == "cloud":
        return build_cloud_role_llm(role, role_cfg, ledger=ledger)

    # ---- local provider ------------------------------------------------- #
    if manager is None:
        raise ValueError(
            "local provider requires a ModelManager for GPU lifecycle handling"
        )
    device_map = role_cfg.get("device_map", "cuda")
    # Optional chat-template kwargs (e.g. Qwen3.5 enable_thinking=False).
    # Only pass them when declared — other templates reject unknown kwargs.
    ctk = role_cfg.get("chat_template_kwargs")
    extra = {"chat_template_kwargs": ctk} if ctk else {}
    # A per-role config key wins over the caller-supplied default, so a mixed
    # stack (e.g. constrained JSON on attacker+evaluator only) stays expressible.
    mode = role_cfg.get("structured_output_mode", structured_output_mode)
    # Backend selection lives here and only here. Unset -> "hf_local", which is
    # the pre-existing behaviour for every role in the frozen configuration.
    backend_kind = _norm_local_backend(role_cfg.get("backend", BACKEND_HF_LOCAL))

    if backend_kind == BACKEND_HF_LOCAL:
        from .local_client import HFLocalChatLLM

        if role_cfg.get("revision"):
            raise ValueError(
                f"{role}: 'revision' is not supported by the hf_local backend "
                f"(transformers pipeline path). Use backend: {BACKEND_GLM4V} for "
                "revision-pinned loading, or drop the revision key — pinning it "
                "here would be silently ignored."
            )
        backend = HFLocalChatLLM(
            model_id=model_id,
            device_map=device_map,
            # ``None`` here means the experiment declares NO output cap
            # (Phase 16.5 F1): generation is then bounded only by the model's
            # remaining context and stops at EOS.
            max_new_tokens=max_new_tokens,
            structured_output_mode=mode,
            # Phase 16.5 F3: declared sampling parameters are passed explicitly;
            # an absent key means "inherited", which the telemetry reports as such.
            top_p=role_cfg.get("top_p"),
            top_k=role_cfg.get("top_k"),
            do_sample=role_cfg.get("do_sample"),
            # Phase 17 Stage 2: opt-in load-time quantisation. Absent key -> None
            # -> the unquantized path every existing config uses.
            quantization=role_cfg.get("quantization"),
            **extra,
        )
    elif backend_kind == BACKEND_GLM4V:
        # Native processor + vision-language model path. Never routed through
        # pipeline("text-generation"), which does not support this architecture.
        from .glm4v_client import GLM4VChatLLM

        backend = GLM4VChatLLM(
            model_id=model_id,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            structured_output_mode=mode,
            revision=role_cfg.get("revision"),
            dtype=role_cfg.get("dtype", "bfloat16"),
            **extra,
        )
    elif backend_kind == BACKEND_QWEN38_NATIVE:
        # Native image-text-to-text path for Qwen3.8-27B. Its own backend because
        # the checkpoint declares Qwen3_5ForConditionalGeneration while the
        # text-generation pipeline resolves Qwen3_5ForCausalLM — a class whose
        # expected parameters do not match the published checkpoint.
        from .qwen38_native_client import Qwen38NativeChatLLM

        backend = Qwen38NativeChatLLM(
            model_id=model_id,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            structured_output_mode=mode,
            revision=role_cfg.get("revision"),
            dtype=role_cfg.get("dtype", "bfloat16"),
            quantization=role_cfg.get("quantization"),
            top_p=role_cfg.get("top_p"),
            do_sample=role_cfg.get("do_sample"),
            **extra,
        )
    else:  # BACKEND_ORNITH15
        # Native multimodal path (AutoProcessor + AutoModelForMultimodalLM). Its
        # own backend because the checkpoint's declared architecture is
        # Qwen3_5ForConditionalGeneration, not the causal-LM class that
        # qwen3_5 also maps to.
        from .ornith15_client import Ornith15ChatLLM

        backend = Ornith15ChatLLM(
            model_id=model_id,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            structured_output_mode=mode,
            revision=role_cfg.get("revision"),
            dtype=role_cfg.get("dtype", "bfloat16"),
            **extra,
        )

    manager.register(role, backend, model_id)
    # Residency is applied only when the role EXPLICITLY declares it. An absent
    # key must not touch the pinned set: the legacy behaviour is that the runner
    # pins the attacker and leaves the target/evaluator evictable, and pinning
    # every role here would stop the target/evaluator rotation from ever
    # releasing anything.
    if "residency" in role_cfg:
        manager.set_residency(role, role_cfg["residency"])
    logger.info("[factory] %s -> local backend=%s model=%s revision=%s "
                "max_new_tokens=%s structured_output_mode=%s residency=%s",
                role, backend_kind, model_id, role_cfg.get("revision"),
                max_new_tokens, mode, manager.residency_mode(role))
    return ManagedLocalChatLLM(backend, manager, role)


def build_three_model_stack(
    models_cfg: dict[str, dict[str, Any]],
    device: str = "cuda",
    default_max_new_tokens: int = 256,
    structured_output_mode: str | None = None,
    ledger=None,
) -> tuple[ModelManager, dict[str, ChatLLM]]:
    """Build attacker/target/evaluator from the ``models:`` config section.

    Returns (manager, {"attacker": llm, "target": llm, "evaluator": llm}).

    Each role resolves its own ``provider``, so a mixed stack (cloud attacker +
    local target + cloud evaluator) is expressible without any special case
    here. The ``ModelManager`` is always created — it simply sees no local
    registrations when every role is cloud.

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
            ledger=ledger,
        )
    return manager, llms
