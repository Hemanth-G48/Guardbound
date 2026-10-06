"""Named attacker-model profiles — the supported way to switch the attacker.

The attacker is the only experimental variable these profiles touch. A profile
rewrites ``models.attacker`` (and nothing else) so the same attack algorithms,
prompts, target, evaluator, NBF stack and success criterion run unchanged:

    --attacker-model qwen3    -> existing HF pipeline backend  (hf_local)
    --attacker-model glm46v   -> native GLM-4.6V backend       (glm4v)

Omitting the selector applies no profile at all, so the default path stays
byte-identical to the frozen configuration.

The registry is intentionally data-only: adding a model is a table entry, never
a source edit to the attack, LLM or manager modules.
"""
from __future__ import annotations

import copy
from typing import Any

# Backend identifiers understood by ``provider_factory.build_role_llm``.
BACKEND_HF_LOCAL = "hf_local"
BACKEND_GLM4V = "glm4v"
BACKEND_ORNITH15 = "ornith15"
# Phase 17 Stage 3: the declared native architecture for Qwen3.8-27B
# (``Qwen3_5ForConditionalGeneration``) instead of the text-generation
# pipeline's ``Qwen3_5ForCausalLM`` substitution.
BACKEND_QWEN38_NATIVE = "qwen38_native"

# Residency modes understood by ``ModelManager.set_residency``.
RESIDENCY_PINNED = "pinned"
RESIDENCY_SEQUENTIAL = "sequential"

QWEN3_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
GLM46V_MODEL_ID = "zai-org/GLM-4.6V-Flash"
GLM46V_REVISION = "411bb4d77144a3f03accbf4b780f5acb8b7cde4e"
ORNITH15_MODEL_ID = "ornith-ai/Ornith-1.5-9B"
# Resolved from the Hub during implementation (2026-08-23 revision).
ORNITH15_REVISION = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"

ATTACKER_PROFILES: dict[str, dict[str, Any]] = {
    "qwen3": {
        "model": QWEN3_MODEL_ID,
        "backend": BACKEND_HF_LOCAL,
        "residency": RESIDENCY_PINNED,
        "dtype": "bfloat16",
        "quantization": "none",
        "revision": None,
    },
    "glm46v": {
        "model": GLM46V_MODEL_ID,
        "revision": GLM46V_REVISION,
        "backend": BACKEND_GLM4V,
        # 20.594 GB measured: GLM and a 7.67 GB target cannot co-reside on a
        # 24.57 GB card, so the attacker is released whenever another role needs
        # the GPU (see ModelManager.set_residency).
        "residency": RESIDENCY_SEQUENTIAL,
        "dtype": "bfloat16",
        "quantization": "none",
        # The same template setting the frozen attacker block already declares.
        # It is load-bearing for this model: GLM-4.6V's template implements
        # ``enable_thinking`` (it emits ``/nothink`` and an empty
        # ``<think></think>``), and with thinking left on the model answers the
        # instruction block instead of the requested JSON schema — measured 0/8
        # correct-schema replies with the default vs 8/8 with thinking disabled.
        # This is a declared template setting, not prompt/parser/template surgery.
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "ornith15": {
        "model": ORNITH15_MODEL_ID,
        "revision": ORNITH15_REVISION,
        "backend": BACKEND_ORNITH15,
        # ~19.3 GB of bf16 weights: like GLM this cannot co-reside with the
        # 7.67 GB target on a 24.57 GB card, so the attacker is released at every
        # run boundary and whenever another role needs the GPU.
        "residency": RESIDENCY_SEQUENTIAL,
        "dtype": "bfloat16",
        "quantization": "none",
        # Measured on this model, 8 goals per condition, through the frozen
        # interface: thinking-default produced the attacker schema in 1/8 replies
        # versus 5/8 with thinking disabled (with constrained_json). The setting
        # is a declared template parameter — the same one the frozen attacker
        # block already declares — not prompt/parser/template surgery.
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "qwen38_native": {
        # Phase 17 Stage 3. Same checkpoint as the text-generation path, loaded
        # through the class it declares. The 4-bit NF4 load measures ~16.5 GiB,
        # so like the other large attackers it cannot co-reside with a 15 GiB
        # target and is released whenever another role needs the GPU.
        "model": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "backend": BACKEND_QWEN38_NATIVE,
        "residency": RESIDENCY_SEQUENTIAL,
        "dtype": "bfloat16",
        # The approved Phase 17 quantization policy (Part E), applied at load time
        # to the official weights.
        "quantization": "nf4",
        # The same template setting the Stage 2 CausalLM baseline ran with, so an
        # architecture comparison is not confounded by a thinking-mode change.
        "chat_template_kwargs": {"enable_thinking": False},
    },
}


class AttackerProfileError(ValueError):
    """An unknown attacker profile was requested."""


def profile_names() -> list[str]:
    return sorted(ATTACKER_PROFILES)


def apply_attacker_profile(cfg: dict, name: str) -> dict:
    """Return a deep copy of ``cfg`` with ``models.attacker`` set from ``name``.

    Only the attacker block is rewritten. Existing attacker keys that the profile
    does not mention (temperature, max_new_tokens, chat_template_kwargs, provider)
    are preserved, so generation parameters remain exactly as configured.
    """
    if name not in ATTACKER_PROFILES:
        raise AttackerProfileError(
            f"unknown attacker profile {name!r}; supported: {profile_names()}"
        )
    profile = ATTACKER_PROFILES[name]
    new_cfg = copy.deepcopy(cfg)
    models = new_cfg.setdefault("models", {})
    attacker = models.setdefault("attacker", {})
    for key, value in profile.items():
        if value is None:
            attacker.pop(key, None)
        else:
            attacker[key] = value
    attacker.setdefault("provider", "local")
    return new_cfg


def attacker_runtime_telemetry(cfg: dict) -> dict[str, Any]:
    """Observational identity of the configured attacker (never mutates cfg)."""
    attacker = (cfg.get("models") or {}).get("attacker") or {}
    return {
        "model_id": attacker.get("model"),
        "revision": attacker.get("revision"),
        "backend": attacker.get("backend", BACKEND_HF_LOCAL),
        "dtype": attacker.get("dtype", cfg.get("hardware", {}).get("dtype", "bfloat16")),
        "quantization": attacker.get(
            "quantization", cfg.get("hardware", {}).get("quantization", "none")
        ),
        "residency_mode": attacker.get("residency", RESIDENCY_PINNED),
    }
