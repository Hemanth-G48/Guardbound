"""Ornith-1.5-9B attacker backend (native processor + multimodal model).

Architecture (resolved from the checkpoint, not assumed)
-------------------------------------------------------
``ornith-ai/Ornith-1.5-9B`` declares

    model_type      : qwen3_5
    architectures   : ["Qwen3_5ForConditionalGeneration"]
    processor_class : Qwen3VLProcessor
    dtype           : bfloat16
    vision_config   : present (image-text-to-text checkpoint)

``qwen3_5`` appears in **both** the causal-LM and the image-text-to-text auto
mappings, so a text-generation pipeline would happily resolve
``Qwen3_5ForCausalLM`` — a class whose weights this checkpoint does not contain.
This backend therefore resolves the model through the normal AutoModel
mechanism (``AutoModelForMultimodalLM``, which is config-driven) and then
**verifies** that the class actually instantiated is the one the checkpoint
declares. A mismatch raises ``Ornith15ArchitectureError``
(``failure_class = MODEL_INTERFACE_INCOMPATIBLE``) instead of silently running
the wrong architecture.

    messages -> processor.apply_chat_template(...) -> tensor inputs
             -> <declared conditional-generation class>.generate(...) -> text

The experiment is text-only: no images are added and the attack prompts are
untouched, keeping the attacker comparable with Qwen and GLM.

Shared behaviour (context trimming, generation parameters, frozen
constrained-JSON machinery, frozen JSON contract — no repair, no retry, no
fallback, no reasoning stripping) lives in
``multimodal_client.MultimodalChatLLM``.
"""
from __future__ import annotations

from typing import Any

from .multimodal_client import (  # noqa: F401  (re-exported for callers/tests)
    _DTYPE_NAMES,
    MultimodalChatLLM,
)


class Ornith15UnavailableError(RuntimeError):
    """The native multimodal auto-class is unavailable in this transformers."""

    failure_class = "MODEL_INTERFACE_INCOMPATIBLE"


class Ornith15ArchitectureError(RuntimeError):
    """The instantiated class is not the architecture the checkpoint declares."""

    failure_class = "MODEL_INTERFACE_INCOMPATIBLE"


# --------------------------------------------------------------------------- #
# Loader seams (patchable in tests: transformers' lazy auto-classes cannot be
# monkeypatched at the package level — see glm4v_client for the same note).
# --------------------------------------------------------------------------- #
def load_processor(model_id: str, revision: str | None = None):
    """Resolve the model's ``AutoProcessor`` (native processor path)."""
    from transformers import AutoProcessor

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if revision is not None:
        kwargs["revision"] = revision
    return AutoProcessor.from_pretrained(model_id, **kwargs)


def load_model_class():
    """Resolve the multimodal auto-class for this checkpoint.

    ``AutoModelForMultimodalLM`` resolves from the checkpoint config, so the
    concrete class follows ``config.architectures`` rather than being hard-coded
    here. ``AutoModelForCausalLM`` is deliberately not used: it maps ``qwen3_5``
    to a causal-LM class this checkpoint has no weights for.
    """
    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError as exc:
        raise Ornith15UnavailableError(
            "transformers in this environment does not provide "
            "AutoModelForMultimodalLM; the Ornith-1.5-9B backend requires a "
            "transformers version with native multimodal auto-class support."
        ) from exc
    return AutoModelForMultimodalLM


def verify_native_architecture(model) -> tuple[str, list[str]]:
    """Assert the loaded class matches the architecture the config declares.

    Returns ``(actual_class_name, declared_architectures)``.
    """
    config = getattr(model, "config", None)
    declared = list(getattr(config, "architectures", None) or [])
    actual = type(model).__name__
    if declared and actual not in declared:
        raise Ornith15ArchitectureError(
            f"loaded {actual} but the checkpoint declares {declared}; refusing "
            "to run an architecture the checkpoint has no weights for"
        )
    return actual, declared


class Ornith15ChatLLM(MultimodalChatLLM):
    """Local ``ChatLLM`` backed by the native Ornith-1.5-9B processor/model path."""

    name = "ornith15-local"
    label = "Ornith-1.5-9B"

    def __init__(
        self,
        model_id: str = "ornith-ai/Ornith-1.5-9B",
        device_map: str = "cuda",
        max_new_tokens: int = 256,
        chat_template_kwargs: dict[str, Any] | None = None,
        structured_output_mode: str | None = None,
        revision: str | None = None,
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__(
            model_id=model_id,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            chat_template_kwargs=chat_template_kwargs,
            structured_output_mode=structured_output_mode,
            revision=revision,
            dtype=dtype,
        )
        # Filled in on load, for telemetry/tests.
        self.loaded_class: str | None = None
        self.declared_architectures: list[str] = []

    def _load_processor(self, model_id: str, revision: str | None):
        return load_processor(model_id, revision=revision)

    def _model_class(self):
        return load_model_class()

    def _verify_model(self, model) -> None:
        actual, declared = verify_native_architecture(model)
        self.loaded_class = actual
        self.declared_architectures = declared
        if declared:
            import logging

            logging.getLogger(__name__).info(
                "Ornith-1.5-9B native class: %s (declared: %s)", actual, declared
            )
