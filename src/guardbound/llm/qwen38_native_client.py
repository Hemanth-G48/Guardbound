"""Qwen3.8-27B native-architecture attacker backend (Phase 17 Stage 3).

Why this backend exists
-----------------------
``Qwen/Qwen3.8-27B`` declares ``Qwen3_5ForConditionalGeneration``. The frozen
text-generation path resolves ``Qwen3_5ForCausalLM`` instead, and the two classes
do **not** expect the same parameters — measured in Stage 3 from tiny instances
of each class versus the checkpoint's own index:

    checkpoint publishes      Qwen3_5ForConditionalGeneration   Qwen3_5ForCausalLM
    model.language_model.*    consumes it                       expects model.layers.*
    model.visual.*            consumes it                       no such module
    mtp.*                     (auxiliary head, not consumed)    (same)

So the CausalLM path is a structural substitution, whatever transformers reports
about missing keys. This adapter resolves the declared class instead:

    messages -> Qwen3VLProcessor.apply_chat_template(...)
             -> Qwen3_5ForConditionalGeneration.generate(...) -> decoded text

Only the loader is model-specific; the pipeline itself (processor call, context
trimming, generation parameters, the frozen JSON stopping criterion and the
frozen JSON contract) is ``MultimodalChatLLM``, shared with the project's other
native-path backends. Nothing here touches prompts, attacks, the rubric, the
parser or the NBF.
"""
from __future__ import annotations

from typing import Any

from .multimodal_client import MultimodalChatLLM

# The revision Stage 3 is pinned to (Part 4 of the stage contract).
QWEN38_MODEL_ID = "Qwen/Qwen3.8-27B"
QWEN38_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
QWEN38_DECLARED_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"


class Qwen38NativeUnavailableError(RuntimeError):
    """The native image-text-to-text auto-class is unavailable in this transformers."""

    failure_class = "MODEL_INTERFACE_INCOMPATIBLE"


class Qwen38NativeArchitectureError(RuntimeError):
    """The instantiated class is not the architecture the checkpoint declares."""

    failure_class = "MODEL_INTERFACE_INCOMPATIBLE"


def load_processor(model_id: str, revision: str | None = None):
    """Resolve the checkpoint's own processor (declared: ``Qwen3VLProcessor``)."""
    from transformers import AutoProcessor

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if revision is not None:
        kwargs["revision"] = revision
    return AutoProcessor.from_pretrained(model_id, **kwargs)


def load_model_class():
    """The image-text-to-text auto-class for this checkpoint.

    ``AutoModelForCausalLM`` is deliberately not used: for a ``qwen3_5`` config it
    maps to a causal-LM class that expects a different parameter structure than
    the checkpoint publishes.
    """
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError as exc:
        raise Qwen38NativeUnavailableError(
            "transformers in this environment does not provide "
            "AutoModelForImageTextToText; the native Qwen3.8-27B path requires a "
            "transformers version with native image-text-to-text support."
        ) from exc
    return AutoModelForImageTextToText


def verify_native_architecture(model) -> tuple[str, list[str]]:
    """Assert the loaded class matches what the checkpoint declares.

    Returns ``(actual_class_name, declared_architectures)``.
    """
    declared = list(getattr(getattr(model, "config", None), "architectures", None) or [])
    actual = type(model).__name__
    if declared and actual not in declared:
        raise Qwen38NativeArchitectureError(
            f"loaded {actual} but the checkpoint declares {declared}; refusing to "
            "run a class the checkpoint does not declare"
        )
    return actual, declared


class Qwen38NativeChatLLM(MultimodalChatLLM):
    """Qwen3.8-27B driven through its declared native architecture."""

    name = "qwen38-native-local"
    label = "Qwen3.8-27B (native)"

    def __init__(
        self,
        model_id: str = QWEN38_MODEL_ID,
        device_map: str = "cuda",
        max_new_tokens: int | None = 256,
        chat_template_kwargs: dict[str, Any] | None = None,
        structured_output_mode: str | None = None,
        revision: str | None = QWEN38_REVISION,
        dtype: str = "bfloat16",
        quantization: str | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            chat_template_kwargs=chat_template_kwargs,
            structured_output_mode=structured_output_mode,
            revision=revision,
            dtype=dtype,
            quantization=quantization,
            top_p=top_p,
            do_sample=do_sample,
        )
        # Filled in on load, for telemetry and for the qualification artefact.
        self.loaded_class: str | None = None
        self.declared_architectures: list[str] = []

    def _load_processor(self, model_id: str, revision: str | None):
        revision = revision if revision is not None else self.revision
        return load_processor(model_id, revision=revision)

    def _model_class(self):
        return load_model_class()

    def _verify_model(self, model) -> None:
        actual, declared = verify_native_architecture(model)
        self.loaded_class = actual
        self.declared_architectures = declared
