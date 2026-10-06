"""GLM-4.6V-Flash attacker backend (native processor + vision-language model).

Why this exists
---------------
``HFLocalChatLLM`` loads every local role through
``pipeline("text-generation", ...)``, which resolves to ``AutoModelForCausalLM``.
``zai-org/GLM-4.6V-Flash`` is an image-text-to-text model
(``Glm4vForConditionalGeneration``) that is **not** in that mapping, so the
pipeline warns *"The model 'Glm4vForConditionalGeneration' is not supported for
text-generation"* and drives it through a generic wrapper. Forcing a VLM through
a causal-LM pipeline is the wrong abstraction; this module drives the model on
its native path instead:

    messages -> processor.apply_chat_template(...) -> tensor inputs
             -> Glm4vForConditionalGeneration.generate(...) -> decoded text

All shared behaviour — context trimming, generation parameters, the frozen
constrained-JSON machinery and the frozen JSON contract (no repair, no retry,
no fallback, no reasoning-wrapper handling) — lives in
``multimodal_client.MultimodalChatLLM``. This module contributes only the
processor/model-class resolution for this model.
"""
from __future__ import annotations

from typing import Any

from .multimodal_client import (  # noqa: F401  (re-exported for callers/tests)
    _DTYPE_NAMES,
    MultimodalChatLLM,
)


class GLM4VUnavailableError(RuntimeError):
    """The native GLM-4.6V model class is not importable in this transformers."""


# --------------------------------------------------------------------------- #
# Loader seams.
#
# transformers exposes its auto-classes through a lazy module whose
# ``__getattr__`` re-resolves on every import, so patching
# ``transformers.AutoProcessor`` does NOT change what
# ``from transformers import AutoProcessor`` returns. These two module-level
# functions are therefore the single place the backend resolves its classes —
# the same pattern ``local_client._maybe_thinking_kwargs`` already uses — which
# keeps the backend unit-testable without weights.
# --------------------------------------------------------------------------- #
def load_processor(model_id: str, revision: str | None = None):
    """Resolve the model's ``AutoProcessor`` (native processor path)."""
    from transformers import AutoProcessor

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if revision is not None:
        kwargs["revision"] = revision
    return AutoProcessor.from_pretrained(model_id, **kwargs)


def load_model_class():
    """Resolve ``Glm4vForConditionalGeneration`` explicitly.

    Never resolves through ``AutoModelForCausalLM``: silently substituting a
    causal-LM class is exactly the mismatch this backend exists to avoid.
    """
    try:
        from transformers import Glm4vForConditionalGeneration
    except ImportError as exc:
        raise GLM4VUnavailableError(
            "transformers in this environment does not provide "
            "Glm4vForConditionalGeneration; the GLM-4.6V backend requires a "
            "transformers version with native glm4v support."
        ) from exc
    return Glm4vForConditionalGeneration


class GLM4VChatLLM(MultimodalChatLLM):
    """Local ``ChatLLM`` backed by the native GLM-4.6V processor/model path."""

    name = "glm4v-local"
    label = "GLM-4.6V"

    def __init__(
        self,
        model_id: str = "zai-org/GLM-4.6V-Flash",
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

    def _load_processor(self, model_id: str, revision: str | None):
        return load_processor(model_id, revision=revision)

    def _model_class(self):
        return load_model_class()
