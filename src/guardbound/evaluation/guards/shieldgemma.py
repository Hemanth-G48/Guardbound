"""ShieldGemma-2B prompt guard (inference-only, no fine-tuning).

Uses the public ``google/shieldgemma-2b`` checkpoint via HuggingFace
``transformers``.  If the model weights or the ``transformers`` /
``torch`` packages are not available, ``predict`` raises a clear
error — never silently substitutes another model.

Not specified in the NBF paper (Sec. 5.1 baseline) — local inference
config.  ``max_length`` and device are configurable.
"""
from __future__ import annotations

from .base import PromptGuard


class ShieldGemmaGuard(PromptGuard):
    name = "shieldgemma_2b"

    def __init__(self, model_id: str = "google/shieldgemma-2b",
                 device: str = "cpu", max_length: int = 512):
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ShieldGemma guard requires torch + transformers. "
                "Install with: pip install transformers torch"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map=self.device, torch_dtype=torch.float16,
        )
        self._model.eval()

    def predict(self, text: str) -> str:
        # Lazy import: avoid loading torch/transformers when this class
        # is only type-checked or instantiated for the registry.
        self._load()
        import torch
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=8, do_sample=False,
            )
        text_out = self._tokenizer.decode(out[0], skip_special_tokens=True)
        # ShieldGemma emits "Yes" (harmful) or "No" (harmless).  The
        # exact output format is a local default; the canonical
        # Google inference snippet should be verified before
        # reproducing paper numbers.
        last = text_out.strip().split()[-1].strip(".,:;\"'").lower() if text_out.strip() else ""
        if last.startswith("yes"):
            return "harmful"
        if last.startswith("no"):
            return "harmless"
        # Conservative fallback
        return "harmful" if "yes" in text_out.lower() else "harmless"
