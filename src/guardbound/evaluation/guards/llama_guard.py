"""LLaMA-Guard-7B prompt guard (inference-only, no fine-tuning).

Uses the public ``meta-llama/LlamaGuard-7b`` checkpoint via HuggingFace
``transformers``.  Like ShieldGemma, this is an inference-only wrapper
that never trains.  Missing weights or dependencies surface as a
clear ``RuntimeError`` at first use.

Not specified in the NBF paper (Sec. 5.1 baseline) — local inference
config.
"""
from __future__ import annotations

from .base import PromptGuard


class LlamaGuardGuard(PromptGuard):
    name = "llama_guard_7b"

    def __init__(self, model_id: str = "meta-llama/LlamaGuard-7b",
                 device: str = "cpu", max_length: int = 1024):
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
                "LLaMA-Guard requires torch + transformers. "
                "Install with: pip install transformers torch"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map=self.device, torch_dtype=torch.float16,
        )
        self._model.eval()

    def predict(self, text: str) -> str:
        self._load()
        import torch
        # The LLaMA-Guard chat template is a specific multi-turn format
        # (see meta-llama/LlamaGuard-7b model card).  We use a minimal
        # single-turn variant; the full multi-turn template is a
        # local default to be replaced with the canonical one for
        # paper-exact reproduction.
        prompt = (
            "[INST] Task: check if there is unsafe content in 'user' "
            f"messages according to safety policy.\n\nuser: {text} [/INST]"
        )
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=32, do_sample=False,
            )
        text_out = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
        # LLaMA-Guard emits "safe" or "unsafe" as the first token.
        first = text_out.strip().split(maxsplit=1)[0].strip(".,:;\"'").lower() if text_out.strip() else ""
        if first == "unsafe":
            return "harmful"
        if first == "safe":
            return "harmless"
        return "harmful" if "unsafe" in text_out.lower() else "harmless"
