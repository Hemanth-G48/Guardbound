"""HuggingFace local backend for open-source targets (Llama-3-8b-instruct, Phi-4).

Requires ``transformers`` (+ torch). Model ids come from config
(``llm.local``). Lazy import keeps the package importable without GPU deps.
"""
from __future__ import annotations

import gc
import threading
import weakref
from pathlib import Path
from typing import Any

import torch

from ..logging_utils import get_logger
from .base import ChatLLM, DEFAULT_TEMPERATURE, Message


def _resolve_model_family(model_id: str) -> str:
    """Best-effort Qwen-family detection for chat-template override.

    Qwen3-family configs (``Qwen3ForCausalLM``, Qwen3.5, Qwen3-4B-Instruct-2507,
    etc.) typically declare ``enable_thinking=True`` by default in their
    ``generation_config.json``, which is picked up by the pipeline and turned into
    a thinking block by their chat template. We disable it at the template layer so
    the model returns usable text inside the 256-token budget.

    Detection is a cheap best-effort heuristic: any model_id whose resolved
    config.json ``model_type`` contains ``Qwen3`` is treated as Qwen3-family.
    """
    try:
        from transformers import AutoConfig

        model_path = Path(model_id)
        actual = (
            str(model_path.resolve())
            if model_path.exists() and model_path.is_dir()
            else model_id
        )
        cfg = AutoConfig.from_pretrained(actual, trust_remote_code=True)
        mt: str = getattr(cfg, "model_type", "")
        if "qwen3" in mt.lower() or "qwen3" in str(model_id).lower():
            return "qwen3"
    except Exception:  # noqa: BLE001 -- best-effort; never hard-fail generation
        pass
    return "other"


def _maybe_thinking_kwargs(model_id: str) -> dict[str, Any]:
    """Return ``{enable_thinking: False}`` for Qwen3-family models.

    This is a *default*, not an override: callers apply it with ``setdefault``
    so an explicitly requested ``enable_thinking`` always wins (Phase 17
    Stage 2, F1).
    """
    return {"enable_thinking": False} if _resolve_model_family(model_id) == "qwen3" else {}


def _termination_reason(
    last_token: int | None,
    eos_token_id: int | list[int] | None,
    generated: int,
    budget: int,
) -> str:
    """Why generation stopped, from the tokens alone.

    ``eos`` — the final token is the model's EOS.
    ``max_new_tokens`` — the output budget was consumed.
    ``stopping_criteria`` — generation ended early on a criterion (the JSON
    object closed), which is neither of the above.
    """
    eos_ids: list[int] = []
    if isinstance(eos_token_id, int):
        eos_ids = [eos_token_id]
    elif eos_token_id:
        eos_ids = list(eos_token_id)
    if last_token is not None and last_token in eos_ids:
        return "eos"
    if generated >= budget:
        return "max_new_tokens"
    return "stopping_criteria"


def _build_quantization_config(quantization: str):
    """The approved load-time quantisation config (Phase 17 Stage 2, Part E).

    NF4 4-bit post-training quantisation via bitsandbytes, applied to the
    official weights at load time. Verified working on this machine in Phase 17
    Stage 0 (``quant_runtime_verification.json``).
    """
    if quantization != "nf4":
        raise ValueError(f"unsupported quantization {quantization!r}; supported: ['nf4']")
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _apply_thinking_default(kwargs: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Fill in the Qwen3-family ``enable_thinking`` default where unset.

    Phase 17 Stage 2 (F1): the previous ``update()`` silently overwrote an
    explicit ``enable_thinking``, so ``enable_thinking=True`` was impossible to
    request through the public constructor and a mode comparison compared one
    mode against itself. Precedence is now explicit-configuration >
    model-family-default, which leaves the default production behaviour
    (thinking off for Qwen3-family models) unchanged.
    """
    for key, value in _maybe_thinking_kwargs(model_id).items():
        kwargs.setdefault(key, value)
    return kwargs


logger = get_logger(__name__)

_pipeline_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()


def pipeline_cache_key(model_id: str, device_map: str = "cuda") -> str:
    """Cache key for a ``(model_id, device_map)`` pair — single source of truth.

    Both the loader (``HFLocalChatLLM._get_pipeline``) and the releaser
    (``release_pipeline``) derive the key here so a load and a release can never
    address different entries.
    """
    return f"{model_id}_{device_map}"


def release_pipeline(model_id: str, device_map: str = "cuda") -> bool:
    """Drop the cached pipeline for ``(model_id, device_map)`` and free its VRAM.

    ``_pipeline_cache`` is the **sole strong owner** of a loaded pipeline:
    backends hold their handle weakly (``HFLocalChatLLM._pipeline``, Phase 17
    Stage 2 F2). Popping the cache entry is therefore *sufficient* to release
    the weights — an idle client cannot keep them resident, so a release no
    longer depends on every caller remembering to drop a second reference.
    Tearing down a model that is still generating is safe: the running call
    frame holds its own strong reference for the duration.

    Returns ``True`` when a cache entry was actually removed.
    """
    cache_key = pipeline_cache_key(model_id, device_map)
    with _cache_lock:
        cached = _pipeline_cache.pop(cache_key, None)
    if cached is None:
        return False
    del cached
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return True


def _extract_json_block(text: str):
    """Return the first ``{...}`` JSON object embedded in ``text`` or None.

    Local instruct models frequently wrap JSON in prose or markdown fences.
    """
    import json

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                block = text[start : i + 1]
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    return None
    return None


class _JSONStoppingCriteria:
    """Stop generation after the model produces a complete JSON object.

    Monitors for ``}`` or newline-after-JSON patterns to end generation
    early, avoiding filling the entire ``max_new_tokens`` budget with
    post-JSON rambling. Falls back to normal generation when the pattern
    is never matched.
    """

    def __init__(self, tokenizer, stop_tokens: list[int]) -> None:
        self.tokenizer = tokenizer
        self.stop_tokens = set(stop_tokens)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        last_id = input_ids[0][-1].item()
        return last_id in self.stop_tokens

    @classmethod
    def build(cls, tokenizer, json_format: bool):
        """Return StoppingCriteriaList if json_format and we can detect JSON end tokens.

        Only stops at a standalone ``}`` token (the JSON object's closing brace),
        NOT at ``\\n`` — newlines appear inside multi-line JSON responses and
        stopping there causes premature truncation. A standalone ``}`` is
        tokenized differently from ``}`` inside a string value, so this is safe.
        """
        if not json_format:
            return None
        # Detect the closing brace token in this tokenizer.
        stop_ids: list[int] = []
        for tok_str in ["}"]:
            ids = tokenizer.encode(tok_str, add_special_tokens=False)
            if ids:
                stop_ids.append(ids[0])
        if not stop_ids:
            return None
        from transformers import StoppingCriteriaList
        return StoppingCriteriaList([cls(tokenizer, stop_ids)])


class HFLocalChatLLM(ChatLLM):
    """Runs a local instruct model with a transformers text-generation pipeline.

    Provider-layer chat-template settings are applied at ``apply_chat_template``
    time, not at generation-param time. For Qwen3-family models the default is
    ``enable_thinking=False`` so the model does not spend its context window on a
    ``<think>...</think>`` block before the JSON answer. That default is a
    *default*, not a mandate: an explicitly supplied ``enable_thinking`` wins
    (Phase 17 Stage 2, F1). This is a harness-level template setting, not a
    change to model weights, generation params, or the attack/NBF prompts.

    ``quantization`` is an **opt-in load-time** setting (Phase 17 Stage 2). Its
    default, ``None``, loads the checkpoint exactly as before; the only supported
    value, ``"nf4"``, applies the bitsandbytes 4-bit NF4 post-training
    quantisation the Phase 17 quantization policy approved, from the official
    weights — never a community quantised checkpoint.
    """

    name = "hf-local"

    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        max_new_tokens: int | None = 512,
        chat_template_kwargs: dict[str, Any] | None = None,
        structured_output_mode: str | None = None,
        # Phase 16.5 (F3): explicit sampling parameters. ``None`` means "not
        # declared by the experiment", in which case the value the framework would
        # otherwise inherit from the model's own generation_config.json is left in
        # place — but it is then *reported* by ``generation_parameters()`` instead
        # of being silently assumed. A declared value is passed to generate()
        # explicitly, so model-specific defaults cannot override the experiment.
        top_p: float | None = None,
        top_k: int | None = None,
        do_sample: bool | None = None,
        # Phase 17 Stage 2: opt-in quantized load. ``None`` (the default) leaves
        # the production load path byte-identical; ``"nf4"`` applies the approved
        # 4-bit NF4 PTQ at load time.
        quantization: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        # ``None`` = no artificial output cap (Phase 16.5 F1). The only remaining
        # bound is the model's own context window, which is the same class of bound
        # the reference API has; generation then stops at EOS as usual.
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        if quantization not in (None, "nf4"):
            raise ValueError(
                f"unsupported quantization {quantization!r}; supported: [None, 'nf4']"
            )
        self.quantization = quantization
        self.chat_template_kwargs: dict[str, Any] = _apply_thinking_default(
            dict(chat_template_kwargs or {}), model_id
        )
        self.structured_output_mode = structured_output_mode  # None | "constrained_json"
        # The loaded pipeline is held weakly — ``_pipeline_cache`` is its sole
        # strong owner (Phase 17 Stage 2, F2). See the ``_pipeline`` property.
        self._pipeline_ref: weakref.ReferenceType[Any] | None = None
        self._pipeline_strong: Any = None
        self._tokenizer: Any = None
        # Per-call telemetry from the most recent generate() (Phase 17 Stage 2).
        self.last_generation_stats: dict[str, Any] | None = None

    @property
    def _pipeline(self) -> Any:
        """The loaded pipeline, held **weakly**.

        Phase 17 Stage 2 (F2): the backend used to keep a second strong
        reference alongside the module-level ``_pipeline_cache``, so
        ``release_pipeline()`` could remove the cache entry and free nothing —
        measured at ~15 GiB still resident, which made the *next* model load
        fail with a CUDA OOM that looked unrelated to the caller's release.
        Holding a weak handle removes the failure mode rather than documenting
        it: there is no second reference for a caller to forget to drop.
        """
        ref = self._pipeline_ref
        if ref is not None:
            resolved = ref()
            if resolved is not None:
                return resolved
        return self._pipeline_strong

    @_pipeline.setter
    def _pipeline(self, pipeline: Any) -> None:
        if pipeline is None:
            self._pipeline_ref = None
            self._pipeline_strong = None
            return
        try:
            self._pipeline_ref = weakref.ref(pipeline)
            self._pipeline_strong = None
        except TypeError:
            # Not weak-referenceable (test doubles, plain containers). Kept
            # strongly so such objects behave exactly as before; they hold no
            # CUDA memory, so they cannot leak VRAM.
            self._pipeline_ref = None
            self._pipeline_strong = pipeline

    @property
    def enable_thinking(self) -> bool | None:
        """Template-layer thinking flag for Qwen3-family models."""
        return self.chat_template_kwargs.get("enable_thinking", False)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            model_path = Path(self.model_id)
            actual_path = str(model_path.resolve()) if model_path.exists() and model_path.is_dir() else self.model_id
            self._tokenizer = AutoTokenizer.from_pretrained(
                actual_path,
                trust_remote_code=True,
            )
        return self._tokenizer

    def _build_gen_config(
        self,
        temperature: float,
        max_new_tokens: int | None,
        do_sample: bool,
    ):
        from transformers import GenerationConfig

        return GenerationConfig(
            # ``None`` means "do not cap output length here" — the call site
            # resolves it to the remaining context budget (Phase 16.5 F1).
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=do_sample,
            pad_token_id=self._get_tokenizer().pad_token_id,
            eos_token_id=self._get_tokenizer().eos_token_id,
            use_cache=True,
        )

    def _context_generation_budget(self, model, input_len: int) -> int:
        """Largest number of tokens this prompt can still generate.

        Used only when the experiment declares no output cap. This is not an
        arbitrary ceiling: it is the model's own context window, the same bound
        the reference API is subject to, so generation still ends at EOS as usual.
        """
        limits = []
        cfg = getattr(model, "config", None)
        for attr in ("max_position_embeddings",):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > 0:
                limits.append(value)
        tok = self._get_tokenizer()
        tml = getattr(tok, "model_max_length", None)
        if isinstance(tml, int) and 0 < tml < 10 ** 9:
            limits.append(tml)
        window = min(limits) if limits else 4096
        return max(1, window - input_len)

    def generation_parameters(self) -> dict[str, Any]:
        """The parameters this backend will actually use, for truthful telemetry.

        ``max_new_tokens=None`` means "context-bounded, no artificial cap";
        ``top_p``/``top_k``/``do_sample`` of ``None`` mean "inherited from the
        model's own generation_config.json" — reported as such rather than
        assumed to equal the experiment's declared value (Phase 16.5 F3).
        """
        return {
            "backend": type(self).__name__,
            "model_id": self.model_id,
            "max_new_tokens": self.max_new_tokens,
            "max_new_tokens_effective": ("context_window" if self.max_new_tokens is None
                                         else self.max_new_tokens),
            "top_p": self.top_p,
            "top_p_source": ("declared" if self.top_p is not None
                             else "inherited_from_generation_config"),
            "top_k": self.top_k,
            "top_k_source": ("declared" if self.top_k is not None
                             else "inherited_from_generation_config"),
            "do_sample": self.do_sample,
            "do_sample_source": ("declared" if self.do_sample is not None
                                 else "derived_from_temperature"),
            "quantization": self.quantization,
            "quantization_method": ("bitsandbytes NF4 PTQ at load time"
                                   if self.quantization == "nf4" else None),
        }

    def _model_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the pipeline's model load.

        Empty of quantisation unless the caller opted in, so the default load
        path is unchanged.
        """
        kwargs: dict[str, Any] = {"attn_implementation": "sdpa"}
        if self.quantization is not None:
            kwargs["quantization_config"] = _build_quantization_config(self.quantization)
        return kwargs

    def _get_pipeline(self):
        loaded = self._pipeline
        if loaded is not None:
            return loaded

        cache_key = pipeline_cache_key(self.model_id, self.device_map)
        with _cache_lock:
            cached = _pipeline_cache.get(cache_key)
            if cached is not None:
                self._pipeline = cached
                logger.info("Reusing cached pipeline for: %s", self.model_id)
                return cached

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                GenerationConfig,
                pipeline,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for HFLocalChatLLM. "
                "Install it with: pip install transformers torch accelerate"
            ) from exc

        model_path = Path(self.model_id)
        if model_path.exists() and model_path.is_dir():
            actual_path = str(model_path.resolve())
            logger.info("Loading local model from: %s", actual_path)
            pipeline_obj = pipeline(
                "text-generation",
                model=actual_path,
                device_map=self.device_map,
                torch_dtype="auto",
                trust_remote_code=True,
                model_kwargs=self._model_kwargs(),
                generation_config=self._build_gen_config(
                    temperature=DEFAULT_TEMPERATURE,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                ),
            )
        else:
            logger.info("Loading HuggingFace model: %s", self.model_id)
            pipeline_obj = pipeline(
                "text-generation",
                model=self.model_id,
                device_map=self.device_map,
                dtype=torch.bfloat16,
                model_kwargs=self._model_kwargs(),
                generation_config=self._build_gen_config(
                    temperature=DEFAULT_TEMPERATURE,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                ),
            )

        if hasattr(pipeline_obj, "model"):
            pipeline_obj.model.eval()
        # The cache takes ownership BEFORE this function's local handle goes
        # away: the client's handle is weak, so at least one strong reference
        # must exist from the moment the pipeline is created until it is cached.
        with _cache_lock:
            _pipeline_cache[cache_key] = pipeline_obj
        self._pipeline = pipeline_obj
        return pipeline_obj

    def __repr__(self) -> str:
        return (
            f"HFLocalChatLLM(model_id={self.model_id!r}, "
            f"device_map={self.device_map!r}, "
            f"max_new_tokens={self.max_new_tokens}, "
            f"chat_template_kwargs={self.chat_template_kwargs!r})"
        )

    def generate(
        self,
        messages: list[Message],
        temperature: float = DEFAULT_TEMPERATURE,
        max_turns_context: int | None = None,
        json_format: bool = False,
        structured_output_mode: str | None = None,
    ) -> str | dict:
        import time as _t
        _t0 = _t.time()
        if max_turns_context is not None and max_turns_context > 0:
            system_msgs = [m for m in messages if m["role"] == "system"]
            non_system = [m for m in messages if m["role"] != "system"]
            trimmed = non_system[-(max_turns_context * 2) :]
            messages = system_msgs + trimmed

        tokenizer = self._get_tokenizer()
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        pipe = self._get_pipeline()
        model = pipe.model

        do_sample = (self.do_sample if self.do_sample is not None
                     else temperature > 0)
        gen_config = self._build_gen_config(
            temperature=temperature,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=131072,
        )
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        # Phase 16.5 F1: when the experiment declares no output cap, bound only by
        # the model's remaining context so generation stops at EOS naturally.
        budget = gen_config.max_new_tokens
        if budget is None:
            budget = self._context_generation_budget(
                model, int(inputs["input_ids"].shape[1]))
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": budget,
            "do_sample": gen_config.do_sample,
            "use_cache": gen_config.use_cache,
            "pad_token_id": gen_config.pad_token_id,
            "eos_token_id": gen_config.eos_token_id,
        }
        if gen_config.temperature is not None:
            gen_kwargs["temperature"] = gen_config.temperature
        # Phase 16.5 F3: declared sampling parameters are passed explicitly, so a
        # model's own generation_config.json cannot silently override them.
        if self.top_p is not None:
            gen_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            gen_kwargs["top_k"] = self.top_k
        # Phase 14.1: constrained JSON decoding via logits processor
        mode = structured_output_mode if structured_output_mode is not None else self.structured_output_mode
        if mode == "constrained_json" and json_format:
            from .json_constrained_decoder import build_json_logits_processor
            logits_processor = build_json_logits_processor(tokenizer)
            if logits_processor:
                gen_kwargs["logits_processor"] = logits_processor
            # Also add } stopping criterion — constrained decoder ensures
            # } only appears at valid JSON positions, so stopping there
            # produces complete JSON with minimal token waste.
            stopping_criteria = _JSONStoppingCriteria.build(tokenizer, True)
            if stopping_criteria:
                gen_kwargs["stopping_criteria"] = stopping_criteria
        else:
            stopping_criteria = _JSONStoppingCriteria.build(tokenizer, json_format)
            if stopping_criteria:
                gen_kwargs["stopping_criteria"] = stopping_criteria
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[0][input_len:]
        reply = tokenizer.decode(new_ids, skip_special_tokens=True)

        # Per-call telemetry (Phase 17 Stage 2, Parts H/K). Recorded only; the
        # return value and the generation parameters are untouched. Nothing is
        # inferred here that was not observed: ``termination`` is "eos" only when
        # the final generated token is the model's EOS, and "max_new_tokens" when
        # the budget ran out first.
        last_token = int(new_ids[-1]) if new_ids.numel() else None
        self.last_generation_stats = {
            "model_id": self.model_id,
            "quantization": self.quantization,
            # The decoded reply exactly as the model produced it, before the
            # client's json_format parsing. Kept because a parse failure is only
            # diagnosable from the raw output (and a parser's return value cannot
            # tell a legitimate answer from a failure).
            "raw_reply": reply,
            "prompt_tokens": int(input_len),
            "generated_tokens": int(new_ids.shape[-1]),
            "context_tokens_total": int(input_len + new_ids.shape[-1]),
            "max_new_tokens_effective": int(budget),
            "hit_output_budget": int(new_ids.shape[-1]) >= int(budget),
            "last_token_id": last_token,
            "eos_token_ids": (
                [gen_config.eos_token_id]
                if isinstance(gen_config.eos_token_id, int)
                else list(gen_config.eos_token_id or [])
            ),
            # Three distinct outcomes, because "the model stopped" and "we cut it
            # off" are different facts: EOS, a stopping criterion (e.g. the JSON
            # object completed), or the output budget running out.
            "termination": _termination_reason(
                last_token, gen_config.eos_token_id,
                int(new_ids.shape[-1]), int(budget),
            ),
            "elapsed_s": round(_t.time() - _t0, 3),
            "json_format": json_format,
            "temperature": temperature,
            "do_sample": bool(gen_config.do_sample),
        }
        if torch.cuda.is_available():
            self.last_generation_stats["vram_allocated_gib"] = round(
                torch.cuda.memory_allocated() / 1024 ** 3, 3
            )
        logger.debug("HF %s replied (%d chars)", self.model_id, len(reply))
        import sys; print(f"[GEN] {self.model_id} temp={temperature} json={json_format} "
              f"ctx_tokens={input_len} new_tokens={new_ids.shape[-1]} "
              f"time={_t.time()-_t0:.1f}s", flush=True, file=sys.stderr)
        if json_format and reply:
            import json

            try:
                return json.loads(reply)
            except json.JSONDecodeError:
                extracted = _extract_json_block(reply)
                if extracted is not None:
                    return extracted
                return reply
        return reply


def build_llm(
    cfg: dict[str, Any],
    kwargs: dict[str, Any] | None = None,
    **gen_kwargs: Any,
) -> ChatLLM:
    """Build a local ``ChatLLM`` from a provider config dict.

    ``kwargs`` can override ``max_new_tokens``, ``device_map``,
    ``chat_template_kwargs``, and ``structured_output_mode``.
    The Qwen3-family thinking default is applied only where the caller did
    not already declare ``enable_thinking`` (Phase 17 Stage 2, F1).
    """
    if not kwargs:
        kwargs = {}
    kwargs.setdefault("max_new_tokens", int(cfg.get("max_new_tokens", 256)))
    kwargs.setdefault("device_map", cfg.get("device_map", "auto"))
    kwargs.setdefault("chat_template_kwargs", {})
    kwargs.setdefault("structured_output_mode", cfg.get("structured_output_mode"))
    kwargs.setdefault("quantization", cfg.get("quantization"))
    kwargs["chat_template_kwargs"] = _apply_thinking_default(
        dict(kwargs["chat_template_kwargs"]), cfg.get("model_id", "")
    )
    return HFLocalChatLLM(cfg["model_id"], **kwargs)
