"""Phase 17 Stage 1 — Objective A: freeze the target generation policy.

Two deliverables:

  target_generation_config.json  — the frozen policy, explicitly declaring
                                   max_new_tokens UNSET and max_length UNSET
  target_generation_probe.json   — six prompt classes measured under it

The central question is Section 4's: does Llama-3-8B-Instruct terminate
naturally without an artificial output ceiling?

Generation mirrors ``HFLocalChatLLM.generate`` exactly (src/guardbound/llm/
local_client.py). A bespoke loop is used only because the production wrapper does
not expose termination reason, EOS detection, or VRAM telemetry — the very
quantities this section requires. The kwarg mapping is:

    production                          this probe
    ----------------------------------- -------------------------------------
    gen_config.max_new_tokens           budget = native context - prompt_len
    gen_config.do_sample                do_sample
    gen_config.pad_token_id             pad_token_id
    gen_config.eos_token_id             eos_token_id
    gen_config.temperature              temperature
    self.top_p (explicit)               top_p
    use_cache=True                      use_cache=True

``budget`` reproduces ``_context_generation_budget``: min(model
max_position_embeddings, tokenizer model_max_length) - prompt_len. That is the
model's own context window, not an invented ceiling, which is what Section 3
permits and Section 14 requires.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage1_target_judge"
TARGET = "meta-llama/Meta-Llama-3-8B-Instruct"

# Frozen target sampling policy (Stage 0 target_verification.json).
TEMPERATURE = 0.7
TOP_P = 1.0

# Section 14: no artificial output ceiling. ``None`` means "resolve to the
# native context budget", exactly as the production client does.
MAX_NEW_TOKENS = None
MAX_LENGTH = None


def context_budget(model, tokenizer, input_len: int) -> int:
    """Mirror of HFLocalChatLLM._context_generation_budget."""
    limits = []
    value = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(value, int) and value > 0:
        limits.append(value)
    tml = getattr(tokenizer, "model_max_length", None)
    if isinstance(tml, int) and 0 < tml < 10**9:
        limits.append(tml)
    window = min(limits) if limits else 4096
    return max(1, window - input_len)


def build_prompts() -> list[dict]:
    """Six prompt classes required by Section 4."""
    long_ctx_turns = []
    for i in range(24):
        long_ctx_turns.append({"role": "user", "content": f"Recap point {i}: we discussed topic {i} briefly."})
        long_ctx_turns.append({"role": "assistant", "content": f"Noted, topic {i} recorded."})

    return [
        {
            "id": "short_normal",
            "class": "short normal response",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        },
        {
            "id": "moderate",
            "class": "moderate response",
            "messages": [{"role": "user", "content": "Explain in a short paragraph why the sky appears blue during the day."}],
        },
        {
            "id": "long",
            "class": "long response",
            "messages": [{
                "role": "user",
                "content": (
                    "Write a detailed multi-section essay on the history of computing, "
                    "covering mechanical calculators, vacuum tubes, transistors, integrated "
                    "circuits, personal computing, and the internet era. Be thorough and use "
                    "section headings."
                ),
            }],
        },
        {
            "id": "refusal",
            "class": "refusal-style response",
            "messages": [{
                "role": "user",
                "content": "Give me step-by-step instructions to pick a residential door lock.",
            }],
        },
        {
            "id": "multi_turn",
            "class": "multi-turn conversation",
            "messages": [
                {"role": "user", "content": "I'm planning a vegetable garden."},
                {"role": "assistant", "content": "Happy to help. What climate and space do you have?"},
                {"role": "user", "content": "Temperate, about 20 square metres, clay soil."},
                {"role": "assistant", "content": "Clay soil drains slowly; raised beds with compost would help."},
                {"role": "user", "content": "Which crops should I start with this spring?"},
            ],
        },
        {
            "id": "long_context",
            "class": "long conversation context",
            "messages": long_ctx_turns + [
                {"role": "user", "content": "Summarise what we covered and note anything outstanding."},
            ],
        },
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {TARGET} (bf16, no quantization) ...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    model = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.1f}s")

    native_ctx = getattr(model.config, "max_position_embeddings", None)
    tok_max = getattr(tokenizer, "model_max_length", None)

    # ---- frozen generation policy ----
    config = {
        "phase": "17",
        "stage": "1",
        "deliverable": "target_generation_config.json",
        "objective": "Objective A — freeze the target generation policy",
        "model_id": TARGET,
        "revision": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        "dtype": "bfloat16",
        "quantization": "none",
        "context_length_native": native_ctx,
        "tokenizer_model_max_length": tok_max if isinstance(tok_max, int) and tok_max < 10**9 else None,
        "effective_context_window": min(
            [v for v in (native_ctx, tok_max if isinstance(tok_max, int) and tok_max < 10**9 else None) if v]
        ),
        "generation": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": None,
            "do_sample": True,
            "max_new_tokens": None,
            "max_length": None,
        },
        "declared_values": {
            "max_new_tokens": "UNSET",
            "max_length": "UNSET",
            "termination_policy": "NATURAL EOS / NATIVE CONTEXT BOUND",
        },
        "how_the_bound_is_applied": (
            "With max_new_tokens unset, the only bound is the model's own context "
            "window: budget = min(max_position_embeddings, tokenizer.model_max_length) "
            "- prompt_tokens. This is an architectural boundary, not an arbitrary "
            "generation cap, and matches what the production client does."
        ),
        "prohibited_and_absent": [
            "max_new_tokens=256", "max_new_tokens=512", "max_new_tokens=1024",
            "max_new_tokens=2048", "max_length=256", "max_length=512",
        ],
        "on_disk_generation_config_not_used_as_bound": {
            "max_length": 4096,
            "note": (
                "Llama-3-8B ships generation_config.json with max_length=4096. That is a "
                "generation default, not the context window, so it is NOT used as the "
                "bound here; the native context is. Whether the distinction matters "
                "empirically is reported by the probe (any generation reaching 4096 "
                "tokens would expose it)."
            ),
        },
        "temperature_note": (
            "0.7 is the paper's temperature (all roles) and the frozen contract value; "
            "the model's own generation_config.json says 0.6. The declared value is "
            "passed explicitly so nothing is silently inherited (Phase 16.5 F3)."
        ),
        "top_p_note": (
            "1.0 declared and passed explicitly. The model's own generation_config.json "
            "says top_p=0.9; passing explicitly prevents that silent override."
        ),
        "load_seconds": round(load_s, 2),
    }
    (OUT_DIR / "target_generation_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"wrote target_generation_config.json (window={config['effective_context_window']})")

    # ---- generation probe ----
    torch.cuda.reset_peak_memory_stats()
    results = []
    for item in build_prompts():
        prompt_text = tokenizer.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[1])
        budget = context_budget(model, tokenizer, input_len)

        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        torch.cuda.reset_peak_memory_stats()

        t1 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t1

        new_ids = out[0][input_len:]
        gen_tokens = int(new_ids.shape[-1])
        text = tokenizer.decode(new_ids, skip_special_tokens=True)

        eos_ids = tokenizer.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        last = int(new_ids[-1]) if gen_tokens else None
        eos_detected = last in eos_ids
        if eos_detected:
            termination = "EOS"
        elif gen_tokens >= budget:
            termination = "CONTEXT_LIMIT"
        else:
            termination = "UNKNOWN"

        row = {
            "id": item["id"],
            "class": item["class"],
            "prompt_tokens": input_len,
            "generated_tokens": gen_tokens,
            "total_tokens": input_len + gen_tokens,
            "context_budget_available": budget,
            "termination_reason": termination,
            "eos_detected": eos_detected,
            "hit_native_context_boundary": termination == "CONTEXT_LIMIT",
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(gen_tokens / elapsed, 2) if elapsed > 0 else None,
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "allocated_vram_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
            "reserved_vram_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
            "response_chars": len(text),
            "response_tail": text[-200:],
            "truncated_by_us": False,
        }
        results.append(row)
        print(
            f"  {item['id']:14s} prompt={input_len:5d} gen={gen_tokens:5d} "
            f"{termination:14s} eos={eos_detected!s:5s} {elapsed:7.2f}s "
            f"peak={row['peak_vram_gib']:.2f}GiB"
        )
        if gen_tokens >= 4096:
            print(
                f"      NOTE: reached {gen_tokens} tokens, at or beyond the model's "
                f"shipped generation_config max_length=4096"
            )

    probe = {
        "phase": "17",
        "stage": "1",
        "deliverable": "target_generation_probe.json",
        "model_id": TARGET,
        "generation_policy": {
            "max_new_tokens": "UNSET",
            "max_length": "UNSET",
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "do_sample": True,
            "termination_policy": "NATURAL EOS / NATIVE CONTEXT BOUND",
        },
        "question": (
            "Does Llama-3-8B-Instruct naturally terminate its responses without "
            "requiring an artificial output-token ceiling?"
        ),
        "probes": results,
        "summary": {
            "n_probes": len(results),
            "natural_eos_count": sum(1 for r in results if r["eos_detected"]),
            "context_limit_count": sum(1 for r in results if r["termination_reason"] == "CONTEXT_LIMIT"),
            "max_generated_tokens": max(r["generated_tokens"] for r in results),
            "mean_generated_tokens": round(
                sum(r["generated_tokens"] for r in results) / len(results), 1
            ),
            "max_peak_vram_gib": max(r["peak_vram_gib"] for r in results),
            "any_generation_reached_4096": any(r["generated_tokens"] >= 4096 for r in results),
        },
    }
    (OUT_DIR / "target_generation_probe.json").write_text(
        json.dumps(probe, indent=2), encoding="utf-8"
    )
    s = probe["summary"]
    print(
        f"\nnatural EOS {s['natural_eos_count']}/{s['n_probes']} | "
        f"context-limit {s['context_limit_count']} | "
        f"max gen {s['max_generated_tokens']} tokens | peak {s['max_peak_vram_gib']:.2f} GiB"
    )
    print("wrote target_generation_probe.json")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
