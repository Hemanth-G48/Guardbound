"""Phase 17 — prove the 4-bit quantization runtime actually works on THIS machine.

Section 10 permits quantization only when the runtime genuinely supports it, and
Section 9 requires the configuration to be recorded rather than described as
"quantized". A successful `pip install` proves neither. This script therefore
performs a real NF4 load and a real generation, and writes what it measured.

Deliberately tested on the smallest cached model rather than a candidate, so a
runtime failure costs seconds instead of a 52 GiB download.

Two machine-specific facts are handled explicitly:
  * torch will not import without KMP_DUPLICATE_LIB_OK on Windows here (a
    duplicate libiomp5md.dll), as documented in crash_investigation/;
  * console capture is unreliable in this environment, so the verdict is written
    to JSON and stdout is treated as a convenience only.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "phase17_model_qualification" / "quant_runtime_verification.json"

# Smallest already-cached instruct model — purely a runtime probe.
PROBE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

report: dict = {
    "phase": "17",
    "deliverable": "quant_runtime_verification.json",
    "probe_model": PROBE_MODEL,
    "purpose": (
        "Prove NF4 4-bit quantization loads and generates on this machine before "
        "committing to the official-checkpoint download for each candidate."
    ),
}


def step(name, fn):
    try:
        value = fn()
        report[name] = value
        print(f"  {name}: {value}")
        return value
    except Exception as exc:  # noqa: BLE001 - the failure itself is the finding
        report[name] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  {name}: FAILED — {type(exc).__name__}: {exc}")
        return None


print("=== environment ===")
import torch  # noqa: E402

step("torch_version", lambda: torch.__version__)
step("torch_cuda_version", lambda: torch.version.cuda)
step("cuda_available", lambda: torch.cuda.is_available())
step("gpu_name", lambda: torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
report["compute_capability"] = f"sm_{cap[0]}{cap[1]}"
print(f"  compute_capability: {report['compute_capability']}")
report["vram_total_gib"] = round(
    torch.cuda.get_device_properties(0).total_memory / 1024**3, 3
)
print(f"  vram_total_gib: {report['vram_total_gib']}")

print("\n=== bitsandbytes ===")
import bitsandbytes as bnb  # noqa: E402

step("bitsandbytes_version", lambda: bnb.__version__)

# The real test: does the 4-bit CUDA path actually engage?
print("\n=== NF4 load + generation ===")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
report["quantization_config"] = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
    "method": "bitsandbytes",
    "tier": "post-training quantization at load time (Section 9 'PTQ')",
}

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

t0 = time.perf_counter()
try:
    tokenizer = AutoTokenizer.from_pretrained(PROBE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        PROBE_MODEL,
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    report["load_ok"] = True
    report["load_seconds"] = round(time.perf_counter() - t0, 2)
    print(f"  load_ok: True ({report['load_seconds']}s)")
except Exception as exc:  # noqa: BLE001
    report["load_ok"] = False
    report["load_error"] = f"{type(exc).__name__}: {exc}"
    print(f"  load_ok: False — {report['load_error']}")

if report.get("load_ok"):
    report["vram_allocated_after_load_gib"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
    report["vram_peak_after_load_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    print(f"  allocated after load: {report['vram_allocated_after_load_gib']} GiB")
    print(f"  peak after load     : {report['vram_peak_after_load_gib']} GiB")

    # Module census, not a single-parameter peek. bitsandbytes deliberately
    # leaves embeddings and the LM head in the compute dtype, so inspecting
    # `next(model.parameters())` reports bf16 even when quantization succeeded.
    import bitsandbytes.nn as bnn  # noqa: E402

    n_linear4bit = sum(1 for m in model.modules() if isinstance(m, bnn.Linear4bit))
    n_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    total_params = sum(p.numel() for p in model.parameters())
    bnb_params = sum(
        p.numel() for m in model.modules() if isinstance(m, bnn.Linear4bit)
        for p in m.parameters()
    )
    report["linear4bit_modules"] = n_linear4bit
    report["plain_linear_modules"] = n_linear
    report["total_params"] = total_params
    report["params_in_4bit_layers"] = bnb_params
    report["frac_params_quantized"] = round(bnb_params / total_params, 4) if total_params else None
    print(
        f"  Linear4bit modules: {n_linear4bit} | plain Linear: {n_linear} | "
        f"{report['frac_params_quantized']:.1%} of params in 4-bit layers"
    )

    messages = [{"role": "user", "content": "Reply with exactly the word: READY"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    t1 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    report["generate_seconds"] = round(time.perf_counter() - t1, 3)

    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    report["generated_text"] = text.strip()[:200]
    report["generate_ok"] = bool(text.strip())
    report["vram_peak_during_generate_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    print(f"  generated: {report['generated_text']!r}")
    print(f"  generate_ok: {report['generate_ok']}")
    print(f"  peak VRAM during generate: {report['vram_peak_during_generate_gib']} GiB")

    # Hard proof that quantization changed the footprint: same model, bf16.
    print("\n=== control: same model in bf16 ===")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        t2 = time.perf_counter()
        model_bf16 = AutoModelForCausalLM.from_pretrained(
            PROBE_MODEL, device_map={"": 0}, dtype=torch.bfloat16
        )
        report["bf16_load_ok"] = True
        report["bf16_load_seconds"] = round(time.perf_counter() - t2, 2)
        report["bf16_vram_allocated_gib"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
        print(f"  bf16 allocated: {report['bf16_vram_allocated_gib']} GiB")
        report["vram_reduction_factor"] = round(
            report["bf16_vram_allocated_gib"] / report["vram_allocated_after_load_gib"], 2
        )
        print(f"  NF4 reduces footprint by {report['vram_reduction_factor']}x")
        del model_bf16
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        report["bf16_load_ok"] = False
        report["bf16_error"] = f"{type(exc).__name__}: {exc}"
        print(f"  bf16 control FAILED — {report['bf16_error']}")

report["verdict"] = (
    "NF4_RUNTIME_VERIFIED"
    if report.get("load_ok") and report.get("generate_ok")
    else "NF4_RUNTIME_UNAVAILABLE"
)
print(f"\nVERDICT: {report['verdict']}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"wrote {OUT.relative_to(REPO_ROOT)}")
