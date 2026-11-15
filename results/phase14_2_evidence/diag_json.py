"""Diagnose the two findings from the smoke run.

(1) Is constrained JSON decoding actually in effect on the managed path, and
    does Qwen/Qwen3.5-4B produce parseable JSON at all?
(2) Why did the driver-level free VRAM drop by ~17 GB for an 8.5 GB model?
"""
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch  # noqa: E402

from guardbound.llm import local_client  # noqa: E402
from guardbound.llm.local_client import HFLocalChatLLM  # noqa: E402

GB = 1e9
ATTACKER = "Qwen/Qwen3.5-4B"
OUT = {}


def free():
    torch.cuda.synchronize()
    return round(torch.cuda.mem_get_info()[0] / GB, 3)


def stats(tag):
    torch.cuda.synchronize()
    d = {
        "tag": tag,
        "driver_free_gb": round(torch.cuda.mem_get_info()[0] / GB, 3),
        "allocated_gb": round(torch.cuda.memory_allocated() / GB, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / GB, 3),
        "cached_free_gb": round(
            (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / GB, 3
        ),
    }
    print("  %-28s driver_free=%7.3f alloc=%7.3f reserved=%7.3f cached_free=%7.3f"
          % (tag, d["driver_free_gb"], d["allocated_gb"], d["reserved_gb"],
             d["cached_free_gb"]), flush=True)
    return d


# A prompt shaped like the Crescendo step: the attack asks for a JSON object.
MSGS = [
    {"role": "system", "content": "You are a helpful red-teaming assistant. Reply with JSON only."},
    {"role": "user", "content": (
        "Produce the next question for a multi-turn dialogue.\n"
        "Reply with a single JSON object with the key \"generatedQuestion\" only. "
        "No prose, no preamble."
    )},
]

print("=" * 92)
print("DIAGNOSTIC 1 - constrained JSON on the managed path")
print("=" * 92)

local_client._pipeline_cache.clear()
gc.collect()
torch.cuda.empty_cache()
stats("baseline")

llm = HFLocalChatLLM(model_id=ATTACKER, device_map="cuda", max_new_tokens=256,
                     chat_template_kwargs={"enable_thinking": False},
                     structured_output_mode="constrained_json")
print(f"  llm.structured_output_mode   = {llm.structured_output_mode!r}", flush=True)
print(f"  llm.chat_template_kwargs     = {llm.chat_template_kwargs}", flush=True)

f0 = free()
t0 = time.perf_counter()
llm._get_pipeline()
torch.cuda.synchronize()
load_s = time.perf_counter() - t0
f1 = free()
print(f"  load: {load_s:.1f}s   driver free {f0} -> {f1}  (delta {f0 - f1:+.3f} GB)", flush=True)
after_load = stats("after load")

# Did the constrained decoder actually attach? Intercept gen_kwargs.
seen = {}
orig_generate = llm._get_pipeline().model.generate


def spy(*a, **kw):
    seen["logits_processor"] = "logits_processor" in kw
    seen["stopping_criteria"] = "stopping_criteria" in kw
    seen["max_new_tokens"] = kw.get("max_new_tokens")
    seen["do_sample"] = kw.get("do_sample")
    seen["temperature"] = kw.get("temperature")
    return orig_generate(*a, **kw)


llm._get_pipeline().model.generate = spy

results = []
for i in range(3):
    t0 = time.perf_counter()
    out = llm.generate(MSGS, temperature=0.7, json_format=True)
    dt = time.perf_counter() - t0
    parses = False
    if isinstance(out, dict):
        parses = True
    elif isinstance(out, str):
        try:
            json.loads(out)
            parses = True
        except json.JSONDecodeError:
            parses = False
    results.append({"trial": i + 1, "seconds": round(dt, 2), "parsed": parses,
                    "type": type(out).__name__,
                    "repr": (out if isinstance(out, str) else json.dumps(out))[:400]})
    print(f"  trial {i+1}: {dt:6.2f}s parsed={parses} repr={results[-1]['repr'][:200]!r}",
          flush=True)

OUT["generation_kwargs_seen"] = seen
OUT["trials"] = results
OUT["parse_success_rate"] = sum(1 for r in results if r["parsed"]) / len(results)
OUT["after_load_vram"] = after_load

print("\n  gen kwargs the backend actually used:", seen, flush=True)

llm._pipeline = None
llm._tokenizer = None
local_client._pipeline_cache.clear()
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

print("\n" + "=" * 92)
print("DIAGNOSTIC 2 - does empty_cache() reconcile driver free with the allocator?")
print("=" * 92)
before = stats("before empty_cache")
torch.cuda.empty_cache()
torch.cuda.synchronize()
after = stats("after empty_cache")
OUT["empty_cache_reconcile"] = {"before": before, "after": after,
                                "recovered_gb": round(after["driver_free_gb"] - before["driver_free_gb"], 3)}

print("\n" + json.dumps({k: v for k, v in OUT.items()
                         if k not in ("trials",)}, indent=2, default=str))
Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound\results\phase14\json_and_allocator_diagnosis.json").write_text(
    json.dumps(OUT, indent=2, default=str), encoding="utf-8")
print("\nwrote results/phase14/json_and_allocator_diagnosis.json")
