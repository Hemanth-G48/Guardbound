"""Part 6 CONFIRMATION - who actually holds the weights, and what does a REAL
eviction cost?

Establishes, with measurements:
  (1) _evict() alone          -> no VRAM freed (cache still holds weights)
  (2) re-activate after _evict -> ~0 s (cache hit) => the model NEVER left the GPU
  (3) cache.clear() alone      -> frees only models whose backend handle was nulled
  (4) backend handle nulling   -> the other required half
  (5) a REAL eviction + reload -> the true swap cost
"""
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from guardbound.llm.model_manager import ModelManager  # noqa: E402
from guardbound.llm.local_client import HFLocalChatLLM, _pipeline_cache  # noqa: E402

GB = 1e9
ATTACKER = "Qwen/Qwen3.5-4B"
TARGET = "microsoft/Phi-4-mini-instruct"
OUT = []


def v():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return round(torch.cuda.memory_allocated() / GB, 3), round(free / GB, 3)


def rec(label):
    a, f = v()
    n = len(_pipeline_cache)
    OUT.append({"label": label, "pytorch_allocated_gb": a, "driver_free_gb": f,
                "cache_entries": n, "cache_keys": list(_pipeline_cache)})
    print(f"  {label:<44} alloc={a:7.3f}GB free={f:7.3f}GB cache={n}", flush=True)


def purge():
    _pipeline_cache.clear()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


print("=" * 92)
print("PART 6 CONFIRMATION - weight ownership + real eviction cost")
print("=" * 92)
purge(); rec("baseline (clean)")

# ---- load both
a_llm = HFLocalChatLLM(model_id=ATTACKER, device_map="cuda", max_new_tokens=32)
t_llm = HFLocalChatLLM(model_id=TARGET, device_map="cuda", max_new_tokens=32)
t0 = time.perf_counter(); a_llm._get_pipeline(); torch.cuda.synchronize()
att_load = time.perf_counter() - t0
rec(f"after loading attacker (cold {att_load:.1f}s)")

t0 = time.perf_counter(); t_llm._get_pipeline(); torch.cuda.synchronize()
tgt_load = time.perf_counter() - t0
rec(f"after loading target   (cold {tgt_load:.1f}s)")

free_before_evict = v()[1]

# ---- (1) ModelManager._evict on the target
mgr = ModelManager(device="cuda")
mgr.register("target", t_llm, TARGET)
mgr._evict("target")
rec("(1) after ModelManager._evict('target')")
free_after_evict = v()[1]
print(f"      -> VRAM freed by _evict(): {free_after_evict - free_before_evict:+.3f} GB", flush=True)

# ---- (2) re-activate the "evicted" model: cache hit or reload?
t0 = time.perf_counter()
t_llm._pipeline = None            # mimic what a fresh activate path would see
t_llm._get_pipeline()
torch.cuda.synchronize()
reactivate_s = time.perf_counter() - t0
rec(f"(2) re-activated target ({reactivate_s:.2f}s)")
print(f"      -> reactivation took {reactivate_s:.2f}s "
      f"({'CACHE HIT - weights never left the GPU' if reactivate_s < 2 else 'RELOAD'})",
      flush=True)

# ---- (3) cache.clear() only: attacker handle is still set, target's was nulled
a_alloc_before = v()[0]
_a_llm_handle = a_llm._pipeline          # deliberately KEEP the attacker handle
t_llm._pipeline = None
_pipeline_cache.clear()
gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
rec("(3) cache.clear(), attacker handle KEPT")
print("      -> attacker still resident (its backend attribute still references it)", flush=True)

# ---- (4) now also drop the backend handle
a_llm._pipeline = None
a_llm._tokenizer = None
_pipeline_cache.clear()
gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
rec("(4) cache.clear() + attacker handle nulled")
_a_llm_handle = None
gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
rec("(4b) local handle also released")

# ---- (5) real cold reload cost after a genuine eviction
t0 = time.perf_counter()
fresh = HFLocalChatLLM(model_id=TARGET, device_map="cuda", max_new_tokens=32)
fresh._get_pipeline()
torch.cuda.synchronize()
reload_s = time.perf_counter() - t0
rec(f"(5) genuine RELOAD of target ({reload_s:.1f}s)")

fresh._pipeline = None; fresh._tokenizer = None
purge(); rec("final cleanup")

summary = {
    "attacker_cold_load_s": round(att_load, 2),
    "target_cold_load_s": round(tgt_load, 2),
    "vram_freed_by_ModelManager_evict_gb": round(free_after_evict - free_before_evict, 3),
    "reactivation_after_evict_s": round(reactivate_s, 2),
    "genuine_reload_after_real_eviction_s": round(reload_s, 2),
    "conclusion": (
        "_evict() frees no VRAM because _pipeline_cache keeps the weights AND "
        "_get_pipeline() then returns them instantly. A real eviction requires "
        "BOTH dropping the backend's _pipeline attribute AND popping the "
        "_pipeline_cache entry; only then does VRAM drop, at the cost of a full "
        "cold reload."
    ),
}
print("\n" + json.dumps(summary, indent=2))
Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound\results\phase14\evict_mechanism_confirmation.json").write_text(
    json.dumps({"steps": OUT, "summary": summary}, indent=2), encoding="utf-8")
print("\nwrote results/phase14/evict_mechanism_confirmation.json")
