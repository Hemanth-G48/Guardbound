"""A/B: does PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 inflate the
allocator's reservation during a model load?

Run twice, with and without the setting, in separate processes.
Usage: set ALLOC_CONF=<value|none> && python ab_alloc.py
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

# torch must not be imported before this is set.
_alloc = os.environ.get("ALLOC_CONF", "none")
if _alloc and _alloc.lower() != "none":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _alloc
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from guardbound.llm.local_client import HFLocalChatLLM, _pipeline_cache  # noqa: E402

GB = 1e9
MODEL = os.environ.get("AB_MODEL", "microsoft/Phi-4-mini-instruct")

_pipeline_cache.clear()
gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.synchronize()


def snap(tag):
    torch.cuda.synchronize()
    return {
        "tag": tag,
        "driver_free_gb": round(torch.cuda.mem_get_info()[0] / GB, 3),
        "allocated_gb": round(torch.cuda.memory_allocated() / GB, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / GB, 3),
    }


before = snap("before")
print(f"ALLOC_CONF={_alloc!r}  model={MODEL}", flush=True)
print("  before:", before, flush=True)

llm = HFLocalChatLLM(model_id=MODEL, device_map="cuda", max_new_tokens=32)
t0 = time.perf_counter()
llm._get_pipeline()
torch.cuda.synchronize()
load_s = time.perf_counter() - t0
after = snap("after")

transient = round(before["driver_free_gb"] - after["driver_free_gb"], 3)
live = round(after["allocated_gb"] - before["allocated_gb"], 3)
print(f"  load: {load_s:.1f}s", flush=True)
print("  after :", after, flush=True)
print(f"  DRIVER-FREE DELTA (transient footprint) : {transient:.3f} GB", flush=True)
print(f"  LIVE-ALLOCATION DELTA (resident weights): {live:.3f} GB", flush=True)
print(f"  reservation overhead                    : "
      f"{after['reserved_gb'] - after['allocated_gb']:.3f} GB", flush=True)

# what does an empty_cache reclaim afterwards?
f_before = torch.cuda.mem_get_info()[0] / GB
torch.cuda.empty_cache()
torch.cuda.synchronize()
f_after = torch.cuda.mem_get_info()[0] / GB
print(f"  empty_cache reclaimed                   : {f_after - f_before:+.3f} GB", flush=True)

out = {"alloc_conf": _alloc, "model": MODEL, "load_s": round(load_s, 2),
       "before": before, "after": after,
       "transient_footprint_gb": transient, "live_footprint_gb": live,
       "empty_cache_reclaimed_gb": round(f_after - f_before, 3)}
tag = "none" if _alloc.lower() == "none" else _alloc.replace(":", "_").replace(",", "-")
Path(f"results/phase14/ab_alloc_{tag}.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print(f"  wrote results/phase14/ab_alloc_{tag}.json", flush=True)
