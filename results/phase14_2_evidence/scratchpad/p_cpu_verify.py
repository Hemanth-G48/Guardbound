"""VERIFY: does device_map='cpu' actually keep the model off the GPU?"""
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from guardbound.llm.local_client import HFLocalChatLLM, _pipeline_cache  # noqa: E402

EVALUATOR = "Qwen/Qwen3-4B-Instruct-2507"

_pipeline_cache.clear()
gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.synchronize()

free0, total0 = torch.cuda.mem_get_info()
print(f"free before: {free0/1e9:.3f} GB (total {total0/1e9:.3f})", flush=True)
print(f"torch.cuda.memory_allocated(): {torch.cuda.memory_allocated()/1e9:.3f} GB", flush=True)

llm = HFLocalChatLLM(model_id=EVALUATOR, device_map="cpu", max_new_tokens=32)
print(f"declared device_map: {llm.device_map}", flush=True)
pipe = llm._get_pipeline()
print(f"pipeline.device: {getattr(pipe, 'device', 'n/a')}", flush=True)
print(f"pipeline.__class__: {type(pipe).__name__}", flush=True)

free1, _ = torch.cuda.mem_get_info()
print(f"free after : {free1/1e9:.3f} GB  (delta {(free0-free1)/1e9:+.3f} GB)", flush=True)
print(f"torch.cuda.memory_allocated(): {torch.cuda.memory_allocated()/1e9:.3f} GB", flush=True)

# where do the parameters and buffers actually live?
devices = {}
for name, p in pipe.model.named_parameters():
    devices[str(p.device)] = devices.get(str(p.device), 0) + p.numel()
for name, b in pipe.model.named_buffers():
    devices[str(b.device)] = devices.get(str(b.device), 0) + b.numel()
print("\nparameter/buffer placement (element counts by device):", flush=True)
for d, n in sorted(devices.items(), key=lambda kv: -kv[1]):
    print(f"  {d:<12} {n:>14,} elements ({n*2/1e9:.3f} GB @ bf16)", flush=True)

# accelerate dispatch record, if any
disp = getattr(pipe.model, "hf_device_map", None)
if disp:
    from collections import Counter
    c = Counter(str(v) for v in disp.values())
    print(f"\nhf_device_map layer placement: {dict(c)}", flush=True)
else:
    print("\nhf_device_map: absent", flush=True)

# generation
msgs = [{"role": "user", "content": "Say OK."}]
import time
t0 = time.perf_counter()
out = llm.generate(msgs, temperature=0.0)
dt = time.perf_counter() - t0
free2, _ = torch.cuda.mem_get_info()
print(f"\ngeneration: {dt:.2f}s, {len(out)} chars", flush=True)
print(f"free after gen: {free2/1e9:.3f} GB", flush=True)

verdict = ("CPU-ONLY (confirmed)" if (free0 - free1) < 1.0
           else f"NOT CPU-ONLY - {(free0-free1)/1e9:.3f} GB landed on the GPU")
print(f"\nVERDICT: {verdict}", flush=True)

Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound\results\phase14\cpu_evaluator_verification.json").write_text(
    json.dumps({
        "declared_device_map": llm.device_map,
        "pipeline_device": str(getattr(pipe, "device", None)),
        "gpu_free_before_gb": round(free0 / 1e9, 3),
        "gpu_free_after_load_gb": round(free1 / 1e9, 3),
        "gpu_delta_gb": round((free0 - free1) / 1e9, 3),
        "param_placement_by_device": devices,
        "verdict": verdict,
        "generation_s": round(dt, 2),
    }, indent=2), encoding="utf-8")
print("wrote results/phase14/cpu_evaluator_verification.json")
