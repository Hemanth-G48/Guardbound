"""Isolate why device_map='cpu' lands on the GPU, and test whether real CPU
placement is achievable (device= vs device_map=)."""
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, pipeline  # noqa: E402

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
OUT = {}


def gpu_free():
    torch.cuda.synchronize()
    free, _ = torch.cuda.mem_get_info()
    return free / 1e9


def placement(pipe):
    devs = {}
    for _, p in pipe.model.named_parameters():
        devs[str(p.device)] = devs.get(str(p.device), 0) + p.numel()
    return devs


def trial(label, **kwargs):
    print(f"\n--- {label} :: kwargs={kwargs} ---", flush=True)
    gc.collect(); torch.cuda.empty_cache()
    f0 = gpu_free()
    try:
        p = pipeline("text-generation", model=MODEL, **kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        OUT[label] = {"error": f"{type(e).__name__}: {e}"}
        return
    f1 = gpu_free()
    devs = placement(p)
    print(f"  pipeline.device = {getattr(p, 'device', 'n/a')}", flush=True)
    print(f"  placement       = {devs}", flush=True)
    print(f"  gpu delta       = {f0 - f1:+.3f} GB", flush=True)
    OUT[label] = {"pipeline_device": str(getattr(p, "device", None)),
                  "placement": devs, "gpu_delta_gb": round(f0 - f1, 3),
                  "kwargs": {k: str(v) for k, v in kwargs.items()}}
    del p
    gc.collect(); torch.cuda.empty_cache()


trial("device_map_cpu", device_map="cpu")
trial("device_cpu", device="cpu")
trial("device_map_cpu_plus_device_cpu", device_map="cpu", device="cpu")

print("\n" + json.dumps(OUT, indent=2), flush=True)
Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound\results\phase14\cpu_placement_probe.json").write_text(
    json.dumps(OUT, indent=2), encoding="utf-8")
print("wrote results/phase14/cpu_placement_probe.json")
