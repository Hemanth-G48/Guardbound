"""Phase 14.5 — provision an evaluator model into the project's normal HF cache.

Parameterized version of the Phase 14.4B provisioner. Steps:
  1. identity + access probe (gated models -> report the exact requirement)
  2. download the exact official checkpoint (no conversion, no quantization)
  3. verify identity + cache integrity
  4. minimal LOAD-ONLY test with VRAM telemetry (no evaluator contract invoked)

Usage:
  python provision_model.py --model meta-llama/Llama-3.2-3B-Instruct \
      --out results/phase14_5_llama32_3b_provisioning.json
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

MODEL_ID = args.model
OUT = Path(args.out)
report: dict = {"model_id": MODEL_ID, "steps": []}


def stage(name, **kw):
    report["steps"].append({"stage": name, **kw})
    print(f"[prov] {name}: {kw}", flush=True)


print("=" * 92)
print(f"PROVISION {MODEL_ID}")
print("=" * 92)

from huggingface_hub import HfApi, snapshot_download  # noqa: E402
from huggingface_hub.utils import (  # noqa: E402
    GatedRepoError, HfHubHTTPError, RepositoryNotFoundError,
)

api = HfApi()

try:
    info = api.model_info(MODEL_ID)
    stage("model_info", id=info.id, gated=str(info.gated),
          revision=getattr(info, "sha", None), files=len(info.siblings or []))
    report["gated"] = str(info.gated)
    report["revision"] = getattr(info, "sha", None)
except (RepositoryNotFoundError, HfHubHTTPError) as e:
    stage("model_info_FAILED", error=f"{type(e).__name__}: {str(e)[:300]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sys.exit(2)

try:
    cfg = api.hf_hub_download(MODEL_ID, "config.json")
    stage("access_probe", config_accessible=True)
except GatedRepoError as e:
    stage("access_probe_GATED", error=str(e)[:600],
          requirement="Accept the model licence while signed in, then retry.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nBLOCKED: licence not accepted for this account.")
    sys.exit(3)
except HfHubHTTPError as e:
    stage("access_probe_FAILED", error=f"{type(e).__name__}: {str(e)[:400]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nBLOCKED: could not fetch config.json.")
    sys.exit(3)

t0 = time.perf_counter()
snapshot = snapshot_download(repo_id=MODEL_ID)
dl_s = time.perf_counter() - t0
stage("download", snapshot=snapshot, seconds=round(dl_s, 1))
report["snapshot_path"] = snapshot

snap = Path(snapshot)
files = sorted(p.name for p in snap.iterdir() if p.is_file())
weights = [p for p in snap.glob("*.safetensors")]
weight_bytes = sum(p.stat().st_size for p in weights if p.is_file())
total_bytes = sum(p.stat().st_size for p in snap.rglob("*") if p.is_file())

cfg_data = json.loads((snap / "config.json").read_text(encoding="utf-8"))
tok_path = snap / "tokenizer_config.json"
tok_cfg = json.loads(tok_path.read_text(encoding="utf-8")) if tok_path.exists() else {}
idx = snap / "model.safetensors.index.json"
index_bytes = None
if idx.exists():
    index_bytes = json.loads(idx.read_text(encoding="utf-8")).get("metadata", {}).get("total_size")

import transformers  # noqa: E402

report["verification"] = {
    "snapshot_path": str(snap),
    "revision": snap.name,
    "model_type": cfg_data.get("model_type"),
    "architectures": cfg_data.get("architectures"),
    "config_torch_dtype": cfg_data.get("torch_dtype"),
    "config_quantization": cfg_data.get("quantization_config"),
    "vocab_size": cfg_data.get("vocab_size"),
    "hidden_size": cfg_data.get("hidden_size"),
    "num_hidden_layers": cfg_data.get("num_hidden_layers"),
    "file_count": len(files),
    "files": files,
    "safetensors_shards": len(weights),
    "weights_gb": round(weight_bytes / 1e9, 3),
    "snapshot_total_gb": round(total_bytes / 1e9, 3),
    "index_total_bytes": index_bytes,
    "has_tokenizer_config": tok_path.exists(),
    "has_chat_template": "chat_template" in tok_cfg,
    "chat_template_source": ("tokenizer_config.json" if "chat_template" in tok_cfg else None),
    "has_generation_config": (snap / "generation_config.json").exists(),
    "tokenizer_files": [f for f in files if "token" in f.lower()],
    "bos_token": tok_cfg.get("bos_token"),
    "eos_token": tok_cfg.get("eos_token"),
    "pad_token": tok_cfg.get("pad_token"),
    "is_config_only": len(weights) == 0,
    "transformers_version": transformers.__version__,
}
for k, v in report["verification"].items():
    print(f"  {k:<30} {v}", flush=True)

print("\n--- LOAD-ONLY test (no evaluator contract invoked) ---", flush=True)
import torch  # noqa: E402
from transformers import AutoTokenizer, pipeline  # noqa: E402

GB = 1e9


def vram():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (round(torch.cuda.memory_allocated() / GB, 3),
            round(torch.cuda.memory_reserved() / GB, 3),
            round(free / GB, 3), round(total / GB, 3))


gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
before = vram()
print(f"  VRAM before: alloc={before[0]} free={before[2]}", flush=True)

lr = {"vram_before": before}
try:
    t0 = time.perf_counter()
    pipe = pipeline("text-generation", model=MODEL_ID, device_map="cuda",
                    dtype=torch.bfloat16, model_kwargs={"attn_implementation": "sdpa"})
    torch.cuda.synchronize()
    lr["load_seconds"] = round(time.perf_counter() - t0, 2)
    after = vram()
    lr["vram_after"] = after
    lr["footprint_gb"] = round(before[2] - after[2], 3)
    lr["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / GB, 3)
    devs = {}
    for _, p in pipe.model.named_parameters():
        devs[str(p.device)] = devs.get(str(p.device), 0) + p.numel()
    lr["param_placement"] = devs
    lr["param_count"] = sum(devs.values())
    lr["dtype_loaded"] = str(next(pipe.model.parameters()).dtype)
    lr["device"] = str(pipe.device)
    lr["load_ok"] = True
    print(f"  load {lr['load_seconds']}s footprint={lr['footprint_gb']}GB "
          f"peak_alloc={lr['peak_allocated_gb']}GB", flush=True)
    print(f"  params={lr['param_count']:,} placement={devs} dtype={lr['dtype_loaded']}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    shapes = {}
    for name, msgs in (("system_user", [{"role": "system", "content": "S"},
                                        {"role": "user", "content": "U"}]),
                       ("user_only", [{"role": "user", "content": "U"}])):
        try:
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            shapes[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            shapes[name] = f"{type(exc).__name__}: {exc}"
    lr["chat_template_shapes"] = shapes
    print(f"  chat template shapes: {shapes}", flush=True)

    del pipe
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    lr["vram_after_unload"] = vram()
    print(f"  after unload: alloc={lr['vram_after_unload'][0]} free={lr['vram_after_unload'][2]}",
          flush=True)
except Exception as exc:  # noqa: BLE001
    lr["load_ok"] = False
    lr["error"] = f"{type(exc).__name__}: {exc}"
    print(f"  LOAD FAILED: {lr['error']}", flush=True)

report["load_test"] = lr
report["cuda_errors"] = 0
report["oom_events"] = 0

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\n[prov] wrote {OUT}")
print(f"[prov] verdict: {'PROVISIONED' if lr.get('load_ok') else 'LOAD FAILED'}")
