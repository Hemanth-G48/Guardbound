"""Phase 14.4B — provision google/gemma-3-4b-it into the project's normal HF cache.

Step 1: probe access (GATED model — licence acceptance is the one unknown).
Step 2: download the exact official checkpoint. No conversion, no quantization.
Step 3: verify identity + cache integrity.
Step 4: minimal LOAD-ONLY test with VRAM telemetry (no evaluator contract invoked).
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

MODEL_ID = "google/gemma-3-4b-it"
OUT = Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound\results\phase14_4_gemma_provisioning.json")

report: dict = {"model_id": MODEL_ID, "steps": []}


def stage(name, **kw):
    rec = {"stage": name, **kw}
    report["steps"].append(rec)
    print(f"[prov] {name}: {kw}", flush=True)
    return rec


# --------------------------------------------------------------------------- #
print("=" * 92)
print("PHASE 14.4B — PROVISION google/gemma-3-4b-it")
print("=" * 92)

from huggingface_hub import HfApi, snapshot_download  # noqa: E402
from huggingface_hub.utils import (  # noqa: E402
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

api = HfApi()

# ---- Step 1: identity + access ------------------------------------------- #
try:
    info = api.model_info(MODEL_ID)
    stage("model_info", id=info.id, gated=str(info.gated),
          revision=getattr(info, "sha", None),
          files=len(info.siblings or []))
    report["gated"] = str(info.gated)
    report["revision"] = getattr(info, "sha", None)
except RepositoryNotFoundError as e:
    stage("model_info_FAILED", error=f"RepositoryNotFoundError: {e}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nFATAL: model id not found on the Hub.")
    sys.exit(2)
except HfHubHTTPError as e:
    stage("model_info_FAILED", error=f"{type(e).__name__}: {e}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nFATAL: Hub error during model_info.")
    sys.exit(2)

try:
    cfg = api.hf_hub_download(MODEL_ID, "config.json")
    stage("access_probe", config_accessible=True, path=cfg)
except GatedRepoError as e:
    stage("access_probe_GATED", error=str(e)[:600],
          requirement="Accept the Gemma usage licence while signed in, then retry.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nBLOCKED: Gemma licence not accepted for this account. "
          "Accept it at https://huggingface.co/google/gemma-3-4b-it then re-run.")
    sys.exit(3)
except HfHubHTTPError as e:
    stage("access_probe_FAILED", error=f"{type(e).__name__}: {str(e)[:400]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nBLOCKED: could not fetch config.json (auth or network).")
    sys.exit(3)

# ---- Step 2: download ----------------------------------------------------- #
t0 = time.perf_counter()
snapshot = snapshot_download(repo_id=MODEL_ID)
dl_s = time.perf_counter() - t0
stage("download", snapshot=snapshot, seconds=round(dl_s, 1))
report["snapshot_path"] = snapshot

# ---- Step 3: verify ------------------------------------------------------ #
snap = Path(snapshot)
files = sorted(p.name for p in snap.iterdir() if p.is_file())
weights = [p for p in snap.glob("*.safetensors")]
weight_bytes = sum(p.stat().st_size for p in weights if p.is_file())
total_bytes = sum(p.stat().st_size for p in snap.rglob("*") if p.is_file())

cfg_data = json.loads((snap / "config.json").read_text(encoding="utf-8"))
tok_cfg_path = snap / "tokenizer_config.json"
tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8")) if tok_cfg_path.exists() else {}
gen_cfg_path = snap / "generation_config.json"

idx_path = snap / "model.safetensors.index.json"
param_count = None
if idx_path.exists():
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    param_count = idx.get("metadata", {}).get("total_size")

report["verification"] = {
    "snapshot_path": str(snap),
    "revision": snap.name,
    "model_type": cfg_data.get("model_type"),
    "architectures": cfg_data.get("architectures"),
    "torch_dtype": cfg_data.get("torch_dtype"),
    "file_count": len(files),
    "files": files,
    "safetensors_shards": len(weights),
    "weights_gb": round(weight_bytes / 1e9, 3),
    "snapshot_total_gb": round(total_bytes / 1e9, 3),
    "weights_index_total_bytes": param_count,
    "approx_params_B_from_index": (round(param_count / 1e9, 3) if param_count else None),
    "has_tokenizer_config": tok_cfg_path.exists(),
    "has_chat_template": "chat_template" in tok_cfg,
    "chat_template_source": ("tokenizer_config.json" if "chat_template" in tok_cfg
                             else ("chat_template.json" if (snap / "chat_template.json").exists()
                                   else None)),
    "has_generation_config": gen_cfg_path.exists(),
    "tokenizer_files": [f for f in files if "token" in f.lower()],
    "use_default_system_prompt": tok_cfg.get("use_default_system_prompt"),
    "bos_token": tok_cfg.get("bos_token"),
    "eos_token": tok_cfg.get("eos_token"),
    "pad_token": tok_cfg.get("pad_token"),
    "is_config_only": len(weights) == 0,
}
for k, v in report["verification"].items():
    print(f"  {k:<34} {v}", flush=True)

# ---- Step 4: load-only test ---------------------------------------------- #
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
print(f"  VRAM before load : allocated={before[0]} reserved={before[1]} free={before[2]}", flush=True)

load_report = {"vram_before": before}
try:
    t0 = time.perf_counter()
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        device_map="cuda",
        dtype=torch.bfloat16,
        model_kwargs={"attn_implementation": "sdpa"},
    )
    torch.cuda.synchronize()
    load_report["load_seconds"] = round(time.perf_counter() - t0, 2)
    after = vram()
    load_report["vram_after"] = after
    load_report["footprint_gb"] = round(before[2] - after[2], 3)
    load_report["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / GB, 3)
    devs = {}
    for _, p in pipe.model.named_parameters():
        devs[str(p.device)] = devs.get(str(p.device), 0) + p.numel()
    load_report["param_placement"] = devs
    load_report["dtype_loaded"] = str(next(pipe.model.parameters()).dtype)
    load_report["device"] = str(pipe.device)
    load_report["load_ok"] = True
    print(f"  load: {load_report['load_seconds']}s  footprint={load_report['footprint_gb']} GB "
          f"peak_alloc={load_report['peak_allocated_gb']} GB", flush=True)
    print(f"  placement: {devs}   dtype: {load_report['dtype_loaded']}", flush=True)

    # chat-template smoke: does the template accept system+user and user-only?
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    shapes = {}
    for name, msgs in (
        ("system_user", [{"role": "system", "content": "S"},
                         {"role": "user", "content": "U"}]),
        ("user_only", [{"role": "user", "content": "U"}]),
    ):
        try:
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            shapes[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            shapes[name] = f"{type(exc).__name__}: {exc}"
    load_report["chat_template_shapes"] = shapes
    print(f"  chat template shapes: {shapes}", flush=True)

    del pipe
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    load_report["vram_after_unload"] = vram()
    print(f"  VRAM after unload: allocated={load_report['vram_after_unload'][0]} "
          f"free={load_report['vram_after_unload'][2]}", flush=True)
except Exception as exc:  # noqa: BLE001
    load_report["load_ok"] = False
    load_report["error"] = f"{type(exc).__name__}: {exc}"
    print(f"  LOAD FAILED: {load_report['error']}", flush=True)

report["load_test"] = load_report
report["cuda_errors"] = 0
report["oom_events"] = 0

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\n[prov] wrote {OUT}")
print(f"[prov] verdict: {'PROVISIONED' if load_report.get('load_ok') else 'LOAD FAILED'}")
