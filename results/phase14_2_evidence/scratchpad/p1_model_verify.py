"""Part 1 - model verification. READ ONLY. No model loading, no VRAM use."""
import json
import os
from pathlib import Path

HUB = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"

TARGETS = {
    "attacker": "Qwen/Qwen3.5-4B-Instruct",
    "attacker_base_alt": "Qwen/Qwen3.5-4B",
    "target": "microsoft/Phi-4-mini-instruct",
    "evaluator": "Qwen/Qwen3-4B-Instruct-2507",
}


def dirname(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def snapshot_dir(model_id: str):
    base = HUB / dirname(model_id)
    snap = base / "snapshots"
    if not snap.exists():
        return None, base
    revs = sorted([d for d in snap.iterdir() if d.is_dir()], key=lambda p: p.name)
    return (revs[-1] if revs else None), base


def inspect(label: str, model_id: str) -> dict:
    snap, base = snapshot_dir(model_id)
    out = {"role": label, "model_id": model_id, "cache_dir": str(base),
           "cached": base.exists()}
    if snap is None:
        out["status"] = "NOT CACHED"
        return out
    out["snapshot_path"] = str(snap)
    # real files (resolve symlinks/blobs)
    total = 0
    kinds = {}
    for f in snap.rglob("*"):
        if f.is_file():
            try:
                sz = f.stat().st_size
            except OSError:
                sz = 0
            total += sz
            ext = "".join(f.suffixes) or f.name
            kinds[ext] = kinds.get(ext, 0) + sz
    out["total_bytes"] = total
    out["total_gb"] = round(total / 1e9, 3)
    out["files_by_ext_gb"] = {k: round(v / 1e9, 3) for k, v in
                              sorted(kinds.items(), key=lambda kv: -kv[1])}

    cfg_path = snap / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        out["architectures"] = cfg.get("architectures")
        out["model_type"] = cfg.get("model_type")
        out["torch_dtype"] = cfg.get("torch_dtype")
        out["hidden_size"] = cfg.get("hidden_size")
        out["num_layers"] = cfg.get("num_hidden_layers")
        out["vocab_size"] = cfg.get("vocab_size")
        out["tie_word_embeddings"] = cfg.get("tie_word_embeddings")

    # parameter count from safetensors index if present
    idx = snap / "model.safetensors.index.json"
    if idx.exists():
        d = json.loads(idx.read_text(encoding="utf-8"))
        out["safetensors_param_count"] = d.get("metadata", {}).get("total_size")
        if out["safetensors_param_count"]:
            out["safetensors_params_B"] = round(
                int(d["metadata"]["total_size"]) / 1e9, 3)
        out["safetensors_shards"] = len(set(d.get("weight_map", {}).values()))
    else:
        sts = sorted(snap.glob("*.safetensors"))
        if sts:
            out["safetensors_shards"] = len(sts)
            out["safetensors_files_gb"] = round(
                sum(s.stat().st_size for s in sts) / 1e9, 3)

    out["has_tokenizer_config"] = (snap / "tokenizer_config.json").exists()
    out["has_chat_template"] = (
        (snap / "tokenizer_config.json").exists()
        and "chat_template" in (snap / "tokenizer_config.json").read_text(
            encoding="utf-8", errors="ignore")
    )
    out["has_generation_config"] = (snap / "generation_config.json").exists()
    if out["has_generation_config"]:
        gc = json.loads((snap / "generation_config.json").read_text(encoding="utf-8"))
        out["gen_config_keys"] = sorted(gc.keys())
    out["status"] = "CACHED"
    return out


print("=" * 78)
print("PART 1 - MODEL VERIFICATION")
print("=" * 78)
print(f"HF hub cache: {HUB}")
print()

# All Qwen/Phi dirs actually present, for completeness
print("--- ALL CACHED DIRS MATCHING Qwen/Phi ---")
for d in sorted(HUB.glob("models--*")):
    if any(k in d.name for k in ("Qwen", "Phi")):
        sz = 0
        for f in d.rglob("*"):
            if f.is_file():
                try:
                    sz += f.stat().st_size
                except OSError:
                    pass
        print(f"  {d.name:<55} {sz/1e9:8.3f} GB")
print()

results = {}
for label, mid in TARGETS.items():
    r = inspect(label, mid)
    results[label] = r
    print(f"--- {label.upper()} : {mid} ---")
    for k, v in r.items():
        if k in ("role", "model_id"):
            continue
        print(f"    {k:<28} {v}")
    print()

missing = [k for k, v in results.items() if v.get("status") != "CACHED"]
print("=" * 78)
print("MISSING / NOT CACHED:", missing if missing else "none")
print("=" * 78)
