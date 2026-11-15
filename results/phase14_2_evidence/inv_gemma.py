"""Phase 14.4 Step 2 — inventory every locally available Gemma-family model.

Read-only. Decides whether the required `Gemma 3 4B IT` is present locally.
No download, no substitution.
"""
import json
import os
from pathlib import Path

HUBS = [
    Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub",
    Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound\models"),
]

print("=" * 96)
print("PHASE 14.4 — LOCAL GEMMA-FAMILY MODEL INVENTORY")
print("=" * 96)

required = ["gemma-3-4b-it", "gemma-3-4B-it"]
found_required = []

for hub in HUBS:
    if not hub.exists():
        print(f"[hub absent] {hub}")
        continue
    print(f"\n### HUB: {hub}")
    dirs = sorted([d for d in hub.iterdir()
                   if d.is_dir() and "gemma" in d.name.lower()])
    if not dirs:
        print("  (no gemma-family directories)")
    for d in dirs:
        print(f"\n  --- {d.name} ---")
        snap_root = d / "snapshots"
        revs = []
        if snap_root.is_dir():
            revs = sorted([p for p in snap_root.iterdir() if p.is_dir()])
        if not revs:
            # maybe it is a plain model directory (not cache layout)
            if (d / "config.json").exists():
                revs = [d]
            else:
                print("    NO SNAPSHOT (empty / locks only)")
                total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                print(f"    bytes on disk: {total}")
                continue
        for rev in revs:
            print(f"    revision: {rev.name}")
            try:
                total = sum(f.stat().st_size for f in rev.rglob("*") if f.is_file())
            except OSError:
                total = 0
            files = sorted(f.name for f in rev.iterdir() if f.is_file())
            weights = [f for f in files if f.endswith(".safetensors")]
            print(f"      size: {total/1e9:.3f} GB   safetensors shards: {len(weights)}")
            cfg_path = rev / "config.json"
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                except Exception as e:  # noqa: BLE001
                    cfg = {}
                    print(f"      config.json unreadable: {e}")
                for k in ("model_type", "architectures", "torch_dtype",
                          "num_hidden_layers", "hidden_size", "vocab_size",
                          "quantization_config"):
                    if k in cfg:
                        v = cfg[k]
                        if k == "quantization_config":
                            v = {kk: v[kk] for kk in list(v)[:4]}
                        print(f"      {k:<20} {v}")
            # tokenizer / template
            tc = rev / "tokenizer_config.json"
            tc_present = tc.exists()
            chat_template = False
            if tc_present:
                try:
                    t = json.loads(tc.read_text(encoding="utf-8"))
                    chat_template = "chat_template" in t
                except Exception:  # noqa: BLE001
                    pass
            print(f"      tokenizer_config.json: {tc_present}  "
                  f"chat_template: {chat_template}")
            gen = rev / "generation_config.json"
            print(f"      generation_config.json: {gen.exists()}")
            if any(r in d.name.lower() for r in required):
                found_required.append(str(rev))

print("\n" + "=" * 96)
print("REQUIRED MODEL: Gemma 3 4B IT  (e.g. google/gemma-3-4b-it)")
print("=" * 96)
cache_hit = [h for h in HUBS
             if (h / "models--google--gemma-3-4b-it").exists()]
print(f"models--google--gemma-3-4b-it present : {bool(cache_hit)}")
print(f"any '*gemma-3-4b-it*' snapshot matched: {found_required or 'NONE'}")
print()
print("VERDICT:", "PRESENT" if (cache_hit or found_required)
      else "ABSENT — required evaluator model is NOT available locally")
