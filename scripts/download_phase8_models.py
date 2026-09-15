"""Download the three Phase 8 models into the HF cache (background helper).

This is the FINAL model set (Gemma-4-12B-it-qat and Llama-3.1-8B removed):

  Qwen/Qwen3.5-4B                   -> attacker   (~8.8 GB bf16, 2 shards)
  microsoft/Phi-4-mini-instruct      -> target    (~7.7 GB bf16, 2 shards)
  Qwen/Qwen3-4B-Instruct-2507       -> evaluator (~7.6 GB bf16, 3 shards)

All three are unquantized bf16. No quantization, no compressed_tensors.
Safe to re-run: existing complete files are skipped.
"""
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen3.5-4B",
    "microsoft/Phi-4-mini-instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
]

ALLOW = ["*.json", "*.safetensors", "*.model", "tokenizer*", "*.py", "vocab.json", "merges.txt"]


def main() -> int:
    for rid in MODELS:
        t0 = time.time()
        print(f"[dl] downloading {rid} ...", flush=True)
        try:
            path = snapshot_download(
                rid,
                allow_patterns=ALLOW,
                max_workers=4,
            )
            dt = time.time() - t0
            print(f"[dl] DONE {rid} -> {path} ({dt/60:.1f} min)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[dl] FAILED {rid}: {type(exc).__name__}: {exc}", flush=True)
            return 1
    print("[dl] ALL COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
