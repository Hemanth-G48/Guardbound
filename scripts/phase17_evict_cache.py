"""Phase 17 — HF cache eviction, recorded before it happens.

Removes only cache entries that no frozen configuration references and that
earlier phases already ruled out. Every removal is written to
``results/phase17_model_qualification/cache_eviction.json`` *before* the delete,
so the record survives even if the delete is interrupted.

Entries deliberately retained (referenced by the frozen stack or by the Phase 17
contract) are listed too, with the reason, so the decision is auditable without
re-deriving it from the config files.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification"
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
GIB = 1024**3

# (cache dir name, reason it is safe to remove)
EVICT = [
    ("models--unsloth--gemma-3-12b-it", "not referenced by any frozen config"),
    ("models--google--gemma-4-12B-it", "not referenced by any frozen config"),
    ("models--meta-llama--Llama-3.1-8B-Instruct", "not in the Phase 17 contract"),
    ("models--zai-org--GLM-4.6V-Flash", "Phase 15 qualification: NOT QUALIFIED"),
    ("models--ornith-ai--Ornith-1.5-9B", "Phase 16.3: pilot NOT AUTHORIZED"),
    ("models--google--gemma-4-12B-it-qat-w4a16-ct", "unused by any config"),
    ("models--Qwen--Qwen3.5-4B", "superseded attacker model"),
    ("models--google--gemma-3-4b-it", "unused by any config"),
    ("models--microsoft--phi-4", "Phase 16.6: MODEL_NOT_FEASIBLE_AT_REQUIRED_PRECISION"),
]

# Retained on purpose. Deleting any of these would break a frozen path or the
# Phase 17 contract itself.
RETAIN = [
    ("models--meta-llama--Meta-Llama-3-8B-Instruct", "Phase 17 fixed TARGET"),
    ("models--meta-llama--Llama-3.2-3B-Instruct", "frozen Track A judge — keeps the reference path runnable"),
    ("models--Qwen--Qwen3-4B-Instruct-2507", "frozen Track A incumbent attacker"),
    ("models--microsoft--Phi-4-mini-instruct", "frozen Track A target"),
    ("models--sentence-transformers--all-mpnet-base-v2", "NBF embedding model"),
    ("models--hanjianghu--NBF-LLM", "author reference artifacts"),
]


def dir_size_gib(path: Path) -> float:
    if not path.is_dir():
        return 0.0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / GIB


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(str(REPO_ROOT)).free

    evicted, retained = [], []
    for name, reason in EVICT:
        path = HF_HUB / name
        size = dir_size_gib(path)
        evicted.append({
            "cache_dir": str(path),
            "name": name,
            "size_gib": round(size, 2),
            "reason": reason,
            "existed": path.is_dir(),
        })
    for name, reason in RETAIN:
        path = HF_HUB / name
        retained.append({
            "cache_dir": str(path),
            "name": name,
            "size_gib": round(dir_size_gib(path), 2),
            "reason": reason,
            "existed": path.is_dir(),
        })

    # Write the record BEFORE deleting anything.
    record = {
        "phase": "17",
        "deliverable": "cache_eviction.json",
        "approval": "user approved the full eviction list",
        "free_before_gib": round(free_before / GIB, 2),
        "evicted": evicted,
        "retained": retained,
        "note": (
            "Authoritative reference: HuggingFace Hub. These are public, "
            "re-downloadable weights that no frozen configuration references."
        ),
    }
    out = OUT_DIR / "cache_eviction.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"record written: {out.relative_to(REPO_ROOT)}")

    print(f"\nfree before: {free_before / GIB:.2f} GiB")
    total_freed = 0.0
    for entry in evicted:
        path = Path(entry["cache_dir"])
        if not path.is_dir():
            print(f"  SKIP (absent)  {entry['name']}")
            continue
        shutil.rmtree(path, ignore_errors=False)
        total_freed += entry["size_gib"]
        print(f"  removed {entry['size_gib']:>7.2f} GiB  {entry['name']}")

    free_after = shutil.disk_usage(str(REPO_ROOT)).free
    print(f"\nfree after : {free_after / GIB:.2f} GiB  (freed {total_freed:.2f} GiB)")

    record["free_after_gib"] = round(free_after / GIB, 2)
    record["freed_gib"] = round(total_freed, 2)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
