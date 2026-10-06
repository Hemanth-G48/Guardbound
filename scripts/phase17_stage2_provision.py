"""Phase 17 Stage 2 — Part E.1: sequential candidate provisioning (official weights).

Downloads ONE candidate at a time, pins the exact revision resolved in Part F,
verifies the local snapshot against the Hub's own file list, and records the
provisioning block in ``model_download_manifest.json``.

The approved quantisation path is NF4 post-training quantisation *at load time*
(bitsandbytes), so the official base weights are what is fetched here — never a
community AWQ/GPTQ/MLX/NVFP4 conversion.

Disk arithmetic is explicit and checked before the transfer starts, because the
HuggingFace cache on this machine cannot use symlinks: a cached model occupies
roughly twice its weight size (blob copy + snapshot copy). Insufficient space is
a STOP, not a reason to delete something.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Set inside the process so the script behaves the same under cmd.exe,
# PowerShell, or the background runner (this machine needs the MKL workaround).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Measured on this machine: the Xet transfer backend opened 37 connections and
# moved 0 bytes in ten minutes, while plain ranged HTTPS requests sustained
# ~7 MiB/s aggregate over 3 parallel streams (single stream: ~0.5 MiB/s).
# The classic HTTP path is therefore pinned explicitly.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Parallelism is required for throughput here, not just for latency: aggregate
# bandwidth scales with the number of streams (measured above).
DOWNLOAD_WORKERS = 8

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage2_attacker"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
MANIFEST_PATH = OUT_DIR / "model_download_manifest.json"

GIB = 1024 ** 3
DISK_MARGIN_GIB = 4.0

CANDIDATES = {
    "qwen38": "Qwen/Qwen3.8-27B",
    "qwen36": "Qwen/Qwen3.6-27B",
    "ministral": "mistralai/Ministral-3-14B-Reasoning-2512",
    "devstral": "mistralai/Devstral-Small-2-24B-Instruct-2512",
}


def curl_json(url: str) -> dict:
    proc = subprocess.run(
        ["curl.exe", "-sS", "-L", "--max-time", "120", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return json.loads(proc.stdout)


def load_manifest() -> dict:
    if MANIFEST_PATH.is_file():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "phase": "17",
        "stage": "2",
        "deliverable": "model_download_manifest.json",
        "policy": (
            "Official base weights only, resolved revision pinned, one candidate at "
            "a time; quantisation happens at load time via NF4 (bitsandbytes), never "
            "by substituting a community quantised checkpoint."
        ),
        "candidates": {},
    }


def snapshot_dir(model_id: str) -> Path:
    return HF_CACHE / ("models--" + model_id.replace("/", "--"))


def local_files(snap: Path) -> dict[str, int]:
    """Local snapshot files by repo-relative path, with sizes."""
    found: dict[str, int] = {}
    snapshots = snap / "snapshots"
    if not snapshots.is_dir():
        return found
    for revision_dir in snapshots.iterdir():
        if not revision_dir.is_dir():
            continue
        for path in revision_dir.rglob("*"):
            if path.is_file():
                found[path.relative_to(revision_dir).as_posix()] = path.stat().st_size
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve, measure and print; download nothing")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_id = CANDIDATES[args.candidate]

    identity_path = OUT_DIR / f"candidate_{args.candidate}_feasibility.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    revision = identity["revision"]

    meta = curl_json(f"https://huggingface.co/api/models/{model_id}?blobs=true")
    hub_files = {
        sib["rfilename"]: (sib.get("size") or 0)
        for sib in meta.get("siblings", [])
    }
    weight_bytes = sum(
        size for name, size in hub_files.items() if name.endswith(".safetensors")
    )

    disk_before = shutil.disk_usage(REPO_ROOT)
    # The pre-flight gate uses the *observed* 2x no-symlink overhead rather than
    # the raw weight size, so an over-budget download is refused up front.
    required_gib = (weight_bytes * 2) / GIB + DISK_MARGIN_GIB
    free_gib = disk_before.free / GIB

    print(f"candidate        : {args.candidate} ({model_id})")
    print(f"pinned revision  : {revision}")
    print(f"hub files        : {len(hub_files)} ({weight_bytes / GIB:.2f} GiB weights)")
    print(f"disk free        : {free_gib:.2f} GiB")
    print(f"required (2x+{DISK_MARGIN_GIB})  : {required_gib:.2f} GiB")

    record = {
        "model_id": model_id,
        "revision": revision,
        "architecture": (identity.get("architecture") or {}).get("architectures"),
        "parameters": (identity.get("safetensors") or {}).get("parameter_class"),
        "reference_precision": identity.get("reference_precision"),
        "download_status": "NOT_STARTED",
        "official_base_weights": True,
        "quantization_applied_at_runtime": "NF4",
        "runtime": "bitsandbytes",
        "disk_free_before_gib": round(free_gib, 2),
        "required_gib_estimate": round(required_gib, 2),
    }

    if free_gib < required_gib:
        record["download_status"] = "INSUFFICIENT_DISK"
        record["note"] = (
            "STOP by policy: not enough space for this candidate under the observed "
            "no-symlink cache overhead. Nothing was deleted."
        )
        _write_record(record)
        print("REFUSED: insufficient disk. Nothing was downloaded or deleted.")
        return 2

    if args.dry_run:
        record["download_status"] = "DRY_RUN_OK"
        _write_record(record)
        print("dry run: preconditions satisfied; no download performed")
        return 0

    from huggingface_hub import snapshot_download

    started = time.time()
    print("downloading official weights (resumable) …", flush=True)
    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=None,
        max_workers=DOWNLOAD_WORKERS,
    )
    elapsed = time.time() - started
    print(f"snapshot_download returned in {elapsed / 60:.1f} min: {path}", flush=True)

    snap = snapshot_dir(model_id)
    found = local_files(snap)
    missing = sorted(name for name in hub_files if name not in found)
    size_mismatch = sorted(
        name for name, size in hub_files.items()
        if name in found and size and found[name] != size
    )
    tokenizer_ok = all(
        name in found for name in (
            "tokenizer.json", "tokenizer_config.json", "config.json",
        )
    )
    weights_ok = not missing and not size_mismatch

    disk_after = shutil.disk_usage(REPO_ROOT)
    record.update(
        download_status="COMPLETE" if (weights_ok and tokenizer_ok) else "INCOMPLETE",
        download_size_gib=round(sum(found.values()) / GIB, 3),
        download_seconds=round(elapsed, 1),
        local_cache_path=str(snap),
        local_cache_verified=bool(found) and not missing,
        tokenizer_verified=tokenizer_ok,
        config_verified="config.json" in found,
        files_expected=len(hub_files),
        files_present=len(found),
        missing_files=missing[:10],
        size_mismatches=size_mismatch[:10],
        disk_free_after_gib=round(disk_after.free / GIB, 2),
    )
    _write_record(record)

    print(f"status           : {record['download_status']}")
    print(f"files            : {record['files_present']}/{record['files_expected']}")
    print(f"on-disk size     : {record['download_size_gib']} GiB")
    print(f"disk free after  : {record['disk_free_after_gib']} GiB")
    if missing:
        print(f"MISSING          : {missing[:5]}")
    return 0 if record["download_status"] == "COMPLETE" else 1


def _write_record(record: dict) -> None:
    manifest = load_manifest()
    manifest["candidates"][record["model_id"]] = record
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
