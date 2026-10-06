"""Phase 17 Stage 0 — attacker model feasibility against the real machine budget.

Read-only reconnaissance. Queries the HuggingFace API for authoritative repo
metadata (no weight download) and compares each candidate against the measured
budgets of this machine.

Corrections learned while building this, both worth recording because the naive
version of each produced wrong numbers:

  * ``usedStorage`` is NOT the download size. It sums every revision and every
    alternative format copy in the repo, so Meta-Llama-3-8B-Instruct reports
    57.9 GiB when the weights actually needed are ~15 GiB. The download figure
    used here is the sum of the file sizes in the ``main`` revision only.
  * a missing or inaccessible repo answers **401**, not 404, on the HF API.
    Accessibility therefore has to be probed explicitly, or an unresolvable
    model silently reads as "no metadata".

Transport is ``curl``: .NET's Invoke-RestMethod returns 401 for every repo on
this machine (including public ones), while curl returns the correct status.

Nothing here loads a model, downloads a weight, or writes outside
``results/phase17_model_qualification/``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification"
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"

HF_API = "https://huggingface.co/api/models/"

MIB = 1024**2
GIB = 1024**3
GPU_TOTAL_MIB = 24570
GPU_TOTAL_GIB = GPU_TOTAL_MIB * MIB / GIB

# Reserved for KV cache, activations, CUDA context and fragmentation. A stated
# assumption, not a measurement — Section 11 replaces it with a measured peak.
RUNTIME_ALLOWANCE_GIB = 3.0

# Roles in the Phase 17 contract. Attacker ids are the resolved-exact ids
# approved for this phase; the spec's shorthand does not resolve on HF.
CANDIDATES = {
    "target": "meta-llama/Meta-Llama-3-8B-Instruct",
    "attacker_A_bonsai": "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
    "attacker_B_qwen38": "Qwen/Qwen3.8-27B",
    "attacker_C_qwen36": "Qwen/Qwen3.6-27B",
    "attacker_D_ministral": "mistralai/Ministral-3-14B-Reasoning-2512",
    "attacker_E_devstral": "mistralai/Devstral-Small-2-24B-Instruct-2512",
}

# Quantized checkpoints probed only for candidates that fail at reference
# precision. Section 9 permits quantization when hardware requires it, and
# Section 10 prefers an official checkpoint over an ad-hoc conversion, so the
# official first-party ids are probed before any community conversion.
QUANTIZED_PROBES = {
    "attacker_B_qwen38": [
        "Qwen/Qwen3.8-27B-FP8",
        "Qwen/Qwen3.8-27B-AWQ",
        "Qwen/Qwen3.8-27B-GPTQ-Int4",
    ],
    "attacker_C_qwen36": [
        "Qwen/Qwen3.6-27B-FP8",
        "Qwen/Qwen3.6-27B-AWQ",
        "Qwen/Qwen3.6-27B-GPTQ-Int4",
    ],
    "attacker_D_ministral": [
        "mistralai/Ministral-3-14B-Reasoning-2512-FP8",
        "mistralai/Ministral-3-14B-Reasoning-2512-GGUF",
    ],
    "attacker_E_devstral": [
        "mistralai/Devstral-Small-2-24B-Instruct-2512-GGUF",
    ],
}

# Bytes per element, by the dtype names safetensors reports. Used so an FP8 or
# 4-bit checkpoint is costed at its real width instead of being charged bf16.
BYTES_PER_DTYPE = {
    "F64": 8.0, "F32": 4.0, "U32": 4.0, "I32": 4.0,
    "BF16": 2.0, "F16": 2.0,
    "F8_E4M3": 1.0, "F8_E5M2": 1.0, "I8": 1.0, "U8": 1.0,
    "I4": 0.5, "U4": 0.5, "I2": 0.25, "U2": 0.25, "I1": 0.125,
}


def curl_json(url: str) -> tuple[int, dict | None, str]:
    """GET a URL with curl, returning (status_code, parsed_json_or_None, raw)."""
    proc = subprocess.run(
        [
            "curl.exe", "-sS", "--max-time", "60",
            "-w", "\n__HTTP__%{http_code}",
            "-H", "User-Agent: curl/8",
            url,
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = proc.stdout
    marker = out.rfind("__HTTP__")
    if marker == -1:
        return 0, None, (proc.stderr or out)[:300]
    body = out[:marker].strip()
    try:
        status = int(out[marker + len("__HTTP__"):].strip())
    except ValueError:
        return 0, None, body[:300]
    if not body:
        return status, None, ""
    try:
        return status, json.loads(body), body
    except json.JSONDecodeError:
        return status, None, body[:300]


def local_cache_dir(repo_id: str) -> Path:
    return HF_HUB / ("models--" + repo_id.replace("/", "--"))


def cache_bytes(repo_id: str) -> int:
    """Bytes actually on disk for this repo — snapshots only, not stale blobs."""
    snap = local_cache_dir(repo_id) / "snapshots"
    if not snap.is_dir():
        return 0
    total = 0
    seen: set[str] = set()
    for path in snap.rglob("*"):
        if not path.is_file():
            continue
        try:
            real = str(path.resolve())
        except OSError:
            continue
        if real in seen:
            continue  # HF points several snapshot files at one blob
        seen.add(real)
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def probe(repo_id: str) -> dict:
    status, meta, raw = curl_json(f"{HF_API}{repo_id}")

    if status == 401:
        return {
            "repo_id": repo_id, "accessible": False, "http_status": status,
            "error": "401 — repo does not exist, or is gated and not accessible with this token",
        }
    if status != 200 or meta is None:
        return {
            "repo_id": repo_id, "accessible": False, "http_status": status,
            "error": f"HTTP {status}: {raw[:200]}",
        }

    config = meta.get("config") or {}
    safetensors = meta.get("safetensors") or {}
    param_map = safetensors.get("parameters") or {}
    total_params = safetensors.get("total") or (sum(param_map.values()) or None)
    shipped = sorted(param_map.keys())
    tinfo = meta.get("transformersInfo") or {}

    cached = cache_bytes(repo_id)

    return {
        "repo_id": repo_id,
        "accessible": True,
        "http_status": status,
        "gated": meta.get("gated", False),
        "revision": meta.get("sha"),
        "pipeline_tag": meta.get("pipeline_tag"),
        "library_name": meta.get("library_name"),
        "architectures": config.get("architectures") or [],
        "model_type": config.get("model_type"),
        "auto_model": tinfo.get("auto_model"),
        "processor": tinfo.get("processor"),
        "total_params": total_params,
        "total_params_billions": round(total_params / 1e9, 2) if total_params else None,
        "param_map": param_map,
        "shipped_dtypes": shipped,
        "is_reference_precision": bool(shipped) and set(shipped) <= {"BF16", "F16"},
        "repo_used_storage_gib": round(meta["usedStorage"] / GIB, 2) if meta.get("usedStorage") else None,
        "local_cache_gib": round(cached / GIB, 2) if cached else 0.0,
    }


def weights_gib(entry: dict) -> float | None:
    """Weight footprint, costed per dtype rather than guessed from one flag.

    Summing each dtype at its real width matters here: an FP8 checkpoint that
    keeps a bf16 copy of the embeddings must not be charged full bf16 for every
    parameter, and a 1-bit/ternary pack must not be charged 2 bytes per weight.

    Returns None for repos that expose no safetensors parameter breakdown
    (MLX / GGUF packs) instead of inventing a figure.
    """
    param_map = entry.get("param_map") or {}
    if not param_map:
        return None
    unknown = [dt for dt in param_map if dt not in BYTES_PER_DTYPE]
    if unknown:
        return None
    total_bytes = sum(
        count * BYTES_PER_DTYPE[dt] for dt, count in param_map.items()
    )
    return total_bytes / GIB


def classify(entry: dict, disk_free_gib: float) -> dict:
    """Decide feasibility for THIS checkpoint, and name what blocks it."""
    if not entry.get("accessible"):
        return {"verdict": "BLOCKED", "blocking_reasons": [entry.get("error", "inaccessible")]}

    w = weights_gib(entry)
    cached = entry.get("local_cache_gib") or 0.0
    is_ref = entry.get("is_reference_precision", False)

    # A checkpoint already resident in the local cache needs no further download.
    already_local = bool(w) and cached > 0.7 * w
    download_gib = 0.0 if already_local else (w or 0.0)

    reasons: list[str] = []
    if w is None:
        reasons.append("no safetensors dtype breakdown — not a transformers-shaped checkpoint")

    fits_vram = None if w is None else (w + RUNTIME_ALLOWANCE_GIB) <= GPU_TOTAL_GIB
    fits_disk = download_gib <= disk_free_gib

    if fits_vram is False:
        reasons.append(
            f"vram: {w:.2f} GiB weights + {RUNTIME_ALLOWANCE_GIB} GiB allowance > {GPU_TOTAL_GIB:.2f} GiB"
        )
    if not fits_disk:
        reasons.append(f"disk: needs {download_gib:.2f} GiB download > {disk_free_gib:.2f} GiB free")

    if not reasons:
        verdict = "FEASIBLE_AT_REFERENCE_PRECISION" if is_ref else "FEASIBLE_QUANTIZED"
    elif not is_ref:
        verdict = "INFEASIBLE_QUANTIZED"
    else:
        verdict = "INFEASIBLE_AT_REFERENCE_PRECISION"

    return {
        "weight_footprint_gib": round(w, 2) if w else None,
        "already_cached": already_local,
        "additional_download_gib": round(download_gib, 2),
        "fits_vram": fits_vram,
        "fits_disk": fits_disk,
        "verdict": verdict,
        "blocking_reasons": reasons,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    disk_free = shutil.disk_usage(str(REPO_ROOT)).free
    disk_free_gib = disk_free / GIB

    print(f"GPU {GPU_TOTAL_MIB} MiB = {GPU_TOTAL_GIB:.2f} GiB | allowance {RUNTIME_ALLOWANCE_GIB} GiB | disk free {disk_free_gib:.2f} GiB")
    print()

    results: dict[str, dict] = {}
    for role, repo_id in CANDIDATES.items():
        entry = probe(repo_id)
        entry.update(classify(entry, disk_free_gib))
        results[role] = entry

        if not entry.get("accessible"):
            print(f"{role:22s} {repo_id:52s} BLOCKED — {entry['error'][:70]}")
            continue

        w = entry.get("weight_footprint_gib")
        print(
            f"{role:22s} {repo_id:50s} "
            f"{str(entry.get('total_params_billions')):>6s}B "
            f"{str(entry.get('shipped_dtypes')):>22s} "
            f"w={str(w):>7s}GiB cache={entry.get('local_cache_gib'):>6.2f}GiB "
            f"{entry['verdict']}"
        )
        for reason in entry.get("blocking_reasons") or []:
            print(f"{'':24s}  -> {reason}")

    print("\n--- quantized checkpoint probes (Section 9/10) ---")
    quant: dict[str, list[dict]] = {}
    for role, repo_ids in QUANTIZED_PROBES.items():
        quant[role] = []
        for repo_id in repo_ids:
            e = probe(repo_id)
            e.update(classify(e, disk_free_gib))
            quant[role].append(e)
            if e.get("accessible"):
                w = e.get("weight_footprint_gib")
                print(
                    f"{role:22s} {repo_id:50s} "
                    f"w={str(w):>7s}GiB {str(e.get('shipped_dtypes')):>24s} {e['verdict']}"
                )
            else:
                print(f"{role:22s} {repo_id:50s} {e.get('error', '')[:60]}")

    payload = {
        "phase": "17",
        "stage": "0_feasibility",
        "budgets": {
            "gpu_total_mib": GPU_TOTAL_MIB,
            "gpu_total_gib": round(GPU_TOTAL_GIB, 3),
            "runtime_allowance_gib": RUNTIME_ALLOWANCE_GIB,
            "disk_free_bytes": disk_free,
            "disk_free_gib": round(disk_free_gib, 3),
        },
        "method": (
            "HF API metadata via curl; no weights downloaded. Download size is the "
            "main-revision weight footprint, NOT usedStorage (which sums all revisions "
            "and format copies). Feasibility compares against measured GPU total and "
            "measured free disk."
        ),
        "candidates": results,
        "quantized_probes": quant,
    }
    out = OUT_DIR / "feasibility_results.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
