"""Phase 17 Stage 2 — Part F: attacker candidate identity verification.

Metadata only. This script downloads **no weights**: it resolves each candidate
against the authoritative HuggingFace API, records the exact revision, and
writes one ``candidate_<name>_feasibility.json`` per candidate. Later stages
merge their measured blocks (loading, VRAM, JSON contract, multi-turn) into the
same files, so every claim about a candidate sits next to its provisioning
evidence.

Why the values are read from the API rather than typed in:

  * ``usedStorage`` is not a download size — it sums every revision and format
    copy (Stage 0 recorded Llama-3-8B at 57.88 GiB that way). The download size
    here is the sum of the files in the resolved revision only.
  * a missing or inaccessible repo answers **401**, not 404, on the HF API, so
    accessibility is probed explicitly rather than inferred from "no metadata".
  * transport is ``curl``: .NET's ``Invoke-RestMethod`` returns 401 for every
    repo on this machine, including public ones (Stage 0 finding).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage2_attacker"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

MIB = 1024 ** 2
GIB = 1024 ** 3

# File keys that make up the official base weights for the approved NF4 path.
WEIGHT_SUFFIXES = (".safetensors",)
CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "model.safetensors.index.json",
)

CANDIDATES = {
    "qwen38": "Qwen/Qwen3.8-27B",
    "qwen36": "Qwen/Qwen3.6-27B",
    "ministral": "mistralai/Ministral-3-14B-Reasoning-2512",
    "devstral": "mistralai/Devstral-Small-2-24B-Instruct-2512",
}

REFERENCE_PRECISION = {
    "qwen38": "BF16",
    "qwen36": "BF16",
    "ministral": "BF16",
    "devstral": "FP8 (native checkpoint)",
}


def curl_json(url: str) -> tuple[int, dict | None, str]:
    """GET a URL with curl, returning (status_code, parsed_json_or_None, raw)."""
    proc = subprocess.run(
        [
            "curl.exe", "-sS", "-L", "--max-time", "120",
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


def _files_from_siblings(siblings: list[dict]) -> dict[str, int]:
    """Main-revision file sizes by path (``siblings`` carry no revision split)."""
    out: dict[str, int] = {}
    for sib in siblings or []:
        name = sib.get("rfilename")
        size = sib.get("size")
        if isinstance(name, str) and isinstance(size, int):
            out[name] = size
    return out


def _weight_bytes(files: dict[str, int]) -> tuple[int, list[str]]:
    weights = sorted(
        path for path in files if path.endswith(WEIGHT_SUFFIXES)
    )
    return sum(files[p] for p in weights), weights


def resolve_candidate(key: str, model_id: str) -> dict:
    record: dict = {
        "candidate": key,
        "requested_model_id": model_id,
        "record_type": "identity_and_provisioning",
        "weights_downloaded_by_this_script": False,
    }

    status, meta, raw = curl_json(f"https://huggingface.co/api/models/{model_id}?blobs=true")
    record["api_status"] = status
    record["exists"] = status == 200 and meta is not None
    if not record["exists"]:
        record["error"] = raw[:300]
        record["verdict_if_unresolvable"] = "BLOCKED (model id does not resolve)"
        return record

    revision = meta.get("sha")
    record.update(
        model_id=meta.get("modelId") or meta.get("id") or model_id,
        revision=revision,
        author=meta.get("author"),
        last_modified=meta.get("lastModified"),
        gated=meta.get("gated"),
        private=bool(meta.get("private")),
        pipeline_tag=meta.get("pipeline_tag"),
        library_name=meta.get("library_name"),
        tags=meta.get("tags", []),
    )

    # Exact revision resolution: the commit the download would be pinned to.
    if revision:
        rev_status, rev_meta, rev_raw = curl_json(
            f"https://huggingface.co/api/models/{model_id}/revision/{revision}"
        )
        record["revision_resolution"] = {
            "status": rev_status,
            "resolved": rev_status == 200 and rev_meta is not None,
            "sha": (rev_meta or {}).get("sha") if rev_meta else None,
            "error": None if rev_meta else rev_raw[:200],
        }

    files = _files_from_siblings(meta.get("siblings", []))
    weight_bytes, weight_files = _weight_bytes(files)
    config_present = {name: name in files for name in CONFIG_FILES}

    record["files"] = {
        "file_count": len(files),
        "weight_file_count": len(weight_files),
        "weight_files": weight_files[:12],
        "weight_bytes": weight_bytes,
        "weight_gib": round(weight_bytes / GIB, 3),
        "config_file_present": config_present,
        "total_revision_bytes": sum(files.values()),
        "total_revision_gib": round(sum(files.values()) / GIB, 3),
    }

    safetensors = meta.get("safetensors") or {}
    params = safetensors.get("parameters") or {}
    record["safetensors"] = {
        "available": bool(safetensors),
        "parameters_by_dtype": params,
        "parameter_count": safetensors.get("total"),
        "parameter_class": (
            f"{round(safetensors['total'] / 1e9, 2)}B" if safetensors.get("total") else None
        ),
    }

    config = meta.get("config") or {}
    record["architecture"] = {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "vocab_size": config.get("vocab_size"),
        "torch_dtype": config.get("torch_dtype"),
        "quantization_config": config.get("quantization_config"),
    }

    # Raw config.json at the resolved revision (authoritative for context length
    # and for whether the checkpoint is already quantized).
    cfg_status, cfg, cfg_raw = curl_json(
        f"https://huggingface.co/{model_id}/raw/{revision}/config.json"
    )
    if cfg:
        record["config_json"] = {
            "status": cfg_status,
            "model_type": cfg.get("model_type"),
            "architectures": cfg.get("architectures"),
            "max_position_embeddings": cfg.get("max_position_embeddings"),
            "num_hidden_layers": cfg.get("num_hidden_layers"),
            "hidden_size": cfg.get("hidden_size"),
            "quantization_config": cfg.get("quantization_config"),
            "torch_dtype": cfg.get("torch_dtype"),
        }
    else:
        record["config_json"] = {"status": cfg_status, "error": cfg_raw[:200]}

    tk_status, tk, tk_raw = curl_json(
        f"https://huggingface.co/{model_id}/raw/{revision}/tokenizer_config.json"
    )
    if tk:
        record["tokenizer_config"] = {
            "status": tk_status,
            "has_chat_template": bool(tk.get("chat_template")),
            "chat_template_kwarg_keys": sorted(
                {
                    token.strip()
                    for token in str(tk.get("chat_template", ""))
                    .replace("{{", " ")
                    .replace("}}", " ")
                    .replace("(", " ")
                    .replace(")", " ")
                    .split()
                    if token.strip()
                }
                if "enable_thinking" in str(tk.get("chat_template", ""))
                else []
            ),
            "enable_thinking_in_template": "enable_thinking" in str(tk.get("chat_template", "")),
            "bos_token": tk.get("bos_token"),
            "eos_token": tk.get("eos_token"),
            "pad_token": tk.get("pad_token"),
            "model_max_length": tk.get("model_max_length"),
        }
    else:
        record["tokenizer_config"] = {"status": tk_status, "error": tk_raw[:200]}

    record["instruction_tuned_evidence"] = {
        "tags_indicating_instruct_or_chat": [
            tag for tag in record["tags"]
            if any(k in tag.lower() for k in ("instruct", "chat", "it", "reasoning"))
        ],
        "card_data": meta.get("cardData", {}),
        "note": (
            "Recorded from the repo's own tags/cardData. No flag is asserted here "
            "that the repository does not itself declare."
        ),
    }
    record["reference_precision"] = REFERENCE_PRECISION[key]

    local_dir = HF_CACHE / ("models--" + model_id.replace("/", "--"))
    record["local"] = {
        "cache_dir": str(local_dir),
        "cache_dir_present": local_dir.is_dir(),
        "present_locally": False,
    }
    if local_dir.is_dir():
        blob_bytes = sum(
            p.stat().st_size for p in (local_dir / "blobs").glob("*") if p.is_file()
        )
        record["local"].update(
            present_locally=True,
            blob_bytes=blob_bytes,
            blob_gib=round(blob_bytes / GIB, 3),
        )

    disk = shutil.disk_usage(REPO_ROOT)
    record["disk_at_resolution"] = {
        "free_bytes": disk.free,
        "free_gib": round(disk.free / GIB, 2),
        "estimated_download_gib": record["files"]["weight_gib"],
        "note": (
            "The HF cache on this machine cannot use symlinks, so a cached model "
            "occupies roughly twice the weight size (blob copy + snapshot copy)."
        ),
    }
    return record


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for key, model_id in CANDIDATES.items():
        record = resolve_candidate(key, model_id)
        path = OUT_DIR / f"candidate_{key}_feasibility.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")

        arch = record.get("architecture") or {}
        weights = (record.get("files") or {}).get("weight_gib")
        params = (record.get("safetensors") or {}).get("parameter_class")
        summary[key] = {
            "model_id": record.get("model_id"),
            "revision": record.get("revision"),
            "exists": record["exists"],
            "architecture": arch.get("architectures"),
            "parameters": params,
            "weight_gib": weights,
            "context_length": (record.get("config_json") or {}).get("max_position_embeddings"),
            "reference_precision": record["reference_precision"],
            "present_locally": record["local"]["present_locally"],
        }
        print(
            f"{key:10s} {model_id:48s} exists={record['exists']} "
            f"rev={str(record.get('revision'))[:12]} "
            f"arch={arch.get('architectures')} params={params} "
            f"weights={weights} GiB "
            f"ctx={(record.get('config_json') or {}).get('max_position_embeddings')}"
        )
        print(f"           wrote {path.relative_to(REPO_ROOT)}")

    manifest_path = OUT_DIR / "candidate_identity_summary.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
