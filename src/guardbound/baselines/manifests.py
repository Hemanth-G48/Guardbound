"""Run manifests for Phase 8 baseline training.

Every training script (SFT/DPO/KTO) writes a JSON manifest capturing:

    - timestamp
    - git commit (if available)
    - base model id + revision
    - dataset path + hash + version
    - paper-specified hyperparameters (lr, epochs)
    - LLaMA-Factory version (when available)
    - LLaMA parameters and any non-paper values, with the literal
      annotation "Not specified in the paper — LLaMA-Factory default."
    - command line + output directory
    - seed, precision, batch settings

Nothing in this module performs training.  It is a write-only
provenance record.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
    except Exception:
        return None


def _file_hash(path: str | Path, algo: str = "sha256") -> str:
    p = Path(path)
    if not p.exists():
        return f"missing:{p}"
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"{algo}:{h.hexdigest()}"


def build_manifest(
    *,
    variant: str,                # "lora_sft" | "lora_dpo" | "lora_kto"
    base_model: str,
    base_model_revision: str | None = None,
    dataset_path: str | Path,
    dataset_version: str | None = None,
    learning_rate: float | None = None,
    num_train_epochs: int | None = None,
    lora_params: dict[str, Any] | None = None,
    batch: dict[str, Any] | None = None,
    scheduler: str | None = None,
    warmup: float | None = None,
    weight_decay: float | None = None,
    precision: str | None = None,
    quantization: str | None = None,
    seed: int | None = None,
    max_length: int | None = None,
    framework_version: str | None = None,
    llama_factory_version: str | None = None,
    replacement_response_provenance: str | None = None,
    chosen_provenance: str | None = None,
    rejected_provenance: str | None = None,
    command_line: list[str] | None = None,
    output_dir: str | Path | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a manifest dict (also JSON-serializable)."""
    paper_specified: dict[str, Any] = {}
    not_specified: dict[str, Any] = {}
    if learning_rate is not None:
        paper_specified["learning_rate"] = learning_rate
    if num_train_epochs is not None:
        paper_specified["num_train_epochs"] = num_train_epochs

    if lora_params is not None:
        for k, v in lora_params.items():
            not_specified[f"lora_{k}"] = v
    if batch is not None:
        not_specified["batch"] = batch
    if scheduler is not None:
        not_specified["scheduler"] = scheduler
    if warmup is not None:
        not_specified["warmup"] = warmup
    if weight_decay is not None:
        not_specified["weight_decay"] = weight_decay
    if precision is not None:
        not_specified["precision"] = precision
    if quantization is not None:
        not_specified["quantization"] = quantization
    if max_length is not None:
        not_specified["max_length"] = max_length

    manifest: dict[str, Any] = {
        "variant": variant,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "dataset": {
            "path": str(dataset_path),
            "hash": _file_hash(dataset_path),
            "version": dataset_version,
        },
        "paper_specified": paper_specified,
        "not_specified_in_paper": not_specified,
        "seed": seed,
        "framework_version": framework_version,
        "llama_factory_version": llama_factory_version,
        "command_line": command_line,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "provenance": {
            "replacement_responses": replacement_response_provenance,
            "chosen": chosen_provenance,
            "rejected": rejected_provenance,
        },
    }
    if extras:
        manifest["extras"] = extras

    # Annotate each non-paper key explicitly
    if not_specified:
        manifest["not_specified_in_paper_note"] = (
            "Keys under 'not_specified_in_paper' use LLaMA-Factory "
            "defaults or other local choices.  See the 'paper_specified' "
            "section for the values that are explicitly in the NBF paper "
            "(arXiv:2503.00187v3, B.1)."
        )
    return manifest


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    """Write the manifest as pretty-printed JSON.  Atomic write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return path
