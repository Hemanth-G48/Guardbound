#!/usr/bin/env python3
"""Experiment manifest writer — reproducibility record for every smoke/experiment run.

Writes a manifest JSON alongside each experiment output file containing:
  timestamp
  git commit / repository revision (if available)
  configuration
  attacker model
  target model
  evaluator model
  embedding model
  NBF checkpoint + sha256
  dataset
  dataset split
  seed
  temperature
  max turns
  threshold (eta)
  device
  dtype
  software versions

Every experiment should write a manifest so the run can be reconstructed later.

Usage:
    from scripts.experiment_manifest import write_manifest
    write_manifest(config, output_path=Path("results/reproduction/smoke_A.jsonl"))
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import transformers
import yaml


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev(repo_root: Path) -> str | None:
    """Return git HEAD rev if available, else None."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(repo_root),
        )
        rev = r.stdout.strip()
        return rev if rev else None
    except Exception:
        return None


def write_manifest(
    config: dict,
    output_path: Path,
    mode: str | None = None,
    limit: int | None = None,
) -> Path:
    """Write a reproducibility manifest next to `output_path`.

    Parameters
    ----------
    config
        The full parsed reproduction.yaml.
    output_path
        The experiment JSONL output file. The manifest is written to
        ``output_path.with_suffix('.manifest.json')``.
    mode
        Experiment mode (A/B/C) — written into the manifest if provided.
    limit
        Goal limit used for this run — written into the manifest if provided.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    nbf = config.get("nbf", {})
    ckpt_path = nbf.get("checkpoint")
    ckpt_sha = None
    if ckpt_path and Path(ckpt_path).exists():
        ckpt_sha = sha256_of(Path(ckpt_path))

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_rev(out.parent),
        "config": {
            "path": str(out.parent / "configs/reproduction.yaml"),
            "content": config,
        },
        "attacker_model": config.get("attacker", {}).get("model"),
        "target_model": config.get("target", {}).get("model"),
        "evaluator_model": (
            config.get("evaluator", {}).get("rubric_model")
            or config.get("experiment_model", {}).get("evaluator")
        ),
        "embedding_model": config.get("embedding", {}).get("model"),
        "nbf_checkpoint": ckpt_path,
        "nbf_checkpoint_sha256": ckpt_sha,
        "dataset": config.get("dataset", {}).get("path"),
        "dataset_split": config.get("dataset", {}).get("split"),
        "seed": config.get("experiment", {}).get("seed"),
        "temperature": config.get("target", {}).get("temperature")
        or config.get("experiment_model", {}).get("temperature"),
        "max_turns": config.get("attacks", {}).get("max_turns"),
        "threshold": nbf.get("threshold") or nbf.get("eta"),
        "mode": mode,
        "limit": limit,
        "device": config.get("hardware", {}).get("device", "cuda"),
        "dtype": config.get("hardware", {}).get("dtype", "bfloat16"),
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
        },
    }

    manifest_path = out.with_suffix(".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path
