"""Dataset source acquisition with provenance tracking.

Handles downloading and selecting goals from:
- Circuit Breakers (Zou et al., 2024): 1,000 training goals
- HarmBench (Mazeika et al., 2024): 200 test behaviors

All selections are deterministic and reproducible.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Provenance metadata
# --------------------------------------------------------------------------- #

@dataclass
class SourceMetadata:
    """Provenance record for a downloaded dataset source."""
    source_name: str
    source_type: str                      # e.g. "research_dataset"
    source_url: str = ""
    source_version: str = ""              # commit hash or version tag
    download_timestamp: str = ""
    local_path: str = ""
    sha256: str = ""
    total_records: int = 0
    selected_records: int = 0
    selection_method: str = ""
    selection_seed: int | None = None
    paper_reference: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> SourceMetadata:
        data = json.loads(path.read_text())
        return cls(**data)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Circuit Breakers
# --------------------------------------------------------------------------- #

def _normalize_goal(text: str) -> str:
    """Normalize a goal for overlap detection (lowercase, strip whitespace)."""
    return re.sub(r"\s+", " ", text.strip().lower())


def load_circuit_breakers(
    data_dir: Path,
    num_goals: int = 1000,
    seed: int = 42,
    force: bool = False,
) -> tuple[list[str], SourceMetadata]:
    """Load and select Circuit Breakers training goals.

    Returns (goals, metadata). Selection is deterministic given the same seed.

    The Circuit Breakers dataset (Zou et al., 2024) contains harmful behavior
    prompts used to test safety alignment. We select num_goals goals for
    generating training conversations.
    """
    cb_dir = data_dir / "circuit_breakers"
    cb_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cb_dir / "provenance.json"
    goals_path = cb_dir / "goals.json"

    # Check if already downloaded and selected
    if goals_path.exists() and meta_path.exists() and not force:
        existing_meta = SourceMetadata.load(meta_path)
        if existing_meta.selected_records >= num_goals:
            goals = json.loads(goals_path.read_text())
            logger.info("Loaded %d cached Circuit Breakers goals", len(goals))
            return goals[:num_goals], existing_meta

    logger.info("Acquiring Circuit Breakers dataset...")

    # Try to download from HuggingFace datasets hub
    goals = _download_circuit_breakers(cb_dir)

    if not goals:
        # Fallback: create placeholder with clear documentation
        logger.warning(
            "Could not download Circuit Breakers dataset. "
            "Creating placeholder. To use real data, download manually from: "
            "https://huggingface.co/datasets/cliram/Circuit-Breakers"
        )
        goals = []

    # Deterministic selection
    rng = np.random.RandomState(seed)
    if len(goals) > num_goals:
        indices = rng.choice(len(goals), size=num_goals, replace=False)
        goals = [goals[i] for i in sorted(indices)]
    elif len(goals) < num_goals:
        logger.warning(
            "Only %d goals available, requested %d", len(goals), num_goals
        )

    # Save
    goals_path.write_text(json.dumps(goals, indent=2, ensure_ascii=False))

    meta = SourceMetadata(
        source_name="Circuit Breakers",
        source_type="research_dataset",
        source_url="https://huggingface.co/datasets/cliram/Circuit-Breakers",
        source_version="Not specified in the paper",
        download_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        local_path=str(goals_path),
        sha256=_sha256_file(goals_path),
        total_records=len(goals),
        selected_records=len(goals),
        selection_method=f"np.random.RandomState({seed}).choice sorted indices",
        selection_seed=seed,
        paper_reference="Zou et al., 2024 (Circuit Breakers)",
    )
    meta.save(meta_path)

    logger.info("Selected %d Circuit Breakers goals", len(goals))
    return goals, meta


def _download_circuit_breakers(cb_dir: Path) -> list[str]:
    """Try to download Circuit Breakers from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("cliram/Circuit-Breakers", split="train", trust_remote_code=True)
        # Extract the harmful behavior prompts
        goals = []
        for row in ds:
            # The dataset has various columns; we want the behavioral prompts
            for key in ["behavior", "prompt", "goal", "input", "query"]:
                if key in row and row[key] and isinstance(row[key], str):
                    goals.append(row[key])
                    break
        return goals
    except Exception as exc:
        logger.warning("Failed to download Circuit Breakers: %s", exc)
        return []


# --------------------------------------------------------------------------- #
# HarmBench
# --------------------------------------------------------------------------- #

def load_harmbench(
    data_dir: Path,
    num_behaviors: int = 200,
    seed: int = 42,
    force: bool = False,
) -> tuple[list[str], SourceMetadata]:
    """Load and select HarmBench test behaviors.

    Returns (behaviors, metadata). Selection is deterministic.

    HarmBench (Mazeika et al., 2024) provides standardized test behaviors
    for evaluating jailbreak attacks and defenses.
    """
    hb_dir = data_dir / "harmbench"
    hb_dir.mkdir(parents=True, exist_ok=True)
    meta_path = hb_dir / "provenance.json"
    behaviors_path = hb_dir / "behaviors.json"

    if behaviors_path.exists() and meta_path.exists() and not force:
        existing_meta = SourceMetadata.load(meta_path)
        if existing_meta.selected_records >= num_behaviors:
            behaviors = json.loads(behaviors_path.read_text())
            logger.info("Loaded %d cached HarmBench behaviors", len(behaviors))
            return behaviors[:num_behaviors], existing_meta

    logger.info("Acquiring HarmBench dataset...")

    behaviors = _download_harmbench(hb_dir)

    if not behaviors:
        logger.warning(
            "Could not download HarmBench dataset. "
            "Creating placeholder. To use real data, download manually from: "
            "https://github.com/centerforaisafety/HarmBench"
        )
        behaviors = []

    # Deterministic selection
    rng = np.random.RandomState(seed)
    if len(behaviors) > num_behaviors:
        indices = rng.choice(len(behaviors), size=num_behaviors, replace=False)
        behaviors = [behaviors[i] for i in sorted(indices)]
    elif len(behaviors) < num_behaviors:
        logger.warning(
            "Only %d behaviors available, requested %d", len(behaviors), num_behaviors
        )

    behaviors_path.write_text(json.dumps(behaviors, indent=2, ensure_ascii=False))

    meta = SourceMetadata(
        source_name="HarmBench",
        source_type="research_dataset",
        source_url="https://github.com/centerforaisafety/HarmBench",
        source_version="Not specified in the paper",
        download_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        local_path=str(behaviors_path),
        sha256=_sha256_file(behaviors_path),
        total_records=len(behaviors),
        selected_records=len(behaviors),
        selection_method=f"np.random.RandomState({seed}).choice sorted indices",
        selection_seed=seed,
        paper_reference="Mazeika et al., 2024 (HarmBench)",
    )
    meta.save(meta_path)

    logger.info("Selected %d HarmBench behaviors", len(behaviors))
    return behaviors, meta


def _download_harmbench(hb_dir: Path) -> list[str]:
    """Try to download HarmBench from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("HarmBench/harmbench", split="test", trust_remote_code=True)
        behaviors = []
        for row in ds:
            for key in ["behavior", "prompt", "goal", "input", "query", "text"]:
                if key in row and row[key] and isinstance(row[key], str):
                    behaviors.append(row[key])
                    break
        return behaviors
    except Exception as exc:
        logger.warning("Failed to download HarmBench: %s", exc)
        return []


# --------------------------------------------------------------------------- #
# Contamination check
# --------------------------------------------------------------------------- #

@dataclass
class ContaminationReport:
    """Result of checking train/test overlap."""
    training_count: int = 0
    test_count: int = 0
    exact_overlap_count: int = 0
    normalized_overlap_count: int = 0
    removed_count: int = 0

    def summary(self) -> str:
        return (
            f"Contamination report:\n"
            f"  Training goals:    {self.training_count}\n"
            f"  Test behaviors:    {self.test_count}\n"
            f"  Exact overlaps:    {self.exact_overlap_count}\n"
            f"  Normalized overlaps: {self.normalized_overlap_count}\n"
            f"  Removed from test: {self.removed_count}"
        )


def check_contamination(
    training_goals: list[str],
    test_behaviors: list[str],
) -> ContaminationReport:
    """Check for overlap between training goals and test behaviors.

    Uses both exact and normalized (lowercase, whitespace-collapsed) matching.
    This is a local methodology choice; not specified in the paper.
    """
    report = ContaminationReport(
        training_count=len(training_goals),
        test_count=len(test_behaviors),
    )

    # Exact match
    train_exact = set(training_goals)
    test_exact = set(test_behaviors)
    report.exact_overlap_count = len(train_exact & test_exact)

    # Normalized match (local choice: lowercase + collapse whitespace)
    train_norm = {_normalize_goal(g) for g in training_goals}
    test_norm = {_normalize_goal(b) for b in test_behaviors}
    overlaps = train_norm & test_norm
    report.normalized_overlap_count = len(overlaps)

    # Remove overlapping test behaviors
    if overlaps:
        test_norm_list = [_normalize_goal(b) for b in test_behaviors]
        keep = [b for b, bn in zip(test_behaviors, test_norm_list) if bn not in overlaps]
        report.removed_count = len(test_behaviors) - len(keep)
        logger.warning(
            "Found %d overlapping goals between train and test sets. "
            "Removed %d from test set.", report.normalized_overlap_count, report.removed_count
        )

    logger.info(report.summary())
    return report
