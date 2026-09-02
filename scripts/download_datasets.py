#!/usr/bin/env python
"""Download official datasets for the NBF Safety Steering reproduction.

Datasets:
1. Circuit Breakers (Zou et al., 2024) — training goals
2. HarmBench (Mazeika et al., 2024) — test behaviors

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --force
    python scripts/download_datasets.py --verify
    python scripts/download_datasets.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Provenance metadata
# --------------------------------------------------------------------------- #

@dataclass
class DatasetInfo:
    """Complete provenance record for a downloaded dataset."""
    name: str
    source_url: str
    source_type: str              # "huggingface", "github", "direct_download"
    version: str                  # commit hash, tag, or version
    download_timestamp: str
    local_path: str
    raw_file_path: str            # path to the raw downloaded file
    sha256: str
    num_records: int
    record_field: str             # which field contains the text
    paper_reference: str
    license: str
    notes: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> DatasetInfo:
        data = json.loads(path.read_text())
        return cls(**data)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Circuit Breakers download
# --------------------------------------------------------------------------- #

# Circuit Breakers (Zou et al., 2024)
# The original CB paper generates harmful behaviors via an uncensored LLM.
# The NBF paper uses these as training goals.
# Multiple HuggingFace mirrors exist; we try several.
CB_SOURCES = [
    ("huggingface", "LLM-LAT/harmful-dataset"),
    ("huggingface", "JailbreakBench/JBB-Behaviors"),
]
CB_GITHUB_URL = "https://github.com/GraySwanAI/circuit-breakers"
CB_PAPER_REF = "Zou et al., 2024 — 'Improving Alignment and Robustness with Circuit Breakers'"
CB_LICENSE = "MIT (per GitHub repo)"


def download_circuit_breakers(
    data_dir: Path,
    force: bool = False,
    dry_run: bool = False,
) -> DatasetInfo | None:
    """Download the Circuit Breakers training goals.

    Tries multiple HuggingFace sources. The original CB paper generates
    harmful behaviors via an uncensored LLM; several mirrors host these.
    """
    cb_dir = data_dir / "circuit_breakers"
    cb_dir.mkdir(parents=True, exist_ok=True)
    info_path = cb_dir / "dataset_info.json"
    raw_path = cb_dir / "raw_data.jsonl"

    if info_path.exists() and raw_path.exists() and not force:
        info = DatasetInfo.load(info_path)
        logger.info("[Circuit Breakers] Already downloaded: %s (%d records)",
                     info.local_path, info.num_records)
        return info

    if dry_run:
        logger.info("[DRY RUN] Would download Circuit Breakers from HuggingFace")
        return None

    try:
        from datasets import load_dataset
    except ImportError:
        _fail_with_clear_message(
            "Circuit Breakers",
            "The 'datasets' package is required for automatic download.",
            "Install it: pip install datasets",
            CB_GITHUB_URL,
        )
        return None

    # Try each source
    records = []
    text_field = None
    source_url = ""
    version_str = "unknown"

    for source_type, source_id in CB_SOURCES:
        logger.info("[Circuit Breakers] Trying %s ...", source_id)
        try:
            ds = load_dataset(source_id, split="train", trust_remote_code=False)
            version = getattr(ds, "version", None)
            version_str = str(version) if version else "unknown"

            for row in ds:
                for key in ["behavior", "prompt", "goal", "input", "query", "text"]:
                    if key in row and row[key] and isinstance(row[key], str):
                        records.append(row[key])
                        if text_field is None:
                            text_field = key
                        break

            if records:
                source_url = f"https://huggingface.co/datasets/{source_id}"
                logger.info("[Circuit Breakers] Got %d records from %s", len(records), source_id)
                break
            else:
                logger.warning("[%s] No text fields found, trying next source", source_id)
                records = []

        except Exception as exc:
            logger.warning("[%s] Failed: %s", source_id, exc)
            continue

    if not records:
        _fail_with_clear_message(
            "Circuit Breakers",
            "Could not download from any HuggingFace source.",
            "The Circuit Breakers training behaviors may need to be generated\n"
            "using the methodology from Zou et al. (2024). See:\n"
            f"  {CB_GITHUB_URL}\n"
            "\n"
            "Alternatively, place a JSONL file with a 'text' field at:\n"
            f"  {raw_path}",
            CB_GITHUB_URL,
        )
        return None

    # Save raw data
    with open(raw_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({"text": rec}, ensure_ascii=False) + "\n")

    info = DatasetInfo(
        name="Circuit Breakers",
        source_url=source_url,
        source_type="huggingface",
        version=version_str,
        download_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        local_path=str(cb_dir),
        raw_file_path=str(raw_path),
        sha256=sha256_file(raw_path),
        num_records=len(records),
        record_field=text_field or "unknown",
        paper_reference=CB_PAPER_REF,
        license=CB_LICENSE,
        notes=f"Downloaded from {source_url}",
    )
    info.save(info_path)

    logger.info("[Circuit Breakers] Downloaded %d records", len(records))
    return info


# --------------------------------------------------------------------------- #
# HarmBench download
# --------------------------------------------------------------------------- #

HB_SOURCE_URL_GITHUB = "https://github.com/centerforaisafety/HarmBench"
HB_SOURCE_URL_HF = "https://huggingface.co/datasets/walledai/HarmBench"
HB_PAPER_REF = "Mazeika et al., 2024 — 'HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal'"
HB_LICENSE = "Apache 2.0 (per GitHub repo)"


def download_harmbench(
    data_dir: Path,
    force: bool = False,
    dry_run: bool = False,
) -> DatasetInfo | None:
    """Download the HarmBench dataset.

    Primary source: HuggingFace datasets hub
    Fallback: GitHub repository
    """
    hb_dir = data_dir / "harmbench"
    hb_dir.mkdir(parents=True, exist_ok=True)
    info_path = hb_dir / "dataset_info.json"
    raw_path = hb_dir / "raw_data.jsonl"

    if info_path.exists() and raw_path.exists() and not force:
        info = DatasetInfo.load(info_path)
        logger.info("[HarmBench] Already downloaded: %s (%d records)",
                     info.local_path, info.num_records)
        return info

    if dry_run:
        logger.info("[DRY RUN] Would download HarmBench from %s", HB_SOURCE_URL_HF)
        return None

    # Try HuggingFace first
    logger.info("[HarmBench] Downloading from %s ...", HB_SOURCE_URL_HF)

    try:
        from datasets import load_dataset
    except ImportError:
        _fail_with_clear_message(
            "HarmBench",
            "The 'datasets' package is required for automatic download.",
            "Install it: pip install datasets",
            HB_SOURCE_URL_GITHUB,
        )
        return None

    records = []
    text_field = None
    version_str = "unknown"

    try:
        ds = load_dataset("walledai/HarmBench", split="test",
                          trust_remote_code=False)
        version = getattr(ds, "version", None)
        version_str = str(version) if version else "unknown"

        for row in ds:
            for key in ["behavior", "prompt", "goal", "input", "query", "text"]:
                if key in row and row[key] and isinstance(row[key], str):
                    records.append(row[key])
                    if text_field is None:
                        text_field = key
                    break
    except Exception as exc:
        logger.warning("HuggingFace download failed: %s", exc)
        logger.info("Trying GitHub fallback...")

        # Try GitHub: download the raw test behaviors file
        try:
            import urllib.request
            github_url = (
                "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
                "/main/data/behavior_datasets/harmbench_behaviors_text_test.csv"
            )
            csv_path = hb_dir / "test_behaviors.csv"
            urllib.request.urlretrieve(github_url, csv_path)

            import csv as csv_mod
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    for key in ["Behavior", "behavior", "Goal", "goal", "text"]:
                        if key in row and row[key]:
                            records.append(row[key])
                            if text_field is None:
                                text_field = key
                            break

            version_str = "github-main"
        except Exception as exc2:
            _fail_with_clear_message(
                "HarmBench",
                f"Failed to download from both HuggingFace and GitHub: {exc}; {exc2}",
                "Download manually from the URL below and place the file in:\n"
                f"  {hb_dir}/raw_data.jsonl\n"
                "Each line should be a JSON object with a text field.",
                HB_SOURCE_URL_GITHUB,
            )
            return None

    if not records:
        _fail_with_clear_message(
            "HarmBench",
            "Dataset downloaded but no text fields found.",
            f"Available columns: {list(ds.column_names) if 'ds' in dir() else 'unknown'}",
            HB_SOURCE_URL_GITHUB,
        )
        return None

    # Save raw data
    with open(raw_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({"text": rec}, ensure_ascii=False) + "\n")

    info = DatasetInfo(
        name="HarmBench",
        source_url=HB_SOURCE_URL_HF,
        source_type="huggingface+github_fallback",
        version=version_str,
        download_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        local_path=str(hb_dir),
        raw_file_path=str(raw_path),
        sha256=sha256_file(raw_path),
        num_records=len(records),
        record_field=text_field or "unknown",
        paper_reference=HB_PAPER_REF,
        license=HB_LICENSE,
        notes=f"Primary: HuggingFace HarmBench/harmbench, fallback: GitHub repo",
    )
    info.save(info_path)

    logger.info("[HarmBench] Downloaded %d records", len(records))
    return info


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

def verify_dataset(data_dir: Path, name: str) -> bool:
    """Verify a downloaded dataset's integrity."""
    info_path = data_dir / name / "dataset_info.json"
    raw_path = data_dir / name / "raw_data.jsonl"

    if not info_path.exists():
        logger.error("[%s] Missing dataset_info.json", name)
        return False
    if not raw_path.exists():
        logger.error("[%s] Missing raw_data.jsonl", name)
        return False

    info = DatasetInfo.load(info_path)
    actual_hash = sha256_file(raw_path)

    if actual_hash != info.sha256:
        logger.error("[%s] SHA256 mismatch: expected %s, got %s",
                     name, info.sha256, actual_hash)
        return False

    # Count records
    count = 0
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1

    if count != info.num_records:
        logger.error("[%s] Record count mismatch: expected %d, got %d",
                     name, info.num_records, count)
        return False

    logger.info("[%s] Verification PASSED: %d records, SHA256=%s",
                name, count, actual_hash[:16])
    return True


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #

def _fail_with_clear_message(
    dataset_name: str,
    error: str,
    instruction: str,
    url: str,
) -> None:
    """Print a clear, actionable error message and fail."""
    logger.error("=" * 60)
    logger.error("DATASET DOWNLOAD FAILED: %s", dataset_name)
    logger.error("=" * 60)
    logger.error("Error: %s", error)
    logger.error("")
    logger.error("Manual download required:")
    logger.error("  URL: %s", url)
    logger.error("  %s", instruction)
    logger.error("")
    logger.error("After downloading, re-run:")
    logger.error("  python scripts/download_datasets.py --verify")
    logger.error("=" * 60)


# --------------------------------------------------------------------------- #
# Summary printer
# --------------------------------------------------------------------------- #

def print_summary(
    cb_info: DatasetInfo | None,
    hb_info: DatasetInfo | None,
) -> None:
    """Print a human-readable summary of downloaded datasets."""
    print("\n" + "=" * 60)
    print("DATASET DOWNLOAD SUMMARY")
    print("=" * 60)

    for info, name in [(cb_info, "Circuit Breakers"), (hb_info, "HarmBench")]:
        print(f"\n{name}")
        print("-" * 40)
        if info is None:
            print("  Status: NOT DOWNLOADED")
            continue
        print(f"  Source:   {info.source_url}")
        print(f"  Version:  {info.version}")
        print(f"  Path:     {info.raw_file_path}")
        print(f"  Records:  {info.num_records}")
        print(f"  SHA256:   {info.sha256[:32]}...")
        print(f"  License:  {info.license}")
        print(f"  Downloaded: {info.download_timestamp}")
        print("  Status: OK")

    print("\n" + "=" * 60)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if already present")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify existing downloads")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded")
    parser.add_argument("--data-dir", default="data/raw",
                        help="Base directory for raw data (default: data/raw)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(level=__import__("logging").DEBUG if args.verbose else __import__("logging").INFO)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.verify:
        cb_ok = verify_dataset(data_dir, "circuit_breakers")
        hb_ok = verify_dataset(data_dir, "harmbench")
        return 0 if (cb_ok and hb_ok) else 1

    # Download
    cb_info = download_circuit_breakers(data_dir, force=args.force, dry_run=args.dry_run)
    hb_info = download_harmbench(data_dir, force=args.force, dry_run=args.dry_run)

    # Summary
    print_summary(cb_info, hb_info)

    # Check success
    if cb_info is None and not args.dry_run:
        logger.error("Circuit Breakers download failed")
        return 1
    if hb_info is None and not args.dry_run:
        logger.error("HarmBench download failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
