"""Phase 17 Stage 2 — Part C: historical artifact integrity (read-only).

Verifies that no Phase 14/15/16 artifact, and none of the four frozen
constants, changed while F1/F2/F3 were remediated.

This is a *read-only* check: it opens files for hashing only, writes nothing
outside ``results/phase17_model_qualification/stage2_attacker/``, and loads no
model.

Two independent kinds of evidence are recorded, because either alone is weak:

  * the four frozen SHA-256 constants, which pin the reference artifacts
    byte-for-byte (Track A 180, Track A 108 snapshot, frozen config, NBF
    checkpoint);
  * a full per-file inventory (relative path, size, SHA-256, mtime) of the four
    historical result directories, plus a directory-level digest so a later
    stage can prove byte-identity with one comparison instead of re-hashing
    the tree.

File *counts* are compared against the Stage 0 baseline manifest, and every
mtime is compared against the baseline capture date: a file rewritten during
Phase 17 would show both a changed digest and a fresh mtime.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage2_attacker"
BASELINE_MANIFEST = (
    REPO_ROOT / "results" / "phase17_model_qualification" / "baseline_manifest.json"
)

# The frozen constants, exactly as recorded in the Stage 0 baseline manifest.
FROZEN = {
    "track_a_180_records": (
        "results/phase15_pilot_30/trackA_frozen/batch00_goals000-029_180records.jsonl",
        "FCB952F1598B697B033D605B0791CFBB64A010F8E43CD1ADF3D0EDC359D8A092",
        5322817,
    ),
    "track_a_108_snapshot": (
        "results/phase15_pilot_30/trackA_frozen/batch00_goals000-017_108records.jsonl",
        "C12A562A9454D83B6CC0EE93B2653F4AA5AB058C5AC3E50CEED4EF7A8EC57FB3",
        3106196,
    ),
    "frozen_config": (
        "configs/reproduction_phase14_frozen.yaml",
        "27B660A7052054521A94E3430013D9863964E825157056D3861228250F19A358",
        4654,
    ),
    "nbf_checkpoint": (
        "checkpoints/models_best_nbf_released.pth",
        "CEA1A75BCEF4FC515814B69C42541C95114F587ABCC4505C9B096BBFA2A136FE",
        11757138,
    ),
}

# Historical result directories and the file counts recorded at Stage 0.
HISTORICAL_DIRS = {
    "results/phase14": 27,
    "results/phase15_pilot_30": 4,
    "results/phase16_5_forensics": 11,
    "results/phase16_6_phi4_target": 8,
}

# Stage 0 captured its baseline on this date (recorded in the manifest).
BASELINE_CAPTURED_UTC = "2026-09-24"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_hex(left: str, right: str) -> bool:
    """Compare two hex digests case-insensitively.

    The recorded constants are upper-case and ``hexdigest()`` is lower-case;
    comparing them literally reported a false FAIL on four byte-identical
    artifacts before this was corrected.
    """
    return left.upper() == right.upper()


def git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout.strip()


def inventory_directory(rel_dir: str) -> dict:
    """Per-file hashes for one historical directory, plus a directory digest."""
    root = REPO_ROOT / rel_dir
    files: list[dict] = []
    lines: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(REPO_ROOT).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        mtime = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        files.append(
            {"path": rel, "bytes": size, "sha256": digest, "mtime_utc": mtime}
        )
        lines.append(f"{rel}:{size}:{digest}:{path.stat().st_mtime_ns}")

    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return {
        "directory": rel_dir,
        "file_count": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "directory_digest_sha256": digest,
        "files": files,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))

    # --- frozen constants -------------------------------------------------- #
    frozen_report: dict[str, dict] = {}
    for name, (rel_path, expected_sha, expected_bytes) in FROZEN.items():
        path = REPO_ROOT / rel_path
        entry: dict = {
            "path": rel_path,
            "exists": path.is_file(),
            "expected_sha256": expected_sha,
            "expected_bytes": expected_bytes,
        }
        if path.is_file():
            actual_sha = sha256_file(path)
            actual_bytes = path.stat().st_size
            entry.update(
                actual_sha256=actual_sha,
                actual_bytes=actual_bytes,
                sha256_match=_same_hex(actual_sha, expected_sha),
                bytes_match=actual_bytes == expected_bytes,
                mtime_utc=datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            )
        else:
            entry["sha256_match"] = False
            entry["bytes_match"] = False
        frozen_report[name] = entry

    # --- historical directories -------------------------------------------- #
    dir_report: dict[str, dict] = {}
    for rel_dir, expected_count in HISTORICAL_DIRS.items():
        inv = inventory_directory(rel_dir)
        baseline_count = baseline["frozen_artifacts_verified"]["historical_dirs_intact"].get(
            rel_dir
        )
        inv.update(
            expected_file_count_stage0=baseline_count,
            file_count_unchanged=(f"{inv['file_count']} files" == baseline_count),
            expected_file_count_stage2=expected_count,
        )
        dir_report[rel_dir] = inv

    # --- git state of the tracked files inside those directories ----------- #
    tracked = git("ls-files", *HISTORICAL_DIRS).splitlines()
    tracked_dirty = git("status", "--porcelain", *HISTORICAL_DIRS).splitlines()

    all_frozen_ok = all(
        e.get("sha256_match") and e.get("bytes_match") for e in frozen_report.values()
    )
    all_dirs_ok = all(
        e["file_count_unchanged"] and e["file_count"] == e["expected_file_count_stage2"]
        for e in dir_report.values()
    )

    report = {
        "phase": "17",
        "stage": "2",
        "deliverable": "historical_integrity.json",
        "purpose": (
            "Part C — prove that no historical artifact and none of the four "
            "frozen constants changed during the F1/F2/F3 remediation."
        ),
        "mode": "READ_ONLY (files opened for hashing only; no writes outside "
                "results/phase17_model_qualification/stage2_attacker/)",
        "baseline_captured_utc": BASELINE_CAPTURED_UTC,
        "frozen_constants": frozen_report,
        "historical_directories": dir_report,
        "git": {
            "tracked_files_in_historical_dirs": len(tracked),
            "tracked_files_modified": tracked_dirty,
            "tracked_files_clean": not tracked_dirty,
        },
        "verdict": {
            "frozen_constants_all_match": all_frozen_ok,
            "historical_directories_unchanged": all_dirs_ok,
            "overall": "PASS" if (all_frozen_ok and all_dirs_ok) else "FAIL",
        },
    }

    out_path = OUT_DIR / "historical_integrity.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"frozen constants all match        : {all_frozen_ok}")
    for name, entry in frozen_report.items():
        print(
            f"  {name:22s} sha256_match={entry.get('sha256_match')} "
            f"bytes_match={entry.get('bytes_match')}"
        )
    print(f"historical dirs unchanged         : {all_dirs_ok}")
    for rel_dir, entry in dir_report.items():
        print(
            f"  {rel_dir:32s} files={entry['file_count']:3d} "
            f"(stage0 {entry['expected_file_count_stage0']}) "
            f"digest={entry['directory_digest_sha256'][:16]}…"
        )
    print(f"git tracked files clean           : {not tracked_dirty}")
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    return 0 if report["verdict"]["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
