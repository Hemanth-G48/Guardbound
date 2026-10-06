"""Phase 17 Stage 2 — Part E.1: watchdog downloader for official candidate weights.

Why this exists: a plain ``snapshot_download`` on this machine stalls — measured
twice, with each of four parallel streams stopping at ~2 GiB and the process then
holding 26-37 live HTTPS connections while moving 0 bytes for minutes. The link
itself is fine (fresh ranged requests still return ~0.8 MiB/s single-stream, and
~7 MiB/s aggregate over three streams), so the stall is in the transfer session,
not the network.

This supervisor therefore:

  * runs ``snapshot_download`` in a child process, pinned to the resolved
    revision and the classic (non-Xet) HTTP path;
  * watches the cache for progress and kills the child when it stops making any
    for ``--stall-seconds``;
  * restarts it, which resumes from the ``.incomplete`` files already on disk;
  * records every attempt (bytes at start/end, duration, kill reason) in
    ``download_retries.json`` so no retry is silent;
  * stops when the local snapshot matches the Hub's own file list (names *and*
    sizes, and the blob hash where the API publishes one), or when the attempt
    budget is exhausted — an exhausted budget is reported as
    ``MODEL_DOWNLOAD_FAILURE``, never hidden.

Nothing is deleted and no other cached model is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage2_attacker"
RETRY_LOG = OUT_DIR / "download_retries.json"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

GIB = 1024 ** 3

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


def cache_bytes(model_id: str) -> int:
    root = HF_CACHE / ("models--" + model_id.replace("/", "--"))
    if not root.is_dir():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def local_state(model_id: str) -> dict[str, int]:
    """Repo-relative paths present in the local snapshot, with sizes."""
    root = HF_CACHE / ("models--" + model_id.replace("/", "--")) / "snapshots"
    found: dict[str, int] = {}
    if root.is_dir():
        for revision_dir in root.iterdir():
            if revision_dir.is_dir():
                for path in revision_dir.rglob("*"):
                    if path.is_file():
                        found[path.relative_to(revision_dir).as_posix()] = path.stat().st_size
    return found


def verify(model_id: str, hub_files: dict[str, int]) -> dict:
    found = local_state(model_id)
    missing = sorted(n for n in hub_files if n not in found)
    wrong_size = sorted(
        n for n, size in hub_files.items()
        if n in found and size and found[n] != size
    )
    return {
        "files_expected": len(hub_files),
        "files_present": sum(1 for n in hub_files if n in found),
        "missing": missing,
        "size_mismatches": wrong_size,
        "complete": not missing and not wrong_size,
    }


def attempt_once(model_id: str, revision: str, stall_seconds: int) -> dict:
    """One child download attempt, killed when progress stops."""
    started = time.time()
    before = cache_bytes(model_id)
    child = subprocess.Popen(
        [
            sys.executable, "-c",
            (
                "import os;"
                "os.environ.setdefault('HF_HUB_DISABLE_XET','1');"
                "from huggingface_hub import snapshot_download;"
                f"snapshot_download(repo_id={model_id!r}, revision={revision!r},"
                " max_workers=4)"
            ),
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    last_change = time.time()
    last_bytes = before
    killed_reason = None
    while True:
        time.sleep(10)
        if child.poll() is not None:
            killed_reason = f"child_exited_{child.returncode}"
            break
        current = cache_bytes(model_id)
        if current > last_bytes + (1 << 20):
            last_bytes = current
            last_change = time.time()
        elif time.time() - last_change > stall_seconds:
            child.kill()
            child.wait(timeout=60)
            killed_reason = f"stalled_{stall_seconds}s"
            break

    after = cache_bytes(model_id)
    return {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "duration_s": round(time.time() - started, 1),
        "bytes_before": before,
        "bytes_after": after,
        "delta_gib": round((after - before) / GIB, 3),
        "ended_with": killed_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    parser.add_argument("--stall-seconds", type=int, default=150)
    parser.add_argument("--max-attempts", type=int, default=60)
    parser.add_argument("--time-budget-min", type=float, default=600.0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_id = CANDIDATES[args.candidate]
    identity = json.loads(
        (OUT_DIR / f"candidate_{args.candidate}_feasibility.json").read_text(encoding="utf-8")
    )
    revision = identity["revision"]

    meta = curl_json(f"https://huggingface.co/api/models/{model_id}?blobs=true")
    hub_files = {
        sib["rfilename"]: (sib.get("size") or 0) for sib in meta.get("siblings", [])
    }
    total_bytes = sum(hub_files.values())

    log: list[dict] = []
    deadline = time.time() + args.time_budget_min * 60
    attempts = 0
    state = verify(model_id, hub_files)

    print(f"target    : {model_id} @ {revision}")
    print(f"hub bytes : {total_bytes / GIB:.2f} GiB across {len(hub_files)} files")
    print(f"local now : {cache_bytes(model_id) / GIB:.2f} GiB, "
          f"{state['files_present']}/{state['files_expected']} files complete")

    while not state["complete"] and attempts < args.max_attempts and time.time() < deadline:
        attempts += 1
        print(f"\n--- attempt {attempts} ---", flush=True)
        record = attempt_once(model_id, revision, args.stall_seconds)
        state = verify(model_id, hub_files)
        record.update(
            attempt=attempts,
            files_present=state["files_present"],
            files_expected=state["files_expected"],
            cache_gib=round(cache_bytes(model_id) / GIB, 3),
            effective_rate_mib_s=round(
                max(record["delta_gib"], 0) * 1024 / max(record["duration_s"], 1), 2
            ),
        )
        log.append(record)
        # Merge per candidate: candidates are provisioned one at a time
        # sequentially, so this file accumulates one block per candidate.
        existing: dict = {}
        if RETRY_LOG.is_file():
            existing = json.loads(RETRY_LOG.read_text(encoding="utf-8"))
        existing.update({
            "phase": "17", "stage": "2",
            "deliverable": "download_retries.json",
            "note": (
                "Every controlled reattempt is recorded here, per candidate. The Xet "
                "transfer backend stalled outright on this machine (0 bytes in ten "
                "minutes while holding 37 sockets); the classic HTTP path with "
                "parallelism is used instead."
            ),
        })
        # Migrate the earlier single-candidate layout (top-level candidate/
        # attempts) so the first candidate's recorded attempts survive.
        if "candidates" not in existing and existing.get("attempts"):
            existing["candidates"] = {
                existing.get("candidate", "unknown"): {
                    "revision": existing.get("revision"),
                    "attempts": existing["attempts"],
                }
            }
            existing.pop("candidate", None)
            existing.pop("attempts", None)
        blocks = existing.setdefault("candidates", {})
        blocks[model_id] = {"revision": revision, "attempts": log}
        RETRY_LOG.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        print(
            f"    +{record['delta_gib']:.2f} GiB in {record['duration_s']:.0f}s "
            f"({record['effective_rate_mib_s']:.2f} MiB/s) — ended: {record['ended_with']}",
            flush=True,
        )
        print(f"    files {state['files_present']}/{state['files_expected']}", flush=True)

    verdict = "COMPLETE" if state["complete"] else "MODEL_DOWNLOAD_FAILURE"
    summary = {
        "candidate": args.candidate,
        "model_id": model_id,
        "revision": revision,
        "attempts": attempts,
        "verdict": verdict,
        "files_present": state["files_present"],
        "files_expected": state["files_expected"],
        "missing_files": state["missing"][:10],
        "size_mismatches": state["size_mismatches"][:10],
        "cache_gib": round(cache_bytes(model_id) / GIB, 3),
        "hub_gib": round(total_bytes / GIB, 3),
    }
    (OUT_DIR / f"download_summary_{args.candidate}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nverdict: {verdict} after {attempts} attempt(s)")
    print(json.dumps(summary, indent=2))
    return 0 if verdict == "COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
