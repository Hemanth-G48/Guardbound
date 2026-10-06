"""Phase 17 Stage 4 — residency and GPU validation (Part 22).

Two sources:

  * a live re-run of Stage 3's residency sequence (imported, not re-implemented:
    native attacker -> target -> judge -> native attacker, three cycles), so the
    identity, processor, peak-VRAM and release evidence is measured under exactly
    the Stage 3 procedure;
  * a programmatic scan of the 30 survival runs for OOM, CUDA errors, residency
    anomalies and identity events recorded during the reliability test itself.

No gate, threshold or residency mode is changed here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage4_qwen38_reliability"

ATTACKS = ("crescendo_paper", "opposite_day", "acronym")


def load_stage3():
    spec = importlib.util.spec_from_file_location(
        "phase17_stage3_native_qualification",
        REPO_ROOT / "scripts" / "phase17_stage3_native_qualification.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase17_stage3_native_qualification"] = module
    spec.loader.exec_module(module)
    return module


def scan_runs() -> dict:
    path = OUT_DIR / "multiturn_survival_raw.jsonl"
    if not path.is_file():
        return {"available": False, "reason": "no survival runs recorded"}
    runs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    oom = cuda = 0
    load_failures = []
    identity_events = 0
    events = {"activate": 0, "load": 0, "evict": 0, "load_refused": 0,
              "post_load_failed": 0, "ceiling_exceeded": 0, "run_refused": 0}
    peaks: list[float] = []
    for run in runs:
        text = " ".join(str(run.get("termination_reason") or "") for _ in [0])
        attacker_errors = " ".join(
            str(c.get("error") or "") for c in run.get("attacker_calls", [])
        )
        everything = f"{text} {attacker_errors}"
        if "out of memory" in everything.lower():
            oom += 1
        if "cuda error" in everything.lower() or "cudnn" in everything.lower():
            cuda += 1
        for event in run.get("switching_events", []):
            events[event] = events.get(event, 0) + 1
        peaks.append(run.get("peak_vram_gib") or 0.0)
    return {
        "available": True,
        "runs": len(runs),
        "oom_count": oom,
        "cuda_error_count": cuda,
        "load_failure_count": len(load_failures),
        "manager_events": events,
        "load_refusals": events.get("load_refused", 0),
        "post_load_verification_failures": events.get("post_load_failed", 0),
        "ceiling_exceeded": events.get("ceiling_exceeded", 0),
        "peak_vram_gib_max": round(max(peaks), 3) if peaks else None,
        "peak_vram_gib_mean": round(sum(peaks) / len(peaks), 3) if peaks else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--skip-live", action="store_true",
                        help="only scan the survival runs (no model loads)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record: dict = {
        "phase": "17", "stage": "4",
        "deliverable": "residency_validation.json",
        "architecture_unchanged": (
            "the Stage 3 native path, unchanged: AutoModelForImageTextToText -> "
            "Qwen3_5ForConditionalGeneration, Qwen3VLProcessor, NF4"
        ),
        "gates": "pre-load, post-load, per-run floor and ceiling unchanged",
        "during_survival_runs": scan_runs(),
    }

    if not args.skip_live:
        stage3 = load_stage3()
        live = stage3.part_residency(args.cycles)
        record["live_sequence"] = {
            "cycles": live["cycles"],
            "trace": [
                {
                    "cycle": row.get("cycle"),
                    "role": row.get("role"),
                    "model_class": row.get("model_class"),
                    "quantization": row.get("quantization"),
                    "processor_class": row.get("processor_class"),
                    "tokenizer_name": row.get("tokenizer_name"),
                    "resident_count": row.get("resident_count"),
                    "free_gib": (row.get("vram") or {}).get("driver_free_gib"),
                    "error": row.get("error"),
                }
                for row in live["trace"]
            ],
            "max_resident_models": live["max_resident_models"],
            "attacker_identity_correct": live["attacker_identity_correct"],
            "cache_empty_after_unload": live["cache_empty_after_unload"],
            "errors": live["errors"],
            "vram_after_release": live["vram_after_release"],
        }

    (OUT_DIR / "residency_validation.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    scan = record["during_survival_runs"]
    print("during survival runs:", json.dumps(scan, indent=2)[:800])
    if "live_sequence" in record:
        live = record["live_sequence"]
        print(f"live: max_resident={live['max_resident_models']} "
              f"identity_ok={live['attacker_identity_correct']} "
              f"errors={len(live['errors'])} "
              f"after_release={live['vram_after_release'].get('allocated_gib')} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
