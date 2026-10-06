"""Rebuild model_switching_validation.json with every candidate's measurements.

Each multi-turn run writes the file for the candidate it just tested, so the
last run wins. The per-candidate blocks are still available in
``attacker_multiturn_qualification.json`` (which merges), so the deliverable is
rebuilt from there, keeping any superseded trace marker.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "phase17_model_qualification" / "stage2_attacker"

switching = {
    "phase": "17",
    "stage": "2",
    "deliverable": "model_switching_validation.json",
    "contract": {
        "sequence": "ATTACKER -> release -> TARGET -> release -> JUDGE -> release -> ATTACKER",
        "checks": [
            "correct model identity (loaded class, model_id attribute)",
            "correct tokenizer",
            "no stale references (cache inspected per step)",
            "max resident models = 1",
            "VRAM returned after the sequence",
        ],
    },
    "candidates": {},
}

mt_path = OUT / "attacker_multiturn_qualification.json"
if mt_path.is_file():
    mt = json.loads(mt_path.read_text(encoding="utf-8"))
    for candidate, block in (mt.get("switching") or {}).items():
        switching["candidates"][candidate] = block

# Keep any previously recorded invalid trace explanation.
sw_path = OUT / "model_switching_validation.json"
if sw_path.is_file():
    previous = json.loads(sw_path.read_text(encoding="utf-8"))
    if previous.get("superseded_switching_traces"):
        switching["superseded_switching_traces"] = previous["superseded_switching_traces"]
    # A block measured for a candidate the merged file does not carry is kept.
    for candidate, block in (previous.get("candidates") or {}).items():
        switching["candidates"].setdefault(candidate, block)

# The first qwen38 switching trace was an invalid harness trace, not a
# measurement: the validation function read manager.get(role)._inner (the
# manager registers the inner backend, not the wrapper) and asked for a role
# named "judge" (the judge is registered as "evaluator"). It was re-measured;
# the invalid trace is kept here, as recorded, so the correction stays visible.
switching.setdefault("superseded_switching_traces", {})["qwen38"] = {
    "recorded": {
        "max_resident_models": 0,
        "vram_after_release": {
            "allocated_gib": 0.008, "reserved_gib": 0.168,
            "peak_allocated_gib": 18.441, "driver_free_gib": 22.495,
            "driver_total_gib": 23.994,
        },
        "trace": [
            {"step": 0, "role": "attacker",
             "error": "AttributeError: 'HFLocalChatLLM' object has no attribute '_inner'"},
            {"step": 1, "role": "target",
             "error": "AttributeError: 'HFLocalChatLLM' object has no attribute '_inner'"},
            {"step": 2, "role": "judge", "error": "KeyError: 'judge'"},
            {"step": 3, "role": "attacker",
             "error": "AttributeError: 'HFLocalChatLLM' object has no attribute '_inner'"},
        ],
    },
    "reason": (
        "Harness defect, not a model or residency result: every step recorded an "
        "exception instead of a load, so no identity, resident-count or VRAM claim "
        "could be read from it. Re-measured by "
        "scripts/phase17_stage2_multiturn_switching_fix.py; the corrected block in "
        "candidates.qwen38 is the measurement."
    ),
}

sw_path.write_text(json.dumps(switching, indent=2), encoding="utf-8")

for candidate, block in switching["candidates"].items():
    print(f"{candidate}: max_resident={block.get('max_resident_models')} "
          f"after_release_free={(block.get('vram_after_release') or {}).get('driver_free_gib')} GiB "
          f"steps={len(block.get('trace') or [])}")
print(f"superseded traces kept: {len(switching.get('superseded_switching_traces') or {})}")
print(f"wrote {sw_path.relative_to(REPO)}")
