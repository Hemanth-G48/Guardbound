"""Merge the per-candidate JSON-qualification blocks into one deliverable.

Each qualification run writes `attacker_json_qualification.json` for the
candidate it just tested, so the file holds only the most recent one. This
rebuilds it from the per-candidate feasibility records, which are the
authoritative store (they carry the block under `json_qualification`).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "phase17_model_qualification" / "stage2_attacker"

CANDIDATES = ("qwen38", "qwen36", "ministral", "devstral")

blocks: dict[str, dict] = {}
for key in CANDIDATES:
    path = OUT / f"candidate_{key}_feasibility.json"
    if not path.is_file():
        continue
    record = json.loads(path.read_text(encoding="utf-8"))
    block = record.get("json_qualification")
    if block:
        blocks[record["model_id"]] = block

summary = {
    "phase": "17",
    "stage": "2",
    "deliverable": "attacker_json_qualification.json",
    "policy": (
        "No JSON repair, no retry, no fallback generation, no reasoning stripping, "
        "no model-specific prompt edits, no regex or fenced-JSON recovery. Raw output "
        "is preserved per candidate under raw_attacker_outputs/. A candidate that "
        "cannot be loaded by the frozen interface has no JSON block at all rather "
        "than a zero."
    ),
    "harness": (
        "Production prompt and call signature: crescendo_paper.generate_crescendo_step "
        "with the real attacker client; classification applies the project's own parse "
        "chain (json.loads then _extract_json_block) and nothing else."
    ),
    "candidates": blocks,
    "not_measured": [
        "prism-ml/bonsai-27b (BLOCKED, Stage 0)",
        "mistralai/Ministral-3-14B-Reasoning-2512 (MODEL_INTERFACE_INCOMPATIBLE)",
        "mistralai/Devstral-Small-2-24B-Instruct-2512 (MODEL_INTERFACE_INCOMPATIBLE)",
    ],
}

path = OUT / "attacker_json_qualification.json"
path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
for model_id, block in blocks.items():
    print(f"{model_id:46s} n={block.get('n_cases')} valid={block.get('valid_json_total')} "
          f"rate={block.get('validity_rate')} parse_errors={block.get('json_parse_error')}")
print(f"wrote {path.relative_to(REPO)} with {len(blocks)} candidate block(s)")
