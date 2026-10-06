"""Re-measure the Part L switching trace with the corrected validation function.

The first multi-turn run produced an invalid switching trace because the
validation function read `manager.get(role)._inner` (the manager holds the inner
backend) and asked for a role named "judge" (the judge is registered as
"evaluator"). Both were harness defects; the trace contained no measurements.
This driver re-runs exactly that function and merges the corrected block into
the Stage 2 artifacts, preserving the invalid trace under a superseded key.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "results" / "phase17_model_qualification" / "stage2_attacker"

spec = importlib.util.spec_from_file_location(
    "phase17_stage2_multiturn", REPO / "scripts" / "phase17_stage2_multiturn.py"
)
multiturn = importlib.util.module_from_spec(spec)
sys.modules["phase17_stage2_multiturn"] = multiturn
spec.loader.exec_module(multiturn)

candidate = sys.argv[1] if len(sys.argv) > 1 else "qwen38"
print(f"re-measuring switching for {candidate}", flush=True)

manager, llms = multiturn.build_stack(candidate)
block = multiturn.validate_switching(manager, llms)

print(json.dumps({
    "max_resident_models": block["max_resident_models"],
    "vram_after_release": block["vram_after_release"],
}, indent=2))
for row in block["trace"]:
    if "error" in row:
        print(f"  step{row['step']} {row['role']}: ERROR {row['error']}")
    else:
        print(f"  step{row['step']} {row['role']:10s} {row.get('loaded_class'):26s} "
              f"resident={row.get('resident_count')} free={row['vram'].get('driver_free_gib')} "
              f"quant={row.get('quantization')}")

# --- merge into model_switching_validation.json ---------------------------- #
sw_path = OUT / "model_switching_validation.json"
sw = json.loads(sw_path.read_text(encoding="utf-8")) if sw_path.is_file() else {}
sw.setdefault("phase", "17")
sw.setdefault("stage", "2")
sw.setdefault("deliverable", "model_switching_validation.json")
sw.setdefault("candidates", {})
previous = sw["candidates"].get(candidate)
if previous:
    sw.setdefault("superseded_switching_traces", {})[candidate] = {
        "trace": previous.get("trace"),
        "reason": (
            "Invalid harness trace, not a measurement: the validation function read "
            "manager.get(role)._inner (the manager registers the inner backend) and "
            "asked for a role named 'judge' (the judge is registered as 'evaluator'), "
            "so every step recorded AttributeError/KeyError instead of a load. "
            "Re-measured by scripts/phase17_stage2_multiturn_switching_fix.py."
        ),
    }
sw["candidates"][candidate] = block
sw_path.write_text(json.dumps(sw, indent=2), encoding="utf-8")

mt_path = OUT / "attacker_multiturn_qualification.json"
if mt_path.is_file():
    mt = json.loads(mt_path.read_text(encoding="utf-8"))
    mt.setdefault("switching", {})[candidate] = block
    mt_path.write_text(json.dumps(mt, indent=2), encoding="utf-8")

print(f"merged corrected switching block for {candidate}")
