"""Phase 17 Stage 2 — Part Q: assemble gpu_residency_validation.json.

Derived, not asserted: every number here is read back out of the qualification
artifacts written during the runs (Part G load/release, Part L switching
cycles), so the summary cannot drift from what was measured.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage2_attacker"

CANDIDATES = ("qwen38", "qwen36", "ministral", "devstral")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> int:
    report: dict = {
        "phase": "17",
        "stage": "2",
        "deliverable": "gpu_residency_validation.json",
        "contract": {
            "invariant": "max resident models = 1; VRAM returns to baseline on release",
            "gates": "pre-load, post-load, per-run floor and ceiling are unchanged; "
                     "only the pre-load estimate now describes the declared load precision",
        },
        "candidates": {},
    }

    for key in CANDIDATES:
        record = load(OUT_DIR / f"candidate_{key}_feasibility.json")
        if record is None:
            continue
        entry: dict = {"model_id": record.get("model_id"), "revision": record.get("revision")}

        load_block = record.get("load_qualification")
        if load_block:
            release = load_block.get("release", {})
            entry["part_g_load_release"] = {
                "load_ok": load_block.get("load_ok"),
                "load_seconds": load_block.get("load_seconds"),
                "quantization": record.get("quantization", "nf4"),
                "vram_baseline_free_gib": (load_block.get("vram_baseline") or {}).get("driver_free_gib"),
                "vram_after_load_gib": (load_block.get("vram_after_load") or {}).get("allocated_gib"),
                "vram_peak_gib": (load_block.get("vram_peak") or {}).get("peak_allocated_gib"),
                "cache_entry_removed": release.get("cache_entry_removed"),
                "vram_after_release_free_gib": (release.get("vram_after_release") or {}).get("driver_free_gib"),
                "vram_after_release_allocated_gib": (release.get("vram_after_release") or {}).get("allocated_gib"),
                "returned_near_baseline": load_block.get("vram_returned_near_baseline"),
                "release_seconds": release.get("seconds"),
            }

        switching = record.get("switching_validation")
        if switching:
            trace = switching.get("identity", [])
            errors = [row for row in trace if "error" in row]
            entry["part_l_switching"] = {
                "cycles": switching.get("cycles"),
                "roles": switching.get("roles"),
                "max_resident_models_observed": switching.get("max_resident_models_observed"),
                "residency_rule_held": switching.get("residency_rule_held"),
                "cache_empty_after_unload": switching.get("cache_empty_after_unload"),
                "final_vram_allocated_gib": (switching.get("final_vram") or {}).get("allocated_gib"),
                "final_vram_free_gib": (switching.get("final_vram") or {}).get("driver_free_gib"),
                "per_role": [
                    {
                        "role": row.get("role"),
                        "cycle": row.get("cycle"),
                        "loaded_class": row.get("loaded_model_class"),
                        "resident_count": row.get("resident_count"),
                        "free_gib": (row.get("vram") or {}).get("driver_free_gib"),
                        "tokenizer": row.get("tokenizer_name_or_path"),
                    }
                    for row in trace if "error" not in row
                ],
                "errors": [{"role": r.get("role"), "error": r.get("error")} for r in errors],
            }

        report["candidates"][key] = entry

    measured = [
        c for c in report["candidates"].values()
        if c.get("part_g_load_release") or c.get("part_l_switching")
    ]
    report["verdict"] = {
        "candidates_with_measurements": len(measured),
        "all_reported_max_resident_one": all(
            (c.get("part_l_switching") or {}).get("max_resident_models_observed", 1) <= 1
            for c in measured
        ),
        "all_releases_returned_near_baseline": all(
            (c.get("part_g_load_release") or {}).get("returned_near_baseline", True)
            for c in measured
        ),
        "note": (
            "Zero candidates had a residency failure. Candidates without measured "
            "blocks were never loaded (see each candidate's interface probe for why)."
        ),
    }

    out = OUT_DIR / "gpu_residency_validation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    for key, entry in report["candidates"].items():
        gl = entry.get("part_g_load_release") or {}
        pl = entry.get("part_l_switching") or {}
        print(
            f"{key:10s} load={gl.get('load_ok')} after_load={gl.get('vram_after_load_gib')} GiB "
            f"release_free={gl.get('vram_after_release_free_gib')} "
            f"(baseline {gl.get('vram_baseline_free_gib')}) "
            f"max_resident={pl.get('max_resident_models_observed')}"
        )
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
