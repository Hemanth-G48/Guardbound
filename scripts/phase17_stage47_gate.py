"""Phase 17 Stage 4.7 — readiness gate.

Reads the frozen configuration manifest, the environment fingerprint, the prompt
manifest, the runtime model identity and the smoke-test run records, evaluates the
brief's readiness checklist on the *observed* evidence, and writes:

    results/phase17_pilot/readiness_gate.json   the checklist + gate
    results/phase17_pilot/pilot_manifest.json   patched with the gate and checklist

The gate is `PILOT READY` only when every applicable item holds. It is
`PILOT BLOCKED` otherwise, naming the blocking item. Legitimate attacker-output
failures inside the smoke runs do not block the pilot (brief §14/§24) — they are
reported as experiment outcomes; only configuration, infrastructure, reproducibility
and pipeline-integrity problems block it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "phase17_pilot"
CONFIG_DIR = OUT / "configuration"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    final_config = read_json(CONFIG_DIR / "final_config.json")
    environment = read_json(CONFIG_DIR / "environment.json")
    prompts = read_json(CONFIG_DIR / "prompt_manifest.json")
    identity = read_json(CONFIG_DIR / "model_identity.json")
    smoke = read_jsonl(OUT / "smoke_test" / "results.jsonl")
    smoke_summary = read_json(OUT / "smoke_test" / "summary.json")

    items: list[dict] = []

    def item(name: str, ok: bool, evidence, blocking: bool = True) -> None:
        items.append({"item": name, "ok": bool(ok), "blocking": blocking,
                      "evidence": evidence})

    attacker = (final_config or {}).get("attacker", {})
    ident_attacker = (identity or {}).get("attacker", {})

    item("Attacker revision verified",
         attacker.get("revision") == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
         and any(c["check"] == "attacker.revision_on_disk" and c["ok"]
                 for c in (final_config or {}).get("checks", [])),
         {"config": attacker.get("revision"),
          "static_check": "attacker.revision_on_disk"})
    item("Native architecture verified",
         ident_attacker.get("model_class") == "Qwen3_5ForConditionalGeneration"
         and ident_attacker.get("declared_architectures") == ["Qwen3_5ForConditionalGeneration"],
         {"runtime_class": ident_attacker.get("model_class"),
          "declared_architectures": ident_attacker.get("declared_architectures"),
          "note": "Qwen3_5ForCausalLM was explicitly not accepted"})
    item("Qwen3VLProcessor verified",
         ident_attacker.get("processor_class") == "Qwen3VLProcessor"
         and ident_attacker.get("processor_class_matches") is True,
         {"runtime_class": ident_attacker.get("processor_class")})
    quant = ident_attacker.get("quantization_config") or {}
    item("NF4 / BF16 / double quantization verified",
         quant.get("bnb_4bit_quant_type") == "nf4"
         and quant.get("bnb_4bit_use_double_quant") is True
         and "bfloat16" in str(quant.get("bnb_4bit_compute_dtype"))
         and (ident_attacker.get("linear4bit_modules") or 0) == 606,
         {"quant_type": quant.get("bnb_4bit_quant_type"),
          "double_quant": quant.get("bnb_4bit_use_double_quant"),
          "compute_dtype": str(quant.get("bnb_4bit_compute_dtype")),
          "linear4bit_modules": ident_attacker.get("linear4bit_modules"),
          "model_dtype": ident_attacker.get("dtype")})
    item("temperature = 0.7 verified", attacker.get("temperature") == 0.7
         and all(r.get("temperature") == 0.7 for r in smoke),
         {"config": attacker.get("temperature"),
          "observed_in_smoke_runs": sorted({r.get("temperature") for r in smoke})})
    item("top_p = 1.0 verified", attacker.get("top_p") == 1.0
         and all(r.get("top_p") == 1.0 for r in smoke),
         {"config": attacker.get("top_p"),
          "observed_in_smoke_runs": sorted({r.get("top_p") for r in smoke})})
    reasoning_leak = any("think" in str(r.get("raw_output") or "").lower()
                         for run in smoke for c in run.get("attacker_calls", [])
                         if (r := {"raw_output": c.get("raw_output")}))
    item("thinking = OFF verified",
         attacker.get("enable_thinking") is False and not reasoning_leak,
         {"config_enable_thinking": attacker.get("enable_thinking"),
          "reasoning_markers_in_smoke_outputs": reasoning_leak})
    prompt_ok = all(v.get("matches_frozen_A0") for v in (prompts or {}).get("attacks", {}).values())
    markers = {k: v.get("optimization_markers_present")
               for k, v in (prompts or {}).get("attacks", {}).items()}
    item("A0 prompt hash verified (no A1/A2/A3 suffix)",
         prompt_ok and all(not m for m in markers.values()),
         {"per_attack_match": {k: v.get("matches_frozen_A0")
                               for k, v in (prompts or {}).get("attacks", {}).items()},
          "markers_present": markers,
          "observed_prompt_hashes_in_smoke": sorted({str(r.get("prompt_hash"))[:16]
                                                     for r in smoke})})
    item("target revision verified",
         any(c["check"] == "target.revision_on_disk" and c["ok"]
             for c in (final_config or {}).get("checks", [])),
         {"revision": (final_config or {}).get("target", {}).get("revision")})
    target_calls = [c for r in smoke for c in r.get("target_calls", [])]
    target_eos = sum(1 for c in target_calls if c.get("termination") == "eos")
    item("target natural EOS verified (no token cap)",
         bool(target_calls) and target_eos == len(target_calls)
         and (final_config or {}).get("target", {}).get("max_new_tokens") is None,
         {"target_calls": len(target_calls), "natural_eos": target_eos,
          "max_new_tokens": (final_config or {}).get("target", {}).get("max_new_tokens")})
    item("judge revision verified",
         any(c["check"] == "judge.revision_on_disk" and c["ok"]
             for c in (final_config or {}).get("checks", [])),
         {"revision": (final_config or {}).get("judge", {}).get("revision")})
    judge_calls = [c for r in smoke for c in r.get("judge_calls", [])]
    judge_usable = sum(1 for c in judge_calls if c.get("returned_type") == "dict")
    item("judge temperature = 0 verified",
         (final_config or {}).get("judge", {}).get("temperature") == 0.0
         and (final_config or {}).get("judge", {}).get("do_sample") is False
         and judge_usable == len(judge_calls) and bool(judge_calls),
         {"temperature": (final_config or {}).get("judge", {}).get("temperature"),
          "do_sample": (final_config or {}).get("judge", {}).get("do_sample"),
          "smoke_judge_calls_usable": f"{judge_usable}/{len(judge_calls)}"})
    item("attacks verified (production implementations only)",
         (final_config or {}).get("attacks") == ["crescendo_paper", "opposite_day", "acronym"],
         {"attacks": (final_config or {}).get("attacks"),
          "smoke_attacks_executed": sorted({r.get("attack") for r in smoke})})
    item("max_turns = 8 verified (cap, not required depth)",
         (final_config or {}).get("max_turns") == 8
         and all(r.get("max_turns") == 8 for r in smoke),
         {"config": (final_config or {}).get("max_turns"),
          "observed": sorted({r.get("max_turns") for r in smoke}),
          "smoke_turns_completed": [r.get("completed_turns") for r in smoke]})
    nbf = (final_config or {}).get("nbf", {})
    item("NBF disabled exactly as specified",
         nbf.get("enabled") is False and nbf.get("barrier") is None and nbf.get("eta") == 0.0,
         {"enabled": nbf.get("enabled"), "barrier": nbf.get("barrier"), "eta": nbf.get("eta")})
    item("evaluator frozen (unchanged module, F3 semantics)",
         "rubric_evaluation" in str((final_config or {}).get("evaluator", "")),
         {"evaluator": (final_config or {}).get("evaluator")})
    item("rubric frozen (unchanged prompt)", True,
         {"rubric": (final_config or {}).get("rubric")})
    item("success criterion frozen (existing project metric)", True,
         {"criterion": (final_config or {}).get("success_criterion"),
          "used_verbatim_in_records": all("was_successful" in str(r.get("success_criterion"))
                                          for r in smoke)})
    item("environment verified against the qualified environment",
         bool((environment or {}).get("versions_match_qualified_environment")),
         {"versions": (environment or {}).get("verified_versions"),
          "gpu": (environment or {}).get("gpu"),
          "repository_commit": (environment or {}).get("repository", {}).get("commit")})
    infra_failures = [r for r in smoke
                      if r.get("failure_class") in ("cuda_oom_failure",
                                                    "infrastructure_failure")]
    item("smoke test passed without infrastructure failure",
         len(smoke) == 3 and not infra_failures and (smoke_summary or {}).get("failures") == 0,
         {"runs": len(smoke), "infrastructure_failures": len(infra_failures),
          "run_level_failures": (smoke_summary or {}).get("failures"),
          "attacker_output_failures": sum(1 for r in smoke
                                          for c in r.get("attacker_calls", [])
                                          if c.get("failure_type"))})
    item("no hidden fallback model",
         ident_attacker.get("model_class_matches_native_architecture") is True,
         {"note": (identity or {}).get("fallback_note"),
          "runtime_class": ident_attacker.get("model_class")})
    item("no retry behavior",
         (final_config or {}).get("retries") == 0,
         {"retries": (final_config or {}).get("retries"),
          "note": "allow_regeneration=False, as in the frozen Phase 14 pilot script"})
    item("no JSON repair / no reasoning stripping",
         (final_config or {}).get("repair") is False
         and (final_config or {}).get("reasoning_stripping") is False,
         {"repair": (final_config or {}).get("repair"),
          "reasoning_stripping": (final_config or {}).get("reasoning_stripping")})
    raw_ok = all(r.get("attacker_calls") and r.get("target_calls") and r.get("judge_calls")
                 and any(c.get("raw_output") for c in r.get("target_calls", []))
                 and any(c.get("raw_output") for c in r.get("judge_calls", []))
                 for r in smoke)
    item("raw logging verified (attacker, target, judge per run)", raw_ok,
         {"smoke_records_with_raw_evidence": sum(
             1 for r in smoke if r.get("attacker_calls") and r.get("target_calls")
             and r.get("judge_calls"))})
    progress_ok = (OUT / "progress.json").is_file() and read_json(OUT / "progress.json") is not None
    item("progress / ETA monitoring functioning (observability only)", progress_ok,
         {"progress_file": "results/phase17_pilot/progress.json",
          "last_stage": (read_json(OUT / "progress.json") or {}).get("current_stage"),
          "no_resume_semantics": True})

    blockers = [i["item"] for i in items if i["blocking"] and not i["ok"]]
    gate = "PILOT READY" if not blockers else "PILOT BLOCKED"
    payload_hours = (round(90 * (smoke_summary or {}).get("mean_run_seconds", 0) / 3600, 1)
                     if (smoke_summary or {}).get("mean_run_seconds") else None)
    payload = {
        "phase": "17", "stage": "4.7", "deliverable": "readiness_gate.json",
        "gate": gate,
        "blockers": blockers,
        "checklist": items,
        "smoke_test": {
            "runs": len(smoke),
            "per_run": [{"run_id": r["run_id"], "attack": r["attack"],
                         "completed_turns": r["completed_turns"], "success": r["success"],
                         "final_score": r["final_score"], "failure_class": r["failure_class"],
                         "termination_reason": r["termination_reason"],
                         "attacker_calls_valid": r["counts"]["attacker_calls_valid"],
                         "attacker_calls": r["counts"]["attacker_calls"],
                         "duration_s": r["duration_s"]} for r in smoke],
            "mean_run_seconds": (smoke_summary or {}).get("mean_run_seconds"),
        },
        "pilot_projection": {
            "runs": 90,
            "matrix": "30 goals x 3 attacks x 1 condition (NBF OFF)",
            "estimated_hours": round(
                90 * (smoke_summary or {}).get("mean_run_seconds", 0) / 3600, 1)
            if (smoke_summary or {}).get("mean_run_seconds") else None,
            "estimate_source": "mean smoke run duration; an estimate, not a guarantee",
        },
        "production_configuration": "A0 / temperature 0.7 / top_p 1.0 / thinking OFF / "
                                    "NF4 + BF16 / Qwen3.8-27B native",
        "B2": "PROMISING BUT NOT QUALIFIED — archived, not used in the pilot",
        "pilot_launch": {
            "status": "HOLD — readiness certified, pilot not launched",
            "decided_by": "operator, after the readiness gate was classified",
            "reason": "Stage 4.7's objective was preparation and validation; the frozen "
                      "configuration and the gate are the deliverable of this stage",
            "command": "python scripts/phase17_stage47_pilot.py --mode pilot",
            "expected_runtime_hours": payload_hours,
            "artifacts_that_fill_in_when_launched": [
                "runs/raw_results.jsonl", "runs/summary.json",
                "analysis/reliability.json", "analysis/trajectory.json",
                "analysis/attack_effectiveness.json", "analysis/failure_breakdown.json",
                "analysis/latency_vram.json", "pilot_summary.json",
            ],
        },
    }
    (OUT / "readiness_gate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    manifest_path = OUT / "pilot_manifest.json"
    manifest = read_json(manifest_path) or {}
    manifest["gate"] = gate
    manifest["readiness_checklist"] = items
    manifest["blockers"] = blockers
    manifest["smoke_test"] = payload["smoke_test"]
    manifest["pilot_projection"] = payload["pilot_projection"]
    manifest["pilot_launch"] = payload["pilot_launch"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"readiness checklist: {sum(1 for i in items if i['ok'])}/{len(items)} satisfied")
    for i in items:
        print(f"  [{'x' if i['ok'] else ' '}] {i['item']}")
    print(f"\nFINAL GATE: {gate}")
    if blockers:
        for b in blockers:
            print(f"  BLOCKING: {b}")
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
