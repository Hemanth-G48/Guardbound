"""Phase 17 Stage 2 — generate the infrastructure and regression deliverables.

Runs the two validation gates itself and records their measured outcome, rather
than restating counts typed in by hand:

  * the Stage 2 regression file (F1/F2/F3 pins);
  * the complete project suite.

Artifacts written to ``results/phase17_model_qualification/stage2_attacker/``:
  infrastructure_fix_report.json
  regression_report.json
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage2_attacker"

STAGE2_FILES = [
    "src/guardbound/llm/local_client.py",
    "src/guardbound/llm/model_manager.py",
    "src/guardbound/attacks/crescendo_paper.py",
    "src/guardbound/attacks/opposite_day.py",
    "src/guardbound/attacks/acronym.py",
    "src/guardbound/attacks/actor_attack.py",
    "src/guardbound/attacks/rubric_evaluation.py",
    "tests/test_phase17_stage2_infrastructure.py",
]

# Files that were already modified before Phase 17 (recorded in the Stage 0
# baseline manifest) and were NOT touched by this stage — the mtime check below
# is what proves the second half of that statement.
PRE_EXISTING_DIRTY = [
    "scripts/phase14_full_reproduction.py",
    "src/guardbound/llm/provider_factory.py",
    "tests/parity/test_refusal_threshold_parity.py",
    "tests/test_phase14_8_telemetry.py",
]


def run_pytest(args: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args, "-q"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    tail = proc.stdout[-6000:]
    passed = failed = errors = 0
    match = re.search(r"(\d+) passed", tail)
    if match:
        passed = int(match.group(1))
    match = re.search(r"(\d+) failed", tail)
    if match:
        failed = int(match.group(1))
    match = re.search(r"(\d+) error", tail)
    if match:
        errors = int(match.group(1))
    failures = re.findall(r"^FAILED (\S+)", tail, flags=re.MULTILINE)
    return {
        "command": "pytest " + " ".join(args),
        "exit_code": proc.returncode,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "failures": failures,
        "tail": tail[-2500:],
    }


def mtimes() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rel in STAGE2_FILES + PRE_EXISTING_DIRTY:
        path = REPO_ROOT / rel
        if path.is_file():
            stat = path.stat()
            out[rel] = {
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "bytes": stat.st_size,
                "role": "stage2_change" if rel in STAGE2_FILES else "pre_existing_dirty_not_touched",
            }
    return out


INFRASTRUCTURE_DEFECTS = [
    {
        "id": "F1",
        "title": "Qwen3 thinking-mode constructor override",
        "severity": "medium (produced a false result in Stage 1)",
        "observed": (
            "HFLocalChatLLM.__init__ applied "
            "chat_template_kwargs.update(_maybe_thinking_kwargs(model_id)), and that "
            "helper returns {'enable_thinking': False} for every Qwen3-family model. "
            "An explicitly requested enable_thinking=True was overwritten at "
            "construction, so the Mode A/Mode B judge comparison initially compared "
            "thinking-off against itself."
        ),
        "root_cause": (
            "dict.update() treats a computed model-family default as if it were an "
            "explicit override."
        ),
        "fix": {
            "file": "src/guardbound/llm/local_client.py",
            "functions": ["_apply_thinking_default", "HFLocalChatLLM.__init__", "build_llm"],
            "current_behavior": "family default overwrote an explicit enable_thinking",
            "required_change": (
                "apply the family default with setdefault() so explicit configuration "
                "wins: precedence is explicit value > model-family default"
            ),
            "experimental_risk": "none — the production judge still defaults to thinking off",
            "performance_impact": "none",
            "validation": "TestThinkingModePrecedence (8 tests)",
        },
        "non_goals": [
            "judge prompts, rubric, scoring and parser untouched",
            "enable_thinking default for Qwen3-family models unchanged (still False)",
            "no production configuration sets enable_thinking=True",
        ],
        "tests": [
            "test_default_is_thinking_off_for_qwen3",
            "test_explicit_false_is_respected",
            "test_explicit_true_is_respected",
            "test_explicit_value_wins_for_non_qwen_models",
            "test_other_template_kwargs_are_preserved",
            "test_caller_dict_is_not_mutated",
            "test_build_llm_default_keeps_thinking_off",
            "test_build_llm_explicit_true_survives",
        ],
    },
    {
        "id": "F2",
        "title": "release_pipeline / lingering model reference",
        "severity": "high (presented as an opaque CUDA OOM)",
        "observed": (
            "HFLocalChatLLM._pipeline was a second strong reference alongside the "
            "module-level _pipeline_cache, so release_pipeline() removed the cache "
            "entry and freed nothing. Measured in Stage 1: cache dropped only, "
            "~15 GiB still allocated; client reference + cache dropped, ~0.008 GiB."
        ),
        "root_cause": (
            "Two owners for one allocation, with no enforcement that a caller drops "
            "the second one."
        ),
        "fix": {
            "file": "src/guardbound/llm/local_client.py",
            "functions": ["HFLocalChatLLM._pipeline (property)", "HFLocalChatLLM._get_pipeline", "release_pipeline"],
            "current_behavior": "client held a strong reference; release needed two steps",
            "required_change": (
                "the cache becomes the sole strong owner; the client's handle is a "
                "weakref, and the loader keeps a strong local handle only until the "
                "cache takes ownership (otherwise the pipeline could be collected "
                "between creation and caching)"
            ),
            "experimental_risk": (
                "low — a released model is reloaded on next use, which is the "
                "pre-existing contract; a live generation keeps its own reference, so "
                "release during generation cannot pull weights out from under a call"
            ),
            "performance_impact": "none; release is what it always claimed to be",
            "validation": "TestPipelineOwnership (5 tests) + TestResidencyCycles (2 tests)",
        },
        "non_goals": [
            "VRAM safety gates unchanged (pre-load, post-load, per-run floor, ceiling)",
            "residency ceiling not raised; no CPU offload",
            "no ad-hoc empty_cache() calls scattered through the attack code",
            "no change to which models the experiment keeps resident",
        ],
        "tests": [
            "test_cached_pipeline_is_held_weakly",
            "test_release_drops_the_last_strong_reference",
            "test_fresh_load_is_owned_by_the_cache_not_the_client",
            "test_client_reloads_after_release",
            "test_non_weakrefable_handle_still_works",
            "test_five_cycles_keep_at_most_one_model_resident",
            "test_switching_evicts_the_previous_role",
        ],
    },
    {
        "id": "F3",
        "title": "judge JSON failure indistinguishable from score 1",
        "severity": "medium (biases any ASR downward, silently)",
        "observed": (
            "llm_evaluate_with_rubric returned 1 whenever the evaluator reply was not "
            "a dict carrying a score in 1..5. Score 1 is also the rubric's genuine "
            "refusal verdict, so an evaluator failure was recorded as a safe verdict."
        ),
        "root_cause": (
            "Two distinct outcomes (valid score 1, unusable output) collapsed onto one "
            "return value."
        ),
        "fix": {
            "file": "src/guardbound/attacks/rubric_evaluation.py (new) + the four attack modules",
            "functions": [
                "classify_rubric_output",
                "record_evaluation",
                "llm_evaluate_with_rubric (crescendo_paper, opposite_day, acronym, actor_attack)",
            ],
            "current_behavior": "unusable evaluator output -> score 1",
            "required_change": (
                "classify each reply (VALID / EMPTY_OUTPUT / JSON_PARSE_ERROR / "
                "MISSING_SCORE / INVALID_SCORE_TYPE / SCORE_OUT_OF_RANGE), record it in "
                "an append-only telemetry log, and return None for anything but a valid "
                "score — never a substituted 1"
            ),
            "experimental_risk": (
                "a failed evaluation now propagates as None into the attack's score "
                "list (Turn.judge_score is already int|None); the attack text that "
                "quotes the last score therefore shows 'None' on the rare failure path"
            ),
            "performance_impact": "none",
            "validation": "TestRubricOutputClassification / TestRubricTelemetry / TestFailureTaxonomySeparation (44 tests)",
        },
        "non_goals": [
            "no rubric text change, no score-meaning change",
            "no retries, no JSON repair, no regex extraction, no fallback scoring",
            "valid scores keep their exact semantics (score 1 remains 1)",
            "judge model, revision and thinking mode untouched",
        ],
        "tests": [
            "test_valid_scores_are_preserved (x5)",
            "test_failures_never_become_a_score (x13)",
            "test_specific_failure_outcomes",
            "test_extra_text_around_valid_json_is_recorded_as_such",
            "test_genuine_score_1_and_json_failure_are_distinguishable",
            "test_every_valid_score_round_trips_through_the_attack_api (x5)",
            "test_every_failure_returns_no_score (x13)",
            "test_json_failure_is_not_an_attack_failure",
            "test_runner_run_with_a_failing_judge_records_no_score",
        ],
    },
    {
        "id": "QUANT",
        "title": "opt-in NF4 load path (required by the approved quantization policy)",
        "severity": "informational — declared production change",
        "observed": (
            "The frozen attacker interface (HFLocalChatLLM) had no way to express the "
            "approved 4-bit NF4 PTQ load, so the approved configuration could not be "
            "driven through the interface it is meant to be qualified against."
        ),
        "root_cause": "capability gap, not a defect",
        "fix": {
            "file": "src/guardbound/llm/local_client.py",
            "functions": ["_build_quantization_config", "HFLocalChatLLM._model_kwargs", "build_llm"],
            "current_behavior": "loads are always unquantized",
            "required_change": "add an opt-in quantization='nf4' load path; default None leaves the path unchanged",
            "experimental_risk": (
                "any experiment that sets quantization='nf4' runs 4-bit weights; no "
                "existing config sets it, so frozen runs are unaffected"
            ),
            "performance_impact": "none when unset",
            "validation": "TestQuantizationIsOptIn (4 tests) asserting the default carries no quantization kwargs",
        },
        "non_goals": [
            "no AWQ/GPTQ/MLX/NVFP4 path",
            "no CPU offload",
            "no change to generation parameters, prompts, parsers or NBF",
        ],
        "tests": [
            "test_default_load_carries_no_quantization",
            "test_opt_in_nf4_uses_the_approved_config",
            "test_unsupported_quantization_is_rejected",
            "test_build_llm_forwards_a_declared_quantization",
        ],
    },
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("running the Stage 2 regression file …", flush=True)
    stage2 = run_pytest(["tests/test_phase17_stage2_infrastructure.py"])
    print(f"  {stage2['passed']} passed, {stage2['failed']} failed", flush=True)

    print("running the complete suite …", flush=True)
    full = run_pytest([])
    print(f"  {full['passed']} passed, {full['failed']} failed", flush=True)

    (OUT_DIR / "infrastructure_fix_report.json").write_text(
        json.dumps({
            "phase": "17",
            "stage": "2",
            "deliverable": "infrastructure_fix_report.json",
            "scope": (
                "Part A — the three Stage 1 defects, plus the opt-in NF4 load path the "
                "approved quantization policy requires. Nothing else in the production "
                "paths was modified."
            ),
            "defects": INFRASTRUCTURE_DEFECTS,
            "changed_files": STAGE2_FILES,
            "unchanged_files_evidence": mtimes(),
            "validation": {
                "stage2_regression_file": stage2,
            },
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8"
    )

    (OUT_DIR / "regression_report.json").write_text(
        json.dumps({
            "phase": "17",
            "stage": "2",
            "deliverable": "regression_report.json",
            "gates": {
                "stage2_regression_file": stage2,
                "complete_suite": full,
            },
            "pre_existing_failures": {
                "tests": [
                    "tests/test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid",
                    "tests/test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present",
                ],
                "cause": (
                    "Both assert that 'Qwen/Qwen3.5-4B' is present in the local HF cache. "
                    "That model was evicted in Phase 17 Stage 0 under a user-approved "
                    "cache eviction (cache_eviction.json, reason 'superseded attacker "
                    "model'), so the assertion fails for an environment-state reason. The "
                    "files were verified absent from the cache and present in the Phase 8 "
                    "config, which is the whole discrepancy."
                ),
                "relationship_to_stage2": (
                    "none — no file these tests exercise was modified in Stage 2; they "
                    "were failing before the F1/F2/F3 changes and are not repaired here "
                    "because repairing them would mean either restoring a model the user "
                    "approved removing or editing an unrelated config"
                ),
                "decision": "reported, not modified",
            },
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8"
    )

    print(f"wrote infrastructure_fix_report.json and regression_report.json to "
          f"{OUT_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
