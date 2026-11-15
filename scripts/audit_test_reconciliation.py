#!/usr/bin/env python3
"""Test-result reconciliation across the three measurement snapshots.

Why this exists
---------------
The readiness reports quoted several totals (758, 878, 922, 971, 974) that were
all real but measured at different moments. This script makes the comparison
mechanical instead of arithmetic-by-hand, and it is explicit about what is
*measured* and what is *recorded history*.

What is knowable
----------------
Snapshot A and B were measured before this audit, and only their **not-passed**
node IDs plus aggregate counts were recorded. The complete collected-test sets
for A and B cannot be reconstructed, so the comparison below is exact on the
not-passed subsets and count-based on the collected sets. Snapshot C is
re-measured from a JUnit XML produced by the current tree.

Provenance
----------
A  758 passed / 44 failed / 4 errors (806 collected) — measured at the start of
   the cloud-integration task, BEFORE the 120 cloud tests were added. Its 48
   not-passed node IDs were captured in that run and are transcribed below.
B  922 passed / 4 failed / 0 errors (926 collected) — measured after the
   environment repair, with the same 926 tests collected as at the end of A's
   task.
C  measured now from ``_snapC.xml``.

Usage
-----
    python -m pytest tests/ -q --tb=no --junitxml=_snapC.xml
    python scripts/audit_test_reconciliation.py --junit _snapC.xml
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

# --------------------------------------------------------------------------- #
# Recorded snapshots (not fabricated: transcribed from the runs that produced
# the quoted totals)
# --------------------------------------------------------------------------- #

SNAPSHOT_A = {
    "label": "A — pre-cloud-tests baseline (before the 120 cloud tests existed)",
    "collected": 806,
    "passed": 758,
    "failed": 44,
    "errors": 4,
    "skipped": 0,
    "not_passed": [
        # --- parity: acronym (7) ------------------------------------------- #
        "tests/parity/test_acronym_parity.py::TestSystemPromptParity::test_system_prompt_matches_after_documented_apostrophes",
        "tests/parity/test_acronym_parity.py::TestAttackerMessageParity::test_round1_messages_identical",
        "tests/parity/test_acronym_parity.py::TestAttackerMessageParity::test_round2_messages_identical",
        "tests/parity/test_acronym_parity.py::TestRefusalDisclaimerParity::test_refusal_only_triggers_backtrack",
        "tests/parity/test_acronym_parity.py::TestRefusalDisclaimerParity::test_disclaimer_only_triggers_backtrack",
        "tests/parity/test_acronym_parity.py::TestRefusalDisclaimerParity::test_neither_triggers_acceptance",
        "tests/parity/test_acronym_parity.py::TestRefusalDisclaimerParity::test_disclaimer_decision_matches",
        # --- parity: actor attack (6) -------------------------------------- #
        "tests/parity/test_actor_attack_parity.py::TestPreAttackParity::test_call_sequence_and_flags_match",
        "tests/parity/test_actor_attack_parity.py::TestPreAttackParity::test_attacker_prompt_contents_match_after_documented_spaces",
        "tests/parity/test_actor_attack_parity.py::TestStepJudgeParity::test_judge_truth_table_matches",
        "tests/parity/test_actor_attack_parity.py::TestStepJudgeParity::test_judge_prompt_identical",
        "tests/parity/test_actor_attack_parity.py::TestCallMultiSemantics::test_retry_budget_and_rewrite_match",
        "tests/parity/test_actor_attack_parity.py::TestNBFParity::test_trial_count_matches",
        # --- parity: crescendo (6) ----------------------------------------- #
        "tests/parity/test_crescendo_parity.py::TestSystemPromptParity::test_system_prompt_identical_after_format",
        "tests/parity/test_crescendo_parity.py::TestAttackerMessageParity::test_round1_messages_identical",
        "tests/parity/test_crescendo_parity.py::TestAttackerMessageParity::test_round2_messages_identical",
        "tests/parity/test_crescendo_parity.py::TestAttackerMessageParity::test_round3_messages_identical_with_refused_score",
        "tests/parity/test_crescendo_parity.py::TestAttackerMessageParity::test_temperature_and_json_mode_match",
        "tests/parity/test_crescendo_parity.py::TestRubricParity::test_rubric_decision_and_prompt_match_after_documented_typos",
        # --- parity: NBF (8) ----------------------------------------------- #
        "tests/parity/test_nbf_parity.py::TestSingleStepNumericalParity::test_ssm_state_transition_matches",
        "tests/parity/test_nbf_parity.py::TestSingleStepNumericalParity::test_nbf_logits_match",
        "tests/parity/test_nbf_parity.py::TestSingleStepNumericalParity::test_softmax_and_score_match",
        "tests/parity/test_nbf_parity.py::TestSingleStepNumericalParity::test_verify_conversion_passes_with_official_weights",
        "tests/parity/test_nbf_parity.py::TestDialogRollParity::test_dialog_score_matches_official",
        "tests/parity/test_nbf_parity.py::TestDialogRollParity::test_calculate_score_indexed_matches_official",
        "tests/parity/test_nbf_parity.py::TestStateUpdateSemantics::test_official_and_guardbound_have_identical_state_semantics",
        "tests/parity/test_nbf_parity.py::TestStateUpdateSemantics::test_scoring_does_not_advance_the_persisted_target_conversation",
        # --- parity: opposite day (6) -------------------------------------- #
        "tests/parity/test_opposite_day_parity.py::TestSystemPromptParity::test_system_prompt_matches_after_documented_whitespace",
        "tests/parity/test_opposite_day_parity.py::TestAttackerMessageParity::test_round1_messages_identical",
        "tests/parity/test_opposite_day_parity.py::TestAttackerMessageParity::test_round2_messages_identical",
        "tests/parity/test_opposite_day_parity.py::TestAttackerMessageParity::test_json_schema_roundtrip",
        "tests/parity/test_opposite_day_parity.py::TestRefusalParity::test_llm_refusal_decision_matches",
        "tests/parity/test_opposite_day_parity.py::TestRefusalParity::test_metadata_threshold_boundary_matches",
        # --- parity: refusal threshold (3) --------------------------------- #
        "tests/parity/test_refusal_threshold_parity.py::TestRefusalTruthTable::test_acronym_combined_decision_matches",
        "tests/parity/test_refusal_threshold_parity.py::TestRefusalTruthTable::test_plain_refusal_decision_matches",
        "tests/parity/test_refusal_threshold_parity.py::TestRefusalTruthTable::test_metadata_90_boundary_is_not_refusal",
        # --- integration (4) ----------------------------------------------- #
        "tests/test_crescendo_integration.py::TestNBFParity::test_original_vs_guardbound_architecture_equivalence",
        "tests/test_crescendo_integration.py::TestCheckpointCompat::test_original_vs_guardbound_architecture_equivalence",
        "tests/test_crescendo_integration.py::TestCheckpointCompat::test_nbf_architecture_matching",
        "tests/test_crescendo_integration.py::TestCheckpointCompat::test_ssm_architecture_matching",
        # --- pre-existing repo/environment state (4) ----------------------- #
        "tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring",
        "tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured",
        "tests/test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid",
        "tests/test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present",
        # --- collection ERRORS (4) ----------------------------------------- #
        "tests/test_phase17_stage2_infrastructure.py::TestPipelineOwnership::test_fresh_load_is_owned_by_the_cache_not_the_client",
        "tests/test_phase17_stage2_infrastructure.py::TestPipelineOwnership::test_client_reloads_after_release",
        "tests/test_phase17_stage2_infrastructure.py::TestResidencyCycles::test_five_cycles_keep_at_most_one_model_resident",
        "tests/test_phase17_stage2_infrastructure.py::TestResidencyCycles::test_switching_evicts_the_previous_role",
    ],
}

SNAPSHOT_B = {
    "label": "B — after the environment repair (same 926 collected as task 1 end)",
    "collected": 926,
    "passed": 922,
    "failed": 4,
    "errors": 0,
    "skipped": 0,
    "not_passed": [
        "tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring",
        "tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured",
        "tests/test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid",
        "tests/test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present",
    ],
}


def parse_junit(path: Path) -> dict:
    """Exact per-test outcomes from a measured run."""
    tree = ET.parse(path)
    root = tree.getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root

    rows, not_passed = [], []
    for case in suite.iter("testcase"):
        classname = case.get("classname", "")
        name = case.get("name", "")
        parts = classname.split(".") if classname else []
        # pytest classnames look like "tests.test_mod.ClassName" (or just
        # "tests.test_mod" for module-level tests).
        if len(parts) >= 3:
            node = f"{'/'.join(parts[:-2])}/{parts[-2]}.py::{parts[-1]}::{name}"
        elif len(parts) == 2:
            node = f"{'/'.join(parts[:1])}/{parts[1]}.py::{name}"
        else:
            node = f"{classname}::{name}"
        outcome = "passed"
        for tag, mapped in (("failure", "failed"), ("error", "error"),
                            ("skipped", "skipped")):
            if case.find(tag) is not None:
                outcome = mapped
                break
        rows.append({"node": node, "outcome": outcome})
        if outcome != "passed":
            not_passed.append(node)

    counts = {}
    for row in rows:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    return {
        "label": "C — current tree, measured now",
        "collected": len(rows),
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0),
        "errors": counts.get("error", 0),
        "skipped": counts.get("skipped", 0),
        "seconds": float(suite.get("time", 0.0)),
        "not_passed": sorted(not_passed),
    }


def compare(a: dict, b: dict) -> dict:
    a_set, b_set = set(a["not_passed"]), set(b["not_passed"])
    return {
        "collected_a": a["collected"],
        "collected_b": b["collected"],
        "collected_delta": b["collected"] - a["collected"],
        "not_passed_a": len(a_set),
        "not_passed_b": len(b_set),
        "recovered": len(a_set - b_set),
        "newly_failing": len(b_set - a_set),
        "remaining": len(a_set & b_set),
        "b_not_passed_is_subset_of_a": b_set <= a_set,
        "identical_not_passed_sets": a_set == b_set,
        "full_collected_set_comparable": False,
        "recovered_ids": sorted(a_set - b_set),
        "newly_failing_ids": sorted(b_set - a_set),
        "remaining_ids": sorted(a_set & b_set),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", default="_snapC.xml")
    parser.add_argument("--out", default="results/audit/test_reconciliation.json")
    args = parser.parse_args()

    snap_c = parse_junit(Path(args.junit))

    matrix = {
        "A vs B": compare(SNAPSHOT_A, SNAPSHOT_B),
        "B vs C": compare(SNAPSHOT_B, snap_c),
        "A vs C": compare(SNAPSHOT_A, snap_c),
    }

    payload = {"snapshots": {"A": SNAPSHOT_A, "B": SNAPSHOT_B, "C": snap_c},
               "matrix": matrix}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Snapshot summary")
    for key, snap in (("A", SNAPSHOT_A), ("B", SNAPSHOT_B), ("C", snap_c)):
        print(f"  {key}: collected={snap['collected']:>4}  passed={snap['passed']:>4}  "
              f"failed={snap['failed']:>3}  errors={snap['errors']:>2}  "
              f"skipped={snap['skipped']:>2}  not_passed={len(snap['not_passed'])}")
    print("\nComparison matrix")
    for name, row in matrix.items():
        print(f"  {name}: recovered={row['recovered']:>3}  "
              f"newly_failing={row['newly_failing']:>2}  remaining={row['remaining']:>2}  "
              f"collected_delta={row['collected_delta']:+5}  "
              f"identical_not_passed={row['identical_not_passed_sets']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
