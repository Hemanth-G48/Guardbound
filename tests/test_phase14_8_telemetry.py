"""Phase 14.8 — telemetry completeness + explicit failure visibility.

Covers the canonical run-record schema introduced in Phase 14.8:
  * a completed run and a failed run produce IDENTICAL key sets
  * a failed run stays unmistakably identifiable as a failure
  * measurements that were never taken are `None`, never 0/[]
  * captured raw attacker output is preserved, and [] means "nothing captured"

Offline only: no model weights are loaded. The failure record is built by the
production `build_failure_record`, not by a replica.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

import phase14_full_reproduction as P14  # noqa: E402

CANON = set(P14.RUN_RECORD_FIELDS)


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class _InnerLLM:
    """Minimal ChatLLM stand-in for CountingChatLLM to wrap."""

    name = "fake-inner"

    def generate(self, messages, temperature=0.7, **kwargs):  # noqa: D102
        return "{}"


class FakeAttack:
    """Stands in for a live attack object at the point of failure."""

    def __init__(self, scores, nbf_scores=(), nbf_verdicts=(), refused=()):
        self._scores = list(scores)
        self.nbf_candidate_scores = list(nbf_scores)
        self.nbf_candidate_verdicts = list(nbf_verdicts)
        self._c_refused = len(refused)

    def get_refusal_count(self) -> int:
        return len(self._history_attacker) + self._c_refused if hasattr(
            self, "_history_attacker") else self._c_refused


class BareAttack:
    """An attack object with none of the optional telemetry attributes."""


def _counting(label="attacker", raw_outputs=()):
    llm = P14.CountingChatLLM(_InnerLLM(), label)
    llm.calls = 3
    llm.purpose_counts = {"generation": 2}
    llm.raw_outputs = list(raw_outputs)
    return llm


def _metadata():
    return {
        "attacker_model": "ATTACKER", "target_model": "TARGET",
        "evaluator_model": "EVALUATOR", "embedding_model": "EMBED",
        "checkpoint_sha256": "ckpt", "dataset_sha256": "data",
        "config_hash": "cfg", "code_commit": "sha", "worktree_sha256": "wt",
        "dtype": "bfloat16", "quantization": "none", "top_p": 1.0,
        "generation_config": {"temperature_attacker": 0.7},
        "filter_trials": {"crescendo": 3},
    }


def _failure_record(**over):
    kwargs = dict(
        run_id="OFF_crescendo_000",
        goal_id=0,
        goal_record={"task": "goal text", "target_system": "sys prompt"},
        condition="off",
        attack_key="crescendo_paper",
        attack_short="crescendo",
        seed=42,
        attack=FakeAttack([3, 1, 3], refused=[None, None]),
        metadata=_metadata(),
        max_turns=8,
        eta=0.0,
        started_at=0.0,
        failure_class="JSON_PARSE_ERROR",
        error_type="AttackGenerationError",
        error_message="Attacker returned a non-JSON response",
        error_text="Attacker returned a non-JSON response: '{not json'",
        attacker_llm=_counting("attacker", [("generation", "str", "{not json")]),
        target_llm=_counting("target"),
        evaluator_llm=_counting("evaluator"),
        model_manager=None,
    )
    kwargs.update(over)
    return P14.build_failure_record(**kwargs)


def _normal_shaped_record():
    """A completed-run record built through the same canonical skeleton.

    Mirrors what run_one_phase14 returns plus the run_batch augmentation, so the
    key-set comparison is against the produced shape rather than a hand-written
    expectation.
    """
    rec = P14.build_run_record(
        run_id="OFF_crescendo_000", goal_id=0, goal="goal text",
        goal_text="goal text", condition="off", attack="crescendo_paper", seed=42,
        attacker_model="ATTACKER", target_model="TARGET",
        evaluator_model="EVALUATOR", embedding_model="EMBED",
        checkpoint_sha256="ckpt", dataset_sha256="data", config_hash="cfg",
        code_commit="sha", worktree_sha256="wt", dtype="bfloat16",
        quantization="none", max_rounds=8, threshold=0.0, filter_trials=3,
        top_p=1.0, generation_config={}, target_system="sys prompt",
        turns=[{"query": "q", "response": "r"}], success=False,
        final_score=3, rubric_scores=[3], refusal_count=0, refused_records=0,
        num_turns=1, termination_reason="max_turns_reached", nbf_enabled=False,
        nbf_scores=[], filtered_queries=0, filter_count=0,
        nbf_stats=P14.nbf_stats_from([], 0), accepted_query_count=1,
    )
    rec.update({
        "runtime": 1.0, "runtime_seconds": 1.0, "failure_class": "ATTACK_FAILURE",
        "error": None, "error_type": None, "error_message": None,
        "peak_vram_gb": 1.0, "residency": {}, "system": {},
        "llm_calls": {"attacker": 1, "target": 1, "evaluator": 1},
        "attacker_purpose_calls": {}, "attacker_parse_failures": {},
        "attacker_raw_outputs": [], "evaluator_purpose_calls": {},
        "evaluator_parse_failures": {}, "target_purpose_calls": {},
    })
    return rec


# --------------------------------------------------------------------------- #
# 1. Structural completeness — key-set equality
# --------------------------------------------------------------------------- #
class TestSchemaCompleteness:
    def test_canonical_field_list_is_unique(self):
        assert len(CANON) == len(P14.RUN_RECORD_FIELDS)

    def test_builder_always_emits_every_field(self):
        assert set(P14.build_run_record()) == CANON

    def test_normal_and_error_key_sets_are_identical(self):
        """The Phase 14.8 contract: structural completeness on both paths."""
        normal = _normal_shaped_record()
        failure = _failure_record()
        assert set(normal.keys()) == set(failure.keys()), (
            f"only-normal={sorted(set(normal) - set(failure))} "
            f"only-error={sorted(set(failure) - set(normal))}"
        )

    def test_failure_record_keys_match_canonical(self):
        assert set(_failure_record().keys()) == CANON

    def test_both_paths_route_through_the_canonical_builder(self):
        """Guard against a path reverting to an ad-hoc dict literal."""
        import inspect
        src = inspect.getsource(P14.build_failure_record)
        assert "build_run_record(" in src
        run_src = inspect.getsource(P14.run_one_phase14)
        assert "return build_run_record(" in run_src
        batch_src = inspect.getsource(P14.run_batch)
        assert "record = build_failure_record(" in batch_src

    def test_serialization_round_trip_preserves_keys_and_nulls(self):
        rec = _failure_record()
        back = json.loads(json.dumps(rec, ensure_ascii=False))
        assert set(back.keys()) == set(rec.keys())
        assert back["num_turns"] is None
        assert back["turns"] is None


# --------------------------------------------------------------------------- #
# 2. Failure visibility — a failed run must stay unmistakable
# --------------------------------------------------------------------------- #
class TestFailureVisibility:
    def test_authoritative_failure_markers_present(self):
        rec = _failure_record()
        assert rec["success"] is False
        assert rec["failure_class"] == "JSON_PARSE_ERROR"
        assert rec["error_type"] == "AttackGenerationError"
        assert rec["error_message"]
        assert rec["error"]

    def test_error_record_is_not_a_legitimate_attack_outcome(self):
        rec = _failure_record()
        assert rec["failure_class"] not in (
            "SUCCESS", "ATTACK_FAILURE", "NBF_SEMANTICS_FAILURE")
        assert rec["success"] is False

    def test_normal_record_carries_error_as_explicit_none(self):
        """Structural symmetry: completed runs set `error` to None, not omit it."""
        assert "error" in _normal_shaped_record()
        assert _normal_shaped_record()["error"] is None


# --------------------------------------------------------------------------- #
# 3. Null semantics — never fabricate a measurement
# --------------------------------------------------------------------------- #
class TestNullSemantics:
    def test_unrecoverable_counts_are_none_not_zero(self):
        rec = _failure_record()
        assert rec["turns"] is None
        assert rec["num_turns"] is None
        assert rec["accepted_query_count"] is None
        # explicitly NOT 0/[] — that would assert "no turns occurred"
        assert rec["num_turns"] != 0
        assert rec["turns"] != []

    def test_final_response_and_termination_reason_are_none(self):
        rec = _failure_record()
        assert rec["final_response"] is None
        assert rec["termination_reason"] is None

    def test_termination_reason_does_not_invent_an_error_enum(self):
        assert _failure_record()["termination_reason"] != "error"


# --------------------------------------------------------------------------- #
# 4. Recovered telemetry — the 17 fields
# --------------------------------------------------------------------------- #
class TestRecoveredTelemetry:
    def test_rubric_telemetry_recovered(self):
        rec = _failure_record(attack=FakeAttack([3, 1, "refused", 4]))
        assert rec["rubric_scores"] == [3, 1, 4]
        assert rec["final_score"] == 4
        assert rec["refused_records"] == 1

    def test_refusal_count_recovered(self):
        rec = _failure_record(attack=FakeAttack([], refused=[None, None]))
        assert rec["refusal_count"] == 2

    def test_configuration_metadata_recovered(self):
        rec = _failure_record()
        assert rec["dtype"] == "bfloat16"
        assert rec["quantization"] == "none"
        assert rec["top_p"] == 1.0
        assert rec["generation_config"] == {"temperature_attacker": 0.7}
        assert rec["filter_trials"] == 3
        assert rec["goal_text"] == "goal text"
        assert rec["target_system"] == "sys prompt"

    def test_model_identity_and_hashes_recovered(self):
        rec = _failure_record()
        for key in ("attacker_model", "target_model", "evaluator_model",
                    "embedding_model", "checkpoint_sha256", "dataset_sha256",
                    "config_hash", "code_commit", "worktree_sha256"):
            assert rec[key], key

    def test_runtime_and_resource_telemetry_present(self):
        rec = _failure_record()
        assert rec["runtime"] is not None
        assert rec["runtime_seconds"] is not None
        assert "residency" in rec
        assert "system" in rec

    def test_bare_attack_object_does_not_crash_recovery(self):
        """An attack exposing none of the optional telemetry still yields a
        complete record, with the NBF/rubric fields genuinely empty."""
        rec = _failure_record(attack=BareAttack())
        assert set(rec.keys()) == CANON
        assert rec["rubric_scores"] == []
        assert rec["nbf_scores"] == []
        assert rec["final_score"] is None


# --------------------------------------------------------------------------- #
# 5. NBF telemetry — presence must reflect actual participation
# --------------------------------------------------------------------------- #
class TestNbfTelemetry:
    def test_off_run_reports_genuinely_empty_nbf(self):
        rec = _failure_record(condition="off",
                              attack=FakeAttack([1], nbf_scores=[], nbf_verdicts=[]))
        assert rec["nbf_enabled"] is False
        assert rec["nbf_scores"] == []
        assert rec["filtered_queries"] == 0
        assert rec["filter_count"] == 0
        assert rec["nbf_stats"]["n"] == 0

    def test_on_run_preserves_scores_recorded_before_the_failure(self):
        rec = _failure_record(
            condition="on",
            attack=FakeAttack([1, 1], nbf_scores=[0.5, -1.0, 0.25],
                              nbf_verdicts=[True, False, True]))
        assert rec["nbf_enabled"] is True
        assert rec["nbf_scores"] == [0.5, -1.0, 0.25]
        assert rec["filtered_queries"] == 1
        assert rec["filter_count"] == 1
        assert rec["nbf_stats"]["n"] == 3
        assert rec["nbf_stats"]["min"] == -1.0
        assert rec["nbf_stats"]["max"] == 0.5

    def test_absent_nbf_is_not_converted_into_a_fake_score(self):
        rec = _failure_record(condition="on", attack=BareAttack())
        assert rec["nbf_scores"] == []


# --------------------------------------------------------------------------- #
# 6. Raw attacker output — diagnostic evidence must survive
# --------------------------------------------------------------------------- #
class TestRawOutputPreservation:
    def test_malformed_output_is_preserved(self):
        rec = _failure_record()
        assert rec["attacker_raw_outputs"], "captured raw output was discarded"
        assert "{not json" in rec["attacker_raw_outputs"][0]["raw"]

    def test_empty_list_means_nothing_was_captured(self):
        rec = _failure_record(attacker_llm=_counting("attacker", []))
        assert rec["attacker_raw_outputs"] == []

    def test_raw_outputs_truncated_to_limit(self):
        many = [("generation", "str", f"out{i}") for i in range(50)]
        rec = _failure_record(attacker_llm=_counting("attacker", many))
        assert len(rec["attacker_raw_outputs"]) == 20


# --------------------------------------------------------------------------- #
# 7. Helper equivalence — no value drift
# --------------------------------------------------------------------------- #
class TestHelperEquivalence:
    @pytest.mark.parametrize("scores", [[], [0.1], [1.0, 2.0, 3.0], [-0.5, 0.0, 0.5]])
    def test_nbf_stats_matches_original_inline_formula(self, scores):
        for filtered in (0, 1, 7):
            expected = (
                {"n": len(scores), "filter_count": filtered,
                 "min": round(min(scores), 4), "max": round(max(scores), 4),
                 "mean": round(sum(scores) / len(scores), 4)}
                if scores else
                {"n": 0, "filter_count": 0, "min": None, "max": None, "mean": None}
            )
            assert P14.nbf_stats_from(scores, filtered) == expected

    def test_rubric_recovery_uses_the_same_source_as_the_normal_path(self):
        rec = _failure_record(attack=FakeAttack([5, "refused", 2]))
        assert rec["rubric_scores"] == [5, 2]
        assert rec["final_score"] == 2


# --------------------------------------------------------------------------- #
# 8. filter_trials — declared-configuration resolution (Phase 14.8.x)
# --------------------------------------------------------------------------- #
# The Phase 14.8.x defect: build_metadata filtered cfg["nbf"]["trials"] by
# `k in ATTACK_KEYS`, comparing config keys (attack-short names) against
# ATTACK_KEYS' KEYS (module names). "crescendo" was therefore dropped before the
# per-run lookup ran, so Crescendo recorded filter_trials == null.
#
# Semantics for this study: filter_trials is the DECLARED frozen configuration
# value (3). It is NOT a claim that three NBF evaluations run per candidate —
# execution is reported by nbf_scores / filtered_queries / filter_count.
CONFIG_CANDIDATES = (
    "configs/reproduction_phase14_frozen.yaml",   # the Phase 14 study's frozen config
    "configs/reproduction_three_model.yaml",      # the script's default config
)


def _metadata_for(config_rel: str) -> dict:
    cfg = P14.load_config(str(_REPO / config_rel))
    return P14.build_metadata(cfg, None, "dataset-sha-placeholder")


@pytest.mark.parametrize("config_rel", CONFIG_CANDIDATES)
class TestFilterTrialsTelemetry:
    def test_resolves_for_every_phase14_attack(self, config_rel):
        ft = _metadata_for(config_rel)["filter_trials"]
        for attack_key in P14.ATTACK_ORDER:
            short = P14.ATTACK_KEYS[attack_key]
            assert ft.get(short) == 3, (
                f"{attack_key} -> attack_short {short!r} resolved to "
                f"{ft.get(short)!r}, expected 3")

    def test_keyed_by_attack_short_names(self, config_rel):
        assert _metadata_for(config_rel)["filter_trials"] == {
            "crescendo": 3, "opposite_day": 3, "acronym": 3}

    def test_actor_attack_remains_absent(self, config_rel):
        assert "actor_attack" not in _metadata_for(config_rel)["filter_trials"]

    def test_values_come_from_the_config_not_a_hardcoded_three(self, config_rel):
        """The value must be read from cfg['nbf']['trials'], not embedded."""
        cfg = P14.load_config(str(_REPO / config_rel))
        expected = {k: v for k, v in cfg["nbf"]["trials"].items()
                    if k in P14.ATTACK_KEYS.values()}
        assert expected == {"crescendo": 3, "opposite_day": 3, "acronym": 3}
        assert _metadata_for(config_rel)["filter_trials"] == expected


class TestNbfTrialsIsDeclarationOnly:
    def test_nbf_trials_is_not_read_by_the_runtime_path(self):
        """Documents why this is telemetry-only: the runner never reads it."""
        import inspect
        from guardbound.attacks import runner
        assert "trials" not in inspect.getsource(runner)


class TestFilterTrialsBothRecordPaths:
    """Step 4: both the completed and the failure record must resolve Crescendo."""

    CONFIG = "configs/reproduction_phase14_frozen.yaml"

    def test_failure_record_resolves_crescendo(self):
        md = _metadata_for(self.CONFIG)
        rec = _failure_record(metadata=md, attack_key="crescendo_paper",
                              attack_short="crescendo")
        assert rec["filter_trials"] == 3

    def test_failure_record_resolves_all_three(self):
        md = _metadata_for(self.CONFIG)
        for attack_key in P14.ATTACK_ORDER:
            rec = _failure_record(metadata=md, attack_key=attack_key,
                                  attack_short=P14.ATTACK_KEYS[attack_key])
            assert rec["filter_trials"] == 3, attack_key

    def test_completed_record_lookup_expression_resolves(self):
        """The completed path uses the identical expression at its own site."""
        md = _metadata_for(self.CONFIG)
        for attack_key in P14.ATTACK_ORDER:
            short = P14.ATTACK_KEYS[attack_key]
            assert (md.get("filter_trials") or {}).get(short) == 3

    def test_both_lookup_sites_use_the_same_expression(self):
        import inspect
        src = inspect.getsource(P14)
        occurrences = src.count(
            'filter_trials=(metadata.get("filter_trials") or {}).get(attack_short)')
        assert occurrences == 2, (
            f"expected the lookup at both record sites, found {occurrences}")


class TestFilterTrialsRegressionGuard:
    """Guard so the `crescendo` vs `crescendo_paper` mismatch cannot return."""

    def test_metadata_predicate_compares_against_attack_keys_values(self):
        import inspect
        src = inspect.getsource(P14.build_metadata)
        assert "ATTACK_KEYS.values()" in src, (
            "build_metadata must filter nbf.trials by ATTACK_KEYS.values()")
        assert "if k in ATTACK_KEYS}" not in src
        assert "if k in ATTACK_KEYS:" not in src

    def test_the_old_predicate_would_drop_crescendo(self):
        """Demonstrates the defect the guard protects against."""
        trials = {"crescendo": 3, "opposite_day": 3, "acronym": 3,
                  "actor_attack": 10}
        old = {k: v for k, v in trials.items() if k in P14.ATTACK_KEYS}
        new = {k: v for k, v in trials.items() if k in P14.ATTACK_KEYS.values()}
        assert "crescendo" not in old, "the old predicate should lose crescendo"
        assert old == {"opposite_day": 3, "acronym": 3}
        assert new == {"crescendo": 3, "opposite_day": 3, "acronym": 3}

    def test_attack_keys_module_names_differ_from_shorts(self):
        """The mismatch is real: module names != attack-short names."""
        assert "crescendo_paper" in P14.ATTACK_KEYS
        assert "crescendo_paper" not in P14.ATTACK_KEYS.values()
        assert "crescendo" in P14.ATTACK_KEYS.values()
        assert "crescendo" not in P14.ATTACK_KEYS


# --------------------------------------------------------------------------- #
# 9. Step 8 — the 55-field schema stays intact with filter_trials added
# --------------------------------------------------------------------------- #
class TestSchemaUnchangedByFilterTrialsPatch:
    REQUIRED_14_8 = (
        "error", "failure_class", "error_type", "error_message", "turns",
        "num_turns", "accepted_query_count", "final_response",
        "termination_reason", "rubric_scores", "final_score", "refusal_count",
        "refused_records", "nbf_scores", "filtered_queries", "filter_count",
        "nbf_stats", "attacker_parse_failures", "target_purpose_calls",
        "attacker_raw_outputs", "filter_trials",
    )

    def test_canonical_field_count_unchanged(self):
        assert len(P14.RUN_RECORD_FIELDS) == 55

    def test_every_named_field_still_present(self):
        missing = [f for f in self.REQUIRED_14_8 if f not in CANON]
        assert not missing, f"schema lost: {missing}"

    def test_failure_record_still_carries_the_full_schema(self):
        rec = _failure_record(metadata=_metadata_for(
            "configs/reproduction_phase14_frozen.yaml"))
        assert set(rec.keys()) == CANON
        assert rec["filter_trials"] == 3
