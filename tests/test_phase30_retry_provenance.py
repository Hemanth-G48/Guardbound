"""Phase 30 — retry alignment, prompt provenance, NBF terminology, utility readiness.

All offline: scripted transports, injected sleep, synthetic fixtures. No provider
request is made and no dataset is read from disk except the repository's own
HarmBench file (read-only) and synthetic JSONL written under ``tmp_path``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
R1_CONFIG = PROJECT_ROOT / "configs" / "paper_reproduction_r1_gpt4o.yaml"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import _cloud_common as common  # noqa: E402


# --------------------------------------------------------------------------- #
# Scripted transport
# --------------------------------------------------------------------------- #

class _Completions:
    def __init__(self, fail_times: int, error: Exception | None = None):
        self.fail_times = fail_times
        self.error = error or RuntimeError("transient provider failure")
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.fail_times:
            self.fail_times -= 1
            raise self.error

        class _Msg:
            content = "SAFE"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _Client:
    def __init__(self, fail_times: int, error: Exception | None = None):
        self.chat = type("C", (), {
            "completions": _Completions(fail_times, error),
        })()


class _StatusError(Exception):
    """Exception carrying an HTTP status, as the SDK raises it."""

    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status_code = status
        self.status = status
        self.body = {"error": {"message": f"HTTP {status}"}}


def _llm(fail_times=0, *, error=None, attempts=None, backoff=None):
    import guardbound.llm.openai_client as oc

    sleeps: list[float] = []

    def fake_sleep(seconds):
        sleeps.append(float(seconds))

    kwargs = {}
    if attempts is not None:
        kwargs["retry_attempts"] = attempts
    if backoff is not None:
        kwargs["retry_backoff"] = backoff
    client = _Client(fail_times, error)
    llm = oc.OpenAIChatLLM(
        model_id="gpt-4o-2024-08-06", client=client, sleep_fn=fake_sleep, **kwargs
    )
    return llm, client.chat.completions, sleeps


# --------------------------------------------------------------------------- #
# 1 — official schedule
# --------------------------------------------------------------------------- #

def test_schedule_matches_the_author_release():
    import guardbound.llm.openai_client as oc

    assert oc.OFFICIAL_RETRY_ATTEMPTS == 10
    # Source-verified (Phase 31): `time.sleep(2*ind_+1)` sits inside the except
    # handler, so ind_ = 0..9 yields ten possible waits — including a 19 s wait
    # *after* the tenth failure.
    assert oc.OFFICIAL_RETRY_BACKOFF_SECONDS == (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
    assert oc.OFFICIAL_RETRY_SUCCESS_BACKOFF_SECONDS == (1, 3, 5, 7, 9, 11, 13, 15, 17)
    assert oc.OFFICIAL_RETRY_EXHAUSTION_BACKOFF_SECONDS == (
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
    assert oc.OFFICIAL_RETRY_TOTAL_BACKOFF_SECONDS == 81
    assert oc.OFFICIAL_RETRY_EXHAUSTION_TOTAL_BACKOFF_SECONDS == 100
    assert oc.APPLIED_RETRY_BACKOFF_SECONDS == oc.OFFICIAL_RETRY_SUCCESS_BACKOFF_SECONDS
    assert oc.APPLIED_RETRY_SLEEPS_AFTER_FINAL_FAILURE is False
    assert oc.RETRY_POLICY_NAME.startswith("official_retry_schedule")


@pytest.mark.parametrize("fail_times", [0, 1, 4, 9])
def test_success_on_attempt_n(fail_times):
    llm, completions, sleeps = _llm(fail_times)
    assert llm.generate([{"role": "user", "content": "hi"}]) == "SAFE"

    assert completions.calls == fail_times + 1
    summary = llm.retry_summary()
    assert summary["calls"] == 1
    assert summary["attempts"] == fail_times + 1
    assert summary["retries"] == fail_times
    assert summary["successful_calls"] == 1
    assert summary["exhausted_calls"] == 0
    assert sleeps == list((1, 3, 5, 7, 9, 11, 13, 15, 17)[:fail_times])
    assert summary["scheduled_backoff_seconds"] == sum(sleeps)


def test_failure_after_all_ten_attempts():
    from guardbound.llm.cloud_errors import RetriesExhausted

    llm, completions, sleeps = _llm(10)
    with pytest.raises(RetriesExhausted) as excinfo:
        llm.generate([{"role": "user", "content": "hi"}])

    error = excinfo.value
    assert completions.calls == 10, "exactly ten attempts"
    assert error.attempts == 10
    assert error.retries == 9
    assert error.total_scheduled_backoff_seconds == 81
    assert error.backoff_seconds == (1, 3, 5, 7, 9, 11, 13, 15, 17)
    assert len(error.error_categories) == 10
    # Applied policy (Phase 31): Guardbound does not wait after the final failure.
    # The reference loop does, and that is recorded separately (100 s vs 81 s).
    assert sleeps == [1, 3, 5, 7, 9, 11, 13, 15, 17], "no sleep after the tenth attempt"
    assert error.to_dict()["exhausted"] is True
    assert llm.retry_summary()["official_exhaustion_total_backoff_seconds"] == 100
    assert llm.last_request_stats["retry"]["exhausted"] is True
    assert llm.last_request_stats["retry"]["succeeded"] is False
    assert llm.last_request_stats["retry"]["actual_attempts"] == 10


def test_no_eleventh_attempt():
    llm, completions, _ = _llm(50)
    with pytest.raises(Exception):
        llm.generate([{"role": "user", "content": "hi"}])
    assert completions.calls == 10


def test_auth_and_permission_failures_are_not_retried():
    from guardbound.llm.cloud_errors import CloudAuthError, CloudPermissionError

    llm, completions, sleeps = _llm(5, error=_StatusError(401))
    with pytest.raises(CloudAuthError):
        llm.generate([{"role": "user", "content": "hi"}])
    assert completions.calls == 1
    assert sleeps == []
    assert llm.retry_summary()["non_retryable_calls"] == 1

    llm, completions, sleeps = _llm(5, error=_StatusError(403))
    with pytest.raises(CloudPermissionError):
        llm.generate([{"role": "user", "content": "hi"}])
    assert completions.calls == 1
    assert sleeps == []


def test_one_usage_row_per_logical_call_even_after_retries():
    from guardbound.llm.usage import UsageLedger

    ledger = UsageLedger()
    import guardbound.llm.openai_client as oc

    client = _Client(2)
    llm = oc.OpenAIChatLLM(
        model_id="gpt-4o-2024-08-06", client=client, ledger=ledger,
        sleep_fn=lambda seconds: None,
    )
    llm.generate([{"role": "user", "content": "hi"}])
    assert client.chat.completions.calls == 3
    assert len(ledger.records()) == 1, "attempts are not separate logical calls"
    assert llm.last_request_stats["retry"]["attempts"] == 3


def test_interrupted_retry_is_represented_accurately():
    import guardbound.llm.openai_client as oc

    calls = {"n": 0}

    def interrupting_sleep(seconds):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("operator interrupt")

    client = _Client(10)
    llm = oc.OpenAIChatLLM(
        model_id="gpt-4o-2024-08-06", client=client,
        sleep_fn=interrupting_sleep,
    )
    with pytest.raises(KeyboardInterrupt):
        llm.generate([{"role": "user", "content": "hi"}])

    summary = llm.retry_summary()
    assert client.chat.completions.calls == 2, "the interrupted attempt is not repeated"
    assert summary["attempts"] == 2
    # Both waits were scheduled; the second was interrupted mid-sleep, so it is
    # counted as a scheduled retry but contributes almost nothing to elapsed time.
    assert summary["retries"] == 2
    assert summary["scheduled_backoff_seconds"] == 4.0  # 1 s then 3 s scheduled
    assert summary["elapsed_backoff_seconds"] < 4.0
    assert summary["successful_calls"] == 0


def test_non_retryable_validation_errors_fail_fast():
    llm, completions, sleeps = _llm(5, error=_StatusError(400))
    with pytest.raises(Exception):
        llm.generate([{"role": "user", "content": "hi"}])
    assert completions.calls == 1
    assert sleeps == []


# --------------------------------------------------------------------------- #
# 2 — ASR denominators (the phase's worked example)
# --------------------------------------------------------------------------- #

def test_phase30_worked_example():
    rows = (
        [{"attack": "actor_attack", "failed": False, "valid_for_asr": True,
          "success": True}] * 50
        + [{"attack": "actor_attack", "failed": True, "valid_for_asr": False,
            "success": False}] * 10
        + [{"attack": "actor_attack", "failed": False, "valid_for_asr": False,
            "success": False}] * 5
        + [{"attack": "actor_attack", "failed": False, "valid_for_asr": True,
            "success": False}] * 135
    )
    overall = common.aggregate_asr_report(
        rows, {"actor_attack": 200})["overall"]
    assert overall["expected_goals"] == 200
    assert overall["successful_goals"] == 50
    assert overall["infrastructure_failure_count"] == 10
    assert overall["invalid_evaluator_count"] == 5
    assert overall["valid_completed_goals"] == 185
    assert round(overall["asr_paper"], 4) == 0.25
    assert round(overall["asr_valid_completed"], 4) == 0.2703


def test_failures_and_invalid_outcomes_stay_separate():
    rows = [{"attack": "crescendo_paper", "failed": True,
             "valid_for_asr": False, "success": False},
            {"attack": "crescendo_paper", "failed": False,
             "valid_for_asr": False, "success": False}]
    overall = common.aggregate_asr_report(
        rows, {"crescendo_paper": 2})["overall"]
    assert overall["infrastructure_failure_count"] == 1
    assert overall["invalid_evaluator_count"] == 1
    assert overall["completed_goals"] == 1
    assert overall["asr_valid_completed"] is None


# --------------------------------------------------------------------------- #
# 3 — ActorAttack prompt provenance
# --------------------------------------------------------------------------- #

def test_actor_prompt_hashes_match_runtime_bytes():
    from guardbound.attacks import actor_attack as aa
    from guardbound.attacks.prompt_variants import actor_attack_prompt_provenance

    block = actor_attack_prompt_provenance("official")
    expected = {
        "extract": aa.EXTRACT_PROMPT,
        "network": aa.NETWORK_PROMPT,
        "actor": aa.ACTOR_PROMPT,
        "more_actor": aa.MORE_ACTOR_PROMPT,
        "queries": aa.QUERIES_PROMPT,
        "json_format_question": aa.JSON_FORMAT_QUESTION_PROMPT,
        "attack_modify": aa.ATTACK_MODIFY_PROMPT,
        "step_judge": aa.ATTACK_STEP_JUDGE_PROMPT,
        "summary_with_details": aa.SUMMARY_WITH_DETAILS_PROMPT,
        "summary_without_details": aa.SUMMARY_WITHOUT_DETAILS_PROMPT,
    }
    for name, text in expected.items():
        entry = block["prompts"][name]
        assert entry["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert entry["chars"] == len(text)
    rendered = aa.ACTOR_PROMPT.format(num_string="3 actors")
    assert block["prompts"]["actor"]["rendered_canonical"]["sha256"] == (
        hashlib.sha256(rendered.encode("utf-8")).hexdigest())
    assert block["prompts"]["actor"]["placeholders"] == ["num_string"]


def test_actor_prompt_reuse_is_recorded_and_hashed():
    from guardbound.attacks.prompt_variants import (
        actor_attack_prompt_provenance,
        prompt_hashes,
    )

    block = actor_attack_prompt_provenance("official")
    rubric = block["prompts"]["rubric"]
    assert rubric["reused_by"] == ["actor_attack", "crescendo_paper", "opposite_day"]
    assert rubric["sha256"] == prompt_hashes()["rubric"]["variants"]["official"]


def test_actor_prompts_are_identical_across_arms():
    from guardbound.attacks.prompt_variants import actor_attack_prompt_provenance

    assert actor_attack_prompt_provenance("official") == \
        actor_attack_prompt_provenance("official")
    assert actor_attack_prompt_provenance("official")["identical_across_arms"] is True

    cfg = yaml.safe_load(R1_CONFIG.read_text(encoding="utf-8"))
    blocks = []
    for name in ("r1_baseline", "r1_paper"):
        _, variant = common.resolve_variant(cfg, name)
        assert "prompt_variant" not in variant
        blocks.append(common.actor_attack_prompt_block(cfg))
    assert blocks[0] == blocks[1]


def test_dynamic_prompts_are_marked_as_such():
    from guardbound.attacks.prompt_variants import actor_attack_prompt_provenance

    dynamic = actor_attack_prompt_provenance("official")["dynamic"]
    assert "queries_chain" in dynamic
    assert "generated by the attacker model" in dynamic["queries_chain"]


# --------------------------------------------------------------------------- #
# 4 — NBF terminology and boundaries
# --------------------------------------------------------------------------- #

def test_nbf_manifest_terms():
    cfg = yaml.safe_load(R1_CONFIG.read_text(encoding="utf-8"))
    block = common.nbf_manifest_block(common.nbf_settings(cfg, {}))
    assert block["eta"] == 0.001
    assert block["raw_admission_threshold"] == -0.001
    assert block["filter_condition"] == "h + eta >= 0"
    assert block["admission_condition"] == "h < -eta"


def test_nbf_boundary_decisions():
    from guardbound.defense.filter_rule import admits, is_admitted, is_filtered

    eta = 0.001
    for epsilon in (1e-9, 1e-6):
        assert admits(-eta, -eta) is False, "h = -eta is filtered (strict <)"
        assert is_filtered(-eta, eta) is True
        assert admits(-eta - epsilon, -eta) is True
        assert is_admitted(-eta - epsilon, eta) is True
        assert admits(-eta + epsilon, -eta) is False
        assert is_filtered(-eta + epsilon, eta) is True


def test_nbf_block_records_both_arms():
    cfg = yaml.safe_load(R1_CONFIG.read_text(encoding="utf-8"))
    baseline = common.nbf_manifest_block(
        common.nbf_settings(cfg, cfg["variants"]["r1_baseline"]))
    paper = common.nbf_manifest_block(
        common.nbf_settings(cfg, cfg["variants"]["r1_paper"]))
    assert baseline["enabled"] is False and paper["enabled"] is True
    assert baseline["eta"] == paper["eta"] == 0.001
    assert baseline["raw_admission_threshold"] == paper["raw_admission_threshold"]


# --------------------------------------------------------------------------- #
# 5 — utility readiness
# --------------------------------------------------------------------------- #

def test_utility_block_is_blocked_and_lists_gaps():
    cfg = yaml.safe_load(R1_CONFIG.read_text(encoding="utf-8"))
    block = common.utility_manifest_block(cfg, root=PROJECT_ROOT)
    assert block["status"] == common.UTILITY_BLOCKED_DATA
    assert block["table1_ready"] is False
    for name in ("mtbench", "xstest"):
        component = block["components"][name]
        assert component["status"] == common.UTILITY_BLOCKED_DATA
        assert component["dataset"]["exists"] is False
        assert component["methodology_gaps"]
        assert component["eta"] == 0.001
        if name == "mtbench":
            assert component["judge_model"].startswith("UNSPECIFIED")
            assert component["dataset"]["schema_status"].startswith("PARTIAL MATCH")
        else:
            assert component["refusal_detector"].startswith("UNSPECIFIED")
            assert component["safe_subset_policy"].startswith("UNSPECIFIED")
    assert any("judge" in gap for gap in block["methodology_gaps"])
    assert any("refusal detector" in gap for gap in block["methodology_gaps"])


def test_a_declaration_cannot_override_missing_data(tmp_path):
    cfg = yaml.safe_load(R1_CONFIG.read_text(encoding="utf-8"))
    cfg["utility"]["mtbench"]["status"] = common.UTILITY_READY
    block = common.utility_manifest_block(cfg, root=tmp_path)
    assert block["components"]["mtbench"]["status"] == common.UTILITY_BLOCKED_DATA


def test_utility_ready_with_synthetic_fixtures(tmp_path):
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/mtbench.jsonl").write_text(
        json.dumps({"question": "q", "reference": "a", "turn2_question": "q2",
                    "category": "writing"}) + "\n", encoding="utf-8")
    (tmp_path / "data/raw/xstest.jsonl").write_text(
        json.dumps({"prompt": "p", "type": "homonym"}) + "\n", encoding="utf-8")
    cfg = yaml.safe_load(R1_CONFIG.read_text(encoding="utf-8"))
    block = common.utility_manifest_block(cfg, root=tmp_path)
    assert block["components"]["mtbench"]["status"] == common.UTILITY_READY
    assert block["table1_ready"] is True
    assert block["components"]["mtbench"]["dataset"]["sha256"]


# --------------------------------------------------------------------------- #
# 6 — both CLI arms still validate
# --------------------------------------------------------------------------- #

def test_both_arms_pass_the_real_cli_validation(capsys):
    import cloud_pilot

    for variant in ("r1_baseline", "r1_paper"):
        code = cloud_pilot.main([
            "--config", str(R1_CONFIG), "--variant", variant,
            "--dry-run", "--limit", "200",
        ])
        assert code == 0
    out = capsys.readouterr().out
    assert out.count("R1 validated") == 2
    assert "NBF enabled        False" in out
    assert "NBF enabled        True" in out


# --------------------------------------------------------------------------- #
# 7 — a failed goal does not stop the run, and records stay unique
# --------------------------------------------------------------------------- #

def test_a_failed_goal_does_not_stop_the_run_and_records_stay_unique(
        tmp_path, monkeypatch):
    import cloud_pilot

    calls = {"n": 0}

    async def flaky_run_one(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated provider outage")
        return {
            "attack": kwargs["attack_name"],
            "goal": kwargs["case"].get("task"),
            "valid_for_asr": True,
            "evaluation_status": "valid",
            "success": False,
            "n_accepted_turns": 1,
            "n_filtered_turns": 0,
            "max_rubric_score": 1,
        }

    monkeypatch.setattr(cloud_pilot, "run_one", flaky_run_one)

    code = cloud_pilot.main([
        "--config", str(R1_CONFIG), "--variant", "r1_baseline",
        "--offline-mock", "--limit", "1", "--out-dir", str(tmp_path),
    ])
    assert code == 1, "the run reports the failure but keeps going"

    rows = [json.loads(line) for line in
            (tmp_path / "r1_baseline.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    assert len(rows) == 3, "one row per planned goal, no duplicates"
    assert len({(r["attack"], r.get("goal")) for r in rows}) == 3
    failed = [r for r in rows if r.get("failed")]
    assert len(failed) == 1
    assert failed[0]["outcome_category"] == "infrastructure_failure"
    assert failed[0]["valid_for_asr"] is False
    assert "retry" in failed[0]
    for row in rows:
        assert "retry" in row, "every goal-level record carries retry telemetry"

    run = json.loads(
        (tmp_path / "r1_baseline.manifest.json").read_text(encoding="utf-8"))
    assert run["run"]["infrastructure_failure_count"] == 1
    assert run["run"]["expected_goals"] == 3
    assert run["run"]["asr_paper_denominator"] == 3
    assert run["run"]["asr_paper"] == 0.0
    assert run["run"]["asr_valid_completed"] == 0.0
    assert run["retry_telemetry"], "per-role retry telemetry is recorded"
    for role, stats in run["retry_telemetry"].items():
        assert stats["policy"].startswith("official_retry_schedule")
        assert stats["max_attempts"] == 10
        assert stats["backoff_schedule"] == [1, 3, 5, 7, 9, 11, 13, 15, 17]
    assert "retry_run" in run
