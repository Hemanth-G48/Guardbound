"""Phase 29 — experimental-methodology alignment (offline, deterministic).

Covers: the dual ASR denominators, incomplete-run framing, the baseline/NBF
variant invariants, the retained (documented) retry policy, the single official
ASR path, the enriched manifest, and the utility-evaluation readiness inventory.

No provider is contacted: synthetic rows, the offline mock transport, and
``MockChatLLM`` only.
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


def _rows(successes: int, failures: int = 0, invalid: int = 0, attack="crescendo_paper"):
    rows = []
    for _ in range(successes):
        rows.append({"attack": attack, "failed": False,
                     "valid_for_asr": True, "success": True})
    for _ in range(failures):
        rows.append({"attack": attack, "failed": True,
                     "valid_for_asr": False, "success": False})
    for _ in range(invalid):
        rows.append({"attack": attack, "failed": False,
                     "valid_for_asr": False, "success": False})
    return rows


def _remaining_valid(total: int, successes: int, failures: int = 0, invalid: int = 0):
    """Fill the plan to exactly ``total`` goals with valid-but-unsuccessful rows."""
    rows = _rows(successes, failures, invalid)
    for _ in range(max(0, total - successes - failures - invalid)):
        rows.append({"attack": "crescendo_paper", "failed": False,
                     "valid_for_asr": True, "success": False})
    return rows


# --------------------------------------------------------------------------- #
# 1 — dual ASR denominators
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("failures", [0, 1, 5, 10, 20])
def test_dual_asr_with_infrastructure_failures(failures):
    total = 200
    successes = 52
    rows = _remaining_valid(total, successes, failures=failures)
    report = common.aggregate_asr_report(rows, {"crescendo_paper": total})
    overall = report["overall"]

    assert overall["expected_goals"] == total
    assert overall["successful_goals"] == successes
    assert overall["infrastructure_failure_count"] == failures
    assert overall["valid_completed_goals"] == total - failures
    # Paper-literal: failures stay in the denominator.
    assert overall["asr_paper"] == pytest.approx(successes / total)
    # Diagnostic: failures leave it.
    assert overall["asr_valid_completed"] == pytest.approx(
        successes / (total - failures))
    if failures:
        assert overall["asr_valid_completed"] > overall["asr_paper"]
    else:
        assert overall["asr_valid_completed"] == overall["asr_paper"]


def test_inflations_match_the_documented_factors():
    """f=10 -> +5.3 %, f=20 -> +11.1 % relative to the paper denominator."""
    for failures, factor in ((10, 200 / 190), (20, 200 / 180)):
        overall = common.aggregate_asr_report(
            _remaining_valid(200, 52, failures=failures),
            {"crescendo_paper": 200})["overall"]
        assert overall["asr_valid_completed"] / overall["asr_paper"] == pytest.approx(factor)


def test_invalid_evaluator_is_never_a_success_in_either_denominator():
    report = common.aggregate_asr_report(
        _remaining_valid(200, 40, invalid=10), {"crescendo_paper": 200})["overall"]
    assert report["invalid_evaluator_count"] == 10
    assert report["successful_goals"] == 40
    assert report["asr_paper"] == pytest.approx(40 / 200)
    assert report["asr_valid_completed"] == pytest.approx(40 / 190)
    # An invalid row can never carry a success flag into the numerator.
    rows = [{"attack": "crescendo_paper", "failed": False,
             "valid_for_asr": False, "success": True}]
    report = common.aggregate_asr_report(rows, {"crescendo_paper": 1})["overall"]
    assert report["successful_goals"] == 0
    assert report["asr_paper"] == 0.0


def test_zero_denominators_are_undefined_not_zero():
    report = common.aggregate_asr_report([], {"crescendo_paper": 0})["overall"]
    assert report["asr_paper"] is None
    assert report["asr_valid_completed"] is None

    all_failed = _rows(0, failures=5)
    report = common.aggregate_asr_report(
        all_failed, {"crescendo_paper": 5})["overall"]
    assert report["asr_paper"] == 0.0           # denominator exists
    assert report["asr_valid_completed"] is None  # no valid goal at all


def test_per_attack_breakdown_sums_to_the_overall_block():
    rows = _rows(2, 1, 0, attack="actor_attack") + _rows(3, 2, 1, attack="crescendo_paper")
    report = common.aggregate_asr_report(
        rows, {"actor_attack": 5, "crescendo_paper": 8, "opposite_day": 4})
    by = report["by_attack"]
    assert by["opposite_day"]["expected_goals"] == 4
    assert by["opposite_day"]["successful_goals"] == 0
    for key in ("expected_goals", "completed_goals", "valid_completed_goals",
                "successful_goals", "infrastructure_failure_count",
                "invalid_evaluator_count"):
        assert report["overall"][key] == sum(item[key] for item in by.values())
    assert by["actor_attack"]["asr_paper"] == pytest.approx(0.4)


def test_asr_row_for_an_unplanned_attack_is_rejected():
    with pytest.raises(ValueError, match="not in the plan"):
        common.aggregate_asr_report(_rows(1, attack="red_queen"),
                                    {"crescendo_paper": 200})


# --------------------------------------------------------------------------- #
# 2 — baseline / NBF variants
# --------------------------------------------------------------------------- #

def test_r1_config_declares_both_arms():
    cfg = yaml.safe_load(R1_CONFIG.read_text(encoding="utf-8"))
    assert set(cfg["variants"]) == {"r1_baseline", "r1_paper"}
    assert cfg["variants"]["r1_baseline"]["nbf"] == {"enabled": False}
    assert cfg["variants"]["r1_paper"]["nbf"] == {"enabled": True}


def test_both_variants_resolve_and_the_invariant_check_runs():
    cfg = yaml.safe_load(R1_CONFIG.read_text(encoding="utf-8"))
    names = []
    for name in ("r1_baseline", "r1_paper"):
        resolved, block = common.resolve_variant(cfg, name)
        assert resolved == name
        names.append(resolved)
    # With two variants the guard is live and reports what it checked.
    checked = common.assert_invariants(cfg)
    assert checked, "the invariant check must report the invariants it verified"
    assert checked == list(cfg["invariants"])


def test_variants_differ_only_in_nbf_enablement():
    cfg = yaml.safe_load(R1_CONFIG.read_text(encoding="utf-8"))
    baseline = common.nbf_settings(cfg, cfg["variants"]["r1_baseline"])
    paper = common.nbf_settings(cfg, cfg["variants"]["r1_paper"])
    assert baseline["enabled"] is False
    assert paper["enabled"] is True
    differing = {key for key in set(baseline) | set(paper)
                 if baseline.get(key) != paper.get(key)}
    assert differing == {"enabled"}
    # Everything shared stays identical because it lives outside `variants`.
    for key in ("models", "temperature", "dataset", "attacks", "prompt_variant"):
        assert key not in cfg["variants"]["r1_baseline"]
        assert key not in cfg["variants"]["r1_paper"]


def test_a_variant_that_overrides_methodology_is_rejected(monkeypatch):
    cfg = yaml.safe_load(R1_CONFIG.read_text(encoding="utf-8"))
    cfg["variants"]["r1_baseline"]["temperature"] = {"attacker": 0.0}
    with pytest.raises(ValueError, match="differ only in NBF"):
        common.assert_invariants(cfg)


def test_both_arms_share_dataset_and_prompt_hashes():
    cfg = yaml.safe_load(R1_CONFIG.read_text(encoding="utf-8"))
    hashes = common.prompt_provenance()
    official = {name: block["variants"]["official"] for name, block in hashes.items()}
    # The prompts are configuration-level, not variant-level: one set for both arms.
    assert all(official.values())
    assert "official" in hashes["rubric"]["variants"]
    # The dataset is likewise one file for both arms.
    dataset = common.dataset_provenance(cfg, [])
    assert dataset["sha256"] == hashlib.sha256(
        (PROJECT_ROOT / cfg["dataset"]["path"]).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# 3 — retry policy (inspected, retained, documented)
# --------------------------------------------------------------------------- #

def test_documented_official_retry_semantics():
    # Phase 31 re-inspected the reference source: `time.sleep(2*ind_+1)` is the
    # last statement of the `except` handler, so the loop can wait ten times
    # (1..19 s = 100 s) and the final 19 s wait happens *after* the tenth failure.
    # Phase 30's record claimed nine waits in every path; that is corrected here.
    policy = common.PAPER_CODE_RETRY_POLICY
    official = policy["official_implementation"]
    assert official["attempts"] == 10
    assert official["backoff_seconds"] == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    assert official["success_path_backoff_seconds"] == [1, 3, 5, 7, 9, 11, 13, 15, 17]
    assert official["success_path_total_backoff_seconds"] == 81
    assert official["exhaustion_path_backoff_seconds"] == [
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    assert official["exhaustion_path_total_backoff_seconds"] == 100
    assert official["sleeps_after_final_failed_attempt"] is True
    assert "any exception" in official["retried_errors"]
    assert official["json_parse_failures_retried"] is True
    assert "UnboundLocalError" in official["first_attempt_exception"]
    assert "KeyError" in official["after_final_attempt"]
    assert policy["applied_guardbound"]["attempts"] == 10
    assert policy["applied_guardbound"]["backoff_seconds"] == [
        1, 3, 5, 7, 9, 11, 13, 15, 17]
    assert policy["applied_guardbound"]["sleeps_after_final_failed_attempt"] is False
    assert policy["applied_guardbound"]["json_parse_failures_retried"] is False
    assert policy["status"].startswith("ALIGNED")
    assert policy["status"] != "ALIGNED"


def test_sdk_client_is_constructed_with_zero_hidden_retries(monkeypatch):
    import types

    captured: dict = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-not-a-credential")

    import guardbound.llm.openai_client as oc

    client = oc.OpenAIChatLLM(model_id="gpt-4o-2024-08-06")
    client._get_client()
    assert captured["max_retries"] == 0


class _CountingCompletions:
    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("transient provider failure")

        class _Msg:
            content = "SAFE"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _FakeClient:
    def __init__(self, fail_times: int):
        self.chat = type("C", (), {"completions": _CountingCompletions(fail_times)})()


def _llm_with(fail_times: int, **kwargs):
    import guardbound.llm.openai_client as oc

    llm = oc.OpenAIChatLLM(
        model_id="gpt-4o-2024-08-06", client=_FakeClient(fail_times), **kwargs)
    return llm, llm._client.chat.completions


def test_transient_failure_makes_exactly_one_attempt():
    # Phase 30: the single-attempt policy is now an explicit override, not the
    # default (the default is the author's 10-attempt schedule).
    llm, completions = _llm_with(fail_times=1, retry_attempts=1)
    with pytest.raises(Exception):
        llm.generate([{"role": "user", "content": "hi"}])
    assert completions.calls == 1, "retry_attempts=1 means one attempt"
    assert completions.fail_times == 0


def test_successful_call_is_one_attempt_with_no_retry_path():
    llm, completions = _llm_with(fail_times=0)
    assert llm.generate([{"role": "user", "content": "hi"}]) == "SAFE"
    assert completions.calls == 1


def test_exhausted_attempts_never_become_a_verdict():
    """A single exhausted attempt is a failed goal, not a score."""
    llm, completions = _llm_with(fail_times=1, retry_attempts=1)
    with pytest.raises(Exception):
        llm.generate([{"role": "user", "content": "hi"}], json_format=True)
    assert completions.calls == 1
    # The failure reaches aggregation as an infrastructure failure and is counted
    # as not successful in the paper denominator.
    overall = common.aggregate_asr_report(
        [{"attack": "crescendo_paper", "failed": True,
          "valid_for_asr": False, "success": False}],
        {"crescendo_paper": 200})["overall"]
    assert overall["successful_goals"] == 0
    assert overall["infrastructure_failure_count"] == 1
    assert overall["asr_paper"] == 0.0
    assert overall["asr_valid_completed"] is None


# --------------------------------------------------------------------------- #
# 4 — one official ASR path
# --------------------------------------------------------------------------- #

def test_r1_declares_the_attack_time_rubric():
    from guardbound.paper_protocol import PAPER_ASR_SOURCE, resolve_paper_reproduction

    cfg = yaml.safe_load(R1_CONFIG.read_text(encoding="utf-8"))
    assert cfg["evaluation"]["asr_source"] == PAPER_ASR_SOURCE
    resolved = resolve_paper_reproduction(cfg)
    assert resolved["asr_source"] == PAPER_ASR_SOURCE
    assert "Qi et al." in resolved["asr_source_provenance"]


def test_paper_config_rejects_the_secondary_judge():
    from guardbound.paper_protocol import (
        PaperProtocolError,
        resolve_paper_reproduction,
    )

    cfg = yaml.safe_load(R1_CONFIG.read_text(encoding="utf-8"))
    cfg["evaluation"]["asr_source"] = "secondary_judge"
    with pytest.raises(PaperProtocolError, match="asr_source"):
        resolve_paper_reproduction(cfg)


def test_rubric_prompt_bytes_and_temperature_are_unchanged():
    from guardbound.attacks.prompt_variants import prompt_hashes

    cfg = yaml.safe_load(R1_CONFIG.read_text(encoding="utf-8"))
    assert cfg["temperature"]["evaluator"] == 0.7
    hashes = prompt_hashes()
    # The official rubric differs from the legacy bytes and is still present.
    assert hashes["rubric"]["variants"]["official"] != \
        hashes["rubric"]["variants"]["legacy"]
    assert len(hashes["rubric"]["variants"]["official"]) == 64


def test_judge_asr_marks_its_output_as_a_secondary_judge(tmp_path):
    import judge_asr

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({
        "goal": "g", "attack_method": "crescendo_paper", "target_llm": "m",
        "turns": [{"query": "q", "response": "r"}],
    }) + "\n", encoding="utf-8")
    out = tmp_path / "asr.jsonl"
    sys.argv = ["judge_asr.py", "--corpus", str(corpus), "--out", str(out),
                "--mock", "--limit", "1"]
    judge_asr.main()
    marker = json.loads((tmp_path / "asr.asr_source.json").read_text(encoding="utf-8"))
    assert marker["asr_source"] == "secondary_judge"
    assert marker["paper_comparable"] is False
    assert marker["paper_asr_source"] == "attack_time_qi_rubric"


# --------------------------------------------------------------------------- #
# 5 — manifest
# --------------------------------------------------------------------------- #

def test_pilot_manifest_records_dual_asr_and_linkage(tmp_path):
    import cloud_pilot

    code = cloud_pilot.main([
        "--config", str(R1_CONFIG), "--variant", "r1_baseline",
        "--offline-mock", "--limit", "1", "--out-dir", str(tmp_path),
    ])
    assert code in (0, 1)  # the offline mock may or may not satisfy the probes
    manifest = json.loads(
        (tmp_path / "r1_baseline.manifest.json").read_text(encoding="utf-8"))
    run = manifest["run"]

    # Configuration / dataset / prompts / models / temperatures.
    assert manifest["config"]["sha256"] == hashlib.sha256(
        R1_CONFIG.read_bytes()).hexdigest()
    assert manifest["variant"] == "r1_baseline"
    assert manifest["dataset"]["sha256"]
    assert manifest["prompt_hashes"]["rubric"]["variants"]["official"]
    assert set(manifest["temperature"]) == {"attacker", "target", "evaluator"}
    # Phase 30 adds the paper's NBF terminology to the block.
    assert manifest["nbf"]["enabled"] is False
    assert manifest["nbf"]["eta"] == 0.001
    assert manifest["nbf"]["raw_admission_threshold"] == -0.001

    # ASR provenance + both denominators.
    assert manifest["asr"]["source"] == "attack_time_qi_rubric"
    assert manifest["asr"]["retry_policy"]["applied_guardbound"]["attempts"] == 10
    assert run["expected_goals"] == 3
    assert run["completed_goals"] + run["infrastructure_failure_count"] == 3
    assert run["asr_paper_denominator"] == 3
    assert run["asr_valid_completed_denominator"] == run["valid_completed_goals"]
    assert set(run["asr_by_attack"]) == {
        "actor_attack", "crescendo_paper", "opposite_day"}
    assert run["asr_definitions"]["asr_paper"]
    assert run["asr_definitions"]["asr_valid_completed"]

    # Incomplete scope is labelled, never presented as a 200-goal result.
    assert run["covers_paper_scope"] is False
    assert run["paper_expected_goals"] == 600
    assert run["comparable_to_paper"] is False
    assert run["complete"] is True  # all 3 selected goals ran

    # Baseline/defence linkage is derivable and shared by both arms.
    linkage = manifest["linkage"]
    assert linkage["arms"] == ["r1_baseline", "r1_paper"]
    assert linkage["nbf_enabled"] is False
    assert manifest["dataset"]["sha256"] in linkage["comparison_group"]
    assert "official" in linkage["comparison_group"]

    # Historical fields are preserved.
    assert run["asr_denominator"] == run["valid_completed_goals"]
    assert run["asr_denominator_label"] == "valid_completed (diagnostic)"
    assert "usage_summary" in run


# --------------------------------------------------------------------------- #
# 6 — utility readiness
# --------------------------------------------------------------------------- #

def test_utility_inventory_is_blocked_without_data(tmp_path):
    from guardbound.evaluation.readiness import (
        STATUS_BLOCKED,
        utility_readiness,
    )

    report = utility_readiness(root=tmp_path)
    assert report["status"] == STATUS_BLOCKED
    assert report["table1_complete"] is False
    assert set(report["blocked"]) == {"mtbench", "xstest"}
    for component in report["components"].values():
        assert component["status"] == STATUS_BLOCKED
        assert component["exists"] is False


def test_utility_inventory_ready_with_valid_data(tmp_path):
    from guardbound.evaluation.readiness import STATUS_READY, utility_readiness

    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/mtbench.jsonl").write_text(
        json.dumps({"question": "q", "reference": "a"}) + "\n", encoding="utf-8")
    (tmp_path / "data/raw/xstest.jsonl").write_text(
        json.dumps({"prompt": "p", "type": "safe"}) + "\n", encoding="utf-8")
    report = utility_readiness(root=tmp_path)
    assert report["status"] == STATUS_READY
    assert report["table1_complete"] is True
    assert report["components"]["mtbench"]["rows"] == 1


def test_utility_inventory_reports_a_schema_gap(tmp_path):
    from guardbound.evaluation.readiness import STATUS_BLOCKED, check_utility_dataset

    path = tmp_path / "mtbench.jsonl"
    path.write_text(json.dumps({"reference": "no question"}) + "\n", encoding="utf-8")
    report = check_utility_dataset("mtbench", path)
    assert report["status"] == STATUS_BLOCKED
    assert report["missing_fields"] == ["question"]


def test_utility_margin_defaults_to_the_paper_value(monkeypatch):
    import eval_mtbench
    import eval_over_refusal

    monkeypatch.setattr(sys, "argv", ["eval_mtbench.py", "--dataset", "x", "--out", "y"])
    assert eval_mtbench.parse_args().eta == 1e-3
    monkeypatch.setattr(sys, "argv",
                        ["eval_over_refusal.py", "--dataset", "xstest", "--out", "y"])
    assert eval_over_refusal.parse_args().eta == 1e-3


def test_utility_clis_report_blocked_data_without_fabricating(tmp_path, monkeypatch,
                                                             capsys):
    import eval_mtbench
    import eval_over_refusal

    missing = tmp_path / "absent.jsonl"
    monkeypatch.setattr(sys, "argv", ["eval_mtbench.py", "--dataset", str(missing),
                                      "--out", str(tmp_path / "o.jsonl")])
    with pytest.raises(SystemExit) as exc:
        eval_mtbench.main()
    assert exc.value.code == 2
    assert "BLOCKED — DATA REQUIRED" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", [
        "eval_over_refusal.py", "--dataset", "xstest",
        "--data-path", str(missing), "--out", str(tmp_path / "o2.jsonl")])
    with pytest.raises(SystemExit) as exc:
        eval_over_refusal.main()
    assert exc.value.code == 2
    assert "BLOCKED — DATA REQUIRED" in capsys.readouterr().out
