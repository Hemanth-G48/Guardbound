"""Phase 31 — retry semantics re-verified against the author's source.

Two independent kinds of evidence:

1. **The reference itself.**  ``_author_generate`` extracts the ``generate()``
   function *from the frozen reference file* and executes those exact bytes
   against a scripted fake client.  Nothing is re-implemented, and ``torch`` /
   ``transformers`` are never imported (only the function body is compiled).
2. **Guardbound.**  Scripted transport, injected sleep and an injected clock, so
   every assertion is deterministic and no test waits in real time.

Also covers the manifest record, the ASR denominators and the upstream dataset
pins.  No network, no credentials, no dataset.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
R1_CONFIG = PROJECT_ROOT / "configs" / "paper_reproduction_r1_gpt4o.yaml"
AUTHOR_SOURCE = (
    PROJECT_ROOT / "nbf_original_stuff" / "orginal_code_give_by_author" /
    "NBF-LLM" / "attacks" / "utils" / "generate.py"
)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import _cloud_common as common  # noqa: E402


# --------------------------------------------------------------------------- #
# Scripted transport
# --------------------------------------------------------------------------- #

class _StatusError(Exception):
    """An exception carrying an HTTP status, as the SDK raises it."""

    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status_code = status
        self.status = status


def _response(text: str):
    class _Msg:
        content = text
        refusal = None

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Resp:
        choices = [_Choice()]
        usage = None
        id = "req-test"

    return _Resp()


class _Completions:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if not self.script:
            raise AssertionError("the client made more calls than the test scripted")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return _response(item)


class _Client:
    def __init__(self, script):
        self.chat = types.SimpleNamespace(completions=_Completions(script))


class _Clock:
    """Injected sleep + clock so elapsed and scheduled backoff are separable."""

    def __init__(self, interrupt_on: int | None = None):
        self.now = 0.0
        self.waits: list[float] = []
        self.interrupt_on = interrupt_on

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(float(seconds))
        if self.interrupt_on is not None and len(self.waits) == self.interrupt_on:
            raise KeyboardInterrupt("operator interrupt")
        self.now += float(seconds)


def _llm(script, *, clock=None, **kwargs):
    import guardbound.llm.openai_client as oc

    client = _Client(script)
    clock = clock or _Clock()
    llm = oc.OpenAIChatLLM(
        model_id="gpt-4o-2024-08-06",
        client=client,
        sleep_fn=clock.sleep,
        **kwargs,
    )
    return llm, client.chat.completions, clock


# --------------------------------------------------------------------------- #
# 1 — the reference source, structurally
# --------------------------------------------------------------------------- #

def _author_tree():
    if not AUTHOR_SOURCE.exists():
        pytest.skip("reference author corpus is not present in this checkout")
    source = AUTHOR_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate"
    )
    return source, tree, function


def _openai_branch(function: ast.FunctionDef):
    """Walk the if/elif chain to the final ``else`` (the OpenAI GPT branch)."""
    node = function.body[0]
    while isinstance(node, ast.If):
        if node.orelse and not isinstance(node.orelse[0], ast.If):
            return node.orelse
        if node.orelse:
            node = node.orelse[0]
            continue
        break
    raise AssertionError("the OpenAI branch was not found")


def test_reference_sleep_is_inside_the_except_handler():
    _, _, function = _author_tree()
    branch = _openai_branch(function)
    loop = next(stmt for stmt in branch if isinstance(stmt, ast.For))
    after_loop = [stmt for stmt in branch if not isinstance(stmt, ast.For)]

    assert ast.unparse(loop.iter) == "range(10)"
    assert ast.unparse(loop.target) == "ind_"

    try_node = next(stmt for stmt in loop.body if isinstance(stmt, ast.Try))
    handler = try_node.handlers[0]
    assert ast.unparse(handler.type) == "Exception"

    handler_calls = [
        (stmt.lineno, ast.unparse(stmt.value))
        for stmt in handler.body if isinstance(stmt, ast.Expr)
    ]
    assert handler_calls[-1][1] == "time.sleep(2 * ind_ + 1)"
    assert try_node.orelse == [] and try_node.finalbody == []

    # `return {}` sits after the loop, not inside the handler or the try body.
    assert [ast.unparse(stmt) for stmt in after_loop] == ["return {}"]
    assert after_loop[0].lineno > handler_calls[-1][0]

    # The handler dereferences `response` before sleeping — the reason a
    # first-attempt transport failure aborts instead of retrying.
    assert "response.choices[0].message.content" in handler_calls[-2][1]


# --------------------------------------------------------------------------- #
# 2 — the reference source, executed
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def author_generate():
    """Compile the reference ``generate`` body and return a callable factory."""
    source, _, function = _author_tree()
    segment = ast.get_source_segment(source, function)
    namespace: dict = {"json": json}
    exec(compile(segment, str(AUTHOR_SOURCE), "exec"), namespace)  # noqa: S102
    return namespace["generate"], namespace


def _run_author(author_generate, script, *, json_format=False):
    generate, namespace = author_generate
    clock = _Clock()
    namespace["time"] = types.SimpleNamespace(sleep=clock.sleep)
    client = _Client(script)
    outcome, error = None, None
    try:
        outcome = generate(
            [{"role": "user", "content": "x"}],
            client,
            "gpt-4o-2024-08-06",
            json_format=json_format,
        )
    except BaseException as exc:  # noqa: BLE001 - the reference raises odd things
        error = exc
    return outcome, error, client.chat.completions, clock


def test_reference_succeeds_on_the_first_attempt(author_generate):
    outcome, error, completions, clock = _run_author(author_generate, ["SAFE"])
    assert error is None and outcome == "SAFE"
    assert completions.calls == 1
    assert clock.waits == []


def test_reference_second_attempt_succeeds(author_generate):
    outcome, error, completions, clock = _run_author(
        author_generate, ["not json", '{"ok": true}'], json_format=True)
    assert error is None and outcome == {"ok": True}
    assert completions.calls == 2
    assert clock.waits == [1]
    assert sum(clock.waits) == 1


def test_reference_fifth_attempt_succeeds(author_generate):
    outcome, error, completions, clock = _run_author(
        author_generate, ["bad"] * 4 + ['{"ok": true}'], json_format=True)
    assert error is None and outcome == {"ok": True}
    assert completions.calls == 5
    assert clock.waits == [1, 3, 5, 7]
    assert sum(clock.waits) == 16


def test_reference_tenth_attempt_succeeds_with_nine_waits(author_generate):
    outcome, error, completions, clock = _run_author(
        author_generate, ["bad"] * 9 + ['{"ok": true}'], json_format=True)
    assert error is None and outcome == {"ok": True}
    assert completions.calls == 10
    assert clock.waits == [1, 3, 5, 7, 9, 11, 13, 15, 17]
    assert sum(clock.waits) == 81


def test_reference_exhaustion_waits_after_the_tenth_failure(author_generate):
    outcome, error, completions, clock = _run_author(
        author_generate, ["bad"] * 10, json_format=True)
    assert error is None, "the reference returns {} instead of raising"
    assert outcome == {}
    assert completions.calls == 10
    assert clock.waits == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    assert sum(clock.waits) == 100


def test_reference_never_makes_an_eleventh_call(author_generate):
    _, _, completions, clock = _run_author(
        author_generate, ["bad"] * 10 + ['{"ok": true}'], json_format=True)
    assert completions.calls == 10
    assert len(clock.waits) == 10


def test_reference_does_not_retry_a_first_attempt_transport_error(author_generate):
    outcome, error, completions, clock = _run_author(
        author_generate, [RuntimeError("transport down"), "SAFE"])
    assert outcome is None
    assert isinstance(error, UnboundLocalError)
    assert completions.calls == 1, "the loop never reaches a second attempt"
    assert clock.waits == []


# --------------------------------------------------------------------------- #
# 3 — Guardbound: schedule constants and success paths
# --------------------------------------------------------------------------- #

def test_guardbound_constants_are_the_reference_schedule():
    import guardbound.llm.openai_client as oc

    expected = tuple(2.0 * index + 1.0 for index in range(10))
    assert oc.OFFICIAL_RETRY_BACKOFF_SECONDS == expected
    assert sum(oc.OFFICIAL_RETRY_SUCCESS_BACKOFF_SECONDS) == 81
    assert sum(oc.OFFICIAL_RETRY_EXHAUSTION_BACKOFF_SECONDS) == 100
    assert oc.OFFICIAL_RETRY_TOTAL_BACKOFF_SECONDS == 81
    assert oc.OFFICIAL_RETRY_EXHAUSTION_TOTAL_BACKOFF_SECONDS == 100
    assert oc.APPLIED_RETRY_BACKOFF_SECONDS == expected[:-1]
    assert oc.APPLIED_RETRY_SLEEPS_AFTER_FINAL_FAILURE is False


@pytest.mark.parametrize(
    "failures, expected_waits",
    [
        (0, []),
        (1, [1]),
        (4, [1, 3, 5, 7]),
        (9, [1, 3, 5, 7, 9, 11, 13, 15, 17]),
    ],
)
def test_guardbound_success_on_a_given_attempt(failures, expected_waits):
    script = [RuntimeError("transient")] * failures + ["SAFE"]
    llm, completions, clock = _llm(script)
    assert llm.generate([{"role": "user", "content": "hi"}]) == "SAFE"

    assert completions.calls == failures + 1
    assert clock.waits == expected_waits
    assert sum(clock.waits) == sum(expected_waits)

    summary = llm.retry_summary()
    assert summary["max_attempts"] == 10
    assert summary["actual_attempts"] == failures + 1
    assert summary["retries"] == failures
    assert summary["scheduled_backoff_seconds"] == sum(expected_waits)
    assert summary["successful_calls"] == 1
    assert summary["exhausted_calls"] == 0
    assert summary["non_retryable_calls"] == 0
    assert summary["error_categories"] == (["provider_error"] if failures else [])

    per_call = llm.last_request_stats["retry"]
    assert per_call["actual_attempts"] == failures + 1
    assert per_call["max_attempts"] == 10
    assert per_call["succeeded"] is True
    assert per_call["exhausted"] is False
    assert per_call["terminal_category"] is None
    assert per_call["backoff_schedule"] == expected_waits


def test_guardbound_never_sleeps_after_a_successful_attempt():
    llm, _, clock = _llm(
        [RuntimeError("a"), RuntimeError("b"), RuntimeError("c"), "SAFE"])
    llm.generate([{"role": "user", "content": "hi"}])
    assert clock.waits == [1, 3, 5]
    assert len(clock.waits) == llm.last_request_stats["retry"]["retries"]


# --------------------------------------------------------------------------- #
# 4 — Guardbound: exhaustion policy
# --------------------------------------------------------------------------- #

def test_guardbound_exhaustion_schedules_no_final_wait():
    from guardbound.llm.cloud_errors import RetriesExhausted

    llm, completions, clock = _llm([RuntimeError("down")] * 10 + ["SAFE"])
    with pytest.raises(RetriesExhausted) as excinfo:
        llm.generate([{"role": "user", "content": "hi"}])

    error = excinfo.value
    assert completions.calls == 10, "no eleventh API request"
    assert clock.waits == [1, 3, 5, 7, 9, 11, 13, 15, 17]
    assert sum(clock.waits) == 81, "applied exhaustion backoff"

    # The reference's tenth wait is recorded, not performed.
    assert error.total_scheduled_backoff_seconds == 81
    assert error.to_dict()["exhausted"] is True
    summary = llm.retry_summary()
    assert summary["official_exhaustion_total_backoff_seconds"] == 100
    assert summary["applied_exhaustion_total_backoff_seconds"] == 81
    assert summary["sleeps_after_final_failed_attempt"] is False
    assert summary["exhausted_calls"] == 1

    per_call = llm.last_request_stats["retry"]
    assert per_call["exhausted"] is True
    assert per_call["succeeded"] is False
    assert per_call["actual_attempts"] == 10
    assert per_call["terminal_category"] == "provider_error"
    assert per_call["error_categories"] == ["provider_error"] * 10


def test_guardbound_scheduled_and_elapsed_backoff_are_distinct(monkeypatch):
    import guardbound.llm.openai_client as oc

    clock = _Clock()
    monkeypatch.setattr(oc.time, "perf_counter", clock.perf_counter)
    llm, _, _ = _llm([RuntimeError("x")] * 10, clock=clock)
    with pytest.raises(Exception):
        llm.generate([{"role": "user", "content": "hi"}])

    summary = llm.retry_summary()
    assert summary["scheduled_backoff_seconds"] == 81
    assert summary["elapsed_backoff_seconds"] == pytest.approx(81.0)


def test_guardbound_elapsed_tracks_real_sleep_duration(monkeypatch):
    import guardbound.llm.openai_client as oc

    class _Instant(_Clock):
        def sleep(self, seconds):
            self.waits.append(float(seconds))  # no time passes

    clock = _Instant()
    monkeypatch.setattr(oc.time, "perf_counter", clock.perf_counter)
    llm, _, _ = _llm([RuntimeError("x")] * 3 + ["SAFE"], clock=clock)
    llm.generate([{"role": "user", "content": "hi"}])

    summary = llm.retry_summary()
    assert summary["scheduled_backoff_seconds"] == 9
    assert summary["elapsed_backoff_seconds"] == pytest.approx(0.0)
    assert llm.last_request_stats["retry"]["scheduled_backoff_seconds"] == 9
    assert llm.last_request_stats["retry"]["elapsed_backoff_seconds"] < 9


def test_guardbound_interrupted_sleep_is_recorded_not_hidden(monkeypatch):
    import guardbound.llm.openai_client as oc

    clock = _Clock(interrupt_on=3)
    monkeypatch.setattr(oc.time, "perf_counter", clock.perf_counter)
    llm, completions, _ = _llm([RuntimeError("x")] * 10, clock=clock)
    with pytest.raises(KeyboardInterrupt):
        llm.generate([{"role": "user", "content": "hi"}])

    assert completions.calls == 3, "the interrupted attempt is not repeated"
    summary = llm.retry_summary()
    assert summary["attempts"] == 3
    assert summary["retries"] == 3
    assert summary["scheduled_backoff_seconds"] == 9.0, "1 + 3 + 5 scheduled"
    assert summary["elapsed_backoff_seconds"] == pytest.approx(4.0)
    assert summary["successful_calls"] == 0
    assert summary["exhausted_calls"] == 0


# --------------------------------------------------------------------------- #
# 5 — non-retryable categories and JSON contract
# --------------------------------------------------------------------------- #

_CASES = [
    ("authentication", _StatusError(401)),
    ("permission", _StatusError(403)),
    ("invalid_request", _StatusError(400)),
    ("context_length", ValueError("This model's maximum context length is 8192")),
    ("content_filter", ValueError("content_filter: blocked by the safety system")),
]


@pytest.mark.parametrize("category, error", _CASES)
def test_guardbound_non_retryable_failures_fail_fast(category, error):
    from guardbound.llm.cloud_errors import CloudProviderError, RetriesExhausted

    llm, completions, clock = _llm([error, "SAFE"])
    with pytest.raises(CloudProviderError) as excinfo:
        llm.generate([{"role": "user", "content": "hi"}])

    assert not isinstance(excinfo.value, RetriesExhausted)
    assert excinfo.value.category == category
    assert completions.calls == 1, "no retry for a permanent failure"
    assert clock.waits == []

    summary = llm.retry_summary()
    assert summary["non_retryable_calls"] == 1
    assert summary["retries"] == 0
    assert summary["exhausted_calls"] == 0
    assert summary["actual_attempts"] == 1
    assert summary["error_categories"] == [category]

    per_call = llm.last_request_stats["retry"]
    assert per_call["actual_attempts"] == 1
    assert per_call["exhausted"] is False
    assert per_call["terminal_category"] == category


def test_guardbound_single_attempt_policy_is_not_exhaustion():
    from guardbound.llm.cloud_errors import CloudServerError, RetriesExhausted

    llm, completions, clock = _llm([_StatusError(500)], retry_attempts=1)
    with pytest.raises(CloudServerError) as excinfo:
        llm.generate([{"role": "user", "content": "hi"}])
    assert not isinstance(excinfo.value, RetriesExhausted)
    assert completions.calls == 1
    assert clock.waits == []
    assert llm.retry_summary()["exhausted_calls"] == 0


def test_guardbound_does_not_regenerate_a_malformed_json_reply():
    llm, completions, clock = _llm(["not json", '{"ok": true}'])
    result = llm.generate([{"role": "user", "content": "hi"}], json_format=True)

    assert result == "not json", "the malformed reply is surfaced, not repaired"
    assert completions.calls == 1, "no silent re-sampling at temperature 0.7"
    assert clock.waits == []
    assert llm.last_request_stats["json_parse_ok"] is False
    assert llm.last_request_stats["outcome"] == "json_parse_error"
    assert llm.retry_summary()["retries"] == 0


def test_guardbound_sdk_hidden_retries_remain_disabled(monkeypatch):
    captured: dict = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-not-a-credential")

    import guardbound.llm.openai_client as oc

    oc.OpenAIChatLLM(model_id="gpt-4o-2024-08-06")._get_client()
    assert captured["max_retries"] == 0


# --------------------------------------------------------------------------- #
# 6 — manifest record
# --------------------------------------------------------------------------- #

def test_manifest_policy_record_separates_official_from_applied():
    policy = common.PAPER_CODE_RETRY_POLICY
    official = policy["official_implementation"]
    applied = policy["applied_guardbound"]

    assert official["attempts"] == applied["attempts"] == 10
    assert official["success_path_total_backoff_seconds"] == 81
    assert official["exhaustion_path_total_backoff_seconds"] == 100
    assert official["sleeps_after_final_failed_attempt"] is True
    assert official["json_parse_failures_retried"] is True

    assert applied["total_scheduled_backoff_seconds"] == 81
    assert applied["sleeps_after_final_failed_attempt"] is False
    assert applied["json_parse_failures_retried"] is False
    assert applied["non_retryable_categories"] == [
        "authentication", "permission", "invalid_request",
        "context_length", "content_filter",
    ]

    assert len(policy["differences"]) >= 4
    assert policy["status"] == "ALIGNED WITH DOCUMENTED DEVIATIONS"
    assert "exactly" not in policy["status"].lower()

    # The record is single-sourced from the client's constants.
    import guardbound.llm.openai_client as oc

    assert official["backoff_seconds"] == list(oc.OFFICIAL_RETRY_BACKOFF_SECONDS)
    assert applied["backoff_seconds"] == list(oc.APPLIED_RETRY_BACKOFF_SECONDS)


def test_retry_delta_exposes_actual_attempts_per_goal():
    def snapshot(**overrides):
        base = {
            "calls": 0, "attempts": 0, "retries": 0,
            "scheduled_backoff_seconds": 0.0, "elapsed_backoff_seconds": 0.0,
            "successful_calls": 0, "exhausted_calls": 0,
            "non_retryable_calls": 0, "by_category": {},
        }
        base.update(overrides)
        return base

    before = {"attacker": snapshot()}
    after = {"attacker": snapshot(
        calls=2, attempts=3, retries=2, scheduled_backoff_seconds=4.0,
        elapsed_backoff_seconds=4.0, successful_calls=2,
        by_category={"provider_error": 2},
    )}
    delta = common.retry_delta(before, after)

    assert delta["attacker"]["attempts"] == 3
    assert delta["attacker"]["actual_attempts"] == 3
    assert delta["attacker"]["retries"] == 2
    assert delta["attacker"]["by_category"] == {"provider_error": 2}
    assert delta["total_retries"] == 2

    # A goal that made no calls contributes nothing — no fabricated row.
    assert common.retry_delta(before, before) == {}


def test_manifest_records_upstream_dataset_pins():
    from guardbound.evaluation.readiness import (
        PAPER_VERSION_UNVERIFIED,
        upstream_provenance,
    )

    mtbench = upstream_provenance("mtbench")
    assert mtbench["repository"] == "https://github.com/lm-sys/FastChat"
    assert mtbench["pinned_tag"] == "v0.2.36"
    assert mtbench["pinned_commit"] == (
        "b494d0c6b4e7935f1764f8439e75da3e66beccc7")
    assert mtbench["dataset_version_status"] == PAPER_VERSION_UNVERIFIED
    assert mtbench["sha256"] is None
    assert mtbench["dataset_files"][
        "fastchat/llm_judge/data/mt_bench/question.jsonl"]["git_blob_sha1"] == (
            "464e2c22f7a021a0f9a584d232338a9a17c40066")

    xstest = upstream_provenance("xstest")
    assert xstest["pinned_tag"] is None
    assert xstest["pinned_commit"] == (
        "475f10bf0a3d6a9dfb174b6de1a38afbfdff98a5")
    assert xstest["dataset_files"]["xstest_prompts.csv"]["git_blob_sha1"] == (
        "ad38204d70b3964387f9690169d11afbc4756d01")
    assert xstest["dataset_version_status"] == PAPER_VERSION_UNVERIFIED

    for name in ("mtbench", "xstest"):
        block = upstream_provenance(name)
        assert len(block["pinned_commit"]) == 40
        int(block["pinned_commit"], 16)
        for entry in block["dataset_files"].values():
            assert len(entry["git_blob_sha1"]) == 40
            int(entry["git_blob_sha1"], 16)


def test_utility_block_reports_pins_and_stays_blocked(tmp_path):
    cfg = yaml.safe_load(R1_CONFIG.read_text(encoding="utf-8"))
    block = common.utility_manifest_block(cfg, root=tmp_path)
    assert block["table1_ready"] is False
    for name in ("mtbench", "xstest"):
        component = block["components"][name]
        assert component["status"] == common.UTILITY_BLOCKED_DATA
        assert component["dataset"]["exists"] is False
        assert component["upstream"]["dataset_version_status"] == (
            "PAPER VERSION UNVERIFIED")
        assert component["upstream"]["pinned_commit"]
        assert component["upstream"]["sha256"] is None


# --------------------------------------------------------------------------- #
# 7 — ASR denominators (unchanged, re-proven)
# --------------------------------------------------------------------------- #

def test_asr_worked_example_is_unchanged():
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
    overall = common.aggregate_asr_report(rows, {"actor_attack": 200})["overall"]
    assert overall["expected_goals"] == 200
    assert overall["successful_goals"] == 50
    assert overall["infrastructure_failure_count"] == 10
    assert overall["invalid_evaluator_count"] == 5
    assert overall["valid_completed_goals"] == 185
    assert round(overall["asr_paper"], 4) == 0.2500
    assert round(overall["asr_valid_completed"], 4) == 0.2703


def test_zero_denominators_stay_none():
    overall = common.aggregate_asr_report([], {"crescendo_paper": 0})["overall"]
    assert overall["asr_paper"] is None
    assert overall["asr_valid_completed"] is None


# --------------------------------------------------------------------------- #
# 8 — one failed goal does not abort the run, and rows stay unique
# --------------------------------------------------------------------------- #

def test_failed_goal_does_not_abort_the_run_or_duplicate_rows(
        tmp_path, monkeypatch):
    import cloud_pilot

    seen: list[str] = []

    async def flaky_run_one(**kwargs):
        seen.append(kwargs["attack_name"])
        if len(seen) == 1:
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
    assert code == 1, "the failure is reported without aborting the run"
    assert len(seen) == 3, "every planned goal was attempted"

    rows = [json.loads(line) for line in
            (tmp_path / "r1_baseline.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    assert len(rows) == 3
    assert len({(r["attack"], r.get("goal")) for r in rows}) == 3
    assert sum(1 for r in rows if r.get("failed")) == 1
    for row in rows:
        assert "retry" in row, "every goal-level record carries retry telemetry"

    manifest = json.loads(
        (tmp_path / "r1_baseline.manifest.json").read_text(encoding="utf-8"))
    run = manifest["run"]
    assert run["expected_goals"] == 3
    assert run["asr_paper_denominator"] == 3, "the failure stays in the paper denominator"
    assert run["infrastructure_failure_count"] == 1
    assert manifest["retry_policy"]["status"] == "ALIGNED WITH DOCUMENTED DEVIATIONS"
    assert manifest["retry_policy"]["official_implementation"][
        "exhaustion_path_total_backoff_seconds"] == 100
    for stats in manifest["retry_telemetry"].values():
        assert stats["max_attempts"] == 10
        assert stats["backoff_schedule"] == [1, 3, 5, 7, 9, 11, 13, 15, 17]
        assert "elapsed_backoff_seconds" in stats
        assert "error_categories" in stats
    for name in ("mtbench", "xstest"):
        assert manifest["utility"]["components"][name]["upstream"]["pinned_commit"]
