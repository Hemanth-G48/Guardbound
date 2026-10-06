"""Phase 17 Stage 2 — infrastructure integrity regression tests (F1, F2, F3).

These pin the three defects found in Stage 1 and fixed in Stage 2:

F1  an explicitly supplied ``enable_thinking`` must win over the Qwen3-family
    default, which the constructor previously overwrote silently.
F2  ``release_pipeline()`` alone must release the weights: the client's pipeline
    handle is weak, so ``_pipeline_cache`` is the only strong owner.
F3  a judge reply that cannot be parsed must be recorded as ``JSON_PARSE_ERROR``
    and must never surface as score ``1``, which the rubric reserves for a
    genuine refusal.

Everything here is offline: no Hub access, no model download, no CUDA
allocation. The residency cycle test uses fake pipelines registered through the
real ``ModelManager``, so the ownership rules under test are the production ones.
"""
from __future__ import annotations

import gc
import weakref

import pytest

from guardbound.attacks import rubric_evaluation
from guardbound.attacks.crescendo_paper import (
    CrescendoAttackPaper,
    llm_evaluate_with_rubric,
)
from guardbound.attacks.runner import run_attack
from guardbound.llm import local_client
from guardbound.llm import model_manager as mm
from guardbound.llm.local_client import (
    HFLocalChatLLM,
    _pipeline_cache,
    pipeline_cache_key,
    release_pipeline,
)
from guardbound.llm.mock import MockChatLLM
from guardbound.llm.model_manager import ModelManager

QWEN3 = "fake/qwen3-family-model"
OTHER = "fake/other-family-model"


# --------------------------------------------------------------------------- #
# F1 — explicit enable_thinking must win over the family default
# --------------------------------------------------------------------------- #

@pytest.fixture
def qwen3_family(monkeypatch):
    """Simulate Qwen3-family detection without touching the Hub."""
    monkeypatch.setattr(
        local_client, "_maybe_thinking_kwargs",
        lambda model_id: {"enable_thinking": False},
    )


@pytest.fixture
def other_family(monkeypatch):
    monkeypatch.setattr(local_client, "_maybe_thinking_kwargs", lambda model_id: {})


class TestThinkingModePrecedence:
    def test_default_is_thinking_off_for_qwen3(self, qwen3_family):
        """The frozen Phase 17 judge policy: unset -> thinking OFF."""
        llm = HFLocalChatLLM(QWEN3)
        assert llm.chat_template_kwargs == {"enable_thinking": False}
        assert llm.enable_thinking is False

    def test_explicit_false_is_respected(self, qwen3_family):
        llm = HFLocalChatLLM(QWEN3, chat_template_kwargs={"enable_thinking": False})
        assert llm.chat_template_kwargs["enable_thinking"] is False

    def test_explicit_true_is_respected(self, qwen3_family):
        """The defect: this used to be silently rewritten to False."""
        llm = HFLocalChatLLM(QWEN3, chat_template_kwargs={"enable_thinking": True})
        assert llm.chat_template_kwargs["enable_thinking"] is True
        assert llm.enable_thinking is True

    def test_explicit_value_wins_for_non_qwen_models(self, other_family):
        llm = HFLocalChatLLM(OTHER, chat_template_kwargs={"enable_thinking": True})
        assert llm.enable_thinking is True

    def test_other_template_kwargs_are_preserved(self, qwen3_family):
        llm = HFLocalChatLLM(
            QWEN3, chat_template_kwargs={"enable_thinking": True, "custom": 1}
        )
        assert llm.chat_template_kwargs == {"enable_thinking": True, "custom": 1}

    def test_caller_dict_is_not_mutated(self, qwen3_family):
        supplied = {"enable_thinking": True}
        HFLocalChatLLM(QWEN3, chat_template_kwargs=supplied)
        assert supplied == {"enable_thinking": True}

    def test_build_llm_default_keeps_thinking_off(self, qwen3_family):
        llm = local_client.build_llm({"model_id": QWEN3})
        assert llm.enable_thinking is False

    def test_build_llm_explicit_true_survives(self, qwen3_family):
        llm = local_client.build_llm(
            {"model_id": QWEN3},
            kwargs={"chat_template_kwargs": {"enable_thinking": True}},
        )
        assert llm.enable_thinking is True


class TestQuantizationIsOptIn:
    """Part E: the approved NF4 load is opt-in; the default path is unchanged."""

    def test_default_load_carries_no_quantization(self):
        llm = HFLocalChatLLM(QWEN3, device_map="cpu")
        assert llm.quantization is None
        assert llm._model_kwargs() == {"attn_implementation": "sdpa"}
        assert llm.generation_parameters()["quantization"] is None

    def test_opt_in_nf4_uses_the_approved_config(self):
        llm = HFLocalChatLLM(QWEN3, device_map="cpu", quantization="nf4")
        config = llm._model_kwargs()["quantization_config"]
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True
        params = llm.generation_parameters()
        assert params["quantization"] == "nf4"
        assert params["quantization_method"] == "bitsandbytes NF4 PTQ at load time"

    def test_unsupported_quantization_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported quantization"):
            HFLocalChatLLM(QWEN3, quantization="awq")

    def test_build_llm_forwards_a_declared_quantization(self):
        llm = local_client.build_llm({"model_id": QWEN3, "quantization": "nf4"})
        assert llm.quantization == "nf4"
        assert local_client.build_llm({"model_id": QWEN3}).quantization is None


class TestQuantizedFootprintEstimate:
    """The pre-load gate must estimate the load that will actually happen.

    Measured in Part L: the bf16 estimate refused the qualified 4-bit attacker
    (55.56 GB estimated against a 16.46 GiB real allocation).
    """

    @staticmethod
    def _local_model(tmp_path, mib: float) -> str:
        """A local snapshot directory whose only weight file has the given size.

        Deliberately megabytes, not gigabytes: an earlier version of this test
        created a 20 GiB file, and pytest keeps its last three temporary
        directories, so repeated runs quietly consumed ~90 GiB of disk.
        """
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        weights = model_dir / "model-00001-of-00001.safetensors"
        with weights.open("wb") as handle:
            handle.truncate(int(mib * 1024 ** 2))
        return str(model_dir)

    def test_nf4_estimate_is_a_quarter_of_the_checkpoint_plus_overhead(self, tmp_path):
        model_id = self._local_model(tmp_path, 4.0)
        bf16 = mm.estimate_model_footprint_gb(model_id)
        nf4 = mm.estimate_model_footprint_gb(model_id, "nf4")
        assert bf16 == pytest.approx(4.0 * 1024 ** 2 / 1e9, rel=1e-6)
        assert nf4 == pytest.approx(bf16 / 4.0 + 4.5, rel=1e-6)

    def test_documented_calibration_covers_the_measured_allocation(self):
        """The NF4 estimate must stay above the measured 4-bit allocation.

        Part F recorded the checkpoint at 55,563,006,776 bytes and Part G
        measured the NF4 load at 16.46 GiB. This pins the documented arithmetic
        (weight bytes / 4 + 4.5 GB) against that measurement, so a later edit to
        the constants cannot silently under-estimate a real load. It is a
        calibration record, not an assertion about the function's code path —
        the code path itself is covered by the two tests around it.
        """
        weight_bytes = 55_563_006_776
        estimated_gb = weight_bytes / 4.0 / 1e9 + 4.5
        measured_gib = 16.46
        assert estimated_gb == pytest.approx(18.39, abs=0.01)
        assert estimated_gb > measured_gib * 1.0, "estimate must not under-state the load"

    def test_quantization_reaches_the_load_event(self, monkeypatch, vram_gauges, tmp_path):
        model_id = self._local_model(tmp_path, 4.0)
        manager = ModelManager()
        backend = HFLocalChatLLM(model_id, device_map="cpu", quantization="nf4")
        fake_pipeline = _FakePipeline(model_id)
        backend._get_pipeline = lambda: fake_pipeline
        manager.register("attacker", backend, model_id)

        manager.ensure_resident("attacker")

        load = [e for e in manager.events if e["event"] == "load"][0]
        # The event rounds to 3 decimals, so compare at that precision.
        assert load["estimated_gb"] == pytest.approx(
            mm.estimate_model_footprint_gb(model_id, "nf4"), abs=1e-3
        )


# --------------------------------------------------------------------------- #
# F2 — release_pipeline must actually release the weights
# --------------------------------------------------------------------------- #

class _FakeModel:
    def eval(self):
        return self


class _FakePipeline:
    """Stands in for a transformers text-generation pipeline."""

    def __init__(self, model_id: str = "fake/model"):
        self.model_id = model_id
        self.model = _FakeModel()


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    model_max_length = 4096

    def encode(self, text, add_special_tokens=False):
        return [7]


@pytest.fixture(autouse=True)
def _clean_pipeline_cache():
    _pipeline_cache.clear()
    yield
    _pipeline_cache.clear()

    rubric_evaluation.reset_evaluator_telemetry()


@pytest.fixture
def fake_tokenizer(monkeypatch):
    monkeypatch.setattr(
        HFLocalChatLLM, "_get_tokenizer", lambda self: _FakeTokenizer()
    )


@pytest.fixture
def fake_pipeline_factory(monkeypatch):
    """Replace the ``transformers`` pipeline factory with an offline stand-in.

    ``transformers`` is a lazy module: its top-level ``pipeline`` name caches
    whatever it resolved on first access, while the function-local
    ``from transformers import pipeline`` in ``local_client`` may resolve
    against the ``transformers.pipelines`` submodule. Both must be patched, or
    a later test silently keeps calling an earlier test's factory.
    """
    import transformers
    import transformers.pipelines as pipelines

    created: list[weakref.ref] = []

    def factory(*args, **kwargs):
        model_id = kwargs.get("model") or (args[1] if len(args) > 1 else "")
        pipe = _FakePipeline(str(model_id))
        created.append(weakref.ref(pipe))
        return pipe

    monkeypatch.setattr(pipelines, "pipeline", factory)
    monkeypatch.setattr(transformers, "pipeline", factory)

    # Fail loudly here rather than as a confusing residency assertion later if
    # the import seam ever stops resolving to this factory.
    from transformers import pipeline as resolved

    assert resolved is factory, "the pipeline factory seam is not patched"
    return created


class TestPipelineOwnership:
    def test_cached_pipeline_is_held_weakly(self):
        llm = HFLocalChatLLM("fake/model", device_map="cpu")
        pipe = _FakePipeline()
        _pipeline_cache[pipeline_cache_key("fake/model", "cpu")] = pipe

        assert llm._get_pipeline() is pipe
        assert llm._pipeline is pipe

        assert release_pipeline("fake/model", "cpu") is True
        assert _pipeline_cache == {}

        del pipe
        gc.collect()
        assert llm._pipeline is None, "the client handle must not outlive the cache"

    def test_release_drops_the_last_strong_reference(self):
        """The F2 invariant: nobody else keeps the weights alive."""
        llm = HFLocalChatLLM("fake/model", device_map="cpu")
        pipe = _FakePipeline()
        _pipeline_cache[pipeline_cache_key("fake/model", "cpu")] = pipe
        llm._get_pipeline()

        ref = weakref.ref(pipe)
        del pipe
        gc.collect()
        assert ref() is not None, "cache still owns it"

        release_pipeline("fake/model", "cpu")
        gc.collect()
        assert ref() is None, "release_pipeline alone must free the pipeline"

    def test_fresh_load_is_owned_by_the_cache_not_the_client(
        self, fake_tokenizer, fake_pipeline_factory
    ):
        created = fake_pipeline_factory

        llm = HFLocalChatLLM("fake/model", device_map="cpu")
        pipe = llm._get_pipeline()
        key = pipeline_cache_key("fake/model", "cpu")
        assert _pipeline_cache[key] is pipe

        ref = weakref.ref(pipe)
        del pipe
        gc.collect()
        assert len(created) == 1
        assert ref() is not None, "the cache, not the client, owns the pipeline"

        release_pipeline("fake/model", "cpu")
        gc.collect()
        assert ref() is None
        assert llm._pipeline is None

    def test_client_reloads_after_release(self, fake_tokenizer, fake_pipeline_factory):
        created = fake_pipeline_factory
        llm = HFLocalChatLLM("fake/model", device_map="cpu")
        first = llm._get_pipeline()
        first_ref = weakref.ref(first)
        release_pipeline("fake/model", "cpu")
        del first
        gc.collect()
        assert first_ref() is None, "the released pipeline must be freed"

        second = llm._get_pipeline()
        assert len(created) == 2, "a released model must be loaded again, not reused"
        assert _pipeline_cache[pipeline_cache_key("fake/model", "cpu")] is second
        assert llm._pipeline is second

    def test_non_weakrefable_handle_still_works(self):
        """Test doubles passed straight into ``_pipeline`` must not raise."""
        llm = HFLocalChatLLM("fake/model", device_map="cpu")
        llm._pipeline = {"model_id": "plain-dict"}
        assert llm._pipeline == {"model_id": "plain-dict"}
        llm._pipeline = None
        assert llm._pipeline is None


class _FreeVram:
    def __init__(self, value: float = 100.0):
        self.value = value


@pytest.fixture
def vram_gauges(monkeypatch):
    monkeypatch.setattr(mm, "_free_vram_gb", lambda: 100.0)
    monkeypatch.setattr(mm, "_reserved_vram_gb", lambda: 0.0)


class TestResidencyCycles:
    """ATTACKER -> TARGET -> JUDGE, repeated: one model resident at a time."""

    ROLES = (
        ("attacker", "fake/attacker"),
        ("target", "fake/target"),
        ("judge", "fake/judge"),
    )

    def _manager(self, monkeypatch) -> ModelManager:
        monkeypatch.setattr(
            HFLocalChatLLM, "_get_tokenizer", lambda self: _FakeTokenizer()
        )

        manager = ModelManager()
        for role, model_id in self.ROLES:
            backend = HFLocalChatLLM(model_id, device_map="cpu")
            manager.register(role, backend, model_id)
            manager.set_residency(role, "sequential")
        return manager

    @staticmethod
    def _live_names(refs: list[weakref.ref]) -> list[str]:
        """Names of the still-live pipelines, without retaining any of them.

        Holding the objects would keep an evicted model alive and make the
        residency assertions vacuous — the measurement must not own what it
        measures.
        """
        names: list[str] = []
        for ref in refs:
            obj = ref()
            if obj is not None:
                names.append(obj.model_id)
        return names

    def test_five_cycles_keep_at_most_one_model_resident(
        self, monkeypatch, vram_gauges, fake_pipeline_factory
    ):
        created = fake_pipeline_factory
        manager = self._manager(monkeypatch)

        for cycle in range(5):
            for role, model_id in self.ROLES:
                manager.ensure_resident(role)

                live = self._live_names(created)
                assert len(live) == 1, (
                    f"cycle {cycle}: {len(live)} models resident, expected 1"
                )
                assert live[0] == model_id, (
                    f"cycle {cycle}: stale weights — loaded {live[0]!r} "
                    f"for role {role!r}"
                )
                assert manager.resident_model_ids == [model_id]

            # One model is resident at a time, so the run-boundary release
            # reports exactly the role that is currently loaded.
            released = manager.release_sequential()
            assert released == ["judge"], (
                f"cycle {cycle}: expected only the resident role to be released, "
                f"got {released}"
            )
            assert self._live_names(created) == [], "release must free every model"
            for role, _ in self.ROLES:
                assert manager.get(role)._pipeline is None

        assert len(created) == 15, "each activation reloads exactly once"
        assert self._live_names(created) == []
        assert _pipeline_cache == {}

    def test_switching_evicts_the_previous_role(
        self, monkeypatch, vram_gauges, fake_pipeline_factory
    ):
        created = fake_pipeline_factory
        manager = self._manager(monkeypatch)

        manager.ensure_resident("attacker")
        manager.ensure_resident("target")
        assert self._live_names(created) == ["fake/target"], (
            f"cache={list(_pipeline_cache)} refs={len(created)} "
            f"handles={[b._pipeline for b in manager._backends.values()]} "
            f"events={[e['event'] for e in manager.events]}"
        )
        assert manager.roles == ["attacker", "target", "judge"]
        assert manager.get("attacker")._pipeline is None

        manager.ensure_resident("judge")
        assert self._live_names(created) == ["fake/judge"]

        manager.unload_all()
        assert self._live_names(created) == []
        assert _pipeline_cache == {}


# --------------------------------------------------------------------------- #
# F3 — JSON failure must be distinguishable from a genuine score of 1
# --------------------------------------------------------------------------- #

VALID_OUTPUTS = [
    ('{"score": 1}', 1),
    ('{"score": 2}', 2),
    ('{"score": 3}', 3),
    ('{"score": 4}', 4),
    ('{"score": 5}', 5),
]

FAILING_OUTPUTS = [
    ("malformed JSON", '{"score": 3,'),
    ("empty output", ""),
    ("whitespace only", "   \n  "),
    ("missing score", '{"reason": "no score here"}'),
    ("wrong score type", '{"score": "high"}'),
    ("boolean score", '{"score": true}'),
    ("out-of-range high", '{"score": 9}'),
    ("out-of-range low", '{"score": 0}'),
    ("natural language", "I would give this response a 3 out of 5."),
    ("reasoning only", "<think>Let me weigh this carefully.</think>"),
    ("truncated JSON", '{"score": 3'),
    ("trailing comma", '{"score": 3,}'),
    ("empty object", "{}"),
]


class TestRubricOutputClassification:
    @pytest.mark.parametrize("raw,expected", VALID_OUTPUTS)
    def test_valid_scores_are_preserved(self, raw, expected):
        ev = rubric_evaluation.classify_rubric_output(raw)
        assert ev.valid
        assert ev.score == expected
        assert ev.outcome == rubric_evaluation.VALID
        assert ev.failure_class is None

    @pytest.mark.parametrize("label,raw", FAILING_OUTPUTS, ids=[
        label for label, _ in FAILING_OUTPUTS
    ])
    def test_failures_never_become_a_score(self, label, raw):
        ev = rubric_evaluation.classify_rubric_output(raw)
        assert not ev.valid
        assert ev.score is None, f"{label} must not yield a score"
        assert ev.outcome in rubric_evaluation.ALL_OUTCOMES
        assert ev.outcome != rubric_evaluation.VALID
        assert ev.failure_class == rubric_evaluation.EVALUATOR_FAILURE_CLASS

    def test_specific_failure_outcomes(self):
        cases = {
            '{"score": 3,': rubric_evaluation.JSON_PARSE_ERROR,
            "I cannot score this.": rubric_evaluation.JSON_PARSE_ERROR,
            "": rubric_evaluation.EMPTY_OUTPUT,
            '{"reason": "x"}': rubric_evaluation.MISSING_SCORE,
            '{"score": "high"}': rubric_evaluation.INVALID_SCORE_TYPE,
            '{"score": true}': rubric_evaluation.INVALID_SCORE_TYPE,
            '{"score": 9}': rubric_evaluation.SCORE_OUT_OF_RANGE,
        }
        for raw, expected in cases.items():
            assert rubric_evaluation.classify_rubric_output(raw).outcome == expected

    def test_extra_text_around_valid_json_is_recorded_as_such(self):
        ev = rubric_evaluation.classify_rubric_output(
            'Sure, here is my evaluation:\n{"score": 4}\nHope that helps.'
        )
        assert ev.valid and ev.score == 4
        assert ev.parsed_via_extraction is True
        assert ev.has_extra_text is True

    def test_parsed_dict_from_the_backend_is_accepted(self):
        ev = rubric_evaluation.classify_rubric_output({"score": 5})
        assert ev.valid and ev.score == 5 and ev.parsed_via_extraction is False


class TestRubricTelemetry:
    def test_genuine_score_1_and_json_failure_are_distinguishable(self):
        rubric_evaluation.reset_evaluator_telemetry()

        judge = MockChatLLM(responses=['{"score": 1}', "not json at all"])
        genuine = llm_evaluate_with_rubric("q", "a", "goal", judge)
        failed = llm_evaluate_with_rubric("q", "a", "goal", judge)

        assert genuine == 1, "a valid score of 1 keeps its meaning"
        assert failed is None, "a parse failure must not be reported as 1"

        events = rubric_evaluation.evaluator_events()
        assert [e.outcome for e in events] == [
            rubric_evaluation.VALID, rubric_evaluation.JSON_PARSE_ERROR,
        ]
        assert [e.score for e in events] == [1, None]

        summary = rubric_evaluation.evaluator_telemetry_summary()
        assert summary["n"] == 2
        assert summary["valid"] == 1
        assert summary["genuine_score_1"] == 1
        assert summary["json_parse_errors"] == 1
        assert summary["evaluator_failures"] == 1
        assert summary["validity_rate"] == 0.5

    @pytest.mark.parametrize("raw,expected", VALID_OUTPUTS)
    def test_every_valid_score_round_trips_through_the_attack_api(self, raw, expected):
        judge = MockChatLLM(responses=[raw])
        assert llm_evaluate_with_rubric("q", "a", "goal", judge) == expected

    @pytest.mark.parametrize("label,raw", FAILING_OUTPUTS, ids=[
        label for label, _ in FAILING_OUTPUTS
    ])
    def test_every_failure_returns_no_score(self, label, raw):
        judge = MockChatLLM(responses=[raw])
        assert llm_evaluate_with_rubric("q", "a", "goal", judge) is None


class TestFailureTaxonomySeparation:
    def test_json_failure_is_not_an_attack_failure(self):
        """A malformed judge reply carries its own class, not ATTACK_FAILURE."""
        ev = rubric_evaluation.classify_rubric_output("nonsense")
        assert ev.failure_class == "JSON_PARSE_ERROR"
        assert ev.failure_class != "ATTACK_FAILURE"

    def test_runner_run_with_a_failing_judge_records_no_score(self):
        """End-to-end: the attack records None, not 1, for a failed evaluation."""
        rubric_evaluation.reset_evaluator_telemetry()

        attacker = MockChatLLM(responses=[
            '{"generatedQuestion": "step one please", "lastResponseSummary": ""}',
        ])
        judge = MockChatLLM(responses=[
            "I am unable to produce the required JSON.",   # rubric -> parse error
            '{"value": false, "metadata": 95}',            # refusal check -> not a refusal
        ])
        attack = CrescendoAttackPaper(max_turns=1, attacker_llm=attacker)
        attack.set_evaluator_llm(judge)

        run_attack(
            attack,
            goal="a goal",
            target_llm=MockChatLLM(responses=["a perfectly ordinary answer"]),
            max_turns=1,
        )

        assert attack._scores == [None]
        assert attack.get_last_score() is None
        assert attack.was_successful() is False

        summary = rubric_evaluation.evaluator_telemetry_summary()
        assert summary["json_parse_errors"] == 1
        assert summary["genuine_score_1"] == 0
        assert summary["score_histogram"] == {}
