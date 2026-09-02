"""Phase 7 — evaluation tests.

All tests are offline and use mocks.  No real API keys, network, or
model weights are required.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardbound.evaluation import (
    ASRJudge,
    BinaryMetrics,
    EvaluationResult,
    GUARDRAIL_LOADERS,
    GuardExample,
    JudgeCache,
    MMLUQuestion,
    NBFPromptGuard,
    OVER_REFUSAL_LOADERS,
    PromptGuard,
    build_asr_prompt,
    compute_asr,
    compute_binary_metrics,
    compute_f1,
    evaluate_asr,
    evaluate_guard,
    evaluate_mmlu,
    evaluate_mtbench,
    evaluate_over_refusal,
    is_refusal,
    judge_mtbench_response,
    judge_single_turn,
    load_jbb_benign,
    load_mtbench_jsonl,
    load_phtest_harmless,
    load_xstest,
    make_cache_key,
    read_jsonl,
    to_latex_table,
    to_markdown_table,
    write_jsonl,
)
from guardbound.llm.mock import MockChatLLM
from guardbound.schemas import Conversation, Turn


# --------------------------------------------------------------------------- #
# ASR — synthetic 2/3 success rate, 8-turn = 1 behavior
# --------------------------------------------------------------------------- #

def _conv(goal: str, success: bool) -> Conversation:
    if success:
        return Conversation(
            goal=goal, attack_method="crescendo", target_llm="mock",
            turns=[Turn(query=f"q{i}",
                        response=f"Sure, here is how to {goal} step {i}.")
                   for i in range(8)],
            max_turns=8,
        )
    return Conversation(
        goal=goal, attack_method="crescendo", target_llm="mock",
        turns=[Turn(query=f"q{i}",
                    response="I'm sorry, but I cannot help with that.")
               for i in range(8)],
        max_turns=8,
    )


class _TokenJudgeLLM(MockChatLLM):
    """Mock judge that returns a pre-set verdict token per call."""
    def __init__(self, tokens: list[str]):
        super().__init__(responses=tokens)


def test_asr_two_out_of_three_succeeds():
    convs = [_conv("g1", success=True),
             _conv("g2", success=True),
             _conv("g3", success=False)]
    judge = ASRJudge(
        judge_llm=_TokenJudgeLLM(["JAILBREAK", "JAILBREAK", "SAFE"]),
        prompt_path="configs/judge_prompts/asr_judge.txt",
        cache_dir=tempfile.mkdtemp(),
    )
    verdicts, agg = evaluate_asr(convs, judge)
    assert agg["total_behaviors"] == 3
    assert agg["successful_behaviors"] == 2
    assert agg["asr"] == pytest.approx(2 / 3)


def test_asr_eight_turns_count_as_one_behavior():
    """An 8-turn successful conversation must count as 1, not 8."""
    convs = [_conv("g1", success=True)]
    judge = ASRJudge(
        judge_llm=_TokenJudgeLLM(["JAILBREAK"] * 100),
        prompt_path="configs/judge_prompts/asr_judge.txt",
        cache_dir=tempfile.mkdtemp(),
    )
    verdicts, agg = evaluate_asr(convs, judge)
    assert agg["total_behaviors"] == 1
    assert agg["successful_behaviors"] == 1
    assert agg["asr"] == 1.0
    # The 8 turns are aggregated into ONE behavior, not 8.


def test_asr_stratify_by_attack():
    convs = [
        _conv("g1", True), _conv("g2", True), _conv("g3", False),
        Conversation(goal="g4", attack_method="actor_attack", target_llm="mock",
                     turns=[Turn(query="q", response="unsafe help")] * 4,
                     max_turns=4),
    ]
    judge = ASRJudge(
        judge_llm=_TokenJudgeLLM(["JAILBREAK", "JAILBREAK", "SAFE", "JAILBREAK"]),
        prompt_path="configs/judge_prompts/asr_judge.txt",
        cache_dir=tempfile.mkdtemp(),
    )
    verdicts, agg = evaluate_asr(convs, judge)
    by = agg["by_attack"]
    assert by["crescendo"]["total"] == 3
    assert by["crescendo"]["success"] == 2
    assert by["actor_attack"]["total"] == 1
    assert by["actor_attack"]["success"] == 1


def test_asr_cache_hit_no_double_billing():
    convs = [_conv("g1", True)]
    judge_llm = _TokenJudgeLLM(["JAILBREAK"] * 100)
    judge = ASRJudge(
        judge_llm=judge_llm,
        prompt_path="configs/judge_prompts/asr_judge.txt",
        cache_dir=tempfile.mkdtemp(),
    )
    evaluate_asr([convs[0]], judge)
    n_calls_after_first = len(judge_llm.calls)
    evaluate_asr([convs[0]], judge)
    n_calls_after_second = len(judge_llm.calls)
    assert n_calls_after_second == n_calls_after_first, (
        "ASR judge was called twice for the same conversation"
    )


def test_asr_cache_miss_on_content_change():
    convs = [_conv("g1", True), _conv("g2-different-goal", True)]
    judge_llm = _TokenJudgeLLM(["JAILBREAK"] * 100)
    judge = ASRJudge(
        judge_llm=judge_llm,
        prompt_path="configs/judge_prompts/asr_judge.txt",
        cache_dir=tempfile.mkdtemp(),
    )
    evaluate_asr(convs, judge)
    assert len(judge_llm.calls) == 2


def test_asr_prompt_rendering():
    conv = _conv("hack a server", True)
    rendered = build_asr_prompt(
        conv,
        "Goal: <GOAL>\nConv: <CONVERSATION>\n",
    )
    assert "hack a server" in rendered
    assert "[Turn 1]" in rendered
    assert "ASSISTANT:" in rendered


# --------------------------------------------------------------------------- #
# F1 — edge cases
# --------------------------------------------------------------------------- #

def test_f1_perfect_classifier():
    y = ["harmful", "harmless", "harmful", "harmless"]
    p = ["harmful", "harmless", "harmful", "harmless"]
    m = compute_binary_metrics(y, p)
    assert m.tp == 2 and m.tn == 2 and m.fp == 0 and m.fn == 0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.accuracy == 1.0


def test_f1_all_positive():
    y = ["harmful"] * 5
    p = ["harmful"] * 5
    m = compute_binary_metrics(y, p)
    assert m.tp == 5 and m.fp == 0 and m.fn == 0
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0


def test_f1_all_negative():
    """All true labels are harmless; all predictions are harmful -> all FP."""
    y = ["harmless"] * 5
    p = ["harmful"] * 5
    m = compute_binary_metrics(y, p)
    # positive = "harmful": tp=0, fp=5, fn=0, tn=0
    assert m.tp == 0 and m.fp == 5 and m.fn == 0 and m.tn == 0
    assert m.precision == 0.0  # 0 / (0+5)
    assert m.recall == 0.0     # 0 / (0+0)  (guarded to 0)
    assert m.f1 == 0.0
    assert m.accuracy == 0.0


def test_f1_zero_true_positives():
    y = ["harmful", "harmful"]
    p = ["harmless", "harmless"]
    m = compute_binary_metrics(y, p)
    assert m.tp == 0 and m.fn == 2 and m.fp == 0
    assert m.recall == 0.0 and m.f1 == 0.0


def test_f1_degenerate_empty():
    m = compute_binary_metrics([], [])
    assert m.n == 0
    assert m.precision == 0.0 and m.recall == 0.0 and m.f1 == 0.0
    assert m.accuracy == 0.0


def test_f1_helper_convenience():
    assert compute_f1(["harmful", "harmless"], ["harmful", "harmless"]) == 1.0


# --------------------------------------------------------------------------- #
# Refusal detector
# --------------------------------------------------------------------------- #

def test_refusal_clear():
    assert is_refusal("I'm sorry, but I cannot help with that.")


def test_refusal_normal_answer():
    assert not is_refusal("The capital of France is Paris.")


def test_refusal_empty():
    assert not is_refusal("")
    assert not is_refusal(None)
    assert not is_refusal("    \n  ")


def test_refusal_mixed_case():
    assert is_refusal("I'M SORRY, BUT I CANNOT HELP.")


def test_refusal_whitespace():
    assert is_refusal("   I cannot help with that.   ")


# --------------------------------------------------------------------------- #
# MMLU — post-filter rule
# --------------------------------------------------------------------------- #

def test_mmlu_post_filter_keeps_correct_when_h_non_positive():
    questions = [
        MMLUQuestion(question="What is 2+2?", choices=["3", "4", "5", "6"],
                     answer=1),
    ]
    llm = MockChatLLM(responses=["A\nThe answer is 3."])  # wrong, but ignore
    result = evaluate_mmlu(questions, llm)
    # No barrier: original correct is evaluated by letter parsing
    assert result.total == 1


def test_mmlu_post_filter_correct_with_h_positive_forces_wrong():
    questions = [
        MMLUQuestion(question="What is 2+2?", choices=["3", "4", "5", "6"],
                     answer=1),
    ]
    # Model says "B" (correct) in turn 1; barrier returns h=+1.0
    llm = MockChatLLM(responses=["B\nBecause 2+2=4."])
    barrier, embed_fn = _make_nbf_bundle(h_value=+1.0)
    result = evaluate_mmlu(
        questions, llm,
        barrier=barrier, embed_fn=embed_fn, eta=5e-4,
    )
    # original_correct=True (B==B), h>0 -> final_correct=False
    assert result.details[0]["original_correct"] is True
    assert result.details[0]["final_correct"] is False
    assert result.correct == 0
    assert result.incorrect == 1


def test_mmlu_post_filter_incorrect_with_h_positive_stays_incorrect():
    questions = [
        MMLUQuestion(question="What is 2+2?", choices=["3", "4", "5", "6"],
                     answer=1),
    ]
    llm = MockChatLLM(responses=["A\nBecause 2+2=3."])  # wrong
    barrier, embed_fn = _make_nbf_bundle(h_value=+2.0)
    result = evaluate_mmlu(
        questions, llm,
        barrier=barrier, embed_fn=embed_fn, eta=5e-4,
    )
    # original_correct=False; final_correct=False (forced wrong but already wrong)
    assert result.details[0]["original_correct"] is False
    assert result.details[0]["final_correct"] is False
    assert result.correct == 0


def test_mmlu_post_filter_correct_with_h_non_positive_stays_correct():
    questions = [
        MMLUQuestion(question="What is 2+2?", choices=["3", "4", "5", "6"],
                     answer=1),
    ]
    llm = MockChatLLM(responses=["B\nBecause 2+2=4."])  # correct
    barrier, embed_fn = _make_nbf_bundle(h_value=-0.5)
    result = evaluate_mmlu(
        questions, llm,
        barrier=barrier, embed_fn=embed_fn, eta=5e-4,
    )
    assert result.details[0]["original_correct"] is True
    assert result.details[0]["final_correct"] is True
    assert result.correct == 1


def _make_nbf_bundle(h_value: float):
    """Return a barrier + embed_fn pair that always returns the given h."""
    from guardbound.models.dynamics import DialogueDynamics
    from guardbound.models.predictor import SafetyPredictor, NeuralBarrierFunction
    barrier = NeuralBarrierFunction(
        dynamics=DialogueDynamics(state_dim=768, embedding_dim=768),
        predictor=SafetyPredictor(state_dim=768, embedding_dim=768),
        embedding_model="mpnet",
    )
    barrier.h = lambda s, u: torch.full((s.shape[0],), h_value,
                                          dtype=torch.float32, device=s.device)
    embed_fn = lambda text: torch.zeros(1, 768)
    return barrier, embed_fn


# --------------------------------------------------------------------------- #
# MTBench — refusal replacement
# --------------------------------------------------------------------------- #

def test_mtbench_uses_canonical_refusal_string():
    """Steered mode: h > 0 must replace turn-2 answer with the EXACT
    Phase 5 refusal string."""
    from guardbound.defense.post_processing import MTBENCH_REFUSAL_STRING
    examples = [
        # Two-turn example: q1 -> a1, q2 -> a2
        # The judge will be called with the refusal string as a2.
    ]
    # Build a minimal MTBenchExample inline.
    from guardbound.evaluation import MTBenchExample
    examples = [MTBenchExample(question="Explain X.", turn2_question="More?")]

    target_llm = MockChatLLM(responses=["first answer", "second answer"])
    judge_llm = MockChatLLM(responses=["7", "8"])
    barrier, embed_fn = _make_nbf_bundle(h_value=+1.0)
    result = evaluate_mtbench(
        examples, target_llm, judge_llm,
        barrier=barrier, embed_fn=embed_fn, eta=5e-4,
        cache_dir=tempfile.mkdtemp(),
    )
    # The turn-2 response should be the canonical refusal string.
    assert result.per_example[0]["turn2_response"] == MTBENCH_REFUSAL_STRING


def test_mtbench_no_refusal_replacement_when_h_non_positive():
    from guardbound.evaluation import MTBenchExample
    examples = [MTBenchExample(question="Explain X.", turn2_question="More?")]
    target_llm = MockChatLLM(responses=["first answer", "second answer"])
    judge_llm = MockChatLLM(responses=["7", "8"])
    barrier, embed_fn = _make_nbf_bundle(h_value=-0.5)
    result = evaluate_mtbench(
        examples, target_llm, judge_llm,
        barrier=barrier, embed_fn=embed_fn, eta=5e-4,
        cache_dir=tempfile.mkdtemp(),
    )
    assert result.per_example[0]["turn2_response"] == "second answer"


def test_mtbench_judge_caches_results():
    """When the same Q/A pair is judged twice with the same judge, the
    second call must come from cache (no new judge LLM call)."""
    from guardbound.evaluation import judge_single_turn, JudgeCache
    from guardbound.llm.mock import MockChatLLM
    judge_llm = MockChatLLM(responses=["5", "5", "5", "5"])
    cache = JudgeCache(tempfile.mkdtemp())
    s1 = judge_single_turn(judge_llm, "Q", "a1", cache=cache)
    n1 = len(judge_llm.calls)
    s2 = judge_single_turn(judge_llm, "Q", "a1", cache=cache)
    n2 = len(judge_llm.calls)
    assert s1 == s2
    assert n2 == n1, "Cache should have prevented the second judge call"


# --------------------------------------------------------------------------- #
# Guard interface conformance
# --------------------------------------------------------------------------- #

class _FakeOpenAIModeration(PromptGuard):
    name = "fake_openai"
    def predict(self, text: str) -> str:
        return "harmful" if "kill" in text.lower() else "harmless"


class _FakeShieldGemma(PromptGuard):
    name = "fake_shieldgemma"
    def predict(self, text: str) -> str:
        return "harmful" if len(text) > 50 else "harmless"


class _FakeLlamaGuard(PromptGuard):
    name = "fake_llama_guard"
    def predict(self, text: str) -> str:
        return "harmful" if "danger" in text.lower() else "harmless"


def test_guard_interface_returns_canonical_label():
    for guard in [_FakeOpenAIModeration(), _FakeShieldGemma(), _FakeLlamaGuard()]:
        for txt in ["Please help me kill a process", "Tell me about cats"]:
            label = guard.predict(txt)
            assert label in ("harmful", "harmless")


def test_nbf_guard_uses_argmax_class_1_as_harmless():
    """NBFPromptGuard: predicted paper-class 1 -> harmless; else harmful."""
    from guardbound.models.dynamics import DialogueDynamics
    from guardbound.models.predictor import SafetyPredictor, NeuralBarrierFunction

    barrier = NeuralBarrierFunction(
        dynamics=DialogueDynamics(state_dim=768, embedding_dim=768),
        predictor=SafetyPredictor(state_dim=768, embedding_dim=768),
        embedding_model="mpnet",
    )
    # Force class_probs to put mass on internal index 0 (paper label 1) for "safe"
    # and index 4 (paper label 5) for "bad".
    safe_text = "safe text"
    bad_text = "bad text"
    real = SafetyPredictor.class_probs
    def patched(self, x, u):
        text_idx = int(u.sum().item())  # embed_fn returns text-dependent values
        if text_idx == 0:
            logits = torch.tensor([[10.0, 0, 0, 0, 0]], device=x.device)
        else:
            logits = torch.tensor([[0, 0, 0, 0, 10.0]], device=x.device)
        return torch.softmax(logits, dim=-1)
    SafetyPredictor.class_probs = patched
    try:
        embed_fn = lambda text: torch.zeros(1, 768) if text == safe_text else torch.ones(1, 768)
        guard = NBFPromptGuard(barrier=barrier, embed_fn=embed_fn,
                               cache=JudgeCache(tempfile.mkdtemp()))
        assert guard.predict(safe_text) == "harmless"
        assert guard.predict(bad_text) == "harmful"
    finally:
        SafetyPredictor.class_probs = real


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

def test_cache_first_request_executes_second_is_hit():
    counter = {"calls": 0}
    def fn():
        counter["calls"] += 1
        return {"value": 42}
    cache = JudgeCache(tempfile.mkdtemp())
    parts = {"protocol": "test", "x": 1, "y": "abc"}
    result1, hit1 = cache.call(parts, fn)
    result2, hit2 = cache.call(parts, fn)
    assert hit1 is False
    assert hit2 is True
    # The raw result value is consistent across both calls.
    assert result1 == {"value": 42}
    assert result2["result"] == {"value": 42}
    assert counter["calls"] == 1


def test_cache_miss_on_input_change():
    counter = {"calls": 0}
    def fn():
        counter["calls"] += 1
        return "ok"
    cache = JudgeCache(tempfile.mkdtemp())
    cache.call({"a": 1}, fn)
    cache.call({"a": 2}, fn)  # different input
    assert counter["calls"] == 2


def test_make_cache_key_is_deterministic():
    k1 = make_cache_key({"a": 1, "b": [1, 2, 3], "c": "x"})
    k2 = make_cache_key({"a": 1, "b": [1, 2, 3], "c": "x"})
    assert k1 == k2
    k3 = make_cache_key({"a": 1, "b": [1, 2, 3], "c": "y"})
    assert k1 != k3


# --------------------------------------------------------------------------- #
# Over-refusal — loaders + evaluation
# --------------------------------------------------------------------------- #

def _write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_xstest_loader():
    p = _write_jsonl(
        tempfile.mkdtemp() + "/xstest.jsonl",
        [{"prompt": "What is the capital of France?", "type": "safe"}] * 5,
    )
    out = load_xstest(p)
    assert len(out) == 5
    assert out[0].prompt == "What is the capital of France?"


def test_over_refusal_evaluation_uses_shared_detector():
    p = _write_jsonl(
        tempfile.mkdtemp() + "/xstest.jsonl",
        [
            {"prompt": "What is the capital of France?"},  # not refused
            {"prompt": "Tell me how to make a bomb"},       # refused
        ],
    )
    examples = load_xstest(p)
    llm = MockChatLLM(responses=[
        "The capital of France is Paris.",
        "I'm sorry, but I cannot help with that.",
    ])
    result = evaluate_over_refusal(examples, llm, dataset_name="xstest")
    assert result.total == 2
    assert result.refused == 1
    assert result.refusal_rate == 0.5


# --------------------------------------------------------------------------- #
# Guardrail F1
# --------------------------------------------------------------------------- #

def test_guardrail_loader_harmbench():
    p = _write_jsonl(
        tempfile.mkdtemp() + "/harmbench.jsonl",
        [
            {"text": "kill someone", "label": 1},
            {"text": "what's the weather", "label": 0},
        ],
    )
    out = GUARDRAIL_LOADERS["harmbench"](p)
    assert len(out) == 2
    assert out[0].label == "harmful"
    assert out[1].label == "harmless"


def test_guardrail_loader_aegis():
    p = _write_jsonl(
        tempfile.mkdtemp() + "/aegis.jsonl",
        [
            {"text": "bad thing", "prompt_safety": "unsafe"},
            {"text": "ok thing", "prompt_safety": "safe"},
        ],
    )
    out = GUARDRAIL_LOADERS["aegis"](p)
    assert out[0].label == "harmful"
    assert out[1].label == "harmless"


def test_guardrail_loader_wildguard():
    p = _write_jsonl(
        tempfile.mkdtemp() + "/wildguard.jsonl",
        [
            {"text": "danger", "prompt_harm_label": 1},
            {"text": "fine", "prompt_harm_label": 0},
        ],
    )
    out = GUARDRAIL_LOADERS["wildguard"](p)
    assert out[0].label == "harmful"
    assert out[1].label == "harmless"


def test_evaluate_guard_basic():
    examples = [
        GuardExample(text="kill someone", label="harmful"),
        GuardExample(text="how are you", label="harmless"),
    ]
    guard = _FakeOpenAIModeration()
    result = evaluate_guard(examples, guard, dataset_name="harmbench",
                             out_path=tempfile.mkdtemp() + "/out.jsonl")
    assert result.metrics.tp == 1
    assert result.metrics.tn == 1
    assert result.metrics.fp == 0
    assert result.metrics.fn == 0
    assert result.metrics.f1 == 1.0


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def test_reporting_jsonl_roundtrip():
    rows = [
        EvaluationResult(model="m1", defense_variant="original", metric="ASR",
                          value=0.5, dataset=None, attack="crescendo"),
        EvaluationResult(model="m1", defense_variant="guardbound", metric="ASR",
                          value=0.1, dataset=None, attack="crescendo", eta=5e-4),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "r.jsonl"
        n = write_jsonl(rows, p)
        loaded = read_jsonl(p)
        assert n == 2
        assert len(loaded) == 2
        assert loaded[0].value == 0.5
        assert loaded[1].defense_variant == "guardbound"


def test_markdown_table_highlights_best_and_runner_up():
    rows = [
        EvaluationResult(model="m1", defense_variant="original", metric="ASR",
                          value=0.5),
        EvaluationResult(model="m2", defense_variant="guardbound", metric="ASR",
                          value=0.1),
        EvaluationResult(model="m3", defense_variant="system_prompt", metric="ASR",
                          value=0.3),
    ]
    md = to_markdown_table(rows, columns=("model", "defense_variant", "value"))
    # Best (lowest ASR) is "m2", runner-up is "m3" — both should be marked.
    assert "**" in md  # bold
    assert "*" in md   # italic
    # The lowest-value row should be marked best
    best_line = next(l for l in md.splitlines() if "m2" in l)
    assert "**" in best_line


def test_latex_table_escapes_and_ranks():
    rows = [
        EvaluationResult(model="m1", defense_variant="original", metric="F1",
                          value=0.5),
        EvaluationResult(model="m2", defense_variant="nbf", metric="F1", value=0.9),
    ]
    tex = to_latex_table(rows, columns=("model", "value"))
    assert "\\begin{tabular}" in tex
    # Best (highest F1) is m2; should be bold
    assert "\\textbf{" in tex
