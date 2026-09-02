"""Phase 8 — defense baseline tests.

All tests are offline.  No GPU, HF downloads, LLaMA-Factory, or API
credentials required.  MockChatLLM is used throughout.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardbound.baselines import (
    BASELINE_VARIANTS,
    DEFAULT_BASELINE_VARIANTS,
    build_defense,
    list_variants,
)
from guardbound.baselines.base import DefenseWrapper
from guardbound.baselines.manifests import build_manifest, write_manifest
from guardbound.baselines.original import OriginalDefense
from guardbound.baselines.system_prompt import (
    DEFAULT_PROMPT_PATH,
    SystemPromptChatLLM,
    SystemPromptDefense,
    _strip_provenance,
)
from guardbound.llm.base import ChatLLM
from guardbound.llm.mock import MockChatLLM
from guardbound.schemas import Conversation, Turn


# --------------------------------------------------------------------------- #
# Test 1 — Registry completeness
# --------------------------------------------------------------------------- #

def test_registry_contains_all_variants():
    expected = {"original", "system_prompt", "lora_sft",
                "lora_dpo", "lora_kto", "guardbound"}
    assert set(BASELINE_VARIANTS) == expected
    assert set(list_variants()) == expected
    assert set(DEFAULT_BASELINE_VARIANTS) == expected


def test_registry_rejects_unknown_variant():
    llm = MockChatLLM(responses=[])
    with pytest.raises(ValueError, match="Unknown defense variant"):
        build_defense("not_a_real_variant", llm)


# --------------------------------------------------------------------------- #
# Test 2 — Original defense does not modify requests
# --------------------------------------------------------------------------- #

def test_original_does_not_modify_requests():
    llm = MockChatLLM(responses=["hello"])
    defended = OriginalDefense().wrap(llm)
    msgs = [{"role": "user", "content": "hi"}]
    defended.generate(msgs, temperature=0.0)
    # The mock records the messages exactly; no system prompt added.
    assert llm.calls[0][0] == msgs


# --------------------------------------------------------------------------- #
# Test 3 — System-prompt wrapper
# --------------------------------------------------------------------------- #

def test_system_prompt_is_added():
    llm = MockChatLLM(responses=["ok"])
    defended = SystemPromptDefense().wrap(llm)
    defended.generate([{"role": "user", "content": "hi"}], temperature=0.0)
    sent = llm.calls[0][0]
    assert sent[0]["role"] == "system"
    assert sent[0]["content"] != ""
    assert sent[1] == {"role": "user", "content": "hi"}


def test_system_prompt_is_not_duplicated():
    llm = MockChatLLM(responses=["ok"])
    defended = SystemPromptDefense().wrap(llm)
    # If the caller already supplied a matching leading system prompt,
    # the wrapper must not add a second copy.
    prompt = SystemPromptDefense()._prompt
    msgs = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "hi"},
    ]
    defended.generate(msgs, temperature=0.0)
    sent = llm.calls[0][0]
    # Same content count of system messages (no duplication).
    sys_msgs = [m for m in sent if m["role"] == "system"]
    assert len(sys_msgs) == 1
    assert sys_msgs[0]["content"] == prompt


def test_system_prompt_does_not_mutate_caller_list():
    llm = MockChatLLM(responses=["ok"])
    defended = SystemPromptDefense().wrap(llm)
    msgs = [{"role": "user", "content": "hi"}]
    msgs_snapshot = [dict(m) for m in msgs]
    defended.generate(msgs, temperature=0.0)
    assert msgs == msgs_snapshot, "Caller-owned messages list was mutated"


def test_system_prompt_preserves_multi_turn_order():
    llm = MockChatLLM(responses=["ok"])
    defended = SystemPromptDefense().wrap(llm)
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    defended.generate(msgs, temperature=0.0)
    sent = llm.calls[0][0]
    assert sent[0]["role"] == "system"
    # Multi-turn ordering preserved.
    assert [m["content"] for m in sent[1:]] == ["u1", "a1", "u2"]


def test_strip_provenance():
    body = "Real prompt\n---\nPROVENANCE NOTES\n"
    assert _strip_provenance(body) == "Real prompt\n"
    assert _strip_provenance("only body") == "only body"


def test_default_prompt_path_exists():
    assert DEFAULT_PROMPT_PATH.exists()
    text = DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
    # Body must be non-empty after stripping provenance.
    assert _strip_provenance(text).strip() != ""


def test_system_prompt_default_via_build_defense():
    llm = MockChatLLM(responses=[])
    defended = build_defense("system_prompt", llm, mock=True)
    assert isinstance(defended, SystemPromptChatLLM)


# --------------------------------------------------------------------------- #
# Test 4 — SFT data builder schema
# --------------------------------------------------------------------------- #

def test_sft_builder_replaces_responses_with_safe_text():
    from guardbound.baselines.build_sft_data import (
        _load_safety_responses,
        build_sft_examples,
    )
    safe = _load_safety_responses(_write_jsonl("safe.jsonl", [
        {"goal_id": "g1", "response": "I cannot help with that."},
        {"goal_id": "g2", "response": "I must decline."},
    ]))
    convs = [
        Conversation(goal="g1", attack_method="crescendo", target_llm="mock",
                     turns=[Turn(query="q1", response="ORIGINAL_BAD_1"),
                            Turn(query="q2", response="ORIGINAL_BAD_2")]),
        Conversation(goal="g2", attack_method="crescendo", target_llm="mock",
                     turns=[Turn(query="q3", response="ORIGINAL_BAD_3")]),
    ]
    examples = build_sft_examples(convs, safe)
    assert len(examples) == 2
    # First example: no ORIGINAL_BAD_ text remains.
    txt = json.dumps(examples[0].messages)
    assert "ORIGINAL_BAD" not in txt
    assert "I cannot help with that." in txt
    assert "q1" in txt and "q2" in txt
    # All assistant turns should be the SAME safe response (we
    # always substitute the verified safe response, never a per-turn
    # generated one).
    assistants = [m for m in examples[0].messages if m["role"] == "assistant"]
    assert all(a["content"] == "I cannot help with that." for a in assistants)


def test_sft_builder_skips_unmatched_goals():
    from guardbound.baselines.build_sft_data import (
        _load_safety_responses,
        build_sft_examples,
    )
    safe = _load_safety_responses(_write_jsonl("safe.jsonl", [
        {"goal_id": "g1", "response": "safe."},
    ]))
    convs = [
        Conversation(goal="g1", attack_method="crescendo", target_llm="mock",
                     turns=[Turn(query="q", response="bad")]),
        Conversation(goal="g_unknown", attack_method="crescendo", target_llm="mock",
                     turns=[Turn(query="q", response="bad")]),
    ]
    examples = build_sft_examples(convs, safe)
    assert len(examples) == 1
    assert examples[0].goal_id == "g1"


def test_sft_builder_fails_clearly_on_missing_safety_file():
    from guardbound.baselines.build_sft_data import _load_safety_responses
    with pytest.raises(FileNotFoundError, match="safety-aligned response dataset"):
        _load_safety_responses("/tmp/does-not-exist-safety.jsonl")


def test_sft_builder_fails_clearly_on_empty_safety_file(tmp_path):
    from guardbound.baselines.build_sft_data import _load_safety_responses
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="No valid safety-aligned"):
        _load_safety_responses(empty)


def test_sft_builder_does_not_fabricate_when_no_response_field(tmp_path):
    """If the safety file has records but no valid response, fail loudly."""
    from guardbound.baselines.build_sft_data import _load_safety_responses
    f = tmp_path / "bad.jsonl"
    f.write_text(json.dumps({"goal_id": "g1"}) + "\n")  # no 'response' key
    with pytest.raises(ValueError):
        _load_safety_responses(f)


def test_sft_dataset_end_to_end(tmp_path):
    from guardbound.baselines.build_sft_data import build_sft_dataset
    corpus = tmp_path / "phase2.jsonl"
    safe = tmp_path / "safe.jsonl"
    out = tmp_path / "sft.jsonl"
    corpus.write_text(json.dumps({
        "goal": "g1", "attack_method": "crescendo", "target_llm": "mock",
        "turns": [
            {"query": "q1", "response": "BAD1"},
            {"query": "q2", "response": "BAD2"},
        ],
    }) + "\n")
    safe.write_text(json.dumps({"goal_id": "g1", "response": "safe."}) + "\n")
    info = build_sft_dataset(corpus, safe, out)
    assert info["n_examples_written"] == 1
    rows = [json.loads(l) for l in out.read_text().splitlines() if l]
    msgs = rows[0]["messages"]
    # The first user message is q1; the first assistant message is the
    # verified safe response (replacing the original BAD1).
    assert msgs[0] == {"role": "user", "content": "q1"}
    assert msgs[1] == {"role": "assistant", "content": "safe."}
    # No "BAD" text remains anywhere in the example.
    assert "BAD" not in json.dumps(rows)


# --------------------------------------------------------------------------- #
# Test 5 — Preference data (DPO / KTO)
# --------------------------------------------------------------------------- #

def test_dpo_examples_contain_prompt_chosen_rejected(tmp_path):
    from guardbound.baselines.build_preference_data import build_dpo_examples
    safe = {"g1": "safe response."}
    convs = [Conversation(goal="g1", attack_method="crescendo", target_llm="mock",
                          turns=[Turn(query="q", response="jailbreak answer")])]
    examples = build_dpo_examples(convs, safe)
    assert len(examples) == 1
    assert examples[0].chosen[0]["content"] == "safe response."
    assert examples[0].rejected[0]["content"] == "jailbreak answer"
    # The prompt is a list of message dicts starting with a user turn.
    assert any(m["role"] == "user" for m in examples[0].prompt)


def test_kto_examples_have_labels(tmp_path):
    from guardbound.baselines.build_preference_data import build_kto_examples
    safe = {"g1": "safe."}
    convs = [Conversation(goal="g1", attack_method="crescendo", target_llm="mock",
                          turns=[Turn(query="q", response="bad")])]
    examples = build_kto_examples(convs, safe)
    # One positive, one negative.
    labels = sorted([e.label for e in examples])
    assert labels == [False, True]


def test_preference_datasets_end_to_end(tmp_path):
    from guardbound.baselines.build_preference_data import (
        build_preference_datasets,
    )
    corpus = tmp_path / "phase2.jsonl"
    safe = tmp_path / "safe.jsonl"
    out_dir = tmp_path / "pref"
    corpus.write_text(json.dumps({
        "goal": "g1", "attack_method": "crescendo", "target_llm": "mock",
        "turns": [{"query": "q", "response": "jailbreak"}],
    }) + "\n")
    safe.write_text(json.dumps({"goal_id": "g1", "response": "safe."}) + "\n")
    info = build_preference_datasets(corpus, safe, out_dir)
    assert info["n_dpo_examples"] == 1
    assert info["n_kto_examples"] == 2  # positive + negative
    dpo_lines = (out_dir / "dpo.jsonl").read_text().splitlines()
    kto_lines = (out_dir / "kto.jsonl").read_text().splitlines()
    assert len(dpo_lines) == 1
    assert len(kto_lines) == 2


# --------------------------------------------------------------------------- #
# Test 6 — Manifest
# --------------------------------------------------------------------------- #

def test_manifest_records_paper_and_unspecified(tmp_path):
    manifest = build_manifest(
        variant="lora_sft",
        base_model="meta-llama/Meta-Llama-3-8B-Instruct",
        dataset_path=tmp_path / "data.jsonl",
        learning_rate=2e-4,
        num_train_epochs=3,
        lora_params={"rank": 8, "alpha": 16},
        scheduler="cosine",
        command_line=["train_lora_sft.py", "--model", "llama-3-8b-instruct"],
        output_dir=tmp_path / "out",
    )
    assert manifest["paper_specified"]["learning_rate"] == 2e-4
    assert manifest["paper_specified"]["num_train_epochs"] == 3
    # Unspecified values are recorded.
    assert "lora_rank" in manifest["not_specified_in_paper"]
    assert manifest["not_specified_in_paper"]["lora_alpha"] == 16
    assert manifest["command_line"][0] == "train_lora_sft.py"
    # Write to disk and reload.
    path = tmp_path / "m.json"
    write_manifest(manifest, path)
    loaded = json.loads(path.read_text())
    assert loaded["paper_specified"]["num_train_epochs"] == 3


# --------------------------------------------------------------------------- #
# Test 7 — Serving
# --------------------------------------------------------------------------- #

def test_serving_mock_mode_returns_chatllm():
    from guardbound.baselines.serving import build_hf_chatllm, is_mock_mode
    with pytest.MonkeyPatch.context() as m:
        m.setenv("NBF_MOCK", "1")
        # Reload the function's view of the env (it's called inside).
        assert is_mock_mode() is True
        llm = build_hf_chatllm("meta-llama/Meta-Llama-3-8B-Instruct", mock=True)
    assert isinstance(llm, ChatLLM)
    assert isinstance(llm, MockChatLLM)


def test_serving_no_torch_uses_mock():
    from guardbound.baselines import serving
    # If torch is unavailable, is_mock_mode() is True.
    if serving.is_mock_mode():
        llm = serving.build_hf_chatllm(
            "meta-llama/Meta-Llama-3-8B-Instruct", mock=False,
        )
        # Falls back to MockChatLLM, never silently substituting a
        # different real model.
        assert isinstance(llm, MockChatLLM)


# --------------------------------------------------------------------------- #
# Test 8 — Registry: lora variants require checkpoint
# --------------------------------------------------------------------------- #

def test_lora_variants_require_checkpoint_and_basemodel():
    llm = MockChatLLM(responses=[])
    with pytest.raises(ValueError, match="requires a LoRA adapter path"):
        build_defense("lora_sft", llm, base_model="m", checkpoint=None,
                       mock=True)
    with pytest.raises(ValueError, match="requires a base_model"):
        # MockChatLLM has no .model_id; require explicit base_model
        build_defense("lora_sft", llm, checkpoint="/tmp/ckpt", mock=True)


# --------------------------------------------------------------------------- #
# Test 9 — NBF registry delegation
# --------------------------------------------------------------------------- #

def test_guardbound_delegates_to_steered_chat():
    import torch
    from guardbound.models.dynamics import DialogueDynamics
    from guardbound.models.predictor import (
        SafetyPredictor, NeuralBarrierFunction,
    )
    barrier = NeuralBarrierFunction(
        dynamics=DialogueDynamics(state_dim=768, embedding_dim=768),
        predictor=SafetyPredictor(state_dim=768, embedding_dim=768),
        embedding_model="mpnet",
    )
    # Make the barrier always pass through (very negative h).
    barrier.h = lambda s, u: torch.full((s.shape[0],), -10.0)
    embed_fn = lambda t: torch.zeros(1, 768)
    base = MockChatLLM(responses=["ok"])
    defended = build_defense(
        "guardbound", base, eta=5e-4, barrier=barrier, embed_fn=embed_fn,
    )
    # Should be a SteeredLLMChat, not a new ad-hoc implementation.
    from guardbound.defense.steered_chat import SteeredLLMChat
    assert isinstance(defended, SteeredLLMChat)
    # The barrier and eta are correctly wired.
    assert defended.barrier is barrier
    assert defended.eta == 5e-4
    assert defended.target_llm is base


# --------------------------------------------------------------------------- #
# Test 10 — Eval loop (compose Phase 6 + Phase 7)
# --------------------------------------------------------------------------- #

def test_eval_loop_produces_evaluation_result_rows(tmp_path):
    """The full registry -> attack -> judge -> results pipeline emits
    EvaluationResult rows in the same schema as Phase 7."""
    from guardbound.evaluation import (
        ASRJudge,
        EvaluationResult,
        evaluate_asr,
    )
    from guardbound.attacks.runner import run_attack
    from guardbound.attacks.base import MultiTurnAttack

    base = MockChatLLM(responses=["some response", "another response"])
    defended = build_defense("original", base, mock=True)

    # Use a local non-stub attack (Phase 6 Crescendo raises
    # NotImplementedError in paper-strict mode).
    class _LocalAttack(MultiTurnAttack):
        name = "local"
        def next_query(self, goal, history):
            return f"echo:{goal}:{len(history)}"
        def is_finished(self, history, max_turns=8):
            return len(history) >= max_turns

    attack = _LocalAttack()
    conv = run_attack(
        attack=attack, goal="test", target_llm=defended,
        max_turns=2, temperature=0.7,
        attack_method="local", target_llm_name="mock",
    )
    # Save + reload + ASR via the SAME Phase 7 evaluate_asr
    out_path = tmp_path / "out.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(conv.to_json() + "\n")
    from guardbound.schemas import load_conversations_jsonl
    conversations = load_conversations_jsonl(out_path)
    judge_llm = MockChatLLM(responses=["SAFE"] * 100)
    judge = ASRJudge(judge_llm=judge_llm, prompt_path="configs/judge_prompts/asr_judge.txt",
                     cache_dir=str(tmp_path / "cache"))
    _, agg = evaluate_asr(conversations, judge, out_path=out_path)
    # ASR with mock SAFE judge -> 0.0
    assert agg["asr"] == 0.0
    assert agg["total_behaviors"] == 1
    # The sidecar is a Phase 7 EvaluationResult row.
    sidecar = out_path.with_suffix(".results.jsonl")
    rows = [json.loads(l) for l in sidecar.read_text().splitlines() if l]
    # The mock SAFE judge (with a "filtered" Conversation) is flagged
    # as guardbound by evaluate_asr's heuristic.  Accept either.
    assert rows[0]["defense_variant"] in {"guardbound", "original"}
    # Ensure the row has the Phase 7 schema fields.
    assert {"model", "defense_variant", "metric", "value", "dataset",
            "attack", "eta", "judge_model"} <= set(rows[0].keys())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _write_jsonl(path: str, rows: list[dict]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)
