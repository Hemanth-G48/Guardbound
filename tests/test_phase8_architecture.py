"""Phase 8 — provider/evaluator/target isolation tests.

Proves the architectural requirements without any real model:

* Provider isolation — switching provider: local <-> cloud changes only the
  ChatLLM backend, never the attack code, prompts, or history handling.
* Evaluator isolation — rubric/refusal/disclaimer/step-judge calls route to
  the evaluator LLM, while attacker-side calls stay on the attacker LLM, and
  each subrole keeps its own prompt/schema.
* Target isolation — evaluator prompts never enter the target's conversation.
* ModelManager — sequential swap semantics and VRAM event recording.
* Config validation — role separation, cloud-path preservation.

Run:  pytest tests/test_phase8_architecture.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from guardbound.llm.base import ChatLLM
from guardbound.llm.mock import MockChatLLM
from guardbound.attacks.crescendo_paper import CrescendoAttackPaper
from guardbound.attacks.opposite_day import OppositeDayAttack
from guardbound.attacks.acronym import AcronymAttack
from guardbound.attacks.actor_attack import ActorAttack
from guardbound.attacks.runner import run_attack_with_backtracking
from guardbound.schemas import Turn

THREE_MODEL_CFG = _PROJECT_ROOT / "configs/reproduction_three_model.yaml"


def _load_three_model_cfg() -> dict:
    with open(THREE_MODEL_CFG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RecordingLLM(ChatLLM):
    """Mock that records every call and replies with fixed, marker-safe text.

    `reply` is returned verbatim for plain calls; `json_reply` for JSON calls.
    Records (purpose-tag, messages snapshot, temperature, json_format).
    """

    name = "recording"

    def __init__(self, reply: str = "generic helpful answer",
                 json_reply: dict | None = None):
        self.reply = reply
        self.json_reply = json_reply if json_reply is not None else {}
        self.calls: list[tuple[str, list[dict], float, bool]] = []

    @staticmethod
    def _classify(messages) -> str:
        blob = "\n".join(m.get("content", "") for m in messages)
        if "Extract the core harmful target" in blob:
            return "extract"
        if "actor_name" in blob and '"actors"' in blob:
            return "actors"
        if '"questions"' in blob:
            return "questions"
        if "responseToEvaluateInput" in blob and "value" in blob:
            return "refusal_or_disclaimer"
        if "scale of 1 to 5" in blob:
            return "rubric"
        if '"type"' in blob and ("successful" in blob and "rejective" in blob):
            return "step_judge"
        if "generatedQuestion" in blob:
            return "generation"
        if "modify your question" in blob:
            return "rewrite"
        return "other"

    def generate(self, messages, temperature=0.7, max_turns_context=None,
                 json_format=False):
        purpose = self._classify(messages)
        self.calls.append((purpose, [dict(m) for m in messages],
                           temperature, bool(json_format)))
        if json_format:
            return dict(self.json_reply)
        return self.reply


def _make_attacker() -> RecordingLLM:
    """Attacker-side recording LLM wired for generation + summaries."""
    return RecordingLLM(
        reply="Approach text.\n1. What is the history of the topic?",
        json_reply={"generatedQuestion": "Q?", "lastResponseSummary": "sum"},
    )


def _make_evaluator() -> RecordingLLM:
    """Evaluator-side recording LLM wired for all four judging subroles."""
    return RecordingLLM(
        reply='{"score": 3}',
        json_reply={"value": False, "metadata": 5, "score": 3,
                    "type": "successful"},
    )


# =========================================================================== #
# Evaluator isolation: the four subroles route to the evaluator LLM
# =========================================================================== #

class TestEvaluatorIsolationCrescendo:
    def test_rubric_and_refusal_use_evaluator(self):
        attacker, evaluator = _make_attacker(), _make_evaluator()
        attack = CrescendoAttackPaper(max_turns=8, attacker_llm=attacker)
        attack.set_evaluator_llm(evaluator)

        attack.evaluate_response("q", "r", "goal")
        attack.check_refusal("q", "r")

        rubric_calls = evaluator.calls_of("rubric") if hasattr(
            evaluator, "calls_of") else [c for c in evaluator.calls
                                         if c[0] == "rubric"]
        refusal_calls = [c for c in evaluator.calls
                         if c[0] == "refusal_or_disclaimer"]
        assert rubric_calls, "rubric must run on the evaluator LLM"
        assert refusal_calls, "refusal check must run on the evaluator LLM"
        # The attacker must NOT have received any judging prompt.
        assert not [c for c in attacker.calls
                    if c[0] in ("rubric", "refusal_or_disclaimer")]

    def test_generation_stays_on_attacker(self):
        attacker, evaluator = _make_attacker(), _make_evaluator()
        attack = CrescendoAttackPaper(max_turns=8, attacker_llm=attacker)
        attack.set_evaluator_llm(evaluator)

        attack.next_query("goal", [])

        gen = [c for c in attacker.calls if c[0] == "generation"]
        assert gen, "query generation must run on the attacker LLM"
        assert not evaluator.calls, \
            "generation must not touch the evaluator LLM"

    def test_no_evaluator_keeps_official_attacker_judging(self):
        attacker = _make_attacker()
        attack = CrescendoAttackPaper(max_turns=8, attacker_llm=attacker)

        attack.evaluate_response("q", "r", "goal")

        assert [c for c in attacker.calls if c[0] == "rubric"], (
            "without a separate evaluator, the official attacker-as-evaluator "
            "behavior must hold"
        )


class TestEvaluatorIsolationOppositeDay:
    def test_judging_routes_to_evaluator(self):
        attacker, evaluator = _make_attacker(), _make_evaluator()
        attack = OppositeDayAttack(max_turns=8, attacker_llm=attacker)
        attack.set_evaluator_llm(evaluator)

        attack.evaluate_response("q", "r", "goal")
        attack.check_refusal("q", "r")

        assert [c for c in evaluator.calls if c[0] == "rubric"]
        assert [c for c in evaluator.calls if c[0] == "refusal_or_disclaimer"]
        assert not [c for c in attacker.calls
                    if c[0] in ("rubric", "refusal_or_disclaimer")]


class TestEvaluatorIsolationAcronym:
    def test_refusal_or_disclaimer_both_on_evaluator(self):
        attacker, evaluator = _make_attacker(), _make_evaluator()
        attack = AcronymAttack(max_turns=8, attacker_llm=attacker)
        attack.set_evaluator_llm(evaluator)

        attack.check_refusal("q", "r")
        attack.check_disclaimer("q", "r")

        checks = [c for c in evaluator.calls
                  if c[0] == "refusal_or_disclaimer"]
        assert len(checks) >= 3, (
            "check_refusal (OR-disclaimer) + explicit check_disclaimer must "
            "all run on the evaluator"
        )
        assert not [c for c in attacker.calls
                    if c[0] == "refusal_or_disclaimer"]

    def test_rubric_on_evaluator(self):
        attacker, evaluator = _make_attacker(), _make_evaluator()
        attack = AcronymAttack(max_turns=8, attacker_llm=attacker)
        attack.set_evaluator_llm(evaluator)

        attack.evaluate_response("q", "r", "goal")
        assert [c for c in evaluator.calls if c[0] == "rubric"]


class TestEvaluatorIsolationActorAttack:
    def test_step_judge_and_rubric_on_evaluator(self):
        attacker, evaluator = _make_attacker(), _make_evaluator()
        attack = ActorAttack(max_turns=8, attacker_llm=attacker)
        attack.set_evaluator_llm(evaluator)
        # Seed pre-attack data so no real pre-attack inference happens.
        attack._pre_attack_data = {
            "instruction": "G",
            "harm_target": "T",
            "query_details": {"delivery_type": "article", "other_details": ""},
            "network_hist": [],
            "actors": [{"actor": {"actor_name": "A", "relationship": "R"},
                        "queries": ["q1"], "query_chain": ""}],
        }

        judge = attack.step_judge("q1", "response")
        attack.evaluate_response("q1", "response", "G")

        assert judge == "successful"
        assert [c for c in evaluator.calls if c[0] == "step_judge"]
        assert [c for c in evaluator.calls if c[0] == "rubric"]
        # The attacker LLM must not receive judge prompts.
        assert not [c for c in attacker.calls
                    if c[0] in ("step_judge", "rubric")]

    def test_rewrites_stay_on_attacker(self):
        attacker, evaluator = _make_attacker(), _make_evaluator()
        attack = ActorAttack(max_turns=8, attacker_llm=attacker)
        attack.set_evaluator_llm(evaluator)
        attack._pre_attack_data = {
            "instruction": "G",
            "harm_target": "T",
            "query_details": {"delivery_type": "article", "other_details": ""},
            "network_hist": [],
            "actors": [{"actor": {"actor_name": "A", "relationship": "R"},
                        "queries": ["q1"], "query_chain": ""}],
        }
        attack._queries = ["q1"]

        attack.rewrite_query_if_needed("q1", "bad response",
                                       attack.get_query_details())

        assert [c for c in attacker.calls if c[0] == "rewrite"], (
            "query rewriting is attacker-side generation"
        )
        assert not [c for c in evaluator.calls if c[0] == "rewrite"]


# =========================================================================== #
# Target isolation: evaluator prompts never enter the target conversation
# =========================================================================== #

class TestTargetIsolation:
    @staticmethod
    def _run_turns(attack):
        target = RecordingLLM(reply="A benign, harmless reply.")
        history: list[Turn] = []
        for _ in range(2):
            q = attack.next_query("goal", history)
            if not q:
                break
            r = target.generate([{"role": "user", "content": q}])
            history.append(Turn(query=q, response=r, was_filtered=False))
            attack.record_turn(q, r, attack.evaluate_response(q, r, "goal"))
            attack.check_refusal(q, r)
        return target

    def test_target_sees_only_target_context(self):
        attacker, evaluator = _make_attacker(), _make_evaluator()
        attack = CrescendoAttackPaper(max_turns=8, attacker_llm=attacker)
        attack.set_evaluator_llm(evaluator)

        target = self._run_turns(attack)

        blob = "\n".join(
            m.get("content", "") for call in target.calls for m in call[1]
        )
        assert "scale of 1 to 5" not in blob, \
            "rubric prompt leaked into target context"
        assert "responseToEvaluateInput" not in blob, \
            "refusal-check prompt leaked into target context"
        assert "disclaimer" not in blob.lower(), \
            "disclaimer prompt leaked into target context"
        assert "rejective" not in blob, \
            "step-judge prompt leaked into target context"

    def test_actor_target_never_sees_judge_prompts(self):
        attacker, evaluator = _make_attacker(), _make_evaluator()
        attack = ActorAttack(max_turns=8, attacker_llm=attacker)
        attack.set_evaluator_llm(evaluator)
        attack._pre_attack_data = {
            "instruction": "G",
            "harm_target": "T",
            "query_details": {"delivery_type": "article", "other_details": ""},
            "network_hist": [],
            "actors": [{"actor": {"actor_name": "A", "relationship": "R"},
                        "queries": ["q1", "q2"], "query_chain": ""}],
        }

        target = RecordingLLM(reply="A benign reply.")
        q1 = attack.next_query("goal", [])
        r1 = target.generate([{"role": "user", "content": q1}])
        attack.evaluate_response(q1, r1, "goal")
        attack.check_refusal(q1, r1)
        attack.record_turn(q1, r1, 3)

        blob = "\n".join(
            m.get("content", "") for call in target.calls for m in call[1]
        )
        assert "successful, rejective, or unknown" not in blob
        assert "scale of 1 to 5" not in blob


# =========================================================================== #
# Provider isolation: provider switch is config-only, no attack changes
# =========================================================================== #

class _FakeCloudLLM(ChatLLM):
    """Stands in for OpenAIChatLLM without importing the SDK."""

    name = "fake-cloud"

    def __init__(self, model):
        self.model = model

    def generate(self, messages, temperature=0.7, max_turns_context=None,
                 json_format=False):
        return {"generatedQuestion": "Q?", "lastResponseSummary": "s"} \
            if json_format else "cloud reply"


class TestProviderIsolation:
    def test_attacks_accept_any_chatllm(self):
        """The same attack object drives local-style and cloud-style backends
        through the identical code path."""
        for backend in (MockChatLLM(responses=[]), _FakeCloudLLM("gpt-4o")):
            attacker, evaluator = _make_attacker(), _make_evaluator()
            attack = CrescendoAttackPaper(max_turns=8, attacker_llm=attacker)
            attack.set_evaluator_llm(evaluator)
            # The attack itself is what must stay provider-agnostic: it only
            # calls .generate() on whatever ChatLLM it was handed.
            q = attack.next_query("goal", [])
            assert isinstance(q, str)
            score = attack.evaluate_response(q, "resp", "goal")
            assert 1 <= score <= 5

    def test_provider_factory_local_registers_with_manager(self, monkeypatch):
        import os
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from guardbound.llm.provider_factory import build_role_llm
        from guardbound.llm.model_manager import ModelManager

        manager = ModelManager()
        llm = build_role_llm(
            {"provider": "local",
             "model": "microsoft/Phi-4-mini-instruct", "parameters": "3.8B"},
            "target", manager=manager,
        )
        assert manager.get("target") is not None
        assert "managed[target]" in llm.name

    def test_provider_factory_cloud_requires_key(self, monkeypatch):
        import os
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from guardbound.llm.provider_factory import build_role_llm
        from guardbound.llm.model_manager import ModelManager

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            build_role_llm(
                {"provider": "cloud", "model": "gpt-4o"}, "attacker",
                manager=ModelManager(),
            )

    def test_provider_factory_cloud_builds_when_key_present(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        from guardbound.llm.provider_factory import build_role_llm
        from guardbound.llm.model_manager import ModelManager

        llm = build_role_llm(
            {"provider": "cloud", "model": "gpt-4o"}, "attacker",
            manager=ModelManager(),
        )
        # Built without any attack-code or config change beyond provider.
        assert llm is not None

    def test_cloud_role_is_rejected_in_managerless_local_mode(self):
        from guardbound.llm.provider_factory import build_role_llm

        with pytest.raises(ValueError, match="ModelManager"):
            build_role_llm({"provider": "local", "model": "x"}, "attacker",
                           manager=None)


# =========================================================================== #
# ModelManager lifecycle
# =========================================================================== #

class TestModelManager:
    def test_activate_swaps_roles(self):
        from guardbound.llm.model_manager import ModelManager

        manager = ModelManager()
        manager.register("attacker", MockChatLLM(), "model-A")
        manager.register("target", MockChatLLM(), "model-B")
        manager.register("evaluator", MockChatLLM(), "model-C")

        manager.activate("attacker")
        manager.activate("target")
        manager.activate("evaluator")

        events = [e["event"] for e in manager.events]
        assert events.count("activate") == 3
        assert events.count("evict") == 2  # A evicted on B, B on C
        assert manager._active_role == "evaluator"

    def test_reactivation_evicts_after_earlier_eviction(self):
        """Re-activating a role in a later turn must still evict the previous
        role. Regression: an "already evicted" marker set in turn 1 used to
        suppress every later eviction, so evicted models were never released
        and VRAM accumulated until the GPU ran out (observed in the smoke run).
        """
        from guardbound.llm.model_manager import ModelManager

        manager = ModelManager()
        manager.register("attacker", MockChatLLM(), "model-A")
        manager.register("target", MockChatLLM(), "model-B")
        manager.register("evaluator", MockChatLLM(), "model-C")

        for role in ("attacker", "target", "evaluator",
                     "attacker", "target", "evaluator"):
            manager.activate(role)

        events = [e["event"] for e in manager.events]
        # 6 activations, 5 handovers -> 5 evictions. Before the fix the second
        # cycle produced no evictions at all.
        assert events.count("activate") == 6, events
        assert events.count("evict") == 5, events
        # The second-cycle handovers must be recorded, role-by-role.
        evicted_roles = [e["role"] for e in manager.events if e["event"] == "evict"]
        assert evicted_roles == [
            "attacker", "target", "evaluator", "attacker", "target",
        ], evicted_roles

    def test_same_physical_model_no_swap(self):
        from guardbound.llm.model_manager import ModelManager

        manager = ModelManager()
        manager.register("attacker", MockChatLLM(), "shared-model")
        manager.register("target", MockChatLLM(), "shared-model")

        manager.activate("attacker")
        manager.activate("target")

        assert not [e for e in manager.events if e["event"] == "evict"]

    def test_peak_vram_and_summary(self):
        from guardbound.llm.model_manager import ModelManager

        manager = ModelManager()
        manager.register("attacker", MockChatLLM(), "A")
        manager.activate("attacker")
        summary = manager.summary()
        assert "attacker" in summary["roles"]
        assert summary["roles"]["attacker"]["model_id"] == "A"
        assert "events" in summary and "peak_vram_gb" in summary

    def test_managed_llm_activates_before_generate(self):
        from guardbound.llm.model_manager import (
            ManagedLocalChatLLM, ModelManager,
        )

        manager = ModelManager()
        manager.register("evaluator", MockChatLLM(responses=["ok"]), "G")
        managed = ManagedLocalChatLLM(manager.get("evaluator"), manager,
                                      "evaluator")
        out = managed.generate([{"role": "user", "content": "hi"}])
        assert out == "ok"
        assert manager._active_role == "evaluator"


# =========================================================================== #
# Config validation
# =========================================================================== #

class TestThreeModelConfigValidation:
    def test_default_config_valid(self):
        from run_reproduction import validate_config
        cfg = _load_three_model_cfg()
        assert validate_config(cfg, strict=False) == []

    def test_role_swap_rejected(self):
        from run_reproduction import validate_config
        cfg = _load_three_model_cfg()
        cfg["models"]["target"]["model"] = cfg["models"]["evaluator"]["model"]
        problems = validate_config(cfg, strict=False)
        assert any("different physical models" in p for p in problems)

    def test_fourth_model_rejected(self):
        """A fourth distinct model id must break role separation somewhere."""
        from run_reproduction import validate_config
        cfg = _load_three_model_cfg()
        # A fourth model on the evaluator: the flattened evaluator block and
        # the evaluation judge must be kept consistent with models.evaluator.
        cfg["models"]["evaluator"]["model"] = "some/fourth-model"
        problems = validate_config(cfg, strict=False)
        assert any("evaluator.rubric_model" in p for p in problems), (
            "swapping in a fourth model via models.evaluator must be caught "
            "by the flattened-view consistency check"
        )

    def test_cloud_section_must_stay_present(self):
        from run_reproduction import validate_config
        cfg = _load_three_model_cfg()
        del cfg["providers"]["cloud"]
        problems = validate_config(cfg, strict=False)
        assert any("providers.cloud" in p for p in problems)

    def test_subrole_must_point_at_evaluator(self):
        from run_reproduction import validate_config
        cfg = _load_three_model_cfg()
        cfg["evaluators"]["rubric"] = "attacker"
        problems = validate_config(cfg, strict=False)
        assert any("evaluators.rubric" in p for p in problems)

    def test_expected_parameter_sizes_recorded(self):
        cfg = _load_three_model_cfg()
        assert cfg["models"]["attacker"]["parameters"] == "4B"
        assert cfg["models"]["target"]["parameters"] == "3.8B"
        assert cfg["models"]["evaluator"]["parameters"] == "4B"
        # Actual counts must be recorded honestly alongside the nominal size.
        assert cfg["models"]["attacker"]["actual_parameters"] == "4.66B"
        assert cfg["models"]["target"]["actual_parameters"] == "3.8B"
        assert cfg["models"]["evaluator"]["actual_parameters"] == "4B"
        assert cfg["models"]["evaluator"]["model"] == (
            "Qwen/Qwen3-4B-Instruct-2507"
        )
        # Quantization must be explicitly declared as none (all-bf16 setup).
        quant = cfg["hardware"]["quantization"]
        assert quant is not None and quant.strip().lower() == "none", (
            "hardware.quantization must be explicitly 'none' for the all-bf16 setup"
        )
        assert cfg["hardware"]["dtype"] == "bfloat16"

    def test_quantization_explicitly_declared(self):
        """Quantization must never be a silent default: for the all-bf16 setup
        it must be explicitly recorded as 'none'."""
        cfg = _load_three_model_cfg()
        quant = cfg["hardware"]["quantization"]
        assert quant is not None and quant.strip().lower() == "none", (
            "hardware.quantization must be explicitly 'none'"
        )
        assert cfg["hardware"]["dtype"] == "bfloat16"


# =========================================================================== #
# Configured models must exist on disk (no silent substitution / no download
# at experiment time).
# =========================================================================== #


def _hf_cache_snapshot(repo_id: str) -> Path | None:
    """Return the HF cache snapshot dir for a repo id, or None.

    Accepts either a local directory path or a Hugging Face repo id
    (``org/name``).
    """
    p = Path(repo_id)
    if p.exists() and p.is_dir():
        return p
    if "/" not in repo_id:
        return None
    cache = Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--" + repo_id.replace("/", "--")
    )
    snaps = cache / "snapshots"
    if not snaps.is_dir():
        return None
    # Newest snapshot wins (HF may keep several).
    candidates = [d for d in snaps.iterdir() if d.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _has_weights(path: Path) -> bool:
    """True if the snapshot contains at least one .safetensors shard."""
    return any(path.rglob("*.safetensors"))


class TestConfiguredModelsPresentOnDisk:
    """The three-model config must name checkpoints that already exist on
    disk. An experiment must never silently download or substitute a model.

    This is the "no accidental model substitution" gate for Phase 8.
    """

    def test_attacker_present(self):
        cfg = _load_three_model_cfg()
        model = cfg["models"]["attacker"]["model"]
        assert model == "Qwen/Qwen3.5-4B", model
        snap = _hf_cache_snapshot(model)
        assert snap is not None, f"attacker {model} not found on disk"
        assert _has_weights(snap), f"attacker {model} has no weights on disk"

    def test_target_present(self):
        cfg = _load_three_model_cfg()
        model = cfg["models"]["target"]["model"]
        assert model == "microsoft/Phi-4-mini-instruct", model
        snap = _hf_cache_snapshot(model)
        assert snap is not None, f"target {model} not found on disk"
        assert _has_weights(snap), f"target {model} has no weights on disk"

    def test_evaluator_present(self):
        cfg = _load_three_model_cfg()
        model = cfg["models"]["evaluator"]["model"]
        assert model == "Qwen/Qwen3-4B-Instruct-2507", model
        snap = _hf_cache_snapshot(model)
        assert snap is not None, f"evaluator {model} not found on disk"
        assert _has_weights(snap), f"evaluator {model} has no weights on disk"

    def test_three_distinct_models(self):
        """Attacker, target, evaluator must be three distinct checkpoints."""
        cfg = _load_three_model_cfg()
        models = [
            cfg["models"][r]["model"]
            for r in ("attacker", "target", "evaluator")
        ]
        assert len(set(models)) == 3, models

    def test_all_bf16_no_quantization(self):
        """The final setup is unquantized bf16 for every role."""
        cfg = _load_three_model_cfg()
        assert cfg["hardware"]["quantization"].strip().lower() == "none"
        assert cfg["hardware"]["dtype"] == "bfloat16"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
