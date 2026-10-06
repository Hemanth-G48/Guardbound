"""Phase 15 — attacker-model selection and the GLM-4.6V native backend.

Covers, with no GPU and no weights:
  * profile selection (qwen3 / glm46v / unknown) and its non-mutation contract
  * local backend dispatch (hf_local unchanged, glm4v native, unknown errors)
  * GLM message conversion — exact role ordering preserved
  * the native processor/model path is the one that runs (never AutoModelForCausalLM)
  * JSON behaviour — malformed output stays a failure, no repair/retry/fallback
  * residency modes (pinned keeps the legacy behaviour; sequential evicts)
  * model switching Qwen -> GLM -> Qwen with no stale identity or resident state
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from guardbound.attacks.crescendo_paper import AttackGenerationError, generate_crescendo_step  # noqa: E402
from guardbound.llm import attacker_profiles as ap  # noqa: E402
from guardbound.llm.attacker_profiles import (  # noqa: E402
    AttackerProfileError,
    apply_attacker_profile,
    attacker_runtime_telemetry,
)
from guardbound.llm.base import ChatLLM  # noqa: E402
from guardbound.llm.glm4v_client import GLM4VChatLLM  # noqa: E402
from guardbound.llm.local_client import HFLocalChatLLM  # noqa: E402
from guardbound.llm.model_manager import ManagedLocalChatLLM, ModelManager  # noqa: E402
from guardbound.llm.provider_factory import build_role_llm  # noqa: E402

FROZEN_CFG = _REPO / "configs" / "reproduction_phase14_frozen.yaml"
GLM_CFG = _REPO / "configs" / "attacker_glm46v.yaml"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Never touch the Hub while resolving a model family."""
    import guardbound.llm.local_client as lc

    monkeypatch.setattr(lc, "_maybe_thinking_kwargs", lambda model_id: {})


def frozen_cfg() -> dict:
    with open(FROZEN_CFG, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# 1. Model selection
# --------------------------------------------------------------------------- #
class TestAttackerProfileSelection:
    def test_default_config_reports_the_legacy_qwen_attacker(self):
        rt = attacker_runtime_telemetry(frozen_cfg())
        assert rt["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
        assert rt["backend"] == "hf_local"
        assert rt["residency_mode"] == "pinned"
        assert rt["revision"] is None

    def test_qwen3_profile_matches_the_default_identity(self):
        cfg = frozen_cfg()
        assert (attacker_runtime_telemetry(apply_attacker_profile(cfg, "qwen3"))
                == attacker_runtime_telemetry(cfg))

    def test_glm46v_profile_pins_model_revision_backend_and_residency(self):
        rt = attacker_runtime_telemetry(apply_attacker_profile(frozen_cfg(), "glm46v"))
        assert rt == {
            "model_id": "zai-org/GLM-4.6V-Flash",
            "revision": "411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
            "backend": "glm4v",
            "dtype": "bfloat16",
            "quantization": "none",
            "residency_mode": "sequential",
        }

    def test_profile_does_not_mutate_the_input_config(self):
        cfg = frozen_cfg()
        before = copy.deepcopy(cfg)
        apply_attacker_profile(cfg, "glm46v")
        assert cfg == before, "apply_attacker_profile mutated the caller's config"

    def test_profile_touches_only_the_attacker_block(self):
        cfg = frozen_cfg()
        glm = apply_attacker_profile(cfg, "glm46v")
        assert glm["models"]["target"] == cfg["models"]["target"]
        assert glm["models"]["evaluator"] == cfg["models"]["evaluator"]
        for section in ("nbf", "dataset", "attacks", "evaluation", "embedding",
                        "hardware", "evaluators"):
            assert glm[section] == cfg[section], f"profile changed {section}"

    def test_glm_profile_declares_the_frozen_attacker_template_setting(self):
        """GLM's template implements enable_thinking; the frozen attacker block
        declares it off, and with it left on the model answers the instruction
        block instead of the attacker schema (measured 0/8 vs 8/8)."""
        glm = apply_attacker_profile(frozen_cfg(), "glm46v")
        assert glm["models"]["attacker"]["chat_template_kwargs"] == {
            "enable_thinking": False
        }
        # ...identical to what the frozen Track A attacker already declares
        assert (frozen_cfg()["models"]["attacker"]["chat_template_kwargs"]
                == glm["models"]["attacker"]["chat_template_kwargs"])

    def test_unknown_profile_is_an_explicit_error(self):
        with pytest.raises(AttackerProfileError, match="unknown attacker profile"):
            apply_attacker_profile(frozen_cfg(), "gpt5")

    def test_cli_selector_defaults_to_none_and_accepts_both_names(self):
        import phase14_full_reproduction as p14

        assert p14.parse_args(["--batch", "0"]).attacker_model is None
        assert p14.parse_args(
            ["--batch", "0", "--attacker-model", "qwen3"]).attacker_model == "qwen3"
        assert p14.parse_args(
            ["--batch", "0", "--attacker-model", "glm46v"]).attacker_model == "glm46v"
        with pytest.raises(SystemExit):
            p14.parse_args(["--batch", "0", "--attacker-model", "llama"])

    def test_shipped_glm_config_agrees_with_the_glm_profile(self):
        with open(GLM_CFG, encoding="utf-8") as f:
            glm_cfg = yaml.safe_load(f)
        assert attacker_runtime_telemetry(glm_cfg) == attacker_runtime_telemetry(
            apply_attacker_profile(frozen_cfg(), "glm46v")
        )
        # and the frozen config on disk was not modified by any of this
        disk = frozen_cfg()
        assert disk["models"]["attacker"]["model"] == "Qwen/Qwen3-4B-Instruct-2507"
        assert "residency" not in disk["models"]["attacker"]


# --------------------------------------------------------------------------- #
# 2. Backend dispatch
# --------------------------------------------------------------------------- #
class TestBackendDispatch:
    def _role_cfg(self, **extra):
        cfg = {"provider": "local", "model": "fake/model", "max_new_tokens": 256}
        cfg.update(extra)
        return cfg

    def test_absent_backend_uses_the_existing_pipeline_backend(self):
        mgr = ModelManager()
        llm = build_role_llm(self._role_cfg(), "attacker", manager=mgr)
        assert isinstance(llm, ManagedLocalChatLLM)
        assert isinstance(llm._inner, HFLocalChatLLM)

    def test_explicit_hf_local_alias_is_the_same_backend(self):
        mgr = ModelManager()
        llm = build_role_llm(self._role_cfg(backend="hf_local"), "attacker", manager=mgr)
        assert isinstance(llm._inner, HFLocalChatLLM)

    def test_glm4v_selects_the_native_backend(self):
        mgr = ModelManager()
        llm = build_role_llm(
            self._role_cfg(backend="glm4v", model="zai-org/GLM-4.6V-Flash",
                           revision="abc123", dtype="bfloat16"),
            "attacker", manager=mgr,
        )
        assert isinstance(llm._inner, GLM4VChatLLM)
        assert llm._inner.revision == "abc123"
        assert llm._inner.dtype_name == "bfloat16"

    def test_unknown_backend_is_an_explicit_error(self):
        with pytest.raises(ValueError, match="unknown local backend"):
            build_role_llm(self._role_cfg(backend="vllm"), "attacker",
                           manager=ModelManager())

    def test_revision_on_hf_local_fails_instead_of_being_ignored(self):
        with pytest.raises(ValueError, match="'revision' is not supported"):
            build_role_llm(self._role_cfg(revision="abc123"), "attacker",
                           manager=ModelManager())

    def test_residency_is_registered_from_config(self):
        mgr = ModelManager()
        build_role_llm(self._role_cfg(residency="sequential"), "attacker", manager=mgr)
        assert mgr.residency_mode("attacker") == "sequential"
        assert "attacker" not in mgr.pinned_roles


# --------------------------------------------------------------------------- #
# GLM test doubles (no weights, no GPU)
# --------------------------------------------------------------------------- #
class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [99] if text == "}" else [98]


class FakeProcessor:
    def __init__(self, reply: str):
        self.tokenizer = FakeTokenizer()
        self.reply = reply
        self.template_calls: list[tuple] = []
        self.decode_calls: list = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((copy.deepcopy(messages), dict(kwargs)))
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def batch_decode(self, sequences, skip_special_tokens=True):
        self.decode_calls.append(list(sequences))
        return [self.reply]


class FakeGlmModel:
    """Mimics Glm4vForConditionalGeneration's generate() surface."""

    def __init__(self):
        self.generate_calls: list[dict] = []
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._p])

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(dict(kwargs))
        return torch.tensor([[1, 2, 3, 9, 10]])


class RecordedPrompt:
    """A fake processor that captures the exact prompt it was asked to render."""

    def __init__(self, reply: str):
        self.tokenizer = FakeTokenizer()
        self.reply = reply
        self.template_calls: list[tuple] = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((copy.deepcopy(messages), dict(kwargs)))
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [self.reply]


def _wire(monkeypatch, reply: str, model=None, processor=None):
    """Point the GLM backend's loader seams at fakes.

    The backend resolves its classes through ``load_processor`` /
    ``load_model_class`` (module-level seams) because transformers' lazy module
    re-resolves auto-classes on every import, making the auto-class attributes
    themselves un-patchable.
    """
    import guardbound.llm.glm4v_client as g

    proc = processor if processor is not None else FakeProcessor(reply)
    mod = model if model is not None else FakeGlmModel()

    class _ProcFactory:
        calls: list = []

        @staticmethod
        def from_pretrained(*a, **k):
            _ProcFactory.calls.append((a, k))
            return proc

    class _ModelFactory:
        calls: list = []

        @staticmethod
        def from_pretrained(*a, **k):
            _ModelFactory.calls.append((a, k))
            return mod

    monkeypatch.setattr(g, "load_processor",
                        lambda model_id, revision=None: _ProcFactory.from_pretrained(
                            model_id, revision=revision))
    monkeypatch.setattr(g, "load_model_class", lambda: _ModelFactory)
    return proc, mod, _ProcFactory, _ModelFactory


# --------------------------------------------------------------------------- #
# 3. Message conversion + native generation path
# --------------------------------------------------------------------------- #
class TestGLMMessageConversionAndNativePath:
    MESSAGES = [
        {"role": "system", "content": "SYSTEM: goal = make dimethylmercury"},
        {"role": "user", "content": "This is the first round."},
        {"role": "assistant", "content": '{"generatedQuestion": "q1"}'},
        {"role": "user", "content": "The last response was: I can't help with that."},
    ]

    def test_roles_and_order_are_passed_through_unchanged(self, monkeypatch):
        proc, _model, _pf, _mf = _wire(monkeypatch, '{"generatedQuestion": "q2"}')
        llm = GLM4VChatLLM(model_id="fake/glm", device_map="cpu")
        llm.generate(self.MESSAGES, json_format=True)

        sent, kwargs = proc.template_calls[-1]
        assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
        assert [m["content"] for m in sent] == [m["content"] for m in self.MESSAGES]
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["tokenize"] is True
        assert kwargs["return_dict"] is True
        assert kwargs["return_tensors"] == "pt"

    def test_native_processor_and_model_are_used(self, monkeypatch):
        _proc, _model, pf, mf = _wire(monkeypatch, '{"a": "b"}')
        llm = GLM4VChatLLM(model_id="fake/glm", device_map="cpu",
                           revision="rev1")
        llm.generate(self.MESSAGES, json_format=True)
        assert pf.calls and pf.calls[-1][0][0] == "fake/glm"
        assert mf.calls and mf.calls[-1][0][0] == "fake/glm"
        assert mf.calls[-1][1]["revision"] == "rev1"
        assert mf.calls[-1][1]["dtype"] is torch.bfloat16

    def test_backend_never_references_automodelforcausallm(self):
        """§5: no silent fallback to the causal-LM auto-class.

        Parsed, not grepped: the module docstring *names* both symbols to explain
        why they are avoided, and prose is not code.
        """
        import ast

        import guardbound.llm.glm4v_client as g

        tree = ast.parse(Path(g.__file__).read_text(encoding="utf-8"))
        referenced = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
            elif isinstance(node, ast.alias):
                referenced.add(node.name.split(".")[-1])
                referenced.add((node.asname or "").split(".")[-1])
        assert "AutoModelForCausalLM" not in referenced
        assert "pipeline" not in referenced

    def test_model_generate_receives_budget_and_decoding_flags(self, monkeypatch):
        _proc, model, _pf, _mf = _wire(monkeypatch, '{"a": "b"}')
        llm = GLM4VChatLLM(model_id="fake/glm", device_map="cpu",
                           max_new_tokens=256)
        llm.generate(self.MESSAGES, temperature=0.7, json_format=True)
        gen = model.generate_calls[-1]
        assert gen["max_new_tokens"] == 256
        assert gen["do_sample"] is True
        assert gen["temperature"] == 0.7
        assert "input_ids" in gen and "attention_mask" in gen

    def test_constrained_json_reuses_the_frozen_decoder(self, monkeypatch):
        _proc, model, _pf, _mf = _wire(monkeypatch, '{"a": "b"}')
        llm = GLM4VChatLLM(model_id="fake/glm", device_map="cpu",
                           structured_output_mode="constrained_json")
        llm.generate(self.MESSAGES, json_format=True)
        gen = model.generate_calls[-1]
        assert "logits_processor" in gen and gen["logits_processor"]
        assert "stopping_criteria" in gen and gen["stopping_criteria"]

    def test_unconstrained_mode_has_no_logits_processor(self, monkeypatch):
        _proc, model, _pf, _mf = _wire(monkeypatch, '{"a": "b"}')
        llm = GLM4VChatLLM(model_id="fake/glm", device_map="cpu")
        llm.generate(self.MESSAGES, json_format=True)
        assert "logits_processor" not in model.generate_calls[-1]

    def test_max_turns_context_keeps_system_and_recent_turns(self, monkeypatch):
        proc, _model, _pf, _mf = _wire(monkeypatch, '{"a": "b"}')
        llm = GLM4VChatLLM(model_id="fake/glm", device_map="cpu")
        llm.generate(self.MESSAGES, max_turns_context=1)
        sent, _ = proc.template_calls[-1]
        assert sent[0]["role"] == "system"
        assert len(sent) == 3  # system + the last (assistant, user) pair


# --------------------------------------------------------------------------- #
# 4. JSON behaviour — the frozen parser stays authoritative
# --------------------------------------------------------------------------- #
class TestGLMJSONBehaviour:
    def test_valid_json_returns_a_dict(self, monkeypatch):
        _wire(monkeypatch, '{"generatedQuestion": "q", "lastResponseSummary": ""}')
        llm = GLM4VChatLLM(model_id="fake/glm", device_map="cpu")
        out = llm.generate([{"role": "user", "content": "hi"}], json_format=True)
        assert isinstance(out, dict) and out["generatedQuestion"] == "q"

    def test_malformed_json_is_returned_raw_as_a_string(self, monkeypatch):
        raw = '{\n"generatedQuestion": "q",\n"\n"lastResponseSummary": ""\n}'
        _wire(monkeypatch, raw)
        llm = GLM4VChatLLM(model_id="fake/glm", device_map="cpu")
        out = llm.generate([{"role": "user", "content": "hi"}], json_format=True)
        assert isinstance(out, str) and out == raw, (
            "malformed JSON must be returned verbatim — no repair, no re-generation"
        )

    def test_malformed_json_fails_through_the_real_attack_interface(self, monkeypatch):
        raw = '{\n"generatedQuestion": "q",\n"\n"lastResponseSummary": ""\n}'
        _wire(monkeypatch, raw)
        llm = GLM4VChatLLM(model_id="fake/glm", device_map="cpu")

        with pytest.raises(AttackGenerationError) as exc:
            generate_crescendo_step(
                round_num=1, goal="goal", history_attacker=[], history_target=[],
                scores=[], last_response="", attacker_llm=llm, max_rounds=8,
            )
        # raw attacker output preserved in the failure, exactly as for Qwen
        assert "lastResponseSummary" in str(exc.value)

    def test_no_second_generation_attempt_is_made(self, monkeypatch):
        raw = "{not json"
        _proc, model, _pf, _mf = _wire(monkeypatch, raw)
        llm = GLM4VChatLLM(model_id="fake/glm", device_map="cpu")
        llm.generate([{"role": "user", "content": "hi"}], json_format=True)
        assert len(model.generate_calls) == 1, "a retry was issued"

    def test_missing_field_is_not_invented(self, monkeypatch):
        _wire(monkeypatch, '{"generatedQuestion": "q"}')
        llm = GLM4VChatLLM(model_id="fake/glm", device_map="cpu")
        with pytest.raises(AttackGenerationError, match="missing"):
            generate_crescendo_step(
                round_num=1, goal="goal", history_attacker=[], history_target=[],
                scores=[], last_response="", attacker_llm=llm, max_rounds=8,
            )

    def test_embedded_json_uses_the_existing_structural_extraction(self, monkeypatch):
        """Pre-existing frozen behaviour (shared with Qwen) — not new repair."""
        _wire(monkeypatch, 'Sure!\n{"generatedQuestion": "q", "lastResponseSummary": ""}\n')
        llm = GLM4VChatLLM(model_id="fake/glm", device_map="cpu")
        out = llm.generate([{"role": "user", "content": "hi"}], json_format=True)
        assert isinstance(out, dict) and out["generatedQuestion"] == "q"

    def test_reasoning_block_is_not_stripped(self, monkeypatch):
        raw = "<think>reasoning here</think>\n"
        _wire(monkeypatch, raw)
        llm = GLM4VChatLLM(model_id="fake/glm", device_map="cpu")
        out = llm.generate([{"role": "user", "content": "hi"}], json_format=True)
        assert out == raw, "the adapter must not special-case reasoning output"


# --------------------------------------------------------------------------- #
# 5. Residency
# --------------------------------------------------------------------------- #
class FakeBackend(ChatLLM):
    """Minimal load/release contract without weights (mirrors HFLocalChatLLM)."""

    name = "fake-local"

    def __init__(self, model_id: str, device_map: str = "cuda"):
        self.model_id = model_id
        self.device_map = device_map
        self.structured_output_mode = None
        self._pipeline = None
        self._tokenizer = None
        self.load_calls = 0

    def _get_pipeline(self):
        self.load_calls += 1
        self._pipeline = {"model_id": self.model_id}
        return self._pipeline

    def generate(self, messages, temperature=0.7, **kwargs):
        return "fake"


class TestResidency:
    def _manager(self, free=100.0, monkeypatch=None):
        import guardbound.llm.model_manager as mm

        if monkeypatch is not None:
            monkeypatch.setattr(mm, "_free_vram_gb", lambda: free)
        mgr = ModelManager()
        return mgr

    def test_default_residency_is_pinned(self, monkeypatch):
        mgr = self._manager(monkeypatch=monkeypatch)
        mgr.register("attacker", FakeBackend("qwen"), "qwen")
        assert mgr.residency_mode("attacker") == "pinned"
        mgr.pin("attacker")
        mgr.activate("attacker")
        mgr.register("target", FakeBackend("phi"), "phi")
        mgr.activate("target")
        assert mgr.resident_model_ids == ["phi", "qwen"], (
            "a pinned attacker must survive another role's activation"
        )

    def test_sequential_attacker_is_evicted_for_the_target(self, monkeypatch):
        mgr = self._manager(monkeypatch=monkeypatch)
        mgr.register("attacker", FakeBackend("glm"), "glm")
        mgr.set_residency("attacker", "sequential")
        mgr.activate("attacker")
        assert mgr.resident_model_ids == ["glm"]

        mgr.register("target", FakeBackend("phi"), "phi")
        mgr.activate("target")
        assert mgr.resident_model_ids == ["phi"], (
            "sequential residency must leave exactly one resident model"
        )
        assert "attacker" not in mgr.pinned_roles

    def test_sequential_role_reloads_when_it_returns(self, monkeypatch):
        mgr = self._manager(monkeypatch=monkeypatch)
        attacker = FakeBackend("glm")
        mgr.register("attacker", attacker, "glm")
        mgr.set_residency("attacker", "sequential")
        mgr.register("target", FakeBackend("phi"), "phi")
        mgr.activate("target")
        for role in ("attacker", "target", "attacker"):
            mgr.ensure_resident(role)
        assert attacker.load_calls == 2, "the released attacker must reload"
        assert mgr.resident_model_ids == ["glm"]

    def test_release_sequential_frees_only_sequential_roles(self, monkeypatch):
        """Run-boundary release: a 20.6 GB sequential attacker must not sit on
        the card between runs, but a pinned stack must be left alone."""
        mgr = self._manager(monkeypatch=monkeypatch)
        mgr.register("attacker", FakeBackend("glm"), "glm")
        mgr.set_residency("attacker", "sequential")
        mgr.register("target", FakeBackend("phi"), "phi")
        mgr.set_residency("target", "pinned")
        # Pinned target first: activating the sequential attacker afterwards
        # must NOT evict it (pins are absolute).
        mgr.ensure_resident("target")
        mgr.ensure_resident("attacker")
        assert mgr.resident_model_ids == ["glm", "phi"]

        released = mgr.release_sequential()
        assert released == ["attacker"]
        assert mgr.resident_model_ids == ["phi"], "the pinned role must survive"
        assert mgr.release_sequential() == [], "nothing sequential left"

    def test_release_sequential_is_a_noop_for_a_pinned_stack(self, monkeypatch):
        """The frozen Track A stack must be untouched by the run-boundary hook."""
        mgr = self._manager(monkeypatch=monkeypatch)
        mgr.register("attacker", FakeBackend("qwen"), "qwen")
        mgr.set_residency("attacker", "pinned")
        mgr.ensure_resident("attacker")
        assert mgr.release_sequential() == []
        assert mgr.resident_model_ids == ["qwen"]

    def test_invalid_residency_mode_is_rejected(self):
        mgr = ModelManager()
        mgr.register("attacker", FakeBackend("glm"), "glm")
        with pytest.raises(ValueError, match="unknown residency mode"):
            mgr.set_residency("attacker", "sometimes")

    def test_unknown_role_is_rejected(self):
        mgr = ModelManager()
        with pytest.raises(KeyError):
            mgr.set_residency("nobody", "sequential")


# --------------------------------------------------------------------------- #
# 6. Qwen -> GLM -> Qwen switching
# --------------------------------------------------------------------------- #
class TestModelSwitching:
    """The attacker ROLE is re-pointed between models, as a profile switch does.

    Both attackers share the role name "attacker" because that is what a single
    process sees when the selection changes — never two simultaneous attackers.
    """

    def _manager(self, monkeypatch, free=100.0):
        import guardbound.llm.model_manager as mm

        monkeypatch.setattr(mm, "_free_vram_gb", lambda: free)
        return ModelManager()

    def _select(self, mgr, name):
        """Bind the attacker role to a model, exactly as build_role_llm does."""
        if name == "qwen":
            backend = FakeBackend("Qwen/Qwen3-4B-Instruct-2507")
            mgr.register("attacker", backend, "Qwen/Qwen3-4B-Instruct-2507")
            mgr.set_residency("attacker", "pinned")
        else:
            backend = FakeBackend("zai-org/GLM-4.6V-Flash")
            mgr.register("attacker", backend, "zai-org/GLM-4.6V-Flash")
            mgr.set_residency("attacker", "sequential")
        return backend

    def test_qwen_glm_qwen_cycle_leaves_no_stale_state(self, monkeypatch):
        mgr = self._manager(monkeypatch)

        qwen = self._select(mgr, "qwen")
        mgr.ensure_resident("attacker")
        assert mgr.resident_model_ids == ["Qwen/Qwen3-4B-Instruct-2507"]

        glm = self._select(mgr, "glm")
        mgr.ensure_resident("attacker")
        assert mgr.resident_model_ids == ["zai-org/GLM-4.6V-Flash"]
        assert qwen._pipeline is None, "the previous attacker must be released"

        qwen2 = self._select(mgr, "qwen")
        mgr.ensure_resident("attacker")
        assert mgr.resident_model_ids == ["Qwen/Qwen3-4B-Instruct-2507"]
        assert glm._pipeline is None and qwen2._pipeline is not None

    def test_switching_does_not_leak_models(self, monkeypatch):
        mgr = self._manager(monkeypatch)
        for name in ("qwen", "glm", "qwen", "glm", "qwen"):
            self._select(mgr, name)
            mgr.ensure_resident("attacker")
        assert mgr.residency_snapshot()["resident_count"] <= 1
        loads = [e for e in mgr.events if e["event"] == "load"]
        evicts = [e for e in mgr.events if e["event"] == "evict"]
        assert len(loads) - len(evicts) <= 1, (len(loads), len(evicts))

    def test_identity_is_never_confused_between_backends(self, monkeypatch):
        mgr = self._manager(monkeypatch)
        allowed = {None, "Qwen/Qwen3-4B-Instruct-2507", "zai-org/GLM-4.6V-Flash"}
        for name in ("qwen", "glm", "qwen"):
            self._select(mgr, name)
            mgr.ensure_resident("attacker")
            assert set(mgr.resident_model_ids) <= {mgr._model_ids["attacker"]}
            for event in mgr.events:
                assert event.get("model_id") in allowed

    def test_qwen_pin_is_restored_after_a_glm_episode(self, monkeypatch):
        """Sequential residency must not leave the role unpinned when Qwen returns."""
        mgr = self._manager(monkeypatch)
        self._select(mgr, "glm")
        mgr.ensure_resident("attacker")
        assert mgr.pinned_roles == []

        self._select(mgr, "qwen")
        mgr.ensure_resident("attacker")
        assert mgr.pinned_roles == ["attacker"]
        assert mgr.residency_mode("attacker") == "pinned"
