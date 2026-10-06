"""Phase 17 Stage 3 — native Qwen3.8-27B backend tests.

Two properties are pinned here:

  * the Stage 3 additions to ``MultimodalChatLLM`` are **opt-in** — every
    existing caller (GLM-4.6V, Ornith-1.5-9B, the frozen config) keeps its
    previous defaults, so the frozen experiments are untouched;
  * the native backend resolves the class the checkpoint declares and refuses
    anything else, rather than silently accepting a substitution.

Everything is offline: no weights are loaded, no GPU is touched.
"""
from __future__ import annotations

import pytest

from guardbound.llm.attacker_profiles import (
    BACKEND_QWEN38_NATIVE,
    apply_attacker_profile,
    profile_names,
)
from guardbound.llm.model_manager import ManagedLocalChatLLM, ModelManager
from guardbound.llm.multimodal_client import MultimodalChatLLM
from guardbound.llm.qwen38_native_client import (
    QWEN38_DECLARED_ARCHITECTURE,
    QWEN38_MODEL_ID,
    QWEN38_REVISION,
    Qwen38NativeArchitectureError,
    Qwen38NativeChatLLM,
    verify_native_architecture,
)


class _StubConfig:
    def __init__(self, architectures):
        self.architectures = architectures


class _StubModel:
    """Just enough model for the architecture assertion."""

    def __init__(self, class_name: str, architectures):
        self.__class__.__name__ = class_name  # type: ignore[misc]
        self.config = _StubConfig(architectures)


class TestStage3AdditionsAreOptIn:
    def test_multimodal_base_defaults_unchanged(self):
        from guardbound.llm.ornith15_client import Ornith15ChatLLM

        llm = Ornith15ChatLLM()
        assert llm.quantization is None
        assert llm.max_new_tokens == 256
        assert llm.top_p is None
        assert llm.do_sample is None
        assert llm.last_generation_stats is None

    def test_glm_backend_defaults_unchanged(self):
        from guardbound.llm.glm4v_client import GLM4VChatLLM

        llm = GLM4VChatLLM()
        assert llm.quantization is None
        assert llm.max_new_tokens == 256

    def test_unsupported_quantization_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported quantization"):
            Qwen38NativeChatLLM(quantization="awq")


class TestNativeBackend:
    def test_defaults_are_the_pinned_checkpoint(self):
        llm = Qwen38NativeChatLLM()
        assert llm.model_id == QWEN38_MODEL_ID
        assert llm.revision == QWEN38_REVISION
        assert llm.label == "Qwen3.8-27B (native)"

    def test_identifier_is_the_declared_architecture(self):
        assert QWEN38_DECLARED_ARCHITECTURE == "Qwen3_5ForConditionalGeneration"

    def test_verify_accepts_the_declared_class(self):
        model = _StubModel("Qwen3_5ForConditionalGeneration", [QWEN38_DECLARED_ARCHITECTURE])
        actual, declared = verify_native_architecture(model)
        assert actual == QWEN38_DECLARED_ARCHITECTURE
        assert declared == [QWEN38_DECLARED_ARCHITECTURE]

    def test_verify_rejects_the_causal_lm_substitution(self):
        """The Stage 2 class must not pass as the native architecture."""
        model = _StubModel("Qwen3_5ForCausalLM", [QWEN38_DECLARED_ARCHITECTURE])
        with pytest.raises(Qwen38NativeArchitectureError, match="does not declare"):
            verify_native_architecture(model)

    def test_backend_records_the_loaded_identity(self):
        llm = Qwen38NativeChatLLM()
        llm._verify_model(
            _StubModel("Qwen3_5ForConditionalGeneration", [QWEN38_DECLARED_ARCHITECTURE])
        )
        assert llm.loaded_class == "Qwen3_5ForConditionalGeneration"
        assert llm.declared_architectures == [QWEN38_DECLARED_ARCHITECTURE]

    def test_architecture_error_carries_the_failure_class(self):
        assert Qwen38NativeArchitectureError.failure_class == "MODEL_INTERFACE_INCOMPATIBLE"


class TestProviderFactoryWiring:
    @staticmethod
    def _role_cfg(**overrides):
        cfg = {
            "provider": "local",
            "backend": "qwen38_native",
            "model": QWEN38_MODEL_ID,
            "revision": QWEN38_REVISION,
            "device_map": {"": 0},
            "max_new_tokens": None,
            "top_p": 1.0,
            "quantization": "nf4",
            "residency": "sequential",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        cfg.update(overrides)
        return cfg

    def test_factory_builds_the_native_backend(self):
        from guardbound.llm.provider_factory import build_role_llm

        manager = ModelManager()
        llm = build_role_llm(self._role_cfg(), "attacker", manager=manager)
        assert isinstance(llm, ManagedLocalChatLLM)
        inner = manager.get("attacker")
        assert isinstance(inner, Qwen38NativeChatLLM)
        assert inner.quantization == "nf4"
        assert inner.max_new_tokens is None
        assert inner.top_p == 1.0
        assert inner.chat_template_kwargs == {"enable_thinking": False}
        assert manager.residency_mode("attacker") == "sequential"

    def test_backend_aliases_resolve(self):
        from guardbound.llm.provider_factory import _norm_local_backend

        for alias in ("qwen38_native", "qwen3_8_native", "qwen38"):
            assert _norm_local_backend(alias) == BACKEND_QWEN38_NATIVE

    def test_unknown_backend_still_fails_loudly(self):
        from guardbound.llm.provider_factory import _norm_local_backend

        with pytest.raises(ValueError, match="unknown local backend"):
            _norm_local_backend("qwen38_native_typo")


class TestAttackerProfile:
    def test_profile_is_registered(self):
        assert "qwen38_native" in profile_names()

    def test_profile_selects_the_native_path_and_preserves_generation_settings(self):
        cfg = {
            "models": {
                "attacker": {
                    "provider": "local",
                    "model": "Qwen/Qwen3-4B-Instruct-2507",
                    "max_new_tokens": None,
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "do_sample": True,
                }
            }
        }
        updated = apply_attacker_profile(cfg, "qwen38_native")["models"]["attacker"]
        assert updated["backend"] == BACKEND_QWEN38_NATIVE
        assert updated["model"] == QWEN38_MODEL_ID
        assert updated["revision"] == QWEN38_REVISION
        assert updated["quantization"] == "nf4"
        assert updated["residency"] == "sequential"
        # Generation parameters are the experiment's, not the profile's.
        assert updated["max_new_tokens"] is None
        assert updated["temperature"] == 0.7
        assert updated["top_p"] == 1.0
        assert updated["do_sample"] is True
        # The caller's config must not be mutated.
        assert cfg["models"]["attacker"]["model"] == "Qwen/Qwen3-4B-Instruct-2507"
