"""Shared helpers for official-vs-Guardbound parity tests (Phase 3).

Everything here is offline: the official NBF code is imported via importlib
with the module-level ``SentenceTransformer``/``tensorboard`` side effects
stubbed out, and the official checkpoint is loaded with ``weights_only=False``
on CPU.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
ORIGINAL_ROOT = REPO / "nbf_original_stuff" / "orginal_code_give_by_author" / "NBF-LLM"
ORIGINAL_CKPT = ORIGINAL_ROOT / "models" / "models_best_nbf_released.pth"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# --------------------------------------------------------------------------- #
# Official module loading (with side-effect stubs)
# --------------------------------------------------------------------------- #

class _DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, *args, **kwargs):
        raise NotImplementedError("SentenceTransformer stubbed in parity tests")


class _DummySummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *a, **k):
        pass

    def add_scalars(self, *a, **k):
        pass

    def add_histogram(self, *a, **k):
        pass

    def close(self):
        pass


def _install_stubs() -> None:
    """Stub module-level side effects of the official ``train.py``."""
    import sentence_transformers

    fake_tb = types.ModuleType("torch.utils.tensorboard")
    fake_tb.SummaryWriter = _DummySummaryWriter
    sys.modules["torch.utils.tensorboard"] = fake_tb
    sentence_transformers.SentenceTransformer = _DummySentenceTransformer
    if str(ORIGINAL_ROOT) not in sys.path:
        sys.path.insert(0, str(ORIGINAL_ROOT))


def load_official_train():
    """Import the official ``NBF-LLM/train.py`` (NeuralStateSpaceModel,
    NeuralBarrierFunction) with module-level side effects stubbed."""
    _install_stubs()
    spec = importlib.util.spec_from_file_location("official_nbf_train", ORIGINAL_ROOT / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_attack(attack: str):
    """Import the official ``attacks/<attack>/run.py`` module (ground truth)."""
    _install_stubs()
    return importlib.import_module(f"attacks.{attack}.run")


def load_official_build():
    """Import the official ``attacks/build_ssm_nbf.py`` (calculate_score*)."""
    _install_stubs()
    return importlib.import_module("attacks.build_ssm_nbf")


def load_official_utils():
    """Import the official ``attacks/utils`` package (check_refusal, rubric, disclaimer)."""
    _install_stubs()
    return importlib.import_module("attacks.utils")


# --------------------------------------------------------------------------- #
# Models from the original checkpoint
# --------------------------------------------------------------------------- #

def load_original_models(device: str | torch.device = "cpu"):
    """Original NeuralStateSpaceModel + NeuralBarrierFunction, official weights.

    Returns (ssm, nbf) in eval mode on CPU.
    """
    train = load_official_train()
    ckpt = torch.load(ORIGINAL_CKPT, map_location="cpu", weights_only=False)
    ssm = train.NeuralStateSpaceModel(768, 768, 768, 512)
    ssm.load_state_dict(ckpt["ssm"])
    ssm.eval()
    nbf = train.NeuralBarrierFunction(768, 768, 32, class_num=5)
    nbf.load_state_dict(ckpt["nbf"])
    nbf.eval()
    if str(device) != "cpu":
        ssm.to(device)
        nbf.to(device)
    return ssm, nbf


def load_guardbound_models(device: str | torch.device = "cpu"):
    """Guardbound DialogueDynamics + SafetyPredictor with the same weights."""
    from guardbound.models.compat import load_original_checkpoint

    return load_original_checkpoint(ORIGINAL_CKPT, device=device)


def make_barrier(dynamics, predictor):
    """Bundle dynamics + predictor into the runner-style barrier object."""
    from guardbound.models.predictor import NeuralBarrierFunction

    return NeuralBarrierFunction(dynamics=dynamics, predictor=predictor)


# --------------------------------------------------------------------------- #
# Deterministic embedding (both APIs)
# --------------------------------------------------------------------------- #

class DeterministicEmbed:
    """Deterministic 768-dim embedder.

    Serves both call conventions:
      official   calculate_score_from_dialog:  embed.encode(text, convert_to_tensor=True)
      guardbound score_query_from_dialog:      embed(text)
    """

    def __init__(self, dim: int = 768, scale: float = 0.1):
        self.dim = dim
        self.scale = scale
        self._vectors: dict[str, torch.Tensor] = {}

    def _vec(self, text: str) -> torch.Tensor:
        if text not in self._vectors:
            digest = hashlib.md5(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "little")
            g = torch.Generator().manual_seed(seed)
            self._vectors[text] = torch.randn(self.dim, generator=g) * self.scale
        return self._vectors[text]

    def encode(self, text: str, convert_to_tensor: bool = True) -> torch.Tensor:
        return self._vec(text)

    def __call__(self, text: str) -> torch.Tensor:
        return self._vec(text)


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #

def assert_allclose(a, b, label: str, atol: float = 1e-6, rtol: float = 1e-5) -> bool:
    """Assert numerical parity and report the first point of divergence."""
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    if a.shape != b.shape:
        raise AssertionError(f"[parity] {label}: shape {tuple(a.shape)} != {tuple(b.shape)}")
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    if not ok:
        diff = (a - b).abs().max().item()
        raise AssertionError(
            f"[parity] {label}: max abs diff {diff:.3e} exceeds atol={atol}, rtol={rtol}"
        )
    return True


def cosmetic_normalize(text: str) -> str:
    """Apply the documented cosmetic normalizations used to compare prompt text.

    Official and Guardbound prompt strings differ only in:
      1. curly apostrophe U+2019 vs ASCII apostrophe (Acronym / NETWORK_PROMPT)
      2. trailing-whitespace-only lines before section headers (Opposite Day)
      3. a handful of single spaces (EXTRACT/QUERIES prompts)
    The normalizer collapses those so substantive content can be compared.
    """
    text = text.replace("\u2019", "'")
    lines = [line.rstrip() for line in text.split("\n")]
    lines = [line for line in lines if line.strip()]
    return "\n".join(lines)


class CopyingMock:
    """Scripted mock that records a DEEP COPY of messages at call time.

    ``MockChatLLM`` records the live list reference, which callers mutate
    afterwards, so recorded messages drift from what the model actually
    received. This wrapper snapshots each call.
    """

    def __init__(self, responses: list[str] | None = None):
        from guardbound.llm.mock import MockChatLLM

        self.inner = MockChatLLM(responses=responses)
        self.calls: list[tuple[list[dict], float, bool]] = []

    def generate(
        self,
        messages,
        temperature: float = 0.7,
        max_turns_context=None,
        json_format: bool = False,
    ):
        self.calls.append(([dict(m) for m in messages], temperature, json_format))
        return self.inner.generate(
            messages,
            temperature=temperature,
            max_turns_context=max_turns_context,
            json_format=json_format,
        )


def scripted_attacker(responses: list[str]):
    """Official-style ``attacker_generate`` with recorded calls.

    Returns (fn, calls) where ``calls`` collects
    ``(messages, json_format, temperature)`` per invocation.
    """
    calls = []
    items = list(responses)

    def fn(messages, json_format=False, temperature=0.7):
        calls.append(([dict(m) for m in messages], json_format, temperature))
        raw = items.pop(0) if items else "{}"
        if json_format:
            return json.loads(raw)
        return raw

    return fn, calls


def same_messages_cosmetic(a: list[dict], b: list[dict], label: str = "attacker messages") -> bool:
    """Assert message parity after the documented cosmetic normalization.

    Used where the official prompt text differs from Guardbound only by
    curly-vs-ASCII apostrophes or trailing whitespace-only lines (Acronym,
    Opposite Day system prompts).
    """
    a_n = [{"role": m["role"], "content": cosmetic_normalize(m["content"])} for m in a]
    b_n = [{"role": m["role"], "content": cosmetic_normalize(m["content"])} for m in b]
    return same_messages(a_n, b_n, label + " (cosmetic-normalized)")


def same_messages(a: list[dict], b: list[dict], label: str = "attacker messages") -> bool:
    """Assert two message lists are identical (roles + contents)."""
    if len(a) != len(b):
        raise AssertionError(
            f"[parity] {label}: message count {len(a)} != {len(b)}\n"
            f"  official: {[m['role'] for m in a]}\n"
            f"  guardbound: {[m['role'] for m in b]}"
        )
    for i, (ma, mb) in enumerate(zip(a, b)):
        if ma["role"] != mb["role"]:
            raise AssertionError(f"[parity] {label}: msg {i} role {ma['role']} != {mb['role']}")
        if ma["content"] != mb["content"]:
            raise AssertionError(
                f"[parity] {label}: msg {i} ({ma['role']}) content differs:\n"
                f"  official:   {ma['content'][:200]!r}\n"
                f"  guardbound: {mb['content'][:200]!r}"
            )
    return True