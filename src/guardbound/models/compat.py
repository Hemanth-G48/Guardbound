"""Checkpoint compatibility layer for original author NBF checkpoints.

Original checkpoint format (from nbf_original_stuff/):
    {
        "ssm": NeuralStateSpaceModel.state_dict(),  # keys: state_transition.X, observation_model.X
        "nbf": NeuralBarrierFunction.state_dict(),   # keys: nbf.X
    }

Guardbound checkpoint format:
    {
        "predictor_state_dict": SafetyPredictor.state_dict(),  # keys: net.X
        "dynamics_state_dict": DialogueDynamics.state_dict(),   # keys: f_theta.net.X, g_theta.net.X
    }

    The models are architecturally identical. Only the state dict key prefixes differ:
    Original "nbf.0.weight"  → Guardbound "predictor.net.0.weight" (when loading into predictor)
    Original "state_transition.0.weight" → "dynamics.f_theta.net.0.weight"
    Original "observation_model.0.weight" → "dynamics.g_theta.net.0.weight"

    Note: The original NBF state dict saves Sequential layer indices [0,2,4] that include
    the inserted ReLU layers. Guardbound's SafetyPredictor also uses Sequential with ReLU
    at indices [1,3]. The weight keys map directly.

    Also note: The original NeuralBarrierFunction.nn.Module has a bug where the Sequential
    module's ReLU layers are index by position. Guardbound uses the same architecture, so
    weights are directly compatible once keys are remapped.

This module provides:
    - remap_original_state_dict(): convert original keys to Guardbound keys
    - load_original_checkpoint(): load an original checkpoint into Guardbound models
    - verify_conversion(): test that converted weights produce identical outputs
"""

from __future__ import annotations

from pathlib import Path

import torch

from .dynamics import DialogueDynamics
from .predictor import SafetyPredictor


def _remap_original_nbf_keys(state_dict: dict) -> dict:
    """Remap NeuralBarrierFunction keys (nbf.X) to SafetyPredictor keys (net.X).

    Original:  nbf.0.weight, nbf.0.bias, nbf.2.weight, nbf.2.bias, nbf.4.weight, nbf.4.bias
    Guardbound: net.0.weight, net.0.bias, net.2.weight, net.2.bias, net.4.weight, net.4.bias
    """
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("nbf."):
            new_key = key.replace("nbf.", "net.", 1)
            remapped[new_key] = value
        else:
            remapped[key] = value
    return remapped


def _remap_original_ssm_keys(state_dict: dict) -> dict:
    """Remap NeuralStateSpaceModel keys to DialogueDynamics keys.

    Original state_transition.X → f_theta.net.X
    Original observation_model.X → g_theta.net.X
    """
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("state_transition."):
            new_key = "f_theta.net." + key[len("state_transition."):]
            remapped[new_key] = value
        elif key.startswith("observation_model."):
            new_key = "g_theta.net." + key[len("observation_model."):]
            remapped[new_key] = value
        else:
            remapped[key] = value
    return remapped


def remap_original_state_dict(original_ckpt: dict) -> dict:
    """Convert an original author checkpoint dict to Guardbound key format.

    Args:
        original_ckpt: dict with keys "ssm" and/or "nbf"

    Returns:
        dict suitable for loading into DialogueDynamics and SafetyPredictor
    """
    result = {}
    if "ssm" in original_ckpt:
        result["dynamics_state_dict"] = _remap_original_ssm_keys(original_ckpt["ssm"])
    if "nbf" in original_ckpt:
        result["predictor_state_dict"] = _remap_original_nbf_keys(original_ckpt["nbf"])
    return result


def load_original_checkpoint(
    checkpoint_path: str | Path,
    dynamics: DialogueDynamics | None = None,
    predictor: SafetyPredictor | None = None,
    device=None,
) -> tuple[DialogueDynamics, SafetyPredictor]:
    """Load an original author checkpoint into Guardbound models.

    Args:
        checkpoint_path: path to the original .pth file (contains {"ssm": ..., "nbf": ...})
        dynamics: optional pre-created DialogueDynamics (creates default if None)
        predictor: optional pre-created SafetyPredictor (creates default if None)
        device: device to load onto

    Returns:
        (dynamics, predictor) with weights loaded from the original checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    original_ckpt = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)

    if dynamics is None:
        dynamics = DialogueDynamics()
    if predictor is None:
        predictor = SafetyPredictor()

    remapped = remap_original_state_dict(original_ckpt)

    if "dynamics_state_dict" in remapped:
        dynamics.load_state_dict(remapped["dynamics_state_dict"])
    if "predictor_state_dict" in remapped:
        predictor.load_state_dict(remapped["predictor_state_dict"])

    dynamics.eval()
    predictor.eval()

    return dynamics, predictor


def verify_conversion(
    original_ckpt_path: str | Path,
    x: torch.Tensor | None = None,
    u: torch.Tensor | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> dict:
    """Verify that the converted Guardbound models produce identical outputs.

    Creates both an original NeuralStateSpaceModel + NeuralBarrierFunction
    and a converted DialogueDynamics + SafetyPredictor, then compares
    their outputs on identical inputs.

    Args:
        original_ckpt_path: path to original checkpoint
        x: input state tensor [1, 768] (random if None)
        u: input embedding tensor [1, 768] (random if None)
        atol, rtol: tolerances for allclose comparison

    Returns:
        dict with comparison results and pass/fail status
    """
    device = torch.device("cpu")

    # Load original models via importlib (NBF-LLM has hyphen in path).
    # The module executes SentenceTransformer(...) and imports tensorboard's
    # SummaryWriter at import time; stub both so this stays cheap and offline —
    # verify_conversion only needs the architectures.
    import importlib.util
    import sys
    import types

    import sentence_transformers

    class _DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *args, **kwargs):
            raise NotImplementedError("stubbed out in verify_conversion")

    class _DummySummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

    # compat.py lives at src/guardbound/models/compat.py; the repo root is
    # three parents up (models -> guardbound -> src -> repo root).
    orig_train_path = Path(__file__).resolve().parents[3]
    orig_train_path = orig_train_path / "nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/train.py"
    original_st = sentence_transformers.SentenceTransformer
    fake_tb = types.ModuleType("torch.utils.tensorboard")
    fake_tb.SummaryWriter = _DummySummaryWriter
    had_fake_tb = "torch.utils.tensorboard" in sys.modules
    original_tb = sys.modules.get("torch.utils.tensorboard")

    sentence_transformers.SentenceTransformer = _DummySentenceTransformer
    sys.modules["torch.utils.tensorboard"] = fake_tb
    try:
        spec = importlib.util.spec_from_file_location("original_train", orig_train_path)
        original_train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original_train)
    finally:
        sentence_transformers.SentenceTransformer = original_st
        if had_fake_tb:
            sys.modules["torch.utils.tensorboard"] = original_tb
        else:
            sys.modules.pop("torch.utils.tensorboard", None)
    NeuralStateSpaceModel = original_train.NeuralStateSpaceModel
    NeuralBarrierFunction = original_train.NeuralBarrierFunction

    original_ckpt = torch.load(original_ckpt_path, map_location=device, weights_only=False)

    original_ssm = NeuralStateSpaceModel(768, 768, 768, 512)
    original_ssm.load_state_dict(original_ckpt["ssm"])
    original_ssm.eval()

    original_nbf = NeuralBarrierFunction(768, 768, 32, class_num=5)
    original_nbf.load_state_dict(original_ckpt["nbf"])
    original_nbf.eval()

    # Load converted Guardbound models
    dynamics, predictor = load_original_checkpoint(original_ckpt_path)

    # Create test inputs
    if x is None:
        x = torch.randn(1, 768)
    if u is None:
        u = torch.randn(1, 768)
    x = x.to(device)
    u = u.to(device)

    # Original: SSM state transition
    with torch.no_grad():
        orig_x_next, orig_y = original_ssm(x, u)

    # Guardbound: f_theta state transition
    with torch.no_grad():
        guard_x_next = dynamics.f_theta(torch.cat([x, u], dim=-1))

    # Original: NBF logits
    with torch.no_grad():
        orig_nbf_logits = original_nbf(x, u)

    # Guardbound: SafetyPredictor logits
    with torch.no_grad():
        guard_logits = predictor(x, u)

    results = {
        "ssm_state_transition_match": torch.allclose(orig_x_next, guard_x_next, atol=atol, rtol=rtol),
        "ssm_state_transition_max_diff": (orig_x_next - guard_x_next).abs().max().item(),
        "nbf_logits_match": torch.allclose(orig_nbf_logits, guard_logits, atol=atol, rtol=rtol),
        "nbf_logits_max_diff": (orig_nbf_logits - guard_logits).abs().max().item(),
        "ssm_shape_ok": orig_x_next.shape == guard_x_next.shape,
        "nbf_shape_ok": orig_nbf_logits.shape == guard_logits.shape,
        "passed": True,
    }

    # Check each
    if not results["ssm_state_transition_match"]:
        results["passed"] = False
    if not results["nbf_logits_match"]:
        results["passed"] = False

    return results
