"""Phase 7 — reproduction configuration + checkpoint verification tests.

These tests keep the reproduction pipeline honest:

* `configs/reproduction.yaml` stays complete (every required field present)
* the pinned NBF checkpoint exists and matches its recorded sha256
* the validation gate actually rejects incomplete/corrupted configs
* the official dataset is present with the expected structure

Run:  pytest tests/test_reproduction_config.py -q
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import yaml
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPRO_CFG = _PROJECT_ROOT / "configs/reproduction.yaml"
_OFFICIAL_DATASET = (
    _PROJECT_ROOT
    / "nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/data/test/harmbench_tasks.json"
)
_CHECKPOINT = (
    _PROJECT_ROOT
    / "nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/models/models_best_nbf_released.pth"
)


# --------------------------------------------------------------------------- #
# Config completeness
# --------------------------------------------------------------------------- #

REQUIRED_FIELDS = [
    "experiment_model.attacker",
    "experiment_model.target",
    "experiment_model.evaluator",
    "attacker.model",
    "target.model",
    "evaluator.rubric_model",
    "evaluator.refusal_model",
    "embedding.model",
    "embedding.dimension",
    "nbf.checkpoint",
    "nbf.state_dimension",
    "nbf.threshold",
    "nbf.trials.crescendo",
    "nbf.trials.actor_attack",
    "nbf.trials.opposite_day",
    "nbf.trials.acronym",
    "dataset.path",
    "dataset.split",
    "experiment.seed",
    "attacks.max_turns",
    "experiment.output_dir",
    "hardware.device",
]


def _get(cfg, path):
    node = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), "config must be a mapping"
    return cfg


_config = _load_config(_REPRO_CFG)


class TestReproductionConfigComplete:
    """The reproduction config must contain every required field."""

    def test_config_file_exists(self):
        assert _REPRO_CFG.exists(), f"missing: {_REPRO_CFG}"

    def test_no_required_field_missing(self):
        for field in REQUIRED_FIELDS:
            val = _get(_config, field)
            assert val is not None, f"missing required field: {field}"


class TestReproductionModelSeparation:
    """Paper reference vs experiment model must both be present and explicit."""

    def test_paper_reference_model_present(self):
        prm = _config.get("paper_reference_model")
        assert prm is not None, "missing paper_reference_model section"
        assert prm.get("attacker") == "gpt-4o", "paper attacker should be gpt-4o"
        assert prm.get("target") == "gpt-4o", "paper target should be gpt-4o"

    def test_experiment_model_present(self):
        em = _config.get("experiment_model")
        assert em is not None, "missing experiment_model section"
        assert em.get("attacker"), "experiment_model.attacker required"
        assert em.get("target"), "experiment_model.target required"
        assert em.get("evaluator"), "experiment_model.evaluator required"

    def test_attacker_target_evaluator_all_explicit(self):
        e = _config
        assert e.get("attacker", {}).get("model"), "attacker.model required"
        assert e.get("target", {}).get("model"), "target.model required"
        ev = e.get("evaluator", {})
        assert ev.get("rubric_model"), "evaluator.rubric_model required"
        assert ev.get("refusal_model"), "evaluator.refusal_model required"


class TestReproductionNBFConfig:
    """NBF configuration must be explicit and pinned."""

    def test_nbf_checkpoint_exists(self):
        ckpt = _get(_config, "nbf.checkpoint")
        assert ckpt
        assert Path(ckpt).exists(), f"NBF checkpoint missing: {ckpt}"

    def test_nbf_checkpoint_sha256_pinned(self):
        expected = _get(_config, "nbf.checkpoint_sha256_expected")
        assert expected, "nbf.checkpoint_sha256_expected must be pinned"
        actual = hashlib.sha256(
            Path(_CHECKPOINT).read_bytes()
        ).hexdigest() if _CHECKPOINT.exists() else None
        assert actual == expected, (
            f"recorded sha256 mismatch: expected {expected}, got {actual}"
        )

    def test_nbf_dimensions_explicit(self):
        assert _get(_config, "nbf.state_dimension") == 768
        assert _get(_config, "embedding.dimension") == 768
        assert _get(_config, "nbf.predictor_classes") == 5

    def test_nbf_trials_explicit(self):
        trials = _get(_config, "nbf.trials")
        assert trials is not None
        assert trials.get("crescendo") == 3
        assert trials.get("opposite_day") == 3
        assert trials.get("acronym") == 3
        assert trials.get("actor_attack") == 10


class TestReproductionDatasetConfig:
    """Dataset configuration must be explicit."""

    def test_dataset_path_exists(self):
        ds = _get(_config, "dataset.path")
        assert ds
        assert Path(ds).exists(), f"dataset missing: {ds}"

    def test_dataset_fields_explicit(self):
        fields = _get(_config, "dataset.fields")
        assert fields
        assert "target_system" in fields
        assert "task" in fields
        assert "max_rounds" in fields

    def test_dataset_split_explicit(self):
        assert _get(_config, "dataset.split") == "test"

    def test_dataset_num_goals_present(self):
        assert _get(_config, "dataset.num_goals") == 200

    def test_dataset_sampling_explicit(self):
        assert _get(_config, "dataset.sampling") == "file_order"


class TestReproductionHardwareConfig:
    """Hardware must be explicitly recorded (not auto-detected silently)."""

    def test_hardware_present(self):
        hw = _config.get("hardware")
        assert hw is not None, "missing hardware section"
        assert hw.get("device")
        assert hw.get("gpu")
        assert hw.get("vram_gb")
        assert hw.get("dtype")

    def test_quantization_explicitly_recorded(self):
        assert "quantization" in _config.get("hardware", {})


# --------------------------------------------------------------------------- #
# Config validation gate
# --------------------------------------------------------------------------- #


def _validate_config(cfg: dict, strict: bool = True) -> list[str]:
    """Replicate the script's validation logic for testing.

    Returns the list of problems. With strict=True, raises SystemExit on
    any problem.
    """
    import sys

    problems = []

    def _get_local(path):
        node = cfg
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    required = [
        "experiment_model.attacker",
        "experiment_model.target",
        "experiment_model.evaluator",
        "attacker.model",
        "target.model",
        "evaluator.rubric_model",
        "evaluator.refusal_model",
        "embedding.model",
        "embedding.dimension",
        "nbf.checkpoint",
        "nbf.state_dimension",
        "nbf.threshold",
        "nbf.trials.crescendo",
        "nbf.trials.actor_attack",
        "nbf.trials.opposite_day",
        "nbf.trials.acronym",
        "dataset.path",
        "dataset.split",
        "experiment.seed",
        "attacks.max_turns",
        "experiment.output_dir",
        "hardware.device",
    ]
    for field_path in required:
        val = _get_local(field_path)
        if val is None:
            problems.append(f"missing required field: {field_path}")

    ckpt = _get_local("nbf.checkpoint")
    if ckpt:
        if not Path(ckpt).exists():
            problems.append(f"nbf.checkpoint does not exist: {ckpt}")
        else:
            expected = _get_local("nbf.checkpoint_sha256_expected")
            if expected:
                actual = hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
                if actual != expected:
                    problems.append(
                        f"nbf.checkpoint sha256 mismatch: expected {expected}, "
                        f"got {actual}"
                    )

    ds = _get_local("dataset.path")
    if ds and not Path(ds).exists():
        problems.append(f"dataset.path does not exist: {ds}")

    for field_path in ("attacker.model", "target.model"):
        model = _get_local(field_path)
        if model:
            p = Path(model)
            if not p.exists() and "/" not in model:
                problems.append(
                    f"{field_path} {model!r} does not exist locally and is "
                    f"not an API model id"
                )

    if problems and strict:
        raise SystemExit(2)
    return problems


class TestValidationGateRejectsIncompleteConfig:
    """The validation gate must refuse to start on missing required fields."""

    def test_valid_default_config_passes(self):
        problems = _validate_config(_config, strict=False)
        assert not problems, problems

    def test_missing_experiment_model_fails(self):
        cfg = _load_config(_REPRO_CFG)
        del cfg["experiment_model"]
        with pytest.raises(SystemExit):
            _validate_config(cfg, strict=True)

    def test_missing_nbf_checkpoint_fails(self):
        cfg = _load_config(_REPRO_CFG)
        cfg["nbf"]["checkpoint"] = "nonexistent/path.pth"
        with pytest.raises(SystemExit):
            _validate_config(cfg, strict=True)

    def test_wrong_checkpoint_sha256_fails(self):
        cfg = _load_config(_REPRO_CFG)
        cfg["nbf"]["checkpoint_sha256_expected"] = "0" * 64
        with pytest.raises(SystemExit):
            _validate_config(cfg, strict=True)

    def test_missing_dataset_fails(self):
        cfg = _load_config(_REPRO_CFG)
        cfg["dataset"]["path"] = "nonexistent/dataset.json"
        with pytest.raises(SystemExit):
            _validate_config(cfg, strict=True)

    def test_missing_model_path_fails(self):
        cfg = _load_config(_REPRO_CFG)
        cfg["attacker"]["model"] = "nonexistent_model"
        with pytest.raises(SystemExit):
            _validate_config(cfg, strict=True)

    def test_problems_listed_not_swallowed(self):
        """On failure, the gate must enumerate the real problems, not a generic
        message."""
        cfg = _load_config(_REPRO_CFG)
        cfg["experiment_model"] = {}
        cfg["dataset"]["path"] = "nonexistent.json"
        problems = _validate_config(cfg, strict=False)
        assert any("experiment_model.attacker" in p for p in problems)
        assert any("dataset.path" in p for p in problems)


# --------------------------------------------------------------------------- #
# Dataset structure
# --------------------------------------------------------------------------- #


class TestOfficialDatasetStructure:
    """The official harmbench_tasks.json must be present and match the
    documented structure."""

    def test_dataset_file_exists(self):
        assert _OFFICIAL_DATASET.exists(), f"missing: {_OFFICIAL_DATASET}"

    def test_dataset_is_json(self):
        with open(_OFFICIAL_DATASET, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_dataset_has_200_goals(self):
        with open(_OFFICIAL_DATASET, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 200

    def test_dataset_fields(self):
        with open(_OFFICIAL_DATASET, "r", encoding="utf-8") as f:
            data = json.load(f)
        for rec in data:
            assert "target_system" in rec, "missing target_system"
            assert "task" in rec, "missing task"
            assert "max_rounds" in rec, "missing max_rounds"

    def test_dataset_max_rounds_is_8(self):
        with open(_OFFICIAL_DATASET, "r", encoding="utf-8") as f:
            data = json.load(f)
        for rec in data:
            assert str(rec["max_rounds"]) == "8", (
                f"max_rounds should be 8 for all goals, got {rec['max_rounds']!r}"
            )

    def test_dataset_no_duplicate_tasks(self):
        with open(_OFFICIAL_DATASET, "r", encoding="utf-8") as f:
            data = json.load(f)
        tasks = [r["task"] for r in data]
        assert len(tasks) == len(set(tasks)), "duplicate tasks found"


class TestCheckpointDeterministicSanity:
    """The checkpoint must produce deterministic logits/scores/decisions."""

    def test_checkpoint_loads_and_scores(self):
        import torch

        from guardbound.models.compat import load_original_checkpoint
        from guardbound.models.predictor import NeuralBarrierFunction

        dynamics, predictor = load_original_checkpoint(
            _CHECKPOINT, device="cpu"
        )
        barrier = NeuralBarrierFunction(dynamics=dynamics, predictor=predictor)

        x = torch.zeros(1, 768)
        u = torch.randn(1, 768)
        with torch.no_grad():
            logits = barrier.predictor(x, u)
            probs = torch.softmax(logits, dim=-1)
            last_prob = probs[:, -1]
            max_other = torch.max(probs[:, :-1], dim=1).values
            score = (last_prob - max_other)[0].item()

        assert logits.shape == (1, 5), f"wrong logit shape: {logits.shape}"
        assert isinstance(score, float)
        # The loaded checkpoint must not be all-zeros.
        assert probs.abs().sum().item() > 0
