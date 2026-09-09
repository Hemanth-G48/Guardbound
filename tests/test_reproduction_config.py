"""Phase 7 — reproduction configuration + checkpoint verification tests.

These tests keep the reproduction pipeline honest:

* ``configs/reproduction.yaml`` stays complete (every required field present)
* the pinned NBF checkpoint exists and matches its recorded sha256
* the validation gate actually rejects incomplete/corrupted configs
* the official dataset is present with the expected structure

Run:  pytest tests/test_reproduction_config.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "configs" / "reproduction.yaml"


def _load_script(name: str):
    """Import a scripts/ module by path (scripts/ has no __init__.py)."""
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def repro_script():
    return _load_script("run_reproduction")


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def _dig(d: dict, dotted: str):
    node = d
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


class TestReproductionConfigComplete:
    """The config must be a complete, explicit single source of truth."""

    def test_config_file_exists(self):
        assert CONFIG_PATH.exists()

    def test_all_required_fields_present(self, cfg):
        missing = [f for f in REQUIRED_FIELDS if _dig(cfg, f) is None]
        assert not missing, f"Missing required config fields: {missing}"

    def test_model_substitution_is_explicit(self, cfg):
        """paper_reference_model and experiment_model must both be declared
        and must differ — the substitution must be documented, not silent."""
        paper = cfg["paper_reference_model"]
        exp = cfg["experiment_model"]
        assert paper["attacker"] and paper["target"]
        assert exp["attacker"] and exp["target"]
        assert exp["provider"] == "hf_local"
        # Documented deviation: experiment models are local Llama, official
        # models are gpt-4o.
        assert "gpt-4o" in paper["attacker"]
        assert "Llama" in exp["attacker"]

    def test_evaluator_is_explicit_not_implicit(self, cfg):
        """Every evaluator role must be named explicitly (never 'same as
        attacker' by accident)."""
        ev = cfg["evaluator"]
        for role in (
            "rubric_model", "refusal_model", "disclaimer_model",
            "actor_step_judge_model",
        ):
            assert ev.get(role), f"evaluator.{role} must be explicit"
        assert ev.get("temperature") == 0.0  # official check_refusal/rubric

    def test_nbf_trials_match_official(self, cfg):
        trials = cfg["nbf"]["trials"]
        assert trials == {
            "crescendo": 3,
            "actor_attack": 10,
            "opposite_day": 3,
            "acronym": 3,
        }

    def test_local_checkpoints_disallowed(self, cfg):
        assert cfg["nbf"]["allow_local_checkpoints"] is False


class TestCheckpointPin:
    """The pinned official checkpoint must exist and match its sha256."""

    def test_checkpoint_exists(self, cfg):
        ckpt = REPO / cfg["nbf"]["checkpoint"]
        assert ckpt.exists(), f"Official NBF checkpoint missing: {ckpt}"

    def test_checkpoint_sha_matches_pin(self, cfg, repro_script):
        ckpt = REPO / cfg["nbf"]["checkpoint"]
        expected = cfg["nbf"].get("checkpoint_sha256_expected")
        assert expected, "checkpoint_sha256_expected must be pinned"
        actual = repro_script.sha256_of(ckpt)
        assert actual == expected, (
            "NBF checkpoint does not match the pinned sha256 — a substituted "
            "checkpoint must never run silently."
        )


class TestValidationGate:
    """The gate must refuse incomplete/corrupted configs."""

    def test_valid_config_passes(self, cfg, repro_script):
        assert repro_script.validate_config(cfg, strict=False) == []

    def test_missing_field_detected(self, cfg, repro_script):
        import copy
        broken = copy.deepcopy(cfg)
        del broken["evaluator"]["rubric_model"]
        problems = repro_script.validate_config(broken, strict=False)
        assert any("evaluator.rubric_model" in p for p in problems)

    def test_checkpoint_substitution_detected(self, cfg, repro_script):
        import copy
        broken = copy.deepcopy(cfg)
        broken["nbf"]["checkpoint_sha256_expected"] = "0" * 64
        problems = repro_script.validate_config(broken, strict=False)
        assert any("sha256 mismatch" in p for p in problems)

    def test_strict_mode_raises(self, cfg, repro_script):
        import copy
        broken = copy.deepcopy(cfg)
        broken["dataset"]["path"] = "does/not/exist.json"
        with pytest.raises(SystemExit):
            repro_script.validate_config(broken, strict=True)


class TestOfficialDataset:
    """The official dataset must be present with the expected structure."""

    def test_dataset_exists_and_shape(self, cfg):
        ds = REPO / cfg["dataset"]["path"]
        assert ds.exists()
        import json
        data = json.loads(ds.read_text(encoding="utf-8"))
        assert len(data) == 200
        assert cfg["dataset"]["num_goals"] == 200
        for rec in data[:5]:
            assert set(rec.keys()) >= {"target_system", "task", "max_rounds"}
            assert rec["max_rounds"] == 8

    def test_no_duplicate_tasks(self, cfg):
        import json
        ds = REPO / cfg["dataset"]["path"]
        data = json.loads(ds.read_text(encoding="utf-8"))
        tasks = [r["task"] for r in data]
        assert len(tasks) == len(set(tasks))

    def test_official_sampling_is_unshuffled(self, cfg):
        """Official steering.py iterates file order with no shuffle/seed —
        the config must record exactly that (or NOT SPECIFIED)."""
        assert cfg["dataset"]["sampling"] == "file_order"
        assert cfg["dataset"]["shuffle"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
