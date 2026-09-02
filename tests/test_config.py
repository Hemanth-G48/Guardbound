"""Tests for typed config loading (Phase 1)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from guardbound.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_config_loads_and_validates():
    cfg = load_config(CONFIG_PATH)
    assert cfg.project == "nbf-safety-steering"
    assert cfg.dims.embedding_dim_n == 768
    assert cfg.dims.state_dim_m == 768
    assert cfg.dialogue.max_turns_k == 8
    assert cfg.dialogue.temperature == 0.7


def test_paper_hyperparameters_present():
    cfg = load_config(CONFIG_PATH)
    s1 = cfg.training.stage1_dynamics
    s2 = cfg.training.stage2_predictor
    assert (s1.optimizer, s1.lr, s1.epochs) == ("adam", 1e-4, 200)
    assert (s2.optimizer, s2.lr, s2.epochs) == ("adam", 1e-3, 200)
    assert cfg.training.loss_weights == {
        "lambda_dyn": 1.0, "lambda_ce": 1.0,
        "lambda_ss": 100.0, "lambda_si": 100.0,
    }
    assert cfg.training.eta_train == 0.0
    assert cfg.training.kappa_noninvariant_turns == 3
    assert cfg.safety_labels.unsafe_score == 5
    assert sorted(cfg.safety_labels.safe_scores) == [1, 2, 3, 4]
    assert cfg.embeddings.default_model == "all-mpnet-base-v2"
    assert cfg.embeddings.alt_model == "all-distilroberta-v1"
    assert cfg.defense.eta_recommended == 5e-4
    assert 5e-4 in cfg.defense.eta_eval_grid
    # paper-pinned model versions
    assert cfg.llm.openai["gpt35_turbo"] == "gpt-3.5-turbo-0125"
    assert cfg.llm.openai["gpt4o"] == "gpt-4o-2024-08-06"
    assert cfg.llm.anthropic["claude35_sonnet"] == "claude-3-5-sonnet-20241022"


def test_invalid_dims_raise(tmp_path: Path):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["dims"]["state_dim_m"] = 512  # contradicts the paper
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="768"):
        load_config(bad)


def test_invalid_loss_weights_raise(tmp_path: Path):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["training"]["loss_weights"]["lambda_ss"] = 10.0
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="lambda_ss"):
        load_config(bad)


def test_non_paper_defaults_documented():
    cfg = load_config(CONFIG_PATH)
    # These exist but are explicitly NOT from the paper; they must stay in the
    # `data` / `training_extra` sections so provenance stays clear.
    assert cfg.data.train_val_split == 0.9
    assert cfg.training_extra.batch_size > 0
    assert cfg.training_extra.weight_decay == 0.0
    assert cfg.training_extra.seed == 0
