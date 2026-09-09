"""Typed configuration loader.

All paper-specified constants live in ``configs/default.yaml`` and are validated
on load. Values the paper does not specify are grouped under ``data`` and
``training_extra`` and flagged in the YAML comments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


# --------------------------------------------------------------------------- #
# Config sections (mirroring configs/default.yaml)
# --------------------------------------------------------------------------- #


@dataclass
class DimsConfig:
    embedding_dim_n: int   # paper: 768
    state_dim_m: int       # paper: 768


@dataclass
class DialogueConfig:
    max_turns_k: int       # paper B.1: 8
    temperature: float     # paper: 0.7 everywhere


@dataclass
class SafetyLabelsConfig:
    scores: list[int]      # paper: {1..5} from GPT-4o judge
    unsafe_score: int      # paper: 5
    safe_scores: list[int] # paper: {1..4}

    @property
    def num_classes(self) -> int:
        return len(self.scores)


@dataclass
class StageConfig:
    optimizer: str
    lr: float
    epochs: int


@dataclass
class TrainingConfig:
    stage1_dynamics: StageConfig     # paper: Adam, lr 1e-4, 200 epochs
    stage2_predictor: StageConfig    # paper: Adam, lr 1e-3, 200 epochs
    loss_weights: dict[str, float]   # paper B.1: dyn=1, ce=1, ss=100, si=100
    eta_train: float                 # paper: 0.0 during training
    kappa_noninvariant_turns: int    # paper default: 3
    freeze_dynamics_in_stage2: bool  # audit decision; --joint overrides


@dataclass
class DefenseConfig:
    eta_eval_grid: list[float]
    eta_recommended: float           # paper Table 15 discussion: 5e-4


@dataclass
class EmbeddingsConfig:
    default_model: str               # paper: all-mpnet-base-v2
    alt_model: str                   # paper: all-distilroberta-v1


@dataclass
class LLMConfig:
    openai: dict[str, str]
    anthropic: dict[str, str]
    local: dict[str, str]
    providers: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class PathsConfig:
    data_raw: Path
    data_processed: Path
    checkpoints: Path
    runs: Path
    results: Path


@dataclass
class DataExtraConfig:
    train_val_split: float = 0.9     # not specified in the paper — default 90/10


@dataclass
class TrainingExtraConfig:
    batch_size: int = 64             # not specified in the paper
    seed: int = 0                    # not specified in the paper
    weight_decay: float = 0.0        # not specified in the paper
    gradient_clip_norm: float | None = None  # not specified in the paper
    scheduler: str = "none"          # not specified in the paper


@dataclass
class ProjectConfig:
    project: str
    dims: DimsConfig
    dialogue: DialogueConfig
    safety_labels: SafetyLabelsConfig
    training: TrainingConfig
    defense: DefenseConfig
    embeddings: EmbeddingsConfig
    llm: LLMConfig
    paths: PathsConfig
    data: DataExtraConfig
    training_extra: TrainingExtraConfig

    # -- validation ----------------------------------------------------------#

    def validate(self) -> None:
        """Fail fast on values inconsistent with the paper."""
        errors: list[str] = []
        if self.dims.embedding_dim_n != 768:
            errors.append("dims.embedding_dim_n must be 768 (paper)")
        if self.dims.state_dim_m != 768:
            errors.append("dims.state_dim_m must be 768 (paper)")
        if self.dialogue.max_turns_k != 8:
            errors.append("dialogue.max_turns_k must be 8 (paper B.1)")
        if abs(self.dialogue.temperature - 0.7) > 1e-9:
            errors.append("dialogue.temperature must be 0.7 (paper)")
        if self.safety_labels.unsafe_score not in self.safety_labels.scores:
            errors.append("safety_labels.unsafe_score must be in scores")
        if sorted(self.safety_labels.safe_scores) != [1, 2, 3, 4]:
            errors.append("safety_labels.safe_scores must be [1, 2, 3, 4] (paper B.1)")
        expected_weights = {"lambda_dyn": 1.0, "lambda_ce": 1.0,
                            "lambda_ss": 100.0, "lambda_si": 100.0}
        for key, val in expected_weights.items():
            if self.training.loss_weights.get(key) != val:
                errors.append(f"training.loss_weights.{key} must be {val} (paper B.1)")
        if not 0.0 < self.data.train_val_split < 1.0:
            errors.append("data.train_val_split must be in (0, 1)")
        if errors:
            raise ValueError("Invalid config:\n  - " + "\n  - ".join(errors))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _build(cls: type, payload: dict[str, Any]):
    return cls(**payload)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Load and validate a YAML config into typed dataclasses."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    cfg = ProjectConfig(
        project=raw["project"],
        dims=_build(DimsConfig, raw["dims"]),
        dialogue=_build(DialogueConfig, raw["dialogue"]),
        safety_labels=_build(SafetyLabelsConfig, raw["safety_labels"]),
        training=TrainingConfig(
            stage1_dynamics=_build(StageConfig, raw["training"]["stage1_dynamics"]),
            stage2_predictor=_build(StageConfig, raw["training"]["stage2_predictor"]),
            loss_weights=dict(raw["training"]["loss_weights"]),
            eta_train=float(raw["training"]["eta_train"]),
            kappa_noninvariant_turns=int(raw["training"]["kappa_noninvariant_turns"]),
            freeze_dynamics_in_stage2=bool(raw["training"]["freeze_dynamics_in_stage2"]),
        ),
        defense=_build(DefenseConfig, raw["defense"]),
        embeddings=_build(EmbeddingsConfig, raw["embeddings"]),
        llm=_build(LLMConfig, raw["llm"]),
        paths=PathsConfig(
            data_raw=Path(raw["paths"]["data_raw"]),
            data_processed=Path(raw["paths"]["data_processed"]),
            checkpoints=Path(raw["paths"]["checkpoints"]),
            runs=Path(raw["paths"]["runs"]),
            results=Path(raw["paths"]["results"]),
        ),
        data=_build(DataExtraConfig, raw.get("data", {})),
        training_extra=_build(TrainingExtraConfig, raw.get("training_extra", {})),
    )
    cfg.validate()
    return cfg
