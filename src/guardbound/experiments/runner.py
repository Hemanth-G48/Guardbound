"""Resumable / idempotent experiment runner (Phase 9).

The runner interprets an ``ExperimentSpec`` and produces a plan
(a list of (variant, model, attack, dataset, eta, seed) tuples)
together with explicit lifecycle statuses for each cell.  The
runner does **not** itself call paid APIs or start training; it
composes the existing Phase 6/7/8 entry points and records
``EvaluationResult`` rows in the project's standard store.

In smoke mode, every cell is executed against mocked LLMs / judges
and results are tagged accordingly so they never mix with real
experiment data.

Lifecycle statuses
------------------
PENDING       not yet executed
RUNNING       currently executing
COMPLETED     finished successfully
FAILED        execution raised
SKIPPED       intentionally skipped (e.g. optional + unavailable)
UNAVAILABLE   required model / dataset / checkpoint missing
UNVERIFIED    executed but result provenance is unclear
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..evaluation import (
    EvaluationResult,
    read_jsonl,
    write_jsonl,
)
from ..logging_utils import get_logger
from .registry import (
    EXPERIMENTS,
    ExperimentSpec,
    ExperimentStatus,
    get_experiment,
    list_experiments,
    validate_experiment,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Plan + cell
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ExperimentCell:
    """One (variant, model, attack, dataset, eta, seed) cell of a spec."""
    experiment_id: str
    variant: str
    model: str
    attack: str | None
    dataset: str | None
    eta: float | None
    seed: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_result(self, metric: str, value: float, *,
                  n: int | None = None, **kwargs) -> EvaluationResult:
        return EvaluationResult(
            model=self.model,
            defense_variant=self.variant,
            metric=metric,
            value=value,
            dataset=self.dataset,
            attack=self.attack,
            eta=self.eta if self.variant == "guardbound" else None,
            seed=self.seed,
            n=n,
            extras=self.extra,
            **kwargs,
        )


@dataclass
class CellState:
    cell: ExperimentCell
    status: ExperimentStatus = ExperimentStatus.PENDING
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    outputs: list[EvaluationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self.cell)
        d["status"] = self.status.value
        d["error"] = self.error
        d["started_at"] = self.started_at
        d["finished_at"] = self.finished_at
        d["outputs"] = [r.to_dict() for r in self.outputs]
        return d


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

@dataclass
class RunnerConfig:
    output_root: str = "results/experiments"
    smoke: bool = False
    resume: bool = True
    dry_run: bool = False
    seed: int = 42


class ExperimentRunner:
    """Resumable / idempotent experiment runner.

    The runner does not perform attacks or evaluations itself.  It
    delegates each cell to an *executor* function that the caller
    provides.  This is intentional: the executor encapsulates the
    Phase 6 + Phase 7 + Phase 8 wiring for the specific metric
    (ASR, MMLU, …).  The runner only records lifecycle state and
    ensures no duplicate logical work is repeated.
    """

    def __init__(self, config: RunnerConfig | None = None):
        self.config = config or RunnerConfig()
        self.states: dict[str, CellState] = {}
        self.results: list[EvaluationResult] = []
        self._manifest_path: Path | None = None

    # ------------------------------------------------------------------ #
    # Plan generation
    # ------------------------------------------------------------------ #

    def plan(self, spec: ExperimentSpec) -> list[ExperimentCell]:
        """Materialize every cell of an experiment into a flat list."""
        attacks = list(spec.attacks) if spec.attacks else [None]
        datasets = list(spec.datasets) if spec.datasets else [None]
        etas = list(spec.eta_values) if spec.eta_values else [None]
        cells: list[ExperimentCell] = []
        for variant in spec.defense_variants:
            for model in spec.models:
                for eta in etas:
                    for seed in spec.seeds:
                        for attack in attacks:
                            for dataset in datasets:
                                cells.append(ExperimentCell(
                                    experiment_id=spec.experiment_id,
                                    variant=variant,
                                    model=model,
                                    attack=attack,
                                    dataset=dataset,
                                    eta=eta,
                                    seed=seed,
                                ))
        return cells

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #

    @staticmethod
    def is_model_available(model: str, smoke: bool) -> bool:
        """Return True if the model can be invoked in this environment.

        In smoke mode, every model is considered available (MockChatLLM
        stands in).  In real mode, closed-source models are only
        available when their API-key env var is set.
        """
        if smoke:
            return True
        m = model.lower()
        if "gpt" in m or m.startswith("o1") or m.startswith("o3") or m == "gpt-5":
            return bool(os.environ.get("OPENAI_API_KEY"))
        if "claude" in m or "anthropic" in m:
            return bool(os.environ.get("ANTHROPIC_API_KEY"))
        # Local / open models assumed available if transformers is
        # importable.  In a strict environment without GPU this would
        # still be available; concrete tests are the caller's
        # responsibility.
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def run(
        self,
        spec: ExperimentSpec,
        executor: Callable[[ExperimentCell, RunnerConfig], list[EvaluationResult]] | None = None,
    ) -> list[EvaluationResult]:
        """Run every cell of ``spec`` and return all produced rows.

        If ``executor`` is None the runner records PENDING status for
        every cell and returns an empty list.  In smoke mode without
        an executor, the runner records PENDING too.
        """
        # Resume from previous manifest if available
        output_dir = self._output_dir(spec)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = output_dir / "manifest.jsonl"
        if self.config.resume and self._manifest_path.exists():
            self._load_manifest(spec)

        new_rows: list[EvaluationResult] = []
        cells = self.plan(spec)
        for cell in cells:
            key = self._cell_key(cell)
            state = self.states.get(key)
            if state is None:
                state = CellState(cell=cell)
                self.states[key] = state

            if self.config.resume and state.status == ExperimentStatus.COMPLETED:
                logger.debug("Resume: skipping completed cell %s", key)
                new_rows.extend(state.outputs)
                continue

            if not self.is_model_available(cell.model, self.config.smoke):
                state.status = ExperimentStatus.UNAVAILABLE
                state.error = f"model '{cell.model}' not available"
                self._record_manifest(state)
                continue

            if spec.optional and self._should_skip_optional(spec, cell):
                state.status = ExperimentStatus.SKIPPED
                state.error = "optional experiment + missing prerequisite"
                self._record_manifest(state)
                continue

            if self.config.dry_run or executor is None:
                state.status = ExperimentStatus.PENDING
                self._record_manifest(state)
                continue

            state.status = ExperimentStatus.RUNNING
            state.started_at = datetime.now(timezone.utc).isoformat()
            try:
                rows = executor(cell, self.config) or []
                # Tag every row with the experiment id and smoke flag.
                for r in rows:
                    r.extras = dict(r.extras or {})
                    r.extras["experiment_id"] = spec.experiment_id
                    r.extras["smoke"] = bool(self.config.smoke)
                state.outputs.extend(rows)
                state.status = ExperimentStatus.COMPLETED
            except Exception as exc:  # pragma: no cover - safety net
                state.status = ExperimentStatus.FAILED
                state.error = repr(exc)
                logger.warning("Cell %s failed: %s", key, exc)
            finally:
                state.finished_at = datetime.now(timezone.utc).isoformat()
                self._record_manifest(state)

            new_rows.extend(state.outputs)
            self.results.extend(state.outputs)

        return new_rows

    # ------------------------------------------------------------------ #
    # Manifest persistence
    # ------------------------------------------------------------------ #

    def _output_dir(self, spec: ExperimentSpec) -> Path:
        rel = spec.output_directory.format(
            experiment_id=spec.experiment_id,
        )
        return Path(self.config.output_root) / Path(rel).name

    def _cell_key(self, cell: ExperimentCell) -> str:
        return "|".join([
            cell.experiment_id, cell.variant, cell.model,
            str(cell.attack), str(cell.dataset),
            f"eta={cell.eta}", f"seed={cell.seed}",
        ])

    def _record_manifest(self, state: CellState) -> None:
        if self._manifest_path is None:
            return
        # Append the latest record for this cell, then dedupe.
        line = json.dumps(state.to_dict(), ensure_ascii=False)
        with open(self._manifest_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _load_manifest(self, spec: ExperimentSpec) -> None:
        if self._manifest_path is None or not self._manifest_path.exists():
            return
        for line in self._manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cell = ExperimentCell(
                experiment_id=d["experiment_id"],
                variant=d["variant"],
                model=d["model"],
                attack=d["attack"],
                dataset=d["dataset"],
                eta=d["eta"],
                seed=d["seed"],
                extra=d.get("extra", {}),
            )
            state = CellState(
                cell=cell,
                status=ExperimentStatus(d["status"]),
                error=d.get("error"),
                started_at=d.get("started_at"),
                finished_at=d.get("finished_at"),
                outputs=[EvaluationResult(**r) for r in d.get("outputs", [])],
            )
            self.states[self._cell_key(cell)] = state
            self.results.extend(state.outputs)

    @staticmethod
    def _should_skip_optional(spec: ExperimentSpec, cell: ExperimentCell) -> bool:
        """For optional experiments: skip if the optional spec is incomplete."""
        if not spec.optional:
            return False
        # If the experiment declares required_models and none are
        # available, skip.  Otherwise run.
        if not spec.required_models:
            return False
        if not any(ExperimentRunner.is_model_available(m, smoke=False)
                   for m in spec.required_models):
            return True
        return False


# --------------------------------------------------------------------------- #
# Convenience: run multiple experiments in sequence
# --------------------------------------------------------------------------- #

def run_all(
    experiment_ids: Iterable[str] | str = "ALL",
    *,
    config: RunnerConfig | None = None,
    executor: Callable[[ExperimentCell, RunnerConfig], list[EvaluationResult]] | None = None,
) -> dict[str, list[EvaluationResult]]:
    """Run a set of experiments and return a {experiment_id: rows} map."""
    cfg = config or RunnerConfig()
    ids = list(EXPERIMENTS.keys()) if experiment_ids == "ALL" else list(experiment_ids)
    runner = ExperimentRunner(cfg)
    out: dict[str, list[EvaluationResult]] = {}
    for eid in ids:
        spec = get_experiment(eid)
        issues = validate_experiment(spec)
        if issues:
            logger.warning("Validation issues for %s: %s", eid, issues)
        out[eid] = runner.run(spec, executor=executor)
    return out
