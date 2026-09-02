"""Phase 9 — experiment orchestration + analysis tests.

All tests are offline.  No real APIs, model weights, or training.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardbound.analysis import (
    ClaimCheck,
    FindingsReport,
    aggregate_by,
    check_claims,
    compute_pareto_front,
    dominates,
    list_default_claims,
    mean_of,
    pivot_results,
    stratify_by_attack,
    stratify_by_model,
    to_pivot_markdown,
)
from guardbound.analysis.pareto import Point
from guardbound.evaluation import (
    EvaluationResult,
    read_jsonl,
    write_jsonl,
)
from guardbound.experiments import (
    EXPERIMENTS,
    ExperimentRunner,
    ExperimentSpec,
    ExperimentStatus,
    RunnerConfig,
    get_experiment,
    list_experiments,
    run_all,
    validate_experiment,
)


# --------------------------------------------------------------------------- #
# Registry tests
# --------------------------------------------------------------------------- #

def test_registry_contains_e1_through_e10():
    ids = list_experiments()
    for required in ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"]:
        assert required in ids, f"missing experiment {required}"


def test_e10_is_optional():
    e10 = get_experiment("E10")
    assert e10.optional is True


def test_validate_experiment_returns_no_issues_for_valid_specs():
    for eid in list_experiments():
        spec = get_experiment(eid)
        issues = validate_experiment(spec)
        assert issues == [], f"{eid} has issues: {issues}"


def test_get_experiment_unknown_id_raises():
    with pytest.raises(ValueError, match="Unknown experiment"):
        get_experiment("E11_NOT_REAL")


def test_e1_main_results_uses_three_evaluation_attacks():
    spec = get_experiment("E1")
    assert set(spec.attacks) == {"actor_attack", "crescendo", "opposite_day"}


def test_e2_threshold_sweep_includes_zero_and_paper_knee():
    spec = get_experiment("E2")
    etas = set(spec.eta_values)
    assert 0.0 in etas
    assert 1e-3 in etas
    assert 5e-3 in etas  # over-refusal extension


def test_e7_adaptive_uses_eta_zero():
    spec = get_experiment("E7")
    assert 0.0 in spec.eta_values
    assert "adaptive" in spec.attacks


def test_e9_pca_has_random_state():
    spec = get_experiment("E9")
    assert spec.configuration.get("random_state") == 42


# --------------------------------------------------------------------------- #
# Pareto
# --------------------------------------------------------------------------- #

def test_pareto_empty():
    assert compute_pareto_front([]) == []


def test_pareto_one_point():
    p = Point(asr=0.5, helpfulness=0.7, label="a")
    front = compute_pareto_front([p])
    assert front == [p]


def test_pareto_invalid_point_filtered():
    p = Point(asr=float("nan"), helpfulness=0.7, label="bad")
    front = compute_pareto_front([p])
    assert front == []


def test_pareto_duplicates_deduplicated():
    a = Point(asr=0.5, helpfulness=0.7)
    b = Point(asr=0.5, helpfulness=0.7)
    c = Point(asr=0.4, helpfulness=0.8)  # dominates a/b
    front = compute_pareto_front([a, b, c])
    # Only c remains (dominates a and b).
    assert c in front
    assert len(front) == 1


def test_pareto_dominated_filtered():
    good = Point(asr=0.2, helpfulness=0.9, label="good")
    bad = Point(asr=0.5, helpfulness=0.5, label="bad")
    front = compute_pareto_front([good, bad])
    assert front == [good]


def test_pareto_equal_asr_higher_helpfulness_wins():
    a = Point(asr=0.5, helpfulness=0.5, label="a")
    b = Point(asr=0.5, helpfulness=0.7, label="b")
    front = compute_pareto_front([a, b])
    assert front == [b]


def test_pareto_equal_helpfulness_lower_asr_wins():
    a = Point(asr=0.5, helpfulness=0.7, label="a")
    b = Point(asr=0.3, helpfulness=0.7, label="b")
    front = compute_pareto_front([a, b])
    assert front == [b]


def test_pareto_two_nondominated():
    a = Point(asr=0.2, helpfulness=0.5, label="a")
    b = Point(asr=0.5, helpfulness=0.9, label="b")
    front = compute_pareto_front([a, b])
    assert set(front) == {a, b}


def test_dominates_strict_inequality_required():
    a = Point(asr=0.2, helpfulness=0.7)
    b = Point(asr=0.2, helpfulness=0.7)  # identical
    assert dominates(a, b) is False
    assert dominates(b, a) is False


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def test_mean_of_ignores_none():
    assert mean_of([0.1, None, 0.3]) == 0.2
    assert mean_of([]) is None
    assert mean_of([None]) is None


def test_aggregate_by_groups_correctly():
    rows = [
        EvaluationResult(model="a", defense_variant="x", metric="ASR", value=0.1),
        EvaluationResult(model="a", defense_variant="x", metric="ASR", value=0.3),
        EvaluationResult(model="a", defense_variant="y", metric="ASR", value=0.5),
    ]
    out = aggregate_by(rows, metric="ASR",
                        key_fields=("model", "defense_variant"))
    assert len(out) == 2
    by_var = {r["defense_variant"]: r["value"] for r in out}
    assert by_var["x"] == 0.2
    assert by_var["y"] == 0.5


def test_stratify_by_attack():
    rows = [
        EvaluationResult(model="m", defense_variant="x", metric="ASR", value=0.1, attack="crescendo"),
        EvaluationResult(model="m", defense_variant="x", metric="ASR", value=0.2, attack="actor_attack"),
        EvaluationResult(model="m", defense_variant="x", metric="ASR", value=0.3, attack="crescendo"),
    ]
    out = stratify_by_attack(rows)
    assert len(out["crescendo"]) == 2
    assert len(out["actor_attack"]) == 1


def test_stratify_by_model():
    rows = [
        EvaluationResult(model="a", defense_variant="x", metric="ASR", value=0.1),
        EvaluationResult(model="b", defense_variant="x", metric="ASR", value=0.2),
    ]
    out = stratify_by_model(rows)
    assert set(out) == {"a", "b"}


# --------------------------------------------------------------------------- #
# Pivot + table
# --------------------------------------------------------------------------- #

def test_pivot_results():
    rows = [
        EvaluationResult(model="m1", defense_variant="x", metric="ASR", value=0.1),
        EvaluationResult(model="m1", defense_variant="y", metric="ASR", value=0.2),
        EvaluationResult(model="m2", defense_variant="x", metric="ASR", value=0.3),
    ]
    pivot = pivot_results(rows, rows_field="model", cols_field="defense_variant", metric="ASR")
    assert len(pivot) == 2
    rows_by_model = {row["model"]: row for row in pivot}
    assert rows_by_model["m1"]["x"] == 0.1
    assert rows_by_model["m1"]["y"] == 0.2
    assert rows_by_model["m2"]["x"] == 0.3
    assert rows_by_model["m2"]["y"] is None


def test_to_pivot_markdown_highlights_best_lower_is_better():
    pivot = [
        {"model": "a", "x": 0.1, "y": 0.5},
        {"model": "b", "x": 0.3, "y": 0.2},
    ]
    md = to_pivot_markdown(pivot, metric="ASR", better_is="lower")
    assert "**" in md  # bold for best
    # The lowest value per row is bolded.
    assert "**0.1000**" in md or "**0.1" in md


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #

def test_findings_unverified_when_no_data():
    report = check_claims([])
    assert len(report.checks) == len(list_default_claims())
    for c in report.checks:
        assert c.status.value == "UNVERIFIED"


def test_findings_pass_when_direction_matches():
    rows = [
        EvaluationResult(model="m", defense_variant="original", metric="ASR", value=0.5, attack="crescendo"),
        EvaluationResult(model="m", defense_variant="guardbound", metric="ASR", value=0.1, attack="crescendo", eta=1e-3),
    ]
    report = check_claims(rows)
    # C1 (steering at eta=1e-3 reduces ASR) should pass.
    c1 = next(c for c in report.checks if c.claim_id == "C1")
    assert c1.status.value in ("PASS", "DIRECTIONALLY-CONSISTENT")


def test_findings_fail_when_direction_contradicts():
    rows = [
        EvaluationResult(model="m", defense_variant="original", metric="ASR", value=0.1, attack="crescendo"),
        EvaluationResult(model="m", defense_variant="guardbound", metric="ASR", value=0.5, attack="crescendo", eta=1e-3),
    ]
    report = check_claims(rows)
    c1 = next(c for c in report.checks if c.claim_id == "C1")
    assert c1.status.value == "FAIL"


def test_findings_never_force_pass_for_missing_data():
    # No relevant rows; must be UNVERIFIED.
    rows = [
        EvaluationResult(model="m", defense_variant="original", metric="MTBench", value=8.0),
    ]
    report = check_claims(rows)
    for c in report.checks:
        assert c.status.value != "PASS" or c.required_result_rows > 0, \
            f"Claim {c.claim_id} marked PASS without supporting rows"


def test_findings_report_to_markdown():
    report = FindingsReport(checks=[
        ClaimCheck(claim_id="X", claim="x", status=__import__("guardbound.analysis.findings", fromlist=["ClaimStatus"]).ClaimStatus.PASS, required_result_rows=2),
    ])
    md = report.to_markdown()
    assert "Phase 9 Findings" in md
    assert "PASS" in md


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def test_runner_plan_materials_all_cells():
    spec = get_experiment("E1")
    runner = ExperimentRunner(RunnerConfig())
    cells = runner.plan(spec)
    # E1 has 6 models x 4 variants x 3 attacks x 3 datasets x 1 eta
    # = 216 cells.  The structural check is that every (variant,
    # model, attack, dataset, eta, seed) tuple is unique.
    assert len(cells) >= 6
    seen = set()
    for c in cells:
        key = (c.variant, c.model, c.attack, c.dataset, c.eta, c.seed)
        assert key not in seen
        seen.add(key)


def test_runner_records_unavailable_for_missing_api_keys():
    spec = get_experiment("E1")
    runner = ExperimentRunner(RunnerConfig(smoke=False, resume=False))
    runner.run(spec)  # no executor: every cell goes PENDING
    # No rows are produced.
    assert runner.results == []


def test_runner_resume_skips_completed_cells(tmp_path):
    """Two consecutive runs with the same output dir: completed cells
    are skipped on the second run; the executor sees only unfinished cells."""
    spec = get_experiment("E1")
    cfg = RunnerConfig(output_root=str(tmp_path), smoke=True,
                        resume=True, dry_run=False, seed=42)
    runner1 = ExperimentRunner(cfg)

    counter = {"calls": 0}

    def _executor(cell, _cfg):
        counter["calls"] += 1
        return [cell.to_result("ASR", 0.1, n=5)]

    runner1.run(spec, executor=_executor)
    n_first = counter["calls"]
    assert n_first > 0

    counter2 = {"calls": 0}

    def _executor2(cell, _cfg):
        counter2["calls"] += 1
        return [cell.to_result("ASR", 0.1, n=5)]

    runner2 = ExperimentRunner(cfg)
    runner2.run(spec, executor=_executor2)
    # The second run skips every cell whose manifest says COMPLETED.
    assert counter2["calls"] == 0


def test_runner_dry_run_records_pending():
    spec = get_experiment("E1")
    cfg = RunnerConfig(smoke=True, dry_run=True, resume=False, output_root=tempfile.mkdtemp())
    runner = ExperimentRunner(cfg)
    runner.run(spec, executor=lambda c, _: [c.to_result("ASR", 0.0)])
    for state in runner.states.values():
        assert state.status == ExperimentStatus.PENDING


# --------------------------------------------------------------------------- #
# Smoke runs the whole registry without errors
# --------------------------------------------------------------------------- #

def test_smoke_all_experiments_runs_end_to_end(tmp_path):
    """Simulate a smoke-mode full registry run."""
    cfg = RunnerConfig(output_root=str(tmp_path), smoke=True, resume=True)
    results = run_all("ALL", config=cfg)
    assert set(results.keys()) == set(EXPERIMENTS.keys())
    # E10 is optional+UNAVAILABLE without API keys; smoke mode
    # considers all models available, so smoke cells should produce
    # results when an executor is provided.
    def _executor(cell, _cfg):
        return [cell.to_result("ASR", 0.1, n=5)]

    cfg2 = RunnerConfig(output_root=str(tmp_path / "smoke2"), smoke=True)
    runner = ExperimentRunner(cfg2)
    rows: list[EvaluationResult] = []
    for eid in list_experiments():
        spec = get_experiment(eid)
        rows.extend(runner.run(spec, executor=_executor))
    # Every non-optional experiment produced at least one row.
    n_per_exp = {eid: 0 for eid in list_experiments()}
    for r in rows:
        n_per_exp[r.extras.get("experiment_id", "?")] += 1
    for eid, spec in EXPERIMENTS.items():
        if not spec.optional:
            assert n_per_exp[eid] > 0, f"{eid} produced no rows in smoke run"


# --------------------------------------------------------------------------- #
# build_final_report smoke
# --------------------------------------------------------------------------- #

def test_build_final_report_smoke(tmp_path):
    out_dir = tmp_path / "src"
    rep_dir = tmp_path / "rep"
    out_dir.mkdir()
    # No result sidecars — should still produce a final report.
    import subprocess
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_final_report.py"
    proc = subprocess.run(
        [sys.executable, str(script),
         "--output-dir", str(out_dir),
         "--report-dir", str(rep_dir),
         "--smoke"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert (rep_dir / "final_report.md").exists()
    assert (rep_dir / "findings.md").exists()
    assert (rep_dir / "final_report.json").exists()
    # final_report.json must record the smoke flag.
    meta = json.loads((rep_dir / "final_report.json").read_text())
    assert meta["smoke"] is True


# --------------------------------------------------------------------------- #
# Tables / figures execute without network
# --------------------------------------------------------------------------- #

def test_plots_pareto_executes(tmp_path):
    from guardbound.analysis import plot_pareto_asr_vs_helpfulness
    pts = [Point(asr=0.5, helpfulness=0.7), Point(asr=0.2, helpfulness=0.9)]
    out = plot_pareto_asr_vs_helpfulness(pts, tmp_path / "pareto.png")
    assert out.exists()
    # Sidecar JSON exists with the figure metadata.
    assert out.with_suffix(".json").exists()


def test_plots_redqueen_executes(tmp_path):
    from guardbound.analysis import plot_redqueen_asr_vs_turns
    out = plot_redqueen_asr_vs_turns(
        {1: 0.4, 3: 0.6, 5: 0.7}, tmp_path / "rq.png",
    )
    assert out.exists()


def test_plots_threshold_curve_executes(tmp_path):
    from guardbound.analysis import plot_threshold_curve
    out = plot_threshold_curve(
        {0.0: 0.5, 5e-4: 0.2, 1e-3: 0.1},
        {0.0: 8.0, 5e-4: 7.5, 1e-3: 7.0},
        tmp_path / "thr.png",
    )
    assert out.exists()


def test_plots_pca_executes(tmp_path):
    from guardbound.analysis import plot_pca_trajectories
    rng = np.random.default_rng(0)
    traj = {
        "original": rng.standard_normal((8, 16)),
        "steered":  rng.standard_normal((8, 16)) + 0.5,
    }
    out = plot_pca_trajectories(traj, tmp_path / "pca.png", random_state=0)
    assert out.exists()
    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert "explained_variance_ratio" in sidecar["extra"]
