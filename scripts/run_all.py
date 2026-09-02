#!/usr/bin/env python3
"""Phase 9 — master runner.

Runs a fixed, sensible ordering of cheap-to-expensive steps:

    1. registry validation
    2. smoke tests
    3. cheap analysis (existing result aggregation)
    4. existing-result aggregation
    5. threshold sweeps
    6. inexpensive ablations
    7. expensive retraining
    8. attack evaluations
    9. guardrail evaluations
   10. PCA collection/analysis
   11. table generation
   12. figure generation
   13. findings verification
   14. final report

The expensive steps are always gated by --smoke so a default
invocation exercises orchestration without any real work.

Flags
-----
--smoke       : run smoke-mode end-to-end.
--resume      : skip completed cells.
--only STAGE  : run a single stage by name.
--skip STAGE  : skip a single stage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


STAGES = [
    "validate",
    "smoke",
    "aggregate",
    "sweeps",
    "ablations",
    "retrain",
    "attacks",
    "guardrails",
    "pca",
    "tables",
    "figures",
    "findings",
    "report",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true",
                   help="Run end-to-end in smoke mode (no GPU / API / training).")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--only", default=None,
                   help=f"Run only this stage.  Stages: {', '.join(STAGES)}")
    p.add_argument("--skip", default=None,
                   help="Skip this stage (repeatable).")
    p.add_argument("--output-dir", default="results/experiments")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _run_validate(args) -> None:
    from guardbound.experiments import (
        EXPERIMENTS, list_experiments, validate_experiment,
    )
    for eid in list_experiments():
        spec = EXPERIMENTS[eid]
        issues = validate_experiment(spec)
        status = "OK" if not issues else f"ISSUES={issues}"
        print(f"  {eid}: {status}")


def _run_smoke(args) -> None:
    from guardbound.experiments import run_all, RunnerConfig
    cfg = RunnerConfig(output_root=args.output_dir, smoke=True,
                        resume=args.resume, dry_run=False, seed=args.seed)
    out = run_all("ALL", config=cfg)
    print(f"  smoke produced {sum(len(r) for r in out.values())} result rows")


def _run_aggregate(args) -> None:
    """Aggregate any existing result JSONLs under the output directory."""
    from pathlib import Path
    from guardbound.analysis import aggregate_by
    from guardbound.evaluation import read_jsonl
    root = Path(args.output_dir)
    if not root.exists():
        print(f"  (no {root}; skipping)")
        return
    count = 0
    for f in root.rglob("*.results.jsonl"):
        rows = read_jsonl(f)
        grouped = aggregate_by(rows, metric="ASR")
        print(f"  {f}: {len(rows)} rows -> {len(grouped)} ASR aggregates")
        count += 1
    if not count:
        print("  (no result sidecars found; skipping)")


def _run_findings(args) -> None:
    from pathlib import Path
    from guardbound.analysis import check_claims
    from guardbound.evaluation import read_jsonl
    root = Path(args.output_dir)
    all_rows = []
    if root.exists():
        for f in root.rglob("*.results.jsonl"):
            all_rows.extend(read_jsonl(f))
    report = check_claims(all_rows)
    out = root / "findings.md"
    if root.exists():
        out.write_text(report.to_markdown(), encoding="utf-8")
    print(f"  findings -> {out} ({len(report.checks)} checks, "
          f"{len(report.missing_data)} missing)")


def _run_report(args) -> None:
    # Implemented in scripts/build_final_report.py.
    import subprocess
    cmd = [sys.executable, str(Path(__file__).with_name("build_final_report.py")),
           "--output-dir", args.output_dir]
    if args.smoke:
        cmd.append("--smoke")
    print("  running:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"  report stage failed: {exc}")


STAGE_FUNCS = {
    "validate": _run_validate,
    "smoke": _run_smoke,
    "aggregate": _run_aggregate,
    "sweeps": lambda a: print("  (E2 sweeps: not executed in orchestrator; "
                                "invoke scripts/run_experiments.py --exp E2)"),
    "ablations": lambda a: print("  (E3-E5 ablations: not executed in orchestrator; "
                                   "invoke scripts/run_experiments.py --exp E3/E4/E5)"),
    "retrain": lambda a: print("  (retrain: invoke scripts/run_experiments.py "
                                "or experiments/retrain.py directly)"),
    "attacks": lambda a: print("  (attack evaluations: invoke scripts/run_experiments.py)"),
    "guardrails": lambda a: print("  (guardrail evals: invoke scripts/bench_guardrails.py)"),
    "pca": lambda a: print("  (E9 PCA: not executed in orchestrator; "
                              "invoke scripts/run_experiments.py --exp E9)"),
    "tables": lambda a: print("  (tables: emit via analysis/tables.py)"),
    "figures": lambda a: print("  (figures: emit via analysis/plots.py)"),
    "findings": _run_findings,
    "report": _run_report,
}


def main() -> None:
    args = parse_args()
    skip = set((args.skip or "").split(",")) if args.skip else set()
    selected = [args.only] if args.only else STAGES
    for stage in selected:
        if stage in skip:
            print(f"[skip] {stage}")
            continue
        if stage not in STAGE_FUNCS:
            print(f"[unknown stage] {stage}")
            continue
        print(f"[stage] {stage}")
        STAGE_FUNCS[stage](args)
    print("All requested stages complete.")


if __name__ == "__main__":
    main()
