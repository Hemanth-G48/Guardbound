#!/usr/bin/env python3
"""Phase 9 — experiment orchestrator.

Composes the existing Phase 1-8 entry points into the paper's
experimental matrix.  Does NOT reimplement any attack, metric,
training, or steering logic; the runner interprets the
declarative experiment specs and records lifecycle state.

Modes
-----
--dry-run : print the plan only, do not invoke any expensive step.
--smoke   : run all cells with mocked LLMs / judges so the
            orchestrator can be exercised end-to-end without GPUs,
            API keys, or model downloads.  Smoke results are tagged
            so they never mix with real experiment data.
--resume  : completed cells are skipped; partial runs continue.
--seed    : default RNG seed for new runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", default="ALL",
                   help="Experiment id (E1..E10) or 'ALL' (default: ALL).")
    p.add_argument("--smoke", action="store_true",
                   help="Run with mocked models / judges.")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan only, do not invoke expensive steps.")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Resume from previous manifest (default: True).")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/experiments")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from guardbound.experiments import (
        EXPERIMENTS, ExperimentRunner, RunnerConfig,
        get_experiment, list_experiments, run_all,
    )

    ids = list(EXPERIMENTS.keys()) if args.exp == "ALL" else [args.exp]
    print(f"Experiments  : {', '.join(ids)}")
    print(f"Smoke        : {args.smoke}")
    print(f"Dry run      : {args.dry_run}")
    print(f"Resume       : {args.resume}")
    print(f"Output dir   : {args.output_dir}")

    cfg = RunnerConfig(
        output_root=args.output_dir,
        smoke=args.smoke,
        resume=args.resume,
        dry_run=args.dry_run,
        seed=args.seed,
    )
    results = run_all(ids, config=cfg)
    for eid, rows in results.items():
        spec = get_experiment(eid)
        n_completed = sum(1 for r in rows if r.value is not None)
        print(f"  {eid}: produced {len(rows)} rows "
              f"({n_completed} with numeric values), "
              f"status={'PENDING' if not rows else 'OK'}")


if __name__ == "__main__":
    main()
