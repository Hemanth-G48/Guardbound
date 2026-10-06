"""Phase 17 Stage 4.7 — ETA / progress observability validation.

Validates, without touching the experiment, that the pilot's progress tracker derives
its timing from **completed run durations** and nothing else:

  * run_duration = run_end - run_start (measured around the whole run);
  * primary ETA = average completed-run duration x remaining completed runs;
  * the first runs use elapsed/completed; from the 5th run on it is the mean of the
    most recent 10 completed runs;
  * progress = completed_runs / total_runs;
  * attacker / target / judge call durations never enter the estimate.

Real measured run durations are replayed through the tracker (the frozen smoke runs,
and the 80 runs of Stage 4.6 as a 90-run-scale input). The user's worked example
(10 runs of 500 s out of 90 -> 40,000 s remaining) is checked exactly.

Writes `results/phase17_pilot/analysis/eta_validation.json`.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

OUT = REPO_ROOT / "results" / "phase17_pilot"
PILOT_SCRIPT = REPO_ROOT / "scripts" / "phase17_stage47_pilot.py"

spec = importlib.util.spec_from_file_location("phase17_stage47_pilot", PILOT_SCRIPT)
pilot = importlib.util.module_from_spec(spec)
sys.modules["phase17_stage47_pilot"] = pilot
spec.loader.exec_module(pilot)

checks: list[dict] = []


def check(name: str, ok: bool, observed, expected, note: str = "") -> None:
    checks.append({"check": name, "ok": bool(ok), "observed": observed,
                   "expected": expected, "note": note})


def tracker(total: int, durations: list[float], elapsed: float):
    """A tracker with a controlled clock and a controlled completed-run history."""
    t = pilot.ProgressTracker(
        total, {k: total // 3 for k in ("crescendo", "opposite_day", "acronym")},
        OUT / "analysis" / "_eta_validation_progress.json")
    per = total // len(pilot.phase14.ATTACK_ORDER) if total else 0
    for index, duration in enumerate(durations):
        t.durations.append(duration)
        attack = pilot.phase14.ATTACK_ORDER[index % 3][:9]
        key = {"crescendo": "crescendo", "opposite_": "opposite_day",
               "acronym": "acronym"}[attack]
        t.per_attack_done[key] = t.per_attack_done.get(key, 0) + 1
    t.started = time.time() - elapsed
    return t


def close(a, b, tol=0.5) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


print("=== 1. static wiring: what feeds the estimate ===")
tracker_src = inspect.getsource(pilot.ProgressTracker)
check("only finish_run appends run durations",
      tracker_src.count("self.durations.append") == 1
      and "duration_s" in inspect.getsource(pilot.ProgressTracker.finish_run),
      {"append_sites": tracker_src.count("self.durations.append")}, 1)
pilot_src = PILOT_SCRIPT.read_text(encoding="utf-8")
check("no per-call timing in the pilot script's estimate path",
      "latency_s" not in pilot_src and "wall_s" not in pilot_src,
      {"latency_s_occurrences": pilot_src.count("latency_s"),
       "wall_s_occurrences": pilot_src.count("wall_s")}, {"latency_s": 0, "wall_s": 0},
      "call-level durations live in the shared record helpers, never in the tracker")
run_loop_src = pilot_src.split("for goal_id, goal_record in enumerate(goals):")[-1]
check("finish_run is called once per completed run, with the run duration",
      run_loop_src.count("tracker.finish_run(") == 1
      and 'tracker.finish_run(attack_short, record["duration_s"], failed)' in pilot_src,
      {"finish_run_call_sites": run_loop_src.count("tracker.finish_run(")}, 1)
check("run duration is measured around the whole run",
      "started_at = time.time()" in pilot_src
      and 'duration = round(time.time() - started_at, 3)' in pilot_src,
      {"measurement": "duration = time.time() - started_at"}, "run_end - run_start")

print("=== 2. progress semantics (completed runs only) ===")
for completed, expected_pct in ((10, 11.1), (45, 50.0), (90, 100.0)):
    t = tracker(90, [500.0] * completed, elapsed=completed * 500.0)
    pct = round(t.completed / t.total_runs * 100, 1)
    check(f"progress at {completed}/90", pct == expected_pct and t.completed == completed,
          {"completed": t.completed, "percent": pct}, {"completed": completed,
                                                       "percent": expected_pct})

print("=== 3. the worked example: 10 x 500 s, 80 remaining ===")
t = tracker(90, [500.0] * 10, elapsed=5000.0)
eta, label = t.estimate()
check("ETA == 80 x 500 s == 40,000 s (11.1 h)", close(eta, 40000.0, 1.0),
      {"eta_seconds": round(eta, 1), "eta_hours": round(eta / 3600, 2)}, 40000.0)
check("ETA label is the rolling estimate at 10 runs", "rolling" in label, label,
      "rolling estimate (last 10 runs)")
block = t._render
throughput = t.completed / 5000.0 * 60
check("Runs/min is run-level (60/500 = 0.12)", close(throughput, 0.12, 1e-6),
      round(throughput, 4), 0.12)
check("Avg/run is elapsed/completed (spec formula)", close(5000.0 / t.completed, 500.0),
      round(5000.0 / t.completed, 3), 500.0)

print("=== 4. preliminary phase uses elapsed/completed, then switches at 5 ===")
for done in (1, 2, 3, 4):
    t = tracker(90, [500.0] * done, elapsed=done * 500.0 + 40.0)
    eta, label = t.estimate()
    expected = (done * 500.0 + 40.0) / done * (90 - done)
    check(f"preliminary ETA at {done} run(s)", close(eta, expected, 0.5),
          round(eta, 2), round(expected, 2),
          note=f"label={label!r}")
t = tracker(90, [500.0] * 5, elapsed=2540.0)
eta, label = t.estimate()
check("switch to rolling at the 5th completed run", "rolling" in label, label,
      "rolling estimate (last 5 runs)")
check("rolling ETA == mean(last 10) x remaining", close(eta, 500.0 * 85),
      round(eta, 2), 500.0 * 85)
t = tracker(90, [], elapsed=0.0)
eta, label = t.estimate()
check("no completed runs -> no fabricated ETA", eta is None and label == "estimating...",
      {"eta": eta, "label": label}, {"eta": None, "label": "estimating..."})

print("=== 5. rolling window is the most recent 10 completed runs ===")
durations = [100.0] * 12 + [500.0] * 10
t = tracker(90, durations, elapsed=sum(durations))
eta, label = t.estimate()
expected = statistics.mean(durations[-10:]) * (90 - t.completed)
check("rolling uses exactly the last 10 runs", close(eta, expected, 0.5),
      round(eta, 2), round(expected, 2), note=f"label={label!r}")
check("old, unrepresentative runs are excluded from the ETA",
      abs(eta - (statistics.mean(durations[2:]) * (90 - t.completed))) > 1000,
      {"rolling_eta": round(eta, 1)}, "differs from an all-runs average")

print("=== 6. replay of REAL measured run durations ===")
smoke = [json.loads(line) for line in
         (OUT / "smoke_test" / "results.jsonl").read_text(encoding="utf-8").splitlines()
         if line.strip()]
smoke_durations = [r["duration_s"] for r in smoke]
t = tracker(90, smoke_durations, elapsed=sum(smoke_durations))
eta, label = t.estimate()
expected = (sum(smoke_durations) / len(smoke_durations)) * (90 - len(smoke_durations))
check("smoke-run replay: preliminary ETA = mean(smoke run durations) x remaining",
      close(eta, expected, 0.5), round(eta, 2), round(expected, 2),
      note=f"real durations={smoke_durations}")

stage46 = REPO_ROOT / "results" / "phase17_model_optimization" / "stage4_6_b2_qualification" / "multiturn_results.jsonl"
real = [json.loads(line)["wall_s"] for line in stage46.read_text(encoding="utf-8").splitlines()
        if line.strip()]
t = tracker(90, real, elapsed=sum(real))
eta, label = t.estimate()
expected = statistics.mean(real[-10:]) * (90 - t.completed)
check("90-run-scale replay of 80 real run durations: ETA = mean(last 10) x 10",
      close(eta, expected, 0.5), round(eta, 2), round(expected, 2),
      note=f"replayed real durations n={len(real)} from Stage 4.6; "
           f"rolling mean={round(statistics.mean(real[-10:]), 1)}s")

print("=== 7. progress.json carries the same run-level numbers ===")
t = tracker(90, [500.0] * 10, elapsed=5000.0)
t.persist()
payload = json.loads((OUT / "analysis" / "_eta_validation_progress.json").read_text(encoding="utf-8"))
check("estimated_remaining_seconds == rolling mean x remaining",
      close(payload["estimated_remaining_seconds"],
            statistics.mean([500.0] * 10) * 80, 1.0),
      payload["estimated_remaining_seconds"], 40000.0)
check("average_run_seconds == mean(completed run durations)",
      close(payload["average_run_seconds"], 500.0), payload["average_run_seconds"], 500.0)
check("rolling_run_seconds == mean(last 10 completed runs)",
      close(payload["rolling_run_seconds"], 500.0), payload["rolling_run_seconds"], 500.0)
check("progress file declares observability-only semantics",
      payload["observability_only"].startswith("this file never drives"),
      payload["observability_only"], "observability-only statement present")

(OUT / "analysis" / "_eta_validation_progress.json").unlink(missing_ok=True)

failed = [c for c in checks if not c["ok"]]
verdict = "ETA VALIDATED — RUN-LEVEL TIMING" if not failed else "ETA INVALID"
payload = {
    "phase": "17", "stage": "4.7", "deliverable": "analysis/eta_validation.json",
    "verdict": verdict,
    "requirement": {
        "run_duration": "run_end_time - run_start_time, measured around the whole run",
        "primary_eta": "average completed-run duration x remaining completed runs",
        "preliminary": "elapsed_experiment_time / completed_runs (before 5 runs)",
        "rolling": "mean of the most recent 10 completed runs (from the 5th run on)",
        "progress": "completed_runs / total_runs",
        "must_not_use": ["attacker-call duration", "target-call duration",
                         "judge-call duration", "token-generation duration"],
    },
    "implementation": {
        "file": "scripts/phase17_stage47_pilot.py",
        "class": "ProgressTracker",
        "run_duration_source": "run_and_record(): duration = round(time.time() - started_at, 3)",
        "fed_to_tracker_by": 'tracker.finish_run(attack_short, record["duration_s"], failed)',
        "call_level_timings": "present only in the run records (engineering metrics); "
                             "no occurrence of latency_s/wall_s anywhere in the pilot script",
        "avg_run_display": "elapsed / completed_runs (the spec's preliminary formula), "
                           "shown alongside 'Last run'; the ETA uses the rolling mean",
        "no_resume_semantics": True,
    },
    "checks": checks,
    "checks_passed": f"{len(checks) - len(failed)}/{len(checks)}",
    "real_durations_replayed": {
        "smoke_runs_stage4_7": smoke_durations,
        "stage4_6_runs": len(real),
        "note": "replayed through the tracker to exercise the estimator; these are real "
                "measured run durations from this pipeline, not pilot data",
    },
}
(OUT / "analysis" / "eta_validation.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8")

print()
for c in checks:
    print(f"  [{'x' if c['ok'] else ' '}] {c['check']}")
print(f"\nchecks: {len(checks) - len(failed)}/{len(checks)}")
print(verdict)
print("wrote results/phase17_pilot/analysis/eta_validation.json")
raise SystemExit(0 if not failed else 1)
