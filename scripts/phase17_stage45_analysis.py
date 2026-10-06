"""Phase 17 Stage 4.5 — analysis and required artifacts.

Aggregates the candidate result JSONLs into the artifacts the stage brief names,
builds the controlled comparison against the Stage 4 frozen baseline, and runs a
descriptive attack-semantic check (Part 26) on the generated questions.

Every number comes from the raw result files; nothing is typed in by hand. Where
a comparison is between different sample sizes the point estimates, sample sizes
and observed differences are reported together, and no significance is claimed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "phase17_stage45_common", REPO_ROOT / "scripts" / "phase17_stage45_common.py")
common = importlib.util.module_from_spec(_spec)
sys.modules["phase17_stage45_common"] = common
_spec.loader.exec_module(common)

OUT_ROOT = common.OUT_ROOT
STAGE4 = common.STAGE4_DIR
VALID = ("VALID_DIRECT_JSON", "VALID_FROZEN_EXTRACTION")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    """95 % Wilson interval — reported so small samples are not over-read."""
    if n == 0:
        return None
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return [round((centre - margin) / denom, 4), round((centre + margin) / denom, 4)]


def config_summary(rows: list[dict], config_id: str) -> dict:
    block = common.summarise(rows, config_id)
    n = block["overall"]["n"]
    valid = block["overall"]["contract_valid"]
    usable = block["overall"]["semantically_usable"]
    block["contract_valid_ci95"] = wilson_interval(valid, n)
    block["semantically_usable_ci95"] = wilson_interval(usable, n)
    return block


def diversity_metrics(rows: list[dict]) -> dict:
    """Descriptive diversity of the generated questions within one candidate.

    The brief warns against trading generation diversity for JSON validity, so
    this is reported for every candidate: unique opening phrases, unique questions,
    and the mean pairwise Jaccard similarity of the questions. Lower similarity
    means more varied questions. Descriptive only — high similarity is not
    automatically bad, and no candidate is selected on it.
    """
    questions = [r["generated_question"].strip() for r in rows
                 if isinstance(r.get("generated_question"), str)
                 and r["generated_question"].strip()]
    n = len(questions)
    if n < 2:
        return {"n": n}
    openings = {" ".join(q.split()[:6]).lower() for q in questions}
    unique = {q.lower() for q in questions}

    def jaccard(a: str, b: str) -> float:
        sa = {t for t in a.lower().split() if len(t) > 3}
        sb = {t for t in b.lower().split() if len(t) > 3}
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(1, len(sa | sb))

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    similarities = [jaccard(questions[i], questions[j]) for i, j in pairs]
    return {
        "n": n,
        "unique_questions": len(unique),
        "unique_question_rate": round(len(unique) / n, 4),
        "unique_openings": len(openings),
        "mean_pairwise_jaccard": round(statistics.mean(similarities), 4),
        "note": ("descriptive: lower Jaccard means more varied questions. Reported so a "
                 "sampling candidate cannot be selected for a validity gain that came "
                 "from collapsing generation diversity."),
    }


def question_metrics(rows: list[dict]) -> dict:
    """Descriptive properties of the generated questions (not a quality score)."""
    questions = [r["generated_question"] for r in rows
                 if isinstance(r.get("generated_question"), str)
                 and r["generated_question"].strip()]
    if not questions:
        return {"n": 0}
    normalized = [q.strip().lower() for q in questions]
    lengths = [len(q) for q in questions]

    def goal_overlap(question: str, goal: str) -> float:
        q_tokens = {t for t in question.lower().split() if len(t) > 4}
        g_tokens = {t for t in goal.lower().split() if len(t) > 4}
        if not g_tokens:
            return 0.0
        return len(q_tokens & g_tokens) / len(g_tokens)

    overlaps = [goal_overlap(r["generated_question"], r["goal"]) for r in rows
                if isinstance(r.get("generated_question"), str)
                and r["generated_question"].strip()]
    # Per-round distinctness at matched cells (attack, round).
    by_cell: dict[tuple, list[str]] = {}
    for row in rows:
        if isinstance(row.get("generated_question"), str) and row["generated_question"].strip():
            by_cell.setdefault((row["attack"], row["round"]), []).append(
                row["generated_question"].strip().lower())
    distinct_ratios = [len(set(v)) / len(v) for v in by_cell.values() if v]
    return {
        "n": len(questions),
        "mean_chars": round(statistics.mean(lengths), 1),
        "median_chars": round(statistics.median(lengths), 1),
        "distinct_question_ratio": round(statistics.mean(distinct_ratios), 4),
        "mean_goal_token_overlap": round(statistics.mean(overlaps), 4),
        "note": ("descriptive only: length, distinctness and lexical overlap with the "
                 "goal. None of these is a quality score, and a candidate is not "
                 "preferred because of them."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "analysis").mkdir(parents=True, exist_ok=True)

    # ---- control baseline, carried forward from the Stage 4 artifacts ------ #
    s4_summary = json.loads((STAGE4 / "attacker_reliability_summary.json").read_text(encoding="utf-8"))
    s4_survival = json.loads((STAGE4 / "multiturn_survival_summary.json").read_text(encoding="utf-8"))
    s4_analysis = json.loads((STAGE4 / "failure_and_context_analysis.json").read_text(encoding="utf-8"))
    control_baseline = {
        "phase": "17", "stage": "4.5",
        "deliverable": "control_baseline.json",
        "source": "results/phase17_model_qualification/stage4_qwen38_reliability/",
        "configuration": {
            "model_id": common.MODEL_ID, "revision": common.REVISION,
            "architecture": "Qwen3_5ForConditionalGeneration",
            "loader": "AutoModelForImageTextToText", "processor": "Qwen3VLProcessor",
            "quantization": "nf4", "compute_dtype": "bfloat16", "double_quant": True,
            "temperature": 0.7, "top_p": 1.0, "max_new_tokens": None,
            "enable_thinking": False, "prompt": "frozen production prompt",
            "structured_decoding": None,
        },
        "isolated": {**s4_summary["overall"],
                     "empty_query": s4_analysis["semantic_usability"]["isolated"]["empty_query"],
                     "semantically_usable":
                         s4_analysis["semantic_usability"]["isolated"]["semantically_usable"]},
        "multi_turn": s4_survival["overall"],
        "note": "the frozen Stage 4 configuration; never modified by this study",
    }
    (OUT_ROOT / "control_baseline.json").write_text(
        json.dumps(control_baseline, indent=2), encoding="utf-8")

    # ---- per-stage candidate summaries ------------------------------------- #
    stage_files = {
        "A": ("prompt_variant", OUT_ROOT / "prompt_variant_results.jsonl"),
        "B": ("sampling_variant", OUT_ROOT / "sampling_variant_results.jsonl"),
        "C": ("structured_decoding", OUT_ROOT / "structured_decoding_results.jsonl"),
    }
    all_candidates: dict[str, dict] = {}
    for stage, (prefix, path) in stage_files.items():
        rows = read_jsonl(path)
        if not rows:
            print(f"{prefix}: no rows yet")
            continue
        by_config: dict[str, list[dict]] = {}
        for row in rows:
            by_config.setdefault(row["candidate_id"], []).append(row)
        payload = {
            "phase": "17", "stage": "4.5",
            "deliverable": f"{prefix}_summary.json",
            "policy": ("contract-valid = the frozen JSON contract is satisfied (direct or via "
                       "the frozen extraction); semantically usable = contract-valid AND a "
                       "non-empty string generatedQuestion"),
            "candidates": {config: config_summary(rows_, config)
                           for config, rows_ in sorted(by_config.items())},
        }
        (OUT_ROOT / f"{prefix}_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        for config, rows_ in by_config.items():
            all_candidates[config] = {
                "stage": stage,
                "summary": payload["candidates"][config]["overall"],
                "contract_valid_ci95": payload["candidates"][config]["contract_valid_ci95"],
                "usable_ci95": payload["candidates"][config]["semantically_usable_ci95"],
                "question_metrics": question_metrics(rows_),
                "diversity_metrics": diversity_metrics(rows_),
                "temperature": rows_[0].get("temperature"),
                "top_p": rows_[0].get("top_p"),
                "prompt_variant": rows_[0].get("prompt_variant"),
                "structured_output_mode": rows_[0].get("structured_output_mode"),
            }
        print(f"wrote {prefix}_summary.json ({len(by_config)} candidate(s))")

    # ---- multi-turn --------------------------------------------------------- #
    mt_rows = read_jsonl(OUT_ROOT / "multiturn_results.jsonl")
    if mt_rows:
        by_candidate: dict[str, list[dict]] = {}
        for row in mt_rows:
            by_candidate.setdefault(row["candidate_id"], []).append(row)

        def mt_block(subset: list[dict]) -> dict:
            n = len(subset)
            survived = sum(1 for r in subset if r["run_survived"])
            usable = sum(1 for r in subset if r["turns_completed"] >= 1)
            calls = sum(r["attacker_calls_attempted"] for r in subset)
            valid = sum(r["attacker_calls_valid"] for r in subset)
            return {
                "runs": n,
                "runs_survived": survived,
                "run_survival_rate": round(survived / n, 4) if n else None,
                "run_survival_ci95": wilson_interval(survived, n),
                "usable_runs": usable,
                "usable_run_rate": round(usable / n, 4) if n else None,
                "usable_run_ci95": wilson_interval(usable, n),
                "attacker_calls": calls,
                "attacker_calls_valid": valid,
                "attacker_calls_usable": sum(
                    1 for r in subset for c in r["attacker_calls"]
                    if c["category"] in VALID and (c.get("generated_question") or "").strip()),
                "empty_query_terminations": sum(
                    1 for r in subset if r.get("termination_mode") == "terminated_by_empty_query"),
                "attacker_failure_terminations": sum(
                    1 for r in subset if not r["run_survived"]),
                "mean_completed_turns": round(
                    statistics.mean(r["turns_completed"] for r in subset), 3),
                "target_calls": sum(r["target_calls"] for r in subset),
                "target_eos": sum(r["target_eos_count"] for r in subset),
                "judge_calls": sum(r["judge_calls"] for r in subset),
                "judge_failures": sum(r["judge_failures"] for r in subset),
                "peak_vram_gib": max(r["peak_vram_gib"] for r in subset),
                "wall_s_mean": round(statistics.mean(r["wall_s"] for r in subset), 1),
            }

        mt_summary = {
            "phase": "17", "stage": "4.5",
            "deliverable": "multiturn_summary.json",
            "setting": {"max_turns": mt_rows[0].get("max_turns_setting"),
                        "note": "identical to the Stage 4 controlled setting"},
            "candidates": {config: mt_block(subset) for config, subset in sorted(by_candidate.items())},
        }
        (OUT_ROOT / "multiturn_summary.json").write_text(
            json.dumps(mt_summary, indent=2), encoding="utf-8")
        # Link each qualified candidate to the isolated configuration it was built
        # from, so the comparison table never shows an orphan "-".
        mt_config_map = {
            "MT_control": {"prompt_variant": "A0_control", "temperature": 0.7, "top_p": 1.0},
            "MT_A1": {"prompt_variant": "A1_json_object_statement", "temperature": 0.7,
                      "top_p": 1.0},
            "MT_B2": {"prompt_variant": "A1_json_object_statement", "temperature": 0.3,
                      "top_p": 1.0},
        }
        isolated_by_config = {
            config: block for config, block in all_candidates.items()
            if block.get("stage") in ("A", "B", "C")
        }
        for config, block in mt_summary["candidates"].items():
            mapping = mt_config_map.get(config, {})
            source_id = {
                ("A0_control", 0.7, 1.0): "A0",
                ("A1_json_object_statement", 0.7, 1.0): "A1",
                ("A1_json_object_statement", 0.3, 1.0): "B2",
            }.get((mapping.get("prompt_variant"), mapping.get("temperature"),
                   mapping.get("top_p")))
            existing = all_candidates.setdefault(config, {"stage": "multiturn"})
            existing["multiturn"] = block
            existing.update({
                "configuration": mapping,
                "isolated_source_candidate": source_id,
                "isolated_reference": (isolated_by_config.get(source_id) or {}).get("summary"),
            })
        print(f"wrote multiturn_summary.json ({len(by_candidate)} candidate(s))")

    # ---- comparison table (Part 25) ---------------------------------------- #
    comparison = {
        "phase": "17", "stage": "4.5",
        "deliverable": "candidate_comparison.json",
        "control": {
            "candidate_id": "stage4_control",
            "contract_valid_rate": control_baseline["isolated"]["valid_rate"],
            "semantically_usable_rate": round(
                control_baseline["isolated"]["semantically_usable"]
                / control_baseline["isolated"]["n"], 4),
            "empty_query_rate": round(
                control_baseline["isolated"]["empty_query"] / control_baseline["isolated"]["n"], 4),
            "prose_failure": control_baseline["isolated"]["failure_shapes"].get("prose", 0),
            "n": control_baseline["isolated"]["n"],
            "multi_turn_survival_rate": control_baseline["multi_turn"]["run_survival_rate"],
            "usable_run_rate": control_baseline["multi_turn"]["usable_run_rate"],
            "mean_completed_turns": control_baseline["multi_turn"]["mean_completed_turns"],
            "latency_s_mean": control_baseline["isolated"]["latency_s"]["mean"],
            "peak_vram_gib": control_baseline["multi_turn"]["peak_vram_gib"],
        },
        "candidates": all_candidates,
        "comparison_rule": ("A candidate is called better only on the specific metric "
                            "where its measurement supports it, at the stated sample "
                            "size. Differences within overlapping 95 % intervals are "
                            "reported as not established."),
    }
    (OUT_ROOT / "candidate_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8")

    # ---- stage 4 vs stage 4.5 ---------------------------------------------- #
    a0 = all_candidates.get("A0", {}).get("summary")
    stage4_vs = {
        "phase": "17", "stage": "4.5",
        "deliverable": "stage4_vs_stage4_5.json",
        "purpose": ("the control was re-measured inside this study (same configuration, "
                    "current session) so candidate comparisons rest on contemporaneous "
                    "measurements; the Stage 4 numbers are carried for reference"),
        "stage4_frozen": {
            "isolated_n": control_baseline["isolated"]["n"],
            "contract_valid_rate": control_baseline["isolated"]["valid_rate"],
            "semantically_usable_rate": round(
                control_baseline["isolated"]["semantically_usable"]
                / control_baseline["isolated"]["n"], 4),
            "multi_turn_runs": control_baseline["multi_turn"]["runs"],
            "run_survival_rate": control_baseline["multi_turn"]["run_survival_rate"],
            "usable_run_rate": control_baseline["multi_turn"]["usable_run_rate"],
        },
        "stage4_5_control_rerun": a0,
        "note": ("Stage 4's isolated sample used a different history construction "
                 "(synthetic round history) than this study's 5-round distribution; the "
                 "comparison is therefore between contemporaneous candidates first, and "
                 "to Stage 4 second."),
    }
    (OUT_ROOT / "stage4_vs_stage4_5.json").write_text(
        json.dumps(stage4_vs, indent=2), encoding="utf-8")

    # ---- paired matched-cell comparison against the control ---------------- #
    stage_a_rows = read_jsonl(OUT_ROOT / "prompt_variant_results.jsonl")
    if stage_a_rows:
        control_rows = {r["call_index"]: r for r in stage_a_rows if r["candidate_id"] == "A0"}
        paired: dict[str, dict] = {}
        for config in sorted({r["candidate_id"] for r in stage_a_rows}):
            if config == "A0":
                continue
            rows_ = [r for r in stage_a_rows if r["candidate_id"] == config]
            identical = 0
            compared = 0
            fixed_cells = []
            broken_cells = []
            for row in rows_:
                ctrl = control_rows.get(row["call_index"])
                if not ctrl:
                    continue
                compared += 1
                ctrl_ok = (ctrl["category"] in VALID and not ctrl.get("empty_query"))
                cand_ok = (row["category"] in VALID and not row.get("empty_query"))
                ctrl_q = (ctrl.get("generated_question") or "").strip()
                cand_q = (row.get("generated_question") or "").strip()
                if ctrl_q and cand_q and ctrl_q == cand_q:
                    identical += 1
                if not ctrl_ok and cand_ok:
                    fixed_cells.append({
                        "call_index": row["call_index"], "attack": row["attack"],
                        "round": row["round"], "control_category": ctrl["category"],
                        "control_raw": ctrl.get("raw_output"),
                        "candidate_question": cand_q[:200],
                    })
                if ctrl_ok and not cand_ok:
                    broken_cells.append({
                        "call_index": row["call_index"], "attack": row["attack"],
                        "round": row["round"], "candidate_category": row["category"],
                        "candidate_raw": row.get("raw_output"),
                    })
            paired[config] = {
                "compared_cells": compared,
                "identical_questions": identical,
                "identical_question_rate": round(identical / compared, 4) if compared else None,
                "cells_control_failed_variant_usable": len(fixed_cells),
                "cells_control_usable_variant_failed": len(broken_cells),
                "fixed_cell_examples": fixed_cells[:5],
                "broken_cell_examples": broken_cells[:5],
            }
        (OUT_ROOT / "analysis" / "paired_matched_cells.json").write_text(
            json.dumps({
                "phase": "17", "stage": "4.5",
                "purpose": ("seeds are paired across candidates, so the same call index is the "
                            "same (attack, round, goal, seed) cell; this reports what changed at "
                            "that level rather than only in aggregate"),
                "control": "A0",
                "comparisons": paired,
            }, indent=2), encoding="utf-8")
        print("wrote analysis/paired_matched_cells.json")

    # ---- artifact index ----------------------------------------------------- #
    index = {
        "phase": "17", "stage": "4.5",
        "layout": {
            "top_level": "the artifacts named in the stage brief",
            "prompt_variants/, sampling_variants/, structured_decoding/":
                "per-candidate raw JSONL and per-candidate summaries",
            "multiturn/": "per-run raw records and conversations",
            "analysis/": "descriptive checks (semantic drift, per-round detail)",
        },
        "artifacts": sorted(p.name for p in OUT_ROOT.glob("*") if p.is_file()),
    }
    (OUT_ROOT / "analysis" / "artifact_index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8")

    print("\n=== candidate overview ===")
    for config, block in sorted(all_candidates.items()):
        s = block.get("summary") or {}
        mt = block.get("multiturn") or {}
        print(f"  {config:6s} n={s.get('n', '-')} valid={s.get('contract_valid_rate', '-')} "
              f"usable={s.get('semantically_usable_rate', '-')} "
              f"empty={s.get('empty_query', '-')} prose={s.get('prose_failure', '-')} "
              f"| mt survival={mt.get('run_survival_rate', '-')} "
              f"usable_runs={mt.get('usable_run_rate', '-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
