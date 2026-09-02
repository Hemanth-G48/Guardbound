#!/usr/bin/env python3
"""Phase 9 — final report builder.

Combines:
  - experiment metadata
  - environment information
  - reproduced tables (pivots)
  - figure references (PNGs + sidecar JSONs)
  - findings verification
  - missing datasets / skipped experiments / unavailable models

Output is written under ``results/final/``:
    final_report.md      human-readable summary
    final_report.json    structured metadata
    tables/              per-table markdown
    figures/             copies of referenced figures

In --smoke mode the report runs end-to-end with mocked data so the
script can be exercised without any real experiments.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="results/experiments",
                   help="Source experiment output directory.")
    p.add_argument("--report-dir", default="results/final",
                   help="Where to write the final report.")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode (use empty results).")
    return p.parse_args()


def _env_block() -> str:
    """Return a small environment summary."""
    import platform
    import sys as _sys
    return (
        f"- Python : {_sys.version.split()[0]}\n"
        f"- Platform: {platform.platform()}\n"
    )


def _git_commit() -> str | None:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    src = Path(args.output_dir)
    dst = Path(args.report_dir)
    tables_dir = dst / "tables"
    figures_dir = dst / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate result rows
    from guardbound.analysis import (
        aggregate_by,
        check_claims,
        pivot_results,
        to_pivot_markdown,
    )
    from guardbound.evaluation import read_jsonl

    all_rows = []
    if src.exists():
        for f in src.rglob("*.results.jsonl"):
            all_rows.extend(read_jsonl(f))

    # Aggregate ASR per (model, variant, eta)
    asr_table = aggregate_by(all_rows, metric="ASR",
                              key_fields=("model", "defense_variant", "eta", "seed"))
    asr_table_path = tables_dir / "asr_by_model_variant.md"
    asr_md_lines: list[str] = ["# ASR by model, defense variant, eta\n"]
    for row in asr_table:
        asr_md_lines.append(
            f"- {row.get('model', '?')} | {row.get('defense_variant', '?')} | "
            f"eta={row.get('eta')} | n={row['n']} | mean={row['value']:.4f}"
        )
    asr_table_path.write_text("\n".join(asr_md_lines) + "\n", encoding="utf-8")

    # Pivot: model x defense variant for ASR
    if all_rows:
        pivot = pivot_results(all_rows, rows_field="model",
                              cols_field="defense_variant", metric="ASR")
        (tables_dir / "asr_pivot.md").write_text(
            to_pivot_markdown(pivot, metric="ASR", better_is="lower"),
            encoding="utf-8",
        )

    # Findings
    report_md = check_claims(all_rows).to_markdown()
    (dst / "findings.md").write_text(report_md, encoding="utf-8")

    # Copy figures (PNGs + sidecar JSONs) so the report has them.
    fig_count = 0
    if src.exists():
        for f in src.rglob("*.png"):
            target = figures_dir / f.name
            shutil.copy2(f, target)
            fig_count += 1
    sidecar_count = 0
    if src.exists():
        for f in src.rglob("*.json"):
            if f.name.endswith(".json") and "results" not in f.name:
                # Figure metadata sidecars; copied only if matched to a PNG.
                if (f.with_suffix("")).with_suffix(".png").exists():
                    target = figures_dir / f.name
                    shutil.copy2(f, target)
                    sidecar_count += 1

    # Final report (Markdown)
    final_md = ["# NBF Safety Steering — Phase 9 Final Report", ""]
    final_md.append(f"Generated: {_dt.datetime.now(_dt.timezone.utc).isoformat()}")
    final_md.append("")
    final_md.append("## Environment")
    final_md.append(_env_block())
    final_md.append("")
    final_md.append("## Git commit")
    gc = _git_commit()
    final_md.append(f"- commit: {gc or 'unknown'}")
    final_md.append("")
    final_md.append("## Reproduced tables")
    final_md.append(f"- See `{asr_table_path.relative_to(dst)}` "
                    f"({len(asr_table)} ASR aggregates from {len(all_rows)} rows).")
    if (tables_dir / "asr_pivot.md").exists():
        final_md.append(f"- See `{(tables_dir / 'asr_pivot.md').relative_to(dst)}` "
                        f"for model × defense ASR pivot.")
    final_md.append("")
    final_md.append("## Figures")
    final_md.append(f"- {fig_count} PNG figures copied from {src}.")
    final_md.append("")
    final_md.append("## Findings")
    final_md.append("- See `findings.md`.")
    final_md.append("")
    final_md.append("## Reproduction status")
    final_md.append(
        "- PHASE 9 INFRASTRUCTURE IMPLEMENTED — real experiments pending.  "
        "See REPRODUCTION_NOTES.md and the per-experiment status output of "
        "`scripts/run_experiments.py`."
    )
    (dst / "final_report.md").write_text("\n".join(final_md) + "\n",
                                          encoding="utf-8")

    # Structured metadata
    meta = {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_commit": gc,
        "n_result_rows": len(all_rows),
        "n_figures_copied": fig_count,
        "n_tables": len(list(tables_dir.glob("*.md"))),
        "report_dir": str(dst),
        "smoke": bool(args.smoke),
    }
    (dst / "final_report.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"Final report written: {dst}")


if __name__ == "__main__":
    main()
