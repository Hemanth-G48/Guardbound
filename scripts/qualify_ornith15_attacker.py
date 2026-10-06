#!/usr/bin/env python3
"""Ornith-1.5-9B attacker qualification (Phase 16) — thin entry point.

One implementation, two entry points: the staged harness lives in
``qualify_glm_attacker.py`` and is attacker-agnostic. This wrapper supplies the
Ornith context (profile, config, output dirs, label) and otherwise behaves
identically — same Stage A/B/C gates, same frozen JSON parser, same "no repair,
no retry, no fallback" contract.

    python scripts/qualify_ornith15_attacker.py --stage qwen        # Qwen regression
    python scripts/qualify_ornith15_attacker.py --stage switching   # Qwen -> Ornith -> Qwen
    python scripts/qualify_ornith15_attacker.py --stage A           # 10 isolated generations
    python scripts/qualify_ornith15_attacker.py --stage B           # 5 multi-turn sequences
    python scripts/qualify_ornith15_attacker.py --stage C           # 3 x 3 attacks (gated)

Any flag supplied on the command line wins: the defaults below are only appended
when the caller did not provide that flag.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from qualify_glm_attacker import main as _harness_main  # noqa: E402

DEFAULTS = {
    "--profile": "ornith15",
    "--config": str(ROOT / "configs" / "attacker_ornith15.yaml"),
    "--out-dir": str(ROOT / "results" / "phase16_ornith15" / "qualification"),
    "--stage-c-dir": str(ROOT / "results" / "phase16_ornith15" / "stage_c"),
    "--label": "Ornith-1.5-9B",
}


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    for flag, value in DEFAULTS.items():
        if flag not in args:
            args += [flag, value]
    # ensure the output directories exist before any redirect/logging happens
    for key in ("--out-dir", "--stage-c-dir"):
        Path(DEFAULTS[key]).mkdir(parents=True, exist_ok=True)
    return _harness_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
