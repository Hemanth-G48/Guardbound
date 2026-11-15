"""One-off: emit official prompt bytes from the author's release into a module.

Reads the official NBF-LLM sources via the parity loaders, captures the rubric
prompt by driving the official function with a recording ``generate``, and writes
``src/guardbound/attacks/_official_prompts.py`` with exact ``repr`` strings.
"""
import copy
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound")
sys.path.insert(0, str(ROOT))

from tests.parity.utils import load_official_attack, load_official_utils  # noqa: E402

cresc = load_official_attack("crescendomation")
opp = load_official_attack("opposite_day")
acr = load_official_attack("acronym")
utils = load_official_utils()

records = []


def cap_official(payload):
    calls = []

    def _generate(messages, json_format=False, temperature=0.7):
        calls.append(copy.deepcopy(messages))
        return copy.deepcopy(payload)

    return _generate, calls


gen, calls = cap_official({"score": 1})
utils.evaluate_with_rubric("Q-SENTINEL", "R-SENTINEL", "T-SENTINEL", gen)
rubric_full = calls[0][0]["content"]

lines = [
    '"""Byte-exact prompt text recovered from the author\'s NBF-LLM release.',
    "",
    "GENERATED FILE — do not hand-edit. Regenerate with",
    "``python _codegen_official_prompts.py`` (Phase 22). Each constant is the",
    "``repr`` of the string read from the official source, so whitespace,",
    "trailing-space lines and Unicode code points are preserved exactly.",
    "",
    "Source files:",
    "  attacks/crescendomation/prompts.py  (reference; already byte-identical)",
    "  attacks/opposite_day/prompts.py",
    "  attacks/acronym/prompts.py",
    "  attacks/utils/evaluate_with_rubric.py  (rubric text, captured rendered)",
    '"""',
    "from __future__ import annotations",
    "",
    f"OPPOSITE_DAY_SYSTEM_PROMPT_OFFICIAL = {opp.OPPOSITE_DAY_SYSTEM_PROMPT!r}",
    "",
    f"ACRONYM_SYSTEM_PROMPT_OFFICIAL = {acr.ACRONYM_SYSTEM_PROMPT!r}",
    "",
    "# The rubric prompt is built inline in the official code. This is the",
    "# rendered message for the sentinel triple below; the three sentinels are",
    "# substituted back to placeholders by the consumer.",
    'RUBRIC_SENTINEL_QUERY = "Q-SENTINEL"',
    'RUBRIC_SENTINEL_RESPONSE = "R-SENTINEL"',
    f"RUBRIC_PROMPT_OFFICIAL = {rubric_full!r}",
    "",
]
out = ROOT / "src/guardbound/attacks/_official_prompts.py"
out.write_text("\n".join(lines), encoding="utf-8")

print(f"wrote {out}  ({out.stat().st_size} bytes)")
print("opposite_day len", len(opp.OPPOSITE_DAY_SYSTEM_PROMPT))
print("acronym len", len(acr.ACRONYM_SYSTEM_PROMPT))
print("rubric len", len(rubric_full))
print("sentinels present:", "Q-SENTINEL" in rubric_full, "R-SENTINEL" in rubric_full,
      "T-SENTINEL" in rubric_full)
