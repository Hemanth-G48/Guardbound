"""MMLU and MTBench post-processing rules.

These are BENCHMARK-SPECIFIC post-processing rules, NOT the runtime Q-filter.

Runtime Q-filter:   h + eta >= 0  => FILTER
MMLU post-process:  h > 0         => WRONG
MTBench post-process: h > 0       => REPLACE with refusal string

Keep these as separate functions. Do not merge with the runtime filter.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# MTBench replacement string
# --------------------------------------------------------------------------- #

# Paper B.1: exact replacement string for MTBench
MTBENCH_REFUSAL_STRING = (
    "I'm sorry, but I cannot provide information on that. "
    "It is important to prioritize safety and ethical considerations "
    "in our conversation. If you have any other questions on a different "
    "topic, feel free to ask!"
)


# --------------------------------------------------------------------------- #
# MMLU post-processing
# --------------------------------------------------------------------------- #

def mmlu_filter_answer(
    h_value: float,
    original_correct: bool,
) -> bool:
    """MMLU post-processing: mark answer wrong if h > 0.

    Paper B.1 rule:
        If h > 0, the answer is considered WRONG.
        Otherwise preserve original_correct.

    Note: This uses h > 0 (strict), NOT h >= 0 or h + eta >= 0.
    """
    if h_value > 0:
        return False  # forced wrong
    return original_correct


# --------------------------------------------------------------------------- #
# MTBench post-processing
# --------------------------------------------------------------------------- #

def mtbench_replace_answer(
    h_value: float,
    original_answer: str,
) -> str:
    """MTBench post-processing: replace answer with refusal string if h > 0.

    Paper B.1 rule:
        When h > 0, replace the answer with the exact refusal string.
        When h <= 0, return the original answer.

    Note: This uses h > 0 (strict), NOT h >= 0 or h + eta >= 0.
    """
    if h_value > 0:
        return MTBENCH_REFUSAL_STRING
    return original_answer
