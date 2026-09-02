"""Tests for MMLU and MTBench post-processing rules."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from guardbound.defense.post_processing import (
    mmlu_filter_answer,
    mtbench_replace_answer,
    MTBENCH_REFUSAL_STRING,
)


class TestMMLU:
    def test_h_negative_preserves(self):
        """h < 0: preserve original correctness."""
        assert mmlu_filter_answer(-0.1, True) is True
        assert mmlu_filter_answer(-0.1, False) is False

    def test_h_zero_preserves(self):
        """h == 0: preserve original correctness (not flagged)."""
        assert mmlu_filter_answer(0.0, True) is True
        assert mmlu_filter_answer(0.0, False) is False

    def test_h_positive_forces_wrong(self):
        """h > 0: force wrong regardless of original."""
        assert mmlu_filter_answer(0.1, True) is False
        assert mmlu_filter_answer(0.1, False) is False


class TestMTBench:
    def test_h_negative_returns_original(self):
        """h <= 0: return original answer."""
        original = "The answer is 42."
        assert mtbench_replace_answer(-0.1, original) == original

    def test_h_zero_returns_original(self):
        """h == 0: return original answer."""
        original = "The answer is 42."
        assert mtbench_replace_answer(0.0, original) == original

    def test_h_positive_returns_exact_string(self):
        """h > 0: return exact refusal string."""
        result = mtbench_replace_answer(0.1, "anything")
        assert result == MTBENCH_REFUSAL_STRING

    def test_exact_string_equality(self):
        """The replacement string must be EXACTLY correct."""
        expected = (
            "I'm sorry, but I cannot provide information on that. "
            "It is important to prioritize safety and ethical considerations "
            "in our conversation. If you have any other questions on a different "
            "topic, feel free to ask!"
        )
        assert MTBENCH_REFUSAL_STRING == expected

    def test_does_not_modify_original(self):
        """When h <= 0, original string is returned unmodified."""
        original = "Test answer with 'quotes' and special chars!"
        assert mtbench_replace_answer(-1.0, original) is original
