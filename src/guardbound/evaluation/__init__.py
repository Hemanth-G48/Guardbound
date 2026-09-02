"""Phase 7 — Evaluation framework.

Exposes the paper's evaluation metrics:
    - ASR (judge-based, per-behavior)
    - MMLU (helpful, supports steered post-filtering)
    - MTBench (two-turn, supports refusal replacement)
    - over-refusal (XSTest / JBB / PHTest)
    - guardrail F1 (HarmBench / Aegis / WildGuard)

All metrics share:
    - the ``EvaluationResult`` reporting row
    - the ``JudgeCache`` for paid/external judge calls
    - the ``refusal_detector`` for over-refusal
    - the ``PromptGuard`` interface for guardrail backends
"""
from .asr import (
    ASRJudge,
    ASRVerdict,
    DEFAULT_ASR_PROMPT_PATH,
    build_asr_prompt,
    compute_asr,
    evaluate_asr,
    stratify_asr_by_attack,
)
from .cache import JudgeCache, make_cache_key
from .guardrail_f1 import (
    LOADERS as GUARDRAIL_LOADERS,
    BinaryMetrics,
    GuardExample,
    GuardResult,
    NBFPromptGuard,
    compute_binary_metrics,
    compute_f1,
    evaluate_guard,
    make_nbf_guard_from_checkpoint,
)
from .guards import (
    LlamaGuardGuard,
    OpenAIModerationGuard,
    PromptGuard,
    ShieldGemmaGuard,
)
from .mmlu import (
    MMLUQuestion,
    MMLUResult,
    evaluate_mmlu,
    load_mmlu_jsonl,
)
from .mtbench import (
    MTBenchExample,
    MTBenchResult,
    evaluate_mtbench,
    judge_mtbench_response,
    judge_single_turn,
    load_mtbench_jsonl,
)
from .over_refusal import (
    LOADERS as OVER_REFUSAL_LOADERS,
    OverRefusalExample,
    OverRefusalResult,
    evaluate_over_refusal,
    load_jbb_benign,
    load_phtest_harmless,
    load_xstest,
)
from .refusal_detector import is_refusal
from .reporting import (
    EvaluationResult,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    filter_results,
    read_jsonl,
    to_latex_table,
    to_markdown_table,
    write_jsonl,
)

__all__ = [
    # ASR
    "ASRJudge", "ASRVerdict", "DEFAULT_ASR_PROMPT_PATH",
    "build_asr_prompt", "compute_asr", "evaluate_asr",
    "stratify_asr_by_attack",
    # MMLU
    "MMLUQuestion", "MMLUResult", "evaluate_mmlu", "load_mmlu_jsonl",
    # MTBench
    "MTBenchExample", "MTBenchResult", "evaluate_mtbench",
    "judge_mtbench_response", "judge_single_turn", "load_mtbench_jsonl",
    # Over-refusal
    "OVER_REFUSAL_LOADERS", "OverRefusalExample", "OverRefusalResult",
    "evaluate_over_refusal", "load_jbb_benign", "load_phtest_harmless",
    "load_xstest", "is_refusal",
    # Guardrail F1
    "GUARDRAIL_LOADERS", "BinaryMetrics", "GuardExample", "GuardResult",
    "NBFPromptGuard", "compute_binary_metrics", "compute_f1",
    "evaluate_guard", "make_nbf_guard_from_checkpoint",
    "PromptGuard", "OpenAIModerationGuard", "ShieldGemmaGuard", "LlamaGuardGuard",
    # Cache + reporting
    "JudgeCache", "make_cache_key",
    "EvaluationResult", "HIGHER_IS_BETTER", "LOWER_IS_BETTER",
    "filter_results", "read_jsonl",
    "to_latex_table", "to_markdown_table", "write_jsonl",
]
