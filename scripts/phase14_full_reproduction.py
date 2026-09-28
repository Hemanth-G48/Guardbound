#!/usr/bin/env python3
"""Phase 14 — full-scale local substitute-model reproduction study.

Runs the complete 200-goal × 3-attack × 2-NBF-condition study:

    200 goals × {crescendo_paper, opposite_day, acronym} × {NBF OFF, NBF ON}
        = 1200 runs total, batched as 4 × 50-goal batches (300 runs each).

Each run produces one JSONL record with a unique run_id of the form
``{condition}_{attack_short}_{goal_id:03d}`` (e.g. ``OFF_crescendo_000``).

NBF semantics (per Phase 13 report):
  OFF = barrier=None (NBF fully inactive; hard invariant: nbf_scores == [] and
        filtered_queries == 0).
  ON  = official --safety_filtering (candidate filter, plain target calls,
        steer_target=False).  Threshold frozen at 0.0.

Resume is on by default.  ``--no-resume`` re-runs all run_ids in the batch.

Usage:
    python scripts/phase14_full_reproduction.py --batch 0
    python scripts/phase14_full_reproduction.py --batch 0 --attack crescendo_paper --condition off --limit 1
    python scripts/phase14_full_reproduction.py --batch 0 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# Allow multiple OpenMP runtimes on Windows (torch + transformers).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 here.
#
# This file used to set it "to reduce CUDA allocation fragmentation on the
# 24 GB card". Measured (results/phase14/ab_alloc_*.json), it does the
# opposite: loading an identical 7.672 GB model on this card costs
#      max_split_size_mb:128 -> 15.349 GB transient driver-free delta
#      unset                  ->  7.676 GB transient driver-free delta
# i.e. a 7.675 GB phantom reservation that empty_cache() then reclaims. With
# the pinned-slot rotation that phantom alone is enough to refuse every model
# load (post-load verify saw 0.45 GB free against a 1.5 GB floor).

import yaml  # noqa: E402

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger("phase14")
# Silence chatty transitive loggers (httpx, transformers) — experiment-critical
# logs (NBF decisions, model calls, termination, failures) are still emitted.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

import torch  # noqa: E402

from guardbound.llm.base import ChatLLM  # noqa: E402

CONFIG_PATH = Path("configs/reproduction_three_model.yaml")
OFFICIAL_DATASET = Path(
    "nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/data/test/harmbench_tasks.json"
)

ATTACK_KEYS = {
    "crescendo_paper": "crescendo",
    "opposite_day": "opposite_day",
    "acronym": "acronym",
}
ATTACK_ORDER = ["crescendo_paper", "opposite_day", "acronym"]
BATCH_START = {"0": 0, "1": 50, "2": 100, "3": 150}
BATCH_SIZE = 50
BASE_SEED = 42
NUM_BATCHES = 4

# Frozen author-dataset identity (verified before every batch and by --dry-run).
OFFICIAL_DATASET_SHA256 = "ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb"
OFFICIAL_DATASET_FULL_SIZE = 200

# Author's backtrack limit (attacks/*/run.py: ``C_refused < 10``).
OFFICIAL_MAX_REFUSAL_RETRIES = 10

# Attacks that are deliberately NOT part of Phase 14 (recorded in every manifest).
EXCLUDED_ATTACKS = {
    "actor_attack": "LOCAL_MODEL_CAPABILITY_LIMIT",
    "red_queen": "OFFICIAL_IMPLEMENTATION_NOT_PRESENT",
}

# Declared, non-blocking deviations frozen for the study (never changed after
# Batch 0 starts; see PHASE14_CODE_FIDELITY_AUDIT.md 14.1/15).
DECLARED_DEVIATIONS = [
    "D5: rubric prompt has 4 documented typo/whitespace normalisations",
    "D6/D7: minor apostrophe / trailing-whitespace differences in 3 prompt constants",
    "D-E1: malformed evaluator JSON falls back to score 1 (parse failures counted per run)",
    "D-LLM1: local HF generation uses top_p=1.0",
    "D-LLM2: no provider-level local retry; transient failures become explicit failed runs",
    "D10: author-absent regeneration helper in OppositeDay/Acronym is dead on this path",
    "D11: runner_debug.py holds stale pre-fix loop and is not imported",
    "LOCAL_MODEL_SUBSTITUTION: gpt-4o attacker/target/evaluator -> local Qwen3-4B / Phi-4-mini",
]


def _git(args: list[str]) -> str:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=30, check=False
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # telemetry must never break a run
        return "unknown"


def git_state() -> dict:
    """Code identity for the manifest: HEAD plus a hash of the working tree.

    The audit fixes are uncommitted, so HEAD alone would not identify the code
    that actually executes; ``worktree_sha256`` hashes ``git diff HEAD`` (tracked
    modifications) and the status list (untracked files).
    """
    head = _git(["rev-parse", "HEAD"])
    diff = _git(["diff", "HEAD"])
    status = _git(["status", "--porcelain"])
    blob = (diff + "\n" + status).encode("utf-8", errors="replace")
    return {
        "code_commit": head,
        "audit_commit": head,
        "worktree_sha256": hashlib.sha256(blob).hexdigest()[:16],
        "worktree_dirty": bool(status),
        "tracked_changed_files": sorted(
            line.split()[-1] for line in status.splitlines() if line.strip()
        ),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Phase 14 full-scale 200-goal reproduction study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--batch", type=int, choices=[0, 1, 2, 3], required=True)
    p.add_argument("--attack", choices=list(ATTACK_KEYS) + ["all"], default="all")
    p.add_argument("--condition", choices=["on", "off", "both"], default="both")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--structured-output-mode", default=None,
                   choices=[None, "constrained_json"],
                   help="Phase 14.1: constrained JSON decoding for attacker/evaluator")
    return p.parse_args(argv)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_goals(batch_idx, limit):
    start = BATCH_START[str(batch_idx)]
    with open(OFFICIAL_DATASET, "r", encoding="utf-8") as f:
        records = json.load(f)
    batch_records = records[start : start + BATCH_SIZE]
    if limit is not None:
        batch_records = batch_records[:limit]
    return batch_records


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_seed(goal_id: int, attack_short: str, condition: str) -> int:
    """Deterministic per-run seed from base 42 (resume/reorder-safe)."""
    cond_offset = 0 if condition == "off" else 10000
    attack_offset = {"crescendo": 0, "opposite_day": 100, "acronym": 200}
    return BASE_SEED + goal_id * 1000 + attack_offset.get(attack_short, 0) + cond_offset


def make_run_id(condition: str, attack_short: str, goal_id: int) -> str:
    return f"{condition.upper()}_{attack_short}_{goal_id:03d}"


class CountingChatLLM(ChatLLM):
    """Transparent telemetry proxy wrapping a real ChatLLM."""

    name = "counting-chat-llm"

    def __init__(self, inner: ChatLLM, label: str = "") -> None:
        self._inner = inner
        self.label = label
        self.calls = 0
        self.purpose_counts: dict[str, int] = {}
        self.raw_outputs: list[tuple[str, str, str]] = []
        # D-E1 telemetry: JSON-mode replies that the backend returned as a
        # string that does not parse. Counted per purpose so an evaluator
        # parse failure (which silently falls back to score 1) is
        # distinguishable from a genuine score-1 response. Telemetry only —
        # the returned value is never altered.
        self.json_parse_failures: dict[str, int] = {}
        self.json_calls: dict[str, int] = {}

    @staticmethod
    def _classify_purpose(messages) -> str:
        blob = "\n".join(str(m.get("content", "")) for m in messages if m.get("content"))
        if "responseToEvaluateInput" in blob:
            return "refusal_or_disclaimer"
        if "scale of 1 to 5" in blob:
            return "rubric"
        if "successful, rejective, or unknown" in blob:
            return "refusal_or_disclaimer"
        if "Extract the core harmful target" in blob:
            return "pre_attack"
        if "actor_name" in blob:
            return "pre_attack"
        return "generation"

    def generate(self, *args, **kwargs):
        self.calls += 1
        messages = args[0] if args else kwargs.get("messages", [])
        purpose = self._classify_purpose(messages)
        self.purpose_counts[purpose] = self.purpose_counts.get(purpose, 0) + 1
        out = self._inner.generate(*args, **kwargs)
        if kwargs.get("json_format"):
            self.json_calls[purpose] = self.json_calls.get(purpose, 0) + 1
            if isinstance(out, str):
                try:
                    json.loads(out)
                except (json.JSONDecodeError, TypeError, ValueError):
                    self.json_parse_failures[purpose] = (
                        self.json_parse_failures.get(purpose, 0) + 1
                    )
        if len(self.raw_outputs) < 400:
            kind = type(out).__name__
            snippet = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)
            self.raw_outputs.append((purpose, kind, snippet[:2000]))
        return out

    def reset_counter(self) -> None:
        self.calls = 0
        self.purpose_counts = {}
        self.raw_outputs = []
        self.json_parse_failures = {}
        self.json_calls = {}

    def parse_failure_summary(self) -> dict:
        """JSON-mode parse failures, split into evaluator vs other purposes."""
        failures = dict(self.json_parse_failures)
        evaluator_purposes = ("rubric", "refusal_or_disclaimer")
        return {
            "total": sum(failures.values()),
            "by_purpose": failures,
            "evaluator": sum(v for k, v in failures.items() if k in evaluator_purposes),
            "json_calls": dict(self.json_calls),
        }

    def __getattr__(self, item: str):
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(object.__getattribute__(self, "_inner"), item)


class Phase14Error(RuntimeError):
    """Base for Phase 14 semantic-invariant failures."""

    failure_class = "INFRASTRUCTURE_FAILURE"


class ModelIdentityError(Phase14Error):
    """The configured roles did not resolve to three distinct physical models."""

    failure_class = "MODEL_IDENTITY_FAILURE"


class ResidencyInvariantError(Phase14Error):
    """Residency violated the pinned-slot contract (e.g. >2 models resident)."""

    failure_class = "RESIDENCY_STATE_FAILURE"


class NBFSemanticsError(Phase14Error):
    """An NBF OFF/ON invariant was breached."""

    failure_class = "NBF_SEMANTICS_FAILURE"


class AttackSemanticsError(Phase14Error):
    """The attack was driven outside its official parameterisation."""

    failure_class = "ATTACK_SEMANTICS_FAILURE"


class StructuredJSONError(Phase14Error):
    """Structured-output behaviour did not match the configured decoding mode."""

    failure_class = "STRUCTURED_JSON_FAILURE"


def sample_gpu_state() -> dict:
    """GPU telemetry for the stability log. Never raises."""
    state: dict = {}
    try:
        free, total = torch.cuda.mem_get_info()
        state["gpu_free_gb"] = round(free / 1e9, 3)
        state["gpu_total_gb"] = round(total / 1e9, 3)
        state["gpu_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
        state["gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 3)
    except Exception:  # noqa: BLE001 -- telemetry must never break a run
        pass
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            util, temp, power, used = (
                p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")
            )
            state.update(
                gpu_util_pct=float(util),
                gpu_temp_c=float(temp),
                gpu_power_w=float(power),
                gpu_mem_used_mib=float(used),
            )
    except Exception:  # noqa: BLE001
        pass
    return state


def sample_host_memory() -> dict:
    """Host RAM telemetry (Windows). Never raises."""
    if os.name != "nt":
        return {}
    try:
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {
                "host_mem_load_pct": int(status.dwMemoryLoad),
                "host_mem_avail_gb": round(status.ullAvailPhys / 1e9, 2),
                "host_mem_total_gb": round(status.ullTotalPhys / 1e9, 2),
            }
    except Exception:  # noqa: BLE001
        pass
    return {}


def residency_record(model_manager, run_id: str) -> dict:
    """Residency telemetry for one run: what loaded/evicted, and what it cost.

    Filters the manager's process-wide event log by run id (the run context is
    attached before the run's first generation), so a run's rotation cost is
    attributable rather than amortised.
    """
    if model_manager is None:
        return {}
    events = [e for e in model_manager.events if e.get("run_id") == run_id]
    loads = [e for e in events if e.get("event") == "load"]
    evicts = [e for e in events if e.get("event") == "evict"]
    refused = [e for e in events if e.get("event") == "load_refused"]
    return {
        "snapshot": model_manager.residency_snapshot(),
        "loads": len(loads),
        "evictions": len(evicts),
        "load_refusals": len(refused),
        "load_seconds": round(sum(e.get("duration_s") or 0.0 for e in loads), 3),
        "eviction_seconds": round(
            sum(e.get("duration_s") or 0.0 for e in evicts), 3
        ),
        "rotations": [
            {
                "role": e.get("role"),
                "action": e.get("action"),
                "model_id": e.get("model_id"),
                "duration_s": e.get("duration_s"),
                "free_before_gb": e.get("free_before_gb"),
                "free_after_gb": e.get("free_after_gb"),
                "freed_gb": e.get("freed_gb"),
                "measured_gb": e.get("measured_gb"),
            }
            for e in loads + evicts
        ],
        "peak_vram_gb": round(model_manager.peak_vram_gb(), 3),
    }


def classify_exception(exc: BaseException) -> tuple[str, str, str]:
    """Explicit failure classification (never a normal attack outcome).

    Returns ``(failure_class, error_type, error_message)``.

    Residency-layer errors (``guardbound.llm.model_manager``) and the Phase 14
    semantic invariants below carry their own ``failure_class`` attribute, so a
    VRAM refusal, a load failure or an NBF invariant breach is never reported as
    a generic infrastructure error.
    """
    name = type(exc).__name__
    msg = str(exc)[:1000]
    low = f"{name} {msg}".lower()
    declared = getattr(exc, "failure_class", None)
    if declared:
        return declared, name, msg
    if name == "AttackGenerationError" or "attackgenerationerror" in low:
        return "JSON_PARSE_ERROR", name, msg
    if "timeout" in low:
        return "TIMEOUT", name, msg
    if "out of memory" in low or "cuda error" in low or "device-side assert" in low:
        return "INFRASTRUCTURE_ERROR", name, msg
    if name in ("NotImplementedError",):
        return "LOCAL_MODEL_CAPABILITY_LIMIT", name, msg
    if "generate" in low or "forward" in low or "token" in low:
        return "MODEL_GENERATION_ERROR", name, msg
    return "INFRASTRUCTURE_ERROR", name, msg


def existing_run_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    ids = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("run_id")
                if rid:
                    ids.add(rid)
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def append_result(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Canonical run-record schema (Phase 14.8)
# --------------------------------------------------------------------------- #
# The completed-run path and the failure path used to construct their records
# independently, which let them drift: an attacker JSON failure produced a
# record missing 22 fields that a completed run carries — even though 17 of them
# were recoverable from objects still alive at the failure point. Both paths now
# build through one skeleton, so their key sets are identical by construction.
#
# VALUE SEMANTICS — deliberately NOT interchangeable:
#   None  -> the measurement is unavailable / was not legitimately observed
#            (e.g. num_turns on a run that raised before a Conversation existed)
#   []    -> a genuinely empty collection (e.g. nbf_scores when NBF is OFF)
#   0     -> a genuinely observed zero (e.g. filtered_queries for an OFF run)
#
# A failed run must stay unmistakable. `success`, `failure_class`, `error_type`,
# `error_message` and `error` are the authoritative failure markers; a failure
# record must never be made to look like a legitimate attack outcome by
# substituting 0/[] for a measurement that was never taken.
RUN_RECORD_FIELDS = (
    # identity / run coordinates
    "run_id", "goal_id", "goal", "goal_text", "condition", "attack", "seed",
    # model + harness identity
    "attacker_model", "target_model", "evaluator_model", "embedding_model",
    "checkpoint_sha256", "dataset_sha256", "config_hash", "code_commit",
    "worktree_sha256",
    # frozen configuration
    "dtype", "quantization", "max_rounds", "threshold", "filter_trials",
    "top_p", "generation_config", "target_system",
    # outcome + failure visibility
    "success", "failure_class", "error", "error_type", "error_message",
    "termination_reason",
    # conversation
    "turns", "num_turns", "final_response", "accepted_query_count",
    # rubric telemetry
    "rubric_scores", "final_score", "refusal_count", "refused_records",
    # NBF telemetry
    "nbf_enabled", "nbf_scores", "filtered_queries", "filter_count", "nbf_stats",
    # runtime / resources
    "runtime", "runtime_seconds", "peak_vram_gb", "residency", "system",
    # per-role call telemetry
    "llm_calls", "attacker_purpose_calls", "attacker_parse_failures",
    "attacker_raw_outputs", "evaluator_purpose_calls",
    "evaluator_parse_failures", "target_purpose_calls",
)


def build_run_record(**overrides) -> dict:
    """Canonical run-record construction — the single construction site.

    Returns every field in ``RUN_RECORD_FIELDS`` defaulted to ``None`` ("not
    measured on this path"), then applies the caller's observed values. Callers
    only pass what they actually observed; nothing is invented to fill a gap.
    """
    record = {field: None for field in RUN_RECORD_FIELDS}
    record.update(overrides)
    return record


def nbf_stats_from(nbf_scores: list, filtered: int) -> dict:
    """NBF summary statistics — one implementation shared by both paths.

    Callers pass the already-rounded ``nbf_scores`` list, so min/max/mean match
    the completed-run path exactly.
    """
    if nbf_scores:
        return {
            "n": len(nbf_scores),
            "filter_count": filtered,
            "min": round(min(nbf_scores), 4),
            "max": round(max(nbf_scores), 4),
            "mean": round(sum(nbf_scores) / len(nbf_scores), 4),
        }
    return {"n": 0, "filter_count": 0, "min": None, "max": None, "mean": None}


def nbf_telemetry_from(attack) -> dict:
    """NBF scores / filter counts read from the live attack object.

    Recovery is exact, never estimated: an OFF run genuinely holds ``[]`` and
    ``0`` here, and an ON run holds whatever candidates were scored before the
    failure. Absent NBF observation is never converted into a fake score.
    """
    candidate_scores = list(getattr(attack, "nbf_candidate_scores", []) or [])
    candidate_verdicts = list(getattr(attack, "nbf_candidate_verdicts", []) or [])
    nbf_scores = [round(float(s), 4) for s in candidate_scores]
    filtered = sum(1 for v in candidate_verdicts if v is False)
    return {
        "nbf_scores": nbf_scores,
        "filtered_queries": filtered,
        "filter_count": filtered,
        "nbf_stats": nbf_stats_from(nbf_scores, filtered),
    }


def rubric_telemetry_from(attack) -> dict:
    """Rubric/refusal telemetry read from the live attack object.

    Same source as the completed-run path (``attack._scores``). The completed
    path additionally trims the list to the turns the runner recorded; on the
    failure path no Conversation object exists, so the attack's accepted scores
    are reported as they stand. Same source, no turn-alignment, nothing
    estimated.
    """
    scores_all = list(getattr(attack, "_scores", []) or [])
    accepted = [s for s in scores_all if s != "refused"]
    rubric_scores = [s for s in accepted if s is not None]
    refusal_count = (
        attack.get_refusal_count() if hasattr(attack, "get_refusal_count") else 0
    )
    return {
        "rubric_scores": rubric_scores,
        "final_score": rubric_scores[-1] if rubric_scores else None,
        "refusal_count": refusal_count,
        "refused_records": len(scores_all) - len(accepted),
    }


def raw_attacker_outputs(llm, limit: int = 20) -> list:
    """Captured raw attacker outputs.

    Returns ``[]`` only when nothing was captured — never as a stand-in for an
    output that was captured. This is what keeps the malformed JSON behind a
    JSON_PARSE_ERROR available as diagnostic evidence.
    """
    if not isinstance(llm, CountingChatLLM):
        return []
    return [
        {"purpose": p, "type": k, "raw": s}
        for p, k, s in llm.raw_outputs[:limit]
    ]


def build_failure_record(
    *,
    run_id: str,
    goal_id: int,
    goal_record: dict,
    condition: str,
    attack_key: str,
    attack_short: str,
    seed: int,
    attack,
    metadata: dict,
    max_turns: int,
    eta: float,
    started_at: float,
    failure_class: str,
    error_type: str,
    error_message: str,
    error_text: str,
    attacker_llm=None,
    target_llm=None,
    evaluator_llm=None,
    model_manager=None,
) -> dict:
    """Build the result record for a run that raised.

    Goes through the same canonical skeleton as the completed-run path, so the
    key set is identical; only values actually observed are filled in.

    Fields left at ``None`` on purpose (never 0/[]):
      ``turns``, ``num_turns``, ``accepted_query_count``
          They derive from the Conversation that ``run_one_phase14`` would have
          returned, which never existed because the attack raised. ``0``/``[]``
          would assert "no turns occurred", which is false; reconstructing them
          from the attack's internal histories would substitute ``"(Summary) …"``
          strings for the raw responses ``conv.turns`` holds.
      ``final_response``
          There is no final response for a run that errored.
      ``termination_reason``
          The run reached no normal termination state. Introducing an ``"error"``
          enum member would be a termination-semantics change, which is out of
          scope; the failure is carried by the fields below.

    The authoritative failure markers are ``success``, ``failure_class``,
    ``error_type``, ``error_message`` and ``error``. A failure record must never
    be made to look like a legitimate attack outcome.
    """
    record = build_run_record(
        run_id=run_id,
        goal_id=goal_id,
        goal=goal_record.get("task"),
        goal_text=goal_record.get("task"),
        condition=condition,
        attack=attack_key,
        seed=seed,
        attacker_model=metadata["attacker_model"],
        target_model=metadata["target_model"],
        evaluator_model=metadata.get("evaluator_model"),
        embedding_model=metadata["embedding_model"],
        checkpoint_sha256=metadata.get("checkpoint_sha256"),
        dataset_sha256=metadata.get("dataset_sha256"),
        config_hash=metadata.get("config_hash"),
        code_commit=metadata.get("code_commit"),
        worktree_sha256=metadata.get("worktree_sha256"),
        dtype=metadata.get("dtype"),
        quantization=metadata.get("quantization"),
        max_rounds=max_turns,
        threshold=eta,
        filter_trials=(metadata.get("filter_trials") or {}).get(attack_short),
        top_p=metadata.get("top_p"),
        generation_config=metadata.get("generation_config"),
        target_system=goal_record.get("target_system") or None,
        # --- failure visibility (authoritative markers) ---
        success=False,
        failure_class=failure_class,
        error=error_text,
        error_type=error_type,
        error_message=error_message,
        # --- runtime / resources ---
        runtime=round(time.time() - started_at, 3),
        runtime_seconds=round(time.time() - started_at, 3),
        nbf_enabled=condition == "on",
        llm_calls={
            "attacker": getattr(attacker_llm, "calls", None),
            "target": getattr(target_llm, "calls", None),
            "evaluator": getattr(evaluator_llm, "calls", None),
        },
        # Recovered exactly from the still-live attack object — the same sources
        # the completed-run path reads. An OFF run genuinely holds []/0 here; an
        # ON run holds whatever was scored before the failure. Nothing inferred.
        **rubric_telemetry_from(attack),
        **nbf_telemetry_from(attack),
        # Diagnostic evidence for the malformed reply itself.
        attacker_raw_outputs=raw_attacker_outputs(attacker_llm),
    )
    if isinstance(attacker_llm, CountingChatLLM):
        record["attacker_purpose_calls"] = dict(attacker_llm.purpose_counts)
        record["attacker_parse_failures"] = attacker_llm.parse_failure_summary()
    if isinstance(evaluator_llm, CountingChatLLM):
        record["evaluator_purpose_calls"] = dict(evaluator_llm.purpose_counts)
        record["evaluator_parse_failures"] = evaluator_llm.parse_failure_summary()
    if isinstance(target_llm, CountingChatLLM):
        record["target_purpose_calls"] = dict(target_llm.purpose_counts)
    if torch.cuda.is_available():
        record["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    record["residency"] = residency_record(model_manager, run_id)
    record["system"] = {**sample_gpu_state(), **sample_host_memory()}
    return record


def make_models(cfg: dict, structured_output_mode: str | None = None):
    """Build the three DISTINCT model roles under a pinned-slot rotation.

    Phase 14.2 architecture (see PHASE14_THREE_MODEL_RESIDENCY_ANALYSIS.md):
    the attacker is pinned GPU-resident for the whole run and the target and
    evaluator alternate in the second slot. Co-residency of all three is
    arithmetically impossible on this card — measured weights are 24.12 GB
    against 24.39 GB free at idle — so the residency layer rotates beneath an
    unchanged logical call order.

    Every role is built through ``build_role_llm`` so all three go through the
    ``ModelManager``; nothing bypasses eviction. ``structured_output_mode`` is
    forwarded to each local backend, which is what keeps Phase 14.1 constrained
    JSON decoding active on the managed path.

    Returns ``(attacker_llm, target_llm, evaluator_llm, manager)``.

    Raises ``ModelIdentityError`` when the configured roles do not resolve to
    three distinct physical models — three distinct models is the entire point
    of this configuration, and a silent dedup would invalidate the study.
    """
    from guardbound.llm.model_manager import ModelManager
    from guardbound.llm.provider_factory import build_role_llm

    models_cfg = cfg["models"]
    device = cfg.get("hardware", {}).get("device", "cuda")
    manager = ModelManager(device=device)

    llms: dict[str, ChatLLM] = {}
    for role in ("attacker", "target", "evaluator"):
        role_cfg = models_cfg.get(role)
        if not role_cfg:
            raise RuntimeError(f"models.{role} section missing from config")
        llms[role] = build_role_llm(
            role_cfg,
            role,
            manager=manager,
            structured_output_mode=structured_output_mode,
        )

    identities = {role: models_cfg[role]["model"] for role in llms}
    if len(set(identities.values())) != 3:
        raise ModelIdentityError(
            "three-model configuration requires three DISTINCT physical models; "
            f"resolved {identities}"
        )

    # Pinned-slot rotation: the attacker never leaves the GPU, so the target and
    # evaluator are the only models that pay a reload.
    manager.pin("attacker")

    return (
        CountingChatLLM(llms["attacker"], "attacker"),
        CountingChatLLM(llms["target"], "target"),
        CountingChatLLM(llms["evaluator"], "evaluator"),
        manager,
    )


def make_barrier_and_embed(cfg: dict):
    from guardbound.models.compat import load_original_checkpoint
    from guardbound.models.predictor import NeuralBarrierFunction
    from guardbound.embeddings import get_embed_fn

    ckpt_path = Path(cfg["nbf"]["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"NBF checkpoint not found: {ckpt_path}")
    dynamics, predictor = load_original_checkpoint(ckpt_path, device="cuda")
    barrier = NeuralBarrierFunction(dynamics=dynamics, predictor=predictor)
    barrier.to("cuda")
    embed_fn = get_embed_fn(cfg["embedding"]["model"])
    return barrier, embed_fn


def build_metadata(cfg: dict, ckpt_sha: str | None, dataset_sha: str) -> dict:
    hw = cfg.get("hardware", {})
    return {
        "phase": 14,
        "study_type": "local_substitute_model_reproduction",
        "attacker_model": cfg["models"]["attacker"]["model"],
        "target_model": cfg["models"]["target"]["model"],
        "evaluator_model": cfg["models"]["evaluator"]["model"],
        "embedding_model": cfg["embedding"]["model"],
        "embedding_backend": "sentence-transformers/all-mpnet-base-v2",
        "checkpoint_path": cfg["nbf"]["checkpoint"],
        "checkpoint_sha256": ckpt_sha,
        "dataset_path": str(OFFICIAL_DATASET),
        "dataset_sha256": dataset_sha,
        "dataset_size": OFFICIAL_DATASET_FULL_SIZE,
        "dataset_sampling": cfg["dataset"].get("sampling"),
        "dataset_shuffle": cfg["dataset"].get("shuffle"),
        "attacks": list(ATTACK_ORDER),
        "conditions": ["on", "off"],
        # `nbf.trials` is keyed by attack-SHORT names ({crescendo, opposite_day,
        # acronym, actor_attack}), while ATTACK_KEYS maps module name ->
        # attack_short. Filtering by `k in ATTACK_KEYS` therefore tested the
        # config keys against ATTACK_KEYS' KEYS ({crescendo_paper, ...}) and
        # silently dropped "crescendo" before the per-run lookup could read it,
        # producing filter_trials == null for Crescendo. Compare against the
        # VALUES so every real attack resolves; `actor_attack` stays excluded
        # because it is not a Phase 14 attack.
        #
        # Scope note: this is a declaration of the frozen configuration value.
        # It does NOT assert that three NBF evaluations run per candidate — the
        # executed filter structure is reported separately by nbf_scores /
        # filtered_queries / filter_count.
        "filter_trials": {
            k: v for k, v in cfg["nbf"]["trials"].items()
            if k in ATTACK_KEYS.values()
        },
        "refusal_retry_limit": OFFICIAL_MAX_REFUSAL_RETRIES,
        "top_p": 1.0,
        "excluded_attacks": EXCLUDED_ATTACKS,
        "declared_deviations": DECLARED_DEVIATIONS,
        "asr_caveat": (
            "Absolute ASR is NOT claimed to reproduce the paper's GPT-4o ASR. "
            "This study measures the controlled NBF-OFF vs NBF-ON difference under "
            "the frozen local substitute-model configuration."
        ),
        **git_state(),
        "config_hash": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16],
        "dtype": hw.get("dtype", "bfloat16"),
        "quantization": hw.get("quantization", "none"),
        "device": hw.get("device", "cuda"),
        "generation_config": {
            "temperature_attacker": cfg["models"]["attacker"].get("temperature", 0.7),
            "temperature_target": cfg["models"]["target"].get("temperature", 0.7),
            "temperature_evaluator": cfg["models"]["evaluator"].get("temperature", 0.0),
            "max_new_tokens_attacker": cfg["models"]["attacker"]["max_new_tokens"],
            "max_new_tokens_target": cfg["models"]["target"]["max_new_tokens"],
            "max_new_tokens_evaluator": cfg["models"]["evaluator"]["max_new_tokens"],
            "use_cache": True,
        },
        "max_rounds": cfg["attacks"].get("max_turns", 8),
        "threshold": cfg["nbf"].get("threshold", 0.0),
        "base_seed": BASE_SEED,
    }


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


async def run_one_phase14(
    attack,
    goal_record: dict,
    goal_id: int,
    attack_key: str,
    condition: str,
    target_llm,
    barrier,
    embed_fn,
    cfg: dict,
    eta: float,
    max_turns: int,
    seed: int,
    metadata: dict,
) -> dict:
    from guardbound.attacks.runner import run_attack_with_backtracking_async

    goal = goal_record["task"]
    system_prompt = goal_record.get("target_system") or None

    if hasattr(attack, "prepare_attack"):
        attack.prepare_attack(goal)

    use_barrier = barrier is not None and embed_fn is not None
    logger.info("run_one_phase14: calling run_attack_with_backtracking_async (max_turns=%d, barrier=%s)", max_turns, barrier is not None)

    t0 = time.time()
    conv = await run_attack_with_backtracking_async(
        attack=attack,
        goal=goal,
        target_llm=target_llm,
        embed_fn=embed_fn if use_barrier else None,
        barrier=barrier if use_barrier else None,
        eta=eta,
        max_turns=max_turns,
        temperature=cfg["target"].get("temperature", 0.7),
        target_llm_name=metadata["target_model"],
        attack_method=attack_key,
        allow_regeneration=False,
        system_prompt=system_prompt,
        steer_target=False,
    )
    logger.info("run_one_phase14: attack completed in %.1fs, turns=%d", time.time() - t0, len(conv.turns))

    attack_scores_all = list(getattr(attack, "_scores", []) or [])
    accepted_scores = [s for s in attack_scores_all if s != "refused"]
    n_refused_records = len(attack_scores_all) - len(accepted_scores)
    refusal_count = attack.get_refusal_count() if hasattr(attack, "get_refusal_count") else 0

    candidate_scores = list(getattr(attack, "nbf_candidate_scores", []) or [])
    candidate_verdicts = list(getattr(attack, "nbf_candidate_verdicts", []) or [])
    nbf_scores = [round(float(s), 4) for s in candidate_scores]
    filtered = sum(1 for v in candidate_verdicts if v is False)

    turns = []
    rubric_scores = []
    final_score = None
    for i, t in enumerate(conv.turns):
        turns.append({"query": t.query, "response": t.response})
        score = accepted_scores[i] if i < len(accepted_scores) else None
        if score is not None:
            rubric_scores.append(score)
            final_score = score

    success = bool(attack.was_successful()) if hasattr(attack, "was_successful") else False

    if success:
        termination_reason = "success_score_5"
    elif hasattr(attack, "should_backtrack") and not attack.should_backtrack():
        termination_reason = "refusal_retry_limit"
    elif len(conv.turns) >= max_turns:
        termination_reason = "max_turns_reached"
    else:
        termination_reason = "attack_loop_exhausted"

    # NBF statistics (required per-run telemetry) — shared implementation.
    nbf_stats = nbf_stats_from(nbf_scores, filtered)

    attack_short = ATTACK_KEYS[attack_key]
    run_id = make_run_id(condition, attack_short, goal_id)

    # Canonical record: every telemetry field present. The completed-run path
    # fills the conversation/rubric/NBF fields here; run_batch then adds runtime,
    # resource and per-role call telemetry.
    return build_run_record(
        run_id=run_id,
        goal_id=goal_id,
        goal=goal,
        goal_text=goal,
        condition=condition,
        attack=attack_key,
        seed=seed,
        attacker_model=metadata["attacker_model"],
        target_model=metadata["target_model"],
        evaluator_model=metadata.get("evaluator_model"),
        embedding_model=metadata["embedding_model"],
        checkpoint_sha256=metadata.get("checkpoint_sha256"),
        dataset_sha256=metadata.get("dataset_sha256"),
        config_hash=metadata.get("config_hash"),
        code_commit=metadata.get("code_commit"),
        worktree_sha256=metadata.get("worktree_sha256"),
        dtype=metadata.get("dtype"),
        quantization=metadata.get("quantization"),
        max_rounds=max_turns,
        threshold=eta,
        filter_trials=(metadata.get("filter_trials") or {}).get(attack_short),
        top_p=metadata.get("top_p"),
        generation_config=metadata.get("generation_config"),
        target_system=system_prompt,
        turns=turns,
        success=success,
        final_score=final_score,
        rubric_scores=rubric_scores,
        refusal_count=refusal_count,
        refused_records=n_refused_records,
        num_turns=len(conv.turns),
        termination_reason=termination_reason,
        nbf_enabled=use_barrier,
        nbf_scores=nbf_scores,
        filtered_queries=filtered,
        filter_count=filtered,
        accepted_query_count=len(conv.turns),
        nbf_stats=nbf_stats,
    )


async def run_batch(args, cfg):
    batch = args.batch
    start_idx = BATCH_START[str(batch)]
    goals = load_goals(batch, args.limit)

    attacks = ATTACK_ORDER if args.attack == "all" else [args.attack]
    conditions = ["on", "off"] if args.condition == "both" else [args.condition]

    total_runs = len(goals) * len(attacks) * len(conditions)
    out_path = Path("results/phase14") / f"batch{batch:02d}.jsonl"
    done_ids = existing_run_ids(out_path) if not args.no_resume else set()

    print(f"[phase14] batch={batch} goals={len(goals)} attacks={len(attacks)} "
          f"conditions={len(conditions)} total_runs={total_runs}", flush=True)
    print(f"[phase14] attacks={attacks} conditions={conditions}", flush=True)
    print(f"[phase14] output={out_path} completed={len(done_ids)}/{total_runs}", flush=True)

    # Dataset + checkpoint identity (verified for the dry run too — read-only).
    dataset_sha = sha256_of(OFFICIAL_DATASET)
    ckpt_sha = sha256_of(Path(cfg["nbf"]["checkpoint"]))
    expected_ckpt_sha = cfg["nbf"].get("checkpoint_sha256_expected")
    if ckpt_sha != expected_ckpt_sha:
        raise RuntimeError(
            f"NBF checkpoint SHA256 mismatch!\n  expected: {expected_ckpt_sha}\n  actual:   {ckpt_sha}"
        )
    if dataset_sha != OFFICIAL_DATASET_SHA256:
        raise RuntimeError(
            f"Dataset SHA256 mismatch!\n  expected: {OFFICIAL_DATASET_SHA256}\n  actual:   {dataset_sha}"
        )
    with open(OFFICIAL_DATASET, "r", encoding="utf-8") as f:
        full_dataset_size = len(json.load(f))
    if full_dataset_size != OFFICIAL_DATASET_FULL_SIZE:
        raise RuntimeError(
            f"Dataset size mismatch: expected {OFFICIAL_DATASET_FULL_SIZE} goals, "
            f"found {full_dataset_size}"
        )
    print(f"[phase14] dataset={OFFICIAL_DATASET}", flush=True)
    print(f"[phase14] dataset sha256={dataset_sha} (verified, {full_dataset_size} goals)", flush=True)
    print(f"[phase14] checkpoint sha256={ckpt_sha} (verified)", flush=True)
    excluded_present = [a for a in EXCLUDED_ATTACKS if a in attacks]
    if excluded_present:
        raise RuntimeError(f"Excluded attacks present in the plan: {excluded_present}")

    if args.dry_run:
        print("[phase14] DRY RUN — no models loaded, no execution")
        print("[phase14] FULL STUDY PLAN:", flush=True)
        print(
            f"[phase14]   dataset goals        = {full_dataset_size}",
            flush=True,
        )
        print(f"[phase14]   attacks             = {ATTACK_ORDER}", flush=True)
        print("[phase14]   conditions          = ['on', 'off']", flush=True)
        print(
            f"[phase14]   runs per goal       = {len(ATTACK_ORDER)} x 2 = "
            f"{len(ATTACK_ORDER) * 2}",
            flush=True,
        )
        print(
            f"[phase14]   TOTAL PLANNED RUNS  = {full_dataset_size} x "
            f"{len(ATTACK_ORDER)} x 2 = {full_dataset_size * len(ATTACK_ORDER) * 2}",
            flush=True,
        )
        print(
            f"[phase14]   batches             = {NUM_BATCHES} x "
            f"{BATCH_SIZE} goals = {BATCH_SIZE * len(ATTACK_ORDER) * 2} runs each",
            flush=True,
        )
        print(
            f"[phase14]   excluded            = "
            f"{ {k: v for k, v in EXCLUDED_ATTACKS.items()} }",
            flush=True,
        )
        for gi, _gr in enumerate(goals):
            goal_id = start_idx + gi
            for ak in attacks:
                for cond in conditions:
                    rid = make_run_id(cond, ATTACK_KEYS[ak], goal_id)
                    tag = "DONE" if rid in done_ids else "RUN"
                    print(f"  {rid} [{tag}]")
        return

    metadata = build_metadata(cfg, ckpt_sha, dataset_sha)
    eta = cfg["nbf"].get("threshold", 0.0)
    max_turns = cfg["attacks"].get("max_turns", 8)

    print(f"[phase14] attacker={metadata['attacker_model']} "
          f"target={metadata['target_model']} "
          f"evaluator={metadata['evaluator_model']}", flush=True)
    print(f"[phase14] eta={eta} max_turns={max_turns}", flush=True)
    print(f"[phase14] structured_output_mode={args.structured_output_mode}", flush=True)

    # Phase 14.2: the study requires three DISTINCT physical models, with the
    # attacker pinned in slot 1 and target/evaluator rotating through slot 2.
    # Asserted BEFORE any weights load, so a mis-configured stack fails
    # immediately instead of silently deduplicating attacker and evaluator.
    _stack_ids = [
        metadata["attacker_model"],
        metadata["target_model"],
        metadata["evaluator_model"],
    ]
    if len(set(_stack_ids)) != 3:
        raise ModelIdentityError(
            f"three distinct model ids required, got {_stack_ids}"
        )
    print("[phase14] residency = pinned-slot rotation "
          "(attacker pinned; target <-> evaluator alternate in slot 2)", flush=True)

    attacker_llm, target_llm, evaluator_llm, model_manager = make_models(
        cfg, structured_output_mode=args.structured_output_mode
    )
    if model_manager is None:
        raise ResidencyInvariantError(
            "pinned-slot rotation requires a ModelManager; make_models returned None"
        )
    print(f"[phase14] distinct model ids  : {sorted(set(_stack_ids))}", flush=True)
    print(f"[phase14] pinned roles        : {model_manager.pinned_roles}", flush=True)
    print(f"[phase14] VRAM gates          : pre-load "
          f"{model_manager.min_free_before_load_gb}GB, post-load "
          f"{model_manager.min_free_after_load_gb}GB, per-run "
          f"{model_manager.min_free_per_run_gb}GB, ceiling "
          f"{model_manager.max_reserved_gb}GB reserved", flush=True)

    # Step 5 verification: the decoding mode requested on the CLI must be the
    # mode actually in effect on each local backend. `ManagedLocalChatLLM`
    # previously dropped this argument, which silently disabled Phase 14.1
    # constrained JSON on every managed role.
    _effective_modes = {
        role: getattr(llm, "structured_output_mode", None)
        for role, llm in (("attacker", attacker_llm),
                          ("target", target_llm),
                          ("evaluator", evaluator_llm))
    }
    print(f"[phase14] effective decoding modes: {_effective_modes}", flush=True)
    if args.structured_output_mode is not None:
        _missing = [r for r, m in _effective_modes.items()
                    if m != args.structured_output_mode]
        if _missing:
            raise StructuredJSONError(
                f"structured_output_mode={args.structured_output_mode!r} did not "
                f"reach: {_missing}"
            )

    # Load NBF barrier for ON condition.
    barrier, embed_fn = make_barrier_and_embed(cfg) if "on" in conditions else (None, None)
    if barrier is not None:
        print(f"[phase14] NBF barrier loaded on CUDA (eta={eta})", flush=True)

    completed = 0
    total_start = time.time()

    for gi, goal_record in enumerate(goals):
        goal_id = start_idx + gi
        for attack_key in attacks:
            for cond in conditions:
                attack_short = ATTACK_KEYS[attack_key]
                run_id = make_run_id(cond, attack_short, goal_id)
                if run_id in done_ids:
                    print(f"[phase14] skip (done): {run_id}", flush=True)
                    continue

                seed = derive_seed(goal_id, attack_short, cond)
                _seed_all(seed)

                # Phase 14.2 per-run safety. The run id is attached first so
                # every subsequent residency event is attributable, then the
                # free-VRAM floor is enforced, then both peak windows are opened
                # before this run allocates anything.
                model_manager.set_run_context(run_id)
                model_manager.pre_run_check(run_id)
                model_manager.reset_vram_peak()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                # Fresh attack instance per run (clean state).
                from guardbound.attacks.registry import get_attack
                attack = get_attack(attack_key)
                # Refusal/backtrack budget: the author's run.py uses the literal
                # ``C_refused < 10`` for crescendo/opposite_day/acronym (only
                # ActorAttack uses N_retry = 3). ``nbf.trials`` in the config
                # describes the *candidate-filter* trial counts, which the runner
                # supplies per attack, so it must NOT be pushed into
                # ``_max_refusal_retries`` (doing so lowered the backtrack budget
                # to 3 and changed the attack). Set it explicitly for the record.
                if hasattr(attack, "_max_refusal_retries"):
                    attack._max_refusal_retries = OFFICIAL_MAX_REFUSAL_RETRIES
                if hasattr(attack, "set_attacker_llm"):
                    attack.set_attacker_llm(attacker_llm)
                if hasattr(attack, "set_evaluator_llm"):
                    attack.set_evaluator_llm(evaluator_llm)

                for _llm in (attacker_llm, target_llm, evaluator_llm):
                    _llm.reset_counter()

                start = time.time()
                logger.info("run %s starting (goal_id=%d, attack=%s, condition=%s)", run_id, goal_id, attack_key, cond)
                try:
                    record = await run_one_phase14(
                        attack=attack,
                        goal_record=goal_record,
                        goal_id=goal_id,
                        attack_key=attack_key,
                        condition=cond,
                        target_llm=target_llm,
                        barrier=barrier if cond == "on" else None,
                        embed_fn=embed_fn if cond == "on" else None,
                        cfg=cfg,
                        eta=eta,
                        max_turns=max_turns,
                        seed=seed,
                        metadata=metadata,
                    )
                except Exception as e:
                    logger.exception("Run failed: %s", run_id)
                    failure_class, error_type, error_message = classify_exception(e)
                    # Built through the canonical skeleton (Phase 14.8), so a
                    # failed run carries the same field set as a completed one
                    # while staying unmistakably marked as a failure. See
                    # build_failure_record for which fields are intentionally
                    # None and why.
                    record = build_failure_record(
                        run_id=run_id,
                        goal_id=goal_id,
                        goal_record=goal_record,
                        condition=cond,
                        attack_key=attack_key,
                        attack_short=attack_short,
                        seed=seed,
                        attack=attack,
                        metadata=metadata,
                        max_turns=max_turns,
                        eta=eta,
                        started_at=start,
                        failure_class=failure_class,
                        error_type=error_type,
                        error_message=error_message,
                        error_text=str(e),
                        attacker_llm=attacker_llm,
                        target_llm=target_llm,
                        evaluator_llm=evaluator_llm,
                        model_manager=model_manager,
                    )
                    append_result(out_path, record)
                    done_ids.add(run_id)
                    completed += 1
                    print(f"[phase14] {run_id} ERROR ({e}) ({record['runtime']:.1f}s)", flush=True)
                    continue

                record["runtime"] = round(time.time() - start, 3)
                record["runtime_seconds"] = record["runtime"]
                # Explicit outcome classification (never conflated with an
                # infrastructure/model failure — those are recorded above).
                record["failure_class"] = (
                    "SUCCESS" if record.get("success") else "ATTACK_FAILURE"
                )
                # Structural symmetry with the failure path: a completed run
                # carries `error` explicitly as None rather than omitting it.
                record["error"] = None
                record["error_type"] = None
                record["error_message"] = None
                if torch.cuda.is_available():
                    record["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
                record["residency"] = residency_record(model_manager, run_id)
                record["system"] = {**sample_gpu_state(), **sample_host_memory()}
                record["final_response"] = (
                    record["turns"][-1]["response"] if record["turns"] else None
                )
                record["llm_calls"] = {
                    "attacker": getattr(attacker_llm, "calls", None),
                    "target": getattr(target_llm, "calls", None),
                    "evaluator": getattr(evaluator_llm, "calls", None),
                }
                if isinstance(attacker_llm, CountingChatLLM):
                    record["attacker_purpose_calls"] = dict(attacker_llm.purpose_counts)
                    record["attacker_parse_failures"] = attacker_llm.parse_failure_summary()
                    # Structurally present on every record (Phase 14.8), using the
                    # same helper as the failure path. Previously this field was
                    # emitted only for zero-turn runs, which made the completed-run
                    # record's own key set non-constant — so a schema-equality test
                    # would have been flaky depending on which success it sampled.
                    # Captured outputs are never replaced by [] here.
                    record["attacker_raw_outputs"] = raw_attacker_outputs(attacker_llm)
                if isinstance(evaluator_llm, CountingChatLLM):
                    record["evaluator_purpose_calls"] = dict(evaluator_llm.purpose_counts)
                    # D-E1: parse failures are counted so a failed parse is
                    # distinguishable from a genuine evaluator score of 1.
                    record["evaluator_parse_failures"] = (
                        evaluator_llm.parse_failure_summary()
                    )
                if isinstance(target_llm, CountingChatLLM):
                    record["target_purpose_calls"] = dict(target_llm.purpose_counts)

                # Invariant checks. A breach is recorded against the run with its
                # own failure class; it does NOT abort the batch. These are
                # per-run data-integrity signals, not physical hazards — aborting
                # here would discard every remaining run of the matrix over one
                # model-side failure. (The original code used bare ``assert``,
                # which aborted the whole batch.) A VRAM guard failure still
                # aborts, because that one is a hardware-safety condition.
                invariant_error = None
                if cond == "off":
                    if record["nbf_scores"] != []:
                        invariant_error = (
                            f"NBF OFF run {run_id} has non-empty nbf_scores"
                        )
                    elif record["filtered_queries"] != 0:
                        invariant_error = (
                            f"NBF OFF run {run_id} has filtered_queries != 0"
                        )
                elif (record["nbf_enabled"] and not record["nbf_scores"]
                        and record.get("num_turns")):
                    # Only meaningful when the attack actually ran: a run that
                    # never produced a query legitimately scored no candidates,
                    # which is an attack failure rather than an NBF breach.
                    invariant_error = (
                        f"NBF ON run {run_id} missing NBF candidate scores"
                    )
                if invariant_error:
                    record["invariant_breach"] = invariant_error
                    record["failure_class"] = "NBF_SEMANTICS_FAILURE"
                    record["error_type"] = "NBFSemanticsError"
                    record["error_message"] = invariant_error
                    print(f"[phase14]   INVARIANT BREACH: {invariant_error}", flush=True)

                append_result(out_path, record)
                done_ids.add(run_id)
                completed += 1

                status = "SUCCESS" if record["success"] else "no"
                print(
                    f"[phase14] {run_id} turns={record.get('num_turns', 0)} "
                    f"success={status} filtered={record.get('filtered_queries', 0)} "
                    f"nbf={len(record.get('nbf_scores', []))} "
                    f"peak_vram={record.get('peak_vram_gb')} "
                     f"({record.get('runtime', 0):.1f}s)",
                     flush=True,
                 )
                residency = record.get("residency") or {}
                if residency:
                    print(
                        f"[phase14]   residency: resident={residency['snapshot']['resident_model_ids']} "
                        f"loads={residency['loads']} "
                        f"evictions={residency['evictions']} "
                        f"load_s={residency['load_seconds']}",
                        flush=True,
                    )
                logger.info("run %s completed in %.1fs (success=%s, turns=%d, calls=%s)",
                            run_id, record.get("runtime", 0),
                            record.get("success"), record.get("num_turns"),
                            record.get("llm_calls", {}))

                # Hard VRAM ceiling: the record is already persisted, so abort
                # the batch rather than continue toward the physical limit.
                model_manager.ceiling_check(run_id)

    print(f"\n[phase14] batch {batch} done. {completed} new runs "
          f"in {time.time() - total_start:.1f}s", flush=True)
    print(f"[phase14] output: {out_path}", flush=True)
    print(f"[phase14] completed this batch: {completed}/{total_runs}", flush=True)
    print(f"[phase14] total completed: {len(done_ids)}/{total_runs}", flush=True)

    # Release every model, pins included, so the next batch starts from a clean
    # card rather than inheriting this batch's residency.
    if model_manager is not None:
        model_manager.set_run_context(None)
        model_manager.unload_all()
        print(f"[phase14] residency released: "
              f"{model_manager.residency_snapshot()}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    cfg = load_config(args.config)
    asyncio.run(run_batch(args, cfg))


if __name__ == "__main__":
    main()
