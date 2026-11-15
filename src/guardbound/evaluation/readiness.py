"""Utility-evaluation readiness inventory (Phase 29).

Read-only. Reports whether the datasets the paper's Table 1 needs are present and
schema-valid, so a missing dataset is a visible ``BLOCKED — DATA REQUIRED`` state
rather than a silent absence. Never downloads, fabricates or substitutes data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STATUS_READY = "READY"
STATUS_BLOCKED = "BLOCKED — DATA REQUIRED"

PAPER_VERSION_UNVERIFIED = "PAPER VERSION UNVERIFIED"

# --------------------------------------------------------------------------- #
# Upstream provenance (Phase 31) — metadata only, no dataset materialised
# --------------------------------------------------------------------------- #
# Every value below was read from published GitHub metadata (repository, refs,
# commit and blob endpoints) during Phase 31. A git blob SHA-1 is a content hash:
# it identifies the exact bytes without needing the file. No data file was
# downloaded, cloned, copied or written. ``sha256`` is therefore ``None`` by
# policy, not by oversight — it can only be computed from bytes we do not hold.
UPSTREAM_PROVENANCE: dict[str, dict[str, Any]] = {
    "mtbench": {
        "repository": "https://github.com/lm-sys/FastChat",
        "pinned_tag": "v0.2.36",
        "pinned_tag_commit": "b21d0f780ca4472a13714262a0790f2ee1ade659",
        "pinned_commit": "b494d0c6b4e7935f1764f8439e75da3e66beccc7",
        "pinned_commit_note": (
            "release commit of the MT-bench code (2023-06-16, 'Release MT-bench "
            "code (#1722)'); also the last commit touching question.jsonl and "
            "reference_answer/gpt-4.jsonl"
        ),
        "dataset_files": {
            "fastchat/llm_judge/data/mt_bench/question.jsonl": {
                "git_blob_sha1": "464e2c22f7a021a0f9a584d232338a9a17c40066",
                "size_bytes": 48929,
                "identical_at_tag": "v0.2.36",
            },
            "fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl": {
                "git_blob_sha1": "8b66387a3129cbb117178fe515491df553dd52df",
                "size_bytes": 51109,
                "identical_at_tag": "v0.2.36",
            },
        },
        "code_files": {
            "fastchat/llm_judge/gen_model_answer.py": {
                "git_blob_sha1": "5468dce0688933d3ccae6f3c632676d9905ff28d",
            },
            "fastchat/llm_judge/gen_judgment.py": {
                "git_blob_sha1": "3d952c8b09459f64397cb23f2f56e121f024b0a4",
            },
            "fastchat/llm_judge/common.py": {
                "git_blob_sha1": "fc223573cbf3ca2633dc022e50dca1e1d180a95f",
                "note": (
                    "last modified 2024-02-04 at "
                    "81785d7ed1d6afb966b464a8ee4689b7413e6313, i.e. later than the "
                    "dataset release; the judge code is *not* pinned by the dataset "
                    "commit"
                ),
            },
        },
        "hash_kind": "git blob SHA-1 (content-addressed; published by the GitHub API)",
        "sha256": None,
        "sha256_reason": "dataset not downloaded — computing it would materialise the data",
        "dataset_version_status": PAPER_VERSION_UNVERIFIED,
        "retrieval": "read-only metadata endpoints and upstream source code; no download",
        "upstream_defaults_not_paper_methodology": [
            "judge model: gpt-4 (gen_judgment.py default) — paper silent",
            "single-answer judge prompt: single-v1-multi-turn from judge_prompts.jsonl",
            "judge temperature 0, max_tokens 1024 on the single path",
            "per-category generation temperature (0/0.1/0.7) — conflicts with the "
            "paper's 'all temperatures 0.7'",
            "both turns come from question['turns']; turn 2 is not model-generated",
            "reference answers only for math/reasoning/coding, from "
            "reference_answer/gpt-4.jsonl",
            "all 80 questions (ids 81-160) are scored; no subset is defined upstream",
        ],
    },
    "xstest": {
        "repository": "https://github.com/paul-rottger/xstest",
        "license": "CC-BY-4.0",
        "pinned_tag": None,
        "pinned_tag_note": "the repository publishes no tags and no releases",
        "pinned_commit": "475f10bf0a3d6a9dfb174b6de1a38afbfdff98a5",
        "pinned_commit_note": (
            "last commit touching xstest_prompts.csv (2025-02-11, 'overdue cleanup'); "
            "with no tags this commit SHA is the only immutable pin"
        ),
        "dataset_files": {
            "xstest_prompts.csv": {
                "git_blob_sha1": "ad38204d70b3964387f9690169d11afbc4756d01",
                "size_bytes": 38719,
            },
        },
        "code_files": {
            "evaluation/classify_completions_strmatch.py": {
                "git_blob_sha1": "aa8fa43a2043c7d400923bd4ffb628154aca1b7c",
            },
            "evaluation/classify_completions_gpt.py": {
                "git_blob_sha1": "b9f911978a4ee5c91aa65239a59c2b979cd61576",
            },
            "readme.md": {
                "git_blob_sha1": "2850b16498d65625fdc1886f83efd029541fcfc4",
            },
        },
        "hash_kind": "git blob SHA-1 (content-addressed; published by the GitHub API)",
        "sha256": None,
        "sha256_reason": "dataset not downloaded — computing it would materialise the data",
        "dataset_version_status": PAPER_VERSION_UNVERIFIED,
        "retrieval": "read-only metadata endpoints and upstream source code; no download",
        "label_semantics": (
            "the README defines unsafe prompts as those whose `type` starts with "
            "'contrast_'; the CSV also carries an explicit `label` column "
            "(safe/unsafe). The paper reports 250 safe + 200 unsafe = 450 prompts."
        ),
        "upstream_detectors_not_paper_methodology": [
            "classify_completions_strmatch.py: lower-case prefix matching, two "
            "classes (1_full_compliance / 2_full_refusal), no partial-refusal class",
            "classify_completions_gpt.py: GPT-4, temperature 0, max_tokens 16, "
            "three classes including 3_partial_refusal",
            "neither detector is selected by the paper; partial refusals are counted "
            "as refusals by the GPT classifier and as compliance by strmatch",
            "the paper does not state whether the denominator is all 450 prompts or "
            "the 250 safe prompts",
        ],
    },
}


def upstream_provenance(name: str) -> dict[str, Any]:
    """Pinned upstream revision for one utility component (copy)."""
    import copy

    return copy.deepcopy(UPSTREAM_PROVENANCE[name])


# Schema each component's loader actually reads (see evaluation/mtbench.py and
# evaluation/over_refusal.py). The paper specifies neither dataset's file layout,
# so these are loader contracts, not paper requirements.
UTILITY_COMPONENTS: dict[str, dict[str, Any]] = {
    "mtbench": {
        "paper_reference": "Table 1 (MTBench, 1-10); B.1 refusal substitution",
        "default_path": "data/raw/mtbench.jsonl",
        "required_fields": ("question",),
        "optional_fields": ("reference", "turn2_question", "category"),
        "notes": (
            "the turn-2 question is model-generated when absent; the paper does "
            "not state how it is sourced, nor which judge model scores it"
        ),
    },
    "xstest": {
        "paper_reference": "Table 1 (XSTest over-refusal); Sec. 5.1",
        "default_path": "data/raw/xstest.jsonl",
        "required_any_of": (("prompt", "text"),),
        "optional_fields": ("type",),
        "notes": (
            "the paper reports 'refused queries over all' but does not specify "
            "the refusal detector; this project uses a documented regex heuristic"
        ),
    },
}


def _resolve(root: str | Path | None, default: str) -> Path:
    path = Path(default)
    if path.is_absolute() or root is None:
        return path
    return Path(root) / path


def check_utility_dataset(
    name: str, path: str | Path | None = None, root: str | Path | None = None
) -> dict[str, Any]:
    """Presence + schema check for one utility dataset."""
    spec = UTILITY_COMPONENTS[name]
    resolved = Path(path) if path else _resolve(root, spec["default_path"])
    report: dict[str, Any] = {
        "component": name,
        "path": str(resolved),
        "exists": resolved.exists(),
        "paper_reference": spec["paper_reference"],
        "notes": spec["notes"],
    }
    if not resolved.exists():
        report.update(status=STATUS_BLOCKED, rows=0, missing_fields=[])
        return report

    rows = 0
    invalid_rows = 0
    missing: set[str] = set()
    with open(resolved, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue
            for field in spec.get("required_fields", ()):
                if not row.get(field):
                    missing.add(field)
            for group in spec.get("required_any_of", ()):
                if not any(row.get(key) for key in group):
                    missing.add("/".join(group))

    report.update(rows=rows, invalid_rows=invalid_rows,
                  missing_fields=sorted(missing))
    report["status"] = (STATUS_READY if rows and not missing
                        else STATUS_BLOCKED)
    return report


def utility_readiness(
    paths: dict[str, str] | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Inventory for every utility component the GPT-4o row needs."""
    paths = paths or {}
    components = {
        name: check_utility_dataset(name, paths.get(name), root)
        for name in UTILITY_COMPONENTS
    }
    ready = [name for name, item in components.items()
             if item["status"] == STATUS_READY]
    blocked = [name for name, item in components.items()
               if item["status"] != STATUS_READY]
    return {
        "components": components,
        "ready": ready,
        "blocked": blocked,
        "table1_complete": not blocked,
        "status": STATUS_READY if not blocked else STATUS_BLOCKED,
    }
