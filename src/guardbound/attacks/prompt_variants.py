"""Prompt-byte variants: paper-reproduction (official) vs historical (legacy).

Phase 21 measured three typographic deviations between Guardbound's prompts and
the author's release: the Opposite-day and Acronym attacker system prompts, and
the rubric prompt. The approved Phase 22 decision is to run the **paper
reproduction** on the official bytes while keeping the historical bytes available
so frozen local experiments remain reproducible.

Two variants therefore exist for every prompt that differs:

``legacy``
    The bytes Guardbound has always sent. Frozen Phase 14-19 local experiments
    were produced with these; they are the default so nothing changes silently.

``official``
    The bytes from ``nbf_original_stuff/.../NBF-LLM/``, stored verbatim in
    :mod:`guardbound.attacks._official_prompts` (generated, not hand-typed).

Prompts that are already byte-identical (Crescendo attacker, refusal, disclaimer)
have no variant: there is exactly one text.
"""
from __future__ import annotations

import hashlib
from typing import Any

from ._official_prompts import (
    ACRONYM_SYSTEM_PROMPT_OFFICIAL,
    OPPOSITE_DAY_SYSTEM_PROMPT_OFFICIAL,
    RUBRIC_PROMPT_OFFICIAL,
    RUBRIC_SENTINEL_QUERY,
    RUBRIC_SENTINEL_RESPONSE,
)

PROMPT_VARIANT_LEGACY = "legacy"
PROMPT_VARIANT_OFFICIAL = "official"
PROMPT_VARIANTS = (PROMPT_VARIANT_LEGACY, PROMPT_VARIANT_OFFICIAL)


def resolve_prompt_variant(name: str | None) -> str:
    """Canonicalise a prompt variant; unknown names fail loudly."""
    variant = (name or PROMPT_VARIANT_LEGACY).strip().lower()
    if variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"unknown prompt variant {name!r}; supported: {list(PROMPT_VARIANTS)}"
        )
    return variant


# --------------------------------------------------------------------------- #
# Attacker system prompts
# --------------------------------------------------------------------------- #

def opposite_day_system_prompt(variant: str | None = None) -> str:
    """Opposite-day attacker system prompt for ``variant``."""
    if resolve_prompt_variant(variant) == PROMPT_VARIANT_OFFICIAL:
        return OPPOSITE_DAY_SYSTEM_PROMPT_OFFICIAL
    from .opposite_day import OPPOSITE_DAY_SYSTEM_PROMPT

    return OPPOSITE_DAY_SYSTEM_PROMPT


def acronym_system_prompt(variant: str | None = None) -> str:
    """Acronym attacker system prompt for ``variant``."""
    if resolve_prompt_variant(variant) == PROMPT_VARIANT_OFFICIAL:
        return ACRONYM_SYSTEM_PROMPT_OFFICIAL
    from .acronym import ACRONYM_SYSTEM_PROMPT

    return ACRONYM_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# Rubric evaluator prompt
# --------------------------------------------------------------------------- #

def rubric_prompt(query: str, response: str, variant: str | None = None) -> str:
    """The rubric evaluator prompt for ``variant``.

    ``legacy`` is left to the call sites (they build it inline, unchanged); this
    function exists for the official bytes and for hashing/auditing both.
    """
    if resolve_prompt_variant(variant) != PROMPT_VARIANT_OFFICIAL:
        raise ValueError(
            "rubric_prompt() produces the official variant only; the legacy text "
            "is built inline by each evaluator for byte-stability."
        )
    return (
        RUBRIC_PROMPT_OFFICIAL
        .replace(RUBRIC_SENTINEL_RESPONSE, response)
        .replace(RUBRIC_SENTINEL_QUERY, query)
    )


# --------------------------------------------------------------------------- #
# Fingerprints
# --------------------------------------------------------------------------- #

#: Fixture used for the rubric fingerprint (the prompt embeds the query and the
#: response, so a fixed pair is needed for a stable hash).
RUBRIC_FIXTURE_QUERY = "Q"
RUBRIC_FIXTURE_RESPONSE = "R"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def system_prompt_templates(variant: str | None = None) -> dict[str, str]:
    """The three attacker system-prompt templates for ``variant``.

    Single resolution point: manifests, audits and tests all read this, so a
    recorded hash cannot describe different bytes from the ones sent.
    """
    from .crescendo_paper import CRESCENDO_SYSTEM_PROMPT

    return {
        "crescendo": CRESCENDO_SYSTEM_PROMPT,
        "opposite_day": opposite_day_system_prompt(variant),
        "acronym": acronym_system_prompt(variant),
    }


def rubric_template(variant: str | None = None) -> str:
    """The rubric prompt for ``variant``, rendered for the fixed fixture."""
    if resolve_prompt_variant(variant) == PROMPT_VARIANT_OFFICIAL:
        return rubric_prompt(
            RUBRIC_FIXTURE_QUERY, RUBRIC_FIXTURE_RESPONSE, PROMPT_VARIANT_OFFICIAL
        )
    return _legacy_rubric_text()


def fingerprint(text: str) -> dict[str, Any]:
    """``{sha256, chars}`` for one prompt text."""
    return {"sha256": _sha256(text), "chars": len(text)}


def prompt_hashes() -> dict[str, Any]:
    """SHA-256 of every prompt text, per variant, for manifests and audits."""
    legacy = system_prompt_templates(PROMPT_VARIANT_LEGACY)
    official = system_prompt_templates(PROMPT_VARIANT_OFFICIAL)
    return {
        "crescendo_attacker": {
            "variants": {
                PROMPT_VARIANT_LEGACY: _sha256(legacy["crescendo"]),
                PROMPT_VARIANT_OFFICIAL: _sha256(official["crescendo"]),
            },
            "identical_across_variants": legacy["crescendo"] == official["crescendo"],
        },
        "opposite_day_attacker": {
            "variants": {
                PROMPT_VARIANT_LEGACY: _sha256(legacy["opposite_day"]),
                PROMPT_VARIANT_OFFICIAL: _sha256(official["opposite_day"]),
            },
            "identical_across_variants":
                legacy["opposite_day"] == official["opposite_day"],
        },
        "acronym_attacker": {
            "variants": {
                PROMPT_VARIANT_LEGACY: _sha256(legacy["acronym"]),
                PROMPT_VARIANT_OFFICIAL: _sha256(official["acronym"]),
            },
            "identical_across_variants": legacy["acronym"] == official["acronym"],
        },
        "rubric": {
            # Both variants are rendered for the SAME fixture so the hashes are
            # directly comparable.
            "variants": {
                PROMPT_VARIANT_LEGACY: _sha256(rubric_template(PROMPT_VARIANT_LEGACY)),
                PROMPT_VARIANT_OFFICIAL: _sha256(rubric_template(PROMPT_VARIANT_OFFICIAL)),
            },
            "identical_across_variants": False,
        },
    }


def _legacy_rubric_text() -> str:
    """The legacy rubric prompt for the fixed fixture, built by the real helper.

    Captured rather than re-typed so the hash reflects what is actually sent.
    """
    from .crescendo_paper import _build_rubric_prompt  # noqa: PLC0415

    return _build_rubric_prompt("Q", "R")


# --------------------------------------------------------------------------- #
# ActorAttack prompt provenance (Phase 30)
# --------------------------------------------------------------------------- #

def _placeholders(text: str) -> list[str]:
    """Field names ``text.format(...)`` consumes, in first-appearance order."""
    import string

    names: list[str] = []
    for _, field, _, _ in string.Formatter().parse(text):
        if field and field not in names:
            names.append(field)
    return names


def actor_attack_prompt_provenance(variant: str | None = None) -> dict[str, Any]:
    """Identity and SHA-256 of every ActorAttack prompt byte string.

    ActorAttack has its own prompt constants (``EXTRACT_PROMPT`` … ``SUMMARY_*``);
    the rubric is the *same* rendered text Crescendo and Opposite-day use. The
    per-actor query chain is generated by the attacker model at run time, so its
    content varies per goal and only its template is hashable — both facts are
    recorded here rather than implied.

    ``ACTOR_PROMPT`` is the only prompt whose bytes depend on a value known
    offline (``num_string``, ``"3 actors"`` for the paper's actor count), so its
    canonical rendering is hashed as well as its template.
    """
    from . import actor_attack as aa

    resolved = resolve_prompt_variant(variant)

    def entry(text: str, kind: str = "template") -> dict[str, Any]:
        return {
            "kind": kind,
            "source": "src/guardbound/attacks/actor_attack.py",
            "placeholders": _placeholders(text),
            **fingerprint(text),
        }

    rendered_actor = aa.ACTOR_PROMPT.format(num_string="3 actors")
    # ``rubric_template`` resolves both variants (official -> the author's bytes,
    # legacy -> the historical text); ``rubric_prompt`` only supports official.
    rubric = rubric_template(resolved)
    return {
        "source": "src/guardbound/attacks/actor_attack.py",
        "variant": resolved,
        "prompts": {
            "extract": entry(aa.EXTRACT_PROMPT),
            "network": entry(aa.NETWORK_PROMPT),
            "actor": {
                **entry(aa.ACTOR_PROMPT),
                "rendered_canonical": {
                    "actor_num": 3,
                    "value": "3 actors",
                    **fingerprint(rendered_actor),
                },
            },
            "more_actor": entry(aa.MORE_ACTOR_PROMPT, kind="constant"),
            "queries": entry(aa.QUERIES_PROMPT),
            "json_format_question": entry(aa.JSON_FORMAT_QUESTION_PROMPT),
            "attack_modify": entry(aa.ATTACK_MODIFY_PROMPT),
            "step_judge": entry(aa.ATTACK_STEP_JUDGE_PROMPT),
            "summary_with_details": entry(aa.SUMMARY_WITH_DETAILS_PROMPT),
            "summary_without_details": entry(aa.SUMMARY_WITHOUT_DETAILS_PROMPT),
            "rubric": {
                "kind": "rendered_template",
                "source": "src/guardbound/attacks/prompt_variants.py::rubric_prompt",
                "reused_by": ["actor_attack", "crescendo_paper", "opposite_day"],
                **fingerprint(rubric),
            },
        },
        "dynamic": {
            "queries_chain": (
                "the per-actor query list is generated by the attacker model and "
                "rewritten on rejection; its bytes vary per goal"
            ),
            "summary_texts": (
                "SUMMARY_* templates are filled with the goal's delivery_type and "
                "other_details, so the rendered text varies per goal"
            ),
        },
        "reuse_notes": (
            "the rubric is the only shared prompt: ActorAttack delegates to the "
            "same function Crescendo/Opposite-day use; every other prompt is "
            "module-local to actor_attack.py"
        ),
        "identical_across_arms": True,
    }
