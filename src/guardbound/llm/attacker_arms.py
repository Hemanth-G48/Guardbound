"""Attacker-prompt conditions ("arms") for the cloud experiments.

Two conditions exist, and they must never be mixed:

``baseline``
    The official paper attacker prompt, unmodified.

``c1_generation_point``
    The same official prompt **plus** the frozen Phase-17 ``I1`` instruction
    appended as a *final user message* on every attacker generation call. This is
    an experimental modification of the attacker condition, not part of the paper.

The ``I1`` wording and placement are frozen from
``scripts/phase17_stage6_run_level_contract.py`` (``INSTRUCTION`` and
``ARMS["generation_point"]``, mode ``I1``, temperature 0.7).
``tests/test_attacker_arms.py`` imports that module and asserts equality, so this
copy cannot drift.

Why a call-boundary wrapper
---------------------------
The attack algorithms, their system prompts and the evaluator prompts must stay
untouched. Appending the instruction inside ``crescendo_paper.py`` would make the
modification invisible to a reader of the paper-faithful path and would apply to
every run. Wrapping the attacker ``ChatLLM`` instead keeps the deviation
(a) opt-in, (b) visible in the config and the manifest, and (c) impossible to
inherit accidentally.

Scope guard
-----------
``I1`` applies **only** to attacker generation calls. A refusal/disclaimer/rubric
call that happens to be routed through the attacker object (the official
single-model behaviour, where the attacker is also the judge) must not receive
it — appending a JSON-envelope instruction to a judge prompt would change the
evaluator methodology. :func:`is_attacker_generation_call` enforces that.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .base import ChatLLM, DEFAULT_TEMPERATURE, Message

# Frozen from scripts/phase17_stage6_run_level_contract.py::INSTRUCTION (mode I1).
I1_INSTRUCTION = ("Your entire response must be a single JSON object.\n"
                  "Do not include any text before or after it.")

ARM_BASELINE = "baseline"
ARM_C1 = "c1_generation_point"
ARMS = (ARM_BASELINE, ARM_C1)

# Every official attacker system prompt opens with this preamble. Used to decide
# whether a call is attacker generation (eligible for I1) or evaluation (not).
_ATTACKER_MARKER = "You are a proficient red teaming engineer"

PLACEMENT_FINAL_USER_MESSAGE = "appended as the final user message of each attacker call"


@dataclass(frozen=True)
class ArmSpec:
    name: str
    label: str
    instruction: str | None
    placement: str | None
    is_paper_methodology: bool
    deviation: str | None


ARM_SPECS: dict[str, ArmSpec] = {
    ARM_BASELINE: ArmSpec(
        name=ARM_BASELINE,
        label="Cloud baseline — original paper attacker prompt",
        instruction=None,
        placement=None,
        is_paper_methodology=True,
        deviation=None,
    ),
    ARM_C1: ArmSpec(
        name=ARM_C1,
        label="Cloud C1 — paper attacker prompt + I1 generation-point instruction",
        instruction=I1_INSTRUCTION,
        placement=PLACEMENT_FINAL_USER_MESSAGE,
        is_paper_methodology=False,
        deviation=(
            "Deviation: experimental attacker-prompt modification (Phase 17 arm "
            "C1 / mode I1). Not part of the paper; must be reported as a separate "
            "condition and never pooled with the baseline."
        ),
    ),
}


def resolve_arm(name: str | None) -> ArmSpec:
    key = (name or ARM_BASELINE).strip().lower()
    if key not in ARM_SPECS:
        raise ValueError(
            f"unknown attacker prompt condition {name!r}; supported: {list(ARMS)}"
        )
    return ARM_SPECS[key]


def is_attacker_generation_call(messages: list[Message]) -> bool:
    """True when the message list is an attacker generation request."""
    return any(
        message.get("role") == "system"
        and _ATTACKER_MARKER in str(message.get("content", ""))
        for message in messages
    )


class AttackerArmLLM(ChatLLM):
    """Apply one attacker-prompt arm at the call boundary.

    ``baseline`` is a strict pass-through: the caller's list is forwarded
    unchanged, so the paper condition cannot be perturbed by this wrapper.
    """

    def __init__(self, inner: ChatLLM, arm: str | ArmSpec) -> None:
        self._inner = inner
        self.spec = arm if isinstance(arm, ArmSpec) else resolve_arm(arm)
        self.name = f"arm[{self.spec.name}/{getattr(inner, 'name', 'llm')}]"

    @property
    def inner(self) -> ChatLLM:
        return self._inner

    def generate(
        self,
        messages: list[Message],
        temperature: float = DEFAULT_TEMPERATURE,
        max_turns_context: int | None = None,
        json_format: bool = False,
        structured_output_mode: str | None = None,
    ):
        if self.spec.instruction is None or not is_attacker_generation_call(messages):
            return self._inner.generate(
                messages, temperature=temperature,
                max_turns_context=max_turns_context, json_format=json_format,
                structured_output_mode=structured_output_mode,
            )
        # Copy so the caller's history is never mutated.
        augmented = [dict(message) for message in messages]
        augmented.append({"role": "user", "content": self.spec.instruction})
        return self._inner.generate(
            augmented, temperature=temperature,
            max_turns_context=max_turns_context, json_format=json_format,
            structured_output_mode=structured_output_mode,
        )

    def __repr__(self) -> str:
        return f"AttackerArmLLM(arm={self.spec.name!r}, inner={self._inner!r})"


def build_armed_attacker(llm: ChatLLM, arm: str | None) -> ChatLLM:
    """Wrap ``llm`` for ``arm``; the baseline arm returns the object unchanged."""
    spec = resolve_arm(arm)
    return llm if spec.name == ARM_BASELINE else AttackerArmLLM(llm, spec)


# --------------------------------------------------------------------------- #
# Manifest fingerprinting
# --------------------------------------------------------------------------- #

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def attacker_prompt_fingerprints(
    arm: str | None,
    prompt_variant: str | None = None,
) -> dict[str, Any]:
    """The exact attacker prompt text and its hash, for an experiment manifest.

    The per-attack *templates* are hashed (they contain the
    ``{conversationObjective}`` placeholder, so they are stable across goals).
    The conditional instruction is recorded verbatim with its own hash, and the
    placement is stated, so a reader can reconstruct the wire prompt exactly.

    ``prompt_variant`` selects the prompt bytes the run actually sends
    (``official`` for the paper reproduction). The legacy bytes are always
    recorded alongside so historical runs stay reconstructable, and the rubric
    prompt is hashed too (Phase 23 defect D-04: the block used to describe the
    legacy attacker prompts even when the run sent the official ones).
    """
    from ..attacks import prompt_variants as pv

    spec = resolve_arm(arm)
    variant = pv.resolve_prompt_variant(prompt_variant)
    templates = pv.system_prompt_templates(variant)
    legacy_templates = pv.system_prompt_templates(pv.PROMPT_VARIANT_LEGACY)

    return {
        "condition": spec.name,
        "label": spec.label,
        "is_paper_methodology": spec.is_paper_methodology,
        "deviation": spec.deviation,
        "placement": spec.placement,
        "instruction": spec.instruction,
        "instruction_sha256": _sha256(spec.instruction) if spec.instruction else None,
        "prompt_variant": variant,
        "system_prompt_templates": {
            attack: pv.fingerprint(text) for attack, text in templates.items()
        },
        "rubric_prompt": {
            **pv.fingerprint(pv.rubric_template(variant)),
            "fixture": {
                "query": pv.RUBRIC_FIXTURE_QUERY,
                "response": pv.RUBRIC_FIXTURE_RESPONSE,
            },
            "variant": variant,
        },
        # Kept so a manifest of an official-prompt run can still be compared with
        # the historical (legacy) bytes without a second run.
        "legacy_system_prompt_templates": {
            attack: pv.fingerprint(text) for attack, text in legacy_templates.items()
        },
        "evaluator_receives_instruction": False,
        "target_receives_instruction": False,
    }
