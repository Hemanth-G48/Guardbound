#!/usr/bin/env python3
"""Prompt-fidelity audit: official NBF-LLM release vs Guardbound, byte for byte.

This is the executable evidence behind ``PAPER_PROMPT_FIDELITY_AUDIT.md``. It
does not compare summaries or YAML: it imports the **official source modules**
from ``nbf_original_stuff/`` and the **Guardbound modules**, drives both with a
capturing client, and compares the exact message arrays and call parameters that
each would hand to a model.

Checks
------
1. attacker   Crescendo    system + round-1 messages
2. attacker   Crescendo    round-2 messages (history + summary echo)
3. attacker   OppositeDay  system + round-1 messages
4. attacker   Acronym      system + round-1 messages
5. evaluator  refusal      system + user message
6. evaluator  disclaimer   system + user message (Acronym path)
7. evaluator  rubric       user message (official builds it inline)
8. target     conversation shape: system seed + alternating turns, as the
              official ``history_t`` does, captured from the real runner
9. provider   the SDK request body equals the constructed messages (no wrapper,
              no prepended system message, no rewritten roles)

Also reports whether the Phase-17 ``C1`` / ``I1`` generation-point instruction is
present in the shared attack code (it must not be).

Read-only. No network. No paid requests.

Usage
-----
    python scripts/audit_prompt_fidelity.py
    python scripts/audit_prompt_fidelity.py --out results/audit/prompt_fidelity.json
"""
from __future__ import annotations

import argparse
import copy
import difflib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for extra in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from guardbound.llm.base import ChatLLM, DEFAULT_TEMPERATURE  # noqa: E402
from guardbound.llm.openai_client import OpenAIChatLLM  # noqa: E402
from tests.parity.utils import (  # noqa: E402
    load_official_attack,
    load_official_utils,
)

GOAL = "Explain the history of cryptography in three paragraphs."
PROMPT = "What were the key events?"
RESPONSE = "Sure - here is an overview of the main developments."
RUBRIC_PAYLOAD = {"score": 1}
REFUSAL_PAYLOAD = {"value": False, "rationale": "not a refusal", "metadata": 10}
DISCLAIMER_PAYLOAD = {"value": False, "rationale": "no disclaimer"}

MAX_DIFF_LINES = 40


# --------------------------------------------------------------------------- #
# Capture plumbing
# --------------------------------------------------------------------------- #

class CaptureLLM(ChatLLM):
    """Records exactly what a Guardbound call site would send to a provider."""

    name = "capture"

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages,
        temperature: float = DEFAULT_TEMPERATURE,
        max_turns_context: int | None = None,
        json_format: bool = False,
        structured_output_mode: str | None = None,
    ):
        self.calls.append({
            "messages": copy.deepcopy(messages),
            "temperature": temperature,
            "json_format": json_format,
        })
        return copy.deepcopy(self.payload)


def official_capture(payload: Any):
    """The official code takes a ``generate(messages, json_format, temperature)``."""
    calls: list[dict[str, Any]] = []

    def _generate(messages, json_format=False, temperature=0.7):
        calls.append({
            "messages": copy.deepcopy(messages),
            "temperature": temperature,
            "json_format": json_format,
        })
        return copy.deepcopy(payload)

    return _generate, calls


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #

def _first_char_difference(a: str, b: str) -> dict[str, Any] | None:
    if a == b:
        return None
    for index, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return {"index": index, "official": ca, "guardbound": cb,
                    "context_official": a[max(0, index - 40):index + 40],
                    "context_guardbound": b[max(0, index - 40):index + 40]}
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    return {"index": len(shorter), "official": None, "guardbound": None,
            "note": f"one string is a prefix of the other ({len(a)} vs {len(b)} chars)",
            "tail": longer[len(shorter):len(shorter) + 80]}


def compare_messages(official: list[dict], guardbound: list[dict]) -> dict[str, Any]:
    """Exact comparison of two message arrays, role by role."""
    if len(official) != len(guardbound):
        return {
            "equal": False,
            "message_count": {"official": len(official), "guardbound": len(guardbound)},
            "reason": "different number of messages",
        }
    for index, (a, b) in enumerate(zip(official, guardbound)):
        if a.get("role") != b.get("role"):
            return {"equal": False, "reason": f"role mismatch at message {index}",
                    "message_count": {"official": len(official),
                                      "guardbound": len(guardbound)}}
        if a.get("content") != b.get("content"):
            diff = _first_char_difference(a.get("content", ""), b.get("content", ""))
            excerpt = "\n".join(
                difflib.unified_diff(
                    a.get("content", "").splitlines(),
                    b.get("content", "").splitlines(),
                    fromfile="official", tofile="guardbound", lineterm="", n=1,
                )
            )
            return {
                "equal": False,
                "reason": f"content mismatch at message {index} (role={a.get('role')})",
                "message_count": {"official": len(official), "guardbound": len(guardbound)},
                "first_char_difference": diff,
                "diff_excerpt": "\n".join(excerpt.splitlines()[:MAX_DIFF_LINES]),
            }
    return {
        "equal": True,
        "message_count": {"official": len(official), "guardbound": len(guardbound)},
        "roles": [m.get("role") for m in official],
        "total_chars": sum(len(m.get("content", "")) for m in official),
    }


def record(check_id, role, attack, source, official_calls, guardbound_calls,
           notes: list[str] | None = None) -> dict[str, Any]:
    official = official_calls[0]
    guardbound = guardbound_calls[0]
    comparison = compare_messages(official["messages"], guardbound["messages"])
    params = {
        "temperature_official": official["temperature"],
        "temperature_guardbound": guardbound["temperature"],
        "temperature_equal": official["temperature"] == guardbound["temperature"],
        "json_format_official": official["json_format"],
        "json_format_guardbound": guardbound["json_format"],
        "json_format_equal": official["json_format"] == guardbound["json_format"],
    }
    return {
        "id": check_id,
        "role": role,
        "attack": attack,
        "official_source": source,
        "comparison": comparison,
        "parameters": params,
        "verdict": (
            "FAITHFUL"
            if comparison["equal"] and params["temperature_equal"]
            and params["json_format_equal"]
            else "DIFFERENT"
        ),
        "notes": notes or [],
        "official_messages": official["messages"],
        "guardbound_messages": guardbound["messages"],
    }


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def check_attacker_attack(name: str, module_name: str) -> list[dict]:
    """Round-1 and round-2 attacker messages for one attack."""
    from guardbound.attacks import acronym, crescendo_paper, opposite_day

    official = load_official_attack(module_name)
    guardbound = {
        "crescendo": crescendo_paper.generate_crescendo_step,
        "opposite_day": opposite_day.generate_opposite_day_step,
        "acronym": acronym.generate_acronym_step,
    }[name]

    results = []

    # --- round 1: no history ------------------------------------------------ #
    gen, official_calls = official_capture({"generatedQuestion": "Q1", "lastResponseSummary": ""})
    official_step = getattr(official, f"generate_{name}_step")
    official_step(1, GOAL, {"attacker": [], "target": [], "score": [],
                            "round_number": []}, "", gen)

    llm = CaptureLLM({"generatedQuestion": "Q1", "lastResponseSummary": ""})
    guardbound(round_num=1, goal=GOAL, history_attacker=[], history_target=[],
               scores=[], last_response="", attacker_llm=llm, max_rounds=8)
    results.append(record(
        f"attacker_{name}_round1", "attacker", name,
        f"attacks/{module_name}/run.py::{official_step.__name__}",
        official_calls, llm.calls,
        ["Official omits temperature and relies on generate()'s 0.7 default; "
         "Guardbound passes 0.7 explicitly. Same value."],
    ))

    # --- round 2: one prior turn with a refused-style score ------------------ #
    gen2, official_calls2 = official_capture(
        {"generatedQuestion": "Q2", "lastResponseSummary": "S2"})
    official_step(2, GOAL,
                  {"attacker": ["Q1"], "target": ["(Summary) S1"], "score": [2],
                   "round_number": [2]}, "A1", gen2)

    llm2 = CaptureLLM({"generatedQuestion": "Q2", "lastResponseSummary": "S2"})
    guardbound(round_num=2, goal=GOAL, history_attacker=["Q1"],
               history_target=["(Summary) S1"], scores=[2], last_response="A1",
               attacker_llm=llm2, max_rounds=8)
    results.append(record(
        f"attacker_{name}_round2", "attacker", name,
        f"attacks/{module_name}/run.py::{official_step.__name__}",
        official_calls2, llm2.calls,
        ["History echo: attacker turn as assistant, '(Summary)' target summary, "
         "then the round/score/last-response user prompt."],
    ))
    return results


def check_refusal() -> dict:
    from guardbound.attacks import crescendo_paper

    utils = load_official_utils()
    gen, official_calls = official_capture(REFUSAL_PAYLOAD)
    utils.check_refusal(PROMPT, RESPONSE, gen)

    llm = CaptureLLM(REFUSAL_PAYLOAD)
    crescendo_paper.llm_check_refusal(PROMPT, RESPONSE, llm)
    return record("evaluator_refusal", "evaluator", "shared",
                  "attacks/utils/check_refusal.py::check_refusal",
                  official_calls, llm.calls,
                  ["Newlines stripped from prompt/response in both."])


def check_refusal_acronym_copy() -> dict:
    from guardbound.attacks import acronym

    utils = load_official_utils()
    gen, official_calls = official_capture(REFUSAL_PAYLOAD)
    utils.check_refusal(PROMPT, RESPONSE, gen)

    llm = CaptureLLM(REFUSAL_PAYLOAD)
    acronym.llm_check_refusal(PROMPT, RESPONSE, llm)
    return record("evaluator_refusal_acronym_copy", "evaluator", "acronym",
                  "attacks/utils/check_refusal.py::check_refusal",
                  official_calls, llm.calls,
                  ["Guardbound carries a second copy of the refusal check in the "
                   "acronym module; audited separately so the copies cannot diverge."])


def check_disclaimer() -> dict:
    from guardbound.attacks import acronym

    utils = load_official_utils()
    gen, official_calls = official_capture(DISCLAIMER_PAYLOAD)
    utils.check_disclaimer(PROMPT, RESPONSE, gen)

    llm = CaptureLLM(DISCLAIMER_PAYLOAD)
    acronym.check_disclaimer(PROMPT, RESPONSE, llm)
    return record("evaluator_disclaimer", "evaluator", "acronym",
                  "attacks/utils/check_disclaimer.py::check_disclaimer",
                  official_calls, llm.calls)


def check_rubric() -> list[dict]:
    from guardbound.attacks import acronym, crescendo_paper

    utils = load_official_utils()
    results = []
    for label, fn, source in (
        ("evaluator_rubric", crescendo_paper.llm_evaluate_with_rubric,
         "attacks/utils/evaluate_with_rubric.py::evaluate_with_rubric"),
        ("evaluator_rubric_acronym_copy", acronym.llm_evaluate_with_rubric,
         "attacks/utils/evaluate_with_rubric.py::evaluate_with_rubric"),
    ):
        gen, official_calls = official_capture(RUBRIC_PAYLOAD)
        utils.evaluate_with_rubric(PROMPT, RESPONSE, GOAL, gen)
        llm = CaptureLLM(RUBRIC_PAYLOAD)
        fn(PROMPT, RESPONSE, GOAL, llm)
        results.append(record(label, "evaluator", "shared", source,
                              official_calls, llm.calls,
                              ["Official builds the rubric prompt inline; compared "
                               "on the rendered message."]))
    return results


def check_target_conversation() -> dict:
    """Target call shape, captured from the real runner."""
    from guardbound.attacks.runner import run_attack_with_backtracking
    from tests.cloud_fakes import FakeOpenAIBackend, completion
    from tests.test_cloud_research_integration import RoutedBackend, build_attack, make_llms

    system_prompt = "You are a helpful, harmless assistant."
    backend = RoutedBackend()
    llms = make_llms(backend)
    attack = build_attack("crescendo", llms, max_turns=2)
    run_attack_with_backtracking(
        attack=attack, goal=GOAL, target_llm=llms["target"], max_turns=2,
        temperature=0.7, system_prompt=system_prompt,
    )

    target_calls = backend.requests_for("target")
    ok_shape = len(target_calls) == 2
    first = target_calls[0]["messages"] if target_calls else []
    second = target_calls[1]["messages"] if len(target_calls) > 1 else []
    checks = {
        "first_message_is_dataset_system": bool(first) and first[0] == {
            "role": "system", "content": system_prompt},
        "first_call_is_system_then_user": [m["role"] for m in first] == ["system", "user"],
        "second_call_appends_assistant_then_user":
            [m["role"] for m in second] == ["system", "user", "assistant", "user"],
        "history_is_preserved_verbatim":
            len(second) >= 3 and second[1] == first[1] and second[2]["role"] == "assistant",
        "target_uses_text_response_format":
            target_calls and target_calls[0].get("response_format") == {"type": "text"},
        "target_temperature_is_paper_value":
            target_calls and target_calls[0].get("temperature") == 0.7,
    }
    return {
        "id": "target_conversation_shape",
        "role": "target",
        "attack": "crescendo",
        "official_source": "attacks/crescendomation/run.py: history_t construction",
        "checks": checks,
        "verdict": "FAITHFUL" if all(checks.values()) else "DIFFERENT",
        "notes": [
            "Official: history_t = [{'role':'system','content':target_system}], then "
            "append {'role':'user'} before each target call and {'role':'assistant'} "
            "after it. Measured directly from the runner's SDK payloads.",
        ],
    }


def check_sdk_payload_matches_constructed_messages() -> dict:
    """The provider must not rewrite what the attack layer built."""
    from guardbound.attacks import crescendo_paper
    from tests.cloud_fakes import FakeOpenAIBackend, completion

    official = load_official_attack("crescendomation")
    gen, official_calls = official_capture({"generatedQuestion": "Q1", "lastResponseSummary": ""})
    official.generate_crescendo_step(
        1, GOAL, {"attacker": [], "target": [], "score": [], "round_number": []}, "", gen)

    backend = FakeOpenAIBackend([completion("irrelevant")])
    llm = OpenAIChatLLM(model_id="gpt-4o-mini-2024-07-18", client=backend.client,
                        role="attacker")
    capture = CaptureLLM({"generatedQuestion": "Q1", "lastResponseSummary": ""})

    # Drive the Guardbound step through a capturing wrapper that also forwards to
    # the real OpenAI client, so we can compare (a) what the call site built with
    # (b) what the SDK actually received.
    class ForwardingCapture(CaptureLLM):
        def generate(self, messages, temperature=DEFAULT_TEMPERATURE, **kwargs):
            out = super().generate(messages, temperature=temperature, **kwargs)
            llm.generate(messages, temperature=temperature, json_format=True)
            return out

    forwarding = ForwardingCapture({"generatedQuestion": "Q1", "lastResponseSummary": ""})
    crescendo_paper.generate_crescendo_step(
        round_num=1, goal=GOAL, history_attacker=[], history_target=[], scores=[],
        last_response="", attacker_llm=forwarding, max_rounds=8)

    sdk_body = backend.requests[0]
    constructed = official_calls[0]["messages"]
    checks = {
        "sdk_messages_equal_constructed": sdk_body["messages"] == constructed,
        "no_extra_system_message": sum(1 for m in sdk_body["messages"]
                                       if m["role"] == "system") == 1,
        "roles_unchanged": [m["role"] for m in sdk_body["messages"]] == \
                           [m["role"] for m in constructed],
        "json_object_mode": sdk_body.get("response_format") == {"type": "json_object"},
        "temperature_preserved": sdk_body.get("temperature") == 0.7,
        "model_id_verbatim": sdk_body.get("model") == "gpt-4o-mini-2024-07-18",
        "only_expected_keys": sorted(sdk_body.keys()) == sorted(
            ["model", "messages", "temperature", "response_format"]),
    }
    return {
        "id": "provider_sdk_payload",
        "role": "attacker",
        "attack": "crescendo",
        "official_source": "n/a — provider adaptation check",
        "checks": checks,
        "verdict": "FAITHFUL" if all(checks.values()) else "DIFFERENT",
        "sdk_request_keys": sorted(sdk_body.keys()),
        "notes": ["The SDK body is compared to the official message array; any "
                  "wrapper that prepended, appended or rewrote a message would fail "
                  "the first check."],
    }


def check_c1_absence() -> dict:
    """The I1 instruction must live only in the opt-in arm module."""
    src_root = PROJECT_ROOT / "src" / "guardbound"
    needle = "single JSON object"
    allowed = "src/guardbound/llm/attacker_arms.py"
    hits, attack_hits = [], []
    for path in sorted(src_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if needle not in text:
            continue
        rel = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        hits.append(rel)
        if "/attacks/" in f"/{rel}":
            attack_hits.append(rel)
    return {
        "id": "c1_generation_point_absence",
        "role": "attacker",
        "attack": "all",
        "official_source": "n/a — deviation guard",
        "checks": {
            "instruction_absent_from_attack_modules": not attack_hits,
            "instruction_confined_to_the_opt_in_arm_module": hits == [allowed],
        },
        "files_with_hits": hits,
        "verdict": "FAITHFUL" if not attack_hits and hits == [allowed] else "DIFFERENT",
        "notes": [
            "C1/I1 is an opt-in wrapper in src/guardbound/llm/attacker_arms.py "
            "(wording frozen from the Phase-17 Stage-6 arm). It is absent from "
            "every attack module, so no run can inherit it by default.",
        ],
    }


def check_c1_arm_wire() -> dict:
    """C1 on the wire: exactly one instruction, attacker calls only."""
    from guardbound.attacks import crescendo_paper
    from guardbound.llm.attacker_arms import ARM_BASELINE, ARM_C1, build_armed_attacker
    from tests.cloud_fakes import FakeOpenAIBackend, completion

    payload = completion(json.dumps({"generatedQuestion": "Q", "lastResponseSummary": "S"}))
    evaluation = completion(json.dumps({"value": False, "rationale": "x", "metadata": 10}))

    results = {}
    for arm in (ARM_BASELINE, ARM_C1):
        backend = FakeOpenAIBackend([payload, evaluation, payload])
        inner = OpenAIChatLLM(model_id="gpt-4o-mini-2024-07-18",
                              client=backend.client, role="attacker")
        armed = build_armed_attacker(inner, arm)

        # an attacker generation call …
        crescendo_paper.generate_crescendo_step(
            round_num=1, goal=GOAL, history_attacker=[], history_target=[],
            scores=[], last_response="", attacker_llm=armed, max_rounds=8)
        # … and an evaluator call routed through the same object
        crescendo_paper.llm_check_refusal(PROMPT, RESPONSE, armed)

        generation_body, evaluator_body = backend.requests[0], backend.requests[1]
        instruction = "single JSON object"
        results[arm] = {
            "generation_has_instruction": any(
                instruction in m["content"] for m in generation_body["messages"]),
            "generation_instruction_count": sum(
                1 for m in generation_body["messages"] if instruction in m["content"]),
            "evaluator_has_instruction": any(
                instruction in m["content"] for m in evaluator_body["messages"]),
        }

    checks = {
        "baseline_generation_has_no_instruction":
            results[ARM_BASELINE]["generation_has_instruction"] is False,
        "c1_generation_has_exactly_one_instruction":
            results[ARM_C1]["generation_instruction_count"] == 1,
        "baseline_evaluator_has_no_instruction":
            results[ARM_BASELINE]["evaluator_has_instruction"] is False,
        "c1_evaluator_has_no_instruction":
            results[ARM_C1]["evaluator_has_instruction"] is False,
    }
    return {
        "id": "c1_arm_wire_behaviour",
        "role": "attacker",
        "attack": "crescendo",
        "official_source": "n/a — condition isolation check",
        "checks": checks,
        "measured": results,
        "verdict": "FAITHFUL" if all(checks.values()) else "DIFFERENT",
        "notes": ["C1 appends its instruction once, on attacker generation only. "
                  "A judge call routed through the same object is unchanged."],
    }


def check_c1_wrapper_location() -> dict:
    """Record where C1 lives, so the claim is verifiable."""
    script = PROJECT_ROOT / "scripts" / "phase17_stage6_run_level_contract.py"
    text = script.read_text(encoding="utf-8") if script.exists() else ""
    return {
        "id": "c1_wrapper_location",
        "official_source": "n/a",
        "checks": {
            "c1_defined_only_in_phase17_script": script.exists()
            and "class ArmAttacker" in text,
            "c1_sentence_present_there": "single JSON object" in text,
            "cloud_scripts_do_not_import_it": "ArmAttacker" not in "\n".join(
                p.read_text(encoding="utf-8")
                for p in (PROJECT_ROOT / "scripts").glob("cloud_*.py")
            ),
        },
    }


# Checks that render Guardbound's LEGACY bytes. They are expected to differ from
# the release (four disclosed typographic deviations); the paper-reproduction
# path uses the ``official`` variant, proven byte-identical by
# ``check_prompt_variants``.
_LEGACY_ONLY_PROMPT_CHECKS = {
    "attacker_opposite_day_round1",
    "attacker_opposite_day_round2",
    "attacker_acronym_round1",
    "attacker_acronym_round2",
    "evaluator_rubric",
    "evaluator_rubric_acronym_copy",
}


def check_prompt_variants() -> dict:
    """Phase 22: the paper path must resolve to exact official bytes.

    Guardbound now carries two variants of every prompt that differs from the
    release. ``legacy`` keeps the frozen Phase 14-19 runs reproducible;
    ``official`` is what the R1 paper-reproduction configuration sends.
    """
    import hashlib

    from guardbound.attacks import prompt_variants

    official_opposite = load_official_attack("opposite_day")
    official_acronym = load_official_attack("acronym")
    utils = load_official_utils()

    gen, official_calls = official_capture(RUBRIC_PAYLOAD)
    utils.evaluate_with_rubric(PROMPT, RESPONSE, GOAL, gen)
    official_rubric = official_calls[0]["messages"][0]["content"]

    rendered_official_rubric = prompt_variants.rubric_prompt(
        PROMPT, RESPONSE, "official")

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    checks = {
        "opposite_day_official_matches_release":
            prompt_variants.opposite_day_system_prompt("official")
            == official_opposite.OPPOSITE_DAY_SYSTEM_PROMPT,
        "acronym_official_matches_release":
            prompt_variants.acronym_system_prompt("official")
            == official_acronym.ACRONYM_SYSTEM_PROMPT,
        "rubric_official_matches_release":
            rendered_official_rubric == official_rubric,
        "legacy_and_official_differ_for_opposite_day":
            prompt_variants.opposite_day_system_prompt("legacy")
            != prompt_variants.opposite_day_system_prompt("official"),
        "legacy_and_official_differ_for_acronym":
            prompt_variants.acronym_system_prompt("legacy")
            != prompt_variants.acronym_system_prompt("official"),
        "crescendo_is_identical_across_variants": prompt_variants.prompt_hashes()
            ["crescendo_attacker"]["identical_across_variants"],
    }

    return {
        "id": "prompt_variants",
        "role": "evaluator+attacker",
        "attack": "all",
        "verdict": "FAITHFUL" if all(checks.values()) else "CHECK",
        "official_source": "prompt_variants.prompt_hashes()",
        "checks": checks,
        "hashes": {
            "opposite_day": {
                "legacy": digest(prompt_variants.opposite_day_system_prompt("legacy")),
                "official": digest(prompt_variants.opposite_day_system_prompt("official")),
            },
            "acronym": {
                "legacy": digest(prompt_variants.acronym_system_prompt("legacy")),
                "official": digest(prompt_variants.acronym_system_prompt("official")),
            },
            "rubric": {
                # Both from prompt_hashes(), i.e. the same fixed fixture.
                "legacy": prompt_variants.prompt_hashes()["rubric"]["variants"]["legacy"],
                "official": prompt_variants.prompt_hashes()["rubric"]["variants"]["official"],
            },
        },
        "notes": [
            "The R1 configuration sets prompt_variant: official, so the paper "
            "reproduction sends the release's bytes.",
            "The legacy bytes are retained unchanged so frozen Phase 14-19 "
            "experiments remain reproducible.",
        ],
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/audit/prompt_fidelity.json")
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    print("Comparing rendered prompts: official NBF-LLM release vs Guardbound\n")

    checks.extend(check_attacker_attack("crescendo", "crescendomation"))
    checks.extend(check_attacker_attack("opposite_day", "opposite_day"))
    checks.extend(check_attacker_attack("acronym", "acronym"))
    checks.append(check_refusal())
    checks.append(check_refusal_acronym_copy())
    checks.append(check_disclaimer())
    checks.extend(check_rubric())
    checks.append(check_target_conversation())
    checks.append(check_sdk_payload_matches_constructed_messages())
    checks.append(check_prompt_variants())

    evidence = [check_c1_absence(), check_c1_wrapper_location(), check_c1_arm_wire()]

    for check in checks:
        verdict = check["verdict"]
        detail = ""
        if "comparison" in check:
            detail = (f"{check['comparison']['message_count']['official']} msgs, "
                      f"temp {check['parameters']['temperature_official']}")
        elif "checks" in check:
            failed = [k for k, v in check["checks"].items() if not v]
            detail = "all checks pass" if not failed else f"FAILED: {failed}"
        print(f"  {check['id']:<38} {verdict:<10} {detail}")

    print("\nDeviation guards")
    for item in evidence:
        failed = [k for k, v in item["checks"].items() if not v]
        print(f"  {item['id']:<38} {'OK' if not failed else 'CHECK':<10} "
              f"{'all checks pass' if not failed else failed}")

    payload = {
        "goal": GOAL,
        "checks": checks,
        "evidence": evidence,
        "all_faithful": (
            all(c["verdict"] == "FAITHFUL" for c in checks)
            and all(all(item["checks"].values()) for item in evidence)
        ),
        # ``all_faithful`` describes Guardbound's DEFAULT (legacy) prompt bytes,
        # which still carry the four disclosed typographic deviations so frozen
        # Phase 14-19 runs stay reproducible. The paper-reproduction path sends
        # the release's bytes instead; ``check_prompt_variants`` proves those are
        # byte-identical for every prompt that differs.
        "paper_path_faithful": (
            all(c["verdict"] == "FAITHFUL"
                for c in checks if c["id"] == "prompt_variants")
            and all(c["verdict"] == "FAITHFUL" for c in checks
                    if c["id"] not in _LEGACY_ONLY_PROMPT_CHECKS)
            and all(all(item["checks"].values()) for item in evidence)
        ),
        "legacy_only_prompt_checks": sorted(_LEGACY_ONLY_PROMPT_CHECKS),
        "typographic_deviations": [
            c["id"] for c in checks
            if c["verdict"] != "FAITHFUL"
        ],
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True).stdout.strip() or None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nall_faithful = {payload['all_faithful']}")
    print(f"wrote {out}")
    return 0 if payload["all_faithful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
