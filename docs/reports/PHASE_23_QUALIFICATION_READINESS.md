# PHASE 23 — QUALIFICATION READINESS

Whether the existing harness can qualify **the R1 stack**, the exact mismatches,
the minimal correction, and a fully specified future qualification protocol.

Nothing was executed against a model. No budget cap exists or was added.

---

## 1. What the harness is

`scripts/cloud_model_qualification.py` + `configs/cloud_qualification.yaml`.
It probes the attacker, target and evaluator roles with the **official prompts and
parsers** and returns `QUALIFIED / NOT_QUALIFIED / INSUFFICIENT_DATA` per role
against thresholds declared in advance.

Probes: `attacker_contract` (all three attacks), `attacker_multi_turn`,
`attacker_consistency`, `attacker_refusal_judge`, `target_system_prompt`,
`target_completion`, `target_multi_turn`, `evaluator_rubric`,
`evaluator_refusal`, `evaluator_disclaimer`, `evaluator_consistency`,
`nbf_integration` (local, free).

---

## 2. Does it test the R1 configuration? — No, in five ways

| # | Item | R1 | Qualification harness as shipped | Consequence |
|---|---|---|---|---|
| 1 | Models | `gpt-4o-2024-08-06` ×3 | `gpt-4o-mini-2024-07-18` ×3 (`cloud_qualification.yaml:33-42`) | qualifies a different model unless `--model gpt-4o-2024-08-06` is passed |
| 2 | Evaluator temperature | 0.7 | **0.0** — probes call `llm_evaluate_with_rubric`, `llm_check_refusal`, `check_disclaimer` **without** a temperature (`:485, :597, :641, :670, :697`), so the helpers' legacy default applies | rubric determinism / agreement measured on a configuration R1 does not use |
| 3 | Target temperature (recall probe) | 0.7 | **0.0** (`:546` `llm.generate(list(history), temperature=0.0)`) | context-recall measured at a temperature R1 does not use |
| 4 | Prompt variant | `official` | **legacy** — the probes call the step generators directly, which default to the legacy constants | the harness measures reproducibility of bytes R1 does not send |
| 5 | NBF threshold | `η = 0.001` | `nbf.threshold: 0.0`, `nbf.eta` absent (`cloud_qualification.yaml:82`) | the NBF probe validates the checkpoint, not the R1 margin |

Plus one structural gap:

| 6 | ActorAttack | in the R1 attack set, composite execution | **no ActorAttack probe at all** (only crescendo / opposite_day / acronym contracts) | the highest-complexity attack is unqualified |

---

## 3. Verification matrix (the 15 items in the brief)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Target model is GPT-4o | **NO** by default (mini); `--model gpt-4o-2024-08-06` works | `cloud_qualification.yaml:37-42`; `--model` override exists |
| 2 | Attacker model is GPT-4o | **NO** by default, same override | same |
| 3 | Evaluator model is GPT-4o | **NO** by default, same override | same |
| 4 | Attacker temperature 0.7 | PASS | `:520` `temperature=0.7` |
| 5 | Target temperature 0.7 | **NO** for the recall probe (0.0); other target probes use `:520`-style defaults | `:546` |
| 6 | Evaluator temperature 0.7 | **NO** (legacy 0.0) | `:485, :597, :641, :670, :697` |
| 7 | Official prompt variant selected | **NO** (legacy constants) | probe call sites |
| 8 | Paper attack set represented | **PARTIAL** — three text attacks probed, ActorAttack absent, RedQueen n/a | `probe_attacker_contract` |
| 9 | ActorAttack handled via the composite path | **NO** | not probed |
| 10 | NBF threshold 0.001 | **NO** (0.0 in the qualification config) | `cloud_qualification.yaml:82` |
| 11 | C1/I1 disabled | PASS | baseline condition default; `c1_enabled` not read by the harness; arm wrapper not used |
| 12 | No budget cap enforced | PASS | no `budget_usd` anywhere; `--budget-usd` rejected |
| 13 | No hidden retry / repair / fallback | PASS | `max_retries=0`; probes call once per sample; no repair branch |
| 14 | Usage metadata persisted | PASS | `qualification.jsonl` + report `usage` block (32-call shape verified in the pilot; qualification test asserts the block) |
| 15 | Failed / unpriced calls reported honestly | PASS | `usage_available=False`, cost `None` (never 0); error rates per role |

---

## 4. Is the mismatch a blocker?

**It is a blocker for "qualify the R1 stack", not for the R1 experiment code.**

The R1 execution path itself is unaffected: `cloud_pilot.py` resolves 0.7 / official
/ η=0.001 from the R1 YAML and passes them to the attacks (verified in
`PHASE_23_R1_CONFIGURATION_TRACE.md`). What is broken is the *gate*: running the
harness today would produce a `QUALIFIED` verdict that describes a stack R1 will
never run, which is worse than no verdict because it looks like evidence.

Classified **P1** (not P0; register **D-05**): no paper-critical execution path is
wrong because of it, but the qualification evidence would be unrepresentative.

### Minimal correction (proposed for Phase 24, not implemented)

1. **Config** — add `configs/qualification_r1_gpt4o.yaml`:
   `models` = `gpt-4o-2024-08-06` for all three roles; `temperature: {attacker:
   0.7, target: 0.7, evaluator: 0.7}`; `prompt_variant: official`;
   `nbf: {threshold: 0.001, eta: 0.001}`; `attacks: {max_turns: 8}`;
   `qualification: {output_dir: results/qualification_r1}` (no cap).
2. **Plumbing** — read the config's per-role temperatures and `prompt_variant` once
   and pass them into every probe:
   `llm_evaluate_with_rubric(..., temperature=evaluator_temperature,
   official_prompt=(variant == "official"))`, `llm_check_refusal(...,
   temperature=...)`, `check_disclaimer(..., temperature=...)`,
   `generate_*_step(..., system_template=..., )`, and
   `llm.generate(history, temperature=target_temperature)` for the recall probe.
3. **ActorAttack probe** — a composite contract probe with scripted *assertions on
   call shape* (one `prepare`, three fresh conversations, `N_retry=3`, one summary
   per non-skipped actor, early stop at 5) using the real attacker model for
   `infer_single` and a deterministic target stub. This tests the model's ability
   to produce a valid pre-attack payload, which is the only ActorAttack-specific
   model risk.
4. **Tests** — assert that each probe receives the config's temperature, that the
   official variant is used, and that η from the config reaches the NBF probe.

Scope: no frozen experiment is touched; no paper-critical file changes; ~4 files.

---

## 5. Future minimal qualification protocol (designed, not executed)

### 5.1 Scope

| Element | Specification |
|---|---|
| Roles | attacker, target, evaluator — each resolved independently and each judged separately |
| Model | `gpt-4o-2024-08-06` (the R1 model), pinned; no fallback, no substitution |
| Prompts | `official` variant; the three attack system prompts + refusal + disclaimer + rubric, all rendered by the production helpers |
| Attacks | crescendo, opposite_day (full contract probe) + ActorAttack pre-attack contract probe |
| Temperatures | 0.7 / 0.7 / 0.7 (the R1 values) |
| NBF | checkpoint identity + score determinism at η = 0.001 (local, free) |
| C1/I1 | disabled; the arm wrapper is not constructed |
| Spend | no cap; usage reported per role; the operator decides when to stop |

### 5.2 Per-role pass criteria (declared before the run)

| Role | Metric | Bound |
|---|---|---|
| attacker | contract validity (3 attacks × N samples) | ≥ 0.95 |
| attacker | attack coverage (≥1 usable round-1 question per attack) | = 1.0 |
| attacker | empty-question rate | ≤ 0.05 |
| attacker | multi-turn usable rate | ≥ 0.80 |
| attacker | request-error rate | ≤ 0.05 |
| target | non-empty rate | ≥ 0.95 |
| target | truncation rate | ≤ 0.10 |
| target | context recall at turn 4 | ≥ 0.90 |
| evaluator | rubric validity | ≥ 0.95 |
| evaluator | rubric determinism (K repeats at 0.7) | ≥ 0.90 |
| evaluator | agreement on labelled fixtures | ≥ 0.80 |
| evaluator | structural violations (out-of-range score / wrong type) | = 0.0 |
| all | request-error rate | ≤ 0.05 |

### 5.3 Invalid response

Counted, never repaired: non-JSON when JSON was requested, missing field, wrong
type, empty question, out-of-range rubric score, refusal/disclaimer contract
violation. Each is recorded with its category; none is retried or re-prompted.

### 5.4 Usage and errors

One `UsageRecord` per call with role, tokens (input / cached / output / total),
duration, request id, outcome, cost and cost basis; `None` cost when the model has
no rate or usage is missing. Errors classified by the eight provider categories and
recorded, never retried.

### 5.5 Safe-stop conditions

Stop the run (do not continue) if: the API key is absent; a model id is rejected as
unknown; the request-error rate exceeds 0.20 in any role after the first 10 calls;
the NBF checkpoint hash differs; or the process is interrupted. The harness writes
one row per probe set, so a partial qualification is visible as missing rows rather
than a partial verdict.

### 5.6 What failure means

A `NOT_QUALIFIED` role is reported as such. The remedy is to choose a different
model as an explicitly labelled substitution — not to relax a threshold, not to
edit a prompt, and not to tune the attack.

---

## 6. Verdict

The harness is well-built but **does not test the R1 stack**: it defaults to a
different model, a different evaluator temperature, the legacy prompt bytes, a
zero margin and one fewer attack. That is a P1 gate defect with a four-file minimal
correction. Until it is corrected, any qualification verdict would not be evidence
about R1.
