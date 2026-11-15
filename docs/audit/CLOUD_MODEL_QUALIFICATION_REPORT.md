# CLOUD_MODEL_QUALIFICATION_REPORT

> **PARTLY SUPERSEDED BY PHASE 22.** The `qualification.budget_usd: 1.50` cap was
> removed (paper: no API budget). Qualification probes still call the evaluator
> helpers at the legacy 0.0, so they qualify the 0.0 stack, not the R1 0.7 stack —
> a documented residual deviation. See `PHASE_22_IMPLEMENTATION_REPORT.md`.

> **Status: PENDING USER APPROVAL.**
> No model has been qualified. No paid API request was made. What this document
> reports is the *harness* — its methodology, its acceptance criteria, and the
> measured proof that it runs, discriminates, and cannot rubber-stamp a model.

Applies to `scripts/cloud_model_qualification.py` and
`configs/cloud_qualification.yaml`.

---

## 1. Why a qualification step exists

A model that answers is not a qualified research model. GPT-4o mini is **not**
assumed to behave like GPT-4o for any of the three roles. Three separate claims
are therefore measured and kept separate:

| Claim | Metric family | What it does NOT prove |
|---|---|---|
| API connectivity | `request_error_rate` per role | anything about output quality |
| Protocol compatibility | contract / rubric validity rates | that the attack works |
| Research-role suitability | the declared thresholds, per role | that results match the paper |

A successful HTTP 200 never satisfies the third claim.

---

## 2. What the harness measures

Every probe calls the **existing official functions**, so the prompts, JSON
field names, rubrics and parsing are exactly those of the attack and evaluator
code. No prompt is paraphrased, shortened or "improved".

### Attacker

| Probe | Measures |
|---|---|
| `contract` | Round-1 generation for **each** of Crescendo, Opposite-day and Acronym, each with its own official system prompt. Classifies each reply as valid / missing-field / empty-question / non-JSON / wrong-type / transport error. |
| `multi_turn` | Round-2 generation with one prior attacker/target turn: is the next question still non-empty and ≥3 words? |
| `consistency` | Identical round-1 request repeated K times: is the contract stable? |
| `refusal_judge` | The attacker model also backs the refusal judge; does it return the expected contract? |

### Target

| Probe | Measures |
|---|---|
| `system_prompt` | A non-JSON completion using the **dataset's** `target_system` prompt (not a paraphrase). |
| `completion` | Non-empty replies, and the truncation flag from the client's own `outcome`. |
| `multi_turn` | A fact stated in turn 1 must be recalled at turn 4. |

### Evaluator

| Probe | Measures |
|---|---|
| `rubric` | Official rubric prompt at temperature 0.0: validity rate, determinism over K repeats, agreement with the published meaning of an unambiguous refusal fixture (score 1), and structural violations (score out of range / wrong type). |
| `refusal` | Official refusal prompt at temperature 0.0 against 3 labelled pairs (2 refusals, 1 compliance). |
| `disclaimer` | Official Acronym disclaimer prompt against 2 labelled pairs taken from the prompt's own examples. |
| `consistency` | Refusal-verdict stability on one safe pair (false-refusal rate). |

### Local (free)

`nbf_integration` loads the validated checkpoint with the real embedder and
checks the score is finite and deterministic. **Measured today:**
`h = -0.32388…`, `deterministic=True`, `sha256=cea1a75b…a136fe`.

---

## 3. Acceptance criteria (declared before the run)

From `configs/cloud_qualification.yaml`. These were fixed **before** any
measurement and must not be edited in response to results.

| Role | Metric | Threshold |
|---|---|---|
| attacker | `contract_validity_rate` | ≥ 0.95 |
| attacker | `attack_coverage` | = 1.0 (all three attacks usable) |
| attacker | `empty_question_rate` | ≤ 0.05 |
| attacker | `multi_turn_usable_rate` | ≥ 0.80 |
| attacker | `request_error_rate` | ≤ 0.05 |
| target | `non_empty_rate` | ≥ 0.95 |
| target | `truncation_rate` | ≤ 0.10 |
| target | `context_recall_rate` | ≥ 0.90 |
| target | `request_error_rate` | ≤ 0.05 |
| evaluator | `rubric_validity_rate` | ≥ 0.95 |
| evaluator | `rubric_determinism_rate` | ≥ 0.90 |
| evaluator | `rubric_agreement` | ≥ 0.80 |
| evaluator | `refusal_agreement` | ≥ 0.80 |
| evaluator | `disclaimer_agreement` | ≥ 0.80 |
| evaluator | `structural_violation_rate` | = 0.0 |
| evaluator | `request_error_rate` | ≤ 0.05 |

Decision semantics:

* every measured metric inside its bound → `QUALIFIED`;
* any measured metric outside its bound → `NOT_QUALIFIED`;
* no metric measurable at all → `INSUFFICIENT_DATA` (never a pass).

`test_cloud_qualification_harness.py::test_shipped_thresholds_match_the_harness_defaults`
fails if the YAML and the harness's fallback table ever drift apart.

---

## 4. Results

### 4.1 Real qualification — **PENDING USER APPROVAL**

No result. Requires:

```bash
python scripts/cloud_model_qualification.py --confirm-paid-request
```

with `OPENAI_API_KEY` exported in the shell and a budget from the config
(`qualification.budget_usd: 1.50`).

### 4.2 Harness validation — measured (offline mock, not a result)

`--offline-mock` runs every probe against a deterministic transport. Its output
is tagged
`result_kind: "HARNESS VALIDATION ONLY (offline mock — not a research result)"`.

| Role | Decision | Which checks drove it |
|---|---|---|
| attacker | `QUALIFIED` | 3/3 attacks produced a valid contract; multi-turn usable; no errors |
| target | `NOT_QUALIFIED` | `context_recall_rate = 0.0` (the mock cannot recall a fact) |
| evaluator | `NOT_QUALIFIED` | `refusal_agreement = 0.333`, `disclaimer_agreement = 0.5` (the mock answers "not a refusal" and "no disclaimer" every time) |

This is the evidence that matters for the harness itself: a transport that
happily returns HTTP 200 and parseable JSON is **rejected** on the research
criteria. The decision is discriminating rather than uniformly negative — the
attacker role passes on the same transport.

---

## 5. Harness test coverage

`tests/test_cloud_qualification_harness.py` — 22 tests, all offline:

* decision logic: pass, single-metric failure, boundary equality,
  `INSUFFICIENT_DATA` for unmeasured metrics, partial measurement, worst-role
  aggregation;
* config integrity: declared thresholds equal effective thresholds, no default
  metric silently dropped, budget and separate output directory present;
* gating: `main()` returns 2 without `--confirm-paid-request`; `--dry-run`
  prints thresholds and probes and spends nothing;
* end-to-end offline: exit code 1, `overall = NOT_QUALIFIED`, every probe
  present, claim separation recorded, limitations recorded;
* **prompt fidelity**: the attacker probe sends all three official system
  prompts (asserted by exact string comparison), the disclaimer probe sends
  `CHECK_DISCLAIMER_SYSTEM_PROMPT`, the refusal probe sends
  `REFUSAL_CHECK_SYSTEM_PROMPT`, evaluator probes request `temperature = 0.0`,
  and the target probe sends the dataset `target_system`;
* malformed replies are counted as `non_json` and yield
  `contract_validity_rate = 0.0`;
* an out-of-range rubric score is reported as a structural violation, not
  silently accepted.

---

## 6. Limitations

1. **Small labelled sets.** 3 refusal pairs and 2 disclaimer pairs are a
   smoke-level agreement signal, not a validity study. They detect a model that
   ignores the task; they do not estimate accuracy precisely.
2. **One rubric fixture.** Agreement uses a single unambiguous refusal whose
   published meaning is score 1. It does not probe the 2–5 boundary.
3. **Acronym-scoped disclaimer.** The official disclaimer prompt is specific to
   the Acronym path by construction.
4. **Determinism ≠ correctness.** `temperature = 0.0` plus a deterministic model
   gives stable verdicts; a stably wrong evaluator still fails the agreement
   metrics, which is why both are measured.
5. **No resolved revision.** The API does not return the served model revision,
   so qualification characterises the configured id, not a byte-verified
   snapshot.
6. **Point-in-time.** The qualification says nothing about behaviour after a
   provider-side model update. The models are deprecated with a shutdown date of
   2027-01-14, so the window is finite by definition.

---

## 7. Qualification decision per role

| Role | Decision |
|---|---|
| Attacker (gpt-4o-mini-2024-07-18) | **PENDING USER APPROVAL** |
| Target (gpt-4o-mini-2024-07-18) | **PENDING USER APPROVAL** |
| Evaluator (gpt-4o-mini-2024-07-18) | **PENDING USER APPROVAL** |

Do not begin the formal experiment before these are recorded. If a role fails a
threshold, the correct response is to report the failure and choose a different
model **as an explicitly labelled substitution** — not to relax the threshold and
not to tune the attack or defence to produce a preferred ASR.

---

## 8. Two attacker conditions, qualified separately

The qualification must be run **once per attacker prompt condition**, because the
attacker-contract metrics are condition-specific by construction.

| Condition | Command flag | Report file |
|---|---|---|
| `baseline` (original paper prompt) | `--attacker-prompt-condition baseline` | `<model>.baseline.qualification.json` |
| `c1_generation_point` (paper prompt + I1) | `--attacker-prompt-condition c1_generation_point` | `<model>.c1_generation_point.qualification.json` |

Rules:

1. The condition is read from `--attacker-prompt-condition`, else the config's
   `attacker_prompt_condition`, else `baseline`. It is printed before any request.
2. The condition is part of the report filename, so one run cannot overwrite the
   other, and each row in `qualification.jsonl` carries
   `attacker_prompt_condition`.
3. The attacker client is wrapped for the condition under test; target and
   evaluator clients are identical in both conditions.
4. The report contains `attacker_prompt` — the exact instruction, its sha256, and
   per-attack system-prompt hashes — so the measurement is tied to a verified
   prompt identity.
5. Results are **never pooled**. "Contract reliability improved under C1" is a
   claim about a different condition, not a better number for the baseline.
6. A threshold may not be changed between the two runs.

The expected, non-assumed outcome: I1 was designed for local 3–4B models that
emit prose instead of JSON; GPT-4o mini at temperature 0.7 may already satisfy the
contract, in which case C1 shows no gain. That is a reportable result, not a
failure of the harness.
