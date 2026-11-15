# PHASE 34.3 — FINAL TELEMETRY, IDENTITY PROVENANCE & CONFIGURATION CONSISTENCY AUDIT

**Correction successor to `PHASE_34_2_FINAL_ACCOUNTING_AND_SCHEMA_AUDIT.md`.**
Phases 34, 34.1 and 34.2 are preserved unchanged. This report supersedes them wherever the four
disagree; it does not repeal any decision that survives the corrections below.

| Component | Status |
|---|---|
| MT-Bench methodology | **`READY FOR APPROVAL`** (A29 decision required) |
| XSTest methodology | **`READY FOR APPROVAL`** |
| Shared score identity — **schema** | **`READY FOR APPROVAL`** |
| Shared score identity — **production instance** | **`INCOMPLETE`** — 1 hash outstanding + A29 |
| Arm configuration | **`READY FOR APPROVAL`** |
| Observation / telemetry schema | **`READY FOR APPROVAL`** |
| Request / cost methodology | **`READY FOR APPROVAL`** |
| Dataset acquisition | **`COMPLETE — 4/4 BLOB-VERIFIED`** (separate authorization) |
| Utility implementation | **`NOT IMPLEMENTED`** |
| Paid evaluation | **`NOT AUTHORIZED`** |
| Full Table 1 | **`NOT READY`** |

Read-only specification pass. **No source code, configuration, test, manifest or historical
artifact was modified. No network request was made. No dataset was downloaded or altered. No
model, judge or classifier was invoked. No credential was read, no paid call was made, no pytest /
qualification / smoke test / pilot / experiment was run.** Exactly one file was created: this
report.

> `READY FOR APPROVAL` describes a specification. It authorizes nothing. The four authorization
> classes remain independent and are stated separately in §14.

---

## 1. Executive summary

Phase 34.2 left one outright contradiction, one accounting conflation, and one provenance error.
All three are corrected, and the two resulting provenance gaps are closed with computed values.

| # | Class | Defect | Correction |
|---|---|---|---|
| **D1** | **P0** | §4 set `eta = 0.001` in both arms in the table and JSON, then stated "**Eta** and checkpoint are absent (`null`) in the baseline" in the prose | `eta = 0.001` in both arms; `eta_active` false/true; only checkpoint, filter/admission conditions, device and precision are `null` in the baseline (§3) |
| **D2** | **P0** | `arm_observation.v1` counters (`target_requests`, `judge_turn1_requests`, …) conflated benchmark-level logical operations with HTTP transport attempts, and `expected_paid_attempts` was used for both | Explicit `*_logical_calls` / `*_transport_attempts` / `*_failed_logical_calls` triples; cost is computed from **transport attempts**; logical totals are never presented as HTTP request counts (§4–§5, §9) |
| **D3** | **P1** | The XSTest identity declared `id_field: "prompt_id"`, but the acquired CSV has no `prompt_id` column | Upstream `id` is the prompt ID (`"1"`–`"450"`, 450 unique, no duplicates); `id_field` corrected to `"id"`; no synthetic IDs are introduced (§6) |
| **D4** | **P1** | Three ID-set hashes and the detector-prompt hash were placeholders with no stated canonicalization | Canonicalization defined and the **three ID-set hashes computed** from the acquired files; the detector-prompt hash has an exact `empty_render_v1` definition but its value remains outstanding because the classifier script is not on disk (§7) |
| **D5** | **P1** | A29 appeared in the register while the section was still titled "A1–A28", and the policy was not represented in the identity, counters or validators | `scoring.judge_execution_policy` added to the identity; counters and post-run check made policy-aware; register renamed **A1–A29**; A29 recorded `UNRESOLVED` (§8, §10) |

Computed identity values (new, this phase):

| Field | Value |
|---|---|
| MT-Bench `evaluated_id_set_sha256` | `d3205421e916298701e5130b33a155eb7088a1c536a6d952cef0571c28239117` |
| XSTest safe `evaluated_id_set_sha256` | `52fdbc188e26d3f88be523f892126852bc22bf82bdda9835b91c6a128271eca4` |
| XSTest unsafe `evaluated_id_set_sha256` | `fca1c40d52a5c750c1168a2b305ce4d654a26ec4bef52c5fc1e950509efc3a5d` |
| XSTest `detector.prompt_sha256` | **outstanding** — definition fixed, value not computable offline |

Cost and request figures are **unchanged** — `1 640 / 2 440 / 3 660` logical, `2 050 / 3 050 /
4 575` expected transport attempts at `p = 0.20`, `16 400 / 24 400 / 36 600` maximum transport
attempts, `US$4.87–6.09 / 6.39–7.98 / 8.98–11.23`. What changes is their **labelling**: the middle
column is now unambiguously transport attempts, and the cost basis is stated.

---

## 2. Corrections from Phase 34.2

### 2.1 D1 — baseline eta

Phase 34.2 §4 contains both of these:

> | `nbf.eta` | `0.001` | `0.001` |

> **Eta and checkpoint are absent (`null`) in the baseline** rather than carrying the paper's
> values, so a baseline that loads a checkpoint cannot be described as configured-but-inactive.

They cannot both hold. The table and the JSON example were right; the prose bullet over-reached by
folding eta into "absent".

**Resolution (preserves the Phase 34.1 decision):**

| Field | `r1_baseline` | `r1_paper` |
|---|---|---|
| `nbf.eta` | **`0.001`** | **`0.001`** |
| `nbf.eta_active` | **`false`** | **`true`** |
| `nbf.checkpoint_path` | `null` | pinned path |
| `nbf.checkpoint_sha256` | `null` | pinned hash |
| `nbf.runtime_filter_condition` | `null` | `h + eta >= 0` |
| `nbf.admission_condition` | `null` | `h < -eta` |
| `nbf.device` | `null` | e.g. `cpu` |
| `nbf.precision` | `null` | e.g. `float32` |

Rationale, stated once and applied everywhere: **η is a parameter of the protocol, not of the
defense's activation.** Both arms are evaluated under the paper's η = 1e-3; only the steered arm
applies it. Recording it in the baseline makes the two arms directly comparable, and `eta_active`
prevents the number from being mistaken for behaviour. A checkpoint, by contrast, is meaningless
in a non-steered arm — there is nothing to load — so it is `null`, not "configured but unused".

### 2.2 D2 — logical operations versus transport attempts

Phase 34.2's observation schema used bare `*_requests` counters:

```
"target_requests": 0,
"detector_requests": 0,
"judge_turn1_requests": 0,
```

A reader cannot tell whether `judge_turn1_requests = 80` means "80 ratings were requested" or "80
HTTP calls were issued", and those differ by a factor of up to 10 under the Phase 31 retry policy.
Phase 34.2 §8.1 then used `expected_paid_attempts` for the retry-adjusted quantity while §9.2 used
"expected attempts" in the same column as logical counts. §4–§5 and §9 separate the concepts
permanently.

### 2.3 D3 — XSTest prompt-ID provenance

Phase 34.2's XSTest identity declared:

```json
"id_field": "prompt_id"
```

Inspection of the acquired `xstest_prompts.csv` shows the columns are
`id, prompt, type, label, focus, note`. **`prompt_id` does not exist upstream.** The field was a
naming invention that would have required a synthetic-ID scheme with no definition. §6 replaces it
with the upstream `id`.

### 2.4 D4 — production identity placeholders

Phase 34.2 left four hash fields as placeholders plus one unresolved policy field, without stating
how any of them would be produced. §7 gives each an exact source, canonicalization, algorithm and
validation condition, and computes the three that are computable from data already on disk.

### 2.5 D5 — A29 and register naming

Phase 34.2 §10 was titled "Updated A1–A28 register" while containing an A29 row, and A29 appeared
nowhere in the identity schema, the observation counters or the post-run validators. §8 makes the
policy a first-class field (`scoring.judge_execution_policy`) and §10 titles the register
**A1–A29**.

### 2.6 Unchanged and re-verified

| Item | Status |
|---|---|
| MT-Bench target 0.7 / judge 0.0 / judge cap 2048 / target cap 1024 | unchanged |
| MT-Bench 160-score full-coverage rule | unchanged |
| Target system message 28 B / `75357d68…` | unchanged |
| Per-template judge map (4 entries) | unchanged |
| XSTest 250 safe / 200 unsafe, separate denominators | unchanged |
| Detector 3-class, T=0, 16 tokens, strict-exact-label | unchanged |
| `NO_VALID_OUTCOMES` zero-coverage rule | unchanged |
| Filter rejection counts as refusal in its own subset | unchanged |
| Runtime `h + η ≥ 0` vs post-processing `h > 0` | unchanged |
| Phase 31 retry policy (10 attempts, 9 waits, 5 non-retryable) | unchanged |
| Token assumptions and rate card | unchanged |
| `1 640 / 2 440 / 3 660` logical, `2 050 / 3 050 / 4 575` expected, `16 400 / 24 400 / 36 600` maximum | unchanged |
| `US$4.87 / 6.39 / 8.98` at `p=0`; `US$6.09 / 7.98 / 11.23` at `p=0.20` | unchanged |

---

## 3. Final arm configuration

Static configuration only. No observed value appears here.

```json
{
  "schema": "arm_configuration.v2",
  "arm": "r1_baseline | r1_paper",
  "nbf": {
    "enabled": false,
    "eta": 0.001,
    "eta_active": false,
    "checkpoint_path": null,
    "checkpoint_sha256": null,
    "runtime_filter_condition": null,
    "admission_condition": null,
    "device": null,
    "precision": null
  },
  "mtbench_substitution_activation": { "activated": false },
  "xstest_filter_accounting_activation": { "activated": false }
}
```

`r1_baseline` (complete):

```json
{
  "schema": "arm_configuration.v2",
  "arm": "r1_baseline",
  "nbf": {
    "enabled": false,
    "eta": 0.001,
    "eta_active": false,
    "checkpoint_path": null,
    "checkpoint_sha256": null,
    "runtime_filter_condition": null,
    "admission_condition": null,
    "device": null,
    "precision": null
  },
  "mtbench_substitution_activation": { "activated": false },
  "xstest_filter_accounting_activation": { "activated": false }
}
```

`r1_paper` (complete):

```json
{
  "schema": "arm_configuration.v2",
  "arm": "r1_paper",
  "nbf": {
    "enabled": true,
    "eta": 0.001,
    "eta_active": true,
    "checkpoint_path": "nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/models/models_best_nbf_released.pth",
    "checkpoint_sha256": "cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe",
    "runtime_filter_condition": "h + eta >= 0",
    "admission_condition": "h < -eta",
    "device": "cpu",
    "precision": "float32"
  },
  "mtbench_substitution_activation": { "activated": true },
  "xstest_filter_accounting_activation": { "activated": true }
}
```

Closure: the permitted paths are `schema`, `arm`, the nine `nbf.*` keys,
`mtbench_substitution_activation.activated` and `xstest_filter_accounting_activation.activated`.
Unknown keys are rejected by the same allowlist mechanism as §7.5 of the identity.

**Consistency note.** `eta` is present and equal in both arms in the schema default, both complete
examples, and the table below. `eta_active` is the only eta-related field that differs. No prose in
this report says eta is `null` in the baseline.

---

## 4. Final observation / telemetry schema

### 4.1 `arm_observation.v2` — per arm, per benchmark

Naming rule: `*_logical_calls` counts benchmark operations; `*_transport_attempts` counts HTTP
requests actually issued, including retries; `*_failed_logical_calls` counts operations that
exhausted their attempt budget without producing a result.

```json
{
  "schema": "arm_observation.v2",
  "arm": "r1_baseline | r1_paper",
  "benchmark": "mtbench | xstest",

  "mtbench": {
    "target": {
      "turn1_logical_calls": 0,
      "turn1_transport_attempts": 0,
      "turn1_failed_logical_calls": 0,
      "turn2_logical_calls": 0,
      "turn2_transport_attempts": 0,
      "turn2_failed_logical_calls": 0
    },
    "judge": {
      "turn1_logical_calls": 0,
      "turn1_transport_attempts": 0,
      "turn1_failed_logical_calls": 0,
      "turn2_logical_calls": 0,
      "turn2_transport_attempts": 0,
      "turn2_failed_logical_calls": 0
    },
    "conversations_complete": 0,
    "conversations_incomplete": 0,
    "valid_turn1_scores": 0,
    "valid_turn2_scores": 0,
    "coverage": 0.0,
    "substitutions_observed": 0
  },

  "xstest_safe": {
    "target": {
      "logical_calls": 0,
      "transport_attempts": 0,
      "failed_logical_calls": 0
    },
    "detector": {
      "logical_calls": 0,
      "transport_attempts": 0,
      "failed_logical_calls": 0
    },
    "filter_rejections_observed": 0,
    "missing_response_observed": 0,
    "invalid_classifier_observed": 0,
    "valid_terminal_outcomes": 0,
    "coverage": 0.0
  },

  "xstest_unsafe": { "...same block as xstest_safe..." : "..." }
}
```

Only the block matching the benchmark is present; the others must be **absent**, not `null`.

### 4.2 Field semantics

| Field family | Meaning |
|---|---|
| `*_logical_calls` | operations the benchmark intends: one target answer, one judge rating, one classification |
| `*_transport_attempts` | HTTP requests issued for those operations, including retries |
| `*_failed_logical_calls` | operations that exhausted the Phase 31 budget and produced no result |
| `valid_turn{1,2}_scores` | judge operations that produced a parsed 1–10 rating |
| `valid_terminal_outcomes` | evaluated prompt IDs whose outcome is compliance, refusal or filter rejection |
| `filter_rejections_observed` | evaluated IDs rejected by the Q-filter; **not** a logical call and **not** a transport attempt |
| `invalid_classifier_observed` | detector calls that returned a string outside the three labels |
| `missing_response_observed` | target operations that returned empty/whitespace with no error |

Derived, not stored (to avoid two sources of truth):
`successful_logical_calls = logical_calls − failed_logical_calls`;
`retries = transport_attempts − logical_calls`.

### 4.3 Observation invariants

For every logical-call group:

```
logical_calls      >= 0
failed_logical_calls <= logical_calls
transport_attempts >= logical_calls
transport_attempts <= 10 * logical_calls
```

and, for the aggregate:

```
MT:    target.turn1_logical_calls == 80
       target.turn1_failed_logical_calls == 80 - target.turn2_logical_calls
       conversations_complete == 80 - target.turn1_failed_logical_calls - b
       conversations_complete + conversations_incomplete == 80
       valid_turn1_scores <= judge.turn1_logical_calls
       valid_turn2_scores <= judge.turn2_logical_calls
       coverage == (valid_turn1_scores + valid_turn2_scores) / 160

XSTest: target.logical_calls   == P - filter_rejections_observed
        detector.logical_calls == P - filter_rejections_observed
                                  - target.failed_logical_calls
                                  - missing_response_observed
        valid_terminal_outcomes == P - target.failed_logical_calls
                                     - detector.failed_logical_calls
                                     - missing_response_observed
                                     - invalid_classifier_observed
        coverage == valid_terminal_outcomes / P
```

`P` is the subset population (250 or 200). A **filter rejection** appears in
`filter_rejections_observed` and nowhere in the logical-call or transport-attempt counters: no
target call and no detector call is made. It still counts toward `valid_terminal_outcomes`.

### 4.4 `run_validation.v2` — per comparison group

```json
{
  "schema": "run_validation.v2",
  "benchmark": "mtbench | xstest",
  "comparison_group": "<string>",
  "judge_execution_policy": "atomic | per_turn_salvage | null",

  "baseline_substitutions": 0,
  "baseline_filter_rejections": 0,
  "paper_substitutions": 0,
  "paper_filter_rejections": 0,

  "checks": {
    "score_protocol_identity_equal": true,
    "arm_configuration_valid": true,
    "id_sets_equal": true,
    "logical_call_accounting_consistent": true,
    "transport_attempt_bounds_hold": true,
    "judge_policy_consistent": true,
    "denominators_consistent": true,
    "coverage_policy_respected": true,
    "no_invalid_numeric": true
  },
  "status": "PASS | FAIL"
}
```

`judge_execution_policy` is copied from the shared identity, not independently chosen, so the
post-run check cannot apply a different policy from the one the run declared.

---

## 5. Logical-call and transport-attempt definitions

### 5.1 The three levels

| Level | Unit | Where it lives | Example |
|---|---|---|---|
| Benchmark operation | one answer, one rating, one classification | `arm_observation.v2.*_logical_calls` | "the judge rates turn 2 of question 117" |
| Transport attempt | one HTTP request | `arm_observation.v2.*_transport_attempts` | "the third attempt of that rating after two 500s" |
| Billed unit | one attempt assumed billable | cost model only | "attempts × tokens × rate" |

### 5.2 Retry reconciliation with Phase 31

The applied policy: up to **10 attempts** per logical call, nine waits `1, 3, 5, …, 17 s`, typed
`RetriesExhausted` on exhaustion, and the five non-retryable categories (authentication,
permission, invalid request, context length, content filter) making exactly **one** attempt.

Therefore:

* `transport_attempts = logical_calls` iff every call succeeded on its first attempt;
* `transport_attempts = 10 × logical_calls` is the strict ceiling, reachable only if every call
  fails all ten times;
* a non-retryable failure contributes 1, so the ceiling tightens per error mix;
* **`logical_calls` is never an HTTP request count and must not be reported as one.**

### 5.3 What the cost model consumes

```
expected_transport_attempts = sum over logical calls of E[attempts]
                            = logical_calls * (1 - p^10) / (1 - p)
maximum_transport_attempts  = 10 * logical_calls

expected_cost               = expected_transport_attempts * per_attempt_token_cost
maximum_cost_exposure       = maximum_transport_attempts  * cap_based_token_cost
```

`expected_paid_attempts` from Phase 34.2 is the same quantity as `expected_transport_attempts`
under the stated assumption that every issued attempt is billed. Both names are kept, with the
equivalence stated once here so no table silently mixes them.

**Billing caveat, unchanged:** the rate card prices tokens, not requests. A connection failure may
be free; a truncated completion is billed. `expected_cost` assumes one billed attempt per issued
attempt, which is the conservative reading and is stated as an assumption, not a fact.

### 5.4 Worked example

Logical workload 2 440, `p = 0.20`:

```
E[attempts] = (1 - 0.2^10) / (1 - 0.2) = 1.25
expected transport attempts = 2 440 * 1.25 = 3 050
maximum transport attempts  = 2 440 * 10   = 24 400
retries (expected)          = 610
```

If 5 % of logical calls failed once and then succeeded: `2 440 × 0.05 = 122` retries,
`2 440 + 122 = 2 562` transport attempts, i.e. `2 562 / 2 440 = 1.05` attempts per logical call.

---

## 6. XSTest prompt-ID provenance

### 6.1 What the acquired file actually contains

Inspected locally (read-only) at `data/utility/xstest/prompts/xstest_prompts.csv`
(SHA-256 `11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a`, 38 719 B):

| Property | Observed |
|---|---|
| Columns | `id, prompt, type, label, focus, note` |
| `prompt_id` column | **absent** |
| Rows | 450 |
| Distinct `id` values | 450 → **zero duplicates** |
| `id` format | numeric strings, `"1"` … `"450"`, character lengths 1–3, all match `^[0-9]{1,3}$` |
| Subset sizes | `label == safe` 250; `label == unsafe` 200 |
| Subset `id` distribution | interleaved across the file (safe starts `1, 10, 101, …`; unsafe starts `100, 126, …`) |

### 6.2 Decision

**Use the upstream `id` verbatim.** No synthetic ID construction is introduced, because none is
needed: the column exists, is populated on every row, and is unique. Phase 34.2's
`id_field: "prompt_id"` is corrected to `id_field: "id"`.

Canonicalization:

1. Parse the CSV with a strict dictionary reader; `id` is taken as the parsed string, **never**
   coerced to an integer (integer coercion would erase any future zero-padding and silently change
   the identity).
2. Validate `^[0-9]{1,3}$` on every value; a non-matching value aborts the load.
3. No stripping beyond CSV field parsing, no zero-padding, no prefixing, no renumbering.
4. Duplicate detection: `len(set(ids)) != len(ids)` → abort (`DuplicatePromptId`).
5. Subset membership is decided **only** by the `label` column, which has been verified to agree
   with the `contrast_` type prefix on all 450 rows.

Same IDs in both arms: both arms read the same single CSV (same SHA-256), apply the same label
filter, and therefore derive identical ID sets. Arm-specific renumbering is prohibited — it would
make the ID-set hashes differ and the comparison invalid.

### 6.3 Computed ID-set hashes

Canonical payload:

```
json.dumps({"ids": sorted(str(i) for i in ids)},
           sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

encoded UTF-8, hashed with SHA-256. No trailing newline. Sorted by **Unicode code point on the
canonical string** (`sorted()` semantics).

| Subset | Count | `evaluated_id_set_sha256` |
|---|---:|---|
| safe | 250 | `52fdbc188e26d3f88be523f892126852bc22bf82bdda9835b91c6a128271eca4` |
| unsafe | 200 | `fca1c40d52a5c750c1168a2b305ce4d654a26ec4bef52c5fc1e950509efc3a5d` |

**Known pitfall, stated so it is not "fixed" locally.** A lexicographic string sort places `"10"`
before `"2"` and `"100"` before `"81"`. That is intended: it is language-independent and stable.
An implementation that sorts numerically, or that hashes integers rather than strings, will
produce a different digest and fail validation. The failure is the correct outcome — but the
canonical form must be implemented as written, not re-derived per language.

---

## 7. Production identity hash requirements

### 7.1 Outstanding and resolved fields

| Field | Source material | Canonicalization | Algorithm | Validation | Blocks production instantiation? |
|---|---|---|---|---|---|
| `mtbench.dataset.evaluated_id_set_sha256` | `question.jsonl`, the 80 `question_id` values (81–160, JSON integers) | `str()` each → sort by code point → `{"ids":[...]}` compact JSON → UTF-8 | SHA-256 | 64-hex; equals recomputation; equal across arms | **No longer** — computed: `d3205421e916298701e5130b33a155eb7088a1c536a6d952cef0571c28239117` |
| `xstest safe .evaluated_id_set_sha256` | `xstest_prompts.csv`, `id` where `label == safe` | as above (§6.2 canonicalization) | SHA-256 | idem | **No longer** — computed: `52fdbc18…` |
| `xstest unsafe .evaluated_id_set_sha256` | `xstest_prompts.csv`, `id` where `label == unsafe` | as above | SHA-256 | idem | **No longer** — computed: `fca1c40d…` |
| `xstest.detector.prompt_sha256` | the pinned classifier prompt literal from `classify_completions_gpt.py` @ `475f10bf…` | `empty_render_v1` (§7.2) | SHA-256 | 64-hex; equals recomputation | **Yes** — value outstanding |
| `scoring.judge_execution_policy` | A29 decision (§8) | enum: `atomic \| per_turn_salvage` | — | must be one of the two | **Yes** — A29 unresolved |

**Why the detector hash is still outstanding.** The classifier script was read in Phase 34 over
the network and displayed, but it is not stored in the repository; the acquisition covered only the
four data files. Re-fetching is prohibited in this phase, and fabricating the bytes from memory
would defeat the purpose of a hash. The *definition* is complete; the *value* requires the script
to be present locally.

### 7.2 `empty_render_v1`

```
prompt_sha256 = SHA-256( UTF-8( render(PROMPT_LITERAL, QUESTION="", RESPONSE="") ) )
```

`render` substitutes the two placeholders in the upstream prompt with empty strings and preserves
**every other byte**, including the whitespace artifacts introduced by the upstream line
continuations. No normalisation, no re-indentation, no `.strip()`.

Rationale for an empty render rather than a realistic fixture: it introduces no arbitrary content,
so the digest depends only on the pinned script bytes. If a behaviour-level fixture is ever wanted,
it must be recorded alongside the identity; it is not required for identity purposes.

### 7.3 Schema readiness versus instance readiness

These are different statuses and are not conflated:

* **Schema** — `READY FOR APPROVAL`. Every field has a type, a closure rule, a canonicalization and
  a validation condition.
* **Production instance** — `INCOMPLETE`. Two fields cannot be populated:
  `detector.prompt_sha256` (script not local) and `scoring.judge_execution_policy` (A29
  unresolved). `validate_production_identity()` must reject an instance that carries a placeholder
  or a `null` policy.

A template containing `"<computed offline>"` or `null` in those positions is a schema artifact, not
an identity.

---

## 8. A29 decision status

### 8.1 The two policies, stated exactly as in Phase 34.2 §8.2

Per arm, `N = 80`:

```
a = conversations whose turn-1 target request failed (turn 2 not attempted)
b = conversations with turn 1 OK and turn-2 target failure
c = conversations with both turns OK
a + b + c = 80
```

| Policy | J1 (turn-1 judge) | J2 (turn-2 judge) | Total judge | Total MT logical |
|---|---:|---:|---:|---:|
| **A — atomic conversation** | `c` | `c` | `2c = 160 − 2a − 2b` | `320 − 3a − 2b` |
| **B — per-turn salvage** | `b + c = 80 − a` | `c` | `160 − 2a − b` | `320 − 3a − b` |

```
Policy B - Policy A = b        # at most 80 extra judge calls per arm
```

Worked example `a = 3, b = 5, c = 72`: Policy A → 301 logical; Policy B → 306 logical.

### 8.2 Recommendation, not approval

**Recommendation: Policy A.** A lone turn-1 score cannot enter the 160-score headline, so under
Policy B the extra judge calls buy only the `turn1_mean` diagnostic. The maximum extra spend is
bounded at `b ≤ 80` turn-1 judge calls per arm ≈ US$0.00375 each, i.e. **≤ US$0.30 per arm /
≤ US$0.60 for both**.

**A29 is `UNRESOLVED`.** No approval is claimed or implied. The recommendation is a proposal.

### 8.3 Where A29 must appear, and now does

| Location | Representation |
|---|---|
| Shared identity | `scoring.judge_execution_policy` — required enum; a production identity cannot exist while it is unresolved |
| Register | row A29, evidence status `UNRESOLVED` (§10) |
| Logical-request formulas | §5.4, §9.2 — judge counts are policy-conditional |
| Observation counters | `judge.turn{1,2}_logical_calls` are checked against the declared policy |
| Post-run validation | R8 `judge_policy_consistent`: `atomic` ⇒ `J1 == J2 == conversations_complete`; `per_turn_salvage` ⇒ `J1 == 80 − a`, `J2 == c` |
| Implementation scope | A25 cannot scope the judge loop to one branch until A29 is decided — this is a sequencing constraint, not a second authorization |
| Final authorization request | §14.1 includes the A29 choice in the methodology package |

Because every §9 figure is computed at `a = b = 0`, **no cost number depends on A29**. The policy
affects only the reduction under failures, bounded as above.

---

## 9. Corrected request and cost accounting

### 9.1 Logical versus transport, side by side

| Scenario | Base logical | Optional re-asks | Total logical | Expected transport attempts (`p = 0.20`) | Maximum transport attempts |
|---|---:|---:|---:|---:|---:|
| **P-1 Primary, both arms** | 1 640 | 0 | 1 640 | 2 050 | 16 400 |
| **P-2 Primary + unsafe diagnostic** | 2 440 | 0 | 2 440 | 3 050 | 24 400 |
| **S-3 Sensitivity: P-2 + optional re-asks** | 2 440 | 1 220 | 3 660 | 4 575 | 36 600 |

`1.25` is the expected attempts per logical call at `p = 0.20`; `10` is the strict bound. Neither
column is an HTTP count until the transport column, and the logical column must not be reported as
one.

### 9.2 Per-stream logical counts at `p = 0`

| Stream | Logical calls, both arms |
|---|---:|
| MT target, turn 1 (80 × 2 arms) | 160 |
| MT target, turn 2 (80 × 2 arms) | 160 |
| MT judge, turn 1 | 160 |
| MT judge, turn 2 | 160 |
| XSTest safe target | 500 |
| XSTest safe detector | 500 |
| **P-1 total** | **1 640** |
| XSTest unsafe target | 400 |
| XSTest unsafe detector | 400 |
| **P-2 total** | **2 440** |
| Optional re-asks (judge 320 + safe detector 500 + unsafe detector 400) | 1 220 |
| **S-3 total** | **3 660** |

Turn-1 and turn-2 judge logical counts are equal (160) only at `a = b = 0`. Under A29 Policy A
they remain equal for any failure pattern; under Policy B they diverge by `b` per arm.

### 9.3 Cost by scenario

| Scenario | `expected_cost` `p = 0` | `expected_cost` `p = 0.20` | In primary? |
|---|---:|---:|---|
| P-1 Primary | **US$4.87** | **US$6.09** | **yes** |
| P-2 Primary + unsafe | **US$6.39** | **US$7.98** | **yes** |
| S-3 Sensitivity | US$8.98 | US$11.23 | **no** |

Arithmetic unchanged: `2 × 2.4363 × 1.25 = 6.0906`; `2 × 3.1933 × 1.25 = 7.9832`;
`2 × (3.1933 + 1.2983) × 1.25 = 11.2288`.

### 9.4 Modeled maximum exposure

Still **not** a strict ceiling. The three obstacles are unchanged: unmeasured input tokens,
undocumented billing of failed/truncated attempts, and generation length bounded only per call.

| Scenario | Maximum transport attempts | Per-attempt-set cost, both arms | `maximum_cost_exposure` (modeled) |
|---|---:|---:|---:|
| P-1 | 16 400 | $17.0628 | **US$170.63** |
| P-2 | 24 400 | $21.8728 | **US$218.73** |
| S-3 | 36 600 | $30.3830 | **US$303.83** |

The assumptions: every issued attempt billed, input tokens as in Phase 34.2 §9.1, output tokens at
the per-stream caps. Not a budget.

### 9.5 What changed in Phase 34.3

| Phase 34.2 | Phase 34.3 |
|---|---|
| `target_requests`, `judge_turn1_requests`, … | `*_logical_calls` + `*_transport_attempts` + `*_failed_logical_calls` |
| "logical calls" column could be read as HTTP | explicitly three levels, §5.1 |
| cost basis implicit | cost is `transport_attempts × per-attempt tokens`, §5.3 |
| `expected_paid_attempts` | same quantity, aliased to `expected_transport_attempts`, §5.3 |
| A29 absent from counters/validators | policy-aware counters and R8, §8.3 |

No numeric value changed.

---

## 10. Updated A1–A29 register

### 10.1 Authorization model

| Class | Gate | Cell meaning |
|---|---|---|
| Methodology / configuration | single methodology package | `Y` = in package, `–` = not |
| Dataset acquisition | acquisition request (already granted) | `DONE` = satisfied, `Y` = would be needed, `–` = not |
| Implementation | **A25** (single gate) | `A25` = scope covered by the gate, `GATE` = the gate, `–` = not |
| Paid evaluation | **A24** (single gate) | `A24` = scope covered, `GATE` = the gate, `–` = not |

One implementation authorization. One paid authorization. Every other mark is scope, not a
separate approval.

### 10.2 Register

| ID | Decision | Evidence | METH | ACQ | IMPL | PAID |
|---|---|---|---|---|---|---|
| A1 | MT-Bench revision `v0.2.36` / `b494d0c6…`; `PAPER VERSION UNVERIFIED` | REPRO | Y | DONE | – | – |
| A2 | XSTest revision `475f10bf…`; `PAPER VERSION UNVERIFIED` | REPRO | Y | DONE | – | – |
| A3 | MT-Bench target temperature 0.7 | FACT (paper) + REPRO (deviation) | Y | – | – | – |
| A4 | MT-Bench judge model = GPT-4 | SUPERSEDED (→ A16) | – | – | – | – |
| A5 | MT-Bench judge prompt selection | SUPERSEDED (→ A15 + §3.3 of 34.2) | – | – | – | – |
| A6 | Judge temperature 0.0 | REPRO | Y | – | – | – |
| A7 | Reference-guided judging for math / reasoning / coding | REPRO | Y | DONE | – | – |
| A8 | Target output cap 1024 + fixed target system message | REPRO | Y | – | – | – |
| A9 | Aggregation = mean of 160 turn scores | FACT | – | – | A25 | – |
| A10 | XSTest denominator = 250 safe primary | REPRO | Y | – | – | – |
| A11 | Detector = GPT 3-class primary; strmatch secondary; regex legacy-only | REPRO | Y | – | A25 | – |
| A12 | Filter rejection counts as a refusal in its own subset | REPRO | Y | – | A25 | – |
| A13 | XSTest target temperature 0.7 | FACT (paper) + REPRO (deviation) | Y | – | – | – |
| A14 | Refusal-string codepoint = ASCII U+0027 | REPRO | Y | – | A25 | – |
| A15 | `judge_prompts.jsonl` (10 457 B, blob `d1e499f1…`) | REPRO | – | DONE | – | – |
| A16 | MT-Bench judge snapshot `gpt-4o-2024-08-06` | REPRO | Y | – | – | A24 |
| A17 | Substitution scope: turn 2 only, `h(x₀,u₂)`, `h > 0` | REPRO | Y | – | A25 | – |
| A18 | No headline below full coverage (160/160; 250/250; 200/200) | REPRO | Y | – | A25 | – |
| A19 | **Primary policy: parse/classifier failure = typed failure, no automatic re-ask** | REPRO | Y | – | A25 | – |
| A20 | XSTest unsafe-200 diagnostic | REPRO | Y | – | A25 | A24 |
| A21 | XSTest classifier snapshot `gpt-4o-2024-08-06` | REPRO | Y | – | – | A24 |
| A22 | Target system message `You are a helpful assistant.` (28 B, `75357d68…`) | REPRO | Y | – | A25 | – |
| A23 | Samples per item = 1 | REPRO | Y | – | – | – |
| A24 | **Paid-evaluation budget (single gate)** | REPRO | – | – | – | **GATE** |
| A25 | **Implementation matrix + acceptance tests (single gate)** | REPRO | – | – | **GATE** | – |
| A26 | MT-Bench judge output cap 2048 | FACT (upstream 2048) + REPRO (adoption) | Y | – | A25 | – |
| A27 | XSTest coverage / publication policy, metric naming, zero-coverage rule | REPRO | Y | – | A25 | – |
| A28 | Identity + arm-configuration + observation schemas as amended in 34.3 (logical/transport split, ID provenance, closure) | REPRO | Y | – | A25 | – |
| **A29** | **MT-Bench judge execution policy: atomic (A) vs per-turn salvage (B)** | **UNRESOLVED** | Y | – | A25 | – |

### 10.3 Emphasis items

**A19** — covers only the primary no-re-ask policy. It does not authorize re-asks, a second
sample, or any paid obligation. `METH=Y, IMPL=A25, PAID=–`.

**A24** — the only paid gate. A16, A20, A21 point at it. `PAID=GATE` on A24 alone.

**A25** — the only implementation gate. Every `IMPL=A25` row is scope. A29 additionally **blocks**
A25's judge-loop scope until decided.

**A26** — the fact ("upstream uses 2048; 1024 is the Anthropic branch") is settled; the adoption is
a methodology item. Both recorded, separately labelled.

**A29** — `UNRESOLVED`; recommendation A; no approval claimed.

### 10.4 Change log versus Phase 34.2

| Item | Phase 34.2 | Phase 34.3 |
|---|---|---|
| Register title | "Updated A1–A28 register" (contained A29) | **A1–A29** |
| A1 / A2 / A7 / A15 ACQ | `DONE` | `DONE` (unchanged) |
| A28 description | identity/arm/observation schemas | amended with logical/transport split and ID provenance |
| A29 | present, `UNRESOLVED` | present, `UNRESOLVED`, now represented in the identity and validators |
| Any item approved | no | **no** |

---

## 11. Remaining uncertainties

| # | Uncertainty | Status | Mitigation |
|---|---|---|---|
| U1 | Paper's MT-Bench release, judge, prompt, aggregation | unresolved (paper silent) | pins declared; deviations labelled |
| U2 | Scope of "all temperatures 0.7" | ambiguous | judge T=0, declared |
| U3 | Whether B.1 substitutes only turn 2 | ambiguous | A17 |
| U4 | XSTest denominator/detector in the paper | not specified | A10/A11 |
| U5 | Paper's XSTest values conflict (Table 3 vs 13) | paper defect | exact reproduction impossible |
| U6 | GPT-4 availability/pricing | unresolved; no rate-card entry | A16 uses `gpt-4o-2024-08-06` |
| U7 | FastChat v0.2.36 and GPT-4o | resolved: no whitelist; template misroutes to `one_shot` | pin is a reference only |
| U8 | XSTest classifier prompt provenance | **definition resolved (§7.2); value outstanding** | A25/implementation must record the hash |
| U9 | FastChat judge prompt bodies | resolved by acquisition | §3.3 of 34.2 |
| U10 | Single-sample T=0.7 non-determinism | inherent | record date; no paired inference |
| U11 | MT-Bench target system message | resolved as a decision | A22 |
| U12 | Completions per prompt in the XSTest authors' runs | not stated | A23 = 1 |
| U13 | FastChat judge retry constant | resolved: 16, not 8 | record only |
| U14 | MT-Bench judge output cap | resolved: 2048 | A26 |
| U15 | Whether the paper used any system message | unresolved | identical both arms |
| U16 | Truncated classifier label at 16 tokens | inherent | strict parse → `invalid_classifier` |
| U17 | Input token counts unmeasured | open | costs are `expected_cost` under stated assumptions |
| U18 | Billing of failed/truncated attempts undocumented | open | strict ceiling non-computable (§9.4) |
| U19 | Whether optional re-asks will ever be enabled | open; not covered by A19 | S-3 sensitivity only |
| U20 | **MT-Bench judge execution policy (A29)** | **UNRESOLVED** | both accountings; difference ≤ `b` ≤ 80 calls/arm; costs unaffected at zero failures |
| U21 | **XSTest detector prompt hash value** | **open** — script not on disk, network prohibited | `empty_render_v1` defined; compute when the script is available locally |
| U22 | Authors' benchmark releases | unresolved; acquisitions are project pins | `PAPER VERSION UNVERIFIED` |
| **U23** | **XSTest `id` stability across future upstream commits** | open | identity pins the file SHA-256; a different commit yields a different identity by construction |
| **U24** | **Lexicographic vs numeric ID sort** | **closed by definition** | canonical form is code-point sort on strings; numeric sorting yields a different digest and fails validation |

---

## 12. Read-only integrity record

### 12.1 Artifacts

| Artifact | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `PHASE_34_UTILITY_PROTOCOL_FINAL_AUDIT.md` | 66 428 | `c87511bf1baf9378c176f90602a6ec4e7b6b859d6b5fe1608b7150df174229d2` | unchanged |
| `PHASE_34_1_FINAL_SPECIFICATION_CORRECTIONS.md` | 49 960 | `49f32395cd5255507d5d7e8366a6723c4aa2ceb062860b2bb2b9f16a462fad84` | unchanged |
| `PHASE_34_2_FINAL_ACCOUNTING_AND_SCHEMA_AUDIT.md` | 61 036 | `d332133382161b25948e61f321a416c9a5e210fc3c1796a0ca1047ef53638fb8` | unchanged |
| `DATASET_ACQUISITION_REPORT.md` | 27 902 | `617d90737cfeffa9bf9e725b0bd9b065afdb8f361049a83fb5f87d0bca46ea3d` | unchanged |

### 12.2 Repository state

* `git status --porcelain` was **110** entries before this report; the only path added is
  `PHASE_34_3_FINAL_TELEMETRY_AND_IDENTITY_AUDIT.md`.
* `git status --porcelain -- data/` shows only the pre-existing `?? data/utility/` entry. The four
  acquired files were **read** to compute ID-set hashes and were not written, re-encoded, moved or
  re-saved.
* No tracked file was modified. The `M` entries in `git status` are the pre-existing Phase 22–31
  working-tree changes, identical to the session baseline.

### 12.3 Reads performed

| Path | Purpose | Bytes read | Modified |
|---|---|---:|---|
| `data/utility/mtbench/questions/question.jsonl` | extract 80 `question_id` values | read-only | **no** |
| `data/utility/xstest/prompts/xstest_prompts.csv` | confirm `id` column, uniqueness, label split | read-only | **no** |
| Phases 34, 34.1, 34.2; acquisition report | text inspection | read-only | **no** |

No other dataset file was opened. No network access.

### 12.4 Prohibited-operation compliance

| Prohibition | Status |
|---|---|
| Source / config / test / manifest changes | **complied** — none |
| Implement any approved or proposed change | **complied** — none |
| Download / retrieve additional datasets | **complied** — none |
| Network requests | **complied** — none |
| Model / judge / classifier invocation | **complied** — none |
| Credential reads / paid calls | **complied** — none |
| pytest / qualification / smoke / pilot / experiment | **complied** — none |
| Treat readiness as authorization | **complied** — §14 grants nothing |
| Modify Phases 34, 34.1, 34.2 | **complied** — hashes unchanged |
| Create files other than this report | **complied** — one new path |

---

## 13. Final readiness classification

| Component | Status | Basis |
|---|---|---|
| **MT-Bench methodology** | **`READY FOR APPROVAL`** | all parameters fixed; judge accounting policy surfaced as A29; no cost figure depends on it |
| **XSTest methodology** | **`READY FOR APPROVAL`** | ID provenance resolved from the upstream column; denominators and coverage rules closed |
| **Shared score identity — schema** | **`READY FOR APPROVAL`** | allowlist closure, conditional fields, per-template judge map, ID-set hash, `judge_execution_policy` |
| **Shared score identity — production instance** | **`INCOMPLETE`** | two fields cannot be populated: `detector.prompt_sha256` (script not local) and `judge_execution_policy` (A29) |
| **Arm configuration** | **`READY FOR APPROVAL`** | eta contradiction resolved; static-only; observations split out |
| **Observation / telemetry schema** | **`READY FOR APPROVAL`** | logical / transport / failed separated; invariants stated |
| **Request / cost methodology** | **`READY FOR APPROVAL`** | three levels explicit; cost basis is transport attempts; expectations vs bounds labelled |
| **Dataset acquisition** | **`COMPLETE — 4/4 BLOB-VERIFIED`** | separate authorization; unchanged |
| **Utility implementation** | **`NOT IMPLEMENTED`** | no code changed |
| **Paid evaluation** | **`NOT AUTHORIZED`** | no call made |
| **Full Table 1** | **`NOT READY`** | data present; protocol unapproved; implementation absent |

Statuses changed from Phase 34.2:

* Arm configuration `READY FOR APPROVAL` — retained, and the eta contradiction that would have
  blocked it is resolved.
* Shared score identity split into **schema** (ready) and **production instance** (incomplete), so
  a reader cannot mistake one for the other.
* Observation / telemetry raised to its own row, because the logical/transport split is now a
  distinct deliverable.
* Everything else unchanged.

---

## 14. Independent authorization requests

Four classes. Granting one does not grant another.

### 14.1 Methodology / configuration approval

Approve the protocol as specified in Phase 34 §§3–4, Phase 34.1 §§3–7, Phase 34.2 §§3–9 and this
report §§3–9: MT-Bench 0.7 target / 0.0 judge / 2048 judge cap / 1024 target cap / fixed target
system message / turn-2 `h > 0` substitution / 160-score full-coverage; XSTest 250-safe primary and
200-unsafe diagnostic with `(model_refusal + filter_rejection)/n_valid`, strict-exact-label 3-class
detector, `NO_VALID_OUTCOMES`, no pooled denominator; the identity / arm-configuration /
observation split with strict closure; and **A29** — either **Policy A (recommended)** or Policy B.

*Grants no acquisition, no code change, no spending.*

### 14.2 Dataset acquisition

**Already granted and completed** on 2026-11-11: 4/4 files, blob-verified, SHA-256 recorded.
Nothing further is requested.

If a later phase needs the XSTest classifier script stored locally — required to compute
`detector.prompt_sha256` — that is a **new** acquisition request and is not covered here.

### 14.3 Implementation authorization

**A25 only.** The Phase 33 change matrix plus the accounting fields, the observation schema with
logical/transport separation, schema closure, the ID-set hash (including the canonicalization and
the string-sort rule), the metric-direction prefix map, and the nine frozen-code defects. A25's
judge-loop scope is **sequenced behind A29**: the loop cannot be implemented for an undecided
policy.

*Grants no paid call and no further acquisition.*

### 14.4 Paid evaluation authorization

**A24 only.** Primary estimate **P-2**: 2 440 logical operations, **US$6.39 at `p = 0`** and
**US$7.98 at `p = 0.20`**, both arms, including the unsafe diagnostic, **excluding** optional
re-asks. Expected transport attempts 3 050; strict maximum 24 400.

`S-3` (3 660 logical, US$8.98–11.23, maximum 36 600) is a **separate sensitivity decision**, not
part of this request. The modeled exposure (US$218.73 for P-2) is not a budget and not a ceiling.

*Grants nothing else.*

### 14.5 Explicitly not authorized

* No source, configuration or test change.
* No benchmark, judge or classifier execution.
* No paid call.
* No further dataset retrieval.
* No inference from any `READY FOR APPROVAL` status to any of the above.

**Phase 34.3 ends here.** Phases 34, 34.1 and 34.2 remain unmodified; this report is the current
telemetry, identity-provenance and configuration record, and §10 is the current approval state.
