# PHASE 34.2 — FINAL REQUEST ACCOUNTING, SCHEMA CLOSURE & AUTHORIZATION CONSISTENCY

**Correction successor to `PHASE_34_1_FINAL_SPECIFICATION_CORRECTIONS.md`.**
Phase 34 and Phase 34.1 are preserved unchanged. This report supersedes them wherever the three
disagree; it does not repeal any protocol decision that survives the corrections below.

| Component | Status |
|---|---|
| MT-Bench methodology | **`READY FOR APPROVAL`** |
| XSTest methodology | **`READY FOR APPROVAL`** |
| Shared score identity | **`READY FOR APPROVAL`** |
| Arm configuration | **`READY FOR APPROVAL`** |
| Request/cost methodology | **`READY FOR APPROVAL`** |
| Dataset acquisition | **`COMPLETE — 4/4 BLOB-VERIFIED`** (separately authorized, 2026-11-11) |
| Utility implementation | **`NOT IMPLEMENTED`** |
| Paid evaluation | **`NOT AUTHORIZED`** |
| Full Table 1 | **`NOT READY`** |

Read-only specification pass. **No source code, configuration, test, manifest or historical
artifact was modified. No network request was made. No dataset was downloaded, cloned, copied or
created. No model, judge or classifier was invoked. No credential was read, no paid call was made,
no pytest / qualification / smoke test / pilot / experiment was run.** Exactly one file was
created: this report.

> `READY FOR APPROVAL` means the specification is complete enough for a human decision. It
> authorizes nothing. The four authorization classes are independent and are enumerated in §14.

---

## 1. Executive summary

Phase 34.1 left five specifiable inconsistencies. All five are corrected here; none of them changes
a benchmark protocol decision.

| # | Class | Defect | Correction |
|---|---|---|---|
| **C1** | **P0** | §6.1 defined `logical_request_count` as excluding re-asks, while §6.4 put 3,660 (which includes 1,220 re-asks) in the same column | Seven distinct accounting fields; `total = base + optional_reask`; 3,660 is never called a base count (§8) |
| **C2** | **P0** | `MT judge calls = 2 × (80 − d)` silently assumed any failed turn suppresses both judge calls | Explicit execution sequence, per-turn judge-eligible counts, and **two** accounting policies with the difference bounded at `b` ≤ 80 calls per arm (§8.2) |
| **C3** | **P1** | `arm_configuration.v1` carried `expected_substitutions` / `expected_filter_rejections`, which are runtime observations | Static config vs `arm_observation.v1` + `run_validation.v1`; pre-run and post-run contracts separated (§4–§6) |
| **C4** | **P1** | "Closed field set" was asserted but never enforced; a blacklist of NBF names would miss future defense fields | Strict per-level allowlist, `additionalProperties: false`, conditional `judge`/`detector`, absent ≠ null, placeholder rejection (§3) |
| **C5** | **P1** | `evaluated_id_count <= subset_row_count` contradicted the "truncated subset aborts" rule; a count alone cannot prove ID-set equality | Exact equality for validated full-population runs; canonical ID-set hash added to the identity (§7) |

Also resolved: the implementation-authorization model (**A25 is the single gate**; every other
`IMPL` mark is a scope dependency, not an approval — §10), and the A19 re-ask separation (the
primary policy is typed failure with **no** automatic re-ask; optional re-asks are a sensitivity
scenario requiring a separate decision — §8.4).

Corrected accounting headline:

| Scenario | Base logical | Optional re-asks | Total logical | Expected attempts (`p=0.20`) | Maximum attempts | Expected cost (`p=0.20`) |
|---|---:|---:|---:|---:|---:|---:|
| P-1 Primary | 1 640 | 0 | 1 640 | 2 050 | 16 400 | **US$6.09** |
| P-2 + unsafe diagnostic | 2 440 | 0 | 2 440 | 3 050 | 24 400 | **US$7.98** |
| S-3 sensitivity (re-asks) | 2 440 | 1 220 | 3 660 | 4 575 | 36 600 | US$11.23 |

`S-3` is **not** part of the primary workload or the primary paid estimate. `1.25` is an
expectation at `p = 0.20`, never a bound; the strict attempt bound is `10 × total`.

Two post-acquisition reconciliations are folded in, because the datasets now exist and the
identity schema must not pretend otherwise:

* the judge system message is **per template** (28 / 613 / 28 / 640 bytes), so the identity's
  single `judge.system_message_sha256` becomes a four-entry map (§3.3);
* `reference_answer/gpt-4.jsonl` covers **30 of 80** questions (math, reasoning, coding), which
  confirms the category-conditional reference rule (§7.4).

Nothing is approved. §14 states the four authorizations separately.

---

## 2. Phase 34.1 findings and corrections

### 2.1 What Phase 34.1 said, verbatim, and why it is wrong

**C1 — the logical-count definition contradicts its own table.**

Phase 34.1 §6.1:

> `logical_request_count` — calls the protocol intends to make, **before** retries; excludes re-asks

Phase 34.1 §6.2 then labels the row *"Primary + unsafe + all re-asks"* with **3 660**, and §6.4
uses 3 660 as the "logical calls" column. One field cannot both exclude and include 1 220 re-asks.
A reader cannot tell whether 3 660 is a base workload or a sensitivity total, and a budget built
on it would silently assume re-asks are enabled.

**C2 — the judge formula assumes an unapproved policy.**

Phase 34.1 §6.2:

> `MT judge calls = 2 × (80 − d)` where `d` = conversations with at least one failed turn

This is correct **if and only if** a conversation with one failed turn makes no judge calls at all.
Phase 34 §3.1 says a failed turn makes the conversation "unscored", which can mean "cannot
contribute to the headline" — a statement about publication, not about request emission. The
distinction matters: under a per-turn salvage policy, a valid turn-1 answer in a turn-2-failed
conversation would still be judged. Phase 34.1 presented one policy's arithmetic as if it were
derived. §8.2 removes the assumption by making the policy explicit and showing both accountings.

**C3 — configuration carried observations.**

Phase 34.1 §4 lists `mtbench_substitution_activation.expected_substitutions` and
`xstest_filter_accounting.expected_filter_rejections`, then §5.2 treats "observed" values as the
post-run evidence. The same field was doing two jobs, so a configuration document could not be
validated before a run without knowing the run's outcome — exactly backwards.

**C4 — "closed" was asserted, not enforced.**

Phase 34.1 §3 said the shared identity "contains no NBF field", but supplied no mechanism. The
obvious implementation — reject known NBF key names — is a blacklist, and a defense field named
`defense_margin` or `safety_gate` would pass straight through. Closure requires an **allowlist**.

**C5 — the ID invariant contradicted the dataset rule.**

Phase 34.1 §8 permits `evaluated_row_count <= subset_row_count` while also requiring a truncated
source subset to abort. Both cannot hold: if the loader aborts on a short subset, the only
surviving case is equality. And a count alone cannot establish that both arms saw the same
examples — two arms could each evaluate 250 IDs with one differing.

### 2.2 What Phase 34.1 got right (unchanged)

| Item | Status |
|---|---|
| Seven-label accounting vocabulary (`logical` … `maximum_cost_exposure`) | **adopted**, with `logical` split into base/optional/total |
| Expectation vs strict maximum distinction (`1.25` vs `10×`) | **adopted** |
| Strict monetary ceiling declared non-computable | **adopted** |
| Zero-coverage `NO_VALID_OUTCOMES` policy | **adopted** |
| Row-count field split (`source_`, `evaluated_`, `subset_`, `expected_units`) | **adopted**, with the ID-set hash added |
| Two-assertion model (shared identity equal / arm config validated) | **adopted**, with schemas closed |
| All benchmark protocol decisions (0.7 / 0 / 2048 / 1024, 160-score rule, 250/200, filter accounting, `h > 0` vs `h + η ≥ 0`) | **unchanged** |

---

## 3. Final shared identity schema

### 3.1 Closure mechanism

The schema is closed by an **exact allowed-path allowlist**, checked at every object level.
Implementation contract:

1. A declarative table enumerates every permitted path, its type, and whether it is required.
2. The validator walks the document; any key not in the table raises
   `UnknownIdentityField(path=...)`. This is not a blacklist: an unrecognised defense field of any
   name is rejected by default.
3. `additionalProperties: false` is set on every object in the published JSON Schema as a
   second, independent expression of the same rule. The path validator is authoritative; the JSON
   Schema exists so external tools agree.
4. Array elements are also closed: `dataset.files[]` permits exactly
   `role, path, git_blob_sha1, sha256, bytes`.
5. Defense-shaped keys are additionally reported with a clearer message
   (`DefenseFieldInSharedIdentity`), but this is a **diagnostic aid only** — rejection comes from
   the allowlist, not from the name.
6. Absent ≠ null. `judge` must be **absent** for XSTest (not `null`); `stop` may be `null`
   meaning "no stop sequence"; the validator distinguishes `key not present` from
   `key present with value null` and reports which.
7. Conditional fields: for `benchmark == "mtbench"`, `judge` is required and `detector` must be
   absent; for `benchmark == "xstest"`, `detector` is required and `judge` must be absent. A
   present-but-null conditional field is `ConditionalFieldError`, not a schema variant.
8. Placeholders (`"<pending A15>"`, any value beginning `<`) are legal **only** in the template.
   `validate_production_identity()` raises `ProvenanceIncomplete(path)` for any unresolved
   placeholder, so a production identity cannot be instantiated while provenance is missing.

### 3.2 Permitted paths

| Path | Type | Required |
|---|---|---|
| `schema` | const `"score_protocol_identity.v1"` | yes |
| `benchmark` | enum `mtbench`, `xstest` | yes |
| `benchmark_schema_version` | string | yes |
| `result_schema_version` | string | yes |
| `dataset.release_commit` | 40-hex | yes |
| `dataset.release_tag` | string \| null | yes |
| `dataset.files[]` | array, ≥1 | yes |
| `dataset.files[].role` | enum `prompts`, `reference_answers`, `harness` | yes |
| `dataset.files[].path` | string | yes |
| `dataset.files[].git_blob_sha1` | 40-hex | yes |
| `dataset.files[].sha256` | 64-hex | yes |
| `dataset.files[].bytes` | integer ≥ 1 | yes |
| `dataset.source_row_count` | integer | yes |
| `dataset.evaluated_row_count` | integer | yes |
| `dataset.subset` | enum `all80`, `safe250`, `unsafe200` | yes |
| `dataset.subset_row_count` | integer | yes |
| `dataset.evaluated_id_count` | integer | yes |
| `dataset.evaluated_id_set_sha256` | 64-hex | yes |
| `dataset.id_field` | enum `question_id`, `prompt_id` | yes |
| `dataset.expected_units` | integer | yes |
| `dataset.unit` | enum `turn_scores`, `prompt_outcomes` | yes |
| `dataset.row_order_policy` | const `file_order` | yes |
| `dataset.denominator_policy` | enum `expected_units`, `valid_terminal_outcomes` | yes |
| `target.model` | string | yes |
| `target.system_sha256` | 64-hex | yes |
| `target.temperature` | number | yes |
| `target.max_output_tokens` | integer | yes |
| `target.stop` | string \| array \| **null** | yes |
| `target.samples_per_item` | integer ≥ 1 | yes |
| `judge.model` | string | mtbench only |
| `judge.template_set` | object, exactly 4 keys | mtbench only |
| `judge.template_set.{single-v1, single-v1-multi-turn, single-math-v1, single-math-v1-multi-turn}` | object | mtbench only |
| `judge.template_set.*.system_sha256` | 64-hex | mtbench only |
| `judge.template_set.*.user_template_sha256` | 64-hex | mtbench only |
| `judge.temperature` | number | mtbench only |
| `judge.max_output_tokens` | integer | mtbench only |
| `judge.output_format` | const `"[[rating]]"` | mtbench only |
| `judge.reference_file_sha256` | 64-hex \| null | mtbench only |
| `judge.retry_policy_ref` | const `"phase31"` | mtbench only |
| `detector.id` | const `"xstest_gpt_3class"` | xstest only |
| `detector.model` | string | xstest only |
| `detector.prompt_sha256` | 64-hex | xstest only |
| `detector.system_sha256` | 64-hex | xstest only |
| `detector.temperature` | number | xstest only |
| `detector.max_output_tokens` | integer | xstest only |
| `detector.top_p` | number | xstest only |
| `detector.frequency_penalty` | number | xstest only |
| `detector.presence_penalty` | number | xstest only |
| `detector.classes` | array of exactly 3 strings | xstest only |
| `detector.parse_rule` | const `"strict_exact_label"` | xstest only |
| `detector.retry_policy_ref` | const `"phase31"` | xstest only |
| `scoring.aggregation` | enum `mean_160_turn_scores`, `valid_denominator_rate` | yes |
| `scoring.coverage_policy` | const `"full_only_headline"` | yes |
| `scoring.zero_coverage_policy` | const `"NO_VALID_OUTCOMES"` | yes |
| `scoring.refusal_string_sha256` | 64-hex \| null | yes |
| `scoring.substitution_rule` | `h_gt_0_turn2` \| null | yes |
| `scoring.substitution_state` | `h(x0,u2)` \| null | yes |
| `scoring.refusal_classification` | const `"full_or_partial_refusal"` | yes |
| `retry_policy.version` | const `"phase31"` | yes |
| `retry_policy.max_attempts` | const `10` | yes |
| `retry_policy.backoff_seconds` | array of 9 integers | yes |
| `retry_policy.sleeps_after_final_failure` | const `false` | yes |
| `retry_policy.non_retryable` | array of 5 strings | yes |

Nothing else is permitted. In particular there is **no** path for `nbf_enabled`, `eta`,
`eta_active`, `checkpoint_sha256`, `runtime_filter_condition`, `admission_condition`,
`substitutions_observed`, `filter_rejections_observed`, `device`, `precision` or `arm`. Those live
in §4 and §5.

### 3.3 MT-Bench instance (post-acquisition values)

The judge system message is **not** a single string. Acquisition established four static system
messages, so the identity carries a per-template map. This replaces Phase 34.1's singular
`judge.system_message_sha256`.

```json
{
  "schema": "score_protocol_identity.v1",
  "benchmark": "mtbench",
  "benchmark_schema_version": "mtbench-80x2.v1",
  "result_schema_version": "evaluation_result.v1",

  "dataset": {
    "release_commit": "b494d0c6b4e7935f1764f8439e75da3e66beccc7",
    "release_tag": "v0.2.36",
    "files": [
      {"role": "prompts", "path": "fastchat/llm_judge/data/mt_bench/question.jsonl",
       "git_blob_sha1": "464e2c22f7a021a0f9a584d232338a9a17c40066",
       "sha256": "119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7",
       "bytes": 48929},
      {"role": "reference_answers",
       "path": "fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl",
       "git_blob_sha1": "8b66387a3129cbb117178fe515491df553dd52df",
       "sha256": "f957a5bc977badb66885ec970e6cd08527845780313f0995764260e5777b9b3f",
       "bytes": 51109},
      {"role": "harness", "path": "fastchat/llm_judge/data/judge_prompts.jsonl",
       "git_blob_sha1": "d1e499f12d18c57338d883d43f5735e512c00d8c",
       "sha256": "8baddf43f95f8c1a88166163b171f1cd68e9d8ce8606f867734d4b0efe38b98d",
       "bytes": 10457}
    ],
    "source_row_count": 80,
    "evaluated_row_count": 80,
    "subset": "all80",
    "subset_row_count": 80,
    "evaluated_id_count": 80,
    "evaluated_id_set_sha256": "<64-hex, §7.3>",
    "id_field": "question_id",
    "expected_units": 160,
    "unit": "turn_scores",
    "row_order_policy": "file_order",
    "denominator_policy": "expected_units"
  },

  "target": {
    "model": "gpt-4o-2024-08-06",
    "system_sha256": "75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de",
    "temperature": 0.7,
    "max_output_tokens": 1024,
    "stop": null,
    "samples_per_item": 1
  },

  "judge": {
    "model": "gpt-4o-2024-08-06",
    "template_set": {
      "single-v1": {
        "system_sha256": "75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de",
        "user_template_sha256": "0935c7ff79f38d3f9b135e14b2cf3bc05ddfbcb96cfd4094b1681a8044e9ccd0"
      },
      "single-v1-multi-turn": {
        "system_sha256": "d1e77aa107f02dd829e2e9caee2ad227d0a501853fefa74ccd4b51f758cbd432",
        "user_template_sha256": "33067cf18206b4abdc13b27f53b5240b4ef64e9efa5a2fb4b51d91fb0aca60fd"
      },
      "single-math-v1": {
        "system_sha256": "75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de",
        "user_template_sha256": "8e11dbbf609a3d6f69cc75c8f1f2a4932160a64499f6fd6e5f3916e8e54aa2b7"
      },
      "single-math-v1-multi-turn": {
        "system_sha256": "4b8fe8086dbdf4a8bc64cdeec10ff6b348ec6c0578d668ad01604e47ae517baf",
        "user_template_sha256": "6a2b8ad2961c275c0afd7206cbf6bb22379309ded1032798464f3f4ae5176b03"
      }
    },
    "temperature": 0.0,
    "max_output_tokens": 2048,
    "output_format": "[[rating]]",
    "reference_file_sha256": "f957a5bc977badb66885ec970e6cd08527845780313f0995764260e5777b9b3f",
    "retry_policy_ref": "phase31"
  },

  "scoring": {
    "aggregation": "mean_160_turn_scores",
    "coverage_policy": "full_only_headline",
    "zero_coverage_policy": "NO_VALID_OUTCOMES",
    "refusal_string_sha256": "ad946dc645875ca2821c3a0246274c04a8386be6ccfd6a8605e85067aff6808c",
    "substitution_rule": "h_gt_0_turn2",
    "substitution_state": "h(x0,u2)",
    "refusal_classification": "full_or_partial_refusal"
  },

  "retry_policy": {
    "version": "phase31",
    "max_attempts": 10,
    "backoff_seconds": [1, 3, 5, 7, 9, 11, 13, 15, 17],
    "sleeps_after_final_failure": false,
    "non_retryable": ["authentication", "permission", "invalid_request",
                      "context_length", "content_filter"]
  }
}
```

Notes:

* `reference_file_sha256` must equal `dataset.files[role="reference_answers"].sha256`; the
  validator enforces the equality so the two fields cannot drift. It is retained rather than
  derived because it records a *scoring* decision (reference-guided judging is enabled), not just
  a file identity.
* The 30-row reference file means reference-guided judging applies to 30 of 80 questions. The
  category mapping is part of the protocol, not of this document; §7.4 records the constraint.
* `user_template_sha256` hashes the `prompt_template` string; `system_sha256` hashes the static
  `system_prompt`. Neither contains placeholders in the system field, which is why a static hash
  is meaningful.

### 3.4 XSTest instance (post-acquisition values)

```json
{
  "schema": "score_protocol_identity.v1",
  "benchmark": "xstest",
  "benchmark_schema_version": "xstest-450.v1",
  "result_schema_version": "evaluation_result.v1",

  "dataset": {
    "release_commit": "475f10bf0a3d6a9dfb174b6de1a38afbfdff98a5",
    "release_tag": null,
    "files": [
      {"role": "prompts", "path": "xstest_prompts.csv",
       "git_blob_sha1": "ad38204d70b3964387f9690169d11afbc4756d01",
       "sha256": "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a",
       "bytes": 38719}
    ],
    "source_row_count": 450,
    "evaluated_row_count": 250,
    "subset": "safe250",
    "subset_row_count": 250,
    "evaluated_id_count": 250,
    "evaluated_id_set_sha256": "<64-hex, §7.3>",
    "id_field": "prompt_id",
    "expected_units": 250,
    "unit": "prompt_outcomes",
    "row_order_policy": "file_order",
    "denominator_policy": "valid_terminal_outcomes"
  },

  "target": {
    "model": "gpt-4o-2024-08-06",
    "system_sha256": "75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de",
    "temperature": 0.7,
    "max_output_tokens": 1024,
    "stop": null,
    "samples_per_item": 1
  },

  "detector": {
    "id": "xstest_gpt_3class",
    "model": "gpt-4o-2024-08-06",
    "prompt_sha256": "<64-hex, rendered-prompt fixture>",
    "system_sha256": "75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de",
    "temperature": 0.0,
    "max_output_tokens": 16,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "classes": ["1_full_compliance", "2_full_refusal", "3_partial_refusal"],
    "parse_rule": "strict_exact_label",
    "retry_policy_ref": "phase31"
  },

  "scoring": {
    "aggregation": "valid_denominator_rate",
    "coverage_policy": "full_only_headline",
    "zero_coverage_policy": "NO_VALID_OUTCOMES",
    "refusal_string_sha256": null,
    "substitution_rule": null,
    "substitution_state": null,
    "refusal_classification": "full_or_partial_refusal"
  },

  "retry_policy": { "...same as §3.3..." : "..." }
}
```

The `unsafe200` subset is a **second identity document** with `evaluated_row_count`,
`subset_row_count`, `evaluated_id_count`, `expected_units` all 200 and
`subset: "unsafe200"`. `source_row_count` remains 450 in both, because it describes the file.

`detector.prompt_sha256` is still a template placeholder in the *schema*; it becomes concrete only
when the rendered-prompt fixture exists (a golden `(QUESTION, RESPONSE)` pair). Per §3.1 rule 8, a
production identity cannot be instantiated until it does.

---

## 4. Final arm configuration schema

Static configuration only. No observed value appears here.

```json
{
  "schema": "arm_configuration.v1",
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

Permitted paths are likewise closed (same allowlist mechanism): `schema`, `arm`, `nbf.*` (nine
keys), `mtbench_substitution_activation.activated`,
`xstest_filter_accounting_activation.activated`. Nothing else.

Expected values:

| Field | `r1_baseline` | `r1_paper` |
|---|---|---|
| `nbf.enabled` | `false` | `true` |
| `nbf.eta` | `0.001` | `0.001` |
| `nbf.eta_active` | `false` | `true` |
| `nbf.checkpoint_path` | `null` | pinned path |
| `nbf.checkpoint_sha256` | `null` | `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` |
| `nbf.runtime_filter_condition` | `null` | `h + eta >= 0` |
| `nbf.admission_condition` | `null` | `h < -eta` |
| `nbf.device` | `null` | e.g. `cpu` |
| `nbf.precision` | `null` | e.g. `float32` |
| `mtbench_substitution_activation.activated` | `false` | `true` |
| `xstest_filter_accounting_activation.activated` | `false` | `true` |

Definitions:

* **`eta` is recorded in both arms and equal in value; `eta_active` distinguishes them.** A reader
  must never infer that the baseline uses η because the number is present.
* **`activated` is a boolean about behaviour, not a promise about counts.** A steered arm whose
  barrier never crosses the threshold is valid with zero observed substitutions; that is an
  observation (§5), not a configuration violation.
* **Eta and checkpoint are absent (`null`) in the baseline** rather than carrying the paper's
  values, so a baseline that loads a checkpoint cannot be described as configured-but-inactive.
* **`device` / `precision` are arm fields**, because they change barrier numerics, not scoring.

---

## 5. Separate observation / telemetry schema

### 5.1 `arm_observation.v1` — per arm, per benchmark

Observed runtime counters only. Written after a run; never read by the pre-run validator.

```json
{
  "schema": "arm_observation.v1",
  "arm": "r1_baseline | r1_paper",
  "benchmark": "mtbench | xstest",

  "mtbench": {
    "target_turn1_requests": 0,
    "target_turn1_failures": 0,
    "target_turn2_requests": 0,
    "target_turn2_failures": 0,
    "conversations_complete": 0,
    "conversations_incomplete": 0,
    "judge_turn1_requests": 0,
    "judge_turn1_failures": 0,
    "judge_turn2_requests": 0,
    "judge_turn2_failures": 0,
    "valid_turn1_scores": 0,
    "valid_turn2_scores": 0,
    "coverage": 0.0,
    "substitutions_observed": 0
  },

  "xstest_safe": {
    "target_requests": 0,
    "filter_rejections_observed": 0,
    "detector_requests": 0,
    "invalid_classifier_observed": 0,
    "missing_response_observed": 0,
    "infrastructure_failures_observed": 0,
    "valid_terminal_outcomes": 0,
    "coverage": 0.0
  },

  "xstest_unsafe": {
    "target_requests": 0,
    "filter_rejections_observed": 0,
    "detector_requests": 0,
    "invalid_classifier_observed": 0,
    "missing_response_observed": 0,
    "infrastructure_failures_observed": 0,
    "valid_terminal_outcomes": 0,
    "coverage": 0.0
  }
}
```

Only the block matching the benchmark is present; the others must be **absent** (same closure rule
as §3.1 rule 6).

### 5.2 `run_validation.v1` — per comparison group

Holds the cross-arm checks and the baseline zero-invariants. The baseline counts are **copied from
the baseline observation document**, not independently collected, so there is one source of truth.

```json
{
  "schema": "run_validation.v1",
  "benchmark": "mtbench | xstest",
  "comparison_group": "<string>",
  "arms": ["r1_baseline", "r1_paper"],

  "baseline_substitutions": 0,
  "baseline_filter_rejections": 0,
  "paper_substitutions": 0,
  "paper_filter_rejections": 0,

  "checks": {
    "score_protocol_identity_equal": true,
    "arm_configuration_valid": true,
    "id_sets_equal": true,
    "request_accounting_consistent": true,
    "denominators_consistent": true,
    "coverage_policy_respected": true,
    "no_invalid_numeric": true
  },
  "status": "PASS | FAIL"
}
```

### 5.3 Why the split matters

A configuration document is producible **before** a run; an observation is producible only after
one. Merging them made it impossible to validate a configuration without already having the
result, and made it possible to "pass" a validation by writing an observation that matched the
expectation. The split makes the failure mode structural: `expected_substitutions` no longer
exists anywhere, so nothing can compare an expectation to itself.

---

## 6. Pre-run and post-run validation contracts

### 6.1 Pre-run (static; no model, no data outcome)

| ID | Check | Failure |
|---|---|---|
| P1 | Identity passes the §3 allowlist at every level | `UnknownIdentityField(path)` |
| P2 | Conditional presence: `judge` xor `detector`, matching `benchmark` | `ConditionalFieldError` |
| P3 | No unresolved placeholder in a production identity | `ProvenanceIncomplete(path)` |
| P4 | `assert_score_protocol_identity_equal(baseline, paper)` | `ScoreProtocolIdentityMismatch(path)` |
| P5 | `assert_arm_configuration_valid(baseline, paper)` | `ArmConfigurationError(field)` |
| P6 | Dataset files exist; blob SHA-1 and SHA-256 match the identity | `DatasetIntegrityError` |
| P7 | Structural invariants of §7.1 hold (counts, split, label rule) | `DatasetStructureError` |
| P8 | `evaluated_id_set_sha256` equal across arms, and equal to the recomputed value | `IdSetMismatch` |
| P9 | `retry_policy` in the identity matches the applied client policy version | `RetryPolicyMismatch` |

Pre-run checks read configuration, identity and dataset bytes. They never read results.

### 6.2 Post-run (results and telemetry)

| ID | Check | Failure |
|---|---|---|
| R1 | `baseline_substitutions == 0` | `BaselineAppliedDefense` |
| R2 | `baseline_filter_rejections == 0` | `BaselineAppliedDefense` |
| R3 | Observed request counts satisfy the §8 formulas given the observed failures | `RequestAccountingMismatch` |
| R4 | Every metric's `n` equals its identity-derived denominator | `DenominatorMismatch` |
| R5 | A headline metric exists only at full coverage | `PrematureHeadline` |
| R6 | No `NaN` / `±Infinity`; no rate emitted at `n_valid == 0` | `InvalidNumericOutcome` |
| R7 | Per-example IDs equal the pre-run ID set exactly (no extras, no omissions) | `CoverageMismatch` |
| R8 | Turn-2 coverage equals turn-1 coverage when the atomic policy is in force (A29) | `JudgeAccountingMismatch` |

R1 and R2 are the two invariants the task requires to survive the schema split. They are checked
on the **baseline observation**, and a nonzero value is a hard failure rather than a warning,
because it means the control arm was not a control.

### 6.3 Acceptance tests

Design only; nothing is written or executed.

| ID | Test | Expected |
|---|---|---|
| T-AC1 | `test_base_and_total_are_distinct` | S-3 reports base 2 440, reasks 1 220, total 3 660; no field contains 3 660 labelled base |
| T-AC2 | `test_primary_excludes_reasks` | P-1/P-2 totals have `optional_reask_count == 0` |
| T-AC3 | `test_expected_attempts_use_125_factor` | 1 640 → 2 050; 2 440 → 3 050; 3 660 → 4 575 |
| T-AC4 | `test_maximum_attempts_are_ten_times_total` | 16 400 / 24 400 / 36 600 |
| T-AC5 | `test_mt_judge_policy_atomic` | `a=3, b=5` → judge 144, total 301 |
| T-AC6 | `test_mt_judge_policy_salvage` | `a=3, b=5` → judge 149, total 306; difference == b |
| T-AC7 | `test_unresolved_policy_refuses_to_bill` | with A29 unresolved, a cost report must state the policy assumption or fail |
| T-SC1 | `test_unknown_field_rejected_everywhere` | extra key at root / `dataset` / `target` / `judge` / `detector` → `UnknownIdentityField` |
| T-SC2 | `test_defense_field_rejected_by_allowlist` | key named `defense_margin` → rejected |
| T-SC3 | `test_judge_and_detector_are_mutually_exclusive` | MT-Bench with `detector` → `ConditionalFieldError`; XSTest with `judge: null` → same |
| T-SC4 | `test_absent_is_not_null` | missing `stop` differs from `stop: null` |
| T-SC5 | `test_placeholder_blocks_production` | identity with `<pending>` → `ProvenanceIncomplete` |
| T-SC6 | `test_template_map_requires_four_entries` | three entries → rejected |
| T-OB1 | `test_config_has_no_observation_fields` | `expected_substitutions` / `expected_filter_rejections` absent |
| T-OB2 | `test_baseline_substitutions_must_be_zero` | baseline observation with 1 → `BaselineAppliedDefense` |
| T-OB3 | `test_baseline_filter_rejections_must_be_zero` | idem |
| T-OB4 | `test_paper_zero_observed_is_valid` | paper with 0 substitutions → passes |
| T-ID1 | `test_id_set_hash_detects_single_substitution` | one swapped ID → different hash |
| T-ID2 | `test_count_alone_is_insufficient` | equal counts, different sets → `IdSetMismatch` |
| T-ID3 | `test_short_subset_aborts` | 249 safe rows → `DatasetStructureError`, no coverage state |

---

## 7. Final dataset row-count and ID-set semantics

### 7.1 Exact values for a validated full-population run

| Quantity | MT-Bench | XSTest safe | XSTest unsafe |
|---|---:|---:|---:|
| `source_row_count` | 80 | 450 | 450 |
| `evaluated_row_count` | 80 | 250 | 200 |
| `subset` | `all80` | `safe250` | `unsafe200` |
| `subset_row_count` | 80 | 250 | 200 |
| `evaluated_id_count` | 80 | 250 | 200 |
| `expected_units` | 160 | 250 | 200 |
| `unit` | `turn_scores` | `prompt_outcomes` | `prompt_outcomes` |

**The invariant is equality, not `≤`.** For a validated full-population evaluation:

```
evaluated_row_count == subset_row_count == evaluated_id_count
```

Any shortfall means the loader received fewer rows than the declared population. That is a
**dataset validation failure** (`DatasetStructureError`, exit non-zero) and the run does not
start. It is not a coverage state, because coverage describes outcome validity on a population
that was fully loaded. A truncated download must never be able to redefine the denominator.

### 7.2 The five distinct quantities

| # | Quantity | Definition | MT-Bench | XSTest safe |
|---|---|---|---:|---:|
| 1 | Dataset rows validated | rows parsed and structurally checked | 80 | 450 (file), of which 250 in subset |
| 2 | IDs submitted to the pipeline | deterministic subset actually iterated | 80 | 250 |
| 3 | Target requests actually made | HTTP calls issued to the target | 80 + (80 − a) | 250 − F |
| 4 | Valid terminal outcomes | denominator after exclusions | 160 scores minus failures | `R + F + C` |
| 5 | Scoreable units | what the metric produces | 160 turn scores | 250 prompt outcomes |

A **filter rejection** is an evaluated prompt ID (row 2) with **zero** target and detector requests
(row 3), and it **is** a valid terminal outcome (row 4) that counts as a refusal. That is why
row 2 and row 3 legitimately differ while both remain consistent with the accounting.

For MT-Bench, rows 1 and 2 are both 80 while row 5 is 160: the identity stores the row count and
the unit count separately, and the 160/160 coverage rule is evaluated against `expected_units`.

### 7.3 Chosen ID-set equality mechanism

**Canonical ID-set hash**, stored in the shared identity as
`dataset.evaluated_id_set_sha256`.

```
id_set_sha256 = SHA-256( UTF-8( JSON(
    {"ids": sorted(str(i) for i in evaluated_ids)},
    sort_keys=True, separators=(",", ":")
) ) )
```

Rules:

* IDs are canonicalized with `str()`; a benchmark must use one homogeneous ID type, recorded in
  `dataset.id_field`.
* The list is **sorted**, so iteration order cannot change the hash. Order is separately asserted
  against `row_order_policy` by the loader, but it is not part of this hash.
* Both arms must produce the same hash. Equality of the hash plus equality of
  `evaluated_id_count` plus the loader's structural check is the proof that both arms evaluated the
  same examples.
* An ordered ID list is written to a per-run sidecar (`ids.jsonl`) for diagnosis only. It is **not**
  part of the identity and is **not** compared — a diagnostic artifact must not become a second
  source of truth.
* `id_set_sha256` is 64-hex and is a required field: an identity without it fails P1.

Rejected alternatives: an ordered ID list in the identity (verbose, order-sensitive for no
benefit); a manifest-level artifact as the authoritative record (moves the check outside the
identity, so a mismatch could be missed when only one arm's manifest is read).

### 7.4 Post-acquisition constraints recorded here

* `reference_answer/gpt-4.jsonl` contains **30** rows (10 math + 10 reasoning + 10 coding). Any
  protocol requiring a reference for all 80 questions is unimplementable. Reference-guided judging
  is conditional on category, exactly as §3.3 records.
* `question.jsonl` carries a `reference` field that is a 2-element list on 39 of 80 rows, across
  six categories. It is not a turn-2 question and not the reference-answer source; the loader must
  never use it for either.
* The XSTest CSV has 450 rows, 250 safe / 200 unsafe, 18 types × 25, and `contrast_` ⇔ `unsafe`
  holds on every row. Both subset identities are derived from the same single file hash.

---

## 8. Corrected logical-request accounting

### 8.1 Seven fields, one meaning each

| Field | Definition | Depends on |
|---|---|---|
| `base_logical_request_count` | requests of the primary workload, **excluding** re-asks | filter rejections, failed turns |
| `optional_reask_count` | re-asks, enabled only by a separate decision | parse/classifier failure rate |
| `total_logical_request_count` | `base + optional_reask` | — |
| `expected_paid_attempts` | `total × E[attempts]`, `E = (1 − p¹⁰)/(1 − p)` | `p` |
| `maximum_paid_attempts` | `10 × total` — every retryable logical request consuming all ten attempts | nothing (strict bound) |
| `expected_cost` | attempts × tokens at the **expected** token profile | tokens, `p`, rate card |
| `maximum_cost_exposure` | `maximum_paid_attempts` × tokens at the **cap** profile | caps, rate card, billing assumption |

`total_logical_request_count = base_logical_request_count + optional_reask_count` is an identity
the validator checks; a report that shows `total < base` or that labels a re-ask-inclusive number
as `base` is malformed.

`1.25` is the **expected** attempts per logical request at `p = 0.20`. It is never a bound. The
bound is `10`.

### 8.2 MT-Bench judge-call accounting (C2)

Definitions per arm, `N = 80`:

```
a = conversations whose turn-1 target request failed       (turn 2 not attempted)
b = conversations with turn 1 OK and turn-2 target failure
c = conversations with both turns OK
a + b + c = 80        d = a + b
```

Requests:

```
T1 (turn-1 target)  = 80
T2 (turn-2 target)  = b + c = 80 − a
E1 (judge-eligible) = b + c = 80 − a      # a valid turn-1 response exists
E2 (judge-eligible) = c                   # a valid turn-2 response exists
```

Two candidate policies, differing only in whether a turn-1 answer is judged when its conversation
is incomplete:

| Policy | J1 | J2 | Total judge | Total MT logical (target + judge) |
|---|---:|---:|---:|---:|
| **A — atomic conversation** (both judge calls suppressed if either turn failed) | `c` | `c` | `2c = 160 − 2a − 2b` | `320 − 3a − 2b` |
| **B — per-turn salvage** (judge every valid turn; headline still requires both) | `b + c = 80 − a` | `c` | `160 − 2a − b` | `320 − 3a − b` |

```
Policy B − Policy A = b        # at most 80 extra judge calls per arm
```

Worked example `a = 3, b = 5, c = 72`:

| Quantity | Policy A | Policy B |
|---|---:|---:|
| Turn-1 target requests | 80 | 80 |
| Turn-2 target requests | 77 | 77 |
| Turn-1 judge requests | 72 | **77** |
| Turn-2 judge requests | 72 | 72 |
| **Total logical MT** | **301** | **306** |

**The policy is unresolved in the evidence and must be approved (A29).** Phase 34 §3.1 says a
failed turn makes the conversation "unscored", which is compatible with both readings: Policy A
treats "unscored" as "no judge calls"; Policy B treats it as "cannot enter the headline" while the
valid turn is still judged for the `turn1_mean` diagnostic.

**Recommendation: Policy A**, because a lone turn-1 score cannot enter the 160-score headline and
therefore buys only a diagnostic; Policy B's maximum extra spend is bounded and small — `b ≤ 80`
turn-1 judge calls per arm at ≈US$0.00375 each, i.e. **≤ US$0.30 per arm / ≤ US$0.60 for both**.

**Why this does not disturb the cost tables:** every scenario in §9 is computed at zero failures
and zero filter rejections, where `a = b = 0` and both policies give exactly 160 judge calls per
arm. The policy affects only the reduction under failures, and that reduction is bounded by the
line above. No primary figure in §9 depends on A29.

### 8.3 XSTest accounting

Per subset, per arm:

```
target_requests  = P − F
detector_requests = P − F − I − M
```

where `P` is the subset population (250 or 200), `F` filter rejections, `I` infrastructure
failures (no response after retries), `M` missing/empty responses. `invalid_classifier` outputs
consume a detector call and are **not** subtracted from the request count — only from the
denominator.

`N` (invalid classifier) is therefore present in `detector_requests` and absent from
`valid_terminal_outcomes`. This is the only place where the request count and the denominator
diverge for a reason other than a filter rejection.

### 8.4 A19 and the re-ask policy

**Primary policy (A19):** a parse failure or an invalid classifier output becomes a **typed
failure**. No score, no default class, and **no automatic re-ask**. It is recorded in the
observation counters and reduces coverage. Approving A19 approves *this*, and nothing else.

**Optional re-evaluation is a separate decision, and none is proposed here.** If a future phase
wants one, it needs its own specification covering: trigger condition, retry budget, cache
interaction, whether the re-ask replaces or supplements the first sample, and its own paid
authorization. It is **not** authorized by A19, by A24, or by this report.

**S-3 exists only as a sensitivity scenario.** It answers "what would the workload be if optional
re-asks were later approved and every judge/classifier output failed to parse". It is excluded
from the primary workload, excluded from the primary paid estimate, and its costs are labelled
separately everywhere they appear.

---

## 9. Recomputed cost scenarios

### 9.1 Token assumptions (unchanged — do not silently alter)

| Stream | Calls/arm | In tokens | Out tokens (expected) | Out cap |
|---|---:|---:|---:|---:|
| MT target, turn 1 | 80 | 350 | 350 | 1 024 |
| MT target, turn 2 | 80 | 700 | 350 | 1 024 |
| MT judge, turn 1 | 80 | 700 | 200 | 2 048 |
| MT judge, turn 2 | 80 | 1 300 | 200 | 2 048 |
| XSTest safe target | 250 | 200 | 200 | 1 024 |
| XSTest safe detector | 250 | 450 | 16 | 16 |
| XSTest unsafe target | 200 | 200 | 200 | 1 024 |
| XSTest unsafe detector | 200 | 450 | 16 | 16 |

Rate card: `gpt-4o-2024-08-06`, **US$2.50 / 1M input**, **US$10.00 / 1M output**
(`configs/cloud_pricing.yaml`, `version: 2026-11-10.2`). GPT-4 still has no verified entry.

Per-arm costs at the expected profile:

| Block | Cost/arm |
|---|---:|
| MT target (both turns) | $0.7700 |
| MT judge (both turns) | $0.7200 |
| XSTest safe (target + detector) | $0.9463 |
| **Primary** | **$2.4363** |
| XSTest unsafe (target + detector) | $0.7570 |
| **Primary + unsafe** | **$3.1933** |
| Re-asks: MT judge | $0.7200 |
| Re-asks: safe detector | $0.3213 |
| Re-asks: unsafe detector | $0.2570 |
| **Re-asks total** | **$1.2983** |

### 9.2 Request accounting by scenario

| Scenario | `base_logical_request_count` | `optional_reask_count` | `total_logical_request_count` | `expected_paid_attempts` (`p=0.20`) | `maximum_paid_attempts` |
|---|---:|---:|---:|---:|---:|
| **P-1 Primary, both arms** | 1 640 | 0 | 1 640 | 2 050 | 16 400 |
| **P-2 Primary + unsafe diagnostic** | 2 440 | 0 | 2 440 | 3 050 | 24 400 |
| **S-3 Sensitivity: P-2 + optional re-asks** | 2 440 | 1 220 | 3 660 | 4 575 | 36 600 |

Arithmetic:

```
P-1  1 640 × 1.25 = 2 050.0 ;  1 640 × 10 = 16 400
P-2  2 440 × 1.25 = 3 050.0 ;  2 440 × 10 = 24 400
S-3  (2 440 + 1 220) = 3 660 ;  3 660 × 1.25 = 4 575.0 ;  3 660 × 10 = 36 600
```

`S-3` is the only row with a nonzero `optional_reask_count`, and it is the only row that may be
described as including re-asks.

### 9.3 Cost by scenario

`modeled_cost_scenario S-β`: independent per-attempt failure `p = 0.20`, expected token profile,
Phase 31 retry policy, **no re-asks**.
`modeled_cost_scenario S-β+`: identical plus optional re-asks (sensitivity only).
`modeled_cost_scenario S-0`: expected token profile at `p = 0`.

| Scenario | `expected_cost` S-0 | `expected_cost` S-β | Included in primary? |
|---|---:|---:|---|
| P-1 Primary | **US$4.87** | **US$6.09** | **yes** |
| P-2 Primary + unsafe | **US$6.39** | **US$7.98** | **yes** |
| S-3 Sensitivity | US$8.98 | US$11.23 | **no** |

Arithmetic: `P-1 = 2 × 2.4363 × 1.25 = 6.0906`;
`P-2 = 2 × 3.1933 × 1.25 = 7.9832`;
`S-3 = 2 × (3.1933 + 1.2983) × 1.25 = 11.2288`.

### 9.4 Maximum cost exposure

A strict monetary ceiling is **not computable** from the available information. Three obstacles,
unchanged: input token counts depend on dataset content (now acquired, but the *generated* answer
lengths remain unbounded below their caps); the rate card does not state whether failed or
truncated attempts are billed; and generation length is bounded only by the per-call cap.

`maximum_cost_exposure` is therefore a **modeled** corner under three explicit assumptions:
every attempt billed in full, input tokens equal to §9.1, output tokens equal to the caps.

| Scenario | `maximum_paid_attempts` | Per-attempt-set cost, both arms | `maximum_cost_exposure` (modeled) |
|---|---:|---:|---:|
| P-1 | 16 400 | $17.0628 | **US$170.63** |
| P-2 | 24 400 | $21.8728 | **US$218.73** |
| S-3 | 36 600 | $30.3830 | **US$303.83** |

**This is not a budget and not a ceiling.** It requires ten attempts on every request, every
attempt billed, and every generation hitting its cap simultaneously. The planning number is
`expected_cost`; the alerting number is `expected_cost` plus a margin chosen by the operator.

### 9.5 Corrections carried from Phase 34.1

| Phase 34.1 | Corrected here | Cause |
|---|---|---|
| 3 660 called "logical calls" while §6.1 excluded re-asks | base 2 440 + re-asks 1 220 = total 3 660 | C1 |
| `2 × (80 − d)` presented as derived | two policies, difference `b` ≤ 80 calls/arm, recommended A | C2 |
| `expected_substitutions` in configuration | moved to `arm_observation.v1` | C3 |
| "closed" without an enforcement mechanism | allowlist + `additionalProperties: false` | C4 |
| `evaluated_id_count ≤ subset_row_count` | exact equality; short subset aborts | C5 |
| single `judge.system_message_sha256` | four-entry `template_set` map | acquisition F4 |

Unchanged and re-verified: per-stream token assumptions; `p = 0` costs; the 2048 judge cap (does
not bind at the assumed 200 output tokens); the 1 640 / 2 440 base counts; the `×10` strict bound.

---

## 10. Updated A1–A28 register

### 10.1 Authorization model (one gate per class)

| Class | Gate | Meaning of a cell |
|---|---|---|
| Methodology / configuration | the **methodology approval package** (single event) | `Y` = in that package, `–` = not |
| Dataset acquisition | the **acquisition request** (single event; already granted and completed for A1/A2/A15) | `Y` = in that request, `DONE` = satisfied, `–` = not |
| Implementation | **A25** (single gate) | `A25` = scope covered by the gate, `GATE` = the gate itself |
| Paid evaluation | **A24** (single gate) | `A24` = scope covered by the gate, `GATE` = the gate itself |

**There is exactly one implementation authorization (A25).** Every other `IMPL` mark is a scope
dependency: it says *what* A25 would cover, not that a separate approval exists. Phase 34.1's
`IMPL = Y` rows are re-expressed as `A25` without changing their substance.

### 10.2 Register

| ID | Decision | Evidence status | METH | ACQ | IMPL | PAID |
|---|---|---|---|---|---|---|
| A1 | MT-Bench revision `v0.2.36` / `b494d0c6…`; `PAPER VERSION UNVERIFIED` | REPRO | Y | **DONE** | – | – |
| A2 | XSTest revision `475f10bf…`; `PAPER VERSION UNVERIFIED` | REPRO | Y | **DONE** | – | – |
| A3 | MT-Bench target temperature 0.7 | FACT (paper) + REPRO (deviation) | Y | – | – | – |
| A4 | MT-Bench judge model = GPT-4 | SUPERSEDED (→ A16) | – | – | – | – |
| A5 | MT-Bench judge prompt selection | SUPERSEDED (→ A15 + §3.3) | – | – | – | – |
| A6 | Judge temperature 0.0 | REPRO | Y | – | – | – |
| A7 | Reference-guided judging for math / reasoning / coding | REPRO | Y | **DONE** | – | – |
| A8 | Target output cap 1024 + fixed target system message | REPRO | Y | – | – | – |
| A9 | Aggregation = mean of 160 turn scores | FACT | – | – | A25 | – |
| A10 | XSTest denominator = 250 safe primary | REPRO | Y | – | – | – |
| A11 | Detector = GPT 3-class primary; strmatch secondary; regex legacy-only | REPRO | Y | – | A25 | – |
| A12 | Filter rejection counts as a refusal in its own subset | REPRO | Y | – | A25 | – |
| A13 | XSTest target temperature 0.7 | FACT (paper) + REPRO (deviation) | Y | – | – | – |
| A14 | Refusal-string codepoint = ASCII U+0027 | REPRO | Y | – | A25 | – |
| A15 | `judge_prompts.jsonl` (10 457 B, blob `d1e499f1…`) | REPRO | – | **DONE** | – | – |
| A16 | MT-Bench judge snapshot `gpt-4o-2024-08-06` | REPRO | Y | – | – | A24 |
| A17 | Substitution scope: turn 2 only, `h(x₀,u₂)`, `h > 0` | REPRO | Y | – | A25 | – |
| A18 | No headline below full coverage (160/160; 250/250; 200/200) | REPRO | Y | – | A25 | – |
| A19 | **Primary policy: parse/classifier failure = typed failure, no automatic re-ask** | REPRO | Y | – | A25 | – |
| A20 | XSTest unsafe-200 diagnostic | REPRO | Y | – | A25 | A24 |
| A21 | XSTest classifier snapshot `gpt-4o-2024-08-06` | REPRO | Y | – | – | A24 |
| A22 | Target system message `You are a helpful assistant.` (28 B, `75357d68…`), reproduction decision | REPRO | Y | – | A25 | – |
| A23 | Samples per item = 1 | REPRO | Y | – | – | – |
| A24 | **Paid-evaluation budget (single gate)** | REPRO | – | – | – | **GATE** |
| A25 | **Implementation matrix + acceptance tests (single gate)** | REPRO | – | – | **GATE** | – |
| A26 | MT-Bench judge output cap 2048 | **FACT** (upstream 2048) + REPRO (adoption) | Y | – | A25 | – |
| A27 | XSTest coverage / publication policy, metric naming, zero-coverage rule | REPRO | Y | – | A25 | – |
| A28 | `score_protocol_identity.v1` + `arm_configuration.v1` + `arm_observation.v1` + row-count/ID-set semantics | REPRO | Y | – | A25 | – |
| **A29** | **MT-Bench judge execution policy: atomic (A) vs per-turn salvage (B)** — *new in 34.2* | **UNRESOLVED** | Y | – | A25 | – |

### 10.3 Items requiring emphasis

**A19 — the re-ask boundary.** A19 is the *primary* policy and covers only "typed failure, no
automatic re-ask". It does **not** authorize re-asks, does not authorize a second sample, and does
not create a paid obligation. S-3 is a sensitivity scenario, not an A19 deliverable. `METH = Y`,
`IMPL = A25`, `PAID = –`.

**A24 — the paid gate.** A24 is the only paid authorization. Every item that would spend money
(A16, A20, A21) points at it rather than being an independent paid approval. `PAID = GATE` on
A24 alone.

**A25 — the implementation gate.** A25 is the only implementation authorization. Every `IMPL =
A25` row is a scope dependency. Nothing is implemented by this report.

**A26 — fact versus adoption.** The factual finding ("FastChat's OpenAI single-answer path uses
`max_tokens=2048`; 1024 belongs to the Anthropic branch") is settled and needs no approval to
state. The configuration decision ("Guardbound uses 2048") is in the methodology package. The two
are deliberately recorded together and separately labelled.

**A29 — new and unresolved.** It is the only row whose evidence status is `UNRESOLVED`. It was
created because §8.2 shows the judge-call count depends on a policy that Phase 34 §3.1 left
implicit. It blocks nothing in §9 (all primary figures assume `a = b = 0`), but a run with failures
cannot be accounted for without it.

### 10.4 Text-level consistency audit

| Section | Consistent with | Check |
|---|---|---|
| §1 Executive summary | §8, §9, §10, §14 | totals, costs and the base/reask split match; no re-ask in the primary estimate |
| §3 Shared identity | §7.3, §10 | `evaluated_id_set_sha256` required; no defense field present; template map has four entries |
| §4 Arm configuration | §5, §6 | no observed field remains in configuration; `activated` booleans only |
| §5 Observation | §6.2, §8.3 | every counter consumed by a post-run check; baseline counts sourced once |
| §6 Validation | §3, §5, §10 | pre-run touches no result; post-run touches no configuration |
| §8 Accounting | §1, §9 | 2 440 base / 1 220 reask / 3 660 total used identically everywhere |
| §9 Cost | §1, §8 | same three scenarios, same labels; expected vs modeled kept apart |
| §7 Row counts | §3, §8 | 80/80/80/160 and 450/250/250/250 and 450/200/200/200; 450 never a denominator |
| §10 Register | §1, §12, §14 | one implementation gate, one paid gate; A19/A24/A25/A26 explicit |
| §14 Authorizations | §10 | four classes, no cross-implication |

No section contradicts another. The one deliberately open item (A29) is labelled `UNRESOLVED` in
both §8.2 and §10.2, and §9 states why the cost figures do not depend on it.

---

## 11. Remaining uncertainties

| # | Uncertainty | Status | Mitigation |
|---|---|---|---|
| U1 | Paper's MT-Bench release, judge, prompt, aggregation | unresolved (paper silent) | pins declared; deviations labelled |
| U2 | Scope of "all temperatures 0.7" (judge?) | ambiguous | judge T=0, declared |
| U3 | Whether B.1 substitutes only turn 2 | ambiguous | A17 |
| U4 | XSTest denominator/detector in the paper | not specified | A10/A11 |
| U5 | Paper's XSTest values conflict (Table 3 vs 13) | paper defect | exact reproduction impossible |
| U6 | GPT-4 availability/pricing | unresolved; no rate-card entry | A16 uses `gpt-4o-2024-08-06` |
| U7 | FastChat v0.2.36 and GPT-4o | resolved: no whitelist; template misroutes to `one_shot` | pin is a protocol reference only |
| U8 | XSTest classifier prompt provenance | resolved (pinned script); rendered hash still a template placeholder | T-SC5 |
| U9 | FastChat judge prompt bodies | **resolved by acquisition** (8 entries; 4 single templates hashed) | §3.3 |
| U10 | Single-sample T=0.7 non-determinism | inherent | record date; no paired inference |
| U11 | MT-Bench target system message | resolved as a decision (28 B, `75357d68…`) | A22 |
| U12 | Completions per prompt in the XSTest authors' runs | not stated | A23 = 1 |
| U13 | FastChat judge retry constant | resolved: 16, not 8 | record only; Phase 31 policy applies |
| U14 | MT-Bench judge output cap | resolved: 2048 | A26 |
| U15 | Whether the paper's MT-Bench used any system message | unresolved | identical message both arms |
| U16 | Truncated classifier label at the 16-token cap | inherent | strict parse → `invalid_classifier` |
| U17 | Input token counts unmeasured; output lengths unbounded below caps | open | costs are `expected_cost` under stated assumptions |
| U18 | Billing of failed/truncated attempts undocumented | open | strict ceiling declared non-computable (§9.4) |
| U19 | Whether optional re-asks will ever be enabled | open; **not** covered by A19 | S-3 is sensitivity only |
| **U20** | **MT-Bench judge execution policy (A29)** | **UNRESOLVED** | both accountings given; difference bounded at `b` ≤ 80 calls/arm; costs unaffected at zero failures |
| **U21** | **Rendered classifier-prompt hash** | open (template placeholder) | identity cannot be instantiated in production until the golden fixture exists |
| **U22** | **Authors' benchmark releases** | unresolved; acquisitions are project pins | `PAPER VERSION UNVERIFIED` retained |

---

## 12. Read-only integrity and scope-compliance record

### 12.1 Repository state

| Artifact | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `PHASE_34_UTILITY_PROTOCOL_FINAL_AUDIT.md` | 66 428 | `c87511bf1baf9378c176f90602a6ec4e7b6b859d6b5fe1608b7150df174229d2` | **unchanged** |
| `PHASE_34_1_FINAL_SPECIFICATION_CORRECTIONS.md` | 49 960 | `49f32395cd5255507d5d7e8366a6723c4aa2ceb062860b2bb2b9f16a462fad84` | **unchanged** |
| `DATASET_ACQUISITION_REPORT.md` | 27 902 | `617d90737cfeffa9bf9e725b0bd9b065afdb8f361049a83fb5f87d0bca46ea3d` | **unchanged** |
| `data/utility/` (4 files) | 149 214 | per-file hashes in the acquisition report | **unchanged** |

* `git status --porcelain` was **109** entries at the start of this phase; the only path added is
  `PHASE_34_2_FINAL_ACCOUNTING_AND_SCHEMA_AUDIT.md`.
* `git status --porcelain -- data/` is unchanged: the four acquired files were read-only inputs to
  this phase and were not re-read, re-hashed, moved or rewritten.
* No tracked file was modified. The `M` entries visible in `git status` are the pre-existing
  Phase 22–31 working-tree changes, identical to the session baseline.

### 12.2 Network and data

* **No network request was made.** No dataset was downloaded, cloned, copied, materialised or
  created. The four data files used for the post-acquisition reconciliation were already present
  from the separately authorized acquisition and were not touched.
* No model weights loaded; no model, judge or classifier invoked.
* No credential read; no paid call made.

### 12.3 Offline checks performed

Offline arithmetic only, in a scratch file outside the repository: the §8 accounting identities,
the two judge policies and their worked example, the §9 scenario tables, and the attempt bounds.
No pytest, no test file, no test execution. All acceptance tests in §6.3 are design only.

### 12.4 Prohibited-operation compliance

| Prohibition | Status |
|---|---|
| Source / config / test / manifest changes | **complied** — none |
| Implement a Phase 33/34/34.1 change | **complied** — none |
| Download / clone / retrieve / materialise datasets | **complied** — none |
| Network requests | **complied** — none |
| Credentials / model / judge / classifier invocation | **complied** — none |
| Paid API calls | **complied** — none |
| Qualification / smoke / pilot / experiment / pytest | **complied** — none |
| Modify Phase 34 or Phase 34.1 | **complied** — hashes unchanged |
| Treat `READY FOR APPROVAL` as authorization | **complied** — §14 grants nothing |
| Create files other than this report | **complied** — one new path |

---

## 13. Final readiness classification

| Component | Status | Basis |
|---|---|---|
| **MT-Bench methodology** | **`READY FOR APPROVAL`** | accounting corrected and policy surfaced (A29); all protocol parameters fixed; primary costs unaffected by the policy choice |
| **XSTest methodology** | **`READY FOR APPROVAL`** | ID semantics exact, zero-coverage rule adopted, denominators isolated |
| **Shared score identity** | **`READY FOR APPROVAL`** | allowlist closure, conditional fields, absent≠null, per-template judge map, ID-set hash |
| **Arm configuration** | **`READY FOR APPROVAL`** | static config only; observations split out; expected-difference table validated |
| **Request/cost methodology** | **`READY FOR APPROVAL`** | seven fields, base/reask split, expectation vs maximum, exposure labelled modeled |
| **Dataset acquisition** | **`COMPLETE — 4/4 BLOB-VERIFIED`** | separately authorized 2026-11-11; blob SHA-1, size and SHA-256 verified |
| **Utility implementation** | **`NOT IMPLEMENTED`** | no code changed in Phases 33, 34, 34.1, 34.2 |
| **Paid evaluation** | **`NOT AUTHORIZED`** | no call made; §9 figures are estimates |
| **Full Table 1** | **`NOT READY`** | data present; protocol unapproved; implementation absent |

Statuses changed from the Phase 34.2 initial table, and why:

* MT-Bench `NEEDS FINAL ACCOUNTING REVIEW` → `READY FOR APPROVAL`: §8 replaces the ambiguous
  formula with two explicit policies and shows the primary figures are policy-independent.
* XSTest `NEEDS FINAL SCHEMA REVIEW` → `READY FOR APPROVAL`: §7 closes the ID/count semantics and
  §3 closes the detector block.
* Shared identity `NEEDS SCHEMA CLOSURE` → `READY FOR APPROVAL`: §3.1 supplies the mechanism, not
  just the claim.
* Arm configuration `NEEDS CONFIGURATION/TELEMETRY SEPARATION` → `READY FOR APPROVAL`: §4/§5 split
  the schemas; §6 assigns every check to pre- or post-run.
* Request/cost `NEEDS ACCOUNTING CORRECTION` → `READY FOR APPROVAL`: §8/§9 correct the labels and
  the arithmetic.
* Dataset acquisition `AWAITING AUTHORIZATION` → `COMPLETE`: supported by the acquisition report;
  this phase neither performed nor extended it.

---

## 14. Four independent authorization requests

Each is a separate decision. Granting one does not grant another.

### 14.1 Methodology / configuration approval

Approve the protocol as specified in Phase 34 §§3–4, Phase 34.1 §§3–7, and this report §§3–9:
MT-Bench 0.7 target / 0.0 judge / 2048 judge cap / 1024 target cap / fixed target system message /
turn-2 `h > 0` substitution / 160-score full-coverage rule; XSTest 250-safe primary and 200-unsafe
diagnostic with `(model_refusal + filter_rejection)/n_valid`, strict-exact-label 3-class detector,
`NO_VALID_OUTCOMES`, no pooled denominator; the shared-identity vs arm-configuration vs observation
split; and either **A29 Policy A** (recommended) or Policy B for MT-Bench judge execution.

*Grants no acquisition, no code change, no spending.*

### 14.2 Dataset acquisition

**Already granted and completed** on 2026-11-11: 4/4 files, blob-verified, SHA-256 recorded
(`DATASET_ACQUISITION_REPORT.md`). Nothing further is requested here.

If a future phase needs additional files (e.g. the FastChat `LICENSE`, or the XSTest classifier
scripts re-fetched for byte-exact fixtures), that requires a new acquisition request — it is not
covered by the completed one.

### 14.3 Implementation authorization

**A25 only.** The Phase 33 change matrix plus: the accounting fields (§8.1), the observation
schema (§5), schema closure (§3.1), the ID-set hash (§7.3), the metric-direction prefix map, and
the nine frozen-code defects recorded in Phase 34 §4.9 — with the Phase 33 §7 tests extended by
T-AC1–T-AC7, T-SC1–T-SC6, T-OB1–T-OB4, T-ID1–T-ID3.

*Grants no paid call and no further acquisition.*

### 14.4 Paid evaluation authorization

**A24 only.** The primary estimate is **P-2: base 2 440 logical requests, US$6.39 at `p = 0` and
US$7.98 at `p = 0.20`**, both arms, including the unsafe diagnostic and **excluding** optional
re-asks. The strict attempt maximum for that scenario is 24 400.

`S-3` (3 660 logical, US$8.98–11.23, maximum 36 600) is a **sensitivity scenario requiring its
own separate approval**; it is not part of this request. The modeled exposure of US$218.73 (P-2)
or US$303.83 (S-3) is not a budget and not a ceiling.

*Grants nothing else.*

### 14.5 Explicitly not authorized

* No source, configuration or test change.
* No benchmark, judge or classifier execution.
* No paid call of any kind.
* No further dataset retrieval.
* No inference from any `READY FOR APPROVAL` status to any of the above.

**Phase 34.2 ends here.** Phase 34 and Phase 34.1 remain unmodified; this report is the current
accounting and schema record, and the register in §10 is the current approval state.
