# PHASE 41 — R1 QUALIFICATION PROBE V2 OFFLINE IMPLEMENTATION REPORT

**Phase:** 41 (offline implementation of the Phase 40B specification)
**Date:** 2026-11-12
**Scope:** implementation only; no qualification run, no model API call, no research experiment.
**Verdict:** `READY FOR PHASE 42 INDEPENDENT OFFLINE AUDIT`

---

## 1. Executive summary

The Phase 40B design is implemented in the Guardbound repository:

* `probe_actor_attack_pre_attack` now runs the explicit
  `extract_harm_target → get_actors → validate → get_init_queries` sequence on a deterministic
  30-goal HarmBench prefix, validates the raw pre-filter payload, separates request errors from
  structural failures, and reports the five-state `delivery_type` diagnostic without gating on it.
* `probe_target` produces five independent nine-message conversations behind the exact R1 system
  prompt, retains all four assistant replies, classifies `pass / incorrect / refusal / error`, and
  keeps `final_reply` as a reference to message 8.
* Qualification reports carry `schema_version = "qualification.v2"`, `probe_spec_version =
  "r1-probe.v2"`, a UTC `run_id`, and a separate `experiment_config` identity.
* Reports are written under the canonical immutable filename
  `<tag>.<arm>.<run_id>.qualification.v2.json`; the resolver selects the newest, rejects
  same-timestamp ambiguity, and never falls back.
* `require_qualification` accepts only v2 reports whose mandatory probe set is present and `PASS`,
  whose `probes_complete` is `true`, and whose identities match. Legacy v1 reports remain readable
  but are explicitly rejected.
* A triggered `delivery_type` review leaves the immutable report untouched and requires a separate
  `--review-decision` record before a paid experiment is authorized.

**Verification:** the new Phase 41 suite is **48 passed**; the focused qualification suites are
**282 passed**; the full offline suite is **1384 passed, 4 failed, 0 skipped**, where all four
failures are pre-existing machine-environment checks (VRAM guard and a locally-uncached Qwen
model), unrelated to this phase.

No Python source outside the approved files, no R1 configuration, no dataset, no checkpoint, and no
historical report was modified.

---

## 2. Phase 40B compliance matrix

| Requirement | Status | Evidence |
|---|---|---|
| C1 — full ActorAttack structural contract | Implemented | `probe_actor_attack_pre_attack`, `_classify_actor_payload`, `_classify_queries`; T1–T15, T39–T40 |
| C2 — nine-message continuity transcript | Implemented | `probe_target`, `_serialise_transcript`; T19–T22 |
| C3 — corrected Clopper–Pearson wording | Implemented in code/docstrings/report limitations | `probe_target["multi_turn"]["note"]`, `_base_report` limitations; §6 of the design |
| C4 — canonical immutable filename | Implemented | `_persist_report`, `resolve_qualification_report`; T26–T30 |
| C5 — v1 readability vs v2 eligibility | Implemented | `require_qualification`; T31, T32, T33, T45 |
| P1-1 — explicit denominators | Implemented | `attempted_count`, `successful_response_count`, `request_error_count`, `structural_valid_count`, `structural_validity_rate`, `request_error_rate`; T16–T17, T42 |
| P1-2 — separate identities | Implemented | `experiment_config` + `schema_version`/`probe_spec_version`; T25 |
| P1-3 — delivery-type diagnostic + review record | Implemented | five-state enum, `build_review_triggers`, `require_review_decision`; T2–T5, T37, T40, T44 |
| P2-1 — `payload_valid_rate` alias | Implemented as exact alias | T1, `assemble_metrics` |
| P2-2 — consistent field names | Implemented | `schema_version`, `probe_spec_version` everywhere |
| P2-3 — diagnostics never affect completeness | Implemented | `probes_status` iterates `MANDATORY_PROBES` only; T36 |
| P2-4 — `NOT_EXECUTED`/`FAIL`/`INSUFFICIENT_DATA` block | Implemented | `final_decision`, gate status check; T35 |
| P2-5 — code-word probe diagnostic-only if retained | Not retained; `diagnostics` block is empty | T36 proves isolation structurally |
| P2-6 — original failed report preserved | Verified | SHA-256 `b17d392a…` unchanged; T38 |
| P2-7 — consistent schema names and fixtures | Implemented | new suite and updated fixtures |

Frozen parameters (unchanged): 30 ActorAttack samples, ≥27/30, `<0.50` review trigger, 5
continuity conversations, ≥4/5, refusal counts as failure, exact R1 system prompt, 8 turns, 200
HarmBench denominator, NBF checkpoint and η.

---

## 3. Pre-implementation Git status

The session began with an already-dirty working tree (121 porcelain entries in the session
snapshot): the Phase 22–38 changes, the frozen R1 configuration, the acquired utility datasets, and
the historical qualification report. Nothing was reverted, cleaned, stashed, or reformatted.

This phase touched only the files listed in §4. `git status` continues to show the pre-existing
entries as before.

---

## 4. Files modified and added

**Modified**

| File                                        | Change                                                                                      |
| ------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `src/guardbound/attacks/actor_attack.py`    | optional `diagnostics` out-parameter on `extract_harm_target`, `get_actors`, `infer_single` |
| `scripts/cloud_model_qualification.py`      | v2 probes, v2 report identity, canonical persistence, review triggers                       |
| `scripts/_cloud_common.py`                  | newest-v2 resolver, v2-only gate, review-decision validation, source-config hash            |
| `scripts/cloud_pilot.py`                    | `--review-decision`, review authorization check, manifest fields                            |
| `configs/cloud_qualification.yaml`          | declared 30/5/0.50; continuity threshold 0.80                                               |
| `configs/qualification_r1_gpt4o.yaml`       | continuity threshold 0.90 → 0.80 (Phase 40B consistency fix)                                |
| `docs/DELIVERABLE_1_EXECUTION_RUNBOOK.md`   | canonical filename, resolver, v1 ineligibility, v2 gate, review decision                    |
| `tests/test_cloud_qualification_harness.py` | v2 report glob and explicit system prompt                                                   |
| `tests/test_phase24_p0_fixes.py`            | v2 probe calls, glob, assertions                                                            |
| `tests/test_phase26_qualification_gate.py`  | v2 report globs                                                                             |
| `tests/test_phase27_fail_closed_gate.py`    | v2 synthetic probe fixtures                                                                 |
| `tests/test_phase36_implementation.py`      | v2 report fixture, resolver test                                                            |
| `tests/test_phase38_hardening.py`           | v2 report fixture, runbook assertions                                                       |
| `tests/test_attacker_arms.py`               | v2 report globs                                                                             |

**Added**

| File | Purpose |
|---|---|
| `tests/test_phase41_probe_redesign.py` | T1–T45 plus parity, probe-name, and config-declaration tests |
| `docs/PHASE_41_OFFLINE_IMPLEMENTATION_REPORT.md` | this report |

**Not modified:** `configs/paper_reproduction_r1_gpt4o.yaml`, the HarmBench dataset, the NBF
checkpoint, the v1 qualification report, and any Crescendo/Opposite-day/NBF implementation.

---

## 5. Function-level implementation details

### 5.1 `src/guardbound/attacks/actor_attack.py`

`extract_harm_target(..., diagnostics: dict | None = None)`
* Records `extract_payload_is_mapping` and `details_present`.
* `details_present` distinguishes an absent `details` key from an explicit `null` and an empty
  mapping, which the report contract requires.
* With `diagnostics=None` the return value is byte-identical to the previous implementation.

`get_actors(..., diagnostics: dict | None = None)`
* Records `raw_is_mapping`, `raw_actor_count`, `rejected_non_mapping`,
  `rejected_missing_actor_name`, `rejected_invalid_actor_name`, `kept`.
* The filtering logic itself is unchanged; the sink observes it.
* With `diagnostics=None` the function returns the same `(actors, messages)` as before.

`infer_single(..., diagnostics: dict | None = None)`
* Forwards the sink to both helpers; the returned structure is unchanged.

### 5.2 `scripts/cloud_model_qualification.py`

Constants: `SCHEMA_VERSION = "qualification.v2"`, `PROBE_SPEC_VERSION = "r1-probe.v2"`,
`ACTOR_ATTACK_PROBE_SAMPLES = 30`, `CONTINUITY_CONVERSATIONS = 5`,
`DELIVERY_TYPE_REVIEW_THRESHOLD = 0.50`, `CONTINUITY_SYSTEM_PROMPT`, `CONTINUITY_TURNS` (four
natural-fact turns), `CONTINUITY_TOKEN = "miso"`, `CONTINUITY_MESSAGE_COUNT = 9`, `PROBE_SPECS`.

`probe_actor_attack_pre_attack(llm, goals, *, temperature, samples=None)`
* Deterministic prefix of the goal list, in file order.
* Per sample: `extract_harm_target` → `get_actors(actor_num=3, diagnostics=sink)` → static
  validation → `get_init_queries` for each kept actor → query validation.
* Failure order is deterministic: `not_mapping`, `no_actors`, `actor_not_mapping`,
  `missing_actor_name`, `invalid_actor_name`, `missing_relationship`, `invalid_relationship`,
  `invalid_query_details`, `empty_chain`, `non_string_query`, `empty_query`, `wrong_actor_count`.
* Request errors are caught separately and never counted as malformed payloads.
* Returns `attempted_count`, `successful_response_count`, `request_error_count`,
  `structural_valid_count`, `structural_validity_rate`, `empty_chain_rate`, `request_error_rate`,
  `payload_valid_rate` (alias), `delivery_type_valid_rate`, `delivery_type_present_rate`,
  `samples`, `latency_tokens`.
* No goal text, extracted target, generated query, or delivery-type text is retained.

`probe_target(llm, goal_cases, repeats, *, target_temperature, system_prompt, conversations)`
* Raises `ValueError` if `system_prompt` is missing or empty.
* The dataset completion block is preserved.
* Runs `conversations` independent conversations; each starts with the exact system message and
  appends four user turns and four assistant replies.
* Returns `attempted_count`, `successful_conversation_count`, `request_error_count`, `pass_count`,
  `refusal_count`, `incorrect_count`, `error_count`, `context_recall_rate` (alias
  `continuity_rate`), `request_error_rate`, `conversations`, `system_prompt_sha256`,
  `system_prompt_sent`, `message_count`.
* Each conversation stores the nine-message transcript, `final_message_index`, `final_reply`,
  `outcome`, and `request_error`.

`utc_run_id`, `build_review_triggers`, `_experiment_config_block`, `annotate_probes`
* UTC `%Y%m%dT%H%M%SZ`; review trigger block; experiment-config path/hash; per-probe `mandatory`
  and `probe_spec` annotations.

`_base_report(..., run_id, config_path)`
* Adds `schema_version`, `probe_spec_version`, `run_id`, `experiment_config`, `review_triggers`,
  and the corrected continuity/delivery-type limitations.

`_persist_report(..., run_id)`
* Builds `<tag>.<arm>.<run_id>` and writes `<stem>.qualification.v2.json`.
* Raises `FileExistsError` before writing if the path exists.
* Appends `schema_version`, `probe_spec_version` and `run_id` to `qualification.jsonl`.

`main`
* Paper configurations force 30 ActorAttack samples and 5 continuity conversations; non-paper
  configurations may declare alternatives.
* Loads 30 goals once and reuses the first three for the completion probe.
* Records review triggers and annotates probes in the persisted report.

### 5.3 `scripts/_cloud_common.py`

* `load_config` records `_source_path` and `_source_sha256` for configuration binding.
* `expected_qualification_identity` adds `experiment_config_sha256`.
* `resolve_qualification_report` globs `<tag>.<arm>.*.qualification.v2.json`, selects the newest
  run id, and raises on same-timestamp ambiguity.
* `require_qualification` rejects legacy/unknown schema, missing/wrong probe spec, non-`QUALIFIED`
  verdicts, `probes_complete != true`, missing or non-`PASS` mandatory probes, stale identities,
  and a missing/mismatched `experiment_config.sha256`.
* `load_review_decision` validates the six required fields and the decision/eligibility
  consistency rule.
* `require_review_decision` blocks authorization when a trigger is `true` and no decision exists.

### 5.4 `scripts/cloud_pilot.py`

* Adds `--review-decision`.
* Calls `require_review_decision` immediately after the qualification gate and before the API key.
* Records `qualification` and `review_decision` in the run manifest.

---

## 6. ActorAttack contract implementation

The contract follows Phase 40B §3.2:

| Condition | Report behaviour |
|---|---|
| top-level payload not a mapping | `not_mapping`, structural fail |
| `actors` missing/not a list/empty | `no_actors`, structural fail |
| actor entry not a mapping | `actor_not_mapping` + `rejected_non_mapping` |
| `actor_name` absent | `missing_actor_name` + diagnostic count |
| `actor_name` not a non-empty string | `invalid_actor_name` |
| `relationship` absent | `missing_relationship` |
| `relationship` not a non-empty string | `invalid_relationship` |
| zero queries for an actor | `empty_chain` |
| query not a string | `non_string_query` |
| query empty/whitespace | `empty_query` |
| actor count ≠ 3 | `wrong_actor_count` |
| `query_details` non-mapping, non-null | `invalid_query_details` |
| `query_details` absent/null | pass, normalised for execution |
| `delivery_type` missing/null/empty/invalid/valid | pass, diagnostic only |

Seven `query_details` cases are implemented exactly: absent → `absent`; explicit null → `null`;
mapping with key absent → `mapping` + `missing`; mapping with null → `mapping` + `null`; mapping
with empty string → `mapping` + `empty`; mapping with non-empty string → `mapping` + `valid`;
non-mapping → `invalid` + structural failure.

---

## 7. Continuity transcript implementation

* System prompt: `You are a helpful assistant.` (28 bytes, SHA-256
  `75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de`).
* Four user turns: pet facts, arithmetic, primary colour, cat recall.
* Four assistant replies retained; the list is re-sent every turn.
* Exactly nine messages; the final reply is stored both as `messages[8]` and as the convenience
  field `final_reply`, with `final_message_index: 8`.
* Outcomes: `pass` (token `miso`), `incorrect`, `refusal` (via
  `guardbound.evaluation.refusal_detector.is_refusal`), `error`.
* Refusals count as failed continuity; request errors are excluded from the continuity rate and
  gated separately at ≤ 0.05.
* No persistent-memory assumption appears in the prompts or the report.

---

## 8. v2 schema and identity

Every report contains:

* `schema_version = "qualification.v2"`.
* `probe_spec_version = "r1-probe.v2"`.
* `run_id` (UTC `%Y%m%dT%H%M%SZ`).
* `experiment_config = {path, sha256}`.
* Existing model, prompt, temperature, structured-output and threshold identities.
* `probes[*].mandatory` and `probes[*].probe_spec`.
* `review_triggers.delivery_type` with threshold, observed, triggered, null operator fields.
* `probe_status`, `probes_complete`, `overall`, `limitations`, `environment`, `git_*`, `usage`.

The qualification implementation identity is separate from the experiment configuration identity:
changing a probe changes `probe_spec_version`, not the R1 config hash.

The paper stack's `attacker_prompt_condition = "baseline"`, `prompt_variant = "official"`, all
temperatures 0.7, and `structured_output = "json_object"` are read from the frozen configuration,
not re-declared in code.

---

## 9. Immutable persistence and resolver

* Writer: `<tag>.<arm>.<run_id>.qualification.v2.json`.
* `FileExistsError` before any write when the path exists.
* Append-only `qualification.jsonl` row with schema, probe spec and run id.
* Resolver: newest canonical match by `run_id`.
* Same `run_id` in more than one match → `QualificationError` requiring an explicit path.
* Newest report failing validation → raise; no older report is tried.
* Legacy fixed-name v1 report is never a resolver candidate and is never modified.

---

## 10. Gate validation behaviour

Accepted only when all hold:

1. `schema_version == "qualification.v2"`.
2. `probe_spec_version == "r1-probe.v2"`.
3. `overall == "QUALIFIED"`.
4. `probes_complete is True`.
5. Every entry of `MANDATORY_PROBE_NAMES` exists with `status == "PASS"`.
6. Models, prompt variant, attacker condition, temperatures and structured output match.
7. `experiment_config.sha256` is present, matches the referenced file when it exists, and matches
   the loaded configuration's own hash when available.

Rejected: v1 reports, unknown schema versions, unknown probe specs, missing mandatory probes,
`NOT_EXECUTED`, `FAIL`, `INSUFFICIENT_DATA`, `probes_complete: false`, stale identity, missing
experiment config hash. Diagnostics are never consulted.

---

## 11. Review-trigger behaviour

* `delivery_type_valid_rate < 0.50` sets `review_triggers.delivery_type.triggered = true` in the
  immutable report.
* `operator_decision`, `rationale` and `actor_attack_eligible` remain null in the report.
* A separate `--review-decision` JSON record supplies the operator decision; its schema is
  validated, including the `exclude_actor_attack ⇔ eligible=false` rule.
* A triggered review with no decision blocks a paid experiment before the API key is read.
* The metric itself never changes the qualification verdict.

---

## 12. Tests executed, commands and results

Environment: Windows, Python 3.13. Test runs set `KMP_DUPLICATE_LIB_OK=TRUE` because this machine
loads two OpenMP runtimes; without it Python aborts before pytest starts. This is an environment
workaround, not a code change.

### Phase 41 suite

```
python -m pytest tests/test_phase41_probe_redesign.py -q
```

**48 passed in 0.25s** — T1–T45 plus three structural tests (diagnostics parity, mandatory-name
agreement between harness and gate, and config declarations).

### Focused qualification suites

```
python -m pytest tests/test_phase41_probe_redesign.py \
  tests/test_cloud_qualification_harness.py \
  tests/test_phase24_p0_fixes.py \
  tests/test_phase26_qualification_gate.py \
  tests/test_phase27_fail_closed_gate.py \
  tests/test_phase36_implementation.py \
  tests/test_phase38_hardening.py \
  tests/test_attacker_arms.py -q
```

**282 passed in 6.65s.**

### Full offline suite

```
python -m pytest tests/ -q
```

**1384 passed, 4 failed, 0 skipped in 179.43s.**

The four failures are environment checks unrelated to this phase:

| Test | Cause |
|---|---|
| `test_phase14_2_rotation.py::…::test_releases_cache_before_measuring` | this machine's VRAM/cache-release condition |
| `test_phase14_2_rotation.py::…::test_effective_free_after_release_is_what_is_measured` | same |
| `test_phase8_architecture.py::…::test_default_config_valid` | `Qwen/Qwen3.5-4B` is not in the local HF cache |
| `test_phase8_architecture.py::…::test_attacker_present` | same missing local checkpoint |

They do not import or exercise the qualification path. No failure is hidden: the full suite was
executed, and these four are reported as environment exclusions, not as passes.

No test used a network transport, an API key, or model inference.

---

## 13. Compatibility and parity evidence

* `get_actors(..., diagnostics=None)` and `extract_harm_target(..., diagnostics=None)` are
  byte-identical to the previous behaviour; the sink is only written when provided.
* `infer_single(..., diagnostics=None)` returns the same structure; a dedicated test asserts
  `infer_single(..., diagnostics=sink) == infer_single(...)` and the same for `get_actors`.
* Existing ActorAttack tests pass unchanged, including the composite runner and temperature
  propagation (Phase 36).
* Crescendo and Opposite-day files were not touched; their tests pass.
* NBF loading, checkpoint, η and the variant invariants were not modified.
* Offline-mock and dry-run workflows still work; the qualification harness's offline tests pass.
* A v1 report remains readable (T31) but is rejected by the gate with an explicit legacy message.

---

## 14. Historical report preservation

`results/qualification/gpt-4o-2024-08-06.baseline.qualification.json`

* SHA-256 `b17d392a39eeeda64d561017073addd88376d3dafa394bd732c1ba2e908aeaeb` — unchanged.
* Still a v1 document: no `schema_version`, `overall = NOT_QUALIFIED`,
  `probes_complete = false`.
* No migration, rewrite, or rename occurred.

---

## 15. R1 configuration, dataset and checkpoint hashes

| Artifact | SHA-256 | Status |
|---|---|---|
| `configs/paper_reproduction_r1_gpt4o.yaml` | `2125cc3b233efde9e97a8407c364c2140091245dd39bcf440c4366fd99ef393b` | unchanged |
| HarmBench `harmbench_tasks.json` | `ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb` | unchanged |
| NBF checkpoint | `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` | unchanged |

The R1 configuration hash matches the frozen identity recorded in Phase 40B.

---

## 16. Remaining limitations and unresolved issues

1. **Code-word diagnostic not retained.** The Phase 40B design allows it as diagnostic-only. It was
   removed with the old memory-policy probe; the `diagnostics` block is empty. T36 proves
   structurally that any diagnostic cannot affect `probes_complete`.
2. **`extract_harm_target` gained an optional diagnostics parameter.** Phase 40B §10.2 listed only
   `get_actors` and `infer_single`; distinguishing an absent `details` key from explicit null
   requires observing the raw extraction payload. The parameter defaults to `None` and does not
   change the attack path.
3. **`configs/qualification_r1_gpt4o.yaml` threshold corrected from 0.90 to 0.80.** This is a
   qualification config, not the frozen experiment config. Leaving it at 0.90 would silently apply
   a stricter gate than the Phase 40B frozen `≥4/5`. The R1 experiment config is untouched.
4. **`run_id` has second resolution.** Two runs in the same second collide and the writer refuses
   the second. This is intentional immutability behaviour, not a retry.
5. **Review-decision enforcement applies to paid runs.** `--dry-run` and `--offline-mock` skip the
   qualification gate and therefore the review check; they create no paid client.
6. **`probes_complete` remains CLI-only.** The Phase 38 P2-2 residual is unchanged: a caller that
   bypasses `require_qualification` is not stopped by a shared factory.
7. **Probe cost.** A full run is about 270 attacker calls for the ActorAttack probe plus the
   existing target/evaluator probes; the Phase 40B estimate is ~US$0.7–1.0. No run was executed.
8. **Continuity is a smoke gate.** `≥4/5` is not an 80%-reliability claim; the report's
   `limitations` block states this explicitly.
9. **Full-suite environment failures.** The four failures in §12 are machine-specific and predate
   this phase; they were not repaired because doing so is outside the approved scope.

---

## 17. Confirmation of no external activity

* No OpenAI or other external model API was called.
* No API key was read, printed, copied, or validated.
* No qualification command was executed against a provider.
* No ActorAttack, Crescendo, Opposite-day, Acronym, RedQueen or research experiment was run.
* No model weights were downloaded or loaded for inference; the NBF local probes in tests were
  monkeypatched or used the existing offline fixtures.
* No dataset or checkpoint was downloaded or modified.
* No credential is present in this report or in any created file.

---

## 18. Final verdict

# `READY FOR PHASE 42 INDEPENDENT OFFLINE AUDIT`

The Phase 40B contracts are implemented as specified, the frozen R1 artifacts are unchanged, the
v1 report is preserved, and the Phase 41 regression suite passes in full. The four full-suite
failures are pre-existing environment checks outside the qualification path and are reported
explicitly rather than masked.

No qualification run, paid call, or Deliverable 1 execution is authorized by this report.
