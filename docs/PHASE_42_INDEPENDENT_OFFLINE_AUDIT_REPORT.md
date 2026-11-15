# PHASE 42 — INDEPENDENT OFFLINE AUDIT OF R1 QUALIFICATION V2

**Audit date:** 2026-11-12
**Auditor posture:** independent, read-only verification of the Phase 41 implementation against
Phase 40B.
**Verdict:** `PASS WITH FINDINGS — CORRECTIONS REQUIRED BEFORE AUTHORIZATION`

A passing audit does **not** authorize a qualification run, a paid API call, or Deliverable 1.

---

## 1. Executive summary

The Phase 41 implementation is substantially present and the R1 gate is structurally fail-closed.
The v2 schema, immutable naming, newest-report resolver, legacy-v1 rejection, mandatory-probe
validation, identity binding, continuity transcript, and review-decision plumbing all behave as
specified in the cases I executed.

The audit nevertheless found **one P0 bypass route, three P1 defects, and three P2 weaknesses** that
must be corrected before any authorization:

1. **P0-1 — paid execution outside `cloud_pilot.py` bypasses qualification.** Only
   `cloud_pilot.main` calls `require_qualification`; `cloud_smoke_test.py`, `run_attacks.py`,
   `run_baseline_evals.py`, the `eval_*` scripts, `judge_asr.py`, `run_full_evaluation.py`,
   `run_reproduction.py`, and `demo_steered_chat.py` construct `OpenAIChatLLM` directly. The R1
   Deliverable 1 path is gated; the repository-wide "no paid client before qualification" claim is
   false.
2. **P1-1 — `delivery_type_valid_rate` and `delivery_type_present_rate` can exceed 1.0.** The
   numerator is incremented before the query-generation request-error check, so a valid-delivery
   sample that errors during `get_init_queries` is counted in the numerator but excluded from the
   denominator. I reproduced `2.0` for 2 attempted / 1 successful / 1 request error. This can
   suppress the `<0.50` review trigger.
3. **P1-2 — duplicate actor names yield fewer than three actor conversations but a structural
   pass.** `get_actors` deduplicates by `actor_name`, while `_classify_actor_payload` checks the
   raw count. A payload with three entries and two distinct names reports `actor_count: 3`,
   `queries_per_actor: [n, n]`, and `structural_valid: true`.
4. **P1-3 — the review decision is not bound to the report/run.** `require_review_decision` checks
   only that the decision's `trigger` is in the triggered set. A stale `accept_degradation` record
   from an earlier run is accepted for a later triggered report. The manifest records the pair, but
   the code does not enforce freshness.
5. **P2-1 — malformed `questions` payloads are silently normalized or filtered.** A string
   `questions` becomes one-character queries; a dict becomes its keys; non-string list items are
   dropped. The probe reports structural validity because the resulting queries are strings.
6. **P2-2 — a non-iterable `questions` value is misclassified as a request error**, not a
   structural failure, and is excluded from the structural denominator.
7. **P2-3 — the gate does not require `run_id` or the `review_triggers` block.** A v2 report that
   omits either is accepted; omitting `review_triggers` silently disables the review requirement.

The continuity probe, report persistence, resolver, frozen artifacts, and the Phase 41 test suite
passed the audit. The four full-suite failures were independently reproduced and their immediate
causes verified as environment limitations (CPU-only torch; missing `Qwen/Qwen3.5-4B` cache). The
claim that they are historically pre-existing is **unverified**, but the failing files are not part
of the Phase 41 working-tree diff.

---

## 2. Audit scope and restrictions followed

**Read:** the Phase 40B specification (`docs/PHASE_40_QUALIFICATION_PROBE_REDESIGN.md`), the
preserved v1 design, the Phase 41 report, all named implementation files, the configs, the runbook,
and the qualification/gate/attack regression tests.

**Executed:** offline pytest suites and local scripted fakes only. No network transport, no provider
call, no credential read, no model load for inference, no research attack, no dataset or checkpoint
download.

**Not done:** no repository file was modified except the creation of this report. No fix was
implemented. No qualification command was run against a provider. No paid or free model inference
occurred. The scratch reproduction scripts used by the audit live in the session scratchpad, outside
the repository.

**Limitation:** the working tree has no Phase 41 commit boundary. `git diff` is cumulative against
the last commit and includes Phases 22–38. Phase 41 attribution therefore relies on the Phase 41
report, the session snapshot (the relevant files were already dirty before Phase 41), and the
identifiable diagnostics hunks.

---

## 3. Phase 40B compliance matrix

| Requirement | Status | Notes |
|---|---|---|
| C1 — complete ActorAttack structural contract | **Partial — P1-2, P2-1, P2-2** | Most clauses implemented; exactly-three is checked on the raw count, not the post-filter count; malformed `questions` is normalized/misclassified |
| C2 — nine-message continuity transcript | **Pass** | T19–T22, T41–T42; P3 wording note on `final_reply` |
| C3 — corrected Clopper–Pearson statements | **Pass** | 0/5 → 45.1% success upper / 54.9% failure lower; 30/30 → 90.5%; 4/5 smoke gate labelled |
| C4 — canonical immutable filename | **Pass** | Writer, collision check, JSONL, resolver; P3 note on unreachable duplicate-timestamp fixture |
| C5 — v1 readability vs v2 eligibility | **Pass** | v1 rejected explicitly; unknown schema/spec rejected |
| P1-1 — explicit denominators | **Partial — P1-1, P2-2** | Structural/request counts correct; delivery-type rates and malformed-`questions` classification are wrong |
| P1-2 — separate identities | **Pass** | `schema_version`, `probe_spec_version`, `experiment_config.sha256` |
| P1-3 — delivery diagnostic + review record | **Partial — P1-1, P1-3, P2-3** | Trigger separation correct; denominator and decision binding are not |
| P2-1 — `payload_valid_rate` alias | **Pass** | Exact alias of `structural_validity_rate` |
| P2-2 — consistent field names | **Pass** | `schema_version` / `probe_spec_version` used consistently |
| P2-3 — diagnostics never affect completeness | **Pass** | `probes_status` iterates mandatory names only; T36 |
| P2-4 — non-PASS statuses block | **Pass** | T35, T43 |
| P2-5 — code-word probe diagnostic-only if retained | **Pass** | Not retained; `diagnostics` is empty; isolation tested structurally |
| P2-6 — original failed report preserved | **Pass** | SHA-256 `b17d392a…` unchanged; T38 |
| P2-7 — consistent schema names/fixtures | **Pass** | New suite and updated fixtures |

Frozen values verified: 30 samples, ≥27/30, `<0.50` trigger, 5 conversations, ≥4/5, refusal counts
as failure, exact system prompt, 8 turns, 200-goal denominator, NBF checkpoint and η.

---

## 4. ActorAttack contract audit

### 4.1 What passed

* The probe performs the specified sequence: `extract_harm_target` → `get_actors` → static
  validation → `get_init_queries` → query validation.
* `get_actors` records the raw pre-filter counts through the optional sink; the filter logic itself
  is unchanged, and `diagnostics=None` preserves the returned `(actors, messages)`.
* `extract_harm_target` distinguishes absent `details` from explicit `null` and a mapping.
* The five `delivery_type` states (`missing`, `null`, `empty`, `invalid_type`, `valid`) are
  reported without gating.
* The seven `query_details` cases are implemented; the non-mapping case fails structurally.
* Missing `actor_name`, `relationship`, empty chains, non-string queries, empty queries,
  `wrong_actor_count`, and `invalid_query_details` have deterministic failure classes.
* Request errors are counted separately from structural failures, and zero successful responses
  yield a `None` rate and `INSUFFICIENT_DATA`.
* The 30-goal prefix is read in file order; `goal_index` follows it; no goal text is retained.
* `payload_valid_rate` is the exact structural-rate alias.
* The `delivery_type` diagnostic is not consulted by the qualification decision.

### 4.2 Reproduced defects

I used a read-only scripted `ChatLLM` fake (session scratchpad) with the same prompt
detection as the test suite.

**P1-2 — duplicate actor names.** Payload:

```json
{"actors": [
  {"actor_name": "a", "relationship": "r"},
  {"actor_name": "a", "relationship": "r"},
  {"actor_name": "b", "relationship": "r"}
]}
```

Observed:

```
structural_valid True
failure_reason None
actor_count 3
queries_per_actor [1, 1]
request_error False
```

`get_actors` deduplicates to two actors, but `_classify_actor_payload` checks the raw count (3) and
`_classify_queries` receives `actor_count=3`. The execution would run only two actor conversations.
The probe therefore awards a structural pass to a payload that violates the exactly-three execution
requirement.

**P2-1 — `questions` string.** `{"questions": "abcd"}` produced
`queries_per_actor: [4, 4, 4]` and `structural_valid: true`: the parser iterated the string into
four one-character queries. A dict `questions` value similarly yields its keys. A mixed list
`["ok", 3]` produced `[1, 1, 1]` and `structural_valid: true`: the integer was silently dropped.
The final queries are strings, so the literal row-7 check passes, but the model did not produce the
required query list.

**P2-2 — non-iterable `questions`.** `{"questions": 5}` raised `TypeError` inside
`get_init_queries`, was caught as a request error, and produced:

```
structural_valid False
failure_reason request_error
request_error True
request_error_rate 1.0
```

The malformed payload is excluded from `successful_response_count` and from the structural
denominator. A single such sample in a 30-goal run inflates `request_error_rate` while hiding the
structural failure; frequent occurrences fail the independent request-error gate.

### 4.3 Diagnostics and parity

* `get_actors(..., diagnostics=None)` and `infer_single(..., diagnostics=None)` are behaviorally
  unchanged; the new parity test asserts equality of returned structures with and without the sink.
* The sink adds `raw_is_mapping`, `raw_actor_count`, `rejected_invalid_actor_name`, and `kept`
  beyond the Phase 40B wording (`raw_items`, `rejected_non_mapping`,
  `rejected_missing_actor_name`, `kept`). The extra counts are benign; the raw items themselves are
  not retained, which is consistent with the report's retention rule.
* The `extract_harm_target` diagnostics parameter is not in the Phase 40B §10.2 inventory. It is
  necessary to distinguish absent from null `details`, defaults to `None`, and is covered by the
  parity test. **Pass with a provenance note (P3-5).**

---

## 5. Target continuity audit

**Pass.**

* Exact system prompt `You are a helpful assistant.` at index 0; 28 bytes; SHA-256
  `75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de` (verified independently).
* Five independent conversations for paper configs; four user turns and four assistant replies per
  conversation.
* Nine messages exactly; roles `system,user,assistant,user,assistant,user,assistant,user,assistant`.
* The full accumulated history is re-sent on every turn (`list(messages)`).
* All four assistant replies are retained; the final reply is at index 8.
* Outcomes: token present → `pass`; absent and `is_refusal` → `refusal`; absent otherwise →
  `incorrect`; exception → `error`.
* Refusals count as continuity failures; request errors are excluded from the continuity
  denominator (`successful_conversation_count = attempted − request_error_count`) and gated
  separately.
* The dataset completion block is preserved; `goal_cases` and `repeats` still drive it.
* No persistent-memory framing remains; the script uses natural pet/arithmetic/colour/recall turns.
* Threshold is exactly 0.80 (≥4/5) in `DEFAULT_THRESHOLDS` and both shipped configs. The report's
  limitations and the probe note state that this is a five-sample smoke gate, not an 80%
  reliability claim.

**P3-1 — `final_reply` wording.** In the in-memory transcript the final assistant message dict is
the same object referenced by `messages[-1]`; in the serialized JSON, `final_reply` and
`messages[8].content` are two equal string values, not an identity reference. The test asserts
equality, which is the meaningful contract. The design/report should say "copy/equal value" rather
than "reference" if object identity is not intended.

**P3-10 — top-level `message_count: 9`.** The top-level field is the design constant; a conversation
that errors mid-turn has a shorter transcript, recorded in its per-conversation `message_count`.
The error is visible, but a consumer reading only the top-level field could overstate completeness.

---

## 6. Schema, persistence, resolver, and identity audit

### 6.1 Passed

* `schema_version == "qualification.v2"` and `probe_spec_version == "r1-probe.v2"` are written by
  `_base_report` and checked by the gate.
* `run_id` uses UTC `%Y%m%dT%H%M%SZ`; the test verifies sortable ordering.
* The writer builds `<tag>.<arm>.<run_id>.qualification.v2.json`; it checks `path.exists()` before
  `_flush_usage` and `write_json`, so an existing report is refused before any write/append.
* The JSONL row includes `schema_version`, `probe_spec_version`, and `run_id`.
* The resolver globs `<tag>.<arm>.*.qualification.v2.json`, sorts by the parsed run id, and returns
  the newest.
* `_select_newest_report` raises when two matches share a run id.
* A newest report that fails validation raises; no older report is tried.
* A v1 report (no `schema_version`) is rejected with an explicit legacy message; unknown schema and
  probe-spec versions are rejected.
* Probe annotations (`mandatory`, `probe_spec`) are present; diagnostics are not in
  `probe_status` and cannot affect `probes_complete`.
* The gate checks the mandatory set, `PASS` statuses, `probes_complete is True`, the verdict, and
  the identity fields.
* `experiment_config.sha256` is checked both against the referenced file (when it exists) and
  against the loaded configuration's `_source_sha256`.

### 6.2 Weaknesses

**P2-3 — missing `run_id` and `review_triggers` are not rejected.** A v2 report with the required
schema/spec/statuses but no `run_id` and no `review_triggers` passes `require_qualification`.
`require_review_decision` then sees an empty trigger map and returns `None`, silently disabling the
review. This is a tamper-evident/provenance gap, not an automatic false QUALIFIED, because the rest
of the gate still applies. Correction: require both fields; validate the run-id format and its
consistency with the filename; require the `delivery_type` trigger block.

**P3-2 — the same-timestamp resolver test is a unit test, not an end-to-end scenario.** With
canonical names in one directory, two reports with the same run id collide at the filename level
(the writer refuses the second). The duplicate check is defensive and correct, but the fixture
calls `_select_newest_report` directly; the resolver's glob cannot produce the duplicate. Coverage,
not behavior, is the issue.

**P3-8 — same-second re-run produces a traceback.** `_persist_report` raises `FileExistsError`;
`main` does not catch it. The report is protected, but the CLI error is not operator-friendly.

---

## 7. Paid-run gate and review-decision audit

### 7.1 Passed

* In `cloud_pilot.main`, `require_qualification` is called at line 518, before `require_api_key`
  (line ~526), `build_ledger` (line ~525), `load_nbf`, and `build_role_llms` (line 541). No paid
  client is constructed before the gate.
* `require_review_decision` is called immediately after the qualification gate and before the API
  key. A triggered review with no decision raises; a review decision cannot override a failed
  qualification because the qualification call raises first.
* `exclude_actor_attack` ⇔ `actor_attack_eligible: false` is enforced; the reverse is enforced for
  the other decisions.
* The manifest records the `qualification` block (report, SHA-256, run id, experiment config,
  review triggers) and the parsed `review_decision`.
* `--dry-run` returns before any client; `--offline-mock` injects `OfflineMockClient` and skips the
  gate. Neither creates a paid client or claims qualification.
* Existing fail-closed behavior is intact in `cloud_pilot`.

### 7.2 Findings

**P0-1 — repository-wide paid bypass.** The gate is only in `cloud_pilot.py`. The following
scripts construct a paid `OpenAIChatLLM` directly without calling `require_qualification`:

| Script | Lines |
|---|---|
| `scripts/cloud_smoke_test.py` | 137–145 |
| `scripts/run_attacks.py` | 212–213, 240–241, 252–253 |
| `scripts/run_baseline_evals.py` | 79–80, 222–223 |
| `scripts/judge_asr.py` | 47–52 |
| `scripts/eval_mmlu.py` | 50–51 |
| `scripts/eval_mtbench.py` | 52–53 |
| `scripts/eval_over_refusal.py` | 48–49 |
| `scripts/run_full_evaluation.py` | 99–118 |
| `scripts/run_reproduction.py` | 302–303 |
| `scripts/demo_steered_chat.py` | 85 |

`cloud_smoke_test.py` explicitly requires `--confirm-paid-request` but has no qualification check.
The Phase 36 claim "no paid client may be constructed until a QUALIFIED report" is therefore true
only for `cloud_pilot`. This is pre-existing and was previously documented as the Phase 38 P2-2
residual, but it is a live P0 route: `run_attacks.py`, `run_baseline_evals.py`, and the `eval_*`
scripts can spend on the R1 models without any qualification report. Correction: centralize the
gate in the paid-client factory or add it to each paid entry point; if the smoke test is a
deliberate exception, it needs its own separate authorization and documentation.

**P1-3 — decision not bound to the report.** `load_review_decision` validates six fields and the
eligibility rule. `require_review_decision` then checks only `decision["trigger"] in required`.
There is no `qualification_report_sha256`, `run_id`, `configuration_sha256`, or `arm` binding. A
decision file from an earlier triggered run can be supplied for a later triggered report. The
manifest records the pair, so the misapplication is visible after the fact, but it is not prevented.
Correction: add a report/run binding to the decision schema and verify it; alternatively require
the decision to echo the observed rate and threshold.

**P3-6 — manifest lacks the decision file path/hash.** The parsed decision content is recorded, but
not the source path or file hash. Correction: record `review_decision_path` and
`review_decision_sha256`.

**P3 observation — a decision supplied when no trigger exists is silently ignored.**
`require_review_decision` returns `None` without reading the file; the manifest records `null`. If
the operator intended the decision to be recorded, it is dropped. Not a safety issue; a provenance
clarity issue.

---

## 8. Frozen configuration and artifact hash verification

Independently computed with SHA-256:

| Artifact | Expected | Observed | Status |
|---|---|---|---|
| `configs/paper_reproduction_r1_gpt4o.yaml` | `2125cc3b…393b` | `2125cc3b…393b` | **match** |
| HarmBench `harmbench_tasks.json` | `ac789de8…14eb` | `ac789de8…14eb` | **match** |
| NBF checkpoint | `cea1a75b…36fe` | `cea1a75b…36fe` | **match** |
| Historical v1 qualification report | `b17d392a…aeb` | `b17d392a…aeb` | **match** |
| Historical Phase 40 v1 design | `0f9371da…7a58` | `0f9371da…7a58` | **match** |

The historical report remains a v1 document (`schema_version` absent, `overall` `NOT_QUALIFIED`,
`probes_complete` false) and was not modified.

**`configs/qualification_r1_gpt4o.yaml` change (0.90 → 0.80).** Phase 40B §4.5 freezes the
continuity gate at ≥4/5 = 0.80; the old 0.90 required 5/5 and contradicted the approved threshold.
The change is justified for this qualification config. It does not touch the frozen experiment
configuration or any R1 model/dataset/attack value. **P3-4** records that the edit is outside the
Phase 40B §9 file inventory but is a necessary consistency fix; the Phase 41 report discloses it.

---

## 9. Git diff and working-tree assessment

* Working tree: **122 porcelain entries** at audit time. This includes the pre-existing Phase 22–38
  changes, the new Phase 41 report, and the Phase 41 files. Nothing was cleaned, stashed, reset, or
  reverted.
* Tracked diff: 18 files changed, 1989 insertions, 207 deletions (cumulative since the last
  commit). The diff is **not** a Phase 41-only diff.
* `src/guardbound/attacks/actor_attack.py` was already modified before Phase 41. Its cumulative
  diff contains the diagnostics hunks (`extract_harm_target`, `get_actors`, `infer_single`) plus
  changes from prior phases (temperature threading, rubric/prompt work). The Phase 41 report
  attributes only the diagnostics to Phase 41; that attribution is consistent with the session
  snapshot but cannot be proven from git alone. **P3-7** records this audit-provenance limitation.
* `test_phase14_2_rotation.py`, `test_phase8_architecture.py`, and
  `src/guardbound/llm/model_manager.py` are **not** in the working-tree diff. Their failures are
  therefore not Phase 41 changes.
* No file outside the audit report was modified by this audit.

---

## 10. Independent test commands and exact results

Environment: Windows, Python 3.13, `torch 2.14.1+cpu`, `torch.cuda.is_available() == False`.
`KMP_DUPLICATE_LIB_OK=TRUE` is required because this machine loads two OpenMP runtimes; without it
Python aborts before pytest starts. This is an environment workaround, not a code change.

### 10.1 Phase 41 suite

```
python -m pytest tests/test_phase41_probe_redesign.py -q
```

**48 passed in 0.26s.**

### 10.2 Focused qualification suites

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

**282 passed in 7.29s.**

### 10.3 Full offline suite

```
python -m pytest tests/ -q
```

**1384 passed, 4 failed, 0 skipped in 184.33s.** Exit code 1.

### 10.4 The four failures, reproduced in isolation

```
python -m pytest \
  "tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring" \
  "tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured" \
  "tests/test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid" \
  "tests/test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present" -q
```

**4 failed in 0.17s.** Verified causes:

* `test_phase14_2_rotation` — `pre_run_check` calls `torch.cuda.empty_cache()` only when
  `torch.cuda.is_available()`; this machine is CPU-only (`torch 2.14.1+cpu`, CUDA false), so the
  test's expected call order is not produced. The guard measures free VRAM via the monkeypatched
  function and never calls `empty_cache`.
* `test_phase8_architecture` — the configured attacker `Qwen/Qwen3.5-4B` is absent from the local
  HF cache. The cache contains other Qwen snapshots (`Qwen3-4B-Instruct-2507`, `Qwen3-8B`,
  `Qwen3.6-27B`, `Qwen3.8-27B`) but not `Qwen3.5-4B`.

These files are not in the Phase 41 working-tree diff. The immediate causes are verified as
environment limitations; the claim that they are historically **pre-existing** is **unverified**
(no baseline checkout was run). They are correctly excluded from Phase 41's pass count, and the
Phase 41 report does not claim a green full suite.

### 10.5 Read-only reproduction of the audit findings

Two session-scratchpad scripts (outside the repository) exercised the probe with scripted `ChatLLM`
fakes. Observed outputs:

```
attempted 2
successful 1
errors 1
delivery_valid_rate 2.0
delivery_present_rate 2.0
structural_rate 1.0
```

```
duplicate actor names: structural_valid True, actor_count 3, queries_per_actor [1, 1]
questions string:      structural_valid True, queries_per_actor [4, 4, 4]
questions integer:     failure_reason request_error, request_error True, request_error_rate 1.0
questions mixed list:  structural_valid True, queries_per_actor [1, 1, 1]
```

No repository test was modified to produce these; they are audit evidence only.

---

## 11. Findings

### P0-1 — Paid execution outside `cloud_pilot.py` bypasses qualification

* **Severity:** P0
* **Location:** `scripts/cloud_smoke_test.py:137-145`; `scripts/run_attacks.py:212-213,240-241,252-253`;
  `scripts/run_baseline_evals.py:79-80,222-223`; `scripts/judge_asr.py:47-52`;
  `scripts/eval_mmlu.py:50-51`; `scripts/eval_mtbench.py:52-53`;
  `scripts/eval_over_refusal.py:48-49`; `scripts/run_full_evaluation.py:99-118`;
  `scripts/run_reproduction.py:302-303`; `scripts/demo_steered_chat.py:85`.
* **Expected:** Phase 36 P1-1 / `_cloud_common.QualificationError`: no paid client may be
  constructed without a current QUALIFIED v2 report. Phase 40B §7.6: the R1 gate is fail-closed.
* **Actual:** `require_qualification` is called only in `cloud_pilot.main:518`. The listed scripts
  instantiate `OpenAIChatLLM` directly and can spend without any qualification report.
* **Evidence:** grep of `OpenAIChatLLM(`/`build_role_llms` across `scripts/`; only
  `cloud_pilot.py:541` calls `build_role_llms`, and only `cloud_pilot.py:518` calls the gate.
* **Impact:** the R1 Deliverable 1 path is gated, but the repository-wide paid path is not. A user
  can run the same attacks/baselines/evals on the R1 models without qualification. This is a live
  spend-risk route, not merely a documentation gap.
* **Recommended correction:** centralize the gate in the paid client factory
  (`_cloud_common.build_role_llms` / `build_single_role_llm`) so every paid constructor inherits it,
  or add an explicit gate to each paid entry point. If the smoke test must remain ungated, give it a
  separate authorization path and stop claiming it is covered by the qualification gate.

### P1-1 — Delivery-type rates can exceed 1.0; denominator is computed before the request-error check

* **Severity:** P1
* **Location:** `scripts/cloud_model_qualification.py:1136-1139` (numerator increments),
  `:1141-1156` (query-phase `try`/`continue`), `:1188-1189` (rates over `successful`).
* **Expected:** `delivery_type_valid_rate` and `delivery_type_present_rate` are computed over
  `successful_response_count`; a request error must not contribute to the numerator.
* **Actual:** the numerator is incremented immediately after the payload validation, before
  `get_init_queries`. If a query-generation call raises, the sample is added to `errors` and
  excluded from `successful`, but its delivery-type contribution remains. Reproduced:
  `attempted=2, successful=1, errors=1, delivery_valid_rate=2.0,
  delivery_present_rate=2.0`.
* **Impact:** the `<0.50` review trigger can be suppressed. Example: five invalid-delivery
  successes plus five valid-delivery query errors report `1.0` instead of the correct `0.0`, so no
  review is required. The structural/request-error gates are unaffected.
* **Recommended correction:** compute both rates from the completed sample rows filtered by
  `request_error is False`, or increment the numerators only after the query phase succeeds.

### P1-2 — Duplicate actor names produce fewer than three actor conversations but a structural pass

* **Severity:** P1
* **Location:** `src/guardbound/attacks/actor_attack.py:384-388` (dedup by `actor_name`),
  `scripts/cloud_model_qualification.py:1004-1026` (`_classify_actor_payload` checks
  `row["actor_count"]`), `:1157` (`_classify_queries` receives the raw count), `:1160`
  (`queries_per_actor` reflects the kept list).
* **Expected:** exactly three actors must survive to execution. The probe must record both the raw
  and post-filter counts and fail when fewer than three distinct actors are kept.
* **Actual:** a payload with three entries and two distinct `actor_name` values yields `kept=2`,
  `actor_count=3` (raw), `queries_per_actor` length 2, and `structural_valid=True`. Reproduced.
* **Impact:** a model that cannot produce three distinct actors can pass the mandatory actor probe,
  and the execution runs only two actor conversations. This is an incorrect qualification decision.
* **Recommended correction:** require `diagnostics["kept"] == 3` in addition to
  `raw_actor_count == 3`; record `kept_actor_count` in the sample; classify the difference as
  `wrong_actor_count` or a new `duplicate_actor_name`.

### P1-3 — Review decision is not bound to the report, configuration, arm, or run

* **Severity:** P1
* **Location:** `scripts/_cloud_common.py:456-498` (`load_review_decision` schema),
  `:501-524` (`require_review_decision`).
* **Expected:** a decision record cannot be misapplied to a different triggered report/run.
* **Actual:** the record schema has `trigger`, `decision`, `rationale`, `decided_by`, `decided_at`,
  `actor_attack_eligible`. `require_review_decision` checks only that `decision["trigger"]` is in
  the triggered set. A decision from an earlier run is accepted for a later triggered report.
* **Impact:** unsafe review authorization. The manifest records the report SHA-256 and the parsed
  decision, so the pairing is auditable after the fact, but the code does not enforce freshness.
* **Recommended correction:** add `qualification_report_sha256` and/or `run_id` to the decision
  schema and verify it against the report block; optionally require the observed rate/threshold to
  match. If the Phase 40B schema is considered frozen, record the gap as a spec amendment rather
  than silently relying on operator discipline.

### P2-1 — Malformed `questions` payloads are silently normalized or filtered

* **Severity:** P2
* **Location:** `src/guardbound/attacks/actor_attack.py:449-453`;
  probe call at `scripts/cloud_model_qualification.py:1143-1155`.
* **Expected:** the probe validates the raw pre-attack payload; malformed query-generation
  responses are not converted into valid queries.
* **Actual:** `{"questions": "abcd"}` becomes four one-character queries per actor; a dict becomes
  its keys; `["ok", 3]` drops the integer and passes. Reproduced.
* **Impact:** the probe can award structural validity to a model that did not produce a query list.
  The final queries are strings, so the literal row-7 check passes; the defect is at the raw
  response boundary. P2 rather than P1 for that reason.
* **Recommended correction:** extend the diagnostics sink to `get_init_queries` (or validate inside
  it) to record the raw `questions` type/presence and the raw item count; fail with
  `invalid_query_payload` on non-list/non-mapping shapes and `non_string_query` on dropped items.

### P2-2 — Non-iterable `questions` is classified as a request error

* **Severity:** P2
* **Location:** `scripts/cloud_model_qualification.py:1149-1156` (catch-all around
  `get_init_queries`); `actor_attack.py:449`.
* **Expected:** a malformed model payload is a structural failure; request errors are
  transport/exhaustion failures.
* **Actual:** `{"questions": 5}` raises `TypeError`, is caught as `request_error`, and is excluded
  from `successful_response_count` and the structural rate. Reproduced.
* **Impact:** a malformed sample is hidden from the structural denominator and inflates the request
  error rate. Frequent occurrences fail the independent request-error gate; a single occurrence does
  not count against structural validity.
* **Recommended correction:** distinguish transport exceptions from payload-shape exceptions
  (`TypeError`, `AttributeError`, `KeyError` from parsing) and classify the latter as a structural
  failure (`invalid_query_payload`).

### P2-3 — The gate does not require `run_id` or the `review_triggers` block

* **Severity:** P2
* **Location:** `scripts/_cloud_common.py:305-450` (`require_qualification`).
* **Expected:** a v2 report carries `run_id` and `review_triggers` (Phase 40B §7.5/§8); the gate
  fails closed if they are missing.
* **Actual:** a v2 report with all mandatory probes `PASS` but no `run_id` and no
  `review_triggers` passes. `require_review_decision` then returns `None`, disabling the review.
* **Impact:** a tampered or older v2 report can bypass the review and lose run provenance. The rest
  of the gate still applies, so this is not an automatic false QUALIFIED.
* **Recommended correction:** require `run_id` to be a sortable UTC string consistent with the
  filename, and require `review_triggers` to be a mapping containing the `delivery_type` block with
  `threshold`, `observed`, and `triggered`.

### P3 findings (low)

| ID | Finding | Evidence / correction |
|---|---|---|
| P3-1 | `final_reply` is described as a "reference"; in JSON it is an equal string copy of `messages[8].content` | Test T22 asserts equality, which is the meaningful check; wording should say "copy/equal value" |
| P3-2 | The same-timestamp resolver test is a unit test of `_select_newest_report`, not an end-to-end resolver scenario | Canonical filenames collide before the glob can return duplicates; keep the defensive check, but do not claim end-to-end coverage |
| P3-3 | `cloud_smoke_test.py` docstring says it is the only real OpenAI caller | False given `cloud_pilot`, the qualification harness, and the eval scripts; correct the docstring |
| P3-4 | `configs/qualification_r1_gpt4o.yaml` changed 0.90 → 0.80 outside the Phase 40B §9 inventory | Justified by §4.5 (≥4/5 = 0.80); the frozen experiment config is untouched; the change is disclosed in the Phase 41 report |
| P3-5 | `extract_harm_target` gained a diagnostics parameter not listed in §10.2 | Necessary to distinguish absent vs null `details`; defaults to `None`; parity test passes |
| P3-6 | The manifest records the parsed review decision but not its file path/hash | Add `review_decision_path` and `review_decision_sha256` |
| P3-7 | The cumulative git diff cannot isolate Phase 41 in `actor_attack.py` | The file was already dirty before Phase 41; the diagnostics hunks are identifiable; the rest predates the phase |
| P3-8 | Same-second re-run raises `FileExistsError` with a traceback | Catch it in `main` and print a clear refusal; the report is still protected |
| P3-9 | `probe_actor_attack_pre_attack` has no internal minimum-sample guard | `main` enforces 30 for paper configs; a direct caller with 5 goals gets a rate; add a guard or document the caller contract |
| P3-10 | Top-level `message_count: 9` is the design constant even when a conversation errors | The per-conversation count records the actual; document the distinction |

---

## 12. Remaining limitations and unverified claims

1. **Pre-existing status of the four full-suite failures is unverified.** I verified the immediate
   causes (CPU-only torch; missing `Qwen/Qwen3.5-4B`) and that the failing files are not in the
   Phase 41 working-tree diff. I did not run a baseline checkout to prove they failed before Phase
   41.
2. **Phase 41 attribution in `actor_attack.py` is partly inferential.** The working tree is
   cumulative; the diagnostics hunks match the Phase 41 inventory, but the file was already dirty.
3. **No paid run was executed**, so the probes' live behavior against GPT-4o is unverified. This
   audit covers the offline implementation and gate logic only.
4. **The `delivery_type` review trigger is an engineering safeguard**, not a paper requirement. Its
   statistical meaning is limited to the observed 30 samples.
5. **The continuity gate is a five-sample smoke gate**, not an 80%-reliability claim. The
   implementation and documentation state this correctly.
6. **The audit did not test malicious filesystem tampering** (forged reports, renamed files). The
   gate is fail-closed against malformed content, but it has no cryptographic signature.
7. **The code-word diagnostic probe is not retained.** Phase 40B permits this; the `diagnostics`
   block is empty and T36 proves structural isolation.

---

## 13. External activity confirmation

* No OpenAI or other model API was called.
* No API key was read, printed, validated, or exposed.
* No qualification command was executed against a provider.
* No model inference or weight download occurred.
* No ActorAttack, Crescendo, Opposite-day, Acronym, RedQueen, or research experiment was run.
* No dataset or checkpoint was downloaded or modified.
* No credential appears in this report or in the audit scratchpad scripts.

---

## 14. Final recommendation

**Verdict: `PASS WITH FINDINGS — CORRECTIONS REQUIRED BEFORE AUTHORIZATION`.**

The v2 qualification implementation is structurally sound and the main R1 `cloud_pilot` path is
fail-closed, but it is **not authorized** for a paid qualification run or Deliverable 1 execution
until the following are resolved:

1. **P0-1** — gate every paid entry point, or formally except and document the ungated scripts.
2. **P1-1** — fix the delivery-type denominator and re-verify the `<0.50` trigger.
3. **P1-2** — require three kept actors, not three raw entries.
4. **P1-3** — bind the review decision to the report/run.
5. **P2-1/P2-2/P2-3** — validate the raw `questions` payload, classify malformed shapes as
   structural failures, and require `run_id`/`review_triggers` in the gate.

The P3 items can be addressed in the same correction pass or documented as accepted deviations.

This audit does not authorize a qualification run, a paid API call, or Deliverable 1. Phase 43 must
not begin automatically.
