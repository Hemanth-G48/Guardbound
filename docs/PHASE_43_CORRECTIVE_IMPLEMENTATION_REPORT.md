# PHASE 43 — R1 QUALIFICATION V2 CORRECTIVE IMPLEMENTATION REPORT

**Phase:** 43 (corrective implementation, offline only)
**Date:** 2026-11-13
**Authority:** Phase 40B specification, Phase 41 implementation, Phase 42 independent audit
**Verdict:** `IMPLEMENTATION COMPLETE — READY FOR PHASE 44 INDEPENDENT OFFLINE AUDIT`

This report does **not** claim independent audit status. A separate Phase 44 audit must verify the
corrections before any authorization decision. No qualification run, paid call, or Deliverable 1
execution is authorized by this phase.

---

## 1. Executive summary

All Phase 42 P0/P1/P2 findings are corrected, with one documented safety amendment to the review
decision record:

* **P0-1** — R1 paid authorization is centralized in `_cloud_common.require_paid_authorization`
  and applied to every audited paid entry point plus the shared `build_role_llms` /
  `build_single_role_llm` factories. The qualification harness retains the only explicit
  exemption. Non-R1 models and offline-mock/dry-run paths remain unguarded.
* **P1-1** — `delivery_type_valid_rate` and `delivery_type_present_rate` are derived from completed
  sample rows with `request_error is False`; both are bounded to `[0, 1]` or `None`.
* **P1-2** — the probe records `raw_actor_count` and `kept_actor_count` (post-filter and
  post-truncation), sets `actor_count` to the executable count, and fails unless both raw and kept
  counts are exactly three.
* **P1-3** — review decisions now require `qualification_report_sha256`, `run_id`,
  `configuration_sha256`, `arm`, `trigger`, `decision`, `rationale`, `decided_by`, `decided_at`,
  and `actor_attack_eligible`; every binding is checked against the verified report. A decision
  supplied with no active trigger is rejected, not discarded.
* **P2-1/P2-2** — `get_init_queries` validates the raw `questions` container before iteration when
  the probe supplies a diagnostics sink, raising a typed `QueryPayloadError` that the probe
  classifies as structural, not transport. With `diagnostics=None` the attack path is unchanged.
* **P2-3** — the gate requires a valid UTC `run_id`, canonical filename consistency, a
  `review_triggers` mapping, a `delivery_type` block, numeric threshold/observed values, a boolean
  or null `triggered`, and internal trigger consistency.
* **P3** — the `final_reply` wording, smoke-test docstring, manifest decision path/hash, same-second
  collision message, sample-count contract, and `message_count` semantics are addressed.
* The report schema remains `qualification.v2` and the probe spec remains `r1-probe.v2`; no
  migration is required because no field was removed or redefined. The review-decision binding is a
  **Phase 43 safety amendment**, documented as such.

**Verification:** Phase 43 suite **57 passed**; Phase 41 suite **48 passed**; existing qualification
and gate suites **234 passed**; full offline suite **1441 passed, 4 failed, 0 skipped**. The four
failures are the same environment-only checks as Phase 42 (CPU-only torch; missing
`Qwen/Qwen3.5-4B` cache).

All frozen hashes and the Phase 40B/41/42 reports are unchanged.

---

## 2. Phase 42 findings-to-fix mapping

| Phase 42 finding | Status | Implementation | Evidence |
|---|---|---|---|
| P0-1 paid bypass outside `cloud_pilot` | **Fixed for audited entry points + factories** | `require_paid_authorization`, `guard_paid_model`, `add_paid_authorization_args`; guards in 10 scripts; `build_role_llms` / `build_single_role_llm` | Phase 43 tests; full suite |
| P1-1 delivery-type rates > 1.0 | **Fixed** | Rates derived from `completed` rows | Phase 43 suite; original reproduction now 1.0 not 2.0 |
| P1-2 duplicate actors false pass | **Fixed** | `raw_actor_count`, `kept_actor_count`, count check | Phase 43 suite; reproduction now `wrong_actor_count` |
| P1-3 unbound review decision | **Fixed (amendment)** | 10-field record + binding checks + unused-decision rejection | Phase 43 suite; T44 |
| P2-1 `questions` normalization | **Fixed** | Strict raw validation with `QueryPayloadError` | Phase 43 suite; reproduction now `invalid_query_payload` / `non_string_query` |
| P2-2 malformed payload as request error | **Fixed** | Typed structural exception before the transport catch | Phase 43 suite; `request_error is False` |
| P2-3 missing `run_id`/`review_triggers` | **Fixed** | Gate provenance and trigger validation | Phase 43 suite |
| P3-1 `final_reply` wording | **Addressed** | Comments say equal-value copy; JSON has no identity | Code inspection |
| P3-3 smoke-test docstring | **Addressed** | Docstring now describes a standalone diagnostic + R1 gate | Code inspection |
| P3-6 manifest decision provenance | **Addressed** | `review_decision_path` + `review_decision_sha256` | `cloud_pilot` manifest |
| P3-8 same-second traceback | **Addressed** | `main` catches `FileExistsError`, prints refusal, returns 2 | Code inspection; `_persist_report` test |
| P3-9 sample-count contract | **Addressed** | Probe docstring states caller supplies 30 for paper configs | Code inspection |
| P3-10 `message_count` semantics | **Addressed** | Comments distinguish expected 9 from actual per-conversation length | Code inspection |
| P3-2 duplicate-timestamp coverage | Accepted | Defensive selector check retained; unit-tested | Documented |
| P3-4 config 0.90→0.80 | Accepted | Justified by Phase 40B §4.5; experiment config untouched | §13 |
| P3-5 `extract_harm_target` diagnostics | Accepted | Necessary for absent/null distinction; parity-tested | §11 |
| P3-7 cumulative diff attribution | Accepted | Working tree has no Phase 41/43 commit boundary | §3, §14 |

---

## 3. Files modified and added

**Modified**

| File | Phase 43 change |
|---|---|
| `scripts/_cloud_common.py` | centralized paid authorization; gate run-id/trigger hardening; bound review decisions; factory guards |
| `scripts/cloud_model_qualification.py` | completed-row delivery rates; raw/kept actor counts; `QueryPayloadError` classification; operator-facing collision error; comments |
| `scripts/cloud_pilot.py` | passes authorization paths to `build_role_llms`; manifest decision path/hash |
| `src/guardbound/attacks/actor_attack.py` | `QueryPayloadError`; strict `questions` validation with diagnostics; post-truncation `kept` |
| `scripts/cloud_smoke_test.py` | R1 guard before key access; auth flags; corrected docstring |
| `scripts/run_attacks.py` | guards for target/attacker cloud clients; auth flags |
| `scripts/run_baseline_evals.py` | guards for target/judge cloud clients; auth flags |
| `scripts/judge_asr.py` | judge guard; auth flags |
| `scripts/eval_mmlu.py` | target guard; auth flags |
| `scripts/eval_mtbench.py` | target/judge guard; auth flags |
| `scripts/eval_over_refusal.py` | target guard; auth flags |
| `scripts/run_full_evaluation.py` | guards before key access; setup-function guards; auth flags |
| `scripts/run_reproduction.py` | guard in single- and three-model paths; auth flags |
| `scripts/demo_steered_chat.py` | guard (non-R1 no-op); auth flags |
| `docs/DELIVERABLE_1_EXECUTION_RUNBOOK.md` | bound decision schema; trigger requirement |
| `tests/test_phase41_probe_redesign.py` | T44 rewritten for report-bound decisions |
| `tests/test_phase36_implementation.py` | canonical report fixture + `review_triggers` |
| `tests/test_phase38_hardening.py` | canonical report fixture + `review_triggers` |

**Added**

| File | Purpose |
|---|---|
| `tests/test_phase43_corrective.py` | 57 offline corrective tests |
| `docs/PHASE_43_CORRECTIVE_IMPLEMENTATION_REPORT.md` | this report |

**Not modified:** `configs/paper_reproduction_r1_gpt4o.yaml`, `configs/cloud_qualification.yaml`,
`configs/qualification_r1_gpt4o.yaml` (its 0.80 change was Phase 41), HarmBench, the NBF checkpoint,
the historical v1 report, and the Phase 40B/41/42 reports.

---

## 4. P0-1 — centralized R1 paid authorization

### 4.1 Mechanism

`_cloud_common` now exposes:

* `R1_PAID_MODEL_IDS = {"gpt-4o-2024-08-06"}` and `R1_EXPERIMENT_CONFIG`.
* `is_r1_paid_model(model_id, cfg)` — true for the frozen R1 model or any `paper_reproduction`
  configuration.
* `add_paid_authorization_args(parser)` — adds `--qualification-report` / `--review-decision`.
* `require_paid_authorization(model_id, *, cfg, qualification_report, review_decision, purpose)`:
  * `purpose="qualification"` is the only exemption (the harness creates the report).
  * Non-R1 models return `{"gated": False}`.
  * R1 loads the frozen experiment configuration when a config is not supplied, resolves the
    newest canonical report, validates it, then validates the bound review decision.
  * Returns `{"gated": True, "qualification": block, "review_decision": ...}`.
* `guard_paid_model(model_id, args, ...)` — CLI wrapper reading the two paths from `args`.

`build_role_llms` calls the guard for any R1 role and skips it only when all three roles carry an
`OfflineMockClient` (offline-mock is not a paid client). `build_single_role_llm` calls the guard
unless `purpose="qualification"`; the harness passes that purpose explicitly.

### 4.2 Entry-point coverage

| Entry point | Guard location | Order |
|---|---|---|
| `cloud_pilot.py` | `require_qualification` + `require_review_decision`; factory also guards | before `require_api_key`, `load_nbf`, `build_role_llms` |
| `cloud_model_qualification.py` | no report gate; `purpose="qualification"`; existing `--confirm-paid-request` | explicit exemption |
| `cloud_smoke_test.py` | `guard_paid_model` in `main` | before `preflight` (key presence) and `OpenAIChatLLM` |
| `run_attacks.py` | guards in `make_target_llm` / `make_attacker_llm` | before `OpenAIChatLLM` |
| `run_baseline_evals.py` | guards in `make_target_llm` and the ASR judge branch | before `OpenAIChatLLM` |
| `judge_asr.py` | guard in `make_judge` | before `OpenAIChatLLM` |
| `eval_mmlu.py` / `eval_mtbench.py` / `eval_over_refusal.py` | guard in `make_llm` | before `OpenAIChatLLM` |
| `run_full_evaluation.py` | guards in `main` and both `setup_*` functions | before `os.environ.get` and `OpenAIChatLLM` |
| `run_reproduction.py` | guard in `make_llm` for both paths | before the key check and `OpenAIChatLLM` |
| `demo_steered_chat.py` | guard before `OpenAIChatLLM` | non-R1 model, no-op |
| `build_role_llms` | centralized guard | before `build_three_model_stack` |
| `build_single_role_llm` | centralized guard | before `build_cloud_role_llm` |

Offline-mock and dry-run paths return before the guard or carry an `OfflineMockClient`; they never
create a paid client or claim qualification. Non-R1 workflows are unaffected.

### 4.3 Evidence

Phase 43 tests assert `QualificationError` before construction for `eval_mmlu`, `eval_mtbench`,
`eval_over_refusal`, `judge_asr`, `run_attacks` (target and attacker), `run_baseline_evals`,
`run_full_evaluation` (target and judge), `run_reproduction`, `cloud_smoke_test`, and
`build_role_llms`. A non-R1 model and the qualification purpose return unguarded; mock factories
return without the gate.

---

## 5. P1-1 — corrected delivery-type metric calculation

`probe_actor_attack_pre_attack` now computes:

```python
completed = [row for row in rows if not row["request_error"]]
successful = len(completed)
delivery_valid = sum(row["delivery_type_state"] == "valid" for row in completed)
delivery_present = sum(row["delivery_type_state"] in
                       ("null", "empty", "invalid_type", "valid") for row in completed)
```

The in-loop counters were removed. `structural_validity_rate` and `payload_valid_rate` keep their
existing denominator (`successful_response_count`), and `request_error_rate` keeps
`request_error_count / attempted_count`.

**Phase 42 reproduction, re-run:** 2 attempted, 1 completed, 1 request error, the errored sample
having valid delivery data. Observed now:

```
delivery_valid_rate 1.0   (was 2.0)
delivery_present_rate 1.0 (was 2.0)
```

The remaining completed sample was valid, so 1/1 is correct. The suppression case (invalid
completed sample + valid errored sample) is covered by
`test_delivery_rates_exclude_a_query_phase_request_error`: **0.0**, not 1.0. Additional tests cover
all-valid, all-invalid, mixed, zero-successful, and request errors at the query stage.

Invariants verified: rates in `[0, 1]` or `None`, errored samples contribute to neither numerator
nor denominator, and the `<0.50` trigger uses the corrected rate.

---

## 6. P1-2 — raw versus kept actor validation

Each sample now records:

* `raw_actor_count` — entries in the model payload before filtering.
* `kept_actor_count` — distinct actors after `get_actors` filtering and truncation.
* `actor_count` — documented as the executable (kept) count.

`get_actors` sets `diagnostics["kept"]` after the `actors[:actor_num]` truncation.
`_classify_actor_payload` fails with `wrong_actor_count` unless **both** counts are exactly 3;
`_classify_queries` receives the kept count. The normal deduplication/attack behavior is unchanged.

**Phase 42 reproduction, re-run:** 3 raw entries with 2 distinct names now yields:

```
structural_valid False
failure_reason wrong_actor_count
actor_count 2
raw_actor_count 3
```

Tests cover 3 distinct / 3 raw with 2 distinct / 3 raw with one repeated name / fewer than 3 /
more than 3 (filter truncates to 3 but raw is 4) / invalid entries (still classified as
`actor_not_mapping` or `missing_actor_name`).

---

## 7. P1-3 — report-bound review decisions

### 7.1 Schema (Phase 43 safety amendment)

```json
{
  "qualification_report_sha256": "…64 hex…",
  "run_id": "20261112T144310Z",
  "configuration_sha256": "…64 hex…",
  "arm": "baseline",
  "trigger": "delivery_type",
  "decision": "accept_degradation",
  "rationale": "…",
  "decided_by": "…",
  "decided_at": "2026-11-12T00:00:00Z",
  "actor_attack_eligible": true
}
```

Phase 40B defined only the last six fields; the first four are the documented amendment. The
immutable qualification report is not modified.

### 7.2 Validation

`load_review_decision` checks all ten fields, 64-hex hashes, UTC run-id format, non-empty arm and
trigger, ISO-8601 `decided_at`, the decision enum, and the eligibility consistency rule.

`require_review_decision` compares the decision against the verified report block:

* report `sha256` → `qualification_report_sha256`
* report `run_id` → `run_id`
* report `experiment_config.sha256` → `configuration_sha256`
* report `attacker_prompt_condition` → `arm`
* decision `trigger` must be in the currently active trigger set.

If no trigger is active and a decision path is supplied, the gate raises instead of discarding it.
Tests cover valid, stale hash, wrong run, wrong config, wrong arm, missing field, invalid
timestamp, eligibility contradiction, inactive trigger, and unused decision.

### 7.3 Manifest provenance

`cloud_pilot` records `qualification` (report path, SHA-256, run id, experiment config, triggers)
plus `review_decision`, `review_decision_path`, and `review_decision_sha256`.

---

## 8. P2-1 / P2-2 — raw `questions` validation

`get_init_queries(..., diagnostics=...)` now validates the raw payload **before** iteration:

* payload must be a mapping; `questions` must be present and a list;
* string and dictionary values are rejected, never iterated into characters/keys;
* each item must be a string or a mapping with a `question` field;
* missing/non-string/empty `question` values are rejected;
* empty list is allowed at this layer and reported as `empty_chain` by the probe.

Failures raise `QueryPayloadError(reason)`. The probe catches it as a **structural** failure;
transport exceptions remain request errors. With `diagnostics=None`, the exact historical parser
behavior is preserved (a string still iterates to characters), which the attack path uses.

**Phase 42 reproductions, re-run:**

| Input | Phase 42 | Phase 43 |
|---|---|---|
| `"abcd"` | structural pass, `[4,4,4]` | `invalid_query_payload`, structural fail |
| `{"a": "q"}` | structural pass | `invalid_query_payload`, structural fail |
| `5` | request error | `invalid_query_payload`, structural fail, `request_error=False` |
| `["ok", 3]` | structural pass | `non_string_query`, structural fail |
| absent | `empty_chain` | `missing_questions` |
| `null` | request error | `invalid_query_payload` |
| `[]` | `empty_chain` | `empty_chain` |
| valid list | pass | pass |

Tests also assert `get_init_queries(..., diagnostics=None)` still returns the historical
character-list for a string payload.

---

## 9. P2-3 — report provenance and trigger structure

`require_qualification` now rejects:

* missing/invalid `run_id` (must be `^\d{8}T\d{6}Z$` and a real UTC timestamp);
* a non-canonical filename;
* a filename run id that differs from the payload run id;
* missing/non-mapping `review_triggers`;
* missing/non-mapping `delivery_type`;
* missing, non-numeric, or out-of-range `threshold` / `observed`;
* non-boolean/non-null `triggered`;
* `triggered` inconsistent with `observed < threshold`.

Existing schema, probe-spec, mandatory-probe, verdict, and identity checks are unchanged. Valid
reports with no trigger and with an active trigger plus a bound decision both pass.

---

## 10. P3 corrections

* **P3-1** — comments now state that `final_reply` is an equal-value copy of `messages[8].content`;
  JSON has no object identity.
* **P3-3** — `cloud_smoke_test.py` no longer claims to be the only real API caller; it is a
  standalone diagnostic and R1 targets are gated.
* **P3-6** — manifest records the decision path and SHA-256.
* **P3-8** — `main` catches `FileExistsError`, prints a clear refusal, and returns 2; the writer
  still never overwrites or silently retries.
* **P3-9** — the probe docstring states the caller supplies 30 goals for a paper configuration.
* **P3-10** — comments distinguish the expected top-level 9 from the actual per-conversation
  length after a request error.
* **P3-2/4/5/7** — accepted and documented; no code change (defensive selector test, justified
  config threshold, necessary extraction diagnostics, cumulative-diff attribution).

---

## 11. Compatibility and frozen methodology

* `diagnostics=None` preserves `get_actors`, `extract_harm_target`, and `get_init_queries`
  behavior; the Phase 41 parity test and the new `get_init_queries` parity test pass.
* ActorAttack execution structures are unchanged; `infer_single` still returns the same keys.
* Crescendo, Opposite-day, Acronym, and RedQueen files were not touched.
* NBF loading, checkpoint, η, filtering, and the frozen R1 configuration were not touched.
* The continuity transcript is exactly the approved nine-message design; the threshold is 0.80.
* Report schema remains `qualification.v2`; probe spec remains `r1-probe.v2`. No field was removed
  or redefined; the review-decision binding is the documented Phase 43 amendment.
* v1 reports remain readable and ineligible; the historical report is byte-identical.
* Offline-mock and dry-run paths remain non-paid; the factory skips the guard only when all roles
  carry `OfflineMockClient`.

---

## 12. Tests executed

Environment: Windows, Python 3.13, `torch 2.14.1+cpu`, `torch.cuda.is_available() == False`.
`KMP_DUPLICATE_LIB_OK=TRUE` is required for the local OpenMP runtime; this is an environment
workaround, not a code change.

```
python -m pytest tests/test_phase43_corrective.py -q
```
**57 passed.**

```
python -m pytest tests/test_phase41_probe_redesign.py -q
```
**48 passed.**

```
python -m pytest tests/test_cloud_qualification_harness.py \
  tests/test_phase24_p0_fixes.py tests/test_phase26_qualification_gate.py \
  tests/test_phase27_fail_closed_gate.py tests/test_phase36_implementation.py \
  tests/test_phase38_hardening.py tests/test_attacker_arms.py -q
```
**234 passed.**

```
python -m pytest tests/ -q
```
**1441 passed, 4 failed, 0 skipped in 185.16s.** Exit code 1.

The four failures are the same environment checks as Phase 42 and are unrelated to Phase 43:

* `test_phase14_2_rotation.py` (2) — `torch.cuda.is_available()` is false (CPU-only torch), so the
  monkeypatched cache-release call order is not produced.
* `test_phase8_architecture.py` (2) — `Qwen/Qwen3.5-4B` is absent from the local HF cache.

They are reported as failed, not passed. No unrelated environment repair was attempted.

---

## 13. Frozen artifact integrity

| Artifact | SHA-256 | Status |
|---|---|---|
| `configs/paper_reproduction_r1_gpt4o.yaml` | `2125cc3b…393b` | unchanged |
| HarmBench `harmbench_tasks.json` | `ac789de8…14eb` | unchanged |
| NBF checkpoint | `cea1a75b…36fe` | unchanged |
| Historical v1 qualification report | `b17d392a…aeb` | unchanged |
| Phase 40 v1 design | `0f9371da…7a58` | unchanged |
| Phase 40B design | `9152f89e…ad3` | unchanged |
| Phase 41 report | `c42a7198…8ff` | unchanged by Phase 43 |
| Phase 42 report | `369617b8…e97` | unchanged by Phase 43 |

No API credential or `sk-…` pattern appears in any Phase 43 file. No configuration was changed in
this phase.

---

## 14. Remaining limitations

1. **Entry-point coverage, not library-level enforcement.** The guard is applied to every audited
   paid script and both shared factories, but a future script that imports `OpenAIChatLLM`
   directly could still bypass it. The durable fix is a client-construction policy at the provider
   factory, outside this phase's approved scope.
2. **R1 detection is model/config based.** `gpt-4o-2024-08-06` and `paper_reproduction`
   configurations are gated. A different non-R1 paid model (including Anthropic/Ollama roles in
   `run_full_evaluation`) is outside the R1 authorization scope and keeps its own confirmation
   path.
3. **Review-decision binding is a Phase 43 amendment.** Phase 40B's six-field schema was
   underspecified; the four binding fields are documented here and in the runbook. The immutable
   report remains unchanged.
4. **The four full-suite failures are environment-only.** Their immediate causes were verified;
   no attempt was made to repair unrelated environment dependencies.
5. **No paid run was executed**, so live provider behavior remains unverified by design.
6. **`run_id` has second resolution.** A same-second re-run is refused with a clear message rather
   than silently retried.
7. **The `probe_actor_attack_pre_attack` sample-count contract is documented, not enforced
   internally.** `main` supplies 30 for paper configurations; direct callers must honour it.
8. **The cumulative working tree cannot isolate Phase 43 from earlier phases in git.** The changes
   in §3 are the Phase 43 set; the rest predates it.

---

## 15. External activity confirmation

* No OpenAI, Anthropic, or other model API was called.
* No API key was read, printed, validated, copied, or exposed.
* No qualification command was executed against a provider.
* No model inference or weight download occurred.
* No actor attack, research experiment, smoke test, pilot, or Deliverable 1 run occurred.
* No dataset or checkpoint was downloaded or modified.
* No credential appears in this report or in any Phase 43 file.

---

## 16. Recommendation

`IMPLEMENTATION COMPLETE — READY FOR PHASE 44 INDEPENDENT OFFLINE AUDIT`.

Phase 44 must independently verify the P0/P1/P2 corrections, the Phase 43 amendment, and the
entry-point coverage before any authorization decision. Completion of this phase is not
authorization for a qualification run, a paid call, or Deliverable 1.
