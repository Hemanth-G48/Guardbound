# PHASE 44 — INDEPENDENT OFFLINE AUDIT OF PHASE 43 CORRECTIVE IMPLEMENTATION

**Audit date:** 2026-11-13
**Mode:** read-only, offline
**Verdict:** `FAIL — BLOCKING CORRECTIONS REQUIRED`

The Phase 43 P1/P2 correctness corrections are independently verified. The audit nevertheless found
**two P0 authorization bypasses** and **two P1 review-trigger bypasses** that must be corrected
before any authorization decision:

1. **P0-1 — the qualification exemption is a caller-controlled string.** Any caller can pass
   `purpose="qualification"` to `require_paid_authorization`, `build_single_role_llm`, or
   `build_role_llms` and construct an R1 paid client with no report. I reproduced the bypass with a
   scripted factory.
2. **P0-2 — unguarded paid R1 construction remains outside the audited scripts.** `final_analysis.py`
   (repo root) constructs the R1 judge directly with no authorization check, and the public
   `build_cloud_role_llm` / `get_llm` library helpers construct paid clients with no guard.
3. **P1-1 — a QUALIFIED report with `observed: null, triggered: null` passes the gate**, and
   `require_review_decision` then returns `None`. A malformed/tampered report can silently skip the
   delivery-type review.
4. **P1-2 — the gate accepts any trigger threshold in `(0, 1]`.** A report with `threshold: 0.01`
   and `observed: 0.1` is accepted as `triggered: false`, suppressing the frozen `< 0.50` review.

A `PASS` is therefore not supportable. Phase 45 or any paid activity is not authorized.

---

## 1. Scope and no-paid-activity confirmation

**Inspected:** the Phase 40B design, the v1 design, the Phase 41/42/43 reports, `_cloud_common.py`,
`cloud_model_qualification.py`, `cloud_pilot.py`, `actor_attack.py`, every paid-model script
identified in Phase 42, the review-decision and report schemas, the runbook, and the Phase 43
regression suite. I also searched the whole repository for `OpenAIChatLLM`, `build_cloud_role_llm`,
`build_three_model_stack`, `build_role_llms`, `build_single_role_llm`, and `get_llm`.

**Executed:** offline pytest suites and scripted local fakes in the session scratchpad only. No
network transport, provider call, credential read, model load, dataset download, smoke test, pilot,
attack, or research experiment occurred. No repository file was modified except the creation of
this report.

**Limitation:** the working tree has no Phase 41/43 commit boundary; the cumulative `git diff`
includes Phases 22–38. Phase 43 attribution relies on the Phase 43 report, the session snapshot,
and the identifiable changes.

---

## 2. Repository and environment

* Windows, Python 3.13, `torch 2.14.1+cpu`, `torch.cuda.is_available() == False`.
* `KMP_DUPLICATE_LIB_OK=TRUE` is required because the machine loads two OpenMP runtimes; without it
  Python aborts before pytest starts. This is an environment workaround, not a code change.
* Working tree: a dirty tree containing the pre-existing Phase 22–38 changes and the Phase 41–43
  artifacts (129 porcelain entries at audit close). The audit added only this report.
* The four full-suite failures are environment-only: two expect a CUDA cache-release call that a
  CPU-only torch build never reaches, and two require `Qwen/Qwen3.5-4B`, which is absent from the
  local Hugging Face cache (other Qwen snapshots exist).

---

## 3. Phase 42 findings-to-Phase 43 verification matrix

| Phase 42 finding | Phase 43 claim | Independent result |
|---|---|---|
| **P0-1** paid bypass outside `cloud_pilot` | centralized guard across audited scripts + factories | **PARTIAL / FAIL** — audited scripts and both `_cloud_common` factories are guarded, but `final_analysis.py` and the public library factories are not; the `purpose` exemption is caller-controlled |
| **P1-1** delivery rates > 1.0 | completed-row denominators | **VERIFIED FIXED** — audit reproduction returns `0.0`, not `1.0`, for the suppression case |
| **P1-2** duplicate actors false pass | raw/kept counts | **VERIFIED FIXED** — raw 3 / kept 2 yields `wrong_actor_count` |
| **P1-3** unbound review decision | ten-field bound decision | **VERIFIED FIXED** — stale hash/run/config/arm, missing fields, invalid timestamp, eligibility and inactive-trigger cases all reject |
| **P2-1** `questions` normalization | strict raw validation | **VERIFIED FIXED** — string/dict/int/mixed/absent/null are structural; empty list is `empty_chain`; valid list passes |
| **P2-2** malformed payload as request error | typed `QueryPayloadError` | **VERIFIED FIXED** — malformed payloads report `request_error == False` |
| **P2-3** missing `run_id`/`review_triggers` | gate provenance | **PARTIAL** — malformed fields, NaN/inf/bool/string/range/missing/inconsistent/date/filename cases reject; **null trigger fields and non-0.50 thresholds are accepted** |

---

## 4. Exhaustive paid-client entry-point table

| Entry point | Client path | Authorization guard | Ordering |
|---|---|---|---|
| `scripts/cloud_pilot.py` | `common.build_role_llms` | explicit `require_qualification` + `require_review_decision`; factory guard | before `require_api_key`, `load_nbf`, clients |
| `scripts/cloud_model_qualification.py` | `common.build_single_role_llm` | `purpose="qualification"` exemption; `--confirm-paid-request` | explicit qualification flow |
| `scripts/cloud_smoke_test.py` | `OpenAIChatLLM` | `guard_paid_model` in `main` | before `preflight` (key presence) and client |
| `scripts/run_attacks.py` | `OpenAIChatLLM` | guards in `make_target_llm` / `make_attacker_llm` | before client |
| `scripts/run_baseline_evals.py` | `OpenAIChatLLM` | guard in `make_target_llm` and the ASR judge branch | before client |
| `scripts/judge_asr.py` | `OpenAIChatLLM` | guard in `make_judge` | before client |
| `scripts/eval_mmlu.py` | `OpenAIChatLLM` | guard in `make_llm` | before client |
| `scripts/eval_mtbench.py` | `OpenAIChatLLM` | guard in `make_llm` (target/judge) | before client |
| `scripts/eval_over_refusal.py` | `OpenAIChatLLM` | guard in `make_llm` | before client |
| `scripts/run_full_evaluation.py` | `OpenAIChatLLM` | guard in `main` before key; guard in both `setup_*` | before `os.environ.get` and client |
| `scripts/run_reproduction.py` | `OpenAIChatLLM` / `build_role_llm` | guard in both `make_llm` paths | before key check and client |
| `scripts/demo_steered_chat.py` | `OpenAIChatLLM` | guard (non-R1 model, no-op) | before client |
| `scripts/run_paper_experiment.py` | delegates to `run_attacks.make_target_llm` | inherited guard | before client; auth flags not forwarded (fail-closed) |
| **`final_analysis.py`** | **`OpenAIChatLLM`** | **NONE** | **default R1 judge, no check** |
| `scripts/audit_prompt_fidelity.py` | `OpenAIChatLLM` with a fake backend | not a paid path | offline audit |
| `scripts/phase14_1_test_structured_output.py` | local HF client | not a paid path | local |
| `src/guardbound/llm/provider_factory.py` | `build_cloud_role_llm` → `OpenAIChatLLM` | **NONE at the library layer** | callers may bypass |
| `src/guardbound/llm/__init__.py` `get_llm` | `OpenAIChatLLM` | **NONE at the library layer** | no production caller found |
| `src/guardbound/llm/provider_factory.py` `build_three_model_stack` | `build_cloud_role_llm` | **NONE at the library layer** | called by guarded `build_role_llms`; direct callers bypass |

The Phase 42 list was not exhaustive: `final_analysis.py` is a concrete unguarded R1 judge script.

---

## 5. Qualification exemption analysis

`require_paid_authorization` (`scripts/_cloud_common.py:716-747`) begins:

```python
if purpose == "qualification":
    return {"gated": False, "purpose": purpose, "model": model_id}
```

`build_single_role_llm` (`:974-1001`) and `build_role_llms` (`:839-881`) accept `purpose` and forward
it. The value is a plain string parameter with no caller authentication, no harness-only token, and
no check that the caller is `cloud_model_qualification.py`.

**Reproduction (scratch, read-only):** monkeypatching `build_cloud_role_llm` to return an object,
then calling:

```python
common.build_single_role_llm(
    "target",
    {"provider": "openai", "model": "gpt-4o-2024-08-06"},
    object(),
    purpose="qualification",
)
```

printed:

```
PURPOSE_EXEMPTION: BYPASS — constructed object
```

No report, no review decision, no qualification run. This is a **P0 caller-controlled authorization
bypass**. The intended harness does need an exemption, but it must be a capability the harness
possesses and ordinary callers cannot name. Recommended correction (not implemented): a separate,
non-exported qualification factory or a harness-issued token/context; at minimum, a check that the
caller is the qualification harness rather than a public string.

---

## 6. Delivery-type metric audit

Inspection of `probe_actor_attack_pre_attack` confirms:

```python
completed = [row for row in rows if not row["request_error"]]
successful = len(completed)
delivery_valid = sum(row["delivery_type_state"] == "valid" for row in completed)
delivery_present = sum(row["delivery_type_state"] in
                       ("null", "empty", "invalid_type", "valid") for row in completed)
```

The pre-query counters were removed; numerators and denominators are derived from the same completed
rows. `_rate(0, 0)` returns `None`; `0 <= numerator <= denominator` bounds the rates to `[0, 1]`.

**Independent reproduction of the Phase 42 suppression scenario** (2 attempted, 1 completed with an
invalid delivery type, 1 query-stage request error with a valid delivery observation):

```
P1_1: successful 1 errors 1 valid_rate 0.0 present_rate 1.0
```

The errored sample contributes nothing; the completed invalid sample yields `0.0`. The Phase 43
suite additionally covers all-valid, all-invalid, mixed, zero-completed, and query-stage error
cases. The `<0.50` trigger uses this corrected value. **Pass.**

Presence semantics are consistent with the five-state classifier: `missing` and a non-mapping
`query_details` are not present; `null`, `empty`, `invalid_type`, and `valid` are.

---

## 7. Actor payload audit

Inspection confirms:

* `raw_actor_count` comes from the pre-filter diagnostics.
* `get_actors` sets `diagnostics["kept"]` **after** the `actors[:actor_num]` truncation.
* The probe sets `kept_actor_count = len(actors)` (the executable list) and
  `actor_count = kept_actor_count`.
* `_classify_actor_payload` fails with `wrong_actor_count` unless **both** raw and kept counts are 3.
* `_classify_queries` receives the kept count.
* `diagnostics=None` leaves the attack path and the returned structures unchanged.

**Independent reproduction:** three raw entries with two distinct names:

```
P1_2: wrong_actor_count raw 3 kept 2 valid False
```

The Phase 43 suite covers 3 distinct, 3 raw/2 distinct, 3 raw/1 repeated, fewer than 3, more than 3
(raw 4, kept 3), invalid entries, and missing actor names. **Pass.**

---

## 8. Review-decision binding audit

`load_review_decision` requires and validates all ten fields: the four binding fields plus
`trigger`, `decision`, `rationale`, `decided_by`, `decided_at`, and `actor_attack_eligible`.
`require_review_decision` compares the decision's report SHA-256, run ID, configuration SHA-256, and
arm against the verified block, requires an active trigger match, and rejects a decision supplied
when no trigger is active. The Phase 43 suite covers valid, stale hash, wrong run, wrong config,
wrong arm, missing fields, invalid timestamp, eligibility contradiction, inactive trigger, and
unused decision.

**Manifest:** `cloud_pilot` records the qualification block (report path, SHA-256, run ID,
experiment config, triggers), the parsed decision, the decision path, and its SHA-256. The values
come from the validated `require_qualification` block, the validated decision, and the supplied
path.

**Gaps (see findings P1-1/P1-2):** a QUALIFIED report with null trigger fields is accepted, and the
threshold is not pinned to the frozen 0.50. Both allow a review to be skipped. The binding of the
decision itself is sound.

---

## 9. Raw query-payload validation audit

`get_init_queries(..., diagnostics=...)` validates before iteration: mapping payload, present and
list-typed `questions`, string/dict items rejected, each mapping must have a non-string/empty
`question` checked, and empty lists allowed. Failures raise `QueryPayloadError`; the probe catches
it before the transport catch and marks the sample structural, not a request error. With
`diagnostics=None` the historical parser behavior is preserved.

**Independent reproductions:**

```
string: invalid_query_payload request_error=False valid=False
dict:   invalid_query_payload request_error=False valid=False
int:    invalid_query_payload request_error=False valid=False
mixed:  non_string_query     request_error=False valid=False
empty:  empty_chain          request_error=False valid=False
```

The Phase 41 parity test and the Phase 43 `get_init_queries` parity test confirm the normal attack
path is unchanged. **Pass.**

---

## 10. Report schema, provenance and persistence

Verified in `require_qualification` and `_persist_report`:

* `schema_version == "qualification.v2"`, `probe_spec_version == "r1-probe.v2"`.
* Mandatory probe set present, each `PASS`; `probes_complete is True`; `overall == QUALIFIED`.
* Identity: models, prompt variant, attacker condition, temperatures, structured output, and
  `experiment_config.sha256` (self-consistency plus the loaded `_source_sha256`).
* `run_id` format `^\d{8}T\d{6}Z$`, real UTC date, canonical filename, filename/payload run-id match.
* `review_triggers` mapping, `delivery_type` block, required fields, finite numeric ranges,
  boolean/null `triggered`, and consistency.
* Writer refuses an existing path before any write; `main` catches `FileExistsError` and returns 2.
* JSONL append includes schema, probe spec, and run ID.
* Resolver selects the newest canonical report; same-run-id peers raise; no fallback; v1 rejected.
* Report SHA-256 is computed from the file bytes.

**Independent edge-case run:** NaN, +inf, negative, >1, boolean, numeric string, zero threshold,
missing fields, missing trigger block, inconsistent trigger, impossible date, malformed run ID, and
filename mismatch all rejected. **Two accepted cases** are the P1 findings: `observed=null,
triggered=null`, and `threshold=0.01, observed=0.1, triggered=false`.

---

## 11. Phase 40B compliance matrix

| Requirement | Status |
|---|---|
| Probe A: 30 HarmBench goals, file order | Pass — `main` uses the frozen 30 for paper configs; `load_goals` row-count validation |
| Frozen dataset hash | Pass — `ac789de8…` |
| Raw pre-filter actor validation | Pass — diagnostics sink, raw/kept counts |
| Exactly three actors | Pass — raw and kept must both be 3 |
| Actor fields non-empty | Pass — `actor_name` and `relationship` |
| Query/query-details validation | Pass — seven cases and strict `questions` |
| Delivery-type diagnostic, not a gate | Pass |
| Request vs structural errors | Pass — `QueryPayloadError` separated |
| No goal/target/query/delivery text retained | Pass — hashes/lengths/counts only |
| Probe B: exact system prompt | Pass — `75357d68…` |
| Nine messages, four user/four assistant | Pass |
| Five conversations, full history, intermediates | Pass |
| Expected token / refusal failure / request separation | Pass |
| Threshold 4/5 = 0.80, smoke-gate wording | Pass |
| No quiet probe redesign | Pass — schema/spec unchanged; report-binding amendment documented |

---

## 12. Regression and compatibility

* `diagnostics=None` preserves `extract_harm_target`, `get_actors`, and `get_init_queries` behavior;
  parity tests pass.
* ActorAttack execution structures are unchanged.
* Crescendo, Opposite-day, Acronym, and RedQueen files were not modified.
* NBF loading/filtering/checkpoint/η were not modified.
* The frozen R1 configuration is byte-identical.
* Historical v1 report and Phase 40B/41/42/43 reports are byte-identical.
* No credential or `sk-` pattern appears in the Phase 43 files.
* Error handling does not swallow `QualificationError`: the guards are outside broad `try` blocks;
  the `ImportError` handlers cannot catch the authorization error.
* CLI auth flags are wired in the audited scripts; `run_paper_experiment.py` forwards a Namespace
  without them, so a triggered review fails closed with no decision path (P2, not a bypass).
* **Residual:** `final_analysis.py` and the library-level constructors are not guarded (P0-2).

---

## 13. Test commands and exact results

All runs used `KMP_DUPLICATE_LIB_OK=TRUE` on a CPU-only machine.

```
python -m pytest tests/test_phase43_corrective.py -q
```
**57 passed in 0.54s.** Exit 0.

```
python -m pytest tests/test_phase41_probe_redesign.py -q
```
**48 passed in 0.38s.** Exit 0.

```
python -m pytest tests/test_cloud_qualification_harness.py \
  tests/test_phase24_p0_fixes.py tests/test_phase26_qualification_gate.py \
  tests/test_phase27_fail_closed_gate.py tests/test_phase36_implementation.py \
  tests/test_phase38_hardening.py tests/test_attacker_arms.py -q
```
**234 passed in 7.72s.** Exit 0.

```
python -m pytest tests/ -q
```
**1441 passed, 4 failed, 0 skipped in 181.77s.** Exit 1.

Failures, independently reproduced in the same run:

* `tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring`
* `tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured`
* `tests/test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid`
* `tests/test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present`

Verified immediate causes: `torch.cuda.is_available() == False` (CPU-only `torch 2.14.1+cpu`), so
the VRAM guard's `empty_cache` branch is skipped; `Qwen/Qwen3.5-4B` is absent from the HF cache
(other Qwen snapshots are present). The failing files are not in the working-tree diff. The claim
that these failures are historically pre-existing is **unverified**; they are not Phase 43 changes.

**Audit scratch evidence (offline):**

```
PURPOSE_EXEMPTION: BYPASS — constructed object
NULL_TRIGGER: ACCEPTED overall=QUALIFIED triggered=None decision=None
THRESHOLD_0_01: ACCEPTED overall=QUALIFIED triggered=False decision=None
THRESHOLD_0_5_CONTROL: rejected QualificationError: ... triggered a review ...
P1_1: successful 1 errors 1 valid_rate 0.0 present_rate 1.0
P1_2: wrong_actor_count raw 3 kept 2 valid False
```

---

## 14. Frozen artifact integrity

| Artifact | Full SHA-256 | Status |
|---|---|---|
| `configs/paper_reproduction_r1_gpt4o.yaml` | `2125cc3b233efde9e97a8407c364c2140091245dd39bcf440c4366fd99ef393b` | unchanged |
| HarmBench `harmbench_tasks.json` | `ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb` | unchanged |
| NBF checkpoint | `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` | unchanged |
| Historical v1 report | `b17d392a39eeeda64d561017073addd88376d3dafa394bd732c1ba2e908aeaeb` | unchanged |
| Phase 40 v1 design | `0f9371da149535ec0d18572dc4150543985ac2cfaae63266bddcb7c306187a58` | unchanged |
| Phase 40B design | `9152f89e5a951f00f6a01efe1cef681875f4c4769864474b912b646cc57f9ad3` | unchanged |
| Phase 41 report | `c42a71983ad27472e9dd1a6b35571db10bb8c979343825f05f762469baa9d8ff` | unchanged |
| Phase 42 report | `369617b8780b424c8a8d4aed3474f527b0908a3def4ca3268fe75fde33249e97` | unchanged |
| Phase 43 report | `8857e515ce638940140c7c8c2cd5226cae0941b1c78b44d6733b608597b1c9e3` | unchanged |

The abbreviated values in the audit brief match these full hashes.

---

## 15. Findings

### P0-1 — Caller-controlled qualification exemption

* **Severity:** P0
* **Location:** `scripts/_cloud_common.py:716-735` (`require_paid_authorization`),
  `:974-1001` (`build_single_role_llm`), `:839-881` (`build_role_llms`).
* **Expected:** only the qualification harness can construct an R1 paid client without a report.
* **Actual:** `purpose` is a public string; any caller passing `purpose="qualification"` bypasses
  the gate. Reproduced: constructed an object with no report or decision.
* **Impact:** unauthorized paid R1 execution. The qualification harness's separate
  `--confirm-paid-request` is correct, but the exemption is not tied to it or to the caller.
* **Correction:** replace the string with a harness-only capability (a private factory or a token
  created by the harness), or verify the caller/module and the explicit confirmation; do not accept
  a caller-supplied purpose string.

### P0-2 — Unguarded paid R1 construction outside the audited scripts

* **Severity:** P0
* **Location:** `final_analysis.py:47-61, 80` (default `gpt-4o-2024-08-06`, no guard, no auth
  flags); `src/guardbound/llm/provider_factory.py:132-213` (`build_cloud_role_llm` →
  `OpenAIChatLLM`); `src/guardbound/llm/__init__.py:62-80` (`get_llm`).
* **Expected:** every R1 paid client construction is gated, including library-level helpers.
* **Actual:** `final_analysis.py` constructs the R1 judge directly. The library factories are
  unguarded; `build_three_model_stack` and direct `build_cloud_role_llm` callers bypass the
  scripts-layer guard.
* **Impact:** a concrete paid R1 judge path and library-level bypasses. The Phase 43 entry-point
  inventory omitted `final_analysis.py`.
* **Correction:** add the centralized guard and auth flags to `final_analysis.py`, and enforce the
  gate at the provider factory (or restrict/remove the public constructors). The Phase 43 report's
  limitation #1 is therefore a blocking defect, not a documentation note.

### P1-1 — QUALIFIED report with null trigger fields bypasses the review

* **Severity:** P1
* **Location:** `scripts/_cloud_common.py:403-450` (trigger shape/consistency) and
  `:590-640` (`require_review_decision`).
* **Expected:** a QUALIFIED report has a numeric `observed` and boolean `triggered`; null is only
  valid for a non-qualified/early-stop report.
* **Actual:** `observed=None, triggered=None` is accepted; `require_review_decision` sees no
  trigger and returns `None`. Reproduced: `NULL_TRIGGER: ACCEPTED overall=QUALIFIED triggered=None
  decision=None`.
* **Impact:** a malformed/tampered report silently skips the delivery-type review.
* **Correction:** for `overall == QUALIFIED` (or always in v2), require `observed` to be a finite
  number and `triggered` a boolean; reject null.

### P1-2 — Trigger threshold is not pinned to the frozen 0.50

* **Severity:** P1
* **Location:** `scripts/_cloud_common.py:409-450`.
* **Expected:** the report threshold is the frozen `0.50` from Phase 40B.
* **Actual:** any value in `(0, 1]` is accepted. `threshold: 0.01, observed: 0.1,
  triggered: false` is accepted and suppresses the review; the control `threshold: 0.5,
  observed: 0.1` correctly requires a decision. Reproduced.
* **Impact:** a tampered report can suppress the review by lowering the threshold.
* **Correction:** require `threshold == DELIVERY_TYPE_REVIEW_THRESHOLD` (0.50) and reject any other
  value.

### P2-1 — R1 detection is an exact-snapshot/config test

* **Severity:** P2
* **Location:** `scripts/_cloud_common.py:698-702` (`is_r1_paid_model`).
* **Expected/actual:** the exact `gpt-4o-2024-08-06` id or a `paper_reproduction` config is gated.
  A standalone script using an alias such as `gpt-4o` is not gated. This is outside the frozen R1
  model identity but is a paid-execution boundary worth documenting or broadening deliberately.
* **Correction:** document the boundary; if the R1 family/aliases are in scope, add them to the
  detection set rather than relying on the exact snapshot string.

### P2-2 — `run_paper_experiment.py` cannot supply an authorization decision

* **Severity:** P2
* **Location:** `scripts/run_paper_experiment.py:132-144`.
* **Expected/actual:** it delegates to the guarded `run_attacks.make_target_llm`, but constructs a
  `Namespace` without `qualification_report`/`review_decision`. A triggered review fails closed
  with no way to supply the decision. Not a bypass; an operability/provenance gap.
* **Correction:** add and forward the auth flags (or reuse `add_paid_authorization_args`).

### P2-3 — Phase 43 test coverage omits the blocking cases

* **Severity:** P2
* **Location:** `tests/test_phase43_corrective.py`.
* **Expected/actual:** the suite validates the fixed metrics, actor counts, decisions, payloads, and
  most gate fields, but has no test for the `purpose="qualification"` bypass, `final_analysis.py`,
  null trigger fields, or a non-0.50 threshold. The audit found them by inspection and scratch
  probes, so the 57-passing suite is not sufficient evidence of P0 closure.
* **Correction:** add regression tests for all four cases.

### P3 findings

| ID | Finding | Evidence / correction |
|---|---|---|
| P3-1 | `_persist_report` uses a non-atomic `write_json`; a crash can leave a partial report | The existence check prevents overwrite and the gate rejects invalid JSON, but a partial file can block resolution until removed; use an atomic temp-file rename |
| P3-2 | Manifest `review_decision_sha256` is computed after validation, so a file changed between validation and manifest write could be hashed differently | Hash the exact validated bytes (read once, parse and hash the same buffer) |
| P3-3 | The same-timestamp resolver ambiguity is unreachable through the canonical glob; only the unit-level selector test covers it | Keep the defensive check; do not claim end-to-end coverage |
| P3-4 | `review_triggers.operator_decision`/`rationale` are not required to be null in the immutable report | Require null; the separate decision record remains the only operator input |
| P3-5 | `run_full_evaluation` guards OpenAI roles but not Anthropic/Ollama | Non-R1 scope; document the boundary |
| P3-6 | The cumulative git diff cannot isolate Phase 43 | The Phase 43 file list is the attribution basis; no commit boundary exists |

---

## 16. Remaining limitations and unverified claims

1. **The four full-suite failures are environment-only.** Immediate causes verified; the historical
   first-failure date is unverified and no repair was attempted.
2. **No paid run was executed**, so live provider behavior and the probes' eventual outcome remain
   unverified by design.
3. **The qualification exemption and library constructors are architectural, not test failures.**
   The Phase 43 suite passes while the bypasses remain; this is why the audit does not infer P0
   closure from the test result.
4. **The review-trigger null and threshold cases require a tampered/malformed report.** They are
   still gate-validation failures: the gate must fail closed.
5. **The report's `operator_decision`/`rationale` fields are informational**; the gate ignores them,
   which is acceptable only if they are required to be null.
6. **The audit did not attempt malicious filesystem tampering beyond the synthetic report cases**;
   the gate has no cryptographic signature and trusts the local filesystem.

---

## 17. Authorization status

This audit does **not** authorize:

* a qualification run or any paid API call;
* any API-key validation, model inference, smoke test, pilot, or experiment;
* Deliverable 1;
* Phase 45 or any subsequent implementation.

The next step is a separately reviewed corrective pass addressing P0-1, P0-2, P1-1, and P1-2, then a
new independent audit. The four P1/P2 correctness fixes from Phase 43 are verified, but the
authorization boundary is not yet trustworthy.

---

## 18. Final verdict

# `FAIL — BLOCKING CORRECTIONS REQUIRED`

The Phase 43 metric, actor-count, review-binding, raw-query, and most gate-provenance corrections
are independently verified. The implementation nevertheless remains fail-open at the authorization
boundary:

* a caller-controlled `purpose="qualification"` string bypasses the R1 paid gate;
* `final_analysis.py` and the public library factories construct paid R1 clients without any guard;
* a QUALIFIED report with null trigger fields skips the review;
* a non-0.50 threshold suppresses the frozen `<0.50` review.

No qualification run, paid call, or research experiment is authorized by this report.
