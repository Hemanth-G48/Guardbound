# PHASE 45 — R1 AUTHORIZATION BOUNDARY AND REVIEW-TRIGGER HARDENING

**Phase:** 45 (offline implementation only)
**Date:** 2026-11-13
**Controlling findings:** Phase 44 independent audit
**Verdict:** `IMPLEMENTATION COMPLETE — READY FOR PHASE 46 INDEPENDENT OFFLINE AUDIT`

This report does not claim independent audit status. Phase 46 must verify the boundary and the
trigger hardening before any authorization decision. No qualification run, paid call, or Deliverable
1 execution is authorized by this phase.

---

## 1. Executive summary

Phase 45 closes the Phase 44 blockers at the client-construction layer rather than adding another
list of script guards:

* **P0-1 closed** — the caller-controlled `purpose="qualification"` string is removed from
  `require_paid_authorization`, `build_single_role_llm`, `build_role_llms`, and `guard_paid_model`.
  The qualification harness uses a separate underscore-private builder with a call-site check and an
  explicit `--confirm-paid-request`/`--offline-mock` requirement. The old string now raises
  `TypeError`; a direct call to the private builder from any other module raises
  `QualificationError`.
* **P0-2 closed at the construction boundary** — `OpenAIChatLLM` itself now refuses the frozen R1
  model without an active authorization context, before any key or client access.
  `build_cloud_role_llm` makes the same check before its key read; `build_three_model_stack` and
  `make_chat_llm` therefore fail closed too. `final_analysis.py` now uses the gated
  `build_paid_chat_llm` and has `--qualification-report`/`--review-decision` flags.
* **P1-1 closed** — a `QUALIFIED` report must have a numeric `observed` and a boolean `triggered`;
  null values are rejected. The non-qualified/early-stop null behavior is retained and documented.
* **P1-2 closed** — the gate accepts only the frozen `threshold == 0.50`. `0.01`, `0.51`, `1.0`,
  booleans, numeric strings, NaN, and infinity are rejected; `0.49` triggers and `0.50` does not.
* **P2-1** — the exact-snapshot detection rule is documented and tested; `gpt-4o` is not silently
  treated as R1. **P2-2** — `run_paper_experiment.py` forwards both auth paths and uses a mock on
  `--dry-run`. **P2-3** — the new suite covers every Phase 44 blocking case.
* All Phase 43 corrections (delivery metrics, kept actor count, ten-field review binding, raw
  `questions`, v2 provenance/persistence) are preserved and re-tested.

**Verification:** Phase 45 suite **27 passed**; Phase 43 + Phase 41 **105 passed**; existing
qualification/gate suites **234 passed**; full offline suite **1468 passed, 4 failed, 0 skipped**.
The four failures are the same environment-only checks (CPU-only torch; missing `Qwen/Qwen3.5-4B`).

**Frozen artifacts:** all hashes match; no config, dataset, checkpoint, historical report, or
Phase 40B/41/42/43/44 report was modified. No credentials or model calls.

---

## 2. Scope and no-paid-activity confirmation

**Implemented offline only.** No OpenAI/Anthropic request, API-key read/validation, qualification
run, model inference, download, smoke test, pilot, attack, or Deliverable 1 execution occurred.

**Inspected:** `_cloud_common.py`, `cloud_model_qualification.py`, `cloud_pilot.py`,
`actor_attack.py`, every paid-model script, `final_analysis.py`, `run_paper_experiment.py`, the
public `guardbound.llm` factories, the constructor, the Phase 43 tests, and the Phase 44 findings.

**Tests:** scripted fake SDK backends, temporary reports, and monkeypatched constructors only. No
test required a credential, network, model download, or paid inference.

---

## 3. Architecture of the new authorization boundary

### 3.1 Shared policy module

`src/guardbound/llm/paid_authorization.py` (new) contains:

* `R1_MODEL_IDS = {"gpt-4o-2024-08-06"}` — the exact frozen snapshot.
* `is_r1_model(model_id, cfg)` — exact snapshot **or** `paper_reproduction: true`; aliases are not
  silently treated as R1.
* `PaidAuthorizationRequired` — the fail-closed error.
* `require_construction_authorization(model_id, cfg=None)` — called at the lowest client layer.
* `_authorized(*model_ids)` — the internal `ContextVar` capability context.
* `authorized_models()` — diagnostics.

There is no import from `scripts/` into this module, so the dependency direction is preserved.

### 3.2 Three enforcement points

1. **`OpenAIChatLLM.__init__`** calls `require_construction_authorization(model_id)` immediately
   after the model-id check and **before** any structured-output validation, key access, or client
   construction. This covers direct construction.
2. **`provider_factory.build_cloud_role_llm`** calls the same check immediately after reading
   `model_id` and **before** `os.environ.get(api_key_env)`. This covers `build_three_model_stack`,
   `build_role_llm`, and direct factory callers.
3. **`scripts/_cloud_common`** validates the report/review decision and opens the context around the
   supported builders:
   * `build_paid_chat_llm` — one direct `OpenAIChatLLM` through the gate + context.
   * `build_paid_role_llm` — one provider-factory role through the gate + context.
   * `build_single_role_llm` — research single-role cloud path.
   * `build_role_llms` — three-role stack; all-offline-mock is exempt from the report gate and still
     runs in the context.

The context is a `ContextVar`, so nested builders accumulate grants and reset cleanly. It is not a
public string and cannot be selected by a caller argument.

### 3.3 What is protected

| Path | Before Phase 45 | Phase 45 |
|---|---|---|
| `OpenAIChatLLM(R1)` direct | constructed | `PaidAuthorizationRequired` before key |
| `build_cloud_role_llm(R1)` | constructed after key read | `PaidAuthorizationRequired` before key read |
| `build_three_model_stack(R1)` | constructed | fails through `build_cloud_role_llm` |
| `make_chat_llm("openai", R1)` | constructed | fails through the constructor |
| `build_single_role_llm(R1)` | string exemption | report gate + context; no string |
| `build_role_llms(R1)` | string exemption | report gate + context; offline-mock exception |
| `final_analysis.py` | direct R1 judge | `build_paid_chat_llm` + auth flags |
| `run_paper_experiment.py` | dropped auth paths | forwards both paths; dry-run mock |

**Residual:** Python cannot make a module private. A caller can import the underscore-prefixed
`_authorized` and open the context manually; this is the explicit internal capability, not a public
API, and is documented in §16. It is not reachable through the removed `purpose` string.

---

## 4. Qualification-harness exemption

`_build_qualification_role_llm` in `_cloud_common.py` is the only path that may construct an R1
client without a report, because the harness produces the first report. It:

* is underscore-private and has no public string/purpose parameter;
* requires a caller stack frame from `cloud_model_qualification.py` (checked with `inspect.stack`);
* requires `confirm_paid_request=True` for a non-offline construction, in addition to the harness's
  existing `--confirm-paid-request` gate;
* allows `offline_mock=True` without confirmation because no paid client is created;
* opens the same `_authorized` context and calls the low-level `_build_cloud_role`.

The normal harness call site is:

```python
common._build_qualification_role_llm(
    role, role_cfg, ledger, attacker_arm=arm.name,
    confirm_paid_request=bool(args.confirm_paid_request),
    offline_mock=bool(args.offline_mock),
)
```

Phase 26/27 tests instrument this private function; all 234 qualification/gate tests pass. A direct
call from any other module is tested to raise `QualificationError`.

---

## 5. Exhaustive R1 paid-client construction inventory

| Entry point | Client path | Guard |
|---|---|---|
| `scripts/cloud_pilot.py` | `build_role_llms` → `build_three_model_stack` | explicit report/decision gate + factory context |
| `scripts/cloud_model_qualification.py` | `_build_qualification_role_llm` → `build_cloud_role_llm` | call-site + confirm/offline + context |
| `scripts/cloud_smoke_test.py` | `build_paid_chat_llm` | pre-`preflight` guard + helper |
| `scripts/run_attacks.py` | `build_paid_chat_llm` | helper |
| `scripts/run_baseline_evals.py` | `build_paid_chat_llm` (target/judge) | helper |
| `scripts/judge_asr.py` | `build_paid_chat_llm` | helper |
| `scripts/eval_mmlu.py` | `build_paid_chat_llm` | helper |
| `scripts/eval_mtbench.py` | `build_paid_chat_llm` (target/judge) | helper |
| `scripts/eval_over_refusal.py` | `build_paid_chat_llm` | helper |
| `scripts/run_full_evaluation.py` | `build_paid_chat_llm` (target/judge) | pre-key guard + helper |
| `scripts/run_reproduction.py` | `build_paid_chat_llm` / `build_paid_role_llm` | pre-branch guard + helper |
| `scripts/demo_steered_chat.py` | `build_paid_chat_llm` (non-R1) | helper |
| `scripts/run_paper_experiment.py` | delegated `run_attacks.make_target_llm` | inherited helper; flags forwarded; dry-run mock |
| `final_analysis.py` | `build_paid_chat_llm` | helper + auth flags |
| `scripts/audit_prompt_fidelity.py` | `OpenAIChatLLM` with a fake backend | not a paid path |
| `scripts/phase14_1_test_structured_output.py` | local HF client | not a paid path |
| `src/guardbound/llm/provider_factory.py` `build_cloud_role_llm` | `OpenAIChatLLM` | construction guard before key |
| `src/guardbound/llm/provider_factory.py` `build_three_model_stack` | `build_cloud_role_llm` | fails closed via the factory |
| `src/guardbound/llm/__init__.py` `make_chat_llm` | `OpenAIChatLLM` | fails closed via the constructor |

There is **no `get_llm` symbol** in the repository; the Phase 44 report's public-factory finding
refers to the actual API `make_chat_llm`, which is now tested explicitly. The direct constructor is
also tested.

---

## 6. `final_analysis.py` changes

* Adds `scripts/` to `sys.path` and imports `_cloud_common`.
* Adds `--qualification-report` and `--review-decision`.
* `make_judge(model_name, mock=False, args=None)`:
  * mock path unchanged;
  * real path uses `common.build_paid_chat_llm(model_name, args=args)`, which gates before key
    access and construction.
* `main` passes `args` to `make_judge`.
* Its existing non-R1 behavior is unchanged; a non-R1 judge is not gated.

Test: `make_judge(R1, mock=False, args=missing_report)` raises `QualificationError`;
`make_judge(R1, mock=True)` returns a judge; no report/key needed.

---

## 7. Public library factory changes

* `OpenAIChatLLM.__init__` — construction guard.
* `build_cloud_role_llm` — guard before the key read.
* `build_three_model_stack` — protected through `build_cloud_role_llm`.
* `make_chat_llm` — protected through the constructor; tested for the R1 fail-closed and authorized
  context paths.

The `_authorized` context is opened only by the audited builders. There is no public
`purpose`/`allow_unqualified` parameter on any factory.

---

## 8. Non-R1 and offline-mock compatibility

* **Non-R1 models** (`TEST_MODEL = "gpt-4o-mini"`, `gpt-3.5`, Claude, Ollama) construct unchanged.
  `is_r1_model` is exact for the frozen snapshot; `paper_reproduction` configs are detected at the
  scripts layer where the config is available.
* **Aliases** — `gpt-4o` is **not** silently treated as R1; the exact rule is documented in
  `paid_authorization.is_r1_model` and tested (`is_r1_paid_model("gpt-4o") is False`).
* **Offline mock** — `build_role_llms` detects all three roles carrying `OfflineMockClient`, skips
  the report gate, and opens the context for the mock stack; tested.
* **Qualification offline mock** — `_build_qualification_role_llm(..., offline_mock=True)` needs no
  report or confirmation; tested by the existing qualification suite.
* **Dry run** — `run_full_evaluation`, `run_reproduction`, `cloud_smoke_test`, `judge_asr`,
  `run_baseline_evals`, `run_model_qualification`, and `run_paper_experiment` return before paid
  construction; `run_paper_experiment` uses a mock on dry-run.
* **No test needs an API key** — all construction tests inject a fake SDK client or use
  `OfflineMockClient`.
* **Legacy test fixture** — `tests/conftest.py` opens the R1 context for legacy unit tests that
  inspect request/retry behavior. `test_phase45_authorization` is excluded from that fixture and
  asserts the fail-closed path itself. No production code uses a test-only bypass.

---

## 9. Report trigger validation changes

`require_qualification` now reads `declared_verdict` before the trigger block and enforces:

* `threshold` is a non-boolean number in `(0, 1]` **and exactly `0.50`**;
* for `overall == "QUALIFIED"`:
  * `observed` is a non-boolean finite number in `[0, 1]`;
  * `triggered` is a boolean;
  * null/missing values raise `QualificationError`;
* for a non-qualified/early-stop report, `observed`/`triggered` may be null **only as a matched
  null/null pair**; the verdict then fails the gate anyway;
* `triggered == (observed < 0.50)` exactly.

A `QUALIFIED` report with `observed=None, triggered=None` is rejected; a `QUALIFIED` report with
`threshold=0.01, observed=0.1, triggered=False` is rejected. Both Phase 44 reproductions now fail
closed (verified with the same scratch probes).

---

## 10. Frozen threshold enforcement

`DELIVERY_TYPE_REVIEW_THRESHOLD = 0.50` remains the single value used by
`build_review_triggers` and the gate. The gate rejects any other threshold, even when the
`triggered` value is mathematically consistent with that other threshold.

Boundary tests:

| Threshold | Observed | Triggered | Result |
|---:|---:|---|---|
| 0.50 | 0.49 | true | accepted |
| 0.50 | 0.50 | false | accepted (strict `<`) |
| 0.01 | 0.10 | false | rejected |
| 0.51 | any | consistent | rejected |
| 0.00 / 1.00 | any | any | rejected |
| bool / "0.5" / NaN / inf | any | any | rejected |

---

## 11. Phase 43 regression-preservation matrix

| Phase 43 correction | Status | Evidence |
|---|---|---|
| Delivery rates over completed rows; request errors excluded; `[0,1]`/None | Preserved | Phase 43 suite 57; Phase 44 reproduction `0.0` |
| Raw vs kept actor counts; exactly three executable actors | Preserved | Phase 43 suite; reproduction `wrong_actor_count` |
| Ten-field report-bound review decision | Preserved | Phase 43 suite; T44 |
| Raw `questions` validation; `diagnostics=None` parity | Preserved | Phase 43 suite; Phase 41 equality test |
| `qualification.v2` / `r1-probe.v2`; canonical filenames; immutable persistence; newest resolver; v1 ineligible | Preserved | Phase 41/43 suites; full suite |

No Phase 43 test was weakened; the only test changes are the old `purpose` exemption test (now
asserts `TypeError`) and the instrumentation target (`_build_qualification_role_llm`).

---

## 12. Phase 44 findings-to-fix mapping

| Phase 44 finding | Status | Evidence |
|---|---|---|
| P0-1 caller-controlled `purpose` | **Closed** | public parameters removed; old string `TypeError`; private builder call-site/confirm tests |
| P0-2 `final_analysis.py` direct R1 judge | **Closed** | gated helper + auth flags; negative test |
| P0-2 `build_cloud_role_llm` unguarded | **Closed** | factory guard before key; test |
| P0-2 `build_three_model_stack` unguarded | **Closed** | fails through the factory; test |
| P0-2 public `get_llm` | **Closed for the actual API `make_chat_llm`** | constructor guard; test (no `get_llm` symbol exists) |
| P1-1 null trigger on QUALIFIED | **Closed** | gate rejects null observed/triggered; tests |
| P1-2 threshold not pinned | **Closed** | exact 0.50; boundary/rejection tests |
| P2-1 R1 aliases | **Addressed** | exact-snapshot rule documented and tested |
| P2-2 `run_paper_experiment` dropped args | **Closed** | flags + namespace forwarding + dry-run mock; test |
| P2-3 missing tests | **Closed** | 27 Phase 45 tests covering all blocking cases |

---

## 13. Tests executed

Environment: Windows, Python 3.13, `torch 2.14.1+cpu`, `torch.cuda.is_available() == False`.
`KMP_DUPLICATE_LIB_OK=TRUE` is required for the local OpenMP runtime (environment workaround).

```
python -m pytest tests/test_phase45_authorization.py -q
```
**27 passed in 1.09s.** Exit 0.

```
python -m pytest tests/test_phase43_corrective.py tests/test_phase41_probe_redesign.py -q
```
**105 passed in 0.61s.** Exit 0.

```
python -m pytest tests/test_cloud_qualification_harness.py \
  tests/test_phase24_p0_fixes.py tests/test_phase26_qualification_gate.py \
  tests/test_phase27_fail_closed_gate.py tests/test_phase36_implementation.py \
  tests/test_phase38_hardening.py tests/test_attacker_arms.py -q
```
**234 passed in 7.47s.** Exit 0.

```
python -m pytest tests/ -q
```
**1468 passed, 4 failed, 0 skipped in 179.90s.** Exit 1.

Failures, independently reproduced:

* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring`
* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured`
* `test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid`
* `test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present`

Verified causes: CPU-only `torch 2.14.1+cpu` (`cuda_available == False`) for the two VRAM-guard
tests; `Qwen/Qwen3.5-4B` absent from the local HF cache for the two architecture tests. The failing
files are not in the Phase 45 working-tree diff. The historical pre-existence claim is **unverified**;
they are not Phase 45 changes.

**Phase 44 reproductions re-run (read-only scratch):**

```
PURPOSE_EXEMPTION: blocked TypeError ... unexpected keyword argument 'purpose'
NULL_TRIGGER: rejected QualificationError: ... requires a numeric ... observed
THRESHOLD_0_01: rejected QualificationError: ... threshold must equal the frozen 0.50
THRESHOLD_0_5_CONTROL: rejected QualificationError: ... triggered a review ...
P1_1: successful 1 errors 1 valid_rate 0.0 present_rate 1.0
P1_2: wrong_actor_count raw 3 kept 2 valid False
P2_questions_*: structural, request_error=False
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
| Phase 44 report | `da9ccdf379bb7eb172da302d8700e69493aed7ee6440253a1ad0955b2ed1e1f7` | unchanged |

No credential or `sk-` pattern was added. No config/dataset/checkpoint was modified.

---

## 15. Files modified and added

**Added**

* `src/guardbound/llm/paid_authorization.py` — shared client-layer boundary.
* `tests/test_phase45_authorization.py` — 27 Phase 45 tests.
* `docs/PHASE_45_AUTHORIZATION_AND_TRIGGER_HARDENING_REPORT.md` — this report.

**Modified**

* `src/guardbound/llm/openai_client.py` — construction guard.
* `src/guardbound/llm/provider_factory.py` — factory guard before key read.
* `scripts/_cloud_common.py` — removed `purpose`; qualification-only builder; gated helpers; exact
  threshold/null trigger validation.
* `scripts/cloud_model_qualification.py` — uses the private qualification builder.
* `scripts/cloud_smoke_test.py` — helper construction.
* `scripts/run_attacks.py` — helper construction.
* `scripts/run_baseline_evals.py` — helper construction (target/judge).
* `scripts/judge_asr.py` — helper construction.
* `scripts/eval_mmlu.py`, `scripts/eval_mtbench.py`, `scripts/eval_over_refusal.py` — helper.
* `scripts/run_full_evaluation.py` — helper in both setup functions.
* `scripts/run_reproduction.py` — helper in both paths; pre-branch guard.
* `scripts/demo_steered_chat.py` — helper.
* `scripts/run_paper_experiment.py` — auth flags, namespace forwarding, dry-run mock.
* `final_analysis.py` — auth flags and gated judge construction.
* `tests/conftest.py` — legacy-test authorization fixture (Phase 45 module excluded).
* `tests/test_phase26_qualification_gate.py` — instrument the private qualification builder.
* `tests/test_phase27_fail_closed_gate.py` — instrument the private qualification builder.
* `tests/test_phase43_corrective.py` — old purpose test asserts rejection.

**Not modified:** `configs/paper_reproduction_r1_gpt4o.yaml`, `configs/cloud_qualification.yaml`,
`configs/qualification_r1_gpt4o.yaml`, HarmBench, NBF checkpoint, historical v1 report, and the
Phase 40B/41/42/43/44 reports. `scripts/cloud_pilot.py` was not changed in Phase 45; it already
passes the auth paths to `build_role_llms`.

**Attribution limitation:** the working tree has no Phase 45 commit boundary; the list above is the
Phase 45 change set, while the rest of the cumulative diff predates it.

---

## 16. Remaining limitations and unverified claims

1. **Python module privacy is not absolute.** The underscore-private `_authorized` context can be
   imported by arbitrary code. The removed public string is no longer a bypass, and the
   qualification builder has a call-site + confirmation check, but a determined caller can still
   open the context manually. A future hardening could isolate the context in a separate process or
   enforce it at a process/OS boundary.
2. **Config-level `paper_reproduction` detection is at the scripts layer.** The client constructor
   sees only the model id. Direct construction of a non-R1 model under a paper-reproduction config is
   not detected at the client layer; the audited scripts and `_cloud_common` builders pass the config
   and gate it. The frozen R1 identity itself is the exact snapshot and is gated at every layer.
3. **Aliases are intentionally excluded.** `gpt-4o` and other aliases are not R1. Broadening the
   rule would need a separate decision and tests.
4. **Legacy tests use a test-only authorization fixture.** It is in `tests/conftest.py`, not
   production, and the Phase 45 module is excluded. A future test asserting fail-closed must live in
   that module or clear the context.
5. **No paid run was executed.** Live provider behavior remains unverified by design.
6. **The four full-suite failures are environment-only.** Immediate causes verified; the historical
   first-failure date is unverified.
7. **Optional P3 hardening was not implemented:** atomic report writes, hashing the exact validated
   decision bytes, requiring report operator fields to be null, and same-timestamp end-to-end
   coverage remain documented gaps, not blockers for the Phase 44 findings.
8. **`run_paper_experiment.py` dry-run uses a mock target.** The runner receives `dry_run=True`; no
   paid request or report is needed. This preserves the non-paid dry-run behavior.

---

## 17. Authorization status

This phase **does not authorize**:

* a qualification run or any paid API call;
* API-key validation, model inference, smoke test, pilot, attack, or experiment;
* Deliverable 1;
* Phase 46 or any subsequent implementation without a separately reviewed prompt.

Phase 46 must independently verify the new construction boundary, the qualification-only builder,
the exact `0.50` threshold, and the null-trigger rejection before any authorization decision.
