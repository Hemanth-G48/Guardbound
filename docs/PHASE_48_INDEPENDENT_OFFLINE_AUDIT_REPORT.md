# PHASE 48 — INDEPENDENT OFFLINE AUDIT OF PHASE 47 AUTHORIZATION LIFECYCLE

**Audit date:** 2026-11-13
**Mode:** independent, read-only, offline
**Controlling implementation:** Phase 47
**Controlling audit:** Phase 46
**Verdict:** `PASS WITH FINDINGS — CORRECTIONS REQUIRED BEFORE AUTHORIZATION`

All Phase 46 P0/P1 defects are independently verified closed **under the adopted trusted in-process
code threat model**. One P2 configuration-forwarding gap remains in `run_reproduction.py`; it does
not permit construction of the frozen R1 model, which remains gated at the client layer. No
qualification run, paid call, or Deliverable 1 execution is authorized.

---

## 1. Scope and no-paid-activity confirmation

**Read:** the Phase 47 implementation and report, the Phase 46 findings, the Phase 40B/41/42/43
documents, `paid_authorization.py`, `openai_client.py`, `provider_factory.py`,
`guardbound/llm/__init__.py`, `_cloud_common.py`, `cloud_model_qualification.py`, `cloud_pilot.py`,
`final_analysis.py`, `run_paper_experiment.py`, `run_reproduction.py`, every paid-model script,
`tests/conftest.py`, and the Phase 47/45/43/41 tests.

**Executed:** offline pytest suites and isolated scratch probes using fake SDK backends, monkeypatched
factories, and temporary files. No network, credential read, qualification run, model inference,
download, smoke test, pilot, attack, or experiment. No repository file was modified except this
report.

---

## 2. Adopted threat model and its limits

The implementation and Phase 47 report consistently state:

* repository-owned code is trusted not to deliberately import and misuse internal capabilities;
* public builders and supported scripts enforce the report/review gate;
* accidental direct construction fails closed;
* offline mock execution cannot create a live provider;
* internal capabilities are implementation details, not security primitives;
* arbitrary hostile Python code in the same process is outside the guarantee;
* a separate process/OS credential and transport broker would be required for an untrusted
  in-process model.

`paid_authorization.py`'s module docstring and `QualificationCapability` docstring explicitly say the
ContextVar, underscore-prefixed functions, and dataclass seal are **not** security boundaries. No
documentation reviewed claims otherwise.

**Direct import result (fake client):**

```
DIRECT_BLOCKED: yes
IMPORT_BYPASS gpt-4o-2024-08-06 after []
```

The import remains possible by design. Under the adopted model this is an accepted internal
capability, not a new defect. If the project requires protection against hostile same-process code,
this remains a P0 and needs the broker; Phase 47 does not claim to solve that.

---

## 3. Phase 46 findings-to-verification matrix

| Phase 46 finding | Independent result | Evidence |
|---|---|---|
| **P0-1** importable `_authorized` | **Accepted limitation under the adopted trusted-code model**; not a new defect | threat model documented; direct import reproduces the documented behavior |
| **P0-2** spoofable builder / unverified offline flag | **Closed** | explicit `QualificationCapability`; no `inspect.stack`; offline requires `OfflineMockClient`; 22 Phase 47 tests |
| **P1-1** async grant inheritance | **Closed** | owner + released scope; `ASYNC_CHILD: blocked`; copied context blocked |
| **P2-1** optional `cfg` | **Partial — P2 finding** | exact R1 gated with `cfg=None`; supplied paper config gates non-R1; `run_reproduction` does not forward its `cfg`; public helpers rely on caller discipline for a paper config with a non-R1 model |

---

## 4. Qualification capability and offline-mock audit

`src/guardbound/llm/paid_authorization.py:59-95` defines a frozen `QualificationCapability` with a
sealed constructor and `_issue_qualification_capability`. `_cloud_common.py:1069-1080` exposes the
private issuer; `_build_qualification_role_llm` (`:1082-1122`) requires the capability and checks:

* `isinstance(capability, QualificationCapability)` — missing → `TypeError`; string/object →
  `QualificationError`;
* offline mode → `isinstance(role_cfg["client"], OfflineMockClient)`;
* live mode → `capability.confirmed`.

There is no `inspect.stack()`, filename, `purpose`, or standalone boolean on the builder. The
harness creates the capability at `cloud_model_qualification.py:2161` after its confirmation/offline
decision and passes it at `:2270`.

**Independent scratch results:**

```
CAP_MISSING rejected
OFFLINE_NONE rejected
OFFLINE_MOCK accepted True after []
```

`OFFLINE_MOCK accepted True` confirms the actual `OfflineMockClient` reached the factory; the
factory recorder saw that exact object. The builder rejects `client=None`, an ordinary SDK client,
and `object()` before the factory (the Phase 47 tests cover all three; the scratch confirms the
`None` case). The accepted mock is the client used downstream: `_build_cloud_role` passes
`role_cfg["client"]` to `build_cloud_role_llm`, which passes it to `OpenAIChatLLM`. There is no
fall-through from offline to a live client: a non-mock client is rejected before
`_build_cloud_role`.

The capability states are consistent: `_issue` rejects an unconfirmed live capability; the builder
also checks `confirmed`; an offline capability with `confirmed=False` is allowed only with an actual
mock.

**Result: pass.**

---

## 5. Authorization lifecycle and async audit

`paid_authorization.py:41-57` defines `_Scope(owner, released)` and `model -> _Scope`.
`require_construction_authorization` (`:122-139`) requires the scope present, unreleased, and owned by
the current execution context (`asyncio.current_task()` or `_SYNC_OWNER`). `_authorized` (`:142-161`)
sets the mapping and, in `finally`, marks the scope released and resets the ContextVar.
`authorized_models` (`:164-172`) returns only active, current-owner models.

**Independent results:**

```
DIRECT_BLOCKED: yes
IMPORT_BYPASS gpt-4o-2024-08-06 after []      # documented internal capability
ASYNC_CHILD blocked
COPIED_CONTEXT blocked
CAP_MISSING rejected
OFFLINE_NONE rejected
OFFLINE_MOCK accepted True after []
```

The Phase 47 suite also verifies normal/exception exit, nested restore, model isolation, failed
construction, and copied-context release. A child task created inside the scope has a different owner
and, after the parent exits, a released scope; it cannot construct. `authorized_models` does not
report the foreign-owner grant.

**Residual:** a caller that imports `_authorized` can create a fresh scope; that is the documented
internal capability. `authorized_models` is read-only and does not expose stale grants as active.

**Result: pass under the adopted model.**

---

## 6. Configuration-contract audit

**Mechanically enforced:**

* The exact frozen snapshot `gpt-4o-2024-08-06` is gated at `OpenAIChatLLM.__init__`
  (`openai_client.py:330`) and `build_cloud_role_llm` (`provider_factory.py:158-159`), even with
  `cfg=None`.
* A supplied `paper_reproduction: true` config gates a non-R1 model through
  `build_paid_chat_llm`, `build_paid_role_llm`, and `build_single_role_llm` (Phase 47 tests;
  scratch `PAPER_NON_R1 rejected`).
* `cloud_pilot` — the supported R1 paper path — passes its config to `build_role_llms`
  (`_cloud_common.py:916`), which applies the config rule to every role.
* `build_role_llms` gates every R1 role when the config says `paper_reproduction`.

**P2 finding — caller discipline and one script forwarding gap:**

* `build_paid_chat_llm` (`_cloud_common.py:789`), `build_paid_role_llm` (`:818`), and
  `build_single_role_llm` (`:1124`) accept `cfg=None`. The docstrings state that a caller with a
  paper-reproduction config must pass it, and `cfg=None` is the non-paper/exact-model path. This is
  a documented caller contract, not a mechanical check for a **non-R1** model under a paper config.
* `run_reproduction.make_llm` (`run_reproduction.py:289,296,308`) has `cfg` in scope but calls
  `build_paid_role_llm(..., manager=manager)`, `guard_paid_model(model, args)`, and
  `build_paid_chat_llm(model, args=args)` **without** `cfg=cfg`. The script's supported
  `configs/reproduction*.yaml` are not `paper_reproduction`; the R1 paper config does not satisfy
  `validate_config`'s schema (`attacker.model`, `evaluator.rubric_model`, `nbf.trials`, …). A hybrid
  config with `paper_reproduction: true` plus the reproduction schema could be accepted, and a
  non-R1 `--target` override would then bypass the config-level rule.
* The exact R1 constructor/factory guard still prevents the **frozen R1 model** from being
  constructed in all these paths. The bypass is a non-R1 substitution under a paper config, not R1
  execution.

**Result: P2; non-blocking for the frozen R1 gate, but the forwarding gap and the caller-discipline
contract should be corrected before a paper-config substitution is authorized.** Minimum correction:
pass `cfg=cfg` in all three `run_reproduction` call sites and/or require `cfg` for any
paper-reproduction helper call.

Alias policy is exact: `gpt-4o-2024-08-06` is R1; `gpt-4o` is not. Tested. An alias is a deliberate
substitution, not the frozen R1 identity.

---

## 7. Review-gate audit (Phase 44 preservation)

Inspection and the Phase 47/43/45 tests confirm:

* `QUALIFIED` reports require a finite numeric `observed` and boolean `triggered`; null/missing
  reject.
* Threshold is exactly `0.50`; comparison strictly `< 0.50`; `0.49` triggers, `0.50` does not.
* `0.01`, `0.51`, `0.00`, `1.00`, booleans, strings, NaN, and infinity reject.
* Ten-field decision binding; stale/mismatched/late/unused decisions reject.
* The gate-edge scratch: baseline accepted; `null_observed_triggered` rejected; all invalid
  threshold types rejected.

**Result: pass.**

---

## 8. Phase 41/43 regression audit

The Phase 47/43/41/qualification suites pass:

* `qualification.v2` / `r1-probe.v2`; UTC run IDs; canonical immutable filenames; no overwrite;
  newest resolver/no fallback; v1 ineligible.
* Delivery metrics from completed rows; kept-actor validation; raw `questions`; `diagnostics=None`
  parity.
* Qualification capability and offline path; Phase 24/26/27 harness offline tests (234).

**Result: pass.**

---

## 9. Test commands and exact results

Environment: Windows, Python 3.13, `torch 2.14.1+cpu`, `torch.cuda.is_available() == False`;
`KMP_DUPLICATE_LIB_OK=TRUE` for the local OpenMP runtime.

```
python -m pytest tests/test_phase47_authorization_lifecycle.py -q
```
**22 passed in 0.93s.** Exit 0.

```
python -m pytest tests/test_phase45_authorization.py -q
```
**27 passed in 1.08s.** Exit 0.

```
python -m pytest tests/test_phase43_corrective.py tests/test_phase41_probe_redesign.py -q
```
**105 passed in 0.53s.** Exit 0.

```
python -m pytest tests/test_cloud_qualification_harness.py \
  tests/test_phase24_p0_fixes.py tests/test_phase26_qualification_gate.py \
  tests/test_phase27_fail_closed_gate.py tests/test_phase36_implementation.py \
  tests/test_phase38_hardening.py tests/test_attacker_arms.py -q
```
**234 passed in 6.65s.** Exit 0.

```
python -m pytest tests/ -q
```
**1490 passed, 4 failed, 0 skipped in 178.32s.** Exit 1.

Failures, independently reproduced:

* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring`
* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured`
* `test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid`
* `test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present`

Verified causes: CPU-only `torch 2.14.1+cpu` and missing `Qwen/Qwen3.5-4B`. The failing files are
not in the working-tree diff. Historical pre-existence remains unverified; they are not Phase 47/48
changes.

---

## 10. Frozen-artifact hash results

| Artifact | Full SHA-256 | Status |
|---|---|---|
| R1 config | `2125cc3b233efde9e97a8407c364c2140091245dd39bcf440c4366fd99ef393b` | match |
| HarmBench | `ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb` | match |
| NBF checkpoint | `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` | match |
| Historical v1 report | `b17d392a39eeeda64d561017073addd88376d3dafa394bd732c1ba2e908aeaeb` | match |
| Phase 40 v1 design | `0f9371da149535ec0d18572dc4150543985ac2cfaae63266bddcb7c306187a58` | match |
| Phase 40B design | `9152f89e5a951f00f6a01efe1cef681875f4c4769864474b912b646cc57f9ad3` | match |
| Phase 41 report | `c42a71983ad27472e9dd1a6b35571db10bb8c979343825f05f762469baa9d8ff` | match |
| Phase 42 report | `369617b8780b424c8a8d4aed3474f527b0908a3def4ca3268fe75fde33249e97` | match |
| Phase 43 report | `8857e515ce638940140c7c8c2cd5226cae0941b1c78b44d6733b608597b1c9e3` | match |
| Phase 44 report | `da9ccdf379bb7eb172da302d8700e69493aed7ee6440253a1ad0955b2ed1e1f7` | match |
| Phase 45 report | `685bb7770994ee8ed7730f7f6c425e344341153d5e17a44944116d7452b9bb81` | match |
| Phase 46 report | `8ddbdf683974cfac72f671c4eb11a45d947a8d756d7fbb05c90e65d4658fe6ee` | match |
| Phase 47 report | `1a332bd1df8881a775e09067b50ef6da58b1ff3abf0fee0347e1233c0cebd44d` | match |

No credential or `sk-` pattern was found in the inspected files. No frozen artifact was modified.

---

## 11. Findings and residual risks

### P2-1 — Configuration-level rule is caller discipline; `run_reproduction` does not forward `cfg`

* **Severity:** P2.
* **Location:** `_cloud_common.py:789,818,1124` (`cfg=None` helpers);
  `run_reproduction.py:289,296,308` (no `cfg=cfg`).
* **Expected:** every paper-reproduction call forwards the configuration; the config-level rule is
  mechanically enforced.
* **Actual:** the exact R1 snapshot is mechanically gated, and a supplied paper config gates non-R1
  models, but the public helpers allow `cfg=None` (documented caller contract), and
  `run_reproduction` has `cfg` but does not forward it. A hybrid `paper_reproduction` reproduction
  config with a non-R1 `--target` could bypass the config-level rule.
* **Impact:** a non-R1 substitution under a paper config can run without the report/review gate; the
  frozen R1 model remains blocked by the constructor/factory guard. Non-blocking for R1 execution.
* **Correction:** pass `cfg=cfg` at all three `run_reproduction` call sites and/or require `cfg` for
  any paper-reproduction helper call.
* **Classification:** Phase 46 P2-1 residual; not a P0/P1 blocker for the frozen R1 stack.

### Accepted threat-model limitation (P0-1)

The importable `_authorized` and `_qualification_capability`/seal remain internal capabilities. The
implementation/docs consistently state this is not a security boundary. Under the trusted-code model
this is accepted; under an untrusted in-process model a process/OS broker is required. Not a new
defect for Phase 48.

### P3 residuals

| ID | Finding | Status |
|---|---|---|
| P3-1 | `common.write_json` is non-atomic; a crash can leave a partial report | Accepted residual; the gate fails closed and never falls back |
| P3-2 | `review_decision_sha256` is computed at manifest-build time, after validation | Accepted residual; hash the exact validated bytes in a future pass |
| P3-3 | Extra review-decision fields are ignored | Accepted; no bypass |
| P3-4 | `tests/conftest.py:27-43` grants R1 for non-Phase-45/47 tests | Test-only; the fail-closed modules are excluded; future fail-closed tests must use an excluded module |
| P3-5 | Same-second run-ID collision raises and `main` catches it; resolver ambiguity is unit-level | Accepted; no silent retry/fallback |
| P3-6 | Live provider behavior is unverified | By design; no paid run authorized |

---

## 12. Authorization status

This audit **does not authorize**:

* a qualification run or any paid API call;
* API-key validation, model inference, downloads, smoke tests, pilots, attacks, or experiments;
* Deliverable 1;
* Phase 49 or any subsequent implementation without a separately reviewed prompt.

The Phase 46 P0/P1 defects are independently verified closed under the adopted trusted in-process
code model. The P2 configuration-forwarding gap should be corrected before a paper-config
substitution is authorized. A `PASS WITH FINDINGS` verdict is not permission to spend money or
access provider credentials; a separate, explicit human authorization decision is required.
