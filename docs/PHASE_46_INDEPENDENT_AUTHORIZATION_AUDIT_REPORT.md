# PHASE 46 — INDEPENDENT AUDIT OF THE R1 AUTHORIZATION BOUNDARY

**Audit date:** 2026-11-13
**Mode:** independent, read-only, offline
**Controlling implementation:** Phase 45
**Verdict:** `FAIL — BLOCKING AUTHORIZATION OR REVIEW-GATE DEFECTS REMAIN`

The Phase 44 review-gate defects are closed and independently verified. The authorization boundary
is not closed against in-process Python code:

1. **P0-1** — any caller can import `guardbound.llm.paid_authorization._authorized` and construct
   the frozen R1 client with no qualification report or review decision. I reproduced this with a
   fake SDK backend.
2. **P0-2** — the qualification-only builder relies on a filename string from `inspect.stack()` and
   a caller-controlled `offline_mock` flag. A scratch module named `cloud_model_qualification.py`
   called it with `offline_mock=True`, `confirm_paid_request=False`, and a real role config; it was
   accepted and reached the factory with `client=None` — in production that is a real provider
   construction with no report or confirmation.
3. **P1-1** — an `asyncio` task created inside `_authorized` retains the grant after the parent
   context exits; the child can construct an R1 client after the builder returns.

A passing audit is therefore not supportable. No qualification run or paid activity is authorized.

---

## 1. Scope and restrictions

**Read:** the Phase 45 implementation and report, the Phase 44 findings, the Phase 40B/41/42/43
documents, `paid_authorization.py`, `openai_client.py`, `provider_factory.py`,
`guardbound/llm/__init__.py`, `_cloud_common.py`, `cloud_model_qualification.py`, `cloud_pilot.py`,
`final_analysis.py`, `run_paper_experiment.py`, every paid-model script named in Phase 45, the Phase
45/43/41 tests, and `tests/conftest.py`.

**Executed:** offline pytest suites and read-only scratch probes using fake SDK backends, temporary
reports, monkeypatched factories, and `ContextVar` inspection. No network, credential read,
qualification run, model load, download, smoke test, pilot, attack, or experiment. No repository
file was modified except the creation of this report.

**Threat model:** the audit distinguishes (a) accidental/ordinary use of public constructors and
supported scripts from (b) a caller deliberately importing internal authorization functions or
arbitrary code in the same Python process. Python cannot enforce a boundary against (b) without
moving the credential/paid transport outside the process. The audit records where the Phase 45
implementation protects (a) and where (b) remains open.

---

## 2. Authorization-context bypass

### P0-1 — Importable `_authorized` context constructs R1 without a report

* **Files/lines:** `src/guardbound/llm/paid_authorization.py:24` (`_ACTIVE` ContextVar),
  `:45-60` (`require_construction_authorization`), `:64-72` (`_authorized`);
  `src/guardbound/llm/openai_client.py:330` (constructor check).
* **Reproduction (offline, fake client):**
  ```python
  from guardbound.llm.paid_authorization import _authorized
  from guardbound.llm.openai_client import OpenAIChatLLM
  with _authorized("gpt-4o-2024-08-06"):
      OpenAIChatLLM("gpt-4o-2024-08-06", client=fake.client)
  ```
  Observed:
  ```
  initial []
  DIRECT: blocked
  IMPORT_BYPASS: constructed gpt-4o-2024-08-06 authorized ['gpt-4o-2024-08-06']
  after_context []
  ```
* **Expected:** R1 construction requires a validated report and, when triggered, a bound decision.
* **Actual:** a deliberate import of the underscore-private context opens the grant; no report or
  review validation occurs.
* **Classification:** new Phase 45 design limitation; the Phase 44 public `purpose` string is closed,
  but the replacement is not a security boundary against in-process code.
* **Impact:** under an untrusted-in-process threat model this is a critical authorization bypass; under
  a trusted-code model it is an accepted internal capability. Phase 45's report documents the
  limitation but the implementation cannot claim a closed boundary against arbitrary Python code.
* **Minimum correction:** move credential access and paid transport behind a process/OS boundary
  (a broker that accepts only authorized requests), or explicitly narrow the threat model to trusted
  in-process code and stop claiming a closed boundary. A Python-level ContextVar is not a security
  primitive. Not implemented in Phase 46.

### Context lifecycle (tested)

```
DIRECT: blocked
after_context []
after_exception []
nested_inside ['gpt-4o-2024-08-06', 'gpt-4o-mini']
nested_after ['gpt-4o-2024-08-06']
nested_final []
ONE_MODEL: isolated
ASYNC_CHILD: GRANT_VISIBLE
```

* Normal exit and exception exit reset the context: **pass**.
* Nested contexts restore the previous set: **pass**.
* A grant for one model does not authorize another: **pass**.
* `authorized_models()` only reads state: **pass** (static inspection).
* **P1-1** — an `asyncio` task created inside the context inherits the grant; after the parent exits,
  the child still sees the R1 authorization. A child task can construct an R1 client after the
  builder returns.
  * **Files/lines:** `paid_authorization.py:24,64-72` (ContextVar and context manager).
  * **Expected:** the grant ends when the builder's context ends.
  * **Actual:** `asyncio` copies the context into the child task; the copied grant survives.
  * **Impact:** unexpected authorization lifetime in asynchronous callers; under a trusted-code model
    a residual, under an untrusted model a bypass.
  * **Minimum correction:** use an explicit token passed to the constructor/transport instead of
    ambient context, or forbid task creation inside the builder. Not implemented.

---

## 3. Qualification-only builder

`_cloud_common._build_qualification_role_llm` (`:1058-1091`) checks:

```python
called_from_harness = any(
    Path(frame.filename).name == "cloud_model_qualification.py"
    for frame in inspect.stack()[1:]
)
if not called_from_harness: raise ...
if not offline_mock and not confirm_paid_request: raise ...
with _authorized(model_id):
    return _build_cloud_role(...)
```

### P0-2 — Filename check is spoofable and `offline_mock` does not verify the client

* **Reproduction (offline):** a scratch module named `cloud_model_qualification.py` (outside the
  repository) called the builder with:
  ```python
  common._build_qualification_role_llm(
      "target",
      {"provider": "openai", "model": "gpt-4o-2024-08-06"},  # no client
      object(),
      confirm_paid_request=False,
      offline_mock=True,
  )
  ```
  with `build_cloud_role_llm` monkeypatched to a recorder. Observed:
  ```
  SPOOFED_HARNESS: ACCEPTED [{'role': 'target', 'model': 'gpt-4o-2024-08-06', 'client': None}]
  AUTHORIZED_AFTER []
  ```
  The recorder saw `client=None`; in production the real factory would read the API key and
  construct a real `OpenAIChatLLM` — no report, no review decision, no confirmation.
* **Expected:** `offline_mock=True` must require an actual `OfflineMockClient`; the exemption must be
  tied to a non-spoofable harness capability.
* **Actual:** the filename check is satisfied by a file name; the boolean skips confirmation and does
  not verify the client. The normal harness sets `role_cfg["client"] = OfflineMockClient()`, so the
  shipped path is safe, but the builder does not enforce its own contract.
* **Classification:** new Phase 45 defect (unresolved internal-capability gap).
* **Impact:** a caller that can name a module `cloud_model_qualification.py` can drive a real
  provider construction without the harness's explicit confirmation. Under a trusted-code model this
  is an internal misuse; under the audit's authorization requirement it is blocking.
* **Minimum correction:** when `offline_mock=True`, require
  `isinstance(role_cfg.get("client"), OfflineMockClient)`; otherwise require
  `confirm_paid_request` and a harness capability that is not a source filename. Replace the
  `inspect.stack` check with a token/handle created by the harness's explicit confirmation path.
  Not implemented.

The unrelated caller path is correctly rejected by the filename check
(`OFFLINE_FLAG_CALL: rejected ... outside cloud_model_qualification.py`), and the context resets
after the builder (`AUTHORIZED_AFTER []`). The intended harness offline path is exercised by the
Phase 24/26/27 tests (234 passed).

---

## 4. Exhaustive construction-boundary inventory

| Path | Guard | Ordering |
|---|---|---|
| `OpenAIChatLLM.__init__` (`openai_client.py:330`) | `require_construction_authorization` | before key/client |
| `build_cloud_role_llm` (`provider_factory.py:158-159`) | same | before `os.environ.get` |
| `build_three_model_stack` | fails via `build_cloud_role_llm` | before key/client |
| `build_role_llm` | cloud path via `build_cloud_role_llm`; local path no paid client | before key/client |
| `make_chat_llm` (`guardbound/llm/__init__.py`) | constructor guard | before key/client |
| `build_paid_chat_llm` (`_cloud_common.py:788`) | report gate + context | before key/client |
| `build_paid_role_llm` | report gate + context | before key/client |
| `build_single_role_llm` (`_cloud_common.py:1095`) | report gate + context; `cfg` optional | before key/client |
| `build_role_llms` (`_cloud_common.py:844`) | report gate + context; all-offline-mock exempt | before key/client |
| `_build_qualification_role_llm` (`:1058`) | stack filename + confirm/offline + context | see P0-2 |
| `final_analysis.py` | `build_paid_chat_llm` + auth flags | before key/client |
| `run_paper_experiment.py` | delegated `make_target_llm` + forwarded flags; dry-run mock | before key/client |
| `scripts/cloud_pilot.py` | explicit report/decision + `build_role_llms` | before key/client |
| `cloud_model_qualification.py` | private builder + `--confirm-paid-request` | before key/client (except P0-2) |
| `cloud_smoke_test.py` | pre-`preflight` guard + helper | before key/client |
| `run_attacks.py`, `run_baseline_evals.py`, `judge_asr.py`, `eval_mmlu/mtbench/over_refusal.py`, `run_full_evaluation.py`, `run_reproduction.py`, `demo_steered_chat.py` | helper + correct branch guard | before key/client |
| `scripts/audit_prompt_fidelity.py` | fake backend | not paid |
| `scripts/phase14_1_test_structured_output.py` | local HF | not paid |

**Search results:** no unguarded `OpenAIChatLLM(...)` remains in production scripts except the
private builder path above; no direct OpenAI SDK construction was found in the audited paid paths.
`cloud_pilot` passes the auth paths to `build_role_llms`; `final_analysis` and
`run_paper_experiment` are covered.

### 4.1 Config-level detection

* `build_role_llms` receives the config and applies `paper_reproduction` detection to every R1
  role; **pass**.
* `build_single_role_llm` / `build_paid_chat_llm` / `build_paid_role_llm` accept `cfg=None`. A caller
  with a `paper_reproduction` config and a **non-R1** model can omit `cfg` and bypass the config-level
  rule. The frozen exact R1 snapshot remains gated at the constructor. P2-1.
* The audited scripts pass exact R1 model ids and use the gated helpers; **pass**.

### 4.2 Alias policy

`is_r1_model` (`paid_authorization.py:32-42`) is exact: `gpt-4o-2024-08-06` or
`paper_reproduction`. `gpt-4o` is **not** silently normalized; tested
(`is_r1_paid_model("gpt-4o") is False`). No provider-side alias-resolution evidence was found, so
the practical consequence is: an alias run is outside the frozen R1 identity and must be treated as
a deliberate substitution, not the frozen R1 stack. P2-2.

---

## 5. Review-trigger and exact-threshold matrix

`require_qualification` (`_cloud_common.py:411-470`) reads `declared_verdict` before the trigger
block and enforces the exact threshold and the qualified-null rules.

| Threshold | Observed | Triggered | Result |
|---:|---:|---|---|
| 0.50 | 0.49 | true | accepted (triggered) |
| 0.50 | 0.50 | false | accepted (strict `<`) |
| 0.01 | 0.10 | false | rejected (`threshold must equal 0.50`) |
| 0.51 | 0.49 | true | rejected |
| 0.00 / 1.00 | any | any | rejected |
| bool / "0.5" / NaN / infinity | any | any | rejected |
| 0.50 | null | null | rejected for a QUALIFIED report |
| 0.50 | 0.1 | null | rejected for a QUALIFIED report |
| 0.50 | bool | any | rejected for a QUALIFIED report |
| 0.50 | 0.1 | false | rejected (inconsistent) |

Verified by the Phase 45 suite (27 passed) and the Phase 44 gate-edge scratch:
`baseline_no_trigger: ACCEPTED`, `null_observed_triggered: rejected`, all invalid threshold types
rejected. A non-qualified/early-stop report may carry a matched null/null pair, but the verdict gate
then rejects it; it cannot pass qualification.

No CLI argument, environment variable, or report field overrides the threshold; both
`DELIVERY_TYPE_REVIEW_THRESHOLD` constants are `0.50`. **Pass.**

---

## 6. Review-decision binding

The ten-field schema and binding are unchanged from Phase 43. The Phase 43 suite (57 passed) covers:

* correct decision;
* stale report SHA-256;
* wrong run ID;
* wrong configuration SHA-256;
* wrong arm;
* missing fields;
* invalid timestamp;
* eligibility contradiction;
* inactive trigger;
* unused decision.

All reject; the bound decision accepts. Extra fields are not disallowed by the schema and are
ignored (P3-3). The manifest records the decision path and SHA-256
(`cloud_pilot.py` manifest block). **Pass**, with the TOCTOU note in P3-1.

---

## 7. Provenance and persistence

Verified by the Phase 41/43 suites and inspection:

* `schema_version == "qualification.v2"`, `probe_spec_version == "r1-probe.v2"`.
* Valid UTC run ID, canonical filename, filename/payload run-id match.
* Mandatory probes present and `PASS`; `probes_complete is True`; verdict/identity checks.
* Writer refuses an existing report before writing; JSONL append with schema/spec/run ID.
* Newest resolver, same-timestamp ambiguity, no fallback, v1 readable but ineligible.
* Report hash from the file bytes.
* **P3-2** — `common.write_json` is not atomic; a crash can leave a partial report. The gate rejects
  invalid JSON and never falls back, so this is a provenance/operability risk, not a gate bypass.
* **P3-1** — `review_decision_sha256` is computed at manifest-build time, after the decision was
  validated; a file changed between validation and manifest hashing could be recorded under a
  different hash. The validated content is the parsed dict; the manifest hash is a later read.

---

## 8. Compatibility and regression

* `final_analysis.py` has the auth flags and uses `build_paid_chat_llm`; negative and mock paths
  pass. The positive real path requires a key and was not executed.
* `run_paper_experiment.py` forwards both auth paths through `target_llm_namespace`; `--dry-run`
  uses `mock=True`, so no paid client is constructed. Test passes.
* `cloud_smoke_test` dry-run returns before the guard/helper; test passes.
* `run_full_evaluation` and `run_reproduction` return before clients on `--dry-run` (inspection).
* Offline-mock stack builds without a report (test); qualification offline path is exercised by the
  234-test gate suite.
* Non-R1 models are not gated (test).
* Phase 43 metrics/actor/query corrections are preserved (105 tests).
* **P2-3** — `tests/conftest.py:27-38` grants the R1 context for all legacy tests except
  `test_phase45_authorization`; the exclusion is a module-name string. It is test-only and does not
  affect production, but a future fail-closed test placed outside that module could be masked.
  The Phase 45 fail-closed tests are in the excluded module and pass for the right reason.

---

## 9. Tests executed

Environment: Windows, Python 3.13, `torch 2.14.1+cpu`, `torch.cuda.is_available() == False`;
`KMP_DUPLICATE_LIB_OK=TRUE` for the local OpenMP runtime.

```
python -m pytest tests/test_phase45_authorization.py -q
```
**27 passed in 1.10s.** Exit 0.

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
**234 passed in 7.60s.** Exit 0.

```
python -m pytest tests/ -q
```
**1468 passed, 4 failed, 0 skipped in 156.58s.** Exit 1.

Failures, independently reproduced:

* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring`
* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured`
* `test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid`
* `test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present`

Verified immediate causes: CPU-only `torch 2.14.1+cpu` (`cuda_available == False`) and
`Qwen/Qwen3.5-4B` absent from the HF cache. The failing files are not in the working-tree diff. The
historical pre-existence claim is **unverified**; they are not Phase 45/46 changes.

**Read-only audit probes:**

```
DIRECT: blocked
IMPORT_BYPASS: constructed gpt-4o-2024-08-06
after_context [] / after_exception []
nested inside/after correct; ONE_MODEL: isolated
ASYNC_CHILD: GRANT_VISIBLE
OFFLINE_FLAG_CALL: rejected (unrelated caller)
SPOOFED_HARNESS: ACCEPTED [{'model': 'gpt-4o-2024-08-06', 'client': None}]
```

---

## 10. Frozen artifact integrity

| Artifact | Full SHA-256 | Status |
|---|---|---|
| `configs/paper_reproduction_r1_gpt4o.yaml` | `2125cc3b233efde9e97a8407c364c2140091245dd39bcf440c4366fd99ef393b` | match |
| HarmBench `harmbench_tasks.json` | `ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb` | match |
| NBF checkpoint | `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` | match |
| Historical v1 report | `b17d392a39eeeda64d561017073addd88376d3dafa394bd732c1ba2e908aeaeb` | match |
| Phase 40 v1 design | `0f9371da149535ec0d18572dc4150543985ac2cfaae63266bddcb7c306187a58` | match |
| Phase 40B design | `9152f89e5a951f00f6a01efe1cef681875f4c4769864474b912b646cc57f9ad3` | match |
| Phase 41 report | `c42a71983ad27472e9dd1a6b35571db10bb8c979343825f05f762469baa9d8ff` | match |
| Phase 42 report | `369617b8780b424c8a8d4aed3474f527b0908a3def4ca3268fe75fde33249e97` | match |
| Phase 43 report | `8857e515ce638940140c7c8c2cd5226cae0941b1c78b44d6733b608597b1c9e3` | match |
| Phase 44 report | `da9ccdf379bb7eb172da302d8700e69493aed7ee6440253a1ad0955b2ed1e1f7` | match |
| Phase 45 report | `685bb7770994ee8ed7730f7f6c425e344341153d5e17a44944116d7452b9bb81` | match |

No credential or `sk-` pattern appears in the inspected files. No frozen artifact was modified.

---

## 11. Findings

### P0-1 — Importable `_authorized` context bypasses the report gate

* **Severity:** P0 under an untrusted-in-process threat model; accepted internal capability under a
  documented trusted-code model.
* **Location:** `src/guardbound/llm/paid_authorization.py:24,45-72`;
  `src/guardbound/llm/openai_client.py:330`.
* **Evidence:** `IMPORT_BYPASS: constructed gpt-4o-2024-08-06` with no report/decision.
* **Expected/actual:** R1 construction requires validated authorization; a deliberate import opens
  the context and constructs.
* **Classification:** new design limitation; Phase 44's public string is closed, but the boundary is
  not enforced against in-process code.
* **Blocking:** yes, unless the project explicitly adopts a trusted-code threat model. A Python
  ContextVar cannot be a security boundary against arbitrary code in the same process.
* **Minimum correction:** move credential/paid transport behind a process/OS boundary or an
  authorization broker; or document a trusted-code model and remove the claim of a closed boundary.

### P0-2 — Qualification builder: spoofable filename and unverified offline flag

* **Severity:** P0/P1.
* **Location:** `scripts/_cloud_common.py:1058-1091`.
* **Evidence:** a scratch module named `cloud_model_qualification.py` called the builder with
  `offline_mock=True`, `confirm_paid_request=False`, and a role config with no client; it was
  accepted and reached the factory with `client=None`.
* **Expected:** offline mode requires an actual `OfflineMockClient`; the harness exemption is a
  non-spoofable capability, not a filename string.
* **Actual:** the filename check is satisfied by naming a file; the boolean skips confirmation and
  does not inspect the client.
* **Classification:** new Phase 45 defect.
* **Blocking:** yes under the audit's authorization requirement; the shipped harness sets the mock
  client, so the normal path is safe, but the builder does not enforce it.
* **Minimum correction:** require `isinstance(role_cfg.get("client"), OfflineMockClient)` when
  `offline_mock=True`; replace `inspect.stack` with a harness-issued capability/token. Not implemented.

### P1-1 — Async child task inherits the grant

* **Severity:** P1 (context lifetime) under an untrusted model; P2 under trusted code.
* **Location:** `paid_authorization.py:24,64-72`.
* **Evidence:** `ASYNC_CHILD: GRANT_VISIBLE` after the parent context exits.
* **Expected/actual:** the grant ends with the builder; an `asyncio` task created inside the context
  retains a copied grant and can construct an R1 client later.
* **Minimum correction:** explicit token passed to the constructor/transport rather than ambient
  context, or forbid task creation inside the builder.

### P2-1 — Config-level detection can be skipped by omitting `cfg`

* **Severity:** P2.
* **Location:** `_cloud_common.py:788,1095` (`cfg=None` on the public helpers).
* **Expected/actual:** `build_role_llms` applies `paper_reproduction`; a caller of
  `build_single_role_llm`/`build_paid_chat_llm` with a paper config and a non-R1 model can omit
  `cfg` and skip the config-level rule. The exact R1 snapshot remains gated at the constructor.
* **Minimum correction:** require `cfg` for paper-reproduction callers or infer identity from a
  mandatory config object; document the caller contract.

### P2-2 — Alias policy is exact, not normalized

* **Severity:** P2/P3.
* **Location:** `paid_authorization.py:32-42`; `_cloud_common.is_r1_paid_model`.
* **Expected/actual:** `gpt-4o-2024-08-06` is R1; `gpt-4o` is not silently R1. This is documented
  and tested. No provider-side alias resolution evidence was found; an alias run is a deliberate
  model substitution outside the frozen R1 identity.

### P2-3 — Test-only authorization fixture is name-scoped

* **Severity:** P3.
* **Location:** `tests/conftest.py:27-38`.
* **Expected/actual:** the fixture grants R1 for legacy tests and excludes `test_phase45_authorization`
  by module-name substring. It is not production code, and the fail-closed tests are in the excluded
  module; a future fail-closed test outside that module could be masked. Document the rule.

### P3 findings

| ID | Finding | Evidence / correction |
|---|---|---|
| P3-1 | `review_decision_sha256` is computed at manifest-build time, after validation | A file changed between validation and manifest hashing could be recorded under a different hash; hash the exact validated bytes |
| P3-2 | `common.write_json` is non-atomic | A crash can leave a partial report; the gate fails closed and never falls back, but the operator must remove it |
| P3-3 | Extra review-decision fields are accepted | The schema does not disallow extras; no bypass, but add `additionalProperties` semantics if strictness is required |
| P3-4 | `final_analysis` positive real path unexecuted | Negative/mock paths pass; a positive construction with a real key was not executed (no paid activity) |

**Review gate and exact threshold: no findings.** The P1 review-trigger defects from Phase 44 are
independently verified fixed (27 Phase 45 tests, gate-edge scratch, boundary matrix). The review
decision binding is verified (Phase 43 suite 57).

---

## 12. Authorization status

No API call, credential read, model inference, download, qualification run, smoke test, pilot,
attack, or experiment occurred. The repository was not modified except for this report.

**A future qualification authorization request may not be considered yet.** The review gate is
technically ready, but the authorization boundary is not closed against in-process Python code:
P0-1 (`_authorized` import), P0-2 (spoofable qualification builder / offline flag), and P1-1 (async
grant inheritance) must be corrected or the threat model must be explicitly narrowed to trusted
in-process code with the offline-mock client enforced. The four full-suite failures are environment
limitations and are not authorization evidence.

A passing Phase 46 audit, had one been issued, would not itself authorize paid activity. This audit
does not pass.
