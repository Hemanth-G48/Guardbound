# PHASE 47 — AUTHORIZATION CAPABILITY AND QUALIFICATION BUILDER CORRECTIONS

**Phase:** 47 (offline implementation only)
**Date:** 2026-11-13
**Controlling audit:** Phase 46
**Verdict:** `IMPLEMENTATION COMPLETE — READY FOR PHASE 48 INDEPENDENT OFFLINE AUDIT`

This phase implements the Phase 46 corrective requirements and formally adopts the trusted
in-process code threat model. It does **not** claim protection against arbitrary hostile Python code
in the same process; a Python `ContextVar`, an underscore-prefixed function, a dataclass seal, or a
filename check is not a security boundary. No qualification run, paid call, or Deliverable 1
execution is authorized.

---

## 1. Executive summary and threat model

### 1.1 Adopted threat model — trusted in-process code

* Repository-owned application code is trusted not to deliberately import and misuse internal
  authorization capabilities.
* Public construction APIs and supported scripts enforce the report/review gate.
* Accidental direct construction fails closed.
* Offline mock execution cannot create a live provider client.
* Internal capabilities are implementation details, not security primitives.
* Arbitrary hostile Python code in the same process is **outside** the protection guarantee.

A Python `ContextVar`, an underscore-prefixed function, a private module, a dataclass seal, or an
inspect-stack check cannot stop arbitrary code in the same process from deliberately bypassing
application policy. If the project later requires protection against untrusted in-process code,
credential access and paid transport must move behind a separate process/OS-level broker. **No broker
is implemented in this phase**, and this report does not claim one.

### 1.2 What changed

* The qualification-only builder no longer uses `inspect.stack()` or a filename. It requires an
  explicit `QualificationCapability` object created by the harness's execution path.
* Offline mode now requires an actual `OfflineMockClient`; `client=None`, an ordinary SDK client,
  and arbitrary objects are rejected before any factory/key access.
* The authorization scope is tied to an owning execution context (current asyncio task or the
  synchronous sentinel) and carries a shared `released` flag. A child task or a copied context that
  outlives the builder sees `released=True` and cannot construct an R1 client.
* The `cfg=None` contract is documented: it is the non-paper/exact-model path; a caller with a
  `paper_reproduction` config must pass it, and the frozen R1 snapshot remains gated at the client
  layer regardless.
* The Phase 44/45 review gate, threshold, decision binding, and Phase 41/43 corrections are
  unchanged and re-tested.

### 1.3 Verification

* Phase 47 suite: **22 passed**.
* Phase 45 suite: **27 passed**.
* Phase 43 + Phase 41: **105 passed**.
* Qualification/gate suites: **234 passed**.
* Full offline suite: **1490 passed, 4 failed, 0 skipped** in 177.25s.
* Phase 46 async reproduction now returns `ASYNC_CHILD: blocked`.

The four full-suite failures are the established environment checks: CPU-only `torch 2.14.1+cpu`
and missing `Qwen/Qwen3.5-4B`. They are not Phase 47 changes.

---

## 2. Phase 46 findings-to-fix mapping

| Phase 46 finding | Status | Correction and evidence |
|---|---|---|
| **P0-1** importable `_authorized` constructs R1 without a report | **Threat model formally adopted; capability remains internal by design** | The trusted in-process code model is documented in `paid_authorization.py` and this report. The import is an internal capability, not a security boundary. The phase does **not** claim protection against arbitrary in-process code. A process/OS broker is the required correction if that threat model is needed; it is not implemented. |
| **P0-2** spoofable filename + unverified `offline_mock` | **Fixed** | `_build_qualification_role_llm` requires `QualificationCapability`; no `inspect.stack`; offline requires `isinstance(role_cfg["client"], OfflineMockClient)`. Tests: spoofed filename without capability → `TypeError`; `client=None`/ordinary/object → `QualificationError`; actual mock → accepted. |
| **P1-1** async child inherits the grant | **Fixed** | `_Scope` with `owner` + `released`; `require_construction_authorization` checks owner and release; the child task and copied context are blocked after the parent exits. Tests. |
| **P2-1** optional `cfg=None` | **Documented contract** | `build_paid_chat_llm`, `build_paid_role_llm`, and `build_single_role_llm` docstrings state: pass `cfg` for any paper-reproduction configuration; `cfg=None` is the non-paper/exact-model path. Tests: exact R1 gated with `cfg=None`; a supplied paper config gates a non-R1 model; `cfg=None` non-R1 path remains compatible. |

---

## 3. Qualification-builder changes

### 3.1 Explicit capability

`src/guardbound/llm/paid_authorization.py` now defines:

```python
@dataclass(frozen=True, slots=True)
class QualificationCapability:
    offline_mock: bool
    confirmed: bool
    _seal: object
```

* Created only through `_issue_qualification_capability` (private by convention).
* `__post_init__` rejects an unsealed construction, non-boolean fields, and a live capability
  without confirmation.
* The seal is **not cryptographic**; arbitrary hostile code can import it. This is consistent with
  the trusted-code model.

`_cloud_common._qualification_capability(offline_mock, confirmed)` is the harness-facing issuer.
The harness creates it only after its existing confirmation/offline decision and passes it to the
builder. The builder has **no** `purpose`, `confirm_paid_request`, or `offline_mock` parameter.

### 3.2 Builder enforcement

```python
def _build_qualification_role_llm(..., *, capability: QualificationCapability):
    if not isinstance(capability, QualificationCapability): raise QualificationError
    if capability.offline_mock:
        if not isinstance(role_cfg.get("client"), OfflineMockClient):
            raise QualificationError("offline qualification requires OfflineMockClient")
    elif not capability.confirmed:
        raise QualificationError("live qualification requires --confirm-paid-request")
    with _authorized(model_id):
        return _build_cloud_role(...)
```

* No `inspect.stack()`; a filename is not a capability.
* The capability type check rejects a string or arbitrary object before any factory/key access.
* Offline mode rejects `client=None`, an ordinary `FakeOpenAIBackend().client`, and `object()`;
  accepts only `OfflineMockClient`.
* Live mode requires `confirmed=True`; the harness's `--confirm-paid-request` gate remains.
* The pattern is tested with a fake factory; no live client or key is touched.

### 3.3 Normal harness path

`cloud_model_qualification.main` creates the capability after its existing
`--confirm-paid-request`/`--offline-mock` decision and passes it to every role builder. The offline
tests set `role_cfg["client"] = OfflineMockClient()` before the builder. The report schema, role
config, arm handling, and identity are unchanged. The existing Phase 24/26/27 qualification suites
pass (234).

---

## 4. Offline-mock enforcement evidence

| Role client | Capability | Result |
|---|---|---|
| `OfflineMockClient` | `offline_mock=True` | accepted; factory receives the mock |
| `None` | `offline_mock=True` | `QualificationError` before factory |
| `FakeOpenAIBackend().client` | `offline_mock=True` | `QualificationError` before factory |
| `object()` | `offline_mock=True` | `QualificationError` before factory |
| `object()` / string | any | `QualificationError` (not a `QualificationCapability`) |
| no capability | any | `TypeError` (required keyword) |

The Phase 46 spoofed-harness case (a module named `cloud_model_qualification.py` with
`offline_mock=True` and `client=None`) no longer reaches the factory: the builder requires the
capability and the offline client. The old scratch called the removed keyword arguments and now
raises `TypeError`; the Phase 47 test asserts the new contract.

---

## 5. Authorization lifecycle and async results

`paid_authorization.py` now stores `model -> _Scope`:

* `_Scope.owner` is the current asyncio task or `_SYNC_OWNER`.
* `_Scope.released` is set in the context manager's `finally`.
* `require_construction_authorization` requires the model's scope to be present, unreleased, and
  owned by the current execution context.

Verified behavior:

| Scenario | Result |
|---|---|
| Direct R1 construction without a scope | `PaidAuthorizationRequired` |
| Synchronous construction inside `_authorized` | succeeds with a fake client |
| Normal exit / exception exit | scope released; subsequent construction blocked |
| Nested scopes | outer state restored correctly |
| One model authorized, another constructed | blocked |
| Child `asyncio` task created inside, constructs after parent exit | `ASYNC_CHILD: blocked` |
| `contextvars.copy_context()` run after parent exit | blocked |
| Failed builder (non-cloud role) | no grant remains |
| `authorized_models()` | read-only, active + current-owner only |

The Phase 46 `ASYNC_CHILD: GRANT_VISIBLE` reproduction now returns `blocked`. The direct
`_authorized` import still constructs an R1 client; that is the documented internal capability, not
a claim of protection against arbitrary in-process code.

---

## 6. Configuration enforcement results

* Exact frozen snapshot `gpt-4o-2024-08-06` remains gated with `cfg=None` at the constructor and
  factory layers.
* A supplied `paper_reproduction: true` config gates a non-R1 model through
  `build_paid_chat_llm`, `build_paid_role_llm`, and `build_single_role_llm`.
* `build_role_llms` applies the config-level rule to every role (Phase 45 behavior preserved).
* `cfg=None` is documented as the non-paper/exact-model path; a caller with a paper config must pass
  it. The non-R1 `cfg=None` path remains compatible.
* Alias rule is exact: `gpt-4o-2024-08-06` is R1; `gpt-4o` and other aliases are not silently
  normalized. An alias run is a deliberate substitution, not the frozen R1 identity.

---

## 7. Review-gate and Phase 41/43 preservation

Unchanged and re-tested:

* `QUALIFIED` reports require finite numeric `observed` and boolean `triggered`; null/missing are
  rejected.
* The threshold is exactly `0.50`; the comparison is strictly `< 0.50`; `0.49` triggers and `0.50`
  does not; `0.01`, `0.51`, `0.00`, `1.00`, booleans, strings, NaN, and infinity are rejected.
* Ten-field review-decision binding, stale/mismatched/late/unused decisions rejected.
* `qualification.v2` / `r1-probe.v2`, UTC run ID, canonical filenames, immutable persistence,
  newest resolver, no fallback, v1 ineligible.
* Delivery metrics over completed rows, kept-actor validation, raw `questions` validation, and
  `diagnostics=None` parity.

The Phase 47 suite includes a compact threshold/null regression; the Phase 41/43/45 suites provide
the full matrix.

---

## 8. Tests executed

Environment: Windows, Python 3.13, `torch 2.14.1+cpu`, `torch.cuda.is_available() == False`;
`KMP_DUPLICATE_LIB_OK=TRUE` for the local OpenMP runtime.

```
python -m pytest tests/test_phase47_authorization_lifecycle.py -q
```
**22 passed in 0.97s.** Exit 0.

```
python -m pytest tests/test_phase45_authorization.py -q
```
**27 passed in 1.06s.** Exit 0.

```
python -m pytest tests/test_phase43_corrective.py tests/test_phase41_probe_redesign.py -q
```
**105 passed in 0.62s.** Exit 0.

```
python -m pytest tests/test_cloud_qualification_harness.py \
  tests/test_phase24_p0_fixes.py tests/test_phase26_qualification_gate.py \
  tests/test_phase27_fail_closed_gate.py tests/test_phase36_implementation.py \
  tests/test_phase38_hardening.py tests/test_attacker_arms.py -q
```
**234 passed in 7.16s.** Exit 0.

```
python -m pytest tests/ -q
```
**1490 passed, 4 failed, 0 skipped in 177.25s.** Exit 1.

Failures, independently reproduced:

* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring`
* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured`
* `test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid`
* `test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present`

Verified causes: CPU-only `torch 2.14.1+cpu` and missing `Qwen/Qwen3.5-4B`. The failing files are
not in the Phase 47 working-tree diff. Historical pre-existence remains unverified; they are not
Phase 47 changes.

**Read-only audit reproductions against Phase 47:**

```
DIRECT: blocked
IMPORT_BYPASS: constructed gpt-4o-2024-08-06   # documented trusted-code internal capability
after_context [] / after_exception []
nested correct; ONE_MODEL: isolated
ASYNC_CHILD: blocked
```

The direct import remains possible and is explicitly outside the protection guarantee; it is not
claimed as a security boundary.

---

## 9. Frozen artifact integrity

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
| Phase 46 report | `8ddbdf683974cfac72f671c4eb11a45d947a8d756d7fbb05c90e65d4658fe6ee` | match |

No credential or `sk-` pattern was added. No frozen artifact was modified.

---

## 10. Files changed

**Added**

* `tests/test_phase47_authorization_lifecycle.py` — 22 tests.
* `docs/PHASE_47_AUTHORIZATION_LIFECYCLE_IMPLEMENTATION_REPORT.md` — this report.

**Modified**

* `src/guardbound/llm/paid_authorization.py` — explicit `QualificationCapability`; owner/released
  scope; async/copied-context lifetime; threat-model docstring.
* `scripts/_cloud_common.py` — `_qualification_capability`; capability-based builder; offline
  client enforcement; `cfg=None` contract documentation.
* `scripts/cloud_model_qualification.py` — creates and passes the capability; no filename/boolean
  exemption.
* `tests/conftest.py` — excludes the Phase 47 fail-closed module from the legacy R1 fixture.
* `tests/test_phase45_authorization.py` — updated builder-API test (capability required).

**Not changed in Phase 47:** `openai_client.py`, `provider_factory.py` (Phase 45 guards remain),
`cloud_pilot.py`, the runbook, all configs/datasets/checkpoints, and all historical reports.

**Attribution limitation:** the working tree has no Phase 47 commit boundary; the list above is the
Phase 47 change set, while the rest of the cumulative diff predates it.

---

## 11. Remaining limitations

1. **No process/OS security boundary.** `_authorized` and `_qualification_capability` are importable
   internal capabilities. The trusted-code model is explicit; arbitrary hostile in-process code is
   out of scope. A future untrusted-code requirement needs a separate credential/transport broker.
2. **The dataclass seal is not cryptographic.** It stops accidental construction, not deliberate
   import of the private issuer/seal.
3. **`cfg=None` is the documented non-paper path.** A caller with a paper-reproduction config must
   pass it; the exact frozen R1 snapshot remains gated at the client layer.
4. **Aliases are not R1.** `gpt-4o` is a deliberate substitution, not the frozen snapshot.
5. **Live provider behavior is unverified.** No paid run, key access, or model inference occurred.
6. **P3 items from Phase 46 remain open:** non-atomic report writes, decision-hash TOCTOU, extra
   decision fields, and environment-only suite failures.
7. **The legacy test fixture** grants R1 for non-Phase-45/47 tests; future fail-closed tests must be
   in an excluded module or clear the context.

---

## 12. No-paid-activity confirmation

* No OpenAI, Anthropic, or other model API call.
* No API-key access, validation, printing, or copying.
* No qualification run, model inference, download, smoke test, pilot, attack, or experiment.
* No Deliverable 1 execution.
* No credential or secret in any modified file.
* No frozen artifact modified.

Phase 47 completion does not authorize paid activity. Phase 48 must independently audit the
capability, offline enforcement, scope lifetime, and configuration contract before any separate
human authorization decision.
