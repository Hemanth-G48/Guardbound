# PHASE 49 — PAPER-REPRODUCTION CONFIGURATION PROPAGATION CORRECTION

**Phase:** 49 (offline implementation only)
**Date:** 2026-11-13
**Controlling audit:** Phase 48
**Verdict:** `IMPLEMENTATION COMPLETE — READY FOR PHASE 50 INDEPENDENT OFFLINE AUDIT`

This phase closes the Phase 48 P2-1 configuration-forwarding gap in
`run_reproduction.make_llm`. It does **not** change the authorization architecture, the
qualification probes, the report schema, the review threshold, or the frozen R1 configuration. No
qualification run, paid call, or Deliverable 1 execution is authorized.

---

## 1. Executive summary

`run_reproduction.make_llm` had the active `cfg` in scope but called three authorization helpers
without forwarding it:

* `build_paid_role_llm` (three-model path),
* `guard_paid_model` (single-model path),
* `build_paid_chat_llm` (single-model path).

A hybrid `paper_reproduction: true` reproduction configuration with a non-R1 target override could
therefore skip the configuration-level report/review requirement. The exact frozen
`gpt-4o-2024-08-06` model remained gated by the client layer, but the config-level rule was not
reliably applied.

The correction forwards the exact same `cfg` object at all three call sites. Tests verify object
identity at the `run_reproduction.make_llm` entry path, the hybrid-paper rejection without
authorization, non-paper compatibility, the exact-R1 guard with `cfg=None` and through the entry
path, and unchanged reproduction validation.

**Verification:** Phase 49 suite **8 passed**; Phase 47 **22 passed**; Phase 45 **27 passed**;
Phase 43 + 41 **105 passed**; qualification/gate **234 passed**; full offline suite **1498 passed,
4 failed, 0 skipped** in 155.71s. The four failures are the established environment checks
(CPU-only torch; missing `Qwen/Qwen3.5-4B`).

No frozen artifact changed. No credential, API call, model inference, or experiment occurred.

---

## 2. Phase 48 P2-1 finding and root cause

**Finding:** the configuration-level rule depended partly on caller discipline.
`run_reproduction.py` had `cfg` in scope at `make_llm` but passed it to none of the three guards.
The helpers cannot infer a paper-reproduction configuration when `cfg=None`.

**Root cause:** the Phase 47 forwarding fix was applied to the helper signatures and the
`cloud_pilot`/`build_role_llms` path, but the three `run_reproduction` call sites were not updated.
The existing `cfg` object was therefore discarded before the config-level check.

**Impact:** a hybrid `paper_reproduction: true` config with a non-R1 target override could run
without the report/review requirement. The frozen R1 model itself remained blocked by the
constructor/factory guard.

---

## 3. Files and functions changed

| File | Change |
|---|---|
| `scripts/run_reproduction.py` | `make_llm` three-model call now passes `cfg=cfg` to `build_paid_role_llm`; single-model path passes `cfg=cfg` to `guard_paid_model` and `build_paid_chat_llm`; docstring/comment records the Phase 49 contract. |
| `tests/test_phase49_config_propagation.py` | **new** — 8 focused tests. |
| `docs/PHASE_49_CONFIG_PROPAGATION_IMPLEMENTATION_REPORT.md` | **new** — this report. |

No helper signatures, configs, datasets, checkpoints, reports, or frozen artifacts were changed.
`cloud_pilot.py`, `build_role_llms`, the qualification capability, and the review gate are
untouched.

---

## 4. Configuration propagation evidence

All forwarding tests call the supported `run_reproduction.make_llm` entry path and capture the
exact object identity (`is`), not just equality.

| Scenario | Result |
|---|---|
| Three-model path | `build_paid_role_llm` receives `cfg is cfg`, the exact role config, and the same manager |
| Single-model path | `guard_paid_model` receives `cfg is cfg`; `build_paid_chat_llm` receives `cfg is cfg` |
| Hybrid `paper_reproduction: true`, three-model, non-R1 | `QualificationError` without a report; no factory/key |
| Hybrid `paper_reproduction: true`, single-model, non-R1 | `QualificationError` without a report; no key |
| Non-paper, non-R1 | construction proceeds (fake builder); unchanged |
| Exact R1 with `cfg=None` | `QualificationError` at the helper |
| Exact R1 through `make_llm` | `QualificationError` before any client |
| Reproduction validation | `validate_config({})` still raises `SystemExit` |

The hybrid tests use the real guards with no report; the gate raises before any provider
construction. The non-paper test uses the real guard (a no-op) and a fake builder; no real client is
created. The exact-R1 tests use a fake client or the guard; no key is read.

---

## 5. Helper contract review

Inspected: `build_paid_chat_llm`, `build_paid_role_llm`, `build_single_role_llm`,
`guard_paid_model`, `build_role_llms`, and the public provider factories.

* The helpers cannot reliably infer a paper-reproduction configuration when `cfg=None`. Treating
  `cfg=None` as paper would break legitimate non-paper callers; treating it as non-paper is the
  documented contract. No safe mechanical inference exists at that boundary without a broader
  signature/semantic change.
* The exact frozen R1 model is still gated independently of `cfg` at `OpenAIChatLLM.__init__` and
  `build_cloud_role_llm`.
* `build_role_llms` (used by `cloud_pilot`) still receives and applies the config to every role.
* `run_reproduction` now forwards its active `cfg`; the config-level rule is invoked for its
  paper-reproduction path.

**Recorded limitation:** the public helpers' `cfg=None` remains a documented caller contract, not a
mechanical paper-config detection. A caller with a paper config must pass it; the exact R1 model is
unconditionally gated.

**Single-model identity note:** the single-mode `run_reproduction` config has top-level
`attacker`/`target`/`evaluator` blocks, not a `models` block. `_r1_authorization_config` therefore
loads the frozen R1 config for identity; the report/review gate is now invoked, but the report's
role-model identity is the frozen R1 configuration. A non-R1 substitution accompanied by a valid
R1 report is not rejected by role-model identity. This is the pre-existing report-identity design,
not the Phase 49 forwarding gap; binding role models to report identities would be a separate
change and is not implemented here.

---

## 6. Test coverage and exact results

Environment: Windows, Python 3.13, `torch 2.14.1+cpu`, `torch.cuda.is_available() == False`;
`KMP_DUPLICATE_LIB_OK=TRUE` for the local OpenMP runtime.

```
python -m pytest tests/test_phase49_config_propagation.py -q
```
**8 passed in 0.94s.** Exit 0.

```
python -m pytest tests/test_phase47_authorization_lifecycle.py -q
```
**22 passed in 0.95s.** Exit 0.

```
python -m pytest tests/test_phase45_authorization.py -q
```
**27 passed in 1.03s.** Exit 0.

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
**234 passed in 7.26s.** Exit 0.

```
python -m pytest tests/ -q
```
**1498 passed, 4 failed, 0 skipped in 155.71s.** Exit 1.

Failures, independently reproduced:

* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring`
* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured`
* `test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid`
* `test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present`

Verified causes: CPU-only `torch 2.14.1+cpu` and missing `Qwen/Qwen3.5-4B`. The failing files are
not in the Phase 49 working-tree diff. Historical pre-existence remains unverified; they are not
Phase 49 changes.

---

## 7. Regression preservation

* **Phase 47** — `QualificationCapability` enforcement, actual `OfflineMockClient`, owner/released
  scope, async-child blocking, exact R1 guard, alias policy: 22 tests pass.
* **Phase 45** — public-factory gates, offline mock, final_analysis/run_paper paths: 27 tests pass.
* **Phase 44** — review-decision binding, exact `0.50`, strict `<`, null rejection: Phase 43/45
  suites and the Phase 49 test matrix pass.
* **Phase 41/43** — v2/probe/run-id/canonical filenames, immutable persistence, newest resolver,
  v1 ineligibility, delivery metrics, kept actors, raw `questions`: 105 tests pass.
* **Qualification/gate** — `cloud_pilot`/`build_role_llms` behavior: 234 tests pass; the Phase 49
  change does not touch those paths.

---

## 8. Frozen artifact integrity

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
| Phase 48 report | `6606c996aee210781c69fe9114c4a29c0b79379d9d67e48c989b8d871e004e9e` | match |

No credential or `sk-` pattern was added. No frozen artifact was modified.

---

## 9. Remaining limitations

1. **Public-helper `cfg=None` remains caller discipline.** The helpers cannot infer a paper config;
   the exact R1 model is always gated. A paper-config caller must pass `cfg`.
2. **Single-model report identity is the frozen R1 configuration.** The single-mode
   `run_reproduction` config lacks a `models` block; `_r1_authorization_config` loads the R1 config
   for identity. The gate is now invoked, but role-model identity is a separate design issue.
3. **P3 residuals from Phase 46 remain:** non-atomic report writes, decision-hash TOCTOU, ignored
   extra decision fields, the test-only R1 fixture, and same-second run-ID handling.
4. **No live provider behavior was verified.** No paid qualification or experiment is authorized.
5. **Environment-only suite failures** remain; no unrelated repair was attempted.

---

## 10. No-paid-activity confirmation

* No OpenAI, Anthropic, or other model API call.
* No API-key access, validation, printing, or copying.
* No qualification run, model inference, download, smoke test, pilot, attack, or experiment.
* No Deliverable 1 execution.
* No credential or secret in any modified file.
* No frozen artifact modified.

Phase 49 completion does not authorize paid activity. Phase 50, if appropriate, must independently
audit this configuration-forwarding correction and the preserved authorization/review contracts
before any separate human authorization decision.
