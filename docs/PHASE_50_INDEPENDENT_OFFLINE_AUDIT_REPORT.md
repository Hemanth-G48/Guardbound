# PHASE 50 — INDEPENDENT OFFLINE AUDIT OF PHASE 49 CONFIGURATION AND MODEL IDENTITY

**Phase:** 50 (independent read-only audit)
**Controlling implementation:** Phase 49
**Controlling audits:** Phase 48, Phase 46
**Verdict:** `FAIL — BLOCKING AUTHORIZATION OR REVIEW-GATE DEFECTS REMAIN`

The Phase 49 configuration forwarding is correct. The audit nevertheless found that a
qualification report and review decision bound to the frozen R1 configuration can authorize a
**different actual runtime model** through the supported single-model `run_reproduction` path. That
is an authorization identity defect, not a documentation issue, and it blocks authorization.

No paid activity, credential access, live client construction, inference, qualification run, or
Deliverable 1 execution occurred. This report is the only new file created.

---

## 1. Executive summary

Phase 49 fixed the Phase 48 P2-1 forwarding defect: `run_reproduction.make_llm` now passes the
exact `cfg` object to `build_paid_role_llm`, `guard_paid_model`, and `build_paid_chat_llm`. The
Phase 49 tests pass and the forwarding is independently reproduced.

That is not sufficient. In the single-model path the supplied config has no `models` block, so
`_r1_authorization_config` silently substitutes the frozen R1 configuration for identity. The gate
then validates an R1 report and an R1 review decision against the **R1 configuration**, while the
actual runtime model comes from the single-model `attacker`/`target`/`evaluator` blocks. The actual
role model is never compared to the report. With a valid R1 report (and, for a triggered report, a
valid R1 decision):

* `run_reproduction.make_llm` constructs `gpt-4o-mini` or the alias `gpt-4o` under a
  `paper_reproduction: true` configuration;
* the construction reaches the provider/client layer with the substituted model ID;
* the three-model path **does** reject the same non-R1 target because its `cfg` carries `models`
  and the report identity is compared against the actual target.

The answer to the required question is **yes**: an R1 report and decision can authorize a different
actual runtime model through the supported single-model paper-reproduction path. This is **P0-1**.

**P0-1 is blocking.** Under the audit's own PASS criterion — the report and decision must authorize
the actual runtime model configuration, not merely a configuration loaded as a substitute for
missing role identity — Phase 50 cannot pass.

---

## 2. Scope and no-paid-activity confirmation

This phase was read-only and offline. I did not:

* read, validate, print, copy, or expose a credential or API key;
* construct a live provider client;
* make a paid API call;
* run qualification, inference, a smoke test, pilot, attack, or experiment;
* download a model, dataset, or checkpoint;
* execute Deliverable 1;
* modify a source file, test, config, dataset, checkpoint, or historical report.

The identity scenarios used synthetic v2 reports and a synthetic review decision written to an
isolated temporary directory outside the repository, plus `FakeOpenAIBackend` from
`tests/cloud_fakes.py`. The only credential-like value was the literal branch marker
`phase50-dummy-not-a-credential`, set to make `run_reproduction.make_llm` enter the API-style model
branch; the real helper received a fake SDK client, so no key was read by a live client.

The working tree contains substantial pre-existing uncommitted Phase 1–49 work. I did not attribute
that cumulative diff to this audit. Phase 50 created only this report.

---

## 3. Adopted threat model

The Phase 47 trusted in-process code threat model remains in force:

* The boundary is **application-policy enforcement against accidental or unsupported misuse of the
  repository's own code**.
* A trusted caller with an arbitrary hostile Python process is **out of scope**. Python
  `ContextVar`s, underscore-private names, sealed dataclasses, and `inspect.stack()` are not
  security boundaries.
* A process/OS-level credential-and-transport broker would be required to defend against hostile
  same-process code.

Within that model, the qualification/review contract must still bind the **actual configuration
being executed**. That is a correctness requirement, not a demand for cryptographic isolation.

---

## 4. Phase 48 P2-1 configuration-forwarding verification

**Source inspected:** `scripts/run_reproduction.py:275-314` (`make_llm`).

The three call expressions now forward the exact object:

| Line | Call | Forwarded |
|---|---|---|
| 294–295 | `common.build_paid_role_llm(cfg["models"][role], role, args=args, manager=manager, cfg=cfg)` | exact `cfg`, role config, manager |
| 302 | `common.guard_paid_model(model, args, cfg=cfg)` | exact `cfg`, actual `model` |
| 314 | `common.build_paid_chat_llm(model, args=args, cfg=cfg)` | exact `cfg`, actual `model` |

There is no `cfg=None` at these sites. The mock path returns before them; the local-override and
local-HF branches occur after the guard.

**Independent test evidence:**

* `tests/test_phase49_config_propagation.py` — **8 passed**.
* The audit wrapper captured the forwarded config by object identity at the `make_llm` entry path:
  the three-model call and the single-model guard/build calls all received the same `cfg` object.
* A hybrid `paper_reproduction: true` three-model config with a non-R1 target and no report raises
  `QualificationError`.
* The same non-R1 construction under a non-paper config proceeds.
* The exact `gpt-4o-2024-08-06` remains blocked with `cfg=None` and through `make_llm`.
* `run_reproduction.validate_config({})` still raises `SystemExit`.

**Conclusion:** the Phase 48 P2-1 forwarding defect is closed. The forwarding is not the identity
binding required by the Phase 50 PASS criterion. None of the Phase 49 tests supplies a valid R1
report plus a non-R1 runtime model; they cover missing authorization, not identity binding. The
passing suite is therefore not evidence that the report authorizes the runtime model.

---

## 5. Runtime model versus qualification identity — full trace

### 5.1 The pieces

| Piece | Location | Behavior |
|---|---|---|
| Runtime role model | `run_reproduction.py:297-298` | `override or cfg[role]["model"]`; passed to the guard and builder |
| R1 detection | `_cloud_common.py:722-729` → `paid_authorization.py:98-108` | Exact `gpt-4o-2024-08-06` **or any model when `cfg.paper_reproduction` is true** |
| Identity config | `_cloud_common.py:732-740` | If `cfg is None` **or `cfg["models"]` is not a dict**, load `configs/paper_reproduction_r1_gpt4o.yaml` |
| Expected identity | `_cloud_common.py:238-260` | Models from `cfg["models"]`; the single-model config has none |
| Report validation | `_cloud_common.py:325-557` | Compares the report's `configuration_under_qualification` to the expected identity |
| Paid authorization | `_cloud_common.py:743-770` | `is_r1_paid_model` → `require_qualification` → `require_review_decision` |
| Direct builder | `_cloud_common.py:789-815` | Gate, then `_authorized(model_id)` and `OpenAIChatLLM(model_id=...)` |
| Role builder | `_cloud_common.py:818-842` | Gate, then `_authorized(model_id)` and `build_role_llm(role_cfg, ...)` |
| Single-role builder | `_cloud_common.py:1124-1148` | Gate, then `_authorized(model_id)` and `_build_cloud_role(...)` |
| Construction guard | `openai_client.py:302-330` | `require_construction_authorization(model_id)`; exact-R1 or `cfg`-paper detection |
| Three-role stack | `_cloud_common.py:941-966` | Uses `cfg["models"]`; the identity and the actual models coincide |

### 5.2 Which model is used where

1. **Locating/validating the report.** The report path is supplied by `--qualification-report` or
   resolved from the identity config. In the failing path the identity config is the loaded R1
   config, so the report is an R1 report. The actual runtime role model is not used to locate or
   validate it.
2. **Binding the review decision.** The decision binds `qualification_report_sha256`, `run_id`,
   `configuration_sha256`, `arm`, `trigger`, `decision`, `rationale`, `decided_by`, `decided_at`,
   and `actor_attack_eligible` (`_cloud_common.py:560-700`). It binds the **R1 report**, not the
   runtime role model.
3. **Authorizing construction.** `is_r1_paid_model(actual_model, cfg)` is true because
   `cfg.paper_reproduction` is true. `_r1_authorization_config(cfg)` loads the R1 config because
   the single config has no `models`. The R1 report and decision pass. `_authorized(actual_model)`
   opens the scope for the actual non-R1 model ID.
4. **Passing the model to the client.** `OpenAIChatLLM(model_id=actual_model, ...)` receives the
   actual non-R1 ID. Its constructor guard (`openai_client.py:330`) checks `is_r1_model(actual_model)`
   without `cfg`; non-R1 is not an R1 construction, so the constructor does not require R1
   authorization. The context exists, but it is for a model the R1 report never qualified.

### 5.3 The three-model contrast

With a `models` block, `_r1_authorization_config` keeps the supplied config. The expected identity
contains the actual non-R1 target. An R1 report's `configuration_under_qualification.models` is the
R1 triple, so `require_qualification` raises a stale-identity error **before** the factory. This is
the behavior the single-model path lacks.

---

## 6. Scenarios A–E

All scenarios used synthetic v2 reports with the R1 identity (models R1, temperatures 0.7,
`prompt_variant: official`, `structured_output: json_object`, experiment hash equal to
`paper_reproduction_r1_gpt4o.yaml`), plus `FakeOpenAIBackend`. The triggered report had
`observed=0.1`, `triggered=true`; the decision was bound to that exact report.

| Scenario | Actual runtime role | Report identity | Decision | Gate outcome | Factory reached | Construction |
|---|---|---|---|---|---|---|
| **A** exact R1 | `gpt-4o-2024-08-06` | R1 | none | pass | yes | R1 client |
| **B** non-R1 target, non-paper | `gpt-4o-mini` | R1 report supplied but not consulted | none | unguarded (non-paper) | yes | `gpt-4o-mini` |
| **C** attacker mismatch, non-paper | attacker `gpt-4o-mini`; target R1 | R1 report supplied, not consulted | none | unguarded (non-paper) | yes | attacker `gpt-4o-mini` |
| **D** alias, non-paper | `gpt-4o` | R1 report supplied, not consulted | none | unguarded (non-paper) | yes | `gpt-4o` |
| **E** hybrid paper, non-R1 target | `gpt-4o-mini` | **R1** | none | **pass against R1** | yes | **`gpt-4o-mini` — DEFECT** |
| **E-override target** | `gpt-4o-mini` via `args.target`; cfg target R1 | **R1** | none | **pass against R1** | yes | **`gpt-4o-mini` — DEFECT** |
| **E-override attacker** | `gpt-4o-mini` via `args.attacker`; cfg attacker R1 | **R1** | none | **pass against R1** | yes | **`gpt-4o-mini` — DEFECT** |
| **E-alias** hybrid paper, alias | `gpt-4o` | **R1** | none | **pass against R1** | yes | **`gpt-4o` — DEFECT** |
| **E-trigger, no decision** | `gpt-4o-mini` | R1, triggered | absent | blocked | no | blocked: decision required |
| **E-trigger, R1 decision** | `gpt-4o-mini` | R1, triggered | **R1 decision** | pass | yes | **`gpt-4o-mini` — DEFECT** |
| **Three-model non-R1 target** | target `gpt-4o-mini`; attacker/evaluator R1 | R1 | none | **stale identity** | no | blocked |

**Interpretation:**

* B, C, and D are legitimate **non-paper** runs. The supplied R1 report is not used; they do not
  demonstrate R1 authorization. They do, however, confirm that the helper cannot distinguish a
  paper workflow from a non-paper workflow when no paper config is supplied.
* E is the critical case. The config explicitly says `paper_reproduction: true`; the target is
  different; the R1 report passes; the actual target is constructed. The report identity is a
  substitute, not the runtime configuration. The E-override variants are the exact form in the
  phase brief: the config still carries R1 `target`/`attacker` IDs, and the actual runtime model is
  the `args.target`/`args.attacker` override; the R1 report still passes and the override is
  constructed.
* E-alias shows that the exact-model alias policy in `is_r1_model` does not protect a paper
  configuration. `gpt-4o` is not treated as R1 at the model layer, but the `paper_reproduction`
  flag makes `is_r1_paid_model` true, and the R1 report authorizes it.
* E-trigger with the R1 decision shows the review decision is correctly bound to the R1 report but
  is nevertheless reusable for a different actual runtime model. The decision schema has no
  runtime-model field.
* The three-model non-R1 target is rejected, proving the identity check can work when the
  configuration exposes the actual role models. The single-model path does not.

### Answer to the required question

**Yes.** A qualification report and review decision bound to the frozen R1 configuration can
authorize a different actual runtime model through the supported single-model
`run_reproduction` paper path (and through the public helpers when a paper config lacks a `models`
block). The actual runtime model is `gpt-4o-mini` (or `gpt-4o`); the report and decision are R1.
This is an authorization identity defect.

**Classification: P0-1, blocking.** It is not a mere configuration-forwarding documentation issue,
and it is not fixed by forwarding `cfg`. The identity must bind the actual role models.

---

## 7. Public-helper configuration contract

**Inspected:** `build_paid_chat_llm` (`_cloud_common.py:789-815`), `build_paid_role_llm`
(`818-842`), `build_single_role_llm` (`1124-1148`), `guard_paid_model` (`773-786`),
`build_role_llms` (`941-966`), `require_paid_authorization` (`743-770`), and the provider
factories.

Findings:

1. **`cfg=None` cannot be mechanically identified as paper-reproduction.** The helper has no
   workflow context. The exact R1 model is independently gated at the client constructor. A
   non-R1 model with `cfg=None` is treated as non-paper and is not gated. That is the documented
   caller contract, not a new discovery.
2. **Many supported scripts call the helpers without `cfg`.** Examples: `run_attacks.py:215,242,253`,
   `eval_mmlu.py:53`, `eval_mtbench.py:55`, `eval_over_refusal.py:51`, `judge_asr.py:50`,
   `run_baseline_evals.py:82,224`, `run_full_evaluation.py:100,117`, `final_analysis.py:61`.
   These are not paper-config-aware; the exact R1 model is still protected. Whether a non-R1 model
   is a legitimate non-paper run or an unauthorized paper substitution is left to the caller.
3. **`build_paid_role_llm`, `build_single_role_llm`, and `build_paid_chat_llm` share the identity
   defect when given a `paper_reproduction: true` config without `models`:** the R1 config is
   loaded as the identity, the R1 report passes, and the actual non-R1 role is constructed inside
   `_authorized(actual_model)`.
4. **`build_role_llms` is different.** It requires `cfg["models"]` and uses those actual models for
   the identity; the three-model non-R1 target is rejected as stale. `cloud_pilot` uses this path.

The configuration contract remains partly trusted-caller discipline. That is a P2 residual, but the
P0 is the **paper-config single-model identity substitution**, which bypasses the contract even when
the caller passes a config.

---

## 8. Phase 47 authorization lifecycle verification

`tests/test_phase47_authorization_lifecycle.py` — **22 passed**.

Confirmed preserved:

* explicit sealed `QualificationCapability`; no `inspect.stack()` filename authorization;
* actual `OfflineMockClient` identity for offline mode; `client=None` rejected;
* no offline-to-live fallthrough;
* live confirmation before key/client access;
* scope owner, release, nested restoration, exception cleanup;
* async child-task and copied-context blocking;
* model isolation and failed-builder cleanup;
* exact R1 constructor guard; alias policy.

The P0 is not a Phase 47 lifecycle regression. The capability correctly grants the model it is told
to grant; the defect is that the gate decided the R1 report authorized the **actual non-R1 model**
in the first place.

---

## 9. Phase 44 review-gate verification

`tests/test_phase45_authorization.py` — **27 passed**; Phase 43/41 — **105 passed**; qualification
and gate suites — **234 passed**.

Confirmed preserved for the R1 report identity:

* `QUALIFIED` reports require finite numeric `observed` and boolean `triggered`;
* null/missing values reject;
* threshold is exactly `0.50`; comparison is strictly `<`;
* 0.49 triggers and 0.50 does not;
* invalid thresholds reject;
* the ten-field decision binding is enforced; stale, mismatched, late, and unused decisions reject;
* the decision binds the report SHA-256, run ID, configuration SHA-256, arm, and trigger.

**However:** the decision has no field for the actual runtime role models, and the report it binds
is the R1 identity substituted for the single-model config. The E-trigger scenario constructs a
non-R1 target with a correctly bound **R1** decision. The review gate is internally correct; the
authorization identity is wrong. This is part of P0-1.

---

## 10. Phase 41/43 regression preservation

Phase 41 + Phase 43 — **105 passed**. The following were exercised and remain intact:

* `qualification.v2` / `r1-probe.v2`;
* UTC run IDs and canonical immutable filenames;
* newest-report resolver; no overwrite; no fallback to an invalid report;
* legacy v1 readable but ineligible;
* delivery metrics from completed rows only;
* kept-actor validation;
* raw `questions` validation;
* `diagnostics=None` parity;
* review-trigger calculation and decision enforcement;
* v2 schema and probe-spec checks.

No regression was found in the probe, schema, or review-gate implementation.

---

## 11. Exact test commands and results

Environment: Windows, Python 3.13, `torch 2.14.1+cpu`,
`torch.cuda.is_available() == False`, `KMP_DUPLICATE_LIB_OK=TRUE`.

```
python -m pytest tests/test_phase49_config_propagation.py -q
```
**8 passed in 0.89s.** Exit 0.

```
python -m pytest tests/test_phase47_authorization_lifecycle.py -q
```
**22 passed in 0.95s.** Exit 0.

```
python -m pytest tests/test_phase45_authorization.py -q
```
**27 passed in 1.04s.** Exit 0.

```
python -m pytest tests/test_phase43_corrective.py tests/test_phase41_probe_redesign.py -q
```
**105 passed in 0.63s.** Exit 0.

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
**1498 passed, 4 failed, 0 skipped in 164.25s.** Exit 1.

The four failures are current environment checks:

* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring`
* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured`
* `test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid`
* `test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present`

Immediate causes verified: CPU-only torch / no CUDA, and `Qwen/Qwen3.5-4B` absent from the local HF
cache. I did not verify that these failures historically predate Phase 49; I do not claim that.

The identity scenarios were run with
`scratchpad/audit_phase50_identity.py`; the output is reproduced in section 6. No real credential or
provider client was used.

---

## 12. Frozen-artifact hashes

| Artifact | SHA-256 | Status |
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
| Phase 49 report | `91a13d7518abf8244b3e8269f338268609f29bc3e9c4c1bf75af87d7cd49c214` | record |

No frozen artifact was modified or regenerated.

---

## 13. Findings and residual risks

### P0-1 — R1 report/decision authorizes a substituted runtime model (blocking)

**Evidence:** scenarios E, E-override target, E-override attacker, E-alias, and E-trigger-with-decision. The actual target is
`gpt-4o-mini`/`gpt-4o`; the report identity and review decision are the frozen R1 configuration.
`guard_paid_model`, `build_paid_chat_llm`, and `OpenAIChatLLM` all accept the substituted model.
The three-model path rejects the same substitution.

**Root cause:** `_r1_authorization_config` substitutes the frozen R1 config when the supplied paper
config has no `models` block (`_cloud_common.py:732-740`); `expected_qualification_identity` uses
that substituted config; the actual single-model role IDs are never compared
(`_cloud_common.py:238-260`, `494-542`). The review decision has no runtime-model field.

**Impact:** a valid R1 qualification and review can authorize a different actual model under a
supported paper-reproduction path. The exact R1 model is still gated, but the substituted model is
not the exact R1 model, and the report does not describe it. This violates the requirement that the
report/decision authorize the actual configuration being executed.

**Correction direction (not implemented):** bind the actual role models into the expected identity
and the review decision, and stop silently substituting the R1 config for a supplied paper config
that lacks `models`. At minimum, a paper config must expose the actual role models and the gate
must compare them; the single-model config must be normalized or rejected rather than treated as
the R1 config. The alias policy must also be considered: a paper config must not turn `gpt-4o` into
an R1-authorized substitution.

### P2-1 — `cfg=None` remains caller discipline

The public helpers cannot infer a paper workflow. The exact R1 snapshot is independently gated.
Many scripts call the helpers without `cfg`; a non-R1 model there is a non-paper run and is not
gated. This is the documented contract, but it is a residual risk for any script intended as
paper reproduction without a config.

### P2-2 — Legacy scripts are not paper-config-aware

The scripts listed in section 7 do not pass a paper configuration. They are safe for the exact R1
snapshot; a non-R1 model is treated as non-paper. This is a caller-contract residual, related to but
distinct from the P0 single-model identity substitution.

### P3 residuals (not fixed)

* Non-atomic qualification-report writes.
* Decision-hash time-of-check/time-of-use exposure.
* Extra review-decision fields are ignored.
* The legacy test fixture grants R1 authorization to selected tests (`tests/conftest.py`).
* Same-second run-ID collision behavior.
* The public-helper `cfg=None` contract.
* The single-model identity substitution is additionally recorded as P0-1, not merely P3.

---

## 14. Authorization status

**No authorization is granted.**

* Phase 49 is not accepted as ready for paid authorization while P0-1 remains.
* No qualification run, paid model call, or Deliverable 1 execution is authorized.
* This FAIL is an audit verdict, not a request to implement a correction in this phase.
* Any correction must be a separate, explicitly reviewed phase; this audit did not implement one.
* After a correction, a new independent audit must verify that the report and review decision bind
  the **actual runtime role models**, including the single-model paper path and the alias case.
* Any future paid qualification still requires a separate explicit human decision.

**Final verdict: `FAIL — BLOCKING AUTHORIZATION OR REVIEW-GATE DEFECTS REMAIN`.**
