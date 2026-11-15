# PHASE 52 — INDEPENDENT OFFLINE AUDIT OF PHASE 51 RUNTIME MODEL IDENTITY BINDING

**Phase:** 52 (independent read-only audit)
**Controlling implementation:** Phase 51
**Controlling audit:** Phase 50
**Verdict:** `FAIL — BLOCKING IDENTITY DEFECT REMAINS`

The Phase 50 P0-1 — an R1 report/decision authorizing a different effective role model — was **not
reproduced** on the gated synthetic single-model shape or the three-model path. The Phase 51 tests
for those shapes pass, and the three-model R1 path constructs the exact R1 model.

However, the audit found a **blocking identity defect in the supported single-model configuration
schema**: the actual `run_reproduction` single-model config uses `evaluator.rubric_model` (plus
refusal/disclaimer/actor models), not `evaluator.model`. `_effective_role_models` requires
`evaluator.model`, so the exact-R1 single-model Scenario A fails closed as
`missing model IDs for ['evaluator']` before any report is validated or factory is reached. The
Phase 51 test for Scenario A uses a synthetic `evaluator.model` shape and does not exercise the
actual single-model schema.

This is a fail-closed identity defect, not an R1-report-to-non-R1 authorization bypass. It is still a
blocking identity finding because Scenario A (the exact R1 single-model path) is a required
acceptance scenario, and the Phase 51 “single-model normalization” does not describe the actual
single-model configuration.

No paid activity, credential access, live client, inference, qualification, or Deliverable 1
execution occurred.

---

## 1. Executive verdict

| Question | Result |
|---|---|
| Can an R1 report authorize a non-R1 target through the tested synthetic single-model shape? | **No** — rejected as stale |
| Can an R1 report/decision authorize a non-R1 target through the three-model path? | **No** — rejected as stale |
| Does the actual single-model schema support the exact-R1 Scenario A? | **No** — `missing evaluator` before report validation |
| Is the final factory model exactly the effective identity for the synthetic/three-model positives? | **Yes** — R1 |
| Is there a supported standard-path post-authorization model substitution? | No; a caller-supplied mapping can, classified P3 under the trusted caller model |
| Is the source hash bound for all supported configs? | No for hand-built configs without `_source_sha256` (P2) |
| Verdict | **FAIL** |

**The blocking finding is P1-1: the single-model effective evaluator identity is not derived from
the actual single-model evaluator configuration.** The Phase 51 test suite does not catch it because
its test config uses `evaluator.model`, a field the actual single-model config does not have and
`run_reproduction.validate_config` does not require.

---

## 2. Scope and no-paid-activity confirmation

Read-only and offline. I did not modify source, tests, configs, datasets, checkpoints, historical
reports, or implementation files. I created only this report; the independent probes were written
and run in an isolated temporary directory.

* No real API key was read, printed, validated, copied, or exposed.
* The probe used the literal `phase52-dummy-not-a-credential` only to enter the API-style branch.
* All construction probes used `FakeOpenAIBackend` or a factory spy; no live provider client.
* No paid call, qualification, inference, smoke test, pilot, attack, experiment, download, or
  Deliverable 1 execution.

---

## 3. Threat model

The Phase 47 trusted in-process application-policy model remains in force. The boundary covers
trusted repository code and accidental or unsupported misuse; arbitrary hostile Python in the same
process is out of scope. `ContextVar`, underscore-private names, sealed dataclasses, and
`inspect.stack()` are not described as security boundaries.

Within that model, the qualification identity must describe the actual runtime role models. That is
the property audited here.

---

## 4. Files and source locations inspected

| Area | Location |
|---|---|
| Effective role normalization | `scripts/_cloud_common.py:238-273` (`_effective_role_models`) |
| Expected qualification identity | `scripts/_cloud_common.py:276-304` (`expected_qualification_identity`) |
| Effective runtime config | `scripts/_cloud_common.py:769-825` (`_effective_runtime_config`) |
| R1 authorization | `scripts/_cloud_common.py:827-870` (`_r1_authorization_config`, `require_paid_authorization`) |
| Paid helpers | `scripts/_cloud_common.py:875-921` (`guard_paid_model`, `build_paid_chat_llm`); `931-947` (`build_paid_role_llm`); `1022-1070` (`build_role_llms`); `1240-1260` (`build_single_role_llm`) |
| Cloud defaults | `scripts/_cloud_common.py:1131-1152` (`apply_cloud_defaults`) |
| Config load and source hash | `scripts/run_reproduction.py:183-190` (`load_config`) |
| Runtime single-model selection | `scripts/run_reproduction.py:280-316` (`make_llm`) |
| Actual single-model evaluator schema | `configs/reproduction.yaml:92-98` (`rubric_model`, `refusal_model`, `disclaimer_model`, `actor_step_judge_model`; no `model`) |
| Required/config validation | `scripts/run_reproduction.py:691`, `725-754` (`evaluator.rubric_model` required; flat map to `models.evaluator.model`) |
| Single manifest metadata | `scripts/run_reproduction.py:604` (`evaluator_model` from `rubric_model`) |
| Execution paths | `scripts/run_reproduction.py:976-988` (three-model evaluator built; single evaluator is `None`) |
| Provider construction | `src/guardbound/llm/provider_factory.py:132-217` (`build_cloud_role_llm`), `220-369` (`build_role_llm`), `369-403` (`build_three_model_stack`) |
| Client constructor | `src/guardbound/llm/openai_client.py:302-330` |
| Authorization lifecycle | `src/guardbound/llm/paid_authorization.py:98-161` |
| Phase 51 tests | `tests/test_phase51_runtime_identity.py` (config helper and Scenarios A–L) |
| Phase 49 tests | `tests/test_phase49_config_propagation.py` |
| Phase 47/45 tests | `tests/test_phase47_authorization_lifecycle.py`, `tests/test_phase45_authorization.py` |
| Test fixture exclusion | `tests/conftest.py:26-44` |

---

## 5. Independent effective-identity and construction trace

### 5.1 Actual single-model path

1. Supplied config: the supported single-model shape, e.g. `configs/reproduction.yaml`, has
   `attacker.model`, `target.model`, and an evaluator block containing `rubric_model`,
   `refusal_model`, `disclaimer_model`, `actor_step_judge_model`; **there is no `evaluator.model`**.
2. Source hash: `run_reproduction.load_config` adds `_source_sha256` over the file bytes.
3. Effective role models: `_effective_role_models` reads the top-level `attacker`, `target`, and
   `evaluator` mappings and requires `evaluator["model"]`. The actual config provides only
   `evaluator["rubric_model"]` and the other evaluator subroles, so the function raises
   `PaidAuthorizationError: cannot establish the effective runtime role models: missing model IDs
   for ['evaluator']`.
4. CLI overrides: `make_llm` computes `override or cfg[role]["model"]`; the evaluator block has no
   `model`, so even the direct evaluator path would raise `KeyError` rather than select a model.
5. Authorization: the guard raises before `require_qualification`; no report is read by the gate.
6. Factory: not reached. The independent probe recorded `factory_calls=[]`.

This is why the exact-R1 single-model Scenario A cannot pass with the actual supported schema. The
Phase 51 test helper writes `evaluator: {model: ...}`; that is a different schema and would also
fail `run_reproduction.validate_config` because it omits the required `evaluator.rubric_model` and
`evaluator.refusal_model`.

### 5.2 Synthetic single-model shape used by Phase 51

With `evaluator.model` added, `_effective_role_models` yields the R1 triple; the synthetic report
matches; `make_llm` passes the report; `build_paid_chat_llm` receives `role="target"`, `model=R1`;
the factory receives R1. The audit reproduced this (`A_SYNTHETIC_SINGLE`), but it is not the actual
single-model schema.

### 5.3 Three-model path

`run_reproduction.make_llm` routes a `models` config through `build_paid_role_llm`; the effective
models come from `cfg["models"]`; the role config and the report must agree. The audit reproduced:

* `A_THREE_MODEL`: R1 config, R1 report, target factory call model `gpt-4o-2024-08-06`.
* The Phase 51 `I` test: non-R1 target + R1 report is stale before `build_role_llm`.

### 5.4 Non-paper path

The actual non-paper `reproduction.yaml` has local non-R1 models and no `paper_reproduction` flag;
`is_r1_paid_model` returns false, the gate is not called, and the non-paper construction proceeds.
This is the documented non-paper contract.

---

## 6. Source/configuration hash analysis

Three distinct identities are visible:

1. **Source-file identity:** `run_reproduction.load_config` computes `_source_sha256` over the
   original file bytes; `common.load_config` does the same for cloud configs. The report stores this
   as `experiment_config.sha256`, and `require_qualification` compares it with the expected source
   hash and, when the report's path exists, with the file's actual hash.
2. **Experiment-configuration identity:** the report's `configuration_under_qualification` (models,
   prompt variant, temperatures, structured output) and the manifest's `config_hash`. The latter is
   a hash of the cfg including `_source_sha256`, so it is **not** the raw file hash; it is a
   normalized manifest hash. It is not the gate's source hash.
3. **Effective runtime model identity:** `_effective_role_models` / `_effective_runtime_config` and
   the actual role model passed to the factory.

Findings:

* **N1 — mismatched source hash:** a valid R1 report with a zeroed `experiment_config.sha256`
  against a config with a real `_source_sha256` is rejected (`stale ... experiment_config.sha256`).
* **N2 — matching source hash, altered roles:** a config with a non-R1 target and the report's
  source hash is rejected on the models. The source hash alone is not sufficient.
* **N3 — hand-built config without `_source_sha256`:** the expected hash is `None`; a report whose
  `experiment_config.path` exists and whose hash matches that path passes with the R1 role triple.
  The **source identity is not bound** for a hand-built config. This is a **P2** caller-contract
  finding; it does not substitute the effective model identity, and a non-R1 effective role still
  fails the models check.
* **N4 — CLI override:** the effective role changes but the file hash does not; the R1 report's
  models no longer match and the gate rejects. The source hash is not used as a substitute for the
  model check.
* **Claimed hash:** a hand-built config can set `_source_sha256` to a report's hash; if the roles and
  other identity fields match, the source claim is accepted. The config contents are not re-hashed
  by the gate. This is the same P2 caller-contract issue.
* **Manifest hash:** adding `_source_sha256` means the manifest `config_hash` now includes the file
  hash field. It is a normalized manifest hash, not the report's source hash; no gate path uses the
  manifest hash. Record as P3.

The source hash is **not** a substitute for the model triple. The A actual failure is not a hash
issue; it is the evaluator schema mapping.

---

## 7. Review-decision binding analysis

The decision validation order is correct: `require_paid_authorization` first calls
`require_qualification` (which includes the effective model comparison), then
`require_review_decision`. A model mismatch raises before a valid decision is consulted.

* The Phase 44 10-field binding remains: report SHA-256, run ID, configuration SHA-256, arm,
  trigger, decision, rationale, decider, timestamp, and `actor_attack_eligible`.
* Threshold is exactly `0.50`, comparison is strictly `<`, and null/missing observed/trigger data
  fails closed.
* The synthetic `H` test (Phase 51) validates an R1-triggered report and correctly bound R1
  decision, then rejects the non-R1 effective target as stale before the decision can authorize it.
* The actual single-model evaluator failure occurs before the decision layer; the decision is never
  consulted.

No decision-reuse defect was found for the synthetic/three-model paths. The single-model evaluator
schema failure is upstream of the decision.

---

## 8. Scenarios A–N

| Scenario | Config / effective roles | Report | Decision | Gate result | Factory reached | Model if constructed |
|---|---|---|---|---|---|---|
| **A actual single** | actual `reproduction.yaml` schema; attacker/target R1; evaluator `rubric_model` R1; no `evaluator.model` | R1 | none | `missing evaluator` | no | — |
| **A synthetic single** | Phase 51 test shape with `evaluator.model` R1 | R1 | none | pass | yes | R1 |
| **A three-model** | R1 `models` block | R1 | none | pass | yes | R1 |
| **B target** | actual single; target `gpt-4o-mini`; evaluator rubric R1 | R1 | none | `missing evaluator` (not the intended target mismatch) | no | — |
| **C attacker** | actual single; attacker `gpt-4o-mini` | R1 | none | `missing evaluator` | no | — |
| **D evaluator** | actual single; evaluator rubric `gpt-4o-mini` | R1 | none | `missing evaluator` | no | — |
| **E target override** | actual single R1; `args.target=gpt-4o-mini` | R1 | none | `missing evaluator` | no | — |
| **F attacker override** | actual single R1; `args.attacker=gpt-4o-mini` | R1 | none | `missing evaluator` | no | — |
| **G alias** | actual single; effective `gpt-4o` | R1 | none | `missing evaluator` | no | — |
| **H R1 decision** | synthetic R1 triggered + valid R1 decision; non-R1 effective target | R1 | valid R1 | stale target (synthetic shape) | no | — |
| **I three-model** | `models` target `gpt-4o-mini` | R1 | none | stale target | no | — |
| **J incomplete** | actual single missing evaluator; paper config also missing target | — | — | missing model IDs / non-empty model guard | no | — |
| **K non-paper** | actual non-paper local config | not used | none | unguarded | yes | local non-R1 |
| **L helpers** | synthetic single/three-model paper configs | R1 | none | stale target on all helpers | no | — |
| **L actual single** | actual single schema | R1 | none | `missing evaluator` | no | — |
| **M mapping flip** | paper `models`; caller `dict` subclass returns R1 then `gpt-4o-mini` | R1 | none | gate R1; construction sees NON_R1 | yes | `gpt-4o-mini` |
| **N1 source mismatch** | R1 roles; source hash mismatch | R1 | none | stale source hash | no | — |
| **N2 altered roles** | non-R1 target; matching source hash | R1 | none | stale models | no | — |
| **N3 no source hash** | hand-built R1 config without `_source_sha256` | R1 | none | pass; source unbound | no construction in probe | — |
| **N4 CLI override** | actual single evaluator; synthetic target override | R1 | none | actual: missing evaluator; synthetic: stale | no | — |

The Phase 51 A–L tests (synthetic shape) pass with the expected outcomes. The **actual single-model
schema** changes the result for A–H and L: they fail at the evaluator identity before the intended
role comparison.

### M — post-authorization model change

The only successful reproduction used a caller-supplied `dict` subclass whose `get("model")`
returns R1 for the first call (the gate) and `gpt-4o-mini` for later calls (the provider factory).
The gate authorized R1; the factory constructed `gpt-4o-mini`. A plain `dict` cannot do this
synchronously, and concurrent mutation by another thread is hostile same-process code outside the
Phase 47 threat model. Classify as **P3**: `build_paid_role_llm`/`build_single_role_llm` pass the
original `role_cfg` to the factory rather than the effective role copy; a defensive copy would remove
the discrepancy. It is not the primary blocker.

---

## 9. Factory/client reachability

* A actual, B–H actual, I, J, L actual: **no** factory call (`factory_calls=[]` or spy empty).
* A synthetic, A three-model, K: **yes**; models R1, R1, local non-R1 respectively.
* M: **yes**; model `gpt-4o-mini` despite the R1 gate.
* N1/N2/N4: **no**.
* All rejection paths checked with fake clients or spies; no live client was constructed.

---

## 10. Phase 47 lifecycle results

`tests/test_phase47_authorization_lifecycle.py` — **22 passed**.

Confirmed intact:

* sealed `QualificationCapability`; no `inspect.stack()` filename authorization;
* actual `OfflineMockClient` identity; `client=None` rejected for offline qualification;
* no offline-to-live fallthrough;
* live confirmation before key/client access;
* scope owner/released checks, nested restoration, exception cleanup;
* async child-task and copied-context blocking;
* model isolation and failed-builder cleanup;
* exact R1 constructor guard and alias policy.

The Phase 51 module is excluded from the legacy autouse R1 fixture, so the synthetic A path tests
the real helper context. The single-model evaluator schema failure is not a Phase 47 lifecycle
regression.

---

## 11. Phase 44 review-gate results

`tests/test_phase45_authorization.py` — **27 passed**; Phase 43/41 — **105 passed**;
qualification/gate — **234 passed**.

Confirmed:

* finite numeric observed; boolean triggered; null/missing reject;
* exact 0.50 threshold and strict `<`; 0.49 triggers, 0.50 does not;
* ten-field decision binding; stale, mismatched, late, and unused decisions reject;
* the decision cannot be used until the report identity matches;
* the H synthetic valid R1 decision still cannot authorize a non-R1 target.

The actual single-model evaluator failure occurs before the review-decision layer.

---

## 12. Phase 41/43 regression results

Phase 41 + Phase 43 — **105 passed**. The v2/probe-spec, UTC run ID, canonical immutable filename,
newest-report resolver, v1 ineligibility, completed-row delivery metrics, kept-actor validation,
raw `questions` validation, `diagnostics=None` parity, trigger calculation, and decision enforcement
remain intact.

---

## 13. Exact test commands and results

Environment: Windows, Python 3.13, `torch 2.14.1+cpu`, no CUDA,
`KMP_DUPLICATE_LIB_OK=TRUE`.

```
python -m pytest tests/test_phase51_runtime_identity.py -q
```
**21 passed in 1.27s.** Exit 0.

```
python -m pytest tests/test_phase49_config_propagation.py -q
```
**8 passed in 0.87s.** Exit 0.

```
python -m pytest tests/test_phase47_authorization_lifecycle.py -q
```
**22 passed in 0.95s.** Exit 0.

```
python -m pytest tests/test_phase45_authorization.py -q
```
**27 passed in 1.11s.** Exit 0.

```
python -m pytest tests/test_phase43_corrective.py tests/test_phase41_probe_redesign.py -q
```
**105 passed in 0.81s.** Exit 0.

```
python -m pytest tests/test_cloud_qualification_harness.py \
  tests/test_phase24_p0_fixes.py tests/test_phase26_qualification_gate.py \
  tests/test_phase27_fail_closed_gate.py tests/test_phase36_implementation.py \
  tests/test_phase38_hardening.py tests/test_attacker_arms.py -q
```
**234 passed in 7.59s.** Exit 0.

```
python -m pytest tests/ -q
```
**1519 passed, 4 failed, 0 skipped in 160.36s.** Exit 1.

The four failures are current CPU-only/missing-cache environment failures:

* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring`
* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured`
* `test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid`
* `test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present`

Their historical pre-existence is not established; I do not claim it. No Phase 52 test failed, but
the full suite is not called successful.

Independent probes: `audit_phase52.py` (isolated temporary directory). Results include
`A_ACTUAL_SINGLE: BLOCKED ... missing model IDs for ['evaluator']`, `A_SYNTHETIC_SINGLE:
CONSTRUCTED gpt-4o-2024-08-06`, `A_THREE_MODEL: ... gpt-4o-2024-08-06`, `N1/N2: BLOCKED stale`,
`N3: PASS` (source unbound), and `M_FLIP: CONSTRUCTED gpt-4o-mini`.

---

## 14. Frozen-artifact hash verification

| Artifact | SHA-256 | Status |
|---|---|---|
| R1 config | `2125cc3b233efde9e97a8407c364c2140091245dd39bcf440c4366fd99ef393b` | match |
| HarmBench | `ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb` | match |
| NBF checkpoint | `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` | match |
| Historical v1 report | `b17d392a39eeeda64d561017073addd88376d3dafa394bd732c1ba2e908aeaeb` | match |
| Phase 40B design | `9152f89e5a951f00f6a01efe1cef681875f4c4769864474b912b646cc57f9ad3` | match |
| Phase 41 report | `c42a71983ad27472e9dd1a6b35571db10bb8c979343825f05f762469baa9d8ff` | match |
| Phase 42 report | `369617b8780b424c8a8d4aed3474f527b0908a3def4ca3268fe75fde33249e97` | match |
| Phase 43 report | `8857e515ce638940140c7c8c2cd5226cae0941b1c78b44d6733b608597b1c9e3` | match |
| Phase 44 report | `da9ccdf379bb7eb172da302d8700e69493aed7ee6440253a1ad0955b2ed1e1f7` | match |
| Phase 45 report | `685bb7770994ee8ed7730f7f6c425e344341153d5e17a44944116d7452b9bb81` | match |
| Phase 46 report | `8ddbdf683974cfac72f671c4eb11a45d947a8d756d7fbb05c90e65d4658fe6ee` | match |
| Phase 47 report | `1a332bd1df8881a775e09067b50ef6da58b1ff3abf0fee0347e1233c0cebd44d` | match |
| Phase 48 report | `6606c996aee210781c69fe9114c4a29c0b79379d9d67e48c989b8d871e004e9e` | match |
| Phase 49 report | `91a13d7518abf8244b3e8269f338268609f29bc3e9c4c1bf75af87d7cd49c214` | match |
| Phase 50 report | `a6f992b1751996db69c3f0494dd924b98ef12342838ef7566dd2ed633b34b7a7` | match |
| Phase 51 report | `478ca1a85e891b4c6159ad9fc16e5aa438f0c11832eb840af5c76c0f2ff688ea` | record |

No frozen artifact was modified or regenerated.

---

## 15. Findings

### P1-1 — Single-model effective evaluator identity does not match the actual schema (blocking)

**Source:** `configs/reproduction.yaml:92-98` defines `evaluator.rubric_model`,
`refusal_model`, `disclaimer_model`, and `actor_step_judge_model`, with no `evaluator.model`;
`run_reproduction.validate_config:691,725-754` requires the rubric/refusal fields and maps the
rubric to `models.evaluator.model`; `_effective_role_models:238-273` requires `evaluator.model`;
`run_reproduction.make_llm:304` also reads `cfg[role]["model"]`; `run_reproduction.main_async:986-988`
does not construct a single-model evaluator; `run_reproduction.py:604` records the rubric as
`evaluator_model`.

**Impact:** the exact-R1 single-model Scenario A fails closed with
`missing model IDs for ['evaluator']`. Scenarios B–H and the actual-schema L likewise fail at the
evaluator before reaching the intended target/attacker override or report check. The Phase 51 A
test uses a synthetic `evaluator.model` config and would fail `run_reproduction.validate_config`,
so it is not evidence for the actual single-model path.

**Classification:** P1 blocking identity finding. It is fail-closed, not an authorization bypass,
but it blocks the required A and the supported single-model path.

**Correction direction (not implemented):** derive the effective evaluator identity from the actual
single-model evaluator role — `evaluator.model` when present, otherwise the configured
`rubric_model` (and the other actual evaluator subroles where applicable) — and use the same value
in `run_reproduction.make_llm`, the qualification identity, and the manifest. Update the Phase 51
tests to use the actual single-model schema and run `validate_config`.

### P2-1 — Single-model evaluator is not constructed

`run_reproduction.main_async:986-988` builds only attacker and target in the single-model path;
`evaluator_llm` remains `None`; the manifest records `rubric_model` as `evaluator_model`. The
evaluator role is in the qualification identity but has no single-model construction. This is a
role-semantics gap, distinct from the P1 mapping.

### P2-2 — Three-model CLI overrides are ignored

`run_reproduction.make_llm:292-295` routes any config with `models` and a manager directly to
`build_paid_role_llm` and never consults `args.target`/`args.attacker`. The single-model path applies
the overrides; the three-model path silently uses the configured models. It does not bypass the
report (the actual config models are what the gate sees), but `--target`/`--attacker` do not change
the three-model effective identity as the phase brief expects.

### P2-3 — Hand-built configs can omit the source hash

`run_reproduction.load_config` records `_source_sha256`, but a hand-built config passed to a public
helper can omit it. The expected source hash is then `None`; a report with a matching
`experiment_config.path` and R1 role triple passes. The effective model identity still applies; the
**source-file identity** is not bound. The manifest `config_hash` also includes `_source_sha256`,
so it is a normalized config hash, not the source hash. Documented caller contract; not a substitute
for the model check.

### P2-4 — `cfg=None` and legacy script caller contract

The exact R1 snapshot remains gated; a non-R1 model with `cfg=None` is a non-paper run and is not
gated. The legacy scripts listed in Phase 51 (`run_attacks.py`, `eval_*.py`, `judge_asr.py`,
`run_baseline_evals.py`, `run_full_evaluation.py`, `final_analysis.py`) do not pass a paper config.
This remains a caller-discipline contract.

### P3-1 — Role-config reference can be changed between gate and factory

`build_paid_role_llm` and `build_single_role_llm` pass the original caller `role_cfg` to the
factory after the gate. A caller-supplied dict subclass whose `get("model")` changes answers can
make the gate see R1 and the factory see `gpt-4o-mini` (reproduced as `M_FLIP`). A plain `dict` and
normal synchronous execution do not provide this; concurrent mutation is outside the trusted
same-process model. A defensive copy of the effective role config for construction would close it.
Classified P3 under the Phase 47 model.

### P3-2 — Config-hash composition

The manifest `config_hash` is computed over the cfg including `_source_sha256`; it is a normalized
config hash, not the raw source hash. The report's source hash is separate. This is not used by the
gate, but the three identities should not be conflated in manifests.

### P3-3 — Prior residuals

Non-atomic report writes; decision-hash TOCTOU; ignored extra decision fields; the legacy test-only
R1 fixture (Phase 51 is excluded); same-second run-ID behavior.

---

## 16. Remaining caller-contract and `cfg=None` risks

* A hand-built config can claim any `_source_sha256` or omit it; the effective model triple and the
  other identity fields are still checked, but the source-file identity is caller-controlled.
* A public helper with `cfg=None` and a non-R1 model is the documented non-paper path.
* A legacy script that does not pass a paper `cfg` cannot invoke the config-level gate for a
  non-R1 model; it remains caller discipline.
* The single-model evaluator and the three-model CLI behavior are the P2 findings above.
* The M custom-mapping path is P3 under the trusted in-process model.

---

## 17. Explicit authorization status

**No authorization is granted.**

* Phase 51 is not accepted as complete for the exact-R1 single-model path. P1-1 blocks a PASS.
* No qualification run, paid API call, smoke test, pilot, attack, experiment, or Deliverable 1
  execution is authorized.
* This audit did not implement a correction. A separate corrective phase and a new independent
  audit are required for the single-model evaluator identity, the source-hash contract, and the
  P2/P3 items.
* Any later paid qualification still requires a separate explicit human decision.

**Final verdict: `FAIL — BLOCKING IDENTITY DEFECT REMAINS`.**
