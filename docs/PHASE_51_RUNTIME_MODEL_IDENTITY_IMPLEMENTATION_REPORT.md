# PHASE 51 — OFFLINE CORRECTION OF R1 RUNTIME MODEL IDENTITY BINDING

**Phase:** 51 (corrective implementation, offline)
**Controlling audit:** Phase 50
**Controlling implementation:** Phase 49
**Verdict:** `IMPLEMENTATION COMPLETE — READY FOR PHASE 52 INDEPENDENT OFFLINE AUDIT`

The Phase 50 P0-1 identity defect is corrected. The qualification report and review decision are
now bound to the **effective runtime role models**, including CLI overrides, single-model role
blocks, three-model `models` blocks, and public-helper calls. An R1 report is rejected when any
actual role differs from the R1 triple, before the model factory or client constructor is reached.

No paid activity, credential access, live client construction, inference, qualification run, or
Deliverable 1 execution occurred.

---

## 1. Executive verdict

The Phase 50 failure was not a configuration-forwarding problem: Phase 49 already forwarded `cfg`.
The gate was validating an R1 report against a **substituted R1 config** while the single-model
path selected the actual role models elsewhere. The actual model could therefore be non-R1 even
though the report and decision were R1.

Phase 51 removes that substitution for supplied configurations and derives one effective three-role
identity used by both authorization and construction:

* single-model configs are normalized from their top-level `attacker`/`target`/`evaluator` blocks;
* a present `models` block is authoritative for three-model configs;
* a missing or empty effective role ID fails closed rather than assuming R1;
* a CLI override replaces the configured role ID in the effective identity before the report is
  checked;
* the actual role/model is passed through `run_reproduction.make_llm`, `guard_paid_model`,
  `build_paid_chat_llm`, `build_paid_role_llm`, and `build_single_role_llm`;
* `build_role_llms` uses the same effective models for its gate and its construction.

Every Phase 51 mismatch scenario rejects before the factory. The exact-R1 positive path and the
legitimate non-paper path still construct with the exact expected model.

---

## 2. Root cause and correction summary

| Phase 50 observation | Phase 51 correction |
|---|---|
| `_r1_authorization_config` loaded the R1 config whenever `cfg["models"]` was absent | A supplied config is normalized from its effective roles; missing role data fails closed |
| Single-model actual role models were selected separately from the identity | `make_llm` forwards the actual role; the helpers derive the effective identity from that role and model |
| CLI `--target`/`--attacker` overrides were invisible to the report identity | The override replaces the configured role model in the effective identity |
| An R1 report and R1 decision could authorize a non-R1 single-model target | The report's models are compared against the effective roles; the mismatch is stale and rejects before the factory |
| The alias `gpt-4o` could pass through `paper_reproduction` | The effective alias is compared as a distinct model ID and is rejected by the R1 report |
| The single-model config had no source SHA bound | `run_reproduction.load_config` records `_source_sha256`, and the report must match it |
| A paper config with missing role data had no identity | It fails closed with `QualificationError`; no R1 substitution |

The review decision remains bound to the exact report, run, configuration SHA, arm, and trigger.
Because the report now describes the effective role models, the decision is transitively bound to
the actual runtime identity. A triggered R1 decision is no longer usable for a non-R1 runtime.

---

## 3. Exact files changed

| File | Phase 51 change |
|---|---|
| `scripts/_cloud_common.py` | Added `_effective_role_models` and `_effective_runtime_config`; `expected_qualification_identity` uses the effective roles; `_r1_authorization_config` no longer substitutes R1 for a supplied config; `require_paid_authorization` accepts `role` and requires a concrete gated model; `guard_paid_model`, `build_paid_chat_llm`, `build_paid_role_llm`, and `build_single_role_llm` forward the role; structured-output identity defaults to the client default; docstring clarified |
| `scripts/run_reproduction.py` | `make_llm` forwards `role=role` to the single-model guard and builder; `load_config` records `_source_sha256`; docstring updated |
| `tests/test_phase51_runtime_identity.py` | **new** — 21 tests covering Scenarios A–L, effective-identity design, source-hash binding, and factory reachability |
| `tests/test_phase49_config_propagation.py` | test fakes accept the new `role` keyword; forwarding assertions unchanged |
| `tests/conftest.py` | Phase 51 module excluded from the legacy autouse R1 test context so the real helper boundary is observed |
| `docs/PHASE_51_RUNTIME_MODEL_IDENTITY_IMPLEMENTATION_REPORT.md` | **new** — this report |

No qualification config, R1 config, dataset, checkpoint, historical report, probe design, report
schema, review threshold, or experiment methodology was modified.

---

## 4. Effective runtime identity design

### 4.1 One role-model normalizer

`_effective_role_models(cfg)` in `_cloud_common.py`:

1. Uses `cfg["models"]` when it is a mapping. A present `models` block is authoritative; a missing
   role inside it is **not** filled from top-level roles.
2. Otherwise reads the top-level `attacker`, `target`, and `evaluator` role mappings, which is the
   `run_reproduction` single-model shape.
3. Requires a non-empty string model ID for all three roles. Missing IDs raise
   `PaidAuthorizationError` (a `QualificationError`) with the missing role names.
4. Never loads `configs/paper_reproduction_r1_gpt4o.yaml` for a supplied config.

### 4.2 One effective runtime config

`_effective_runtime_config(cfg, role=None, model_id=None)`:

* returns a copy; the caller's config is not mutated;
* keeps the `models` block authoritative or normalizes the top-level roles;
* when `role` is supplied and `model_id` is the actual model, replaces that role's model in the
  effective copy;
* when no role is supplied and the model is not present in the effective roles, fails closed
  (`cannot determine which effective role ...; pass the role explicitly`);
* when no model/role is supplied for a paper config, fails closed;
* `cfg=None` retains only the legacy exact-R1 contract: the frozen R1 config is the identity for an
  exact-R1 client, and the exact-R1 constructor guard remains independent.

`expected_qualification_identity(cfg)` now calls the same normalizer, so the report check and the
construction gate cannot drift. The structured-output identity uses the same `json_object` default
as the client when the config omits the setting.

### 4.3 Gate and construction use the same values

* `run_reproduction.make_llm` passes the selected role and actual model.
* `guard_paid_model` and `build_paid_chat_llm` accept `role`.
* `build_paid_role_llm` and `build_single_role_llm` already know the role; they forward it with the
  actual `role_cfg["model"]`.
* `build_role_llms` passes its `models`-derived model into the shared gate; the gate's identity and
  `build_three_model_stack` use the same `cfg["models"]`.
* The actual model passed to `OpenAIChatLLM` / `build_role_llm` / `_build_cloud_role` is the value
  used in the effective identity.

### 4.4 Frozen R1 and alias policy

`is_r1_model` remains exact: `gpt-4o` is not the frozen snapshot. Under a paper config the
config-level gate applies to the actual model, but the report's `configuration_under_qualification.models`
must equal the effective triple. An R1 report therefore rejects `gpt-4o`, `gpt-4o-mini`, or any
other role substitution. No alias normalization was added.

---

## 5. How CLI overrides enter the identity

`run_reproduction.make_llm` computes the actual model as before:

```python
override = args.attacker if role == "attacker" else args.target
model = override or cfg[role]["model"]
```

It now passes `role=role` to `guard_paid_model` and `build_paid_chat_llm`, and
`build_paid_role_llm`/`build_single_role_llm` pass their role. The effective identity replaces that
role's model with the actual override before the report is validated. A `--target gpt-4o-mini`
override therefore changes the expected target in the identity; the R1 report becomes stale and the
guard raises before the API-key branch or provider construction.

---

## 6. Report and review-decision binding

The report's `configuration_under_qualification.models` is compared to the effective roles in
`require_qualification`. A mismatch reports the recorded and expected models and raises before the
factory. The review decision is then required only for a triggered report and is still checked
against:

* report SHA-256;
* run ID;
* experiment configuration SHA-256;
* arm and trigger;
* decision, rationale, decider, and timestamp;
* `actor_attack_eligible`;
* exact frozen threshold `0.50`, strict `<`, finite `observed`, and boolean `triggered`.

Because the report now records the effective role IDs, a decision bound to an R1 report cannot
authorize a non-R1 runtime. Scenario H validates the R1 decision against a matching R1 report and
then demonstrates that the actual non-R1 target is still rejected before the decision can be
reused.

---

## 7. Scenarios A–L

| Scenario | Supplied paper config / effective roles | Report identity | Decision | Gate result | Factory reached | Model passed |
|---|---|---|---|---|---|---|
| A exact R1 | single; attacker/target/evaluator R1 | R1 | none (non-triggered) | pass | yes | R1 |
| B non-R1 target | single; target `gpt-4o-mini`, others R1 | R1 | none | stale models; reject | no | — |
| C non-R1 attacker | single; attacker `gpt-4o-mini`, others R1 | R1 | none | stale models; reject | no | — |
| D non-R1 evaluator | single; evaluator `gpt-4o-mini`, others R1 | R1 | none | stale models; reject | no | — |
| E target override | single R1; `args.target=gpt-4o-mini` | R1 | none | stale target; reject | no | — |
| F attacker override | single R1; `args.attacker=gpt-4o-mini` | R1 | none | stale attacker; reject | no | — |
| G alias | single; target `gpt-4o` | R1 | none | stale target; reject | no | — |
| H R1 decision | single; target `gpt-4o-mini` | R1, triggered | valid R1-bound decision | stale target; reject | no | — |
| I three-model | `models` target `gpt-4o-mini`, attacker/evaluator R1 | R1 | none | stale target; reject | no | — |
| J missing identity | paper single without target | — | — | missing model IDs; fail closed | no | — |
| K non-paper | non-paper; target `gpt-4o-mini` | not used | none | pass (unguarded) | yes | `gpt-4o-mini` |
| L helpers | paper `models`; target `gpt-4o-mini` | R1 | none | stale target on all helpers | no | — |

Additional checks:

* A non-triggered R1 report requires no decision and constructs R1.
* A triggered report with no decision is rejected with the review-decision error.
* Scenario H's R1 decision is validated with `common.require_review_decision` before the actual
  mismatch is tested.
* Scenario J also covers `build_paid_chat_llm` and `build_role_llms`; the latter fails before
  `build_three_model_stack`.
* Scenario L covers `guard_paid_model`, `build_paid_chat_llm` (with and without a role),
  `build_paid_role_llm`, `build_single_role_llm`, and `build_role_llms`.
* The no-role/unmatched-model case fails with `cannot determine which effective role`; the
  no-role/model-present case is still rejected as stale against the R1 report.
* Every rejection test uses a **synthetically valid R1 report**, not merely a missing report.

Factory-reachability was checked with spies on `OpenAIChatLLM`, `build_role_llm`,
`_build_cloud_role`, and `build_three_model_stack`; all reject scenarios record zero factory calls.

---

## 8. Three-model path preservation

The three-model path was already rejecting a non-R1 target against an R1 report. Phase 51 preserves
that behavior and routes both paths through `_effective_role_models`:

* `build_paid_role_llm` keeps the role and the actual role config;
* the `cfg["models"]` block is authoritative;
* the report's model triple is compared against all three effective models;
* Scenario I confirms a non-R1 target is rejected as stale before `build_role_llm`;
* `build_role_llms` uses the same identity and remains gated for a paper config.

No three-model model ID, provider, device, or stack behavior changed.

---

## 9. Phase 47 authorization lifecycle

`tests/test_phase47_authorization_lifecycle.py` — **22 passed**.

Preserved:

* explicit sealed `QualificationCapability`; no `inspect.stack()` filename authorization;
* actual `OfflineMockClient` identity; `client=None` rejected for offline qualification;
* no offline-to-live factory fallthrough;
* live confirmation before key/client access;
* scope owner and release, nested restoration, exception cleanup;
* async child-task and copied-context blocking;
* model isolation and failed-builder cleanup;
* exact R1 constructor guard and exact alias policy.

The Phase 51 module is excluded from the legacy autouse R1 fixture in `tests/conftest.py`, so the
positive Scenario A exercises the real helper-opened context rather than inheriting one. The
trusted in-process threat model is unchanged.

---

## 10. Test commands and results

Environment: Windows, Python 3.13, `torch 2.14.1+cpu`,
`torch.cuda.is_available() == False`, `KMP_DUPLICATE_LIB_OK=TRUE`.

```
python -m pytest tests/test_phase51_runtime_identity.py -q
```
**21 passed in 1.23s.** Exit 0.

```
python -m pytest tests/test_phase49_config_propagation.py -q
```
**8 passed in 0.90s.** Exit 0.

```
python -m pytest tests/test_phase47_authorization_lifecycle.py -q
```
**22 passed in 0.94s.** Exit 0.

```
python -m pytest tests/test_phase45_authorization.py -q
```
**27 passed in 1.06s.** Exit 0.

```
python -m pytest tests/test_phase43_corrective.py tests/test_phase41_probe_redesign.py -q
```
**105 passed in 0.65s.** Exit 0.

```
python -m pytest tests/test_cloud_qualification_harness.py \
  tests/test_phase24_p0_fixes.py tests/test_phase26_qualification_gate.py \
  tests/test_phase27_fail_closed_gate.py tests/test_phase36_implementation.py \
  tests/test_phase38_hardening.py tests/test_attacker_arms.py -q
```
**234 passed in 7.27s.** Exit 0.

```
python -m pytest tests/ -q
```
**1519 passed, 4 failed, 0 skipped in 167.65s.** Exit 1.

The four failures are current environment checks, not Phase 51 regressions:

* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring`
* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured`
* `test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid`
* `test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present`

Immediate causes: CPU-only torch / no CUDA, and `Qwen/Qwen3.5-4B` absent from the local HF cache.
Their historical pre-existence is **not** verified; I do not claim it. No Phase 51 test failed and
no new failure was introduced. The full suite is not described as successful.

---

## 11. Frozen-artifact integrity

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
| Phase 49 report | `91a13d7518abf8244b3e8269f338268609f29bc3e9c4c1bf75af87d7cd49c214` | match |
| Phase 50 report | `a6f992b1751996db69c3f0494dd924b98ef12342838ef7566dd2ed633b34b7a7` | match |

No frozen artifact was modified or regenerated.

---

## 12. Remaining P2/P3 risks

* **`cfg=None` caller contract.** The exact R1 model remains gated and the R1 config is its legacy
  identity. A non-R1 model with `cfg=None` is a non-paper run and is not gated. The legacy scripts
  that call the public helpers without `cfg` retain this contract.
* **Legacy paper-reproduction scripts without `cfg`.** `run_attacks.py`, `eval_*.py`,
  `judge_asr.py`, `run_baseline_evals.py`, `run_full_evaluation.py`, and `final_analysis.py` do not
  pass a paper config. The exact R1 model is still protected; a non-R1 model there is treated as
  non-paper. This is documented caller discipline, not the Phase 50 identity defect.
* **Single paper config completeness.** The `prompt_variant` default is `legacy`; a paper single
  config must set `prompt_variant: official` to match an official R1 report. This fails closed, not
  open. `run_reproduction.load_config` now records `_source_sha256`; a hand-built config without
  that key cannot bind the source hash, though the effective model identity still applies.
* **Non-R1 paper config with a matching non-R1 report.** The frozen R1 report is always rejected for
  non-R1 effective roles. A separate, deliberately qualified non-R1 report could describe that
  non-R1 configuration; that is not the frozen R1 authorization and would require its own
  methodology decision.
* **P3 residuals:** non-atomic qualification-report writes; decision-hash time-of-check/time-of-use
  exposure; ignored extra decision fields; the legacy test-only R1 fixture (Phase 51 is excluded);
  same-second run-ID collision behavior; public-helper `cfg=None` contract.

No P2/P3 item was fixed beyond what was necessary for the P0 identity correction. The
`run_reproduction.load_config` source hash and the structured-output default are the two adjacent
changes; both are identity-related and are covered by tests.

---

## 13. No-paid-activity confirmation

* No OpenAI, Anthropic, or other model API call.
* No API-key access, validation, printing, copying, or exposure. Tests use the literal
  `phase51-dummy-not-a-credential` only to enter the API-style branch; the real helper receives a
  `FakeOpenAIBackend` client.
* No live provider client; no network request.
* No qualification, inference, smoke test, pilot, attack, experiment, or download.
* No Deliverable 1 execution.
* No credentials or secrets in any modified file.
* No frozen artifact, config, dataset, checkpoint, or historical report modified.

Phase 51 completion is **not** paid-execution authorization. Phase 52 must independently audit this
correction. Any later qualification still requires a separate explicit human decision.
