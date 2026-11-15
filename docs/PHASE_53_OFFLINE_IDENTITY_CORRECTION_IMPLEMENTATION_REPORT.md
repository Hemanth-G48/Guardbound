# PHASE 53 — OFFLINE CORRECTION OF SINGLE-MODEL EVALUATOR IDENTITY AND CONFIGURATION CONSISTENCY

**Phase:** 53 (corrective implementation, offline)
**Controlling audit:** Phase 52
**Controlling implementation:** Phase 51
**Verdict:** `IMPLEMENTATION COMPLETE — READY FOR PHASE 54 INDEPENDENT OFFLINE AUDIT`

The Phase 52 P1-1 is corrected. The actual single-model schema now resolves its evaluator identity
from `evaluator.rubric_model`, constructs the single evaluator through the gated `make_llm` path,
and rejects every R1-report/non-R1-runtime mismatch before the factory. The source identity is
loader-bound and re-hashed; hand-built paper configs fail closed; three-model CLI overrides fail
explicitly; and the role configuration is snapshotted before construction.

No paid activity, credential access, live client, inference, qualification, smoke test, pilot,
attack, experiment, download, or Deliverable 1 execution occurred.

---

## 1. Executive verdict

| Phase 52 issue | Phase 53 result |
|---|---|
| P1-1 actual single evaluator schema (`rubric_model`) not used | **Fixed** — effective evaluator is the rubric; actual LLM constructed |
| P1-1 Scenario A actual schema failed `missing evaluator` | **Fixed** — actual schema A constructs target and evaluator as R1 |
| P1-1 B–H actual failed at evaluator, not intended role | **Fixed** — source/labelled actual schema rejects on the intended role |
| P2-1 single evaluator `None` / manifest rubric inconsistency | **Fixed** — single evaluator LLM built; manifest records effective model and subroles |
| P2-2 three-model CLI override silently ignored | **Fixed** — explicit `ValueError`; single-model overrides applied |
| P2-3 source hash optional/fabricated | **Fixed** — `_source_path` + re-hash; hand-built/fabricated source fails |
| P3 custom mapping changes model after gate | **Fixed** — effective role snapshot used for construction |
| Phase 47 lifecycle / Phase 44 / Phase 41/43 | **Pass** |

One non-blocking P2 remains: the identity's temperatures are still taken from the top-level
`temperature` block, while the actual single-model reproduction sets
`evaluator.temperature: 0.0`. The exact-R1 Scenario A test uses 0.7. This is a role-temperature
identity consistency finding, not a model-ID or authorization bypass.

---

## 2. Exact files changed

| File | Phase 53 change |
|---|---|
| `scripts/_cloud_common.py` | Single-model evaluator identity uses `rubric_model`; added `require_verified_source`; qualification/paid authorization require verified source; `require_paid_authorization` returns `effective_models`; role builders use the effective-model snapshot; `build_role_llms` snapshots the models; source path recorded |
| `scripts/run_reproduction.py` | `make_llm` evaluator resolution and single/three CLI semantics; single evaluator construction in `main_async`; evaluator subrole/config-hash metadata in `build_metadata`/`write_manifest`/`run_one`; `validate_config` rejects single `evaluator.model`; source path recorded |
| `tests/test_phase53_identity_correction.py` | **new** — 23 tests for actual schema A–V, W/X/Y, AC |
| `tests/test_phase51_runtime_identity.py` | test config updated to the actual evaluator rubric schema |
| `tests/test_phase45_authorization.py`, `tests/test_phase47_authorization_lifecycle.py` | test fixtures provide the verified R1 source identity |
| `tests/conftest.py` | Phase 53 module excluded from the legacy autouse R1 context |
| `docs/PHASE_53_OFFLINE_IDENTITY_CORRECTION_IMPLEMENTATION_REPORT.md` | this report |

No R1 config, `configs/reproduction.yaml`, `configs/reproduction_three_model.yaml`, dataset,
checkpoint, historical report, probe design, report schema, or review threshold was modified.

---

## 3. Actual single-model schema and evaluator trace

The actual single-model config `configs/reproduction.yaml:70-98` defines:

* `attacker.model`, `target.model`;
* `evaluator.rubric_model`;
* `evaluator.refusal_model`;
* `evaluator.disclaimer_model`;
* `evaluator.actor_step_judge_model`;
* `evaluator.temperature: 0.0`.

It does not define `evaluator.model`. `run_reproduction.validate_config:691,725-754` requires
`evaluator.rubric_model` and `evaluator.refusal_model` and maps the rubric to the evaluator role.

Current actual consumer:

1. `_effective_role_models` (`_cloud_common.py:238-282`) selects the evaluator model from
   `rubric_model` for the single-model schema and from `models.evaluator.model` for the three-model
   schema. Missing/empty/non-mapping values fail closed.
2. `run_reproduction.make_llm:300-315` resolves the evaluator from that effective identity. It no
   longer falls through `cfg[role]["model"]` and no longer applies `args.target` to the evaluator.
3. `run_reproduction.main_async:1045-1051` constructs `evaluator_llm = make_llm(cfg, "evaluator", ...)`
   through the same gated helper as attacker/target and calls `set_evaluator_llm` for every attack.
4. The attack classes' `get_evaluator_llm()` return the explicit evaluator LLM; the single-model
   path no longer silently falls back to the attacker.
5. `build_metadata`/`write_manifest` record `evaluator_model` from the effective identity and add
   `evaluator_subroles`, distinguishing the declared subrole field from the actual consumer model.

**Which model performs which function:** the single-model evaluator LLM is the rubric model. It
performs rubric, refusal/disclaimer, and actor-step-judge calls for the attack. The declared
`refusal_model`/`disclaimer_model`/`actor_step_judge_model` fields are not separately instantiated
clients; `evaluator_subroles` exposes the declared value and the actual `consumer_model` so the
manifest cannot imply a different consumer. The three-model `models.evaluator.model` is the single
evaluator LLM, and `validate_three_model` requires every `evaluators.*` subrole to route to it.

The evaluator identity is now resolved by the same function used by the report check. The `A`
actual test constructs both target and evaluator as R1; the `D` test changes only the rubric to
non-R1 and rejects before construction.

---

## 4. Effective identity resolution rules

1. A present `models` block is authoritative. Missing roles fail closed; top-level roles are not
   used as a fallback.
2. Otherwise the single-model top-level roles are normalized. `attacker.model` and `target.model`
   are required; the evaluator model is `evaluator.rubric_model`, not `evaluator.model`.
3. `evaluator.model` in a single-model config is rejected by `validate_config` as unsupported; the
   single-model schema uses the rubric.
4. Empty, whitespace-only, non-string, or non-mapping active role models fail closed before report
   validation.
5. No missing role is populated from `configs/paper_reproduction_r1_gpt4o.yaml`.
6. `_effective_runtime_config` substitutes the caller's actual role model when a role/override is
   supplied; all other effective role models come from the same effective config.
7. The effective role snapshot is used for both authorization and construction.
8. The effective model ID is passed to `OpenAIChatLLM`/`build_role_llm`, and recorded in the manifest.

---

## 5. Evaluator subrole treatment

* **Single-model:** one evaluator LLM, built from `rubric_model`; all evaluator calls route through
  it. `refusal_model`, `disclaimer_model`, and `actor_step_judge_model` are declaration-only for
  this path, not separate clients. `evaluator_subroles` records each declared value and the actual
  `consumer_model`. The `E` test sets the rubric to R1 and refusal/disclaimer/actor to non-R1 and
  proves the actual evaluator is R1 while the manifest records the declaration/consumer split.
* **Three-model:** one evaluator LLM from `models.evaluator.model`; `validate_three_model` requires
  `evaluators.*` to route to `"evaluator"`. `L` confirms all three roles reach the factory as R1;
  `M` rejects a non-R1 target with an R1 report.
* No evaluator subrole is silently given an independent unqualified model. The non-rubric single
  fields are demonstrably not separate consumers; they remain visible in the manifest.

---

## 6. Single-model and three-model CLI semantics

* **Single-model:** `--target` and `--attacker` are supported; the actual override enters the
  effective identity before authorization. There is no evaluator CLI override; `--target` no longer
  leaks into the evaluator. `H`/`I` show non-R1 overrides rejected against an R1 report.
* **Three-model:** `--target`/`--attacker` are **not supported**; `make_llm` raises a clear
  `ValueError` instead of ignoring them. `J` proves no factory call.
* `N`/`O` are therefore: single overrides applied and validated; three-model overrides explicitly
  rejected. This is the documented mode distinction.
* The report is always computed from the model that will be passed to the factory; a CLI override
  cannot leave an old report passing.

---

## 7. Source-hash contract

Three identities are kept distinct:

1. **Source-file identity:** `run_reproduction.load_config` and `common.load_config` record
   `_source_path` and `_source_sha256` over the actual file bytes.
2. **Normalized experiment hash:** `config_hash` in the metadata/manifest hashes the cfg (including
   the source fields). It is not the source hash and is not used by the gate.
3. **Effective runtime identity:** the effective role model triple and associated identity fields.

`require_verified_source`:

* requires both `_source_path` and `_source_sha256` for a supplied paper/exact-R1 config;
* re-hashes the file at the path;
* rejects a missing path/file, missing hash, or mismatch.

Results:

* `P` loaded config + matching report passes.
* `Q` file change after load is rejected (`hash changed`).
* `R` hand-built paper config without a source identity fails closed.
* `S` fabricated `_source_sha256` is rejected against the file hash.
* `T` source matches but effective model differs → stale models.
* `U` effective model matches but report source hash differs → stale source.
* `V` old/non-v2 report schema is not silently accepted.
* CLI overrides change the effective model but not the file hash; the model check rejects an old
  report, as `H`/`I`/`E` show.

The trusted mechanism is the normal loader's source record + re-hash, under the Phase 47 trusted
in-process code model. A hand-built mapping with a valid path/hash can still claim the same source
file; the effective model identity remains mandatory. This is documented as a P2 caller contract,
not a model-identity bypass.

---

## 8. Manifest/report identity consistency

`build_metadata` and `write_manifest` now use the effective role models:

* single `evaluator_model` = `rubric_model`;
* three-model `evaluator_model` = `models.evaluator.model`;
* `evaluator_subroles` records the declared field and the actual `consumer_model`;
* `config_hash` is explicitly the normalized configuration hash, separate from `_source_sha256`;
* attacker/target/evaluator manifest fields use the effective models, not a top-level field that
  may differ from the factory input.

`K` verifies the single-schema manifest; `E` verifies the declared-vs-consumer subroles; `L`
verifies the three-model model IDs.

---

## 9. Authorization-to-factory model trace

For every accepted or rejected case, the same effective model is used by the report comparison and
the factory:

1. config load records source path/hash;
2. `_effective_role_models` normalizes the roles;
3. `make_llm` selects the override/evaluator/role model;
4. `require_paid_authorization` builds the effective config, verifies source, compares the report,
   and returns `effective_models`;
5. `build_paid_chat_llm` constructs exactly `model_id`;
6. `build_paid_role_llm`/`build_single_role_llm` construct from the effective role snapshot;
7. `build_role_llms` builds the three-model stack from the effective snapshot;
8. the manifest records the same effective model.

`A`, `W`, `L`, and `Y` provide the positive evidence; all rejection tests assert empty factory
calls or empty spies.

---

## 10. Scenarios A–AC

| Scenario | Result |
|---|---|
| **A** actual schema exact R1 | target/evaluator constructed as R1; both factory calls R1 |
| **B** non-R1 target | rejected stale before factory |
| **C** non-R1 attacker | rejected stale before factory |
| **D** non-R1 evaluator rubric | rejected stale before factory |
| **E** evaluator subroles | rubric active; refusal/etc. declared but same evaluator consumer; manifest records it |
| **F** missing rubric | fail closed; no factory |
| **G** empty rubric | fail closed; no factory |
| **H** target override | effective target non-R1; R1 report stale; no factory |
| **I** attacker override | effective attacker non-R1; R1 report stale; no factory |
| **J** three-model override | explicit `ValueError`; no factory |
| **K** manifest | effective evaluator and subroles recorded; config hash separated |
| **L** three-model exact R1 | target/attacker/evaluator factory models R1 |
| **M** three-model non-R1 | stale before `build_role_llm`; spy empty |
| **N** single supported overrides | applied and rejected against R1 report |
| **O** three unsupported overrides | explicit error |
| **P** loaded config/source/report | pass |
| **Q** source file changed | rejected |
| **R** hand-built no source | rejected |
| **S** fabricated source hash | rejected |
| **T** source match, model differs | rejected |
| **U** model match, report source differs | rejected |
| **V** old schema | rejected |
| **W** authorization/factory model | same R1 evaluator |
| **X** factory reachability | all rejection tests have zero factory calls/spies |
| **Y** mapping flip | snapshot constructs R1, not the later non-R1 answer |
| **Z** Phase 47 | 22 passed |
| **AA** Phase 44 | 27/105 passed; valid R1 decision still cannot authorize non-R1 (Phase 51 H preserved) |
| **AB** Phase 41/43 | 105/234 passed |
| **AC** non-paper local | `make_llm` mock evaluator works; no R1 report required |

---

## 11. Test commands and results

Environment: Windows, Python 3.13, `torch 2.14.1+cpu`, no CUDA,
`KMP_DUPLICATE_LIB_OK=TRUE`.

```
python -m pytest tests/test_phase53_identity_correction.py -q
```
**23 passed in 4.52s.** Exit 0.

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
**1542 passed, 4 failed, 0 skipped in 167.99s.** Exit 1.

The four failures are current environment checks, not Phase 53 regressions:

* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring`
* `test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured`
* `test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid`
* `test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present`

Causes: CPU-only torch/no CUDA and `Qwen/Qwen3.5-4B` absent from the local cache. Historical
pre-existence is not claimed. The full suite is not described as successful.

---

## 12. Frozen-artifact hash verification

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
| Phase 51 report | `478ca1a85e891b4c6159ad9fc16e5aa438f0c11832eb840af5c76c0f2ff688ea` | match |
| Phase 52 report | `da7308bc385c82b417a767899a74591d858a6131a9dfb0770032e05341380f37` | match |

No frozen artifact was modified or regenerated. The `configs/reproduction.yaml` and
`reproduction_three_model.yaml` were read, not modified.

---

## 13. Remaining P0/P1/P2/P3 findings

### P2-1 — Role-temperature identity remains top-level

`expected_qualification_identity` derives temperatures from the top-level `temperature` block
(default 0.7). The actual single-model `reproduction.yaml` sets `evaluator.temperature: 0.0`.
The exact-R1 Phase 53 A fixture uses 0.7, which matches the R1 report; the local non-paper config
is 0.0. This is a role-specific temperature identity inconsistency, not a model-ID bypass. A later
correction should resolve temperatures per role without changing the pinned R1 temperatures.

### P2-2 — `cfg=None` and legacy script caller contract

The exact R1 snapshot remains gated; a non-R1 model with `cfg=None` is a non-paper run. Legacy
scripts that pass no paper config retain this documented caller contract.

### P2-3 — Single `evaluator.model` is rejected

The actual single-model schema uses `rubric_model`; `validate_config` now rejects a single-config
`evaluator.model` rather than silently interpreting it. This is intentional schema clarity.

### P2-4 — Hand-built source path/hash trust

A hand-built mapping can claim a valid source path/hash pair; the file is re-hashed and must match,
but the config contents are not independently hashed. The effective model identity is still
mandatory. This is a trusted-caller contract, not a model bypass.

### P3 residuals

Non-atomic qualification-report writes; decision-hash TOCTOU; ignored extra decision fields; the
legacy test-only R1 fixture (Phase 51 and 53 are excluded); same-second run-ID behavior. The
Phase 52 custom-mapping reproduction is now blocked by the effective-role snapshot.

---

## 14. No-paid-activity confirmation

* No API credential was read, printed, validated, copied, or exposed. Tests use the literal
  `phase53-dummy-not-a-credential` only to enter the API-style branch.
* No live provider client; all construction used `FakeOpenAIBackend` or a factory spy.
* No paid request, qualification, inference, smoke test, pilot, attack, experiment, or download.
* No Deliverable 1 execution.
* No frozen artifact, config, dataset, checkpoint, or historical report modified.
* Phase 53 completion is **not** paid-execution authorization.

Phase 54 must independently audit the actual-schema evaluator identity, source contract, CLI
semantics, manifest, and snapshot. Any later qualification requires a separate explicit human
decision.
