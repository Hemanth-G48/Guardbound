# PHASE 24 — TEST RESULTS

All runs offline. `OPENAI_API_KEY` unset for every command; no `.env` present; no
network access from any test (all provider interactions use `httpx.MockTransport`
or in-process scripted doubles).

---

## 1. Full suite

```powershell
Set-Location "C:\Users\CSLAB\Desktop\hemanth\Guardbound"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
python -m pytest tests/ -q --no-header -p no:cacheprovider --tb=line --junitxml=_phase24_final.xml
```

| Moment | Collected | Passed | Failed | Errors | Skipped | Time |
|---|---:|---:|---:|---:|---:|---:|
| Phase 23 baseline | 1067 | 1063 | 4 | 0 | 0 | 172.4 s |
| Phase 24 (final) | **1131** | **1127** | **4** | **0** | **0** | 198.4 s |

`1131 = 1067 + 64` (the new `tests/test_phase24_p0_fixes.py`).

The four failures are the **same four** environmental tests as the Phase 23
baseline, node for node, and were not touched:

```
tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring
tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured
tests/test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid
tests/test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present
```

Two require CUDA (`torch` is a CPU-only build); two require `Qwen/Qwen3.5-4B`
weights that are not on disk. Reported separately, **not** fixed, not skipped.

**No newly introduced failure.** Two tests were updated because they asserted the
defects this phase removed (see §4); both keep their original intent.

---

## 2. Focused runs

| Run | Result |
|---|---|
| `pytest tests/test_phase24_p0_fixes.py -q` | **64 passed** |
| `pytest tests/test_phase22_paper_fidelity.py tests/parity -q` | **103 passed** |
| `pytest tests/test_cloud_qualification_harness.py tests/test_attacker_arms.py tests/test_cloud_research_integration.py -q` | all passed |
| `python scripts/audit_prompt_fidelity.py` | `paper_path_faithful = true`, `prompt_variants` FAITHFUL |
| `python scripts/cloud_pilot.py --config configs/paper_reproduction_r1_gpt4o.yaml --variant r1_paper --dry-run --limit 2` | 0.7/0.7/0.7, paper attack set, official prompts, η 0.001, `R1 validated` |
| `python scripts/cloud_model_qualification.py --config configs/qualification_r1_gpt4o.yaml --offline-mock --skip-nbf` | runs every probe incl. ActorAttack pre-attack; report records the R1 stack (asserted in tests) |

---

## 3. New coverage by defect

| Defect | Tests | Count |
|---|---|---|
| D-01 threshold semantics | `test_filter_rule_*`, `test_steered_llm_chat_uses_the_same_rule`, `test_attack_level_filter_*`, `test_runner_converts_eta*`, `test_r1_configuration_propagates_eta*`, `test_r1_rejects_*` | 17 |
| D-02 composite steering | `test_composite_*` (8) + `test_pilot_passes_the_r1_steering_flags` | 9 |
| D-03 / D-09 aggregation | `test_aggregate_scores_*` (3), `test_asr_loader_excludes_failed_goals`, `test_asr_denominator_counts_only_valid_completed_goals`, `test_empty_denominator_is_undefined_not_zero` | 6 |
| D-04 / D-06 / D-07 / D-08 manifest | `test_manifest_prompt_hashes*`, `test_arm_manifest_block*`, `test_dataset_provenance*` (2), `test_git_state_helpers*`, `test_paper_reference*`, `test_offline_pilot_manifest_is_self_describing` | 7 |
| D-05 qualification | `test_r1_qualification_config*` (2), `test_evaluator_probes*`, `test_legacy_temperature*`, `test_target_probe*`, `test_attacker_contract_probe*`, `test_actor_attack_pre_attack*`, `test_offline_qualification_run*`, `test_qualification_metrics*`, `test_missing_actor_attack_probe*` | 10 |
| Boundary arithmetic | parametrisations (6 rows × 3 mechanisms) | counts inside the above |

All 64 are deterministic: no sleeps, no network, no randomness, no external files
beyond the shipped dataset/checkpoint metadata.

---

## 4. The two updated tests (and why it is not a weakened suite)

| Test | Before | After | Why |
|---|---|---|---|
| `tests/parity/test_refusal_threshold_parity.py::TestThresholdBoundary::test_sign_convention_threshold_is_negated` | asserted the literal source text `"safety_threshold = -eta if eta > 0 else 0.0"` | asserts `threshold_from_eta(0.5) == -0.5` **and** that the runner source calls `threshold_from_eta(eta)` | the literal was replaced by the shared helper; the invariant (raw threshold = −η) is unchanged and now checked at both levels |
| `tests/test_phase22_paper_fidelity.py::test_composite_requires_both_barrier_and_embed_fn_to_steer` | passed `barrier=object()` because the composite never used the barrier | supplies a minimal barrier with a `predictor` and asserts the wrapper is constructed with `eta=1e-3` and a per-actor turn guard | the composite now installs and uses the attack-level filter; the test's original intent (eta reaches the Q-filter, guard is sized per actor) is preserved |

No assertion was deleted, no test was skipped, no tolerance was widened, and the
four environmental failures were not touched.

---

## 5. Phase 23 baseline comparison

| Comparison | Result |
|---|---|
| Collected | 1067 → 1131 (+64) |
| Failed node IDs | identical set (4) |
| Newly failing | **none** |
| Recovered | not applicable (no previous failure was "fixed" — the threshold, steering and aggregation defects were not covered by failing tests before this phase) |
| Errors | 0 → 0 |
| Skipped | 0 → 0 |

A green suite is not the evidence for the fixes: each P0 is demonstrated by a
direct behavioural probe (boundary table, steering counters, ASR denominator
arithmetic) in addition to its tests.

---

## 6. Reproducing this report

```powershell
Set-Location "C:\Users\CSLAB\Desktop\hemanth\Guardbound"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue

python -m pytest tests/ -q --no-header -p no:cacheprovider --tb=line --junitxml=_phase24_final.xml
python -m pytest tests/test_phase24_p0_fixes.py -q
python scripts/audit_prompt_fidelity.py
python scripts/cloud_pilot.py --config configs/paper_reproduction_r1_gpt4o.yaml `
    --variant r1_paper --dry-run --limit 2
python scripts/cloud_model_qualification.py --config configs/qualification_r1_gpt4o.yaml --dry-run
```

Every command above is free and offline.
