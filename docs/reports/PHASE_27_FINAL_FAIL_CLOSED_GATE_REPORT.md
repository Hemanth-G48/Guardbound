# PHASE 27 — FINAL FAIL-CLOSED GATE REPORT

**Status: `READY FOR MINIMAL QUALIFICATION`.**

The mandatory local qualification gate now fails closed. A qualification run
constructs provider clients only when the local NBF decision is `QUALIFIED`; every
other outcome — `NOT_QUALIFIED`, `INSUFFICIENT_DATA`, a skipped probe, a missing or
corrupt checkpoint, an embedder failure, an invalid margin or score,
non-determinism, or a candidate-filter failure — stops the run before any client
exists, persists the report with `stopped_before_paid_probes: true`, and records zero
usage calls.

No paid request, qualification, smoke test, pilot or experiment was run. No API key
was read, requested or exposed. No budget cap exists or was introduced.

Date: 2026-11-10 · Windows 11 · Python 3.13.9 · branch `main` @ `89ed03f`
(+ uncommitted Phases 20-27).

---

## 1. Executive summary

| Question | Answer |
|---|---|
| Does `INSUFFICIENT_DATA` still permit client construction? | **No** — verified: 0 clients, 0 usage calls, exit 1 |
| Does `NOT_QUALIFIED` stop before clients? | **No change** — already did; re-verified (0 clients) |
| Does a passing gate still reach the probe sequence? | **Yes** — 3 clients, 60 usage calls, full probe set |
| Can the harness still return `QUALIFIED`? | **Yes** — a fully passing synthetic stack yields `overall: QUALIFIED`, exit 0 |
| Are the Phase 26 controls preserved? | **Yes** — NBF gate, probe completeness, CLI rejection, NBF-first ordering, usage JSONL |
| Offline suite | **1189 collected / 1185 passed / 4 failed / 0 errors / 0 skipped** — same four environmental node IDs |
| Classification | **`READY FOR MINIMAL QUALIFICATION`** |

The fix is a two-line semantic change in one function plus the stop-report's verdict
and reason; the rest of this phase is regression tests and independent
instrumentation.

---

## 2. Root cause and exact correction

### Root cause

`scripts/cloud_model_qualification.py::main` gated client construction on one
specific failure state:

```python
local_verdict = final_decision(local_decisions, local_statuses, config_intact,
                               required=("nbf_integration", "candidate_filter"))
if local_verdict == NOT_QUALIFIED:      # <-- only this state stopped
    … persist report; return 1
clients = {…}                            # INSUFFICIENT_DATA fell through to here
```

Phase 26 made `INSUFFICIENT_DATA` a *non-passing* verdict for the qualification
(`probes_complete=false`, overall never `QUALIFIED`) but still allowed it to continue,
so the run built three clients and issued 60 paid probes before reporting
`NOT_QUALIFIED`/`INSUFFICIENT_DATA`. Phase 26 §10 recorded exactly that
(`--offline-mock --skip-nbf`: clients 3, usage 60).

### Correction

```python
# Fail closed (Phase 27): the gate is mandatory, so *any* verdict other than
# QUALIFIED stops the run here. INSUFFICIENT_DATA is not permission to proceed.
if local_verdict != QUALIFIED:
    reason = ("local NBF readiness gate failed"
              if local_verdict == NOT_QUALIFIED
              else "local NBF readiness gate incomplete (INSUFFICIENT_DATA)")
    … results = {… "overall": local_verdict,            # verbatim, not coerced
                     "stopped_before_paid_probes": True,
                     "stop_reason": reason, …}
    _persist_report(…)
    return 1
clients = {…}
```

Two deliberate details:

* **`overall` is the local verdict, verbatim** — `NOT_QUALIFIED` for a measured
  failure, `INSUFFICIENT_DATA` for an unmeasurable one. Coercing everything to
  `NOT_QUALIFIED` would hide *why* the gate stopped.
* **The stop reason distinguishes the two cases** so an operator reading only the
  console still sees whether something failed or something could not be measured.

No other production line changed: the local probes, the metrics, the probe-status
model, the report shape (`stopped_before_paid_probes`, `stop_reason`, `probe_status`,
`usage_jsonl`), the paid-probe path, R1 values, attack code, NBF code, prompts,
dataset and provider layer are untouched.

---

## 3. Before / after execution sequence

```
BEFORE (Phase 26)                          AFTER (Phase 27)
------                                     ------
parse args                                 parse args
load config                                load config
reject paper overrides (exit 2)            reject paper overrides (exit 2)
validate paper protocol                    validate paper protocol
paid-confirmation gate (exit 2)            paid-confirmation gate (exit 2)
key presence check                         key presence check
usage ledger                               usage ledger
local NBF probe                            local NBF probe
candidate-filter probe                     candidate-filter probe
local decision + gate                      local decision + gate
  NOT_QUALIFIED  -> report, exit 1           any verdict != QUALIFIED
                                               -> report (exact verdict),
  INSUFFICIENT   -> CONTINUE                     stopped_before_paid_probes: true,
                    build 3 clients              usage 0 calls, exit 1
                    run paid probes (60 calls)  QUALIFIED -> build 3 clients
                    report NOT_QUALIFIED                   -> paid probes
                                                           -> report
```

Observed transition for the same command (`--offline-mock --skip-nbf`):

| | Phase 26 | Phase 27 |
|---|---|---|
| Local gate | `INSUFFICIENT_DATA` | `INSUFFICIENT_DATA` |
| Clients constructed | **3** | **0** |
| Usage calls / rows | **60 / 60** | **0 / 0** |
| Report `overall` | `NOT_QUALIFIED` | `INSUFFICIENT_DATA` (verbatim) |
| `stopped_before_paid_probes` | `false` | **`true`** |
| Exit code | 1 | 1 |

---

## 4. Regression test evidence

New file `tests/test_phase27_fail_closed_gate.py` — **25 tests**, all passing:

| Requirement (phase brief §3) | Test |
|---|---|
| 1. `QUALIFIED` permits client construction | `test_qualified_local_gate_reaches_client_construction` (3 clients, probe set present), `test_passing_gate_with_a_synthetic_qualified_stack_is_qualified` (exit 0, `overall: QUALIFIED`) |
| 2. `NOT_QUALIFIED` stops | parametrized `non_finite_score`, `threshold_mismatch`, `threshold_sign`, `nondeterministic`, `candidate_filter_failure` |
| 3. `INSUFFICIENT_DATA` stops | parametrized `skip_nbf`, `unmeasurable_determinism`, `candidate_filter_untested` |
| 4. `--skip-nbf` stops | `test_skip_nbf_stops_before_client_construction` (also asserts the console says "gate incomplete" and "no paid probe was run") |
| 5. Missing checkpoint stops | parametrized `missing_checkpoint` |
| 6. Embedder failure stops | parametrized `embedder_failure` |
| 7. Candidate-filter failure stops | parametrized `candidate_filter_failure` / `candidate_filter_untested` |
| 8. Every non-passing outcome persists a report with a non-success overall, `stopped_before_paid_probes: true`, local statuses, zero usage | parametrized `test_non_passing_outcome_persists_report_status_and_zero_usage` (10 scenarios) |
| 9. No client constructed in any non-passing scenario | parametrized `test_non_passing_local_gate_never_constructs_a_client` (10 scenarios, `clients == 0`) |
| 10. Passing gate still runs the existing probe sequence | `test_qualified_local_gate_reaches_client_construction`, `test_passing_gate_with_a_synthetic_qualified_stack_is_qualified` |

Additional ordering guards:

* `test_gate_runs_before_the_key_check_and_client_factory` — a failing gate with
  `build_single_role_llm` replaced by a function that raises: `main` returns 1
  without touching it.
* `test_paid_mode_gate_failure_never_reaches_the_provider` — without
  `--offline-mock` (i.e. the real path, `--confirm-paid-request`), a failing gate
  returns 1 with `clients == 0`, `usage.calls == 0`. A dummy literal key is used; no
  request is made.

The ten scenarios are exactly the list the brief enumerated: skip, missing
checkpoint, embedder failure, invalid/non-finite score, threshold mismatch, threshold
sign, non-determinism, unmeasurable checks, candidate-filter failure, and
insufficient candidate-filter coverage.

### Three tests updated (directly related, intent preserved)

Phase 27 changes the semantics these tests encoded. Each kept its subject and gained
a passing local gate via the new `passing_local_gate` fixture in `tests/conftest.py`:

| Test | Before | After |
|---|---|---|
| `test_phase26…::test_usage_jsonl_has_one_row_per_call` | ran with `--skip-nbf` | passing gate, no skip flag; still asserts rows == calls and the 21 fields |
| `test_phase26…::test_usage_jsonl_survives_an_interruption` | `--skip-nbf` + interrupt | passing gate + interrupt; still asserts retained rows |
| `test_phase26…::test_report_records_probe_status_and_completeness` | asserted the skipped NBF was `INSUFFICIENT` while probes ran | passing gate; asserts `nbf_integration: PASS`, `stopped_before_paid_probes: false`, and the mock still not `QUALIFIED` |
| `test_cloud_qualification_harness…::test_offline_run_reports_usage_without_a_cap` | `--skip-nbf` | passing gate; still asserts `cost_basis == "rate_card"` and a persisted ledger |
| `test_cloud_qualification_harness…::test_offline_mock_run_executes_probes_and_does_not_pass` | `--skip-nbf` | passing gate; still asserts the attacker passes and the target fails on the mock |
| `test_cloud_qualification_harness…::test_offline_mock_records_model_identity` | `--skip-nbf` | passing gate; unchanged assertions |
| `test_phase24…::test_offline_qualification_run_reports_the_r1_stack` | `--skip-nbf` | passing gate; still asserts the R1 stack block and the ActorAttack probe |

No unrelated test was modified, skipped, deleted or weakened. The four environmental
failures are untouched (§7).

---

## 5. Independent client-construction instrumentation

Standalone probe (not the test module) that instruments
`common.build_single_role_llm`, calls the real `main`, and reads the persisted
report:

| Case | Local gate | Exit | **Clients** | Overall | Stopped | Usage calls / rows | Provider lines |
|---|---|---:|---:|---|---|---:|---:|
| `--skip-nbf` | `INSUFFICIENT_DATA` | 1 | **0** | `INSUFFICIENT_DATA` | **true** | 0 / 0 | 0 |
| Missing checkpoint | `NOT_QUALIFIED` | 1 | **0** | `NOT_QUALIFIED` | **true** | 0 / 0 | 0 |
| Embedder failure | `NOT_QUALIFIED` | 1 | **0** | `NOT_QUALIFIED` | **true** | 0 / 0 | 0 |
| Local gate passes (mock roles fail) | `QUALIFIED` | 1 | 3 | `NOT_QUALIFIED` | false | 60 / 60 | 0 |
| Fully passing synthetic stack | `QUALIFIED` | **0** | 3 | **`QUALIFIED`** | false | 0 / 0 | 0 |

Probe-status evidence (from the same runs):

* skip: `nbf_integration: INSUFFICIENT_DATA`, `candidate_filter: INSUFFICIENT_DATA`,
  all four paid probes `NOT_EXECUTED`;
* missing checkpoint / embedder failure: `nbf_integration: FAIL`,
  `candidate_filter: INSUFFICIENT_DATA`, paid probes `NOT_EXECUTED`;
* local gate passing: all six probes executed —
  `attacker PASS`, `actor_attack_pre_attack PASS`, `nbf_integration PASS`,
  `candidate_filter PASS`, `target FAIL`, `evaluator FAIL` (the offline mock cannot
  recall context or classify refusals);
* fully passing synthetic stack: all six `PASS`, `probes_complete: true`.

Console evidence (verbatim):

```
local NBF gate      INSUFFICIENT_DATA
Refusing to run paid probes: local NBF readiness gate incomplete (INSUFFICIENT_DATA).
OVERALL: INSUFFICIENT_DATA (no paid probe was run)

local NBF gate      NOT_QUALIFIED
Refusing to run paid probes: local NBF readiness gate failed.
OVERALL: NOT_QUALIFIED (no paid probe was run)
```

`clients_constructed` is the only path to a provider request in this codebase
(`build_single_role_llm` creates the client that `generate()` calls), so
“0 clients” ⇒ “0 provider requests”. The probe process also logged **0** lines
matching `api.openai` / `HTTP Request`, and `OPENAI_API_KEY` was unset for every run
except the in-process fake used by one regression test.

---

## 6. Usage-ledger and report persistence

Every non-passing outcome persists:

| Artifact | Content |
|---|---|
| `<model>.baseline.qualification.json` | `overall` = the local verdict (`NOT_QUALIFIED` or `INSUFFICIENT_DATA`), `stopped_before_paid_probes: true`, `stop_reason`, `probe_status` for all six mandatory probes (local ones `FAIL`/`INSUFFICIENT_DATA`, paid ones `NOT_EXECUTED`), `metrics.local`, `decisions.local`, `probes_complete: false`, `usage` |
| `<model>.baseline.usage.jsonl` | created, **0 rows** (no request was made) |
| `qualification.jsonl` | one run row with `overall`, `probes_complete: false`, `stopped_before_paid_probes: true`, `calls: 0`, `usage_jsonl` |

The passing path still produces `usage.jsonl` with one row per call (60 rows for the
offline-mock run), flushed after each probe group, and the report's `usage` summary
matches the row count.

---

## 7. Full offline suite

```powershell
python -m pytest tests/ -q --no-header -p no:cacheprovider --tb=line --junitxml=_phase27_run1.xml
```

| | Phase 26 baseline | Phase 27 |
|---|---:|---:|
| Collected | 1164 | **1189** (+25) |
| Passed | 1160 | **1185** |
| Failed | 4 | **4** |
| Errors | 0 | **0** |
| Skipped | 0 | **0** |
| Time | 105.3 s | 128.2 s |

The failing node IDs are **identical** in both runs:

```
tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring
tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured
tests/test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid
tests/test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present
```

Two require CUDA (torch is CPU-only); two require `Qwen/Qwen3.5-4B` weights that are
not on disk. They were not modified, skipped or deselected.

**Newly introduced failures: none.** Tests not executed: none. The five test files
touched are listed in §8; every change preserves the test's subject.

---

## 8. Files changed and frozen-artifact integrity

| File | Change |
|---|---|
| `scripts/cloud_model_qualification.py` | gate condition `== NOT_QUALIFIED` → `!= QUALIFIED`; stop report records the exact local verdict and a case-specific `stop_reason` |
| `tests/test_phase27_fail_closed_gate.py` | **new**, 25 tests |
| `tests/conftest.py` | new `passing_local_gate` fixture (documented as “use only when the gate is not the subject”) |
| `tests/test_phase26_qualification_gate.py` | three tests updated to supply a passing gate instead of `--skip-nbf` |
| `tests/test_cloud_qualification_harness.py` | three tests updated the same way |
| `tests/test_phase24_p0_fixes.py` | one test updated the same way |

Nothing else was touched — in particular no attack, NBF, prompt, dataset, provider,
evaluation or R1 configuration file.

Frozen-artifact verification:

```
git status --porcelain -- results runs checkpoints data \
    configs/reproduction_phase12.yaml configs/reproduction_three_model.yaml \
    configs/reproduction_phase16_5_forensics.yaml nbf_original_stuff
# -> empty
```

| Artifact | State |
|---|---|
| `results/`, `runs/`, `checkpoints/`, `data/` | unchanged |
| Historical `reproduction*.yaml` | unchanged |
| R1 config (`paper_reproduction_r1_gpt4o.yaml`) | unchanged |
| Qualification config (`qualification_r1_gpt4o.yaml`) | unchanged (Phase 26 state) |
| Official checkpoint | `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` (re-hashed) |
| Original author code / historical prompt variants | unchanged |
| Phase 22-26 reports | not edited (only `HANDOFF.md`'s status line) |

No git write operation was performed.

---

## 9. Remaining discrepancies

| ID | Sev | Item | Note |
|---|---|---|---|
| — | — | **Legacy (non-paper) qualification config can no longer skip NBF** | `configs/cloud_qualification.yaml` has `nbf.threshold: 0.0` but no `nbf.eta`, so under the strict gate its local decision is `NOT_QUALIFIED`; its tests now supply a passing gate. Intended consequence of fail-closed (which applies to every qualification config), not an R1 issue. If the legacy harness is ever used again, complete its `nbf` block (`eta: 0.0`) or use the R1 config. |
| P2-a | P2 | `h = −η` float-noise boundary | decision at exactly −η follows the floating-point score (~1e-7 band vs η = 1e-3); shared with the official implementation; documented in Phase 26 |
| P3-a | P3 | Real-score filtered branch not observed in the candidate-filter probe | the three probe queries score below −η; the filtered branch is proven by the exact-boundary stage (h = 0 → 0 target calls) |
| P3-b | P3 | Per-probe wire temperature not recorded | the probes pass the configured temperatures; the report records the configuration, not each payload |
| P3-c | P3 | Interrupted **pilot** leaves `goals_completed: 0` | `complete: false` keeps the status unambiguous; outside this phase's scope |
| — | — | Phase 23 D-10 (empty chain can score) | faithful to `attack_single`; accepted |
| — | — | Phase 23 D-16 (upstream HarmBench byte check) | needs network; file hash pinned |

No P0 and no P1 remains.

---

## 10. Final readiness classification

# `READY FOR MINIMAL QUALIFICATION`

| Acceptance criterion (brief §7) | Met? | Evidence |
|---|---|---|
| Every local-gate outcome other than `QUALIFIED` stops before client construction | **yes** | §5: 10 regression scenarios + 3 independent runs, all **0 clients** |
| All regression tests pass | **yes** | 25/25 in the new file; the five updated tests pass |
| No new offline test failures | **yes** | 1189 / 1185 / 4, identical node IDs to Phase 26 |
| Existing R1 configuration and methodology unchanged | **yes** | no R1/NBF/attack/prompt/dataset change; §8 |
| No paid request or experiment executed | **yes** | key unset, offline mock only, 0 provider-request lines |

The gate is fail-closed by construction: `if local_verdict != QUALIFIED` is the only
condition guarding `build_single_role_llm`, and the verdict itself is derived from
per-probe statuses (Phase 26) rather than a caller-supplied flag. A skipped,
unmeasurable or failing local check therefore cannot reach a provider.

This classification is an implementation readiness decision. **It does not authorise
an API request.** The first paid qualification still requires a separate, explicit
decision, using the exact command from the Phase 26 report §14:

```powershell
Set-Location "C:\Users\CSLAB\Desktop\hemanth\Guardbound"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:OPENAI_API_KEY = "<paste in this shell only>"
python scripts/cloud_model_qualification.py `
    --config configs/qualification_r1_gpt4o.yaml `
    --confirm-paid-request
```

Expected on a passing stack: exit 0, `overall: QUALIFIED`, `probes_complete: true`,
and `<model>.baseline.usage.jsonl` with one row per provider call. On any local
failure: exit 1, `stopped_before_paid_probes: true`, **zero usage rows**, and the
failing local check named in `probe_status` and `stop_reason`.
