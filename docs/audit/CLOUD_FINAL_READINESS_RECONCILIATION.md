# CLOUD_FINAL_READINESS_RECONCILIATION

> **§3 SUPERSEDED BY PHASE 22.** The `$1.50` budget cap and all budget-based
> request blocking were removed: the paper specifies no API budget and the
> author's release defines none. §1's test counts (1013) describe the Phase 21
> tree; Phase 22 ends at 1067. See `PHASE_22_IMPLEMENTATION_REPORT.md`.

Resolves the four reporting inconsistencies in the previous readiness report, and
is the single authoritative source for test counts.

**Measured 2026-11-10 on branch `main`, HEAD `89ed03f` + uncommitted work,
Python 3.13.9, conda `base`, Windows 11, `OPENAI_API_KEY` unset.**

No number in this document is a transcription. Every count comes from an executed
run; where a historical snapshot cannot be re-executed, that is stated.

---

## 1. Test counts — measured, not asserted

### 1.1 The authoritative run

```powershell
python -m pytest tests/ -q --no-header -p no:cacheprovider --tb=no --junitxml=_snapC.xml
```

Verbatim pytest summary:

```text
4 failed, 1009 passed in 152.91s (0:02:32)
```

| Field | Value |
|---|---|
| Collected | **1013** (`pytest --collect-only` → `1013 tests collected`) |
| Passed | **1009** |
| Failed | **4** |
| Errors | **0** |
| Skipped | **0** |
| Duration | **152.91 s** (JUnit run) |
| Exit code | 1 (expected — 4 failing tests) |

Wall-clock duration is machine-load dependent, not a metric: a confirmation run of
the identical tree produced the identical outcome counts in **171.53 s**
(`4 failed, 1009 passed in 171.53s`). Only the outcome counts are treated as
evidence; the two durations are reported so neither reads as a discrepancy later.

JUnit cross-check: `tests="1013" failures="4" errors="0" skipped="0"
time="152.911"`.

### 1.2 Why the earlier report said 971 in one place and 974 in another

**Root cause: an ordering/transcription error, not a collection difference.**

The sequence at the end of that task was:

1. Full suite measured → **971 passed**, 4 failed, 0 skipped.
2. Three tests were added afterwards
   (`tests/test_cloud_pricing_and_snapshots.py` gained
   `test_cloud_defaults_reach_the_clients_so_the_manifest_cannot_lie`,
   `test_role_level_setting_wins_over_the_cloud_default`,
   `test_cloud_defaults_do_not_touch_local_roles`; that file went 22 → 25 tests).
3. Suite re-measured → **974 passed**. §3.1 of the report was updated to 974.
4. **§1 of the same report still showed 971.** That line was never refreshed.

Evidence that the delta is exactly those three tests:

```text
pytest tests/test_cloud_pricing_and_snapshots.py --collect-only -q
25 tests collected                                   # was 22
971 + 3 = 974
```

Ruled out, explicitly:

| Hypothesis | Verdict |
|---|---|
| Different test collection | **No** — same `tests/` target, same rootdir, same `pyproject.toml` (`testpaths = ["tests"]`) |
| Skipped tests | **No** — 0 skipped in every run reported here |
| Different execution command | **No** — the same `python -m pytest tests/ -q` in both cases |
| Configuration or markers | **No** — no markers, no `addopts`, no deselection |
| Reporting/transcription error | **Yes** — §1 was stale |

### 1.3 The deeper problem: two different "baselines" were conflated

The previous report's §3.1 labelled **758** as the "Phase-0 baseline". That is
wrong for that task. 758 was measured *before* the 120 cloud tests existed; the
actual state at the start of the environment-repair task was **878 / 44 / 4**.
Both numbers are real, at different moments:

| # | Moment | Collected | Passed | Failed | Errors |
|---|---|---:|---:|---:|---:|
| A0 | Before any cloud tests existed | 806 | 758 | 44 | 4 |
| A1 | After +120 cloud tests (end of task 1 = start of task 2) | 926 | 878 | 44 | 4 |
| B | After the environment repair (same 926 collected) | 926 | 922 | 4 | 0 |
| C | Current tree | 1013 | 1009 | 4 | 0 |

`A1` and `B` share a collected count (926) precisely because the repair added and
removed no tests — it only changed outcomes. That is the cleanest evidence that
the 44 recoveries are attributable to the dependency repair.

### 1.4 If a 971 figure is ever quoted

971 was a real measurement of an intermediate tree (step 1 above). It is not the
current state and no longer describes any document in the repository.

---

## 2. Regression comparison — exact node IDs

### 2.1 What is knowable, stated honestly

* **A** and **B** were measured before this audit. Their **aggregate counts** and
  their **not-passed node IDs** were recorded; their *complete* collected sets
  were not. So the comparison is **exact on the not-passed subsets** and
  **count-based** on the collected sets.
* **C** is measured now from `_snapC.xml` (per-test outcomes).
* **A cannot be re-executed.** Its 44 failures and 4 errors were import-time
  failures caused by the `torchvision`/`torchaudio` defect, which has since been
  repaired. Re-running today would not reproduce them, and simulating the defect
  would be a fabricated comparison. A's node IDs below are the recorded output of
  the run that produced `758 / 44 / 4`.

Reproduce the comparison with:

```powershell
python scripts/audit_test_reconciliation.py --junit _snapC.xml
```

### 2.2 Snapshot A — 758 passed / 44 failed / 4 errors (806 collected)

48 not-passed node IDs, by cause:

| Cause | Count | Node IDs |
|---|---:|---|
| `import transformers` failed (torchvision ABI) — parity suite | 36 | `tests/parity/test_{acronym,actor_attack,crescendo,nbf,opposite_day,refusal_threshold}_parity.py::*` (7 + 6 + 6 + 8 + 6 + 3) |
| Same cause — integration | 4 | `tests/test_crescendo_integration.py::{TestNBFParity,TestCheckpointCompat}::*` |
| Caused by the same import failure at fixture setup | 4 (ERROR) | `tests/test_phase17_stage2_infrastructure.py::{TestPipelineOwnership,TestResidencyCycles}::*` |
| CUDA-gated VRAM guard (`torch.cuda.is_available()` is False) | 2 | `tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::*` |
| Model weights absent from the local HF cache | 2 | `tests/test_phase8_architecture.py::*` |

The full 48 are listed in `scripts/audit_test_reconciliation.py::SNAPSHOT_A`, so
the historical record is machine-readable rather than prose.

### 2.3 Snapshot B — 922 passed / 4 failed / 0 errors (926 collected)

The 4 remaining:

```text
tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring
tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured
tests/test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid
tests/test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present
```

### 2.4 Snapshot C — 1009 passed / 4 failed / 0 errors (1013 collected)

The same four node IDs, byte-identical to B's set:

```text
tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured
tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring
tests/test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present
tests/test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid
```

### 2.5 Comparison matrix (computed)

| Comparison | Same not-passed tests? | Collected Δ | Recovered | Newly failing | Remaining failures |
|---|---|---:|---:|---:|---:|
| **A → B** | No (subset relation: B ⊂ A) | +120 | **44** | **0** | **4** |
| **B → C** | **Yes — identical sets** | +87 | 0 | **0** | **4** |
| **A → C** | No (subset relation: C ⊂ A) | +207 | **44** | **0** | **4** |

Accounting checks:

```
not_passed:   48 − 44 recovered = 4 remaining          ✓ (equivalently 44 + 4 = 48)
collected:   806 + 120 + 87 = 1013                      ✓
C passed:    1013 − 4 failed = 1009                     ✓
```

### 2.6 The distinction the brief asked for

A test that **errored during import** and now **passes** is counted as
**recovered**, not merely as "the error count went down". Example:
`tests/test_phase17_stage2_infrastructure.py::TestResidencyCycles::*` were 4
collection *errors* in A and are 4 passing tests in C — they appear in
`recovered`, and the error count fell 4 → 0 independently.

Mechanism, stated once: the 40 import-time failures and the 4 import-time errors
all trace to `import transformers` raising
`RuntimeError: operator torchvision::nms does not exist`. Repairing
`torchvision` removed the cause; no test source was edited to make them pass.

### 2.7 The 4 remaining failures

| Test | Cause | Fixable by |
|---|---|---|
| `test_phase14_2_rotation::…::test_releases_cache_before_measuring` | `ModelManager.pre_run_check` gates the cache release behind `if torch.cuda.is_available():`; torch is CPU-only | a CUDA torch build (approval-gated; see `CLOUD_ENVIRONMENT_COMPATIBILITY_REPORT.md` §6.2) |
| `…::test_effective_free_after_release_is_what_is_measured` | same | same |
| `test_phase8_architecture::TestThreeModelConfigValidation::test_default_config_valid` | `configs/reproduction_three_model.yaml` names `Qwen/Qwen3.5-4B`, absent from the local HF cache; validation correctly reports it | downloading the weights |
| `…::TestConfiguredModelsPresentOnDisk::test_attacker_present` | asserts those weights are on disk | downloading the weights |

All four were present in the original 48. None is caused by the cloud work.

---

## 3. Budget enforcement — audited

### 3.1 Answers to the ten questions

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | Does the qualification script accept `--budget-usd`? | **Yes** | `argparse` in `scripts/cloud_model_qualification.py`; `tests/test_cloud_qualification_harness.py::test_budget_flag_is_accepted_and_overrides_the_config` |
| 2 | Is the $1.50 cap in YAML or hardcoded? | **YAML** — `configs/cloud_qualification.yaml → qualification.budget_usd: 1.50`. The code has no default dollar figure. | same test asserts `cfg["qualification"]["budget_usd"] == 1.50` |
| 3 | Is the budget checked before **every** request? | **Yes** — `OpenAIChatLLM.generate` calls `ledger.check_budget(model)` before building the client or the request | `test_budget_exhausted_blocks_the_request_before_it_is_sent` asserts `backend.calls == 0` |
| 4 | Does it account for all three roles? | **Yes** — one shared `UsageLedger` is passed to all three clients | `test_one_ledger_enforces_the_budget_across_all_three_roles` (attacker spend stops the target) |
| 5 | Are cached input tokens handled correctly? | **Yes** — `estimate_cost` bills `cached` at `cached_input_per_1m` and `input − cached` at the full rate, clamping `cached ≤ input` | `test_cost_uses_separate_cached_input_rate`, `test_cached_tokens_cannot_exceed_input_in_the_calculation` |
| 6 | What if the rate is unknown? | **Refuses to run.** `check_budget` raises `BudgetExceededError` ("cannot be enforced") and no request is sent. Without a budget the cost is `null`, never `0`. | `test_budget_without_a_rate_refuses_to_run_blind`, `test_cost_is_none_not_zero_when_price_is_unknown` |
| 7 | What if a request succeeds but its usage record is unavailable? | **Was a defect — now fixed.** Previously a 200 with no `usage` object produced `0` tokens → **$0.00**, silently under-counting. Now the record is marked `usage_available=False`, its cost is `None` (`cost_basis: unavailable`), and the next `check_budget` **refuses** further requests on the grounds that spend can no longer be tracked. | `test_budget_refuses_when_a_response_carries_no_usage` (asserts cost is `None` and `backend.calls == 1`), `test_summary_counts_unpriceable_calls` |
| 8 | Can concurrent requests exceed the budget? | **Yes in principle.** `check_budget` is a pre-flight comparison against *recorded* spend, not a reservation. Sequential callers overshoot by at most one call; N concurrent calls bound the overshoot at N × one call. Both paid scripts issue requests strictly sequentially (single event loop, no `gather`), so N = 1 today. | documented in `UsageLedger.check_budget`; `test_budget_guard_is_a_preflight_check_against_recorded_spend` |
| 9 | Does it stop new calls at the limit? | **Yes** — raises before the request; `spent_usd >= budget_usd` triggers it, and the caller surfaces the error | `test_budget_blocks_once_exhausted` |
| 10 | Is the final spend persisted? | **Yes** — the pilot writes `<variant>.usage.json` + `.usage.jsonl`; the qualification writes `usage` into its report and one row per run into `qualification.jsonl` | `test_offline_run_passes_the_config_budget_to_the_guard` asserts budget, remaining, cost basis and the JSONL |

### 3.2 Exact mechanism, end to end

```
configs/cloud_qualification.yaml : qualification.budget_usd = 1.50
        │  (or --budget-usd, which wins)
        ▼
_cloud_common.build_ledger()  ->  UsageLedger(rate_card=configs/cloud_pricing.yaml, budget_usd=1.50)
        │
        ├── shared by the attacker, target and evaluator clients
        ▼
OpenAIChatLLM.generate():
    messages = trim(...)
    ledger.check_budget(model)          <── raises BEFORE any network I/O
        ├── spent >= budget            -> BudgetExceededError
        ├── any record had no usage    -> BudgetExceededError
        └── model has no rate          -> BudgetExceededError
    response = client.chat.completions.create(...)
    cost = estimate_cost(rate, input, cached, output)     (None if usage missing)
    ledger.record(UsageRecord(..., estimated_cost_usd=cost, usage_available=...))
```

### 3.3 Corrected command

The previous report proposed `--confirm-paid-request` alone and claimed a $1.50
cap. The cap does apply through the config, but it must be visible in the command.
Use the explicit form:

```powershell
python scripts/cloud_model_qualification.py --confirm-paid-request --budget-usd 1.50
```

### 3.4 Verdict

**PASS.** The cap is enforced before each request, shared across roles, refuses
on unknown pricing, refuses once spend is untrackable, and is persisted.
Documented limitation: not a reservation (question 8), which does not apply to
the sequential scripts shipped here.

---

## 4. PowerShell execution guide (Windows 11 · Python 3.13.9 · conda `base`)

PowerShell 5.1-compatible: no `&&`, no ternary. Run blocks in order.

```powershell
# 0. Go to the repository
Set-Location "C:\Users\CSLAB\Desktop\hemanth\Guardbound"

# 1. Session environment. KMP_DUPLICATE_LIB_OK only if you hit "OMP: Error #15".
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# 2. Provide the key in THIS shell only. Never in a file, never as an argument.
$env:OPENAI_API_KEY = "YOUR_API_KEY"

# 3. Confirm the key exists without printing it.
if ($env:OPENAI_API_KEY) { "OPENAI_API_KEY present (length " + $env:OPENAI_API_KEY.Length + ")" } else { "OPENAI_API_KEY ABSENT" }

# 4. Free: confirm the environment.
python -m pip check
python -c "import torch, torchvision, transformers, sentence_transformers; print(torch.__version__, torchvision.__version__, transformers.__version__, sentence_transformers.__version__)"

# 5. Free: full test suite, JUnit written for the reconciliation audit.
python -m pytest tests/ -q --no-header -p no:cacheprovider --tb=no --junitxml=_snapC.xml

# 6. Free: test-count reconciliation (expect A/B/C matrix, 4 remaining failures).
python scripts/audit_test_reconciliation.py --junit _snapC.xml

# 7. Free: prompt-fidelity audit (official release vs Guardbound, both conditions).
python scripts/audit_prompt_fidelity.py

# 8. Free: plan the qualification for EACH attacker condition.
python scripts/cloud_model_qualification.py --dry-run --attacker-prompt-condition baseline
python scripts/cloud_model_qualification.py --dry-run --attacker-prompt-condition c1_generation_point

# 9. Free: exercise both conditions against a fake transport.
python scripts/cloud_model_qualification.py --offline-mock --skip-nbf --attacker-prompt-condition baseline --out-dir _qual_mock
python scripts/cloud_model_qualification.py --offline-mock --skip-nbf --attacker-prompt-condition c1_generation_point --out-dir _qual_mock

# 10. Free: confirm the two reports are separate files.
Get-ChildItem _qual_mock -Filter *.qualification.json | Select-Object Name, Length

# ---------------------------------------------------------------- PAID below --
# 11. PAID — qualify the paper-methodology condition first (≈$1.50 cap).
python scripts/cloud_model_qualification.py --confirm-paid-request --budget-usd 1.50 --attacker-prompt-condition baseline

# 12. PAID — qualify C1 as a SEPARATE condition.
python scripts/cloud_model_qualification.py --confirm-paid-request --budget-usd 1.50 --attacker-prompt-condition c1_generation_point

# 13. Inspect both decisions side by side.
Get-ChildItem results\qualification -Filter *.qualification.json | ForEach-Object {
    $r = Get-Content $_.FullName -Raw | ConvertFrom-Json
    [pscustomobject]@{
        Report    = $_.Name
        Condition = $r.attacker_prompt_condition
        Overall   = $r.overall
        Attacker  = $r.decisions.attacker.decision
        Target    = $r.decisions.target.decision
        Evaluator = $r.decisions.evaluator.decision
        CostUSD   = $r.usage.estimated_cost_usd
        BudgetUSD = $r.usage.budget_usd
    }
} | Format-Table -AutoSize

# 14. Inspect the per-run ledger rows.
Get-Content results\qualification\qualification.jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Format-Table model, attacker_prompt_condition, overall, calls, estimated_cost_usd
```

Notes:

* Step 2 is the only place a secret appears. Do not add it to `configs/*.yaml`,
  a `.env` committed to git, or a command-line argument (arguments land in shell
  history).
* `$env:` values are per-session. A new terminal needs step 2 again.
* Steps 11–12 are **not** to be run until approved. Nothing above them spends
  money.
* Clean up the mock output when finished: `Remove-Item -Recurse -Force _qual_mock`.

---

## 5. Documents updated to the corrected counts

| Document | Change |
|---|---|
| `CLOUD_FINAL_READINESS_REPORT.md` | §1 and §3 counts replaced with the reconciled values; links to this document; §3.1 timeline corrected |
| `CLOUD_ENVIRONMENT_COMPATIBILITY_REPORT.md` | §5.5 "971 passed" → 1009, with a pointer here |
| `CLOUD_READINESS_REPORT.md` | superseded banner added; its 878 intersection is historical, not current |
| `CLOUD_MODEL_QUALIFICATION_REPORT.md` | test count for the harness refreshed |
| `CLOUD_INTEGRATION_AUDIT.md` | unchanged (its "120 tests" statement refers to that phase and remains correct) |

No number was hand-adjusted to make documents agree; every figure above is an
executed measurement, and the reconciliation script is the tool that produces the
comparison table.
