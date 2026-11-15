# CLOUD_FINAL_READINESS_REPORT

> **SUPERSEDED BY PHASE 22.** The budget cap, the "typographic deviations are
> preserved" decision, the 0.0 evaluator temperature and the Acronym-inclusive
> attack set were all changed in Phase 22. See
> `PHASE_22_IMPLEMENTATION_REPORT.md` and `HANDOFF.md` for the current state.

**Status: `READY WITH DOCUMENTED METHODOLOGY DEVIATIONS`**

Every number here is measured. **Zero paid API requests were made.** Cloud
inference is still **not** verified — the harness is validated and the methodology
is audited; the model is not.

- Date: 2026-11-10 · branch `main`, HEAD `89ed03f` + uncommitted work (§8)
- Python 3.13.9, conda `base`, Windows 11, NVIDIA RTX 4500 Ada (driver 595.79)
- Supersedes the previous revision of this file and `CLOUD_READINESS_REPORT.md`

**Companion documents**

| Document | Contents |
|---|---|
| `CLOUD_FINAL_READINESS_RECONCILIATION.md` | authoritative test counts, A/B/C regression matrix, budget audit (10 questions), PowerShell guide |
| `PAPER_PROMPT_FIDELITY_AUDIT.md` | 13 prompt slots compared byte-for-byte against the official release; the two attacker conditions |
| `PAPER_METHODOLOGY_FIDELITY_AUDIT.md` | attacks, target, evaluator, NBF, parameters, provider adaptation, deviations register |
| `CLOUD_EXPERIMENT_PROTOCOL_FINAL.md` | exact experiment configuration, budget, separation rules, run order |
| `CLOUD_ENVIRONMENT_COMPATIBILITY_REPORT.md` | torch/torchvision diagnosis and repair |
| `CLOUD_MODEL_QUALIFICATION_REPORT.md` · `CLOUD_PRICING_VERIFICATION.md` · `CLOUD_MODEL_SNAPSHOT_AUDIT.md` | role qualification harness; verified prices; snapshot audit |

---

## 1. Acceptance criteria — evidence

| Requirement | Result |
|---|---|
| PyTorch/torchvision compatibility | **PASS** — `torchvision 0.29.2+cpu` against `torch 2.14.1+cpu`; `pip check` conflict gone |
| Transformers compatibility | **PASS** — `from transformers import PreTrainedModel` OK |
| NBF embedder | **PASS** — real `[1, 768]` float32, deterministic, L2-normalized |
| NBF checkpoint | **PASS** — sha256 `cea1a75b…a136fe`; `h` finite, deterministic, formula verified |
| Parity tests | **PASS** — `tests/parity` **41 passed** |
| Attacker prompt fidelity | **DEVIATION (typographic)** — Crescendo byte-identical; Opposite-day and Acronym differ by blank-line whitespace / 3 apostrophe code points |
| Target prompt fidelity | **PASS** — dataset `target_system` verbatim; conversation shape verified |
| Evaluator prompt fidelity | **DEVIATION (typographic)** — refusal and disclaimer byte-identical; rubric differs by 4 whitespace chars, criteria unchanged |
| Attack implementation fidelity | **PASS** — logic faithful; loop quirks pinned by tests |
| NBF methodology fidelity | **PASS** — unchanged; real-barrier accept/filter/suppression verified |
| Cloud provider fidelity | **PASS** — SDK body equals the constructed messages; nothing added or rewritten |
| GPT-4o mini qualification | **HARNESS PASS / REAL PENDING** — 22 offline tests; mock correctly rejects target + evaluator |
| Pricing | **VERIFIED** — two official sources, 2026-11-10 |
| Model snapshots | **PASS (with deprecation)** — dated snapshots configured; both Deprecated, shutdown 2027-01-14 |
| Model substitution documented | **PASS** — Configuration B label + manifest fields; C1 adds a second label |
| Offline regression | **PASS** — 806 → 1013 collected; 48 → 4 not-passed; **0 newly failing** |
| Frozen experiment integrity | **PASS** — `git status` clean for `results/`, Phase 18/18-ext/19, all `reproduction*.yaml` |
| API expenditure | **ZERO** — every request went to an in-process `httpx.MockTransport` |

---

## 2. Test results (reconciled)

See `CLOUD_FINAL_READINESS_RECONCILIATION.md` for the per-node-ID matrix.

| Moment | Collected | Passed | Failed | Errors |
|---|---:|---:|---:|---:|
| A0 — before any cloud tests | 806 | 758 | 44 | 4 |
| A1 — after +120 cloud tests | 926 | 878 | 44 | 4 |
| B — after the environment repair | 926 | 922 | 4 | 0 |
| **C — current tree** | **1013** | **1009** | **4** | **0** |

Final run: `4 failed, 1009 passed in 152.91s (0:02:32)`, 0 skipped, JUnit
`tests="1013"`.

| Comparison | Recovered | Newly failing | Remaining | Collected Δ |
|---|---:|---:|---:|---:|
| A → B | 44 | 0 | 4 | +120 |
| B → C | 0 | 0 | 4 | +87 |
| A → C | 44 | 0 | 4 | +207 |

**The earlier 971-vs-974 discrepancy is resolved:** 971 was measured before three
`apply_cloud_defaults` tests existed; §1 of the old report was never refreshed.
Cause = transcription, not collection.

Cloud test inventory: **207 tests** across `test_cloud_provider.py` (57),
`test_cloud_usage.py` (35), `test_cloud_research_integration.py` (34),
`test_cloud_pricing_and_snapshots.py` (25),
`test_cloud_qualification_harness.py` (22), `test_cloud_nbf_integration.py` (7),
`test_attacker_arms.py` (27). 120 existed before the repair; 87 were added after.

### The 4 remaining failures (environmental, all in the original 48)

| Test | Cause |
|---|---|
| `test_phase14_2_rotation::…::test_releases_cache_before_measuring` | `pre_run_check` gates its cache release behind `torch.cuda.is_available()`; torch is CPU-only |
| `…::test_effective_free_after_release_is_what_is_measured` | same |
| `test_phase8_architecture::…::test_default_config_valid` | `Qwen/Qwen3.5-4B` absent from the local HF cache |
| `…::test_attacker_present` | asserts those weights are on disk |

---

## 3. What changed since the previous revision

### 3.1 Budget enforcement — one real defect found and fixed

Audited against ten questions (`CLOUD_FINAL_READINESS_RECONCILIATION.md` §3).
Nine were already correct: `--budget-usd` exists, the cap is in YAML (1.50), the
guard runs before **every** request, one ledger covers all three roles, cached
tokens are priced separately, unknown pricing refuses to run, the limit stops new
calls, and spend is persisted.

**Defect:** a `200` response carrying **no `usage` object** produced zero tokens →
**$0.00**, so the guard would keep spending while believing nothing had been
spent. Fixed: the record is marked `usage_available=False`, its cost is `None`
(never `0`), and the next `check_budget` **refuses** because spend can no longer
be tracked. Pinned by
`test_budget_refuses_when_a_response_carries_no_usage`.

Documented limitation: the guard is a pre-flight comparison, not a reservation, so
N concurrent calls could overshoot by N × one call. Both paid scripts issue
requests strictly sequentially.

### 3.2 Prompt fidelity — measured, and one earlier claim corrected

The previous report asserted the attacker prompts were byte-identical. That was
true for **Crescendo only**. An executable audit
(`scripts/audit_prompt_fidelity.py`) compares rendered message arrays between the
official release and Guardbound:

* **byte-identical:** Crescendo attacker (all rounds), refusal check (both copies),
  disclaimer check, target conversation shape, provider SDK payload;
* **typographic deviation:** Opposite-day attacker (2 blank lines lose 36 spaces),
  Acronym attacker (3 × `’`→`'`, 1 blank line), rubric prompt (1 collapsed
  double-space, 3 inserted spaces: `acton`→`act on`, `content,which`→
  `content, which`, `deviation.It`→`deviation. It`).

No wording, instruction, field name, example or rubric criterion differs. These
deviations are **pre-existing** (not introduced by the cloud work), documented by
the parity suite's own test names, and were **not "fixed"** because editing
prompts is forbidden and would fork the cloud run from the frozen local runs.

Also corrected: `CLOUD_INTEGRATION_AUDIT.md` previously described C1 as a judge
prompt. It is not — C1 is the Phase-17 attacker generation-point instruction.

### 3.3 Two attacker conditions, kept separate

| Condition | Attacker prompt | Purpose |
|---|---|---|
| `baseline` | original paper prompt | paper-methodology reproduction under model substitution |
| `c1_generation_point` | paper prompt + frozen I1 instruction, appended as a final user message | test whether I1's local-model gain transfers to cloud models |

Implementation: `src/guardbound/llm/attacker_arms.py`, an opt-in wrapper at the
call boundary. The paper path is a strict pass-through (the object is returned
unchanged). Properties, all tested:

* I1 wording equals the frozen
  `scripts/phase17_stage6_run_level_contract.py::INSTRUCTION` (drift guard);
* exactly one I1 message per attacker generation call;
* **never** sent to the target or the evaluator, even when a judge call is routed
  through the attacker object;
* separate config (`configs/cloud_research_gpt4o_mini_c1.yaml`), separate output
  directory, condition in the manifest, condition in the qualification filename;
* the two configs differ only in identity/label/condition —
  `_cloud_common.assert_arm_comparability` fails the suite otherwise;
* no assumption that I1 helps GPT-4o mini.

### 3.4 GPU

Unchanged: `torch 2.14.1+cpu`, CUDA unavailable. There is no CUDA build of torch
2.14.1 (`cu128` tops out at `2.11.0`), so restoring the GPU is a downgrade
decision. Not needed for the cloud work — the NBF runs correctly on CPU. Exact
command and trade-offs: `CLOUD_ENVIRONMENT_COMPATIBILITY_REPORT.md` §6.2.

---

## 4. Remaining blockers

| # | Blocker | Severity | Resolution |
|---|---|---|---|
| 1 | Model qualification not run (either condition) | Gate on the formal experiment | `PENDING USER APPROVAL` |
| 2 | Both models deprecated, shutdown 2027-01-14 | Time-boxes reproducibility | Run before that date; `gpt-6-luna` would be a new, separately labelled substitution |
| 3 | Three typographic prompt deviations | Disclosed, not blocking | Optionally restore the official bytes — needs authorization, and would fork from the frozen local runs |
| 4 | CUDA unavailable | Blocks local HF serving + 2 tests; not cloud work | Approval decision (§3.4) |
| 5 | `Qwen/Qwen3.5-4B` weights absent | Blocks 2 pre-existing tests | Download, or accept as environment-dependent |
| 6 | No API-side revision exposure | Reproducibility limitation | Snapshot pinned; run date recorded |

---

## 5. Next approved step

PowerShell only. Full sequence with inspection commands:
`CLOUD_FINAL_READINESS_RECONCILIATION.md` §4.

```powershell
Set-Location "C:\Users\CSLAB\Desktop\hemanth\Guardbound"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OPENAI_API_KEY = "YOUR_API_KEY"

# free
python -m pytest tests/ -q --no-header -p no:cacheprovider --tb=no --junitxml=_snapC.xml
python scripts/audit_test_reconciliation.py --junit _snapC.xml
python scripts/audit_prompt_fidelity.py
python scripts/cloud_model_qualification.py --dry-run --attacker-prompt-condition baseline
python scripts/cloud_model_qualification.py --dry-run --attacker-prompt-condition c1_generation_point

# PAID — one condition at a time, cap stated explicitly
python scripts/cloud_model_qualification.py --confirm-paid-request --budget-usd 1.50 --attacker-prompt-condition baseline
python scripts/cloud_model_qualification.py --confirm-paid-request --budget-usd 1.50 --attacker-prompt-condition c1_generation_point
```

Proceed to the pilot only if each condition's three role decisions are
`QUALIFIED`. Prompts must not be edited in response to the results.

---

## 6. Working-tree state

Modified: `.gitignore`, `requirements.txt`, `src/guardbound/llm/openai_client.py`,
`src/guardbound/llm/provider_factory.py`, `src/guardbound/llm/usage.py`,
`tests/conftest.py`, `tests/test_llm_clients_mocked.py`,
`tests/test_cloud_research_integration.py`, `tests/test_cloud_qualification_harness.py`.

New: `src/guardbound/llm/{cloud_errors.py,usage.py,attacker_arms.py}`, four
`configs/cloud_*.yaml` (+ `cloud_qualification.yaml`,
`cloud_research_gpt4o_mini_c1.yaml`), four cloud scripts, two audit scripts,
seven cloud test modules, `tests/__init__.py`, and the audit documents.

Pre-existing uncommitted work was **not** reset, reverted, stashed or overwritten.
No `git` operation beyond read-only inspection was performed. Frozen artifacts
(`results/`, Phase 18/18-ext/19, `reproduction*.yaml`, the NBF checkpoint bytes)
are untouched.

---

## 7. Status

**`READY WITH DOCUMENTED METHODOLOGY DEVIATIONS`**

Why not unconditional readiness:

1. **Documented prompt deviations** — three typographic normalizations in
   Opposite-day, Acronym and the rubric prompt (§3.2). Semantically neutral,
   measured, hashed, and disclosed rather than hidden.
2. **A model substitution and, optionally, a prompt-condition deviation** —
   `gpt-4o-mini` for all roles, and C1's I1 instruction as a separate arm.
3. **Qualification has not run** — a paid step pending approval.

Why not "not ready": every infrastructure and methodology blocker from the
previous rounds is closed. The environment is repaired and validated, the
parity suite and the real-checkpoint NBF path pass, pricing is verified,
snapshots are pinned, the provider is proven not to rewrite prompts, the two
attacker conditions are isolated and controlled, and the regression comparison is
exact with zero newly failing tests.

Nothing in this report claims that cloud inference works, that GPT-4o mini is
qualified, or that the prompts are byte-identical to the paper's.
